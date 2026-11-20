import Accelerate
import Foundation
import Metal

public enum ComputeBackend: String, Sendable {
    case metal
    case cpu
}

public enum MetalEngineError: Error, LocalizedError, Sendable {
    case metalUnavailable
    case libraryCompilationFailed(String)
    case missingKernelFunction(String)
    case commandBufferFailed(String)
    case invalidDimensions(String)

    public var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            return "Metal device or command queue is unavailable on this host."
        case .libraryCompilationFailed(let msg):
            return "Runtime MSL library compilation failed: \(msg)"
        case .missingKernelFunction(let name):
            return "Missing compiled MSL kernel function: \(name)"
        case .commandBufferFailed(let msg):
            return "Metal command buffer execution failed: \(msg)"
        case .invalidDimensions(let msg):
            return "Invalid tensor dimensions: \(msg)"
        }
    }
}

private struct SimilarityParams {
    var numQueries: UInt32
    var numDocs: UInt32
    var dim: UInt32
    var normalizeInputs: UInt32
}

private struct RouteParams {
    var numMessages: UInt32
    var numSubscribers: UInt32
    var dim: UInt32
}

private struct PrefixParams {
    var numQueries: UInt32
    var numWorkers: UInt32
    var stride: UInt32
}

/// Thread-safe Apple Silicon Metal compute engine with runtime-compiled MSL shaders
/// and an Accelerate-backed CPU reference fallback.
public final class MetalComputeEngine: @unchecked Sendable {
    public let device: (any MTLDevice)?
    public let deviceName: String
    public let hasUnifiedMemory: Bool
    public let activeBackend: ComputeBackend

    private let commandQueue: (any MTLCommandQueue)?
    private let psoCosineF32: (any MTLComputePipelineState)?
    private let psoCosineF16: (any MTLComputePipelineState)?
    private let psoDotInt8: (any MTLComputePipelineState)?
    private let psoRoute: (any MTLComputePipelineState)?
    private let psoPrefix: (any MTLComputePipelineState)?

    // Serializes command submission & completion against shared unified-memory slabs.
    private let dispatchQueue = DispatchQueue(label: "ai.hyphn8.metal-agent-bus.compute")

    /// Initializes the Metal runtime engine by compiling embedded MSL source at runtime.
    /// Set `preferCPU = true` to force the CPU reference path (useful for parity checks & benchmarks).
    public init(preferCPU: Bool = false) throws {
        if preferCPU {
            self.device = nil
            self.commandQueue = nil
            self.deviceName = "CPU Reference (Accelerate)"
            self.hasUnifiedMemory = true
            self.activeBackend = .cpu
            self.psoCosineF32 = nil
            self.psoCosineF16 = nil
            self.psoDotInt8 = nil
            self.psoRoute = nil
            self.psoPrefix = nil
            return
        }

        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue()
        else {
            self.device = nil
            self.commandQueue = nil
            self.deviceName = "CPU Fallback (No Metal Device)"
            self.hasUnifiedMemory = true
            self.activeBackend = .cpu
            self.psoCosineF32 = nil
            self.psoCosineF16 = nil
            self.psoDotInt8 = nil
            self.psoRoute = nil
            self.psoPrefix = nil
            return
        }

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        let library: any MTLLibrary
        do {
            library = try dev.makeLibrary(source: MSLShaders.source, options: options)
        } catch {
            throw MetalEngineError.libraryCompilationFailed(error.localizedDescription)
        }

        func pipeline(_ name: String) throws -> any MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw MetalEngineError.missingKernelFunction(name)
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        self.device = dev
        self.commandQueue = queue
        self.deviceName = dev.name
        self.hasUnifiedMemory = dev.hasUnifiedMemory
        self.activeBackend = .metal
        self.psoCosineF32 = try pipeline("batch_cosine_similarity_f32")
        self.psoCosineF16 = try pipeline("batch_cosine_similarity_f16")
        self.psoDotInt8 = try pipeline("batch_dot_int8_quantized")
        self.psoRoute = try pipeline("semantic_route_threshold")
        self.psoPrefix = try pipeline("prefix_overlap_score")
    }

    // MARK: - Similarity Kernels

    /// Computes `[numQueries x numDocs]` fp32 similarity scores directly against a unified-memory corpus slab.
    public func batchSimilarityF32(
        queries: [Float],
        numQueries: Int,
        corpus: SharedSlab<Float>,
        numDocs: Int,
        dim: Int,
        normalizeInputs: Bool = false
    ) throws -> [Float] {
        guard numQueries > 0, numDocs > 0, dim > 0 else { return [] }
        guard queries.count == numQueries * dim else {
            throw MetalEngineError.invalidDimensions("queries.count (\(queries.count)) != \(numQueries * dim)")
        }
        guard corpus.count >= numDocs * dim else {
            throw MetalEngineError.invalidDimensions("corpus.count (\(corpus.count)) < \(numDocs * dim)")
        }

        return try dispatchQueue.sync {
            if activeBackend == .metal,
               let dev = device,
               let queue = commandQueue,
               let pso = psoCosineF32,
               let corpusBuf = corpus.metalBuffer
            {
                let qBytes = queries.count * MemoryLayout<Float>.stride
                let outCount = numQueries * numDocs
                let outBytes = outCount * MemoryLayout<Float>.stride
                guard let qBuf = queries.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: qBytes, options: .storageModeShared) }),
                      let outBuf = dev.makeBuffer(length: outBytes, options: .storageModeShared),
                      let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder()
                else {
                    throw MetalEngineError.commandBufferFailed("Failed to allocate buffers or encoder for f32 similarity.")
                }

                var params = SimilarityParams(
                    numQueries: UInt32(numQueries),
                    numDocs: UInt32(numDocs),
                    dim: UInt32(dim),
                    normalizeInputs: normalizeInputs ? 1 : 0
                )

                enc.setComputePipelineState(pso)
                enc.setBuffer(qBuf, offset: 0, index: 0)
                enc.setBuffer(corpusBuf, offset: 0, index: 1)
                enc.setBuffer(outBuf, offset: 0, index: 2)
                enc.setBytes(&params, length: MemoryLayout<SimilarityParams>.stride, index: 3)

                dispatchSimdgroups(encoder: enc, pso: pso, cols: numDocs, rows: numQueries)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                if cmd.status != .completed {
                    throw MetalEngineError.commandBufferFailed(cmd.error?.localizedDescription ?? "Metal command buffer status=\(cmd.status.rawValue)")
                }

                let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
                return Array(UnsafeBufferPointer(start: ptr, count: outCount))
            }

            // CPU Accelerate reference path
            var scores = [Float](repeating: 0, count: numQueries * numDocs)
            let cPtr = corpus.pointer
            queries.withUnsafeBufferPointer { qBuf in
                let qBase = qBuf.baseAddress!
                for q in 0..<numQueries {
                    let qVec = qBase.advanced(by: q * dim)
                    var qNorm: Float = 1
                    if normalizeInputs {
                        var sq: Float = 0
                        vDSP_svesq(qVec, 1, &sq, vDSP_Length(dim))
                        qNorm = sqrt(sq)
                    }
                    for d in 0..<numDocs {
                        let dVec = cPtr.advanced(by: d * dim)
                        let dotVal = VectorMath.dot(qVec, dVec, count: dim)
                        if normalizeInputs {
                            var dsq: Float = 0
                            vDSP_svesq(dVec, 1, &dsq, vDSP_Length(dim))
                            let denom = qNorm * sqrt(dsq)
                            scores[q * numDocs + d] = denom > 1e-12 ? (dotVal / denom) : 0
                        } else {
                            scores[q * numDocs + d] = dotVal
                        }
                    }
                }
            }
            return scores
        }
    }

    /// Computes `[numQueries x numDocs]` fp16 dot-product scores (pre-normalized embeddings) with fp32 accumulation.
    public func batchSimilarityF16(
        queries: [Float16],
        numQueries: Int,
        corpus: SharedSlab<Float16>,
        numDocs: Int,
        dim: Int
    ) throws -> [Float] {
        guard numQueries > 0, numDocs > 0, dim > 0 else { return [] }
        guard queries.count == numQueries * dim else {
            throw MetalEngineError.invalidDimensions("queries.count (\(queries.count)) != \(numQueries * dim)")
        }
        guard corpus.count >= numDocs * dim else {
            throw MetalEngineError.invalidDimensions("corpus.count (\(corpus.count)) < \(numDocs * dim)")
        }

        return try dispatchQueue.sync {
            if activeBackend == .metal,
               let dev = device,
               let queue = commandQueue,
               let pso = psoCosineF16,
               let corpusBuf = corpus.metalBuffer
            {
                let qBytes = queries.count * MemoryLayout<Float16>.stride
                let outCount = numQueries * numDocs
                let outBytes = outCount * MemoryLayout<Float>.stride
                guard let qBuf = queries.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: qBytes, options: .storageModeShared) }),
                      let outBuf = dev.makeBuffer(length: outBytes, options: .storageModeShared),
                      let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder()
                else {
                    throw MetalEngineError.commandBufferFailed("Failed to allocate buffers or encoder for f16 similarity.")
                }

                var params = SimilarityParams(
                    numQueries: UInt32(numQueries),
                    numDocs: UInt32(numDocs),
                    dim: UInt32(dim),
                    normalizeInputs: 0
                )

                enc.setComputePipelineState(pso)
                enc.setBuffer(qBuf, offset: 0, index: 0)
                enc.setBuffer(corpusBuf, offset: 0, index: 1)
                enc.setBuffer(outBuf, offset: 0, index: 2)
                enc.setBytes(&params, length: MemoryLayout<SimilarityParams>.stride, index: 3)

                dispatchSimdgroups(encoder: enc, pso: pso, cols: numDocs, rows: numQueries)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                if cmd.status != .completed {
                    throw MetalEngineError.commandBufferFailed(cmd.error?.localizedDescription ?? "Metal command buffer status=\(cmd.status.rawValue)")
                }

                let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
                return Array(UnsafeBufferPointer(start: ptr, count: outCount))
            }

            // CPU fallback
            var scores = [Float](repeating: 0, count: numQueries * numDocs)
            let cPtr = corpus.pointer
            for q in 0..<numQueries {
                let qOff = q * dim
                for d in 0..<numDocs {
                    let dOff = d * dim
                    var acc: Float = 0
                    for i in 0..<dim {
                        acc += Float(queries[qOff + i]) * Float(cPtr[dOff + i])
                    }
                    scores[q * numDocs + d] = acc
                }
            }
            return scores
        }
    }

    /// Computes `[numQueries x numDocs]` similarity scores against an int8-quantized unified-memory corpus.
    public func batchSimilarityInt8(
        queries: [Float],
        numQueries: Int,
        corpusQuantized: SharedSlab<Int8>,
        corpusScales: SharedSlab<Float>,
        numDocs: Int,
        dim: Int
    ) throws -> [Float] {
        guard numQueries > 0, numDocs > 0, dim > 0 else { return [] }
        guard queries.count == numQueries * dim else {
            throw MetalEngineError.invalidDimensions("queries.count (\(queries.count)) != \(numQueries * dim)")
        }
        guard corpusQuantized.count >= numDocs * dim, corpusScales.count >= numDocs else {
            throw MetalEngineError.invalidDimensions("int8 corpus or scale slab too small")
        }

        return try dispatchQueue.sync {
            if activeBackend == .metal,
               let dev = device,
               let queue = commandQueue,
               let pso = psoDotInt8,
               let corpusBuf = corpusQuantized.metalBuffer,
               let scaleBuf = corpusScales.metalBuffer
            {
                let qBytes = queries.count * MemoryLayout<Float>.stride
                let outCount = numQueries * numDocs
                let outBytes = outCount * MemoryLayout<Float>.stride
                guard let qBuf = queries.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: qBytes, options: .storageModeShared) }),
                      let outBuf = dev.makeBuffer(length: outBytes, options: .storageModeShared),
                      let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder()
                else {
                    throw MetalEngineError.commandBufferFailed("Failed to allocate buffers or encoder for int8 similarity.")
                }

                var params = SimilarityParams(
                    numQueries: UInt32(numQueries),
                    numDocs: UInt32(numDocs),
                    dim: UInt32(dim),
                    normalizeInputs: 0
                )

                enc.setComputePipelineState(pso)
                enc.setBuffer(qBuf, offset: 0, index: 0)
                enc.setBuffer(corpusBuf, offset: 0, index: 1)
                enc.setBuffer(scaleBuf, offset: 0, index: 2)
                enc.setBuffer(outBuf, offset: 0, index: 3)
                enc.setBytes(&params, length: MemoryLayout<SimilarityParams>.stride, index: 4)

                dispatchSimdgroups(encoder: enc, pso: pso, cols: numDocs, rows: numQueries)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                if cmd.status != .completed {
                    throw MetalEngineError.commandBufferFailed(cmd.error?.localizedDescription ?? "Metal command buffer status=\(cmd.status.rawValue)")
                }

                let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
                return Array(UnsafeBufferPointer(start: ptr, count: outCount))
            }

            // CPU fallback
            var scores = [Float](repeating: 0, count: numQueries * numDocs)
            let cPtr = corpusQuantized.pointer
            let sPtr = corpusScales.pointer
            for q in 0..<numQueries {
                let qOff = q * dim
                for d in 0..<numDocs {
                    let dOff = d * dim
                    var acc: Float = 0
                    for i in 0..<dim {
                        acc += queries[qOff + i] * Float(cPtr[dOff + i])
                    }
                    scores[q * numDocs + d] = acc * sPtr[d]
                }
            }
            return scores
        }
    }

    // MARK: - Semantic Pub/Sub Routing Kernel

    /// Evaluates `[numMessages x numSubscribers]` message embeddings against subscriber intent embeddings and thresholds.
    public func semanticRoute(
        messages: [Float],
        numMessages: Int,
        subscribers: SharedSlab<Float>,
        thresholds: SharedSlab<Float>,
        numSubscribers: Int,
        dim: Int
    ) throws -> (scores: [Float], matches: [UInt8]) {
        guard numMessages > 0, numSubscribers > 0, dim > 0 else { return ([], []) }
        guard messages.count == numMessages * dim else {
            throw MetalEngineError.invalidDimensions("messages.count mismatch")
        }

        return try dispatchQueue.sync {
            let total = numMessages * numSubscribers
            if activeBackend == .metal,
               let dev = device,
               let queue = commandQueue,
               let pso = psoRoute,
               let subBuf = subscribers.metalBuffer,
               let thrBuf = thresholds.metalBuffer
            {
                let mBytes = messages.count * MemoryLayout<Float>.stride
                guard let mBuf = messages.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: mBytes, options: .storageModeShared) }),
                      let scoreBuf = dev.makeBuffer(length: total * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let matchBuf = dev.makeBuffer(length: total * MemoryLayout<UInt8>.stride, options: .storageModeShared),
                      let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder()
                else {
                    throw MetalEngineError.commandBufferFailed("Failed to allocate buffers for semantic routing.")
                }

                var params = RouteParams(
                    numMessages: UInt32(numMessages),
                    numSubscribers: UInt32(numSubscribers),
                    dim: UInt32(dim)
                )

                enc.setComputePipelineState(pso)
                enc.setBuffer(mBuf, offset: 0, index: 0)
                enc.setBuffer(subBuf, offset: 0, index: 1)
                enc.setBuffer(thrBuf, offset: 0, index: 2)
                enc.setBuffer(scoreBuf, offset: 0, index: 3)
                enc.setBuffer(matchBuf, offset: 0, index: 4)
                enc.setBytes(&params, length: MemoryLayout<RouteParams>.stride, index: 5)

                dispatchSimdgroups(encoder: enc, pso: pso, cols: numSubscribers, rows: numMessages)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                if cmd.status != .completed {
                    throw MetalEngineError.commandBufferFailed(cmd.error?.localizedDescription ?? "Metal command buffer status=\(cmd.status.rawValue)")
                }

                let sPtr = scoreBuf.contents().bindMemory(to: Float.self, capacity: total)
                let mPtr = matchBuf.contents().bindMemory(to: UInt8.self, capacity: total)
                return (
                    Array(UnsafeBufferPointer(start: sPtr, count: total)),
                    Array(UnsafeBufferPointer(start: mPtr, count: total))
                )
            }

            // CPU Accelerate fallback
            var scores = [Float](repeating: 0, count: total)
            var matches = [UInt8](repeating: 0, count: total)
            let sPtr = subscribers.pointer
            let tPtr = thresholds.pointer
            messages.withUnsafeBufferPointer { mBuf in
                let mBase = mBuf.baseAddress!
                for m in 0..<numMessages {
                    let mVec = mBase.advanced(by: m * dim)
                    for s in 0..<numSubscribers {
                        let score = VectorMath.dot(mVec, sPtr.advanced(by: s * dim), count: dim)
                        let idx = m * numSubscribers + s
                        scores[idx] = score
                        matches[idx] = score >= tPtr[s] ? 1 : 0
                    }
                }
            }
            return (scores, matches)
        }
    }

    // MARK: - KV-Prefix Affinity Kernel

    /// Computes longest common prefix chunk counts `[numQueries x numWorkers]`.
    public func prefixOverlap(
        queryHashes: [UInt64],
        queryLens: [UInt32],
        workerHashes: [UInt64],
        workerLens: [UInt32],
        stride: Int
    ) throws -> [UInt32] {
        let numQueries = queryLens.count
        let numWorkers = workerLens.count
        guard numQueries > 0, numWorkers > 0, stride > 0 else { return [] }
        guard queryHashes.count == numQueries * stride, workerHashes.count == numWorkers * stride else {
            throw MetalEngineError.invalidDimensions("prefix hash buffer dimensions mismatch")
        }

        return try dispatchQueue.sync {
            let total = numQueries * numWorkers
            if activeBackend == .metal,
               let dev = device,
               let queue = commandQueue,
               let pso = psoPrefix
            {
                guard let qhBuf = queryHashes.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: queryHashes.count * MemoryLayout<UInt64>.stride, options: .storageModeShared) }),
                      let qlBuf = queryLens.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: queryLens.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) }),
                      let whBuf = workerHashes.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: workerHashes.count * MemoryLayout<UInt64>.stride, options: .storageModeShared) }),
                      let wlBuf = workerLens.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: workerLens.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) }),
                      let outBuf = dev.makeBuffer(length: total * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                      let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder()
                else {
                    throw MetalEngineError.commandBufferFailed("Failed to allocate buffers for prefix overlap.")
                }

                var params = PrefixParams(
                    numQueries: UInt32(numQueries),
                    numWorkers: UInt32(numWorkers),
                    stride: UInt32(stride)
                )

                enc.setComputePipelineState(pso)
                enc.setBuffer(qhBuf, offset: 0, index: 0)
                enc.setBuffer(qlBuf, offset: 0, index: 1)
                enc.setBuffer(whBuf, offset: 0, index: 2)
                enc.setBuffer(wlBuf, offset: 0, index: 3)
                enc.setBuffer(outBuf, offset: 0, index: 4)
                enc.setBytes(&params, length: MemoryLayout<PrefixParams>.stride, index: 5)

                let tgW = min(16, numWorkers)
                let tgH = min(16, numQueries)
                let grid = MTLSize(width: numWorkers, height: numQueries, depth: 1)
                let tg = MTLSize(width: tgW, height: tgH, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                if cmd.status != .completed {
                    throw MetalEngineError.commandBufferFailed(cmd.error?.localizedDescription ?? "Metal command buffer status=\(cmd.status.rawValue)")
                }

                let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: total)
                return Array(UnsafeBufferPointer(start: ptr, count: total))
            }

            // CPU fallback
            var out = [UInt32](repeating: 0, count: total)
            for q in 0..<numQueries {
                let qOff = q * stride
                let qLen = Int(queryLens[q])
                for w in 0..<numWorkers {
                    let wOff = w * stride
                    let limit = min(min(qLen, Int(workerLens[w])), stride)
                    var matched: UInt32 = 0
                    for i in 0..<limit {
                        if queryHashes[qOff + i] != workerHashes[wOff + i] {
                            break
                        }
                        matched += 1
                    }
                    out[q * numWorkers + w] = matched
                }
            }
            return out
        }
    }

    // MARK: - Dispatch Helper

    private func dispatchSimdgroups(
        encoder: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        cols: Int,
        rows: Int
    ) {
        let simdWidth = max(pso.threadExecutionWidth, 32)
        let maxThreads = min(256, pso.maxTotalThreadsPerThreadgroup)
        let simdgroupsPerTG = max(1, maxThreads / simdWidth)
        let tgWidth = simdgroupsPerTG * simdWidth
        let numTGX = (cols + simdgroupsPerTG - 1) / simdgroupsPerTG

        encoder.dispatchThreadgroups(
            MTLSize(width: numTGX, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
        )
    }
}
