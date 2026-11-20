import Foundation
import MetalAgentBus
import MetalAgentBusCore

private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
    mutating func nextFloat() -> Float {
        let u = Double(next() >> 11) * (1.0 / Double(1 << 53))
        return Float(u * 2.0 - 1.0)
    }
}

private func makeNormalizedVectors(count: Int, dim: Int, seed: UInt64) -> [Float] {
    var rng = SplitMix64(seed: seed)
    var flat = [Float](repeating: 0, count: count * dim)
    for i in 0..<count {
        var vec = [Float](repeating: 0, count: dim)
        for d in 0..<dim { vec[d] = rng.nextFloat() }
        let norm = VectorMath.normalize(vec)
        let off = i * dim
        for d in 0..<dim { flat[off + d] = norm[d] }
    }
    return flat
}

private func measureMs(warmup: Int = 3, iterations: Int = 15, _ body: () throws -> Void) throws -> (median: Double, p95: Double) {
    for _ in 0..<warmup { try body() }
    var samples: [Double] = []
    samples.reserveCapacity(iterations)
    let clock = ContinuousClock()
    for _ in 0..<iterations {
        let elapsed = try clock.measure { try body() }
        let c = elapsed.components
        let ms = Double(c.seconds) * 1_000.0 + Double(c.attoseconds) / 1e15
        samples.append(ms)
    }
    samples.sort()
    let median = samples[samples.count / 2]
    let p95 = samples[min(samples.count - 1, Int(Double(samples.count) * 0.95))]
    return (median, p95)
}

private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var m: Float = 0
    for i in 0..<a.count {
        let d = abs(a[i] - b[i])
        if d > m { m = d }
    }
    return m
}

private func col(_ text: String, _ width: Int, right: Bool = false) -> String {
    if text.count >= width { return text }
    let pad = String(repeating: " ", count: width - text.count)
    return right ? (pad + text) : (text + pad)
}

private func fmt(_ value: Double, decimals: Int) -> String {
    String(format: "%.\(decimals)f", value)
}

private func fmtSci(_ value: Float) -> String {
    String(format: "%.2e", Double(value))
}

@main
struct MetalAgentBusBench {
    static func main() throws {
        let gpu = try MetalComputeEngine(preferCPU: false)
        let cpu = try MetalComputeEngine(preferCPU: true)

        print("metal-agent-bus benchmark")
        print("Device: \(gpu.deviceName) | Unified Memory: \(gpu.hasUnifiedMemory) | Backend: \(gpu.activeBackend.rawValue)")
        print(String(repeating: "-", count: 96))
        print("\(col("Workload", 34)) \(col("CPU (ms)", 10, right: true)) \(col("GPU (ms)", 10, right: true)) \(col("Speedup", 9, right: true)) \(col("GB/s", 10, right: true)) \(col("MaxErr", 11, right: true))")
        print(String(repeating: "-", count: 96))

        let dim = 1536
        let numDocs = 32_768
        let numQueries = 32

        let corpusF32Raw = makeNormalizedVectors(count: numDocs, dim: dim, seed: 42)
        let queriesF32 = makeNormalizedVectors(count: numQueries, dim: dim, seed: 99)

        // 1. FP32 Batched Similarity
        let slabF32 = SharedSlab<Float>(device: gpu.device, initialCapacity: corpusF32Raw.count)
        slabF32.append(contentsOf: corpusF32Raw)

        let cpuF32Scores = try cpu.batchSimilarityF32(queries: queriesF32, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim)
        let gpuF32Scores = try gpu.batchSimilarityF32(queries: queriesF32, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim)
        let errF32 = maxAbsDiff(cpuF32Scores, gpuF32Scores)

        let cpuF32T = try measureMs(warmup: 1, iterations: 5) {
            _ = try cpu.batchSimilarityF32(queries: queriesF32, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim)
        }
        let gpuF32T = try measureMs(warmup: 3, iterations: 15) {
            _ = try gpu.batchSimilarityF32(queries: queriesF32, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim)
        }
        let bytesF32 = Double(numDocs * dim * 4 + numQueries * dim * 4 + numQueries * numDocs * 4)
        let gbsF32 = (bytesF32 / 1e9) / (gpuF32T.median / 1000.0)
        print("\(col("Cosine f32 (32q x 32k x 1536d)", 34)) \(col(fmt(cpuF32T.median, decimals: 2), 10, right: true)) \(col(fmt(gpuF32T.median, decimals: 2), 10, right: true)) \(col(fmt(cpuF32T.median / gpuF32T.median, decimals: 1) + "x", 9, right: true)) \(col(fmt(gbsF32, decimals: 1), 10, right: true)) \(col(fmtSci(errF32), 11, right: true))")

        // 2. FP16 Batched Similarity
        let corpusF16Raw = VectorMath.toFloat16(corpusF32Raw)
        let queriesF16 = VectorMath.toFloat16(queriesF32)
        let slabF16 = SharedSlab<Float16>(device: gpu.device, initialCapacity: corpusF16Raw.count)
        slabF16.append(contentsOf: corpusF16Raw)

        let gpuF16Scores = try gpu.batchSimilarityF16(queries: queriesF16, numQueries: numQueries, corpus: slabF16, numDocs: numDocs, dim: dim)
        let errF16 = maxAbsDiff(cpuF32Scores, gpuF16Scores)

        let gpuF16T = try measureMs(warmup: 3, iterations: 15) {
            _ = try gpu.batchSimilarityF16(queries: queriesF16, numQueries: numQueries, corpus: slabF16, numDocs: numDocs, dim: dim)
        }
        let bytesF16 = Double(numDocs * dim * 2 + numQueries * dim * 2 + numQueries * numDocs * 4)
        let gbsF16 = (bytesF16 / 1e9) / (gpuF16T.median / 1000.0)
        print("\(col("Cosine f16 (32q x 32k x 1536d)", 34)) \(col(fmt(cpuF32T.median, decimals: 2), 10, right: true)) \(col(fmt(gpuF16T.median, decimals: 2), 10, right: true)) \(col(fmt(cpuF32T.median / gpuF16T.median, decimals: 1) + "x", 9, right: true)) \(col(fmt(gbsF16, decimals: 1), 10, right: true)) \(col(fmtSci(errF16), 11, right: true))")

        // 3. INT8 Quantized Batched Similarity
        let slabI8 = SharedSlab<Int8>(device: gpu.device, initialCapacity: numDocs * dim)
        let slabScales = SharedSlab<Float>(device: gpu.device, initialCapacity: numDocs)
        for d in 0..<numDocs {
            let slice = Array(corpusF32Raw[(d * dim)..<((d + 1) * dim)])
            let (q, s) = VectorMath.quantizeInt8(slice)
            slabI8.append(contentsOf: q)
            slabScales.append(s)
        }
        let gpuI8Scores = try gpu.batchSimilarityInt8(queries: queriesF32, numQueries: numQueries, corpusQuantized: slabI8, corpusScales: slabScales, numDocs: numDocs, dim: dim)
        let errI8 = maxAbsDiff(cpuF32Scores, gpuI8Scores)

        let gpuI8T = try measureMs(warmup: 3, iterations: 15) {
            _ = try gpu.batchSimilarityInt8(queries: queriesF32, numQueries: numQueries, corpusQuantized: slabI8, corpusScales: slabScales, numDocs: numDocs, dim: dim)
        }
        let bytesI8 = Double(numDocs * dim + numDocs * 4 + numQueries * dim * 4 + numQueries * numDocs * 4)
        let gbsI8 = (bytesI8 / 1e9) / (gpuI8T.median / 1000.0)
        print("\(col("Dot int8   (32q x 32k x 1536d)", 34)) \(col(fmt(cpuF32T.median, decimals: 2), 10, right: true)) \(col(fmt(gpuI8T.median, decimals: 2), 10, right: true)) \(col(fmt(cpuF32T.median / gpuI8T.median, decimals: 1) + "x", 9, right: true)) \(col(fmt(gbsI8, decimals: 1), 10, right: true)) \(col(fmtSci(errI8), 11, right: true))")

        // 4. Semantic Pub/Sub Routing (64 messages x 4,096 agent intent subscriptions)
        let numMsgs = 64
        let numSubs = 4_096
        let msgs = makeNormalizedVectors(count: numMsgs, dim: dim, seed: 101)
        let subs = makeNormalizedVectors(count: numSubs, dim: dim, seed: 202)
        let subSlab = SharedSlab<Float>(device: gpu.device, initialCapacity: subs.count)
        subSlab.append(contentsOf: subs)
        let thrSlab = SharedSlab<Float>(device: gpu.device, initialCapacity: numSubs)
        thrSlab.append(contentsOf: [Float](repeating: 0.5, count: numSubs))

        let cpuRoute = try cpu.semanticRoute(messages: msgs, numMessages: numMsgs, subscribers: subSlab, thresholds: thrSlab, numSubscribers: numSubs, dim: dim)
        let gpuRoute = try gpu.semanticRoute(messages: msgs, numMessages: numMsgs, subscribers: subSlab, thresholds: thrSlab, numSubscribers: numSubs, dim: dim)
        let errRoute = maxAbsDiff(cpuRoute.scores, gpuRoute.scores)

        let cpuRouteT = try measureMs(warmup: 1, iterations: 5) {
            _ = try cpu.semanticRoute(messages: msgs, numMessages: numMsgs, subscribers: subSlab, thresholds: thrSlab, numSubscribers: numSubs, dim: dim)
        }
        let gpuRouteT = try measureMs(warmup: 3, iterations: 15) {
            _ = try gpu.semanticRoute(messages: msgs, numMessages: numMsgs, subscribers: subSlab, thresholds: thrSlab, numSubscribers: numSubs, dim: dim)
        }
        let bytesRoute = Double((numMsgs + numSubs) * dim * 4 + numMsgs * numSubs * 5)
        let gbsRoute = (bytesRoute / 1e9) / (gpuRouteT.median / 1000.0)
        print("\(col("Route f32  (64m x 4k subs x 1536d)", 34)) \(col(fmt(cpuRouteT.median, decimals: 2), 10, right: true)) \(col(fmt(gpuRouteT.median, decimals: 2), 10, right: true)) \(col(fmt(cpuRouteT.median / gpuRouteT.median, decimals: 1) + "x", 9, right: true)) \(col(fmt(gbsRoute, decimals: 1), 10, right: true)) \(col(fmtSci(errRoute), 11, right: true))")

        // 5. Prefix-Affinity Routing (128 prompts x 256 warm worker slots x 256 chunks)
        let pQ = 128
        let pW = 256
        let pStride = 256
        var rng = SplitMix64(seed: 777)
        let sharedBase = (0..<128).map { _ in rng.next() }
        var qHashes = [UInt64](repeating: 0, count: pQ * pStride)
        var wHashes = [UInt64](repeating: 0, count: pW * pStride)
        let qLens = [UInt32](repeating: UInt32(pStride), count: pQ)
        let wLens = [UInt32](repeating: UInt32(pStride), count: pW)
        for q in 0..<pQ {
            let matchLen = (q * 7) % 128
            for i in 0..<pStride {
                qHashes[q * pStride + i] = i < matchLen ? sharedBase[i] : rng.next()
            }
        }
        for w in 0..<pW {
            let matchLen = (w * 5) % 128
            for i in 0..<pStride {
                wHashes[w * pStride + i] = i < matchLen ? sharedBase[i] : rng.next()
            }
        }

        let cpuPref = try cpu.prefixOverlap(queryHashes: qHashes, queryLens: qLens, workerHashes: wHashes, workerLens: wLens, stride: pStride)
        let gpuPref = try gpu.prefixOverlap(queryHashes: qHashes, queryLens: qLens, workerHashes: wHashes, workerLens: wLens, stride: pStride)
        let prefMismatch = zip(cpuPref, gpuPref).filter { $0 != $1 }.count

        let cpuPrefT = try measureMs(warmup: 2, iterations: 15) {
            _ = try cpu.prefixOverlap(queryHashes: qHashes, queryLens: qLens, workerHashes: wHashes, workerLens: wLens, stride: pStride)
        }
        let gpuPrefT = try measureMs(warmup: 3, iterations: 15) {
            _ = try gpu.prefixOverlap(queryHashes: qHashes, queryLens: qLens, workerHashes: wHashes, workerLens: wLens, stride: pStride)
        }
        print("\(col("Prefix LCP (128q x 256w x 256c)", 34)) \(col(fmt(cpuPrefT.median, decimals: 2), 10, right: true)) \(col(fmt(gpuPrefT.median, decimals: 2), 10, right: true)) \(col(fmt(cpuPrefT.median / gpuPrefT.median, decimals: 1) + "x", 9, right: true)) \(col("-", 10, right: true)) \(col("\(prefMismatch) diff", 11, right: true))")
        print(String(repeating: "-", count: 96))
    }
}
