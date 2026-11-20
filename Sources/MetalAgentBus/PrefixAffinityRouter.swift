import Foundation
import MetalAgentBusCore

/// Represents a warm local or remote inference worker slot (MLX, Ollama, llama.cpp, etc.).
public struct WorkerSlot: Sendable, Equatable {
    public let workerID: String
    public var cachedChunkHashes: [UInt64]
    public var inFlight: Int
    public var maxConcurrency: Int

    public init(
        workerID: String,
        cachedChunkHashes: [UInt64] = [],
        inFlight: Int = 0,
        maxConcurrency: Int = 4
    ) {
        self.workerID = workerID
        self.cachedChunkHashes = cachedChunkHashes
        self.inFlight = max(0, inFlight)
        self.maxConcurrency = max(1, maxConcurrency)
    }
}

/// Routing decision produced by `PrefixAffinityRouter`.
public struct RoutingDecision: Sendable, Equatable {
    public let workerID: String
    public let matchedChunks: Int
    public let matchedTokens: Int
    public let workerLoad: Int

    public init(workerID: String, matchedChunks: Int, matchedTokens: Int, workerLoad: Int) {
        self.workerID = workerID
        self.matchedChunks = matchedChunks
        self.matchedTokens = matchedTokens
        self.workerLoad = workerLoad
    }
}

/// Model-agnostic, Metal-accelerated KV-cache-aware prefix-affinity router.
///
/// Tracks warm prompt prefix chunk hashes across local inference workers (MLX, `llama.cpp`, Ollama)
/// and scores incoming agent prompts in parallel on Metal GPU to maximize KV-cache hits while
/// balancing worker concurrency.
public actor PrefixAffinityRouter {
    public let chunkSize: Int
    public let maxChunks: Int
    public let engine: MetalComputeEngine

    private var workers: [WorkerSlot] = []

    public init(
        chunkSize: Int = 16,
        maxChunks: Int = 512,
        engine: MetalComputeEngine
    ) {
        precondition(chunkSize > 0 && maxChunks > 0)
        self.chunkSize = chunkSize
        self.maxChunks = maxChunks
        self.engine = engine
    }

    public var workerCount: Int { workers.count }

    /// Registers or updates a worker slot with its currently warm prompt token sequence.
    public func registerWorker(
        workerID: String,
        warmTokens: [UInt32] = [],
        inFlight: Int = 0,
        maxConcurrency: Int = 4
    ) {
        let hashes = Array(VectorMath.chunkHashes(tokens: warmTokens, chunkSize: chunkSize).prefix(maxChunks))
        let slot = WorkerSlot(
            workerID: workerID,
            cachedChunkHashes: hashes,
            inFlight: inFlight,
            maxConcurrency: maxConcurrency
        )
        if let idx = workers.firstIndex(where: { $0.workerID == workerID }) {
            workers[idx] = slot
        } else {
            workers.append(slot)
        }
    }

    /// Updates a worker's active load count.
    public func setLoad(workerID: String, inFlight: Int) {
        if let idx = workers.firstIndex(where: { $0.workerID == workerID }) {
            workers[idx].inFlight = max(0, inFlight)
        }
    }

    /// Routes a batch of incoming prompt token sequences to the best available workers.
    ///
    /// - Parameter loadPenaltyChunks: Equivalent prefix-chunk penalty per in-flight request when
    ///   trading off KV-prefix reuse against worker queueing load.
    public func route(
        prompts: [[UInt32]],
        loadPenaltyChunks: Int = 2,
        reserveLoad: Bool = false
    ) throws -> [RoutingDecision?] {
        guard !prompts.isEmpty else { return [] }
        guard !workers.isEmpty else { return prompts.map { _ in nil } }

        let qChunks = prompts.map { Array(VectorMath.chunkHashes(tokens: $0, chunkSize: chunkSize).prefix(maxChunks)) }
        let maxQ = qChunks.map(\.count).max() ?? 0
        let maxW = workers.map(\.cachedChunkHashes.count).max() ?? 0
        let stride = max(1, max(maxQ, maxW))

        var flatQ = [UInt64](repeating: 0, count: prompts.count * stride)
        var qLens = [UInt32](repeating: 0, count: prompts.count)
        for (qi, chunks) in qChunks.enumerated() {
            qLens[qi] = UInt32(chunks.count)
            for (ci, h) in chunks.enumerated() {
                flatQ[qi * stride + ci] = h
            }
        }

        var flatW = [UInt64](repeating: 0, count: workers.count * stride)
        var wLens = [UInt32](repeating: 0, count: workers.count)
        for (wi, worker) in workers.enumerated() {
            wLens[wi] = UInt32(worker.cachedChunkHashes.count)
            for (ci, h) in worker.cachedChunkHashes.enumerated() {
                flatW[wi * stride + ci] = h
            }
        }

        let overlaps = try engine.prefixOverlap(
            queryHashes: flatQ,
            queryLens: qLens,
            workerHashes: flatW,
            workerLens: wLens,
            stride: stride
        )

        var decisions: [RoutingDecision?] = []
        decisions.reserveCapacity(prompts.count)
        let numW = workers.count

        for q in 0..<prompts.count {
            var bestIdx: Int?
            var bestUtility = Int.min
            var bestMatched = 0

            for w in 0..<numW {
                let slot = workers[w]
                guard slot.inFlight < slot.maxConcurrency else { continue }
                let matched = Int(overlaps[q * numW + w])
                let utility = matched - slot.inFlight * loadPenaltyChunks
                if utility > bestUtility || (utility == bestUtility && matched > bestMatched) {
                    bestUtility = utility
                    bestMatched = matched
                    bestIdx = w
                }
            }

            if let idx = bestIdx {
                if reserveLoad {
                    workers[idx].inFlight += 1
                    if !qChunks[q].isEmpty && qChunks[q].count >= workers[idx].cachedChunkHashes.count {
                        workers[idx].cachedChunkHashes = qChunks[q]
                    }
                }
                let chosen = workers[idx]
                decisions.append(
                    RoutingDecision(
                        workerID: chosen.workerID,
                        matchedChunks: bestMatched,
                        matchedTokens: bestMatched * chunkSize,
                        workerLoad: chosen.inFlight
                    )
                )
            } else {
                decisions.append(nil)
            }
        }

        return decisions
    }
}
