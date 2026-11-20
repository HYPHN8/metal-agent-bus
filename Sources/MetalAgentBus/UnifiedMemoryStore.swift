import Foundation
import MetalAgentBusCore

/// A single episodic memory entry stored in the unified-memory multi-agent store.
public struct MemoryRecord: Sendable, Equatable {
    public let id: String
    public let agentID: String
    public let content: String
    public let tokenCount: Int
    public let createdAt: Date
    public let metadata: [String: String]

    public init(
        id: String,
        agentID: String,
        content: String,
        tokenCount: Int,
        createdAt: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.agentID = agentID
        self.content = content
        self.tokenCount = max(0, tokenCount)
        self.createdAt = createdAt
        self.metadata = metadata
    }
}

/// Ranked retrieval result returned by `UnifiedMemoryStore`.
public struct MemoryHit: Sendable, Equatable {
    public let record: MemoryRecord
    public let score: Float

    public init(record: MemoryRecord, score: Float) {
        self.record = record
        self.score = score
    }
}

/// Search options supporting agent filtering, Maximal Marginal Relevance (MMR) diversity,
/// and strict context-window token budgeting.
public struct SearchOptions: Sendable {
    public let topK: Int
    public let filterAgentIDs: Set<String>?
    public let minScore: Float?
    /// When set in `(0, 1)`, enables MMR diversity re-ranking (`1.0` = pure relevance, `0.0` = pure diversity).
    public let mmrLambda: Float?
    /// Candidate pool multiplier used prior to MMR / token-budget selection.
    public let mmrCandidateMultiplier: Int
    /// Optional token-budget cap: stops packing hits once adding the next hit would exceed the budget.
    public let tokenBudget: Int?

    public init(
        topK: Int = 8,
        filterAgentIDs: Set<String>? = nil,
        minScore: Float? = nil,
        mmrLambda: Float? = nil,
        mmrCandidateMultiplier: Int = 4,
        tokenBudget: Int? = nil
    ) {
        self.topK = max(1, topK)
        self.filterAgentIDs = filterAgentIDs
        self.minScore = minScore
        self.mmrLambda = mmrLambda
        self.mmrCandidateMultiplier = max(1, mmrCandidateMultiplier)
        self.tokenBudget = tokenBudget
    }
}

public enum StoreError: Error, LocalizedError, Sendable {
    case dimensionMismatch(expected: Int, got: Int)

    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let expected, let got):
            return "Embedding dimension mismatch: expected \(expected), got \(got)."
        }
    }
}

/// Zero-copy unified-memory episodic store shared across concurrent Swift actors and Metal compute shaders.
public actor UnifiedMemoryStore {
    public let dimension: Int
    public let precision: VectorPrecision
    public let engine: MetalComputeEngine

    private var records: [MemoryRecord] = []
    private var activeMask: [Bool] = []
    private var idToSlot: [String: Int] = [:]

    private let slabF32: SharedSlab<Float>?
    private let slabF16: SharedSlab<Float16>?
    private let slabInt8: SharedSlab<Int8>?
    private let slabInt8Scales: SharedSlab<Float>?

    public init(
        dimension: Int,
        precision: VectorPrecision = .float16,
        initialCapacity: Int = 1024,
        engine: MetalComputeEngine
    ) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
        self.precision = precision
        self.engine = engine

        let elemCap = max(initialCapacity, 64) * dimension
        switch precision {
        case .float32:
            self.slabF32 = SharedSlab<Float>(device: engine.device, initialCapacity: elemCap)
            self.slabF16 = nil
            self.slabInt8 = nil
            self.slabInt8Scales = nil
        case .float16:
            self.slabF32 = nil
            self.slabF16 = SharedSlab<Float16>(device: engine.device, initialCapacity: elemCap)
            self.slabInt8 = nil
            self.slabInt8Scales = nil
        case .int8:
            self.slabF32 = nil
            self.slabF16 = nil
            self.slabInt8 = SharedSlab<Int8>(device: engine.device, initialCapacity: elemCap)
            self.slabInt8Scales = SharedSlab<Float>(device: engine.device, initialCapacity: max(initialCapacity, 64))
        }
    }

    public var totalSlots: Int { records.count }

    public var activeCount: Int {
        activeMask.reduce(0) { $0 + ($1 ? 1 : 0) }
    }

    /// Appends (or upserts by `record.id`, tombstoning the prior slot) a single normalized memory vector.
    public func append(record: MemoryRecord, embedding: [Float]) throws {
        guard embedding.count == dimension else {
            throw StoreError.dimensionMismatch(expected: dimension, got: embedding.count)
        }
        if let oldSlot = idToSlot[record.id] {
            activeMask[oldSlot] = false
        }

        let norm = VectorMath.normalize(embedding)
        switch precision {
        case .float32:
            slabF32?.append(contentsOf: norm)
        case .float16:
            slabF16?.append(contentsOf: VectorMath.toFloat16(norm))
        case .int8:
            let (q, scale) = VectorMath.quantizeInt8(norm)
            slabInt8?.append(contentsOf: q)
            slabInt8Scales?.append(scale)
        }

        let slot = records.count
        records.append(record)
        activeMask.append(true)
        idToSlot[record.id] = slot
    }

    /// Appends a batch of records directly into unified memory.
    public func appendBatch(_ items: [(record: MemoryRecord, embedding: [Float])]) throws {
        for item in items {
            try append(record: item.record, embedding: item.embedding)
        }
    }

    /// Marks an existing memory entry inactive without relocating GPU slab slots.
    @discardableResult
    public func tombstone(id: String) -> Bool {
        guard let slot = idToSlot.removeValue(forKey: id) else { return false }
        activeMask[slot] = false
        return true
    }

    /// Runs a batched multi-query similarity search across the unified-memory slab on Metal GPU.
    public func search(
        queries: [[Float]],
        options: SearchOptions = SearchOptions()
    ) throws -> [[MemoryHit]] {
        guard !queries.isEmpty, !records.isEmpty else {
            return queries.map { _ in [] }
        }

        for q in queries {
            guard q.count == dimension else {
                throw StoreError.dimensionMismatch(expected: dimension, got: q.count)
            }
        }

        let numQueries = queries.count
        let numDocs = records.count

        // Normalize all query embeddings once.
        var flatF32: [Float] = []
        flatF32.reserveCapacity(numQueries * dimension)
        for q in queries {
            flatF32.append(contentsOf: VectorMath.normalize(q))
        }

        let rawScores: [Float]
        switch precision {
        case .float32:
            rawScores = try engine.batchSimilarityF32(
                queries: flatF32,
                numQueries: numQueries,
                corpus: slabF32!,
                numDocs: numDocs,
                dim: dimension,
                normalizeInputs: false
            )
        case .float16:
            let flatF16 = VectorMath.toFloat16(flatF32)
            rawScores = try engine.batchSimilarityF16(
                queries: flatF16,
                numQueries: numQueries,
                corpus: slabF16!,
                numDocs: numDocs,
                dim: dimension
            )
        case .int8:
            rawScores = try engine.batchSimilarityInt8(
                queries: flatF32,
                numQueries: numQueries,
                corpusQuantized: slabInt8!,
                corpusScales: slabInt8Scales!,
                numDocs: numDocs,
                dim: dimension
            )
        }

        var results: [[MemoryHit]] = []
        results.reserveCapacity(numQueries)

        for q in 0..<numQueries {
            let rowOffset = q * numDocs
            var candidates: [(slot: Int, score: Float)] = []
            candidates.reserveCapacity(min(numDocs, options.topK * options.mmrCandidateMultiplier))

            for slot in 0..<numDocs {
                guard activeMask[slot] else { continue }
                if let allowed = options.filterAgentIDs, !allowed.contains(records[slot].agentID) {
                    continue
                }
                let score = rawScores[rowOffset + slot]
                if let minScore = options.minScore, score < minScore {
                    continue
                }
                candidates.append((slot, score))
            }

            candidates.sort { $0.score > $1.score }

            let selectedSlots: [(slot: Int, score: Float)]
            if let lambda = options.mmrLambda, lambda > 0, lambda < 1, candidates.count > 1 {
                let poolSize = min(candidates.count, options.topK * options.mmrCandidateMultiplier)
                let pool = Array(candidates.prefix(poolSize))
                selectedSlots = selectMMR(pool: pool, topK: options.topK, lambda: lambda)
            } else {
                selectedSlots = Array(candidates.prefix(options.topK))
            }

            var hits: [MemoryHit] = []
            hits.reserveCapacity(selectedSlots.count)
            var usedTokens = 0
            for item in selectedSlots {
                let rec = records[item.slot]
                if let budget = options.tokenBudget {
                    if usedTokens + rec.tokenCount > budget {
                        continue
                    }
                    usedTokens += rec.tokenCount
                }
                hits.append(MemoryHit(record: rec, score: item.score))
            }
            results.append(hits)
        }

        return results
    }

    // MARK: - MMR Diversity Re-ranking

    private func selectMMR(
        pool: [(slot: Int, score: Float)],
        topK: Int,
        lambda: Float
    ) -> [(slot: Int, score: Float)] {
        let vectors = pool.map { loadVectorF32(slot: $0.slot) }
        var chosenIndices: [Int] = []
        chosenIndices.reserveCapacity(min(topK, pool.count))
        var remaining = Array(0..<pool.count)

        while chosenIndices.count < topK, !remaining.isEmpty {
            if chosenIndices.isEmpty {
                chosenIndices.append(remaining.removeFirst())
                continue
            }

            var bestPos = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for (pos, candIdx) in remaining.enumerated() {
                var maxSim: Float = -1.0
                for chosenIdx in chosenIndices {
                    let sim = vectors[candIdx].withUnsafeBufferPointer { a in
                        vectors[chosenIdx].withUnsafeBufferPointer { b in
                            VectorMath.dot(a.baseAddress!, b.baseAddress!, count: dimension)
                        }
                    }
                    if sim > maxSim { maxSim = sim }
                }
                let mmr = lambda * pool[candIdx].score - (1.0 - lambda) * maxSim
                if mmr > bestScore {
                    bestScore = mmr
                    bestPos = pos
                }
            }
            chosenIndices.append(remaining.remove(at: bestPos))
        }

        return chosenIndices.map { pool[$0] }
    }

    private func loadVectorF32(slot: Int) -> [Float] {
        let off = slot * dimension
        switch precision {
        case .float32:
            let ptr = slabF32!.pointer.advanced(by: off)
            return Array(UnsafeBufferPointer(start: ptr, count: dimension))
        case .float16:
            let ptr = slabF16!.pointer.advanced(by: off)
            var out = [Float](repeating: 0, count: dimension)
            for i in 0..<dimension { out[i] = Float(ptr[i]) }
            return out
        case .int8:
            let ptr = slabInt8!.pointer.advanced(by: off)
            let scale = slabInt8Scales!.pointer[slot]
            var out = [Float](repeating: 0, count: dimension)
            for i in 0..<dimension { out[i] = Float(ptr[i]) * scale }
            return out
        }
    }
}
