import Testing
@testable import MetalAgentBus
@testable import MetalAgentBusCore
@testable import MetalAgentBusC

private func deterministicVectors(count: Int, dim: Int, seed: UInt64) -> [Float] {
    var state = seed
    func nextFloat() -> Float {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z = z ^ (z >> 31)
        return Float(Double(z >> 11) / Double(1 << 53) * 2.0 - 1.0)
    }
    var out = [Float](repeating: 0, count: count * dim)
    for i in 0..<count {
        var v = [Float](repeating: 0, count: dim)
        for d in 0..<dim { v[d] = nextFloat() }
        let n = VectorMath.normalize(v)
        for d in 0..<dim { out[i * dim + d] = n[d] }
    }
    return out
}

@Test("Runtime MSL compiles and activates Metal GPU on Apple Silicon")
func metalCompilesAtRuntime() throws {
    let engine = try MetalComputeEngine(preferCPU: false)
    #expect(engine.activeBackend == .metal)
    #expect(engine.hasUnifiedMemory)
}

@Test("GPU vs CPU parity across fp32, fp16, int8, unnormalized cosine, and odd dimensions")
func similarityParityAcrossPrecisionsAndDimensions() throws {
    let gpu = try MetalComputeEngine(preferCPU: false)
    let cpu = try MetalComputeEngine(preferCPU: true)

    for dim in [128, 384, 387] {
        let numDocs = 257
        let numQueries = 7
        let corpusRaw = deterministicVectors(count: numDocs, dim: dim, seed: UInt64(dim * 13))
        let queriesRaw = deterministicVectors(count: numQueries, dim: dim, seed: UInt64(dim * 29))

        // FP32 with forced slab reallocation
        let slabF32 = SharedSlab<Float>(device: gpu.device, initialCapacity: 64)
        slabF32.append(contentsOf: corpusRaw)
        #expect(slabF32.generation > 1)

        let cpuF32 = try cpu.batchSimilarityF32(queries: queriesRaw, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim)
        let gpuF32 = try gpu.batchSimilarityF32(queries: queriesRaw, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim)
        let maxErrF32 = zip(cpuF32, gpuF32).map { abs($0 - $1) }.max() ?? 0
        #expect(maxErrF32 < 1e-4)

        // Unnormalized cosine mode
        let cpuNorm = try cpu.batchSimilarityF32(queries: queriesRaw.map { $0 * 3.5 }, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim, normalizeInputs: true)
        let gpuNorm = try gpu.batchSimilarityF32(queries: queriesRaw.map { $0 * 3.5 }, numQueries: numQueries, corpus: slabF32, numDocs: numDocs, dim: dim, normalizeInputs: true)
        let maxErrNorm = zip(cpuNorm, gpuNorm).map { abs($0 - $1) }.max() ?? 0
        #expect(maxErrNorm < 1e-4)

        // FP16
        let slabF16 = SharedSlab<Float16>(device: gpu.device, initialCapacity: corpusRaw.count)
        slabF16.append(contentsOf: VectorMath.toFloat16(corpusRaw))
        let gpuF16 = try gpu.batchSimilarityF16(queries: VectorMath.toFloat16(queriesRaw), numQueries: numQueries, corpus: slabF16, numDocs: numDocs, dim: dim)
        let maxErrF16 = zip(cpuF32, gpuF16).map { abs($0 - $1) }.max() ?? 0
        #expect(maxErrF16 < 5e-3)

        // INT8
        let slabI8 = SharedSlab<Int8>(device: gpu.device, initialCapacity: corpusRaw.count)
        let slabScales = SharedSlab<Float>(device: gpu.device, initialCapacity: numDocs)
        for d in 0..<numDocs {
            let slice = Array(corpusRaw[(d * dim)..<((d + 1) * dim)])
            let (q, s) = VectorMath.quantizeInt8(slice)
            slabI8.append(contentsOf: q)
            slabScales.append(s)
        }
        let gpuI8 = try gpu.batchSimilarityInt8(queries: queriesRaw, numQueries: numQueries, corpusQuantized: slabI8, corpusScales: slabScales, numDocs: numDocs, dim: dim)
        let maxErrI8 = zip(cpuF32, gpuI8).map { abs($0 - $1) }.max() ?? 0
        #expect(maxErrI8 < 2e-2)
    }
}

@Test("UnifiedMemoryStore supports search, MMR diversity, token budget, and tombstones")
func unifiedMemoryStoreBehavior() async throws {
    let engine = try MetalComputeEngine(preferCPU: false)
    let store = UnifiedMemoryStore(dimension: 4, precision: .float32, initialCapacity: 4, engine: engine)

    try await store.append(record: MemoryRecord(id: "m1", agentID: "planner", content: "axis-0 primary", tokenCount: 100), embedding: [0.94, 0.34, 0, 0])
    try await store.append(record: MemoryRecord(id: "m2", agentID: "planner", content: "axis-0 duplicate", tokenCount: 100), embedding: [0.95, 0.31, 0, 0])
    try await store.append(record: MemoryRecord(id: "m3", agentID: "coder", content: "axis-1 diverse relevant", tokenCount: 100), embedding: [0.25, 0.97, 0, 0])
    try await store.append(record: MemoryRecord(id: "m4", agentID: "reviewer", content: "orthogonal", tokenCount: 100), embedding: [0, 0, 1, 0])

    let query: [Float] = [0.707, 0.707, 0, 0]
    let plain = try await store.search(queries: [query], options: SearchOptions(topK: 2))
    #expect(plain[0].map(\.record.id) == ["m1", "m2"])

    let diverse = try await store.search(queries: [query], options: SearchOptions(topK: 2, mmrLambda: 0.5))
    #expect(diverse[0].map(\.record.id) == ["m1", "m3"])

    let budgeted = try await store.search(queries: [query], options: SearchOptions(topK: 4, tokenBudget: 150))
    #expect(budgeted[0].count == 1)
    #expect(budgeted[0][0].record.id == "m1")

    let removed = await store.tombstone(id: "m1")
    #expect(removed)
    #expect(await store.activeCount == 3)
    let afterDelete = try await store.search(queries: [query], options: SearchOptions(topK: 2))
    #expect(afterDelete[0].first?.record.id == "m2")
}

@Test("Semantic AgentBus fans out messages over Metal GPU to matching AsyncStreams")
func semanticAgentBusFanOut() async throws {
    let engine = try MetalComputeEngine(preferCPU: false)
    let bus = AgentBus(dimension: 4, engine: engine)

    let coderStream = try await bus.subscribe(agentID: "coder", intentEmbedding: [1, 0, 0, 0], threshold: 0.8)
    let qaStream = try await bus.subscribe(agentID: "qa", intentEmbedding: [0, 1, 0, 0], threshold: 0.8)

    let msg = BusMessage(id: "msg-1", senderID: "planner", payload: "Implement Metal kernel", embedding: [0.98, 0.2, 0, 0])
    let delivered = try await bus.publish(msg)

    #expect(delivered.count == 1)
    #expect(delivered[0].recipientID == "coder")
    #expect(delivered[0].similarity > 0.95)

    await bus.unsubscribe(agentID: "coder")
    await bus.unsubscribe(agentID: "qa")

    var coderReceived: [RoutedMessage] = []
    for await item in coderStream { coderReceived.append(item) }
    var qaReceived: [RoutedMessage] = []
    for await item in qaStream { qaReceived.append(item) }

    #expect(coderReceived.count == 1)
    #expect(coderReceived[0].message.id == "msg-1")
    #expect(qaReceived.isEmpty)
}

@Test("PrefixAffinityRouter maximizes KV-prefix overlap and respects worker saturation")
func prefixAffinityRouting() async throws {
    let engine = try MetalComputeEngine(preferCPU: false)
    let router = PrefixAffinityRouter(chunkSize: 4, maxChunks: 64, engine: engine)

    let sysPromptA: [UInt32] = Array(1...32)
    let sysPromptB: [UInt32] = Array(101...132)

    await router.registerWorker(workerID: "mlx-slot-0", warmTokens: sysPromptA, inFlight: 0, maxConcurrency: 1)
    await router.registerWorker(workerID: "mlx-slot-1", warmTokens: sysPromptB, inFlight: 0, maxConcurrency: 2)

    let incoming1: [UInt32] = Array(1...24) + [999, 998, 997, 996]
    let incoming2: [UInt32] = Array(1...24) + [888, 887, 886, 885]

    let decisions = try await router.route(prompts: [incoming1, incoming2], loadPenaltyChunks: 2, reserveLoad: true)
    #expect(decisions[0]?.workerID == "mlx-slot-0")
    #expect(decisions[0]?.matchedChunks == 6)
    #expect(decisions[0]?.matchedTokens == 24)
    #expect(decisions[1]?.workerID == "mlx-slot-1")
}

@Test("C ABI roundtrip creates, appends, searches, and destroys a unified-memory store")
func cABIRoundTrip() throws {
    guard let store = mab_store_create(4, 16) else {
        Issue.record("mab_store_create returned nil")
        return
    }
    defer { mab_store_destroy(store) }

    let v0: [Float] = [1, 0, 0, 0]
    let v1: [Float] = [0, 1, 0, 0]
    #expect(v0.withUnsafeBufferPointer { mab_store_append_f32(store, $0.baseAddress, 4) } == 0)
    #expect(v1.withUnsafeBufferPointer { mab_store_append_f32(store, $0.baseAddress, 4) } == 1)
    #expect(mab_store_count(store) == 2)

    let q: [Float] = [0.95, 0.31, 0, 0]
    var indices = [Int32](repeating: -1, count: 2)
    var scores = [Float](repeating: 0, count: 2)
    let n = q.withUnsafeBufferPointer { qp in
        indices.withUnsafeMutableBufferPointer { ip in
            scores.withUnsafeMutableBufferPointer { sp in
                mab_store_search_f32(store, qp.baseAddress, 4, 2, ip.baseAddress, sp.baseAddress)
            }
        }
    }
    #expect(n == 2)
    #expect(indices[0] == 0)
    #expect(scores[0] > 0.9)
}
