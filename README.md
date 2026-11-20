# metal-agent-bus

**Zero-copy unified-memory multi-agent semantic bus, retrieval engine, and KV-prefix-affinity router for Apple Silicon.**

Built in native **Swift 6** with **runtime-compiled Metal Shading Language (MSL)** kernels—no Xcode installation, no Python runtime, and no vector-database sidecars required.

## Why `metal-agent-bus`?

Local multi-agent systems on Apple Silicon spend most of their coordination budget copying embeddings across processes, querying heavyweight sidecar vector databases, or routing prompts blindly across warm LLM workers. `metal-agent-bus` sits cleanly **above any inference backend** (MLX, `llama.cpp`, Ollama, or hosted APIs) and exploits Apple Silicon unified memory (`.storageModeShared` `MTLBuffer`) directly from Swift actors:

- **Zero-Copy Unified-Memory Slabs (`SharedSlab`):** Swift actors append normalized vectors directly into shared `MTLBuffer` pages that Metal compute shaders read with zero copies.
- **Runtime-Compiled MSL Shaders:** All five compute kernels compile at initialization via `MTLDevice.makeLibrary(source:options:)`—pure `swift build` and `swift test` using macOS Command Line Tools, with an automatic Accelerate CPU fallback when no Metal device is present.
- **Multi-Precision Episodic Memory (`UnifiedMemoryStore`):** `fp32`, `fp16` (2× bandwidth), and symmetric per-vector `int8` (4× memory reduction) batched similarity search with tombstones, Maximal Marginal Relevance (MMR) diversity re-ranking, and context-window token-budget packing.
- **GPU Semantic Pub/Sub (`AgentBus`):** Fan out agent messages against thousands of subscribed agent intent vectors in a single Metal dispatch, delivering matches over Swift `AsyncStream`.
- **KV-Cache-Aware Routing (`PrefixAffinityRouter`):** Compute longest-common-prefix (LCP) token-chunk overlaps across warm inference worker slots on Metal GPU to maximize KV-cache reuse while respecting worker concurrency limits.
- **C ABI Seam (`MetalAgentBusC`):** Minimal `@_cdecl` handle API (`Headers/metal_agent_bus.h`) ready for Python `ctypes`, MLX, or Rust bindings.

---

## Benchmarks (Apple M5 Max · 128 GB Unified Memory)

Measured with `swift run -c release metal-agent-bus-bench` (CPU baseline uses single-threaded Apple Accelerate `vDSP_dotpr`; GPU times include command encoding, Metal execution, and synchronization):

| Workload | CPU (ms) | Metal GPU (ms) | Speedup | Effective GB/s | Max Abs Error vs. CPU |
|---|---:|---:|---:|---:|---:|
| Cosine `fp32` (`32q × 32,768 docs × 1536d`) | 76.29 | **3.13** | **24.4×** | 65.8 | `2.24e-08` |
| Cosine `fp16` (`32q × 32,768 docs × 1536d`) | 76.29 | **2.04** | **37.4×** | 51.5 | `4.10e-05` |
| Dot `int8` (`32q × 32,768 docs × 1536d`) | 76.29 | **1.53** | **49.8×** | 35.8 | `5.33e-04` |
| Semantic Route `fp32` (`64 msgs × 4,096 subs × 1536d`) | 18.32 | **1.02** | **18.0×** | 26.4 | `2.24e-08` |
| Prefix LCP (`128 prompts × 256 workers × 256 chunks`) | 0.80 | **0.23** | **3.4×** | — | `0 diff` |

---

## Quick Start

### Requirements
- macOS 14+ on Apple Silicon (M1–M5)
- Swift 6.0+ (`xcode-select --install` is sufficient; full Xcode is not required)

### Build, Test & Benchmark

```bash
swift test
swift run -c release metal-agent-bus-bench
```

### Swift Package Dependency

```swift
dependencies: [
    .package(url: "https://github.com/HYPHN8/metal-agent-bus.git", from: "0.1.0")
]
```

### Usage Example

```swift
import MetalAgentBus
import MetalAgentBusCore

let engine = try MetalComputeEngine()

// 1. Unified-memory episodic retrieval with MMR diversity & token-budget packing
let store = UnifiedMemoryStore(dimension: 1536, precision: .float16, engine: engine)
try await store.append(
    record: MemoryRecord(id: "m1", agentID: "researcher", content: "KV-cache prefix notes", tokenCount: 120),
    embedding: myEmbedding
)
let hits = try await store.search(
    queries: [queryEmbedding],
    options: SearchOptions(topK: 8, mmrLambda: 0.6, tokenBudget: 2048)
)

// 2. GPU-accelerated semantic pub/sub bus
let bus = AgentBus(dimension: 1536, engine: engine)
let stream = try await bus.subscribe(agentID: "coder", intentEmbedding: coderIntent, threshold: 0.78)
try await bus.publish(BusMessage(senderID: "planner", payload: "Optimize shader", embedding: taskEmbedding))

// 3. KV-prefix-affinity routing across warm local LLM workers
let router = PrefixAffinityRouter(chunkSize: 16, engine: engine)
await router.registerWorker(workerID: "mlx-worker-0", warmTokens: systemPromptTokens, maxConcurrency: 2)
let decisions = try await router.route(prompts: [incomingPromptTokens], reserveLoad: true)
```

---

## Package Architecture

- **`MetalAgentBusCore`** — `MSLShaders` (runtime-compiled MSL 3.x SIMDgroup kernels), `SharedSlab<T>` (`.storageModeShared` unified-memory arena), `MetalComputeEngine`, and `VectorMath` (Accelerate normalization, `Float16`, symmetric `Int8` quantization, 64-bit FNV-1a token-chunk hashing).
- **`MetalAgentBus`** — Swift 6 `actor` runtime: `UnifiedMemoryStore`, `AgentBus`, and `PrefixAffinityRouter`.
- **`MetalAgentBusC`** — C ABI (`@_cdecl`) wrapper and `Headers/metal_agent_bus.h`.
- **`MetalAgentBusBench`** — Reproducible CPU-vs-Metal benchmark executable.

## License

MIT — see [LICENSE](LICENSE).
