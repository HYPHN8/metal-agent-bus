import Foundation
import MetalAgentBusCore

/// Opaque synchronous handle backing the C ABI for Python (`ctypes`), Rust, or C callers.
///
/// Operations (`append`, `count`, `search`) are internally serialized by `lock` so concurrent
/// reads/writes from multiple host threads cannot race on `vectorCount` or `SharedSlab` growth.
/// Callers must ensure `mab_store_destroy` does not overlap with any in-flight call on the same handle.
private final class MABStoreHandle {
    let dim: Int
    let engine: MetalComputeEngine
    let slab: SharedSlab<Float>
    let lock = NSLock()
    var vectorCount: Int = 0

    init(dim: Int, initialCapacity: Int) throws {
        self.dim = dim
        self.engine = try MetalComputeEngine(preferCPU: false)
        self.slab = SharedSlab<Float>(device: engine.device, initialCapacity: max(initialCapacity, 64) * dim)
    }
}

/// Creates a new unified-memory fp32 vector store handle. Returns NULL on error.
@_cdecl("mab_store_create")
public func mab_store_create(_ dim: Int32, _ initialCapacity: Int32) -> UnsafeMutableRawPointer? {
    guard dim > 0 else { return nil }
    do {
        let handle = try MABStoreHandle(dim: Int(dim), initialCapacity: max(Int(initialCapacity), 64))
        return Unmanaged.passRetained(handle).toOpaque()
    } catch {
        return nil
    }
}

/// Destroys a store handle previously created with `mab_store_create`.
@_cdecl("mab_store_destroy")
public func mab_store_destroy(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr else { return }
    Unmanaged<MABStoreHandle>.fromOpaque(ptr).release()
}

/// Appends one fp32 vector (normalized internally) and returns its 0-based slot index, or -1 on error.
@_cdecl("mab_store_append_f32")
public func mab_store_append_f32(
    _ ptr: UnsafeMutableRawPointer?,
    _ values: UnsafePointer<Float>?,
    _ len: Int32
) -> Int32 {
    guard let ptr, let values else { return -1 }
    let handle = Unmanaged<MABStoreHandle>.fromOpaque(ptr).takeUnretainedValue()
    guard Int(len) == handle.dim else { return -1 }
    let vec = Array(UnsafeBufferPointer(start: values, count: handle.dim))
    let norm = VectorMath.normalize(vec)
    handle.lock.lock()
    defer { handle.lock.unlock() }
    handle.slab.append(contentsOf: norm)
    let slot = handle.vectorCount
    handle.vectorCount += 1
    return Int32(slot)
}

/// Returns the number of vectors currently stored.
@_cdecl("mab_store_count")
public func mab_store_count(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr else { return 0 }
    let handle = Unmanaged<MABStoreHandle>.fromOpaque(ptr).takeUnretainedValue()
    handle.lock.lock()
    defer { handle.lock.unlock() }
    return Int32(handle.vectorCount)
}

/// Searches for the top-`topK` nearest vectors to `query` and writes slot indices + scores into caller buffers.
/// Returns the number of results written, or -1 on error.
@_cdecl("mab_store_search_f32")
public func mab_store_search_f32(
    _ ptr: UnsafeMutableRawPointer?,
    _ query: UnsafePointer<Float>?,
    _ dim: Int32,
    _ topK: Int32,
    _ outIndices: UnsafeMutablePointer<Int32>?,
    _ outScores: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr, let query, let outIndices, let outScores, topK > 0 else { return -1 }
    let handle = Unmanaged<MABStoreHandle>.fromOpaque(ptr).takeUnretainedValue()
    guard Int(dim) == handle.dim else { return -1 }

    let qVec = VectorMath.normalize(Array(UnsafeBufferPointer(start: query, count: handle.dim)))
    handle.lock.lock()
    defer { handle.lock.unlock() }
    guard handle.vectorCount > 0 else { return 0 }

    do {
        let scores = try handle.engine.batchSimilarityF32(
            queries: qVec,
            numQueries: 1,
            corpus: handle.slab,
            numDocs: handle.vectorCount,
            dim: handle.dim,
            normalizeInputs: false
        )
        let k = min(Int(topK), scores.count)
        let ranked = scores.enumerated().sorted { $0.element > $1.element }.prefix(k)
        for (i, pair) in ranked.enumerated() {
            outIndices[i] = Int32(pair.offset)
            outScores[i] = pair.element
        }
        return Int32(k)
    } catch {
        return -1
    }
}
