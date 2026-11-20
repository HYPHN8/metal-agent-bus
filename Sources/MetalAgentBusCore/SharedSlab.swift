import Foundation
import Metal

/// Storage precision for vectors stored inside a unified-memory slab.
public enum VectorPrecision: String, Sendable, CaseIterable {
    case float32
    case float16
    case int8
}

/// Zero-copy unified-memory slab backed by a `.storageModeShared` `MTLBuffer` on Apple Silicon.
///
/// Both Swift CPU actors and Metal compute shaders access the exact same memory pages without
/// staging copies. Growth allocates a fresh buffer generation (doubling capacity) while previous
/// generations remain safely retained by any in-flight Metal command buffer.
public final class SharedSlab<Element>: @unchecked Sendable {
    public private(set) var capacity: Int
    public private(set) var count: Int = 0
    public private(set) var generation: UInt64 = 1

    private let device: (any MTLDevice)?
    private var mtlBuffer: (any MTLBuffer)?
    private var fallbackStorage: UnsafeMutablePointer<Element>?

    public init(device: (any MTLDevice)?, initialCapacity: Int) {
        let cap = max(initialCapacity, 64)
        self.device = device
        self.capacity = cap
        let byteLength = cap * MemoryLayout<Element>.stride
        if let device, let buf = device.makeBuffer(length: byteLength, options: .storageModeShared) {
            self.mtlBuffer = buf
        } else {
            self.fallbackStorage = UnsafeMutablePointer<Element>.allocate(capacity: cap)
        }
    }

    deinit {
        fallbackStorage?.deallocate()
    }

    /// Raw pointer to the active unified-memory backing store.
    public var mutablePointer: UnsafeMutablePointer<Element> {
        if let mtlBuffer {
            return mtlBuffer.contents().bindMemory(to: Element.self, capacity: capacity)
        }
        return fallbackStorage!
    }

    public var pointer: UnsafePointer<Element> {
        UnsafePointer(mutablePointer)
    }

    /// Underlying `MTLBuffer` if Metal unified memory is active.
    public var metalBuffer: (any MTLBuffer)? {
        mtlBuffer
    }

    /// Ensures space for at least `minCapacity` elements, allocating a fresh generation buffer on growth.
    public func ensureCapacity(_ minCapacity: Int) {
        guard minCapacity > capacity else { return }
        var newCap = capacity
        while newCap < minCapacity {
            newCap *= 2
        }
        let byteLength = newCap * MemoryLayout<Element>.stride
        if let device, let newBuf = device.makeBuffer(length: byteLength, options: .storageModeShared) {
            if count > 0 {
                memcpy(newBuf.contents(), mutablePointer, count * MemoryLayout<Element>.stride)
            }
            mtlBuffer = newBuf
            fallbackStorage?.deallocate()
            fallbackStorage = nil
        } else {
            let newPtr = UnsafeMutablePointer<Element>.allocate(capacity: newCap)
            if count > 0 {
                newPtr.update(from: mutablePointer, count: count)
            }
            fallbackStorage?.deallocate()
            fallbackStorage = newPtr
        }
        capacity = newCap
        generation &+= 1
    }

    /// Appends a contiguous slice of elements directly into unified memory.
    public func append(contentsOf slice: [Element]) {
        guard !slice.isEmpty else { return }
        ensureCapacity(count + slice.count)
        let dst = mutablePointer.advanced(by: count)
        slice.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(dst, base, slice.count * MemoryLayout<Element>.stride)
        }
        count += slice.count
    }

    /// Appends a single element into unified memory.
    public func append(_ value: Element) {
        ensureCapacity(count + 1)
        mutablePointer.advanced(by: count).pointee = value
        count += 1
    }

    /// Overwrites an existing contiguous slice starting at `offset`.
    public func write(_ slice: [Element], at offset: Int) {
        precondition(offset >= 0 && offset + slice.count <= capacity, "SharedSlab write out of bounds")
        guard !slice.isEmpty else { return }
        let dst = mutablePointer.advanced(by: offset)
        slice.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(dst, base, slice.count * MemoryLayout<Element>.stride)
        }
    }

    /// Resets logical count without releasing unified memory pages.
    public func removeAll(keepingCapacity: Bool = true) {
        count = 0
        if !keepingCapacity {
            generation &+= 1
        }
    }
}
