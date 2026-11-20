import Accelerate
import Foundation

/// Vector normalization and quantization helpers using Apple Accelerate where applicable.
public enum VectorMath {
    /// Returns an L2-normalized copy of `vector`. Zero vectors remain zero.
    public static func normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }
        var sumSq: Float = 0
        vDSP_svesq(vector, 1, &sumSq, vDSP_Length(vector.count))
        let norm = sqrt(sumSq)
        guard norm > 1e-12 else { return [Float](repeating: 0, count: vector.count) }
        var invNorm = 1.0 / norm
        var out = [Float](repeating: 0, count: vector.count)
        vDSP_vsmul(vector, 1, &invNorm, &out, 1, vDSP_Length(vector.count))
        return out
    }

    /// Converts an fp32 vector to IEEE-754 binary16 (`Float16` on Apple Silicon).
    public static func toFloat16(_ vector: [Float]) -> [Float16] {
        vector.map { Float16($0) }
    }

    /// Symmetric per-vector int8 quantization: `x ≈ Float(q) * scale`.
    public static func quantizeInt8(_ vector: [Float]) -> (values: [Int8], scale: Float) {
        guard !vector.isEmpty else { return ([], 1.0) }
        var maxAbs: Float = 0
        vDSP_maxmgv(vector, 1, &maxAbs, vDSP_Length(vector.count))
        guard maxAbs > 1e-12 else {
            return ([Int8](repeating: 0, count: vector.count), 1.0)
        }
        let scale = maxAbs / 127.0
        let invScale = 127.0 / maxAbs
        var quantized = [Int8](repeating: 0, count: vector.count)
        for i in 0..<vector.count {
            let scaled = (vector[i] * invScale).rounded()
            let clamped = max(-127.0, min(127.0, scaled))
            quantized[i] = Int8(clamped)
        }
        return (quantized, scale)
    }

    /// Exact CPU dot product via Accelerate `vDSP_dotpr`.
    public static func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count: Int) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    /// Deterministic 64-bit FNV-1a hash over a slice of token IDs (chunked KV-prefix fingerprint).
    public static func hashTokenChunk(_ tokens: ArraySlice<UInt32>) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for token in tokens {
            var v = token
            for _ in 0..<4 {
                hash ^= UInt64(v & 0xff)
                hash &*= 0x100000001b3
                v >>= 8
            }
        }
        return hash
    }

    /// Splits a token sequence into fixed-size chunk hashes for KV-prefix matching.
    public static func chunkHashes(tokens: [UInt32], chunkSize: Int) -> [UInt64] {
        precondition(chunkSize > 0, "chunkSize must be positive")
        let fullChunks = tokens.count / chunkSize
        guard fullChunks > 0 else { return [] }
        var hashes: [UInt64] = []
        hashes.reserveCapacity(fullChunks)
        for c in 0..<fullChunks {
            let start = c * chunkSize
            let end = start + chunkSize
            hashes.append(hashTokenChunk(tokens[start..<end]))
        }
        return hashes
    }
}
