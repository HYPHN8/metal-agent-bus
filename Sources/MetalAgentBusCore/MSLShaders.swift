import Foundation

/// Runtime-compiled Metal Shading Language (MSL) compute kernels for multi-agent workloads.
///
/// Designed for Apple Silicon unified memory (`.storageModeShared`) and compiled at runtime
/// via `MTLDevice.makeLibrary(source:options:)` so SwiftPM packages build cleanly using
/// macOS Command Line Tools without requiring full Xcode.
public enum MSLShaders {
    public static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct SimilarityParams {
        uint num_queries;
        uint num_docs;
        uint dim;
        uint normalize_inputs;
    };

    struct RouteParams {
        uint num_messages;
        uint num_subscribers;
        uint dim;
    };

    struct PrefixParams {
        uint num_queries;
        uint num_workers;
        uint stride;
    };

    // Each SIMDgroup (32 lanes on Apple Silicon) computes one (query, doc) similarity score.
    // Threadgroup size: 256 threads = 8 SIMDgroups computing 8 docs in parallel per query.
    kernel void batch_cosine_similarity_f32(
        device const float* queries      [[buffer(0)]],
        device const float* corpus       [[buffer(1)]],
        device float*       scores       [[buffer(2)]],
        constant SimilarityParams& p     [[buffer(3)]],
        uint2 tg_id                      [[threadgroup_position_in_grid]],
        uint  sg_id                      [[simdgroup_index_in_threadgroup]],
        uint  lane                       [[thread_index_in_simdgroup]],
        uint  sgs_per_tg                 [[simdgroups_per_threadgroup]],
        uint  simd_width                 [[threads_per_simdgroup]]
    ) {
        uint doc = tg_id.x * sgs_per_tg + sg_id;
        uint q   = tg_id.y;
        if (doc >= p.num_docs || q >= p.num_queries) {
            return;
        }

        device const float* q_ptr = queries + (ulong)q * p.dim;
        device const float* d_ptr = corpus  + (ulong)doc * p.dim;

        float dot_acc = 0.0f;
        float q_norm  = 0.0f;
        float d_norm  = 0.0f;

        uint vec4_count = ((p.dim & 3u) == 0u) ? (p.dim >> 2) : 0u;
        device const float4* q4 = (device const float4*)q_ptr;
        device const float4* d4 = (device const float4*)d_ptr;

        if (p.normalize_inputs != 0u) {
            for (uint i = lane; i < vec4_count; i += simd_width) {
                float4 qv = q4[i];
                float4 dv = d4[i];
                dot_acc += dot(qv, dv);
                q_norm  += dot(qv, qv);
                d_norm  += dot(dv, dv);
            }
            for (uint i = (vec4_count << 2) + lane; i < p.dim; i += simd_width) {
                float qv = q_ptr[i];
                float dv = d_ptr[i];
                dot_acc += qv * dv;
                q_norm  += qv * qv;
                d_norm  += dv * dv;
            }
            dot_acc = simd_sum(dot_acc);
            q_norm  = simd_sum(q_norm);
            d_norm  = simd_sum(d_norm);
            if (lane == 0u) {
                float denom = sqrt(q_norm * d_norm);
                scores[(ulong)q * p.num_docs + doc] = (denom > 1e-12f) ? (dot_acc / denom) : 0.0f;
            }
        } else {
            for (uint i = lane; i < vec4_count; i += simd_width) {
                dot_acc += dot(q4[i], d4[i]);
            }
            for (uint i = (vec4_count << 2) + lane; i < p.dim; i += simd_width) {
                dot_acc += q_ptr[i] * d_ptr[i];
            }
            dot_acc = simd_sum(dot_acc);
            if (lane == 0u) {
                scores[(ulong)q * p.num_docs + doc] = dot_acc;
            }
        }
    }

    // Half-precision (fp16) corpus & queries with fp32 accumulation for 2x bandwidth on Apple Silicon.
    kernel void batch_cosine_similarity_f16(
        device const half* queries       [[buffer(0)]],
        device const half* corpus        [[buffer(1)]],
        device float*      scores        [[buffer(2)]],
        constant SimilarityParams& p     [[buffer(3)]],
        uint2 tg_id                      [[threadgroup_position_in_grid]],
        uint  sg_id                      [[simdgroup_index_in_threadgroup]],
        uint  lane                       [[thread_index_in_simdgroup]],
        uint  sgs_per_tg                 [[simdgroups_per_threadgroup]],
        uint  simd_width                 [[threads_per_simdgroup]]
    ) {
        uint doc = tg_id.x * sgs_per_tg + sg_id;
        uint q   = tg_id.y;
        if (doc >= p.num_docs || q >= p.num_queries) {
            return;
        }

        device const half* q_ptr = queries + (ulong)q * p.dim;
        device const half* d_ptr = corpus  + (ulong)doc * p.dim;

        float dot_acc = 0.0f;
        uint vec4_count = ((p.dim & 3u) == 0u) ? (p.dim >> 2) : 0u;
        device const half4* q4 = (device const half4*)q_ptr;
        device const half4* d4 = (device const half4*)d_ptr;

        for (uint i = lane; i < vec4_count; i += simd_width) {
            float4 qv = float4(q4[i]);
            float4 dv = float4(d4[i]);
            dot_acc += dot(qv, dv);
        }
        for (uint i = (vec4_count << 2) + lane; i < p.dim; i += simd_width) {
            dot_acc += float(q_ptr[i]) * float(d_ptr[i]);
        }

        dot_acc = simd_sum(dot_acc);
        if (lane == 0u) {
            scores[(ulong)q * p.num_docs + doc] = dot_acc;
        }
    }

    // Symmetric per-vector int8 quantized corpus x fp32 queries (4x memory reduction).
    kernel void batch_dot_int8_quantized(
        device const float* queries      [[buffer(0)]],
        device const char*  corpus_q     [[buffer(1)]],
        device const float* corpus_scale [[buffer(2)]],
        device float*       scores       [[buffer(3)]],
        constant SimilarityParams& p     [[buffer(4)]],
        uint2 tg_id                      [[threadgroup_position_in_grid]],
        uint  sg_id                      [[simdgroup_index_in_threadgroup]],
        uint  lane                       [[thread_index_in_simdgroup]],
        uint  sgs_per_tg                 [[simdgroups_per_threadgroup]],
        uint  simd_width                 [[threads_per_simdgroup]]
    ) {
        uint doc = tg_id.x * sgs_per_tg + sg_id;
        uint q   = tg_id.y;
        if (doc >= p.num_docs || q >= p.num_queries) {
            return;
        }

        device const float* q_ptr = queries  + (ulong)q * p.dim;
        device const char*  d_ptr = corpus_q + (ulong)doc * p.dim;

        float dot_acc = 0.0f;
        uint vec4_count = ((p.dim & 3u) == 0u) ? (p.dim >> 2) : 0u;
        device const float4* q4 = (device const float4*)q_ptr;
        device const char4*  d4 = (device const char4*)d_ptr;

        for (uint i = lane; i < vec4_count; i += simd_width) {
            dot_acc += dot(q4[i], float4(d4[i]));
        }
        for (uint i = (vec4_count << 2) + lane; i < p.dim; i += simd_width) {
            dot_acc += q_ptr[i] * float(d_ptr[i]);
        }

        dot_acc = simd_sum(dot_acc);
        if (lane == 0u) {
            scores[(ulong)q * p.num_docs + doc] = dot_acc * corpus_scale[doc];
        }
    }

    // Semantic pub/sub fan-out routing: [M x D] messages x [S x D] subscriber intent vectors.
    kernel void semantic_route_threshold(
        device const float* messages     [[buffer(0)]],
        device const float* subscribers  [[buffer(1)]],
        device const float* thresholds   [[buffer(2)]],
        device float*       scores       [[buffer(3)]],
        device uchar*       matches      [[buffer(4)]],
        constant RouteParams& p          [[buffer(5)]],
        uint2 tg_id                      [[threadgroup_position_in_grid]],
        uint  sg_id                      [[simdgroup_index_in_threadgroup]],
        uint  lane                       [[thread_index_in_simdgroup]],
        uint  sgs_per_tg                 [[simdgroups_per_threadgroup]],
        uint  simd_width                 [[threads_per_simdgroup]]
    ) {
        uint sub = tg_id.x * sgs_per_tg + sg_id;
        uint msg = tg_id.y;
        if (sub >= p.num_subscribers || msg >= p.num_messages) {
            return;
        }

        device const float* m_ptr = messages    + (ulong)msg * p.dim;
        device const float* s_ptr = subscribers + (ulong)sub * p.dim;

        float dot_acc = 0.0f;
        uint vec4_count = ((p.dim & 3u) == 0u) ? (p.dim >> 2) : 0u;
        device const float4* m4 = (device const float4*)m_ptr;
        device const float4* s4 = (device const float4*)s_ptr;

        for (uint i = lane; i < vec4_count; i += simd_width) {
            dot_acc += dot(m4[i], s4[i]);
        }
        for (uint i = (vec4_count << 2) + lane; i < p.dim; i += simd_width) {
            dot_acc += m_ptr[i] * s_ptr[i];
        }

        dot_acc = simd_sum(dot_acc);
        if (lane == 0u) {
            ulong idx = (ulong)msg * p.num_subscribers + sub;
            scores[idx]  = dot_acc;
            matches[idx] = (dot_acc >= thresholds[sub]) ? uchar(1) : uchar(0);
        }
    }

    // Longest common prefix length (in chunks) between [Q x stride] query hashes and [W x stride] worker hashes.
    kernel void prefix_overlap_score(
        device const ulong* query_hashes   [[buffer(0)]],
        device const uint*  query_lens     [[buffer(1)]],
        device const ulong* worker_hashes  [[buffer(2)]],
        device const uint*  worker_lens    [[buffer(3)]],
        device uint*        overlap_out    [[buffer(4)]],
        constant PrefixParams& p           [[buffer(5)]],
        uint2 gid                          [[thread_position_in_grid]]
    ) {
        uint w = gid.x;
        uint q = gid.y;
        if (w >= p.num_workers || q >= p.num_queries) {
            return;
        }

        uint limit = min(min(query_lens[q], worker_lens[w]), p.stride);
        device const ulong* q_ptr = query_hashes  + (ulong)q * p.stride;
        device const ulong* w_ptr = worker_hashes + (ulong)w * p.stride;

        uint matched = 0u;
        for (uint i = 0u; i < limit; ++i) {
            if (q_ptr[i] != w_ptr[i]) {
                break;
            }
            matched += 1u;
        }
        overlap_out[(ulong)q * p.num_workers + w] = matched;
    }
    """
}
