#ifndef METAL_AGENT_BUS_H
#define METAL_AGENT_BUS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Thread safety: mab_store_append_f32, mab_store_count, and mab_store_search_f32
 * are internally synchronized per store handle. Callers must ensure mab_store_destroy
 * is invoked only after all concurrent operations on the handle have returned.
 */
void* mab_store_create(int32_t dim, int32_t initial_capacity);
void mab_store_destroy(void* store);
int32_t mab_store_append_f32(void* store, const float* values, int32_t len);
int32_t mab_store_count(void* store);
int32_t mab_store_search_f32(
    void* store,
    const float* query,
    int32_t dim,
    int32_t top_k,
    int32_t* out_indices,
    float* out_scores
);

#ifdef __cplusplus
}
#endif

#endif /* METAL_AGENT_BUS_H */
