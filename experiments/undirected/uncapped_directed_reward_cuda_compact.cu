// Compact work lists change which independent pairs are visited, preserving
// all FP16 storage roundings and the original-position diagnostic delta sum.
using half_bits = unsigned short;
using offset_t = unsigned long long;

__device__ __forceinline__ float from_half(half_bits bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

__device__ __forceinline__ half_bits to_half(float value) {
    half_bits bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}

__device__ __forceinline__ half_bits lerp(half_bits previous, float coactivity) {
    const float old = from_half(previous);
    return to_half(__fmaf_rn(0.1f, coactivity - old, old));
}

extern "C" __global__ void directed_compact_build_pairs(
    const bool* __restrict__ mask,
    int* __restrict__ active_pairs,
    int* __restrict__ active_pair_counts,
    int pairs
) {
    const unsigned actor = blockIdx.x;
    const offset_t list_base = static_cast<offset_t>(actor) * pairs;
    const offset_t arc_base = 2 * list_base;
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();

    // Warp-aggregated integer appends avoid a dense scan on ordinary ticks.
    // List order may differ across warps, but each pair occurs exactly once;
    // no floating-point reduction or shared arc update follows list order.
    const unsigned lane = threadIdx.x % 32;
    for (int tile = 0; tile < pairs; tile += blockDim.x) {
        const int pair = tile + threadIdx.x;
        const bool present = pair < pairs && (mask[arc_base + pair] || mask[arc_base + pairs + pair]);
        const unsigned ballot = __ballot_sync(0xffffffffu, present);
        int start = 0;
        if (lane == 0 && ballot) start = atomicAdd(&count, __popc(ballot));
        start = __shfl_sync(0xffffffffu, start, 0);
        if (present) {
            const unsigned preceding = ballot & ((1u << lane) - 1u);
            active_pairs[list_base + start + __popc(preceding)] = pair;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) active_pair_counts[actor] = count;
}

template<bool direct>
__device__ __forceinline__ float pair_activity(
    const half_bits* features, const long long* u, const long long* v,
    offset_t activity_base, offset_t node_base, int pair
) {
    if constexpr (direct) {
        const float first = from_half(features[node_base + u[pair]]);
        const float second = from_half(features[node_base + v[pair]]);
        return from_half(to_half(first * second));
    } else {
        return from_half(features[activity_base + pair]);
    }
}

template<bool direct>
__device__ __forceinline__ void update_eligibility(
    const half_bits* features, const long long* u, const long long* v,
    const bool* mask, half_bits* eligibility,
    const int* active_pairs, const int* active_pair_counts, int pairs, int nodes
) {
    const unsigned actor = blockIdx.x;
    const offset_t activity_base = static_cast<offset_t>(actor) * pairs;
    const offset_t arc_base = 2 * activity_base;
    const offset_t node_base = static_cast<offset_t>(actor) * nodes;
    const int count = active_pair_counts[actor];
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const int pair = active_pairs[activity_base + index];
        const offset_t first = arc_base + pair;
        const offset_t second = first + pairs;
        const float activity = pair_activity<direct>(features, u, v, activity_base, node_base, pair);
        if (mask[first]) eligibility[first] = lerp(eligibility[first], activity);
        if (mask[second]) eligibility[second] = lerp(eligibility[second], activity);
        // Absent eligibility is already zero from reset or retirement.
    }
}

extern "C" __global__ void directed_compact_eligibility_only(
    const half_bits* __restrict__ features,
    const long long* __restrict__ u,
    const long long* __restrict__ v,
    const bool* __restrict__ mask,
    half_bits* __restrict__ eligibility,
    const int* __restrict__ active_pairs,
    const int* __restrict__ active_pair_counts,
    int pairs, int nodes, int direct
) {
    if (direct) {
        update_eligibility<true>(features, u, v, mask, eligibility,
                                active_pairs, active_pair_counts, pairs, nodes);
    } else {
        update_eligibility<false>(features, u, v, mask, eligibility,
                                 active_pairs, active_pair_counts, pairs, nodes);
    }
}

template<bool receipt>
__device__ __forceinline__ void update_arc(
    offset_t edge, int arc, bool present, float activity, float scale,
    half_bits bound, float limit, bool scatter,
    half_bits* eligibility, half_bits* weights, half_bits* matrix,
    const long long* indices, half_bits* deltas, unsigned& attempted, unsigned& changed
) {
    // The full scratch was cleared before launch; inactive directions stay 0.
    if (!present) return;
    const half_bits trace = lerp(eligibility[edge], activity);
    eligibility[edge] = trace;
    const half_bits prior = weights[edge];
    if constexpr (receipt) {
        const float signal = from_half(trace);
        const float old = from_half(prior);
        const half_bits raw = to_half(__fmaf_rn(signal, scale, old));
        const float rounded = from_half(raw);
        const half_bits next = rounded > limit ? bound : rounded < -limit ? bound | 0x8000u : raw;
        const float value = from_half(next);
        attempted += signal != 0.0f;
        changed += value != old;
        deltas[edge] = to_half(value - old) & 0x7fffu;
        if (next != prior) {
            weights[edge] = next;
            if (scatter) matrix[indices[arc]] = next;
        }
    } else if ((prior & 0x7c00u) == 0x7c00u) {
        // Preserve the prepared backend's old-vs-old NaN/Inf diagnostics.
        changed += (prior & 0x03ffu) != 0;
        deltas[edge] = 0x7e00u;
    }
}

template<bool receipt, bool direct>
__device__ __forceinline__ void update_pairs(
    offset_t activity_base, offset_t node_base, int pairs, float scale, half_bits bound, float limit,
    bool scatter, const half_bits* features, const long long* u, const long long* v,
    const bool* mask, half_bits* eligibility,
    half_bits* weights, half_bits* matrix, const long long* indices, half_bits* deltas,
    const int* active_pairs, int count, unsigned& attempted, unsigned& changed
) {
    const offset_t arc_base = 2 * activity_base;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const int pair = active_pairs[activity_base + index];
        const offset_t first = arc_base + pair;
        const offset_t second = first + pairs;
        const float activity = pair_activity<direct>(features, u, v, activity_base, node_base, pair);
        update_arc<receipt>(first, pair, mask[first], activity, scale, bound, limit, scatter,
                            eligibility, weights, matrix, indices, deltas, attempted, changed);
        update_arc<receipt>(second, pair + pairs, mask[second], activity, scale, bound, limit, scatter,
                            eligibility, weights, matrix, indices, deltas, attempted, changed);
    }
}

__device__ __forceinline__ unsigned warp_sum(unsigned value) {
    for (int distance = 16; distance > 0; distance >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, distance);
    }
    return value;
}

extern "C" __global__ void directed_compact_reward_update(
    const half_bits* __restrict__ features,
    const long long* __restrict__ u,
    const long long* __restrict__ v,
    const bool* __restrict__ mask,
    half_bits* __restrict__ eligibility,
    half_bits* __restrict__ weights,
    half_bits* __restrict__ matrix,
    const long long* __restrict__ indices,
    const half_bits* __restrict__ reward,
    const float* __restrict__ rate,
    const bool* __restrict__ event,
    long long* __restrict__ attempted_writes,
    long long* __restrict__ changed_writes,
    half_bits* __restrict__ deltas,
    const int* __restrict__ active_pairs,
    const int* __restrict__ active_pair_counts,
    int pairs, int actors, int nodes, int direct, int scatter, float weight_limit
) {
    const unsigned actor = blockIdx.x;
    const offset_t activity_base = static_cast<offset_t>(actor) * pairs;
    const offset_t node_base = static_cast<offset_t>(actor) * nodes;
    const int count = active_pair_counts[actor];
    unsigned attempted = 0, changed = 0;
    half_bits bound = 0;
    float scale = 0.0f;
    if (event[actor]) {
        bound = to_half(weight_limit);
        scale = rate[actor / actors] * from_half(reward[actor]);
    }
    half_bits* actor_matrix = matrix + static_cast<offset_t>(actor) * nodes * nodes;
    if (event[actor] && direct) {
        update_pairs<true, true>(activity_base, node_base, pairs, scale, bound, from_half(bound), scatter,
                                features, u, v, mask, eligibility, weights, actor_matrix, indices,
                                deltas, active_pairs, count, attempted, changed);
    } else if (event[actor]) {
        update_pairs<true, false>(activity_base, node_base, pairs, scale, bound, from_half(bound), scatter,
                                 features, u, v, mask, eligibility, weights, actor_matrix, indices,
                                 deltas, active_pairs, count, attempted, changed);
    } else if (direct) {
        update_pairs<false, true>(activity_base, node_base, pairs, 0.0f, 0, 0.0f, false,
                                 features, u, v, mask, eligibility, weights, actor_matrix, indices,
                                 deltas, active_pairs, count, attempted, changed);
    } else {
        update_pairs<false, false>(activity_base, node_base, pairs, 0.0f, 0, 0.0f, false,
                                  features, u, v, mask, eligibility, weights, actor_matrix, indices,
                                  deltas, active_pairs, count, attempted, changed);
    }

    // The list order only changes unsigned integer additions. One CTA owns
    // each actor's int64 counters, matching the prepared backend exactly.
    __shared__ unsigned attempted_warps[32], changed_warps[32];
    attempted = warp_sum(attempted);
    changed = warp_sum(changed);
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        attempted_warps[warp] = attempted;
        changed_warps[warp] = changed;
    }
    __syncthreads();
    if (warp == 0) {
        attempted = warp_sum(lane < blockDim.x / 32 ? attempted_warps[lane] : 0u);
        changed = warp_sum(lane < blockDim.x / 32 ? changed_warps[lane] : 0u);
        if (lane == 0) {
            if (attempted) attempted_writes[actor] += attempted;
            if (changed) changed_writes[actor] += changed;
        }
    }
}
