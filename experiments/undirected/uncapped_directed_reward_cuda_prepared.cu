// PyTorch CUDA FP16 storage, with FP32 opmath between the same rounding points.
// Inline conversions make these kernels independent of toolkit headers.
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

// PyTorch local_activity is retained, avoiding any transcendental change.
// Each endpoint product rounds to half before float opmath for eligibility.
__device__ __forceinline__ float pair_activity(
    const half_bits* local_activity, const long long* u, const long long* v,
    offset_t node_base, int pair
) {
    const float first = from_half(local_activity[node_base + u[pair]]);
    const float second = from_half(local_activity[node_base + v[pair]]);
    return from_half(to_half(first * second));
}

extern "C" __global__ void directed_prepare_trace(
    const half_bits* __restrict__ local_activity,
    half_bits* __restrict__ node_trace, int count
) {
    const offset_t i = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<offset_t>(count)) {
        // mul_ stores half before add_(alpha=.05), which uses FP32 FMA.
        const float decayed = from_half(to_half(from_half(node_trace[i]) * .95f));
        node_trace[i] = to_half(__fmaf_rn(from_half(local_activity[i]), .05f, decayed));
    }
}

extern "C" __global__ void directed_prepared_eligibility_only(
    const half_bits* __restrict__ local_activity,
    const long long* __restrict__ u,
    const long long* __restrict__ v,
    const bool* __restrict__ mask,
    half_bits* __restrict__ eligibility,
    int pairs, int nodes
) {
    const offset_t activity_base = static_cast<offset_t>(blockIdx.x) * pairs;
    const offset_t arc_base = 2 * activity_base;
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const offset_t first = arc_base + pair;
        const offset_t second = first + pairs;
        const bool forward = mask[first], reverse = mask[second];
        if (forward || reverse) {
            const float activity = pair_activity(local_activity, u, v, static_cast<offset_t>(blockIdx.x) * nodes, pair);
            // Masked eligibility is zeroed by reset and after each retirement.
            if (forward) eligibility[first] = lerp(eligibility[first], activity);
            if (reverse) eligibility[second] = lerp(eligibility[second], activity);
        }
    }
}

template<bool receipt>
__device__ __forceinline__ void update_arc(
    offset_t edge, int arc, bool present, float activity, float scale,
    half_bits bound, float limit, bool scatter,
    half_bits* eligibility, half_bits* weights, half_bits* matrix,
    const long long* indices, half_bits* deltas, unsigned& attempted, unsigned& changed
) {
    deltas[edge] = 0;
    if (!present) return;
    const half_bits trace = lerp(eligibility[edge], activity);
    eligibility[edge] = trace;
    const half_bits prior = weights[edge];

    if constexpr (receipt) {
        const float signal = from_half(trace);
        const float old = from_half(prior);
        const half_bits raw = to_half(__fmaf_rn(signal, scale, old));
        const float rounded = from_half(raw);
        // Torch clamps after rounding the FP32 addcmul to FP16 and propagates NaN.
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
        // The normal inactive path does no reward math. Match the reference's
        // old-vs-old comparison/difference if a non-finite arc is present.
        changed += (prior & 0x03ffu) != 0;
        deltas[edge] = 0x7e00u;
    }
}

template<bool receipt>
__device__ __forceinline__ void update_pairs(
    offset_t activity_base, offset_t node_base, int pairs, float scale, half_bits bound, float limit,
    bool scatter, const half_bits* local_activity, const long long* u, const long long* v,
    const bool* mask, half_bits* eligibility,
    half_bits* weights, half_bits* matrix, const long long* indices, half_bits* deltas,
    unsigned& attempted, unsigned& changed
) {
    const offset_t arc_base = 2 * activity_base;
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const offset_t first = arc_base + pair;
        const offset_t second = first + pairs;
        const bool forward = mask[first], reverse = mask[second];
        const float activity = forward || reverse ? pair_activity(local_activity, u, v, node_base, pair) : 0.0f;
        update_arc<receipt>(first, pair, forward, activity, scale, bound, limit, scatter,
                            eligibility, weights, matrix, indices, deltas, attempted, changed);
        update_arc<receipt>(second, pair + pairs, reverse, activity, scale, bound, limit, scatter,
                            eligibility, weights, matrix, indices, deltas, attempted, changed);
    }
}

__device__ __forceinline__ unsigned warp_sum(unsigned value) {
    for (int distance = 16; distance > 0; distance >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, distance);
    }
    return value;
}

extern "C" __global__ void directed_prepared_reward_update(
    const half_bits* __restrict__ local_activity,
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
    int pairs, int actors, int nodes, int scatter, float weight_limit
) {
    const unsigned actor = blockIdx.x;
    const offset_t activity_base = static_cast<offset_t>(actor) * pairs;
    const offset_t node_base = static_cast<offset_t>(actor) * nodes;
    unsigned attempted = 0, changed = 0;
    if (event[actor]) {
        const half_bits bound = to_half(weight_limit);
        const float scale = rate[actor / actors] * from_half(reward[actor]);
        update_pairs<true>(activity_base, node_base, pairs, scale, bound, from_half(bound), scatter,
                           local_activity, u, v, mask, eligibility, weights,
                           matrix + static_cast<offset_t>(actor) * nodes * nodes,
                           indices, deltas, attempted, changed);
    } else {
        update_pairs<false>(activity_base, node_base, pairs, 0.0f, 0, 0.0f, false,
                            local_activity, u, v, mask, eligibility, weights, nullptr,
                            indices, deltas, attempted, changed);
    }

    // One CTA owns each actor's counters, so replay order is deterministic.
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
