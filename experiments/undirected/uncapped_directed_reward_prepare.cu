// Appended to the encoded compact reward source for its packing and FP16 helpers.
constexpr int prepare_nodes = 96;
constexpr int prepare_pairs = 4560;
constexpr int prepare_arcs = 2 * prepare_pairs;

template<bool receipt, bool check_nonfinite, bool store_deltas = true, bool track_nonfinite = false>
__device__ __forceinline__ unsigned prepare_arc(
    int arc, int matrix_index, float local, float scale, half_bits bound,
    half_bits* eligibility, half_bits* weights, half_bits* matrix, half_bits* deltas,
    bool& result_nonfinite
) {
    const half_bits trace = lerp(eligibility[arc], local);
    eligibility[arc] = trace;
    if constexpr (receipt) {
        const half_bits prior = weights[arc];
        const float signal = from_half(trace), old = from_half(prior);
        const half_bits raw = to_half(__fmaf_rn(signal, scale, old));
        const float rounded = from_half(raw), limit = from_half(bound);
        const half_bits next = rounded > limit ? bound : rounded < -limit ? bound | 0x8000u : raw;
        const float value = from_half(next);
        if constexpr (track_nonfinite) result_nonfinite |= (next & 0x7c00u) == 0x7c00u;
        if constexpr (store_deltas) deltas[arc] = to_half(value - old) & 0x7fffu;
        if (next != prior) {
            weights[arc] = next;
            matrix[matrix_index] = next;
        }
        // At most 9120 arcs. The 16-bit halves can be reduced together.
        return (static_cast<unsigned>(signal != 0.0f) << 16) | (value != old);
    } else if constexpr (check_nonfinite) {
        const half_bits prior = weights[arc];
        if ((prior & 0x7c00u) == 0x7c00u) {
            if constexpr (store_deltas) deltas[arc] = 0x7e00u;
            return 0x10000u | ((prior & 0x03ffu) != 0);
        }
    }
    return 0;
}

template<bool receipt, bool check_nonfinite, bool store_deltas = true, bool track_nonfinite = false>
__device__ __forceinline__ unsigned prepare_update_pairs(
    const float* local, const int* active_pairs, int count, float scale, half_bits bound,
    half_bits* eligibility, half_bits* weights, half_bits* matrix, half_bits* deltas,
    bool* result_nonfinite = nullptr
) {
    unsigned counters = 0;
    bool nonfinite = false;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const unsigned metadata = active_pairs[index];
        const int pair = metadata & pair_mask;
        const int u = (metadata >> u_shift) & node_mask;
        const int v = (metadata >> v_shift) & node_mask;
        const float coactivity = from_half(to_half(local[u] * local[v]));
        if (metadata & forward_bit) {
            counters += prepare_arc<receipt, check_nonfinite, store_deltas, track_nonfinite>(
                pair, u * prepare_nodes + v, coactivity, scale, bound,
                eligibility, weights, matrix, deltas, nonfinite);
        }
        if (metadata & reverse_bit) {
            counters += prepare_arc<receipt, check_nonfinite, store_deltas, track_nonfinite>(
                pair + prepare_pairs, v * prepare_nodes + u, coactivity, scale, bound,
                eligibility, weights, matrix, deltas, nonfinite);
        }
    }
    if constexpr (track_nonfinite) *result_nonfinite = nonfinite;
    return counters;
}

__device__ __forceinline__ void prepare_sum_deltas(
    const half_bits* deltas, const bool* mask, float* weight_change
) {
    constexpr int vectors = prepare_arcs / 8;
    constexpr int logical_warps = 16;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ float warp_values[logical_warps];
    for (unsigned logical_warp = warp; logical_warp < logical_warps; logical_warp += blockDim.x / 32) {
        // Match the 512-thread diagnostic even with a smaller physical CTA.
        float accumulators[8] = {};
        for (unsigned vector = logical_warp * 32 + lane; vector < vectors; vector += logical_warps * 32) {
            #pragma unroll
            for (int element = 0; element < 8; ++element) {
                const unsigned arc = vector * 8 + element;
                const float value = mask[arc] ? from_half(deltas[arc]) : 0.0f;
                accumulators[element] = __fadd_rn(accumulators[element], value);
            }
        }
        float value = accumulators[0];
        #pragma unroll
        for (int element = 1; element < 8; ++element) {
            value = __fadd_rn(value, accumulators[element]);
        }
        #pragma unroll
        for (int distance = 16; distance > 0; distance >>= 1) {
            value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, distance));
        }
        if (lane == 0) warp_values[logical_warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < logical_warps ? warp_values[lane] : 0.0f;
        #pragma unroll
        for (int distance = 8; distance > 0; distance >>= 1) {
            value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, distance));
        }
        if (lane == 0) *weight_change = __fadd_rn(*weight_change, value);
    }
}

template<bool fused_sum>
__device__ __forceinline__ void reward_prepare_update(
    const half_bits* __restrict__ h,
    const half_bits* __restrict__ obs,
    const bool* __restrict__ pending,
    half_bits* __restrict__ reward,
    bool* __restrict__ event,
    long long* __restrict__ reward_events,
    half_bits* __restrict__ node_trace,
    half_bits* __restrict__ local_activity,
    half_bits* __restrict__ eligibility,
    half_bits* __restrict__ weights,
    half_bits* __restrict__ matrix,
    const float* __restrict__ rate,
    long long* __restrict__ attempted_writes,
    long long* __restrict__ changed_writes,
    half_bits* __restrict__ deltas,
    bool* __restrict__ nonfinite_flags,
    const int* __restrict__ active_pairs,
    const int* __restrict__ active_pair_counts,
    const bool* __restrict__ mask,
    float* __restrict__ weight_change,
    int actors, int receipt_possible, int save_local, float weight_limit
) {
    const unsigned actor = blockIdx.x;
    const unsigned thread = threadIdx.x;
    const offset_t node_offset = static_cast<offset_t>(actor) * prepare_nodes;
    __shared__ float local[prepare_nodes];
    __shared__ float receipt_value;
    __shared__ unsigned warp_counters[32];

    if (thread == 0) {
        const offset_t obs_offset = static_cast<offset_t>(actor) * NODE_INPUTS;
        const float difference = from_half(to_half(__fsub_rn(
            from_half(obs[obs_offset + NODE_ACK]), from_half(obs[obs_offset + NODE_REJ]))));
        const half_bits receipt = to_half(difference * static_cast<float>(pending[actor]));
        receipt_value = from_half(receipt);
        reward[actor] = receipt;
        const bool occurred = receipt_value != 0.0f;
        event[actor] = occurred;
        if (occurred) reward_events[actor] += 1;
    }
    for (int node = thread; node < prepare_nodes; node += blockDim.x) {
        const offset_t i = node_offset + node;
        const float scaled = from_half(to_half(4.0f * from_half(h[i])));
        const half_bits local_bits = to_half(tanhf(scaled));
        const float local_value = from_half(local_bits);
        local[node] = local_value;
        if (save_local) local_activity[i] = local_bits;
        const float decayed = from_half(to_half(from_half(node_trace[i]) * .95f));
        node_trace[i] = to_half(__fmaf_rn(local_value, .05f, decayed));
    }
    __syncthreads();

    const offset_t pair_offset = static_cast<offset_t>(actor) * prepare_pairs;
    eligibility += 2 * pair_offset;
    weights += 2 * pair_offset;
    deltas += 2 * pair_offset;
    matrix += node_offset * prepare_nodes;
    active_pairs += pair_offset;
    const int count = active_pair_counts[actor];
    if (!receipt_possible) {
        prepare_update_pairs<false, false>(local, active_pairs, count, 0.0f, 0,
                                           eligibility, weights, matrix, deltas);
        return;
    }

    const bool received = receipt_value != 0.0f;
    unsigned counters = received
        ? prepare_update_pairs<true, false>(local, active_pairs, count,
                                            rate[actor / actors] * receipt_value,
                                            to_half(weight_limit), eligibility, weights, matrix, deltas)
        : prepare_update_pairs<false, true>(local, active_pairs, count, 0.0f, 0,
                                            eligibility, weights, matrix, deltas);
    const unsigned lane = thread % 32, warp = thread / 32;
    if (!received && !__any_sync(0xffffffffu, counters != 0)) {
        if (lane == 0) warp_counters[warp] = 0;
    } else {
        counters = warp_sum(counters);
        if (lane == 0) warp_counters[warp] = counters;
    }
    __syncthreads();
    if (warp == 0) {
        counters = lane < blockDim.x / 32 ? warp_counters[lane] : 0;
        if (!received && !__any_sync(0xffffffffu, counters != 0)) {
            if (lane == 0) nonfinite_flags[actor] = false;
            return;
        }
        counters = warp_sum(counters);
        if (lane == 0) {
            const unsigned primary = counters >> 16, changed = counters & 0xffffu;
            if (!received) nonfinite_flags[actor] = primary != 0;
            if (received && primary) attempted_writes[actor] += primary;
            if (changed) changed_writes[actor] += changed;
            if constexpr (fused_sum) {
                // All false-receipt contributions are +0 or canonical NaN.
                if (!received && primary) {
                    weight_change[actor] = __fadd_rn(weight_change[actor], from_half(0x7e00u));
                }
            }
        }
    }
    if constexpr (fused_sum) {
        if (received) prepare_sum_deltas(deltas, mask + 2 * pair_offset, weight_change + actor);
    }
}

#define PREPARE_UPDATE_ENTRY(name, fused_sum)                                              \
extern "C" __global__ void name(                                                          \
    const half_bits* __restrict__ h, const half_bits* __restrict__ obs,                    \
    const bool* __restrict__ pending, half_bits* __restrict__ reward,                      \
    bool* __restrict__ event, long long* __restrict__ reward_events,                       \
    half_bits* __restrict__ node_trace, half_bits* __restrict__ local_activity,            \
    half_bits* __restrict__ eligibility, half_bits* __restrict__ weights,                  \
    half_bits* __restrict__ matrix, const float* __restrict__ rate,                         \
    long long* __restrict__ attempted_writes, long long* __restrict__ changed_writes,       \
    half_bits* __restrict__ deltas, bool* __restrict__ nonfinite_flags,                     \
    const int* __restrict__ active_pairs, const int* __restrict__ active_pair_counts,       \
    const bool* __restrict__ mask, float* __restrict__ weight_change,                      \
    int actors, int receipt_possible, int save_local, float weight_limit                   \
) {                                                                                      \
    reward_prepare_update<fused_sum>(h, obs, pending, reward, event, reward_events,         \
        node_trace, local_activity, eligibility, weights, matrix, rate, attempted_writes,  \
        changed_writes, deltas, nonfinite_flags, active_pairs, active_pair_counts,          \
        mask, weight_change, actors, receipt_possible, save_local, weight_limit);          \
}

PREPARE_UPDATE_ENTRY(directed_reward_prepare_update, false)
PREPARE_UPDATE_ENTRY(directed_reward_prepare_update_and_sum, true)
