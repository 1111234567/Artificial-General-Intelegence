// One block owns an actor's support state and compacts its final direction
// bits only if support changed. The arc arithmetic matches support_tick.py.
using half_bits = unsigned short;
using offset_t = unsigned long long;

constexpr int nodes = 96;
constexpr int pairs = nodes * (nodes - 1) / 2;
constexpr int warp_size = 32;
constexpr int threads = 128;
constexpr int warps = threads / warp_size;
constexpr int tiles = (pairs + warp_size - 1) / warp_size;
constexpr int tiles_per_warp = (tiles + warps - 1) / warps;
constexpr unsigned all_lanes = 0xffffffffu;
constexpr unsigned node_mask = (1u << ENCODED_NODE_BITS) - 1;
constexpr unsigned v_shift = ENCODED_PAIR_BITS + ENCODED_NODE_BITS;
constexpr unsigned forward_bit = 1u << (v_shift + ENCODED_NODE_BITS);
constexpr unsigned reverse_bit = forward_bit << 1;

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

__device__ __forceinline__ float round_half(float value) {
    return from_half(to_half(value));
}

// Torch CUDA reduces the contiguous 2-, 4-, and 8-element inputs with a
// descending shuffle tree. Each input is first added to the +0 identity.
__device__ __forceinline__ float sum_input(half_bits value) {
    return __fadd_rn(0.0f, from_half(value));
}

__device__ __forceinline__ float begin_attempt(const half_bits* job) {
    return round_half(sum_input(job[12]) + sum_input(job[13]));
}

__device__ __forceinline__ float finish_attempt(const half_bits* job) {
    const float even = (sum_input(job[14]) + sum_input(job[18]))
                     + (sum_input(job[16]) + sum_input(job[20]));
    const float odd = (sum_input(job[15]) + sum_input(job[19]))
                    + (sum_input(job[17]) + sum_input(job[21]));
    return round_half(even + odd);
}

__device__ __forceinline__ float mean_attempt(float first, float second, float first_status, float second_status) {
    const float a = __fadd_rn(0.0f, round_half(first * first_status));
    const float b = __fadd_rn(0.0f, round_half(second * second_status));
    return round_half((a + b) * .5f);
}

__device__ __forceinline__ float receipt_coefficient(const half_bits* obs, const half_bits* rule) {
    const half_bits* other = obs + SUPPORT_JOB_FEATURES;
    const float first_begin = begin_attempt(obs);
    const float second_begin = begin_attempt(other);
    const float first_finish = finish_attempt(obs);
    const float second_finish = finish_attempt(other);
    const float first_ack = from_half(obs[22]);
    const float second_ack = from_half(other[22]);
    const float first_rej = from_half(obs[23]);
    const float second_rej = from_half(other[23]);
    const float begin_ack = __fadd_rn(0.0f, round_half(
        mean_attempt(first_begin, second_begin, first_ack, second_ack) * from_half(rule[5])));
    const float finish_ack = __fadd_rn(0.0f, round_half(
        mean_attempt(first_finish, second_finish, first_ack, second_ack) * from_half(rule[6])));
    const float begin_rej = __fadd_rn(0.0f, round_half(
        mean_attempt(first_begin, second_begin, first_rej, second_rej) * from_half(rule[7])));
    const float finish_rej = __fadd_rn(0.0f, round_half(
        mean_attempt(first_finish, second_finish, first_rej, second_rej) * from_half(rule[8])));
    return round_half((begin_ack + begin_rej) + (finish_ack + finish_rej));
}

__device__ __forceinline__ float arc_drive(
    const float* rule, float weight, float coactivity, float difference, float trace
) {
    float drive = round_half(__fmaf_rn(weight, rule[1], rule[0]));
    drive = round_half(__fmaf_rn(coactivity, rule[2], drive));
    drive = round_half(__fmaf_rn(trace, rule[3], drive));
    drive = round_half(__fmaf_rn(difference, rule[4], drive));
    return round_half(tanhf(drive));
}

__device__ __forceinline__ unsigned warp_sum(unsigned value) {
    #pragma unroll
    for (int distance = 16; distance > 0; distance >>= 1) {
        value += __shfl_down_sync(all_lanes, value, distance);
    }
    return value;
}

extern "C" __global__ __launch_bounds__(threads) void directed_support_tile(
    const half_bits* __restrict__ local_activity,
    const half_bits* __restrict__ node_trace,
    const int* __restrict__ metadata,
    const half_bits* __restrict__ obs,
    const half_bits* __restrict__ rules,
    const half_bits* __restrict__ weight_scale,
    const half_bits* __restrict__ birth_positive,
    const half_bits* __restrict__ birth_negative,
    half_bits* __restrict__ gates,
    half_bits* __restrict__ weights,
    half_bits* __restrict__ matrix,
    const long long* __restrict__ arc_indices,
    half_bits* __restrict__ eligibility,
    bool* __restrict__ mask,
    int* __restrict__ last_change,
    long long* __restrict__ births,
    long long* __restrict__ deaths,
    long long* __restrict__ edge_count,
    long long* __restrict__ edge_ticks,
    int* __restrict__ active_pairs,
    int* __restrict__ active_pair_counts,
    int actors, int step, int span, int cooldown,
    float gate_rate, float birth_threshold, float death_threshold
) {
    const unsigned actor = blockIdx.x;
    const unsigned genome = actor / actors;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int tile_start = warp * tiles_per_warp;
    const int tile_end = min(tile_start + tiles_per_warp, tiles);
    const offset_t pair_base = static_cast<offset_t>(actor) * pairs;
    const offset_t arc_base = 2 * pair_base;
    const offset_t node_base = static_cast<offset_t>(actor) * nodes;
    const offset_t matrix_base = node_base * nodes;
    const offset_t obs_base = static_cast<offset_t>(actor) * 2 * SUPPORT_JOB_FEATURES;

    __shared__ float activity[nodes], traces[nodes];
    __shared__ float support_rule[10];
    if (threadIdx.x < nodes) {
        activity[threadIdx.x] = from_half(local_activity[node_base + threadIdx.x]);
        traces[threadIdx.x] = from_half(node_trace[node_base + threadIdx.x]);
    }
    if (threadIdx.x < 2) {
        const unsigned kind = threadIdx.x;
        const half_bits* source = rules + (2 * genome + kind) * 9;
        float* dest = support_rule + 5 * kind;
        dest[0] = from_half(source[0]);
        dest[1] = from_half(weight_scale[2 * genome + kind]);
        dest[2] = from_half(source[2]);
        dest[3] = round_half(from_half(source[3]) + receipt_coefficient(obs + obs_base, source));
        dest[4] = from_half(source[4]);
    }
    __syncthreads();

    const float admit_at = round_half(birth_threshold);
    const float retire_at = round_half(death_threshold);
    __shared__ unsigned forward_masks[tiles], reverse_masks[tiles];
    unsigned admitted = 0, retired = 0, active_count = 0;
    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int pair = tile * warp_size + lane;
        bool forward = false, reverse = false;
        if (pair < pairs) {
            const unsigned packed = metadata[pair];
            const unsigned first = (packed >> ENCODED_PAIR_BITS) & node_mask;
            const unsigned second = (packed >> v_shift) & node_mask;
            const float a = activity[first], b = activity[second];
            const float active = round_half(a * b);
            const float distinct = fabsf(round_half(a - b));
            const float trace = round_half(traces[first] * traces[second]);

            #pragma unroll
            for (int direction = 0; direction < 2; ++direction) {
                const int arc = direction * pairs + pair;
                const offset_t edge = arc_base + arc;
                const float weight = from_half(weights[edge]);
                const float target = arc_drive(support_rule + 5, weight, active, distinct, trace);
                const float previous_gate = from_half(gates[edge]);
                const half_bits next_gate = to_half(
                    __fmaf_rn(gate_rate, target - previous_gate, previous_gate));
                gates[edge] = next_gate;
                const bool ready = last_change[edge] <= step - cooldown;
                bool present = mask[edge];
                const float gate = from_half(next_gate);
                if (ready && present && gate <= retire_at) {
                    weights[edge] = 0;
                    matrix[matrix_base + arc_indices[arc]] = 0;
                    eligibility[edge] = 0;
                    mask[edge] = present = false;
                    last_change[edge] = step;
                    ++retired;
                } else if (ready && !present && gate >= admit_at) {
                    const float birth_drive = arc_drive(support_rule, weight, active, distinct, trace);
                    const float offset = from_half(
                        birth_drive >= 0.0f ? birth_positive[0] : birth_negative[0]);
                    const half_bits newborn = to_half(round_half(.08f * birth_drive) + offset);
                    weights[edge] = newborn;
                    matrix[matrix_base + arc_indices[arc]] = newborn;
                    mask[edge] = present = true;
                    last_change[edge] = step;
                    ++admitted;
                }
                if (direction == 0) forward = present;
                else reverse = present;
            }
        }
        const unsigned forward_mask = __ballot_sync(all_lanes, forward);
        const unsigned reverse_mask = __ballot_sync(all_lanes, reverse);
        if (lane == 0) {
            forward_masks[tile] = forward_mask;
            reverse_masks[tile] = reverse_mask;
            active_count += __popc(forward_mask | reverse_mask);
        }
    }

    __shared__ unsigned birth_warps[warps], death_warps[warps], counts[warps];
    admitted = warp_sum(admitted);
    retired = warp_sum(retired);
    if (lane == 0) {
        birth_warps[warp] = admitted;
        death_warps[warp] = retired;
        counts[warp] = active_count;
    }
    __syncthreads();
    unsigned total_admitted = 0, total_retired = 0;
    #pragma unroll
    for (int i = 0; i < warps; ++i) {
        total_admitted += birth_warps[i];
        total_retired += death_warps[i];
    }
    const bool changed = total_admitted != 0 || total_retired != 0;
    if (threadIdx.x == 0) {
        const long long prior_count = edge_count[actor];
        edge_ticks[actor] += prior_count * static_cast<long long>(span);
        births[actor] += total_admitted;
        deaths[actor] += total_retired;
        edge_count[actor] = prior_count + total_admitted - total_retired;
        if (changed) active_pair_counts[actor] = counts[0] + counts[1] + counts[2] + counts[3];
    }
    if (!changed) return;

    // Warp tiles cover disjoint consecutive ranges. Prefixes need only one
    // count per warp, and each writer reuses its final direction ballots.
    unsigned cursor = 0;
    for (int i = 0; i < warp; ++i) cursor += counts[i];
    const unsigned lane_bit = 1u << lane;
    for (int tile = tile_start; tile < tile_end; ++tile) {
        const unsigned forward = forward_masks[tile];
        const unsigned reverse = reverse_masks[tile];
        const unsigned present = forward | reverse;
        if (present & lane_bit) {
            const unsigned direction = ((forward & lane_bit) ? forward_bit : 0u)
                                     | ((reverse & lane_bit) ? reverse_bit : 0u);
            const unsigned rank = __popc(present & (lane_bit - 1u));
            active_pairs[pair_base + cursor + rank] = metadata[tile * warp_size + lane] | direction;
        }
        cursor += __popc(present);
    }
}
