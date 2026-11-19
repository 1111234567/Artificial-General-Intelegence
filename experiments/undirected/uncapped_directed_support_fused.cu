// FP16 support-state fusion. Keep all reference half rounding boundaries;
// NVRTC's default --fmad=false prevents unintended contraction between them.
// Explicit FMAs match Torch addcmul and lerp's FP32 opmath.
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

__device__ __forceinline__ float round_half(float value) {
    return from_half(to_half(value));
}

__device__ __forceinline__ float arc_drive(
    const float* rule, float weight, float coactivity,
    float difference, float trace
) {
    float drive = round_half(__fmaf_rn(weight, rule[1], rule[0]));
    drive = round_half(__fmaf_rn(coactivity, rule[2], drive));
    drive = round_half(__fmaf_rn(trace, rule[3], drive));
    drive = round_half(__fmaf_rn(difference, rule[4], drive));
    return round_half(tanhf(drive));
}

__device__ __forceinline__ unsigned warp_sum(unsigned value) {
    for (int distance = 16; distance > 0; distance >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, distance);
    }
    return value;
}

extern "C" __global__ void directed_support_update(
    const half_bits* __restrict__ coactivity,
    const half_bits* __restrict__ difference,
    const half_bits* __restrict__ node_trace,
    const long long* __restrict__ u,
    const long long* __restrict__ v,
    const half_bits* __restrict__ rules,
    const half_bits* __restrict__ weight_scale,
    const half_bits* __restrict__ birth_coefficient,
    const half_bits* __restrict__ gate_coefficient,
    const half_bits* __restrict__ birth_positive,
    const half_bits* __restrict__ birth_negative,
    half_bits* __restrict__ gates,
    half_bits* __restrict__ weights,
    half_bits* __restrict__ eligibility,
    bool* __restrict__ mask,
    int* __restrict__ last_change,
    long long* __restrict__ births,
    long long* __restrict__ deaths,
    long long* __restrict__ edge_count,
    long long* __restrict__ edge_ticks,
    int pairs, int actors, int nodes, int rule_features,
    int step, int span, int cooldown,
    float gate_rate, float birth_threshold, float death_threshold
) {
    const unsigned actor = blockIdx.x;
    const unsigned genome = actor / actors;
    const offset_t activity_base = static_cast<offset_t>(actor) * pairs;
    const offset_t arc_base = 2 * activity_base;
    const offset_t node_base = static_cast<offset_t>(actor) * nodes;
    __shared__ float support_rule[10];
    if (threadIdx.x < 2) {
        const unsigned kind = threadIdx.x;
        const half_bits* source = rules + (2 * genome + kind) * rule_features;
        float* dest = support_rule + 5 * kind;
        const float coefficient = from_half(
            kind == 0 ? birth_coefficient[actor] : gate_coefficient[actor]);
        dest[0] = from_half(source[0]);
        dest[1] = from_half(weight_scale[2 * genome + kind]);
        dest[2] = from_half(source[2]);
        dest[3] = round_half(from_half(source[3]) + coefficient);
        dest[4] = from_half(source[4]);
    }
    __syncthreads();

    // Torch half/scalar comparisons round the scalar to half first.
    const float admit_at = round_half(birth_threshold);
    const float retire_at = round_half(death_threshold);
    unsigned admitted = 0, retired = 0;
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const float active = from_half(coactivity[activity_base + pair]);
        const float distinct = from_half(difference[activity_base + pair]);
        const float trace = round_half(
            from_half(node_trace[node_base + u[pair]]) *
            from_half(node_trace[node_base + v[pair]]));
        for (int direction = 0; direction < 2; ++direction) {
            const offset_t edge = arc_base + direction * pairs + pair;
            const float weight = from_half(weights[edge]);
            const float target = arc_drive(support_rule + 5, weight, active, distinct, trace);
            const float previous_gate = from_half(gates[edge]);
            const half_bits next_gate = to_half(
                __fmaf_rn(gate_rate, target - previous_gate, previous_gate));
            gates[edge] = next_gate;
            const bool ready = last_change[edge] <= step - cooldown;
            const bool present = mask[edge];
            const float gate = from_half(next_gate);
            const bool retire = ready && present && gate <= retire_at;
            const bool admit = ready && !present && gate >= admit_at;
            if (retire) {
                weights[edge] = 0;
                eligibility[edge] = 0;
                mask[edge] = false;
                last_change[edge] = step;
                ++retired;
            } else if (admit) {
                // Birth drive uses the pre-change weight. Multiplication
                // by .08 stores half before the separately rounded offset
                // addition; there is no post-birth weight clamp.
                const float birth_drive = arc_drive(support_rule, weight, active, distinct, trace);
                const float offset = from_half(
                    birth_drive >= 0.0f ? birth_positive[0] : birth_negative[0]);
                weights[edge] = to_half(round_half(.08f * birth_drive) + offset);
                mask[edge] = true;
                last_change[edge] = step;
                ++admitted;
            }
        }
    }

    // One CTA owns every counter for an actor. No atomics or reduction
    // ordering can affect the exact integer support statistics.
    __shared__ unsigned birth_warps[32], death_warps[32];
    admitted = warp_sum(admitted);
    retired = warp_sum(retired);
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        birth_warps[warp] = admitted;
        death_warps[warp] = retired;
    }
    __syncthreads();
    if (warp == 0) {
        admitted = warp_sum(lane < blockDim.x / 32 ? birth_warps[lane] : 0u);
        retired = warp_sum(lane < blockDim.x / 32 ? death_warps[lane] : 0u);
        if (lane == 0) {
            const long long prior_count = edge_count[actor];
            edge_ticks[actor] += prior_count * static_cast<long long>(span);
            births[actor] += admitted;
            deaths[actor] += retired;
            edge_count[actor] = prior_count + admitted - retired;
        }
    }
}
