// Half observations are copied bit-for-bit except for the same three masked
// public fields as rollout_continuous. Private targets enter only post-action.
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

extern "C" __global__ void directed_world_observation(
    const half_bits* __restrict__ frame,
    const half_bits* __restrict__ table,
    const long long* __restrict__ receipt_idx,
    const bool* __restrict__ flags,
    half_bits* __restrict__ observation,
    int actors, int batch, int first_tick
) {
    const offset_t index = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<offset_t>(batch) * 56) return;
    const offset_t item = index / 56;
    const int feature = index % 56;
    const int slot = feature / 28;
    const int within_slot = feature % 28;
    if (within_slot >= 11 && within_slot < 24) {
        const offset_t row = receipt_idx[item * 2 + slot];
        observation[index] = table[row * 13 + within_slot - 11];
    } else {
        half_bits value = frame[(item % actors) * 56 + feature];
        if (slot == 0 && (within_slot == 3 || within_slot == 4 || within_slot == 10)) {
            const bool incomplete = first_tick || !flags[item];
            // Multiplication, rather than assigning +0, preserves -0.
            value = to_half(from_half(value) * static_cast<float>(incomplete));
        }
        observation[index] = value;
    }
}

extern "C" __global__ void directed_world_action(
    const long long* __restrict__ actions,
    const unsigned char* __restrict__ stages,
    const long long* __restrict__ targets,
    int* __restrict__ counts,
    long long* __restrict__ receipt_idx,
    bool* __restrict__ flags,
    bool* __restrict__ outcomes,
    int batch, int actors, int requests, int request,
    int first_tick, int last_tick,
    int action_stride0, int action_stride1, int action_stride2
) {
    const offset_t item = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= batch) return;
    const int actor = item % actors;
    const offset_t action_base = (item / actors) * action_stride0
                                + static_cast<offset_t>(actor) * action_stride1;
    const long long act0 = actions[action_base];
    const long long act1 = actions[action_base + action_stride2];
    const bool attempt0 = act0 != 0, attempt1 = act1 != 0;
    const bool valid0 = act0 >= 0 && act0 < 11;
    const bool valid1 = act1 >= 0 && act1 < 11;
    bool complete = !first_tick && flags[item];
    bool had_error = !first_tick && flags[batch + item];
    bool covered = !first_tick && flags[2 * batch + item];
    bool version_right = !first_tick && flags[3 * batch + item];
    bool symbol_right = !first_tick && flags[4 * batch + item];
    bool timing_violation = !first_tick && flags[5 * batch + item];
    const unsigned char stage = stages[actor];
    // Both masks use the old complete flag, before this action is accepted.
    const bool active = (stage == 1 || stage == 2) && !complete;
    const bool ready = stage == 2 && !complete;
    const bool legal = ready && act0 >= 3 && valid0;
    const bool first_finish = legal && !covered;
    const bool accepted = ready && act0 == targets[actor];
    const bool rejected0 = attempt0 && !accepted;
    const bool mistimed = !ready && attempt0;
    if (first_finish) {
        // A legal action is in [3,10], so C++ integer division/remainder
        // matches Torch floor division/remainder without negative operands.
        const long long code = act0 - 3;
        const long long target_code = targets[actor] - 3;
        version_right = version_right || code / 4 == target_code / 4;
        symbol_right = symbol_right || code % 4 == target_code % 4;
    }
    complete = complete || accepted;
    had_error = had_error || (active && rejected0);
    covered = covered || legal;
    timing_violation = timing_violation || mistimed || attempt1;
    flags[item] = complete;
    flags[batch + item] = had_error;
    flags[2 * batch + item] = covered;
    flags[3 * batch + item] = version_right;
    flags[4 * batch + item] = symbol_right;
    flags[5 * batch + item] = timing_violation;
    counts[item] += static_cast<int>(!active && attempt0) + static_cast<int>(attempt1);
    counts[batch + item] += static_cast<int>(!active) + 1;
    counts[2 * batch + item] += static_cast<int>(rejected0) + static_cast<int>(attempt1);
    counts[3 * batch + item] += static_cast<int>(attempt0) + static_cast<int>(attempt1);
    counts[4 * batch + item] += static_cast<int>(mistimed);
    counts[5 * batch + item] += static_cast<int>(ready);
    counts[6 * batch + item] += static_cast<int>(attempt1);
    counts[7 * batch + item] += static_cast<int>(legal);
    receipt_idx[item * 2] = (valid0 ? act0 : 22) + (accepted ? 11 : 0);
    receipt_idx[item * 2 + 1] = valid1 ? act1 : 22;
    if (last_tick) {
        const offset_t base = static_cast<offset_t>(request) * batch + item;
        const offset_t stride = static_cast<offset_t>(requests) * batch;
        outcomes[base] = complete;
        outcomes[stride + base] = complete && !had_error;
        outcomes[2 * stride + base] = covered;
        outcomes[3 * stride + base] = covered && !timing_violation;
        outcomes[4 * stride + base] = version_right;
        outcomes[5 * stride + base] = symbol_right;
    }
}
