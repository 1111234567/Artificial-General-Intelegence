// Actor-local private-symbol state for a CUDA kernel that runs several ticks.
// Initialize once per life. The same thread writes the state each tick. If
// other threads read it to observe, synchronize after initialization and action.
struct PrivateWorldState {
    int counts[8];
    unsigned char receipt[2];
    bool complete;
    bool had_error;
    bool covered;
    bool symbol_right;
    bool timing_violation;
};

using private_world_offset = unsigned long long;
using private_world_half_bits = unsigned short;

__device__ __forceinline__ private_world_half_bits private_world_mask(
    private_world_half_bits bits, bool visible
) {
    float value;
    private_world_half_bits result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    value *= static_cast<float>(visible);
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ private_world_half_bits private_world_observe(
    const PrivateWorldState& state,
    const private_world_half_bits* frame,
    const private_world_half_bits* table,
    int feature, bool first_tick
) {
    // frame points to the current tape actor's two 28-feature ports.
    const int slot = feature >= 28;
    const int within_slot = feature - slot * 28;
    if (within_slot >= 11 && within_slot < 24) {
        return table[state.receipt[slot] * 13 + within_slot - 11];
    }
    const auto bits = frame[feature];
    if (feature == 3 || feature == 4 || feature == 10) {
        // Match the sealed Half multiply, including NaNs and signed zero.
        return private_world_mask(bits, first_tick || !state.complete);
    }
    return bits;
}

template<bool score = true>
__device__ __forceinline__ void private_world_action(
    PrivateWorldState& state, long long primary, long long spare,
    unsigned char stage, long long target, bool first_tick,
    long long* trace_value = nullptr
) {
    // trace_value, when supplied, is the raw int64 [request,tick,population,actor] element.
    if (trace_value != nullptr) *trace_value = primary;
    const bool incomplete = first_tick || !state.complete;
    const bool ready = stage == 2 && incomplete;
    const bool valid_primary = primary >= 0 && primary < 11;
    const bool valid_spare = spare >= 0 && spare < 11;
    const bool legal = ready && primary >= 3 && valid_primary;
    const bool accepted = legal && (primary - 3) % 4 == (target - 3) % 4;
    if (score) {
        const bool attempt_primary = primary != 0;
        const bool attempt_spare = spare != 0;
        const bool active = (stage == 1 || stage == 2) && incomplete;
        const bool rejected = attempt_primary && !accepted;
        const bool mistimed = !ready && attempt_primary;
        const bool covered = !first_tick && state.covered;
        state.symbol_right = ((!first_tick && state.symbol_right) || (legal && !covered && accepted));
        state.had_error = (!first_tick && state.had_error) || (active && rejected);
        state.timing_violation = (!first_tick && state.timing_violation) || mistimed || attempt_spare;
        state.covered = covered || legal;

        state.counts[0] += static_cast<int>(!active && attempt_primary) + static_cast<int>(attempt_spare);
        state.counts[1] += static_cast<int>(!active) + 1;
        state.counts[2] += static_cast<int>(rejected) + static_cast<int>(attempt_spare);
        state.counts[3] += static_cast<int>(attempt_primary) + static_cast<int>(attempt_spare);
        state.counts[4] += static_cast<int>(mistimed);
        state.counts[5] += static_cast<int>(ready);
        state.counts[6] += static_cast<int>(attempt_spare);
        state.counts[7] += static_cast<int>(legal);
    }
    state.complete = !incomplete || accepted;
    state.receipt[0] = (valid_primary ? primary : 22) + (accepted ? 11 : 0);
    state.receipt[1] = valid_spare ? spare : 22;
}

__device__ __forceinline__ void private_world_store_outcomes(
    const PrivateWorldState& state, bool* outcomes,
    int request, int requests, int item, int batch
) {
    // Call once after the last action of each request.
    const private_world_offset base = static_cast<private_world_offset>(request) * batch + item;
    const private_world_offset stride = static_cast<private_world_offset>(requests) * batch;
    outcomes[base] = state.complete;
    outcomes[stride + base] = state.complete && !state.had_error;
    outcomes[2 * stride + base] = state.covered;
    outcomes[3 * stride + base] = state.covered && !state.timing_violation;
    // The private task's version outcome is true after any first legal answer.
    outcomes[4 * stride + base] = state.covered;
    outcomes[5 * stride + base] = state.symbol_right;
}

__device__ __forceinline__ void private_world_store_counts(
    const PrivateWorldState& state, int* counts, int item, int batch
) {
    // Counts span the entire life; write them once after its last action.
    #pragma unroll
    for (int counter = 0; counter < 8; ++counter) {
        counts[static_cast<private_world_offset>(counter) * batch + item] = state.counts[counter];
    }
}

template<bool score = true>
__device__ __forceinline__ PrivateWorldState private_world_load_batches(
    const int* counts, const bool* flags, const long long* receipt_idx,
    int item, int batch, bool first_tick = false
) {
    // Import the sealed rollout's buffers before a multi-tick span. The first
    // request can begin with uninitialized flags which must never be read.
    PrivateWorldState state{};
    const private_world_offset offset = item;
    const private_world_offset stride = batch;
    state.complete = !first_tick && flags[offset];
    state.receipt[0] = receipt_idx[2 * offset];
    state.receipt[1] = receipt_idx[2 * offset + 1];
    if (score) {
        state.had_error = !first_tick && flags[stride + offset];
        state.covered = !first_tick && flags[2 * stride + offset];
        state.symbol_right = !first_tick && flags[4 * stride + offset];
        state.timing_violation = !first_tick && flags[5 * stride + offset];
        #pragma unroll
        for (int counter = 0; counter < 8; ++counter) {
            state.counts[counter] = counts[counter * stride + offset];
        }
    }
    return state;
}

template<bool score = true>
__device__ __forceinline__ void private_world_store_batches(
    const PrivateWorldState& state, int* counts, bool* flags, long long* receipt_idx,
    int item, int batch
) {
    // Export state before a caller resumes the original one-tick world path.
    // Request outcomes are a separate write at the actual request boundary.
    const private_world_offset offset = item;
    const private_world_offset stride = batch;
    flags[offset] = state.complete;
    receipt_idx[2 * offset] = state.receipt[0];
    receipt_idx[2 * offset + 1] = state.receipt[1];
    if (score) {
        flags[stride + offset] = state.had_error;
        flags[2 * stride + offset] = state.covered;
        flags[3 * stride + offset] = state.covered;
        flags[4 * stride + offset] = state.symbol_right;
        flags[5 * stride + offset] = state.timing_violation;
        private_world_store_counts(state, counts, item, batch);
    }
}
