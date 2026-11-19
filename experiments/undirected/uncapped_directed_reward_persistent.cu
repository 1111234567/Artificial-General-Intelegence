// Append after uncapped_directed_reward_prepare.cu. The caller supplies one
// actor per CTA and publishes its rounded FP16 local activities as float[96].
// Each pointer is already offset to the actor row. Matrix and delta storage
// may reside in shared memory. The caller owns the barrier before consuming
// the updated traces, weights or matrix in a later phase of the same CTA.
// When the rollout omits weight-change diagnostics, both delta pointers may
// be null. Attempted and changed counts still accumulate.
// A supplied all_weights_finite is initialized by the caller for the full
// actor row. The hook clears it on the first nonfinite reward write.
template<bool store_deltas>
__device__ __forceinline__ unsigned persistent_receipt_pairs(
    const float* local, const int* active_pairs, int count, float scale, half_bits bound,
    half_bits* eligibility, half_bits* weights, half_bits* matrix, half_bits* deltas,
    bool track_nonfinite, bool& result_nonfinite
) {
    if (track_nonfinite) {
        return prepare_update_pairs<true, false, store_deltas, true>(
            local, active_pairs, count, scale, bound, eligibility, weights, matrix, deltas, &result_nonfinite);
    }
    return prepare_update_pairs<true, false, store_deltas>(
        local, active_pairs, count, scale, bound, eligibility, weights, matrix, deltas);
}

__device__ __forceinline__ void persistent_reward_update(
    const float* local, const int* active_pairs, int count, const bool* mask,
    half_bits* eligibility, half_bits* weights, half_bits* matrix, half_bits* deltas,
    bool receipt_possible, float receipt, float rate, float weight_limit,
    long long* attempted_writes, long long* changed_writes, float* weight_change,
    bool include_weight_change = true, bool* all_weights_finite = nullptr
) {
    const bool received = receipt != 0.0f;
    const bool weights_finite = all_weights_finite && *all_weights_finite;
    if (!receipt_possible || (!received && weights_finite)) {
        prepare_update_pairs<false, false>(local, active_pairs, count, 0.0f, 0,
                                           eligibility, weights, matrix, deltas);
        return;
    }

    unsigned counters;
    bool result_nonfinite = false;
    if (received) {
        const float scale = rate * receipt;
        const half_bits bound = to_half(weight_limit);
        if (include_weight_change) {
            counters = persistent_receipt_pairs<true>(local, active_pairs, count, scale, bound,
                eligibility, weights, matrix, deltas, weights_finite, result_nonfinite);
        } else {
            counters = persistent_receipt_pairs<false>(local, active_pairs, count, scale, bound,
                eligibility, weights, matrix, nullptr, weights_finite, result_nonfinite);
        }
    } else if (include_weight_change) {
        counters = prepare_update_pairs<false, true>(local, active_pairs, count, 0.0f, 0,
                                                     eligibility, weights, matrix, deltas);
    } else {
        counters = prepare_update_pairs<false, true, false>(local, active_pairs, count, 0.0f, 0,
                                                            eligibility, weights, matrix, nullptr);
    }
    if (!received && !__syncthreads_or(counters != 0)) return;

    constexpr unsigned nonfinite_bit = 1u << 31;
    __shared__ unsigned partials[32];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    counters = warp_sum(counters);
    if (weights_finite && __any_sync(0xffffffffu, result_nonfinite)) counters |= nonfinite_bit;
    if (lane == 0) partials[warp] = counters;
    __syncthreads();
    if (warp == 0) {
        const unsigned partial = lane < blockDim.x / 32 ? partials[lane] : 0;
        const bool lost_finite = weights_finite && __any_sync(0xffffffffu, partial & nonfinite_bit);
        counters = warp_sum(partial & ~nonfinite_bit);
        if (lane == 0) {
            const unsigned primary = counters >> 16, changed = counters & 0xffffu;
            if (lost_finite) *all_weights_finite = false;
            if (received && primary) *attempted_writes += primary;
            if (changed) *changed_writes += changed;
            // All false-receipt contributions are +0 or canonical NaN.
            if (include_weight_change && !received && primary) {
                *weight_change = __fadd_rn(*weight_change, from_half(0x7e00u));
            }
        }
    }
    if (include_weight_change && received) prepare_sum_deltas(deltas, mask, weight_change);
}
