// This source is appended to the guarded compact-updater derivative.
// PyTorch 2.11.0/cu130 sum_functor<Half,float,float>, vt0=4/input_vec=8:
// 9120 contiguous half values, aligned rows, >=16 outputs => (32,16),
// input_mult=(1,32,0), step_input=512, no inter-CTA reduction.

__device__ __forceinline__ float sparse_delta_value(
    offset_t edge, bool receipt, const half_bits* deltas,
    const bool* mask, const half_bits* weights
) {
    if (!mask[edge]) return 0.0f;
    if (receipt) return from_half(deltas[edge]);
    // The compact false-receipt path writes canonical half NaN for old
    // NaN and Inf weights, zero otherwise. Reconstruct it without clearing
    // or reading stale delta slots. This path is entered only if needed.
    return (weights[edge] & 0x7c00u) == 0x7c00u ? from_half(0x7e00u) : 0.0f;
}

extern "C" __global__ void directed_sparse_diagnostic_sum(
    const half_bits* __restrict__ deltas,
    const bool* __restrict__ mask,
    const half_bits* __restrict__ weights,
    const bool* __restrict__ event,
    const bool* __restrict__ nonfinite_flags,
    float* __restrict__ weight_change
) {
    const unsigned actor = blockIdx.x;
    const bool receipt = event[actor];
    // A full CTA takes the same branch, before the synchronization points.
    if (!receipt && !nonfinite_flags[actor]) return;

    const offset_t base = static_cast<offset_t>(actor) * 9120;
    const unsigned lane = threadIdx.x;
    const unsigned warp = threadIdx.y;
    const unsigned thread = lane + warp * 32;
    float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned vector = thread; vector < 1140; vector += 512) {
        #pragma unroll
        for (int element = 0; element < 8; ++element) {
            const float value = sparse_delta_value(
                base + vector * 8 + element, receipt, deltas, mask, weights
            );
            accumulators[element] = __fadd_rn(accumulators[element], value);
        }
    }
    float value = accumulators[0];
    #pragma unroll
    for (int element = 1; element < 8; ++element) {
        value = __fadd_rn(value, accumulators[element]);
    }
    // Reduce.cuh block_x_reduce: descending 16,8,4,2,1.
    #pragma unroll
    for (int distance = 16; distance > 0; distance >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, distance));
    }

    __shared__ float warp_values[16];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (warp == 0) {
        // Only lane zero's final value is stored. This lane follows exactly
        // block_y_reduce's 8,4,2,1 addition tree for the 16 warp partials.
        value = lane < 16 ? warp_values[lane] : 0.0f;
        #pragma unroll
        for (int distance = 8; distance > 0; distance >>= 1) {
            value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, distance));
        }
        if (lane == 0) {
            weight_change[actor] = __fadd_rn(weight_change[actor], value);
        }
    }
}
