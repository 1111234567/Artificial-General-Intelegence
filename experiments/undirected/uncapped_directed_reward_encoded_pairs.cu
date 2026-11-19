// Packed active-pair entry: pair ID, two endpoint IDs, and two direction bits.
// The sparse updater retains the original per-arc arithmetic and diagnostics.
constexpr unsigned pair_bits = ENCODED_PAIR_BITS;
constexpr unsigned node_bits = ENCODED_NODE_BITS;
constexpr unsigned u_shift = pair_bits;
constexpr unsigned v_shift = u_shift + node_bits;
constexpr unsigned forward_bit = 1u << (v_shift + node_bits);
constexpr unsigned reverse_bit = forward_bit << 1;
constexpr unsigned pair_mask = (1u << pair_bits) - 1;
constexpr unsigned node_mask = (1u << node_bits) - 1;

extern "C" __global__ void directed_encoded_build_pairs(
    const bool* __restrict__ mask,
    const int* __restrict__ metadata,
    const bool* __restrict__ changed,
    int* __restrict__ active_pairs,
    int* __restrict__ active_pair_counts,
    int pairs
) {
    const unsigned actor = blockIdx.x;
    if (changed && !changed[actor]) return;

    const auto list_base = static_cast<unsigned long long>(actor) * pairs;
    const auto arc_base = 2 * list_base;
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();

    const unsigned lane = threadIdx.x % 32;
    for (int tile = 0; tile < pairs; tile += blockDim.x) {
        const int pair = tile + threadIdx.x;
        const bool forward = pair < pairs && mask[arc_base + pair];
        const bool reverse = pair < pairs && mask[arc_base + pairs + pair];
        const unsigned ballot = __ballot_sync(0xffffffffu, forward || reverse);
        int start = 0;
        if (lane == 0 && ballot) start = atomicAdd(&count, __popc(ballot));
        start = __shfl_sync(0xffffffffu, start, 0);
        if (forward || reverse) {
            const unsigned preceding = ballot & ((1u << lane) - 1u);
            const unsigned directions = (forward ? forward_bit : 0u) | (reverse ? reverse_bit : 0u);
            active_pairs[list_base + start + __popc(preceding)] = metadata[pair] | directions;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) active_pair_counts[actor] = count;
}
