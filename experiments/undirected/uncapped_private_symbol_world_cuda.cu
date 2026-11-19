// One warp emits the 28 aligned FP16 pairs for an actor's two ports.
extern "C" __global__ void private_packed_observation(
    const half_bits* __restrict__ frame,
    const half_bits* __restrict__ table,
    const long long* __restrict__ receipt_idx,
    const bool* __restrict__ flags,
    half_bits* __restrict__ observation,
    int actors, int batch, int first_tick
) {
    const offset_t index = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const offset_t item = index / 32;
    const int pair = index % 32;
    if (item >= batch || pair >= 28) return;

    const int slot = pair >= 14;
    const int slot_pair = pair % 14;
    const auto public_pairs = reinterpret_cast<const unsigned int*>(frame);
    const offset_t public_index = (item % actors) * 28 + pair;
    unsigned int bits;
    if (slot_pair >= 6 && slot_pair < 12) {
        const offset_t row = receipt_idx[item * 2 + slot];
        const auto receipt = table + row * 13 + slot_pair * 2 - 11;
        bits = static_cast<unsigned int>(receipt[0]) | (static_cast<unsigned int>(receipt[1]) << 16);
    } else {
        bits = public_pairs[public_index];
        if (!slot && (slot_pair == 1 || slot_pair == 2 || slot_pair == 5)) {
            const bool incomplete = first_tick || !flags[item];
            // Multiply the same half as the sealed kernel, including NaNs and signed zero.
            if (slot_pair == 1) {
                const half_bits masked = to_half(from_half(bits >> 16) * static_cast<float>(incomplete));
                bits = (bits & 0xFFFFu) | (static_cast<unsigned int>(masked) << 16);
            } else {
                const half_bits masked = to_half(from_half(bits & 0xFFFFu) * static_cast<float>(incomplete));
                bits = (bits & 0xFFFF0000u) | masked;
            }
        }
        if (slot_pair == 5) {
            const offset_t row = receipt_idx[item * 2 + slot];
            bits = (bits & 0xFFFFu) | (static_cast<unsigned int>(table[row * 13]) << 16);
        }
    }
    reinterpret_cast<unsigned int*>(observation)[item * 28 + pair] = bits;
}
