// Compact each warp's surviving arcs across consecutive loop iterations.
// Shared delta slots retain the reference lane and iteration for exact sums.
extern "C" __global__ __launch_bounds__(ARC_THREADS) void fast_local_arcs_compact(
    const int* __restrict__ clock,
    const int* __restrict__ indices,
    const float* __restrict__ local_activity,
    const float* __restrict__ node_trace,
    const float* __restrict__ gain,
    const float* __restrict__ receipts,
    const float* __restrict__ plastic,
    const float* __restrict__ legacy_rule,
    const float* __restrict__ reward_rate,
    const half_bits* __restrict__ initial_w,
    float* __restrict__ weight,
    float* __restrict__ gate,
    half_bits* __restrict__ eligibility,
    half_bits* __restrict__ strength,
    bool* __restrict__ mask,
    int* __restrict__ last_change,
    half_bits* __restrict__ effective,
    half_bits* __restrict__ matrix,
    long long* __restrict__ counts,
    float* __restrict__ totals,
    int actors, int nodes, int arcs
) {
    const unsigned actor = blockIdx.x;
    const unsigned genome = actor / actors;
    const unsigned batch = gridDim.x;
    const int step = *clock;
    if (step == 0) {
        if (threadIdx.x == 0)
            counts[COUNT_EDGE_TICKS * batch + actor] += counts[COUNT_EDGE_COUNT * batch + actor];
        return;
    }

    constexpr int rows = 4;
    constexpr unsigned full_warp = 0xffffffffu;
    constexpr int no_delta = (-2147483647 - 1);  // raw negative zero
    /* CACHE */
    __shared__ int tasks[fast_warps][rows * 32];
    __shared__ int deltas[fast_warps][rows * 32];
    auto* warp_tasks = tasks[warp];
    auto* warp_deltas = deltas[warp];

    for (int base = warp * 32; base < fast_arcs; base += rows * ARC_THREADS) {
        int task_count = 0;
        #pragma unroll 1
        for (int row = 0; row < rows; ++row) {
            const int arc = base + row * ARC_THREADS + lane;
            const int slot = row * 32 + lane;
            warp_deltas[slot] = no_delta;
            bool apply = false;
            if (arc < fast_arcs) {
                /* GATE */
                apply = live || (!(next_gate < BIRTH_THRESHOLD)
                               && step - last_change[i] >= COOLDOWN);
            }
            const unsigned enabled = __ballot_sync(full_warp, apply);
            if (apply) {
                const unsigned lower = (1u << lane) - 1u;
                warp_tasks[task_count + __popc(enabled & lower)] = arc | (slot << 14);
            }
            task_count += __popc(enabled);
        }
        __syncwarp(full_warp);
        for (int task = lane; task < task_count; task += 32) {
            const int packed = warp_tasks[task];
            const int arc = packed & 0x3fff;
            const int slot = packed >> 14;
            /* UPDATE */
        }
        __syncwarp(full_warp);
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            const int bits = warp_deltas[row * 32 + lane];
            if (bits != no_delta)
                change += __int_as_float(bits);
        }
        __syncwarp(full_warp);
    }

    const int outcomes[] = {births, deaths, attempted, changed, updates};
    #pragma unroll
    for (int field = 0; field < fast_count_fields; ++field) {
        const int sum = __reduce_add_sync(full_warp, outcomes[field]);
        if (lane == 0)
            block_stats[field * fast_warps + warp] = sum;
    }
    const float sum = warp_sum(change);
    if (lane == 0)
        block_stats[fast_count_fields * fast_warps + warp] = __float_as_int(sum);
    __syncthreads();
    fast_commit_arc_stats(block_stats, counts, totals, batch, actor);
}
