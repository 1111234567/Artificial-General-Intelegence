__device__ __forceinline__ long long post_answer_warp_sum(long long value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void private_symbol_post_answer(
    const long long* __restrict__ actions, const unsigned char* __restrict__ stages,
    const long long* __restrict__ targets, long long* __restrict__ counts,
    int requests, int ticks, int population, int actors
) {
    const int candidate = blockIdx.x;
    const long long action_stride = static_cast<long long>(population) * actors;
    const long long items = static_cast<long long>(requests) * actors;
    long long total = 0;
    // Each thread handles an actor/request, including its tail up to the next NEW.
    for (long long item = threadIdx.x; item < items; item += blockDim.x) {
        const int request = item / actors;
        const int actor = item % actors;
        const long long first_frame = static_cast<long long>(request) * ticks;
        const long long action_item = static_cast<long long>(candidate) * actors + actor;
        const auto target_symbol = (static_cast<unsigned long long>(targets[item])
                                    - POST_ANSWER_FIRST_FINISH) % POST_ANSWER_SYMBOLS;
        bool complete = false;
        for (int tick = 0; tick < ticks; ++tick) {
            const long long frame = first_frame + tick;
            const long long action = actions[frame * action_stride + action_item];
            if (complete) {
                total += action != POST_ANSWER_WAIT;
            } else if (stages[frame * actors + actor] == 2 && action >= POST_ANSWER_FIRST_FINISH
                       && action < POST_ANSWER_ACTIONS) {
                complete = (action - POST_ANSWER_FIRST_FINISH) % POST_ANSWER_SYMBOLS == target_symbol;
            }
        }
        if (complete && request + 1 < requests) {
            for (int tick = 0; tick < ticks; ++tick) {
                const long long frame = first_frame + ticks + tick;
                if (stages[frame * actors + actor] == 0) {
                    total += actions[frame * action_stride + action_item] != POST_ANSWER_WAIT;
                }
            }
        }
    }

    __shared__ long long warp_counts[POST_ANSWER_THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    total = post_answer_warp_sum(total);
    if (lane == 0) warp_counts[warp] = total;
    __syncthreads();
    if (warp == 0) {
        total = post_answer_warp_sum(lane < POST_ANSWER_THREADS / 32 ? warp_counts[lane] : 0);
        if (lane == 0) counts[candidate] = total;
    }
}
