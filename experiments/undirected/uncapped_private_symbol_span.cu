// One CUDA block owns an actor between topology updates. The recurrent matrix
// remains on-chip through the span; weights and diagnostics retain their
// original persistent global layout for the support ticks and journal.

__device__ __forceinline__ int span_best_action(const half_bits* values) {
    int best_index = 0;
    float best = from_half(values[0]);
    for (int index = 1; index < 11; ++index) {
        const float value = from_half(values[index]);
        if ((value != value && best == best) || value > best) {
            best_index = index;
            best = value;
        }
    }
    return best_index;
}

__device__ __forceinline__ void advance_span_eligibility(
    const half_bits history[][96], int start, int stop, const int* active, int count, half_bits* eligibility
) {
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const unsigned metadata = active[index];
        const int pair = metadata & pair_mask;
        const int u = (metadata >> u_shift) & node_mask, v = (metadata >> v_shift) & node_mask;
        const int primary = metadata & forward_bit ? pair : pair + 4560;
        const bool both = (metadata & (forward_bit | reverse_bit)) == (forward_bit | reverse_bit);
        half_bits first = eligibility[primary];
        half_bits second = both ? eligibility[pair + 4560] : 0;
        const bool separate = both && second != first;
        for (int tick = start; tick < stop; ++tick) {
            const float coactive = from_half(to_half(from_half(history[tick][u]) * from_half(history[tick][v])));
            first = lerp(first, coactive);
            if (separate) second = lerp(second, coactive);
        }
        eligibility[primary] = first;
        if (both) eligibility[pair + 4560] = separate ? second : first;
    }
}

__device__ __forceinline__ void advance_span_node(
    int node, half_bits current, const half_bits* bias, const half_bits* obs,
    half_bits* state, half_bits* activation, const half_bits* tanh_table
) {
    const float decayed = from_half(to_half(from_half(state[node]) * .75f));
    const float drive = from_half(to_half(__fadd_rn(from_half(current), from_half(bias[node]))));
    half_bits updated = to_half(__fmaf_rn(drive, .25f, decayed));
    if (node < 56) updated = to_half(__fmaf_rn(from_half(obs[node]), .25f, from_half(updated)));
    state[node] = updated;
    activation[node] = SPAN_TABLE ? __ldg(tanh_table + updated) : to_half(tanhf(from_half(updated)));
}

extern "C" __global__ void directed_private_span(
    const half_bits* __restrict__ public_frames,
    const unsigned char* __restrict__ stages,
    const long long* __restrict__ targets,
    const half_bits* __restrict__ receipts,
    const half_bits* __restrict__ tanh_table,
    int* __restrict__ counts,
    long long* __restrict__ receipt_indices,
    bool* __restrict__ flags,
    bool* __restrict__ outcomes,
    long long* __restrict__ raw_actions,
    const half_bits* __restrict__ all_bias,
    half_bits* __restrict__ all_matrix,
    half_bits* __restrict__ all_h,
    half_bits* __restrict__ all_activation,
    half_bits* __restrict__ all_trace,
    bool* __restrict__ all_tried,
    bool* __restrict__ all_pending,
    half_bits* __restrict__ all_reward,
    long long* __restrict__ all_reward_events,
    const float* __restrict__ rates,
    const int* __restrict__ all_active_pairs,
    const int* __restrict__ pair_counts,
    const bool* __restrict__ all_mask,
    half_bits* __restrict__ all_eligibility,
    half_bits* __restrict__ all_weight,
    half_bits* __restrict__ all_deltas,
    long long* __restrict__ all_attempted,
    long long* __restrict__ all_changes,
    float* __restrict__ all_weight_change,
    int batch, int actors, int requests, int ticks, int first_step, int stop_step,
    int first_receipt, float weight_limit
) {
    constexpr int nodes = 96, arcs = 9120, pairs = 4560;
    const int item = blockIdx.x;
    const int actor = item % actors;
    const int genome = item / actors;
    const int thread = threadIdx.x;
    const auto node_start = static_cast<offset_t>(item) * nodes;
    const auto arc_start = static_cast<offset_t>(item) * arcs;
    const auto matrix_start = node_start * nodes;
    const auto* bias = all_bias + genome * nodes;
    const int* active = all_active_pairs + static_cast<offset_t>(item) * pairs;
    const int active_count = pair_counts[item];
    constexpr int row_words = SPAN_MATRIX_STRIDE / 2;
    __shared__ __align__(4) half_bits dense[nodes * SPAN_MATRIX_STRIDE];
    __shared__ __align__(4) half_bits activation[SPAN_FUSED_ROWS ? 2 : 1][nodes];
    __shared__ half_bits state[nodes], trace[nodes], current[nodes], obs[56];
    __shared__ half_bits local_deltas[SPAN_SHARED_DELTAS ? arcs : 1];
    __shared__ half_bits history[SPAN_LAZY ? 11 : 1][nodes];
    __shared__ float local[nodes], reward;
    __shared__ half_bits receipt_bits;
    __shared__ bool tried, pending, matrix_dirty, matrix_finite;
    __shared__ PrivateWorldState world_state;
    auto* matrix_words = reinterpret_cast<unsigned*>(dense);
    auto* global_words = reinterpret_cast<unsigned*>(all_matrix + matrix_start);
    half_bits* deltas = SPAN_SHARED_DELTAS ? local_deltas : all_deltas + arc_start;
    bool nonfinite = false;
    for (int word = thread; word < nodes * nodes / 2; word += blockDim.x) {
        const unsigned bits = global_words[word];
        matrix_words[word / 48 * row_words + word % 48] = bits;
        nonfinite |= (bits & 0x7c00u) == 0x7c00u || (bits & 0x7c000000u) == 0x7c000000u;
    }
    if (thread < nodes) {
        state[thread] = all_h[node_start + thread];
        activation[0][thread] = all_activation[node_start + thread];
        trace[thread] = all_trace[node_start + thread];
    }
    if (thread == 0) {
        world_state = private_world_load_batches(counts, flags, receipt_indices, item, batch,
                                                 first_step % ticks == 0);
        tried = all_tried[item];
        pending = all_pending[item];
        matrix_dirty = false;
    }
    const bool finite = !__syncthreads_or(nonfinite);
    if (thread == 0) matrix_finite = finite;

    int reward_events = 0;
    int eligible_since = 0;
    for (int step = first_step; step < stop_step; ++step) {
        const int tick = step % ticks;
        const int request = step / ticks;
        const int span_tick = step - first_step;
        half_bits* current_activation = activation[SPAN_FUSED_ROWS ? span_tick % 2 : 0];
        half_bits* next_activation = activation[SPAN_FUSED_ROWS ? (span_tick + 1) % 2 : 0];
        const auto frame_actor = static_cast<offset_t>(step) * actors + actor;
        if (thread < 56) {
            obs[thread] = private_world_observe(world_state, public_frames + frame_actor * 56,
                                                receipts, thread, tick == 0);
        }
        if (thread < nodes) {
            const half_bits rounded = to_half(4.0f * from_half(state[thread]));
            const half_bits activity_bits = SPAN_TABLE ? __ldg(tanh_table + rounded)
                                                     : to_half(tanhf(from_half(rounded)));
            const float activity = from_half(activity_bits);
            if (SPAN_LAZY) history[span_tick][thread] = activity_bits;
            local[thread] = activity;
            const float decayed = from_half(to_half(from_half(trace[thread]) * .95f));
            trace[thread] = to_half(__fmaf_rn(activity, .05f, decayed));
        }
        if (thread == 0) {
            // These two features always come from the primary receipt row.
            const auto* received = receipts + world_state.receipt[0] * 13;
            const float receipt_delta = from_half(to_half(__fsub_rn(from_half(received[11]), from_half(received[12]))));
            receipt_bits = to_half(receipt_delta * static_cast<float>(pending));
            reward = from_half(receipt_bits);
            reward_events += reward != 0.0f;
            matrix_dirty |= reward != 0.0f;
        }
        __syncthreads();

        const bool receipt_possible = tick >= first_receipt;
        if (!SPAN_LAZY || (receipt_possible && (reward != 0.0f || !matrix_finite))) {
            if (SPAN_LAZY && eligible_since < span_tick) {
                advance_span_eligibility(history, eligible_since, span_tick, active, active_count,
                                         all_eligibility + arc_start);
                // The writer visits the same pair indices in the same lanes.
            }
            persistent_reward_update(local, active, active_count, all_mask + arc_start,
                all_eligibility + arc_start, all_weight + arc_start, dense, deltas,
                receipt_possible, reward, rates[genome], weight_limit,
                all_attempted + item, all_changes + item, all_weight_change + item, true, &matrix_finite);
            eligible_since = span_tick + 1;
            __syncthreads();
        }
        const auto* vector = reinterpret_cast<const unsigned*>(current_activation);
        if (SPAN_FUSED_ROWS) {
            if (thread < 192) {
                const auto product = multiply_row_pair<row_words>(matrix_words, vector, thread / 32, thread % 32);
                if (thread % 4 == 0) {
                    const int node = (thread / 32) * 16 + (thread % 32) / 4;
                    advance_span_node(node, to_half(product.first), bias, obs, state, next_activation, tanh_table);
                    advance_span_node(node + 8, to_half(product.second), bias, obs, state, next_activation, tanh_table);
                }
            }
        } else {
            if (thread < 192) {
                multiply_rows<row_words>(matrix_words, vector, current, thread / 32, thread % 32);
            }
            __syncthreads();
            if (thread < nodes) {
                advance_span_node(thread, current[thread], bias, obs, state, next_activation, tanh_table);
            }
        }
        __syncthreads();
        if (thread == 0) {
            const int primary = span_best_action(state + 56);
            const int spare = span_best_action(state + 67);
            const bool was_tried = tried && from_half(obs[0]) == 0.0f;
            pending = from_half(obs[10]) != 0.0f && !was_tried && primary >= 3 && primary < 11;
            tried = was_tried || pending;
            const long long target = targets[static_cast<offset_t>(request) * actors + actor];
            long long* out = raw_actions == nullptr ? nullptr : raw_actions + static_cast<offset_t>(step) * batch + item;
            private_world_action(world_state, primary, spare, stages[frame_actor], target, tick == 0, out);
            if (tick == ticks - 1) private_world_store_outcomes(world_state, outcomes, request, requests, item, batch);
        }
        __syncthreads();
    }

    if (SPAN_LAZY && eligible_since < stop_step - first_step) {
        advance_span_eligibility(history, eligible_since, stop_step - first_step,
                                 active, active_count, all_eligibility + arc_start);
    }
    if (matrix_dirty) {
        for (int word = thread; word < nodes * nodes / 2; word += blockDim.x) {
            global_words[word] = matrix_words[word / 48 * row_words + word % 48];
        }
    }
    if (thread < nodes) {
        all_h[node_start + thread] = state[thread];
        all_activation[node_start + thread] = activation[SPAN_FUSED_ROWS ? (stop_step - first_step) % 2 : 0][thread];
        all_trace[node_start + thread] = trace[thread];
    }
    if (thread == 0) {
        private_world_store_batches(world_state, counts, flags, receipt_indices, item, batch);
        all_tried[item] = tried;
        all_pending[item] = pending;
        all_reward[item] = receipt_bits;
        all_reward_events[item] += reward_events;
    }
}
