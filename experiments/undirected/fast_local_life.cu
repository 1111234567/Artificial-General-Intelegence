// Each block carries one life between its request boundaries. Arc state lives
// in global memory while node, matrix, genetic and world state stay on chip.
constexpr int life_nodes = 96;
constexpr int matrix_stride = 104;
constexpr int life_arcs = life_nodes * (life_nodes - 1);
constexpr int arc_warps = ARC_THREADS / 32;
constexpr int prepare_warps = NODE_THREADS / 32;
constexpr int arc_count_fields = 5;

struct LifeNodes {
    half_bits h[life_nodes];
    half_bits activation[2][life_nodes];
    half_bits observation[INPUTS];
    half_bits anchor[life_nodes];
    float bias[life_nodes];
    float trace[life_nodes];
    float activity[life_nodes];
    float gain[life_nodes];
};

struct LifeGenes {
    float plastic[PLASTIC_SIZE];
    float rules[2][RULE_SIZE];
    float reward_rate;
};

struct LifeReceipts {
    float ack, rej, reward;
    float legacy[2];
};

struct LifeDiagnostics {
    long long count[COUNT_FIELDS];
    float total[TOTAL_FIELDS];
    int bias_updates[prepare_warps];
    float bias_changes[prepare_warps];
    int arc_counts[arc_count_fields][arc_warps];
    float weight_changes[arc_warps];
};

__device__ __forceinline__ void prepare_life_nodes(
    LifeNodes& node, const LifeGenes& genes, LifeReceipts& receipts,
    LifeDiagnostics& stats, const PrivateWorldState& world, const half_bits* frame,
    const half_bits* table, bool started, bool first_tick, bool pending, bool modulated
) {
    const int thread = threadIdx.x;
    const auto* gene = genes.plastic;
    if (thread < INPUTS)
        node.observation[thread] = private_world_observe(world, frame, table, thread, first_tick);
    constexpr int first_receipt_feature = 11;
    const auto* primary = table + world.receipt[0] * 13;
    const auto* spare = table + world.receipt[1] * 13;
    const float primary_ack = from_half(primary[ACK - first_receipt_feature]);
    const float primary_rej = from_half(primary[REJ - first_receipt_feature]);
    const float ack = started ? primary_ack + SPARE_RECEIPT_RATE * from_half(spare[ACK - first_receipt_feature]) : 0.f;
    const float rej = started ? primary_rej + SPARE_RECEIPT_RATE * from_half(spare[REJ - first_receipt_feature]) : 0.f;
    if (thread == 0) {
        const float reward = started && pending ? primary_ack - primary_rej : 0.f;
        float begin_ack = 0.f, finish_ack = 0.f, begin_rej = 0.f, finish_rej = 0.f;
        if (started) {
            for (int job = 0; job < JOBS; ++job) {
                const auto* port = table + world.receipt[job] * 13;
                const float begin = from_half(port[12 - first_receipt_feature]) + from_half(port[13 - first_receipt_feature]);
                const float job_ack = from_half(port[ACK - first_receipt_feature]);
                const float job_rej = from_half(port[REJ - first_receipt_feature]);
                float finish = 0.f;
                for (int i = 14; i < 22; ++i)
                    finish += from_half(port[i - first_receipt_feature]);
                begin_ack += begin * job_ack;
                finish_ack += finish * job_ack;
                begin_rej += begin * job_rej;
                finish_rej += finish * job_rej;
            }
        }
        receipts.ack = ack;
        receipts.rej = rej;
        receipts.reward = reward;
        for (int row = 0; row < 2; ++row) {
            const auto* rule = genes.rules[row];
            receipts.legacy[row] = rule[5] * (begin_ack / JOBS) + rule[6] * (finish_ack / JOBS)
                                 + rule[7] * (begin_rej / JOBS) + rule[8] * (finish_rej / JOBS);
        }
        stats.count[COUNT_REWARD_EVENTS] += reward != 0.f;
        stats.count[COUNT_PUBLIC_ACK_EVENTS] += ack != 0.f;
        stats.count[COUNT_PUBLIC_REJ_EVENTS] += rej != 0.f;
    }
    if (thread < NODE_THREADS) {
        int updates = 0;
        float change = 0;
        if (thread < life_nodes) {
            const float activity = tanhf(4.f * from_half(node.h[thread]));
            const float trace_old = node.trace[thread];
            const float trace = started ? trace_old + (activity - trace_old) * TRACE_RATE : trace_old;
            const float gain = modulated ? 1.f + MOD_SCALE * tanhf(gene[MOD_ACTIVITY] * activity
                                                                 + gene[MOD_TRACE] * trace) : 1.f;
            node.activity[thread] = activity;
            node.trace[thread] = trace;
            node.gain[thread] = gain;
            if (started) {
                const float old = node.bias[thread];
                const float anchor = from_half(node.anchor[thread]);
                const float local = BIAS_RATE * (gene[BIAS_ACTIVITY] * activity + gene[BIAS_TRACE] * trace)
                                  + BIAS_RECEIPT_RATE * (gene[BIAS_ACK] * ack + gene[BIAS_REJ] * rej) * trace;
                const float delta = gain * local - BIAS_RATE * gene[BIAS_DECAY] * (old - anchor);
                const float updated = clip(old + delta, anchor - BIAS_LIMIT, anchor + BIAS_LIMIT);
                node.bias[thread] = updated;
                const float effective_old = from_half(to_half(old));
                const float effective_new = from_half(to_half(updated));
                updates += effective_new != effective_old;
                change += fabsf(effective_new - effective_old);
            }
        }
        updates = warp_sum(updates);
        change = warp_sum(change);
        if (thread % 32 == 0) {
            stats.bias_updates[thread / 32] = updates;
            stats.bias_changes[thread / 32] = change;
        }
    }
    __syncthreads();
    if (thread == 0) {
        int updates = stats.bias_updates[0];
        float change = stats.bias_changes[0];
        #pragma unroll
        for (int warp = 1; warp < prepare_warps; ++warp) {
            updates += stats.bias_updates[warp];
            change += stats.bias_changes[warp];
        }
        stats.count[COUNT_BIAS_UPDATES] += updates;
        stats.total[TOTAL_BIAS_CHANGE] += change;
    }
}

__device__ __forceinline__ void advance_life_arcs(
    const LifeNodes& node, const LifeGenes& genes, const LifeReceipts& receipts,
    LifeDiagnostics& stats, const int* indices, const half_bits* baseline,
    float* weight, float* gate, half_bits* eligibility, half_bits* strength,
    bool* mask, int* last_change, half_bits* effective, half_bits* matrix, int step
) {
    if (step == 0) {
        if (threadIdx.x == 0)
            stats.count[COUNT_EDGE_TICKS] += stats.count[COUNT_EDGE_COUNT];
        return;
    }
    const auto* gene = genes.plastic;
    const auto* birth_rule = genes.rules[0];
    const auto* prior = genes.rules[1];
    const float ack = receipts.ack;
    const float rej = receipts.rej;
    const float reward = receipts.reward;
    const float legacy_birth_trace = birth_rule[3] + receipts.legacy[0];
    const float legacy_trace = prior[3] + receipts.legacy[1];
    const float weight_receipt = genes.reward_rate * reward
                               + WEIGHT_RECEIPT_RATE * (gene[WEIGHT_ACK] * ack + gene[WEIGHT_REJ] * rej);
    const float gate_receipt = gene[GATE_ACK] * ack + gene[GATE_REJ] * rej;
    int births = 0, deaths = 0, attempted = 0, changed = 0, updates = 0;
    float change = 0;
    for (int arc = threadIdx.x; arc < life_arcs; arc += ARC_THREADS) {
        const int flat = indices[arc];
        const int dest = flat / life_nodes;
        const int src = flat % life_nodes;
        const float pre = node.trace[src];
        const float source_activity = node.activity[src];
        const float post = node.activity[dest];
        const float coactivity = pre * post;
        const float product = source_activity * post;
        const float trace_product = pre * node.trace[dest];
        const float endpoint = fabsf(source_activity - post);
        const float modulation = node.gain[dest];
        const bool live = mask[arc];
        const float old = live ? weight[arc] : 0.f;
        const float legacy_support = prior[0] + prior[1] * (old / WEIGHT_LIMIT) + prior[2] * product
                                   + legacy_trace * trace_product + prior[4] * endpoint;
        const float support = tanhf(legacy_support) + gene[GATE_BIAS] + gene[GATE_COACTIVITY] * coactivity
                            + gene[GATE_TRACE] * trace_product
                            + gene[GATE_WEIGHT] * (fabsf(old) / WEIGHT_LIMIT) + gate_receipt * coactivity;
        const float previous = gate[arc];
        const float next_gate = previous + (clip(modulation * support, -1.f, 1.f) - previous) * GATE_RATE;
        gate[arc] = next_gate;
        if (!live && next_gate < BIRTH_THRESHOLD)
            continue;
        const bool ready = step - last_change[arc] >= COOLDOWN;
        if (!live && !ready)
            continue;
        float new_weight;
        float new_strength;
        if (!live) {
            const float drive = tanhf(birth_rule[0] + birth_rule[2] * product
                                     + legacy_birth_trace * trace_product + birth_rule[4] * endpoint);
            new_weight = from_half(to_half(SEED_RANGE * drive + (drive >= 0.f ? SEED_OFFSET : -SEED_OFFSET)));
            new_strength = RAMP_RATE;
            eligibility[arc] = to_half(0.f);
            mask[arc] = true;
            last_change[arc] = step;
            ++births;
        } else {
            const float old_strength = from_half(strength[arc]);
            const bool fade = ready && next_gate <= DEATH_THRESHOLD;
            new_strength = clip(old_strength + (fade ? -RAMP_RATE : RAMP_RATE), 0.f, 1.f);
            if (new_strength == 0.f) {
                new_weight = 0.f;
                eligibility[arc] = to_half(0.f);
                mask[arc] = false;
                last_change[arc] = step;
                ++deaths;
            } else {
                const float e = from_half(eligibility[arc]);
                const half_bits next_eligibility = to_half(e + ELIGIBILITY_RATE * (product - e));
                eligibility[arc] = next_eligibility;
                const float receipt = weight_receipt * from_half(next_eligibility);
                const float rest = WEIGHT_RATE * gene[WEIGHT_DECAY] * (old - from_half(baseline[arc]));
                const float local = WEIGHT_RATE * (gene[WEIGHT_PRE] * pre + gene[WEIGHT_POST] * post
                                                   + gene[WEIGHT_COACTIVITY] * coactivity);
                new_weight = clip(old + (modulation * (local + receipt) - rest), -WEIGHT_LIMIT, WEIGHT_LIMIT);
                const float intrinsic = from_half(to_half(new_weight * old_strength));
                const float visible = from_half(effective[arc]);
                if (receipt != 0.f) {
                    const float no_receipt = clip(old + (modulation * local - rest), -WEIGHT_LIMIT, WEIGHT_LIMIT);
                    const float control = from_half(to_half(no_receipt * old_strength));
                    ++attempted;
                    changed += intrinsic != control;
                }
                updates += intrinsic != visible;
                change += fabsf(intrinsic - visible);
            }
        }
        weight[arc] = new_weight;
        strength[arc] = to_half(new_strength);
        const half_bits written = to_half(new_weight * new_strength);
        if (written != effective[arc]) {
            effective[arc] = written;
            matrix[dest * matrix_stride + src] = written;
        }
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int outcomes[arc_count_fields] = {births, deaths, attempted, changed, updates};
    #pragma unroll
    for (int field = 0; field < arc_count_fields; ++field) {
        const int sum = warp_sum(outcomes[field]);
        if (lane == 0)
            stats.arc_counts[field][warp] = sum;
    }
    const float sum = warp_sum(change);
    if (lane == 0)
        stats.weight_changes[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int field = 0; field < arc_count_fields; ++field) {
            outcomes[field] = 0;
            #pragma unroll
            for (int warp_id = 0; warp_id < arc_warps; ++warp_id)
                outcomes[field] += stats.arc_counts[field][warp_id];
        }
        const int fields[arc_count_fields] = {COUNT_BIRTHS, COUNT_DEATHS, COUNT_ATTEMPTED_WRITES,
                                             COUNT_CHANGED_WRITES, COUNT_WEIGHT_UPDATES};
        #pragma unroll
        for (int field = 0; field < arc_count_fields; ++field)
            stats.count[fields[field]] += outcomes[field];
        auto& edges = stats.count[COUNT_EDGE_COUNT];
        edges += outcomes[0] - outcomes[1];
        stats.count[COUNT_EDGE_TICKS] += edges;
        change = 0;
        #pragma unroll
        for (int warp_id = 0; warp_id < arc_warps; ++warp_id)
            change += stats.weight_changes[warp_id];
        stats.total[TOTAL_WEIGHT_CHANGE] += change;
    }
}

__device__ __forceinline__ void finish_life_node(LifeNodes& node, int index, int next, half_bits product) {
    const float decayed = from_half(to_half(from_half(node.h[index]) * .75f));
    const float drive = from_half(to_half(from_half(product) + from_half(to_half(node.bias[index]))));
    half_bits updated = to_half(__fmaf_rn(drive, .25f, decayed));
    if (index < INPUTS)
        updated = to_half(__fmaf_rn(from_half(node.observation[index]), .25f, from_half(updated)));
    node.h[index] = updated;
    node.activation[next][index] = to_half(tanhf(from_half(updated)));
}

__device__ __forceinline__ int life_best_action(const half_bits* ports) {
    int best_index = 0;
    float best = from_half(ports[0]);
    for (int index = 1; index < ACTIONS; ++index) {
        const float candidate = from_half(ports[index]);
        if ((candidate != candidate && best == best) || candidate > best) {
            best = candidate;
            best_index = index;
        }
    }
    return best_index;
}

extern "C" __global__ __launch_bounds__(ARC_THREADS, 2) void fast_local_life(
    const half_bits* __restrict__ public_frames, const unsigned char* __restrict__ stage,
    const long long* __restrict__ targets, const half_bits* __restrict__ table,
    long long* __restrict__ trace, long long* __restrict__ spare_trace, bool* __restrict__ outcomes,
    int* __restrict__ world_counts, bool* __restrict__ world_flags,
    long long* __restrict__ world_receipts, int* __restrict__ clock,
    const int* __restrict__ indices, const float* __restrict__ plastic,
    const float* __restrict__ rules, const float* __restrict__ reward_rate,
    const half_bits* __restrict__ initial_bias, const half_bits* __restrict__ initial_weights,
    half_bits* __restrict__ h, half_bits* __restrict__ activation, float* __restrict__ node_bias,
    float* __restrict__ node_trace, float* __restrict__ node_activity, float* __restrict__ gain,
    float* __restrict__ policy_receipts, bool* __restrict__ tried, bool* __restrict__ pending,
    float* __restrict__ weights, float* __restrict__ gate, half_bits* __restrict__ eligibility,
    half_bits* __restrict__ strength, bool* __restrict__ mask, int* __restrict__ last_change,
    half_bits* __restrict__ effective, unsigned* __restrict__ matrix,
    long long* __restrict__ counts, float* __restrict__ totals,
    int actors, int requests, int ticks, int start, int stop, int modulated
) {
    const int item = blockIdx.x, batch = gridDim.x;
    const int actor = item % actors, genome = item / actors;
    const int thread = threadIdx.x;
    const auto node_base = static_cast<offset_t>(item) * life_nodes;
    const auto arc_base = static_cast<offset_t>(item) * life_arcs;
    const auto* baseline = initial_weights + static_cast<offset_t>(genome) * life_arcs;
    auto* matrix_words = matrix + node_base * (life_nodes / 2);
    __shared__ unsigned dense[life_nodes * matrix_stride / 2];
    __shared__ __align__(4) LifeNodes node;
    __shared__ LifeGenes genes;
    __shared__ LifeReceipts receipts;
    __shared__ LifeDiagnostics stats;
    __shared__ PrivateWorldState world;
    bool was_tried = false, expect_receipt = false;
    if (thread == 0) {
        world = private_world_load_batches(world_counts, world_flags, world_receipts, item, batch, true);
        was_tried = tried[item];
        expect_receipt = pending[item];
        genes.reward_rate = reward_rate[genome];
    }
    if (thread < life_nodes) {
        node.h[thread] = h[node_base + thread];
        node.activation[0][thread] = activation[node_base + thread];
        node.bias[thread] = node_bias[node_base + thread];
        node.trace[thread] = node_trace[node_base + thread];
        node.anchor[thread] = initial_bias[genome * life_nodes + thread];
    }
    if (thread < PLASTIC_SIZE)
        genes.plastic[thread] = plastic[genome * PLASTIC_SIZE + thread];
    if (thread < 2 * RULE_SIZE)
        genes.rules[thread / RULE_SIZE][thread % RULE_SIZE] = rules[genome * 2 * RULE_SIZE + thread];
    if (thread < COUNT_FIELDS)
        stats.count[thread] = counts[thread * batch + item];
    if (thread < TOTAL_FIELDS)
        stats.total[thread] = totals[thread * batch + item];
    for (int word = thread; word < life_nodes * (life_nodes / 2); word += ARC_THREADS) {
        dense[word / (life_nodes / 2) * (matrix_stride / 2) + word % (life_nodes / 2)] = matrix_words[word];
    }
    __syncthreads();

    int current = 0;
    for (int request = start; request < stop; ++request) {
        const long long target = targets[static_cast<offset_t>(request) * actors + actor];
        for (int tick = 0; tick < ticks; ++tick) {
            const int step = request * ticks + tick;
            const auto frame_item = static_cast<offset_t>(step) * actors + actor;
            prepare_life_nodes(node, genes, receipts, stats, world, public_frames + frame_item * INPUTS,
                               table, step != 0, tick == 0, expect_receipt, modulated);
            advance_life_arcs(node, genes, receipts, stats, indices, baseline, weights + arc_base,
                              gate + arc_base, eligibility + arc_base, strength + arc_base, mask + arc_base,
                              last_change + arc_base, effective + arc_base, reinterpret_cast<half_bits*>(dense), step);

            const int next = current ^ 1;
            if (thread < 192) {
                const int warp = thread / 32, lane = thread % 32;
                const auto product = multiply_row_pair<matrix_stride / 2>(
                    dense, reinterpret_cast<const unsigned*>(node.activation[current]), warp, lane);
                if (lane % 4 == 0) {
                    const int row = warp * 16 + lane / 4;
                    finish_life_node(node, row, next, rounded_half(product.first));
                    finish_life_node(node, row + 8, next, rounded_half(product.second));
                }
            }
            current = next;
            __syncthreads();
            if (thread == 0) {
                const int primary = life_best_action(node.h + INPUTS);
                const int spare = life_best_action(node.h + INPUTS + ACTIONS);
                was_tried = was_tried && from_half(node.observation[NEW]) == 0.f;
                expect_receipt = from_half(node.observation[READY]) != 0.f && !was_tried
                                 && primary >= FIRST_FINISH && primary < N_ACTIONS;
                was_tried = was_tried || expect_receipt;
                const auto trace_item = static_cast<offset_t>(step) * batch + item;
                private_world_action(world, primary, spare, stage[frame_item], target, tick == 0, trace + trace_item);
                if (spare_trace != nullptr) spare_trace[trace_item] = spare;
                if (outcomes != nullptr && tick == ticks - 1)
                    private_world_store_outcomes(world, outcomes, request, requests, item, batch);
            }
            __syncthreads();
        }
    }
    if (thread < life_nodes) {
        h[node_base + thread] = node.h[thread];
        activation[node_base + thread] = node.activation[current][thread];
        node_bias[node_base + thread] = node.bias[thread];
        node_trace[node_base + thread] = node.trace[thread];
        node_activity[node_base + thread] = node.activity[thread];
        gain[node_base + thread] = node.gain[thread];
    }
    if (thread < COUNT_FIELDS)
        counts[thread * batch + item] = stats.count[thread];
    if (thread < TOTAL_FIELDS)
        totals[thread * batch + item] = stats.total[thread];
    if (thread == 0) {
        private_world_store_batches(world, world_counts, world_flags, world_receipts, item, batch);
        tried[item] = was_tried;
        pending[item] = expect_receipt;
        auto* last_receipts = policy_receipts + static_cast<offset_t>(item) * 5;
        last_receipts[0] = receipts.ack;
        last_receipts[1] = receipts.rej;
        last_receipts[2] = receipts.reward;
        last_receipts[3] = receipts.legacy[0];
        last_receipts[4] = receipts.legacy[1];
        if (item == 0) *clock = stop * ticks;
    }
    for (int word = thread; word < life_nodes * (life_nodes / 2); word += ARC_THREADS) {
        matrix_words[word] = dense[word / (life_nodes / 2) * (matrix_stride / 2) + word % (life_nodes / 2)];
    }
}
