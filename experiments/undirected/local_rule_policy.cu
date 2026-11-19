// Fused CUDA implementation of the every-tick local rule. Python provides the
// shared constants and genetic column positions. One block owns each actor.
#if !TENSOR_TRANSITION
using half_bits = unsigned short;
#endif
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

__device__ __forceinline__ float clip(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

template<typename T>
__device__ __forceinline__ T warp_sum(T value) {
    #pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, shift);
    return value;
}

extern "C" __global__ void local_rule_prepare(
    const int* __restrict__ clock,
    const half_bits* __restrict__ obs,
    const half_bits* __restrict__ h,
    const bool* __restrict__ pending,
    const float* __restrict__ plastic,
    const float* __restrict__ legacy_rule,
    const half_bits* __restrict__ initial_bias,
    float* __restrict__ bias,
    float* __restrict__ node_trace,
    float* __restrict__ local_activity,
    float* __restrict__ gain,
    float* __restrict__ receipts,
    long long* __restrict__ counts,
    float* __restrict__ totals,
    int actors, int nodes, int modulated
) {
    const unsigned actor = blockIdx.x;
    const unsigned genome = actor / actors;
    const unsigned batch = gridDim.x;
    const bool started = *clock != 0;
    const auto* input = obs + static_cast<offset_t>(actor) * INPUTS;
    const auto* gene = plastic + static_cast<offset_t>(genome) * PLASTIC_SIZE;
    const auto* prior = legacy_rule + static_cast<offset_t>(genome) * 2 * RULE_SIZE;
    const auto* baseline = initial_bias + static_cast<offset_t>(genome) * nodes;
    const offset_t base = static_cast<offset_t>(actor) * nodes;
    const float primary_ack = from_half(input[ACK]);
    const float primary_rej = from_half(input[REJ]);
    const float ack = started ? primary_ack + SPARE_RECEIPT_RATE * from_half(input[PORT_SIZE + ACK]) : 0.f;
    const float rej = started ? primary_rej + SPARE_RECEIPT_RATE * from_half(input[PORT_SIZE + REJ]) : 0.f;
    if (threadIdx.x == 0) {
        const float reward = started && pending[actor] ? primary_ack - primary_rej : 0.f;
        float begin_ack = 0.f, finish_ack = 0.f, begin_rej = 0.f, finish_rej = 0.f;
        if (started) {
            for (int job = 0; job < JOBS; ++job) {
                const auto* port = input + job * PORT_SIZE;
                const float begin = from_half(port[12]) + from_half(port[13]);
                float finish = 0.f;
                for (int i = 14; i < 22; ++i)
                    finish += from_half(port[i]);
                begin_ack += begin * from_half(port[ACK]);
                finish_ack += finish * from_half(port[ACK]);
                begin_rej += begin * from_half(port[REJ]);
                finish_rej += finish * from_half(port[REJ]);
            }
        }
        receipts[actor * 5] = ack;
        receipts[actor * 5 + 1] = rej;
        receipts[actor * 5 + 2] = reward;
        for (int row = 0; row < 2; ++row) {
            const auto* rule = prior + row * RULE_SIZE;
            receipts[actor * 5 + 3 + row] = rule[5] * (begin_ack / JOBS) + rule[6] * (finish_ack / JOBS)
                                           + rule[7] * (begin_rej / JOBS) + rule[8] * (finish_rej / JOBS);
        }
        counts[COUNT_REWARD_EVENTS * batch + actor] += reward != 0.f;
        counts[COUNT_PUBLIC_ACK_EVENTS * batch + actor] += ack != 0.f;
        counts[COUNT_PUBLIC_REJ_EVENTS * batch + actor] += rej != 0.f;
    }
    int updates = 0;
    float change = 0;
    for (int node = threadIdx.x; node < nodes; node += blockDim.x) {
        const offset_t i = base + node;
        const float activity = tanhf(4.f * from_half(h[i]));
        const float trace_old = node_trace[i];
        const float trace = started ? trace_old + (activity - trace_old) * TRACE_RATE : trace_old;
        const float gain_value = modulated ? 1.f + MOD_SCALE * tanhf(gene[MOD_ACTIVITY] * activity
                                                                    + gene[MOD_TRACE] * trace) : 1.f;
        local_activity[i] = activity;
        node_trace[i] = trace;
        gain[i] = gain_value;
        if (!started)
            continue;
        const float old = bias[i];
        const float anchor = from_half(baseline[node]);
        const float local = BIAS_RATE * (gene[BIAS_ACTIVITY] * activity + gene[BIAS_TRACE] * trace)
                            + BIAS_RECEIPT_RATE * (gene[BIAS_ACK] * ack + gene[BIAS_REJ] * rej) * trace;
        const float delta = gain_value * local - BIAS_RATE * gene[BIAS_DECAY] * (old - anchor);
        const float updated = clip(old + delta, anchor - BIAS_LIMIT, anchor + BIAS_LIMIT);
        bias[i] = updated;
        const float effective_old = from_half(to_half(old));
        const float effective_new = from_half(to_half(updated));
        updates += effective_new != effective_old;
        change += fabsf(effective_new - effective_old);
    }
    updates = warp_sum(updates);
    change = warp_sum(change);
    __shared__ int warp_updates[32];
    __shared__ float warp_changes[32];
    if (threadIdx.x % 32 == 0) {
        warp_updates[threadIdx.x / 32] = updates;
        warp_changes[threadIdx.x / 32] = change;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int warp = 1; warp < blockDim.x / 32; ++warp) {
            updates += warp_updates[warp];
            change += warp_changes[warp];
        }
        counts[COUNT_BIAS_UPDATES * batch + actor] += updates;
        totals[TOTAL_BIAS_CHANGE * batch + actor] += change;
    }
}

extern "C" __global__ __launch_bounds__(ARC_THREADS) void local_rule_arcs(
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
    const auto* gene = plastic + static_cast<offset_t>(genome) * PLASTIC_SIZE;
    const auto* birth_rule = legacy_rule + static_cast<offset_t>(genome) * 2 * RULE_SIZE;
    const auto* prior = birth_rule + RULE_SIZE;
    const auto* baseline = initial_w + static_cast<offset_t>(genome) * arcs;
    const offset_t n = static_cast<offset_t>(actor) * nodes;
    const offset_t arc_base = static_cast<offset_t>(actor) * arcs;
    const offset_t matrix_base = n * nodes;
    const float ack = receipts[actor * 5];
    const float rej = receipts[actor * 5 + 1];
    const float reward = receipts[actor * 5 + 2];
    const float legacy_birth_trace = birth_rule[3] + receipts[actor * 5 + 3];
    const float legacy_trace = prior[3] + receipts[actor * 5 + 4];
    const float weight_receipt = reward_rate[genome] * reward
                                 + WEIGHT_RECEIPT_RATE * (gene[WEIGHT_ACK] * ack + gene[WEIGHT_REJ] * rej);
    const float gate_receipt = gene[GATE_ACK] * ack + gene[GATE_REJ] * rej;
    int births = 0, deaths = 0, attempted = 0, changed = 0, updates = 0;
    float change = 0;
    for (int arc = threadIdx.x; arc < arcs; arc += blockDim.x) {
        const offset_t i = arc_base + arc;
        const int flat = indices[arc];
        const int dest = flat / nodes;
        const int src = flat % nodes;
        const float pre = node_trace[n + src];
        const float source_activity = local_activity[n + src];
        const float post = local_activity[n + dest];
        const float coactivity = pre * post;
        const float product = source_activity * post;
        const float trace_product = pre * node_trace[n + dest];
        const float endpoint = fabsf(source_activity - post);
        const float modulation = gain[n + dest];
        const bool live = mask[i];
        const float old = live ? weight[i] : 0.f;
        const float legacy_support = prior[0] + prior[1] * (old / WEIGHT_LIMIT) + prior[2] * product
                                   + legacy_trace * trace_product + prior[4] * endpoint;
        const float support = tanhf(legacy_support) + gene[GATE_BIAS] + gene[GATE_COACTIVITY] * coactivity
                            + gene[GATE_TRACE] * trace_product
                            + gene[GATE_WEIGHT] * (fabsf(old) / WEIGHT_LIMIT) + gate_receipt * coactivity;
        const float previous = gate[i];
        const float next_gate = previous + (clip(modulation * support, -1.f, 1.f) - previous) * GATE_RATE;
        gate[i] = next_gate;
        if (!live && next_gate < BIRTH_THRESHOLD)
            continue;
        const bool ready = step - last_change[i] >= COOLDOWN;
        if (!live && !ready)
            continue;
        float new_weight;
        float new_strength;
        if (!live) {
            // The inactive arc has zero prior weight. Its genetic seed starts
            // at 1/32 strength and enters the recurrent matrix gradually.
            const float drive = tanhf(birth_rule[0] + birth_rule[2] * product
                                     + legacy_birth_trace * trace_product + birth_rule[4] * endpoint);
            new_weight = from_half(to_half(SEED_RANGE * drive + (drive >= 0.f ? SEED_OFFSET : -SEED_OFFSET)));
            new_strength = RAMP_RATE;
            eligibility[i] = to_half(0.f);
            mask[i] = true;
            last_change[i] = step;
            ++births;
        } else {
            const float old_strength = from_half(strength[i]);
            const bool fade = ready && next_gate <= DEATH_THRESHOLD;
            new_strength = clip(old_strength + (fade ? -RAMP_RATE : RAMP_RATE), 0.f, 1.f);
            if (new_strength == 0.f) {
                new_weight = 0.f;
                eligibility[i] = to_half(0.f);
                mask[i] = false;
                last_change[i] = step;
                ++deaths;
            } else {
                const float e = from_half(eligibility[i]);
                const half_bits next_eligibility = to_half(e + ELIGIBILITY_RATE * (product - e));
                eligibility[i] = next_eligibility;
                const float receipt = weight_receipt * from_half(next_eligibility);
                const float rest = WEIGHT_RATE * gene[WEIGHT_DECAY] * (old - from_half(baseline[arc]));
                const float local = WEIGHT_RATE * (gene[WEIGHT_PRE] * pre + gene[WEIGHT_POST] * post
                                                   + gene[WEIGHT_COACTIVITY] * coactivity);
                new_weight = clip(old + (modulation * (local + receipt) - rest), -WEIGHT_LIMIT, WEIGHT_LIMIT);
                const float intrinsic = from_half(to_half(new_weight * old_strength));
                const float visible = from_half(effective[i]);
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
        weight[i] = new_weight;
        strength[i] = to_half(new_strength);
        const half_bits written = to_half(new_weight * new_strength);
        if (written != effective[i]) {
            effective[i] = written;
            matrix[matrix_base + flat] = written;
        }
    }

    constexpr int warps = ARC_THREADS / 32;
    constexpr int count_fields = 5;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int outcomes[count_fields] = {births, deaths, attempted, changed, updates};
    __shared__ int warp_counts[count_fields][warps];
    __shared__ float warp_changes[warps];
    #pragma unroll
    for (int field = 0; field < count_fields; ++field) {
        const int sum = warp_sum(outcomes[field]);
        if (lane == 0)
            warp_counts[field][warp] = sum;
    }
    const float sum = warp_sum(change);
    if (lane == 0)
        warp_changes[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int field = 0; field < count_fields; ++field) {
            outcomes[field] = 0;
            #pragma unroll
            for (int warp_id = 0; warp_id < warps; ++warp_id)
                outcomes[field] += warp_counts[field][warp_id];
        }
        const int fields[count_fields] = {COUNT_BIRTHS, COUNT_DEATHS, COUNT_ATTEMPTED_WRITES,
                                          COUNT_CHANGED_WRITES, COUNT_WEIGHT_UPDATES};
        #pragma unroll
        for (int field = 0; field < count_fields; ++field)
            counts[fields[field] * batch + actor] += outcomes[field];
        auto& edges = counts[COUNT_EDGE_COUNT * batch + actor];
        edges += outcomes[0] - outcomes[1];
        counts[COUNT_EDGE_TICKS * batch + actor] += edges;
        change = 0;
        #pragma unroll
        for (int warp_id = 0; warp_id < warps; ++warp_id)
            change += warp_changes[warp_id];
        totals[TOTAL_WEIGHT_CHANGE * batch + actor] += change;
    }
}

extern "C" __global__ void local_rule_finish(
    const half_bits* __restrict__ input,
    const float* __restrict__ bias,
    const half_bits* __restrict__ obs,
    half_bits* __restrict__ h,
    half_bits* __restrict__ activity,
    bool* __restrict__ tried,
    bool* __restrict__ pending,
    long long* __restrict__ actions,
    int* __restrict__ clock,
    int nodes
) {
    const unsigned actor = blockIdx.x;
    const offset_t base = static_cast<offset_t>(actor) * nodes;
    const auto* frame = obs + static_cast<offset_t>(actor) * INPUTS;
    __shared__ half_bits ports[JOBS * ACTIONS];
#if TENSOR_TRANSITION
    __shared__ half_bits products[96];
    multiply_rows(reinterpret_cast<const unsigned*>(input) + base * 48,
                  reinterpret_cast<const unsigned*>(activity) + base / 2,
                  products, threadIdx.x / 32, threadIdx.x % 32);
    __syncthreads();
#else
    const half_bits* products = input + base;
#endif
    for (int node = threadIdx.x; node < nodes; node += blockDim.x) {
        const offset_t i = base + node;
        const float decayed = from_half(to_half(from_half(h[i]) * .75f));
        const float drive = from_half(to_half(from_half(products[node]) + from_half(to_half(bias[i]))));
        half_bits updated = to_half(__fmaf_rn(drive, .25f, decayed));
        if (node < INPUTS)
            updated = to_half(__fmaf_rn(from_half(frame[node]), .25f, from_half(updated)));
        h[i] = updated;
        activity[i] = to_half(tanhf(from_half(updated)));
        if (node >= INPUTS && node < INPUTS + JOBS * ACTIONS)
            ports[node - INPUTS] = updated;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int primary = 0;
        for (int job = 0; job < JOBS; ++job) {
            int best_index = 0;
            float best = from_half(ports[job * ACTIONS]);
            for (int action = 1; action < ACTIONS; ++action) {
                const float candidate = from_half(ports[job * ACTIONS + action]);
                if ((candidate != candidate && best == best) || candidate > best) {
                    best = candidate;
                    best_index = action;
                }
            }
            actions[static_cast<offset_t>(actor) * JOBS + job] = best_index;
            if (job == 0)
                primary = best_index;
        }
        const bool was_tried = tried[actor] && from_half(frame[NEW]) == 0.f;
        const bool next_pending = from_half(frame[READY]) != 0.f && !was_tried
                                  && primary >= FIRST_FINISH && primary < N_ACTIONS;
        pending[actor] = next_pending;
        tried[actor] = was_tried || next_pending;
        if (actor == 0)
            *clock += 1;
    }
}
