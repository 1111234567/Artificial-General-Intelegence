// 96-node specialization of the directed local-rule update. Each physical
// warp still owns one of the reference block's eight logical warps.
constexpr int fast_nodes = 96;
constexpr int fast_arcs = fast_nodes * (fast_nodes - 1);
constexpr int fast_warps = ARC_THREADS / 32;
constexpr int fast_count_fields = 5;
constexpr int fast_stat_stride = (fast_count_fields + 1) * fast_warps;

__device__ __forceinline__ void fast_commit_arc_stats(
    const int* warp_stats, long long* counts, float* totals, unsigned batch, unsigned actor
) {
    const int lane = threadIdx.x;
    if (lane >= 8)
        return;
    if (warp_stats[0] == -1) {
        if (lane == 0)
            counts[COUNT_EDGE_TICKS * batch + actor] += counts[COUNT_EDGE_COUNT * batch + actor];
        return;
    }

    int sum = 0;
    if (lane < fast_count_fields) {
        #pragma unroll
        for (int warp = 0; warp < fast_warps; ++warp)
            sum += warp_stats[lane * fast_warps + warp];
    }
    const int deaths = __shfl_sync(0xffu, sum, 1);
    if (lane == 0) {
        auto& edges = counts[COUNT_EDGE_COUNT * batch + actor];
        edges += sum - deaths;
        counts[COUNT_EDGE_TICKS * batch + actor] += edges;
    }
    if (lane < fast_count_fields) {
        constexpr int fields[] = {COUNT_BIRTHS, COUNT_DEATHS, COUNT_ATTEMPTED_WRITES,
                                  COUNT_CHANGED_WRITES, COUNT_WEIGHT_UPDATES};
        counts[fields[lane] * batch + actor] += sum;
    }
    if (lane == fast_count_fields) {
        float change = 0.f;
        #pragma unroll
        for (int warp = 0; warp < fast_warps; ++warp)
            change += __int_as_float(warp_stats[fast_count_fields * fast_warps + warp]);
        totals[TOTAL_WEIGHT_CHANGE * batch + actor] += change;
    }
}

template<int tiles>
__device__ __forceinline__ void fast_update_arcs(
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
    int actors,
    int* __restrict__ partial
) {
    const unsigned actor = blockIdx.x;
    const unsigned genome = actor / actors;
    const unsigned batch = gridDim.x;
    const int step = *clock;
    if (step == 0) {
        if (threadIdx.x == 0 && blockIdx.y == 0) {
            if constexpr (tiles == 1)
                counts[COUNT_EDGE_TICKS * batch + actor] += counts[COUNT_EDGE_COUNT * batch + actor];
            else
                partial[static_cast<offset_t>(actor) * fast_stat_stride] = -1;
        }
        return;
    }

    const offset_t node_base = static_cast<offset_t>(actor) * fast_nodes;
    const offset_t arc_base = static_cast<offset_t>(actor) * fast_arcs;
    const offset_t matrix_base = node_base * fast_nodes;
    const auto* baseline = initial_w + static_cast<offset_t>(genome) * fast_arcs;
    const auto* genetic_plastic = plastic + static_cast<offset_t>(genome) * PLASTIC_SIZE;
    const auto* genetic_legacy = legacy_rule + static_cast<offset_t>(genome) * 2 * RULE_SIZE;
    const auto* actor_receipts = receipts + actor * 5;
    const int thread = threadIdx.x;
    constexpr int tile_threads = ARC_THREADS / tiles;
    const int first_arc = blockIdx.y * tile_threads + thread;
    const int lane = thread % 32;
    const int warp = first_arc / 32;
    __shared__ float activity[fast_nodes];
    __shared__ float trace[fast_nodes];
    __shared__ float node_gain[fast_nodes];
    __shared__ float gene[PLASTIC_SIZE];
    __shared__ float legacy[2 * RULE_SIZE];
    __shared__ float receipt[5];
    __shared__ float rate;
    __shared__ int block_stats[tiles == 1 ? fast_stat_stride : 1];
    for (int node = thread; node < fast_nodes; node += tile_threads) {
        activity[node] = local_activity[node_base + node];
        trace[node] = node_trace[node_base + node];
        node_gain[node] = gain[node_base + node];
    }
    if (thread < PLASTIC_SIZE)
        gene[thread] = genetic_plastic[thread];
    if (thread < 2 * RULE_SIZE)
        legacy[thread] = genetic_legacy[thread];
    if (thread < 5)
        receipt[thread] = actor_receipts[thread];
    if (thread == 0)
        rate = reward_rate[genome];
    __syncthreads();

    const auto* birth_rule = legacy;
    const auto* prior = birth_rule + RULE_SIZE;
    const float ack = receipt[0];
    const float rej = receipt[1];
    const float reward = receipt[2];
    const float legacy_birth_trace = birth_rule[3] + receipt[3];
    const float legacy_trace = prior[3] + receipt[4];
    const float weight_receipt = rate * reward
                                 + WEIGHT_RECEIPT_RATE * (gene[WEIGHT_ACK] * ack + gene[WEIGHT_REJ] * rej);
    const float gate_receipt = gene[GATE_ACK] * ack + gene[GATE_REJ] * rej;
    int births = 0, deaths = 0, attempted = 0, changed = 0, updates = 0;
    float change = 0.f;
    for (int arc = first_arc; arc < fast_arcs; arc += ARC_THREADS) {
        const offset_t i = arc_base + arc;
        const int flat = indices[arc];
        const int dest = flat / fast_nodes;
        const int src = flat % fast_nodes;
        const float pre = trace[src];
        const float source_activity = activity[src];
        const float post = activity[dest];
        const float coactivity = pre * post;
        const float product = source_activity * post;
        const float trace_product = pre * trace[dest];
        const float endpoint = fabsf(source_activity - post);
        const float modulation = node_gain[dest];
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
        if (!live && (next_gate < BIRTH_THRESHOLD || step - last_change[i] < COOLDOWN))
            continue;
        float new_weight;
        float new_strength;
        half_bits old_strength_bits;
        const half_bits visible_bits = effective[i];
        if (!live) {
            const float drive = tanhf(birth_rule[0] + birth_rule[2] * product
                                     + legacy_birth_trace * trace_product + birth_rule[4] * endpoint);
            new_weight = from_half(to_half(SEED_RANGE * drive + (drive >= 0.f ? SEED_OFFSET : -SEED_OFFSET)));
            new_strength = RAMP_RATE;
            old_strength_bits = strength[i];
            eligibility[i] = to_half(0.f);
            mask[i] = true;
            last_change[i] = step;
            ++births;
        } else {
            old_strength_bits = strength[i];
            const float old_strength = from_half(old_strength_bits);
            const bool fade = next_gate <= DEATH_THRESHOLD && step - last_change[i] >= COOLDOWN;
            new_strength = clip(old_strength + (fade ? -RAMP_RATE : RAMP_RATE), 0.f, 1.f);
            if (new_strength == 0.f) {
                new_weight = 0.f;
                eligibility[i] = to_half(0.f);
                mask[i] = false;
                last_change[i] = step;
                ++deaths;
            } else {
                const half_bits old_eligibility = eligibility[i];
                const float e = from_half(old_eligibility);
                const half_bits next_eligibility = to_half(e + ELIGIBILITY_RATE * (product - e));
                if (next_eligibility != old_eligibility)
                    eligibility[i] = next_eligibility;
                const float receipt = weight_receipt * from_half(next_eligibility);
                const float rest = WEIGHT_RATE * gene[WEIGHT_DECAY] * (old - from_half(baseline[arc]));
                const float local = WEIGHT_RATE * (gene[WEIGHT_PRE] * pre + gene[WEIGHT_POST] * post
                                                   + gene[WEIGHT_COACTIVITY] * coactivity);
                new_weight = clip(old + (modulation * (local + receipt) - rest), -WEIGHT_LIMIT, WEIGHT_LIMIT);
                const float intrinsic = from_half(to_half(new_weight * old_strength));
                const float visible = from_half(visible_bits);
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
        const half_bits written_strength = to_half(new_strength);
        if (written_strength != old_strength_bits)
            strength[i] = written_strength;
        const half_bits written = to_half(new_weight * new_strength);
        if (written != visible_bits) {
            effective[i] = written;
            matrix[matrix_base + flat] = written;
        }
    }

    int* warp_stats = tiles == 1 ? block_stats : partial + static_cast<offset_t>(actor) * fast_stat_stride;
    const int outcomes[] = {births, deaths, attempted, changed, updates};
    #pragma unroll
    for (int field = 0; field < fast_count_fields; ++field) {
        const int sum = __reduce_add_sync(0xffffffffu, outcomes[field]);
        if (lane == 0)
            warp_stats[field * fast_warps + warp] = sum;
    }
    const float sum = warp_sum(change);
    if (lane == 0)
        warp_stats[fast_count_fields * fast_warps + warp] = __float_as_int(sum);
    if constexpr (tiles == 1) {
        __syncthreads();
        fast_commit_arc_stats(block_stats, counts, totals, batch, actor);
    }
}

#define FAST_ARC_KERNEL(name, tiles) \
extern "C" __global__ __launch_bounds__(ARC_THREADS / tiles) void name( \
    const int* clock, const int* indices, const float* local_activity, const float* node_trace, \
    const float* gain, const float* receipts, const float* plastic, const float* legacy_rule, \
    const float* reward_rate, const half_bits* initial_w, float* weight, float* gate, half_bits* eligibility, \
    half_bits* strength, bool* mask, int* last_change, half_bits* effective, half_bits* matrix, \
    long long* counts, float* totals, int actors, int nodes, int arcs, int* partial \
) { \
    fast_update_arcs<tiles>(clock, indices, local_activity, node_trace, gain, receipts, plastic, \
        legacy_rule, reward_rate, initial_w, weight, gate, eligibility, strength, mask, last_change, \
        effective, matrix, counts, totals, actors, partial); \
}

FAST_ARC_KERNEL(fast_local_arcs_one_tile, 1)
FAST_ARC_KERNEL(fast_local_arcs_two_tiles, 2)
FAST_ARC_KERNEL(fast_local_arcs_four_tiles, 4)
