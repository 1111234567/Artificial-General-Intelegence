// Optional FP16 node transition. Numeric constants come from the Python module.
// No fast math: PyTorch 2.11.0+cu130, commit
// 70d99e998b4955e0049d13a98d77ae1b14db1f45, uses ::tanh(a) for Half in
// aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu. Half converts to float;
// tanhf here follows the same libdevice path before the final half conversion.
// The companion Torch-tanh policy can disable both transcendental substitutions.
using half_bits = unsigned short;
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

extern "C" __global__ void directed_node_prepare(
    const half_bits* __restrict__ h,
    const half_bits* __restrict__ obs,
    const bool* __restrict__ pending,
    half_bits* __restrict__ reward,
    bool* __restrict__ event,
    long long* __restrict__ reward_events,
    half_bits* __restrict__ node_trace,
    half_bits* __restrict__ local_activity,
    int nodes, int fuse_trace
) {
    const unsigned actor = blockIdx.x;
    const offset_t obs_base = static_cast<offset_t>(actor) * NODE_INPUTS;
    if (threadIdx.x == 0) {
        // Subtraction is a separate Half tensor before multiplication by the
        // bool pending flag. Multiplying by zero must still propagate NaN.
        const float difference = from_half(to_half(__fsub_rn(
            from_half(obs[obs_base + NODE_ACK]), from_half(obs[obs_base + NODE_REJ]))));
        const half_bits receipt = to_half(difference * static_cast<float>(pending[actor]));
        reward[actor] = receipt;
        const bool occurred = from_half(receipt) != 0.0f;
        event[actor] = occurred;
        if (occurred) reward_events[actor] += 1;
    }
    if (!fuse_trace) return;
    const offset_t base = static_cast<offset_t>(actor) * nodes;
    for (int node = threadIdx.x; node < nodes; node += blockDim.x) {
        const offset_t i = base + node;
        // (4. * h) materializes in FP16 before its tanh.
        const float scaled = from_half(to_half(4.0f * from_half(h[i])));
        const half_bits local = to_half(tanhf(scaled));
        local_activity[i] = local;
        // mul_(.95) writes Half, then add_(alpha=.05) uses FP32 FMA.
        const float decayed = from_half(to_half(from_half(node_trace[i]) * .95f));
        node_trace[i] = to_half(__fmaf_rn(from_half(local), .05f, decayed));
    }
}

extern "C" __global__ void directed_node_finish(
    const half_bits* __restrict__ current,
    const half_bits* __restrict__ bias,
    const half_bits* __restrict__ obs,
    half_bits* __restrict__ h,
    half_bits* __restrict__ activity,
    bool* __restrict__ tried,
    bool* __restrict__ pending,
    long long* __restrict__ actions,
    int nodes, int actors, int fuse_tanh
) {
    const unsigned actor = blockIdx.x;
    const offset_t base = static_cast<offset_t>(actor) * nodes;
    const offset_t bias_base = static_cast<offset_t>(actor / actors) * nodes;
    const offset_t obs_base = static_cast<offset_t>(actor) * NODE_INPUTS;
    __shared__ half_bits action_ports[NODE_JOBS * NODE_ACTIONS];
    for (int node = threadIdx.x; node < nodes; node += blockDim.x) {
        const offset_t i = base + node;
        // Preserve each Half materialization in the original Torch sequence:
        // h.mul_(.75), current + bias, h.add_(...,alpha=.25), input add_.
        const float decayed = from_half(to_half(from_half(h[i]) * .75f));
        const float drive = from_half(to_half(__fadd_rn(
            from_half(current[i]), from_half(bias[bias_base + node]))));
        half_bits updated = to_half(__fmaf_rn(drive, .25f, decayed));
        if (node < NODE_INPUTS) {
            updated = to_half(__fmaf_rn(from_half(obs[obs_base + node]), .25f, from_half(updated)));
        }
        h[i] = updated;
        if (fuse_tanh) activity[i] = to_half(tanhf(from_half(updated)));
        if (node >= NODE_INPUTS && node < NODE_INPUTS + NODE_JOBS * NODE_ACTIONS) {
            action_ports[node - NODE_INPUTS] = updated;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int primary = 0;
        for (int job = 0; job < NODE_JOBS; ++job) {
            int best_index = 0;
            float best = from_half(action_ports[job * NODE_ACTIONS]);
            for (int action = 1; action < NODE_ACTIONS; ++action) {
                const float candidate = from_half(action_ports[job * NODE_ACTIONS + action]);
                // PyTorch GreaterOrNan prefers NaNs, then the lowest index
                // for equal values or multiple NaNs, including signed zeros.
                if ((candidate != candidate && best == best) || candidate > best) {
                    best = candidate;
                    best_index = action;
                }
            }
            actions[static_cast<offset_t>(actor) * NODE_JOBS + job] = best_index;
            if (job == 0) primary = best_index;
        }
        const bool was_tried = tried[actor] && !(from_half(obs[obs_base + NODE_NEW]) != 0.0f);
        const bool next_pending = (from_half(obs[obs_base + NODE_READY]) != 0.0f)
            && !was_tried && primary >= NODE_FIRST_FINISH && primary < NODE_N_ACTIONS;
        pending[actor] = next_pending;
        tried[actor] = was_tried || next_pending;
    }
}
