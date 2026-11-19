# 可复用的神经元与训练模块

新任务从仓库根目录按类型导入。环境提供输入、回执或损失，调用方持有状态并决定何时更新。只用 NumPy 的稀疏 LIF 和双层 Elman 可以单独导入。

| 想训练的模型 | 入口 | 可直接沿用的实现 |
|---|---|---|
| 稀疏 LIF | [lif.py](lif.py) 的 `build_network`、`train` | 原 `Network` 的稀疏传播、活动迹、权重和偏置更新、生长与修剪；原框架的六个流程共用建网入口 |
| 适应性 LIF 或带第二隔室的脉冲图 | [lif_population.py](lif_population.py) 的 `evaluate_population` | 64 节点、每节点 12 条输入边的种群评估。CPU 参考与原 CUDA 内核可选择，返回每个基因的答案、正确率和放电率 |
| 普通／共享循环权重 GRU | [gru.py](gru.py) 的 `GRU`、带分类读出的 `GRUAgent`、`fit_gru_sequence` | 原生 `nn.GRU` 整段计算、按无序节点对存储三个门的循环权重、Adam 续训及密集权重导出 |
| tanh RNN、泄漏单元、连续时间水库 | [rate.py](rate.py) 的 `TanhRNN`、`PulseRateRNN` | 输入批量投影、固定或可学习泄漏、Heun 积分。水库可冻结循环权重并继续训练输入映射 |
| NumPy 双层 Elman | [rate_numpy.py](rate_numpy.py) 的 `NumpyElman2` | 两层状态轨迹、手写 BPTT、读出梯度遮罩、谱归一化；旧 [grow_rnn.py](../../experiments/prototypes/grow_rnn.py) 调用同一实现 |
| 连续线性感觉图 | [linear.py](linear.py) 的 `LinearGraph`、`linear_sequence` | 有向旋转图和对角泄漏、逐 tick 更新、并行前缀扫描及稀疏脉冲块采样。已有导出的转移矩阵和感觉投影可以直接传给 `linear_sequence` |
| 有向可塑图、无向时间信用 | [plastic_graph.py](plastic_graph.py) | 逐 tick 的 `LocalRulePolicy`、CUDA 弧更新，以及原无向模型、`TemporalCreditEngine` 和结构操作 |

### 新任务里训练 LIF

下面的两神经元例子使用 NumPy。先让输入、输出神经元依次放电，再把该动作得到的反馈施加到权重。此例中唯一的一条边由 0.5 变为 0.625。

```python
import numpy as np
from src.neurons.lif import NetSpec, build_network

spec = NetSpec(N=2, n_in=1, e_ratio=1.0, e0_init=20)
net = build_network(spec, np.random.default_rng(7), out_idx=[1])
net.step(np.array([1.25, 0.0], dtype=np.float32))
net.step(np.array([0.0, 0.75], dtype=np.float32))
spikes = net.state[net.out_idx].copy()
net.apply_weight(m_vec=np.array([0.5], dtype=np.float32), m_glob=None, eta=0.25)
```

同一生命继续对同一个 `net` 调用 `step`。权重、偏置和拓扑允许分别更新。已有框架配置可以直接交给 `src.neurons.lif.train(config)`，其环境和更新时机沿用[原配置的 flow](../README.md)。

### 在序列上训练

下面用泄漏 tanh 单元学习一个标量读出。梯度沿段内时间传播，下一段接收本段的末状态。优化器可以同时持有神经元和外部读出的参数。

```python
import torch
from src.neurons.rate import TanhRNN
from src.neurons.training import train_chunk

model = TanhRNN(input_size=3, hidden_size=16, leak=0.4)
readout = torch.nn.Linear(16, 1)
optimizer = torch.optim.Adam([*model.parameters(), *readout.parameters()], lr=0.001)
x = torch.randn(32, 4, 3)  # 时间、并行生命、输入通道
y = x[..., :1]
state = model.initial_state(4)
for inputs, targets in zip(x.split(8), y.split(8)):
    loss, state = train_chunk(model, optimizer, inputs, state,
                              lambda activity: (readout(activity) - targets).square().mean())
```

`LinearGraph(input_size=3, width=16)` 和 `GRU(input_size=3, width=16)` 可使用相同训练循环。连续时间 `PulseRateRNN` 需要给 `train_chunk` 传入 `elapsed=`，其输出是活动值，末状态是原始内部状态。直接调用各模型的 `forward` 可自行决定反向传播的跨度。

GRU 自带分类读出和按段训练的便捷入口。下面的 `labels` 对应每个生命中每个任务的动作类别。

```python
import torch
from src.neurons.gru import GRUAgent, fit_gru_sequence

model = GRUAgent(width=32, input_size=5, n_jobs=1, n_actions=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(12, 4, 5)
labels = torch.randint(3, (12, 4, 1))
result = fit_gru_sequence(model, optimizer, x, labels, model.initial_state(4), block_size=6)
state = result["hidden"]
```

训练共享权重 GRU 时，在构造优化器前调用 `share_recurrent_weights(model)`。保存打包的模型及优化器状态可继续训练；`deploy_dense_gru(model)` 可生成普通 GRU 布局的部署副本。多轮 `fit_gru_sequence` 会从同一段历史的起始锚点重放，新的连续输入段应显式传入上段的 `hidden`。

### 反馈驱动的快权重

[fast_weights.py](fast_weights.py) 可以接在任一单元输出的键上。在线 RLS 用 `initial_rls_state` 建立状态，每步先用 `read_fast_weights` 读取，收到回执后用 `rls_update` 写入当时保存的键。批训练可用 `rls_prefix_predictions` 的 Cholesky 前缀解，或用 `delta_prefix_predictions` 的下三角求解复算归一化 delta。当前步的读数只依赖更早的回执。布尔遮罩控制哪些步骤允许写入。这里的键按 `[..., 时间, 键宽]` 排列，标量目标与遮罩按 `[..., 时间]` 排列。长轨迹可改用逐步更新控制内存。

### 选择原有加速后端

- **脉冲种群**。`evaluate_population(genomes, graph_seed, schedule, device="cuda:0", batch_size=32)` 接受归一化基因矩阵和调用方构造的试次。每批只把指定数量的候选交给 GPU，结果保留整个种群的顺序。选择 `mode="add"`、`"gain"` 或 `"both"` 时使用第二隔室基因范围。原内核的输入是两种背景、四种线索、起始信号，读取的是四选一结果。CPU 有参考实现，现有 CUDA 编译器使用 Windows 的 PyTorch/NVRTC，第二隔室 CUDA 后端使用设备 0。[基因顺序与时间表](lif_population.py)
- **GRU 缓存**。`attach_frozen_encoder_cache(model)` 复用已有的冻结编码器缓存，适用于带原生 `model.encoder` 的模型、可复用的只读输入和相容的 CPU FP32 执行模式。[原缓存及失效条件](../../experiments/seed_evolution/encoder_cache.py)
- **可塑图**。`make_local_policy(..., arc_kernel="tiled")` 或 `"compact"` 选择已验证的弧更新。96 节点、设备能力和固定端口等条件决定是否启用该 CUDA 路径。原私有符号世界还能使用 `ResidentLocalRuleEvaluator` 或 `FastLocalTrainingEvaluator`。无向图的完整 RTRL、记录窗口和截断信用使用 `TemporalCreditEngine`。[入口与约束](plastic_graph.py)
- **整局游戏评分**。[linear_game.py](linear_game.py) 提供成对游戏、统一游戏和长生命三种原生 CUDA 评分器，直接复用物理动力学、已准备的感觉缓存和种群评分。成对游戏的 `PairedPolicyCohort.run` 在普通计分时使用融合计数与 warp 归约。专用世界和端口须与相应实验一致。[端到端吞吐对照](../../data/pulse-game/research-2026-11-19/physical-throughput/report.md)
- **精度**。16 位感觉缓存曾在长生命上节约一半缓存空间，也改变过少量真实动作。该可选实现和测量见[硬件缓存试验](../../experiments/pulse_game/precision_cache.py)。通用训练沿用调用者选的设备与 dtype。

从仓库根目录运行 `python -m pytest -q test`，可检查新模块与旧实现的状态、更新和检查点兼容性。冻结报告和独立审计继续指向各自的历史源码。
