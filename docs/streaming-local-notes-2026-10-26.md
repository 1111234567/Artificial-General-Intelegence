# 局部活动学习与小规模结构实验补充说明（2026-10-26）

本文是[总报告](streaming-experiment-2026-10-26.md)的补充，仅包含已冻结的 local-v1 与 structure-v1 实测、对应代码及原 LIF 规则。较大持续流和自由连接渐进实验的结果见总报告。本文中的“局部”描述更新所需的信息和计算路径，不表示完全无梯度，也不等于已经验证了生物 LIF、表示学习或终身学习。

已冻结实验支持两个有限结论：带真实任务反馈的活动相关局部更新能学会这个小分类任务，但没有天然消除 A→B→A 干扰；任务相关重连在相同有效连接预算下改善当前任务拟合，但没有证明改善旧任务保持。仅把输出频率调到近均匀，未学会标签映射。这是对本次频率稳态规则的结果，不能推广为所有 Hebbian、STDP 或活动相关学习都无效。

共同任务由 [make_task](../experiments/streaming/local_learning.py#L100) 生成。每个种子分别生成一个固定线性 teacher 和独立的 train/val/test 原始输入，规模为 4096/1024/2048。原始输入维数为 32，类别数为 3；A 的标签是 teacher 得分的 argmax，B 对同一个原始输入的标签为 (A+1) mod 3。观察到的 context 用 c=+1/-1 表示，输入模型的固定特征为 [1,x,c,0.5cx]，共 66 维。

context 与标签冲突不是不可辨识问题：模型确实看到 context；预先构造的交互特征已使两个任务都可由单层线性分类器表示，且有独立的精确 teacher 测试证明。共享 x 与半幅 cx 保留了任务更新的干扰。这里没有证明网络自己学出了这种表示。代码与测试见 [固定特征](../experiments/streaming/local_learning.py#L66)、[语义测试](../experiments/streaming/test_local_learning.py)。

两组正式实验都使用种子 0/1/2、每阶段 800 批、每批 128 个样本、学习率 0.2，顺序为 A1→B→A2；每 10 步记录验证集曲线。各方法共享原始任务与阶段采样流，正式配置在只看验证集的 pilot 后固定，不按测试集调整，也不根据验证曲线选择中间 checkpoint。报告以下比例时均乘以 100，± 为三个种子的样本标准差，不是置信区间。

局部学习组的完整依据是 [manifest](../data/streaming/local-v1/manifest.json)、[summary](../data/streaming/local-v1/summary.json)、[源码](../experiments/streaming/local_learning.py) 和 [12 项 CPU 测试](../experiments/streaming/test_local_learning.py)。五组都有同一个 198 参数的单层 softmax，初始权重在每个种子内完全相同；常量特征对应的第一列权重就是输出偏置。

| 方法 | 更新可见的信息 | 实际更新 |
|---|---|---|
| frozen | 仅做前向和诊断记录 | 权重、偏置及辅助学习状态均不变 |
| frequency | 抽样输出类别的次数及其 EMA | 只调整偏置，使输出类别频率靠近 1/3；更新函数没有标签或奖励参数 |
| local_delta | 输入活动、预测概率、外部真实标签 | ΔW=η·mean[(onehot(y)−p)⊗x]，直接计算，不调用反向自动微分 |
| reward_local | 输入活动、抽样输出及概率、环境二值正确性反馈 | e=(onehot(a)−p)⊗x；ΔW=η·mean[(r−b)e]，b 为历史奖励均值 |
| bp | 与 local_delta 相同的输入和完整标签 | 对单层交叉熵做标准 SGD 反传 |

局部 delta 与这个单层 BP 数学等价，是更新实现的阳性控制，不是一个新学习算法。多步数值测试在 float64 下验证了等价性；正式各阶段权重最大差为 7.15e-7，各准确率完全相同。reward_local 是用局部活动构造的解析 policy-gradient 估计，仍然有梯度含义。它只接收环境的 0/1 正确性，不接收类别答案；因此与 BP 的差别同时包含反馈信息量和抽样方差，不能把表现差异单归因于“是否局部”。

reward_local 只使用当前试次的即时资格量，没有测试跨长延迟的资格迹。b 只是运行奖励均值，没有自评、价值或正确性预测头。输入是有符号的 rate surrogate，输出是 softmax 抽样类别；这不是脉冲发放的生物实现。训练器为记录指标会读取真实标签，但标签置换测试证明 frequency 的权重轨迹完全不受这些记录用标签影响。

| 方法 | A1 后 A | B 后旧 A | B 后新 B | A2 后 A | A2 后旧 B |
|---|---:|---:|---:|---:|---:|
| frozen | 38.51±4.22 | 38.51±4.22 | 32.65±2.38 | 38.51±4.22 | 32.65±2.38 |
| frequency | 38.82±3.97 | 37.43±3.35 | 33.17±3.00 | 39.10±4.34 | 33.33±2.82 |
| local_delta / bp | 98.16±0.58 | 52.62±0.75 | 97.93±0.07 | 98.10±0.18 | 81.80±1.27 |
| reward_local | 96.92±1.07 | 55.31±0.71 | 96.55±0.69 | 97.14±0.25 | 79.18±1.43 |

学 B 后，local_delta/BP 的旧 A 降低 45.54±1.33 个百分点，reward_local 降低 41.62±1.74 个百分点。frequency 的三个输出类别的平均概率已接近各 1/3，但任务准确率仍接近其冻结初始对照；控制活跃度不等于获得了类别映射的任务价值信号。冻结组没有遗忘，也是因为没有学会新任务，不能据此说它解决了稳定性与可塑性的矛盾。

另设独立的 A/B 混合训练，只作同时可学性的检查；没有把它作为顺序训练的 replay。它与顺序实验总更新数相同，但每个 context 的样本分配比例不同，不是数学上的性能上界。混合训练测试集结果为 local_delta/BP A=97.51±0.37、B=97.74±0.64，reward_local A=96.63±0.59、B=96.37±0.87。可见顺序后的旧任务下降不能简单归因于这个表示完全容纳不了两个任务。

重新学习使用验证集首次到达阈值的步数，分辨率为 10 步。fresh_A 从相同随机初始权重和全新辅助状态开始，使用与 A2 完全相同的输入采样流。没有以测试集的达标时间选择模型。

| 方法 | A1 到 90% | A2 再到 90% | fresh_A 到 90% | A2 恢复本 seed 的 A1 末验证准确率 | fresh_A 到相同末准确率 |
|---|---:|---:|---:|---:|---:|
| local_delta / bp | 20±0 | 60±10 | 20±0 | 316.7±110.2 | 370.0±168.2 |
| reward_local | 40±0 | 206.7±15.3 | 40±10 | 413.3±51.3 | 340.0±91.7 |

低阈值和接近收敛的阈值给出的速度比较不同，且只有三个种子，因此不能选择其中一列宣称普遍再学习加速或普遍没有残留知识。frozen 与 frequency 在预算内都没有达到 90%。

local-v1 正式运行使用项目 .venv 的 PyTorch 2.11.0+cu130，但设备明确为 CPU、单线程，总计 34.109 秒。源码运行中未变化，[artifact_hashes](../data/streaming/local-v1/artifact_hashes.json) 中 131 个文件的 SHA256 全部核验一致。目录约 19.29 MB，包含数据快照、checkpoint、完整曲线、配置及计时；程序拒绝覆盖已有输出。

小规模结构组由 [structure.py](../experiments/streaming/structure.py) 实现，依据为 [manifest](../data/streaming/structure-v1/manifest.json)、[summary](../data/streaming/structure-v1/summary.json)、[结构测试](../experiments/streaming/test_structure.py)。它复用相同任务与逐阶段输入采样流，但使用自己记录的初始化种子和 ±0.01 初始权重；不能把它与 local-v1 当作初始参数完全相同的一次运行。

fixed、learned、random 三个稀疏组在种子内共享完全相同的初始权重与 mask，每个输出 32 条有效连接，总计 96 条。所有组都用真实外部标签残差乘输入活动更新现存权重。learned 每 50 步、每输出立即删除 2 条最小 |w| 的边，再从原先 inactive 候选中按窗口累计残差×输入相关量的绝对值选择 2 条新边；random 在同一频率与数量下随机选择。新边以非零小权重出生。dense 有 198 条连接，是更大预算的可学性控制。

这套小结构对照是固定预算下的即时成对替换，尚未实现用户随后提出的“新连接先出生、旧连接逐渐失活、出生和死亡数量可不同”。后续更自由的渐进方案必须独立报告，不能用本表替代其验证。

| 结构组 | 有效连接 | A1 后 A | B 后旧 A | B 后新 B | A2 后 A | A2 后旧 B |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 96 | 86.62±2.44 | 35.47±8.47 | 85.58±6.13 | 86.51±2.35 | 38.64±12.69 |
| learned | 96 | 97.30±0.40 | 34.91±11.58 | 96.58±0.63 | 97.28±0.20 | 42.51±10.10 |
| random | 96 | 78.26±7.51 | 37.16±6.99 | 82.96±2.06 | 81.04±0.87 | 34.57±4.59 |
| dense | 198 | 98.16±0.58 | 52.59±0.90 | 97.93±0.07 | 98.10±0.18 | 81.90±1.19 |

learned 在当前任务拟合上接近 dense，并优于相同预算的 fixed/random；它没有显示更好的 B 后旧 A 保持。这里的“learned”指根据任务误差信号选择新连接，不是让模型编写自己的结构规则。删除和出生次数确实反映 mask 的变化：learned/random 每个 seed 都发生 48 次改变，合计各新增 288、删除 288 条连接，结尾始终是 96 个非零权重，inactive 权重全为零；fixed 的 mask 不变。来源见 [重连规则与实际计数](../experiments/streaming/structure.py#L113)。

所有组仍分配 198 个权重数组位置和 198 个候选统计量；这是有效连接稀疏，不是事件驱动或真正稀疏硬件的算力、带宽优势。三种子全组正式运行约 5.636 秒 CPU 单线程。manifest 的三份源码哈希及全部 12 个 checkpoint 哈希已复核一致。这里只有有限次干扰与重学，不能据此宣称终身学习。

原 LIF 实现同时存在几种可独立开关的机制，不能统称为“按频率训练”：

| 原机制 | 代码中的实际信息与规则 | 本轮小实验覆盖范围 |
|---|---|---|
| rate 与活动迹 | 每神经元 rate EMA；trace 按时间常数衰减再累加 spike | frequency 仅测试输出次数的稳态偏置调节，不复刻 LIF 动力学 |
| 权重可塑性 | trace_pre × (trace_post−θ) 再乘外来标量/逐输出 m | local_delta 与 reward_local 只提取“局部活动+真实反馈”的受控问题，不声称复现原规则 |
| 反馈来源 | 配置可取环境动作方向、episode 成败、RPE 或 teacher_word 等 | 外部完整标签或二值环境正确性；没有自评头 |
| 预算型生长/剪枝 | 活跃 trace 选候选对，随机出生；超预算后按相对权重与年龄淘汰 | structure-v1 是另外定义的误差相关重连，不执行旧规则 |
| 结构 homeostasis | 高于目标率时删 E-E 入边，低于目标率时按迹加权选择源并生长 | 本轮未执行，也未证明率目标自动等价于任务价值 |

代码位置：[rate/trace](../src/core.py#L279)、[权重更新](../src/core.py#L372)、[活跃区出生](../src/core.py#L428)、[预算剪枝](../src/core.py#L458)、[结构稳态](../src/core.py#L312)、[反馈来源配置](../src/config.py#L289)。原始设计说明中对内生一致性与外部正确性的区分也见 [归档设计](archived/design-2026-10-25.md)。

一个具体的旧实现差别值得保留：after_m 模式先令 e=m·el，再除以 max|e|；当 m 为统一非零标量时，其绝对大小会被归一化抵消，只保留符号。before_m 先归一化 el 再乘 m，才保留该标量幅度。逐神经元反馈不能简单套用“全部幅度被抵消”的说法。另一个历史基线问题是某些 frozen 曾只关闭权重和拓扑，bias 仍更新；本次冻结测试明确覆盖了权重、偏置和辅助状态，见 [Plasticity 注释](../src/config.py#L229)。

概念与原始研究的对应如下；这些文献提供机制背景，不是本项目成绩的外部验证：

- [Oja 1982](https://doi.org/10.1007/BF00275687) 的受约束 Hebbian 规则分析了从平稳输入中提取主成分。它说明活动相关规则可以学习输入统计结构，不能从“没有任务标签”推断“没有任何学习”。
- [Izhikevich 2007](https://izhikevich.org/publications/dastdp.pdf) 将 STDP 触发的慢资格迹与多巴胺调制结合，用模型研究延迟奖励的归因；这不表示任意活动频率都能自动知道任务价值。
- [Bellec 等 2020，e-prop](https://doi.org/10.1038/s41467-020-17236-y) 将前向可计算的局部资格迹与学习信号结合。在线实现对理想学习信号作近似，并非完全不需要信用分配或导数。本项目没有实现或复现 e-prop。
- [Mocanu 等 2018，SET](https://doi.org/10.1038/s41467-018-04316-3) 使用弱权重删除和等量随机出生来更新稀疏连接；[Evci 等，RigL](https://arxiv.org/abs/1911.11134) 使用梯度信息指导连接更新。可修改拓扑已有实证先例，但这不自动证明抗遗忘，也不是任意自由生长的保证。
- [Hinton 的 Forward-Forward](https://arxiv.org/abs/2212.13345) 用正负样本与逐层 goodness 目标；[Baydin 等的 forward gradients](https://arxiv.org/abs/2202.08587) 用前向自动微分计算方向导数再估计梯度。它们与用户澄清的活动频率/局部可塑性是不同分支，本次没有运行这两种训练方法。

复现时使用新的输出目录，避免覆盖冻结结果：

~~~powershell
.\.venv\Scripts\python.exe -B -m unittest experiments.streaming.test_local_learning -v
.\.venv\Scripts\python.exe -B -m experiments.streaming.local_learning --seeds 0,1,2 --phase-steps 800 --eval-every 10 --output data/streaming/local-reproduction
.\.venv\Scripts\python.exe -B -m experiments.streaming.structure --seeds 0,1,2 --fan-in 32 --steps-per-phase 800 --batch-size 128 --lr 0.2 --eval-every 10 --rewire-every 50 --replace-per-output 2 --output data/streaming/structure-reproduction
~~~

总报告合并时应保留三条界限：局部可计算不等于无梯度；会学当前任务不等于会保存旧任务；固定预算即时重连不等于新连接出生后旧连接渐进失活。大规模模型及渐进结构的新结果须由各自冻结的配置、测试集指标与实际连接轨迹补充，不能从以上小任务外推。
