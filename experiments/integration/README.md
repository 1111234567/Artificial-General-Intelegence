# 完整连续状态中的旧经历、时间信用与局部共活动

本实验从上一轮 `online128` 的完整检查点继续运行，拆开内容可读性、合法动作时机和任务结束后的状态更新，并比较旧经历通过监督重放或前向局部规则参与学习的效果。36 组训练、180 个自主评估分支和 72 个额外诊断分支已完成，另有 36 个重复基线用于一致性核验。[完整结果](../../docs/integration-experiment-2026-10-27.md)记录重放的保持/适应取舍、有限梯度收益，以及本次共激活规则导致的严重行为退化。

冻结协议：[main-study-v1.json](../../docs/evidence/integration-2026-10-27/main-study-v1.json)。上一轮的源码与证据保留在 [continuity](../continuity/README.md)。与可塑性论文及 CBP 的区别另见[研究笔记](../../docs/evidence/integration-2026-10-27/paper-plasticity-notes.json)；本实验没有实现 CBP 的单元替换。

## 保留了哪些连续状态

种子为 0、1、2，各自从 A 阶段最终 `online128` 完整检查点分出 12 个分支。起点为每个 actor 已经历 131,072 tick、2,048 次更新的生命时刻。网络仍为 128 维简单连续循环，共 34,348 个参数；四个并发任务通道、96 维输入、每通道 11 个动作，与冻结环境一致。

参数、Adam、实时隐藏状态 `h`、外部时钟、未完成合同、预定事件、待呈现回执、随机数状态和有限 A 经历库一同延续。64 个 actor 保持各自世界与状态，共享权重，不重新配对。所有分支执行自身 argmax 动作；监督标签来自该分支行动前的因果世界状态。

训练块边界只截断计算图，保留 `h` 数值；没有世界或实时隐藏状态重置。新规则的辅助活动迹 `q` 在分支建立时初始化，此后随实时流和完整检查点继续。改变等待及截止后的空档分布只影响未来 NEW，已有合同和预定替换保持原条件。梯度覆盖长度与数值状态的持续长度分别报告。

## 冻结的 12 个分支

每个分支继续 65,536 个实时 tick，`dt=0.25`、batch=64。Adam 学习率为 0.001，WAIT 损失权重为 0.1。主线十个分支使用等待 4–8、最终合同截止到下次 NEW 的空档 4–12；两个 coverage 分支分别改变其中一项。`gap_range` 不是相邻 NEW 的到达间隔；后者还包含当前合同的时长、窗口及修改时机。取消周期以 CANCEL 命令时刻作为这一空档的起点。

| 分支 | 每块实时 tick / 梯度跨度 | 每块学习更新 | 旧经历的作用 |
|---|---|---|---|
| `frozen` | 64 / 无梯度 | 无 | 不读取旧块 |
| `bp64` | 64 / 64 | 当前监督 Adam 一次 | 不读取旧块 |
| `repeat64` | 64 / 64 | 当前块学两遍，Adam 两次 | 不读取旧块 |
| `replay64` | 64 / 64 | 当前块一次，旧块一次，Adam 两次 | 旧监督标签参与梯度 |
| `accum128` | 128 / 64 | 两段已截断图的损失合并，Adam 一次 | 不读取旧块 |
| `bp128` | 128 / 128 | 当前 128 tick，Adam 一次 | 不读取旧块 |
| `coact_live` | 64 / 64 | 当前 Adam 一次，局部 W 更新一次 | 读取旧块以匹配前向预算，丢弃其局部统计 |
| `coact_mix` | 64 / 64 | 当前 Adam 一次，局部 W 更新一次 | 当前与旧块局部统计各占一半 |
| `coact_mix_shuffled` | 64 / 64 | 同 `coact_mix` | 打乱局部规则中跨 actor 的前后活动配对 |
| `coact_only_mix` | 64 / 无梯度 | 仅局部 W 更新一次 | 当前与旧块局部统计各占一半 |
| `coverage_long_dense` | 64 / 64 | 当前 Adam 一次 | 等待 4–8，截止后空档 0.5–2 |
| `coverage_short_sparse` | 64 / 64 | 当前 Adam 一次 | 等待 1–4，截止后空档 4–12 |

表中的 64/128 只指实时块长度；每次旧块的 128-tick 前向另计。固定实时经历下，64-tick 分支共有 1,024 个块；128-tick 分支共有 512 个块。因此 `accum128` 与 `bp128` 才匹配更新频率、前向和监督暴露、只改变图截断位置。直接比较 `bp64` 与 `bp128` 同时改变了 Adam 次数。

A 库保留最多 32 个 128-tick 历史块，每块含全部 64 个 actor 的真实观察、原始因果标签、时间增量和历史 `start_h`。继续训练与测试数据不写入该库。每次旧块处理先无梯度运行 64 tick，再用后 64 tick 计算监督损失或局部统计；临时历史状态不覆盖实时 `h`，也不推进实时世界。

四个局部分支都执行相同长度的旧块前向。`coact_only_mix` 继承有监督训练得到的编码、读出和循环权重，只在本阶段停止标签梯度并局部改变 W；它不是从零开始的无监督能力获得。环境回执仍是公共输入。

## 前向局部规则实际做什么

对 post-by-pre 循环矩阵 W，代码以模型步进前后的状态收集统计：

```text
q_t = exp(-dt/tau) q_(t-1) + (1-exp(-dt/tau)) tanh(h_before)
C   = sum(dt * mean_batch[tanh(h_after) outer q_t]) / sum(dt)
s   = sum(dt * mean_batch[tanh(h_after)^2]) / sum(dt)
delta W = eta * (C - s[:, None] * W)
```

正式 `eta=0.001`、`tau=16`。混合分支先分别按物理时间归一化当前和旧块统计，再各取一半，最后只写一次 W。有监督混合分支的实时统计来自当前 Adam 之前的自身轨迹，旧统计来自随后独立的历史前向；局部更新最后使用当时的 W 施加。两类统计不保证由同一组权重生成。实时活动迹不在块边界清空。

shuffled 对照逐 tick 将 `q` 沿独立 actor 轴做随机非零循环移位，保持当前 tick 的活动边际和 post-squared 衰减统计，破坏原有的同轨迹配对；它不保证更新范数完全相等。历史辅助迹从零重建并预热，64 tick 约为一个 `tau`，不能声称完全恢复原历史迹或消除了历史隐藏状态与当前参数的差异。

这是非线性循环上的 Oja 风格启发式，不是时间信用的精确梯度、e-prop 资格迹、单元效用评分或永久保护机制。经典线性平稳输入的 PCA 保证不适用于这里。监督混合分支仍通过 BPTT 获得标签信用。

## 完整分叉与自身轨迹诊断

每个最终检查点冻结权重和学习附加状态，在原时刻分出完整 `model/h/world` 副本，各继续 2,048 tick。正式主评估显式指定以下五个条件；CLI 默认条件还包含 `conflict`，因此不能用默认值代替本协议。

| 条件 | 未来 NEW 等待 | 最终合同截止→下次 NEW 空档 |
|---|---|---|
| `iid` | 1–4 | 0.5–2 |
| `long_wait` | 4–8 | 0.5–2 |
| `longer` | 8–16 | 0.5–2 |
| `sparse` | 1–4 | 4–12 |
| `long_sparse` | 4–8 | 4–12 |

主指标为最终有效请求的无错误尝试完成率；普通完成率、FINISH 精确率和非法尝试另报。主分母仅含分叉后由测试 NEW 合同产生、截止窗口已经成熟的请求；继承请求、旧合同的后续替换和未成熟窗口分别记为 `inherited`、`carryover_contract`、`censored`。已经完成但窗口未结束的请求也暂不进入主分母。

`integration-own-trace-v2` 在模型自身真实动作造成的状态上旁观现有八个 FINISH logits，按任务年龄与阶段记录内容可读性；同时追踪每个成熟请求在 NEW/REPLACE 后、行动前的 cue 读出与首次观测到合法 FINISH 机会的读出。`both_correct`、`lost`、`gained`、`both_wrong` 划分同一批成对请求；缺 cue、从未到达合法机会和未成熟请求另外报告。初始 FINISH 头读错只表示该阶段不可读，不能证明内容未被编码；成对读出也不包含从未合法的请求。

首次错误对每个成熟请求只记一次，按 `invalid_code → stale_version → duplicate_begin → before_begin → early → late → wrong_payload` 给互斥主因，同时保留例如 `early+wrong_payload` 的联合模式。无任务动作单独按完成、取消、超时、尚无请求和继承原因未知分类，同时报告每 1,000 个空闲机会与每 1,000 个总通道机会的次数。正确读出旧内容且选择 WAIT 不是错误；实际重复退休版本/内容与单纯仍能读出旧内容分开记录。

协议只对 `frozen`、`bp64`、`replay64`、`coact_mix` 的三个种子执行额外配对干预：

| 干预 | 操作 | 协议条件 |
|---|---|---|
| `finish_content_oracle` | 只纠正该干预分支当前 active 通道已选的 FINISH 版本/内容，包括之后将取消或替换的请求 | `longer`、`long_sparse` |
| `timing_oracle` | 提供因果中点动作日程，保留模型条件 BEGIN 版本和八类 FINISH 内容 | `longer`、`long_sparse` |
| `completion_receipt_ablation` | 仅去掉一次已接受 FINISH 的前动作/结果观察特征 | `sparse` |
| `cancel_command_ablation` | 仅去掉一次 CANCEL 的命令特征，保留其他回执 | `sparse` |

所有干预都是诊断分支，会改变后续回执与内部状态；不能计入自主表现，也不能当作纯粹隔离内容或计时的证明。内容 oracle 在当下不直接改变 WAIT/BEGIN/FINISH 动作种类，但反馈会改变后续尝试及其时机，并非沿用固定基线时间表。时间 oracle 按构造在无任务时 WAIT，零空闲误动作不是模型学到的能力。各干预核对共同起点、外部生成器状态、事件指纹与成熟请求指纹；不同 coverage 分布之间不能要求未来事件完全相同。诊断文件中重复的 `none` 基线必须与主评估一致，汇总只计一次。

## CLI 与产物

从仓库根目录运行。以下使用新的复现目录；正式文件位于 `data/integration/main-v1`、`evaluation-v1`、`diagnostics-v1`，不要覆盖。

```powershell
$source = 'data/continuity/main-20261027-v1/seed_000/online128/checkpoint.pt'
.venv\Scripts\python.exe -m experiments.integration.run --from-checkpoint $source --output data/integration/reproduce-v1/seed_000/bp64 --mode bp64 --ticks 65536 --device cuda --threads 1 --checkpoint-every 256
```

把 `--mode` 改为表中任一分支，并为种子 1、2 使用各自的源检查点和新输出目录。默认分支配置与协议一致；`--duration`、`--gap`、`--live-ticks`、`--gradient-horizon`、`--local-lr`、`--local-tau` 覆盖值会改变协议，须另行标记。

完整恢复继续相同世界时刻；`--ticks` 是本继续阶段的累计终点，不是再追加的 tick 数。

```powershell
.venv\Scripts\python.exe -m experiments.integration.run --resume data/integration/reproduce-v1/seed_000/bp64/checkpoint.pt --output data/integration/reproduce-v1/seed_000/bp64 --ticks 65536 --device cuda --threads 1 --checkpoint-every 256
```

`--stop-after-updates N` 在完整块及其全部更新之后保存并停止。`checkpoint.pt` 包含完整继续状态；`checkpoint_NNNNNN.pt` 是按块编号保留的快照。`training.jsonl` 保存学习过程，`report.json` 保存配置、阶段计数、计算和更新范数；`sources/` 保存对应源码。严格加载核对冻结的 continuity 与 integration 源码，不能通过修改旧运行源码延续原协议。

```powershell
.venv\Scripts\python.exe -m experiments.integration.diagnose --checkpoint data/integration/reproduce-v1/seed_000/bp64/checkpoint.pt --output data/integration/reproduce-eval/bp64-seed000.json --conditions iid long_wait longer sparse long_sparse --interventions none --steps 2048 --dt .25 --device cuda
.venv\Scripts\python.exe -m experiments.integration.diagnose --checkpoint data/integration/reproduce-v1/seed_000/bp64/checkpoint.pt --output data/integration/reproduce-diagnostics/bp64-seed000-longer.json --conditions longer --interventions none finish_content_oracle timing_oracle --steps 2048 --dt .25 --device cuda
```

诊断 CLI 会读完整旧版或新版检查点，保留调用进程 RNG 设置并核对源状态未改变。`--keep-trace` 额外输出逐 tick 观察/动作/隐藏状态指纹及评分字段，文件会明显增大。诊断 CLI 没有 `--threads` 参数；正式调度 wrapper 在进程中设置 PyTorch 和环境线程数。

汇总命令在对应输入完成后执行，使用同一冻结 manifest 进行计数、预算、源文件与分叉审计：

```powershell
.venv\Scripts\python.exe -m experiments.integration.report --root data/integration/main-v1 --evaluations data/integration/evaluation-v1 data/integration/diagnostics-v1 --manifest docs/evidence/integration-2026-10-27/main-study-v1.json --output docs/evidence/integration-2026-10-27/aggregate
.venv\Scripts\python.exe -m pytest experiments/integration -q
```

论文启发的终点状态观察另用 `python -m experiments.integration.inspect_state`；它先要求全部 36 个正式终点及报告就绪，再在 CPU 直接读取保存的 W、h 和世界标志，不调用模型或推进环境。定义见[事后观察方案](../../docs/evidence/integration-2026-10-27/state-inspection-plan.json)。单次终点的跨 actor 协方差秩不等于任务表征秩，不能用来挑选模型或单独宣称可塑性改善。

## 解释边界

分别报告实时经历、额外当前前向、旧块前向及 burn-in、监督 tick、梯度次数、局部写次数、更新范数和显式存储。相同梯度次数不代表相同 FLOPs；活动迹、经历库及优化器的存储不包含在 34,348 个模型参数中。共享 GPU 的墙钟耗时不能作为独占硬件速度或能耗基准。

三种子是训练重复，64 个 actor 与同一起点的多次分叉不是独立训练样本。模型、参数和实时状态在各自轨迹中共同演化；跨生命时刻的成绩差异不是单独的权重知识测量。有限监督任务上的改善、旧条件保持、长等待外推与持续可塑性是不同问题，不能从本轮某个总分推出通用终身学习或长期生物机制。
