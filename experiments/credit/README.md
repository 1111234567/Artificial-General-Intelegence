# 连续状态、独立读出与前向时间信用

本目录接续 `experiments/integration/`，不修改旧实验。模型仍是 128 维连续状态网络：

```
dh/dt = -h + W tanh(h)
h <- h + encoder(observation)
logits = head(tanh(h))
```

完整的世界、隐藏状态、回执、时间、随机数和 Adam 状态持续继承。训练每 64 tick 更新一次参数；这不是每 64 tick 重置记忆。

## 两个独立问题

`probe.py` 只观察上一轮冻结模型的自主轨迹。它分别从原始 h、模型实际计算的 tanh(h)、公开输入训练线性读出器；目标包括内容、版本、任务是否活跃和是否开始。训练/验证/测试按完整 actor 划分，读出器不能看到阶段、时间、请求身份或私有世界信息。它不替模型行动。

`run.py` 对同一批 A 阶段完整存档继续学习。主要比较如下。

|模式|学习规则|旧经验|
|---|---|---|
|frozen|不更新|无|
|readout_only|解析交叉熵梯度，只训练读出|无|
|bp64|64 tick 反向传播|无|
|bp_replay|当前和旧经验梯度各占一半|A 阶段固定库|
|diag_sym / diag_sym_replay|对角近似前向资格迹；读出权重传回误差信号|无 / A 库|
|diag_random / diag_random_replay|同上；固定随机反馈传回误差信号|无 / A 库|
|diag_reset64|每块清空资格迹|无|
|diag_instant|每 tick 清空资格迹|无|
|diag_shuffle|打乱不同 actor 的神经元学习信号|无|

后三组是信用分配对照，清空的仅是新增资格迹，不清空模型隐藏状态。随机反馈和打乱对照依然保留精确的读出层梯度。

`readout_only` 借用 reservoir/LSM 的“动态状态供读出学习”思路，但内部网络已由 A 阶段训练，并非随机脉冲 LSM。所有学习组都有外部因果标签；本轮不声称实现自主自评。

## 前向资格迹的近似边界

`local.py` 对实际 Heun 数值积分推导前向敏感度，不把原来的活动 EMA 当作导数。每个突触保留自身输入对本神经元状态的影响 E；编码器也有对应 F。误差与这些资格迹相乘，得到手写梯度估计。无需保存反向时间图。

为限制状态开销，每个 Heun 阶段都丢弃跨神经元传播的敏感度。因此它不是完整 RTRL，也不是 RFLO 或 e-prop 的严格复现。参数更新后继续携带的敏感度另有陈旧近似。迹数值连续并不保证很久以前的信用不会衰减。

旧经验的 128 tick 在更新参数前单独运行，前 64 tick 无直接损失、后 64 tick 计损失。反向组的图保留整个 128 tick；前向组的旧分支从零资格迹开始经过全部 128 tick。两者都只做一次混合梯度 Adam 更新。旧分支不覆盖活跃状态和活跃资格迹。

## 运行与证据

`prepare_probe.py` 冻结独立读出协议；`batch.py` 只执行协议内的有限命令列表。`evaluate.py` 复用旧实验已验证的自主轨迹观察器。`cycle.py` 按预先指定的 A/B 顺序反复进入熟悉环境，检查再学习和保留；这不是新任务可塑性的充分证明。

协议和源码哈希在 `docs/evidence/credit-2026-10-27/`，原始结果在 `data/credit/`。主实验、独立读出、熟悉环境复访分别报告，不能把观察器正确率当作模型自主完成率。

|文件|职责|
|---|---|
|`run.py`、`local.py`|冻结的连续训练与前向敏感度实现|
|`probe.py`、`evaluate.py`|冻结自主轨迹的观察与读出，不介入行动|
|`cycle.py`|预定 A/B 复访序列，保留完整学习状态|
|`cycle_cpu.py`|v2 测量适配：训练保留 CUDA 完整状态，冻结副本统一在 CPU 评估；引用不可变 v1 保存点|
|`launch.py`|从完整原点开始或从完整检查点恢复单个运行|
|`batch.py`、`windows_runner.py`、`evaluate_ready.py`|有限命令队列、Windows 状态文件重试、完成后才评估|
|`report.py`、`probe_report.py`、`figures.py`|审计、种子级汇总与科学图表|
|`statistics.py`、`tangent_audit.py`|固定表征的统计等价、固定轨迹的敏感度数值验证|
|`prepare_probe.py`、`prepare_study.py`|在运行前生成一次协议；拒绝覆盖已冻结协议|

在仓库根目录使用项目环境。以下命令执行已冻结的有限列表；完成项按清单和输出核验后跳过，未完成主分支由 `launch.py` 恢复其完整状态。Windows 使用重试包装入口，避免并发读取时状态文件短暂被占用导致队列中断。

```powershell
.\.venv\Scripts\python.exe -m experiments.credit.windows_runner --kind batch --plan docs/evidence/credit-2026-10-27/main-commands-v1.json --status data/credit/main-v1/execution.json --workers 6
.\.venv\Scripts\python.exe -m experiments.credit.windows_runner --kind evaluation --plan docs/evidence/credit-2026-10-27/evaluation-commands-v1.json --status data/credit/evaluation-v1/execution.json --training-status data/credit/main-v1/execution.json --workers 4
.\.venv\Scripts\python.exe -m experiments.credit.windows_runner --kind batch --plan docs/evidence/credit-2026-10-27/cycles-commands-v2.json --status data/credit/cycles-v2/execution.json --workers 4
.\.venv\Scripts\python.exe -m experiments.credit.report --protocol docs/evidence/credit-2026-10-27/study-protocol-v1.json --root data/credit/main-v1 --evaluations data/credit/evaluation-v1 --cycles data/credit/cycles-v2 --cycle-measurement-revision docs/evidence/credit-2026-10-27/cycle-measurement-revision-v2.json --output docs/evidence/credit-2026-10-27/aggregate
```

复访命令需要主训练已完成。当前正式复访结果使用 v2 统一 CPU 测量；v1 保留原 CUDA 分叉作附加对照，不参与主成绩。主评估队列可以在主训练期间启动，只读取带完整完成标记的不可变检查点。运行边界、原始日志和失败后恢复记录均保留；没有后台定时任务。

全量独立读出汇总位于 `data/credit/probe-aggregate-v1/`，文档证据目录保留按固定维度选取的紧凑视图、全部三个种子与完整原始汇总哈希。选择不依据分数高低；所有分组和配对比较仍在全量结果中。可用 `probe_report --compact-evidence FULL_DIR --output NEW_DIR` 重建新目录中的紧凑视图。

```powershell
.\.venv\Scripts\python.exe -m pytest experiments/credit experiments/integration experiments/continuity --import-mode=importlib -q
.\.venv\Scripts\python.exe scripts/check_docs.py
```

## 相关方法的区别

活动频率、活动相关统计、带奖励的局部规则、带导数的资格迹、前向模式自动微分和 Forward-Forward 是不同方法。重放提供经验，可以与其中多种学习规则组合。它不等同于反向传播，也不等同于共激活。

固定表征下的线性平方损失是一个清楚的特例：累计 XᵀX 和 XᵀY 可以代替保留所有样本来求同一个岭回归解。单有 XᵀX 不包含目标关联；表征持续改变时，原统计通常也不能代表重新计算后的经验。
