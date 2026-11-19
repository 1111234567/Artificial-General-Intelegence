# 持续循环网络的连接渐进退出与再生长

本目录把 [streaming](../streaming/README.md) 中允许新旧连接共存的思路接入当前持续状态控制器。模型仍用 `dh/dt=-h+W tanh(h)`，本轮直接改变循环矩阵 W 的实际非零支持。**21 组训练、42 个阶段端点及 210 个评估分叉已完成并通过审计。** 定向再生长确实发生，但没有超过同强度只退出对照；部分保持收益与空闲误动作增加并存。分种子结果、图表及边界见[本轮报告](../../docs/rewiring-experiment-2026-10-28.md)。

## 更新规则

起点继承完整 A 存档。128×128 的 W 初始全连接，首批新生边必然是曾经退出位置的重新激活。没有预先剪掉旧权重来腾出空位。

每 64 tick 对完整 Heun 计算求 BPTT 梯度。前向直接使用 W，失活位置的权重精确为零，因此这些位置仍有普通的潜在连接导数。活跃位置的梯度进入原 Adam，并在联合裁剪前排除失活位置的梯度；候选梯度使用同一活跃裁剪系数。

已有边先做 Adam 更新，再做每步固定幅度的软收缩；到零才退出，退出时清空这条边的旧动量。失活位置用原 Adam 的学习率、beta 和 epsilon 累积候选更新，但具有各自从退出后开始的时间计数。每 8 次优化器更新，将候选累积软收缩后仍非零的位置作为新边。窗口中途退出的边按其实际收到信号的次数计成本，新生不要求同时删除其他边。

这是明确指定的 Adam 加收缩规则，未宣称它是某个单一 L1 目标的精确 Adam 近端算法。候选缓冲、计数、私有随机状态和全部训练状态一起存档。正式学习没有使用旧经历重放。

## 对照与预算

三个种子，每个 7 组：原稠密 Adam，及两个收缩幅度 `5e-5 / 2e-4` 下的只退出、学习信号生长、随机位置生长，共 21 组。两个强度预先声明，全部报告，不按评估成绩选择。

随机组将自己提出的新边数量及带符号权重按接收神经元保留，随机分配到已有空位；后续死亡次数、实际容量轨迹仍可能不同。原稠密组提供最大连接容量对照。初始全连接使“峰值静态容量”退化为稠密组；本轮不引入另一个被硬剪的起点。

每组从完整 A 继续 65,536 tick 的 B（等待 4–8、空档 4–12），再返回 A 条件（等待 1–4、空档 0.5–2）继续 65,536 tick。h、世界、待发生事件、合同、回执、优化器和候选记忆持续保留；分段仅截断梯度。B 和返回后的 A2 各保存一个完整端点，分别在五种环境上冻结评估，共 42 个端点、210 个评估分叉。

## 文件与入口

| 文件 | 职责 |
|---|---|
| [plasticity.py](plasticity.py) | 有效连接、候选更新、出生与退出、完整结构状态 |
| [run.py](run.py) | 继承旧控制器与 Adam、跨阶段训练和完整恢复 |
| [study.py](study.py) | 有限协议、来源绑定、批量训练和只读评估 |
| [report.py](report.py) | 端点、来源、分母、结构计数与三种子统计核验 |

使用仓库 `.venv\Scripts\python.exe`。正式协议生成后不覆盖，恢复沿原清单继续；生成新实验应使用新输出路径。运行状态位于 `data/rewiring/main-v1/`，证据归档位于 `docs/evidence/rewiring-2026-10-28/`。

```powershell
.venv\Scripts\python.exe -m experiments.credit.windows_runner --kind batch --plan docs/evidence/rewiring-2026-10-28/training-commands-v1.json --status data/rewiring/main-v1/training-execution.json --workers 10
.venv\Scripts\python.exe -m experiments.rewiring.evaluate_ready --plan docs/evidence/rewiring-2026-10-28/evaluation-commands-v1.json --status data/rewiring/main-v1/evaluation-execution.json --training-status data/rewiring/main-v1/training-execution.json --workers 12
.venv\Scripts\python.exe -m experiments.rewiring.report --protocol docs/evidence/rewiring-2026-10-28/study-protocol-v1.json --root data/rewiring/main-v1 --output data/rewiring/recheck
.venv\Scripts\python.exe -m pytest -q experiments/rewiring
```

实际非零边变少并不节省此实现的稠密矩阵计算；全部潜在连接和候选统计仍占存储。这里检验连接组织与任务表现，不将图上的环长度换算成记忆时间。
