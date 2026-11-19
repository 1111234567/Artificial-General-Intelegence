# 软收缩、稀疏允许集合、连接更替与空闲惩罚

本轮从三个已有的稀疏 B 存档继续学习 A，完整继承参数、h、世界、Adam、候选历史与 RNG。**54 组训练及 285 个冻结评估分叉已完成并通过审计。** 固定稀疏允许集合加软收缩已复现只退出的主要保持表现；额外空闲惩罚大幅减少误动作，同时降低了普通完成率。全量强度、配对统计和控制取舍见[实验报告](../../docs/factors-experiment-2026-10-29.md)。

| 模式 | 允许更新的位置 | 每步软收缩 | 变成零之后 |
|---|---|---:|---|
| `dense` | 全部位置 | 0 | 仍由普通 Adam 更新 |
| `dense_shrink` | 全部位置 | 2e-4 | 仍由普通 Adam 更新 |
| `fixed` | 源 B 的非零位置，集合固定 | 0 | 原允许位置仍可更新 |
| `fixed_shrink` | 同一固定集合 | 2e-4 | 原允许位置仍可更新，保留矩 |
| `retire` | 随退出缩小 | 2e-4 | 沿用旧永久退出规则 |
| `grow` | 可退出和再生长 | 2e-4 | 沿用旧候选更新与交接规则 |

`dense` 表示全位置允许学习，起始权重仍是共同的稀疏 B。`fixed` 表示固定允许集合，实际非零数可改变。`retire`、`grow` 的候选历史和绝对结构计数从旧存档继承，新阶段变化另计偏移。

训练损失为原加权 CE（WAIT 权重 0.1）加 `idle_lambda * mean_idle(-log p(WAIT))`，系数取 0、0.1、1。idle 是 `observe` 后、`act` 前捕获的已启用但无活跃任务位置；没有任务时正确等待与有任务时等待分开。这个掩码只用于训练反馈，模型仍接收原 96 维输入，自主选择全部 11 类动作。

单分支训练入口通过 `--idle-lambda` 调节该项，配置默认值为 0；更改系数应创建新的分支目录。当前正式表格只覆盖预先声明的 0、0.1、1，不把未运行的其他值称为已验证设置。

| 文件 | 职责 |
|---|---|
| [structure.py](structure.py) | 六种结构策略、允许集合与实际支持、状态恢复 |
| [run.py](run.py) | 原 B 存档延续、空闲损失与训练诊断 |
| [study.py](study.py) | 冻结协议、来源绑定、训练与冻结评估 |
| [queue.py](queue.py) | 有限 GPU/CPU 队列与可调整的共同进程预算 |
| [report.py](report.py) | 最多六进程的完整存档审计、三种子配对统计和空闲原因/年龄诊断 |
| [plot_results.py](plot_results.py) | 从通过审计的汇总生成科学图 |

运行时使用仓库 `.venv\Scripts\python.exe`。既有运行中切勿再启动同一队列；中断恢复前先确认原进程已退出。源码或研究目标改变后应建立新版本，不能改旧协议使其接受新实现。

```powershell
.venv\Scripts\python.exe -m experiments.factors.queue --training-plan docs/evidence/factors-2026-10-29/training-commands-v1.json --evaluation-plan docs/evidence/factors-2026-10-29/evaluation-commands-v1.json --baseline-plan docs/evidence/factors-2026-10-29/baseline-commands-v1.json --status-root data/factors/main-v1 --limits data/factors/main-v1/worker-limits.json
.venv\Scripts\python.exe -m experiments.factors.report --protocol docs/evidence/factors-2026-10-29/study-protocol-v1.json --root data/factors/main-v1 --output data/factors/recheck --workers 6
.venv\Scripts\python.exe -m pytest -q experiments/factors
```

进程上限初始为训练 10、CPU 12、共同总数 14，防止进程数相加挤入交换空间；新作业开始前检查可用内存。可改 `worker-limits.json` 调整调度，无需改变模型或重新开始学习。阶段损失桶中的 `legal_begin/finish` 按教师标签分组，教师在窗口中点选择 FINISH，它们不等于环境的所有合法完成机会。
