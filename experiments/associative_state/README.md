# 关联状态、重复与改写

在已训练的 GRU 动作执行器前比较三种符号记忆适配器：共激活累加、关联残差修正和同为 64 状态标量的 GRU。输入证据来自公开的成功 FINISH 回执；目标不从 oracle 标签直接写入记忆。原执行器继续决定 BEGIN、WAIT 和 FINISH。

[第一版方案](analysis-plan.md) · [共享地址后续方案](shared-plan.md) · [研究来源](../../docs/memt-research-2026-11-07.md)

两版训练、144 份行为评分与独立审计均已完成。[完整报告和结果边界](../../docs/associative-state-experiment-2026-11-07.md)

| 文件 | 用途 |
|---|---|
| [model.py](model.py) | add / delta 和普通递归状态；独立地址向量 |
| [bridge.py](bridge.py) | 公开回执绑定器、逆置换与每 actor 新映射世界 |
| [train.py](train.py) | 固定预算的随机关联预训练及来源冻结 |
| [evaluate.py](evaluate.py) | 八个固定分支、真实动作、完整状态和请求账本 |
| [launch.py](launch.py) | 九个 CPU 工作进程；先全部训练、再全部评分 |
| [shared.py](shared.py) | 独立的共享地址后续；不改第一版源码与工件 |
| [audit.py](audit.py) | 全部指标独立复算与指定分支的完整原生重放 |
| [results.py](results.py) | 全部检查点汇总，不选最好终点 |

第一版数据位于 `data/associative-state/v1/`；共享地址后续位于 `data/associative-state/shared-v2/`。两版均保留，原实验工件只读。

这是监督符号适配器：公开 cue 与成功回执的绑定、写入时机和符号逆置换由程序提供；预训练的内容损失不经过实际动作执行器。它不构成自主学习绑定、MemT 完整复现或原 GRU 端到端持续学习。

复现第一版：

```powershell
.\.venv\Scripts\python.exe -m pytest experiments/associative_state -q
.\.venv\Scripts\python.exe -m experiments.associative_state.train declare
.\.venv\Scripts\python.exe -m experiments.associative_state.launch train --workers 9
.\.venv\Scripts\python.exe -m experiments.associative_state.launch eval --workers 9
.\.venv\Scripts\python.exe -m experiments.associative_state.audit --replay-workers 9
.\.venv\Scripts\python.exe -m experiments.associative_state.results
```

共享地址后续：

```powershell
.\.venv\Scripts\python.exe -m experiments.associative_state.shared declare
.\.venv\Scripts\python.exe -m experiments.associative_state.shared train --workers 9
.\.venv\Scripts\python.exe -m experiments.associative_state.shared eval --workers 9
.\.venv\Scripts\python.exe -m experiments.associative_state.audit --root data/associative-state/shared-v2 --replay-workers 9
.\.venv\Scripts\python.exe -m experiments.associative_state.results --variant shared-v2
```

声明、训练与评分不覆盖已有运行；审计和汇总可以重新生成。正式工件的源码指纹必须匹配冻结协议。原执行器来源需要本机已有的 teacher-timing 三个中点动作终点。
