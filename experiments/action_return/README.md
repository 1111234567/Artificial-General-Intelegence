# 自主行动与旧任务返回

这组实验从两类已经训练好的无向图存档出发。统一端口组冻结骨干，改变从公开观察或连续内部状态到实际动作的读出。另一组能完成部分短任务，继续训练时保持各自继承的连接集合，测旧任务撤出与返回。[中文研究报告](../../docs/reports/actions/action-return-experiment-2026-11-03.md)

| 文件 | 作用 | 输出 |
|---|---|---|
| [readout.py](readout.py) | 比较原生端口与三种启动读出，检查自由行动和动作回执。[拟合细节](readout-README.md) | [训练和评估协议](../../data/action-return/readout-v1/protocol.json) |
| [readout_swap.py](readout_swap.py) | 在相同状态保留是否启动的选择，交换两个启动版本 | [配对结果](../../data/action-return/readout-swap-v1/summary.json) |
| [finish_readout.py](finish_readout.py) 与 [crossed_readout.py](crossed_readout.py) | 加入条件 FINISH 读出，分别接非输入状态和公开观察的启动读出 | [结束动作](../../data/action-return/finish-readout-v1/summary.json) · [四格结果](../../data/action-return/crossed-readout-v1/summary.json) |
| [refit_policy.py](refit_policy.py) | 保持初始化、训练预算和外部事件相同，比较帮助轨迹与自主行动轨迹 | [逐种子结果](../../data/action-return/refit-policy-v1/summary.json) |
| [absence_study.py](absence_study.py) 与 [absence.py](absence.py) | 六个分支连续训练，统计撤出期间实时与历史复习暴露 | [分母与配对差](../../data/action-return/absence-v1/report.md) |
| [edge_identity.py](edge_identity.py) | 在未剪枝对照的三个真实端点，瞬时静默同一组曾退出的无向边 | [消融协议](../../data/action-return/edge-identity-v1/protocol.json) |

所有决策使用 FP64。自主行动的主分析取各分叉中未参加读出拟合的四个 actor。原骨干见过这四个 actor。旧任务撤出和瞬时删边实验都沿用原评估器的全部 actor。两条实验线各自以三个训练种子作为独立重复。

## 核对本机存档

从仓库根目录运行。独立核对会重新计算指标并写入相应的核验 JSON。绘图命令从已完成的汇总读取数据。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -q experiments/action_return
.\.venv\Scripts\python.exe -m experiments.action_return.independent_audit --study refit
.\.venv\Scripts\python.exe -m experiments.action_return.independent_edge_audit
.\.venv\Scripts\python.exe -m experiments.action_return.plot_refit
```

## 从源存档复现

重跑需要对应的原始完整存档、逐块审计轨迹及协议中记录的文件。协议绑定本机实际绝对路径和 SHA256。保留原结果，在独立工作副本中按依赖顺序生成新协议，准备确切的来源路径。模块用 `python -m experiments.action_return.<模块>` 调用。每个动作实验从三个种子分别运行，各个家族和难度的评估会由程序生成。

| 顺序 | 模块与命令 |
|---|---|
| 启动读出 | `readout prepare` → `run --seed 0/1/2`（各运行一次）→ `summary` |
| 版本交换 | `readout_swap prepare` → 各种子 `run` → `summary` |
| 结束与交叉对照 | 先 `finish_readout`，后 `crossed_readout`，各用 `declare` → 各种子 `run` → `summarize` |
| 等预算复拟合 | `refit_policy prepare` → `launch` → `summarize` |
| 缺席与返回 | 先运行 `absence_tests` 生成来源和恢复验证，再 `absence_study prepare` → `launch` → `report` |
| 相同边的瞬时干预 | `edge_identity prepare` → 各种子 `run` → `summary` |
