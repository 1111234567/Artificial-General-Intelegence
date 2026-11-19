# 从提示到结束窗口的内容与计时

三个实验使用上轮自主轨迹复拟合组的同一批 FP64 模型。它们回答不同的问题，取样和动作处理也不同。

| 实验 | 观察对象 | 数据和程序 |
|---|---|---|
| 自由行动 | 模型在持续世界里自行选择的动作、每个成熟请求三个时刻的八种 FINISH 得分，以及没有到达窗口的请求 | [旁路记录与分母](natural.py) · [协议](../../data/content-timing/natural-v1/protocol.json) |
| 固定动作的配对分叉 | 从同一真实背景改变原始内容或处理时长。正确启动后等候，保留环境回执，读取冻结模型原本想执行的动作 | [分叉与精确干预](matched_flow.py) · [协议](../../data/content-timing/matched-v1/protocol.json) |
| 跨 actor 的状态读出 | 在同一批配对分叉的 24 个非输入状态上，分别拟合原始符号、家族规则映射后的内容，并比较同阶段拟合及提示读出跨时间使用 | [拟合与阴性对照](probe.py) · [事先规定的分析](probe-plan.md) |

## 本机入口

在仓库根目录运行，各种子使用独立的单线程 CPU 进程。正式结果绑定原始存档、程序和协议。已有协议和结果保留在各自的版本目录。完整重跑需要在独立工作副本中准备这些源存档和依赖协议。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -q experiments/content_timing
.\.venv\Scripts\python.exe -m experiments.content_timing.natural --help
.\.venv\Scripts\python.exe -m experiments.content_timing.matched_flow --help
```

自由行动的程序重新执行原有 24 个评估分叉，逐个核对历史动作、请求和结果。保持同一分叉的所有背景和内容变体在同一训练或测试侧，是状态读出成立的条件。程序同时检查将四个内容变体换成同一个平均状态后，四选一测试准确率是否恰为 25%。分析中的 actor 4–7 对上轮策略读出的拟合留出，早期循环骨干训练曾见过这些 actor。

首次传播由 [onset.py](onset.py) 在保存的背景上单独补测。迁移核对用 [transfer.py](transfer.py) 将配对读出应用到真实自由轨迹，连续特征重放按 [摘要算法勘误](../../data/content-timing/transfer-v1/replay_correction.json) 执行。域内校准用 [natural_probe.py](natural_probe.py) 在其他真实 actor 上拟合相同的读出。汇总见 [研究报告](../../docs/reports/actions/content-timing-experiment-2026-11-03.md)，原始分母与输入指纹可用 [独立核验程序](audit.py) 检查。
