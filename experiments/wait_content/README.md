# 在等待中学习结束内容

比较两种同预算训练。两者在同一批真实轨迹上训练原有共享边和原有 FINISH 读出，都保留每个位置的动作监督。第二组把一半附加的八类内容标签从结束时移到同一请求的处理阶段。这个位置的动作目标仍是 WAIT。最终指标来自策略自由行动。[研究报告](../../docs/reports/actions/wait-content-experiment-2026-11-03.md) · [预定比较口径](analysis-plan.md)

| 程序 | 职责 |
|---|---|
| [data.py](data.py) | 从完整端点持续采集冻结原策略真正遇到的公开观察、自己执行的动作和教师标签。在同请求、同更新块内配对内容位置 |
| [train.py](train.py) | 两组共同的 FP64 截断反传、动作损失、掩码、参数和优化器。保存逐步预算、完整端点及第一步核验快照 |
| [eval.py](eval.py) | 在同一段公共历史上为每套权重恢复连续状态，随后分别执行真正的十一动作策略，按请求身份和实际空闲机会计数 |
| [common_history.py](common_history.py) | 让三个策略读取原策略的同一份测试经历，分开诊断等待中的内容排序和首次教师结束位置的出手意愿 |
| [long_followup.py](long_followup.py) 与 [long_audit.py](long_audit.py) | 复核长任务低基率结果。延续原分叉到 8,192 tick，单独统计原窗口后发起且成熟的请求，逐条核对原前缀 |
| [data_audit.py](data_audit.py) 与 [audit.py](audit.py) | 复核采集轨迹、监督预算、抽取的优化步骤，以及逐 tick 的实际行为 |
| [figure.py](figure.py) | 用从行为轨迹独立重建的分子和分母绘图 |

从仓库根目录使用 `.venv`。旧协议记录了本机真实存档路径与哈希，重建实验需具备对应来源。正式实验保留已有结果；各模块提供自己的命令行帮助。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -q experiments/wait_content/train_tests.py experiments/wait_content/eval_test.py
.\.venv\Scripts\python.exe -m experiments.wait_content.audit
.\.venv\Scripts\python.exe -m experiments.wait_content.figure
```
