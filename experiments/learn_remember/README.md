# 实际习得、旧规则保持与一次性命令记忆

从相同任务的两条既有线索出发。第一条在原 80 节点图上等预算比较自主经历校正和使用原策略经历的校正。第二条用公开输入直接训练循环策略，在新世界里独立检验习得、旧家族撤出和定额历史复习。文档中的规则编号从 0 开始。[研究报告](../../docs/reports/memory/learn-remember-experiment-2026-11-03.md)

| 程序 | 用途 |
|---|---|
| [correction.py](correction.py)、[correction_audit.py](correction_audit.py) | 从上轮旧图继续学习、匹配未来测试及独立审计 |
| [gru_train.py](gru_train.py)、[gru_review.py](gru_review.py) | 单层 GRU 从零习得短/宽任务并撤出旧规则，再从相同存档运行固定复习分支 |
| [symmetric_gru.py](symmetric_gru.py)、[symmetric_review.py](symmetric_review.py) | 使三个门的隐藏连接分别共享无序节点对系数，保持相同的训练预算，使用原 GRU 算法收集及评估 |
| [evaluate_gru.py](evaluate_gru.py)、[evaluate_symmetric.py](evaluate_symmetric.py) | 在与训练分离的相同请求上测自主动作。按种子、规则和符号分别计数 |
| [memory_assay.py](memory_assay.py)、[memory_audit.py](memory_audit.py) | 同一背景下改变原始符号及家族，固定后续输入，在等待中途交换内部状态，并核对公开输入及真实教师窗口 |
| [evaluate_deployment.py](evaluate_deployment.py)、[state_transfer.py](state_transfer.py) | 对比留出世界自行预热与训练端点延续，并在同一陌生世界中单独换入训练端点 h |
| [results.py](results.py) | 合计留出请求、核对模型间的请求身份，并绘制三种子旧规则保持图 |

训练需要 CUDA 环境与原有公共任务实现。已有结果固定在 `data/learn-remember` 的各版本目录。重新训练请使用独立工作副本和输出目录，避免覆盖正式检查点。既有文件可从仓库根目录用下列命令复核。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m experiments.learn_remember.symmetric_gru test
.\.venv\Scripts\python.exe -m pytest -q experiments/learn_remember/evaluate_gru_test.py
.\.venv\Scripts\python.exe -m experiments.learn_remember.memory_audit
.\.venv\Scripts\python.exe -m experiments.learn_remember.reciprocity_audit
.\.venv\Scripts\python.exe -m experiments.learn_remember.results
```
