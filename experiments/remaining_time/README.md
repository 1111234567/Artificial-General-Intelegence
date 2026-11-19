# 保留原策略，辅助预测剩余时间

复制原 GRU 循环和完整十一动作头，附加一个每槽标量时间读出。时间值只接受监督和用于诊断，不进入模型输入、动作解码或合法性门控。[分析方案](analysis-plan.md) · [研究报告](../../docs/remaining-time-experiment-2026-11-06.md)

| 程序 | 用途 |
|---|---|
| [model.py](model.py) | 保持原前向函数，四组梯度路由与分组裁剪 |
| [targets.py](targets.py) | 从已有真实教师流派生有符号时间标签，逐行重建 accepted-BEGIN 时刻 |
| [train.py](train.py) | probe、time、action、joint 四组同 256 次更新，数值 h 连续 |
| [evaluate.py](evaluate.py) | 原生自主行为，以及不干预动作的时间预测侧录 |
| [common_history.py](common_history.py) | 在同一份真实教师历史上只读重放各终点 |
| [train_audit.py](train_audit.py) | 独立复算实际 CUDA 梯度、分别裁剪、Adam 和 h |
| [eval_audit.py](eval_audit.py)、[eval_replay.py](eval_replay.py) | 独立重建请求分母、时间统计、原生世界、原策略与辅助读出 |
| [results.py](results.py) | 提取逐种子行为和时间误差，生成便携汇总与图 |

协议与完整数组在 `data/remaining-time/`，训练公开历史来自已冻结的 `data/action-split/streams-v1/`，原有源码和工件保持原样。各运行命令见模块 `--help`，正式输出不覆盖。使用 CUDA float32 训练，CPU 单线程多进程评估。时间误差以环境时间单位报告，一个 tick 为 0.25 单位。

测试入口是 [model_test.py](model_test.py)、[targets_test.py](targets_test.py)、[train_test.py](train_test.py)、[eval_test.py](eval_test.py)、[common_history_test.py](common_history_test.py)。probe 的原策略始终冻结；time 只冻结动作头，循环改变仍会改变实际策略。
