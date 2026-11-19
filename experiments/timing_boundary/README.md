# 结束边界的状态读出

冻结上一轮三种 FINISH 监督位置的 GRU，在新的连续混合任务中分开检查时间边界与内容。比较原动作头、线性读出、一个固定的小型非线性读出，以及随机 GRU 和当前输入对照。实验定义见 [判断方案](analysis-plan.md)，独立核验范围见 [审计方案](audit-plan.md)。

这是只读状态诊断。拟合的读出不接管世界中的动作，原模型的自由完成率与读出准确率分列。连续状态仅在每条独立测试流的起点初始化，流内不按请求重置。

结果见 [研究报告](../../docs/reports/actions/timing-boundary-experiment-2026-11-04.md)。数据写入独立的 `data/timing-boundary/`，旧轮源码、检查点与协议保持原样。运行前阅读各模块的命令帮助和已冻结协议，已有正式输出不覆盖重跑。

| 模块 | 用途与存档 |
|---|---|
| [data.py](data.py) | 新持续流、共同教师历史与自身行动历史，`data/timing-boundary/v1/` |
| [probe.py](probe.py) | 固定的时间、内容和剩余时间读出，全部拟合后统一冻结，`data/timing-boundary/probe-v1/` |
| [audit.py](audit.py) | 独立原生世界重放、请求选择、状态与读出复算，`data/timing-boundary/audit-v1/` |
| [results.py](results.py) | 不拟合模型，提取已有计数与作图，`docs/evidence/timing-boundary-2026-11-04/` |

三个测试模块分别是 [data_test.py](data_test.py)、[probe_test.py](probe_test.py)、[test_audit.py](test_audit.py)。数值程序使用 CPU 单线程，多种子独立进程并行。完整数据、模型和预测数组仍在本机数据目录，便携 JSON 保留来源 SHA。
