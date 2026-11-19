# 分开学习时机和内容，冻结或更新循环部分

接续 [时间边界诊断](../../docs/reports/actions/timing-boundary-experiment-2026-11-04.md)，用相同持续示范、监督行和更新预算，比较只学习新动作读出与同时学习原 GRU 循环部分。主要解释边界见 [预定方案](analysis-plan.md)。新头从零开始，原策略作为另列参照。

| 程序 | 用途 |
|---|---|
| [model.py](model.py) | 三类模式、两类启动版本、八类结束内容。解码成原生动作，推理没有私有合法性门控 |
| [data.py](data.py) | 一次连续的最早合法教师流，保留会取消/替换的所有活跃训练行 |
| [train.py](train.py) | 同起点、同 256 更新的 frozen/full，完整数值状态跨块延续 |
| [evaluate.py](evaluate.py) | 所有成熟请求的原生行动评估，分开漏启动、无窗口、窗口漏发、抢答与空闲 |
| [train_audit.py](train_audit.py) | 独立重建实际 CUDA 更新、Adam、梯度和状态连续性 |
| [eval_audit.py](eval_audit.py)、[eval_replay.py](eval_replay.py) | 请求状态机与完整世界/策略重放的两套独立核验 |
| [diagnose.py](diagnose.py) | 冻结终点，在原示范、新示范和自身历史上分开读模式与内容 |
| [results.py](results.py) | 提取分母、配对行为变化与绘图 |

正式数据位于 `data/action-split/`，逐程序协议在运行前冻结。已有正式目录不覆盖重跑。测试模块是 [model_test.py](model_test.py)、[data_test.py](data_test.py)、[train_test.py](train_test.py) 和 [eval_test.py](eval_test.py)。训练用 CUDA float32，CPU 单线程进程并行采集、评估和复核。

原模式损失下全训练组没有学会结束，因此另有一个明确事后选择的固定三类等权归一化对照。它仍从相同原循环参数和零新头开始，保留同一数据、各项原始标签及更新预算。原失败结果保留，补测采用全新测试请求。没有把同样本预算说成同样的损失权重或运算量。

补测入口是 [balanced_train.py](balanced_train.py)、[balanced_eval.py](balanced_eval.py)、[balanced_results.py](balanced_results.py)。[balanced_audit.py](balanced_audit.py)、[balanced_eval_audit.py](balanced_eval_audit.py)、[balanced_eval_replay.py](balanced_eval_replay.py) 分别核对更新、指标与完整运行。[完整结果与限制](../../docs/reports/actions/action-split-experiment-2026-11-05.md)
