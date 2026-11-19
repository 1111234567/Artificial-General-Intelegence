# 连续状态下的时长覆盖与规则撤出

从上一轮三个中点动作组终点继续，完整保留模型、Adam、数值 h 和原生世界。比较保持原 A/B 任务与时长、扩大时长覆盖、只继续 B 三个分支。[预定方案](analysis-plan.md) · [研究报告](../../docs/coverage-retention-experiment-2026-11-07.md)

| 程序 | 用途 |
|---|---|
| [data.py](data.py) | 恢复完整世界，仅改变未来 NEW 的时长/家族，记录继承任务与真实教师回执 |
| [train.py](train.py) | 从源 Adam256 和源 h 接续 256 次更新，不执行新预热 |
| [evaluate.py](evaluate.py) | 新世界中三个时长范围、全部成熟请求的统一评估 |
| [live_eval.py](live_eval.py) | 从训练终点原世界和 h 让 A 返回，不重置状态、不更新权重 |
| [train_audit.py](train_audit.py)、[eval_audit.py](eval_audit.py) | 独立重建更新、原生轨迹、动作、末态和指标 |
| [error_signature.py](error_signature.py)、[error_signature_audit.py](error_signature_audit.py) | 事后只读区分正确规则、另一规则的答案及其它错误 |
| [results.py](results.py) | 便携计数、配对结果、训练暴露和图 |

数据位于 `data/coverage-retention/`。原世界完整状态包含仍活跃的请求、已有回执和调度；future B-only 不删除边界遗留的 A 任务。主训练流的数值 h 跨块延续，128 tick 仅限制梯度图。

Fresh 和 live 使用不同请求，不能直接作配对状态干预。time 侧录中的历史字段名 `overdue_beyond_training` 只表示就绪后超过 0.25 单位，当前中点训练已经覆盖其中一部分，不按字段名判定未见状态。

测试入口为 [data_test.py](data_test.py)、[train_test.py](train_test.py)、[eval_test.py](eval_test.py)。协议在运行前冻结，已有产物不覆盖。事后错误签名的首个解析器缺字段检查在生成分类前失败，修订及原声明另存于该诊断目录。
