# 固定比例旧经验复习与后缀保持

从同一个完整学习终点，比较连续学 B、不时复习旧 A、等量复习旧 B。主方案每八次更新有一次复习，旧 A/B 片段逐对配平；另固定一段停止复习后的 B 学习后缀。[主方案](analysis-plan.md) · [后缀保持方案](retention-plan.md)

| 程序 | 用途 |
|---|---|
| [bank.py](bank.py) | 固定旧 A/B 配对片段与选取顺序，保留真实完成后的 WAIT |
| [train.py](train.py) | 256 次同预算更新，当前权重重建旧前史，主 B 状态始终连续 |
| [train_audit.py](train_audit.py) | 独立重建真实 B/复习更新、prefix anchors、Adam 与状态链 |
| [retention_data.py](retention_data.py) | 从同一真实 B 世界接续 8,192 tick，无新 A 或复习 |
| [retention_train.py](retention_train.py)、[retention_audit.py](retention_audit.py) | 各模型接续 64 次 B 更新，并独立核验 |
| [evaluate.py](evaluate.py)、[eval_audit.py](eval_audit.py) | 两个固定终点的新世界与原世界返回，原生行为、时间侧录和规则答案签名 |

数据根为 `data/episodic-review/`，已有学习与撤出工件只读。三种子、三组均保留初始模型、Adam、h 和世界。复习时的辅助 h 不覆盖主状态；主世界继续接收原来的 B 教师回执。

这里的连续性指教师轨迹、模拟时钟与数值状态的接续。训练消费已冻结的真实教师轨迹，不是在物理实时环境中异步执行复习。

每个重放窗口有 32 tick 前史和 96 tick 目标槽监督，覆盖正确 BEGIN、等待、FINISH 及其后空闲。其它槽与更早历史仍是原始观察，但不提供它们的梯度标签。监督库每类 256 条，重建当前 h 还要读取完整旧公开历史，不能将存储成本说成只有这些窗口。

两种复习组有完全相同的更新日程、监督掩码和标签数量。不复习组使用更多实时 B 梯度标签，重放组则增加完整前史重建的前向成本，报告会分别记账。固定更新预算不代表相同 FLOPs、墙钟时间或全部监督行数。

第 256 次更新和无复习后缀结束后的 study320 都保存、冻结和评估，不根据成绩选择其中一个。时间预测从不参与实际动作选择，错误答案分类只读取已经执行的真实行为。

[完整结果、成本和解释边界](../../docs/reports/memory/episodic-review-experiment-2026-11-07.md)已整理。测试模块为 [bank_test.py](bank_test.py)、[train_test.py](train_test.py)、[retention_data_test.py](retention_data_test.py)、[retention_train_test.py](retention_train_test.py)、[eval_test.py](eval_test.py)。
