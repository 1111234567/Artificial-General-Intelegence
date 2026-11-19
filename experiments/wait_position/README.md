# 固定旧请求，移动处理期 WAIT 监督

使用同一套 A 存档、旧请求库及真实 B 示范，对比中点、随机和最高风险时刻，另以就绪前末步作补充。部署中的模型只接收公开输入。[研究报告](../../docs/reports/actions/wait-position-experiment-2026-11-03.md) · [预定指标](analysis-plan.md)

| 程序 | 用途 |
|---|---|
| [bank.py](bank.py)、[bank_audit.py](bank_audit.py) | 绑定原来选中过的 A 请求，重放真实世界，标出已经启动且尚未就绪的 WAIT 候选 |
| [train_review.py](train_review.py)、[train_audit.py](train_audit.py) | 配平复习样本、梯度与候选扫描，逐步只改 WAIT 所在的位置，核对模型/Adam 及连续状态 |
| [bank_diagnostics.py](bank_diagnostics.py) | 从真实选点日志统计距离就绪时间、风险最高时是否真想发 FINISH、位置覆盖情况 |
| [eval_position.py](eval_position.py)、[eval_timing.py](eval_timing.py) | B 的早期曲线、混合任务中旧 A/新 B 的冲突、实际抢答及合法窗口漏发 |
| [switch_pairs.py](switch_pairs.py)、[fixed_history.py](fixed_history.py) | 原配对世界中的真实行动，以及同一参考轨迹上的动作意向 |
| [last_train.py](last_train.py)、[eval_last.py](eval_last.py)、[last_checks.py](last_checks.py) | 固定选末步的单独训练和两种行为测试，使用相同原始请求和分母 |
| [results.py](results.py) | 合计逐种子分母、复习条件的早期曲线与末端结果，生成便携 JSON 和图 |

本机产物位于 `data/wait-position/`。新目录保留原试验、补充试验及源码绑定；检查文档或查已归档结果无需启动训练。读取表格时区分任意有错后成功的普通完成、全程无错的严格完成，以及两条请求都无错的成对严格完成。
