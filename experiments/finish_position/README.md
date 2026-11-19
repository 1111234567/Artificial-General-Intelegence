# 固定末步 WAIT 后移动 FINISH 正例

比较复习旧 A 时，在原窗口中点或刚刚合法时教 FINISH。新 B 仍采用相同演示，不改两组的训练样本身份、预算或部署动作规则。[研究报告](../../docs/finish-position-experiment-2026-11-03.md) · [分析口径](analysis-plan.md)

| 程序 | 用途 |
|---|---|
| [bank.py](bank.py) | 从原 A 请求重放两种合法正例的时刻，保留原始教师标签、实际回执和正确结束类别 |
| [train.py](train.py)、[train_audit.py](train_audit.py) | 两组等预算训练，核对实际移动的监督标签与独立更新 |
| [interior_train.py](interior_train.py)、[eval_interior.py](eval_interior.py) | 两端结果已知后，对同一旧请求单独补测两者之间的一个固定整数位置 |
| [eval_finish.py](eval_finish.py) | 原生动作闭环评估，分开统计是否启动、抢答、合法出手和窗口漏发 |
| [switch_test.py](switch_test.py) | 在相同的 317 对冲突请求上自由行动，从真实动作与回执复算两次是否都正确 |
| [boundary_readout.py](boundary_readout.py) | 在相同公开示范轨迹上读取同一请求三个相邻阶段的动作倾向 |

冻结协议、检查点、逐步事实与分母放在 `data/finish-position/`。执行训练需要原有库、上轮的完整存档、项目 CUDA 环境及新的输出目录。日常查阅可以直接读取报告及图表。
