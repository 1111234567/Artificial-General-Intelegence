# 实际教师的结束时机

保留原模型与训练方法，比较最早合法结束与合法窗口中点结束的真实教师示范。两种教师分别使用动作监督和动作加时间监督，固定三种子、256 次更新。[分析方案](analysis-plan.md)

这里改变的是完整示范方式，包含标签与实际回执历史。两侧外部事件、请求身份和总输入预算相同，时间有效标签数与负时间覆盖单独计数。earliest 分支必须逐位复现 [上一轮](../remaining_time/README.md) 对应模型、Adam 和隐藏态。

| 程序 | 用途 |
|---|---|
| [data.py](data.py) | 绑定原 earliest 数据，实际采集中点教师流，核对外部事件、当前监督与标签覆盖 |
| [train.py](train.py) | 四条件同预算继续训练，earliest 历史控制复现 |
| [evaluate.py](evaluate.py) | 新世界中的原生自主行动及只读时间侧录 |
| [eval_audit.py](eval_audit.py) | 请求状态机与时间指标的独立重建 |
| [common_history.py](common_history.py) | 在两种实际教师历史上交叉重放四个冻结模型 |
| [phase_readout.py](phase_readout.py) | 从已有中点历史事后提取同请求两个固定合法时刻，不重新推理或训练 |
| [train_audit.py](train_audit.py)、[common_audit.py](common_audit.py)、[eval_replay.py](eval_replay.py)、[phase_audit.py](phase_audit.py) | 更新、共同历史、原生运行与阶段读出的独立核验 |
| [results.py](results.py) | 分种子汇总、配对差异和图 |

数据根为 `data/teacher-timing/`。已冻结原脚本、检查点和流不修改；新增程序可以导入旧函数或在独立进程内绑定输出位置，避免覆盖旧实验。完整数值 h 在每条流中持续，128 tick 只截断梯度。

[完整研究报告](../../docs/teacher-timing-experiment-2026-11-06.md)同时保留主要自主成绩、外推失败和事后阶段读出的限制。测试入口为 [data_test.py](data_test.py)、[train_test.py](train_test.py)、[common_history_test.py](common_history_test.py)、[phase_readout_test.py](phase_readout_test.py)。
