# 单图内部计算的复现入口

[正式审计包](audit-data.zip)含本轮训练终点和冻结干预。需要的上游亲本可按[复算包目录](../reproduce-2026-11-20.md)取得。

[阶段报告](../../native-rehearsal-2026-11-20.md)也记录了先行的单草稿外置回执对照。这里是完整循环图实验的证据。

| 问题 | 方案、动作与结果 |
|---|---|
| 没有额外输入时，20 或 40 节点的图能否自行改进后续动作 | [事前协议](protocol.md) · [两套新世界和逐出生读数](behavior.json) · [时长干预](wait.json) · [原始对照与训练终点核验](primary-audit.json) |
| 模型运行是否依赖另外保存的菜单或外部重演器 | [跨进程恢复动作与完整末态](stream-resume.json) · [流式验证程序](../../../experiments/pulse_game/native_rehearsal_stream.py) |
| 副区的反馈是否令本体想起特定旧经历 | [预定方向与针对性控制](reactivation-protocol.md) · [回流、实际动作与局限](reactivation-results.md) · [物理复验](reactivation-audit.json) |
| 静默时的活动能否预测改写的行为后果 | [方案](prospect-plan.md) · [新几何与不同干预](prospect-results.md) |
| 生命内让另一区提前试算自己的下一笔权重改动 | [共享当前参数与试算路径](trial-plan.md) · [逐拍执行程序](../../../experiments/pulse_game/native_rehearsal_trial.py) |

15 份主训练检查点与优化器保存在本机 `data/native-rehearsal-20261120/<kind>/seed-<birth>.pt`，真实行动账本在 `eval/<seed>/`。逐个体可用 [加载入口](../../../experiments/pulse_game/native_rehearsal_eval.py) 恢复。两组生命内试算的检查点在同一数据目录的 `trial-training` 下。主训练只学已有边权重，初始连接支持及跨区连线由实验提供。
