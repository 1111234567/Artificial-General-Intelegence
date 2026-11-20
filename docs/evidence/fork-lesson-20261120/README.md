# 整个个体使用克隆区预测的证据

[正式审计包](audit-data.zip)含出生参数、末步检查点和正式路线记录。原封存加载器的运行环境及上游亲本见[复算包目录](../reproduce-2026-11-20.md)。

[阶段报告](../../fork-lesson-2026-11-20.md)列出了预测位置、获得梯度的参数、真实已选路线和冻后才执行的备选路线之间的区别。

| 问题 | 原始记录与实现 |
|---|---|
| 两条路线在什么时候兑现，主区看见哪些输入 | [世界说明](world.md) · [任务和神经接口](../../../experiments/pulse_game/fork_lesson_world.py) |
| 整体训练能否预告未走门，并在自己选门时使用两条桥 | [训练协议](fit-protocol.json) · [开发轨迹审计](fit-audit.json) · [独立部署与剪断通路](eval.md) |
| 主区发往副区的真实电流是否有对象选择性 | [询问、不同私有经历和固定主区的交叉干预](dialogue-results.md) |
| 只教本体的发送轴突能否改进预测 | [限定80条连接](sender-plan.md) · [跨阶段梯度与窗口互换](sender-credit.md) |
| 额外开放一次内试探，能否让同一主区预告随未执行动作而变化 | [试探与真实走门的交叉](whatif.md) · [冻结数值和端口核验](whatif-fit-audit.json) · [只传一位侧别的对照](bit-relay.md) |
| 不增加读出网络，用原生神经元读出试探收益 | [严格 40 节点复现](native-request.md) · [主区单元14及逐出生区间](native-request-results.json) · [固定侧别替换真实副本回流](native-relay.md) |
| 是否重现了先前未显示在门口的感官细节 | [跨期校准和晚期静默检验](reinstatement-results.md) |

本机完整开发轨迹、推理参数和冻结评估在 `data/fork-lesson-20261120`。六种原生整体教学条件各有三份 `fit/<kind>/seed-<birth>.pt`，另一次内试探的检查点在 `whatif-fit`。通过训练代码中的 `load_final` 还原。图和外层优化器的参数只在开发期训练；生内快权重按神经规则照常更新。
