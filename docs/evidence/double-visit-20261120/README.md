# 双复查实验的复算入口

复算时先解压[改道包](../reroute-20261120/audit-data.zip)，再解压[本轮审计包](audit-data.zip)。[在新检出中恢复与计分](../reproduce-2026-11-20.md)

[总报告](../../double-visit-2026-11-20.md)给出首次真实选门、旧事实的存留和状态移植的主要结论。

| 问题 | 材料 |
|---|---|
| 两份历史只差什么，旧图是否真的留出 | [物理世界与信息审计](world.md) · [环境](../../../experiments/pulse_game/double_visit_world.py) |
| 神经元从哪些亲自走过的路线取得什么梯度 | [训练与核验](fit-results.md) · [原训练的地图与继承指纹](fit-inheritance-audit.json) |
| 整个个体是否在两份相反任务中都选择正确 | [正式评估及同钟空试](eval-results.md) · [事前判据](eval-plan.md) |
| 第一次地方性更新有没有被覆盖 | [公布目标前的读数](memory.md) · [互换状态的因果影响](trace-swap-results.md) |
| 直接给房间线索或给出正确预测后如何行动 | [直接房与门询问](localization.md) · [受约束的正确/错误报告](rescue-results.md) |

本机检查点、真实开发轨迹和冻结观察在 `data/double-visit-20261120`。用[训练程序](../../../experiments/pulse_game/double_visit_fit.py)的 `load_final(birth, condition)` 加载六个出生之一。条件名为 `baseline` 或 `identity`，部署期间不需要外部解码器或干预。
