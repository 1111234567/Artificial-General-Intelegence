# 房内改道与首次选门的证据

[正式审计包](audit-data.zip)含本轮 30 份训练终点和所需的六个上游身体。[校验和使用方式](../reproduce-2026-11-20.md)

[阶段报告](../../reroute-2026-11-20.md)说明哪些指标支持中间房预测、哪些指标检验了实际选门。

| 问题 | 入口 |
|---|---|
| 从身体真正获得的感觉是否足以重建地图 | [环境、覆盖和分割](world.md) · [物理模型](../../../experiments/pulse_game/reroute_world.py) |
| 教中间房能否改善第一次行动 | [训练账本](fit-results.md) · [原始冻结结果与区间](eval-results.md) · [冻结评估脚本](../../../experiments/pulse_game/reroute_eval.py) |
| 从未训练的旧地图组合能否迁移 | [严格地图结果](novel-map.md) · [全场景检验](../../../experiments/pulse_game/reroute_novel.py) · [改变一笔观察的配对结果](update-results.json) |
| 中途换入房间身份会怎样影响预测和动作 | [供体、同归控制与物理结果](midpoint-results.md) · [冻结干预的实现](../../../experiments/pulse_game/reroute_midpoint.py) |
| 副区自身能否被读出“下一房”，这一活动方向是否影响真实走门 | [跨图读数和等能量对照](axis-results.md) · [协议](axis-plan.md) |
