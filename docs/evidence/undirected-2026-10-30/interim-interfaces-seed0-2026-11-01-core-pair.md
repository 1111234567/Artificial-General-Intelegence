# seed_000 四种接口/方向的中期行为子集

生成时间 UTC：2026-11-01T11:06:03.185630+00:00；已完成 2/4 分支。

仅限 recorded512 / fixed / warmup，主训练终点 tick196608，每族 short 与 long_both 各2048步冻结续跑。每种配置只有一个训练 seed；下表是四个 family 各自比率的等权算术均值，不是跨 seed 均值，没有跨 seed 方差、置信区间或最终胜者结论。主终点早于撤回/回归，不代表最终保持率；本子集独立于42分支的完整审计。

|接口|方向|条件|完成率 %|干净完成率 %|每千次实际闲置机会的动作数|
|---|---|---|---:|---:|---:|
|core|undirected|short|28.5107|18.2692|49.2082|
|core|undirected|long_both|0.0000|0.0000|0.0485|
|core|untied|short|32.3345|21.2657|46.0238|
|core|untied|long_both|0.0000|0.0000|0.0000|

逐族 matured / completed / clean_completed、实际机会与闲置动作原始计数、重算比率、训练及评估命令、队列完成记录和输出 SHA256 均保存在同名 JSON。对读取的评估执行冻结的 `_checked_evaluation` 校验，并核对自身内容哈希、声明 spec、终态和 checkpoint 绑定；实际训练输出及所报主终点 checkpoint 只流式哈希，没有加载。源码指纹按冻结协议和两个命令清单核验。未重跑评估、模型、世界或训练。

待完成：
- `seed_000__ports__undirected__recorded512__fixed__warmup`
- `seed_000__ports__untied__recorded512__fixed__warmup`
