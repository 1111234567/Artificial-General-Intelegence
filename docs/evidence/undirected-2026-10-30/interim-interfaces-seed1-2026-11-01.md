# seed_001 四种接口/方向的中期行为子集

生成时间 UTC：2026-11-01T11:35:22.332347+00:00；已完成 4/4 分支。

仅限 recorded512 / fixed / warmup，主训练终点 tick196608，每族 short 与 long_both 各2048步冻结续跑。每种配置只有一个训练 seed；下表是四个 family 各自比率的等权算术均值，不是跨 seed 均值，没有跨 seed 方差、置信区间或最终胜者结论。主终点早于撤回/回归，不代表最终保持率；本子集独立于42分支的完整审计。

|接口|方向|条件|完成率 %|干净完成率 %|每千次实际闲置机会的动作数|
|---|---|---|---:|---:|---:|
|core|undirected|short|30.2983|17.2529|39.2452|
|core|undirected|long_both|0.0000|0.0000|0.0000|
|core|untied|short|38.3571|19.5459|69.2897|
|core|untied|long_both|0.0000|0.0000|0.0992|
|ports|undirected|short|0.0000|0.0000|0.0000|
|ports|undirected|long_both|0.0000|0.0000|0.0000|
|ports|untied|short|0.0000|0.0000|0.0000|
|ports|untied|long_both|0.0000|0.0000|0.0000|

## long_both 的已成熟 primary 请求诊断

下表是四族的原始计数之和；requests / completed / clean_completed 已逐族核对主分母。FINISH 尝试计数可以对同一请求累计多次。首次合法机会的内容读出是现有八个 FINISH logits 的条件 argmax，实际行动仍由完整十一个动作的 argmax 决定；该读出不等于状态信息上限，也不等于整体完成行为。

|接口|方向|requests|completed / clean|已 BEGIN|观察到合法机会|活跃期 FINISH 尝试|合法时 FINISH 尝试|首次合法内容正确 / 读出数|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|core|undirected|684|0 / 0|684|684|6|0|289 / 684|
|core|untied|684|0 / 0|684|684|12|0|291 / 684|
|ports|undirected|684|0 / 0|0|0|0|0|0 / 0|
|ports|untied|684|0 / 0|0|0|0|0|0 / 0|

逐族 matured / completed / clean_completed、实际机会与闲置动作原始计数、重算比率、训练及评估命令、队列完成记录和输出 SHA256 均保存在同名 JSON。对读取的评估执行冻结的 `_checked_evaluation` 校验，并核对自身内容哈希、声明 spec、终态和 checkpoint 绑定；实际训练输出及所报主终点 checkpoint 只流式哈希，没有加载。源码指纹按冻结协议和两个命令清单核验。未重跑评估、模型、世界或训练。
