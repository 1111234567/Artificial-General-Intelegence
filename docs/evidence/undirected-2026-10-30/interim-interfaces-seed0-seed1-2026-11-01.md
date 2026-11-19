# 两个训练种子分别列值的接口/方向中期子集

生成时间 UTC：2026-11-01T11:35:22.343485+00:00。

预定范围：seed0 和 seed1 各四种接口/方向，recorded512 / fixed / warmup，主训练终点 tick196608；每族 short、long_both 各2048步冻结续跑。每行只代表相应训练种子，完成率和闲置率都是该种子四族各自比率的等权平均。没有对两个种子求均值、方差、置信区间或走势；第三个种子和42分支完整审计尚待总表。此处也不宣布最终胜者或最终保持率。

|训练 seed|接口|方向|条件|完成率 %|干净完成率 %|闲置动作/千次实际闲置机会|
|---:|---|---|---|---:|---:|---:|
|0|core|undirected|short|28.5107|18.2692|49.2082|
|0|core|undirected|long_both|0.0000|0.0000|0.0485|
|0|core|untied|short|32.3345|21.2657|46.0238|
|0|core|untied|long_both|0.0000|0.0000|0.0000|
|0|ports|undirected|short|0.0000|0.0000|0.0000|
|0|ports|undirected|long_both|0.0000|0.0000|0.0000|
|0|ports|untied|short|0.0000|0.0000|0.0000|
|0|ports|untied|long_both|0.0000|0.0000|0.0000|
|1|core|undirected|short|30.2983|17.2529|39.2452|
|1|core|undirected|long_both|0.0000|0.0000|0.0000|
|1|core|untied|short|38.3571|19.5459|69.2897|
|1|core|untied|long_both|0.0000|0.0000|0.0992|
|1|ports|undirected|short|0.0000|0.0000|0.0000|
|1|ports|undirected|long_both|0.0000|0.0000|0.0000|
|1|ports|untied|short|0.0000|0.0000|0.0000|
|1|ports|untied|long_both|0.0000|0.0000|0.0000|

## long_both 请求诊断，按训练 seed 分列

每行是相应种子的四族原始计数之和；requests/completed/clean_completed 已逐族核对 primary 主分母。首次合法内容读出只是在实际状态上取现有八个 FINISH logits 的条件 argmax，实际行动仍取十一个动作的 argmax。读出不代表状态信息上限或整体完成行为；0/0 表示没有读出，不是0%准确率。

|训练 seed|接口|方向|requests|completed / clean|已 BEGIN|观察到合法机会|活跃 FINISH 尝试|合法 FINISH 尝试|首次合法内容正确/读出数|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|0|core|undirected|636|0 / 0|636|636|1|0|272 / 636|
|0|core|untied|636|0 / 0|636|636|8|0|269 / 636|
|0|ports|undirected|636|0 / 0|0|0|0|0|0 / 0|
|0|ports|untied|636|0 / 0|0|0|0|0|0 / 0|
|1|core|undirected|684|0 / 0|684|684|6|0|289 / 684|
|1|core|untied|684|0 / 0|684|684|12|0|291 / 684|
|1|ports|undirected|684|0 / 0|0|0|0|0|0 / 0|
|1|ports|untied|684|0 / 0|0|0|0|0|0 / 0|

同名 JSON 保留逐族原始分母、计数及两份独立证据的 SHA256。各种子的独立 JSON 包含命令清单、源码、保存的队列完成记录、实际输出及 checkpoint 字节指纹和冻结验证器的校验记录。seed0 原脚本及产物未修改。
