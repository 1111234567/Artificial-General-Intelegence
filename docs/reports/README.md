# 分主题研究报告

[实验总览](../experiments.md)汇总研究结论和适用边界，[证据索引](../evidence/README.md)定位协议、读数和原始档案。新报告按问题放入相应主题目录，可在下方添加入口。已进入发布或完成清单的报告及这些报告依赖的相对链接保留原址。

- [脉冲游戏、真实反馈与跨游戏迁移](#pulse-game)
- [局部可塑性、拓扑和原生习得](#structure)
- [初始化搜索和执行性能](#initialization)
- [内容、行动与时机](#actions)
- [记忆读写、规则保持与复习](#memory)
- [连续状态、时间信用与规划](#continuous-state)

<a id="pulse-game"></a>

## 脉冲游戏、真实反馈与跨游戏迁移

- 11-19 [晨间结果](../pulse-game-morning-2026-11-19.md)汇总 12 个独立训练出生、实际反馈学习、情境冲突、回访、主动探查和长生命验证。
- 11-19 [夜间实验记录](../pulse-game-night-2026-11-19.md)按路线保留研究问题、对照、负结果与后续检查。
- 11-19 [异步回执与跨游戏迁移](../pulse-game-transfer-boundaries-2026-11-19.md)涵盖回执归因、物体辨别和同一生命反复切换。
- 11-19 [证据与复现入口](../pulse-game-evidence-2026-11-19.md)和[本机数据入口](../../data/pulse-game/README.md)提供逐路线证据、发布包与校验信息。

<a id="structure"></a>

## 局部可塑性、拓扑和原生习得

- 11-19 [有向图局部可塑性与回执记忆](../local-rule-experiment-2026-11-19.md)记录三组 128 代终点、配对留出和回执分叉。
- 11-02 [启动覆盖、删边候选与 FP64 速度](structure/coverage-exit-experiment-2026-11-02.md)研究成功启动后怎样选退出候选。
- 10-30 [共享无向图的实验设计](../undirected-experiment-2026-10-30.md) · [完整对照结果](../undirected-results-2026-10-30.md)。
- 10-30 [原生习得与功能性组织](../organization-experiment-2026-10-30.md)对照非线性支路、嵌套耦合、生长和退出。
- 10-29 [随机出生后的交错课程与长间隔返回](../lifelong-experiment-2026-10-29.md) · [收缩、连接资格、退出与空闲损失](../factors-experiment-2026-10-29.md)。
- 10-28 [已有回路与非线性扩展](../loops-experiment-2026-10-28.md) · [连接渐退与再生长](../rewiring-experiment-2026-10-28.md)。

<a id="initialization"></a>

## 初始化搜索和执行性能

- 11-11 [搜索协议与恢复入口](initialization/seed-evolution-experiment-2026-11-11.md) · [FP32 审计通过的完整结果](../evidence/seed-evolution-2026-11-11-large/report.md) · [CPU/GPU 执行与性能](initialization/seed-evolution-performance-2026-11-11.md)。
- 11-15 BF16 扩展在 11-16 暂停，已记录 17,224／23,424 个候选。[进度](../../data/seed-evolution/2026-11-15-scale-bf16-v1/progress.json) · [暂停记录](../../data/seed-evolution/2026-11-15-scale-bf16-v1/pause-20261116.json)。

<a id="actions"></a>

## 内容、行动与时机

- 11-06 [实际教师的结束时机](../teacher-timing-experiment-2026-11-06.md) · [时间读出与原策略](../remaining-time-experiment-2026-11-06.md)。
- 11-05 [分开训练时机与动作内容](actions/action-split-experiment-2026-11-05.md)。
- 11-04 [相邻时间边界与内容读出](actions/timing-boundary-experiment-2026-11-04.md)。
- 11-03 [等待监督放在哪一步](actions/wait-position-experiment-2026-11-03.md) · [首次合法就教结束的影响](../finish-position-experiment-2026-11-03.md)。
- 11-03 [何时读错结束内容](actions/content-timing-experiment-2026-11-03.md) · [等待中学习结束内容](actions/wait-content-experiment-2026-11-03.md)。
- 11-03 [可读状态、行动与旧任务返回](actions/action-return-experiment-2026-11-03.md) · [从读出接到自主 BEGIN](actions/action-return-readout-2026-11-03.md)。

<a id="memory"></a>

## 记忆读写、规则保持与复习

- 11-10 [固定循环特征，只训练记忆读写](memory/frozen-trace-experiment-2026-11-10.md)。
- 11-07 [学习请求绑定与读写地址](memory/learned-binding-experiment-2026-11-07.md) · [关联状态及共享地址](../associative-state-experiment-2026-11-07.md) · [MemT 来源与适用边界](../memt-research-2026-11-07.md)。
- 11-07 [旧经历复习及后缀保持](memory/episodic-review-experiment-2026-11-07.md) · [时长覆盖及旧规则撤出](../coverage-retention-experiment-2026-11-07.md)。
- 11-03 [长任务习得和旧规则保持](memory/learn-remember-experiment-2026-11-03.md) · [旧经历对新规则的迁移](memory/experience-transfer-experiment-2026-11-03.md) · [等预算复习选哪些旧请求](memory/review-selection-experiment-2026-11-03.md)。
- 10-26 [反复经历与巩固](memory/consolidation-experiment-2026-10-26.md) · [持续记忆与原生循环](../persistent-experiment-2026-10-26.md)。
- 10-26 [持续输入、学习读写与结构渐变](../streaming-experiment-2026-10-26.md) · [局部活动学习补充](../streaming-local-notes-2026-10-26.md)。

<a id="continuous-state"></a>

## 连续状态、时间信用与规划

- 10-27 [完整连续状态、长等待与经验重放](../continuity-experiment-2026-10-27.md) · [内容、时间与局部共激活](../integration-experiment-2026-10-27.md) · [前向信用与独立读出](../credit-experiment-2026-10-27.md)。
- 10-26 [持续动力学与双任务时间控制](../dynamics-experiment-2026-10-26.md) · [消耗式规划与共享内部计算](../planning-experiment-2026-10-26.md)。
