# 研究证据与本机数据索引

本页给出代码、协议、逐种子指标和原始存档的位置。研究结论见[实验总览](../experiments.md)，逐轮报告见[分主题索引](../reports/README.md)，恢复与发布包见[本机数据索引](../../data/README.md)。各轮报告注明预试验、机械检查、阶段读数和正式结果各自适用的范围。

## 阅读结果与取得完整材料

[局部迹与反馈实验](../local-trace-feedback-2026-11-19.md)提供终局资源训练、同预算旧图、等时分叉、数值写入落点干预及分量消融。

[一次反馈、编辑选择与标记含义](../revisit-choice-feedback-2026-11-19.md)检验模型是否能从一生中的资源变化重新解释标记，并比较实际自编辑与外部即时梯度的方向。

[自编辑回执与后果归因](../edit-receipt-learning-2026-11-19.md)比较自身感觉、正确配对和打乱后果在四个训练时点的表现，并通过实际动作和写入端移植检验旧任务保持。

[相关经历与脉冲顺序试验](../learning-transfer-2026-11-19.md)提供两套证据。第一套包含[新旧规律的逐世界曲线与快矩阵回滚](learning-transfer-2026-11-19/relation.md)，第二套包含[首轮和独立复测的冻结拓扑、逐请求计分与检查点](learning-transfer-2026-11-19/order.md)。
本页另附 11 月 3 日四项研究的[逐种子计数摘录](selected-counts-2026-11-03.json)。它覆盖自主轨迹复拟合的长任务、提示后第一步的状态读出、等待期监督的长测试后缀，以及同世界移入训练状态的干预。每组都保存原文件相对路径、SHA256、所取行及原字段名。在这里可以分开核对成功数与成熟请求数、空闲误动作与实际空闲机会、探针答对数与测试行数。复算长测试总请求时，还要加上[首个测试窗口](wait-content-2026-11-03/free-behavior.json)。这份摘录用于读数，恢复训练另需本机检查点和完整协议。

先读研究报告的指标定义和限制，再看对应的 JSON/CSV、图和协议。汇总中的文件路径与 SHA 可以追溯来源，但不包含来源文件本身；检查点、逐 tick 轨迹、优化器、世界和 RNG 状态通常仍在本机 `data/`。后续研究依赖完整前序存档，单独复制一张图或一份汇总不能恢复训练，也不能完成全部独立审计。

[.gitignore](../../.gitignore) 忽略所有 `data/` 目录，以及 `*.npy`、`*.npz`、`*.log` 和 `__pycache__/` 等文件。因此，文档能链接到本机文件不等于这些文件会进入 Git 或随代码迁移。`docs/evidence/` 也可能包含被忽略的日志/缓存。下面列的是当前工作树中可以单独携带的材料入口；实际分享、提交或备份时仍需核对文件清单。便携汇总的完整性和原始存档的完整性分别判断。

## 研究线、数据根与可携带材料

所有 `data/...` 路径均相对仓库根目录，受上述忽略规则影响。表中 JSON/CSV/图只是已保存的证据入口，不承诺每个目录都含完整检查点或完整审计链。

| 研究线与报告 | 代码说明 | 本机原始数据根 | 本目录下的证据入口与范围 |
|---|---|---|---|
| [持续记忆与循环训练](../persistent-experiment-2026-10-26.md) | [persistent](../../experiments/persistent/README.md) | `data/persistent/` | [目录](persistent-2026-10-26/)；[results.json](persistent-2026-10-26/results.json)、[表格](persistent-2026-10-26/tables.md) 与图，含检查点指纹；完整模型/开发日志在本机 |
| [持续输入与结构变化](../streaming-experiment-2026-10-26.md) | [streaming](../../experiments/streaming/README.md) | `data/streaming/` | [目录](streaming-2026-10-26/)；[流结果](streaming-2026-10-26/stream-results.json)、[结构结果](streaming-2026-10-26/structure-results.json)、预算和来源核验及图 |
| [交错经历与巩固](../reports/memory/consolidation-experiment-2026-10-26.md) | [consolidation](../../experiments/consolidation/README.md) | `data/consolidation/` | [目录](consolidation-2026-10-26/)；[summary.json](consolidation-2026-10-26/summary.json)、[表格](consolidation-2026-10-26/tables.md) 与图；每次运行的完整状态在本机 |
| [消耗式规划](../planning-experiment-2026-10-26.md) | [planning](../../experiments/planning/README.md) | `data/planning/` | [目录](planning-2026-10-26/)；[协议](planning-2026-10-26/main-study-v1.json)、[summary.json](planning-2026-10-26/summary.json)、生成器/预测核验、压缩测试题与预测计划及图 |
| [持续动力学与计时](../dynamics-experiment-2026-10-26.md) | [dynamics](../../experiments/dynamics/README.md) | `data/dynamics/` | [目录](dynamics-2026-10-26/)；[正式协议](dynamics-2026-10-26/main-study-v2.json)、[summary.json](dynamics-2026-10-26/summary.json)、单列的 pilots 与图 |
| [完整状态连续运行](../continuity-experiment-2026-10-27.md) | [continuity](../../experiments/continuity/README.md) | `data/continuity/` | [目录](continuity-2026-10-27/)；[协议](continuity-2026-10-27/main-study-v1.json)、[汇总及逐种子 CSV](continuity-2026-10-27/aggregate/)、执行/分叉/恢复核验 |
| [旧经历与时间信用整合](../integration-experiment-2026-10-27.md) | [integration](../../experiments/integration/README.md) | `data/integration/` | [详细证据索引](integration-2026-10-27/README.md) 区分正式汇总、执行、预实验和事后端点观察；完整训练存档不在该目录 |
| [独立读出与前向时间信用](../credit-experiment-2026-10-27.md) | [credit](../../experiments/credit/README.md) | `data/credit/` | [目录](credit-2026-10-27/)；[主/复访汇总](credit-2026-10-27/aggregate/)、[紧凑读出证据](credit-2026-10-27/probe-aggregate-v1/overview.md)、协议/修订/完成清单及图；全量读出汇总仍在本机 |
| [既有回路与非线性扩展](../loops-experiment-2026-10-28.md) | [loops](../../experiments/loops/README.md) | `data/loops/` | [目录](loops-2026-10-28/)；[既有回路审计](loops-2026-10-28/audit-aggregate-v1/report.md)、[扩展汇总](loops-2026-10-28/extension-aggregate-v1/report.json)、协议/完成清单及图 |
| [渐退与再生长](../rewiring-experiment-2026-10-28.md) | [rewiring](../../experiments/rewiring/README.md) | `data/rewiring/` | [目录](rewiring-2026-10-28/)；[报告及 CSV](rewiring-2026-10-28/aggregate-v1/)、协议/执行修订/完成清单及图 |
| [结构因素与空闲惩罚](../factors-experiment-2026-10-29.md) | [factors](../../experiments/factors/README.md) | `data/factors/` | [目录](factors-2026-10-29/)；[报告及 CSV](factors-2026-10-29/aggregate-v1/)、协议/完成清单及图；逐请求分母、年龄和历史暴露分列 |
| [随机出生与持续课程](../lifelong-experiment-2026-10-29.md) | [lifelong](../../experiments/lifelong/README.md) | `data/lifelong/` | [目录](lifelong-2026-10-29/)；[报告及 CSV](lifelong-2026-10-29/aggregate-v1/)、主协议、补充覆盖协议、完成清单及图 |
| [原生习得与功能性组织](../organization-experiment-2026-10-30.md) | [organization](../../experiments/organization/README.md) | `data/organization/` | [目录](organization-2026-10-30/)；[主研究](organization-2026-10-30/aggregate-v1/)、[习得对照](organization-2026-10-30/acquisition-aggregate-v1/)、各自协议、完成清单及图；运行笔记和机械试验单列 |
| [共享无向图](../undirected-results-2026-10-30.md) | [undirected](../../experiments/undirected/README.md) | `data/undirected/` | [目录](undirected-2026-10-30/) 保留协议、命令、[运行记录](undirected-2026-10-30/run-status.md)、中期结果/图及独立核验记录；完整最终报告、分析和完成清单在本机 `data/undirected/`，中期文件不替代最终结果 |
| [启动经历与功能性删边](../reports/structure/coverage-exit-experiment-2026-11-02.md) | [coverage_exit](../../experiments/coverage_exit/README.md) | 正式 `data/coverage-exit/`；机械检查另在 `data/coverage_exit/` | [目录](coverage-exit-2026-11-02/) 有执行成本[输入](coverage-exit-2026-11-02/execution-inputs.json)、图及 [RTRL 后端复测材料](coverage-exit-2026-11-02/fast-rtrl/README.md)；行为协议、训练/删边结果和完整恢复材料主要仍在本机 |
| [自主行动与旧任务返回](../reports/actions/action-return-experiment-2026-11-03.md) | [action_return](../../experiments/action_return/README.md) | `data/action-return/` | [目录](action-return-2026-11-03/) 当前保存 [配对复拟合图](action-return-2026-11-03/paired-refit.png)；协议、逐种子计数、审计和撤出/删边结果仍在本机 |
| [内容与计时](../reports/actions/content-timing-experiment-2026-11-03.md) | [content_timing](../../experiments/content_timing/README.md) | `data/content-timing/` | [目录](content-timing-2026-11-03/) 当前保存 [绘图数值](content-timing-2026-11-03/figure-values.json) 和图；完整自然轨迹、配对分叉、读出与迁移核验仍在本机 |
| [等待中学习内容](../reports/actions/wait-content-experiment-2026-11-03.md) | [wait_content](../../experiments/wait_content/README.md) | `data/wait-content/` | [目录](wait-content-2026-11-03/) 当前保存 [独立重建的行为数值](wait-content-2026-11-03/free-behavior.json) 和图；完整训练、长任务追踪与审计仍在本机 |
| [实际习得与旧规则保持](../reports/memory/learn-remember-experiment-2026-11-03.md) | [learn_remember](../../experiments/learn_remember/README.md) | `data/learn-remember/` | [目录](learn-remember-2026-11-03/) 当前保存 [种子级汇总与来源 SHA](learn-remember-2026-11-03/results.json) 和图；训练、复习、自主评估、记忆/部署/状态移植和独立审计仍在本机 |
| [旧经历迁移与状态持续](../reports/memory/experience-transfer-experiment-2026-11-03.md) | [experience_transfer](../../experiments/experience_transfer/README.md) | `data/experience-transfer/` | [迁移、保持与来源 SHA](experience-transfer-2026-11-03/paired-curves.json) · [状态配对分子与分母](experience-transfer-2026-11-03/state-release.json) · [图表目录](experience-transfer-2026-11-03/) |
| [等预算旧经验复习选择](../reports/memory/review-selection-experiment-2026-11-03.md) | [review_selection](../../experiments/review_selection/README.md) | `data/review-selection/` | [逐种子学习曲线、切换计数及来源 SHA](review-selection-2026-11-03/counts-and-curves.json) · [三种复习方式的自由行为图](review-selection-2026-11-03/review-selection.png) |
| [等待监督的位置](../reports/actions/wait-position-experiment-2026-11-03.md) | [wait_position](../../experiments/wait_position/README.md) | `data/wait-position/` | [逐种子时机错误、切换对和来源校验值](wait-position-2026-11-03/counts.json) · [行为对比图](wait-position-2026-11-03/wait-position.png) |
| [结束正例的位置](../finish-position-experiment-2026-11-03.md) | [finish_position](../../experiments/finish_position/README.md) | `data/finish-position/` | [分母、时机错误、学习速度与来源校验值](finish-position-2026-11-03/counts.json) · [自由行为对照](finish-position-2026-11-03/finish-position.png) |
| [相邻时间边界与内容读出](../reports/actions/timing-boundary-experiment-2026-11-04.md) | [timing_boundary](../../experiments/timing_boundary/README.md) | `data/timing-boundary/` | [逐种子读出、真实行为、分母与来源校验值](timing-boundary-2026-11-04/results.json) · [冻结状态诊断图](timing-boundary-2026-11-04/timing-boundary.png) |
| [时机与内容分别学习](../reports/actions/action-split-experiment-2026-11-05.md) | [action_split](../../experiments/action_split/README.md) | `data/action-split/` | [原配方数值](action-split-2026-11-05/results.json) · [新请求归一化补测](action-split-2026-11-05/balanced-results.json) · [主图](action-split-2026-11-05/action-split.png) · [补测图](action-split-2026-11-05/balanced-action-split.png) |
| [保留原策略后预测时间](../remaining-time-experiment-2026-11-06.md) | [remaining_time](../../experiments/remaining_time/README.md) | `data/remaining-time/` | [逐种子行为、时间误差与来源校验值](remaining-time-2026-11-06/results.json) · [共同历史和自主行为图](remaining-time-2026-11-06/remaining-time.png) |
| [实际教师结束时机](../teacher-timing-experiment-2026-11-06.md) | [teacher_timing](../../experiments/teacher_timing/README.md) | `data/teacher-timing/` | [逐种子行为、配对变化、标签覆盖与来源](teacher-timing-2026-11-06/results.json) · [自主完成与时机错误图](teacher-timing-2026-11-06/teacher-timing.png) |
| [连续状态下覆盖与保持](../coverage-retention-experiment-2026-11-07.md) | [coverage_retention](../../experiments/coverage_retention/README.md) | `data/coverage-retention/` | [新世界/原世界计数、训练暴露、答案签名与来源](coverage-retention-2026-11-07/results.json) · [时长与规则保持图](coverage-retention-2026-11-07/coverage-retention.png) |
| [配对旧经验复习与后缀保持](../reports/memory/episodic-review-experiment-2026-11-07.md) | [episodic_review](../../experiments/episodic_review/README.md) | `data/episodic-review/` | [两个阶段的逐种子行为、错误签名、预算与来源](episodic-review-2026-11-07/results.json) · [复习及停止后的保持图](episodic-review-2026-11-07/episodic-review.png) |

关联状态的新一轮：[研究报告](../associative-state-experiment-2026-11-07.md) · [代码](../../experiments/associative_state/README.md) · [第一版逐种子指标](associative-state-2026-11-07/results.json) · [独立审计](associative-state-2026-11-07/audit.json) · [共享地址版指标](associative-state-2026-11-07/shared-v2/results.json) · [共享版审计](associative-state-2026-11-07/shared-v2/audit.json)。本机原始工件在 `data/associative-state/`，两版协议与结果分别保留。

学习绑定的后续：[研究报告](../reports/memory/learned-binding-experiment-2026-11-07.md) · [代码](../../experiments/learned_binding/README.md) · [第一版全部行为与训练来源](learned-binding-2026-11-07/results.json) · [动作/奖励/优化器审计](learned-binding-2026-11-07/audit.json) · [动作损失诊断](learned-binding-2026-11-07/action-loss.json) · [绑定信息只读诊断](learned-binding-2026-11-07/binding-probe.json)。完整原始记录在 `data/learned-binding/`。

11/10固定encoder的追加：[报告](../reports/memory/frozen-trace-experiment-2026-11-10.md) · [全部结果](frozen-trace-2026-11-10/results.json) · [审计](frozen-trace-2026-11-10/audit.json)。手写控制复用，不重复计算实验数量；该追加没有使内容翻译摆脱近随机水平。

`coverage_exit` 是代码目录名；`coverage-exit` 是正式数据/证据路径名。两个本机数据目录均实际存在且用途不同，不能统一改名或合并。同样，`action_return`、`content_timing`、`wait_content`、`learn_remember` 的代码名使用下划线，其数据与证据路径使用连字符。

## 初始化、局部可塑性与连续游戏

| 研究线 | 证据入口 |
|---|---|
| 11-11 FP32 初始化搜索 | [正式结果和适用边界](seed-evolution-2026-11-11-large/report.md) · [逐条件结果](seed-evolution-2026-11-11-large/results.json) · [本机全量审计](../../data/seed-evolution/2026-11-11-large-v1/audit.json) · [搜索协议与恢复](../reports/initialization/seed-evolution-experiment-2026-11-11.md) |
| 11-15 BF16 扩展 | 11-16 暂停，已记录 17,224／23,424 个候选。[进度](../../data/seed-evolution/2026-11-15-scale-bf16-v1/progress.json) · [暂停记录](../../data/seed-evolution/2026-11-15-scale-bf16-v1/pause-20261116.json) |
| 11-19 有向图局部可塑性 | [终代与回执分叉](../local-rule-experiment-2026-11-19.md) · [终代配对结果](../../data/undirected/2026-11-19-local-rule-v1/final_comparison.json) · [执行提速及生产回放](local-rule-2026-11-19/fast-training.md) · [精度核验](local-rule-2026-11-19/fast_local_precision_probe.md) |
| 11-19 脉冲游戏与迁移 | [逐研究问题的证据](../pulse-game-evidence-2026-11-19.md) · [跨游戏边界](../pulse-game-transfer-boundaries-2026-11-19.md) · [发布包与本机冻结数据](../../data/pulse-game/README.md) |

## 共享框架与历史材料

[共享 LIF 框架](../../src/README.md) 通过 [scripts/run.py](../../scripts/README.md) 把运行结果追加到本机 `data/results.jsonl`，文件任务写 `data/env/`。这些历史框架记录不属于上表某条独立研究的正式结果，配置指纹也不含种子、episode 预算或代码版本等全部条件。

[旧实现](../../experiments/legacy/README.md)、[原型](../../experiments/prototypes/README.md)、[局部诊断](../../experiments/diagnostics/README.md)、[基线](../../experiments/baselines/README.md) 与 [退役实现](../../experiments/retired/README.md) 的用途和兼容问题分别说明；旧版本材料见 [历史三卷索引](../archived/README.md)。`docs/evidence/` 中保存的分析脚本、源码快照、性能探针和中期说明用于保留当时过程，不能作为批量执行入口。

## 整理、验证与复现

只核对本页的链接和 UTF-8 编码，不读取检查点、不运行训练：

```powershell
python scripts/check_docs.py docs/evidence/README.md
```

完整复现从各线 README 和冻结协议开始。先确认所需原始存档、来源 SHA、源码快照及执行环境齐全，再在新工作副本和输出位置运行；包含绝对路径的协议不能仅改文字后视作原实验。旧协议、图和汇总保留其版本边界。很多报告、审计和绘图模块会向默认位置写产物，查资料时直接读取既有文件；本索引的维护不要求重跑这些模块。
