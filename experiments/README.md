# 实验脚本索引

11 月 19 日的连续游戏代码从 [Pulse game 导航](pulse_game/INDEX.md) 进入。要复现原符号任务，使用[已封存的运行说明](pulse_game/README.md)。历轮研究的协议与结果见[分主题报告](../docs/reports/README.md)，原始存档位置见[数据目录](../data/README.md)。

[src](../src/README.md) 是早期 LIF/SNN 实验的共享框架，主入口为 [scripts/run.py](../scripts/run.py)，[test/run.py](../test/README.md) 是兼容入口。下列研究线有各自的模型、任务、训练和审计流程，并非 `scripts/run.py list` 中的同名配置；后续研究会复用先前模块与完整存档，具体来源以各线协议为准。共享命令见 [工具说明](../scripts/README.md)，旧结果边界见 [复核记录](../docs/review-2026-10-26.md)。

| 目录 | 用途 | 主要依赖 |
|---|---|---|
| [pulse_game](pulse_game/INDEX.md) | 游戏中的生内学习、冲突规则回访、主动探查和连续状态恢复。[11 月 19 日报告](../docs/pulse-game-morning-2026-11-19.md) | 冻结运行包用 NumPy，训练与 GPU 对照使用 PyTorch |
| [persistent](persistent/README.md) | 可寻址记忆、共享循环、梯度截断、自评与奖励适应的受控实验 | PyTorch；CUDA 可选，绘图另需 Matplotlib |
| [streaming](streaming/README.md) | 持续输入下的学习读写/停止、活动相关学习与连接渐进退出 | PyTorch；较大正式训练使用 CUDA，绘图需 Matplotlib |
| [consolidation](consolidation/README.md) | 重复经历、部分共享、条件冲突与规则修订下的交错学习和同计算量重放对照 | PyTorch；正式训练使用 CUDA，绘图需 Matplotlib |
| [planning](planning/README.md) | 消耗式资源规划、共享内部计算、独立多层及有实际候选执行的外部搜索对照 | PyTorch；正式训练使用 CUDA，报告需 Matplotlib |
| [dynamics](dynamics/README.md) | 持续双任务、实际时间间隔、可学习循环动力学与冻结循环/时间感知 GRU 对照 | PyTorch、NumPy；正式训练使用 CUDA，报告需 Matplotlib |
| [continuity](continuity/README.md) | 完整世界与隐藏状态连续运行、容量与重置对照、长等待适应及有限历史重放 | PyTorch、NumPy；正式训练使用 CUDA，报告需 Matplotlib |
| [integration](integration/README.md) | 完整连续状态中的内容、计时与结束状态诊断；旧经历梯度重放、较长梯度和前向共激活对照 | PyTorch、NumPy；正式训练使用 CUDA，报告需 Matplotlib |
| [credit](credit/README.md) | 独立内容读出、对角前向时间信用、重放与 A/B 连续复访；固定表征统计量验证 | PyTorch、NumPy；训练使用 CUDA、主测量使用 CPU，报告需 Matplotlib |
| [loops](loops/README.md) | 已有多节点回路的参数干预、两种等参数非线性场扩展与完整状态继续学习 | PyTorch、NumPy；训练使用 CUDA、统一 CPU 评估，绘图需 Matplotlib |
| [rewiring](rewiring/README.md) | 同一持续循环模型中的连接渐进退出、独立再生长、随机位置控制与 B→A 返回学习 | PyTorch、NumPy；CUDA 训练、CPU 评估与严格存档审计 |
| [factors](factors/README.md) | 同一稀疏 B 来源上的收缩/允许集合/更替 × 空闲罚项完整对照，分开严格与普通完成、空闲原因和年龄 | PyTorch、NumPy；CUDA 训练、CPU 评估，六进程存档审计 |
| [lifelong](lifelong/README.md) | 从随机出生运行四套条件规则，交叉任务课程、近期重复/历史复习、时间覆盖，测旧任务撤出和返回及原生连接策略 | PyTorch、NumPy；绘图另需 Matplotlib、ijson |
| [organization](organization/README.md) | 原生零收缩习得对照；同参数非线性回路；允许共存的生长/渐退及实际损失复核、完整事件寿命 | PyTorch、NumPy；CPU组织训练、CUDA旧基线配对；Matplotlib、ijson绘图 |
| [baselines](baselines/README.md) | 非脉冲任务基线 | Python、NumPy |
| [undirected](undirected/README.md) | 10 月的共享无向图与时间信用，以及 11 月的有向图演化和前向局部可塑性。[无向图结果](../docs/undirected-results-2026-10-30.md) · [局部规则实验](../docs/local-rule-experiment-2026-11-19.md) | PyTorch，部分后端使用 CUDA |
| [coverage_exit](coverage_exit/README.md) | 成功启动经历、短时状态读出、功能性删边、双精度更新优化及同预算调度实测。[报告](../docs/reports/structure/coverage-exit-experiment-2026-11-02.md) | PyTorch、NumPy、psutil、Matplotlib |
| [action_return](action_return/README.md) | 将状态读出接入真实行动，在模型自身经历上复拟合，比较弱边退出后的旧任务撤出、返回及瞬时删边。[报告](../docs/reports/actions/action-return-experiment-2026-11-03.md) | PyTorch、NumPy，FP64 CPU，绘图需 Matplotlib |
| [content_timing](content_timing/README.md) | 跟踪同一请求在提示、处理中点和合法结束时的内容得分，配对改变提示符号和时长，检验连续状态中可读的信息。[报告](../docs/reports/actions/content-timing-experiment-2026-11-03.md) | PyTorch、NumPy，FP64 CPU，绘图需 Matplotlib |
| [wait_content](wait_content/README.md) | 在同一请求内将一半辅助内容标签从结束移到等待，等预算训练现有无向图，比较真实完成和空闲动作。[报告](../docs/reports/actions/wait-content-experiment-2026-11-03.md) | PyTorch、NumPy，FP64 CPU，绘图需 Matplotlib |
| [learn_remember](learn_remember/README.md) | 原图自主经历校正、单层循环策略习得与撤出、旧经历复习及隐藏状态交换，评估实际无错完成。[报告](../docs/reports/memory/learn-remember-experiment-2026-11-03.md) | PyTorch、NumPy，GRU 用 CUDA 训练，独立 CPU 评估，绘图需 Matplotlib |
| [experience_transfer](experience_transfer/README.md) | 配对学习前史、换边后的新规则习得、旧冲突保持，以及相同输入下的状态演化和自由释放。[报告](../docs/reports/memory/experience-transfer-experiment-2026-11-03.md) | PyTorch、NumPy，CUDA 训练、CPU 自由评估，绘图需 Matplotlib |
| [review_selection](review_selection/README.md) | 用同等预算均匀复习、只选冲突旧请求、或按当前预测误差复习，独立测 B 学习曲线及同符号切换情境后的真实行为。[报告](../docs/reports/memory/review-selection-experiment-2026-11-03.md) | PyTorch、NumPy，CUDA 训练、CPU 自由评估，绘图需 Matplotlib |
| [wait_position](wait_position/README.md) | 固定旧 A 请求与三个监督标签，只移动 WAIT 位置，分开测抢答、合法窗口出手和真实切换。[报告](../docs/reports/actions/wait-position-experiment-2026-11-03.md) | PyTorch、NumPy，CUDA 训练、CPU 自由评估，绘图需 Matplotlib |
| [finish_position](finish_position/README.md) | 保留最后一个应等待时刻，比较三个合法位置上的旧 FINISH 正例，核对提前结束、漏过窗口和相邻冲突任务完成情况。[报告](../docs/finish-position-experiment-2026-11-03.md) | PyTorch、NumPy，CUDA 训练，CPU 独立测试，绘图需 Matplotlib |
| [timing_boundary](timing_boundary/README.md) | 冻结循环状态，分开检查相邻就绪边界与内容的可读性，比较相同教师历史和自身行动历史。[判断方案](timing_boundary/analysis-plan.md) | PyTorch、NumPy、scikit-learn；CPU 诊断，绘图需 Matplotlib |
| [action_split](action_split/README.md) | 同预算比较仅训练新动作读出与同时更新循环，分开监督模式和内容，再补测固定的三类模式归一化。[报告](../docs/reports/actions/action-split-experiment-2026-11-05.md) | PyTorch、NumPy，CUDA 训练，CPU 独立评估，绘图需 Matplotlib |
| [remaining_time](remaining_time/README.md) | 保留原策略，比较只读时间、时间更新循环、动作监督及联合训练，分开测回归误差与自主行为。[报告](../docs/remaining-time-experiment-2026-11-06.md) | PyTorch、NumPy，CUDA 训练，CPU 独立评估，绘图需 Matplotlib |
| [teacher_timing](teacher_timing/README.md) | 相同外部事件下比较实际 earliest/midpoint 教师，交叉动作/联合目标，并复现历史控制。[报告](../docs/teacher-timing-experiment-2026-11-06.md) | PyTorch、NumPy，CUDA 训练，CPU 多进程原生评估，绘图需 Matplotlib |
| [coverage_retention](coverage_retention/README.md) | 完整状态接续后分开测时长覆盖与 A 撤出，另做原世界无重置返回和事后答案签名检查。[报告](../docs/coverage-retention-experiment-2026-11-07.md) | PyTorch、NumPy，CUDA 训练，CPU 多进程原生评估，绘图需 Matplotlib |
| [episodic_review](episodic_review/README.md) | 固定配对旧 A/B 复习、当前权重完整前史重建，以及前瞻 B-only 无复习后缀。[报告](../docs/reports/memory/episodic-review-experiment-2026-11-07.md) | PyTorch、NumPy，CUDA 训练和辅助前向，CPU 原生评估，绘图需 Matplotlib |
| [associative_state](associative_state/README.md) | 公开成功回执驱动的同容量快状态；比较累加、delta、递归与共享地址学习，原执行器保持冻结。[报告](../docs/associative-state-experiment-2026-11-07.md) | PyTorch、NumPy；FP32 CPU 并行，绘图需 Matplotlib |
| [learned_binding](learned_binding/README.md) | 自动生成绑定状态、地址和写门，用完整动作CE及自身公开回报学习；与手写hard/soft及只改读写顺序的版本对照。[报告](../docs/reports/memory/learned-binding-experiment-2026-11-07.md) | PyTorch、NumPy；严格FP32 CUDA/CPU训练，CPU自身行动及审计 |
| [seed_evolution](seed_evolution/README.md) | 初始化局部进化、等预算随机搜索，以及多机调度和吞吐测量。11 月 11 日 FP32 正式轮已完成。[审计与最终结果](../docs/evidence/seed-evolution-2026-11-11-large/report.md) | PyTorch、NumPy，后端按运行协议 |
| [prototypes](prototypes/README.md) | RNN、连续流、量化与 SNN 映射原型 | NumPy；部分依赖当前 `src` |
| [diagnostics](diagnostics/README.md) | 针对旧实现的局部测量 | NumPy、`legacy`；环境探测可选 Torch/SciPy |
| [legacy](legacy/README.md) | 旧 LIF 实验、标定、成本记录与移植对齐参照 | NumPy；GPU 成本分支可选 Torch |
| [retired](retired/README.md) | 退役版本和依赖已消失接口的诊断 | 见各脚本；不作为默认运行入口 |

## 按用途找入口

下面是入口导航，不是顺序执行清单。研究线的命令通常从仓库根目录以 `.venv/Scripts/python.exe -m experiments.<目录>.<模块>` 调用；早期直接脚本调用与完整参数见各目录 README。`evaluate.py` 等也可能只有 Python 接口，不能只凭文件名假定有 CLI。

训练、继续训练、队列和 `prepare` / `declare` 会创建或改变实验产物。评估、审计、报告和绘图也可能耗时并写文件；“冻结评估”指参数不学习，不代表不推进模型或世界。各协议绑定源码、来源路径和 SHA，复跑应使用独立工作副本及新输出目录，并同步设置依赖；不要把下表中的正式目录当作可覆盖目标。

| 研究线 | 运行、训练或调度入口 | 看结果、评估和导出入口 | 验证入口 |
|---|---|---|---|
| persistent | [run.py](persistent/run.py)、奖励适应 [adapt.py](persistent/adapt.py) | 新世界复测 [evaluate.py](persistent/evaluate.py)；[summarize.py](persistent/summarize.py) | `test_tasks.py`、`test_models.py` |
| streaming | [run_stream.py](streaming/run_stream.py)；[calibrate_stop.py](streaming/calibrate_stop.py)、[reinforce_writer.py](streaming/reinforce_writer.py) 继续训练；[local_learning.py](streaming/local_learning.py)、[structure.py](streaming/structure.py)、[structure_free.py](streaming/structure_free.py) 为独立对照 | [evaluate_long.py](streaming/evaluate_long.py)、[diagnose.py](streaming/diagnose.py)；[summarize_stream.py](streaming/summarize_stream.py)、[summarize_structure.py](streaming/summarize_structure.py) | 目录内 `test_*.py`；`reinforce_writer --self-test` |
| consolidation | [run.py](consolidation/run.py) | [summarize.py](consolidation/summarize.py) | 目录内 `test_*.py` |
| planning | [run.py](planning/run.py) 的 `train` / `evaluate` | [report.py](planning/report.py)；事后无学习搜索 [search_control.py](planning/search_control.py) | 目录内 `test_*.py` |
| dynamics | [run.py](dynamics/run.py) | [report.py](dynamics/report.py) | 目录内 `test_*.py` |
| continuity | [run.py](continuity/run.py) 的主训练、适应和完整恢复 | [evaluate.py](continuity/evaluate.py)、[report.py](continuity/report.py) | 目录内 `test_*.py`；保存恢复证据见该线协议 |
| integration | [run.py](integration/run.py) | 轨迹诊断 [diagnose.py](integration/diagnose.py)、[report.py](integration/report.py)；不推进状态的端点观察 [inspect_state.py](integration/inspect_state.py) | 目录内 `test_*.py` |
| credit | [prepare_study.py](credit/prepare_study.py)、[prepare_probe.py](credit/prepare_probe.py) 生成协议；[launch.py](credit/launch.py)、[batch.py](credit/batch.py)、[windows_runner.py](credit/windows_runner.py) 调度；[cycle.py](credit/cycle.py)、[cycle_cpu.py](credit/cycle_cpu.py) 连续复访 | [evaluate_ready.py](credit/evaluate_ready.py)、[evaluate.py](credit/evaluate.py)、[probe.py](credit/probe.py)；[report.py](credit/report.py)、[probe_report.py](credit/probe_report.py)、[figures.py](credit/figures.py) | `test_*.py`；[statistics.py](credit/statistics.py)、[tangent_audit.py](credit/tangent_audit.py) 分别核验固定表征统计与敏感度 |
| loops | [prepare_audit.py](loops/prepare_audit.py)、[audit_run.py](loops/audit_run.py) 为既有回路诊断；[extension_study.py](loops/extension_study.py)、[train.py](loops/train.py) 为扩展继续学习；复用 credit 队列 | [report.py](loops/report.py)、[extension_report.py](loops/extension_report.py)、[plot_results.py](loops/plot_results.py) | 目录内 `test_*.py` |
| rewiring | [study.py](rewiring/study.py)、[run.py](rewiring/run.py)；复用 credit 队列 | [evaluate_ready.py](rewiring/evaluate_ready.py)、[report.py](rewiring/report.py)、[plot_results.py](rewiring/plot_results.py) | 目录内 `test_*.py` |
| factors | [study.py](factors/study.py)、[run.py](factors/run.py)、[queue.py](factors/queue.py) | [report.py](factors/report.py)、[plot_results.py](factors/plot_results.py) | 目录内 `test_*.py` |
| lifelong | [study.py](lifelong/study.py)、[run.py](lifelong/run.py)、[queue.py](lifelong/queue.py)；[supplement.py](lifelong/supplement.py) 为补充冻结评估 | [report.py](lifelong/report.py)、[plot.py](lifelong/plot.py)、[finalize.py](lifelong/finalize.py)；`evaluate.py` 仅提供 Python 接口 | 目录内 `test_*.py` |
| organization | [acquisition.py](organization/acquisition.py) 为习得对照；[study.py](organization/study.py)、[run.py](organization/run.py) 为主研究；复用 lifelong 队列 | [acquisition_report.py](organization/acquisition_report.py)、[report.py](organization/report.py)、[plot.py](organization/plot.py)、[finalize.py](organization/finalize.py) | 目录内 `test_*.py` |
| undirected | [study.py](undirected/study.py)、[run.py](undirected/run.py)、[queue.py](undirected/queue.py) | [analysis_batch.py](undirected/analysis_batch.py)、[sensitivity_audit.py](undirected/sensitivity_audit.py)、[report.py](undirected/report.py)；[finish.py](undirected/finish.py) 收尾；[results.py](undirected/results.py)、[plot.py](undirected/plot.py) | 目录内 `test_*.py`；完整历史校验与独立 JVP 校准分开记账 |
| coverage_exit | [coverage.py](coverage_exit/coverage.py)、[retirement_study.py](coverage_exit/retirement_study.py)、[retirement_magnitude_study.py](coverage_exit/retirement_magnitude_study.py) 为继续训练 | [coverage_check.py](coverage_exit/coverage_check.py)、[retirement_report.py](coverage_exit/retirement_report.py)；[retirement_ablation.py](coverage_exit/retirement_ablation.py)、[coverage_version_probe.py](coverage_exit/coverage_version_probe.py)、[action_margins.py](coverage_exit/action_margins.py) 为诊断 | `test_*.py`、`coverage_tests.py`、`retirement_magnitude_tests.py`、[retirement_magnitude_recovery.py](coverage_exit/retirement_magnitude_recovery.py)；`benchmark_*` / [schedule_credit.py](coverage_exit/schedule_credit.py) 含真实更新，属于成本实验 |
| action_return | [readout.py](action_return/readout.py)、[readout_swap.py](action_return/readout_swap.py)、[finish_readout.py](action_return/finish_readout.py)、[crossed_readout.py](action_return/crossed_readout.py)、[refit_policy.py](action_return/refit_policy.py)；[absence_study.py](action_return/absence_study.py)、[edge_identity.py](action_return/edge_identity.py) | 各模块的 `summary` / `summarize` / `report`；[plot_refit.py](action_return/plot_refit.py) | `test_*.py`、[absence_tests.py](action_return/absence_tests.py)、[absence_provenance.py](action_return/absence_provenance.py)、[independent_audit.py](action_return/independent_audit.py)、[independent_edge_audit.py](action_return/independent_edge_audit.py) |
| content_timing | [natural.py](content_timing/natural.py)、[matched_flow.py](content_timing/matched_flow.py) 收集冻结模型轨迹；[probe.py](content_timing/probe.py)、[natural_probe.py](content_timing/natural_probe.py) 拟合离线读出 | [onset.py](content_timing/onset.py)、[transfer.py](content_timing/transfer.py)、[transfer_replay.py](content_timing/transfer_replay.py) 诊断；[natural_tables.py](content_timing/natural_tables.py)、[figure.py](content_timing/figure.py) | `test_probe.py`、[audit.py](content_timing/audit.py) |
| wait_content | [data.py](wait_content/data.py) 采集；[train.py](wait_content/train.py) 配对训练 | [eval.py](wait_content/eval.py)、[common_history.py](wait_content/common_history.py)、[long_followup.py](wait_content/long_followup.py)；[figure.py](wait_content/figure.py) | `train_tests.py`、`eval_test.py`、`eval_serialization_test.py`；[data_audit.py](wait_content/data_audit.py)、[audit.py](wait_content/audit.py)、[eval_audit.py](wait_content/eval_audit.py)、[long_audit.py](wait_content/long_audit.py) |
| learn_remember | 原图校正、普通/双向共享 GRU 训练与历史复习，见下表 | 自主评估、记忆交换、部署和种子级结果，见下表 | 世界重放、训练、复习、共享参数及状态移植审计，见下表 |
| experience_transfer | [前史与 B 训练](experience_transfer/train_transfer.py) · [B′ 换边](experience_transfer/reciprocal_train.py) | [自由行为](experience_transfer/eval_transfer.py) · [状态播放与释放](experience_transfer/state_clamp.py) · [配对汇总](experience_transfer/results.py) | [训练回放](experience_transfer/train_audit.py) · [行为测试](experience_transfer/eval_transfer_test.py) · [状态核验](experience_transfer/state_audit.py) |
| review_selection | [旧请求库](review_selection/bank.py) · [等预算训练](review_selection/train_review.py) | [交错评估](review_selection/eval_review.py) · [同槽请求对](review_selection/pairs.py) · [首次错误诊断](review_selection/errors.py) | [训练复算](review_selection/train_replay_check.py) · [成对回执核验](review_selection/pair_audit.py) |
| wait_position | [三种位置的训练](wait_position/train_review.py) · [静态末步](wait_position/last_train.py) | [交错评估](wait_position/eval_position.py) · [切换请求对](wait_position/switch_pairs.py) · [同一历史打分](wait_position/fixed_history.py) | [可用时刻核验](wait_position/bank_audit.py) · [时机诊断](wait_position/eval_timing.py) |
| finish_position | [两端训练](finish_position/train.py) · [中间位置补测](finish_position/interior_train.py) | [自由行为](finish_position/eval_finish.py) · [同槽切换](finish_position/switch_test.py) · [共同示范读出](finish_position/boundary_readout.py) | [位置核验](finish_position/bank_audit.py) · [更新复算](finish_position/train_audit.py) |
| timing_boundary | [持续流采集](timing_boundary/data.py) · [冻结状态读出](timing_boundary/probe.py) | [数值汇总与图](timing_boundary/results.py) · [研究报告](../docs/reports/actions/timing-boundary-experiment-2026-11-04.md) | [独立复算](timing_boundary/audit.py) · 目录内三个测试模块 |
| action_split | [固定连续流](action_split/data.py) · [冻结/全训练](action_split/train.py) · [归一化补测](action_split/balanced_train.py) | [自主行动](action_split/evaluate.py) · [只读诊断](action_split/diagnose.py) · [新请求补测](action_split/balanced_eval.py) | [真实更新](action_split/train_audit.py) · [请求状态机](action_split/eval_audit.py) · [世界与策略重放](action_split/eval_replay.py) |
| remaining_time | [时间标签](remaining_time/targets.py) · [四组梯度](remaining_time/model.py) · [同预算训练](remaining_time/train.py) | [原生自主行动](remaining_time/evaluate.py) · [共同教师历史](remaining_time/common_history.py) · [结果与图](remaining_time/results.py) | [更新复算](remaining_time/train_audit.py) · [分母与时间统计](remaining_time/eval_audit.py) · [完整原生重放](remaining_time/eval_replay.py) |
| teacher_timing | [实际教师流](teacher_timing/data.py) · [同预算训练](teacher_timing/train.py) | [自主评估](teacher_timing/evaluate.py) · [交叉共同历史](teacher_timing/common_history.py) · [事后固定阶段](teacher_timing/phase_readout.py) | [更新复算](teacher_timing/train_audit.py) · [原生重放](teacher_timing/eval_replay.py) · [共同历史复核](teacher_timing/common_audit.py) · [阶段计数复核](teacher_timing/phase_audit.py) |
| coverage_retention | [完整世界接续](coverage_retention/data.py) · [原 Adam/h 继续学习](coverage_retention/train.py) | [新世界](coverage_retention/evaluate.py) · [原世界返回](coverage_retention/live_eval.py) · [答案签名](coverage_retention/error_signature.py) | [更新复算](coverage_retention/train_audit.py) · [行为与原生重放](coverage_retention/eval_audit.py) · [签名独立核验](coverage_retention/error_signature_audit.py) |
| episodic_review | [配对旧库](episodic_review/bank.py) · [主训练](episodic_review/train.py) · [固定 B 后缀](episodic_review/retention_train.py) | [两个阶段的原生评估](episodic_review/evaluate.py) · [结果与图](episodic_review/results.py) | [主更新复算](episodic_review/train_audit.py) · [后缀复算](episodic_review/retention_audit.py) · [原生与规则指标核验](episodic_review/eval_audit.py) |

### 学会与保持线的完整入口

这一线的后续程序多于早期训练入口。以下覆盖 `learn_remember/` 的 25 个 Python 文件；它们都属于同一研究的不同步骤，不应把审计分支或状态诊断再计为独立训练重复。

| 步骤 | 程序 | 用途与边界 |
|---|---|---|
| 旧图继续校正 | [correction.py](learn_remember/correction.py)、[correction_audit.py](learn_remember/correction_audit.py) | 自主经历/原策略经历的等预算训练与独立核验 |
| 普通 GRU 习得与复习 | [gru_train.py](learn_remember/gru_train.py)、[gru_review.py](learn_remember/gru_review.py) | 从零习得、旧家族撤出及固定历史复习；包含 CUDA 训练 |
| 普通 GRU 训练核验 | [gru_audit.py](learn_remember/gru_audit.py)、[gru_report.py](learn_remember/gru_report.py) | 重放保存的执行、核对公开观察/教师标签，汇总训练证据；训练拟合成绩不替代自主测试 |
| 双向共享 GRU | [symmetric_gru.py](learn_remember/symmetric_gru.py)、[symmetric_review.py](learn_remember/symmetric_review.py) | 共享无序节点对参数的训练与复习；`symmetric_gru test` 为结构检查，`train` 为实际训练 |
| 共享参数及复习审计 | [symmetric_training_audit.py](learn_remember/symmetric_training_audit.py)、[symmetric_review_audit.py](learn_remember/symmetric_review_audit.py)、[reciprocity_audit.py](learn_remember/reciprocity_audit.py) | 核对执行、密集/压缩存档、Adam 状态、对称权重和真实完成回执 |
| 自主行动测试 | [evaluate_gru.py](learn_remember/evaluate_gru.py)、[evaluate_symmetric.py](learn_remember/evaluate_symmetric.py)、[evaluate_gru_test.py](learn_remember/evaluate_gru_test.py) | 与训练分离的同请求冻结评估及评估器单元测试 |
| 独立计数与世界重放 | [evaluate_audit.py](learn_remember/evaluate_audit.py)、[evaluate_world_replay.py](learn_remember/evaluate_world_replay.py)、[evaluate_tables.py](learn_remember/evaluate_tables.py) | 向量化重建分母、用已保存动作重放原世界、保留种子导出表格 |
| 一次性命令与状态交换 | [memory_assay.py](learn_remember/memory_assay.py)、[run_memory.py](learn_remember/run_memory.py)、[memory_audit.py](learn_remember/memory_audit.py)、[review_checks.py](learn_remember/review_checks.py) | 配对内容/家族改变、等待中途状态交换、随阶段/复习端点开展相同测试与核验 |
| 部署与状态移植 | [evaluate_deployment.py](learn_remember/evaluate_deployment.py)、[state_transfer.py](learn_remember/state_transfer.py)、[state_transfer_audit.py](learn_remember/state_transfer_audit.py) | 比较陌生世界预热、训练端点延续和单独换入 h；用原生评估器独立复算移植 |
| 最终汇总 | [results.py](learn_remember/results.py) | 核对模型间请求身份，按种子合计并生成 `results.json`、`retention.png`；会写入现有证据目录，整理资料时直接读取已有文件 |

## 先检查还是重跑

文档导航和配置查看不需要训练：

```powershell
python scripts/run.py list
python scripts/run.py show base
python scripts/run.py check base
python scripts/check_docs.py experiments/README.md scripts/README.md docs/evidence/README.md
```

这些命令只检查声明、已实现的前置条件、文档链接与编码，不能复现实验结论。`check` 的退出码边界见 [工具说明](../scripts/README.md)。独立研究线的单元测试按各目录 README 选择；部分测试会运行小型训练或恢复过程，成本和验证范围各不相同，不把“测试通过”当作正式实验结果。

历史原型的 `grow_rnn --gradcheck` 只检查局部梯度；`rnn2snn --check` 会先训练 RNN。基线默认命令、诊断导入和历史脚本也可能训练、扫描或写环境文件。已知不能直接运行的接口见 [diagnostics](diagnostics/README.md) 与 [retired](retired/README.md)，不要统一批量 import 或执行全部脚本。研究结果应同时保留脚本版本、完整配置、种子、训练预算与测量定义。

## 旧路径对照（2026-10-26）

文件名保持不变；除 `test/run.py` 的兼容入口外，旧路径不保留包装脚本。

| 原位置 | 新位置 | 文件名 |
|---|---|---|
| `test/` | `scripts/` | `run.py`（原实现；`test/run.py` 仍可启动同一模块） |
| `test/` | `experiments/legacy/` | `grow_agent.py`、`grow_goal.py`、`grow_world.py`、`grow_talk.py`、`grow_homeo.py`、`grow_cond.py`、`grow_teach.py`、`grow_func.py`、`grow_hebb.py`、`net.py`、`calib5.py`、`bench_lif2.py`、`scan_base.py` |
| `test/` | `experiments/legacy/` | `CONSTANTS.md`、`baseline-2026-10-23-simulator.md` |
| `test/` | `experiments/prototypes/` | `grow_rnn.py`、`grow_stream.py`、`grow_snn_convert.py`、`grow_loop.py`、`grow_lif.py`、`rnn2snn.py`、`probe_g.py` |
| `test/` | `experiments/diagnostics/` | `probe_decay.py`、`probe_env.py`、`probe_homeo.py`、`probe_in.py`、`probe_mem.py`、`probe_out.py`、`probe_persist.py`、`probe_sim_noise.py` |
| `test/` | `experiments/baselines/` | `baseline_linear.py` |
| `test/archived/` | `experiments/retired/` | 全部原 `.py` 文件；原 README 改写为当前退役索引 |
| `scripts/` | `scripts/historical/` | `check_draft.py`、`check_claims.py`、`diff_volumes.py` |

上述历史脚本迁移只调整目录、导入、文件路径与说明，保留实验计算和参数。另行新增的 `persistent/` 有独立训练与评估记录。历史章节编号与脚本注释中的旧判断按当时记录理解，不自动成为现行设计规则。
