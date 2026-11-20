# Pulse game 代码导航

[发布版复现命令](README.md) · [研究证据](../../docs/pulse-game-evidence-2026-11-19.md) · [报告与汇总脚本](reporting/README.md) · [回归测试](tests/README.md)

## 公用入口

| 任务 | 代码 |
|---|---|
| 复算期限提示、实际单路线、反馈冲突和写入 | [单路训练](noon_horizon_online.py)、[路径账本核验](noon_horizon_online_audit.py)、[回执核验](noon_agreement_eval.py)、[写入或跳过的真实后续](noon_commit_eval.py)、[研究报告](../../docs/active-review-2026-11-20.md) |
| 运行冻结控制器，保存或恢复同一生命的状态 | [lifelong_agent.py](lifelong_agent.py)、[live_continuity.py](live_continuity.py)、[lifetime_retention_fork.py](lifetime_retention_fork.py) |
| 复算旧符号控制器封存结果 | [legacy_grounded_release.py](legacy_grounded_release.py) |
| 生成 CPU 包，校验新进程续跑与晨间发布证据 | [export_reference.py](export_reference.py)、[portable_reboot.py](portable_reboot.py)、[morning_release_check.py](morning_release_check.py) |
| 查主图与个体行为轨迹的生成过程 | [morning_digest.py](morning_digest.py)、[morning_figure.py](morning_figure.py)、[morning_trace.py](morning_trace.py)、[morning_lifetime_review.py](morning_lifetime_review.py) |
| 查批量物理实现与吞吐对照 | [unified_physical.py](unified_physical.py)、[unified_gpu.cu](unified_gpu.cu)、[physical_counts.py](physical_counts.py)、[physical_throughput_bench.py](physical_throughput_bench.py) |

## 按研究问题找代码

跨对象学习从[冻结容量回放](interleave_capacity.py)、[真实回访目标对照](interleave_revisit_train.py)、[混合生命训练](interleave_train.py)与[全新对象第一次行动](interleave_induction.py)进入。各对照与使用口径见[研究记录](../../docs/interleaved-learning-2026-11-20.md)。

| 跨对象学习的复现入口 | 实现和已有证据 |
|---|---|
| 看一张具体图怎样吃、写入和行动 | [固定个体回放](interleave_triangle_example.py) · [两个真实世界实例](../../docs/evidence/interleaved-learning-2026-11-20/triangle-lived-example-64000.md) |
| 比较训练经历怎样改变学习能力 | [不同气味几何的同预算训练](interleave_geometry_curriculum.py) · [两种技能的训练顺序](interleave_cross_skill.py) · [一生七次接触和两次实际行动](interleave_one_life.py) |
| 复算同一个状态能答对几件事 | [普通世界与几何泛化](interleave_geometry_eval.py) · [六项记忆分叉](interleave_cross_skill_eval.py) · [连续双答与独立留出](interleave_one_life_eval.py) |
| 定位已有权重写入、低精度摘要与慢速巩固 | [同一存档的写入分量](interleave_triangle_components.py) · [习得稳定力度](interleave_adaptive_reference.py) · [慢副本的低精度实测](../../docs/evidence/interleaved-learning-2026-11-20/dynamic-quantized-reference.md) |
| 对冻结图加速和核对源码 | [逐味觉展开八种赋值](interleave_triangle_prefix.py) · [配对测速和位级验证](../../docs/evidence/interleaved-learning-2026-11-20/triangle-prefix-bench.md) · [研究证据索引](../../docs/evidence/interleaved-learning-2026-11-20/README.md) |

局部可塑图从实际终局收益学习。[训练](local_trace_train.py) · [冻结评估](local_trace_eval.py) · [等时读出](local_trace_timing.py) · [物理分量消融](local_trace_components.py)

| 研究问题 | 环境或机制 | 实验与测量 |
|---|---|---|
| 多经历能否积累更好的写入方法 | [全参数写入](self_param_writer.py) · [连续活动](self_dose_continuous.py) | [连续生命](self_dose_life.py) · [启蒙样本与时间](self_dose_boot.py) · [梯度精确枚举](self_dose_expected_credit.py) |
| 图能否在启蒙后生成对自身边、偏置与写入端的修改 | [原生写入和读回](self_write_core.py) · [梯度启蒙](self_write_core_bootstrap.py) | [长生命](self_write_core_lifetime.py) · [写入器移植](self_write_transfer.py) · [信息选择](self_write_inquiry.py) · [局部自我探测](self_write_view_probe.py) |
| 无端图能否在生命中形成少量有用的入边交互 | [稀疏系数与候选采样](sparse_pairs_model.py) · [多次回执印证](sparse_alt_model.py) | [出生起训练](sparse_pairs_study.py) · [切换化学关系](sparse_pair_switch.py) · [成熟功能定位](sparse_location_roles.py) |
| 一张内部没有端点的图能否从反馈学习新规律 | [统一感觉动作图](unified_live_graph.py) · [连续图和快速回执学习器](open_life_model.py) | [改变偏好方向并移植突触图](open_graph_orthogonal.py) · [同一气味对在两条生命里有相反答案](open_graph_remap.py) |
| 出生图能否自由分配入边、出边，或先在多个环内独立循环 | [自由强连通图](free_graph_model.py) · [分环和随机跨环边](module_graphs.py) | [自由图对照](free_graph_study.py) · [分环对照](module_graph_study.py) · [集中输出干预](free_hub_audit.py) |
| 模型能否依据实际行动结果修改接线 | [训练期选择增退边](plastic_topology_modules.py) · [生内接线变化](live_module_edits.py) | [稳定与反转世界的同出生对照](live_module_edits_study.py) |
| 生长、调权和退出能否在同一生命中协作 | [所有活跃参数的回执更新](joint_live_model.py) · [从零决定新边的位置与首个权重](coupled_birth.py) | [联合模型选参和确认](joint_live_integrated.py) · [新生边的数值干预](live_edge_probe.py) |
| 能否按近期证据决定旧经验的共享强度 | [共享系数](trust_gate.py) · [真实回执重放](trust_replay.py) | [训练和留出](trust_study.py) · [连续改变旧经历的相似程度](transfer_overlap.py) |
| 自环、单环、多环分别如何处理顺序和对称规则 | [图结构和训练](ring_logic_model.py) · [20 项输入表](ring_logic_world.py) | [等预算对照](ring_logic_study.py) · [固定端口的表达能力](ring_witness.py) |
| 相关经历能否加快陌生关系的生内学习 | [配对世界](relation_world.py) · [快矩阵学习器](relation_learner.py) | [训练与封存](relation_study.py) · [逐世界分析](relation_analysis.py) |
| 需要辨认脉冲顺序时，有向环能带来什么 | [顺序游戏](order_world.py) · [两种拓扑](order_model.py) | [原试验](order_study.py) · [独立复测](order_replication.py) · [动作重放核验](order_verify.py) |
| 脉冲如何改变运动并产生实际接触 | [world.py](world.py)、[mixture_world.py](mixture_world.py)、[mixture_flash.py](mixture_flash.py) | [paired_choice.py](paired_choice.py)、[mixture_benchmark.py](mixture_benchmark.py) |
| 固定网络能否在不同任务里保留能力 | [motor_lif.py](motor_lif.py)、[spike_policy.py](spike_policy.py)、[attractor_motif.py](attractor_motif.py) | [transfer.py](transfer.py)、[confirm_spiking.py](confirm_spiking.py)、[attractor_motif_find.py](attractor_motif_find.py) |
| 旧控制器如何接入真实回执和新动作 | [legacy_bridge.py](legacy_bridge.py)、[legacy_grounded.py](legacy_grounded.py) | [legacy_bridge_fork.py](legacy_bridge_fork.py)、[legacy_grounded_outcomes.py](legacy_grounded_outcomes.py)、[legacy_grounded_graph_control.py](legacy_grounded_graph_control.py) |
| 隐藏规律翻转后，状态如何更新和返回 | [unified_world.py](unified_world.py)、[longrun_world.py](longrun_world.py)、[habitat_deploy.py](habitat_deploy.py) | [independent_births.py](independent_births.py)、[longrun_study.py](longrun_study.py)、[repeated_game_switch.py](repeated_game_switch.py) |
| 局部地址如何分开、复用和保护经历 | [adaptive_local.py](adaptive_local.py)、[context_local.py](context_local.py)、[context_local_active.py](context_local_active.py) | [adaptive_local_audit.py](adaptive_local_audit.py)、[context_local_run.py](context_local_run.py)、[old_context_fork.py](old_context_fork.py) |
| 探查的时机和代价如何改变真实收益 | [active_symmetry_world.py](active_symmetry_world.py)、[active_symmetry_agent.py](active_symmetry_agent.py) | [active_symmetry_train.py](active_symmetry_train.py)、[active_symmetry_final.py](active_symmetry_final.py) |
| 无标签感觉、场所与接触历史如何影响行为 | [ring_world.py](ring_world.py)、[ring_alt_chemical.py](ring_alt_chemical.py)、[ring_alt_habitats.py](ring_alt_habitats.py) | [ring_alt_chemical_run.py](ring_alt_chemical_run.py)、[ring_alt_chemical_shift.py](ring_alt_chemical_shift.py)、[ring_alt_matched.py](ring_alt_matched.py) |
| 对称变换与脉冲巩固是否保留可迁移结构 | [symmetry_spike.py](symmetry_spike.py)、[spike_consolidate.py](spike_consolidate.py) | [symmetry_spike_matrix.py](symmetry_spike_matrix.py)、[spike_consolidate_run.py](spike_consolidate_run.py) |
| 实际回执延迟时，写入如何找到正确事件 | [credit_wait.py](credit_wait.py)、[delayed_receipts.py](delayed_receipts.py) | [credit_wait_study.py](credit_wait_study.py)、[delayed_receipt_study.py](delayed_receipt_study.py)、[receipt_fork.py](receipt_fork.py) |
| 经历能否跨对象复用，以及何时需要复习 | [object_world.py](object_world.py)、[content_routing.py](content_routing.py)、[recall_world.py](recall_world.py) | [object_transfer_study.py](object_transfer_study.py)、[replay_study.py](replay_study.py)、[recurring_fork.py](recurring_fork.py) |
| 主动行动后的感觉与价值如何训练 | [scout_world.py](scout_world.py)、[scout_neural.py](scout_neural.py)、[meta_scout_model.py](meta_scout_model.py) | [scout_neural_train.py](scout_neural_train.py)、[scout_neural_reward.py](scout_neural_reward.py)、[meta_scout_train.py](meta_scout_train.py) |
| 两步路径上如何绑定转移与终点价值 | [twostep_world.py](twostep_world.py)、[twostep_latent.py](twostep_latent.py)、[twostep_rewire.py](twostep_rewire.py) | [twostep_reward.py](twostep_reward.py)、[twostep_rewire_matched.py](twostep_rewire_matched.py) |
| 背景与食物共用感觉端口时，怎样认出旧背景 | [untyped_senses.py](untyped_senses.py)、[untyped_revisit.py](untyped_revisit.py) | [混合感觉口实验](untyped_revisit_study.py) · [角色识别与回访结果](../../data/pulse-game/research-2026-11-19/untyped_revisit/report.md) |

按 [研究证据索引](../../docs/pulse-game-evidence-2026-11-19.md) 查各路线的协议、样本与结果。发布资料中列出的源码和测试路径可用于对应的封存复现。
