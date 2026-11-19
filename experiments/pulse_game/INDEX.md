# Pulse game 代码导航

[发布版复现命令](README.md) · [研究证据](../../docs/pulse-game-evidence-2026-11-19.md) · [报告与汇总脚本](reporting/README.md) · [回归测试](tests/README.md)

## 公用入口

| 任务 | 代码 |
|---|---|
| 运行冻结控制器，保存或恢复同一生命的状态 | [lifelong_agent.py](lifelong_agent.py)、[live_continuity.py](live_continuity.py)、[lifetime_retention_fork.py](lifetime_retention_fork.py) |
| 复算旧符号控制器封存结果 | [legacy_grounded_release.py](legacy_grounded_release.py) |
| 生成 CPU 包，校验新进程续跑与晨间发布证据 | [export_reference.py](export_reference.py)、[portable_reboot.py](portable_reboot.py)、[morning_release_check.py](morning_release_check.py) |
| 查主图与个体行为轨迹的生成过程 | [morning_digest.py](morning_digest.py)、[morning_figure.py](morning_figure.py)、[morning_trace.py](morning_trace.py)、[morning_lifetime_review.py](morning_lifetime_review.py) |
| 查批量物理实现与吞吐对照 | [unified_physical.py](unified_physical.py)、[unified_gpu.cu](unified_gpu.cu)、[physical_counts.py](physical_counts.py)、[physical_throughput_bench.py](physical_throughput_bench.py) |

## 按研究问题找代码

| 研究问题 | 环境或机制 | 实验与测量 |
|---|---|---|
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
