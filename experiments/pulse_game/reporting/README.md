# 报告与汇总脚本

这些脚本从已有实验结果生成表、统计量和图。在仓库根目录以模块方式运行，例如查看策略探查报告的参数。

```powershell
.\.venv\Scripts\python.exe -m experiments.pulse_game.reporting.active_symmetry_report --help
```

| 脚本 | 产物或职责 |
|---|---|
| [active_symmetry_report.py](active_symmetry_report.py) | 独立世界上的策略探查、等预算对照和实际收益 |
| [attractor_dendrite_summary.py](attractor_dendrite_summary.py) | 吸引子和树突对照汇总 |
| [credit_transfer_summary.py](credit_transfer_summary.py) | 延迟回执、信用归因和迁移的汇总 |
| [legacy_grounded_report.py](legacy_grounded_report.py) | 旧符号控制器的真实接触结果和配对效应 |
| [night_aggregate.py](night_aggregate.py) | 夜间多条研究线的分组汇总 |
| [ring_alt_report.py](ring_alt_report.py) | 环形觅食的门槛检验与对照图 |
| [ring_alt_chemical_report.py](ring_alt_chemical_report.py) | 化学感觉和受体机制对照 |
| [ring_alt_habitats_report.py](ring_alt_habitats_report.py) | 多场所觅食、复习和持续学习对照 |
| [spike_consolidate_report.py](spike_consolidate_report.py) | 脉冲巩固机制和配对门槛 |
| [symmetry_spike_report.py](symmetry_spike_report.py) | 脉冲控制器的对称迁移与随机旋转对照 |
| [topology_summary.py](topology_summary.py) | 等参数条件下的感觉拓扑对照 |

带数据目录参数的脚本可在证据副本上复算。默认读写路径见各脚本入口。晨间发布和已发布报告的入口列在 [代码导航](../INDEX.md)。
