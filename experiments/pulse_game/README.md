# Pulse game 研究代码

[晨间结果](../../docs/pulse-game-morning-2026-11-19.md) · [按问题查证据](../../docs/pulse-game-evidence-2026-11-19.md) · [只需 NumPy 的可运行包](../../data/pulse-game/morning-20261119/continuous-learning-cpu.zip)

## 最短复现

在仓库根目录使用项目的 Python。第一条命令以一个完整生命运行冻结检查点，第二条重跑旧符号模拟器的已封存评估，第三条运行游戏目录的测试。

```powershell
.\.venv\Scripts\python.exe -m experiments.pulse_game.lifelong_agent --game transfer --address local --worlds 24
.\.venv\Scripts\python.exe -m experiments.pulse_game.legacy_grounded_release --step replay
.\.venv\Scripts\python.exe -m pytest -q experiments/pulse_game
```

| 代码 | 职责 |
|---|---|
| [world.py](world.py)、[mixture_world.py](mixture_world.py)、[mixture_flash.py](mixture_flash.py) | 身体如何受到脉冲驱动，食物何时真正被碰到，公开观察中何时出现气味或回执 |
| [unified_world.py](unified_world.py)、[longrun_world.py](longrun_world.py) | 两地规律迁移、冲突、返回，以及连续长生命的隐藏规律翻转 |
| [habitat_deploy.py](habitat_deploy.py)、[adaptive_local.py](adaptive_local.py) | 可逐 tick 调用的 CPU 控制器和自组织局部地址。在线状态更新没有访问私有答案 |
| [unified_train.py](unified_train.py)、[unified_evolution.py](unified_evolution.py)、[independent_births.py](independent_births.py) | 真实接触历史的慢参数训练、按真实动作结果精调、预登记的 12 次独立出生 |
| [unified_gpu.cu](unified_gpu.cu)、[physical_counts.py](physical_counts.py) | 已与逐拍 CPU 环境核对的批量物理运行，以及有序归约 |
| [adaptive_local_audit.py](adaptive_local_audit.py)、[old_context_fork.py](old_context_fork.py)、[longrun_study.py](longrun_study.py) | 同检查点读写作用域、同历史回访和长期可塑性消融 |
| [lifelong_agent.py](lifelong_agent.py)、[live_continuity.py](live_continuity.py)、[lifetime_retention_fork.py](lifetime_retention_fork.py) | 可移植权重的推理与生内快照、逐段生成的长时间环境、完整状态与冷状态的未来行为分叉 |
| [morning_digest.py](morning_digest.py)、[morning_figure.py](morning_figure.py) | 按独立世界与训练出生聚合成对结果，生成主结果图 |

其他文件是具体问题的模拟器、搜索、独立复测或历史变体。按问题阅读证据索引，可以找到各项手写先验、监督来源、失败对照和冻结文件。
