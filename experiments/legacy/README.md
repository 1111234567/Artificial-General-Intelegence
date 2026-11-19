# 旧 LIF 实验与对齐参照

这里保留旧实验实现，供历史复现、局部诊断和框架移植比对。现行实验说明见 [实验记录](../../docs/experiments.md)，已撤回与待核验的判断见 [复核记录](../../docs/review-2026-10-26.md)。

| 文件 | 用途 |
|---|---|
| [grow_agent.py](grow_agent.py)、[grow_goal.py](grow_goal.py)、[grow_world.py](grow_world.py) | 文件环境智能体、目标闭环、早期闭环 |
| [grow_teach.py](grow_teach.py)、[grow_talk.py](grow_talk.py)、[grow_cond.py](grow_cond.py) | 固定窗口、first-spike 与条件化任务版本 |
| [grow_func.py](grow_func.py)、[grow_homeo.py](grow_homeo.py)、[grow_hebb.py](grow_hebb.py) | 功能测量、结构稳态和 Hebbian 实验 |
| [net.py](net.py) | 部分旧脚本的共享参数与网络基元；与当前 `src/config.py` 分开维护 |
| [calib5.py](calib5.py)、[scan_base.py](scan_base.py)、[bench_lif2.py](bench_lif2.py) | 动力学标定、参数扫描和模拟器成本测量 |
| [CONSTANTS.md](CONSTANTS.md)、[成本基线](baseline-2026-10-23-simulator.md) | 当时的参数台账和机器测量，保留历史上下文 |

依赖 Python 与 NumPy。`bench_lif2.py` 的 Torch/GPU 分支还需要相应环境。`grow_agent.py` 运行会在仓库 `data/env/<seed>/` 中创建、重置并清理文件环境，不宜同时运行占用相同种子的旧脚本。导入这些模块不会启动其主训练。

轻量导入检查与构造对齐：

```powershell
python -c "from experiments.legacy import net, grow_agent, grow_goal, grow_hebb; print('legacy imports OK')"
python scripts/align_layers.py construct 1
```

前一条命令在仓库根目录执行；后一条也支持脚本绝对路径。构造对齐只证明该比较中的初始字段相同。旧常数是历史工作点，参数标签、迹时间尺度和成功率注释均不能单独证明一般机制或持续学习能力。
