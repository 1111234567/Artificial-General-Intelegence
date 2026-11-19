# 启动经历、功能性删边与 FP64 执行优化

从仓库根目录使用项目的 `.venv/Scripts/python.exe`。两项继续训练取自上一轮的 131,072 tick 完整存档。每个臂推进 16,384 tick，共 128 次更新，覆盖三个训练种子。训练完成后从各自的端点状态自由行动。

| 入口 | 用途 |
|---|---|
| `python -m experiments.coverage_exit.coverage` | 端口图训练时提供部分成功 BEGIN 经历。协议保存在 [coverage-v1](../../data/coverage-exit/coverage-v1/protocol.json) |
| `python -m experiments.coverage_exit.coverage_check` | 对齐模型自选动作、实际动作、回执、标签与历史对照，导出逐种子统计 |
| `python -m experiments.coverage_exit.retirement_study` | 可学习接口模型上，比较只允许渐退与允许实际全删。协议保存在 [retirement-v1](../../data/coverage-exit/retirement-v1/protocol.json) |
| `python -m experiments.coverage_exit.retirement_report` | 汇总删边事件、近期及历史损失、端点自由评估 |
| `python -m experiments.coverage_exit.retirement_ablation` | 在端点副本上分别消融排名候选和最弱四条边，测短任务的实际行为差异 |
| `python -m experiments.coverage_exit.coverage_version_probe` | 在不同 actor 和时间段上验证版本的独立线性读出，保留原模型动作 |
| `python -m experiments.coverage_exit.retirement_magnitude_study` | 更换在线退出候选的排序，沿用相同的全删及回退检验，复用已完成的配对对照 |

原始存档和继续学习产生的文件分目录保存。研究结论、逐种子数据和性能图见[本轮报告](../../docs/reports/structure/coverage-exit-experiment-2026-11-02.md)。

## 执行后端

在优化器更新边界使用以下接口。CPU 复现时从 PowerShell 设置 `$env:CUDA_VISIBLE_DEVICES = '-1'` 再启动项目 Python，与原存档的全局 RNG 环境一致。窗口后端用于本轮的删边继续训练。完整 RTRL 后端可以对原始存档做独立对照。

```python
from experiments.undirected.run import NativeTrainer
from experiments.coverage_exit.fast_window import install_window
from experiments.coverage_exit.fast_credit import install_fast_credit, backend_manifest

trainer = NativeTrainer.from_checkpoint(path, device="cpu")
# 在 recorded512 实验中，历史对齐、旧参数快照和辅助复习仍由同一个 trainer 管理。
install_window(trainer)
trainer.train_update()

# 单独使用解析 RTRL 的例子。rtrl_path 必须来自 RTRL 或 RTRL-reset 组。
rtrl_trainer = NativeTrainer.from_checkpoint(rtrl_path, device="cpu")
install_fast_credit(rtrl_trainer)
execution_provenance = backend_manifest()
rtrl_trainer.train_update()
```

窗口执行器在每段固定参数内展开一次循环矩阵。每个历史参数块、探测时临时替换的一批权重、辅助复习的预热和求导阶段都各自构造矩阵。RTRL 执行器沿每个 Heun 子步传播完整导数，并用批量矩阵乘法汇总所有时刻对共享权重的导数。两者都使用 FP64，状态容量、信用跨度与步长沿用存档。优化会改变部分求和顺序，复核包括数值误差、Adam、实际动作以及世界轨迹。载入检查点后重新选择执行后端，并随训练结果保存后端来源。

## 复测速度

以下示例从仓库根目录运行。基准要求新输出文件，CPU 训练按单线程执行，CUDA 对照使用可见的设备。真实存档指定了图、时间跨度和初始敏感度。对比耗时段包含完整学习更新，存档恢复和文件落盘单独处理。

```powershell
$py = '.\.venv\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
$cp = 'data/undirected/main-v1/training/seed_000/ports__undirected__rtrl__fixed__warmup/checkpoints/tick_131072.pt'
& $py -m experiments.coverage_exit.benchmark_fast_credit --checkpoint $cp --interface ports --mode undirected --workload trainer --repeats 4 --output data/coverage-exit/perf/local-rtrl-repeat.json
```

在同一 PowerShell 环境中可复测两个窗口组存档。

```powershell
& $py -m experiments.coverage_exit.benchmark_window --repeats 6 --cpu 2 --output data/coverage-exit/perf/local-window-repeat.json
```

调度基准 `python -m experiments.coverage_exit.schedule_credit --help` 提供四个存档副本在同一四核预算下的吞吐比较，各方案分别有 1、2、4 个并发作业。单作业关注延迟，多作业关注全部更新用时。[单作业线程测量](../../data/coverage-exit/perf/block-rtrl-cpu-thread-sweep.json) · [多作业等资源测量](../../data/coverage-exit/perf/block-rtrl-four-core-scheduling.json)
