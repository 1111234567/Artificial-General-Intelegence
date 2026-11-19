# 初始化进化搜索

正式规模为 **64个新候选×32代×3个配对重复**，与同预算随机搜索比较。首代共享后共12,096个独立候选，每方法每重复2,048次评估。原4×4只保留为流程与吞吐预试验，不作为有效性结论。

本轮三个配对重复与终训审计均已完成。512 步自主评估的宏平均严格率为进化组 25.31%、随机组 24.97%，三个配对差值方向不一致。[最终报告](../../docs/evidence/seed-evolution-2026-11-11-large/report.md) · [正式方案](large-plan.md) · [预试验方案](analysis-plan.md)

**11/11运行时已按用户要求修订为CPU/GPU分工。** 该轮使用的恢复入口是 `python -m experiments.seed_evolution.execution_runtime run`；旧CPU-only pipeline仅保留历史，不应用于恢复新后缀。[性能测量、精度边界与恢复说明](../../docs/reports/initialization/seed-evolution-performance-2026-11-11.md)

| 文件 | 用途 |
|---|---|
| [core.py](core.py) | 初始化/局部变异、完整64步学习与真实行动评分 |
| [runner.py](runner.py) | 已完成搜索的4×4预试验，未做终训/终评 |
| [large_runner.py](large_runner.py) | 大规模可恢复队列、候选及代际提交、最终训练/测试 |
| [large_audit.py](large_audit.py) | 谱系、预算、独立梯度与闭环结果核验 |
| [large_results.py](large_results.py) | 只从完整通过且SHA匹配的审计生成结果 |
| [pipeline.py](pipeline.py) | 搜索→终训→持出测试→审计→结果，失败即停 |
| [execution_runtime.py](execution_runtime.py) | CPU缓存/惰性加载、20进程前缀及18+2后缀调度 |
| [execution_audit.py](execution_audit.py) | 按实际后端独立重算并核对配对分配 |
| [execution_results.py](execution_results.py) | 带执行修订与后端计数的发布 |

## 执行、迁移与查看结果

| 用途 | 文件及记录 |
|---|---|
| 查 11 月 11 日正式轮的数值与后端分段 | [结果表](../../docs/evidence/seed-evolution-2026-11-11-large/report.md) · [数值与审计来源](../../docs/evidence/seed-evolution-2026-11-11-large/results.json) |
| 找局域网主副机的任务分配和工作进程 | [调度](lan_dispatch.py) · [监督器](lan_supervisor.py) · [工作进程](lan_worker.py) |
| 找 Debian CPU 续跑工具和包模板 | [CPU 运行器](cloud_cpu.py) · [打包工具](../../scripts/package_seed_evolution_cloud.py) · [包内说明](../../scripts/cloud_cpu_bundle/README.md) |
| 查批量 GPU 训练与持续流水线测速 | [批量训练](gpu_batch_train.py) · [流水线测速](gpu_pipeline_benchmark.py) |
| 查 11 月 15 日 BF16 扩展与运行界面 | [规模搜索](scale_search.py) · [暂停记录](../../data/seed-evolution/2026-11-15-scale-bf16-v1/pause-20261116.json) · [控制台](dashboard.py) |

### 11 月 11 日运行的恢复命令

这条命令用于当时的运行目录。

```powershell
.\.venv\Scripts\python.exe -u -m experiments.seed_evolution.execution_runtime run
```

同一执行修订命令可恢复未完成运行，已提交工件验证后复用。不要在正在运行时启动第二个调度器，OS锁会拒绝。候选文件不因失败被替换为新的随机个体。原`large_runner declare`及`pipeline`仅用于原CPU协议的历史复现；当前已声明目录不要重复声明。

查看进展：

```powershell
.\.venv\Scripts\python.exe -m experiments.seed_evolution.large_runner status
.\.venv\Scripts\python.exe -m experiments.seed_evolution.large_audit --metadata-only
```

根目录为 `data/seed-evolution/2026-11-11-large-v1/`。`progress.json`记录阶段与数量；`pipeline-process.json`记录真实Python进程；`pipeline-failure.json`/`failure.json`记录中断原因。`pipeline-completion.json`仅在完整审计和汇总完成后生成。运行中metadata-only检查不代表实验完成。

最终报告在 `docs/evidence/seed-evolution-2026-11-11-large/report.md`。它会分别列开发代际分布、与随机重启的差值、0/64/512步测试以及禁读/B间隔条件；累计最佳上升不能独自证明迭代有效。搜索只改变初始化，没有同时降低任务难度。
