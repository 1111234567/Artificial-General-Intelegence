# 初始化搜索的CPU/GPU性能调整

2026-11-11。用户要求继续优化性能，并指出原14个单线程进程时CPU总占用仅约70%。本次调整执行方式，不减少12,096候选、每候选64步学习、终训512步或实际行动评分预算。

## 测到的瓶颈

同一候选完整搬到GPU，64步约105.2秒，早先CPU记录约88.7秒；这两次后台负载不同，不能把GPU利用率当作吞吐量。单步分段检查显示：GPU擅长原192维GRU执行器，但逐tick的小记忆更新在GPU上消耗大量调用时间。

因此采用分工：encoder、读写头、记忆更新和Adam留在CPU，冻结的GRU执行器放到CUDA，保留输入与输出复制的梯度。

| 实测 | 结果与边界 |
|---|---|
| 三个真实源各8步，CPU对混合 | 混合训练快约1.71–2.05倍；当时原14进程还在运行 |
| 单候选混合64步 | 48.11秒，早先CPU记录88.72秒；负载非同期，不当成整体搜索加速比 |
| 64步CPU对混合数值 | 可训练参数最大绝对差约1.74e-6；该候选两开发世界的动作哈希、计数和fitness一致，不能推广成所有候选逐位等价 |
| 两个独立混合64步 | 模型、Adam、日志逐位一致；缓存开关及65步运行的第64步前缀也核对一致 |
| 冻结encoder缓存 | CPU64步模型、Adam、日志完全一致；单次配对80.69→77.39秒，训练用时约降4.1% |
| 按需加载训练流 | 64步仅需8流；一个worker只保留当前source。三source加载对照峰值864→571 MiB，减少约293 MiB |

基准原件：[GPU整体](../../../data/seed-evolution/runtime-2026-11-11/gpu-benchmark.json)、[CPU/GPU拆分](../../../data/seed-evolution/runtime-2026-11-11/hybrid-benchmark.json)、[encoder缓存](../../../data/seed-evolution/runtime-2026-11-11/encoder-cache.json)、[按需加载](../../../data/seed-evolution/runtime-2026-11-11/lazy-training-streams.json)。不同负载和不同工作进程的数字不混成一个严格的总加速比。

## 正式执行修订

旧进程树在保留1,170个完整候选后有计划退出，原工件和已发计划未修改。[中断记录](../../../data/seed-evolution/runtime-2026-11-11/transition-stop.json)

新运行已恢复。当前已发计划的第4代（内部编号3）仍为CPU；从第5代（编号4）起，每个重复、每种搜索方法的对应候选序号0–7使用混合后端，其余仍为CPU。选择不依赖fitness、完成速度或搜索方法。默认当前CPU前缀20个worker，后续18个CPU worker与2个混合worker分别排队。

CPU缓存优化保留原模型/Adam/日志的逐位复现要求。混合后端保持FP32、关闭TF32和AMP，但CPU/CUDA运算顺序不同；训练元数据真实记录设备，旧CPU证据不改写。最终胜者在其搜索后端从初始化重新训练，仍要求第64步精确复现。

独立审计按实际后端重建训练，CUDA只使用一个审计进程，保留谱系、预算、世界和持出账本检查。最终发布须核验执行修订的SHA，并披露旧CPU前缀与配对混合后缀。[执行修订](../../../data/seed-evolution/2026-11-11-large-v1/execution-amendment.json)

恢复后的20进程CPU前缀完成174个新候选，约9.97个/分钟；旧第4代片段约7.86个/分钟，观测值约提高27%。source、负载和启动边界不同，这不是严格单因素加速证明。系统总CPU占用实测100%，可用内存有波动，记录在[资源样本](../../../data/seed-evolution/runtime-2026-11-11/live-resource-samples.json)。

第5代首个真实混合候选`rep_0/evolution_g04_i00`已完成：训练58.98秒、CPU评分31.05秒，元数据与执行记录均显示CPU策略/Adam及CUDA执行器，FP32且无AMP/TF32。同代CPU候选训练约86秒，属于不同候选的运行参考，不当成严格配对加速比。18个CPU及2个混合worker现已实际投入正式搜索，监控时显存约4.65GiB/8GiB。

切换后的只读检查已经核对1,466个候选的来源、后端分配和记录，其中包括真实CPU/GPU后缀；这仅是运行中检查，不是全量实验通过。[阶段检查](../../../data/seed-evolution/runtime-2026-11-11/live-execution-check.json)

## 11 月 12 日本机吞吐

原 10 CPU + 2 混合进程在两段热机窗口内合计 6 分 45 秒完成 54 个候选，包括训练与评分，吞吐 8.0 个/分钟。测第三个混合进程时，64 步训练及评分耗时 68.4 秒，权重哈希和评分与同一候选的存档一致。试跑期间全卡显存峰值 5,327 MiB，最低可用物理内存 4,704 MiB。

切换到 10 CPU + 3 混合后，同代首个 6 分 26 秒热机窗口完成 47 个完整候选，吞吐 7.3 个/分钟。该窗口跨过混合队列结束。混合队列结束后，现场同时有 13 个纯 CPU 候选在运行，3 分 35 秒内完成 30 个，吞吐 8.4 个/分钟。这些窗口覆盖的候选和队列阶段不同。每代改分 108 个混合候选的整代耗时需等第 13 代完成后测量。[切换记录](../../../data/seed-evolution/2026-11-11-large-v1/execution-balanced-20261112-203756.launch.json)

## 恢复命令

2026-11-12，本机运行增加了[补充排程](../../../data/seed-evolution/2026-11-11-large-v1/balanced-schedule.json)。第 12 代沿用原设备分配。从第 13 代起，每代 384 个候选中，108 个使用混合后端，276 个使用 CPU。进化组和随机组按相同序号配对。搜索使用 10 个 CPU 进程和 3 个混合进程。混合队列发完后，空出的进程接手纯 CPU 候选，按其原定后端训练。

从 `F:\agi` 恢复本机这次运行：

```powershell
.\.venv\Scripts\python.exe -u -c "from experiments.seed_evolution.balanced_runtime import run; run(cpu_workers=10, hybrid_workers=3)"
```

同一目录不要启动第二个调度器，进程锁会拒绝。已完成的候选经校验复用，失败不自动改用另一后端。当前阶段以[progress.json](../../../data/seed-evolution/2026-11-11-large-v1/progress.json)和[执行进程记录](../../../data/seed-evolution/2026-11-11-large-v1/execution-process.json)为准。正式遗传有效性结论仍须等待整场搜索、独立测试与审计完成。

11 月 13 日 00:10，本机搜索在第 16 代遇到 cuDNN 反向传播错误，已提交 5,861 个候选。检查时显卡仍占用约 6,926 MiB 显存，`nvidia-smi` 报告约 1,025 MiB 可用。恢复时启用 8 个 CPU 进程，并让 1 个混合进程等待至少 1,600 MiB 可用显存后再启动。进程会轮询可用显存，候选继续按既定设备分配运行。[恢复记录](../../../data/seed-evolution/2026-11-11-large-v1/execution-recover-20261113-004431.launch.json)

```powershell
.\.venv\Scripts\python.exe -u -c "from experiments.seed_evolution.balanced_runtime import run; run(cpu_workers=8, hybrid_workers=1, hybrid_start_free_mib=1600)"
```

## 双机调度与恢复

11 月 14 日启用双机管理器。主机负责生成计划、核验并提交候选、选择父本和胜者。副机在本地副本中执行分配给它的纯 CPU 候选。每代最多分配八对，配对的进化和随机候选使用同一台机器。主机优先复用已提交的候选和本机的部分工件。副机任务中断时，主机导入已完成且通过校验的结果，再接手其余候选。

管理器持续记录[两机进度](../../../data/seed-evolution/2026-11-11-large-v1/lan-manager-status.json)，随搜索推进自动启动终训、持出测试和审计。发生已识别的可恢复故障时，管理器自动续跑。连续三次续跑没有新增候选，或审计发现证据不一致时，状态会变为 `needs_attention`。成功结束以完整审计及 `pipeline-completion.json` 为准。

副机 5070 Ti Laptop 的独立基准中，两个混合进程通过 SSH 同跑，后台同时有两个正式 CPU 进程。两个混合候选各用约 226 秒，模型哈希一致。采样中显存峰值为 2,666 MiB，最低可用系统内存为 2,718 MiB。此前桌面单进程跑同一候选用了 48.2 秒。两次的会话和并发负载不同，当前数据只能说明这组混跑没有带来提速。正式混合候选由主机的 4060 Ti 跑，副机混合结果保存为独立基准。[并发报告](../../../data/seed-evolution/transfer/DESKTOP-T3C3GMF-probe-result-offline-20261114-193753-17964.json)

主机重启后，从 `F:\agi` 执行以下命令。若副机重启，需在副机管理员 PowerShell 运行 `Start-Service sshd`，以恢复远端执行通道。

```powershell
.\.venv\Scripts\python.exe -u -m experiments.seed_evolution.lan_supervisor `
  --start-if-inactive `
  --remote-root '\\DESKTOP-T3C3GMF\C$\Users\Jerry\Desktop\agi-lan-probe-20261114' `
  --key 'C:/Users/Jerry/.ssh/agi_lan_20261114' `
  --known-hosts 'F:/agi/data/seed-evolution/transfer/desk-t3c3gmf-known-hosts' `
  --cpu-workers 12 --hybrid-workers 3 --remote-workers 2 --pairs 8 --hybrid-start-free-mib 4800
```
