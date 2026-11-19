# Debian 云电脑续跑

这个归档带有本机已完成的 4,032 个候选、已提交的 11 代、下一代的计划、训练数据、冻结执行器，以及 Linux x86_64 的 Python 3.13 CPU 依赖。解包后使用系统 Python 3.13 即可离线安装依赖，无需 sudo 或 GPU。

```bash
tar -xf seed-evolution-debian-cpu.tar.xz
cd seed-evolution-debian-cpu
bash run.sh check
nohup bash run.sh run > cloud.log 2>&1 &
```

`check` 核对来源并做两步 CPU 训练。`run` 从第 12 代继续，默认开 6 个工作进程，适配约 9.7 GiB 内存且无 Swap 的机器。如需调整可给 `run` 传 `--workers N`。中断后重新执行同一条 `run` 命令会复用已提交候选。进程运行期间只保留一个调度器。

```bash
bash run.sh status
tail -n 30 cloud.log
```

搜索完成后自动接上终训、测试、审计与报告。终训从胜者保存的第 64 步模型和 Adam 继续到 512 步。历史候选保留原分数和原后端记录，迁移审计核对历史档案，并在云端重放新 CPU 结果。

进度和错误记录在 `data/seed-evolution/2026-11-11-large-v1/`。最终报告在 `docs/evidence/seed-evolution-2026-11-11-large/report.md`。
