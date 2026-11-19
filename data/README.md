# 本机数据索引

`data/` 保存原始运行结果、检查点、恢复材料及发布包。Git 收录这份索引、[Pulse game 数据索引](pulse-game/README.md)和[晨间 CPU 分发材料](pulse-game/morning-20261119/README.md)，包括参考个体、图表和校验记录。其他本机入口需要相应的原始档案。按研究问题查报告、汇总和逐种子读数，见 [研究证据索引](../docs/evidence/README.md)。选择训练、复现或恢复命令，见 [实验代码索引](../experiments/README.md)。

## 研究档案与分发

| 目的 | 本机入口 | 关联说明 |
|---|---|---|
| 找某条研究线的逐种子数据和原始存档 | `data/<研究名>/<运行版本>/`，具体路径见证据索引 | 先看报告的指标口径，再核对对应协议、逐种子结果和来源 SHA。[研究证据索引](../docs/evidence/README.md) |
| 查共享图、私有符号或局部学习规则 | [undirected/](undirected/) | 首轮共享图为 `main-v1/`。后续运行按起始日期与实验命名。[共享图结果](../docs/undirected-results-2026-10-30.md) · [局部学习规则](../docs/local-rule-experiment-2026-11-19.md) |
| 查初始化搜索的代际进度和恢复材料 | [11 月 11 日 FP32 主研究](seed-evolution/2026-11-11-large-v1/) · [11 月 15 日 BF16 放大](seed-evolution/2026-11-15-scale-bf16-v1/) | `progress.json` 是阶段状态。按对应协议和调度修订辨认恢复入口。[FP32 完整结果](../docs/evidence/seed-evolution-2026-11-11-large/report.md) · [初始化搜索代码说明](../experiments/seed_evolution/README.md) |
| 下载或复验 Pulse game 晨包，查训练及冻结证据 | [pulse-game/](pulse-game/README.md) | 有 CPU 运行包、发布 SHA-256 清单以及各阶段原始材料的入口 |
| 查配置与源码参考件 | [literature/memt-2026-11-07/](literature/memt-2026-11-07/) | 存放外部实现的配置、源码和来源清单。对应 [MEMT 研究笔记](../docs/memt-research-2026-11-07.md) |

移交完整实验时，按该轮协议一并携带原始结果、恢复存档、源检查点和源码快照。结果 JSON、图和摘录适用于读数。恢复还可能需要世界、随机数、优化器及训练状态，具体由各研究的检查点和协议定义。新复跑使用独立输出目录，已有报告和审计可直接读现存文件。

## Coverage 两个目录

| 路径 | 内容与用途 |
|---|---|
| [coverage-exit/](coverage-exit/) | 正式结果。[启动覆盖协议](coverage-exit/coverage-v1/protocol.json) · [删边配对报告](coverage-exit/retirement-v1/report.json) · [弱边退出恢复核验](coverage-exit/magnitude-v1/actual_retirement_recovery.json) · [性能实测](coverage-exit/perf/) |
| [coverage_exit/](coverage_exit/) | [早期覆盖烟雾检查](coverage_exit/coverage-smoke-v1/protocol.json)及 [删边算子验证](coverage_exit/retirement-validation-v1.json)。正式删边协议通过这个确切路径读取并记录验证文件 SHA-256。[协议生成代码](../experiments/coverage_exit/retirement_study.py) |

完整命令及后续复访用到的恢复材料见 [Coverage 实验入口](../experiments/coverage_exit/README.md)与 [研究报告](../docs/reports/structure/coverage-exit-experiment-2026-11-02.md)。

## 共享框架与历史日志

- [results.jsonl](results.jsonl) 是 [共享框架 CLI](../scripts/README.md) 追加和查询的登记文件，使用文件环境时写入 `env/`。
- [results-pre-2026-10-24.jsonl](results-pre-2026-10-24.jsonl) 保留旧版登记。两份登记的来历见 [变更记录](../CHANGELOG.md)。
- 根目录原有的 27 份控制台日志按研究线归入 [consolidation](logs/consolidation/)、[continuity](logs/continuity/)、[dynamics](logs/dynamics/)和 [integration](logs/integration/)。文件名与字节内容可用 [SHA-256 清单](logs/SHA256SUMS)核对，清单路径相对 `data/logs/`。各运行目录内的日志随原档案查找。
