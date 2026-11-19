# Pulse game 数据索引

从 [晨间报告](../../docs/pulse-game-morning-2026-11-19.md)读主要结果，从 [证据与复现入口](../../docs/pulse-game-evidence-2026-11-19.md)按研究问题找到逐项证据。实验命令、依赖及复现范围见 [研究代码说明](../../experiments/pulse_game/README.md)。晨间 CPU 运行包及其图表、校验记录随仓库分发，其他实验目录指向本机原始档案。

## 分发与 CPU 复验

1. 下载 [continuous-learning-cpu.zip](morning-20261119/continuous-learning-cpu.zip)，解压到新目录。依照[运行包说明](morning-20261119/README.md)复验迁移、长生命及跨游戏任务。此前的干净环境复验使用 Python 3.14.5 和 NumPy 2.5.4。[干净环境数值与 SHA-256](morning-20261119/portable-verification.json)
2. 校验下载的 ZIP 时，将它的 SHA-256 与上述记录的 `archive_sha256` 比较。包内 `manifest.json` 记录参考个体的选择和各成员的 SHA-256。[发布清单](morning-20261119/release-verification.json)另记录本机发布涉及的报告和代表性证据的 SHA-256，`files` 中的路径从仓库根目录解析。
3. 从代码运行时，[参考检查点](morning-20261119/reference-agent.npz)与[读出配置](morning-20261119/readout-options.json)在 `morning-20261119/`。要恢复某个中途生命，按包内 `save_life` / `restore_life` 接口使用该生命自己的状态文件，并让环境提供同一时刻的后续观察。[独立进程恢复记录](morning-20261119/fresh-process-restart.json)

## 读数、训练与冻结存档

| 想查什么 | 入口 |
|---|---|
| 晨间统计与图 | [learning-and-retention.json](morning-20261119/learning-and-retention.json) · [出生统计](morning-20261119/independent-birth-statistics.json) · [拓扑配对](morning-20261119/topology-comparison.json) · [学习结果 PDF](morning-20261119/learning-results.pdf) |
| 12 次独立出生及训练来历 | [出生登记](independent-lifelong-births-v1/registered.json) · [训练审计](independent-lifelong-births-v1/whole_recipe_verified.json) · [逐出生档案](independent-lifelong-births-v1/) |
| 生内长跑的状态与恢复位置 | [分布式地址](native-long-life-v1/) · [局部地址](native-long-life-local-v1/) · [选择性读写](native-long-life-selective-v1/)。各目录分别保存结果与队列分片状态 |
| 11 月 19 日冻结对照、原游戏复测与性能核验 | [research-2026-11-19/](research-2026-11-19/) 收纳冻结分支、复测、汇总和精度/吞吐核验。[独立审计](research-2026-11-19/independent-audit.md) · [按问题查证据](../../docs/pulse-game-evidence-2026-11-19.md) |
| 情境地址、主动探查与两步世界 | [局部地址](context-local-20261119/README.md) · [连续主体的付费探查](context-active-20261119/README.md) · [探标游戏](scout-20261119/README.md) · [两步世界](twostep-20261119/README.md) · [隐式路线翻转](twostep-adaptive-20261119/README.md) · [严格实际奖励](twostep-actual-reward-20261119/README.md) |
| 回找早期搜索、协议与来源 | 根目录的 `ring-alt-*`、`spike-*` 文件保留发现、确认与冻结阶段。各次 `meta-*`、`attractor-*`、`dendrite-*` 保存分支工件。[环形任务冻结配置](ring-value-search-v1/frozen.json) · [早期证据索引](meta-evidence/README.md) · [来源校验](meta-evidence/source-checksums.json) |

再分析或移交单条分支时，从相应报告找出该分支的冻结协议、源检查点、逐种子结果与完成清单，按清单保留路径。仅复验晨包时，使用上面的 ZIP 和运行说明即可。
