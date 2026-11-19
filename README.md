<a id="agi"></a>

# agi · 持续学习研究

我们想知道，一个智能体能否在时间中持续运行，让自己的行动、反馈和过去的经历改变以后的能力。远期目标是 AGI，眼前的问题是经验如何积累，以及学新事物时怎样保住旧能力。[研究目标与当前判断](agi.md)

<a id="目前测到了什么"></a>

## 最近一次实验

11 月 19 日，12 个独立出生的网络在 B 连续学 48 对冲突样本后回访 A。使用分布式地址时，首次正确率为 27.3%，用当前情境的局部地址写入时为 90.3%。另一项同预算拓扑对照中，独立训练的感觉单元也能达到有向循环图的水平。[晨间报告](docs/pulse-game-morning-2026-11-19.md) · [结果图](data/pulse-game/morning-20261119/learning-results.png) · [实验对照与证据](docs/pulse-game-evidence-2026-11-19.md)

<a id="文档与目录"></a>

## 从哪里读起

| 想做什么 | 入口 |
|---|---|
| 追溯最初的想法和后来的修订 | [10 月 22 日草案](docs/archived/design-memory-learning-2026-10-22.md) · [设计复核](docs/review-2026-10-26.md) · [现行设计](docs/design.md) |
| 按问题或日期查历轮实验 | [文档目录](docs/README.md) · [分主题报告](docs/reports/README.md) · [实验总索引](docs/experiments.md) · [证据目录](docs/evidence/README.md) · [变更记录](CHANGELOG.md) |
| 给新任务选神经元、沿用旧训练和加速方式 | [按神经元类型导入，含 LIF 和序列训练示例](src/neurons/README.md) |
| 找训练、评估、复现代码 | [游戏实验脚本](experiments/pulse_game/INDEX.md) · [全部实验代码](experiments/README.md) |
| 找结果、检查点或可分发的包 | [本机数据目录](data/README.md) · [CPU 运行包](data/pulse-game/morning-20261119/continuous-learning-cpu.zip) · [运行包用法](data/pulse-game/morning-20261119/README.md) |
| 查看早期共享 LIF 框架和维护命令 | [共享框架](src/README.md) · [维护工具](scripts/README.md) |

<a id="最少命令"></a>

## 本机查看

在仓库根目录运行以下命令，分别检查文档链接、列出共享框架配置和查看冻结游戏个体的运行参数。

```powershell
.\.venv\Scripts\python.exe -B scripts/check_docs.py
.\.venv\Scripts\python.exe -B scripts/run.py list
.\.venv\Scripts\python.exe -B -m experiments.pulse_game.lifelong_agent --help
```
