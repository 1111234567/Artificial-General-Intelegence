<a id="agi"></a>

# agi · 持续学习研究

我们想知道，一个智能体能否在时间中持续运行，让自己的行动、反馈和过去的经历改变以后的能力。远期目标是 AGI，眼前的问题是经验如何积累，以及学新事物时怎样保住旧能力。[研究目标与当前判断](agi.md)

<a id="目前测到了什么"></a>

## 最近一次实验

局部可塑图在一生中自行选择、接收反馈并修改已用通路。只用实际终局资源启蒙后，面对陌生气味，一次反馈、等待四拍的留出正确率为 98.61%；关闭生内写入后为 50.11%。[局部迹与反馈实验](docs/local-trace-feedback-2026-11-19.md) · [上一轮对反馈含义的检验](docs/revisit-choice-feedback-2026-11-19.md)

自改写样本试验把同一个体的连续经历延长到 512 道新题，并在相同训练预算下比较独立历史和梯度时间范围。[结果、完整写入权限与梯度审计](docs/self-edit-dose-2026-11-19.md) · [项目参数术语](docs/graph-parameter-terms.md)

自改写实验先用梯度教图通过一次接触写入偏好，再停止外部优化，观察它怎样修改普通边、偏置和产生写入命令的连接。[启蒙、写入器移植和结构读回](docs/self-writing-bootstrap-2026-11-19.md)

16 节点无端图从零个二阶系数开始，仅凭自己的接触回执选择哪些入边活动应当组合。一生中平均只安装约 40 项，仍学会了需要成分组合才能判断的食物偏好。[稀疏生长、功能定位与换规律测试](docs/sparse-native-learning-2026-11-19.md) · [此前的无端图学习与经验移植](docs/end-free-learning-2026-11-19.md)

自由拓扑试验放开了逐节点度数，并比较了先形成多个环、再用跨环边连接的出生结构。训练期和生命内的连接增退分别测量。[接线、实际成绩和结构变化](docs/free-graph-2026-11-19.md)

同一天的后续研究把自环、有向单环、无端点多环放到相同参数和训练预算下，逐项检查三张真值表。另一项实验让个体根据实际接触决定要借用多少旧经验。[共享旧经验和闭合拓扑的结果](docs/feedback-and-closed-topologies-2026-11-19.md)

11 月 19 日的后续试验中，相关经历使个体在陌生规律上的早期学习增益多出 **8.52 个百分点**。独立的顺序辨别复测中，64 对新网络在相同训练预算下，有向环和自环的长间隔正确率分别为 **90.15% 和 69.76%**。[结果、机制对照与适用范围](docs/learning-transfer-2026-11-19.md)

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
