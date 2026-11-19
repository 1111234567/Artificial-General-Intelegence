# 交错经历与记忆巩固

接续 [streaming](../streaming/README.md)。本分支把环境中的共同规律、上下文冲突、可观察的规则修订和反复返回放到同一个简单学习问题中。目的在于分开测量外界再次提供经验与内部重放的作用，不预设不可修改的长期连接或固定的重要性等级。

已完成三种子、两种日程、三种更新方式共 18 组正式测试。结果与适用边界见[完整报告](../../docs/reports/memory/consolidation-experiment-2026-10-26.md)，可移植数据与图见[证据目录](../../docs/evidence/consolidation-2026-10-26/)。

## 最小系统

- [world.py](world.py)：三个可观察上下文共享原始连续属性。A 使用基础分类规律，B 对规律作部分扰动，C 使用条件化的不同答案。A 后来出现可观察的版本变化；新版本与 C 也不同。模型看不到“重合/冲突”标签。
- [model.py](model.py)：两层隐藏层的共享 MLP，用环境给出的类别反馈训练。没有自评、睡眠阶段、结构生长或受保护参数。
- [memory.py](memory.py)：有限原始经历，随机优先级构成均匀历史样本。写入快、参数整合慢是功能分工；不主张复现生物短期/长期记忆。
- [run.py](run.py)：按真实发生顺序训练。每批外界经历先预测，再接收反馈；历史重放只能读取此前已经发生的经历。评估只读模型权重，不读取记忆库，也不更新系统。

三个对照都维护同样大小的经历库。`online` 对当前批次更新一次；`repeat` 更新当前批次两次；`replay` 更新当前批次后，再更新一批过去经历，空库时回退当前批次。主要因果比较是计算量相同的 `repeat` 与 `replay`；`online` 的训练计算量只有一半。训练次数固定，不按测试结果选检查点。

## 环境与日程

输入为 32 维正态属性、三个上下文的 one-hot 和两个规则版本的 one-hot。A0 的四分类答案来自随机线性教师，B0 来自部分共享教师，C0 为 A0 答案循环偏移 1，A1 为偏移 2。版本只在 A 修订时变化；B/C 保持原版本。上下文和版本是显式条件，不代表系统自主发现情境边界或变化。

训练、验证和测试样本独立。普通样本排除前两维同时为正的区域，转移样本只来自这个联合区域；单个属性取值在训练中仍有覆盖。这只是一个有限分布外推检查，不代表一般组合推理。

每个区块默认 100 次外界输入，每次 256 个样本：

- 集中日程：A0 六块、B0 六块、C0 六块，随后 A1 三块、B0 三块、C0 三块。
- 交替日程：`A0 B0 C0` 重复六轮，随后 `A1 B0 C0` 重复三轮。

两日程每个条件内部的样本序列完全相同，只改变穿插顺序。最终距离各条件最后一次训练的间隔不同，因此日程终点差异不能单独解释成纯间隔巩固。日志保存条件年龄、每次返回前后的独立评估、反馈前流成绩和真实重放次数。A0 历史知识与 A1 当前规则始终分别评估，旧记录不改标签。

## 复现

先仅使用验证集确定可学性与预算；正式运行使用固定配置和未参与选模的种子。输出目录必须为空。

```powershell
.venv/Scripts/python.exe -m pytest experiments/consolidation -q
.venv/Scripts/python.exe experiments/consolidation/run.py --output data/consolidation/NEW-pilot --seeds 101 --methods repeat,replay --steps-per-block 30 --eval-split val
.venv/Scripts/python.exe experiments/consolidation/run.py --output data/consolidation/NEW-main --seeds 0,1,2 --eval-split test
.venv/Scripts/python.exe experiments/consolidation/summarize.py --inputs data/consolidation/NEW-main --output data/consolidation/NEW-summary
```

已完成的正式输出按种子并行分开保存为 `data/consolidation/main-seed0-v1`、`main-seed1-v1`、`main-seed2-v1`。汇总这些记录时，把三条路径一起传给 `--inputs`；生成器验证完整性、源码与配对哈希，要求每个日程/方法都有相同的三个种子。

每次运行保存完整配置、源代码快照及哈希、逐区块结果、最终参数/优化器/经历库、外界输入序列哈希。`results.json` 中的 `prequential_accuracy` 是训练流反馈前的成绩，可能含过去见过的样本，不能代替独立测试结果。

这里先检验真实经历的均匀重放，没有实现学出来的重要性、选择性回忆或生成式梦境。相关动机来自[互补学习系统](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)与[提取练习研究](https://doi.org/10.1126/science.1152408)，本分支不声称复现这些生物实验。
