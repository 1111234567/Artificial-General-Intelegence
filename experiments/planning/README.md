# 消耗式配方规划

这一实验把记忆暂时固定为完整可见的规则和库存，单独检查共享内部计算是否有助于规划。执行配方会真实消耗输入物品；模型每次选一个合法动作，完整计划必须由独立执行器验证。

[完整实测报告](../../docs/planning-experiment-2026-10-26.md)：三个种子未显示当前共享循环的稳定优势；同一配对子集的浅层贪心 / 共享四轮 / 浅层引导搜索分别为 27.58% / 21.82% / 88.33%。搜索的额外转移工作及无学习对照分别报告。

## 文件

| 文件 | 用途 |
|---|---|
| [world.py](world.py) | 整数库存、消耗式规则、有限资源分配生成器、精确图搜索和独立计划检查器 |
| [model.py](model.py) | 单层、共享循环、独立多层及截断反馈梯度的规则 Transformer |
| [heuristics.py](heuristics.py) | 仅使用公开输入的资源保护与忽略未来消耗的启发式，用于暴露任务捷径 |
| [run.py](run.py) | 冻结数据、配对训练、实际计划执行及浅层模型引导的外部搜索 |
| [report.py](report.py) | 汇总已完成运行，保存测试题、所有预测计划和图表 |
| [search_control.py](search_control.py) | 明确标为事后诊断的无学习搜索，扩展数与神经调用分开计量 |

## 实验边界

- 精确求解器用于生成验收和训练最优动作集合；推理策略不能访问未来距离、可解性标签或教师计划。
- 模型输入只有当前库存、目标、配方及当前可执行性。可执行性由模拟器检查，不会排除合法但必败的动作。达成目标后的停止也由模拟器提供。
- 每次环境动作前将模型内部状态置零。此项研究内部计算，尚未检验跨任务经验积累、长期隐状态或自主停止。
- 所有最优下一步都进入监督集合，不把另一条正确最短计划标错。
- 浅层模型在同一个外部批次上更新四次；共享四轮、独立四层和截断梯度模型各更新一次。匹配的是核心层处理样本数，不是严格 FLOPs、反向传播量或实际耗时。
- 浅层外部搜索实际扩展候选库存，使用策略概率排序和公共状态转移；另计模型调用、模拟转移和搜索工作。它不调用精确求解器。外部搜索与贪心策略只在同一测试子集上比较。
- 增加内部轮数需要增加推理计算；超出训练轮数后的表现属于外推，不预设会改善。

## 运行

从仓库根目录运行，需要 PyTorch；绘图另需 Matplotlib。新运行目录必须为空，默认不读取正式测试结果。

```powershell
.venv/Scripts/python.exe -m pytest experiments/planning -q
.venv/Scripts/python.exe -m experiments.planning.run train --help
.venv/Scripts/python.exe -m experiments.planning.run evaluate --help
```

训练将配置、代码副本、数据和指纹写入指定目录，使用最终检查点；验证集可以监测，独立测试需显式调用 `evaluate --split test`。再次评估同一划分会拒绝覆盖。完整实测配置和结论以本轮实验报告为准。

正式训练完成后，无学习诊断与证据汇总可用：

```powershell
.venv/Scripts/python.exe -m experiments.planning.search_control --out data/planning/main-seed0-v1 data/planning/main-seed1-v1 data/planning/main-seed2-v1 --threads 1
.venv/Scripts/python.exe -m experiments.planning.report data/planning/main-seed0-v1 data/planning/main-seed1-v1 data/planning/main-seed2-v1 --out docs/evidence/planning-reproduced
```

这些已完成的原目录不会重复写入；复现时使用新的训练和证据目录。
