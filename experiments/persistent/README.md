# 持续记忆、循环训练与真实反馈的最小实验

2026-10-26 实现并在本机 RTX 4060 Ti 上运行。结果与解释见[实验报告](../../docs/persistent-experiment-2026-10-26.md)，可保存的指标与图见[证据目录](../../docs/evidence/persistent-2026-10-26/)。

本实验只回答：固定状态的长延迟保持、可寻址记忆的读取、多步计算的训练方式，以及真实二值反馈能否改善一个已有模型。不提供语言能力、自由探索、终身学习或意识的结论。

## 结构与任务

每个样本是重新随机生成的键值置换图。模型先看到乱序事实，再收到起点和跳数，回答沿事实走指定步数后的终点。训练词表 64；主实验含 8/16/24 条事实、1–4 跳。扩大到 64 条事实和 8 跳用于外推。没有中间节点教师强制。

`MemoryLoop` 由一个嵌入、一个共享内容寻址器、小型状态更新网络和可选自评头组成，共 33,346 参数。记录保存原始键值，写入和保留规则固定；训练学习表示、读取、计算和输出。它没有学会存储分配或删除策略。

`CompressedGRU` 有 211,905 参数，所有过去的事实必须经固定 128 维隐状态进入回答；查询到来后不能重读事实。它支持分段调用 `encode_chunk` 并传递状态。GRU 与记忆模型的比较同时改变存储和计算结构，不是参数/内存容量严格匹配的对照。

NOOP 延迟不会增加有效事实数。GRU 会执行空输入状态更新；记录存储会跳过 NOOP，因此后者的时间保持是预设存储能力。容量外推单独测量，不能把两者混为一谈。

## 代码入口

| 文件 | 用途 |
|---|---|
| [tasks.py](tasks.py) | 随机世界、独立字典 oracle 与数据契约 |
| [models.py](models.py) | GRU、共享读取循环、自评与反馈 |
| [run.py](run.py) | 监督机制对照、训练日志、最后一个检查点 |
| [evaluate.py](evaluate.py) | 用独立的新世界复测已保存模型 |
| [adapt.py](adapt.py) | 只用采样动作的真实成功/失败奖励做适应 |
| [summarize.py](summarize.py) | 导出完整 JSON 指标和 PNG/SVG 图 |
| [test_tasks.py](test_tasks.py)、[test_models.py](test_models.py) | 任务真值、标签隔离、梯度和状态边界检查 |

## 运行环境

本次使用 Python 3.14.5、PyTorch 2.11.0+cu130、NumPy 2.5.1、Matplotlib 3.10.9。测试使用 pytest。CUDA PyTorch 安装在仓库 `.venv/`，系统原有 CPU PyTorch 未替换。

在新环境中，先安装适合设备的 PyTorch，再安装 NumPy、Matplotlib、pytest。以下命令从仓库根目录执行；其它工作目录可使用脚本绝对路径。相对输出路径均以仓库根目录解析。

```powershell
.venv/Scripts/python.exe -m pytest experiments/persistent/test_tasks.py experiments/persistent/test_models.py -q
.venv/Scripts/python.exe experiments/persistent/run.py --output data/persistent/main-v1 --steps 1000 --log-every 250
.venv/Scripts/python.exe experiments/persistent/run.py --output data/persistent/read-budget-v1 --modes single --steps 4000 --log-every 1000
.venv/Scripts/python.exe experiments/persistent/run.py --output data/persistent/retention-v1 --modes gru,single --seeds 10,11,12 --steps 5000 --max-hops 1 --train-records 2,4,8 --train-gaps 0,4,16 --log-every 1000
.venv/Scripts/python.exe experiments/persistent/evaluate.py --input data/persistent/main-v1 --output data/persistent/heldout-main-v1 --suite core --seed-offset 2900000 --batches 16
.venv/Scripts/python.exe experiments/persistent/evaluate.py --input data/persistent/read-budget-v1 --output data/persistent/heldout-budget-v1 --suite core --seed-offset 2900000 --batches 16
.venv/Scripts/python.exe experiments/persistent/evaluate.py --input data/persistent/retention-v1 --output data/persistent/heldout-retention-v1 --suite retention --seed-offset 5900000 --batches 16
.venv/Scripts/python.exe experiments/persistent/adapt.py --output data/persistent/adapt-v1 --eval-batches 16 --log-every 100
.venv/Scripts/python.exe experiments/persistent/summarize.py
```

上述目录已包含本次结果。复跑时换用新的输出目录，且同步修改后续 `--input` / `--checkpoint-dir` / 汇总数据根目录；程序拒绝覆盖非空输出目录。`run.py --resume` 只复用配置、环境、源码和结果指纹全部匹配的已完成运行，不从中间步恢复优化器。

完整检查点与开发日志在被 Git 忽略的 `data/persistent/`。证据目录中的 JSON 保存本次所有正式配置、种子、检查点哈希与逐条件指标；它也包含开发评估，报告和图使用明确标出的 heldout 结果。pilot 与 CPU smoke 不进入正式汇总。

## 对照解释

| 模式 | 训练方式 | 能回答的问题 |
|---|---|---|
| `single` | 单跳样本、单步训练；测试时重复同一读取器 | 单步能力能否直接组合成多步能力 |
| `detach` | 同样多步前向，步间只截断状态梯度 | 与 `joint` 比较跨步信用分配 |
| `joint` | 多步最终答案监督、完整跨步反传 | 同一读取器在展开计算中学习 |
| `judge_aux` | `joint` 加候选正确性的监督，不反馈自评分 | 辅助判断目标的作用 |
| `judge` | 同时将预测的自评分送入下一步 | 自评辅助与反馈的总体作用 |
| `gru` | 固定状态编码事实，再进行循环回答 | 压缩状态基线 |

主循环实验中每组 128,000 个训练样本。`single` 的 4,000 步补充对照有 512,000 个样本，实际读取调用次数与其它 1,000 步循环组相同，但样本数、更新次数、FLOPs 和耗时不全相等。不能把它称为严格计算预算完全匹配。

自评监督由环境真值产生；推理只能使用模型自己的分数。判断候选是否正确与判断需要几步不同，结果记录分步 Brier、最终步校准和仅依赖步位置的验证集基线。循环次数由任务给定，尚未训练自主停止。

`adapt.py` 中 `reward` 使用 leave-one-out 基线的 REINFORCE；`critic` 用共享核心的价值预测及反馈。两组均只从真实二值奖励学习，目标答案不进入交叉熵或模型输入。`critic` 的共享辅助损失和反馈同时改变，不能单独归因于自评。这个实现不是 DeepSeek 自反馈 RL 的复现。

默认训练参数都是本次工程取值，没有被确立为项目通用常数。数据生成器允许自环及短环，所以报告还包含始终回答查询本身的基线和无短环子集成绩；不能只用 1/64 作为机会水平。
