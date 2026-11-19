# 原生任务交错、历史复习与长间隔实验

方法和正式结果入口：[实验报告](../../docs/lifelong-experiment-2026-10-29.md)。正式执行依据是冻结的[主协议](../../docs/evidence/lifelong-2026-10-29/study-protocol-v1.json)及[覆盖补充协议](../../docs/evidence/lifelong-2026-10-29/coverage-probe-supplement-v1.json)。本文件说明接口与解释边界；行为结论与数表以通过严格审计的主报告为准。

## 模块与命令

在仓库根目录运行 `python -m experiments.lifelong.<module>`；使用当前项目中安装了 PyTorch 的 Python 环境。绘图另需 `matplotlib` 与 `ijson`；完成清单和精选诊断也使用 `ijson` 流式读取，避免一次加载完整汇总 JSON。

| 模块 | 职责 |
|---|---|
| `world.py` | 四种 family 的公开提示、payload 映射、真实命令曝光和最近接触时间 |
| `schedule.py` | 连续生涯的 blocked/mixed 主训练、family 0 撤出与返回 |
| `replay.py` | 真实历史的 256 tick 块、容量 32 的均匀 reservoir、历史完整性校验 |
| `run.py` | 从随机出生训练、完整恢复、固定检查点与计算量记账 |
| `evaluate.py` | 完整状态冻结分叉；由 Python 接口调用，不提供独立 CLI |
| `study.py` | 主协议与命令清单准备、绑定的单分支训练和主评估 |
| `supplement.py` | 独立冻结的同 horizon、支撑内补充评估，不追加训练 |
| `queue.py` | 按命令清单执行、限并发、等待完整训练依赖和恢复执行记录 |
| `report.py` | 读取主测量和补充测量，通过严格审计后输出 JSON/CSV |
| `plot.py` | 仅从通过审计的汇总绘制科学图，保存逐种子绘图数据和来源 SHA |
| `finalize.py` | 核验汇总、队列、成功测试与图表，生成最终完成清单和文件指纹 |

独立训练入口适用于机械检查或另立的新实验。新运行要求新的输出目录；`--stop-at` 是累计 live tick，必须对齐 128 tick 且不超过完整协议。

```powershell
python -m experiments.lifelong.run --output data/lifelong/example-new --seed 0 --architecture fixed_shrink --schedule mixed --learning history_replay --coverage wide --device cpu --stop-at 1024
python -m experiments.lifelong.run --output data/lifelong/example-new --resume data/lifelong/example-new/latest.pt --device cpu --stop-at 32768
```

`--resume` 从检查点继承分支配置、模型、h、世界、优化器、结构、历史与 RNG；命令行上的出生配置不改写恢复状态。上述例子的独立输出没有主协议的启动与完成绑定，不能冒充正式分支。

新的协议目录与原始数据目录可由以下入口准备；目标协议文件若存在会拒绝覆盖。准备操作绑定当时的执行源码和依赖，不能用它回写已经冻结的正式协议。

```powershell
python -m experiments.lifelong.study prepare --output docs/evidence/lifelong-example-new --raw data/lifelong/example-main-new
python -m experiments.lifelong.supplement prepare --parent-protocol docs/evidence/lifelong-example-new/study-protocol-v1.json --output docs/evidence/lifelong-example-new --raw data/lifelong/example-coverage-new
```

正式分支使用生成清单中的完整 `command` 数组。`study train` 接受 `--protocol` 与 `--branch-id`，训练设备取协议；`study evaluate` 另需 `--checkpoint` 指向该分支的 `latest.pt`，以及尚不存在的 `--output` 文件。`supplement evaluate` 使用补充协议，参数接口相同。评估会核验完成绑定，再逐一读取预定快照；不是只评估传入的最终权重。

批量执行入口如下；三份清单必须属于同一运行。limits JSON 需包含正整数 `training`、`cpu`、`total`，可另设 `min_available_bytes`。训练和 CPU 任务共用 `total` 上限；修改限制会保留时间记录。补充队列使用 `coverage-` 前缀的三份清单，它们的训练和基线作业列表为空，依赖已经存在的主训练完成标记。

```powershell
python -m experiments.lifelong.queue --training-plan docs/evidence/lifelong-example-new/training-commands-v1.json --baseline-plan docs/evidence/lifelong-example-new/baseline-commands-v1.json --evaluation-plan docs/evidence/lifelong-example-new/evaluation-commands-v1.json --status-root data/lifelong/example-main-new/status --limits data/lifelong/example-main-new/limits.json
```

严格汇总必须同时提供主协议、主数据根目录、补充协议与输出目录。补充数据根目录由补充协议绑定；`--workers` 控制检查点审计并发。输入不完整或审计失败不会发布结果 artifact。

```powershell
python -m experiments.lifelong.report --protocol docs/evidence/lifelong-2026-10-29/study-protocol-v1.json --root data/lifelong/main-v1 --supplement docs/evidence/lifelong-2026-10-29/coverage-probe-supplement-v1.json --output docs/evidence/lifelong-2026-10-29/aggregate-v1 --workers 6
```

汇总通过后生成图表；完成主报告、回归测试和图表核验后，再生成完成清单。`finalize` 会绑定当时文件的 SHA，之后若改动报告或分析文件，应重新生成清单。

```powershell
python -m experiments.lifelong.plot --report docs/evidence/lifelong-2026-10-29/aggregate-v1/report.json --output docs/evidence/lifelong-2026-10-29/figures
python -m experiments.lifelong.finalize --evidence docs/evidence/lifelong-2026-10-29
```

主报告中的精选请求/动作诊断可用 [diagnosis_v1.py](../../docs/evidence/lifelong-2026-10-29/diagnosis_v1.py) 复算。该脚本核验所用汇总 CSV 和原始分叉的 SHA，输出每个种子的分子、分母及三种子汇总，不产生新经验或训练。

```powershell
python docs/evidence/lifelong-2026-10-29/diagnosis_v1.py
```

## 设计与解释边界

- 三个独立训练种子，每种子十个分支。固定结构完整交叉两种课程、两种复习口径和两种时间覆盖，共 24 个分支；grow/open 只在 mixed、history_replay、wide 中各运行三个种子。没有完整的结构交互矩阵。
- 每个种子共享初始编码器、动作头和稀疏循环函数：宽度 128、初始每行 32 个输入，共 4,096 个循环非零。结构规则从第一次更新开始执行；没有从旧 B 状态移植结构。
- `fixed_shrink` 保留初始可更新集合，`grow_shrink` 使用原有退出和影子候选再生长，`open_shrink` 允许全部循环位置普通更新。三者均用相同软收缩。分配参数、允许坐标、实际非零和累计生灭是不同数量；后续容量不匹配，比较只约束完整更新策略。
- 四种 family 共享请求处理协议，包含部分相同、部分冲突的符号映射。family one-hot 仅在 NEW/REPLACE 提示时出现。撤出的 family 0 是原样映射；family 1 的 raw 2/3、family 2 的 raw 0/1 仍使用原样答案，撤出期间仍有共享案例。它们超出纯时间分布切换，但不构成独立领域专有知识、复杂现实领域、层级任务或组合推理的验证。
- 完整生涯为 172,032 live tick：主训练 131,072、撤出 future NEW 的 family 0 共 32,768、返回 family 0 共 8,192。切换不重置 h、世界、已有合同或优化器。撤出不保证立刻没有旧 family 0 合同、live 标签或历史复习。
- `recent_repeat` 与 `history_replay` 都进行辅助学习；它们比较最近历史重复与全生涯历史抽样，没有纯 live-only 条件。两种方法都从保存的真实起始 h 出发，使用 128 tick 无梯度 warmup 和 128 tick 有梯度 suffix；首次启动为共同的短回退。
- 历史复习使用自身旧行动轨迹中的观察和原始因果教师标签，有限 warmup 由当前权重执行。旧 h 不等于当前参数下的精确无限历史状态；辅助状态不覆盖 live h，也不生成新的闭环经验。
- 每 128 live tick，以同一份更新前参数分别计算 live 和辅助的各自归一化目标梯度，等权平均后处理允许集合、联合裁剪、执行一次 Adam 和结构规则。辅助计算有成本；同 live tick 不保证同实际 NEW 数、family 曝光或候选集合。
- narrow 与 wide 同时改变时间支撑和事件密度。比较须展示真实 NEW、REPLACE、监督位置和最后接触时间，不能仅凭名义 tick 宣称请求曝光相同。
- 主协议保留每分支 46 个分叉，其中 horizon 为 1,024 或 2,048 tick；补充协议另加 16 个 2,048 tick 分叉。tick 0 仅核验共同出生参数，没有行为测量。纵向主要使用 `matched_memory_*` 和 `matched_temporal_*`，在同条件、同 2,048 tick horizon 下比较。原 `temporal_*` 保留来源 horizon，不应直接当作完全匹配比较。
- short 在两种训练支撑内；long_both 在 wide 支撑内、在 narrow 支撑外。先核对撤出前是否已获得该函数，再解释撤出后的下降；支撑外的低分或低分地板本身不能证明遗忘。
- 冻结评估继承完整 h、世界、时钟、待发生事件和 RNG，参数更新为零，执行全部 11 类动作的 argmax。它测量完整状态的延续功能，不单独隔离权重知识；评估期间仍会收到新任务提示和正常回执。
- 主指标的分母是成熟、最终有效的分叉后请求；普通完成、严格完成、继承请求、旧合同 carryover、右删失与空闲误动作分别保留。actor、请求、family、检查点与分叉不增加独立训练种子数。
- 长任务等待、实际空闲年龄、真实提示缺席、live 监督缺席和辅助监督缺席使用不同时间轴。保持与返回曲线不能自动解释为无复习、纯间隔效应、冷启动加速或一般智能。

