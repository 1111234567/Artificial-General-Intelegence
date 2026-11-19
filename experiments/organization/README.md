# 原生习得、非线性组织与功能性连接试验

方法与正式结果统一进入[实验报告](../../docs/organization-experiment-2026-10-30.md)。本目录的执行源码、测试及正式协议已经冻结，两套正式汇总均已通过严格审计。本 README 说明研究问题与复现接口；行为结论以通过审计的汇总和主报告为准。

## 三个研究问题

1. **从出生开始的收缩是否影响能力获得？** `acquisition.py` 新训三个种子、三种条件，共 9 个 CUDA 分支，仅把相应旧原生条件的收缩从 `2e-4` 改为 `0`。条件为 fixed/wide、open/wide、fixed/narrow，均 mixed/history。同种子的旧出生函数、优化与时长保持匹配；旧 9 条完整生涯及其冻结分叉作为已存在的绑定证据借用，不重新训练或生成旧评估。
2. **相同状态、参数与矩阵运算预算下，非线性转移的组织是否有用？** 主实验比较 `plain / parallel / coupled` 的 passive 条件。它们在本研究内从相同块对角函数出生。`parallel` 与 `coupled` 连同 tanh 数量也相同，比较是否把跨 bank 反馈放入进一步的联合非线性。该出生函数与旧 lifelong/acquisition 的随机跨 bank 稀疏出生不同。
3. **按下一段真实学习损失选择连接，是否优于固定、幅度或随机候选？** plain 场比较 `functional_fixed / functional_task / functional_magnitude / functional_random`；coupled 场另做 fixed/task。出生相同、候选与探测规则匹配，允许接受后的容量轨迹不同。任务引导是否改善行为、是否伤害历史分量，都须实测。

主实验共有 `3 seeds × 9 variants = 27` 个新分支：

| 场 | 结构规则 |
|---|---|
| plain、parallel、coupled | passive，各 3 个种子 |
| plain | functional_fixed、functional_task、functional_magnitude、functional_random，各 3 个种子 |
| coupled | functional_fixed、functional_task，各 3 个种子 |

主实验全部使用 CPU；其队列进程设 `CUDA_VISIBLE_DEVICES=-1`。单独的 9 个习得对照保持 CUDA 训练。设备安排见[主实验启动环境](../../docs/evidence/organization-2026-10-30/main-launch-environment.json)，在查看正式行为成绩前确定。两项研究分别解释；CPU 主实验与 CUDA 习得对照之间的差异不能直接当成单一架构或收缩的因果效应。

## 共同模型与连续生涯

- 输入 112 维，32 个 actor、每个 actor 四槽，每槽 11 类动作。family 提示只随 NEW/REPLACE 出现；模型不接收私有计时器、映射后答案、实时空闲掩码或持续任务编号，仍收到原协议的动作和接受/拒绝回执。
- 持久状态 `h∈R^128`，两 bank 各 64 维。编码器和动作头连接全部状态；不预先给 bank 分配任务、记忆或计时角色。主实验初始每行仅在本 bank 随机保留 32 个循环输入，共 4,096 个非零；跨 bank 块为零。
- 每个模型分配 36,396 个参数，其中循环张量为一个 `128×128` 参数。各主实验场每次 field 计算均为四次 `64×64` 矩阵乘，16,384 MAC；plain 使用 128 个 tanh，parallel/coupled 使用 256 个。相同 MAC 不保证相同运行时间，也不代表稀疏计算已实现。
- 原 Heun 积分、`dt=.25`、最大子步 `.25`、固定 `-h` 漏项不变。每个子步计算两次 field；观察脉冲在积分后只注入一次。主训练、撤出、返回与优化边界均不清空数值 h、世界、合同、优化器、历史或 RNG。
- 每条生涯 172,032 live tick、1,344 次 Adam 更新：主训练 131,072，撤出 future NEW 的 family 0 共 32,768，返回 family 0 共 8,192。TBPTT 为 128；h 的数值跨块持续。
- 每次 Adam 前，使用同一份参数计算 live128 梯度与辅助 suffix128 梯度，按各自归一化目标等权平均，联合裁剪至 1，学习率 `.001`，无 Adam 权重衰减。辅助历史通常为无梯度 warmup128＋有梯度 suffix128；首次启动使用原共同回退。容量 32 的 reservoir 保留真实旧轨迹及原因果标签；辅助状态不覆盖 live h。
- 主实验均 mixed/history/wide：处理时长 1–16、空档 0.5–32；窗口 1.5–2.5，取消概率 `.20`、替换概率 `.30`。四个 family 共享请求协议且包含部分相同、部分冲突的符号映射。

写 `h=(x,y)`，循环矩阵分成 `[[A,B],[C,D]]`。三个场为：

```text
plain:    dx=-x+A tanh(x)+B tanh(y)
          dy=-y+D tanh(y)+C tanh(x)
parallel: dx=-x+A tanh(x)+tanh(B tanh(y))
          dy=-y+D tanh(y)+tanh(C tanh(x))
coupled:  dx=-x+A tanh(x+B tanh(y))
          dy=-y+D tanh(y+C tanh(x))
```

`B=C=0` 时三者出生函数完全一致；coupled 学到跨 bank 作用后可以产生跨 bank 混合偏导。原稠密循环图本来就有多种反馈路径，因此这里检验转移函数的非线性组织，不能把增加环计数当成机制证据。

功能性规则让所有循环坐标持续积累真实裁剪梯度和 Adam 矩，包括暂未获准的坐标；Adam 后将未允许的权重恢复为零，保留其矩。每 8 次更新产生最多 4 个出生与 4 个退出候选，等到下一真实 live 块及当时选中的辅助历史才验证。每次待执行试验另算四份损失：基线、出生候选、完整消融、半强度退出；每份都重放 live 与辅助窗口，不驱动世界，也不替换 h。

出生与退出分别决定是否接受；退出基线包含已接受的出生。退出试验将权重减半，绝对值不超过当时学习率时置零。接受要求等权 mixed loss 严格下降，live 或 history 分量仍可能变差。`functional_fixed` 做相同探测而不接受结构改变。没有总容量硬上限或一进一出约束。额外 forward、资格变化、实际非零、跨 bank 坐标和运行时间分别记账；passive 的数值过零与 functional 的资格生灭不混为同一事件。

候选数约束每次变动的幅度和探索速度，四次联合前向是另一固定预算。有限的 167 轮、每轮最多四条出生，使本次允许边最多增加 668 条，不能把最终未变稠密单独归因于自主选择。半幅更新也改变现有权重；本轮尚无长期 birth-only/fade-only 对照。

## 冻结执行接口

在仓库根目录使用项目 Python。下列入口是复现说明；不要对当前冻结目录重新 prepare 或改写已绑定文件。另立实验必须使用全新的证据目录与 raw 目录。

| 模块 | 职责 |
|---|---|
| `model.py` | 三种状态转移与同参数/MAC 预算 |
| `plasticity.py` | 候选、下一真实块损失试验、资格与 Adam 矩历史 |
| `run.py` | 共同出生、持续训练、完整保存与恢复 |
| `acquisition.py` | 9 个 CUDA 零收缩对照及旧证据绑定 |
| `study.py` | 27 个主实验分支、命令清单、训练与冻结分叉 |
| `acquisition_report.py` | 新旧习得对照的严格配对审计与 CSV |
| `report.py` | 27 条生涯、容量、损失损伤及分叉的严格审计与 CSV |
| `plot.py` | 只消费通过审计的报告与 CSV，保存逐种子图数据及 SHA |
| `finalize.py` | 绑定两个完整汇总、测试、图表与执行历史，生成完成清单 |

另立 CPU 主研究的准备接口为：

```powershell
.\.venv\Scripts\python.exe -m experiments.organization.study prepare --training-device cpu --output docs/evidence/organization-example-new --raw data/organization/example-main-new
```

`study train` 使用 `--protocol` 和 `--branch-id`；`--stop-at` 是对齐 chunk 的累计 live tick。`study evaluate` 另给该分支的 `latest.pt` 与未存在的预定输出。正式批处理使用生成的完整 command 数组，由冻结的 `experiments.lifelong.queue` 执行。

CPU 主研究须在独立 PowerShell 进程中设置设备屏蔽；习得对照的 CUDA 队列进程不设置这个屏蔽值：

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
.\.venv\Scripts\python.exe -m experiments.lifelong.queue --training-plan docs/evidence/organization-2026-10-30/organization-training-commands-v1.json --baseline-plan docs/evidence/organization-2026-10-30/organization-baseline-commands-v1.json --evaluation-plan docs/evidence/organization-2026-10-30/organization-evaluation-commands-v1.json --status-root data/organization/main-v1 --limits data/organization/main-v1/worker-limits.json
```

CUDA 习得对照使用同目录 `acquisition-*-commands-v1.json`，status/raw 根为 `data/organization/acquisition-v1`。两套队列各自保存并发限制与执行记录，不能混用清单。

每条生涯只保留 0、131,072、163,840、172,032 四份阶段快照，并周期保存 latest。后三份各做四 family × short/long_both × 2,048 tick：每分支 24 个完整状态冻结分叉。主实验为 108 份快照、648 个分叉；习得新分支为 36 份快照、216 个新分叉，另配对 216 个已有旧分叉。tick 0 只核验出生。

所有快照恢复保留模型、原 h、世界、合同、优化器、结构、reservoir 和 RNG。评估只改变未来 NEW 的 family 与时长范围，执行 11 类 argmax；参数、结构更新与隐藏状态重置均为零。新 organization checkpoint 与旧 lifelong checkpoint 版本互不冒充。

## 严格汇总与绘图

数据完整后先运行两项严格审计。主报告支持 `--workers 1` 到 `6`、默认 `6`，按独立分支并行恢复，最后按协议顺序合并并做跨分支核验；内存紧张时可用 `--workers 1`。此过程不重新训练，也不把机械测试结果当正式结果。

完整 RNG 验证也要求对应的设备可见性：习得检查点保存了 CUDA RNG，审计时 CUDA 必须可见，尽管模型恢复到 CPU；主研究在 `CUDA_VISIBLE_DEVICES=-1` 下保存，严格汇总也使用这个环境。

```powershell
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m experiments.organization.acquisition_report --protocol docs/evidence/organization-2026-10-30/acquisition-protocol-v1.json --root data/organization/acquisition-v1 --output docs/evidence/organization-2026-10-30/acquisition-aggregate-v1 --workers 3
$env:CUDA_VISIBLE_DEVICES = "-1"
.\.venv\Scripts\python.exe -m experiments.organization.report --protocol docs/evidence/organization-2026-10-30/organization-protocol-v1.json --root data/organization/main-v1 --output docs/evidence/organization-2026-10-30/aggregate-v1 --workers 6
```

只有两报告都 `complete=true`、审计 passed 且无错误、CSV 与 `artifacts.json` 的 SHA 一致时才绘图。绘图依赖 `matplotlib` 和 `ijson`，仅使用 CSV 和小型/流式报告元数据，不 import 训练模块、加载 checkpoint 或读取未审计 raw 表现。

```powershell
.\.venv\Scripts\python.exe -m experiments.organization.plot --acquisition docs/evidence/organization-2026-10-30/acquisition-aggregate-v1 --main docs/evidence/organization-2026-10-30/aggregate-v1 --output docs/evidence/organization-2026-10-30/figures --family-view mean4
```

`--family-view mean4` 先在每个种子内对四 family 的率等权平均，再对三个训练种子求均值和样本 SD；可改用 `family0` 专看撤出规则。不把四 family 的分母混池，也不把事件、actor、窗口或分叉当独立 n。图保留种子点、配对差及逐种子 JSON；缺失分母保留 undefined，不补零。输入报告、CSV、协议和绘图源码 SHA 随 PNG/SVG 一同绑定。

正式[图目录](../../docs/evidence/organization-2026-10-30/figures)包含三组 PNG/SVG、逐种子 `figure-data-v1.json`、SHA 清单及[视觉与统计核验](../../docs/evidence/organization-2026-10-30/figures/qa-v1.json)。现有目标目录会拒绝覆盖；复算须另给新目录。

主报告、最终测试与全部图表完成后，运行 `python -m experiments.organization.finalize` 生成[完成清单](../../docs/evidence/organization-2026-10-30/completion.json)。文档或分析文件后续变更时应重新绑定，不能让旧摘要冒充当前版本。

## 解释边界

- 首先看主训练末是否学会，再看撤出与返回。地板附近的低分不能证明先前已获得的能力被遗忘。
- 严格完成要求途中无错；普通完成允许先错后完成。空闲误动作以实际空闲行动机会为分母。合法 FINISH 行动机会未选 FINISH 与“有合法机会但最终未完成”的成熟请求指标不同，后者也包含错内容 FINISH。
- 被撤出的 family 0 仍可能存在旧合同、历史复习和其他 family 的共享案例。预定撤出 tick 不等于完全没有该规则的信息。
- narrow 的 long_both 超出训练时长支撑；wide 的 long_both 在支撑内。宽窄覆盖还改变事件密度，固定 tick 不保证相同实际请求数。
- 功能试验用的是下一真实训练块，且可能重用辅助历史，不是独立留出集。混合损失改善也不保证历史分量改善；报告接受率时必须同时给条件分母。
- `functional_fixed` 在真实控制/辅助学习和保存边界保持 `B=C=0`，plain 与 coupled 属于同一受约束前向函数族；休眠跨 bank 梯度仍以不同形式进入联合裁剪与 Adam。固定组分差不能归因于实际跨 bank 非线性，fixed/task 的差中之差只能称完整参数化 × 结构学习策略交互。数学与 CPU 机械证据见[固定场边界核验](../../docs/evidence/organization-2026-10-30/fixed-field-boundary.md)。
- 比较 functional 与 passive 会改变资格规则、候选验证和额外计算；比较不同 functional 策略也可能改变最终边数。行为、容量、出生退出、跨 bank 桥及额外计算须联合报告。
- 跨 bank 边和容量增长本身不能证明形成了功能模块、一般长期记忆或智能突破。四种规则仍共享同一请求协议，本研究没有验证任意层级组合或跨现实领域学习。
- 冻结分叉评价完整状态策略的延续，不能独立分离权重记忆、状态内容与既有动作头的读取。三个训练种子始终是独立重复的数量。
