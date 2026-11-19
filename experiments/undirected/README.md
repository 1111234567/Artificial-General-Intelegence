# 无向连续图、接口与时间信用对照

本目录分别检验共享无向边、接口、时间信用、结构操作和课程。42 个训练分支、1,680 个评估分叉均已完成。[正式结果与未通过的对照预期](../../docs/undirected-results-2026-10-30.md)

## 按轮次找代码

| 研究阶段 | 入口 | 记录 |
|---|---|---|
| 10 月，无向共享边与时间信用 | [世界与训练](run.py)、[正式分支](study.py)、[结果汇总](report.py) | [三种子结果](../../docs/undirected-results-2026-10-30.md) |
| 11 月 16–17 日，从无向出生图到有向图 | [出生与局部更新](evolved_plasticity.py)、[有向训练入口](uncapped_directed_reward_run.py) | 本页下方保留各阶段的参数、运行目录和命令 |
| 11 月 18–19 日，私有符号与可塑规则 | [私有符号世界](uncapped_private_symbol_world.py)、[局部规则训练](local_rule_run.py)、[终代比较](local_final_comparison.py) | [11 月 19 日实验](../../docs/local-rule-experiment-2026-11-19.md) · [终代数值](../../data/undirected/2026-11-19-local-rule-v1/final_comparison.json) |
| 连续游戏中的原符号复测 | [游戏与回放入口](../pulse_game/INDEX.md) | [冻结复测结果](../../data/pulse-game/research-2026-11-19/legacy_grounded-report.md) |

控制台的 Python 服务在 [plasticity_console.py](plasticity_console.py)，浏览器页面在 [plasticity_webui](plasticity_webui/)。查看历史结果可以直接打开对应报告。下列训练与续跑命令会写入各自的运行目录。

## 2026-11-16 进化局部可塑性

新一轮以 80 个持续状态节点、双向共享边权和固定端口为起点。外层运行 32 代，每代评估 1,024 个候选，分为 8 个岛。每个候选遇到 16 个独立环境。外层选择并变异出生图、初始权重和局部更新规则。一次生命内由该规则根据活动迹和公开回执调整权重与拓扑。切换任务时保留全部状态、权重和拓扑。代际只继承出生参数与更新规则。内层用前向计算，外层用遗传选择和变异。

首阶段每次只有一个请求，等待 1–3 tick，环境公开可完成窗口。任务 A、B 的私有映射对所有符号都冲突，再切回原来的 A。严格完成要求实际成功且本请求内此前无误动作。回归 A 的严格率只取每个符号的首次回归探测。适应度同时计入完成、严格完成、旧情境恢复和误动作。各岛还保留偏重实际完成率的亲本。候选用同一组开发环境比较。最终入选候选在独立新环境进行配对评估，分别开启完整更新、仅改权、冻结、扰动可塑性回执。各对照仍从环境接收真实回执做动作。

从仓库根目录启动训练与控制台。重复运行相同训练命令会从已提交批次续跑。日志与配置保存在 `data/undirected/2026-11-16-evolved-plasticity-v1/`，控制台监听 [127.0.0.1:8766](http://127.0.0.1:8766/)。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.evolved_plasticity_run
.venv/Scripts/python.exe -B -u -m experiments.undirected.plasticity_console
```

### 1,024 代分阶段对照

完整更新和仅改权各自进化，每组每代 1,024 个候选，两组共享出生种子和每代的 16 条开发环境轨迹。第 1–64 代固定版本和符号，评估出手时机。第 65–256 代增加随机版本。第 257–640 代使用单情境私有符号映射和持续到窗口的原始提示。第 641–1,024 代恢复单次提示与 A→B→A 冲突。课程序号是固定训练预算，阶段留出分数用于判断各能力达到的程度。

时机覆盖率按有窗口内 FINISH 的请求计。版本和符号准确率分别以每个已尝试请求的第一次窗口内 FINISH 为分母，同时记录以所有请求为分母的正确覆盖率。每阶段结束保存独立环境上的两组冠军、原位更新消融和固定猜测参照。运行目录为 `data/undirected/2026-11-16-plasticity-curriculum-v1/`。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.plasticity_curriculum_run
.venv/Scripts/python.exe -B -u -m experiments.undirected.plasticity_console --run-dir data/undirected/2026-11-16-plasticity-curriculum-v1 --port 8767
```

### 单任务、无边数上限

完整可塑性单组运行 1,024 代，每代 1,024 个候选。每个候选遇到 16 个独立环境。每次生命都用同一套任务，交替访问两套私有映射共八次，随机版本和短等待贯穿全程。状态和当生塑性跨情境延续，代际传递出生基因和更新规则。

80 节点出生时有 640 条边，门控从零起步。各节点对独立积累局部信号，达到正负阈值时分别生长或退出。全部 3,160 个无序节点对都有资格。同代按实际严格完成、两个情境的回访表现和总体完成排序。两情境均完成过严格请求时，较少的全程平均活跃边可以在行为成绩相同的情况下提高排名。固定猜符号是留出参照。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_plasticity_run
.venv/Scripts/python.exe -B -u -m experiments.undirected.plasticity_console --run-dir data/undirected/2026-11-16-plasticity-uncapped-v1 --port 8768
```

已验证的 CUDA Graph 加速入口沿用同一运行目录、批次和断点。切换及配对测速证据存放在运行目录的 `speed-switch-*.json` 和 `speed-benchmark-*.json`。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_acceleration
```

本轮从第 171 代开始用 BF16。续跑使用下面的入口。原 FP32 批次保存在同一运行目录，精度分界记录在 `precision-segments.json`。循环状态、矩阵和局部迹使用 BF16，FP32 主权重和门控继续累加小更新。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_bf16_run
```

### 出生拓扑可遗传增删

新入口把出生拓扑表示为 3,160 个无向节点对的连接开关。创始个体独立抽取每条边，平均约 640 条。亲本突变可直接打开或关闭任意节点对的开关，因此出生时的边数和节点度都能随谱系改变。出生图、初始权重和更新规则是遗传基因。每次生命内的变化在该次评估中积累。

默认每条生命连续交替访问 A、B 共 8 次，每次访问处理 4 个请求。每个请求占 7 个环境步，共 32 个请求和 224 tick。每个候选基因在 16 个独立环境中各评估一次。运行目录为 `data/undirected/2026-11-16-plasticity-variable-birth-v1/`。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_birthmask_run
```

### 延长连续判断时间

长窗口入口使用可变出生拓扑。默认在 NEW 后等待 1–3 tick，随后给 8 tick 的合法完成窗口。WAIT 在窗口内可以多次使用。默认 8 次 A/B 访问、32 个请求共 416 tick。每条生命从出生基因初始化，状态在各次访问之间连续演化。NEW 上的等待提示在可配置范围内归一化到 1–3，环境按实际 tick 决定就绪与过期。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_roomy_run
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_roomy_run --min-wait 8 --max-wait 16 --finish-window 8 --out data/undirected/2026-11-16-plasticity-long-wait-v1
```

默认输出目录为 `data/undirected/2026-11-16-plasticity-roomy-v1/`。不同时间配置应使用不同的 `--out` 目录。时间参数与每条生命的 tick 数写入协议，候选和固定猜测参照使用同一时钟。

### READY 信号与答错成本

NEW 后在 1–8 tick 内随机出现 READY，提交窗口为 16 tick。NEW 只提供请求线索，等待时长由 READY 揭晓。每条生命共 832 tick。每次严格完成计 1 分，每份请求的首次合法 FINISH 若报错版本或符号扣 0.05，最后除以请求数。后续尝试继续产生回执供模型学习。默认每代 1,024 个候选，使用可遗传增删的出生图。

普通留出与训练使用同一时钟。另一组配对留出将 READY 从 1–8 tick 推迟到 20–27 tick，固定请求内容、随机种子和每个请求分配的总时间。单独记录 READY 前出手的请求率、窗口内首次出手正确率和旧情境回访。结果存为 `audits/*-ready-shift.json`。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_ready_signal_run
```

默认输出目录为 `data/undirected/2026-11-16-plasticity-ready-signal-v1/`。罚分可通过 `--wrong-penalty` 指定，不同取值使用独立的 `--out` 目录。

### 四群交叉

NEW 后随机等待 8–12 tick，READY 起允许 4 tick 完成。每条生命处理 32 个请求，共 576 tick。每代 4,096 个候选分到四个群。首次合法答错扣 0.05，从严格完成积分中扣除。

首代从上一轮五个检查点按代次和谱系选取 256 套不同的出生基因并分配到各群，其余 3,840 套新建。从第二代起，每群评估 64 个精英原样重测、416 个局部变异、64 个广幅变异、384 个群内交叉、32 个群间交叉和 64 个新个体。每个交叉子代从供体取八分之一的无向节点对，连接开关及出生权重一起继承。各类交叉名额中的四分之一再随机继承供体的一整行规则，两行分别控制权重更新和结构生删。节点偏置和其余规则来自第一亲本。候选计划记录两个亲本的身份、谱系和供体规则行。

```powershell
.venv/Scripts/python.exe -B -u -m experiments.undirected.uncapped_island_crossover_run
```

默认输出目录为 `data/undirected/2026-11-16-plasticity-island-crossover-v1/`。

## 1. 真正共享的无向边

全部 80 个持续状态节点位于同一张潜在图中，无预分 bank、层、记忆槽或计时节点。场为

```text
dh/dt = -h + W tanh(h)
```

每个无序 pair `{u,v}` 只有一个共享坐标 `s`。无向组中 `W[u,v]=W[v,u]=s/√2`。解绑定控制增加独立坐标 `a`，令 `W[u,v]=(s+a)/√2`、`W[v,u]=(s-a)/√2`；出生时 `a=0`。这里 `W[i,j]` 表示节点 `j` 对节点 `i` 的作用。两种参数化共用 pair 资格；控制组比较方向权重解绑定，没有额外开放独立方向的生灭决策。

80 节点有 3,160 个潜在 pair；出生为连通 degree-16 regular 图，含 640 个资格 pair、1,280 个物理非零矩阵条目。每个 pair 的两个方向同时出生或退出。零对角是本实验的配置，不是无向图的定义。

出生使用私有 RNG：随机重标号 ring 起点加保持度数的 double-edge switches，检查连通。该有限采样过程不宣称均匀采样全部 regular 图。一次全局缩放使 `||W||F = 0.6√80`，平均每行平方范数为 `0.36`；这不把谱范数或谱半径限制为 `0.6`。训练后也不进行谱投影。

共享/斜对称坐标满足 `||W||F²=||s||²+||a||²`，共享梯度为 `(G[u,v]+G[v,u])/√2`。相同坐标学习率具有明确的几何意义，但 Adam 不保证共享与解绑定优化轨迹等价。`||W-Wᵀ||F=2||a||₂`；记录中的 `antisymmetric_frobenius_norm` 指这个差值的范数。

`W` 对称不意味着状态 Jacobian 对称：`J=-I+W diag(sech²(h))`。对固定对称 `W` 的无外源连续流，经典 [Hopfield 能量](https://authors.library.caltech.edu/records/rhj8z-8dj90)单调下降，因此这个具体场没有非平凡的持续自主极限环。观察 pulse、学习改变权重和有限步数值解不具有同一保证。这是模型边界，不是一般无向系统的能力结论。

## 2. 两种接口，同一个完整出生函数

`UndirectedModel`（`interface=core`）的内部状态图共享无向边，输入编码器和读出头可训练。`PortGraphModel`（`interface=ports`）没有可训练编码器、读出权重或 bias：输入注入节点 `0..55`，输出直接读取节点 `56..77` 的原始状态作为 22 个 logits，其余两个节点没有预设功能。

端口节点持续演化并参与任意 pair；输出节点可以反向影响输入节点及其他状态。端口不钳制、不重置，也不构成单向输出层。端口位置表示固定外部接口，不是自发功能分组。

共同输入为两个 job 的 56 个公开特征。每 job 的 duration 特征索引 9 除以 16、window 索引 10 除以 2.5，其余保持原单位。固定 pulse gain 为 `0.25`。积分先自主演化，再注入一次 pulse；Heun 的最大子步为 `0.25`。没有把 oracle 标签输入控制器。

Core 出生时把编码器设为同一 `0.25` 输入 selector，把线性读出头设为同一输出 selector、bias 设为零。相同图、权重、状态和接口尺度，使 core/ports × shared/untied 四组的完整出生状态与任意多步输出逐位相同；之后自由参数不同：

| 接口 / 参数化 | 循环自由参数 | 总自由参数 | 动态信用参数 | 局部读出参数 |
|---|---:|---:|---:|---:|
| Ports / shared | 3,160 | 3,160 | 3,160 | 0 |
| Ports / untied | 6,320 | 6,320 | 6,320 | 0 |
| Core / shared | 3,160 | 9,422 | 7,640 | 1,782 |
| Core / untied | 6,320 | 12,582 | 10,800 | 1,782 |

所有潜在坐标都参与梯度、裁剪和 Adam，休眠值在真实前向边界投影为零，矩和信用轨迹不清零。资格数、数值非零数和自由参数数分开统计。当前实现使用 dense `80×80` 场乘法，每个 Heun 子步计算两次场；稀疏 support 不等于节省对应的实际计算。

## 3. 信用越过 128 tick 的方式

实时世界每 tick 仅推进一次，动作由完整十一类 logits 的 argmax 产生，实时前向在 `no_grad` 下执行。信用引擎重建真实记录的参数轨迹，检查原始状态；它不重新生成闭环经验。

| 方法 | 正式实时信用范围 | 说明 |
|---|---|---|
| `tbptt128` | 128 tick | 一个更新块 |
| `recorded512` | 最多 512 tick | 四个真实参数块，早期只有已存在的历史 |
| `rtrl` | 全部已实现的历史 | 完整 `state × dynamic-parameter` 敏感度，按 128 tick 批处理 |
| `rtrl_reset` | 128 tick | 同样计算完整 Jacobian 与乘法，但前一敏感度置零的成本对照 |

`recorded1024` 用于机械校验和 API；它不属于本次 14 条正式变体。

导数约定是在每个真实记录的参数使用处施加同一个加性扰动。已实现的优化器更新、结构决策、动作、观察、标签和世界作为常量；不对它们求导。这不是把当前参数反复用于整段历史，也不是对完整训练算法求导。RTRL 使用完整敏感度，不采用局部或对角近似；128 tick 是批处理边界，不是其信用截断边界。

每次更新还计算等权辅助梯度：历史块先做 detached warmup128，再对 suffix128 求导；启动阶段用当前块。最终梯度为实时/历史各 `0.5`，共同裁剪后执行一次 Adam。辅助学习和结构探测不替换实时 h、不推进世界。历史辅助的信用上限仍为 128 tick。

[1152-tick 机械校验](../../docs/evidence/undirected-2026-10-30/credit-path-check.json)仅在 tick 0 给 cue，监督末段 `[1024,1152)`，期间手动改变参数八次。它检验记录轨迹导数、跨边界信用和有限差分，不是行为学习成绩。独立审计再用两个固定随机方向的 JVP 校准保存的 RTRL 敏感度投影；两个投影不构成全体 S 元素或全部梯度的证明。

## 4. 生长与渐退

`PairOperator` 每八次更新累积 raw shared 梯度均值。生长按 `-gmean*Adam_delta` 从休眠 pair 排名，渐退按 `-theta*gmean` 从有资格 pair 排名，各至多四个。完整排名向量、参数和提案保留用于离线复算。pending 在下一个真实更新块执行。

`fixed` 没有额外探测。`fixed_probe/grow/retire/both` 使用同样候选算法和四次真实损失测量：基点、生长组、全删消融、有限减弱。接受开关分别为无、仅生长、仅减弱、二者。全删只作诊断并回滚；有限减弱为半幅，只有 `|theta|<=lr` 时才试置零。生长与减弱各自要求混合损失严格下降，不固定总边数，也不强制等量替换。

渐退基点包含已接受的生长。混合目标改善允许历史分量变差；接受一批有限减弱也不证明整批可以删除。允许资格变化与普通数值跨零分别记录；新旧边可以并存，有限减弱后的权重仍可由普通学习恢复。自适应结构仅用于 shared 组。

## 5. 正式预算与对照

三个独立训练 seed 为 `0/1/2`，共 14 变体、42 分支。全部 CPU、float64、每进程单线程、batch8、两个 job、80 状态、degree16、出生 gain0.6、更新块128、Adam lr0.001、梯度 clip1、WAIT 权重0.1。

主训练 196,608 tick，之后 family 0 撤出 32,768 tick，再只返回 family 0 共 8,192 tick；总计 237,568 tick、1,856 次更新。warmup 课程前 65,536 tick 使用 narrow 时间范围，之后改 wide；wide 对照从出生起使用 wide。改变只影响未来 NEW，不改变已排定后果、不重置 h/世界/信用。

| 维度 | 声明对照 |
|---|---|
| 接口 × 权重共享 | Core/Ports × shared/untied，固定图、recorded512、warmup |
| 时间信用 | Ports/shared/fixed/warmup：128、512、RTRL、reset |
| RTRL 方向解绑定 | Ports/untied/fixed/warmup |
| 结构操作 | Ports/shared/RTRL/warmup：fixed_probe、grow、retire、both |
| 时间课程 | Ports/shared/fixed：128、RTRL 各有 wide 对照 |

六个完整 checkpoint 为 `0, 65536, 131072, 196608, 229376, 237568`。五个非零 checkpoint 各评估四种 family × `short/long_both` × 2,048 tick，成功分支计划 40 个冻结分叉。分叉、actor 和 checkpoint 不增加独立 seed 数。新任务的状态数、job 数、接口、课程与旧组织实验不同，不能直接按分数给新旧架构排序。

数值失败保留原 seed、最后合法状态和失败记录，不换 seed、不把缺失评价填零、不裁剪或重置敏感度以继续。执行故障与科学数值失败分开处理。完成审计与分母检查前不作效果结论。

## 6. 精确 API

```python
from experiments.undirected.model import UndirectedModel, PortGraphModel
from experiments.undirected.plasticity import PairConfig, PairOperator

model = PortGraphModel().double()  # fixed ports, shared weights
mask, birth = model.initialize_birth(mask_seed=0, weight_seed=0, degree=16, gain=.6)
operator = PairOperator(model, mask, PairConfig(mode="both"))
h = model.initial_state(batch_size=8)
logits, next_h = model(observation, h, dt=.25)
```

两类模型提供 `vector_field/flow/step/forward/recurrent_matrix`、`initial_state/config/parameter_report`、`dynamic_parameter_names`、`normalized_observation/read_state`。Core 另外提供 `initialize_port_matched_interfaces_()`。`forward(obs,h,dt)` 支持无 batch 或带 batch 的输入及 `functional_call/jacrev/vmap`，世界 float32 观察会转换为参数 dtype。

资格由 operator 单独持有。真实前向前调用 `project_and_validate()`；Adam 后调用 `after_optimizer(optimizer, raw_gradients, clip_scale, probe=closure)`。closure 无参数，返回有限 `live/history/mixed` 标量且 `mixed=.5*(live+history)`。`PairOperator.from_state(model,state)` 恢复完整 pending、窗口和事件，不清矩、不重建出生图。

记录轨迹用 `TemporalCreditEngine(model,CreditConfig(...)).gradients(chunk,preupdate_parameters)` 提供梯度。checkpoint、私有 RNG、世界、replay、完整信用和实时连续性由 [run.py](run.py) 管理；完整只读参数轨迹由 [ledger.py](ledger.py) 保存。正式启动入口与源绑定见 [study.py](study.py) 及 [冻结协议](../../docs/evidence/undirected-2026-10-30/protocol-v1.json)。不要为分析修改执行源码或旧实验。

## 7. 审计数据到图

`plot.py` 不读取 checkpoint、raw 分数或训练目录，也不导入报告/训练模块。它只消费审计输出 `report.json`、SHA 绑定的 CSV、全 80 节点 `graphs.json` 和独立 JVP 审计 JSON。报告必须 `audit.status=passed`；有已完成实时块的 RTRL/reset 分支必须完成独立两投影校准。窗口方法没有保存全 S，不要求独立 S 校准；若另提供 primal/history 审计，则明确记录该检查范围。零个已完成块的分支保留不适用标记，不声称通过 S 校准。

行为率先在每个 seed 内等权平均四种 family；四个分母均须有定义。随后只有三个声明 seed 全成功且有定义时才给均值和样本 SD。失败保留 `NF` 标记，分母未定义保留 `U` 标记，不显示假零或两个 seed 的均值。严格/普通完成率分母为成熟有效的分叉后请求；空闲错误使用实际空闲机会。报告保留各 family 的计数和分母，平均率不改成合并请求计数后的率。

```powershell
# 两项审计就绪后先检查数据契约；此命令不生成图片。
& F:\agi\.venv\Scripts\python.exe -m experiments.undirected.plot validate --report <audited-report.json>

# 完成审计后才执行；默认展示主训练末和最终端点。
& F:\agi\.venv\Scripts\python.exe -m experiments.undirected.plot render --report <audited-report.json> --output <new-figures-directory>
```

预备图包括固定随机圆布局的出生图、实际 `W` 与 `W-Wᵀ`、实际参数数、接口/信用/结构/课程行为对照和计算成本。圆布局只为显示连接，不声称功能聚集。各图输出 PNG、SVG 和输入数据 SHA 清单。1152-tick 信用图单独注明“机械验证，无行为学习结果”。

## 8. 运行和查结果

从仓库根目录运行，使用 `.venv/Scripts/python.exe`。当前启动状态和恢复记录以[运行记录](../../docs/evidence/undirected-2026-10-30/run-status.md)为准。三个入口各有一个职责：

| 入口 | 做什么 | 何时使用 |
|---|---|---|
| `python -m experiments.undirected.queue` | 执行冻结的训练、评估和基线命令清单，核对已有终态，从检查点恢复 | 同一输出目录只运行一个队列；本轮基线清单为空 |
| `python -m experiments.undirected.finish` | 绑定正在运行的队列，调度独立历史校验，待全部完成后生成报告、图和中文结果表 | Windows 后台收尾；不启动训练。不要对同一分析目录另外再启动 `analysis_batch --watch` |
| `python -m experiments.undirected.results` | 从已通过审计的结果表生成同种子差值、实际生长/退出数、撤出及返回指标 | 仅重建中文结果表时使用，参数为 `--report` 与 `--output` |

最终状态为 `complete=true`。所有预定分支、评估、审计及产物均已核对。固定图与回滚探测固定图在种子 0 的逐位一致性检查记录为 `failed`。[最终状态及产物路径](../../data/undirected/completion-v4/finish-status.json)

分析 v2 对有限窗口方法每个已完成块仅做一次状态前向核对，不构造它们从未保存的历史敏感度。RTRL/reset 分支仍保留两方向投影校准。详见[本轮方法与限制](../../docs/undirected-experiment-2026-10-30.md)。
