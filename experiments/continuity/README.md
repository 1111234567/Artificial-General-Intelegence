# 世界与内部状态跨训练块连续保留

本实验检验：在同一个持续环境里保留内部状态，是否改善跨清空边界的请求、长等待和分布改变后的学习。沿用[上一轮动力学实验](../dynamics/README.md)的简单模型，不增加预设记忆槽或神经调节机制。三种子、33 组训练与 315 次冻结分叉已完成，结果及限制见[完整报告](../../docs/continuity-experiment-2026-10-27.md)，正式协议见 [main-study-v1.json](../../docs/evidence/continuity-2026-10-27/main-study-v1.json)；本说明记录实现与复现方法。

每个 actor 有独立的世界、事件生成器和隐藏向量，64 个 actor 共享模型参数。每个世界有四条并发订单流。命令只出现一次，输入为版本、四选一内容、时长、完成窗口及一次动作回执，共 96 维；模型对每条订单选择 WAIT、BEGIN 或 FINISH(版本、内容)。输入没有待办表、READY、进度或合法动作掩码。外部到达不随动作改变，执行状态和回执取决于实际动作。

连续模型仍为 `dx/dt = -x + W tanh(x)`，观察脉冲到达时 `x += B u`，使用 Heun 积分。冻结循环只冻结 W，输入编码与读出继续训练；GRU 收到同一观察和真实 elapsed time。同宽不等于同参数或同计算量，实际数量保存在运行报告中。

每个 actor 的世界和隐藏向量只在开始时初始化一次。64 tick 的 TBPTT 边界只 `detach`，保留数值；世界时钟、未完成请求、版本和回执均继续。`reset128` 对照每 512 tick 只清空隐藏向量，世界不变。梯度截断仍限制跨时段的信用分配，保留状态不代表梯度覆盖整个生命期。

## 正式训练与学习预算

A 阶段为三个种子 0/1/2、七个配置，共 21 次训练。每次 batch=64、四条订单、TBPTT=64、2,048 次更新，每个 actor 连续接收 131,072 tick；观察间隔固定为 0.25，学习率 0.001，WAIT 损失权重 0.1。使用最终检查点。

| 配置 | 状态宽度 | 循环 | 执行世界动作 | 隐藏状态清空 |
|---|---:|---|---|---|
| `c64` / `c128` / `c256` | 64 / 128 / 256 | 学习 | 教师 | 无 |
| `reset128` | 128 | 学习 | 教师 | 每 512 tick |
| `reservoir128` | 128 | 冻结 W | 教师 | 无 |
| `gru128` | 128 | GRU | 教师 | 无 |
| `online128` | 128 | 学习 | 模型自身 | 无 |

教师分支执行当前世界的因果教师动作，模型预测只用于损失。`online128` 执行自己的 argmax 动作，下一步收到自己的回执；监督标签来自该分支当时的世界，不复制教师轨迹标签。两者都使用 Adam 和教师交叉熵监督，不能称为仅凭奖励学习。教师立即 BEGIN，在合法完成区间中点附近 FINISH。

B 阶段从每个种子 `online128` 的完整 A 检查点分出四个副本，保留参数、Adam、隐藏状态、世界、随机数状态和 A 数据缓存。未来 NEW 的时长分布由 1–4 改为 4–8；已有请求和预定替换保留原合同。每个分支继续 1,024 个实际运行块，每块 64 tick，全程执行自身动作。

| B 分支 | 每个运行块的参数更新 |
|---|---|
| `frozen` | 0 次；继续演化隐藏状态与世界 |
| `current` | 当前块 1 次 |
| `current_repeat` | 当前块 1 次，再处理同一块 1 次 |
| `replay` | 当前块 1 次，再处理过去 A 数据 1 次 |

缓存用均匀 reservoir 抽样保留最多 32 个连续、不重叠的 A 数据块，每块 128 tick，含真实观察、当时监督标签、elapsed time 和历史 `start_h`。B 与评估数据不加入缓存。回放从历史 `start_h` 出发，用当前参数无梯度运行前 64 tick，再对后 64 tick 计算损失；历史状态可能已与当前参数不一致，burn-in 不保证消除这种差异。

重复和回放使用单独的训练临时状态，结果不覆盖正在运行的隐藏向量，不推进实际世界。`current_repeat` 与 `replay` 都有两次梯度更新、相同监督 tick 数，但回放额外运行 64 tick burn-in，前向计算量并不相同。回放还使用显式数据存储，不能把收益全部归于内部状态。

## 冻结副本评估

评估复制完整检查点中的世界和隐藏状态，继续原绝对时钟与待呈现回执，冻结权重，每个条件运行 2,048 tick，dt=0.25。副本不覆盖继续学习的主体。A 测五个条件，B 测 `iid`、`long_wait`、`longer`。

| 条件 | 未来 NEW 的分布 |
|---|---|
| `iid` | 时长 1–4，普通到达与冲突概率 |
| `long_wait` | 时长 4–8 |
| `longer` | 时长 8–16 |
| `sparse` | 到达间隔 4–12 |
| `conflict` | 到达间隔 0.25–0.75，取消概率 0.30、替换概率 0.55 |

主指标只计算分叉后、由评估分布 NEW 产生、截至结束已走完截止窗口的最终有效请求，包括它们的替换。原有请求记为 `inherited`；分叉后才发出、但仍履行旧分布合同的预定替换记为 `carryover_contract`；未走完窗口的请求记为 `censored`。已经完成但窗口尚未结束的请求也暂不进入主分母。严格完成率要求请求完成前没有错误尝试，FINISH 精确率和各类错误另报。

IID 另做每 512 个绝对 tick 清空隐藏状态的配对干预，保留同一世界和回执。跨边界请求按 `发令时刻 < 清空时刻 < 发令时刻 + 声明时长` 选取，不依赖模型是否已经 BEGIN。评估核对外部事件及请求身份指纹，避免模型各自选择不同人群。

`--content-probe` 另外执行教师动作，在教师 FINISH 时刻只比较现有八个 FINISH 内容/版本 logits。它排除了 WAIT/BEGIN 竞争，仍同时受到编码、保持和读出的影响，不是自主计时成功率，也不能证明其他隐藏信息不存在。

这些评估保留参数与内部状态共同形成的行为；A/B 前后差值不是单独的权重知识测量。不同条件从同一历史和随机状态分叉，改变时长后未来到达时间也会改变；后续 B 训练可能经历 A 评估副本看过的同一外部前缀。学习比较需要同时看 B 的冻结分支，不能把所有记录当作互相独立的新世界。

## 运行、暂停与恢复

从仓库根目录运行以下 PowerShell 命令；示例使用新的复现目录。CLI 的默认训练时钟为混合间隔，默认 B 长度为 512 块，正式复现必须保留下面的 `--train-dts .25` 与 `--adapt-updates 1024`。

```powershell
# A：七个配置 × 三个种子。
.\.venv\Scripts\python.exe -m experiments.continuity.run --seeds 0 1 2 --variants c64 c128 c256 reset128 reservoir128 gru128 online128 --updates 2048 --batch 64 --tbptt 64 --train-dts .25 --device cuda --output data/continuity/reproduce-A

# B：每个 online128 最终检查点分出四个学习条件。
0..2 | ForEach-Object {
    $continuitySeedTag = '{0:D3}' -f $_
    .\.venv\Scripts\python.exe -m experiments.continuity.run --adapt-from "data/continuity/reproduce-A/seed_$continuitySeedTag/online128/checkpoint.pt" --adapt-modes frozen current current_repeat replay --adapt-updates 1024 --adapt-durations 4 8 --device cuda --output data/continuity/reproduce-B
}

# 冻结评估 A 与 B；每个 JSON 可以保存多个检查点的记录。
$continuityACheckpoints = @(Get-ChildItem -Path 'data/continuity/reproduce-A/seed_*/*/checkpoint.pt' | Select-Object -ExpandProperty FullName)
.\.venv\Scripts\python.exe -m experiments.continuity.evaluate --checkpoint $continuityACheckpoints --conditions iid long_wait longer sparse conflict --steps 2048 --dt .25 --device cuda --content-probe --output data/continuity/reproduce-eval-A.json
$continuityBCheckpoints = @(Get-ChildItem -Path 'data/continuity/reproduce-B/seed_*/B_*/checkpoint.pt' | Select-Object -ExpandProperty FullName)
.\.venv\Scripts\python.exe -m experiments.continuity.evaluate --checkpoint $continuityBCheckpoints --conditions iid long_wait longer --steps 2048 --dt .25 --device cuda --content-probe --output data/continuity/reproduce-eval-B.json

# 核对协议、分母与配对指纹，按种子汇总。
.\.venv\Scripts\python.exe -m experiments.continuity.report --root data/continuity/reproduce-A --evaluations data/continuity/reproduce-eval-A.json --adapt-root data/continuity/reproduce-B --adapt-evaluations data/continuity/reproduce-eval-B.json --manifest docs/evidence/continuity-2026-10-27/main-study-v1.json --output data/continuity/reproduce-summary
```

运行目录保存 `checkpoint.pt`、`report.json`、`training.jsonl` 及源码副本。汇总输出 `summary.json`、`audit.json`、`summary.md`、CSV 和 PNG/SVG 图；只汇总 A 时省略两个 adapt 参数。缺少运行时可用 `--allow-incomplete` 生成明确标记为未完成的报告，完整性错误仍会使命令失败。训练 `report.json` 中 `world_metrics` 是整个生命期累计量，不能直接当作 B 阶段完成率。

需要计划暂停时，可在首次训练命令追加 `--checkpoint-every 256 --stop-after-updates 256`。完整检查点只在一个运行块及其全部优化结束后保存，含模型、Adam、隐藏状态、世界/调度器/回执、缓存和全部随机数状态。恢复时继续该状态，而非只加载权重：

```powershell
.\.venv\Scripts\python.exe -m experiments.continuity.run --resume data/continuity/reproduce-A/seed_000/c128/checkpoint.pt --updates 2048 --device cuda
.\.venv\Scripts\python.exe -m experiments.continuity.run --resume data/continuity/reproduce-B/seed_000/B_replay/checkpoint.pt --adapt-updates 1024 --device cuda
```

`--updates` / `--adapt-updates` 是相应阶段的总目标，不是恢复后追加的数量。停止进程期间模拟时间也停止；恢复没有冒充停止期间仍在思考或接收现实事件。

程序接口为 `Trainer(Variant 或配置名, TrainConfig)`、`train_update()`、`save_checkpoint()`、`Trainer.from_checkpoint()` 和 `begin_adaptation()`；`semantic_state()` 与 `state_digest()` 用于核对续接内容。`evaluate_checkpoint()` 返回只读分叉记录，`evaluate_fork()` 可对已有完整运行状态做相同评估。实现分别见 [world.py](world.py)、[model.py](model.py)、[run.py](run.py)、[evaluate.py](evaluate.py) 和 [report.py](report.py)。

本轮已完成的实现检查为 **55 项测试通过**，覆盖连续调度、一次观察、动作因果、跨块状态、完整保存/恢复、回放预算、分母与清空配对。可自行复核：

```powershell
.\.venv\Scripts\python.exe -m pytest experiments/continuity -q
```

本任务仍有明确的队列与计时器解法，采用有限状态宽度和因果教师监督；公开版本只有两个值，不能区分任意久远的历史版本。持续运行与学习表现需由正式结果判断，不能据此推断开放环境中的长期知识积累或一般智能优势。并行运行的耗时包含硬件争用，不能当作独占速度或功耗基准。
