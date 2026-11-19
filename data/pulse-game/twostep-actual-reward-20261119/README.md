# 实际端点回报口径对照

实验日期为 2026-11-19，时区 Asia/Shanghai。此前两个封存包的模型在环境中依旧只看到真实触果后的 ±。优化器将未触果也判为任务失败 −1。本组用相同随机初值、训练生命、主要超参数与预算分离目标口径。

| 外环目标 | 实际正回执 | 实际负回执 | 未触果 |
|---|---:|---:|---:|
| 失败效用基准 | +1 | −1 | −1 |
| 只计实际正回执 | +1 | 0 | 0 |
| 实际有符号回报 | +1 | −1 | 0 |

只计正回执时，负回执仍由实际接触产生并被模型的联结记忆接收。上表前两个目标的期望效用相差正仿射变换。第三个目标会区分真实选错与未行动。追加课程先优化实际正回执，后切换到实际有符号回报。跨回合版本把实际未来有符号回报折现，不向无奖路测填入伪奖励。

在完全不知道当前表时，若两步都均匀随机选择并完成接触，核心任务有 1/4 概率得正回执，逐回合有符号期望为 −1/2，未触果则为 0。这个回报结构让早期的接触探索有代价。模型是否能靠后续实得回报形成探索仍由训练与独立评估检验。

核心价值重估模型从随机权重学习快权向量键、负载和稀疏运动网络，感知实际入口、房间和果实回执，保留原始阶段提示。每个组 128×96 个训练生命，前 12 轮端点较短。统一触觉动态模型取消四个阶段提示，把入口和果实的实际接触身份合并到同一输入，使用可见房间、同一触觉、实得回报和公共回合起点学习。原生动态组每个模型 112×96 生命，课程组每个模型 144×96 生命。所有候选动作由模型产生，训练没有教师动作或手写去目标指令。

确认使用没有训练交集的世界。探针无回报。核心保持/翻转用每生命两个条件。动态转移与价值四格用每生命四个条件。

## 核心任务的同预算结果

同一模型种子 345774、训练世界 2700000–2712287。独立确认 3640000–3640255，探针策略随机种子 801227。每个转移条件有 512 次首入口机会。

| 优化目标 | 确定转移的首入口 | 80/20 的首入口 | 80/20 的实际入口接触 | 80/20 两支都正确 |
|---|---:|---:|---:|---:|
| 失败效用 | 512/512 | 511/512 | 512/512 | 255/256 |
| 实际正回执 | 511/512 | 510/512 | 512/512 | 254/256 |
| 实际有符号回报 | 0/512 | 0/512 | 0/512 | 0/256 |

实际正回执组相对失败效用组的稀有条件均值差为 −0.195 个百分点。基于生命的配对 95% 区间 [−0.586, 0]。实际有符号回报组贪心执行时没有到达入口。额外 64 个探针的原始运动记录中，29 个生命实际发过脉冲，960 个活动 tick 中有 240 个发脉冲，位移最大 0.3225，实际最近入口仍在 0.78 之外。原始配对和运动记录分别为 `reports/core-actual-reward-3640000.json`、`reports/latent-strict-core-motion-3640000.json`。

核心组的正回执→有符号课程与只计正回执在前 48 轮的训练日志逐字段相同，计时字段除外。切换目标并继续至同一预算后，80/20 只剩 63/512 正确，126/512 实际接触入口，0/256 两支都对。相关原始记录在 `reports/latent-positive-signed-prefix.json` 与 `reports/core-credit-curriculum-3640000.json`。

## 无阶段标签和统一触觉的原生动态任务

同一模型种子 477317、训练世界 3300000–3310751，共 112×96 生命。没有早段短端点课程。独立确认 3660000–3660255，探针策略种子 620724。每种转移有 1024 次四格机会。

| 优化目标 | 确定转移正确 | 80/20 正确 | 80/20 四格皆对 |
|---|---:|---:|---:|
| 失败效用 | 992/1024 | 876/1024 | 141/256 |
| 只计实际正回执 | 980/1024 | 855/1024 | 135/256 |
| 实际有符号回报 | 0/1024 | 0/1024 | 0/256 |

正回执组独立 80/20 实际入口接触为 1024/1024。有符号组实际入口接触为 0/1024。其 64 个独立运动轨迹中，21 个发过脉冲，最大位移 0.3208，开放的最近入口为 0.7803。原始记录为 `reports/untyped-actual-reward-3660000.json`、`reports/untyped-strict-cold-motion-3660000.json`。

正回执模型在相同独立样本中，只遮挡无奖路测到达房或后续访值房，80/20 都是 512/1024。用均匀角色注意替代学到的接触角色选择，降至 739/1024。另用独立的 3680000–3680255 四格同体检验，首次探针前的实际脉冲共用，同一对世界的物理轨迹和除到达房与实际 ± 之外的观测逐项相同。随后各自自主发脉冲，确定型为 971/1024，80/20 为 861/1024，140/256 生命四格皆对。文件为 `reports/untyped-positive-controls-3660000.json` 和 `reports/untyped-positive-matched-3680000.json`。

## 一样的训练前缀，改变真实结果的信用时间

统一触觉的三个 112 轮模型，前 64 轮的优化损失、接触成功率与开发输出等全部日志字段逐项相同，计时字段除外。后三组分别继续用实际正回执、改为每回合实际有符号回执、改为折现后续实得有符号回执。回报为零的未触果事件三个组都没有补写人工负奖。折现版本的 γ 为 0.96，无奖路测可以用后来真实触果的结果获得信用。

| 后 48 轮目标 | 80/20 首入口 | 80/20 四格皆对 | 只转移变化时实际入口接触 |
|---|---:|---:|---:|
| 继续实际正回执 | 855/1024 | 135/256 | 256/256 |
| 逐回合实际有符号回报 | 502/1024 | 0/256 | 246/256 |
| 跨回合折现实际有符号回报 | 738/1024 | 63/256 | 255/256 |

独立世界与上述统一触觉原生组相同。前缀校验见 `reports/untyped-positive-signed-prefix.json`、`reports/untyped-positive-signed-lifelong-prefix.json`，最终配对为 `reports/untyped-positive-to-signed-3660000.json`。较长期的真实结果信用在已有运动和探索后部分维持动态重估。逐回合有符号条件的入口接触率较高，表现退化体现在选择和适应。

折现有符号目标相对逐回合有符号目标的配对增益为 23.05 个百分点，以生命做 95% bootstrap 区间为 [20.51, 25.59]。配对原始结果见 `reports/untyped-long-minus-local-signed-3660000.json`。

折现模型在相同独立 80/20 下只遮路测房为 512/1024，只遮访值房为 511/1024，均匀角色注意为 673/1024。在 3680000–3680255 四格同体条件下，确定型为 850/1024，稀有型为 759/1024、77/256 四格皆对。两个模型在探针前共用身体历史，原始探针观察逐项相同。文件为 `reports/untyped-long-credit-controls-3660000.json` 和 `reports/untyped-long-credit-matched-3680000.json`。

## 另一随机初值的统一触觉复现

模型种子 477318 与训练世界 3370000–3383823 在每组固定。预算为 144×96。前 24 轮保持原映射与价值，前 32 轮使用确定转移，前 12 轮缩短端点距离，之后恢复完整原生条件。所有回报方式使用相同课程。独立评估为 3700000–3700255。

| 优化目标 | 确定型首入口 | 80/20 首入口 | 80/20 四格皆对 | 只转移变化时实际入口接触 |
|---|---:|---:|---:|---:|
| 失败效用 | 1013/1024 | 903/1024 | 172/256 | 256/256 |
| 只计实际正回执 | 1022/1024 | 926/1024 | 189/256 | 256/256 |
| 实际有符号逐回合 | 0/1024 | 0/1024 | 0/256 | 0/256 |
| 实际有符号跨回合折现，从随机权重 | 0/1024 | 0/1024 | 0/256 | 0/256 |

在测试的三个独立模型初值 345774、477317、477318 上，逐回合实际有符号冷启动全部到不了入口。477318 另测了跨回合折现的冷启动，也没有到达入口。统一触觉的实际正回执目标在 477317 和 477318 上均产生了可测的双因素重估。课程组原始结果为 `reports/untyped-curriculum-actual-reward-3700000.json`。

## 复现入口

从 `source` 下运行。冻结文件与训练过程见 `models`、`reports` 和 `manifest.json`。

```powershell
python -m pytest experiments/pulse_game/twostep_test.py experiments/pulse_game/twostep_neural_test.py experiments/pulse_game/twostep_rewire_test.py -q
python -m experiments.pulse_game.twostep_latent_reward --train-start 2700000 --model-seed 345774 --lives 96 --rounds 128 --short-rounds 12 --positive-actual --output ../models/reproduction-latent-positive-core.pt
python -m experiments.pulse_game.twostep_latent_reward --train-start 2700000 --model-seed 345774 --lives 96 --rounds 128 --short-rounds 12 --terminal-only --output ../models/reproduction-latent-signed-core.pt
python -m experiments.pulse_game.twostep_credit_eval --model failure_utility ../models/latent-failure-utility.pt --model observed_positive ../models/latent-positive-core.pt --model signed_only ../models/latent-strict-core.pt --seed-start 3640000
python -m experiments.pulse_game.twostep_credit_trace --model ../models/latent-strict-core.pt --seed-start 3640000
python -m experiments.pulse_game.twostep_untyped train --train-start 3300000 --model-seed 477317 --lives 96 --rounds 112 --positive-actual --output ../models/reproduction-untyped-positive-cold.pt
python -m experiments.pulse_game.twostep_untyped train --train-start 3300000 --model-seed 477317 --lives 96 --rounds 112 --terminal-only --output ../models/reproduction-untyped-strict-cold.pt
python -m experiments.pulse_game.twostep_rewire_compare --model failure_utility ../models/untyped-failure-utility.pt untyped --model observed_positive ../models/untyped-positive-cold.pt untyped --model signed_only ../models/untyped-strict-cold.pt untyped --seed-start 3660000
python -m experiments.pulse_game.twostep_untyped confirm --model ../models/untyped-positive-cold.pt --seed-start 3660000 --policy-seed 620724
python -m experiments.pulse_game.twostep_rewire_matched --model ../models/untyped-positive-cold.pt --untyped --seed-start 3680000
python -m experiments.pulse_game.twostep_untyped train --train-start 3300000 --model-seed 477317 --lives 96 --rounds 112 --terminal-only --positive-warm-rounds 64 --future-discount .96 --output ../models/reproduction-untyped-actual-warm-long.pt
python -m experiments.pulse_game.twostep_rewire_compare --model observed_positive ../models/untyped-positive-cold.pt untyped --model positive_then_local_signed ../models/untyped-actual-warm.pt untyped --model positive_then_life_signed ../models/untyped-actual-warm-long.pt untyped --seed-start 3660000
python -m experiments.pulse_game.twostep_untyped train --train-start 3370000 --model-seed 477318 --lives 96 --rounds 144 --stable-rounds 24 --deterministic-rounds 32 --short-rounds 12 --positive-actual --output ../models/reproduction-untyped-positive-curriculum.pt
python -m experiments.pulse_game.twostep_rewire_compare --model failure_utility ../models/untyped-curri-failure-utility.pt untyped --model observed_positive ../models/untyped-positive-curriculum.pt untyped --model signed_only ../models/untyped-strict-curriculum.pt untyped --model signed_lifelong_cold ../models/untyped-strict-long.pt untyped --seed-start 3700000
```
