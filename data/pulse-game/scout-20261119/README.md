# Scout 实验复现包

实验始于 2026-11-19，时区为 Asia/Shanghai。冻结时使用 Windows、Python 3.15.2、NumPy 2.5.4 和 PyTorch 2.15.0+cpu。`source` 中包含环境、评估器和训练代码的副本。`manifest.json` 记录种子和文件摘要。

## 手写运动原语

`scout_agent.py` 使用手写双通道脉冲原语，从公开反馈更新 cue 条件偏好与探查动作价值。更换或屏蔽传感后，网络仍沿自身实际轨迹接收触标和触端回执。

训练生命为 470000–470127，开发生命为 490000–490127，独立确认生命为 914000–914255。独立确认中，自适应策略全程成功率为 93.646%，末 16 回合为 100%。全探策略为 93.860%，平均每回合多用 1.063 个运动 tick。只用回报估计探查价值的分支达到 91.968%，同频率固定脚本为 90.057%。

## 直接神经动作

`scout_neural.py` 将 24 个公开观测、4 个公开结果更新的快权概率和 2 个随机探索输入映射到左右运动脉冲。循环群体有 64 个单元，每个单元有 12 条掩码连接。行为克隆和两轮 DAgger 在训练时使用公开历史教师动作。执行时从模型直接读出左、右或静默，没有去标或折返原语。

克隆和 DAgger 生命为 780000–780255、805000–805255 与 807000–807255。开发生命为 853000–853127。独立确认生命为 906000–906255，策略随机种子更换为 615200。端点距中心扰动为 ±0.22，单阵风速度扰动为 ±0.12，第 32 回合起互换左右目标身份。

冻结示范模型在独立确认中成功率为 93.610%，末 16 回合为 99.976%，必须折返子集的末 16 回合正确折返率为 100%。同一神经状态下只翻转 A/B 探标脉冲，1024 次可配对事件全部改发另一侧脉冲。关闭和异生命定时重放传感分别为 71.887% 和 71.570%，关闭和错配端点奖励分别为 49.500% 和 49.213%。六轮独立行动回报 PPO 微调后全程为 93.591%，未见可辨认收益。

身体时点 sham 促使假信号仍在本体实际触标或触端的时点到达。独立供体生命为 918000–918255。只替换背景 A/B 后成功率为 71.381%。只替换端点奖励后成功率为 49.622%，展示的正奖励比例为 93.652%。两个信号均替换后成功率为 49.896%。

## 随机初始化与真实回报

两组随机初始化各使用 56 轮、每轮 128 个生命，前 8 轮为无扰动场地。样本中只用真实端点结果标记成功，错误和超时均记为失败。两组各有 100% 到端点，并在公开回报驱动的快权下使用私有偏好。独立确认生命为 921000–921255，随机输入种子为 625590。A 组成功率为 75.513%，末段为 77.881%，必须折返子集全程为 10.331%。B 组分别为 72.992%、74.585% 和 0.515%。A 组换成无传感后为 73.431%，可看到局部传感利用。两组探标率仍接近 50%，指向顺路穿标和少量折返。

全敏感课程先训练 112 轮，每个 cue 均受 A/B 背景影响，再恢复每生恰两种敏感 cue 64 轮。用于后续训练的 warm 模型从 A 组继续，独立确认 924000–924255 中成功率为 83.215%，末段为 86.987%。同一快权和神经状态下只移动标位，879/2046 次敏感回合的首个运动脉冲随标换边，2050 次不敏感回合中没有观察到这种改变。

从该模型继续训练 112 轮，网络对左右镜像与任意 A/B 标签交换共享参数。对四个等价输入运行同一稀疏循环网络，按物理运动通道对齐再平均动作 logits。每一步仍只有端点回报提供训练信号。训练时身体从中心开始，侦察信息仍由实际穿标获得。独立确认 930000–930255 中，该纯回报模型的成功率为 **93.549%**，末 16 回合为 99.878%，晚段必须折返子集的正确折返率为 99.804%，全程端点接触率为 100%。敏感回合有 1240/2051 次首拍随标位反事实换边，不敏感回合有 0/2045 次。1016 次应折返且自然脉冲已正确反向的探标事件，在只翻转 A/B 输入后均改变了运动方向。

纯回报模型的传感关闭和异生命重放分别为 72.552% 和 72.229%，奖励关闭和异生命错配分别为 49.707% 和 48.956%。同身体时点的独立供体替换传感为 72.235%，换奖励为 48.956%，后者仍显示 93.732% 的正奖励。第二批独立生命 937000–937255 同时增加端点扰动到 ±0.28、阵风扰动到 ±0.122，并在中途移动标位，成功率为 93.823%。与标准场逐回合物理镜像的配对，端点身份、端点时点和探标时点逐项一致。

第二个随机初始化 285105 由课程模型继续训练，在 934000–934255 上复现了 93.774% 全程成功率、99.927% 末段成功率和 99.796% 晚段必须折返率。仅增加物理左右对称的初始分支也达到 93.652% 全程和 99.902% 末段成功率。从随机权重直接训练共享对称网络并提供过渡身体姿态的对照取得 60.938% 成功率，必须折返率为 0%。仅使用过渡身体姿态课程的独立冷启动为 59.723%，warm 继续训练为 83.746%。训练阶段和初值影响能否到达高分，相关阴性轨迹均保存在 `reports`。

另一批独立生命 944000–944255 中，纯回报网络的冻结完整模型为 93.87%。在保留快权前提下，切断稀疏循环边后为 76.43%，每 tick 清除连续隐状态后为 73.20%。三项末段必须折返率分别为 99.80%、13.63% 和 0%。

946000–946255 的配对确认比较了同一 warm 检查点、相同训练场景和 112×96 生命预算。物理与 A/B 共享达到 93.835%，关掉共享为 83.008%，只共享物理镜像为 93.628%。另一个对照保留共享和个体可见的真实奖励，只把外环 PPO 目标在独立生命间置换，成功率为 72.412%，且晚段必须折返率为 0%。真实外环奖励的配对差为 +21.42 个百分点，生命层级 bootstrap 95% 区间为 [20.78, 22.11]。对同批示范模型成功率差为 −0.049 个百分点，区间为 [−0.378, +0.299]，平均少用 0.128 个运动 tick。

独立冷启动对照继续获得 187×96 个原生训练生命，累计 34,848 个，与主模型的 34,816 个接近。最终对照在同一 946000–946255 批次的成功率为 83.557%，末段必须折返率为 50.681%。与主模型配对差为 10.28 个百分点，95% 区间 [9.75, 10.82]。这是两个训练流程的比较，冷启动路径的早期姿态课程和主模型的早期原生运动经验也不同。

## 未见隐藏表

额外训练一份独立神经模型，训练与开发只用 A 行四个二元好目标的偶校验模式，确认只用奇校验模式。两个划分在完整 A 行上严格不重合，单 cue 的二元身份仍为公平分布。种子选择只约束环境取样，网络输入和反馈保持原样。独立确认的 256 个奇校验生命达到 93.860% 成功率，末 16 回合为 99.951%，晚段必须折返子集为 100%。关闭传感为 72.125%，错配反馈为 50.513%。当次选中的精确生命种子记录在检查点 metadata 与评估 JSON。

从随机权重重新训练的纯回报流程同样约束所有训练和开发生命为偶校验设置。最终检查点在独立的 256 个奇校验生命上达到 **93.811%**，末 16 回合为 99.927%，晚段必须折返子集为 99.899%。992 次该类物理探标中，991 次在真实背景脉冲到达当拍发出正确反向脉冲，保持同状态翻转背景标签后这些动作全部改变。关闭传感为 73.499%，异生命重放为 72.644%，错配反馈为 50.024%。这一流程的快权更新规则仍使用公开上下文、接触身份和结果的既定含义。

## 运行

以下命令从 `source` 目录执行。新训练产物可写入新路径。

```powershell
python -m pytest experiments/pulse_game/test_world.py experiments/pulse_game/scout_neural_test.py -q
python -m experiments.pulse_game.scout_search --recalibrate --seed-start 914000 --lives 256
python -m experiments.pulse_game.scout_neural_train eval --model ../models/neural-bc-dagger.pt --seed-start 906000 --lives 256 --policy-seed 615200
python -m experiments.pulse_game.scout_neural_train eval --model ../models/neural-ppo-from-dagger.pt --seed-start 906000 --lives 256 --policy-seed 615200
python -m experiments.pulse_game.scout_neural_sham --model ../models/neural-bc-dagger.pt --subject-start 906000 --peer-start 918000 --lives 256
python -m experiments.pulse_game.scout_neural_train train --output ../models/reproduced-bc.pt
python -m experiments.pulse_game.scout_neural_reward --input ../models/reproduced-bc.pt --output ../models/reproduced-ppo.pt
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1100000 --rounds 56 --temperature 1.0 --lr 0.001 --output ../models/reproduced-scratch-a.pt
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285104 --train-start 1150000 --rounds 56 --temperature 1.25 --lr 0.0007 --output ../models/reproduced-scratch-b.pt
python -m experiments.pulse_game.scout_neural_train eval --model ../models/scratch-a.pt --seed-start 921000 --lives 256 --policy-seed 625590
python -m experiments.pulse_game.scout_neural_train eval --model ../models/scratch-b.pt --seed-start 921000 --lives 256 --policy-seed 625590
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1200000 --rounds 176 --lives 96 --nominal-rounds 0 --all-sensitive-rounds 112 --temperature 0.9 --warm-start ../models/scratch-a.pt --output ../models/reproduced-curriculum.pt
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1430000 --rounds 112 --lives 96 --nominal-rounds 0 --symmetry physical_context --lr 0.0006 --warm-start ../models/reproduced-curriculum.pt --output ../models/reproduced-pure-symmetric.pt
python -m experiments.pulse_game.scout_neural_train eval --model ../models/pure-reward-sym-warm.pt --seed-start 930000 --lives 256 --policy-seed 625596
python -m experiments.pulse_game.scout_neural_sham --model ../models/pure-reward-sym-warm.pt --subject-start 930000 --peer-start 931000 --lives 256 --policy-seed 625596 --peer-policy-seed 625597
python -m experiments.pulse_game.scout_neural_transfer --model ../models/pure-reward-sym-warm.pt --seed-start 937000
python -m experiments.pulse_game.scout_neural_dynamics --model ../models/pure-reward-sym-warm.pt --seed-start 944000
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1430000 --rounds 112 --lives 96 --nominal-rounds 0 --symmetry physical_context --lr 0.0006 --outer-sham --warm-start ../models/curriculum-warm.pt --output ../models/reproduced-outer-sham.pt
python -m experiments.pulse_game.scout_neural_compare --model physical_context ../models/pure-reward-sym-warm.pt --model no_symmetry ../models/no-sym-ref.pt --model physical ../models/mirror-only.pt --model outer_sham ../models/sym-outer-sham.pt --seed-start 946000
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285109 --train-start 1660000 --rounds 187 --lives 96 --nominal-rounds 0 --temperature 1.05 --lr 0.0008 --warm-start ../models/sym-cold.pt --output ../models/reproduced-cold-budget-matched.pt
python -m experiments.pulse_game.scout_neural_compare --model pure ../models/pure-reward-sym-warm.pt --model comparable_budget_cold ../models/cold-budget-matched.pt --seed-start 946000
python -m experiments.pulse_game.scout_neural_train eval --model ../models/sym-warm-b.pt --seed-start 934000 --lives 256 --policy-seed 625600
python -m experiments.pulse_game.scout_neural_train eval --model ../models/mirror-only.pt --seed-start 934000 --lives 256 --policy-seed 625600
python -m experiments.pulse_game.scout_neural_partition train --output ../models/reproduced-disjoint-tables.pt
python -m experiments.pulse_game.scout_neural_partition confirm --model ../models/neural-disjoint-tables.pt --seed-start 1590000 --lives 256 --policy-seed 631450
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1700000 --rounds 56 --map-parity 0 --output ../models/reproduced-pure-parity-base.pt
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1800000 --rounds 176 --lives 96 --nominal-rounds 0 --all-sensitive-rounds 112 --temperature 0.9 --warm-start ../models/reproduced-pure-parity-base.pt --output ../models/reproduced-pure-parity-curriculum.pt
python -m experiments.pulse_game.scout_neural_scratch --model-seed 285103 --train-start 1900000 --rounds 112 --lives 96 --nominal-rounds 0 --symmetry physical_context --lr 0.0006 --warm-start ../models/reproduced-pure-parity-curriculum.pt --output ../models/reproduced-pure-parity-final.pt
python -m experiments.pulse_game.scout_neural_partition confirm --model ../models/pure-parity-final.pt --seed-start 2010000 --lives 256 --policy-seed 631451
```
