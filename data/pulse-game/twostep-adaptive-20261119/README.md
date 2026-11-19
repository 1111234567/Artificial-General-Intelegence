# 动态转移重学与短读出扩展

实验日期为 2026-11-19，时区 Asia/Shanghai。已封存的常开与短读出证据见相邻 `twostep-20261119` 目录。本目录记录更严格的接触通道条件、转移变化，以及额外的纯回报训练。

**回报口径。** 已封存和本目录模型的公开 ± 脉冲只在实际触果后出现。外环效用是 `2 × 实际正回执 − 1`，未触果按任务失败计为 −1。逐回合优化时，无奖路测从 PPO 计分中排除。独立追加的逐字端点回报条件另存相邻的 `twostep-actual-reward-20261119` 目录，未触果回报为 0。两套目标的实现见 `terminal_returns` 与 `discounted_life_returns`。

## 单次隐式转移变化

先经历 32 个完整两步回合。随后 16 回合可自主实际接触两种入口并观察到达房，但果实关闭且没有回报。转移若要反转，发生在这一阶段内逐生随机的未提示时点。之后 12 次无入口的房内探访揭示果实价值。最后只读第一次从同一起点自主运动并实际接触的入口，首探针不显示奖励。

同一世界的四格条件是两者保持、只改变转移、只改变房果价值、两者都改变。正确首入口的目标签名依次为原入口、另一入口、另一入口、原入口。四格共享种子、目标布局和环境扰动。没有任何一次探访提供新入口的终端成绩。

连续图先验参照从真实入口和房身份更新变化容忍的后验。在 3160000–3160255 独立生命中，确定型四格 1024 次均正确。80/20 下平衡准确率为 92.38%，236/256 个生命四格都正确。把状态表改成长生累计转移计数后为 50.29%。仅遮挡无奖路测中房身份或仅遮挡价值探访中房身份，均为 50%。原始结果为 `reports/rewire-reference-3160000.json`。

直接动作分支学习向量地址、负载与转移写速率。训练中，没奖励的路测回合参与未来记忆计算；在这些回合的动作上跳过 PPO 得分。候选的入口阶段、房阶段、阶段开始、价值探访四个标签通道置零，仍可感知实际房和真实接触。进一步变体把入口和果实的触觉回执合成同一身份通道。哪一条观测写进转移快权由可训练的软选择学出。该变体仍使用公共回合起点、可见房间与终端符号，保有两种可关联内容的架构先验。

## 同预算的纯回报动态重学

自适应写入与累计记忆两个模型使用相同的纯回报母检查点、训练种子和 112×96 个生命。修改的是是否有一个可训练的转移写入速率下界。母模型与后续训练均使用实际接触的终端奖励。独立 3420000–3420255 的确定型中，自适应为 1024/1024，累计为 514/1024。80/20 中，自适应为 926/1024，194/256 生命四格都正确；累计为 541/1024。配对差为 37.60 个百分点，生命层级 bootstrap 95% 区间为 [35.25, 39.84]。同预算随机冷启动、全程同一困难场景的组为 543/1024，四格全对为 0。自适应模型遮挡任一种需要的房间信息均回到 50%。

用另两批未见生命拟合并验证了一个诊断线性读出。隐式反转后 0、2、4、6、8、10 次实际无奖到达之后，解码到新转移的比例依次为 0.52%、7.29%、33.85%、50.00%、69.27%、81.63%，每批的实际机会数记录在 JSON。到首探针前，转移槽解码准确率为 90.625%，最高房果价值槽为 100%。把实际可获得的转移新证据压缩到 2 或 6 次时，80/20 首入口在另外独立样本中为 53.13% 或 74.02%。

统一触觉的严格组从随机权重开始，预算 112×96 个生命，全程使用原生几何和 80/20 转移。入口与果实共用一个接触身份通道，实际接触时的房间和得分仍可观察。纯 PPO 联合学习事件角色、键值、写速率与运动网络。在 3420000–3420255 的确定型独立样本中为 995/1024，227/256 个生命四格皆对。80/20 为 872/1024，138/256 四格皆对。用均匀注意替换学到的角色选择，80/20 降至 768/1024。遮挡任一种房间信息均为 50%。对同一冻结模型的诊断关联权重，实际入口、伴随得分的实际触觉、无入口房访的空连接权重分别平均为 0.524、0.131、0.715。原始机会数为 12288、8192、2821。

从纯回报母模型继续训练的统一触觉分支同预算在同一批独立世界达到确定型 1012/1024，80/20 为 881/1024。该分支的均匀注意干预为 846/1024，角色选择贡献小于冷启动组。两组使用公共回合起点来提交上一回合中的真实事件，房间身份来自实际到达。阶段标签和区分入口果实的专用触觉口都没有进入模型。

统一触觉冷启动检查点再用 96×96 个新原生生命训练，累计 208×96。阶段输入和事件接口保持相同，训练增加了首探针实得端点目标的权重。独立 3420000–3420255 中，80/20 为 935/1024，193/256 生命四格皆对，均匀注意干预为 841/1024，遮任一种房间信息为 512/1024。这个继续训练结果使用了额外预算。

四格也做了共享身体历史的独立控制。3540000–3540255 中，每对条件从习得到路测、访值共享事前随机指定的一个条件的控制器实际产生的脉冲。已逐项核验实际入口和果实的身份、侧、时点，以及除真实到达房与终端回报外的观测轨迹。每对的首探针原始输入逐项相同，之后各自独立发脉冲。阶段标签清零的自适应模型确定型为 1024/1024、80/20 为 941/1024，206/256 四格皆对。统一触觉冷启动模型为 998/1024、898/1024，162/256 四格皆对。

另两个冷启动加入早期课程。前 24 轮保持旧映射和价值，前 32 轮使用确定性转移，前 12 轮缩短端点距离，余下训练恢复原生动态，且全程只以实际选中的终端回报训练。独立 3420000–3420255 中，保留区分接触来源的模型在 80/20 为 938/1024、203/256 四格皆对，统一触觉模型为 908/1024、171/256。统一触觉均匀注意对照为 870/1024。这两组为 144×96 的独立种子复现，前述 112×96 的暖启动与累计对照仍构成同预算对照。

原始价值重估任务另做了严格同预算的阶段提示对照。使用相同随机初始权重种子 345775、同一组 128×96 的全程原生训练世界与优化器设置。清除四个阶段通道的组，从随机开始在独立 3100000–3100255 得到确定型 507/512，与有标签组完全相同。80/20 无奖首入口为 464/512、213/256 两支都对。有标签为 494/512、238/256。对已训练有标签的网络，仅在整个评估期清除为 481/512，仅在最终首测清除为 474/512。这里保留真实入口和果实的各自事件口。直接在训练中清标签的组学习较慢，第 77 轮开发阶段仍是 50%，最终达到 90% 量级。

## 短读出追加复现

第二组从随机权重、混合常开与短读出的行为克隆模型，在独立 3060000–3060255 的 8/16/32 tick 间隔为 100%/100%/82.62%。只切静默期循环边均为 50%，交换活动后为 0%/0%/17.38%，等幅扰动为 99.41%/99.41%/77.93%。继续提供读出时，清活动后各为 99.02%。

只用真实回报训练的母模型，在固定的候选结构中继续用实际终端奖励训练短读出，三档独立准确率均为 99.80%。短读出清活动后各为 50%，交换状态后各为 0.20%。这一组只切静默边仍为 99.61%/99.02%/90.82%，交换等幅噪声则降到 60.16%/53.71%/48.05%。结果主要支持其持续活动记忆，状态扰动特异性弱于行为克隆混训组。另一个从随机权重只看短读出的纯回报组跑完 128 轮后各为 50%。

第三个纯回报实现从随机权重开始，混合持续和短读出，前三阶段只改变脑内等待间隔，身体始终用原生几何。160×64 个生命之后，在独立 3500000–3500255 中，8/16/32 tick 的准确率为 509/512、510/512、511/512。清活动均为 50%，交换保持和翻转脑状态后分别为 3/512、2/512、1/512。只切静默边为 99.41%、98.24%、70.12%，交换等幅扰动为 49.02%、38.28%、32.42%。它实现了纯回报的短期状态保持，最长延迟时同时依赖一部分复发边。一般扰动敏感，等幅对照结果与此前的行为克隆混训组不同。

## 复现入口

从 `source` 目录运行。训练和确认的精确世界种子、代码摘要及模型配置见 `manifest.json`。

```powershell
python -m pytest experiments/pulse_game/twostep_rewire_test.py -q
python -m experiments.pulse_game.twostep_rewire
python -m experiments.pulse_game.twostep_rewire_train train --train-start 3200000 --model-seed 477312 --lives 96 --rounds 112 --warm-start ../models/latent-reward.pt --output ../models/reproduction-rewire-adaptive.pt
python -m experiments.pulse_game.twostep_rewire_train train --train-start 3200000 --model-seed 477312 --lives 96 --rounds 112 --cumulative --warm-start ../models/latent-reward.pt --output ../models/reproduction-rewire-cumulative.pt
python -m experiments.pulse_game.twostep_rewire_compare --model adaptive ../models/rewire-adaptive.pt typed --model cumulative ../models/rewire-cumulative.pt typed --model cold ../models/rewire-cold.pt typed --seed-start 3420000
python -m experiments.pulse_game.twostep_rewire_inspect --model ../models/rewire-adaptive.pt --fit-start 3440000 --test-start 3450000 --lives 192
python -m experiments.pulse_game.twostep_untyped train --train-start 3300000 --model-seed 477317 --lives 96 --rounds 112 --output ../models/reproduction-untyped-cold.pt
python -m experiments.pulse_game.twostep_untyped confirm --model ../models/untyped-cold.pt --seed-start 3420000 --policy-seed 620724
python -m experiments.pulse_game.twostep_rewire_matched --model ../models/untyped-cold.pt --untyped --seed-start 3540000
python -m experiments.pulse_game.twostep_untyped train --train-start 3370000 --model-seed 477318 --lives 96 --rounds 144 --stable-rounds 24 --deterministic-rounds 32 --short-rounds 12 --output ../models/reproduction-untyped-cold-curriculum.pt
python -m experiments.pulse_game.twostep_untyped train --train-start 3520000 --model-seed 477317 --lives 96 --rounds 96 --warm-start ../models/untyped-cold.pt --from-untyped --lr .0006 --probe-weight 12 --output ../models/reproduction-untyped-cold-extended.pt
python -m experiments.pulse_game.twostep_latent_reward --train-start 2960000 --model-seed 345775 --lives 96 --rounds 128 --short-rounds 0 --erase-stage-tags --output ../models/reproduction-latent-no-phase-native.pt
python -m experiments.pulse_game.twostep_phase_ablation --tagged ../models/latent-reward-native.pt --trained-masked ../models/latent-no-phase-native.pt --seed-start 3100000
python -m experiments.pulse_game.twostep_flash_control confirm --model ../models/flash-window-mixed-b.pt --seed-start 3060000
python -m experiments.pulse_game.twostep_flash_reward --train-start 3140000 --model-seed 670483 --lives 64 --rounds 96 --curriculum-rounds 12 --lr .0006 --hold-fraction .35 --warm-start ../models/reward-native.pt --output ../models/reproduction-flash-reward-warm.pt
python -m experiments.pulse_game.twostep_flash_control confirm --model ../models/flash-reward-warm.pt --seed-start 3060000
python -m experiments.pulse_game.twostep_flash_reward --train-start 3120000 --model-seed 670482 --lives 64 --rounds 160 --curriculum-rounds 16 --hold-fraction .5 --output ../models/reproduction-flash-reward-mixed.pt
python -m experiments.pulse_game.twostep_flash_control confirm --model ../models/flash-reward-mixed.pt --seed-start 3500000 --policy-seed 229360
```
