# 两步物理选择与价值重估

实验日期为 2026-11-19，时区为 Asia/Shanghai。环境、冻结权重、实际种子、训练和评估记录均随包保存。初始训练使用 Windows 和 PyTorch CPU。

## 实验规则

每个生命有随机的入口身份到房间的映射。在确定型条件，入口必到一间房。在 80/20 条件，每次接触入口后有 20% 概率到另一间房。入口和房内果实分别使用同样的左右神经脉冲、位置和惯性运动，第二次真实接触完成才发出回报。每回合两个运动阶段各有 16 个观测时点，转场时身体回中心，控制器的内部状态与快权继续保存。

32 个普通回合之后，安排一组没有入口阶段的房内探访，随机顺序均衡覆盖房间和果实配对。被开放的果实仍要靠运动实际触及。配对的保持与翻转两个分支共享同一世界、布局、速度扰动和已获得的训练历史。探访回报指出房间价值是否换边。随后只看第一次从中心实际选择的入口，两个分支的这次探针均不显示回报。合格的配对变化需要保持分支选对旧入口，同时翻转分支选对新入口。

## 明示有向图参照与直接神经动作

连续图参照从实际入口回执与随后可见房间学习转移，从房果接触和回报更新局部价值。果实、房间和入口群体双向传递连续活动，竞争入口间有侧抑制。运动原语在该参照中手写。256 个独立生种 2150000–2150255 上，确定型和 80/20 均为 100% 配对入口正确。缓存入口价值、访视时遮挡房身份或独立生命同接触时点重放结果，三种控制均为 50%，没有定向正确换入口的配对。

第二分支训练稀疏循环网络直接输出左、右或静默。输入为原始公开观察、公开事件得到的转移与房果快权读数，加一个每回合随机输入。行为克隆与 DAgger 使用上一分支的公开历史动作，运行于独立世界时直接读出网络运动。256 个独立生命 2360000–2360255 中，物理/房名等变模型在确定型为 99.805%，80/20 为 98.047%。包括对象名字等变的模型在两种条件均为 99.023%。两种模型在遮挡房间、错配访视回报和冻结起手缓存时约为 50%。

在 80/20 生命的首入口，仅倒换已学转移或房价值读数使准确率降到约 1%–2%，清除转移约为 50%。同时倒换使输出恢复。对于带房名等变的模型，同时倒换是架构对齐检查。未加入该先验的早期独立模型也显示同方向的双倒换复原。

## 稀有转移后的实质第二选择

双果变体让两间房偏好不同果实。高价值房触及相应果必得正回报，低价值房触及其不同的偏好果有 25% 概率得正回报。两分支探访期间都会换高值果实的身份，只有翻转组同时改变高价值房。这保证访视时遮住房间仍无从仅靠果实与结果的边际规律辨识是否换了入口。引入的稀有转移使正确入口也能落到低值房，第二动作必须从到达房重新选择。

连续图参照在独立 2320000–2320255 中正确处理了 61 次低值房到达，共有 256 次正确选果。保留同一路线而使用全局果实身份的控制在这 61 次均选错。直接神经模型在另一批 2380000–2380255 独立生命中，两分支第一入口和实际到达房的果实选择率均为 100%。每分支有 57 次低值房到达，均选了该房偏好的不同果。只在最终房替换显示的房身份时，512 次第二动作全部换果，第一入口保持逐生相同。

双果直接神经模型在独立 2610000–2610255 的首次探针两边换位后仍为 100%。更强几何扰动 ±0.30 与阵风 ±0.16 下为 99.805%。改为 35% 稀有转移后，学习到的最佳首入口为 91.016%，到达房的最优果选择仍为 100%。标准场和它的物理镜像中，身体选择身份、接触 tick、到达房和回报逐项相同。

## 只用行为回报训练

两次随机初始化通过真实终端回报训练稀疏运动网络。其一在前 12 轮用较短端点距离帮助接触，随后恢复原生几何。第二次全程原生。两者只在训练时让末回合实际触果后也能得到回报。独立的确定型探针两者均为 100%。同一批 2650000–2650255 的 80/20 无奖探针，第一组在配对保持与翻转试验中的入口准确率为 99.805%，第二组为 100%。对第一组同初值、同回合预算和同训练种子的外环对照，把 PPO 目标在独立生命间置换，首探针为 50%，且没有定向正确换入口的配对。

双果网络也从随机权重只用原生终端回报训练，在独立 2810000–2810255 的 80/20 条件中达到 99.805% 配对入口准确率。两支共 109 次低值房到达全部选了该房偏好的果。只替换到达房的显示身份，则两个分支的条件果实选择率均降至 0%。

在保持和翻转的同体配对控制中，探访阶段每对共享事前随机指定一侧控制器产生的运动脉冲，两边各自读取实际接触回报。记录逐项校验两个身体和所有非回报观测相同，首探针恢复各自自主运动。在 2770000–2770255 纯回报模型和 2780000–2780255 双果模型中，确定型及稀有型的配对首入口均为 100%。

常开快权条件中，保留转移和房果记忆而在首入口执行期切循环边、清除跨 tick 活动或只清初始活动，入口准确率均保持 100%。冻结访视价值写入降至 50%。两步强策略还通过了探访与首探针间 2048 个无输入 tick 的等待。这个条件显示可塑快权及学习到的动作读出足以完成重估。

## 学到的快权地址和负载

另一个稀疏循环分支从公共入口回执、房间身份、果实触及与结果绑定学习向量地址与负载。没有给出转移概率或房果值表。公开接触事件决定两组快权的写入时点，模型从过去整生的事件反向传播学习编码，动作端持续发出真实脉冲。使用公开历史教师和公共回执辅助目标的模型，在独立 2740000–2740255 上得到确定型 99.414%、稀有型 98.438% 的首入口准确率。只用动作模仿损失和键分离正则的双果模型，在 2750000–2750255 的稀有型首入口为 91.211%。两支低值房果实选择率分别为 87.04% 和 86.67%。

联合学习键值和运动网络的纯回报模型从随机权重开始，前 12 轮使用短端点课程，之后恢复原场，实际端点结果通过采样动作进入 PPO。它在独立的 2940000–2940255 上确定型首入口为 100%，80/20 为 **99.219%**，稀有型 98.828% 的保持与翻转两支都正确。遮房、同接触时点错配访视回报和冻结访值写入都回到 50%。只换转移槽或只换房果负载分别降至 0.195% 和 0.391%，同时换后恢复为 99.805%。被预先指定的部分包括事件绑定时点、入口关联与房果关联的地址域和匹配几何的模型结构。

第二个随机初值在全程原生几何中只用回报训练，独立 3100000–3100255 的确定型首入口为 99.023%，80/20 为 96.484%。对第一组的冻结记忆做诊断，在另一个生命批次拟合线性读出，再在新批次评估，可从转移槽读出 100% 的常见到达房，从价值槽读出 99.219% 的当前最高价值房果配对。诊断读出在运动模型之外拟合与使用。

## 短读出与循环活动

这一变体保留两步运动和同体重估协议。在每段身体开始运动之前，先运行一个总长 33 tick 的等待窗口。快权读出在其中以单帧脉冲出现，之后沉默 8、16 或 32 tick。持续读出对照保留相同等待时长，从相同的脉冲时点起一直提供快权读出。两份身体此前探访的非回报输入与轨迹逐项相同，最终入口仍为两边各自输出的实际动作。

同一稀疏网络从随机权重出发，混合持续与短读出范例训练。冻结后在独立 3060000–3060255 的 80/20 样本得到以下首入口准确率。状态交换只动第一次实际运动之前的神经活动，环境状态和读出调度保持原样。等幅扰动对照匹配交换引起的每生命 L2 位移与目标范数，并使用正交方向。

| 输入与干预 | 8 tick | 16 tick | 32 tick |
|---|---:|---:|---:|
| 短读出完整 | 99.41% | 99.41% | 90.82% |
| 短读出后清活动 | 50.00% | 50.00% | 50.00% |
| 短读出后仅在静默间隔切有向边 | 50.00% | 50.00% | 50.00% |
| 短读出后交换保持与翻转的活动 | 0.59% | 0.59% | 9.18% |
| 短读出后加交换等幅扰动 | 98.83% | 98.63% | 90.04% |
| 短读出后加清零等幅扰动 | 78.91% | 80.86% | 74.80% |
| 持续读出完整 | 99.61% | 99.02% | 98.63% |
| 持续读出后清活动 | 99.80% | 99.80% | 99.80% |
| 持续读出且仅在静默间隔切有向边 | 99.41% | 99.41% | 99.41% |

两组从随机权重只看短读出范例、采用普通泄漏初值的 28 轮对照，各个延迟均为 50%。它们能实际接触入口和果实。成功训练条件包含对持续与短读出的混合暴露。

另外两组只看短读出范例的冷启动采用可训练的慢漏初值，先学习 0/2/4 tick 间隔，再学习 8/16/32 tick。其独立确认分别为 73.24%/73.05%/68.36% 和 94.14%/90.62%/85.35%。清活动均为 50%，交换活动均逐档反转。第二组只在静默期切边仍分别为 86.33%、79.30%、70.70%。这一阶段从随机权重开始，时间常数初值与训练课程属于实验先验。

使用同一慢漏初值和快慢课程的混训组，短读出为 96.48%/99.02%/96.09%，只切静默期有向边为 92.19%/89.45%/82.03%。该组持续读出完整时各约 99.8%，清活动后也为 50%。它学到的准备状态在输入仍可用时仍有作用。默认时间常数的混训组在相同持续读出干预下保留 99.80%。这些个体的循环边依赖程度和持续输入用法存在实测差异。

## 复现入口

下面的命令从 `source` 目录执行。冻结权重和指标在相邻的 `models` 和 `reports` 文件夹。

```powershell
python -m pytest experiments/pulse_game/twostep_test.py experiments/pulse_game/twostep_neural_test.py -q
python -m experiments.pulse_game.twostep_eval --record-proof
python -m experiments.pulse_game.twostep_distinct
python -m experiments.pulse_game.twostep_neural_train confirm --model ../models/mirror-room.pt --seed-start 2360000 --lives 256 --policy-seed 845324 --record-proof
python -m experiments.pulse_game.twostep_neural_train confirm --model ../models/full.pt --seed-start 2360000 --lives 256 --policy-seed 845324 --record-proof
python -m experiments.pulse_game.twostep_neural_train confirm --model ../models/distinct-neural.pt --seed-start 2380000 --lives 256 --policy-seed 845325 --record-proof
python -m experiments.pulse_game.twostep_neural_train train --output ../models/reproduction-mirror-room.pt --lives 384 --epochs 16 --dagger-epochs 6 --symmetry physical_context --hidden 88 --degree 14 --batch 224 --probe-weight 12
python -m experiments.pulse_game.twostep_neural_train train --output ../models/reproduction-distinct-neural.pt --distinct-fruit --lives 384 --epochs 22 --dagger 0 --symmetry physical_context --hidden 88 --degree 14 --batch 224 --probe-weight 12
python -m experiments.pulse_game.twostep_reward --train-start 2480000 --model-seed 503574 --rounds 128 --short-rounds 0 --lives 96 --lr 0.0011 --extra-probe 4 --output ../models/reproduction-reward-native.pt
python -m experiments.pulse_game.twostep_neural_train confirm --model ../models/reward-native.pt --seed-start 2650000 --policy-seed 427335
python -m experiments.pulse_game.twostep_neural_train confirm --model ../models/distinct-reward.pt --seed-start 2810000 --policy-seed 851097
python -m experiments.pulse_game.twostep_matched --model ../models/reward-cold.pt --seed-start 2770000
python -m experiments.pulse_game.twostep_dynamics --model ../models/reward-cold.pt --seed-start 2670000
python -m experiments.pulse_game.twostep_delay --model ../models/reward-native.pt --seed-start 2800000
python -m experiments.pulse_game.twostep_transfer --model ../models/distinct-neural.pt --seed-start 2610000
python -m experiments.pulse_game.twostep_latent_train train --world-start 2560000 --seed 79489 --lives 384 --epochs 24 --aux 0.15 --output ../models/reproduction-latent-core.pt
python -m experiments.pulse_game.twostep_latent_train confirm --model ../models/latent-core.pt --seed-start 2740000 --policy-seed 487218
python -m experiments.pulse_game.twostep_latent_reward --train-start 2700000 --lives 96 --rounds 128 --short-rounds 12 --model-seed 345774 --output ../models/reproduction-latent-reward.pt
python -m experiments.pulse_game.twostep_latent_train confirm --model ../models/latent-reward.pt --seed-start 2940000 --policy-seed 487220
python -m experiments.pulse_game.twostep_latent_inspect --model ../models/latent-reward.pt --fit-start 3030000 --test-start 3040000 --lives 192
python -m experiments.pulse_game.twostep_flash_control train --world-start 2880000 --seed 938134 --protocol mixed --unbalanced-delay --lives 384 --epochs 28 --output ../models/reproduction-flash-window-mixed.pt
python -m experiments.pulse_game.twostep_flash_control confirm --model ../models/flash-window-mixed.pt --seed-start 3060000
python -m experiments.pulse_game.twostep_flash_control train --world-start 2910000 --seed 938136 --protocol flash --lives 384 --epochs 24 --warm-epochs 10 --quiet-leak-init -3.0 --output ../models/reproduction-flash-retain-b.pt
python -m experiments.pulse_game.twostep_flash_control confirm --model ../models/flash-retain-b.pt --seed-start 3060000
```
