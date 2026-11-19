# 快权元学习复核记录

下列成绩来自冻结检查点在 NumPy 参考环境中的实际闭环运行。开发集只用于选模型。这里记录的本地复核与另行执行的独立盲评使用不同生命种子。各路线的监督来源和适用范围见[全日审计清单](audit-boundaries.md)。

## 原游戏

| 条件 | 冻结检查点 | 证据 |
|---|---|---|
| 线索持续可见，360 步全由自身动作产生经验 | [模型](../meta-aux-autonomous-seed1127/frozen-visible-autonomous.pt) | [256 生复核](../meta-aux-autonomous-seed1127/audit-visible-autonomous.json) |
| 线索和情境只在 tick 0 出现，480 步全由自身动作产生经验 | [模型](../meta-delay-no-gate-1431-cpu/frozen-delay-480.pt) | [512 生及精确错地址复核](../meta-delay-no-gate-1431-cpu/audit-delay-480-exact-address.json) |
| 从单拍模型分叉，继续训练遇到签名被遮的回执 | [模型](../meta-delay-drop-finetune-1908/frozen-retention-360.pt) | [256 生×96 回合的停奖复核](../meta-delay-drop-finetune-1908/audit-retention-360.json) |
| 独立生成 A/B 私有映射，单拍呈现 | [模型](../meta-delay-independent-deep-2019/frozen-independent-deep.pt) | [256 生及错地址复核](../meta-delay-independent-deep-2019/audit-independent-exact-address.json) |

主单拍模型有 32 个慢活动节点，每节点 6 条有效入边，以及每生命单独维护的 16×8 快权。所有公开观测通道共同进入循环与地址、内容、门的计算。模型在可见线索消失后通过活动和地址痕迹保持当前线索。在真正接触后，快权沿形成的地址接收模型给出的内容修正。每个生命开始时清空内部动态状态。

训练可利用模型已经收到的回执。正回执把被接触身份标为偏好目标，负回执把另一个身份标为偏好目标。历史动作目标只在同一情境和线索先前有可见回执时产生，读出辅助由同一历史确定。单次回执内容辅助用当前真实收到的身份和符号。主单拍训练没有开闸时刻目标和示范动作。独立 A/B 模型额外用公开线索和情境约束地址稳定及分离。评价适配器只使用真实公开观测。原游戏的动作头每拍选左、右一个方向，Scout 另有静默类别。

主单拍模型回访 28,616/28,672，隔至少八道其他题回访 8,345/8,380。遮回执、关闭快读写或用另一线索在本拍算出的地址替换实际写地址，回访均约一半。反转回执后回访 60/28,672。独立 A/B 模型回访 14,317/14,336，关闭快写或移除慢循环都降至 7,275/14,336。

同一单拍结构只用历史动作监督时，开发集最好回访 51.74%，终点运动不稳定。真实回报 REINFORCE 和拟合实际动作回报的从头对照均约 50%，包含接触奖励塑形。见各自训练历史。[动作对照](../meta-delay-action-only-1431/history.jsonl) · [REINFORCE](../meta-delay-reinforce-1431/history.jsonl) · [拟合回报](../meta-delay-reward-value-1431/history.jsonl)

停奖实验保留实接触身份，第 16 回合后只去掉正负符号。在 96 回合中，原始单拍模型回访 69.79%，同时冻结后续快写可恢复 97.72%。缺奖训练所得模型自行保持 98.21%。[同模型冻结快写对照](../meta-delay-no-gate-1431-cpu/audit-delay-480-retention-freeze.json)

## Scout

| 条件 | 冻结检查点 | 实际闭环证据 |
|---|---|---|
| 原始 24 通道，公开老师加自身闭环采集 | [模型](../meta-scout-keyvalue-2233/frozen-scout-main.pt) | [256 生×64 回合](../meta-scout-keyvalue-2233/audit-scout-main.json) |
| 同条件，从头去掉开写时序标签 | [模型](../meta-scout-gate-free-2233/frozen-scout-gate-free-600.pt) | [相同种子的闭环与实际快权 Δ](../meta-scout-gate-free-2233/audit-scout-gate-free-write-times.json) |
| 只在偶校验 A/B 目标表上训练 | [模型](../meta-scout-even-2313/frozen-scout-even-520.pt) | [256 个未选过的奇表生命](../meta-scout-even-2313/audit-unseen-odd.json) |
| 从随机权重全程自己行动，由本生命公开历史构造外层标签 | [模型](../meta-scout-self-observed-2328/frozen-self-preview.pt) | [256 生×64 回合](../meta-scout-self-observed-2328/audit-self-preview.json) |
| 第二种子，公开自标签，只在偶校验图训练 | [模型](../meta-scout-self-even-2338/frozen-self-even-early.pt) | [256 个奇图新生命](../meta-scout-self-even-2338/audit-unseen-odd-self-even.json) |
| 高水平模型继续用实际奖励微调 | [模型](../meta-scout-reward-strong-2292/frozen-reward-strong-240.pt) | [512 生命配对](../meta-scout-reward-strong-2292/paired-reward-strong-240.json) |

Scout 有 56 个慢活动节点，每节点 10 条有效入边，12×16 每生命快权，三类动作头。地址、内容、逐值写增益及左右或静默均为模型计算。主路径先回放公开历史老师自己的实际闭环，再用混合历史，最后在模型自主动作所得的真实历史上接受老师的动作标签。老师只接收与学生相同的公开观测。

训练辅助使用已经出现过的公开情境脉冲、当生命内先前相同线索及情境下的签名回执、同拍回执值和线索地址稳定性。主模型开写辅助使用公开的签名回执事件，去门分支将该项设为零。没有情境脉冲的实接触无法提供情境专属目标标签。两种冻结策略的动作都从 24 个公开通道及自身动态状态产生。

从随机权重的自标签分支没有调用历史老师或使用示范动作。外层运动目标使用自己已观察到的情境和回执。情境尚未观察且两种历史偏好已知且一致时，目标为对应的公开远端方向。仍需信息时，外层以公开探标方向引导探索。情境已知而偏好未知时，若当前轨迹实际触到一个端点，只以该端点的身份训练运动一致性。该运动一致性目标与本次接触正负无关。步 320 在 256 个新生命中整体 93.75%，最后 16 回合为 4,096/4,096，必须折返 1,015/1,015，可直达 1,033/1,033。关写为 49.11%，错写地址为 52.00%。

第二种子在偶校验私图用相同公开自标签训练，冻结步 384 后的奇校验新生命整体 93.73%，末段 4,093/4,096，必须折返 1,054/1,054，关写后整体 48.94%。首个自标签种子的 1,015 个物理探标时刻只换一拍情境会把正确折返方向全部反转，1,033 个不敏感探标时刻则保持原方向。[物理探标脉冲分叉](../meta-scout-self-observed-2328/audit-self-sensor-fork.json)

主模型整体做对 15,273/16,384，末 16 回合做对 4,091/4,096。必须先探再折返的末段问题做对 1,000/1,003。无需侦查且正确终点在探标反侧时，直接到达 1,012/1,013。关快写后整体 49.30%，关慢循环边后仅完成 68/1,003 次末段必须折返。末 16 回合交换真实情境脉冲时，敏感题从 2,070/2,075 降到 1,005/2,075，非敏感题仍 2,021/2,021。无开写时序标签模型的整体成绩为 93.19%，签名回执、情境脉冲和普通运动时都可发生实际快权变化。

Scout 训练每生有两个敏感线索。改变为零至四个敏感线索的测试各有 256 生×96 回合，所有组的末 16 回合均做对 4,096/4,096。只用偶校验表训练的独立模型在奇表留出上整体 93.52%，末 16 为 99.93%。[敏感数和规则改写](../meta-scout-keyvalue-2233/audit-scout-transfer-causal.json)

原本不必探查的线索中途变敏感后，在两臂第一次可见分歧为未探查时的真实负回执、且立即下次同线索出现探标在旧正确目标反侧的 69 生中，模型越标 66/69，原规则对照 0/69。只擦掉这次快写，或只把这笔快写换成由正符号生成的修正，两组都为 0/69。真实慢状态、当前运动和其他回执保持不动。公开老师在其对应的 69 生也没有重启探查。首次重启只做对 29/69，较宽自然分组在随后的两次机会做对 187/219，后来做对 254/267。

实际奖励微调从已训练模型出发。每回合奖励含实际正负回执、接触收益、运动与探标代价。没有老师动作、动作标签或快权语义辅助。两次微调各 240 步，在各自初始模型的相同 512 个新生命上，较早模型 81.84%→84.99%，高水平模型 93.26%→93.67%。高水平模型平均多用 0.148 拍并增加 2.42 个百分点探查。区间以生命配对自举，仅比较固定检查点。[较早模型对照](../meta-scout-reward-2271/paired-reward-240.json)

另从随机权重只用实际奖励学习。步 912 冻结实验在 256 个新生命的整体为 63.78%，错配奖励和去写各约 49%，说明学到依赖自身回报的关联。在此检查点撤掉背景仍为 63.94%，末段只有一个必须折返成功，因此该证据适用于回报驱动的线索学习。原始奖励训练的主动探查与利用情境另在更长预算、全敏感课程和相关探索下检查。[从头实际奖励](../meta-scout-raw-reward-scratch-2341/audit-scratch-preview.json)

同网络、同初始化和同预算分别用 A2C 和额外的端点结果预测训练 1,248 步，追加损失只使用自己实际触到的端点及其正负回执。配对 512 个新生命中，标准 A2C 为 73.17%，加预测损失后为 51.26%。接触覆盖从 99.99% 变为 68.21%，末段必须折返从 2/2,062 变为 160/2,062。预测臂在已观察两种情境回执并实际探标的 1,222 个新敏感事件上，端点价值头排对 934 次，真正当拍运动方向对 646 次。该损失学得了部分关键信息，完整动作能力仍受限。全敏感条件下，探后相关探索能增加真实折返经历，但两份 816 步的冻结贪心策略在 256 个新生命均为 50.11% 正确且无末段折返。[配对预测对照](../meta-scout-tried-outcome-2344/paired-outcome-prediction-1248.json) · [端点值读出](../meta-scout-tried-outcome-2344/offline-outcome-prediction-1248.json) · [全敏感探索对照](../meta-scout-raw-reward-post-noise-2348/paired-exploration-816.json)

另从随机权重训练显式触端列控制，快权的两列各代表一个端点身份，写地址和动作由学习产生。实际碰到某个端点的负回执只向该列写 −1，正回执只向它写 +1，未试的列不获得标签。实际触端有 0.25 的最小写入增益。全敏感的新 256 生中，冻结步 1,440 的正确率 62.36%、接触 100%、无情境 50.66%、关写 50.45%。末段必须折返 547/2,068，真实探标时只换情境脉冲可以将其中 508 个正确方向改成相反方向。普通二敏感世界整体 55.07%，末段必须折返 250/1,047。该控制的写入归属和最低增益都有明示结构。[全敏感冻结复核](../meta-scout-indexed-floor-2357/audit-indexed-1440-all-sensitive.json) · [二敏感迁移](../meta-scout-indexed-floor-2357/audit-indexed-1440-two-sensitive.json)

真实探到背景且两种背景各有历史回执的 8,129 个事件中，触端列模型自身当拍价值排对 8,125 次，神经运动当拍选对 4,854 次。独立 192 生用固定的物理中继执行其已经形成的端点选择时，全敏感为 95.88%，原神经动作 62.23%，中继关写 49.70%。普通二敏感世界的中继为 72.21%。中继仅用于诊断已有价值表征能否推动可达的高分运动。[冻结状态读出](../meta-scout-indexed-floor-2357/offline-indexed-1440.json) · [公开值到动作的干预](../meta-scout-indexed-floor-2357/audit-indexed-value-routing.json)

在同一生命内连续三个全图改写后，每段 64 回合，末 16 可回到 90%–96%。最后一次全图反转后首八题约 29%–32%，显示旧记忆的短期代价。固定规则下的长程停奖会下降，冻结快写可在末段恢复到 99.95%。[连续阶段与停奖](../meta-scout-keyvalue-2233/audit-scout-regimes-freeze.json) · [实际奖励分叉的连续阶段](../meta-scout-reward-strong-2292/audit-reward-regimes.json)

在随机单 cue 背景规则变更的世界用实际回报微调，留出世界三次全图改写后的末 16 回合依次为 99.34%、99.44% 和 99.29%，第三次全翻后的前八回合仍只有 30.47%。另从无门模型继续训练缺奖经历，第 64 回合以后持续遮签名直至 256 回合，末 16 为 99.63%。每个回合开始恢复第 64 回合快权快照并允许轮内更新，末段仍为 99.88%。整拍停写的末段为 90.60%。有轮内更新时，跨回合快权固定也能实现保持。[规则切换微调](../meta-scout-rule-shift-2315/audit-rule-shift-regimes.json) · [缺奖和轮内写入隔离](../meta-scout-dropout-2321/audit-dropout-with-boundary-control.json)

已观察到每种情境各有一次真实签名回执时，无门模型做对 1,380/1,382，必须折返 253/253。只观察到另一情境时，模型倾向沿用已见偏好，这使敏感线索的正确率低。分层基于此候选真实获得的事件。[512 生逐类回执计数](../meta-scout-gate-free-2233/audit-receipt-exposures.json)

自标签模型在端点扰动从 ±0.22 增至 ±0.38 而保留阵风 ±0.12 时整体 93.66%。探标距离从 0.35 移至 0.21 或 0.48 时分别为 93.70% 和 92.94%。探标距离 0.55 且端点扰动 ±0.38 时整体降至 88.46%，末段 94.78%；使用已知私图的同运动参考在该条件为 95.02%。[物理迁移](../meta-scout-self-observed-2328/audit-scout-sensor-geometry.json)

从随机权重开始只按实际物理结果奖励的两个种子，于第 192 步在未知正负符号的压缩场分别有 99.58% 和 98.01% 的机会完成探标后到对端。追加未来物理观测预测的两臂分别为 50.29% 和 48.97%。预测确实降低了下一拍位置和速度误差。压缩场中两个预测臂的 x/v RMSE 分别为 .112/.083 和 .076/.088，复制当前值的比较为 .207/.171 和 .211/.174。原场无符号诊断中，预测臂接触覆盖为 85.65% 和 99.48%，纯物理奖励臂为 65.48% 和 72.02%，后者完成探标后到对端的机会更多。原场按私有规则选择目标的能力要另看真实签名阶段。[2361 压缩场](../meta-scout-physical-predict-2361/paired-body-192-short.json) · [2370 压缩场](../meta-scout-physical-predict-2370/paired-body-192-short.json) · [2361 原场](../meta-scout-physical-predict-2361/paired-body-192-nominal.json) · [2370 原场](../meta-scout-physical-predict-2370/paired-body-192-nominal.json)

恢复原场检查点后，另测只在当回合分配真实签名和以 .85 折扣继续分配本生命后续签名。两边都从同一权重开新优化器并采同样的后续场景。终点新增 720 步的冻结配对里，2361 的正确率为 24.26%→4.18%，接触覆盖 48.35%→8.73%；2370 为 50.29%→35.38%，接触覆盖 99.79%→71.01%。这些终点没有保留第 432 步开发集一度出现的覆盖优势。[2361 收益范围](../meta-scout-life-credit-2361/paired-cross-trial-720.json) · [2370 收益范围](../meta-scout-life-credit-2370/paired-cross-trial-720.json)

物理课程四臂在原场只用真实正负回执继续训练至 1,152 步。独立冻结配对中，2361 的纯回报和未来物理预测两臂都为 50.10%，覆盖完整，末段必须折返均为 0/1,068。预测臂实际下一拍 x/v RMSE 为 .072/.074，复制当前值为 .215/.152。2370 的两臂分别为 8.16% 和 2.88%，接触覆盖为 16.66% 和 5.71%；加预测的准确率差 −5.28 个百分点，按生命配对区间为 −5.78 至 −4.78。物理预训练的动作技能并未在恢复任务签名并继续训练后成为情境条件下的正确折返。[2361 终点](../meta-scout-physical-predict-2361/paired-body-1152.json) · [2370 终点](../meta-scout-physical-predict-2370/paired-body-1152.json)

在可靠的触端列价值模型上进一步冻结表征并只训练神经运动头与 critic。两臂均接收身体真实签名。额外内部进展奖只用实际探标时形成的自身价值差和下一拍真实位移。在新 256 生全敏感世界，两臂分别为 56.69% 和 55.63%，进展奖差 −1.06 个百分点，生命配对区间 −1.52 至 −0.60。进展奖增加 1.21 个百分点探后到另一侧，接触减少 1.61 个百分点。在同一探标状态下只交换两列读数，两个臂各 513 次真实正确方向分别有 346 和 259 次变为反向；换异生读数分别为 236 和 208 次。闭环使用交换数值分别降低总成功率 32.42 和 32.97 个百分点。学到的价值确实能控制已实现的部分运动。[运动训练配对](../meta-scout-motor-value-progress-2381/paired-motor-432.json) · [真实签名臂错值干预](../meta-scout-motor-signed-only-2381/audit-wrong-values-432.json) · [内部进展臂错值干预](../meta-scout-motor-value-progress-2381/audit-wrong-values-432.json)

## 复现入口

```powershell
.\.venv\Scripts\python.exe -m experiments.pulse_game.meta_eval data/pulse-game/meta-delay-no-gate-1431-cpu/frozen-delay-480.pt --output data/pulse-game/meta-evidence/repeat-brief.json --lives 512 --seed 1431600000 --cue-ticks 1
.\.venv\Scripts\python.exe -m experiments.pulse_game.meta_scout_eval data/pulse-game/meta-scout-keyvalue-2233/frozen-scout-main.pt --output data/pulse-game/meta-evidence/repeat-scout.json --lives 256 --seed 4262000000
.\.venv\Scripts\python.exe -m experiments.pulse_game.meta_scout_transfer data/pulse-game/meta-scout-keyvalue-2233/frozen-scout-main.pt --output data/pulse-game/meta-evidence/repeat-scout-switch.json --trials 96
.\.venv\Scripts\python.exe -m pytest -q experiments/pulse_game/test_meta.py experiments/pulse_game/test_meta_scout.py experiments/pulse_game/test_world.py
```

每个训练目录中的 `settings.jsonl` 记录了启动参数，`history.jsonl` 区分训练历史和另行进行的自主策略评估。`last.pt` 含优化器状态，冻结文件的文件名和校验和由相应报告固定。

本组 `.venv` 运行时为 Python 3.14.5、NumPy 2.5.4、Torch 2.11.0+cu130。[源文件校验和](source-checksums.json)
