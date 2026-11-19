# 11 月 19 日游戏实验 · 证据与复现入口

[晨间结论](pulse-game-morning-2026-11-19.md) · [按时间写下的决策](pulse-game-night-2026-11-19.md) · [参考个体运行包](../data/pulse-game/morning-20261119/continuous-learning-cpu.zip)

| 问题 | 观察到的结果和适用范围 | 证据 |
|---|---|---|
| 能从自己实际作答的回执学会符号映射吗 | 在原模拟器的 READY、接受、拒绝和严格记分下，用实际 ACK/REJ 学得的地址头超过 91% 严格完成。冻结感觉图加未训练地址约 48%，停写约 25%。用私表评分时才读取正确答案。 | [原符号复测与回放](../data/pulse-game/research-2026-11-19/legacy_grounded-report.md) · [单次回执正反分支](../data/pulse-game/research-2026-11-19/legacy-bridge-receipt-forks.json) |
| 吃过不同东西，能选从未见过的组合吗 | 左右各呈现一份新的四维配比。持续显示时，在交互型偏好上，从实际味道学来的感觉地址超过 80% 的末段最优选择。训练只给单成分且测试不再反馈时，纯交互项在观测中不可识别，选择约 50%。 | [连续食物](../data/pulse-game/research-2026-11-19/mixture-visible-independent-summary.json) · [单拍脉冲](../data/pulse-game/research-2026-11-19/mixture-circuit-v1-external-summary.json) |
| 有哪些学习发生在生内 | 新感觉图、世界、地址和运动搜索共做了 12 次完整出生。测量同一生命从 A 迁移至 B、依据相反回执改判、回 A 的变化。局部与分布式地址共享同一批感觉与运动参数。 | [12 次出生登记](../data/pulse-game/independent-lifelong-births-v1/registered.json) · [出生与世界双重不确定度](../data/pulse-game/morning-20261119/independent-birth-statistics.json) · [局部地址对照](../data/pulse-game/morning-20261119/learning-and-retention.json) |
| 长时间只教 B，错在读还是写 | 切回 A 时，只恢复规则权重在旧 A 方向上的分量即可挽回大部分成绩。只恢复以前的读地址，收益小得多。相同原生感觉编码下，分别存放规则权重与协方差也有效。 | [同一历史的最后选择](../data/pulse-game/research-2026-11-19/A-return-forks) · [原型与分库存储](../data/pulse-game/context-local-20261119/README.md) · [公开情境读出与长课程的二乘二对照](../data/pulse-game/research-2026-11-19/old-habitat-repair-sealed) |
| 稳定很久后，规则突然变了怎么办 | 常规 RLS 的更新逐渐缩小。持续开放当前地址的更新方向，旧经验仍影响更新方向；规律翻转后数次访问即可修正。高噪声稳定世界存在代价。按单次意外、按误差方向持续一致性择时，其表现随条件变化。 | [长期图与统计](../data/pulse-game/research-2026-11-19/lifetime-review-report.md) · [开放起点与只遮 B 关系教学](../data/pulse-game/research-2026-11-19/signed-innovation-sealed) · [仅给步长设下限](../data/pulse-game/research-2026-11-19/update-floor-sealed) |
| 持续运行是否真的保存了经验 | 环境按段生成，脑跨段不断。36 个生命在同一检查点上各走 96,000 请求，另外用冷启动、只保留事件表和只保留规则等干预检查无新教学的表现。 | [分布式地址长跑](../data/pulse-game/native-long-life-v1) · [局部地址长跑](../data/pulse-game/native-long-life-local-v1) · [相同未来的分支](../data/pulse-game/native-long-life-v1/retention-fork) |
| 为什么 20 个事件槽在干扰后会失效 | 大量全新事件会覆盖旧条目。干扰反复出现时，合并相近条目能让旧地址仍在；衰减过快的读权重又可能使其无法影响选择。同前缀下分别恢复旧条目、读权重或高阶规律，可以区分这些原因。 | [同前缀修复](../data/pulse-game/research-2026-11-19/recurring-identical-prefix) · [同容量的读写校准](../data/pulse-game/research-2026-11-19/recurring-read-write-sealed) · [内部复习旧样本和旧规则预测](../data/pulse-game/research-2026-11-19/old-experience-sealed) |
| 是否需要持续放电的复杂小环 | 64 节点延迟网络中，早期递归传播有时有用；一组长等待期间已无放电，用慢活动均值仍能读出线索。局部小环在不同出生图上的作用不稳定。第二隔室的增益调制在串行干扰任务上改善了表现。 | [多种子时延与第二隔室](../data/pulse-game/research-2026-11-19/attractor-dendrite-summary.md) · [匹配边干预与状态互换](../data/pulse-game/research-2026-11-19/attractor-motif-mechanism.md) |
| 已训练图依赖交叉边，是否等于从头训练也有优势 | 六组同预算配对。在长 B 后回 A，有向图 92.3%，各单元独立泄漏 92.4%。训后切边、打乱感觉对齐和从头训练给出不同结论。 | [完整重训与同模型消融](../data/pulse-game/morning-20261119/topology-comparison.json) |
| 能否在没有事件表时学习规则 | 单个方向突触加两组分布式权重，在固定变换的配平试验中达到 85.48% 首答。每生随机旋转时，标量受限，小型元突触矩阵随实际回执数增加而改善。事后对已尝气味做迭代匹配、分配空闲 tick 的收益又随样本数和几何改变。 | [局部 LIF 与旋转压力测试](../data/pulse-game/research-2026-11-19/symmetry-spike-summary.md) · [逐物理 tick 的复习](../data/pulse-game/research-2026-11-19/continuous-replay-sealed) |
| 原始感官没有注明“背景”和“食物”怎么办 | 两个随机顺序的脉冲共用四维感觉口。若长期原型参与熟悉度判断，旧背景回来时更容易被认出。角色决定仍假定每个请求有两次脉冲。没有持久的背景身份、或感官噪声很强时，改善受限。 | [主动分配感觉角色](../data/pulse-game/research-2026-11-19/untyped_revisit/report.md) |
| 会为信息支付实际游戏成本吗 | 支付票和费用才获得完整关系样本。闭环学习的决策头胜过同次数同费用随机时机，对低置信、较长剩余生命更倾向探查。公开规划参照更强。 | [净收益、费用反事实及负例](../data/pulse-game/research-2026-11-19/active-symmetry-final.md) |
| 能先走向线索、再掉头走向食物吗 | 探标游戏中，去掉慢递归有时使高分模型大幅下降。区分公开假设快表、带模仿辅助的神经读写、真实回报训练，以及只能从自己实际预测产生辅助信号的路线。纯回报训练也有卡在固定行为的出生。 | [跨模型同世界测试](../data/pulse-game/research-2026-11-19/scout-common-worlds.json) · [感知和动作监督边界](../data/pulse-game/meta-evidence/audit-boundaries.md) · [归档](../data/pulse-game/scout-20261119/README.md) |
| 未再试入口，能否因终点变化改路线 | 两步世界分别翻转房间转移和终点价值，包含两者抵消的配对。只给短时间的读出再等待时，保持期的循环对部分策略很重要。只优化实际正触果可以从随机起步；实际有符号收益、超时零分的冷启动则可能一直未触及入口。 | [短读出](../data/pulse-game/twostep-20261119/摘要.md) · [隐式路线翻转](../data/pulse-game/twostep-adaptive-20261119/摘要.md) · [严格的实际奖励对照](../data/pulse-game/twostep-actual-reward-20261119/摘要.md) |
| 回执能否跨请求归因 | 按顺序晚到时，暂存实际活动即可重新绑定。乱序时，只按先后明显下降，两维有噪感觉回声可帮助重认。若锚味仍未送达，即使实验者知道对应关系也帮不上当前决定。 | [异步回执与跨游戏](pulse-game-transfer-boundaries-2026-11-19.md) |
| 原有的比较关系适用于新游戏吗 | Harlow 式物体辨别暴露了对“变换匹配”的依赖。提供同物读取再按实际误差选择，并保护未参与解释的旧规则，连续换游戏时能同时改善新学习与旧规则回访。两种候选比较关系由实验者提供。 | [跨游戏完整断言与结果](pulse-game-transfer-boundaries-2026-11-19.md) · [同一生命 32 次来回切换](../data/pulse-game/research-2026-11-19/repeated-game-switches) |
| 同一脑能否在旧规则学习后主动买新信息 | 付费决策读取真实局部规则幅值、协方差与地址熟悉度，牌侧电流接入同一对运动神经元。同次数同费用的购票时机对照，以及固定首探同分的结果均保留。 | [连续主体与净回报](../data/pulse-game/context-active-20261119/README.md) |
| 需要多少精度保存长序列感觉缓存 | 9,216 请求的公开感觉缓存实际以 FP16 或 BF16 存储时字节减半，准备加评分约 1.11 倍。少量真实脉冲和探题结果变化。默认精确路径维持原归约与数值。 | [硬件缓存核验](../data/pulse-game/research-2026-11-19/precision-cache-report.md) |
| 提速改动是否保住真实动作 | CUDA 按生命并行，加上物理计分归约、相同世界复用和 CPU 预取。直接比较原生 NumPy、旧 GPU 与新 GPU 的真实脉冲、状态、首触和评分。 | [实际吞吐](../data/pulse-game/research-2026-11-19/physical-throughput/report.md) · [CPU 包精确复验](../data/pulse-game/morning-20261119/portable-verification.json) |

原型为相近气味建立感觉地址，和原项目中按基因进化增删递归边，是两类不同的结构变化。这一轮主要验证可塑权重、写入位置、旧规则的读取和行动后果。三维导航、开放式语言与任意任务迁移均未成为本轮的评估目标。


