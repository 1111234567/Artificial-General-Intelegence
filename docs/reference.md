# agi — 参考资料与证据边界

> 整理：2026-11-03。本页整合仓库已有的来源核验和本地文献记录；本次没有新增网络检索或论文复现。下文的论文结论沿用所链接报告的核验范围，材料是否在本地与结论是否成立分开说明。
> 项目设计见 [design.md](design.md)，实验见 [experiments.md](experiments.md)，旧推论撤销依据见 [review-2026-10-26.md](review-2026-10-26.md)。更广的旧资料索引保存在[归档参考卷](archived/reference-2026-10-25.md)。

## 按问题阅读

| 想查的问题或旧称 | 对应来源入口 |
|---|---|
| 连续状态为何不要求放弃 BPTT | [持续状态与训练边界](#持续状态与训练边界) |
| LSM / 储备池能保存什么，读出与信用分配有什么关系 | [LSM 与储备池](#lsm-与储备池) → [前向时间信用](#前向时间信用与经验统计) |
| Dreamer、R2I、Odyssey 等是否已经研究过 | [世界模型与持续控制](#世界模型与持续控制)；保留原材料库位置、记录日期和论文/官方页区别 |
| 旧稿的“新动力学”到底是哪一种机制 | [神经元—突触耦合](#神经元与突触的耦合动力学)、[近期动力学与异步计算](#近期动力学与异步计算)、[无向相互作用](#无向相互作用与时间信用)分别阅读 |
| “那篇自然论文”指哪篇 | [Nature 系列论文的名称定位](#nature-系列论文的名称定位)，不要跨论文借用结论 |
| J-space、Hayek、可解释性与主体性是什么关系 | [工作区、可解释性与未定位名称](#工作区可解释性与未定位名称) |
| VISTA、DeepSeek 自反馈与 SkyPile 在哪一轮出现 | [显式记忆与自反馈的来源边界](#显式记忆与自反馈的来源边界) |
| 生长、退火、模块化和无向学习是否已有现成保证 | [循环计算](#循环计算的深度与模块化)、[无向相互作用](#无向相互作用与时间信用)及其链接的原始研究笔记 |

## 来源层次与本地材料

理论初稿、用户提出的想法、外部 AI 评议、论文核验与本地实测是不同来源。前两者的归属限制见[设计中的分层](design.md#愿景建议与证据的层次)。网页 AI 的 LIF/异步材料在[动力学报告](dynamics-experiment-2026-10-26.md#问题从哪里来)作为问题背景；连接增删评议在[因素实验](factors-experiment-2026-10-29.md)中逐项核验，来源由其[协议](evidence/factors-2026-10-29/study-protocol-v1.json)绑定；因素实验后的评议由[终身交错协议](evidence/lifelong-2026-10-29/study-protocol-v1.json)另行绑定。附件在本机仍存在，但不把私人附件路径作为仓库的阅读入口。

除本仓库的报告、证据 JSON 和研究笔记外，旧卷还引用本机 `E:\knowledge-base\`。本次确认其中 `raw/papers/world-models/INDEX.md`、7 篇 PDF 与 3 份 Markdown 记录仍在；`knowledge/ai/concepts/agi-memory-and-learning.md` 及其他主题的 `INDEX.md` 也在。它们是另一个目录中的历史材料，不随本仓库分发。该知识条目已有 2026-10-30 注记，承认项目侧若干推论撤销；其正文仍混有旧推导，不能因被称作“稳定知识”就整体采纳。旧记录中的“未入 RAG 索引”只说明记录当时的状态，本次没有检查索引服务。

## 持续状态与训练边界

**PyTorch 官方 word_language_model 示例**：[main.py](https://github.com/pytorch/examples/blob/main/word_language_model/main.py)。

`repackage_hidden` 对张量调用 `detach()`，训练循环随后将该 `hidden` 继续传给模型。这是“截断反向计算图，同时保留隐状态数值”的直接代码例子。

它反驳“BPTT / TBPTT 必须清空状态”。它不证明保留状态就能长期记忆、持续学习，也不证明 TBPTT 可以学习任意长的依赖。数据流、状态重置和梯度窗口仍需分别声明。

## LSM 与储备池

**Maass、Natschläger 与 Markram（2002），Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations**：[论文 DOI](https://doi.org/10.1162/089976602760407955)。仓库中的实际引用与比较见[信用报告 §2](credit-experiment-2026-10-27.md#2-前向训练不只有共激活)。

LSM（Liquid State Machine，液态状态机）的动态时序表征与可训练读出，为“状态中有信息但现有动作头不会用”提供了背景；它本身不解决每条循环连接的长期任务信用。本地 `readout_only` 冻结的是已经训练的非脉冲循环网络与编码器，仍训练动作头，**不是随机脉冲 LSM 的复现**。[动力学报告](dynamics-experiment-2026-10-26.md)的冻结 reservoir 也仍训练编码器和读出。比较时须说明随机或预训练、冻结哪些部位、反馈及状态重置边界。

## 世界模型与持续控制

[旧参考 §12.12](archived/reference-2026-10-25.md#1212-世界模型与持续控制2026-10-25-补录)是实际存在的整条研究记录，包含世界模型、持续控制、记忆基准与 RNN→SNN；不应因现行实验换了题目而遗失。下表只索引 2026-10-25 的本地材料和当时解释，本次未更新外部发布状态或重新核验基准数字。

| 研究线 | 本地实际材料与可用于提出的问题 | 保留的边界 |
|---|---|---|
| DreamerV3 | `2301.04104_dreamerv3.pdf` 与 `nature-2025_dreamerv3-record.md`；后者是 [Nature 正式版](https://www.nature.com/articles/s41586-025-08744-2)的元数据、方程和局部正文记录，不是全文。已有记录区分递归状态、离散潜变量、重建与 KL 目标。 | Nature PDF 当时未取得，arXiv PDF 与发表记录不能默认为同一文本。世界模型目标如何影响信息保留，是可比较机制，不是本项目必须重建图像的理由。 |
| Dreamer 4 | `2509.24527_dreamer4.pdf`；[旧 §12.12.2](archived/reference-2026-10-25.md#12122-dreamer-4arxiv-250924527-2025-09-29hafner--yan--lillicrap)记录扩散/动力学 transformer、shortcut forcing 与离线代理训练，区别于 V3 的 RSSM。 | 旧条目的第三方 LIBERO 复现只是单独证据，不能据此断言能力“来自规模而非目标”；“代码未发布”是当时记录。长动作序列也不直接证明跨经历积累知识。 |
| R2I / SSM 记忆 | `2403.04253_r2i-memory-tasks.pdf`；[旧 §12.12.8](archived/reference-2026-10-25.md#12128-记忆任务的基准读数dreamerv3-vs-r2i2026-10-25-补录)记录以 SSM 替换 RSSM 递归部件及外部记忆基准。 | “约 30 / 100 步”属于指定任务与评分，不能直接和本地简单延迟标签、物理时间、流长比较，更不是方法的记忆上限。 |
| TD-MPC2 / V-JEPA 2 | `2310.16828_tdmpc2.pdf`、`2506.09985_vjepa2.pdf`；已有材料包含无像素重建的潜空间控制/表征预测，供比较生成压力和控制需要。 | 不重建并不等于没有学习目标；其规划预算、预训练来源与本地任务不同，未在仓库复现。 |
| Odyssey-3 / Genie 3 / Cosmos | `odyssey-2026_odyssey3-page.md`、`deepmind-2025_genie3-blog.md` 和 `2501.03575_cosmos-wfm.pdf`；[旧 §12.12.9](archived/reference-2026-10-25.md#12129-生成式世界模型的记忆形态genie-3--odyssey-3--cosmos)区分持续生成的一致性与代理控制。 | Odyssey 是[官方发布页](https://odyssey.systems/meet-odyssey-3)转录、Genie 是博客，不能当同行评审论文；Odyssey 自报分数与当时未公开机制/权重的限制须随引用保留。不能从几项产品的时长限制推出全领域不存在持续时间轴。 |
| reset-free / continual RL | [旧 §12.12.6](archived/reference-2026-10-25.md#12126-reset-free--continual-rl这条线项目里完全没有)有 Single-Life RL、LiSP、持续 RL 定义及基准等线索。 | 层次不齐；例如 POPGym 明标当时仅标题级阅读。它们提供任务与重置边界的候选，未直接验证本地持续主体。 |

世界模型迫使表征服务于预测或控制，是旧路线的重要动机；它不证明任意“压缩”都会保留任务所需信息，也不能把受控单变量记忆实验一概说成不测记忆。[设计问题表](design.md#实现选择仍然开放)将表征保持、参数学习与行为控制继续分开。

## 显式记忆与自反馈的来源边界

[持续记忆报告的专节](persistent-experiment-2026-10-26.md#与-vista-和-deepseek-的关系)区分上下文管理版与视觉记录版 VISTA，并说明 DeepSeek 官方自反馈发布没有披露足够的 critic、奖励和更新细节。本地实验比较的是透明的真实奖励与自评器，不能称为复现 DeepSeek。SkyPile-150B 在该轮只检查了目录和格式，没有参与训练；它属于文本预训练来源，不直接提供精确行动后果。

## 时间常数与初始化

**Perez-Nieves et al. (2021), Neural heterogeneity promotes robust learning**，Nature Communications 12, 5791：[论文](https://www.nature.com/articles/s41467-021-26022-3)。

论文为每个神经元分别设置膜与突触时间常数，并在相应实验中独立优化这些参数。它是“τ 不可逐单元设计或训练”的反例。该结论不意味着异质性在所有网络、任务或训练条件下都更好。

**Gu et al. (2022), On the Parameterization and Initialization of Diagonal State Space Models**，S4D，NeurIPS 2022：[论文](https://arxiv.org/abs/2206.11893)。

该工作比较对角状态空间模型的参数化与初始化，在所测条件下展示初始化的重要性。它支持研究初始化、频率与衰减参数化的影响；不能推出神经网络一般不能显式设计时间常数，也不能把某个消融的优劣推广到所有记忆系统。

参数中的膜 / 突触时间常数、活动轨迹的衰减时间、任务中可用的信息保持时间是不同对象，需要用对应实验建立联系。

## 在线学习与信用分配

**Bellec et al. (2020), A solution to the learning dilemma for recurrent networks of spiking neurons**，e-prop，Nature Communications 11, 3625：[论文](https://www.nature.com/articles/s41467-020-17236-y)。

原文将梯度表示为资格迹 `e_ji^t` 与神经元特异学习信号 `L_j^t` 的乘积求和，并构造可在线计算的近似。奖励型 e-prop 还结合奖励预测误差。神经元特异反馈是其信用分配的一部分。

因此，e-prop 不能简化成“任意全局标量 × 任意 Hebbian 更新就接近 BPTT”。引用其结果时，必须交代具体资格迹、反馈投影、损失、神经元模型与近似；本仓库的局部规则需要自己的验证。

## 局部关联与持续可塑性

**Oja (1982), Simplified neuron model as a principal component analyzer**：[论文](https://link.springer.com/article/10.1007/BF00275687)。

活动相关加权重归一化提供了一个局部规则的来源；其线性、平稳输入分析不能直接证明非线性循环中的长时内容保持或新旧语义融合。[2026-10-27 整合实验](integration-experiment-2026-10-27.md)因此将活动迹、旧经历重放、标签监督与任务成绩分开测量。

**Dohare et al. (2024), Loss of plasticity in deep continual learning**，Nature 632, 768–774：[论文及 Fig. 1](https://www.nature.com/articles/s41586-024-07711-7#Fig1)、[官方实现](https://github.com/shibhansh/loss-of-plasticity)。

该研究把新任务的可学习性下降与旧任务遗忘分开，并测试持续 BP 中的少量低效用单元替换。它支持单独测量长期可塑性，不证明保留记忆就会自动保持学习能力，也不为任意神经元增删或无梯度共激活规则提供保证。整合实验当时的一次分布转换不足以复现其长期学习问题；后来的交错与撤出实验也未复现该论文的完整长期协议。

## RNN→SNN 转换

**Diehl et al. (2016), Conversion of Artificial Recurrent Neural Networks to Spiking Neural Networks for Low-power Neuromorphic Hardware**：[论文](https://arxiv.org/abs/1601.04187)。

该工作在特定硬件约束下，先用 BPTT 训练 Elman RNN，再离散化并匹配脉冲神经元响应，以问题分类任务展示转换。它提供可研究的先例，不能为任意 tanh RNN→LIF 映射提供等价性保证，也不能把其性能损失当作本项目的通用通过门槛。

本仓库当前原型的结果与限制见 [experiments.md](experiments.md)。

## 神经元与突触的耦合动力学

**Clark & Abbott (2024), Theory of coupled neuronal-synaptic dynamics**，Physical Review X 14, 021001：[论文版本与摘要](https://arxiv.org/abs/2302.08985)。

论文研究具有特定随机连接和 Hebbian / anti-Hebbian 突触动力学的循环模型，分析其相图、时间尺度和不动点。它说明在该模型里，突触动力学可以改变活动与记忆机制。

旧稿把模型特定结论推广为“固定权重网络不可能有多吸引子”“只有结构增删就绝不可能有记忆”，超出了论文范围；这些普遍命题已撤销。将其机制用于本仓库，需要先检查模型、相区和测量定义是否对应。

旧卷还列出 **Wakhloo、Clark 与 Abbott，Associative synaptic plasticity creates dynamic persistent activity**，PNAS 123(40), e2526156123（2026），及 bioRxiv `2025.08.20.671131`。[原记录](archived/reference-2026-10-25.md#神经元突触耦合动力学2026-10-24-补录多吸引子这个词的正主)摘录了刺激撤除后的持续振荡、突触动力学和连接模式解释。它与 PRX 的特定静态/动态相不能混成同一种记忆；本次只恢复这条既有出处，没有补做原文核验，也没有在本地复现。该旧记录随后写成的“两条硬约束”仍按上段撤销/收窄处理。

## 近期动力学与异步计算

“近期”指[2026-10-26 报告](dynamics-experiment-2026-10-26.md#近期研究及采用范围)当时的专项核验，不表示本页重新追踪了最新论文。该表分别记录 Asabuki 与 Clopath 的 Predictive Alignment（PA）、Wagner 等的稳定性与流动、iTDS 输出学习稳定、Adaptive E–I balance、Reservoir associative memory，以及无全局时钟通用计算的理论构造；原始论文与官方代码链接随各条列出。

其中 PA 有明确的可塑循环更新及实际读出反馈 `Qz`，不能替换为目标反馈 `Qf`；官方单输出例子与三比特演示的披露范围也不同。本地动力学实验先测 BPTT 训练循环，不复现 PA；活动稳态、输出学习稳定、内容保持、异步可计算性与持续经验学习各自需要证据。“新动力学”因此是多条候选机制的统称，不能据名称把它们拼成已有验证的架构。

## 前向时间信用与经验统计

[2026-10-27 信用报告](credit-experiment-2026-10-27.md)核对并区分了 [RFLO](https://elifesciences.org/articles/43299)、[e-prop](https://www.nature.com/articles/s41467-020-17236-y)、[SnAp](https://arxiv.org/abs/2006.07232)、[前向模式随机梯度](https://arxiv.org/abs/2202.08587)及 [Forward-Forward](https://arxiv.org/abs/2212.13345)。它们保留的状态、使用的反馈和优化目标不同；活动 EMA 不能直接借用导数资格迹的保证。

[Cascading Eligibility Traces（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e647dad9086b5a4cc136e1d1926cc172-Abstract-Conference.html)研究延迟信用的时间表示。[FORCE / RLS](https://pmc.ncbi.nlm.nih.gov/articles/PMC2756108/)则提供累积相关统计的另一种来源；本项目只验证了固定表征、线性平方损失下的统计等价，没有复现其闭环 FORCE 控制。

## 循环计算的深度与模块化

[Pascanu 等的深循环研究](https://arxiv.org/abs/1312.6026)区分状态转移内部深度、循环状态层数与时间展开。信用实验所用的 128 维稠密循环也存在大量反馈路径，切块命名本身不改变动力学。历史[等参数结构方案](evidence/credit-2026-10-27/topology-design-v1.json)加入浅分解控制，分开非线性深度、秩瓶颈和参数化影响；该方案仍未运行。

[2026-10-28 实际运行的回路干预与零初始化非线性扩展](loops-experiment-2026-10-28.md)核对不同节点的简单环，区分可重复节点的加权闭合游走，并比较状态依赖的连接作用。该本地实验没有复现上面任何论文，也没有据图上的环或 Jacobian 指标宣称功能模块已经形成。

[Driscoll 等（2024）](https://www.nature.com/articles/s41593-024-01668-6)分析多任务 RNN 中复用的动态计算模式；[Béna 与 Goodman（2025）](https://www.nature.com/articles/s41467-024-55188-9)在受控任务中说明结构模块化并不自动产生功能专门化。[Wu 等（2026）](https://www.nature.com/articles/s42256-026-01306-9)进一步研究任务驱动的模块组织，已核对摘要、作者预印本及[官方代码](https://github.com/brain-intelligence-lab/modular-brain-rnn)。这些结果为结构比较提供线索，不能直接推断更多层或更复杂拓扑会解决本项目的长等待失败。

[循环连接渐进增删实测](rewiring-experiment-2026-10-28.md)继续把结构存在与结构可塑性分开：使用当前循环控制器的任务导数产生候选边，保持完整状态并允许独立退出与再生长。它是本地指定规则的对照实验，没有复现某篇自组织论文，也没有验证一般的自由拓扑演化或终身学习。

[结构因素和空闲损失报告](factors-experiment-2026-10-29.md)对用户提供的 AI 评议进行了本地代码与数值核验：候选 Adam 的归一化、新生坐标零矩配合旧全局步号，以及原 WAIT 目标都在实际 PyTorch 2.11 环境检查。公式成立与任务失败的因果归因分别处理；评议提出的混合 A+B 和深转移不能冒称该轮已做，实际对照集中于共同稀疏来源、结构策略与空闲目标。后续任务交错及非线性组织由各自报告说明。

[从随机出生开始的交错、复习与长间隔实验](lifelong-experiment-2026-10-29.md)接续上述本地证据，加入四套有共同与冲突映射的规则、全生涯均匀历史库及匹配计算量的近期重复对照。它没有复现生物 REM、自主重要性选择或某篇特定终身学习论文；冻结测量继承完整状态，因此评价的是系统保持，未单独分离权重存储与当前状态的作用。该轮用户提供的 AI 评议用于提出问题，附件摘要及方法来源由协议绑定。

[非线性组织与功能性增删报告](organization-experiment-2026-10-30.md)和[原始研究核验笔记](evidence/organization-2026-10-30/research-notes.md)补充了 DeepR/soft-DEEP R、RigL、GateON、模块化增长、几何稀疏及不同网络部分的动态训练策略。文献中的采样保证、固定预算、冻结课程与几何偏好分别处理；本地方法是损失复核的候选搜索，没有复现完整退火，也没有依据连接图宣称功能模块形成。

## 无向相互作用与时间信用

[2026-10-30 研究核对](evidence/undirected-2026-10-30/research-notes.md)区分连续 Hopfield 场、欧氏能量梯度流、Equilibrium Propagation、Temporal Contrastive Learning 与 GLE 的实际假设。共享无向边本身不提供任意长时序信用；平衡或准静态学习也不能直接替代持续行动中的监督目标。所选最小对称场在固定权重、无外源连续流条件下有下降能量，该限制不推广到所有无向动力系统。

本地实验采用记录轨迹上的完整 block-RTRL，并与有限窗口、主动清空敏感度作对照。其行为结果和跨块导数的数值校验分别报告。[实验方法](undirected-experiment-2026-10-30.md) · [正式结果](undirected-results-2026-10-30.md)

## Nature 系列论文的名称定位

“自然论文”在已有讨论中不足以唯一识别来源。以下均在本地记录中出现，引用时应写标题/作者，而不是把一篇的结论移给另一篇。

| 若讨论的是 | 对应论文与本页入口 | 它不直接证明什么 |
|---|---|---|
| 时间常数可训练 | Perez-Nieves 等（2021），Neural heterogeneity promotes robust learning，Nature Communications；[时间常数](#时间常数与初始化) | 不证明异质性普遍更好。 |
| 脉冲循环的在线信用 | Bellec 等（2020），A solution to the learning dilemma for recurrent networks of spiking neurons，Nature Communications；[e-prop](#在线学习与信用分配) | 不为任意全局标量乘共激活提供保证。 |
| 长期训练后学不动 | Dohare 等（2024），Loss of plasticity in deep continual learning，Nature；[持续可塑性](#局部关联与持续可塑性) | 不等同旧任务遗忘；本地一次分布切换未复现其长期问题。 |
| 用世界模型跨任务学习 | Hafner 等（2025），Mastering diverse control tasks through world models，Nature；[DreamerV3](#世界模型与持续控制) | 不说明本地长等待任务、永不重置或持续知识积累已经解决。 |
| 学习吸引子与动态保持 | Asabuki 与 Clopath（2025），Predictive Alignment，Nature Communications；[动力学核验表](dynamics-experiment-2026-10-26.md#近期研究及采用范围) | 本地 BPTT 成功不是 PA 复现，公开代码例子也不等于全部演示均可还原。 |
| 模块化、E–I、储备池或连续时间信用 | [模块化来源](#循环计算的深度与模块化)、[动力学表](dynamics-experiment-2026-10-26.md#近期研究及采用范围)、[无向学习核验笔记](evidence/undirected-2026-10-30/research-notes.md)中的具体条目 | 属于不同任务与假设；例如 GLE 的层级网络证明不能直接用于任意无向循环图。 |

Clark / Abbott 的 PRX 与 Wakhloo 等的 PNAS 是另外的期刊来源，不应统称为 Nature 论文。

## 工作区、可解释性与未定位名称

**J-space / J-lens** 在[旧参考 §12.2](archived/reference-2026-10-25.md#122-状态与工作区)与旧设计中出现。本地知识条目 `E:\knowledge-base\knowledge\ai\concepts\agi-memory-and-learning.md` 还有较详细摘要，称来源为 *Verbalizable Representations Form a Global Workspace in Language Models*（Anthropic，2026-07-06），并列出 `github.com/anthropics/jacobian-lens`。但材料库 `raw/papers/memory-and-learning/INDEX.md` 明标该工作未入库；本次没有原论文或代码核验。因此“6–7% 方差”“几十个概念”“决定可报告性”等保留为**旧摘要中的待核验说法**，不作为本项目的容量限制、工作区设计依据或主体性证据。

同一索引另有 `2609.01924_looped-transformers-jlens.pdf`（*Looped Transformers under the Jacobian Lens*）。本次确认索引和该 PDF 均在，没有重新审读 PDF；它与上面的 Anthropic 工作是不同条目。模型能否维持信息、能否由指定读法报告、能否用它控制行动，仍须分别检验。

**Hayek / 哈耶克** 在本次检查的仓库文本、上述知识库的 Markdown/TXT 及论文索引中没有找到可定位记录。现有材料不足以确定指作者、论文还是某个模型；暂记为未定位名称，不据此补造一条理论路线。这不是断言相关文献不存在。

**本地可解释性证据**主要是[独立线性读出](credit-experiment-2026-10-27.md)、[提示到结束的阶段追踪](reports/actions/content-timing-experiment-2026-11-03.md)与[状态交换干预](reports/memory/learn-remember-experiment-2026-11-03.md#一次性命令是否真的靠内部状态保存)。线性可读说明指定特征和划分下能取回信息，线性失败不能排除其他编码；状态交换显示后来选择依赖所带状态，也未区分保存的是原符号还是已算出的内容。图上有回路、边聚集或活动低维，都不能单独证明功能模块、自然语言解释、意识或稳定自我。

## 可训练的关联记忆状态

**MemTensor，MemT-1.0: Towards Memory-Native Evolution**：[论文 v2](https://arxiv.org/html/2610.26825v2)、[官方代码](https://github.com/MemTensor/MemT-1.0)、[2026-11-07 核验笔记](memt-research-2026-11-07.md)。本次核对选定章节和官方实现，未运行模型。在线变化的是局部记忆矩阵，骨干与读写参数保持固定；关联误差修正值得作为前向状态更新的候选，但不等同持续修改共享权重，也未解决本项目的旧规则遗忘。笔记保留了更新/重置边界、状态与权重规模、应用附录的配置不一致，以及尚未执行的最小对照。

上述机制的本地后续见[关联状态实验](associative-state-experiment-2026-11-07.md)：在同一个冻结执行器前测试简单累加/delta 和等状态递归参照，另设共享地址版本；不是 MemT 复现或全部策略参数的持续学习。

## 怎样使用旧索引

[旧参考卷](archived/reference-2026-10-25.md)保留了 SNN、语言模型、世界模型、外部记忆、生物学习与其他工作的线索，其中包含预印本、报道、未充分核验条目和已撤销的解释。它可以帮助寻找原文，不能整体当作现行证据。

继续采用旧条目之前，应核对原始论文或官方代码，记录版本、任务、训练方式、重置边界、对照和适用范围。未找到原文、只读摘要或来自二手报道的条目应保留相应状态。有限检索中没有找到先例，不能写成“研究真空”；一项 benchmark 或某种脉冲模型的结果，也不能定义整个领域的语言能力或计算成本上限。

本页是与当前争议直接相关的核验清单，不是完整综述，也没有据此选定新的架构。
