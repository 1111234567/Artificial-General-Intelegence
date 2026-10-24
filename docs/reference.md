# agi — 参考：文献与领域定位

> **本文件规定** ｜ 用途：依据在哪、这在已有领域里是什么位置
> 写：文献索引（§12，含方法学来源）/ 与 SNN 领域的关系（§15：已有答案的 / 被修正的 / 真空在哪 / 对草案的后果）
> 不写：设计推理（→ `design.md`）、实验设计与结果（→ `experiments.md`）、项目现状（→ `../agi.md`）
> 来源：原设计草案 `design-memory-learning-2026-10-22.md` 的 §12 / §15；2026-10-24 迁入本卷，编号未改
> 维护：读到新的相关工作时追加；被推翻的结论要标出来源，不删

---

## 章节地图（全项目）

| 章 | 卷 |
|---|---|
| 项目定位 / 判据 / 现状 / 下一步 | `../agi.md` |
| `design.md` §0–`design.md` §9、`design.md` §14 | `design.md` |
| `experiments.md` §10、`experiments.md` §11、`experiments.md` §13、`experiments.md` §16 | `experiments.md` |
| **§12、§15** | **本文件** |

> 章节号（§X.Y）在全项目内唯一，三卷共用一套编号。跨文件引用时写出文件名。

---

## 本卷怎么读

| 想知道 | 读 |
|---|---|
| 某个论点有没有出处、出处说了什么 | §12（按主题分组；每条带具体数字与它推翻了什么） |
| 这套东西是不是在重复别人二十年前的工作 | **§15.1 术语对照** → §15.2 已有答案的 |
| 为什么说阶段 B 是真空 | §15.5（三个维度：连续状态 / 训练方式 / 单向）+ §15.6 |
| 语言能力的天花板在哪、是不是「过几年就行」 | §15.4 |

**一条使用提醒：** §12 与 §15 里标「预印本」的是未同行评审的工作；标「报道」的是二手媒体，未读原文。引用时保留这个标注——它是可信度的一部分。

---

## 12. 文献索引

### 12.1 训练与遗忘（AGI 直接相关）

- **Local Support Learning**（LSL，局部门控防遗忘）：arXiv 2610.02126
- **LoPT**（梯度边界后训练）：arXiv 2605.04913
- **Cortex**（上下文 → 可合并参数化记忆，50 次编辑 88.9%/99.8%）：2026-10
- **In-Parameter Memory 综述**（两轴分类：放置 × 获取时间）：arXiv 2610.08630
- **Infinite-Parameter LLMs**（data-to-weights vs in-context 对比）：arXiv 2609.18842
- **Doc-to-LoRA / Text-to-LoRA**（超网络生成更新）：Sakana 2026
- **FF 的代价**（优化地板 + 表征坍塌）：arXiv 2609.33240
- **严格评测（FF / CaFo / MF）**：arXiv 2511.01061
- **COLA**（临界态局部规则 ≈ BPTT）：ICML 2026

### 12.2 状态与工作区

- **J-space**（可言说表征形成全局工作区，6~7% 方差决定可报告性）：Anthropic 2026-07（github.com/anthropics/jacobian-lens）
- **Looped Transformers under the Jacobian Lens**（循环架构的工作区持续性）：arXiv 2609.01924
- **Externalizing the Workspace**（持久自我状态：1.00 / 0.61 / 0.00）：Zenodo 2026-07——测的是承诺在价值压力下的持久性，不是「自我在哪」；`design.md` §2.1 原表的读法已修，见 `design.md` §2.9
- **几何记忆**（模型存事实不只靠查找，还靠全局关系图谱）：ICML 2026

### 12.3 反馈与判据

- **BMC**（双向流形一致性，无监督错误诊断 AUROC 0.889）：ICML 2026，arXiv 2604.16565
- **DiSE**（序列再生成置信度）：ACL 2026，arXiv 2603.02760
- **Spilled Energy**（EBM 视角的幻觉检测）：ICLR 2026，arXiv 2602.18671
- **GRPO is Secretly a PRM**（结果奖励隐含过程信号）：ICML 2026
- **InT**（自提议单步修正）：arXiv 2601.14209
- **First-Token Cueing**（只喂首 token 可纠正 1/5~1/2 的多 token 错）：arXiv 2610.23400

### 12.4 扩散与多 token

- **ME-DLM**（编辑修正）：ICML 2026
- **DAPD**（依赖图独立集）：ICML 2026
- **SDAR**（块扩散 + AR，1.7B~30B）：ACL 2026
- **Eso-LMs**（首次可算精确似然 + KV cache）：ICML 2026
- **并行度与生成顺序评测**（8 MDLM / 58 benchmark）：ACL 2026 Findings

### 12.5 记忆与生物 engram

> ⚠️ 本节两个 engram 同名不同物：**生物的 engram** 是分布式的激活模式（没有地址），**DeepSeek 的 Engram** 是哈希表条目（按 token n-gram 取一行）。不可混用。

- **DeepSeek Engram**（条件记忆，N-gram 哈希查表）：arXiv 2601.07372
- **Memory for Large Language Models**（三维分类综述）：arXiv 2607.25380
- **Mixture of Chapters**（可学习 token bank + chapter 路由）：ICLR 2026
- **Hypermem**（超网络 → FFN 参数更新）：ICML 2026
- **MemT 1.0**（并行记忆通路，288B/36B）：arXiv 2610.26825
- **Titans**（MLP 快权重 + surprise 门控）：arXiv 2501.00663
- **Fast Weight Programmers 综述**：Irie & Gershman, TMLR 2026
- **Distributed Engrammics**（fast-weight 状态可转移）：Zenodo 2026-06
- **Tensor Cache**（外积 L2 范数）：arXiv 2605.22884
- **动态 engram 与记忆辨别**：Nature Neuroscience 2024
- **engram 亚群 F-RAM / N-RAM**：Cell Research 2025
- **STM trace 数天 + 再暴露巩固**：Commun Biol 2022

#### 神经元–突触耦合动力学（2026-10-24 补录——**「多吸引子」这个词的正主**）

> 起因：2026-10-24 场次三里，我把「多吸引子」当成了自制判据（「关输入后活动模式两两余弦 < 0.5」），
> 但没读过它的原始出处。这两篇是同一条线的一前一后，**判据完全不同**。

**① Clark & Abbott, *Theory of Coupled Neuronal-Synaptic Dynamics*, Phys. Rev. X 14, 021001 (2024)**
（arXiv 2302.08985）

模型：速率网络 `(1+∂t)x_i = Σ_j [J_ij + A_ij(t)]·φ(x_j)`，`J ~ N(0, g²/N)` 是冻动的随机骨架，
`(1+p∂t)A_ij = (k/N)·φ(x_i)φ(x_j)` 是**持续进行的 Hebbian 可塑性**（带遗忘）。**突触是动态变量**。

正文原句（本节判断的直接依据）：

> 「For g<1, the trivial neuronal fixed point of the nonplastic network, x(t)=0_N, **is globally stable and the network is quiescent**.」
> 「…if Hebbian plasticity is strong, there exist **stable fixed points** to which finite-size networks settle following transient chaos…」
> 「We compute … their number, showing **exponential growth with network size**.」
> 「In chaotic states with strong Hebbian plasticity, a stable fixed point of neuronal dynamics is **destabilized by synaptic dynamics**, allowing **any neuronal state to be stored as a stable fixed point by halting the plasticity**. This phase of **freezable chaos** offers a new mechanism for working memory.」

**② Wakhloo, Clark & Abbott, *Associative synaptic plasticity creates dynamic persistent activity*,
PNAS 123(40) e2526156123 (2026)**（bioRxiv 2025.08.20.671131）

同一模型的另一面：外部**振荡**刺激（`h_i = I·cos(2πft+θ_i)`，`t<0`）撤除后，网络**自己继续振荡**：

> 「following oscillatory stimulation, neurons continue to oscillate **long after the input is removed**. This creates a dynamic form of memory that **has no explicit storage or retrieval phases** and that requires no prior knowledge of the input.」
> 「We trace the mechanism of these **persistent oscillations** to an interaction between neurons and synapses that creates **complex outlier eigenvalues** of the connectivity matrix.」
> 「…the network anchors the input to **pre-existing modes of the connectivity**.」
> 「even **microscopic changes in synaptic weights** can have dramatic effects on neuronal dynamics.」

**这两篇对项目的两条硬约束：**

| | 含义 |
|---|---|
| **权重可塑性是前提，不是可选** | 固定的网络（无论结构怎么长）在 g<1 时**只有平凡不动点**——全局静息。多不动点 / 持续振荡都只出现在突触是动态变量的相区。**只有结构增删、权重恒定的网络，不可能有多吸引子。** |
| **记忆的形态可能不是静态态** | 两篇各给一种：静态的（freezable chaos，冻结可塑性→当前状态成为不动点）与动态的（persistent oscillations，复数 outlier 特征值→自持振荡）。**判据不同：前者的「模式相似度」，对后者没意义；后者的判据是「活动是否持续」+「频率是否匹配输入」。** |

**与 `experiments.md` §16.11 的关系**：`grow_homeo.py` 在 probe 期权重是常量（无 Hebbian 更新），按此分类属 nonplastic；
实测「关输入 50 步归零」正是「quiescent 相、只有平凡不动点」的形态。**那边说的「多吸引子」用词需按此收窄。**

### 12.6 绑定与容量

- **HeRR**（绑定容量是紧致群的性质，非向量表示）：Zenodo 2026-09（未同行评审）
- **OSC**（正交子空间雕刻，阶数解耦深度）：ICML 2026
- **VSA 容量分析**：JAIR 2026
- **ABCA 相位编码绑定**（swap errors 与 alpha 相位保持）：Commun Biol 2026

### 12.7 睡眠与巩固

- **Language Models Need Sleep**（Knowledge Seeding + Dreaming）：arXiv 2606.03979
- **Do Language Models Need Sleep**（离线循环 + 清空 KV）：arXiv 2605.26099
- **SleepGate**（睡眠三机制 → KV 管理）：arXiv 2603.14517

### 12.8 计算形态与生物对应（2026-10-22 追加）

> 以下为 2026-10-22 检索所得。标注「报道」者为新闻/二手中文报道，未读原文；其余为论文/会议页。

**连接组与全脑仿真**
- **FlyWire 全脑连接组**（Dorkenwald et al., Nature 634, 2024）：成年雌蝇 139,255 神经元 / 约 5000 万突触（v783 数据）；Shiu 的计算模型用了其中约 12.5 万；同期 Nature 九篇
- **Shiu et al.**（Nature 2024）：连接组 + LIF 全脑计算模型；网络本身不训练（权重来自连接组；神递质类型由 ML 预测），运动行为预测 ~95%
- **Eon Systems 具身演示**（2026-03，报道）：连接组 + NeuroMechFly v2 + MuJoCo，15 ms 同步闭环
- **flyGNN**（NeurIPS 2025）：连接组 → 循环消息传递网络 → 全身运动控制
- **MaleCNS v1.0**（Cell 2026-09，报道）：雄蝇中枢神经系统，16.67 万神经元 / 1.25 亿突触
- **DeepSoma**（2026，报道）：树突 / 胞体 / 离子通道级精细神经元全脑平台

**无层拓扑与连续深度**
- **C-DESN**（Continuously-Deep Echo State Network，ECML PKDD 2024）：无层、按距离衰减连接的储备池，深度由单一超参控制
- **储备池 / 液态状态机**（Jaeger 2001; Maass 2002）：固定循环网络 + 线性读出
- **Neural Cellular Automata**（Mordvintsev et al. 2020）：局部规则 + 时间演化，无层

**连续时间与事件驱动**
- **Liquid Neural Networks**（Hasani et al. 2021; Lechner et al. 2020）：ODE 神经元、可学习时间常数
- **CTRNN / Neural ODE**（Chen et al. 2018）
- **DONN**（Deep Oscillatory Neural Network，Sci Rep 2025）：Hopf 振荡器神经元
- **SNN / LIF**（脉冲网络与事件驱动仿真）

**局部学习与临界态**
- **三因子学习**：Izhikevich 2007（多巴胺调制 STDP）；Frémaux & Gerstner 2016（三因子规则综述）；Schultz 1997（多巴胺 = 奖励预测误差）
- **Predictive coding**（Rao & Ballard 1999; Whittington & Bogacz 2017）
- **Equilibrium propagation**（Scellier & Bengio 2017）；**Forward-Forward**（Hinton 2022）
- **COLA**（ICML 2026）：临界态局部规则 ≈ BPTT
- **NALSM**（arXiv 2111.01760, 2021）：星形胶质细胞调制 STDP 到临界态
- **神经元雪崩 / 临界脑**（Beggs & Plenz 2003）

**深度循环与表达力**
- **Looped Transformers**（Saunshi et al., ICLR 2025）：p 层循环 k 次 ≈ pk 层；T 次循环可模拟 kL 步 CoT
- **Recurrent Depth**（Geiping et al., arXiv 2502.05171）：prelude-core-coda，KL 收敛退出
- **Ouro / LoopLM**（arXiv 2510.25741）：1.4B / 2.6B，7.7T token，知识容量不变、操作能力增强
- **STARS**（ICML 2026）：循环深度过大导致性能崩溃，Jacobian 谱半径正则
- **LoopFormer**（ICLR 2026）/ **LoopMDM**（arXiv 2605.26106）/ **LT2**（arXiv 2605.20670）：弹性深度 / 扩散循环 / 线性时间循环
- **Coconut**（arXiv 2412.06769）：连续思维，隐空间 BFS
- **Merrill & Sabharwal**：CoT 表达力（ICLR 2024）；Padding（NeurIPS 2025）；Log-Depth（NeurIPS 2025）

**教学与脚手架**
- **Scaffolding**（Wood, Bruner & Ross 1976）；最近发展区（Vygotsky）
- **行为主义**：Pavlov（经典条件反射）；Skinner（操作性条件反射、塑造、强化程序）

### 12.9 时间尺度与人工培养大脑（2026-10-23 补搜）

> 本轮补搜的起因：`design.md` §2.2 原稿写「时间常数必须显式给定」，这个前提从未被验证过。下面每条都标了它推翻什么。

**反面（时间尺度那一侧）**

| 工作 | 事实 | 推翻 |
|---|---|---|
| **S4D**（NeurIPS 2022，arXiv 2206.11893） | 消融（Table 3a）：实部固定 **-1/2** 就够（原文「a good default」），真正要铺开的是虚部（频率）。纯实部谱 `A_n = -(n+1)`（S4D-Real）在三项消融数据集上分别降 **5.45 / 10.17**、升 0.066；S4D-Random 降 **6.32 / 1.95**、升 0.034 | 「给衰减率谱有用」 |
| **S4D-DFouT**（NeurIPS 2025，arXiv 2508.20441） | 衰减与频率经 Δ 纠缠（两者都随 Δ 线性）；Prop. 1：S4D-Lin 下 Δ > 2/τ 就做不出 τ 处 spike；旧初始化的衰减率是 **hardcoded** | 「设定 τ 这件事成立」 |
| **HOPE**（OpenReview RZwtbg3qYD） | SSM 要小心初始化与训练：A 要降学习率、要重参数化才能稳定 | 「显式给定是稳的」 |
| **Stern / Istrate / Mazzucato**（eLife 2023，12:e86552，含 peer review） | 单细胞 τ 的异质性在强递归下**被抹平**；多尺度来自**异质组装的大小**；振荡时大/小组装分别 entrain 输入慢/快成分（**频谱分解**，不是记忆长度分级） | 「给每个单元设 τ 有用」；也给反面：「reviewer 指出线性近似下 τ = 1/(1-s) 也能得到长时间常数——作者承认其他机制可能，但批评它们需要外源噪声驱动」 |
| **Yamaguti & Nakamura**（arXiv 2605.14388） | 20 个独立训练的 RNN 在同一任务上**学出不同机制**（degeneracy of learned solutions）；低频由多神经元分布式参与，高频由少数短 τ 神经元主导 | 「一个任务对应一种机制」 |
| **Leaky CORnet**（PLOS Comp Biol 2026，10.1371/journal.pcbi.1014653） | 随机初始化网络的内左尺度层级**塔塌**，只有训练后才出现 | 反对「纯结构决定时间尺度」——结构 + 学习缺一不可 |
| **Mamba 实践**（docs.rs / 代码解析） | 初始化就把谱铺开（`A_n=-(n+1)`），理由是 the optimizer doesn't have to **discover from scratch** that the model needs both fast and slow dimensions | 「从零学得出快慢两套」 |

**人工培养大脑（in vitro 神经元网络）**

| 工作 | 事实 | 对本草案的意义 |
|---|---|---|
| **DishBrain**（Kagan et al., Neuron 110, 2022） | 80 万神经元养在 MEA；输入 = 球位置/速度编成电极刺激；输出 = 从自发活动解码左/右；反馈 = 接住给 100 Hz 正弦波 100 ms（可预测），漏了给随机 150 mV / 5 Hz / 4 s（不可预测）；**5 分钟内**出现学习；效果只有「比一开始好」 | 最干净的「窄进窄出 + 回路学习」实例；组 2 对照（只训读出）的原型 |
| **Organoid intelligence 的务实追问**（Trends in Biotechnology 2026-09） | 2D 培养物有 closed-loop learning；3D 类器官只有 **reservoir computing**（本体不训、只训读出）；**「closed-loop adaptation is a function of a whole OI system (substrate plus interface plus feedback), not of the tissue itself」** | `experiments.md` §11.4 必须带组 2 / 组 3 对照；也解释了为什么「学会了」不够当结论 |
| **批次变异**（Directed In Vitro Evolution，Zenodo 2025） | 「Even under identical culture conditions, organoids exhibit different learning rates and neural architectures」；「every brain organoid is unique」；提出用选择压力代替随机生长 | **必须 N ≥ 5 个种子**看分化是否稳定出现，不挑一个成功的报 |
| **技术限制**（ScienceDirect 2026 综述） | 深层无法记录、中心缺氧（>1 mm 死）、批次变异、无标准化协议、伦理未定 | 数字量级：一个实验的可靠结论很少，所以判据要粗、对照要少而硬 |

**新增可引用的工程事实（不是生物）**

- 储备池（Jaeger 2001；Maass 2002）：本体不训也做序列任务——但性能上限低于训练本体，且**读出宽度是主要代价**
- **三因子学习的分子链**（Directed In Vitro Evolution 综述引作机制）：电刺激 → 去极化 → Ca²⁺ 内流 → cAMP → PKA → CREB → 基因表达 → 新突触蛋白——**「临时信号 → 永久结构」的生物实例**

---

### 12.10 SNN 领域（2026-10-23 补搜）

> 本节是 §15 的依据来源。检索方式：15 轮 web_search，读摘要与部分全文，未逐篇精读；KB 侧未查（服务停着）。标注「预印本」者为未同行评审。
>
> **PDF 已归档：** `E:\knowledge-base\raw\papers\spiking-neural-networks\`（**44 篇 / 135.8 MB**，2026-10-23；完整清单与未获取说明见同目录 `INDEX.md`）。另有 **8 篇未获取**（PNAS 2 / bioRxiv 1 / HAL 1 / PMC 4，均 Cloudflare 403 或接口失效）。⚠️ **未入 RAG 索引**——KB 服务当时停止；下次启动需 `reindex(paths=[该目录])`。

**综述与全貌**

- **Spiking Neural Networks: Model, Learning Algorithms, and Hardware Implementations**（IEEE，2026-01）：神经元模型 / 学习算法 / 硬件三条线
- **Spike-based neuromorphic computing: overview**（ScienceDirect，2025-12）
- **Large-Scale Spiking Neural Networks and Their Hardware Platforms**（HAL，2025）：规模化的记忆与通信按 O(N·k) 计
- **Energy-Efficient NLP with SNN: A Comprehensive Review**（Zenodo，2025-12）：三个关键数字——现有 spiking 模型在 GLUE/SuperGLUE 上约达非脉冲对应物的 **85%~95%**；**最大 SLM 仍在 1B 参数以下**（典型 BERT-base 110M）；SGL 训练需常规反传的 **2~10 倍**计算
- **Predictive coding with spiking neural networks: A survey**（Neural Networks，2025-11）：预测编码 SNN 的三类（显式误差神经元 / 膜电位即误差 / 隐式）

**多时间尺度与异质性（含反例）**

- **LSRSNN**（Neural Networks，2026-03）：**显式双峰 τ 分群 + 专门学习算法**；原文「直接把宽分布混合在一起，常常无效甚至比单峰更差」「仅扩大异质性范围是不够的」；在工作记忆任务与语音识别上稳定优于单峰
- **DH-LIF / DH-SNN**（Nature Communications 15:277，2024）：**每个树突分支各自 τ**；单房室 LIF 在两路不同频率输入下产生不了 burst
- **HetSyn**（arXiv 2508.11644，2025）：**τ 放在突触上**（synapse-specific decay），SHD **92.36%**；原文称突触级异质性此前「largely underexplored」
- **Neural heterogeneity controls computations in SNN**（PNAS 121(3)，2024）：**反例**——同质网络的**工作记忆容量更大**，异质网络函数生成能力更强；是权衡不是单调改进
- **Optimizing Working Memory with Heterogeneity of Recurrent Cortical Excitation**（J Neurosci，2013）：异质性降低 bump 扩散但限制可存储状态数，**存在最优异质度**
- **A reservoir of timescales emerges in recurrent circuits with heterogeneous neural assemblies**（arXiv 2110.09165；eLife 12:e86552，2023）：单细胞 τ 的异质性在强递归下**被抹平**；多尺度来自**组装大小**；大/小组装分别 entrain 输入的慢/快成分
- **Neural timescales from a computational perspective**（arXiv 2409.02684）：明确警告**活动时间尺度可能显著不同于模型里设的膜/突触时间常数**
- **PLIF**（ICCV 2021）：每层共享一个**可学习** τ

**结构与演化**

- **ELSM**（Emergence of brain-inspired small-world SNN through neuroevolution，2024）：多目标演化液态状态机，目标 = **small-world 系数 + 临界性**；NMNIST 97.23% / MNIST 98.12% / Fashion-MNIST 88.81%；自发长出 hub / 短路径 / 长尾度分布 / 社群
- **NeuEvo**（PNAS 120(39)，2023）：STDP 演化出四类回路（前馈兴奋 / 前馈抑制 / 反馈抑制 / 侧抑制）+ 全局误差更新权重；CIFAR10 / DVS-CIFAR10 / DVS-Gesture / N-Caltech101 SOTA，ImageNet 可用，RL 可用
- **CortexZero**（Zenodo，2025-11-28，**预印本**）：发育型 + 演化型 + 树突 SNN；基因 = 区域树上的概率分布（膜参数 / 树突形态 / 可塑性系数 / 突触形成概率 / 神经调质受体 / E-I 比）；不用反传、不做静态网络、不同质神经元；**「Empirical behavioral results are deferred to future work」——只有规格，没有结果**
- **Evolving Spiking Circuit Motifs Using Weight Agnostic Neural Networks**（Sandia，2022）：WANN 迁到脉冲回路基元
- **Evolution of Plastic Learning in Spiking Networks via Memristive Connections**（IEEE TEVC 16(5)，2012）：演化**可塑性规则本身**（不只是权重）
- **Self-organization of multi-layer spiking neural networks**（arXiv 2006.06902，2020）：从单个 unit 生长 + STDP 自组织出分层，MNIST 约 90%
- **ModNet**（Zenodo，2026）：演化无层布尔电路网络，全加器 / 比较器 / 排序网络 100%
- **An Intrinsically Knowledge-Transferring Developmental SNN**（IEEE TNNLS 37(4)，2026）：按输入增量招募隐藏神经元，持续学习优于现有 CL 算法，比 BP 及其变体快 **1~3 个数量级**

**在线 / 局部学习**

- **e-prop**（Bellec et al., Nature Communications 11:3625，2020）：纯局部资格迹 + 一个全局学习信号 + 随机反馈；不需要存整条时间链；已上 SpiNNaker / Loihi
- **braintrace**（chaobrain，2026-10）：一行 `compile(model, 'eprop' / 'three_factor' / 'd_rtrl' / 'pp_prop' / 'dni')`
- **A spiking neural network with continuous local learning for robust online BMI**（J Neural Eng 20(6):066042，2024）：连续输入流上训练**不必周期性暂停做反传**，内存恒定，总内存降 **92%**；用 Decolle 式局部学习（每层固定权重读出 + 局部损失）
- **DelRec**（arXiv 2509.24852，2025）：原文「**Currently, all state-of-the-art spiking approaches on these benchmarks rely on surrogate gradient learning (SGL)**」；同时指出循环连接上的延迟「largely unexplored」
- **FidLIF**（arXiv 2610.16924，2026）：Newton 迭代 + 关联扫描，把精确 BPTT 做成 O(log T) 时间深度；同一神经元支持全序列并行训练与**逐步增量推理**
- **Burstprop / Burst-CCN**（bioRxiv，2024）：前馈（soma 上行）+ 反馈（dendrite burst 下行）双通道，局部 burst 依赖规则；**「Neuronal state variables and eligibility traces are reset to zero at the end of each sample presentation」**
- **GLSNN**（Frontiers Comput Neurosci，2020）：随机反馈 + 局部 STDP

**临界与稳态**

- **Synaptic balance due to homeostatically self-organized quasicritical dynamics**（Phys Rev Research 2, 012042，2020）：**两个稳态机制**（抑制性突触短时抑郁 + 阈值适应）即可让网络悬在临界点附近，同时给出 E/I 平衡与异步不规则放电，**不需要外部调参**
- **A unified theory of E/I synaptic balance, quasicritical avalanches and AI spiking**（J Phys Complexity 2, 045001，2021）
- **Self-Organized Criticality in the Brain**（Frontiers in Physics 9:639389，2021）：综述；雪崩斜率 -3/2、分支参数 1
- **Representation drift under spontaneous activity: SOC enhances representation reliability**（arXiv 2509.11545，2025）：**临界态最大化跨 session 低维表征可靠性**，即使单神经元表征在漂移
- **NALSM**（arXiv 2111.01760，2021）：星形胶质调制 STDP 到临界态

**持续学习（挤占）**

- **Continual Learning With Neuromorphic Computing**（IEEE，2025）：综述，分 8 类（STDP 增强 / 预测编码 / 活跃树突 / 贝叶斯 / 架构 / 重放 / 正则 / Hebbian）
- **CoLaNET**（Optical Memory and Neural Networks，2025-12）：列状 SNN + **局部学习规则**；10 个 permuted MNIST 任务各 **92%**，训练 9 个后续任务后第一个只掉 **4%**
- **CH-HNN**（Nature Communications 16:1272，2025）：皮层-海马环路 + 元可塑性；ANN 生成调制信号把 SNN 划成子网
- **NIMBLE**（Eng Appl AI，2025）：演化特征空间 + 兴奋/抑制中间神经元组，无任务标签、无回放
- **SAFA-SNN**（ICLR 2026）/ **RTK-WTA**（ICLR 2026）/ **PS-SNN**（Sci Rep 2026）/ **HLML-SNN**（AAAI 2026）：稀疏感知 / 随机时序 k-WTA / 模式分离 / 多层多标签

**语言与规模**

- **SpikeGPT**（TMLR 2024；arXiv 2302.13939）：45M / 120M / 216M；WikiText-103 test **39.75** vs GPT-2 Small 124M **29.41**、GPT-2 Medium 345M **26.37**；WikiText-2 test **18.01**（超过两个 GPT-2 对照，但训练集扩充过）；Enwik8 BPC 1.283 vs Transformer 1.137；SynOps 约 1/20
- **SpikingSSM**（AAAI 2025）：75M，WikiText-103 **33.94 PPL**；spiking 作为 S4D 之上的「quantization layer」
- **SAPU-LM / PSAPU**（Zenodo，2026-03，**预印本**）：78M，WikiText-103 **30.03 val / 31.11 test**；多 τ tile 的并行 reservoir，**recurrent weights 由 surrogate gradient 训练**；作者称「state of the art for pure spiking, no-attention language modeling」
- **NeuronSpark**（arXiv，2026-03，**预印本**）：0.9B 纯脉冲 LM，中文混合语料，预训练 loss 3.6
- **SpikeLM**（ICML 2024）：GLUE 平均 **76.5** vs BERT_base **83.2**（差 6.7 点）；无蒸馏超过 SpikeBERT 16.8 点
- **SpikingBERT**（AAAI 2024）：首个在 GLUE 多任务上跑的 spiking LM（蒸馏 + 隐式微分）
- **Otters++**（2026-06）：TTFS + 光学硬件模型，GLUE 平均 **84.17**
- **TTFS-GPT-2**（arXiv 2609.05151，2026-09）：1.5B，WikiText PPL **26.6**（SpikeGPT 39.8），作者自述「仍显著落后于 ANN 基线」
- **SymbolicLight V1**（arXiv 2605.21333，2026-09，**预印本**）：194M / 3B token，PPL 8.90 vs GPT-2 201M 8.27（+7.7%）；RTX 2080 Ti 上吞吐 **22.8 vs 91.5 tok/s**、能耗 **2848 vs 905 mJ/token**（作者注明未做数据泄漏验证、未做功率积分）
- **Compute-Optimal Scaling Laws for Spiking Language Models**（arXiv 2610.15146，2026-10）：首个 Chinchilla 式 IsoFLOP 研究（SpikeGPT 家族，width 384~1536，3e17~1e19 FLOPs）；**N_opt ∝ C^0.56**（稠密 0.50）；损失差 = **backbone 0.16 nats（不随算力变）+ spiking 0.34→0.23 nats**
- **SpikeLLM**（ICLR 2025）：**spike-driven 量化**（Optimal Brain Spiking），7B~70B；不是原生训练
- **SpikingBrain**（arXiv 2509.05276，2025-09）：7B 线性 + 76B 混合线性 MoE；**conversion-based training** + 150B token 继续预训练；4M token 序列 TTFT **100 倍**；69.15% 稀疏度
- **LAS**（AAAI 2026）：**loss-less ANN-SNN 转换**，6 个语言模型 + 2 个视觉语言模型，OPT-66B 上 WSC +2%
- **E-SpikeFormer / SFA**（TPAMI 2025）：integer training + spike inference；ImageNet 10M / 19M / 83M / 173M 参数 -> **78.5 / 79.8 / 84.0 / 86.2%**
- **Learning long sequences in spiking neural networks**（Scientific Reports，2024）：原文「不可微的**二值激活在准确率上被连续饱和激活所上界**；而后者又远远落后于深度 SSM 里的非饱和激活」；解法 GSU 放弃纯二值（ternary + 连续流）

**发育与生长**

- **Self-organization of multi-layer SNN**（2020）：见上
- **BDNN**（TNNLS 2026）：见上
- **CortexZero**（2025-11）：见上
- **cadence-net**（PyPI 0.16.0，2026-09）：工程实现，含 15 万神经元果蝇连接组落地案例

### 12.11 个体性测量的方法学来源（2026-10-23 补）——**⏸️ 悬置**

> **获取情况：本节引用的 4 篇均未入库。** 已试途径：① DOI 直接 GET ② Unpaywall API（四个 DOI 均无 OA 位置）③ PMC（**全线 403**）④ 作者机构仓（ANU openresearch 返回截断的 1 页残件，已删）⑤ HAL（hal-00739058 多个路径均 404）。原因：均在 Elsevier（Animal Behaviour）/ Wiley（Biological Reviews）付费墙后。**因 `experiments.md` §11.5 已悬置，不影响任何结论，不建空目录。**
>
> 未获取清单：Bell, Hankison & Laskowski 2009（重复性元分析，Animal Behaviour 77:771）；Carter, Feeney, Marshall et al. 2013（动物人格方法论批判，Biol Rev 88:465）；David, Auclair & Cézilly 2012（斑胸草雀短/长期重复性，Ethology 118:932）；黑猩猩 25 年稳定性（Animal Behaviour 227, 2025）。

> **⏸️ 随 `experiments.md` §11.5 一并悬置。** 下面的材料本身没错（animal personality 是成熟范式），**错在它测的东西不是自我**。保留理由：如果将来重开，这里省一次搜索；但要注意它已经不是「依据」，而是「一段弯路的方法学记录」。

> `experiments.md` §11.5 不是自创测法，它借的是行为生态学的成熟范式。本节记下借了什么、借的边界在哪里。

**核心范式：animal personality / behavioural syndrome**

| 概念 | 含义 | 在 `experiments.md` §11.5 里的对应 |
|---|---|---|
| **repeatability（重复性）** | 同一个体反复测出的行为是否比随机更一致 | 子测 A |
| **cross-situational consistency** | 不同情境下是否表现同一倾向 | 子测 C |
| **temporal stability** | 跨时间点的稳定度（不等于 1，允许变化） | 子测 A 的 T 个场合 |
| **interindividual vs intraindividual variation** | 个体间差异 vs 个体内波动，两者分开量 | 子测 B + A 的分解 |
| **test-retest reliability** | 同一测试重测的相关 | 子测 A 的 ICC |

**实测区间（用作阈值参照，不是直接挪用）**

| 对象 | 重测信度 | 出处 |
|---|---|---|
| 灵长类（多研究元分析） | **r ≈ 0.35 ~ 0.88** | David et al. 2012 引用的综述；猩猩 76 项行为 Cronbach's α 均值 **0.86** / Pearson **0.78**（Uher 等 2008） |
| 犬 | **r ≈ 0.43** | 同上 |
| 斑胸草雀（短/长期） | 短期两个特质均可重复；**长期只有一个可重复**（探索倾向稳定、挣扎率下降） | David, Auclair & Cézilly 2012，Ethology 118:932-942 |
| 黑猩猩（25 年跨度） | 人格特质长期稳定 | Animal Behaviour 227 (2025) |

**两个重要提醒**

1. **「长期稳定」不等于「所有特质都稳定」。** 斑胸草雀那篇正好证了反例：两个不相干的特质里，长期只有一个可重复。→ `experiments.md` §11.5 的子测必须**分维度报告**，不能只给一个总体 ICC。
2. **个体内变异本身是信号，不是噪声。** 该文原文指出「不同个体在短期稳定性上存在差异」——即 intraindividual variability 本身可以是个体特征。

**损伤/消融侧的方法学（为何不用它做判据）**

| 概念 | 含义 | 为何不能当 D 判据 |
|---|---|---|
| **graceful degradation** | 分布式表征下局部损伤对整体影响小（APA 定义） | 平滑 = 分布式，不是「中心」 |
| **局灶损伤 → 特定能力崩溃** | Coltheart 1991 举的病例：只能理解不能产出专有名词 | 特定崩溃 = 局灶模块，不是「中心」 |
| **diffuse vs catastrophic damage** | Aitken 1991 的反驳：弥漫损伤才会平滑 | **两者都测不出「分布式里的中心」** |

**一个可用的旁路观测**：损伤曲线仍可用来量「模块化程度」（陡崖位置 → 偏模块化），但它不回答 D。


---


### 12.12 世界模型与持续控制（2026-10-25 补录）

> 起因：用户问「Dreamer v3 和 v4、以及各种其他连续状态模型」。查完后发现**这一支在项目里几乎是空白**——
> §12.8 有 C-DESN / LNN / 储备池 / 循环深度，但没有 Dreamer / RSSM / TD-MPC 这一整个「显式连续状态 + 在想象中学习」的主线。
> 而 `design.md` §3.2 说「密度维靠世界模型」、`experiments.md` §16.8 / `experiments.md` §16.13 测预测误差全部失败——
> 这条线的现成答案正好在那里。
>
> **命名澄清：** 官方没有「Dreamer **V**4」。第三代官方写作 **DreamerV3**（带 V），第四代论文标题是
> *Training Agents Inside of Scalable World Models*，作者与 PyPI 包都叫 **Dreamer 4**（无 V）。
> 社区（博客 / wiki 聚合）写成 V4 是沿用 V1-V3 的习惯，不是官方措辞。

#### 12.12.1 DreamerV3（Nature 640(8059):647-653, 2025）

世界模型 = **RSSM**，六个部件（原文公式）：

```
sequence model   h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})     确定性递归状态
encoder          z_t ~ q_φ(z_t | h_t, x_t)                后验（训练时）
dynamics pred.   ẑ_t ~ p_φ(ẑ_t | h_t)                      先验（想象时用这个）
reward / continue / decoder    ~ p_φ(· | h_t, z_t)
```

损失：`L = Σ_t (β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep)`，**β_pred = 1 / β_dyn = 1 / β_rep = 0.1**；
dyn 与 rep 用不同的 stop-gradient，并用 **free bits 裁在 1 nat（≈1.44 bit）** 以下——
原文目的：防止「dynamics 平凡可预测但没信息」的退化解。

**五项鲁棒技术（消融结论：KL balancing 与 free bits 最显著，其次是 return normalization 与 symexp twohot）：**

| 技术 | 形式 / 作用 |
|---|---|
| **symlog / symexp** | `symlog(x) = sign(x)·ln(|x|+1)`；编码器输入与解码器目标都变换 |
| **twohot 回归** | 连续值 → 分类，解决多模态分布 |
| **KL balancing** | 动态调整世界模型目标权重，防表征崩塌 |
| **free bits** | KL 项裁在 1 nat 以下就关掉 |
| **百分位回报归一化** | 回报缩放到 [0,1] |

战绩：单一配置、8 领域 **150+ 任务**、Minecraft 从零采钻石（**1 亿**环境步）。

#### 12.12.2 Dreamer 4（arXiv 2509.24527, 2025-09-29；Hafner / Yan / Lillicrap）

与 V3 **不是同一套架构**：从 RSSM 换成 tokenizer + 交互式扩散 / 动力学 transformer。

- **tokenizer**：因果注意力 MAE，把帧压成连续 latent（低维 tanh 投影）
- **动力学**：在「动作 ⊕ shortcut 噪声水平 ⊕ 步长 ⊕ token 表示」的交错序列上跑，
  用 **shortcut forcing** 去噪 → 少量前向传递即可交互式生成，且**抑制误差随时间累积**
- **微调成 agent**：在动力学 transformer 里插 **task token**，从中间时预测 action / reward / value
- 结果：**单 GPU 实时交互**；**首个纯离线**（零环境交互）在 Minecraft 采到钻石，动作序列 **>20,000** 步；
  **比 OpenAI VPT 的离线 agent 用少 100 倍数据**却更好；世界模型的特征用于 BC **优于 Gemma 3**
- 官方代码**未发布**；非官方实现：`nicklashansen/dreamer4`（PyTorch）、`edwhu/dreamer4-jax`

**一条反面对照（第三方复现，可信度中等）：** HuggingFace `Dreamer-VLA/dreamer-vla-dreamer4-libero` 在 LIBERO 上
复现 5 个变体（64M~276M，含 DMControl 预训练与辅助逆动力学损失），**action-probe R² 全为负**
（−0.036 ~ −0.040）；而 V-JEPA 2 ViT-L **+0.845**、LAPA **+0.510**。
→ 可能结论：**Dreamer 4 的能力来自规模 + Minecraft 数据，不一定来自那个重建目标。**

#### 12.12.3 TD-MPC2（arXiv 2310.16828，ICLR 2024 Spotlight）

与 Dreamer 最大的差别一句话：**decoder-free，不重建像素。**

- 隐式世界模型：只学 `latent + reward + value + policy prior`，**没有任何生成 / 重建头**
- 潜空间里做**局部轨迹优化**（MPPI 风格），执行首步后重规划
- 用**学出来的 terminal value** bootstrap 超出 horizon 的回报
- 104 个连续控制任务（DMControl / Meta-World / ManiSkill2 / MyoSuite）、**单一超参**；
  单 agent **317M 参数跑 80 任务**；**300+ checkpoint + 数据集开源**（MIT）

原文明写的动机：**「精确预测原始未来观测（图像 / 本体感觉）在长时程上很难，而且对控制没有帮助」**——所以干脆不要 decoder。

#### 12.12.4 V-JEPA 2 / V-JEPA 2-AC（arXiv 2506.09985, 2025-06）

- **V-JEPA 2**：action-free 联合嵌入预测，**1M+ 小时**互联网视频 + 1M 图；
  Something-Something v2 **77.3** top-1；Epic-Kitchens-100 **39.7** recall@5
- **V-JEPA 2-AC**：在 **<62 小时**无标注机器人视频（Droid）上后训练成动作条件世界模型；
  300M transformer predictor，**block-causal attention**；损失 = teacher-forcing L1 + **rollout loss**
  （原文：rollout loss 是稳定多步预测的**关键**）；规划 = CEM 800 样本 / 10 轮精化 / 13 cm 动作约束
- 结果：零样本部署到两个实验室的 Franka 臂，拾放 **65%~80%**（无环境数据、无任务训练、无奖励）

#### 12.12.5 连续状态载体（LLM 侧）

| 工作 | 形态 | 关键结论 |
|---|---|---|
| **TTT**（Sun et al., ICML 2025） | **隐状态本身是一个模型**（线性 / 2 层 MLP），更新规则 = 一步自监督学习；**推理时也在更新** | 125M~1.3B；「Transformer 与 TTT 能靠更多 token 持续降 PPL，而 **Mamba 在 16k 之后不能**」 |
| **⚠️ TTT 的修正**（ICML 2026, NVIDIA） | 带 KV binding 的 TTT **可化归为「学出来的线性注意力」** | 「测试时记忆」这个解读是**错的**；它不是记忆查找，是表达力更强的线性注意力 |
| **MesaNet**（Google, 2025） | 把 DeltaNet / Mamba / xLSTM 统一成「in-context 回归目标的在线近似优化」；MesaNet 每步用共轭梯度**最小化到最优** | 十亿参数规模，长上下文优于前代 RNN；代价是**推理时更多 FLOPs** |
| **Atlas**（arXiv 2505.23735） | 记忆 = 在线优化问题；线性记忆退化为 **gated linear attention**：`W_t = α_t W_{t-1} + v_t k_tᵀ` | 提供「记忆更新 = 优化步」的统一写法 |

**四个都是同一个形态**：「状态是一个随每个 token 更新的模型」——与 `design.md` §2.10「记忆就是连接」是同一件事的两种说法。
**但它们全部在序列结束时重置**——这是与本项目的关键边界。

#### 12.12.6 reset-free / continual RL（这条线项目里完全没有）

| 工作 | 干什么 |
|---|---|
| *You Only Live Once: Single-Life RL*（Chen & Finn, 2022） | 一次机会、无 episode 重置；分布匹配从新状态恢复（成功率 +20~60%） |
| *Reset-Free Lifelong Learning with Skill-Space Planning*（LiSP, 2020） | 非平稳、非 episodic 环境下的技能空间规划 |
| *NeoRL* / *STEEL*（Levine & Zhang） | 单条轨迹上的探索；用环境里的确定性循环「等掉」相关噪声 |
| *A Definition of Continual RL*（arXiv 2401.15059） | 给「持续」下定义（不是「多任务」的别名） |
| 平均奖励（average-reward）替代折扣 | 折扣在无限时程下对 γ 敏感；持续任务改用平均每步收益 |
| **forward-backward policy** | 训一个「倒回去」的策略来自动 reset，人不用插手 |
| 评测基准：**Jelly Bean World / AgarCL / POPGym** | 逼着 agent 处理非平稳与不可逆转移 |

**注意：** POPGym 是专门测「记忆」的基准，可作为 `experiments.md` §11.4 那六条判活里「有记忆」一项的外部参照（本轮只看到标题级，未读原文）。

#### 12.12.7 RNN → SNN 转换（阶段 0.5 的直接先例）

**Conversion of Artificial Recurrent Neural Networks to Spiking Neural Networks for Low-power Neuromorphic Hardware**（arXiv **1601.04187**，2016）——用 Elman RNN 转 SNN 做问答分类。三段损失拆开看：

| 离散化对象 | 性能变化 | 机制 |
|---|---|---|
| 突触权重 | **最大损失来源** | 可用「训练时就用量化权重」或概率化取整缓解 |
| 隐藏状态 → 4 bit | **不降反升** | 离散化**阻止状态漂移**：以 "where" 开头的问句，答案应是地点，这个状态必须在后续无关词上保持住 |
| 映射到脉冲 | **−4.4%** | ReLU→spike 的损失随积分时间趋近零；但**循环层不能靠「把样例呈现更久」换精度**，要靠 delay 机制，而 delay 长度受轴突资源限制 |

**现代 ANN→SNN（前馈侧）已经相当成熟：**

| 工作 | 做了什么 | 数字 |
|---|---|---|
| **Closed-Loop CKA Distillation**（ICML 2026，湖南大学） | 冻结源 ANN → 转一次 → **再 fine-tune SNN**（全局 logits 蒸馏 + 局部 CKA 特征对齐） | ResNet-18/CIFAR-10：T=2 **90.05%**（基线 10%）、T=8 95.33%、T=32 **96.38%**（源 ANN 96.39%） |
| **IDSNN**（Biomimetics 2023） | 用 ANN 参数**初始化** SNN → 再蒸馏训练 | CIFAR-10 **94.03%**（T=4） |
| **Bi-Spectrum Distillation**（AAAI 2025） | 指出 **LIF 是低通滤波器**，高频被抑制 → 「spectral mismatch」是转换的核心障碍 | CNN 与 **Transformer** 上都报 SOTA |
| **Training-free conversion**（arXiv 2409.03368） | 局部阈值平衡，最小化转换误差界 | —— |
| **Spiking LLM 双转换 + 逐层校准** | LLaMA 转 SNN，整数脉冲神经元 + 多阈值 | 与 SOTA 量化可比 |

**所以「转换」不是新的，「转换后继续训」也不是新的**（闭环蒸馏就是）。
真正的空白是更窄的一条：**转换后的 SNN 在一条永不重置的时间轴上、用局部规则、在线继续学**（无 teacher / 无 dataset / 无 epoch）。

**一个原理性障碍必须先承认：LIF 是低通。** 所以「转成 SNN 就获得 RNN 级智能」要打折成「获得**低通版本**」。
与 §15.4 引的 Scientific Reports 2024 是同一件事的另一面：「二值激活在准确率上被连续饱和激活所上界」。
可证伪推论：回路里高频被反复低通 → 首尾相连后系统会**趋向慢变 / 同步**——
既可给更长的记忆，也可能吃掉区分力（测法：`probe_decay` + 4 cue 相似度，两个装置都是现成的）。

## 15. 与 SNN 领域的关系（2026-10-23 第三次修正）

> 本节回答一个把草案打回原点的问题：「这不就是个 SNN？」
>
> 答案是：**机制清单上逐条对应，而且已有 20 年积累；但草案要的那一支，恰好是 SNN 主流（BPTT + 固定 T + 按样本重置）在工程上排除掉的那一支。** 所以既不是重复劳动，也不是新发明——是「在 SNN 的地盘上做一件它的工程范式不允许做的事」。
>
> 本节分四件写：已有答案的、被外部证据修正的、真空在哪、对草案的直接后果。依据见 §12.10。

### 15.1 定位：机制清单逐条对应的术语

| 草案里的说法 | 该领域的标准术语 | 起始时间 |
|---|---|---|
| 稀疏图 + 节点动力学 + τ | **RSNN**（recurrent spiking neural network） | 2000s |
| 基因 = 结构超参数，变异 + 选择 | **neuroevolution / evolutionary SNN（ESNN）** | 2012 起有连续工作 |
| 发育 / 从单节点长 | **developmental / self-organizing SNN** | 2019-2020 |
| 局部规则 + 资格迹 + 全局调制 | **e-prop / three-factor learning** | 2007（三因子）/ 2020（e-prop） |
| 临界态、雪崩、稳态 | **self-organized (quasi-)criticality（SOC / SOqC）** | 2003 起 |
| 写入局部化、不挤占 | **continual learning in SNN** | 已有 8 类方法的综述 |

**一处偏差要点明：** `experiments.md` §10.8 N3 写「倾向 rate 或 LIF」。选 LIF 就是 SNN；选 rate 就是连续时间 RNN——同一家族，但草案一直没把「脉冲」定死。这一条现在必须定，因为它决定 §15.2 里哪些答案能直接用。

### 15.2 已有答案的（按待定项对照）

| 待定项 | 现成答案 | 出处 |
|---|---|---|
| **S1 的一部分**：多时间尺度要不要分家 | **LSRSNN：显式双峰 τ 分群 + 专门学习算法**，工作记忆与语音任务稳定优于单峰；且原文说「直接把宽分布混合在一起，常常无效甚至比单峰更差」 | Neural Networks 2026-03 |
| 同上，另一条路线 | **DH-LIF：每个树突分支各自 τ**；单房室 LIF 在两路不同频率输入下产生不了 burst | Nature Comms 15:277 (2024) |
| 同上，第三条路线 | **HetSyn：τ 放在突触上**（synapse-specific decay），SHD 92.36% | arXiv 2508.11644 |
| 同上，第四条路线 | **PLIF：每层共享一个可学习 τ** | ICCV 2021 |
| **τ 设定 vs 实测** | 综述明确警告：活动时间尺度可能显著不同于模型里设的膜/突触时间常数 | arXiv 2409.02684 |
| **S2 稳态形式** | **两个机制就够**：抑制性突触短时抑郁 + 阈值适应 -> 准临界 + E/I 平衡 + 幂律雪崩，不需外部调参 | Phys Rev Research 2, 012042 (2020) |
| **阶段 A 顺带项**：会不会自己跑到临界 | **ELSM 直接把 small-world 系数 + 临界性当演化目标**，自发长出 hub / 短路径 / 长尾度分布 / 社群 | 2024 |
| **S7 / S8 结构能不能自己长** | **NeuEvo：STDP 演化出四类回路 + 全局误差更新权重**，四个数据集 SOTA，还能做 RL | PNAS 120(39) 2023 |
| **S8 基因丰度（最宽档）** | **CortexZero：基因 = 区域树上的概率分布**（膜参数 / 树突形态 / 可塑性系数 / 突触形成概率 / 神经调质受体 / E-I 比），发育 = 采样，演化 = 改区域与分布，不用反传。**但只有规格，没有结果** | Zenodo 2025-11-28（预印本） |
| **S1 的另一半**：演化可塑性规则本身 | 已有人演化**可塑性规则**（不只是权重），用忆阻器做可变连接 | IEEE TEVC 16(5) 2012 |
| **`design.md` §7.3 / `design.md` §14.6 三因子 + 资格迹** | **e-prop**：纯局部资格迹 + 一个全局学习信号 + 随机反馈，有理论保证，已上 SpiNNaker / Loihi | Nature Comms 11:3625 (2020) |
| 同上，工程化 | **braintrace 一行**：`compile(model, 'eprop' / 'three_factor' / 'd_rtrl' / 'pp_prop')` | chaobrain 2026-10 |
| **`experiments.md` §11.4 测量三（挤占）** | CoLaNET：10 个任务各 92%，训练 9 个后续任务后第一个只掉 **4%**；CH-HNN；NIMBLE；SAFA-SNN；RTK-WTA | 见 §12.10 |
| **发育 + 增量招募** | BDNN：按输入增量长神经元，持续学习优于现有 CL 算法，比 BP 快 **1~3 个数量级** | IEEE TNNLS 37(4) 2026 |

**一句话：S1 之外的待定项，有一半已经被别人做过并且有数字了。**

### 15.3 被外部证据修正的三条

前两轮推出的结论里，有三条站不住：一条要收窄（τ）、一条漏了代价（异质性）、一条漏了实现成本（算力）。

#### ① 「τ 不可设计」要收窄为「逐单元连续铺开不可设计」

`design.md` §2.2 现在写「τ 不可设计」。外部证据给出的更准的说法是四档：

| 做法 | 结论 | 证据 |
|---|---|---|
| 以**单元**为单位连续铺开 τ | **无效**（强递归下被抹平） | S4D 消融（实部谱降 5.45 / 10.17）；Stern eLife 2023 |
| **单峰宽分布**直接混在一起 | **无效甚至更差** | LSRSNN 原文 |
| 按**组**分离成双峰 + 专门算法 | **有效** | LSRSNN（工作记忆 + 语音） |
| **每层**共享一个可学习 τ | **有效** | PLIF（ICCV 2021） |

所以准确表述是：**τ 不能逐单元连续设计，但可以按组分离或按层共享。** 而「按组分离」正好就是 `design.md` §2.2 那三件可设计事项里的「分区」——草案其实已经隐含了，但没把这个具体形态写出来，而它恰恰是**唯一有正面证据的形态**。

#### ② 「结构异质性有好处」不能写成无条件命题

| 证据 | 结论 |
|---|---|
| PNAS 121(3) 2024（Neural heterogeneity controls computations in SNN） | **同质网络的工作记忆容量更大**；异质网络的函数生成能力更强。是**权衡**，不是单调改进 |
| J Neurosci 2013 | 异质性降低活动 bump 的扩散，但**限制可存储的稳定状态数**——存在一个**最优异质度** |
| Stern eLife 2023 | 同一件事的另一条机制路径 |

`design.md` §2.2 把「结构异质性」列为可设计的三件之一，方向没错，**但没写它有代价**。`experiments.md` §11.4 阶段 A 如果用「有记忆」当适应度，演化会自己找到权衡点——这一点设计是对的；但判据里若只写「记忆长度 + 区分度」，会偏向同质。**要加一条反向观测：异质度与容量的权衡点落在哪。**

#### ③ SNN 的「省算力」在 GPU / CPU 上是假的

| 事实 | 数字 | 出处 |
|---|---|---|
| 同规模生成吞吐 | SNN **22.8** vs transformer **91.5** tok/s（慢 4 倍） | SymbolicLight V1 |
| 同规模每 token 能耗 | **2848** vs **905** mJ/token（高 3 倍） | 同上 |
| 综述结论 | 能耗优势**只在 event-driven 神经形态硬件上成立**，GPU 模拟的 SNN 不成立 | IEEE 综述 2026 |

**对 `experiments.md` §11.4 的直接后果：** 草案写「总成本本机 CPU 小时级」——如果按稠密矩阵实现，这个估计可能差一个数量级；只有真按事件驱动稀疏模拟才成立。**阶段 A 的种群评估与阶段 B 的长步数模拟，要先把模拟器的实现形态定下来（稠密 vs 事件驱动），否则成本估计是假的。**

### 15.4 语言能力边界（这是「transformer 等级」这个问题的答案）

必须分两类，混着看会得出错误结论。

| 类别 | 规模上限 | 最好成绩 | 对照 |
|---|---|---|---|
| **原生训练（from scratch）** | **1.5B**（TTFS-GPT-2）；0.9B（NeuronSpark，预印本） | WikiText-103 **30.03 PPL**（SAPU-LM 78M，预印本）/ 33.94（SpikingSSM 75M）/ 39.75（SpikeGPT 216M）；GLUE **76.5**（SpikeLM）/ **84.17**（Otters++） | GPT-2 Small 124M = **29.41**；GPT-2 Medium 345M = 26.37；BERT_base GLUE = **83.2** |
| **转换 / 量化（ANN->SNN）** | 66B（LAS）/ 76B（SpikingBrain）；SpikeLLM 7~70B | 「loss-less」（LAS）；SpikingBrain「与开源 Transformer 基线可比」 | **能力是借来的**：把已训好的权重压成脉冲执行，不是脉冲训练长出来的 |

**结论：**

1. **原生 SNN 已经到「小规模 transformer 的 90%~95%」**（综述原话：在 GLUE / SuperGLUE 上约达非脉冲对应物的 85%~95%），**在几个单点基准上追平或略超 GPT-2 Small / BERT_base**。
2. **离当代 LLM 差一个代际**，而且没有任何指令跟随 / 多步推理 / 工具使用的公开评测——**「没找到证据」不等于「做不到」，但确实说明这个领域还没走到那一步。**
3. **转换路径能「无损」，但那是借来的能力**：它证明的是「脉冲硬件可以无损执行 ANN 的计算」，不是「脉冲训练能长出 ANN 的能力」。

#### 任务上界 vs 存在上界（2026-10-23 拍板「persona 不设定」后的必读）

上文的「1.5B / GPT-2 Small 级」是**任务上界**，不是**存在上界**。两个必须分开，否则会把「benchmark 打不过」误读成「这个方向有天花板」。

| | 任务上界 | 存在上界 |
|---|---|---|
| 定义 | 给定 benchmark 上的最好成绩 | 能不能持续存在、持续变强（A + D） |
| 谁给 | 数据集 + 指标 + 非脉冲对照 | 反馈是否持续、可塑性是否还在、结构是否还在长 |
| 现成数字 | 原生 SNN 到 GPT-2 Small / BERT_base 级；0.16 nats 固定代价 | **没有数字**——`design.md` §4 的过程四度量就是为此设计的 |
| 会不会被「达到」 | 会（同规模 transformer 已达到） | 只有「还在不在」——没有「到顶」这个状态 |

**为什么这个区分在「persona 不设定」之后变得必要：** 若目标不是「完成任务的网络」，那「任务分布」这个框架本身就不适用——它没有一组固定任务可以定义上界。这时能问的只剩 `design.md` §4 的四条过程量与 `design.md` §2.9 的自我判据。

**但要说清楚：这不取消 0.16 nats。** 那个数字约束的是「用脉冲表达语言」这一件事——它仍然是它，只是不再是这个系统的总上限：

> **0.16 nats 限制「它把话说得多准」；不限制「它能不能持续变强」。**

#### 三条机制证据：为什么不是「再等等就行」

| 证据 | 内容 | 含义 |
|---|---|---|
| **固定代价** | 首个 Chinchilla 式 IsoFLOP 研究：损失差 = **backbone 0.16 nats（不随算力变）** + **spiking 0.34->0.23 nats（随算力降但不到 0）**；N_opt ∝ C^0.56（稠密 0.50） | 0.16 nats 换算成 PPL 约 **1.17 倍**——**哪怕算力无限也追不平的部分** |
| **表达上界** | Scientific Reports 2024 原文：「不可微的**二值激活在准确率上被连续饱和激活所上界**；而后者又远远落后于深度 SSM 里的非饱和激活」 | 这是表达能力层面的，不是训练技巧；他们的解法 GSU 其实就是**放弃纯二值** |
| **规模上限** | 综述：**最大 SLM 仍在 1B 参数以下**（典型 BERT-base 110M）；SGL 训练需常规反传的 **2~10 倍**计算；T 步展开带来额外内存 | 深而宽时近似误差累积 |

### 15.5 三个维度的现状（连续状态 / 训练方式 / 单向）

这三个问题决定了阶段 B 到底是不是新的，所以分开量。每一条的结论都一样：**主流做法与阶段 B 的前提相反。**

#### 维度一：连续内部状态——有，但只是「序列内连续」

| 层 | 状态怎么处理 | 证据 |
|---|---|---|
| **绝大多数** | T 步展开，**每个样本结束清零** | Burst-CCN 原文：「Neuronal state variables **and eligibility traces are reset to zero at the end of each sample presentation**」 |
| 同上，视觉流任务 | **每帧重置**是默认做法，改掉它本身成了一篇 CVPR 论文的贡献 | VISTREAM（CVPR 2025）：现有方法「leading to their **reset-per-frame**」 |
| **少数（SSM / linear-attention 系）** | 有跨 token 的递归状态（SpikingSSM 的 S4D、SpikeGPT 的 RWKV 状态、SpikingBrain 的 linear attention、SAPU 的多 τ reservoir） | SpikingBrain 4M token 上下文 |
| 同上但要注意 | **那个状态是 SSM 的状态，不是脉冲动力学的状态**；SpikingSSM 自己说 spiking 是「a **quantization layer on top**」 | 同上 |
| **几乎没有** | 永不重置。接近的两条：① Taeckens & Shah（J Neural Eng 2024）的内存恒定在线 SNN（但只是 BMI 回归小网络）② VISTREAM 的 LoCC-SNN（跨帧保留历史） | 两者都是专用 decoder |

**关键区分：SNN 说的「连续状态」是「一条序列内不清空」，不是「一个东西一直在活」。** 序列结束、batch 结束、请求结束，状态就没了。

> **⚠️ 2026-10-25 口径修正（重要）：** 上面这个「几乎没有」的判断**只在 SNN / LLM 这一支内部成立**。
> 在 **continual RL / reset-free RL** 那一支里，「永不重置」是**标准设定，已有 20 年积累**——
> Single-Life RL（Chen & Finn 2022）、LiSP（2020）、NeoRL、STEEL、平均奖励替代折扣、
> forward-backward policy、Jelly Bean World / AgarCL / POPGym 基准（详见 §12.12.6）。
>
> **所以真空不是「永不重置」，是三合一：永不重置 + 无任务 + 无准确率。**
> 本卷与 `design.md` 里写「没有人做永不重置」的地方，都应读成这四字的缩写。

#### 维度二：训练方式——主流就是梯度 + BPTT

最直接的一句（DelRec 2025）：**「Currently, all state-of-the-art spiking approaches on these benchmarks rely on surrogate gradient learning (SGL)」**。

| 学习规则 | 用梯度吗 | 要存时间链 | 能持续在线 | 代表 |
|---|---|---|---|---|
| **BPTT + surrogate gradient** | 是（完整反向） | **是**（O(LxT)） | **否**（要暂停才能反传） | 几乎所有 SOTA |
| **ANN->SNN 转换** | 在 ANN 上算 | 在 ANN 上 | 否 | SpikeLLM / SpikingBrain / LAS |
| **局部学习（Decolle 式）** | 是（每层局部梯度） | 否（O(1)） | **是** | Taeckens BMI（内存降 92%） |
| **e-prop** | 是（迹近似梯度） | 否（O(1)） | **是** | Bellec 2020 |
| **三因子调制** | **否**（标量 m 调制） | 否 | **是** | braintrace `'three_factor'` |
| STDP / Hebbian | 否 | 否 | 是 | Diehl & Cook 等 |
| 演化 | 否 | 否 | 否（代际） | ELSM / NeuEvo 的结构部分 |
| 纯稳态（无学习） | 否 | 否 | 是 | SOqC 网络 |

**要纠正一个常见混淆：「不用反向传播」不等于「不用梯度」。** 局部学习和 e-prop 都是**梯度类**，只是不用 BPTT、不存整条时间链。真正无梯度的只有演化、STDP、纯三因子调制。

#### 维度三：是不是单向的——默认是

SpNCN（arXiv 1908.08655）开场一句：**「unlike in standard spiking neural networks, which are typically designed to be feedforward in nature」**。

- **主流**：层堆叠 + 前馈。即使块内有递归（RWKV / SSM / 自循环），也是**层内递归**，不是**跨层反馈**。
- **有跨层反馈的只有几支**：预测编码 SNN（SpNCN、PLOS CB 2025 的多房室模型）、Burstprop（soma 上行 / dendrite burst 下行双通道）、LISR（自循环 + 侧抑制）、CH-HNN（皮层-海马双向）、NeuEvo（反馈抑制 / 侧抑制）。**这些全是分支，不是主线。**

### 15.6 两个阶段的重划

本轮搜证把 `experiments.md` §11.4 的两个阶段拉成了一高一低：阶段 A 是在已有赛道上重跑，阶段 B 才是真正空着的。**（2026-10-23 当天撤回了「阶段 A 降级」这一半——见下。）**

#### 阶段 A 应该降级 —— ⚠️ 2026-10-23 当天撤回

> **撤销记录（写在这里，防后人照着旧结论做）：** 本节的理由三条与「复用 ELSM」的建议**已不成立**。Jerry 当天拍定「连接必须能自己生长 / 裁剪」——加了生长之后，阶段 A 不再是 neuroevolution of SNN 的重复，而是另一件事：下面这些工作全部以「出生时结构定死」为前提，测的是「哪组超参能活」，测不到「结构能不能自己形成」。ELSM 的结构是演化出来然后冻结的，**复用不了**。
>
> **仍成立的部分：** 「阶段 B 是唯一的真空」（见下一节）。**阶段 A 的新形态与判据**见 `experiments.md` §11.4「阶段 A：演化生长规则」。
>
> 下面三条理由与建议**保留原文不删**——它们记了当时的推理，也记了理由三那个真缺陷（基因型与实测 τ_d 不是同一个量），后者在新形态里仍然要处理（`experiments.md` §11.4「阶段 A 的定位」）。

以下为原文（2026-10-23 第三次修正时写入，同日撤回）：

理由三条：

1. **它不是新实验**——是 neuroevolution of SNN，2012 年起有连续工作；ELSM、NeuEvo、WANN-Spiking、CortexZero 都在这条线上。
2. **结论基本可预期**——ELSM 已经把「小世界 + 临界 + hub / 长尾度分布」跑出来了；再跑一遍只能复现。
3. **它有一处自己的缺陷（§15.2）**：基因里的「τ 分布」与实测的 τ_d 不是同一个量（`design.md` §2.2：单细胞 τ 会被强递归抹平）。**只记录基因型、却从行为读结果，会让「收敛到广谱」有两种相反解释**——「选择偏好广谱基因」或「给了也白给」。

**建议改为（已作废）：** 「复用 ELSM 的演化设置 + 只补我们需要的那个观测（每个存活个体的**实测 τ_d 分布**，而不只是基因型）」。算力全部让给阶段 B。

#### 阶段 B 才是真空

| 草案 `experiments.md` §11.4 阶段 B 要求 | SNN 主流 | 是否冲突 |
|---|---|---|
| 一条永不重置的时间轴 | 每样本 / 每帧重置 | **直接冲突** |
| 不开学习也要有状态 | BPTT 必须有固定 T 窗口 | **直接冲突** |
| 无任务、无 episode、无准确率 | 全部有任务 + 准确率 | **直接冲突** |

而且这三条是**互相锁死**的：

```
要 BPTT  ->  必须有有限的展开窗口 T
         ->  所以必须按样本重置状态（否则没法算梯度）
         ->  所以必须有「一个样本 + 一个标签」
         ->  所以必须有任务和准确率
```

反过来：

```
要「永不重置」  ->  没有 T，BPTT 不可用
                ->  只剩：局部规则 / e-prop / 三因子 / 演化 / 稳态
```

**这就是为什么 SNN 领域没人做阶段 B——不是没想到，是被 BPTT 锁死了。** 这不是「研究不足」，是「设计前提互斥」。

> **2026-10-25 补：** 这句话要加一个范围——它只解释「**SNN 领域**为什么没人做」。
> continual RL 那一支用 TD / 策略梯度照样做「一条命」（§12.12.6），它们没有 BPTT 锁；
> 但它们保留了**任务与奖励**。所以完整的真空表述仍是三合一（§15.5 修正块）。

### 15.7 对草案的直接后果（四条）

四条都是「改了这一处，别处也动」的连带项，不是新增设计。

#### ① 阶段 B 提为唯一第一优先（阶段 A「降级复用」已撤回）

「阶段 B 是唯一真空」成立；「阶段 A 降为复用 ELSM」**不成立**（2026-10-23 当天撤回，见 §15.6 的撤销记录）——加了生长 / 裁剪之后，阶段 A 是另一件事。

现状：阶段 B 仍是第一优先，但它的前提是「一个活着的结构」，所以阶段 A 的稳态标定（`experiments.md` §11.4.0）在它之先。

#### ② S1 从「待定项」变成「唯一剩下的设计选择」

一旦放弃 BPTT（阶段 B 的前提），可塑性规则的**可选集被压缩到五种**：

| 候选 | 在线 | 内存 | 有信用分配吗 | 备注 |
|---|---|---|---|---|
| **e-prop** | 是 | O(1) | 有（近似梯度） | 有理论 + 有现成库；**最接近「能用」** |
| 局部学习（Decolle） | 是 | O(1) | 有（每层局部） | 需要每层有目标——与「无任务」冲突 |
| 三因子调制 | 是 | O(1) | **弱**（只有一个标量 m） | 最干净，但归因能力最弱 |
| 演化 | 否（代际） | — | 无 | 只能调结构 / 规则，不能在线学 |
| 稳态 + Hebbian | 是 | O(1) | 无 | SOqC 证明了能自组织，但不学任务 |

**按草案自己的判据（无任务、无 episode、要能测「挤占」），e-prop 是唯一同时满足「在线 + O(1) + 有信用分配」的候选；三因子是最「干净」但归因最弱的。** 这两个可以做成组 1 与组 4 的对照。

#### ③ e-prop / braintrace 不需要重新发明

`design.md` §14.6 的三因子（迹 -> 巩固 -> 稳定）与 `design.md` §7.3 的三因子规则，第一层已经被 e-prop 做完并且工程化了。**自己写一套没有意义——要么直接用，要么说明为什么不用。**

#### ④ 本体与语言：已定「共用」——只有一个完整个体

原写「三选一，未评估」。**2026-10-23 拍板：共用。**

| 原选项 | 结论 |
|---|---|
| A. 本体管自我，语言外包 | **否**——语言能力就不在自我里；`design.md` §2.9 的「自我」与 `design.md` §6 的「策略」之间会留下一个没人负责的缺口 |
| B. 本体自己长出语言 | **是**——但这是一个**要付代价的决策**，不是免费的选项 |
| C. 跨形态分阶段 | 否——它沿用 A 的假设（阶段 1 语言在外部），且多一次跨形态迁移 |

##### 代价（必须写明白：不是「选一个」，是「接受一个」）

1. **语言能力受 SNN 上限约束**：原生 SNN 目前最好到 GPT-2 Small / BERT_base 级（§15.4），且有一个**不随规模消失的 0.16 nats 固定代价**（约 1.17 倍 PPL）。
2. **阶段 1 不能外挂预训练 transformer 底座** → M8 收敛到「从零」。
3. **「脉冲化的块扩散」没有先例**：块扩散（SDAR / Eso-LMs）是 transformer 家族，SNN 侧只有「SSM + 脉冲量化层」（SpikingSSM）。两者的交集 15 轮检索未找到直接工作。M1 因此多一个候选并标风险。
4. **Pando 不是这个个体的雏形**：DeepSeek 权重冻结、记忆在外部文件 → 它不能承载累积修改（`design.md` §2.9）。在「只有一个完整个体」的前提下，Pando 的定位是**环境 / 工具 / 教师**，不是早期版本。这条要写清，否则会有「渐进演化」的错觉。

##### 两个派生结论（自洽性检查的收获）

**① 「说话」不是一次前向，是持续状态上发出的事件。**

如果只有一个完整个体，而它的状态永不重置（阶段 B 的前提），那语言就不能是「独立的推理周期」——它只能是持续状态上偶尔发出的一个输出动作。

这反过来把三条「倾向」变成了「必须」：

| 章 | 原状态 | 共用决策后 |
|---|---|---|
| `experiments.md` §10.8 N5 输出粒度 | 倾向事件 | **必须事件** |
| `design.md` §14.6 奖励粒度 | 奖励是事件，不是流 | **必须事件**（与输出同粒度） |
| `experiments.md` §10.8 N2 时间形态 | 倾向固定 tick | 固定 tick 作为**仿真步长**保留，但「一轮」不再是时间单位 |

**② per-step 注入的 persona 与「只有一个完整个体」不相容。**

每步注入等于承认「有一个外部条件在塑造它」——那它就是「本体 + 外部条件」两件，不是一个个。**persona 不设定（已定）**——不是早期教师后撤掉，是从一开始就不给；理由与两个后果见 `design.md` §2.9 拍板段。

### 15.8 一句话

> **这不是一个新架构，是一个「SNN 主流不允许做」的实验设计。**
> 已有的一半（结构演化、多尺度分群、稳态自组织、e-prop）直接用；
> 被修正的两条（τ 的收窄、异质性的代价）改口径；
> 剩下真正空着的一条：一个**永不清空、无任务、无准确率**的持续状态，会不会自己形成结构。

---

*（原草案至此结束。三卷的尾部各自往下追加）*
