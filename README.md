# agi

训练一个 AGI（不是一个更好的语言模型）。

**状态：阶段 1/2（教）已跑通（25% → 75.0%，v2 脉冲形态）；工作记忆跨度只有 20 步（= τ）；阶段 2（巩固）尚未开始。** 设计推演已完成（七轮修正），模拟器成本已实测，动力学已标定（rate 1.99%）；结构自生长已跑通（rate **1.55% ± 0.44**）。

**统一框架（`src/agi/`）已建成并开始对齐（2026-10-24）**：35 个实验声明 + 七命令，
**已移植 3 个旧脚本 / 20 项，全部逐位一致**（`grow_agent` 9/9、`grow_goal` 6/6、`grow_world` 5/5）；
剩 8 个脚本未移植——详见 `docs/experiments.md` §16.18 与 `src/agi/README.md` 的「移植覆盖度」。

**闭环三步（`grow_goal.py`，§16.13）：当时的阴性结论已不可信 —— 因为它建在一个不可完成的任务上。**
真正的发现（§16.16）：① **任务在参数上不可完成**——6 个动作 × 1 步 = 6，而目标差距 30~40；
② **`frozen` 名不副实**——它只关了权重与拓扑，bias 还在学。

修正后的真读数：

| 组 | 成功率 |
|---|---|
| `none`（三层全关） | **5.9%** |
| random（无网络） | 20.6% |
| 只学 bias | **42.0%** |
| bias + 权重 + 拓扑 | 36.2% |
| linear（非脉冲上界） | 99.8% |

**bias 是学习的全部贡献（+36.1 pp），权重/拓扑是负贡献（−5.8 pp）。**
**并且 2026-10-24 建立了统一框架（`src/agi/`）**，替代 21 个各实现一遍本体的手写脚本；
15 个实验声明 + 七命令。**⚠️ 行为尚未与旧脚本对齐**（旧 discrete 36.2% vs 框架 6.9~8%），
对齐前框架不能用来复现历史结论——详见 `src/agi/README.md` 的「已知缺陷」。
阶段 B（真正的真空）未开始。

## 文档

| 文件 | 内容 |
|---|---|
| `agi.md` | 主文档——定位、判据、当前状态、下一步、文档索引 |
| `docs/design.md` | 设计：判断与机制（§0–§9、§14） |
| `docs/experiments.md` | 验证：待定项 / 实验设计 / 准则与自评 / 实验记录（§10、§11、§13、§16） |
| `docs/reference.md` | 参考：文献索引与领域定位（§12、§15） |
| `src/agi/README.md` | 框架：结构、七命令、三个「不能再犯」的机制、**移植覆盖度** |
| `scripts/check_docs.py` | 文档结构校验：章节完整性 / 引用可解析 / 关键内容逐字存在 / 编码 |
| `scripts/diff_volumes.py` | 把三卷重建回原文，与 `docs/archived/` 的草案逐行 diff（验证信息不丢） |
| `scripts/align_layers.py` | **移植对齐工具**：construct / forward / run / seq / modes / contrib / verify / goal / world |
| `CHANGELOG.md` | 版本历程 |

> 原设计草案 `docs/archived/design-memory-learning-2026-10-22.md` 已退役（2026-10-24 拆入三卷，内容逐行迁移）。

## 目录

```
F:\agi\
  agi.md            主文档
  docs\             设计 / 实验 / 参考（三卷）
    archived\       已退役文档
  src\agi\          **统一框架**（config / core / env / signal / runner / store / experiments）
  scripts\          核验脚本
  test\
    run.py          **唯一入口**（list / show / check / run / results / gaps / align）
    archived\       17 个退役版本脚本 + README（退役理由）
    net.py          旧共享参数（框架将其收敛进 config.py）
    CONSTANTS.md    常数台账 + 场次四 12 条核对表
    *.py            旧实验脚本（实验记录，不再新增）
  data\             运行时数据（环境目录 / results.jsonl）
```

## 运行

**实验一律走框架**（旧脚本保留作实验记录，不再新增）：

```bash
python test/run.py list           # 有哪些实验，每个开了什么（含指纹）
python test/run.py check          # 前置检查（任务可行性）—— 不过就不跑
python test/run.py run base 8 8   # 跑 8 种子并登记到 data/results.jsonl
test/run.py 另有：show <name> / results / gaps / align
```

**移植对齐（判据 = 行为一致，全套 28 秒）：**

```bash
python scripts/align_layers.py modes      # mode → 三层开关（代码 + 实测交叉）
python scripts/align_layers.py verify     # grow_agent 9 项（动作序列逐项）
python scripts/align_layers.py goal       # grow_goal 6 项（16 字段逐位）
python scripts/align_layers.py world      # grow_world 5 项（19 字段逐位）
python scripts/align_layers.py contrib    # 单变量消融（从已对齐基准往下关）
```

**旧脚本（实验记录，可复现历史）：**

```bash
# 非脉冲线性基线（任务可学性的上界参照）
python test/baseline_linear.py 400 8 random,greedy,linear

# 模拟器成本基线（约 1 分钟）
python test/bench_lif2.py

# 动力学标定（五个版本逐次加机制）
python test/calib5.py

# 闭环实验：输出 -> 世界 -> 输入（五组对照，含五个实现缺陷的修正记录）
python test/grow_world.py 50000 8 8 closed,open,randact,noise_in,frozen

# 阶段 1/2：教它说话（4 个词 one-hot + 侧抑制，教师信号按神经元分配）
python test/grow_teach.py 10000 8 8 teach,rand_m,no_teach,frozen
python test/probe_out.py     # 诊断：输出神经元能不能区分四个词

# 阶段 3：功能测量（不改动力学，只加测量）：预测误差 / 动作增益 / 前史因果
python test/grow_func.py 50000 8 8 closed,open,frozen   # 三项 + 离线最优读出
python test/grow_func.py 50000 8 8 --diag                # 因果方向（a 领先还是 o 领先）
python test/grow_func.py 50000 8 8 --div                 # 前史因果（1e-3 扰动）

# 旧版（结构自生长，已停用——它的 15 版死网已定位为五个实现缺陷，见 experiments.md §16.6）
# python test/archived/grow_v2.py 50000 8 8
```

文档校验（章节编号 / 悬空引用 / 表格列数 / 乱码与行尾 / 待定项编号）：

```bash
# 第一个参数是要检查的卷；其余两卷自动作为交叉引用的底簿（章节号全项目唯一）
python scripts/check_draft.py docs/design.md docs/experiments.md docs/reference.md
python scripts/check_draft.py docs/experiments.md docs/design.md docs/reference.md
python scripts/check_draft.py docs/reference.md docs/design.md docs/experiments.md

# 三卷整合校验（章节归属 / 跨卷引用前缀 / 逐字块 / 编码）
python scripts/check_docs.py

# 拆分无损证明（三卷重建回单文件，与归档原件逐行 diff）
python scripts/diff_volumes.py
```

## 技术栈

Python + numpy（阶段 A/B 的模拟器）。实测：**单个体用 CPU 稀疏实现**（10⁷ 步 / 7.2 min），GPU 只在 batch ≥ 16 时赢。手写 kernel 未做。
