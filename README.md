# agi

训练一个 AGI（不是一个更好的语言模型）。

**状态：阶段 1/2（教）已跑通，且形态改成了脉冲 + 事件发射 + 连续状态；工作记忆存在但跨度只有 20 步（= τ）；阶段 2（巩固）尚未开始。** 设计推演已完成（七轮修正），模拟器成本已实测，动力学已标定（rate 1.99%）；**结构自生长已跑通**（rate **1.55% ± 0.44**）；**阶段 1/2 已学到**：v2（`test/grow_talk.py`）脉冲刺激 + 事件发射 + 永不清空的状态，delay=20 时正确率 **25% → 75.0%（+50 pp）**，四组对照清晰（`docs/experiments.md` §16.10）。**但记忆跨度只有 20 步**（延迟 50/100/150 全部停在 25%）——只有状态衰减，没有巩固。阶段 B（真正的真空）未开始。

## 文档

| 文件 | 内容 |
|---|---|
| `agi.md` | 主文档——定位、判据、当前状态、下一步、文档索引 |
| `docs/design.md` | 设计：判断与机制（§0–§9、§14） |
| `docs/experiments.md` | 验证：待定项 / 实验设计 / 准则与自评 / 实验记录（§10、§11、§13、§16） |
| `docs/reference.md` | 参考：文献索引与领域定位（§12、§15） |
| `scripts/check_docs.py` | 文档结构校验：章节完整性 / 引用可解析 / 关键内容逐字存在 / 编码 |
| `scripts/diff_volumes.py` | 把三卷重建回原文，与 `docs/archived/` 的草案逐行 diff（验证信息不丢） |
| `CHANGELOG.md` | 版本历程 |

> 原设计草案 `docs/archived/design-memory-learning-2026-10-22.md` 已退役（2026-10-24 拆入三卷，内容逐行迁移）。

## 目录

```
F:\agi\
  agi.md            主文档
  docs\             设计 / 实验 / 参考（三卷）
    archived\       已退役文档
  scripts\          核验脚本
  test\             实验脚本 + 基线数据
  src\              源码（尚未开始）
  data\             运行时数据
```

## 运行

实验脚本在 `test/` 下（Python + numpy，无 GPU 依赖）：

```bash
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
# python test/grow_v2.py 50000 8 8
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
