# agi

训练一个 AGI（不是一个更好的语言模型）。

**状态：机制验证中。** 设计推演已完成（七轮修正），模拟器成本已实测，动力学已标定（rate 1.99%）；**结构自生长未跑通**（15 个版本，开结构可塑性后 rate 掉到 0.08%）。阶段 B（真正的真空）未开始。

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

> 原设计草案 `docs/design-memory-learning-2026-10-22.md` 已退役（2026-10-24 拆入三卷，内容逐行迁移）。

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

# 结构自生长（当前未跑通）
python test/grow_v2.py 50000 8 8
```

文档校验（章节编号 / 悬空引用 / 旧断言残留）：

```bash
python scripts/check_draft.py docs/design.md
python scripts/check_draft.py docs/experiments.md
python scripts/check_draft.py docs/reference.md
```

## 技术栈

Python + numpy（阶段 A/B 的模拟器）。实测：**单个体用 CPU 稀疏实现**（10⁷ 步 / 7.2 min），GPU 只在 batch ≥ 16 时赢。手写 kernel 未做。
