# 历史文档工具

这些工具从 `scripts/` 移入，用于查阅旧草案的格式和文本演变，依赖 Python 标准库。默认目标是 `docs/archived/` 中的 [design](../../docs/archived/design-2026-10-25.md)、[experiments](../../docs/archived/experiments-2026-10-25.md)、[reference](../../docs/archived/reference-2026-10-25.md) 三卷。

| 工具 | 当前用途 |
|---|---|
| [check_draft.py](check_draft.py) | 标题编号、历史章节引用、表格与编码的诊断报告；默认检查 design，以另外两卷作引用底簿，不以退出码汇总全部问题 |
| [check_claims.py](check_claims.py) | 归档中的重复标题、子节编号和替代字符检查；已取消用当前目录计数约束历史数字的规则 |
| [diff_volumes.py](diff_volumes.py) | 将归档三卷按章节拼回，与 2026-10-22 草案比较；差异只是历史报告 |

无训练的冒烟命令：

```powershell
python scripts/historical/check_claims.py
python scripts/historical/check_draft.py
python scripts/historical/diff_volumes.py
```

三个工具的相对文件参数均以仓库根目录解析，启动目录不影响默认目标。它们不能证明实验结果成立，不能要求现行文档继续保留旧结论。现行维护检查使用 [scripts/check_docs.py](../check_docs.py)，内容复核见 [复核记录](../../docs/review-2026-10-26.md)。
