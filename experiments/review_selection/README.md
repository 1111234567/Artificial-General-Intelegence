# 等预算的旧经历选择

同一普通 GRU 从学完 A 的存档继续学习 B，比较均匀复习、只复习冲突和按模型旧标签预测误差复习。[研究报告](../../docs/reports/memory/review-selection-experiment-2026-11-03.md) · [预定分析](analysis-plan.md)

| 入口 | 作用 |
|---|---|
| [bank.py](bank.py)、[bank_audit.py](bank_audit.py) | 从真实旧轨迹构造目标槽请求库并重放世界、标签和窗口起始状态 |
| [train_review.py](train_review.py)、[train_replay_check.py](train_replay_check.py) | 恢复 A 存档，以相同候选和预算训练三臂，核对第一次真实复习优化及选择 |
| [eval_review.py](eval_review.py)、[eval_runtime.py](eval_runtime.py) | 在训练外的真实 B 流和随机 A/B 流评估与核验，后者处理元数据序列化 |
| [pairs.py](pairs.py)、[pair_audit.py](pair_audit.py) | 生成同槽同符号的换情境和重复情境对，用实际回执重建两次是否全程无错 |
| [results.py](results.py) | 导出三种子学习曲线、随机交错表现、配对成功及复习覆盖，并绘制对比图 |

各程序从仓库根目录按模块调用。运行训练需要项目的 CUDA 环境、上轮已绑定的完整存档和轨迹。已保存的正式结果在 `data/review-selection/`，查阅统计或报告可以直接读取汇总。为避免覆盖已有协议，完整复现应使用独立工作副本和输出目录。

```powershell
.\.venv\Scripts\python.exe -m pytest -q experiments/review_selection/bank_test.py experiments/review_selection/eval_review_test.py
.\.venv\Scripts\python.exe -m experiments.review_selection.pairs check
.\.venv\Scripts\python.exe scripts/check_docs.py
```
