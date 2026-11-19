# 前史迁移与相同输入下的状态恢复

主实验使用已经学过基本行动的普通 GRU，在同一前史预算下比较 A 和 C 对新规则 B 习得的影响。换边的 B′ 从相同前史终点开始。另一项实验冻结上一轮的普通和双向共享模型，在相同输入下演化两个状态，再释放成真正的自由行动。[研究报告](../../docs/reports/memory/experience-transfer-experiment-2026-11-03.md) · [分析方案](analysis-plan.md)

| 问题 | 实验与验证入口 |
|---|---|
| 同背景的两组训练是否匹配 | [train_transfer.py](train_transfer.py) 定义情境映射、连续真实世界、共同 B 示范和早期快照。[train_audit.py](train_audit.py) 重放训练经历并核对早期更新 |
| 新规则学习曲线和旧规则回测 | [eval_transfer.py](eval_transfer.py) 核对新请求身份、训练外的自主动作、成熟与合法窗口。[eval_report.py](eval_report.py) 汇总冻结评估 |
| 保持前史存档不变，交换哪一组与新目标部分重合 | [reciprocal_train.py](reciprocal_train.py) 和 [reciprocal_eval.py](reciprocal_eval.py) |
| 固定示范下的内容排序 | [content.py](content.py) 记录首次教师 FINISH 时模型自身的动作分数，[content_audit.py](content_audit.py) 用逐 tick 路径核对 |
| 不同状态在同一输入下是否靠拢 | [state_clamp.py](state_clamp.py) 分开记录固定历史与释放后的行为，[state_audit.py](state_audit.py) 从实际输入重放并与原评估器交叉检查 |
| 从实验文件生成图表 | [results.py](results.py) 汇合两个目标的曲线面积和保持结果，[state_figure.py](state_figure.py) 保留三个种子的状态差异 |

实验按种子保存在 `data/experience-transfer/`。数据根、检查点及动作轨迹由各版本协议绑定。本轮主张主要依据新的真实自由请求，读出诊断、高频位置数和反复训练同一批数据的次数都不能代替这个分母。

在仓库根目录检查评估计数器与报告链接：

```powershell
.\.venv\Scripts\python.exe -m pytest -q experiments/experience_transfer/eval_transfer_test.py
.\.venv\Scripts\python.exe scripts/check_docs.py
```
