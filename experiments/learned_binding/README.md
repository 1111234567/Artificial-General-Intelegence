# 从公开经历学习绑定、地址与写入

接续[手写关联适配器](../associative_state/README.md)。新模型只读取原公开56维输入，使用共享的每槽GRU生成连续读写地址、value与gate，整个过程由原十一动作损失训练。随后让模型自己采样动作，以公开成功/拒绝结果做 policy gradient。原执行器权重保持冻结。

[冻结方案](analysis-plan.md)。正式工件在 `data/learned-binding/v1/`；旧手写模型和所有前序工件只读。

| 文件 | 职责 |
|---|---|
| [model.py](model.py) | 可微原生动作接口、学习的请求/回执历史、自由读写 |
| [data.py](data.py) | 实际教师轨迹、独立留出世界、只读取公开回执的奖励函数 |
| [common.py](common.py) | 固定设置、来源与源码绑定 |
| [pretrain.py](pretrain.py) | 1,024 tick完整动作CE、数据冻结和终点冻结 |
| [interaction.py](interaction.py) | 自身采样、完整状态、pending回执、行为评价 |
| [run.py](run.py) | 两槽联合回报PG、独立分支与存档 |
| [audit.py](audit.py) | 全量世界/回报/状态链及独立PG/Adam重算 |
| [results.py](results.py) | 审计门槛、逐种子/合并统计与图 |
| [binding_probe.py](binding_probe.py) | 事后只读检查请求身份在状态中是否可恢复 |
| [read_first.py](read_first.py) | 只改变读写顺序的独立后续、复用手写控制 |
| [frozen_trace.py](frozen_trace.py) | 11/10仅固定encoder参数的对照，z照常演化 |

模型不调用手写binder，不按context×raw查地址，也不以成功回执硬屏蔽写入。固定的soft逆置换接口仍然存在；有监督CE仍使用教师动作，不能将该阶段称为只靠自身奖励学习。真正的自身结果学习单独放在outcome阶段，与同起点的参数冻结分支比较。

动态状态需完整计账：native为原h192、encoder z64、M64，共320标量；手写版另外使用八个绑定缓存量，两者不是等参数或等总状态比较。manual_hard保留旧接口，manual_soft单列相同软接口的参照。

```powershell
.\.venv\Scripts\python.exe -m pytest experiments/learned_binding -q
.\.venv\Scripts\python.exe -m experiments.learned_binding.run declare
.\.venv\Scripts\python.exe -m experiments.learned_binding.pretrain data --workers 12
.\.venv\Scripts\python.exe -m experiments.learned_binding.pretrain train --device mixed
.\.venv\Scripts\python.exe -m experiments.learned_binding.run all --workers 9
.\.venv\Scripts\python.exe -m experiments.learned_binding.audit
.\.venv\Scripts\python.exe -m experiments.learned_binding.results
```

[读写顺序后续方案](read-first-plan.md)复用第一版冻结数据和手写工件，仅新训与运行学习者：

```powershell
.\.venv\Scripts\python.exe -m experiments.learned_binding.read_first declare
.\.venv\Scripts\python.exe -m experiments.learned_binding.read_first train
.\.venv\Scripts\python.exe -m experiments.learned_binding.read_first behavior
.\.venv\Scripts\python.exe -m experiments.learned_binding.read_first audit
.\.venv\Scripts\python.exe -m experiments.learned_binding.results --root data/learned-binding/read-first-v2 --output docs/evidence/learned-binding-2026-11-07/read-first-v2
```

声明、训练和评分不覆盖已有运行，审计与汇总可重建。恢复运行需要本机已有的三份teacher-timing动作执行器和三份shared-v2 delta检查点。[实验结果和限制](../../docs/reports/memory/learned-binding-experiment-2026-11-07.md)

11/10的[固定特征方案](frozen-trace-plan.md)及[报告](../../docs/reports/memory/frozen-trace-experiment-2026-11-10.md)另行记录，沿用原冻结数据与手写参照：

```powershell
.\.venv\Scripts\python.exe -m experiments.learned_binding.frozen_trace declare
.\.venv\Scripts\python.exe -m experiments.learned_binding.frozen_trace train
.\.venv\Scripts\python.exe -m experiments.learned_binding.frozen_trace behavior
.\.venv\Scripts\python.exe -m experiments.learned_binding.frozen_trace audit
.\.venv\Scripts\python.exe -m experiments.learned_binding.results --root data/learned-binding/frozen-trace-v3 --output docs/evidence/frozen-trace-2026-11-10
```
