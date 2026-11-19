# 退役实验索引

这些文件从原 `test/archived/` 移入，保留历史版本、失败装置与中间标定过程。它们不作为默认实验入口；退役原因和旧读数应结合 [实验记录](../../docs/experiments.md)、[归档三卷](../../docs/archived/experiments-2026-10-25.md) 与 [复核记录](../../docs/review-2026-10-26.md) 阅读。

| 文件 | 退役用途 / 依赖 |
|---|---|
| `grow_snn.py`、`grow_snn2.py`、`grow_snn3.py`、`grow_snn4.py`、`grow_snn5.py`、`grow_snn6.py`、`grow_snn7.py`、`grow_snn8.py`、`grow_final.py`、`grow_v2.py` | 自生长实验的中间实现；后续版本见 `legacy` |
| `calib_dynamics.py`、`calib2.py`、`calib3.py`、`calib4.py` | 早期动力学标定；后续版本为 `legacy/calib5.py` |
| `bench_lif.py`、`bench_lif3.py` | 成本基准的早期与补测版本；历史机器读数见 [成本基线](../legacy/baseline-2026-10-23-simulator.md) |
| [grow_trace.py](grow_trace.py) | 独立 trial 形态；导入 `prototypes/grow_rnn.py`。后续连续流装置见 [grow_stream.py](../prototypes/grow_stream.py) |
| [probe_ask.py](probe_ask.py) | **不可运行**：依赖已经删除的三段式抢答接口，现存 `grow_talk` 为 first-spike 形态；启动时会明确退出 |

基本依赖是 Python、NumPy；旧成本工具的 Torch/CUDA 分支需要对应安装。这些文件多数没有低成本自检入口，重新运行前需要阅读代码和确认目标接口。本次迁移不重新认可旧脚本注释里的因果判断，也不调整其实验计算。

只检查语法，不执行退役实验：

```powershell
python -m compileall -q experiments/retired
```
