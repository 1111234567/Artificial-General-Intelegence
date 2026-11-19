# 局部诊断

这里的探针主要针对 [legacy](../legacy/README.md) 中的具体 LIF 实现，依赖 Python、NumPy。它们测到的是对应装置与参数下的局部行为；活动自持、模式相似度或短时可分辨性均不足以单独证明记忆或持续学习。当前读数边界见 [复核记录](../../docs/review-2026-10-26.md)。

| 文件 | 目标与依赖 |
|---|---|
| [probe_in.py](probe_in.py) | 输入驱动；读取 `legacy/grow_talk.py` 的定义并直接执行小段诊断 |
| [probe_out.py](probe_out.py) | 历史输出驱动诊断；读取 `legacy/grow_teach.py`，但当前接口不匹配，不能直接运行 |
| [probe_persist.py](probe_persist.py) | 撤输入后的活动；读取 `legacy/grow_talk.py` 定义 |
| [probe_decay.py](probe_decay.py)、[probe_sim_noise.py](probe_sim_noise.py) | 无输入衰减、测量噪声；导入 `legacy/grow_homeo.py` |
| [probe_homeo.py](probe_homeo.py)、[probe_mem.py](probe_mem.py) | 稳态阈值、输入痕迹的历史诊断 |
| [probe_env.py](probe_env.py) | Python/NumPy/SciPy/Torch 环境信息；可选 CUDA 矩阵乘冒烟 |

`probe_in`、`probe_out` 与 `probe_env` 在导入时就会执行，不能用批量 import 当作无副作用检查。源码读取路径由 `__file__` 定位，不依赖启动目录。默认训练或扫描可能耗时。

已知历史兼容问题：`probe_out` 按 2 个输出累加，而现存 `grow_teach.OUT_IDX` 有 4 项，会报数组形状不匹配；本次目录迁移不改它的实验计算。`probe_in` 的输入到输出边集合为空时会打印 `nan` 并产生 NumPy 空均值警告，不能把该项读数当作有效测量。

最小输入探针（不训练）：

```powershell
python experiments/diagnostics/probe_in.py
```

如需只检查语法，可以执行 `python -m compileall -q experiments/diagnostics`。环境探针可另外运行，但 CUDA 冒烟成功也不证明实验训练或性能可复现。
