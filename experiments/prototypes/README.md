# 独立原型

这些脚本研究特定任务装置与转换方式，使用 Python 与 NumPy。它们保留各自的训练循环和评估口径；当前结论边界见 [实验记录](../../docs/experiments.md) 与 [复核记录](../../docs/review-2026-10-26.md)。

| 文件 | 用途与边界 |
|---|---|
| [grow_rnn.py](grow_rnn.py) | Elman RNN、BP 与固定任务基线；`--gradcheck` 只验证局部解析梯度 |
| [grow_stream.py](grow_stream.py) | 连续事件流上的条件化装置；历史读数不能直接当作记忆跨度 |
| [grow_snn_convert.py](grow_snn_convert.py) | tanh 值量化原型；历史转换表现不能直接归因为脉冲实现的性质 |
| [grow_loop.py](grow_loop.py) | 首尾连接与 population 原型；历史读数受装置与时间切分影响 |
| [grow_lif.py](grow_lif.py) | 独立 LIF 原型，用于历史比较，与 `src/core.py` 的框架实现不同 |
| [rnn2snn.py](rnn2snn.py) | 将 RNN 矩阵映射到当前 `src.core.Network`；结构映射不等于行为等价 |
| [probe_g.py](probe_g.py) | 参数增益与差异放大的局部诊断 |

`grow_stream`、`grow_snn_convert`、`grow_loop`、`grow_lif` 与 `probe_g` 依赖同目录 `grow_rnn`；`rnn2snn` 还依赖当前 `src`。跨目录导入由脚本位置解析。

轻量冒烟：

```powershell
python experiments/prototypes/grow_rnn.py --gradcheck
python -c "from experiments.prototypes import grow_stream, grow_snn_convert, grow_loop, grow_lif, rnn2snn, probe_g; print('prototype imports OK')"
```

导入命令在仓库根目录执行，不调用主训练。`rnn2snn.py --check` 会先训练 RNN，并非轻量结构检查；其它默认命令和部分 `--sanity` 分支也包含训练。局部冒烟通过不代表历史科学判断得到复现。
