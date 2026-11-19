# 神经元回归测试与旧入口

从仓库根目录执行 `python -m pytest -q test` 可验证[可复用神经元](../src/neurons/README.md)的状态推进、训练更新、原实现对拍和可用的 CUDA 路径。

## 共享框架入口

[run.py](run.py) 是薄兼容入口，会导入 [scripts/run.py](../scripts/run.py) 的 `main()`。这使旧命令继续可用，也让 Windows 多进程通过可导入模块定位工作函数。

```powershell
python test/run.py list
python test/run.py check base
```

框架命令说明见 [scripts/README.md](../scripts/README.md)，独立脚本及旧路径对照见 [experiments/README.md](../experiments/README.md)。原 `test/` 中的实验脚本按用途迁至 `experiments/`，原 `test/archived/` 迁至 `experiments/retired/`。

上述两条 `run.py` 命令查看或检查框架配置。`check` 通过输出报告发现的问题。
