# Pulse game 回归测试

在仓库根目录运行整个游戏的测试，包含发布资料引用的顶层测试与本目录里的回归测试。

```powershell
.\.venv\Scripts\python.exe -m pytest -q experiments/pulse_game
```

| 范围 | 测试 |
|---|---|
| 生内快照、冷启动对照、延迟回执与在线读写 | [test_portable_continuation.py](test_portable_continuation.py)、[test_online_learning_paths.py](test_online_learning_paths.py) |
| 环形环境、实际接触与感觉训练目标 | [test_ring_world.py](test_ring_world.py)、[test_ring_metric.py](test_ring_metric.py)、[ring_alt_test.py](ring_alt_test.py) |
| 受体生长、跨场所经历与连续状态 | [ring_alt_chemical_test.py](ring_alt_chemical_test.py)、[ring_alt_habitats_test.py](ring_alt_habitats_test.py) |
| 对称探查、旋转对照与脉冲巩固 | [active_symmetry_test.py](active_symmetry_test.py)、[symmetry_spike_test.py](symmetry_spike_test.py)、[test_spike_policy.py](test_spike_policy.py)、[spike_consolidate_test.py](spike_consolidate_test.py) |
| 吸引子编码与因果前窗 | [attractor_motif_test.py](attractor_motif_test.py) |
