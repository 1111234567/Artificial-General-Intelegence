# 非脉冲任务基线

[baseline_linear.py](baseline_linear.py) 用 random、greedy、linear 三档检查固定状态—目标任务的可达性及局部学习表现。依赖 Python 与 NumPy，不依赖 LIF 框架。

两次 episode、单种子的入口冒烟：

```powershell
python experiments/baselines/baseline_linear.py 2 1 greedy
```

完整用法为 `baseline_linear.py [episode数] [种子数] [逗号分隔模式] [学习率]`，默认运行包含训练。固定任务上的高成功率只能说明该任务装置的表现，不能作为持续学习、跨任务泛化或其它网络形态的上界证明。比较时应统一观测、动作、训练预算与评估任务；见 [实验记录](../../docs/experiments.md)。
