# 连续学习网络 · CPU 运行包

已用 Python 3.14.5 和 NumPy 2.5.4 完成干净环境验证。在解压目录下运行。两个命令分别运行一次迁移与回访实验，以及有隐藏规则变化的连续生命。

```powershell
python -m lifelong.lifelong_agent --game transfer --worlds 24
python -m lifelong.lifelong_agent --game long --worlds 16 --blocks 192
```

在迁移游戏中，A 和 B 有时采用相同的气味关系，有时相反。被试在 B 先尝一个新食物，再进行首次选择，之后才开始获得 B 的成对关系样本。各行按相同或冲突规律分别计分。长生命游戏每次访问包含 12 个请求。`--blocks 192` 对应每个生命 2,304 个请求。每次真正碰到食物才得到味觉。在测试选择中，环境照常评分并隐藏味觉。改变 `--seed` 会生成新气味、私有偏好与物理场景。

已有自己的环境时，可以直接加载个体。以下接口只依赖 NumPy。

```python
from lifelong.lifelong_agent import new_lives

brain = new_lives([101, 202])
brain.reset(2)
# observations 是形状 (2, 20) 的 float32 数组，与下表的两个身体一一对应。
motor_pulses = brain.act(observations)
```

| 通道，从 0 开始 | 公开含义 |
|---|---|
| 0–1 | 栖息地的两条感觉电流 |
| 2–5、6–9 | 左、右食物的四种化学成分，脉冲消失时各为 0.5 |
| 10、11、12 | 身体位置、速度、当前能否运动 |
| 13、14 | 上次真实接触的正味、负味 |
| 15–16 | 上次实际碰到左侧或右侧的标记 |
| 17、18–19 | 新请求脉冲，两端距离 |

两个输出分别是向左和向右的运动脉冲。环境根据脉冲改变速度和位置。出生时清空神经状态和经验，一生内持续调用 `act`。冷冻检查点保存感觉连接、读写参数和运动参数。每次实际试吃会改变该生命的短期记忆和学习规则读出。

`--fixed-covariance` 可关闭持续释放可塑性的步骤，便于在同样的场景中复验长期翻转。包内 `manifest.json` 记载参考个体的选择方式和各文件 SHA-256。

面对互相冲突且能由感觉辨认的栖息地，可用同一检查点比较两种写入方式。默认的 `--address distributed` 使用训练学得的上下文方向。`--address local` 在一生中从真实出现的两路背景脉冲分配最多六个局部地址。新地址复制最先形成的规则作为初值，以后的回执只改当前地址。这里的最近原型、角距阈值和复制策略由研究者提供。

```powershell
python -m lifelong.lifelong_agent --game transfer --address local --worlds 64
```

应用程序可以在任意一次 `act` 返回后调用 `save_life("life.npz", brain)`，再用 `restore_life("life.npz")` 从相同检查点恢复。快突触、感觉活动、运动单元、随机数进度和已生长的局部地址一起保存。环境需要提供同一时刻接下来的观察。

复验跨游戏时，可在同一条生命内先学习两处规律，再辨认反复出现的新物体，最后返回旧规律。附加的 `local-selective` 读写策略让同一经验支持“原物再次出现”和“变换后的物体出现”两种读取，由当前线索的匹配程度与自身实际预测误差决定如何读、如何写。两种可选关系由研究者提供，五个系数按真实行动结果在开发世界选定。

```powershell
python -m lifelong.lifelong_agent --game cross-game --address local-selective --worlds 24
```

