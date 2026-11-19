# 持续输入、活动学习与结构变化

这组实验接续 [persistent](../persistent/README.md)。它分别检验有限记忆的读写与停止、顺序冲突任务中的局部学习，以及连接新建和退出。不同分支的反馈与参数更新方式不同，不能把它们合称为已经实现的终身学习系统。

三种子结果与长流测试见[完整报告](../../docs/streaming-experiment-2026-10-26.md)，指标和图见[证据目录](../../docs/evidence/streaming-2026-10-26/)。

## 持续事实流

[stream.py](stream.py) 生成随机上下文中的双链查询。各上下文使用相同的键但有不同的答案，符号、事实顺序、链长、干扰和第三阶段回到哪个任务均随机化。旧任务返回时先查询，再重新提供或修订事实。正确答案和路径长度只用于训练损失与评估，不进入模型。

[cache.py](cache.py) 包含有限的原始记录槽位、需求历史状态、共享内容寻址器、保留评分与停止头。学习型写入选择保留、替换或丢弃；空槽分配及查找同一个 `(context,key)` 是固定的缓存接口。写入前向使用真实完整记录，训练用 straight-through 导数。需求时间戳来自已经发生的请求，不是重要性标签；它们是显式提供的历史特征，不能声称由网络从裸输入学出。

读取时不给正确步数。训练允许停止概率混合的替代损失或期望候选损失，并加入逻辑读取成本；部署使用首次越过停止阈值的候选。正式指标只看硬停止后的答案。批量推理没有裁剪已经停下的行，因此逻辑读取次数不等于 GPU 计算量节省。

[run_stream.py](run_stream.py) 每个完整任务循环后更新参数，只截断计算图，继续携带缓存和需求状态。评估流中参数冻结，记录状态仍持续变化。这个边界很重要：评估的是已经训练过的记忆策略，不是评估阶段持续修改全部参数的自主系统。`mru` 是保留最近两个被请求上下文的明确规则，作为强基线；FIFO 和随机写入用于干预诊断。

大型配置为 12 个上下文、64 个符号、每个上下文 8 条事实、16 个记录槽位、256 维需求状态，最多 10 次读取，共 431,109 个参数。正式训练 batch 为 512。数据与动作采样分别管理；奖励组的初始/最终测试与长流比较成对，ST 与奖励组的训练随机流不同。固定最后检查点用于测试，不按测试结果选模。

```powershell
.venv/Scripts/python.exe experiments/streaming/run_stream.py --output data/streaming/NEW-common --policies mru --seeds 0,1,2 --steps 800 --log-every 200 --batch 512 --eval-batch 256 --contexts 12 --vocab 64 --nodes 8 --slots 16 --hidden 256 --width 256 --read-dim 128 --max-reads 10
```

用 `--initial-dir` 指向共同训练目录，可以从相同读取器开始比较写入策略；加载会校验种子、架构、任务配置和模型/任务源码。`--writer-only` 只训练保留评分与需求状态，必须配合 `--policies learned`。不加它时进行联合更新。两种训练目标由 `--halt-loss mixture|expected` 明确选择。

### 后续训练与诊断

读取训练完成后，可以继续本流模型的停止和写入训练：

```powershell
.venv/Scripts/python.exe experiments/streaming/calibrate_stop.py --input data/streaming/NEW-common --output data/streaming/NEW-stopping --steps 200
.venv/Scripts/python.exe experiments/streaming/reinforce_writer.py --initial-dir data/streaming/NEW-stopping --output data/streaming/NEW-rl --seeds 0,1,2 --steps 200 --batch 512 --eval-batch 256 --eval-cycles 8 --log-every 50 --device cuda
.venv/Scripts/python.exe experiments/streaming/evaluate_long.py --input data/streaming/NEW-rl --output data/streaming/NEW-long --policy learned --device cpu
```

[calibrate_stop.py](calibrate_stop.py) 仅训练停止头；[reinforce_writer.py](reinforce_writer.py) 仅用后续查询的实际二值正确性训练写入，冻结读取器/停止头，没有答案交叉熵或自评 critic。[evaluate_long.py](evaluate_long.py) 检验连续 64 个循环；[diagnose.py](diagnose.py) 只读核对链覆盖、正确槽位注意力和分深度成绩，oracle 仅用于离线诊断。

## 活动相关的局部更新

[local_learning.py](local_learning.py) 比较冻结、频率稳态、局部 delta、真实二值奖励调制的局部资格迹，以及单层反传。相同输入在 A/B 上下文的标签冲突，按 A→B→A 顺序训练。表征已经提供上下文交互项，所以本分支没有验证表示学习。单层局部 delta 与交叉熵 SGD 数学等价；二值奖励规则是局部可实现的策略梯度估计，不能称为无梯度原理。

```powershell
.venv/Scripts/python.exe -m experiments.streaming.local_learning --seeds 0,1,2 --phase-steps 800 --eval-every 10 --output data/streaming/NEW-local
```

## 结构变化

[structure.py](structure.py) 是已完成的小规模固定预算对照：固定稀疏、学习重连、随机重连始终具有相同的 96 条非零连接，另列稠密可学性对照。它回答连接位置能否影响学习，不足以回答新旧连接应怎样交接。

[structure_free.py](structure_free.py) 已完成较大对照：1,024 维原输入、32 类、65,600 个潜在权重，允许连接数净增减。已有边逐步减弱至零才退出；固定小网络、立即预算替换、峰值容量静态网络及稠密网络共同区分容量与过渡冲击。

```powershell
.venv/Scripts/python.exe experiments/streaming/structure_free.py --output data/streaming/NEW-structure-free --seeds 0,1,2 --input-dim 1024 --classes 32 --n-train 262144 --n-val 8192 --n-test 8192 --steps-per-phase 3000 --batch-size 2048 --fan-in 256 --lr .2 --connection-cost .0001 --window 50 --device cuda
```

连接数量、出生/退出次数、纯删除反事实冲击及新旧任务曲线均已保存。实现使用稠密数组记录候选统计，没有稀疏计算加速的声明。

## 运行与证据

使用本项目 `.venv` 中的 PyTorch；GPU 版本为 2.11.0+cu130。小规模局部/结构对照使用 CPU，大型流实验使用 RTX 4060 Ti。检查点和逐步日志位于被 Git 忽略的 `data/streaming/`；所有运行拒绝覆盖非空输出目录。

```powershell
.venv/Scripts/python.exe -m pytest experiments/streaming -q
.venv/Scripts/python.exe experiments/streaming/reinforce_writer.py --self-test
.venv/Scripts/python.exe experiments/streaming/summarize_stream.py
.venv/Scripts/python.exe experiments/streaming/summarize_structure.py
```

导出脚本默认读取本轮正式目录，输出 JSON 与 PNG/SVG。新运行使用新的目录，导出时相应调整数据目录映射，避免覆盖已有结果。

数值平滑、替代梯度、读取成本、学习率、连接更新间隔都是本次实现选择，不是项目通用约束。先前 pilot 包括任务捷径修正前的版本以及未成功的联合训练，不能混入正式测试统计。
