# 感觉、动作和实际味觉共用一张可塑图

16 个节点在同一张强连通有向图中逐拍更新。8 个节点分别接收两侧物体的 4 路连续浓度，2 个动作节点的瞬时状态比较直接给身体左、右脉冲，4 个接触与味觉节点接收身体的实际回执，其余 2 个节点只接收图内信号。所有节点都有输入与输出，使用各自随机出生的更新速度和偏置。出生接线由随机采样得到，边有独立的带符号权值，生命内没有固定的环或受保护的边。图内信号与身体动作同时推进，每个请求结束后保留神经状态、权值和接线。[模型](../../../../experiments/pulse_game/unified_live_graph.py) · [运行器](../../../../experiments/pulse_game/unified_live_graph_study.py)

每个生命从随机出生开始。在 640 次实体取食中，身体最多行进 15 拍，抵达一侧才读取该物体的一份随机味觉。两侧物体独立抽取，正负味觉按当前生命的私有线性偏好随机产生。负味觉也可能发生在期望更好的物体上。学习器每次只收到自身接触的侧和味觉。环境在隐蔽的时点改变偏好方向，浓度、距离、风和探索电流的外生序列均固定。没有规则名、情境标志或另一侧味觉的输入。末 48 个请求照常移动和计分，同时停用公开味觉与预置的探索电流，测量已获得的能力。[预登记](registered.json) · [独立开发配方](development.json)

味觉和接触节点在接收回执当拍的响应给出标量学习信号。更新规则通过刚才的图状态计算活动期间两个动作节点平均差的截断梯度，并按有无回执调整各边独立权值。在缺边位置，根据梯度的移动平均选择新增边。退边优先考虑弱边或梯度支持减少的边，并逐次校验强连通。增、退决策独立，学习和结构编辑可以在同一生命中持续发生。信用分配依赖预先指定的单请求反传规则和试次时序。

## 生命内习得、反转和恢复

确认试验独立交叉 8 个出生根和 12 个物理世界。每种情境、每个试验臂有 96 个配对生命。每个试验臂驱动自己的身体。以实际最先接触的一侧是否拥有更高期望为正确率，超时计错。安静测验各情境联合臂的接触率为 100%。[完整分段、接触和随机实得味觉](summary.json) · [逐生命核验](verified.json)

| 情境与窗口 | 联合学权重和改线 | 只学权重 | 全冻结 | 无味觉 | 独立随机味觉 |
|---|---:|---:|---:|---:|---:|
| 稳定，首 64 次 | 51.42% | 50.99% | 49.87% | 49.87% | 49.90% |
| 稳定，末 48 次安静测验 | 83.81% | 73.85% | 50.28% | 50.28% | 49.44% |
| 反转前 64 次 | 61.13% | 55.49% | 50.72% | 50.72% | 49.92% |
| 反转后首 64 次 | 42.32% | 45.88% | 50.08% | 50.08% | 51.22% |
| 反转后末 48 次安静测验 | 82.36% | 72.59% | 49.70% | 49.70% | 50.56% |
| 恢复原偏好，末 48 次安静测验 | 70.79% | 63.82% | 50.28% | 50.28% | 49.44% |

稳定末窗联合臂比全冻结高 33.53 个百分点，比独立味觉高 34.38 点。反转末窗差值分别为 +32.66 与 +31.79 点。四项预登记对比的配对 95% 同时区间下界分别为 27.15、25.53、23.75、23.90 点。三个情境的 288 个联合生命发生了 8,928 次增边和 9,792 次退边，所有新增边在后续回执下至少获得一次有别于被动衰减的权值改写。逐事件重建了连接，所有出生图、退边后及终图均保持强连通。变点前的同臂轨迹逐项相同。无味觉臂和独立随机味觉臂的接触侧、接触拍和权值演变不受真值变点影响。[配对区间](summary.json) · [手段与轨迹核验](verified.json)

## 同频随机改线与端口传递

另一批 8 个新出生根和 12 个新世界比较选边规则。对照在接线组每次增边、退边的同一时刻随机选择合法地址，照常依据自身真实回执更新权值，并优先从可删除的弱边中抽取退边。每种情境精确匹配 2,976 次增边和 3,264 次退边。安静测验中，主动选边比随机选址高 6.10 点（稳定，95% 同时区间 [1.78, 11.06]）和 7.96 点（反转，[3.41, 13.59]）。[扩展预登记](ablation-registered.json) · [配对数据](ablation-summary.json)

固定实际观测做逐拍反事实。每个抽查窗口内的 16 个节点均有可测的状态变化。隔离 108 次真实回执时，下一拍其他节点受回执的最大绝对状态变化为 0.007–0.166。仅切断味觉发送弧后，这一单步变化精确为零。仅切断动作发送弧也改变了其他节点状态，并改变了 4 个运动拍的输出。另做实际闭环安静测验，两种发送弧切断都未检出准确率损失。安静测验中的切断检验覆盖获得能力后的使用阶段，味觉发送弧在持续学习中的作用仍需单独测量。[固定观测干预](summary.json) · [实体受控切断](ablation-summary.json)

## 复现

当前工作区的 [主试验逐生命输出](../../../../data/pulse-game/unified-live-20261119) 和 [扩展逐生命输出](../../../../data/pulse-game/unified-live-ablation-rerun-20261119) 保存实际接触、接触拍、味觉、逐次编辑及起终接线。摘要包含各 NPZ 的 SHA-256。扩展首轮的全部 NPZ 写入后，汇总写入因缺少导入而失败。修复汇总写入后用相同种子完整重跑，8 个原始文件逐字节相同。[首轮登记](ablation-first-registration.json) · [哈希复核](verified.json)

```powershell
$primary = 'F:\unified-graph-replay'
$ablation = 'F:\unified-graph-replay-ablation'
.venv/Scripts/python.exe -m pytest -q experiments/pulse_game/tests/test_unified_live_graph.py
.venv/Scripts/python.exe -m experiments.pulse_game.unified_live_graph_study register --out $primary
.venv/Scripts/python.exe -m experiments.pulse_game.unified_live_graph_study develop --out $primary
.venv/Scripts/python.exe -m experiments.pulse_game.unified_live_graph_study confirm --out $primary
.venv/Scripts/python.exe -m experiments.pulse_game.unified_live_graph_ablation register --primary $primary --out $ablation
.venv/Scripts/python.exe -m experiments.pulse_game.unified_live_graph_ablation run --primary $primary --out $ablation
.venv/Scripts/python.exe -m experiments.pulse_game.unified_live_graph_verify --primary $primary --ablation $ablation --out "$primary/verified.json"
```
