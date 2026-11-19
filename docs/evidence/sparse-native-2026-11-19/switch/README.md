# 从跨组乘积切换到组内乘积

前一段 A 的味觉取决于气味坐标的四种乘积：0×2、0×3、1×2、1×3。B 仅依赖 0×1 和 2×3，各世界使用新抽取的带符号系数。图只接收原始浓度和自己取得的味觉。两条孪生生命从完全相同的 A 状态出发，在 B 的新世界中有相反的味觉偏好。所有新取值都保留在物理世界内部，模型没有得到变化通知。[预登记与选型规则](protocol.json)

8 个出生和 8 个新世界根交叉。每次测量从已教学状态复制主体，给 80 个新食物对，禁止公开味觉和探索驱动，按首次接触的实际食物计分。分叉的末状态与静默食物不会带回主生命。B 第一次教学前，同一算法的两个孪生逐题接触侧和时刻完全一致，而 B 要求它们选不同侧。继续生长时，640 次 B 教学的静默成绩为 82.36%，暂停新增二阶系数为 73.48%，两者都仍可修改已有系数和普通边。回 A 并再次教学 640 次后，分别为 75.22% 和 75.15%。[配对区间与计数](summary.json) · [逐生命分叉归档](raw.zip)

另两项试验分别控制旧交互的库存，以及早年在 A 是否得到过有效教学。[保住或复用历史地址](history.md) · [未经复习的旧规律回访](reacquire.md)

用下列命令在仓库根目录复算。第一次准备时从证据目录恢复独立算法开发选中的配方。仅核验发布的原始数组可使用最后一条命令，它还会对首两个出生和全部算法重新运行整段生命。[完整核验记录](checks.json)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m experiments.pulse_game.sparse_switch_verify prepare
python -m experiments.pulse_game.sparse_pair_switch run --workers 4
python -m experiments.pulse_game.sparse_switch_verify verify
```
