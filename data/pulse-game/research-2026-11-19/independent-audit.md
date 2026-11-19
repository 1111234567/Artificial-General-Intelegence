# 独立复核 · 旧题本桥接、时延搜索及圆轨度量

2026-11-19，北京时间。复核了 `legacy_bridge.py`、`legacy_bridge_fork.py`、`bridge_controls.py`、`attractor_search.py`、`attractor_gpu.cu`、`ring_metric*.py`，并沿到实际环境计分代码。以下复现使用 CPU。CUDA 部分检查了接口、共享内存屏障和更新顺序。

## 会改变结论范围的项目

| 工件 | 核对结果 | 适用解释 |
|---|---|---|
| 旧题本静默 | 签名回执可屏蔽，但答对仍原生退休并清除公开 `READY`。仅置换私有目标后，同一静默代理有 846 个 actor-tick 的 `READY` 和 852 个 actor-tick 的动作不同，两组写权均为零。 | 「屏蔽 ACK/REJ」保留任务退休这一公开后果。冻结权重是独立的写入对照。 |
| `rewarded-interference` 分叉 | 同一原生后缀重放给两条分叉，后缀反馈绑定参考轨迹。在一个 8 生世界的 71 条后缀回执中，ACK 枝 2 条、REJ 枝 3 条前驱命令和参考不同，其中两枝各 2 条实际写入了当前分支的不同地址。 | 固定后缀的受控干预，或共同的参考轨迹回执重放。两个分支各自获得闭环真回执需要独立地重新执行世界。 |
| 时延搜索 | `encode_lives` 先算完整生命的编码，核接口只含公开感觉与时间。`score_lives` 先记录所选答案，再比较私有目标并只对选择建立的权重列使用一位回执。 | 公开脉冲到固定时刻的符号决策实验。原题本适配仅复用 NEW、首次 READY 和记分目标，原生旧题本和圆轨实际行动另有闭环实验。跨代遗传的是 15 个发育参数，每代重新抽图。 |
| 圆轨训练回执分母 | `event_batch(events, trials)` 保留每生命前 `trials` 条真实咬合事件。4 生、每生 12 题的完整重试中，实际 108 口，感知到的 108 条回执均匹配上一拍咬合，其中训练张量只保留 48 条。 | 重试训练只使用按咬合排序的前缀；训练文件的 `event_count` 是保留事件数。实际评估首次正确率仍以全部题目计，超时按错。 |
| 化学混训分配 | `ring-meta-chemical-mse` 初始 `[40]` 时 onehot/novel 全是单口，anchored/noisy 全是多口，各 320 生。`ring-meta-flash-chemical` 的 novel `[1,4]` 中，一拍全单口、四拍全多口，各 512 生。 | 独立评估的环境操纵可以配对比较。训练配方之间的成绩差不能单独归因于化学风格、曝光时长或多口训练。 |

## 角标、私有状态与分母

- 旧世界桥接实际读主端口 `NEW=0`、`RAW=5:9`、`READY=10`、`ACK=22`、`REJ=23`、`CONTEXT=24:26`。试验题本 64×18×8，所有公开 RAW 都在本生 NEW 脉冲，未退休时 `READY` 逐元素等于 `stage==2`，每题四个可作答 tick。`BridgeAgent.act` 只接收公开帧；在真回执测试中 751 次写入全部匹配前一拍的 `READY` 合法 FINISH，额外写入和漏写都是零。
- 两个完整新旧世界种子的严格率和完成率分别为 `.7500/.9570`、`.7461/.9414`。装配函数自身会核对动作重计和原生计分，两个结果通过。
- 旧归档末代与桥接确认使用同一组 32 个世界，都是每世界 8 生、16 visits、每生 64 题。每个确认臂有 128 条唯一 `(world, graph)` 行，配对汇总的独立抽样单位是世界（32），图在世界内重复。桥接开发所记的原 ring 配置指纹与当前文件吻合。
- 时延搜索中，将四生 `serial_noise` 课表第 9 题起的私有目标循环置换，特征、放电率和前 9 次选择完全相同；随后有 10 个选择改变。关闭回执时所有选择相同。另对旧世界适配验证 18 tick、NEW、首次 READY 和目标编码与原题本逐元素相符。
- 圆轨使用 `[L,R,bite]`、`REACH=12`、`POSITIVE=13`、`NEGATIVE=14`、`TASTED=15:19`、`NEW=19`、`ACTIVE=20`。同一快权 checkpoint 在 4 生×12 题置换私有目标、屏蔽回执时，单口的首咬身份、时刻和运动逐元素没变。允许多口后，真成功仍退休，运动记录 48 个元素变化。完整重试里 108 条签名与身体的 TASTED 同拍且与上一拍有效咬合匹配。
- `ring_metric.Metric.predict_events` 的批量形式只使用严格下三角 Gram 项，蒸馏目标是实际咬到的签名。原有单元测试核验顺序预测、梯度及未来标签因果性；没有看到训练给未咬果实标签的路径。
- 三份圆轨 odd 确认档案共用同一 128 世界，分别与各自训练/刷新种子无交集，checkpoint 指纹当前一致。比较风格与 checkpoint 时应按这批世界配对。错配回执以相邻两世界为一抽样簇，所以单档是 64 个世界对。5 个度量训练目录已归档的初始及刷新种子在各目录内均无重复，训练感觉文件指纹均与当前源码一致。
- 复核时已有的 `attractor-*/search.json` 每个检索内无世界种子重复，也无和开发验证 96 世界的交集。同一种子下递归与断环的共用世界可配对，重复开发验证分数不作为独立确认。世界种子留出与整对私图值留出是两个概念；圆轨 metric 用 A 排列奇偶实现后者，时延搜索只按世界种子留出。

## 当前实现中仍需留意的边界

- `legacy_bridge.py` 保存了开发时的 ring 配置指纹，确认入口没有主动比对新传入 `--ring-file`。本次实际文件相符；自行复跑时需要按归档指纹核对。`bridge_controls.PublicMemory` 是逐键消除参考，未利用四答案排列和 A/B 齐异的全局结构。
- `ring_metric_eval.py` 的相邻错配用 `index ^ 1`，给奇数世界会越界；正式 128 世界不触发。`ring_audit.life_metrics` 的 A/B 配对率直接除以配对数，如果用极短测试题本，配对分母可能为零。将首咬、完成率、接触率与 A/B 配对率各自保留定义。
- 脚本化 `shadow` 回执来自同生命的独立伪映射；该映射没有原任务的排列约束，因而是错目标条件，不等同于保持真实正负比例和结构的跨生命错配。

## 可复现检查

下面是时延搜索的私有目标反事实。直接在项目根目录以安装了 NumPy、Torch 的解释器运行。

```python
from dataclasses import replace
import numpy as np
from experiments.pulse_game.attractor_search import RANGES, cpu_answers, cpu_features, task

tape = task([19741, 20271, 20331, 20557], 'serial_noise', trials=16)
parameters = np.full(len(RANGES), .5)
z, spikes = cpu_features(parameters, 12, tape)
goals = tape.goal.copy()
goals[:, 8:] = (goals[:, 8:] + 1) % 4
alternative = replace(tape, goal=goals)
z2, spikes2 = cpu_features(parameters, 12, alternative)
assert np.array_equal(z, z2) and spikes == spikes2
first = cpu_answers(parameters, z, tape)
second = cpu_answers(parameters, z, alternative)
assert np.array_equal(first[:, :9], second[:, :9])
assert np.count_nonzero(first[:, 9:] != second[:, 9:]) == 10
assert np.array_equal(cpu_answers(parameters, z, tape, feedback=False),
                      cpu_answers(parameters, z, alternative, feedback=False))
```

旧世界静默对照用 `local_rule_world.schedule(seed=202771, actors=8, visits=16)`，将 `target` 替换为 `FIRST_FINISH + (target - FIRST_FINISH + 1) % 4`，两边都用第 3 号出生、已选 `continuous_lif` 且 `feedback='none'`。通过 `private.rollout_with_table(_receipt_table('cpu').half())` 录下 `obs[0,:,READY]` 和主动作后比较。分叉复验从同一 seed/图的第 33 题建立 `native_inputs`，依 `legacy_bridge_fork.group` 的两种回执分叉重放，并在每个干扰帧比较上一拍分支动作与原录制动作。

圆轨对照用 `ring-meta-flash-odd` 记录的 checksum checkpoint、`make_tape([1722, 2091, 2966, 3200], trials=12)`、`cue_ticks=1`，重复 `replace(tape, goal=(tape.goal+1)%4)`。实际事件截断计数用 `MetricForager(..., explorer=True, record=True, appetite=Appetite(random_bite=.43))`、`retry=True`，对照 `result.bites.sum()` 和 `event_batch(actor.events, tape.trials)[2].sum()`。
