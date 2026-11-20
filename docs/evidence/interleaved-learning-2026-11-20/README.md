# 交错学习证据索引

完整复算最终审计时，用 Git LFS 取得[审计数据与检查点](audit-data.zip)，解压到仓库根目录，再运行 `python -m experiments.pulse_game.reporting.interleaved_audit`。两条固定三角形回放示例所需的个体和参照轨迹也随普通检出提供。[审计范围](final-audit.json)

从[主报告](../../interleaved-learning-2026-11-20.md)进入。原训练出生为 731–736，独立复验出生为 811–816。每项结果的分母以对应协议为准，同一世界内交叉的私有规律保留配对。逐轨迹数组和检查点存于本地 `data/interleaved-learning-20261120`，结果文件记录相关指纹。

## 能记住多少件事

| 问题 | 试验 | 证据 |
| --- | --- | --- |
| A 的回访受等待、接触还是 B 的新写入影响？ | 冻结图干预。比较正常写入、接触禁写和等时空等，从同一状态分别测试 A 与各个 B。 | [容量曲线](capacity.png) · [冻结容量结果](frozen-capacity.json) |
| 末题从最近对象改成任一已见对象，联合答对率会变吗？ | 两组续训。原出生另换测试世界，新出生另训复验。 | [原出生与换世界](retention-objective-summary.json) · [新出生](retention-objective-confirmation-summary.json) · [出生与世界交叉区间](capacity-crossed.json) |
| 末题效应来自较早回访，还是来自一次接触就要学会？ | 先冻结旧模型测四格，再以四种相同接触预算的目标续训，比对两队列的主效应和交互。 | [旧模型四格诊断](retention-factorial-original-diagnostic.json) · [两队列四臂结果](retention-factorial-cross-cohort.json) |
| 复习 A 时，B 的表现怎样变化？ | 冻结图分叉。真实复习与同味觉、等时禁写暴露配对。 | [原出生复习分叉](retention-objective-matched-008000.json) · [新出生复习分叉](retention-objective-confirmation-008000.json) |

## 能在怎样的环境里用上经验

| 问题 | 试验 | 证据 |
| --- | --- | --- |
| 一个身体能同时答对冲突场景的 A、B 和给料器标记吗？ | 训后物理留出。按真实接触端点计分，三项从同一状态分别分叉。熟悉气味的冲突子集每出生有 96 个世界根、384 条轨迹。 | [方法](heldout-method.md) · [原出生逐项对照](heldout-physical-endpoint-contrasts.json) · [新出生冷启动](heldout-replication-cold/summary-primary-aba.json) · [新出生续训](heldout-replication-cont-full-direct/summary-primary-aba.json) |
| 直接奖励三项全对会怎样？ | 单独续训比较联合成功目标与逐项平均目标，分别测原生动作与物理端点。 | [训练与分割协议](terminal-quest/protocol.json) · [原生结果](terminal-quest/native-summary.json) · [物理结果](terminal-quest/physical-summary.json) |
| 当味觉有噪声或私有规律反转时，图如何更新？ | 可靠性训练设有干净、随机滑误和混合反转三种条件。干净训练图另做冻结滑误及再访问诊断。 | [训练规程](../../../experiments/pulse_game/interleave_reliability.py) · [本地训练清单](../../../data/interleaved-learning-20261120/reliability-fit/run.json) · [干净训练图的诊断](receipts-trained-clean.json) |
| 有关联的旧经历能帮助初见对象，也会妨碍纠正例外吗？ | 冻结出生参数，在新生命中正常写入；另用相同动作及味觉交换所配对的气味。 | [归纳和例外的原、独立出生结果](induction-checked.json) · [内容交换](transfer-binding-uniform.json) · [内容交换复验](transfer-binding-uniform-independent.json) |

## 写入通路怎样承载这些结果

| 问题 | 试验 | 证据 |
| --- | --- | --- |
| 更好的写法和适合写入的出生图各起多少作用？ | 冻结组件交叉移植；移植后仍允许生内写入。 | [原出生](transplant-original.json) · [独立出生](transplant-independent.json) |
| 一笔 B 的写入会怎样改动 A？ | 冻结图干预，按气味夹角、价值关系和改变量配对，回退部分改动。 | [方向与剂量](interference-objective-uniform-8000.json) · [单行回退](row-credit-objective-uniform-8000.json) |
| 长历史后还能学新对象吗？ | 冻结图分量恢复、限定写入和外加慢参考，分别测旧 A 与新 X。 | [可塑性诊断](plasticity-prior-odor.json) · [长期分量对照](structure-objective-uniform-8000.json) · [慢参考复验](slow-reference-contrasts.json) · [混合长生命复验](integrated-stability-joint-confirmation.json) |
| 三个零和气味对能分别保留各自的价值吗？ | 冻结图交叉八种私有历史，另做直接、渐变与长程续训。两种运动方向从同一状态读取。 | [冻结读出](code-probe-objective-uniform-8000.json) · [两种 8k 续训](triangle-objective-summary.json) · [长程训练、独立出生与全部顺序](triangle-orders-long-independent-32000.json) |

## 固定终点与独立复核

- [17 件和 33 件的实际容量](wide-capacity.json)汇总两批出生、各两套同预算对照与原始轨迹校验。
- [物理混合行为](mixed-results.md)列出真实运动的场景冲突与无提示反转；另列连续三次选侧、最后才结算收益的原生试验。 [原组通用能力卡](model-cards-original.json) · [复验能力卡](model-cards-confirmation.json)
- [三角形等待压力](triangle-wait-stress.md)和[接收行移植](triangle-row-graft.md)覆盖 12 个独立出生。 [原组角度响应](triangle-angular-original.json) · [新组角度响应](triangle-angular-independent.json) · [逐拍反应图](triangle-angular.png)
- [压缩出生参考](compact-reference.json)、[逐地址静态量化](quantized-reference.json)和[动态慢副本的实际改码](dynamic-quantized-reference.md)检验哪种空间细节必须保存。[奖赏学得的稳定力度](adaptive-reference.json)与[迁移到混合生命](adaptive-deployment.json)给出原生 A/B/X 实测。
- [反馈可靠性的外部读出](meta-evidence-state-readout.md)区分图的状态含有何种信息，以及实际行动是否用上了它。
- [具体个体的动作与味觉](triangle-lived-example-64000.md)附逐拍运动记录和复现指纹。[冻结评估的前缀共享测速](triangle-prefix-bench.md)给出动作复现与训练数值误差。
- [原组三角形专训后的跨任务实测](triangle-transfer-original.json)与[新出生复验](triangle-transfer-confirmation.json)在同一个体上测旧能力、跨对象归纳和普通对象记忆。

- [同预算的几何课程与两技能交错](curricula-and-joint.md)给出难度调整、训练生命间的交替及同一生命的六项联合分母。

- [同一生命内的两次有偿行动与六项记忆](one-life-results.md)区分真正的串答收益、同快照六项联合及旧技能退步。[长短时钟和同反馈复查](one-life-padding.md) · [出生身体与写法互换](one-life-body-rule-swap.md) · [主要轨迹与最终检查点核验](final-audit.json)

- [把信用给短任务真实收益的修复](one-life-repair.md)比较等预算长题训练及长短混合训练。

[首轮计划](plan.json) · [后续试验](followups.json) · [独立复验计划](confirmation-plan.json)
