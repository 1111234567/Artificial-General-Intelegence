# 连续状态与旧经历整合：证据入口

[研究报告](../../integration-experiment-2026-10-27.md)解释任务、学习规则、结果与限制；[运行说明](../../../experiments/integration/README.md)给出复现命令。

## 正式实验

- [冻结协议](main-study-v1.json)：三个种子、12 个分支、65,536 个继续训练 tick、五个最终测试条件及预先指定的诊断干预。
- [汇总与逐种子结果](aggregate/summary.json)、[汇总审计](aggregate/audit.json)：36 个训练终点、180 个自主分支、72 个额外诊断分支；243,590 项检查通过。重复基线只核对一致性，不增加统计样本。
- [逐种子指标](aggregate/seed_metrics.csv)、[学习预算](aggregate/budgets.csv)、[成对比较](aggregate/paired_comparisons.csv)、[诊断干预差值](aggregate/diagnostic_delta_summary.csv)。
- [真实轨迹的内容读出](aggregate/own_state_content_by_phase_age.csv)、[首次错误与联合模式](aggregate/first_error_counts.csv)、[结束后的活动](aggregate/inactive_history.csv)。条件读出与自主动作不同；缺少合法机会的读出保持未定义。
- [执行汇总](execution/execution-summary.json)、[最终检查点审计](execution/final-audit.json)、[文件与图表核验索引](execution/index.json)。完整检查点和原始诊断文件仍在 `data/integration/`，汇总保存其 SHA256 和路径；本目录不复制大体积权重。
- [旧控制轨迹复现](execution/prior-current-control-reproduction.json)：`coverage_long_dense` 与上一轮 `B_current` 的参数、h、Adam、动作和数据相同，不增加独立证据。

## 预实验与后续只读观察

[预实验索引](preflight/index.json)记录独立种子 101 的数值选择协议、真实 CUDA 保存恢复、旧检查点探索性诊断和观察器版本一致性。旧检查点探索只使用一个种子，不与正式三种子结果合并。

用户提供的论文见[可塑性研究笔记](paper-plasticity-notes.json)。[只读终点观察方案](state-inspection-plan.json)在正式训练启动后、读取终点状态前记录，是明确标注的事后诊断。其[结果](state-inspection/summary.json)、[概览](state-inspection/overview.csv)和[验证](state-inspection/validation.json)覆盖全部 36 个最终检查点；不推进模型、环境或优化器，不用于模型选择。单个终点的活动饱和与协方差谱不能独立证明长期可塑性丧失。

`report-layoutcheck` 等合成排版测试保存在 `data/integration/`，不属于正式结果。图表使用三种子均值与样本标准差；actor、神经元和同一检查点的多个分叉不作为独立训练重复。
