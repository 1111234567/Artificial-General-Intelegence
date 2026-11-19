# Undirected study operational continuation

## 2026-11-02 收尾复核

10:15 最终收尾及发布已完成。`completion-v4` 复验了在上一次报告内存门槛触发前保存的 21 份分支记录，完成余下 21 份并汇总。全部六个收尾阶段标为 verified，原固定图对照的逐位一致性标为 failed。中文表与发布记录的 SHA 一致，17 幅图同时提供 PNG 和 SVG。[正式结果](../../undirected-results-2026-10-30.md) · [最终状态](../../../data/undirected/completion-v4/finish-status.json)

42 个终点训练、42 份冻结评估及 42 份独立历史校验均已完成。第一次报告处理完所有分支，在父进程比较固定图与回滚探测固定图时退出，发布助手随即停止。原日志及原分析源码已保存在 `data/undirected/completion-v2`。原定逐位一致性条件在种子 0 失败。[原检查日志](../../../data/undirected/completion-v2/report-001.log)

随后读取了配对的实际张量和评估记录。最初记录的差异在 144,896 tick 的信用状态诊断，原报告比较的字段在 145,024 tick 首次不同。三个种子的实际训练动作均相同。种子 0 的部分内容诊断随学习状态出现分歧，起因未确定。[逐块配对核验](fixed-probe-pair-audit-2026-11-02.json)

修订只涉及报告层。原对照预期的结果继续标为 failed，来源、分母和分支内轨迹校验分别报告。独立历史校验以原清单和各输出的 SHA 复验后复用。报告的完整世界与模型轨迹复核按分支保存，支持在后续汇总或绘图失败时重用已验证内容。

Work in progress. Protocol date2026-10-30. User explicitly requested genuinely undirected connections and both interface arrangements, challenged128tick truncated credit, and confirmed the comparison was retire-only versus grow+retire. Do not treat a later status/speed message as cancelling the study.

## 2026-11-01 续跑与交付入口

- 本次恢复先校验 11 个训练终态，再从原状态续跑其余分支；已完成者不重训。按实际可用内存分批恢复，从 8 训＋2 评逐步提高。11 月 1 日晚设置总上限 15，其中训练最多 14、评估最多 3；旧评估积压排完后可将空出来的槽给训练。另有一条独立历史校验，变更时间保存在各队列的 `limits_history`。
- 一次性资源助手 `data/undirected/post-queue-budget-20261101.py` 只等待本次已绑定身份的队列退出，不监视或更改任何模型。仅当 42/42 训、42/42 评和空基线均执行完毕、后处理仍在进行且审计预算未被人工改动时，它才按释放的实际可用内存将审计并发由 1 提至最多 6；保留原有 4 GiB 准入门槛。队列中断或收尾失败则不提高并发。
- 源码和数据约定见原 [协议](protocol-v1.json)。后处理改用 `data/undirected/analysis-v2/analysis-plan-v1.json`：窗口组每块只做一次真实状态前向重建，零次 JVP；RTRL/reset 仍做两方向 JVP，并与保存的完整敏感度投影核对。旧版分析源码已归档；17 项分析测试、12 项报告测试通过。
- 已在后台启动一次有限的后处理任务。进度见[收尾状态](../../../data/undirected/completion-v2/finish-status.json)及[训练队列](../../../data/undirected/main-v1/training-execution.json)、[评估队列](../../../data/undirected/main-v1/evaluation-execution.json)。待正式训练/评估与独立校验都齐备，会依次生成审计表、图、中文结果表，并把实际文件路径写入收尾状态。科学结论以完成的产物为准；进度文件中 `complete=false` 不是完成凭证。
- [正式结果入口](../../undirected-results-2026-10-30.md)在收尾完成并复核实际成功标记、报告/图表/结果文件指纹后由一次性工作区发布助手更新。如果用户在此期间主动编辑该入口，其内容会保留，核验的结果仍在收尾目录。助手不发送通知或外部消息。
- 若再次重启机器，先检查进程和队列状态。训练使用原三份命令清单；新收尾进程需绑定实际存活的新训练队列 PID，并选新的收尾目录。不要清空训练目录或同时启动第二个队列。训练和后处理均使用项目 Python、`CUDA_VISIBLE_DEVICES=-1` 与单线程 CPU。

## Frozen current execution

- Workspace F:/agi; project Python F:/agi/.venv/Scripts/python.exe, Python3.14.5/Torch2.11.0+cu130. All subagents explicitly GPT6.1 Sol Ultra (never Astra).
- Formal `protocol-v1.json`, training/evaluation/empty-baseline command lists in this evidence folder. Raw root `F:/agi/data/undirected/main-v1`.
- Queue `python -m experiments.undirected.queue ...` running in exec session71246. Environment CVD=-1, OMP/MKL/OpenBLAS=1. All formal models CPU float64. Initial shared maxworkers14, minimum free RAM1.5GiB; dynamic limits at rawroot/worker-limits.json.
- All execution modules and their explicitly pinned tests are frozen. Do not edit model/credit/plasticity/run/study/queue/ledger or pinned tests while any job runs. New report/plot/sensitivity analysis files are excluded from execution pins and can be corrected independently.
- Before any restart inspect actual OS processes and execution JSON; never duplicate live jobs. Numerical failure is a scientific terminal (no score0, no replacing seed/resetting S); execution failure remains nonzero and is resumed from the same last complete state.

## Fixed design

**2026-11-01 18:50 CST — resume after host restart.** No training, evaluation, or coordinator process from the original launch remained. All 14 previously recorded open PIDs were dead, the 11 successful training output sets/terminals validated, and all 26 protocol source hashes were unchanged. No execution failure was reclassified as a numerical outcome. Live manifest metadata for the interrupted branches was consistent; the frozen runner replays any progress after its last durable state and checks the resulting records against previously committed bytes. No clearing of training history, hidden state or sensitivity.

Started exactly one new hidden coordinator through the project interpreter (launcher PID 60252), using the original three command plans and status root. Launch logs `data/undirected/main-v1/queue-resume-20261101-185045.{stdout,stderr}.log`. At launch about 13.4 GiB available RAM; initial limits 8 training plus 2 evaluation workers, 3 GiB dispatch threshold, to admit the expensive restores gradually. CPU/F64/CUDA visibility, 1 thread per worker, branch budgets and data unchanged. Evaluation of completed branches now overlaps ongoing training. The old 120 s A/B/A affinity trial restored every affinity mask; no CPU pinning inherited on resume.

80 persistent nodes, B8 actors, two concurrent job slots, four contextual rules, input56/output22. Ports case has no trainable encoder/head/bias: input56 nodes, output22 nodes, two without external ports; all80 evolve and interact. Core case has trainable interfaces initialized to the same fixed selectors. Normalize public duration/16, window/2.5 and pulse gain.25. No labels or private timers enter input.

Undirected has one shared coordinate per unordered pair; untied control adds a skew coordinate. Wuv=(s+a)/sqrt2,Wvu=(s-a)/sqrt2, with a absent in undirected and0 atuntied birth. No layers/banks, diagonal0 as experimental configuration. Regular16 connected birth =640pairs, same physical W/function across interfaces/modes perseed; world birth additionally grouped bycourse. Gain.6 sets Frob rowRMS atbirth, no later spectral projection. Independent parameters: ports3160/6320, core9422/12582.

Main196608; warmup course usesnarrow first65536 thenwide; wide-only control never switches coverage. Withdrawfuture family0 for32768 thenreturn8192; total237568,1856 updates. Snapshot ticks0,65536,131072,196608,229376,237568. Five nonzero snapshots ×4family×short,long_both×2048 =40fork/run.14variants×3seeds=42runs,252planned canonical CP,1680planned forks.

Variants: four core/ports×undirected/untied at recorded512/fixed/warmup; ports undir tbptt128/rtrl/rtrl_reset fixed/warmup; ports untied rtrl/fixed/warmup; ports undir rtrl fixed_probe/grow/retire/both warmup; ports undir tbptt128/rtrl fixed/wide. Not a full Cartesian factorial.

Live credit:128 update cadence,128 suffix loss; recorded512 uses actual oldparameter snapshots with shared currentdelta and restores actual primal. RTRL stores full S[B,H,Pdyn] through all blocks, no diagonal approximation/trace clipping/reset. Reset control computes sameJ/K/J@S but zerosprevious contribution. Derivative conditions on actualparameter path, actions/world/optimizer/structure decisions; not a meta-gradient through them. Aux remains common warmup128+loss128; do not call it unbounded.

Pair policies share full-coordinate Adam moments, project inactive parameters to0 atreal boundaries; no mask straight-through. Four probe policies run same4 actual forward losses onnextblock; only birth/fade acceptance switches. Full deletion is always diagnostic rollback; finitehalf/zero can be accepted bymixed loss. No one-for-one or constant edge count. K4/window8 is finite search amplitude. Fixed(no probes) vsfixed_probe should have same learning effects up to declarednumerics, but differentwork.

## Evidence before formal launch

- Eight-module runtime suite126passed,18 warnings (Torch/Python3.14 JVP JIT deprecation), no skips/failures. XML/log in evidence.
- credit-path-check.py/json: one cue attick0, onlylast128 losses at1024–1151,8parameter changes; finite128/512/1024 encodergrad0, RTRLnonzero andfullrecordBPTT match~2.7e-15. Mechanical slowmode, not learned behavior.
- Production-smoke v1 found restore validator sorting mistake (proposalrank vsactualcanonical pair order). No training update mistake: rawevent weights agreed. Fixedvalidator+regression; original artifacts preserved.
- Production both32updates v2 matchesoldv1 model/Adam/h/world/operator/credit exactly; onlyvalidator changed. Separately fixedoptimizer loader aliasing caller CPU moment tensors withdeepcopy.
- Final-source smoke-v3: actualwarmup/rtrl/both H80/B8/F64,32updates thenexact33rd restoration, savedinput unchanged, whole32-record independentJVP matchescheckpointS. JSON/log saved.
- Stable maxabs-scaled L2 calculations avoidfalse overflow onfinite1e200 gradients; no S clipping/reset. TypedFPE onlyfor runtime numerical failure; callbackIO/schema/OOM/interruption classifiedexecution.

## Full recorded history and independent checks

Every successfulupdate archives preparams/buffers, actualobs/targets/h0/hend/dt/actions/context asatomic gzip(mtime0), withcontenthash andappend-only manifest. Ledger is audit-only, neverextra learner input. It supportsahead record verification/reuse andcallbackfailure recovery. Canonical CP retainfullS; do not saveS everyupdate.

Independent command:
`python -m experiments.undirected.sensitivity_audit --terminal <branch>/terminal.json --output <audit.json>`
It uses torch.func.jvp at each recordedactualprimal and carries2private Rademacher direction tangents across ALL blocks, comparesCP S@direction. No creditengineJacobian/update called, no optimizer/world rerun. Two projections are numericalcalibration, not everyentryproof. Finite-window/noS andbirth-only cases explicitlyskipped forfullS claim. DefaultF64 atol1e-9/rtol1e-8. Sourcepinexecution remainsunchanged.

## Remaining work

Finish42terminal jobs andall expected evaluations (numerical failures haveemptyrecords andendpoint incomparable). Report author undirected_research_ultra ownsnew report.py/test_report.py; it is stillbeingwritten, no formal scores read. Requireall42accounted, realfullrestores, same-tick latest/canonical identity, actualmodel/world/labels/exposures/rawledger integrity; no meanunless3/3successful seeds withdefined denominator. Plan independentperbranchworkers1–6 aftertraining.

Core agent will prepare README/plots; report schema must be shared beforeplotting. Needactualgraph symmetry/parameter counts, interfaces, credit andcost, fourstructuralflags, curriculum, failurecounts, n3seed statistics. Do not inferfunctionalmodules frominitialregulargraph ordrawinglayout.

After dataaudit, runindependentJVP onallappropriate RTRL/reset terminals inboundedCPU pool, bindresults; then render/inspectfigures. Complete docs/undirected-experiment-2026-10-30.md, updatecurrentindices/changelog, createcompletionmanifest andcheckdocs/gitdiff. Existingoldstudies andhuge preexistingdirtytree arepreserved; no commitrequested. Do not endwithonlytrainingstarted ifworkcancontinue.
