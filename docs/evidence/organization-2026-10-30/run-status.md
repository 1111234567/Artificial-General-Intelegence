# Operational continuation record

The study completed on2026-10-30. This operational log preserves earlier in-progress snapshots below; the final status is recorded at the end. Scientific findings belong to the main report and passed aggregate artifacts.

## Frozen running studies

- Acquisition: `acquisition-protocol-v1.json`, data root `F:/agi/data/organization/acquisition-v1`. Nine CUDA runs change only shrink to0 from matched old lifelong branches. 4CP/run,216 new2048tick forks,216 borrowed old forks. Queue exec session9233. Limits file currently training9,cpu8,total9; nine training jobs active at this note. Use actual OS liveness and execution JSON before any restart.
- Organization: `organization-protocol-v1.json`, data root `F:/agi/data/organization/main-v1`. 27 CPU runs,108CP,648 new2048tick forks. `mechanical_test=false`, training_device=cpu. Queue exec session7488 with `CUDA_VISIBLE_DEVICES=-1`, OMP/MKL1. Limits initially3 then4 shared workers, minimum free memory1.5GiB. All main variants use the same device. Protocol selected before inspecting any new formal behavior score.
- Both queues reuse frozen `experiments/lifelong/queue.py`, with respective `acquisition-*` and `organization-*` training/evaluation/empty-baseline plans. Restores preserve world/h/optimizer/replay/pending/RNG. All execution files and explicitly pinned tests are frozen. Do not edit them while jobs run.
- Source files excluded from execution pins and still editable for analysis: acquisition_report.py/test_acquisition_report.py, report.py/test_report.py, plot.py, README.md and project documentation.

## Validation to date

- Acquisition runner9 tests passed; acquisition report7 tests passed.
- Organization CPU runtime selection:70 passed,3 CUDA-skipped;2 outdated test fixture assertions were corrected and both passed separately, then full CPU selection passed. Added formal-CPU preparation test afterward; study8 tests passed.
- Functional runner5 tests passed with CUDA visible, including full pending/window/phase exact resume for CPU and CUDA across all functional policies and plain/coupled fields. GPU allocator cap1.5% applied only to these tiny mechanical tests while formal GPU jobs ran.
- Organization report16 tests passed with CPU-only tiny27/108/648 data,122.54s. No formal behavior scores read for parameter selection.
- Root first redundant CUDA integration pytest was stopped due memory pressure; identified processes archived in interrupted-integration-test-processes.json. Formal training processes were not stopped.
- An initially generated organization plan used the mechanical template API, which selected8tick probes. It was never executed; original four JSON files are archived in unexecuted-mechanical-cpu-prepare with a record. Fixed study.prepare/CLI to accept a formal training_device argument, tested CPU remains2048 and full budget, then prepared the current formal plan.

## Ready analysis commands

After all acquisition9 training+evaluation jobs succeed:

```powershell
python -m experiments.organization.acquisition_report --protocol docs/evidence/organization-2026-10-30/acquisition-protocol-v1.json --root data/organization/acquisition-v1 --output docs/evidence/organization-2026-10-30/acquisition-aggregate-v1
```

Keep CUDA visible for the acquisition auditor: the formal CUDA checkpoints include CUDA RNG, even though models restore to CPU. Wait until enough resource is free. It is serial and restores new36CP plus old birth CP; it verifies216 borrowed raw probes.

After all organization27 training+evaluation jobs succeed:

```powershell
python -m experiments.organization.report --protocol docs/evidence/organization-2026-10-30/organization-protocol-v1.json --root data/organization/main-v1 --output docs/evidence/organization-2026-10-30/aggregate-v1
```

Main data is CPU; CVD=-1 is valid. Report emits header.json and CSVs for plotting without loading full JSON. plot.py/README are being prepared by topology_design_ultra; growth_design_ultra authored main report, structure_research_ultra authored acquisition runner/report and research notes. All agents explicitly Sol6.1 Ultra, never Astra.

## Remaining work

Finish both queues; verify actual process/statuses, source bindings and no missing branches. Run strict reports, address analysis-only failures without changing executed code. Generate and visually inspect figures. Complete Chinese docs/organization-experiment-2026-10-30.md with honest n3 results and actual compute, capacity, lifetime and old-history damage. Update current project indices, README, changelog. Save completion manifest binding all data/audit/test/figure sources; run check_docs and git diff --check. Preserve old frozen evidence and the preexisting dirty tree; no commit/deployment requested.

Interpretation: growth/exit share a mixed task objective but can individually worsen historical loss. Candidate groups are tested jointly, not each edge proven harmless. No hard one-for-one replacement or capacity ceiling; K4 is a search/probe budget. Exact numerical support and eligibility are distinct. Main's block-diagonal birth differs from old lifelong's random cross-bank birth. The nonlinear comparison matches state, parameters and matrix MAC; functional policies additionally pay4 probes per pending window. Initial bank labels are not evidence of spontaneous functional modules. Assess acquisition at131072 before claiming forgetting or retention.

## Subsequent operational updates

- User clarified a separate GPU workload had been active. GPU memory dropped from about7.9GiB to3.8GiB. The isolated-v2 benchmark suspended only this study's30 launcher/worker/queue processes, kept all states resident, measured CPU/GPU pairs, then resumed all30 after82.2766s. Journal active=false and resume count verified. No source or scientific budget changed. Results seconds/update: nativeCPU .3232/CUDA .5400; plainCPU .4186/CUDA1.0428; coupledCPU .4448/CUDA1.1122. Each pair has identical initial parameter hash, batch32,width128,chunk128,warmup4+measured12 updates. This is whole-program throughput, not general GPU capability. GPU allocations of suspended jobs remained resident. Training elapsed counters can include this pause.
- Main CPU shared limits raised4→6 after memory checks; still CVD=-1 and one thread/process. Acquisition remains CUDA and total9.
- Acquisition seed001 fixed/narrow alone remained at birth while siblings reached98304+. Actual worker143800 had only one thread and increasing CPU time; py-spy0.4.2 read a healthy sibling but could not find its Python interpreter. Saved restart-v1 record, original checkpoint SHA and profiler outputs. Terminated only this verified owned worker; independently restarted same frozen acquisition train command from existing latest.pt (birth state) in exec session4251, output retry_seed_001_fixed_narrow.log. Original acquisition queue9233 will mark that old attempt failed, finish its other running jobs, and exit nonzero without launching new evals. **Wait for both original queue and independent replacement before restarting the same frozen queue**; it will validate completed markers and continue evaluations. Never launch a duplicate while replacement is live. Preserve failed attempt status/history; no scientific seed/setting change.
- Main current study runtime validation:70 CPU passed/3 CUDA skipped plus corrected fixture tests; formal CPU study8 passed; functional runner5 CPU/CUDA integration passed. Main report16 passed and acquisition report7 passed. All formal execution sources remain pinned. Latest organization protocol explicitly mechanical_test=false,CPU,2048 probes; archived mechanical plan was never run.

## Final status

- Both queues complete: acquisition9/9 training and9/9 evaluation; organization27/27 training and27/27 evaluation; empty baseline queues complete. All current execution return codes0. Original failed attempt remains archived by the queue; independent retry completed and the restarted coordinator validated its completed marker without another learning pass.
- Main CPU concurrency increased to12; acquisition evaluation used2 concurrent CPU jobs. One-thread model execution and immutable budgets remained unchanged. No scientific seed was replaced or selected by score.
- Acquisition strict audit passed1,649,206 checks using3workers; organization strict audit passed7,413,293 using6workers. Parallel reports were separately checked against serial output on real tiny fixtures. Both formal aggregate directories are complete.
- The final new-module test suite contains112 unique cases, all passed with CUDA visible, no skips/failures/errors, in354.53s. Incremental runs are overlapping evidence, not additional unique tests.
- All three scientific figures are rendered and visually inspected; plots/CSV statistics/input SHA independently checked. Main report and project indices contain both positive and negative results, bounded by3trained seeds. The report includes fixed-field parameterization, finite candidate-budget growth cap and group-level ablation limitations.
- Scientific work and archival checks for this iteration are complete. Documentation validation passed45current documents and624local links with0issues; git diff --check passed. completion.json binds the final artifacts and checks. Future training-length, curriculum, event-budget, birth-only/fade-only and functional-clustering studies are recommendations, not secretly executed extensions.
