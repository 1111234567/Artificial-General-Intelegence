# Exact block RTRL execution evidence

The opt-in backend is `experiments/coverage_exit/fast_credit.py`. It replaces
the block state/parameter Jacobian construction with a dense analytic reverse
pass through every Heun substep. It expands the same potential coupling matrix
once, carries all output-by-state adjoints, and sums the matrix-weight outer
products over time with `bmm`. Shared/skew coordinate projection is unchanged.
Trainable core interfaces are supported, including the encoder derivative at
each post-flow observation pulse; fixed ports remain ordinary evolving nodes.

The native engine still computes the live loss and its reverse derivatives,
the full cross-boundary correction, and `S_next = J @ S_previous + K`. The
runner still owns the realized world/actions, auxiliary gradient, equal live
and auxiliary combination, clipping, Adam, and structural projection. No
diagonal approximation, trace truncation, sensitivity clipping, lower dtype,
AMP or TF32 is introduced. `jacobian_reverse_chunks` is now zero for the
analytic path, with actual analytic reverse-substep counts added.

For a Heun substep `f = -x + W tanh(x)`, `y = x + d f`,
`g = -y + W tanh(y)`, `z = x + .5 d (f+g)`, let `M` be the complete
final-state derivative with respect to `z`. Then the implementation uses:

```
g_bar = .5 d M
y_bar = (g_bar W) diag(1 - tanh(y)^2) - g_bar
f_bar = .5 d M + d y_bar
x_bar = M + y_bar + (f_bar W) diag(1 - tanh(x)^2) - f_bar
W_bar += g_bar outer tanh(y) + f_bar outer tanh(x)
```

There is one output-coordinate axis in every bar tensor; no state coupling is
dropped. Matrix-coordinate derivatives are then projected to all independent
shared/skew coordinates, including currently zero potential edges.

Floating reduction order changes, so the contract is FP64 numerical
equivalence, not universal bitwise identity. Every actual-dimension comparison
records J/K, full S, live/auxiliary/combined gradients, model and Adam state,
hidden state and action/world/data invariants separately. Close checks use
`atol=3e-12, rtol=3e-11`; finite values are checked too. The 25 scoped tests
cover both interfaces and both graph parameterizations, multiple/zero solver
steps, pulse timing, zero-valued edges, changed-parameter blocks, large finite
state and S, clipping/Adam, installation and resume. Latest project-interpreter
run: `25 passed in 5.81s`.

## CPU actual-dimension calibrated starters

Project interpreter `.venv/Scripts/python.exe`, Torch `2.11.0+cu130`, CPU,
FP64, H80/B8/T128, one thread, affinity logical CPU0. Other study agents were
in preparation rather than formal benchmark phases. Each source-faithful
native starter performs one calibration update, then every measured call
starts from the same calibrated state. One warmup per backend and four
alternating AB/BA repetitions; medians in seconds:

| Ports mode | Workload | Native | Analytic | Native / analytic |
| --- | --- | ---: | ---: | ---: |
| shared | block J/K | 2.81698 | 0.28553 | 9.87 |
| shared | complete credit + S | 2.73240 | 0.28524 | 9.58 |
| shared | complete native train_update | 2.98501 | 0.73183 | 4.08 |
| untied | block J/K | 3.49131 | 0.18589 | 18.78 |
| untied | complete credit + S | 3.82844 | 0.32512 | 11.78 |
| untied | complete native train_update | 4.33430 | 0.80618 | 5.38 |

Raw evidence: `data/coverage-exit/perf/block-rtrl-cpu-ports.json`.
Its exact measured backend source, before the work-counter-only adjustment,
is retained as `fast-credit-measured-v1.py` in this directory; its SHA256
`38bcf6b6ccd3c1e02a1aa67556a4e5cbabe69aca6ce24f283734c722ff1549ce`
matches the result's backend hash. That initial measurement did not record
process memory peaks. No historical peak is inferred after process exit.

## CUDA real checkpoint and memory guards

Source checkpoint:
`data/undirected/main-v1/training/seed_000/ports__undirected__rtrl__fixed__warmup/checkpoint_tick_131072.pt`.
The saved native source and dependency hashes match the current archived
implementation. It contains 1024 completed updates and full S with maximum
absolute value `1.569426455451371`. H80/B8/T128, FP64, batch and solver
unchanged. Both backends were compared on the same GPU after explicit CPU to
CUDA checkpoint migration, with strict source validation.

Separate fresh-process guards on one complete update observed CUDA peak
allocated memory of 146,079,744 bytes (native) and 331,703,808 bytes (analytic).
Peak reserved memory was 190,840,832 and 369,098,752 bytes respectively. The
analytic coefficient cache uses more CUDA memory in this case. Process peak
working sets were about 1.55 and 1.53 GiB; these process-lifetime figures also
include imports and checkpoint validation. Cold guard times of 4.946 and
3.402 seconds are not the warmed speed comparison.

The actual CUDA comparison uses four alternating AB/BA repetitions after one
warmup per backend. CP load/validation/clone, adapter installation, collection
and pre-call device synchronization are outside timing. Timed calls synchronize
at completion; these are end-to-end workload times, not CUDA-event kernel-only
times. Host affinity is logical CPU18, an individual physical core according
to Windows core topology, whereas the CPU calibrated-starter benchmark uses
CPU0. External GPU use was roughly 3.1 GiB and 44-55% utilization, with
authorized concurrent CPU research work. No external process was changed.
The GPU results are conditional on that environment; the CPU/GPU time ratio
must not be interpreted as a pure device effect.

| Real shared-port CP workload | Native | Analytic | Native / analytic |
| --- | ---: | ---: | ---: |
| block J/K | 2.01921 | 0.24016 | 8.41 |
| complete credit + S | 2.56372 | 0.72430 | 3.54 |
| complete native train_update | 4.80938 | 3.04396 | 1.58 |

All numerical checks passed over actual consecutive updates ending at ticks
131200, 131328 and 131456. Hidden state, model parameters, actions, world,
data and replay were bitwise identical for these sampled updates. Live
gradient maximum absolute difference was `1.30e-17`; full-S maximum absolute
difference was `4.88e-15`. J/K maximum absolute differences were `5.96e-19`
and `4.44e-15`. This does not establish arbitrary-length bitwise equivalence.

Raw evidence and exact measured source snapshots:

* `data/coverage-exit/perf/block-rtrl-gpu-real-cp.json`
* `data/coverage-exit/perf/block-rtrl-gpu-memory-native.json`
* `data/coverage-exit/perf/block-rtrl-gpu-memory-analytic.json`
* Corresponding `.fast-credit-source.py` files beside each result.

## CPU real-checkpoint comparison and thread scheduling

The same source-validated tick-131072 checkpoint was measured on CPU0, one
thread, with one warmup and four alternating AB/BA repetitions. Original
formal A/B training had finished; the remaining project audit was restricted
to logical CPU12, and study C had not started. External system activity was
left unchanged. A one-second observation during the window is saved in
`block-rtrl-cpu-window-load-sample.json`; CPU0's sibling CPU1 was at 4.8%,
while some other logical processors were busy. Available RAM in that sample
was about 8.4 GiB. The comparison is conditional on this machine state.
Recovery/validation/clone, installation, collection, and preparation are all
outside the timed calls.

| Same real shared-port CP workload, CPU | Native | Analytic | Native / analytic |
| --- | ---: | ---: | ---: |
| block J/K | 2.68384 | 0.22393 | 11.98 |
| complete credit + S | 2.91611 | 0.28582 | 10.20 |
| complete native train_update | 2.85284 | 0.66297 | 4.30 |

The three actual continuation updates ending at 131200, 131328 and 131456
passed all finite FP64 close checks. J was bitwise equal; K maximum absolute
error was `4.44e-15`, S `4.88e-15`, live gradients `1.47e-17`, and model
parameters `6.94e-18`. Hidden state, actions, action/data hashes, world,
auxiliary source and replay were bitwise equal in these sampled updates.
These real-CP updates had clip_scale=1 in both backends; separate tests and
the calibrated-starter benchmark cover active clipping. Process lifetime
peak working set for the entire comparison was 1,331,892,224 bytes, with
peak private commit 2,502,733,824 bytes. This joint-process peak cannot be
attributed independently to either backend.

Raw evidence: `data/coverage-exit/perf/block-rtrl-cpu-real-cp.json`, with exact
backend and benchmark source snapshots beside it.

The optimized-only thread test uses that same CP, one job at a time, one
warmup per count, and four timed rounds ordered `1,2,4`, then `4,2,1`, then
`1,2,4`, then `4,2,1`. Windows physical-core masks confirm that logical
0/1, 2/3, 4/5, 6/7 are four different SMT core pairs. Actual affinities are:

| Threads | Logical CPU affinity | Full train_update median seconds | Process CPU usage range |
| ---: | --- | ---: | ---: |
| 1 | 0 | 0.60314 | 65-94% |
| 2 | 0,2 | 0.53002 | 169-185% |
| 4 | 0,2,4,6 | 0.52724 | 342-368% |

The first one-thread timed call was a 1.204-second outlier during broad
system activity, with CPU1 at 77.2%; the other one-thread calls were
0.589-0.615 seconds. All samples are retained. The medians show about a
1.14x improvement for two versus one thread in this measured single-job
environment, and nearly identical two/four-thread latency despite roughly
double process CPU use. This does not measure parallel-job throughput or
establish a universal thread-count optimum. Model/h/S/action outputs for the
warmup checks at all three thread counts were bitwise identical. Complete
per-logical-processor load samples for every timed call are included in
`data/coverage-exit/perf/block-rtrl-cpu-thread-sweep.json`.

## Four independent jobs on the same four physical cores

`experiments/coverage_exit/schedule_credit.py` is a separate scheduling CLI;
the previous benchmark was not changed for this measurement. Every
configuration restores four independent copies of the same learned RTRL CP.
All four are loaded and each receives one warmup update before the total wall
clock starts. Every copy then executes three complete optimized FP64 native
`train_update` calls, so every configuration completes exactly 12 measured
updates. Normal native update bookkeeping is included. Additional comparison,
tensor hashing and state serialization begin after all four completion
messages and are excluded from the total timed interval.

The budget is always the four verified physical cores represented by logical
CPUs 0,2,4,6. `1x4` runs one job at a time with four threads and all four
CPUs; `2x2` runs two at a time with affinities [0,2] and [4,6]; `4x1` runs
all four with one thread and affinities [0], [2], [4], [6]. Four processes
are preloaded in all configurations, including the configurations with only
one or two active jobs. The coordinator uses logical CPU12. The first round
orders `1x4,2x2,4x1`, and the second reverses that order.

| Configuration | Round 1 wall seconds / updates per second | Round 2 wall seconds / updates per second | Both rounds pooled updates per second |
| --- | ---: | ---: | ---: |
| 1 job x 4 threads | 14.4338 / 0.8314 | 11.8413 / 1.0134 | 0.9134 |
| 2 jobs x 2 threads | 6.0384 / 1.9873 | 5.5331 / 2.1688 | 2.0741 |
| 4 jobs x 1 thread | 3.6126 / 3.3217 | 2.8497 / 4.2109 | 3.7138 |

All 24 final payloads have the same content hash. Model parameters, hidden
state, full S and Adam state are bitwise equal across these sampled jobs and
thread settings; action sequence hash, final action tensor, world, data and
clock also agree. The predetermined FP64 close tolerance is unchanged, with
observed maximum absolute errors zero for these comparisons. The retained
`validation-reference.pt` is the compact final reference. Per-job scratch
payloads were compared and individually removed after their hashes and
comparison outcomes were recorded.

Admission required at least 8 GiB system RAM available. The run would abort
if available RAM fell below 2 GiB or sampled aggregate worker RSS exceeded
8 GiB. Observed maximum sampled sum of four worker RSS was 3.876 GiB, and
minimum system RAM available was 9.192 GiB. Per-worker lifetime peak memory
before extra validation is also recorded. Aggregate RSS includes shared
mapped pages in each process; it is not a unique physical-memory count.

**This was not an exclusive-machine measurement.** The selected cores'
other SMT threads, logical CPUs 1,3,5,7, had sample-average utilization of
roughly 85-98% across cases. Actual four-worker CPU use was about 295-351%
when measured as process CPU seconds per total wall second. External system
activity was retained and every timed interval records per-logical-CPU load.
A read-only post-run observation found another project's training process,
`E:\Carbon League\...\train_bc.py`, started before this window and allowed on
all 20 logical CPUs. It was not paused or modified. That post-run process
observation is separate from the load samples, and does not assign exact
historical utilization to that process. The ranking is consistent in both
rounds, but the exact rates are conditional on this load and four preloaded
processes. They do not predict an entire-machine optimum or scaling beyond
this four-physical-core budget.

For this machine and this measured H80/B8/T128 FP64 workload:

* For one latency-sensitive job, two threads on logical CPUs [0,2] use about
  half the CPU time of four threads with nearly the same measured latency
  (0.530 versus 0.527 seconds in the separate single-job experiment).
* For four independent jobs under this four-core budget, use four processes,
  one thread each, assigned logical CPU0/2/4/6 separately. Its measured
  throughput exceeded the other two configurations in both scheduling rounds.

Raw evidence:

* `data/coverage-exit/perf/block-rtrl-four-core-scheduling.json`
* `data/coverage-exit/perf/block-rtrl-four-core-scheduling-summary.json`
* `data/coverage-exit/perf/block-rtrl-scheduling-external-process-observation.json`
* `data/coverage-exit/perf/block-rtrl-four-core-scheduling/` contains exact
  scheduling/backend source snapshots and the validation reference.

Reproduce the fixed-budget test with:

```
.venv/Scripts/python.exe -m experiments.coverage_exit.schedule_credit \
  --checkpoint data/undirected/main-v1/training/seed_000/ports__undirected__rtrl__fixed__warmup/checkpoint_tick_131072.pt \
  --cores 0,2,4,6 --coordinator-cpu 12 --updates 3 --rounds 2 \
  --output data/coverage-exit/perf/block-rtrl-four-core-scheduling.json
```

## Research training opt-in

```
from experiments.coverage_exit.fast_credit import install_fast_credit, backend_manifest

trainer = NativeTrainer.from_checkpoint(source_checkpoint)
install_fast_credit(trainer)
# Save backend_manifest() alongside research output, then use normal updates.
trainer.train_update()
```

The helper preserves and validates the complete native credit state, including
S, tape and age. It changes only that trainer instance. Archived files are
not replaced. Native checkpoint format remains readable; explicitly reinstall
the helper after restoring an optimized research run, and preserve its separate
backend source manifest. Unknown models/dtypes use the native engine path;
the explicit trainer-install helper rejects an unsupported model/dtype.

Reproduce the real-CP comparison with:

```
.venv/Scripts/python.exe -m experiments.coverage_exit.benchmark_fast_credit \
  --checkpoint data/undirected/main-v1/training/seed_000/ports__undirected__rtrl__fixed__warmup/checkpoint_tick_131072.pt \
  --interface ports --mode undirected --device cpu --cpu 0 \
  --continuation-updates 3 --warmups 1 --repeats 4 --workload all \
  --output data/coverage-exit/perf/block-rtrl-cpu-real-cp.json
```

Use `--device cuda --cpu 18` for the recorded GPU workload, and record external
load through `--environment-note`. The isolated memory path is
`--probe-backend native|analytic --probe-phase trainer` in separate processes.
Thread scheduling is an independent optimized-job test with
`--thread-sweep 1,2,4 --cpus 0,2,4,6`; each count receives that many distinct
physical cores rather than oversubscribing one logical CPU. Use the same
`--checkpoint`, `--output`, `--warmups 1` and `--repeats 4` arguments.
