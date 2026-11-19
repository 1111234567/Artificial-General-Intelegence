# Independent frozen-code review and regression verification

Review date: 2026-10-30. Scope: the frozen lifelong execution files, their critical continuity/replay/native semantics, and the combined regression suite. No frozen execution source or frozen test was changed by this review. The only code correction was to the non-frozen report test fixture, as described below. Existing evaluation processes were left running.

## Frozen-source identity

All 29 source entries in `study-protocol-v1.json` and both source entries in `coverage-probe-supplement-v1.json` matched their recorded byte SHA-256 digests during this review. Main protocol SHA-256: `7cb8cb76eee72759bb12687242fe4221705387cf32280ed8e08d4b6cbbda86e8`; supplement: `140903657a65833567e397b8adde6e89ea25f7e88a76a402e01ed77854323a69`.

## Critical semantics

No blocking runtime defect was identified in the reviewed paths.

- **Real contextual rules and input separation:** `world.py:31,125-174` maps private targets using four explicit permutations, while replacing the public payload observation with the raw symbol. Context accompanies NEW/REPLACE once and is absent from idle observations. Both agreement and conflicting cases exist between the mappings. This is a shared request protocol with contextual payload rules, not four unrelated domains.
- **Continuous curricula:** `schedule.py:110-145` selects only future NEW families. `world.py:116-123` changes policy metadata, and `run.py:329-335` applies it immediately before the next real observation. Pending modifications retain their assigned context. No curriculum boundary resets the world, live hidden state, clocks, optimizer, reservoir, or RNGs. Complete-state restoration uses `__new__` rather than rerunning native birth (`run.py:591-690`).
- **Causal historical replay:** `replay.py:224-290` admits contiguous real history into nonoverlapping 256-tick blocks and uses separate admission/sampling RNGs. The trainer samples only blocks whose endpoint is at or before the current live chunk's start. It admits the current chunk after the update (`run.py:395-410,495-515`). Original observations, actions, elapsed times, stored initial hidden state, and original oracle labels are retained. Replay does not generate new world feedback or replace the live state.
- **Matched auxiliary compute:** `run.py:412-427,503-513` computes both gradients before any parameter update. Except for the shared first-update startup, each auxiliary source has a detached 128-tick warmup and a differentiable 128-tick suffix. Live and auxiliary normalized-loss gradients are averaged 1:1, then eligibility is applied, then one joint clip and one Adam step occur. Recent-repeat recomputes its own recent observations under current parameters; historical replay remains an off-policy replay of the original receipts and labels, not a new closed-loop rollout.
- **Native structural policies:** `run.py:263-312` builds random models with a private 32-input-per-row recurrent mask and rescales by 2 at width 128. The three policies share the same initial sparse function per seed and differ from their first update. `factors/structure.py:67-68,111-163` preserves the distinction between allowed coordinates and current numerical support. The growth mechanism retains its declared dormant shadow Adam and scalar global Adam-age handoff; the review did not alter either.
- **Frozen pre-return evaluation:** `evaluate.py:297-342` clones the full model/world/hidden continuation under inference mode, changes only future draws, and uses the model's native eleven-action argmax. It checks parameter and source-state hashes afterward (`evaluate.py:550-555`). Inherited, carryover, primary, censored, ordinary/strict completion, content readouts, and idle errors remain distinct. Main and supplemental execution wrappers verify snapshot byte and semantic hashes before evaluation and preservation afterward (`study.py:155-176`; `supplement.py:60-85`).

## Independent checks of existing training records

Read all 30 final `report.json` files and each branch's existing trace through tick 163840. All reports were complete at 172032 ticks, with zero reported hidden resets. Each seed had exactly one initial parameter hash across all ten conditions.

All 30 branches had the same full work vector: 172032 live forward/loss ticks; 343936 auxiliary forward ticks; 171904 detached auxiliary warmup ticks; 172032 auxiliary loss ticks; 1344 optimizer steps; 2688 backward calls; 344064 BPTT graph ticks. This verifies the logged nominal compute match, not equal hardware wall time or sparse-kernel savings.

Using trainer boundary tick 163840 minus the zero-based saved timestamps gives command-index gaps of 32740–57338 and live-supervision-index gaps of 32608–57311. Clock clarification from the final aggregate review: the actual world-clock convention is `world.tick - last_tick`, giving 32739–57337 and 32607–57310 respectively; the main report uses this convention. Pending contracts can survive the nominal withdrawal boundary, so these measured recencies are more precise than asserting exactly 32768 ticks without any family-0 feedback. Historical replay continued to supervise family 0 during withdrawal; that is the intended experimental manipulation and is separately clocked.

At tick 1024 (update 8), the matched mixed/history-replay/wide branches had active recurrent edge counts as follows:

| Seed | Fixed | Grow | Open |
| --- | ---: | ---: | ---: |
| 0 | 4089 | 15282 | 16078 |
| 1 | 4093 | 14935 | 16074 |
| 2 | 4095 | 14844 | 16060 |

Fixed eligibility was 4096, open eligibility 16384, and grow eligibility followed its active mask. The fast early expansion is observable from the original update trace. Native structural results compare complete update policies with different capacity trajectories; they do not isolate topology placement at equal active capacity.

## Recoverable structural history and lifetime limit

The update trace retains each 128-tick update's births/deaths and row counts, capacity/density, cumulative event counts, continuously surviving initial-edge counts, ever-born location counts, and support hashes (`run.py:521-537`; `rewiring/plasticity.py:250-281`). Full checkpoints additionally retain exact support, initial/survival/ever-born masks, and dormant shadow buffers.

The existing records support update-resolution event-count/capacity curves, initial-edge continuous survival at saved times, unique ever-born positions at saved times, and coordinate-level net changes between the 13 saved checkpoints. They do **not** directly retain every edge's birth/death timestamp or all repeated activation cycles, so a complete per-edge active-lifetime distribution or mean has not been measured. A currently dormant dynamic-growth edge's `shadow_count` describes its latest dormant spell; it is not the lifetime of an active connection. Prior to another birth, that counter can locate the latest retirement under the frozen dynamic rule, but it does not recover the full edge history. Support hashes alone cannot invert the intervening masks. Report rapid expansion and high event turnover without claiming measured individual connection lifetimes.

## Combined regression

The previous factors final JUnit contains 598 cases from continuity, credit, factors, integration, loops, and rewiring. This run uses those same directories and adds lifelong, with OMP/MKL/OpenBLAS/NumExpr/VecLib thread limits set to 1:

```text
.venv\Scripts\python.exe -m pytest experiments\continuity experiments\credit experiments\factors experiments\integration experiments\loops experiments\rewiring experiments\lifelong -q --junitxml=docs\evidence\lifelong-2026-10-29\tests-final.xml
```

Attempt 1 completed with **714 passed, 2 failed, 0 errors, 0 skipped; 716 total**, in pytest wall time **405.88 s** (JUnit suite time 405.868 s). All 598 prior `(classname, name)` case identities were present, with exactly 118 lifelong additions. Directory counts were continuity 55, credit 93, factors 81, integration 76, loops 183, rewiring 110, and lifelong 118.

Both failures were in the non-frozen report integration tests: `test_formal_protocol_preparation_and_source_set_are_independently_audited` and `test_all_supplement_files_are_required_before_any_expensive_checkpoint_audit`. They failed at `report.py:506` with `COMPLETE_PROTOCOL_SOURCE_SET`. A read-only reproduction found that today's synthetic `study.prepare` glob collects the newly added analysis modules `finalize.py` and `plot.py`; those two modules are outside the report auditor's permitted main/supplement source sets. There were no missing required sources and no frozen-runtime test failures. The existing formal protocol's original 29-source set still matched exactly. This analysis/test integration issue was reported to the owning agent for correction without modifying frozen `study.py`.

Attempt-1 evidence is preserved in `tests-regression-attempt-001.xml` and `tests-regression-attempt-001.log`.

The authorized correction changes only `experiments/lifelong/test_report.py`: its synthetic protocol fixture locally limits `study.prepare`'s lifelong source glob to the original 14 native execution/test files; the unchanged preparation function then adds its original 15 dependencies. All three synthetic preparation sites use that helper. The existing formal protocol test now additionally checks exactly 29 sources and rejects injected `plot.py` or `finalize.py` entries under `COMPLETE_PROTOCOL_SOURCE_SET`. The strict report auditor, runtime modules, and saved formal protocols were not changed.

The focused report suite passed **28/28 in 9.50 s**, with evidence in `tests-report-fixture-fix.xml` and `tests-report-fixture-fix.log`. These 28 are a rerun subset, not additional unique tests.

The final combined regression passed **716/716 in 310.10 s** (JUnit suite time 310.078 s), with **0 failures, 0 errors, and 0 skipped**. JUnit timestamp: `2026-10-30T08:07:33.205838+08:00`. All original 598 `(classname, name)` case identities are present, with exactly 118 lifelong additions and 716 unique identities. The final directory counts match those above. Evidence: `tests-final.xml` and `tests-final.log`.

After the completed regression, all 29 main and 2 supplemental frozen source digests still matched; both saved protocol hashes remained unchanged. The corrected non-frozen `test_report.py` SHA-256 is `e9a9262f95b884b0ff814a46544a3e265b9bfa591a2f1cc1c25a4fc9b5e88cda`. No unresolved regression failure remains in this scope.

| Evidence | SHA-256 |
| --- | --- |
| `tests-final.xml` | `950cd1337910fc81a42f60c4def2ad211a2e979a244127c2788687adf17618bc` |
| `tests-final.log` | `a79c0e75c8d348ab695ac1b53d2107f7c9ed4505d2195e39a2a939a05f3c54be` |
| `tests-report-fixture-fix.xml` | `036710c2364d43aa2ef17d1d7105e855d83a2fb4f895170f930f6891abcdbd8b` |
| `tests-report-fixture-fix.log` | `a9174da89e517faccc6beed51e33c4b672d278e696a87dbb0f8dca2375c532e3` |
| `tests-regression-attempt-001.xml` | `32ef8da2ee0ba3d8bd45d8bd71fb835f9ed71615d780f93b496dabc7ebcf9862` |
| `tests-regression-attempt-001.log` | `ff17ff067f55468a21e020f09478e94b552d3c1a3cbd31829db787c00e285c39` |
