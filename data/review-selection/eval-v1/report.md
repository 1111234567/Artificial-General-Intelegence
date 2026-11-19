# Independent review-selection behavior evaluation

Protocol `b4d79ee047fbf1fc016c3c02cb104155969989ddf412c7394cbb62f8e0f94820`; 168 / 168 bindings; complete=True.

All tests use native11-action argmax, no parameter updates and no action gate. Each checkpoint consumes the same independent public WAIT warmup before an autonomous long-heldout test. The same external cases and eligible matured-request denominators are checked against pre-existing WAIT references.

The mixed stream samples A/B independently at each NEW and job; it is random interleaving, not strict alternation. A and B can be pending simultaneously. Explicit same-slot/raw alternation is separately reported by the paired-control evaluation.

Every mean below weights the three seed rates equally. Per-seed counts, numerators, denominators, all raw groups and each conflicting symbol appear in curves.csv and summary.json. Ordinary completion allows a prior rejected action; strict completion does not. Idle non-WAIT is per1000 inactive slot opportunities and retains the previous slot family/raw.

Actual B NEW counts come from the serialized real world, minus original B0. They include the no-loss boundary exposure. work.B_unique_new_commands is separately retained and is lower by2/1/3 for seed0/1/2 in the original experiment. Repeated review or temporal replay does not add independent cases.

## Actual independent B NEW cases

| Snapshot | seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|
| B000 | 0 | 0 | 0 | 0.0 |
| Bfirst002 | 305 | 303 | 303 | 303.7 |
| Bfirst008 | 1241 | 1199 | 1226 | 1222.0 |
| B001 | 1241 | 1199 | 1226 | 1222.0 |
| B002 | 2454 | 2398 | 2458 | 2436.7 |
| B004 | 4913 | 4843 | 4905 | 4887.0 |
| B008 | 9822 | 9731 | 9776 | 9776.3 |

## Independent long B curve

| Snapshot | Condition | Ordinary | Strict | Idle /1000 |
|---|---|---:|---:|---:|
| B000 | uniform | 17.6% | 16.1% | 13.18 |
| B000 | conflict | 17.6% | 16.1% | 13.18 |
| B000 | error | 17.6% | 16.1% | 13.18 |
| B000 | no_review | 17.6% | 16.1% | 13.18 |
| Bfirst002 | uniform | 44.4% | 22.9% | 3.63 |
| Bfirst002 | conflict | 44.4% | 22.9% | 3.63 |
| Bfirst002 | error | 44.4% | 22.9% | 3.63 |
| Bfirst002 | no_review | 44.4% | 22.9% | 3.63 |
| Bfirst008 | uniform | 19.7% | 13.5% | 2.73 |
| Bfirst008 | conflict | 19.7% | 13.5% | 2.73 |
| Bfirst008 | error | 19.7% | 13.5% | 2.73 |
| Bfirst008 | no_review | 19.7% | 13.5% | 2.73 |
| B001 | uniform | 89.8% | 47.4% | 2.90 |
| B001 | conflict | 87.6% | 37.4% | 4.67 |
| B001 | error | 82.6% | 38.1% | 14.25 |
| B001 | no_review | 75.1% | 57.5% | 2.38 |
| B002 | uniform | 71.0% | 33.1% | 22.07 |
| B002 | conflict | 75.9% | 38.6% | 7.81 |
| B002 | error | 94.4% | 44.5% | 2.07 |
| B002 | no_review | 90.5% | 84.7% | 0.68 |
| B004 | uniform | 85.5% | 54.9% | 12.46 |
| B004 | conflict | 62.8% | 35.4% | 12.53 |
| B004 | error | 88.7% | 58.8% | 5.83 |
| B004 | no_review | 86.9% | 84.7% | 1.56 |
| B008 | uniform | 85.3% | 66.6% | 9.31 |
| B008 | conflict | 96.2% | 77.2% | 0.73 |
| B008 | error | 91.9% | 66.3% | 4.35 |
| B008 | no_review | 99.2% | 98.0% | 0.30 |

## Same-world random-interleaving conflict curves

These two family columns must be assessed separately: a good B rate does not compensate for failed old A requests.

| Snapshot | Condition | A ordinary | A strict | A idle /1000 | B ordinary | B strict | B idle /1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| B000 | uniform | 78.4% | 67.2% | 0.97 | 0.9% | 0.0% | 11.45 |
| B000 | conflict | 78.4% | 67.2% | 0.97 | 0.9% | 0.0% | 11.45 |
| B000 | error | 78.4% | 67.2% | 0.97 | 0.9% | 0.0% | 11.45 |
| B000 | no_review | 78.4% | 67.2% | 0.97 | 0.9% | 0.0% | 11.45 |
| B001 | uniform | 89.7% | 27.9% | 1.05 | 83.3% | 45.2% | 4.45 |
| B001 | conflict | 88.0% | 22.5% | 1.86 | 93.3% | 41.7% | 3.87 |
| B001 | error | 80.5% | 13.5% | 9.23 | 79.8% | 37.4% | 21.64 |
| B001 | no_review | 40.8% | 6.2% | 0.36 | 69.1% | 55.0% | 3.04 |
| B002 | uniform | 86.2% | 18.4% | 6.58 | 71.5% | 31.1% | 19.78 |
| B002 | conflict | 94.2% | 19.6% | 2.98 | 77.6% | 41.1% | 8.70 |
| B002 | error | 91.5% | 14.6% | 0.97 | 90.7% | 34.4% | 3.28 |
| B002 | no_review | 18.8% | 0.9% | 0.61 | 86.5% | 79.1% | 1.24 |
| B004 | uniform | 92.0% | 16.5% | 4.43 | 83.7% | 38.5% | 12.04 |
| B004 | conflict | 71.6% | 3.1% | 4.56 | 64.1% | 24.3% | 12.84 |
| B004 | error | 92.4% | 13.7% | 2.60 | 90.0% | 45.4% | 7.47 |
| B004 | no_review | 8.5% | 0.0% | 0.36 | 87.2% | 86.3% | 2.21 |
| B008 | uniform | 61.1% | 6.5% | 2.22 | 82.8% | 63.0% | 8.64 |
| B008 | conflict | 44.1% | 0.0% | 2.88 | 97.6% | 69.1% | 2.33 |
| B008 | error | 67.9% | 1.5% | 1.98 | 81.9% | 50.0% | 4.71 |
| B008 | no_review | 8.7% | 0.0% | 0.97 | 99.3% | 98.3% | 0.92 |

## B8 mixed conflicting-symbol detail by seed

| seed | Condition | Family | Raw | Matured | Completed | Strict completed | Ordinary | Strict | Idle actions / opportunities | Idle /1000 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | uniform | A | 1 | 14 | 7 | 0 | 50.0% | 0.0% | 5 / 1667 | 3.00 |
| 0 | uniform | A | 3 | 22 | 11 | 1 | 50.0% | 4.5% | 4 / 2838 | 1.41 |
| 0 | uniform | B | 1 | 25 | 25 | 25 | 100.0% | 100.0% | 1 / 3026 | 0.33 |
| 0 | uniform | B | 3 | 20 | 18 | 15 | 90.0% | 75.0% | 5 / 3119 | 1.60 |
| 0 | conflict | A | 1 | 14 | 9 | 0 | 64.3% | 0.0% | 1 / 1683 | 0.59 |
| 0 | conflict | A | 3 | 22 | 16 | 0 | 72.7% | 0.0% | 0 / 2867 | 0.00 |
| 0 | conflict | B | 1 | 25 | 25 | 24 | 100.0% | 96.0% | 2 / 2943 | 0.68 |
| 0 | conflict | B | 3 | 20 | 18 | 18 | 90.0% | 90.0% | 12 / 3092 | 3.88 |
| 0 | error | A | 1 | 14 | 9 | 0 | 64.3% | 0.0% | 1 / 1682 | 0.59 |
| 0 | error | A | 3 | 22 | 15 | 0 | 68.2% | 0.0% | 1 / 2856 | 0.35 |
| 0 | error | B | 1 | 25 | 23 | 22 | 92.0% | 88.0% | 9 / 2946 | 3.05 |
| 0 | error | B | 3 | 20 | 10 | 8 | 50.0% | 40.0% | 16 / 3039 | 5.26 |
| 0 | no_review | A | 1 | 14 | 0 | 0 | 0.0% | 0.0% | 2 / 1618 | 1.24 |
| 0 | no_review | A | 3 | 22 | 5 | 0 | 22.7% | 0.0% | 5 / 2786 | 1.79 |
| 0 | no_review | B | 1 | 25 | 24 | 24 | 96.0% | 96.0% | 3 / 2953 | 1.02 |
| 0 | no_review | B | 3 | 20 | 20 | 20 | 100.0% | 100.0% | 2 / 3098 | 0.65 |
| 1 | uniform | A | 0 | 18 | 8 | 2 | 44.4% | 11.1% | 0 / 2438 | 0.00 |
| 1 | uniform | A | 2 | 20 | 9 | 0 | 45.0% | 0.0% | 0 / 2447 | 0.00 |
| 1 | uniform | B | 0 | 17 | 13 | 3 | 76.5% | 17.6% | 17 / 2618 | 6.49 |
| 1 | uniform | B | 2 | 19 | 19 | 13 | 100.0% | 68.4% | 0 / 2412 | 0.00 |
| 1 | conflict | A | 0 | 18 | 2 | 0 | 11.1% | 0.0% | 27 / 2400 | 11.25 |
| 1 | conflict | A | 2 | 20 | 15 | 0 | 75.0% | 0.0% | 12 / 2489 | 4.82 |
| 1 | conflict | B | 0 | 17 | 17 | 5 | 100.0% | 29.4% | 0 / 2657 | 0.00 |
| 1 | conflict | B | 2 | 19 | 18 | 9 | 94.7% | 47.4% | 19 / 2436 | 7.80 |
| 1 | error | A | 0 | 18 | 10 | 0 | 55.6% | 0.0% | 24 / 2448 | 9.80 |
| 1 | error | A | 2 | 20 | 11 | 0 | 55.0% | 0.0% | 3 / 2461 | 1.22 |
| 1 | error | B | 0 | 17 | 16 | 6 | 94.1% | 35.3% | 2 / 2643 | 0.76 |
| 1 | error | B | 2 | 19 | 13 | 3 | 68.4% | 15.8% | 29 / 2409 | 12.04 |
| 1 | no_review | A | 0 | 18 | 1 | 0 | 5.6% | 0.0% | 1 / 2387 | 0.42 |
| 1 | no_review | A | 2 | 20 | 1 | 0 | 5.0% | 0.0% | 0 / 2398 | 0.00 |
| 1 | no_review | B | 0 | 17 | 17 | 17 | 100.0% | 100.0% | 0 / 2621 | 0.00 |
| 1 | no_review | B | 2 | 19 | 19 | 18 | 100.0% | 94.7% | 0 / 2385 | 0.00 |
| 2 | uniform | A | 0 | 22 | 20 | 5 | 90.9% | 22.7% | 19 / 2173 | 8.74 |
| 2 | uniform | A | 1 | 22 | 19 | 0 | 86.4% | 0.0% | 3 / 2550 | 1.18 |
| 2 | uniform | B | 0 | 17 | 11 | 11 | 64.7% | 64.7% | 67 / 2408 | 27.82 |
| 2 | uniform | B | 1 | 19 | 12 | 9 | 63.2% | 47.4% | 43 / 2695 | 15.96 |
| 2 | conflict | A | 0 | 22 | 0 | 0 | 0.0% | 0.0% | 0 / 2059 | 0.00 |
| 2 | conflict | A | 1 | 22 | 8 | 0 | 36.4% | 0.0% | 2 / 2477 | 0.81 |
| 2 | conflict | B | 0 | 17 | 17 | 14 | 100.0% | 82.4% | 5 / 2494 | 2.00 |
| 2 | conflict | B | 1 | 19 | 19 | 13 | 100.0% | 68.4% | 0 / 2787 | 0.00 |
| 2 | error | A | 0 | 22 | 16 | 1 | 72.7% | 4.5% | 0 / 2156 | 0.00 |
| 2 | error | A | 1 | 22 | 20 | 1 | 90.9% | 4.5% | 0 / 2554 | 0.00 |
| 2 | error | B | 0 | 17 | 17 | 13 | 100.0% | 76.5% | 8 / 2481 | 3.22 |
| 2 | error | B | 1 | 19 | 16 | 8 | 84.2% | 42.1% | 12 / 2753 | 4.36 |
| 2 | no_review | A | 0 | 22 | 0 | 0 | 0.0% | 0.0% | 2 / 2059 | 0.97 |
| 2 | no_review | A | 1 | 22 | 3 | 0 | 13.6% | 0.0% | 3 / 2427 | 1.24 |
| 2 | no_review | B | 0 | 17 | 17 | 17 | 100.0% | 100.0% | 7 / 2457 | 2.85 |
| 2 | no_review | B | 1 | 19 | 19 | 19 | 100.0% | 100.0% | 3 / 2738 | 1.10 |

## Endpoint differences from no review

Paired differences are percentage points, each seed compared with itself on the identical external cases. B gain is a measured difference between snapshots; it is not an isolated learning-rate parameter.

| Condition | Long B strict delta | Mixed A-conflict strict delta | Mixed B-conflict strict delta |
|---|---:|---:|---:|
| uniform | -31.47 | +6.47 | -35.37 |
| conflict | -20.84 | +0.00 | -29.26 |
| error | -31.75 | +1.52 | -48.33 |

Selection policies use only old candidate experience; error-based selection can favor timing/control errors rather than a conflicting semantic symbol. The selection-audit report identifies which examples were actually chosen. Shared budget and case identity do not establish that one policy will generalize outside these3 fixed seeds and one heldout stream per seed.

Original no-review results and original training files remain unchanged. audit.json checks every checkpoint/evidence binding, reconstructed request outcomes, public raw/context pulse identity, frozen external references and equal actual B exposure counts.
