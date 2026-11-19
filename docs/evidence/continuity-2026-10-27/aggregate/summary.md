# Continuity study aggregation

Audit: **passed**; 0 errors, 0 warnings.

Cells show percentages, mean ± sample SD across seeds. Autonomous scores use model actions. Conditional teacher readout appears separately. Raw counts, fingerprints and input SHA-256 values are in summary.json.

| Stage | Controller | Condition | Control | Clean completion | Boundary clean | FINISH precision | n |
|---|---|---|---|---:|---:|---:|---:|
| A | c128 | conflict | continuous | 82.33 ± 0.96 | 82.79 ± 0.43 | 82.12 ± 1.04 | 3 |
| A | c128 | iid | continuous | 92.40 ± 0.44 | 94.44 ± 1.99 | 89.68 ± 1.26 | 3 |
| A | c128 | iid | hidden_reset | 89.74 ± 0.16 | 2.59 ± 0.84 | 85.13 ± 1.13 | 3 |
| A | c128 | long_wait | continuous | 20.27 ± 1.84 | 15.96 ± 4.06 | 16.47 ± 1.42 | 3 |
| A | c128 | longer | continuous | 0.01 ± 0.02 | 0.00 ± 0.00 | 3.37 ± 0.07 | 3 |
| A | c128 | sparse | continuous | 88.73 ± 2.28 | 87.05 ± 4.67 | 27.73 ± 0.40 | 3 |
| A | c256 | conflict | continuous | 97.23 ± 0.36 | 97.13 ± 0.23 | 97.14 ± 0.35 | 3 |
| A | c256 | iid | continuous | 99.11 ± 0.14 | 99.74 ± 0.22 | 96.94 ± 0.27 | 3 |
| A | c256 | iid | hidden_reset | 96.32 ± 0.14 | 2.83 ± 0.93 | 92.47 ± 0.36 | 3 |
| A | c256 | long_wait | continuous | 29.84 ± 1.42 | 23.55 ± 0.29 | 23.29 ± 0.90 | 3 |
| A | c256 | longer | continuous | 0.01 ± 0.01 | 0.00 ± 0.00 | 3.85 ± 0.15 | 3 |
| A | c256 | sparse | continuous | 97.37 ± 1.08 | 95.85 ± 1.67 | 29.61 ± 0.32 | 3 |
| A | c64 | conflict | continuous | 42.35 ± 1.92 | 42.76 ± 1.87 | 42.42 ± 2.05 | 3 |
| A | c64 | iid | continuous | 49.48 ± 3.70 | 49.12 ± 2.98 | 47.74 ± 4.03 | 3 |
| A | c64 | iid | hidden_reset | 47.59 ± 3.26 | 2.70 ± 0.34 | 45.04 ± 3.48 | 3 |
| A | c64 | long_wait | continuous | 10.61 ± 0.88 | 6.89 ± 2.81 | 9.78 ± 0.80 | 3 |
| A | c64 | longer | continuous | 0.02 ± 0.02 | 0.00 ± 0.00 | 2.52 ± 0.07 | 3 |
| A | c64 | sparse | continuous | 45.92 ± 5.83 | 40.62 ± 6.60 | 14.94 ± 2.06 | 3 |
| A | gru128 | conflict | continuous | 12.12 ± 0.30 | 10.84 ± 3.21 | 9.39 ± 0.18 | 3 |
| A | gru128 | iid | continuous | 11.04 ± 0.16 | 11.10 ± 1.53 | 6.87 ± 0.21 | 3 |
| A | gru128 | iid | hidden_reset | 10.85 ± 0.46 | 6.97 ± 1.47 | 6.79 ± 0.05 | 3 |
| A | gru128 | long_wait | continuous | 6.80 ± 0.38 | 5.75 ± 2.45 | 6.08 ± 0.30 | 3 |
| A | gru128 | longer | continuous | 0.01 ± 0.01 | 0.00 ± 0.00 | 2.28 ± 0.09 | 3 |
| A | gru128 | sparse | continuous | 8.46 ± 0.94 | 6.44 ± 4.33 | 1.38 ± 0.08 | 3 |
| A | online128 | conflict | continuous | 83.77 ± 0.10 | 82.84 ± 0.96 | 82.70 ± 0.63 | 3 |
| A | online128 | iid | continuous | 93.20 ± 0.05 | 93.04 ± 0.28 | 91.51 ± 0.35 | 3 |
| A | online128 | iid | hidden_reset | 90.18 ± 0.03 | 4.52 ± 1.30 | 77.89 ± 0.79 | 3 |
| A | online128 | long_wait | continuous | 12.36 ± 1.12 | 9.54 ± 1.98 | 7.88 ± 0.51 | 3 |
| A | online128 | longer | continuous | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.35 ± 0.14 | 3 |
| A | online128 | sparse | continuous | 82.81 ± 2.17 | 80.27 ± 5.03 | 9.04 ± 0.82 | 3 |
| A | reservoir128 | conflict | continuous | 70.48 ± 0.97 | 67.68 ± 2.95 | 71.04 ± 1.31 | 3 |
| A | reservoir128 | iid | continuous | 81.35 ± 0.79 | 79.91 ± 3.40 | 77.34 ± 1.03 | 3 |
| A | reservoir128 | iid | hidden_reset | 78.84 ± 0.70 | 2.71 ± 0.43 | 73.00 ± 1.01 | 3 |
| A | reservoir128 | long_wait | continuous | 12.39 ± 1.05 | 10.76 ± 1.98 | 11.49 ± 0.80 | 3 |
| A | reservoir128 | longer | continuous | 0.08 ± 0.07 | 0.07 ± 0.12 | 2.77 ± 0.12 | 3 |
| A | reservoir128 | sparse | continuous | 62.73 ± 3.88 | 59.25 ± 4.52 | 22.34 ± 1.66 | 3 |
| A | reset128 | conflict | continuous | 80.65 ± 1.79 | 81.18 ± 2.81 | 80.10 ± 1.81 | 3 |
| A | reset128 | iid | continuous | 90.97 ± 0.85 | 91.75 ± 0.66 | 87.29 ± 1.30 | 3 |
| A | reset128 | iid | hidden_reset | 88.84 ± 0.75 | 5.83 ± 3.82 | 85.21 ± 1.29 | 3 |
| A | reset128 | long_wait | continuous | 19.81 ± 1.53 | 13.86 ± 3.33 | 15.92 ± 1.02 | 3 |
| A | reset128 | longer | continuous | 0.03 ± 0.02 | 0.00 ± 0.00 | 3.46 ± 0.11 | 3 |
| A | reset128 | sparse | continuous | 88.36 ± 2.36 | 87.32 ± 4.27 | 25.83 ± 0.59 | 3 |
| B | current | iid | continuous | 60.09 ± 2.95 | 56.89 ± 2.25 | 42.53 ± 3.91 | 3 |
| B | current | iid | hidden_reset | 59.29 ± 2.72 | 4.31 ± 1.01 | 41.08 ± 3.36 | 3 |
| B | current | long_wait | continuous | 82.09 ± 2.98 | 78.20 ± 3.67 | 77.10 ± 3.93 | 3 |
| B | current | longer | continuous | 3.66 ± 1.20 | 2.30 ± 0.31 | 8.44 ± 0.54 | 3 |
| B | current_repeat | iid | continuous | 52.65 ± 1.08 | 56.76 ± 2.24 | 33.17 ± 0.50 | 3 |
| B | current_repeat | iid | hidden_reset | 51.65 ± 0.91 | 4.96 ± 1.41 | 31.97 ± 0.54 | 3 |
| B | current_repeat | long_wait | continuous | 76.70 ± 1.43 | 74.67 ± 0.23 | 72.54 ± 0.86 | 3 |
| B | current_repeat | longer | continuous | 3.41 ± 0.31 | 2.24 ± 0.57 | 7.67 ± 0.88 | 3 |
| B | frozen | iid | continuous | 92.99 ± 0.43 | 92.87 ± 1.93 | 87.24 ± 0.22 | 3 |
| B | frozen | iid | hidden_reset | 90.10 ± 0.40 | 3.53 ± 2.61 | 75.51 ± 0.85 | 3 |
| B | frozen | long_wait | continuous | 12.32 ± 0.15 | 10.64 ± 1.45 | 7.57 ± 0.30 | 3 |
| B | frozen | longer | continuous | 0.01 ± 0.01 | 0.00 ± 0.00 | 2.36 ± 0.13 | 3 |
| B | replay | iid | continuous | 91.97 ± 0.37 | 90.78 ± 0.96 | 87.24 ± 0.49 | 3 |
| B | replay | iid | hidden_reset | 89.66 ± 0.36 | 4.11 ± 1.79 | 81.34 ± 1.15 | 3 |
| B | replay | long_wait | continuous | 63.82 ± 2.97 | 57.56 ± 2.35 | 60.85 ± 2.66 | 3 |
| B | replay | longer | continuous | 1.11 ± 0.40 | 0.27 ± 0.31 | 6.85 ± 0.29 | 3 |

## Parameters

| Controller | State scalars | Total parameters | Trainable parameters | Training actions |
|---|---:|---:|---:|---|
| c64 | 64 | 13,100 | 13,100 | teacher |
| c128 | 128 | 34,348 | 34,348 | teacher |
| c256 | 256 | 101,420 | 101,420 | teacher |
| reset128 | 128 | 34,348 | 34,348 | teacher |
| reservoir128 | 128 | 34,348 | 17,964 | teacher |
| gru128 | 128 | 117,420 | 117,420 | teacher |
| online128 | 128 | 34,348 | 34,348 | online |

## Adaptation

| Arm | Gradients / live chunk | A before | A after | A change | B before | B after | B gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0 | 93.20 ± 0.05 | 92.99 ± 0.43 | -0.21 ± 0.48 | 12.36 ± 1.12 | 12.32 ± 0.15 | -0.05 ± 1.04 |
| current | 1 | 93.20 ± 0.05 | 60.09 ± 2.95 | -33.10 ± 2.89 | 12.36 ± 1.12 | 82.09 ± 2.98 | 69.72 ± 2.93 |
| current_repeat | 2 | 93.20 ± 0.05 | 52.65 ± 1.08 | -40.55 ± 1.06 | 12.36 ± 1.12 | 76.70 ± 1.43 | 64.34 ± 2.05 |
| replay | 2 | 93.20 ± 0.05 | 91.97 ± 0.37 | -1.23 ± 0.42 | 12.36 ± 1.12 | 63.82 ± 2.97 | 51.46 ± 2.53 |

## Conditional FINISH readout

Teacher timing and receipts; argmax over the eight existing FINISH logits. These are diagnostic readout rates.

| Stage | Controller | Condition | Readout accuracy | n |
|---|---|---|---:|---:|
| A | c128 | conflict | 84.48 ± 1.18 | 3 |
| A | c128 | iid | 93.62 ± 0.61 | 3 |
| A | c128 | long_wait | 43.80 ± 1.69 | 3 |
| A | c128 | longer | 14.71 ± 0.39 | 3 |
| A | c128 | sparse | 86.48 ± 1.26 | 3 |
| A | c256 | conflict | 97.86 ± 0.27 | 3 |
| A | c256 | iid | 99.35 ± 0.14 | 3 |
| A | c256 | long_wait | 59.02 ± 2.45 | 3 |
| A | c256 | longer | 17.51 ± 1.84 | 3 |
| A | c256 | sparse | 97.62 ± 0.34 | 3 |
| A | c64 | conflict | 48.21 ± 2.22 | 3 |
| A | c64 | iid | 56.94 ± 2.75 | 3 |
| A | c64 | long_wait | 25.61 ± 1.92 | 3 |
| A | c64 | longer | 12.94 ± 0.28 | 3 |
| A | c64 | sparse | 47.50 ± 3.11 | 3 |
| A | gru128 | conflict | 35.40 ± 1.31 | 3 |
| A | gru128 | iid | 36.55 ± 0.85 | 3 |
| A | gru128 | long_wait | 13.91 ± 0.60 | 3 |
| A | gru128 | longer | 12.47 ± 0.46 | 3 |
| A | gru128 | sparse | 17.46 ± 1.74 | 3 |
| A | online128 | conflict | 84.31 ± 0.46 | 3 |
| A | online128 | iid | 92.93 ± 0.32 | 3 |
| A | online128 | long_wait | 40.39 ± 3.22 | 3 |
| A | online128 | longer | 15.54 ± 0.85 | 3 |
| A | online128 | sparse | 92.45 ± 0.15 | 3 |
| A | reservoir128 | conflict | 78.89 ± 1.70 | 3 |
| A | reservoir128 | iid | 89.67 ± 1.04 | 3 |
| A | reservoir128 | long_wait | 32.54 ± 1.04 | 3 |
| A | reservoir128 | longer | 13.50 ± 0.68 | 3 |
| A | reservoir128 | sparse | 76.56 ± 3.24 | 3 |
| A | reset128 | conflict | 83.16 ± 1.38 | 3 |
| A | reset128 | iid | 92.45 ± 0.94 | 3 |
| A | reset128 | long_wait | 43.05 ± 1.55 | 3 |
| A | reset128 | longer | 13.87 ± 0.36 | 3 |
| A | reset128 | sparse | 85.84 ± 1.26 | 3 |
| B | current | iid | 72.74 ± 3.46 | 3 |
| B | current | long_wait | 91.21 ± 1.52 | 3 |
| B | current | longer | 36.00 ± 3.20 | 3 |
| B | current_repeat | iid | 69.75 ± 2.84 | 3 |
| B | current_repeat | long_wait | 86.99 ± 1.97 | 3 |
| B | current_repeat | longer | 35.94 ± 4.12 | 3 |
| B | frozen | iid | 92.98 ± 0.29 | 3 |
| B | frozen | long_wait | 41.01 ± 2.80 | 3 |
| B | frozen | longer | 16.08 ± 1.08 | 3 |
| B | replay | iid | 96.16 ± 0.36 | 3 |
| B | replay | long_wait | 92.57 ± 1.81 | 3 |
| B | replay | longer | 43.19 ± 2.32 | 3 |

Current-repeat and replay each use two gradient passes. Replay adds a 64-tick no-grad burn-in; forward FLOPs differ. A/B pre/post differences describe functional behavior of weights plus numerical state on different future requests. The A pre-B long-wait evaluation shares an exogenous prefix with B training. Paired contrasts against the frozen-B endpoint are the primary adaptation comparison; these also combine parameter and live-state effects. Training counters are lifetime totals; evaluation cohorts determine these rates.
