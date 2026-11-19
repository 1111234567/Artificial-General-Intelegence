# Consolidation experiment tables

All accuracies are percentages, reported as **mean ± sample SD across three independent seeds**. Every seed is weighted equally; evaluation examples are not pooled across seeds. Gain values are percentage-point changes, paired within each seed. NA means the requested metric is undefined for at least one seed.

**Online uses half the optimizer updates and training-example presentations. Only repeat versus replay is the compute-matched comparison.** External observations, initialization, teachers, and per-context streams are paired. All methods retain the same buffer; only replay uses its historical records for learning.

Evaluation split: `test`; seeds: `[0, 1, 2]`. The version cue is an observed context-local rule change, not autonomous change detection. The held-out positive-positive conjunction is a narrow transfer probe.

`current_mean` = mean(A1, B0, C0); `all4_mean` = mean(A0, A1, B0, C0). A0 measures retention of the superseded historical rule. A1 is evaluated separately; historical-rule retention is not required for present-rule correctness.

## Final checkpoint

### test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 4.47 ± 1.95 | 5.58 ± 2.68 | 10.33 ± 2.05 | 92.67 ± 4.79 | 36.19 ± 0.79 | 28.26 ± 0.53 |
| blocked | repeat | 3.43 ± 1.56 | 2.73 ± 1.20 | 9.78 ± 1.38 | 95.67 ± 0.78 | 36.06 ± 0.81 | 27.91 ± 0.99 |
| blocked | replay | 94.37 ± 0.66 | 93.98 ± 0.66 | 95.93 ± 0.45 | 96.57 ± 0.34 | 95.49 ± 0.40 | 95.21 ± 0.44 |
| recurrent | online | 42.97 ± 7.96 | 93.83 ± 0.22 | 94.78 ± 0.13 | 96.79 ± 0.65 | 95.13 ± 0.25 | 82.09 ± 2.10 |
| recurrent | repeat | 49.03 ± 6.44 | 93.70 ± 0.62 | 94.80 ± 1.09 | 96.57 ± 0.48 | 95.02 ± 0.68 | 83.52 ± 2.07 |
| recurrent | replay | 94.63 ± 0.08 | 95.49 ± 0.13 | 96.26 ± 0.12 | 96.41 ± 0.31 | 96.05 ± 0.16 | 95.69 ± 0.10 |

### transfer_test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 5.12 ± 2.33 | 6.30 ± 3.44 | 10.85 ± 1.90 | 92.44 ± 4.40 | 36.53 ± 0.18 | 28.68 ± 0.70 |
| blocked | repeat | 3.57 ± 0.91 | 2.74 ± 1.48 | 9.86 ± 1.10 | 95.61 ± 0.24 | 36.07 ± 0.74 | 27.94 ± 0.78 |
| blocked | replay | 94.03 ± 0.33 | 93.92 ± 0.23 | 95.67 ± 0.52 | 96.55 ± 0.15 | 95.38 ± 0.27 | 95.04 ± 0.18 |
| recurrent | online | 40.16 ± 4.42 | 94.27 ± 0.78 | 94.91 ± 0.04 | 96.92 ± 0.64 | 95.37 ± 0.47 | 81.56 ± 1.10 |
| recurrent | repeat | 44.51 ± 1.41 | 93.64 ± 0.50 | 94.75 ± 1.07 | 96.45 ± 0.36 | 94.95 ± 0.58 | 82.34 ± 0.72 |
| recurrent | replay | 94.82 ± 0.22 | 95.24 ± 0.22 | 96.26 ± 0.10 | 96.46 ± 0.07 | 95.99 ± 0.05 | 95.70 ± 0.07 |

## Revision end: immediately after the last A1 block

### test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 1.54 ± 0.38 | 95.74 ± 0.15 | 8.74 ± 1.22 | 2.77 ± 0.66 | 35.75 ± 0.54 | 27.20 ± 0.49 |
| blocked | repeat | 1.39 ± 0.25 | 96.02 ± 0.25 | 8.63 ± 1.26 | 2.37 ± 0.19 | 35.67 ± 0.32 | 27.10 ± 0.30 |
| blocked | replay | 93.13 ± 1.13 | 96.18 ± 0.54 | 93.44 ± 0.13 | 93.78 ± 0.78 | 94.47 ± 0.36 | 94.13 ± 0.43 |
| recurrent | online | 20.52 ± 6.70 | 96.61 ± 0.37 | 90.44 ± 0.83 | 93.73 ± 1.75 | 93.59 ± 0.66 | 75.33 ± 2.16 |
| recurrent | repeat | 26.06 ± 8.69 | 96.55 ± 0.10 | 90.24 ± 0.46 | 93.68 ± 0.50 | 93.49 ± 0.10 | 76.63 ± 2.17 |
| recurrent | replay | 94.23 ± 0.36 | 96.36 ± 0.31 | 95.93 ± 0.27 | 95.87 ± 0.07 | 96.05 ± 0.03 | 95.60 ± 0.08 |

### transfer_test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 1.57 ± 0.44 | 95.65 ± 0.23 | 9.34 ± 0.67 | 3.07 ± 1.53 | 36.02 ± 0.75 | 27.41 ± 0.67 |
| blocked | repeat | 1.46 ± 0.36 | 96.09 ± 0.12 | 9.31 ± 0.66 | 2.36 ± 0.40 | 35.92 ± 0.24 | 27.31 ± 0.27 |
| blocked | replay | 92.96 ± 0.99 | 95.79 ± 0.12 | 93.25 ± 0.40 | 93.41 ± 0.67 | 94.15 ± 0.37 | 93.85 ± 0.34 |
| recurrent | online | 20.18 ± 2.30 | 96.71 ± 0.50 | 90.51 ± 0.27 | 93.20 ± 1.57 | 93.47 ± 0.66 | 75.15 ± 1.03 |
| recurrent | repeat | 23.43 ± 3.86 | 96.42 ± 0.30 | 90.37 ± 0.67 | 93.52 ± 0.21 | 93.44 ± 0.25 | 75.94 ± 0.88 |
| recurrent | replay | 94.23 ± 0.76 | 96.38 ± 0.06 | 95.62 ± 0.23 | 95.87 ± 0.08 | 95.96 ± 0.04 | 95.53 ± 0.18 |

## First A1 block: before feedback

### test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 1.14 ± 0.16 | 1.66 ± 0.34 | 8.36 ± 0.88 | 96.85 ± 0.28 | 35.62 ± 0.16 | 27.00 ± 0.16 |
| blocked | repeat | 1.18 ± 0.04 | 1.35 ± 0.10 | 8.48 ± 0.80 | 96.84 ± 0.21 | 35.56 ± 0.29 | 26.96 ± 0.23 |
| blocked | replay | 95.08 ± 0.43 | 1.73 ± 0.16 | 94.33 ± 0.33 | 96.40 ± 0.35 | 64.15 ± 0.19 | 71.89 ± 0.23 |
| recurrent | online | 83.31 ± 1.44 | 5.51 ± 1.19 | 81.17 ± 9.74 | 96.06 ± 0.36 | 60.91 ± 3.55 | 66.51 ± 2.89 |
| recurrent | repeat | 85.57 ± 0.26 | 5.13 ± 0.73 | 88.34 ± 3.03 | 96.28 ± 0.21 | 63.25 ± 1.25 | 68.83 ± 0.94 |
| recurrent | replay | 96.08 ± 0.50 | 1.63 ± 0.40 | 95.92 ± 0.24 | 96.64 ± 0.33 | 64.73 ± 0.13 | 72.57 ± 0.11 |

### transfer_test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 1.09 ± 0.22 | 1.50 ± 0.23 | 7.98 ± 0.68 | 97.01 ± 0.17 | 35.49 ± 0.17 | 26.89 ± 0.15 |
| blocked | repeat | 1.25 ± 0.17 | 1.26 ± 0.04 | 8.01 ± 0.59 | 96.93 ± 0.11 | 35.40 ± 0.21 | 26.86 ± 0.16 |
| blocked | replay | 94.94 ± 0.25 | 1.64 ± 0.07 | 94.08 ± 0.19 | 96.43 ± 0.31 | 64.05 ± 0.07 | 71.77 ± 0.07 |
| recurrent | online | 82.75 ± 2.04 | 5.75 ± 0.64 | 81.67 ± 9.70 | 96.13 ± 0.28 | 61.18 ± 3.41 | 66.57 ± 3.01 |
| recurrent | repeat | 85.09 ± 0.38 | 5.52 ± 0.17 | 88.09 ± 3.69 | 96.32 ± 0.31 | 63.31 ± 1.32 | 68.76 ± 1.09 |
| recurrent | replay | 96.16 ± 0.43 | 1.45 ± 0.12 | 95.65 ± 0.36 | 96.73 ± 0.12 | 64.61 ± 0.14 | 72.50 ± 0.08 |

## First A1 block: after training

### test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 1.65 ± 0.94 | 89.88 ± 3.11 | 7.67 ± 1.77 | 5.39 ± 1.04 | 34.31 ± 1.28 | 26.15 ± 1.19 |
| blocked | repeat | 1.83 ± 0.42 | 94.32 ± 0.51 | 8.55 ± 1.27 | 3.27 ± 0.45 | 35.38 ± 0.16 | 26.99 ± 0.22 |
| blocked | replay | 89.10 ± 2.87 | 93.56 ± 0.60 | 92.36 ± 0.51 | 91.45 ± 1.30 | 92.46 ± 0.79 | 91.62 ± 1.06 |
| recurrent | online | 2.57 ± 0.12 | 95.11 ± 0.10 | 8.88 ± 0.79 | 44.98 ± 8.95 | 49.66 ± 3.25 | 37.89 ± 2.46 |
| recurrent | repeat | 1.92 ± 0.19 | 95.73 ± 0.09 | 9.03 ± 0.97 | 46.41 ± 8.50 | 50.39 ± 3.15 | 38.27 ± 2.41 |
| recurrent | replay | 91.53 ± 0.62 | 94.14 ± 0.36 | 93.25 ± 0.44 | 91.54 ± 0.25 | 92.98 ± 0.23 | 92.61 ± 0.28 |

### transfer_test

| Order | Method | A0 | A1 | B0 | C0 | current_mean | all4_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 1.73 ± 1.18 | 89.03 ± 4.12 | 8.02 ± 1.91 | 6.60 ± 0.81 | 34.55 ± 1.74 | 26.34 ± 1.58 |
| blocked | repeat | 1.82 ± 0.59 | 94.30 ± 0.56 | 9.17 ± 0.70 | 3.37 ± 0.21 | 35.61 ± 0.20 | 27.16 ± 0.29 |
| blocked | replay | 89.03 ± 2.27 | 93.46 ± 0.69 | 92.15 ± 0.88 | 91.04 ± 0.78 | 92.21 ± 0.77 | 91.42 ± 0.91 |
| recurrent | online | 2.67 ± 0.42 | 95.21 ± 0.35 | 9.29 ± 0.75 | 43.49 ± 9.78 | 49.33 ± 3.49 | 37.66 ± 2.57 |
| recurrent | repeat | 1.89 ± 0.30 | 95.70 ± 0.19 | 9.31 ± 0.62 | 41.91 ± 10.63 | 48.97 ± 3.78 | 37.20 ± 2.91 |
| recurrent | replay | 91.11 ± 1.32 | 93.86 ± 0.29 | 93.05 ± 0.47 | 91.45 ± 0.74 | 92.78 ± 0.30 | 92.37 ± 0.20 |

## B0 transfer before its first feedback

B0 is evaluated just before its first B0 block. The agreement subsets use the environment's actual A0/B0 agreement on each evaluation input. Gain subtracts the same seed's untrained initial B0 accuracy; it is not a separate training stage.

### test

| Order | Method | B0 overall | A/B agree | Agree gain (pp) | A/B disagree | Disagree gain (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 74.24 ± 0.80 | 97.56 ± 0.32 | 70.58 ± 3.80 | 6.01 ± 0.52 | -19.49 ± 0.58 |
| blocked | repeat | 74.31 ± 1.01 | 97.57 ± 0.27 | 70.59 ± 3.79 | 6.28 ± 0.88 | -19.22 ± 1.56 |
| blocked | replay | 74.29 ± 0.54 | 97.68 ± 0.46 | 70.71 ± 3.48 | 5.85 ± 0.19 | -19.65 ± 0.86 |
| recurrent | online | 74.21 ± 0.55 | 95.98 ± 0.51 | 69.00 ± 3.42 | 10.54 ± 0.50 | -14.96 ± 1.10 |
| recurrent | repeat | 73.81 ± 0.86 | 96.41 ± 0.19 | 69.44 ± 3.74 | 7.68 ± 0.76 | -17.82 ± 1.13 |
| recurrent | replay | 74.22 ± 0.53 | 96.72 ± 0.58 | 69.75 ± 3.51 | 8.38 ± 0.38 | -17.12 ± 0.39 |

### transfer_test

| Order | Method | B0 overall | A/B agree | Agree gain (pp) | A/B disagree | Disagree gain (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 73.82 ± 0.51 | 97.45 ± 0.63 | 74.29 ± 5.03 | 5.71 ± 0.50 | -16.50 ± 4.32 |
| blocked | repeat | 73.62 ± 0.37 | 97.26 ± 0.51 | 74.11 ± 5.19 | 5.48 ± 0.16 | -16.73 ± 4.58 |
| blocked | replay | 73.76 ± 0.44 | 97.41 ± 0.35 | 74.25 ± 5.31 | 5.59 ± 0.26 | -16.62 ± 4.27 |
| recurrent | online | 73.40 ± 0.22 | 95.73 ± 0.85 | 72.57 ± 5.47 | 9.06 ± 0.43 | -13.15 ± 4.09 |
| recurrent | repeat | 73.55 ± 0.36 | 96.42 ± 0.41 | 73.26 ± 5.39 | 7.61 ± 0.45 | -14.60 ± 4.79 |
| recurrent | replay | 73.38 ± 0.37 | 96.35 ± 0.84 | 73.19 ± 4.81 | 7.19 ± 0.63 | -15.02 ± 4.94 |

## Accuracy before returning to each condition

A return has visit >= 2. First average return accuracies within each seed, then compute the three-seed mean and sample SD. Consecutive blocked visits are included; their gap is zero. Detailed per-return measurements follow.

### test

| Order | Method | A0 | A1 | B0 | C0 |
| --- | --- | --- | --- | --- | --- |
| blocked | online | 95.86 ± 0.27 | 92.46 ± 1.48 | 80.52 ± 5.02 | 81.49 ± 4.07 |
| blocked | repeat | 95.72 ± 0.34 | 94.97 ± 0.55 | 85.06 ± 0.36 | 84.79 ± 0.45 |
| blocked | replay | 96.21 ± 0.23 | 94.61 ± 0.50 | 94.49 ± 0.34 | 95.21 ± 0.33 |
| recurrent | online | 17.18 ± 1.11 | 48.27 ± 7.31 | 64.64 ± 1.83 | 47.52 ± 2.07 |
| recurrent | repeat | 24.43 ± 4.46 | 51.96 ± 0.49 | 66.26 ± 1.57 | 50.95 ± 4.23 |
| recurrent | replay | 93.65 ± 0.92 | 91.56 ± 0.78 | 93.48 ± 0.35 | 95.13 ± 0.17 |

### transfer_test

| Order | Method | A0 | A1 | B0 | C0 |
| --- | --- | --- | --- | --- | --- |
| blocked | online | 95.90 ± 0.22 | 92.04 ± 2.05 | 80.41 ± 4.64 | 81.20 ± 4.46 |
| blocked | repeat | 95.72 ± 0.33 | 95.02 ± 0.30 | 85.11 ± 0.24 | 84.78 ± 0.31 |
| blocked | replay | 96.25 ± 0.33 | 94.49 ± 0.38 | 94.31 ± 0.46 | 94.98 ± 0.14 |
| recurrent | online | 18.87 ± 1.10 | 49.32 ± 7.31 | 64.61 ± 2.08 | 47.79 ± 2.07 |
| recurrent | repeat | 25.45 ± 5.50 | 53.12 ± 1.24 | 66.13 ± 1.21 | 51.12 ± 4.36 |
| recurrent | replay | 93.58 ± 0.66 | 91.32 ± 1.29 | 93.26 ± 0.14 | 95.20 ± 0.19 |

### Every return

Gap is the number of intervening external minibatch steps, not optimizer updates. The example count is measured before the returning block.

| Order | Method | Condition | Visit | External examples | Gap steps | test | transfer_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | A0 | 2 | 25600 | 0.00 ± 0.00 | 94.41 ± 0.47 | 94.59 ± 0.19 |
| blocked | online | A0 | 3 | 51200 | 0.00 ± 0.00 | 95.50 ± 0.49 | 95.49 ± 0.54 |
| blocked | online | A0 | 4 | 76800 | 0.00 ± 0.00 | 96.23 ± 0.36 | 96.42 ± 0.15 |
| blocked | online | A0 | 5 | 102400 | 0.00 ± 0.00 | 96.46 ± 0.21 | 96.53 ± 0.26 |
| blocked | online | A0 | 6 | 128000 | 0.00 ± 0.00 | 96.68 ± 0.27 | 96.49 ± 0.20 |
| blocked | online | A1 | 2 | 486400 | 0.00 ± 0.00 | 89.88 ± 3.11 | 89.03 ± 4.12 |
| blocked | online | A1 | 3 | 512000 | 0.00 ± 0.00 | 95.04 ± 0.17 | 95.05 ± 0.08 |
| blocked | online | B0 | 2 | 179200 | 0.00 ± 0.00 | 96.39 ± 0.16 | 96.23 ± 0.29 |
| blocked | online | B0 | 3 | 204800 | 0.00 ± 0.00 | 96.77 ± 0.42 | 96.74 ± 0.29 |
| blocked | online | B0 | 4 | 230400 | 0.00 ± 0.00 | 96.95 ± 0.28 | 96.89 ± 0.28 |
| blocked | online | B0 | 5 | 256000 | 0.00 ± 0.00 | 97.01 ± 0.37 | 96.87 ± 0.37 |
| blocked | online | B0 | 6 | 281600 | 0.00 ± 0.00 | 97.30 ± 0.28 | 97.07 ± 0.21 |
| blocked | online | B0 | 7 | 537600 | 900.00 ± 0.00 | 8.74 ± 1.22 | 9.34 ± 0.67 |
| blocked | online | B0 | 8 | 563200 | 0.00 ± 0.00 | 67.82 ± 20.19 | 65.97 ± 21.65 |
| blocked | online | B0 | 9 | 588800 | 0.00 ± 0.00 | 83.22 ± 18.62 | 84.18 ± 16.57 |
| blocked | online | C0 | 2 | 332800 | 0.00 ± 0.00 | 94.72 ± 0.24 | 94.63 ± 0.33 |
| blocked | online | C0 | 3 | 358400 | 0.00 ± 0.00 | 96.09 ± 0.32 | 96.20 ± 0.24 |
| blocked | online | C0 | 4 | 384000 | 0.00 ± 0.00 | 96.44 ± 0.16 | 96.48 ± 0.08 |
| blocked | online | C0 | 5 | 409600 | 0.00 ± 0.00 | 96.70 ± 0.40 | 96.84 ± 0.28 |
| blocked | online | C0 | 6 | 435200 | 0.00 ± 0.00 | 96.83 ± 0.37 | 96.90 ± 0.32 |
| blocked | online | C0 | 7 | 614400 | 600.00 ± 0.00 | 8.58 ± 1.35 | 7.93 ± 0.80 |
| blocked | online | C0 | 8 | 640000 | 0.00 ± 0.00 | 75.68 ± 21.75 | 75.02 ± 21.57 |
| blocked | online | C0 | 9 | 665600 | 0.00 ± 0.00 | 86.84 ± 13.05 | 85.58 ± 14.63 |
| blocked | repeat | A0 | 2 | 25600 | 0.00 ± 0.00 | 95.00 ± 0.22 | 94.94 ± 0.55 |
| blocked | repeat | A0 | 3 | 51200 | 0.00 ± 0.00 | 95.50 ± 0.34 | 95.51 ± 0.48 |
| blocked | repeat | A0 | 4 | 76800 | 0.00 ± 0.00 | 95.89 ± 0.48 | 95.78 ± 0.26 |
| blocked | repeat | A0 | 5 | 102400 | 0.00 ± 0.00 | 96.02 ± 0.58 | 95.90 ± 0.68 |
| blocked | repeat | A0 | 6 | 128000 | 0.00 ± 0.00 | 96.19 ± 0.36 | 96.48 ± 0.40 |
| blocked | repeat | A1 | 2 | 486400 | 0.00 ± 0.00 | 94.32 ± 0.51 | 94.30 ± 0.56 |
| blocked | repeat | A1 | 3 | 512000 | 0.00 ± 0.00 | 95.63 ± 0.60 | 95.75 ± 0.09 |
| blocked | repeat | B0 | 2 | 179200 | 0.00 ± 0.00 | 96.30 ± 0.20 | 96.22 ± 0.53 |
| blocked | repeat | B0 | 3 | 204800 | 0.00 ± 0.00 | 96.21 ± 0.72 | 96.51 ± 0.71 |
| blocked | repeat | B0 | 4 | 230400 | 0.00 ± 0.00 | 96.79 ± 0.14 | 96.61 ± 0.28 |
| blocked | repeat | B0 | 5 | 256000 | 0.00 ± 0.00 | 96.83 ± 0.32 | 96.60 ± 0.43 |
| blocked | repeat | B0 | 6 | 281600 | 0.00 ± 0.00 | 96.90 ± 0.14 | 96.90 ± 0.22 |
| blocked | repeat | B0 | 7 | 537600 | 900.00 ± 0.00 | 8.63 ± 1.26 | 9.31 ± 0.66 |
| blocked | repeat | B0 | 8 | 563200 | 0.00 ± 0.00 | 93.63 ± 0.52 | 93.39 ± 0.54 |
| blocked | repeat | B0 | 9 | 588800 | 0.00 ± 0.00 | 95.19 ± 0.50 | 95.33 ± 0.34 |
| blocked | repeat | C0 | 2 | 332800 | 0.00 ± 0.00 | 95.57 ± 0.21 | 95.71 ± 0.20 |
| blocked | repeat | C0 | 3 | 358400 | 0.00 ± 0.00 | 96.42 ± 0.34 | 96.41 ± 0.37 |
| blocked | repeat | C0 | 4 | 384000 | 0.00 ± 0.00 | 96.58 ± 0.34 | 96.61 ± 0.35 |
| blocked | repeat | C0 | 5 | 409600 | 0.00 ± 0.00 | 96.72 ± 0.26 | 96.79 ± 0.23 |
| blocked | repeat | C0 | 6 | 435200 | 0.00 ± 0.00 | 96.74 ± 0.24 | 97.08 ± 0.19 |
| blocked | repeat | C0 | 7 | 614400 | 600.00 ± 0.00 | 8.21 ± 0.91 | 7.99 ± 1.08 |
| blocked | repeat | C0 | 8 | 640000 | 0.00 ± 0.00 | 92.81 ± 1.24 | 92.62 ± 1.50 |
| blocked | repeat | C0 | 9 | 665600 | 0.00 ± 0.00 | 95.30 ± 0.54 | 95.04 ± 0.36 |
| blocked | replay | A0 | 2 | 25600 | 0.00 ± 0.00 | 95.39 ± 0.16 | 95.44 ± 0.66 |
| blocked | replay | A0 | 3 | 51200 | 0.00 ± 0.00 | 96.22 ± 0.18 | 96.13 ± 0.30 |
| blocked | replay | A0 | 4 | 76800 | 0.00 ± 0.00 | 96.29 ± 0.23 | 96.39 ± 0.14 |
| blocked | replay | A0 | 5 | 102400 | 0.00 ± 0.00 | 96.57 ± 0.37 | 96.46 ± 0.43 |
| blocked | replay | A0 | 6 | 128000 | 0.00 ± 0.00 | 96.59 ± 0.34 | 96.83 ± 0.41 |
| blocked | replay | A1 | 2 | 486400 | 0.00 ± 0.00 | 93.56 ± 0.60 | 93.46 ± 0.69 |
| blocked | replay | A1 | 3 | 512000 | 0.00 ± 0.00 | 95.67 ± 0.46 | 95.53 ± 0.20 |
| blocked | replay | B0 | 2 | 179200 | 0.00 ± 0.00 | 90.42 ± 0.59 | 90.19 ± 0.61 |
| blocked | replay | B0 | 3 | 204800 | 0.00 ± 0.00 | 93.88 ± 1.01 | 93.92 ± 0.97 |
| blocked | replay | B0 | 4 | 230400 | 0.00 ± 0.00 | 95.35 ± 0.16 | 95.07 ± 0.42 |
| blocked | replay | B0 | 5 | 256000 | 0.00 ± 0.00 | 95.28 ± 0.54 | 95.17 ± 0.59 |
| blocked | replay | B0 | 6 | 281600 | 0.00 ± 0.00 | 95.72 ± 0.45 | 95.48 ± 0.60 |
| blocked | replay | B0 | 7 | 537600 | 900.00 ± 0.00 | 93.44 ± 0.13 | 93.25 ± 0.40 |
| blocked | replay | B0 | 8 | 563200 | 0.00 ± 0.00 | 95.83 ± 0.19 | 95.52 ± 0.36 |
| blocked | replay | B0 | 9 | 588800 | 0.00 ± 0.00 | 96.00 ± 0.21 | 95.91 ± 0.22 |
| blocked | replay | C0 | 2 | 332800 | 0.00 ± 0.00 | 91.02 ± 0.70 | 89.00 ± 2.10 |
| blocked | replay | C0 | 3 | 358400 | 0.00 ± 0.00 | 95.32 ± 0.60 | 95.06 ± 0.43 |
| blocked | replay | C0 | 4 | 384000 | 0.00 ± 0.00 | 95.87 ± 0.32 | 96.04 ± 0.28 |
| blocked | replay | C0 | 5 | 409600 | 0.00 ± 0.00 | 96.17 ± 0.24 | 96.27 ± 0.16 |
| blocked | replay | C0 | 6 | 435200 | 0.00 ± 0.00 | 96.45 ± 0.28 | 96.48 ± 0.11 |
| blocked | replay | C0 | 7 | 614400 | 600.00 ± 0.00 | 94.31 ± 0.50 | 94.30 ± 0.52 |
| blocked | replay | C0 | 8 | 640000 | 0.00 ± 0.00 | 96.06 ± 0.46 | 96.11 ± 0.34 |
| blocked | replay | C0 | 9 | 665600 | 0.00 ± 0.00 | 96.49 ± 0.12 | 96.54 ± 0.27 |
| recurrent | online | A0 | 2 | 76800 | 200.00 ± 0.00 | 2.25 ± 0.26 | 2.43 ± 0.37 |
| recurrent | online | A0 | 3 | 153600 | 200.00 ± 0.00 | 5.85 ± 1.51 | 5.88 ± 1.88 |
| recurrent | online | A0 | 4 | 230400 | 200.00 ± 0.00 | 9.69 ± 4.50 | 10.58 ± 4.60 |
| recurrent | online | A0 | 5 | 307200 | 200.00 ± 0.00 | 14.22 ± 2.15 | 17.68 ± 2.18 |
| recurrent | online | A0 | 6 | 384000 | 200.00 ± 0.00 | 53.91 ± 9.42 | 57.78 ± 10.76 |
| recurrent | online | A1 | 2 | 537600 | 200.00 ± 0.00 | 13.29 ± 5.60 | 14.42 ± 4.31 |
| recurrent | online | A1 | 3 | 614400 | 200.00 ± 0.00 | 83.25 ± 10.35 | 84.21 ± 10.47 |
| recurrent | online | B0 | 2 | 102400 | 200.00 ± 0.00 | 73.38 ± 0.66 | 72.59 ± 0.64 |
| recurrent | online | B0 | 3 | 179200 | 200.00 ± 0.00 | 74.08 ± 0.96 | 73.48 ± 1.56 |
| recurrent | online | B0 | 4 | 256000 | 200.00 ± 0.00 | 75.16 ± 0.79 | 74.69 ± 0.60 |
| recurrent | online | B0 | 5 | 332800 | 200.00 ± 0.00 | 76.77 ± 0.23 | 75.89 ± 0.77 |
| recurrent | online | B0 | 6 | 409600 | 200.00 ± 0.00 | 78.59 ± 0.31 | 78.28 ± 1.01 |
| recurrent | online | B0 | 7 | 486400 | 200.00 ± 0.00 | 8.88 ± 0.79 | 9.29 ± 0.75 |
| recurrent | online | B0 | 8 | 563200 | 200.00 ± 0.00 | 39.81 ± 17.44 | 42.16 ± 15.89 |
| recurrent | online | B0 | 9 | 640000 | 200.00 ± 0.00 | 90.44 ± 0.83 | 90.51 ± 0.27 |
| recurrent | online | C0 | 2 | 128000 | 200.00 ± 0.00 | 8.44 ± 0.91 | 8.28 ± 0.85 |
| recurrent | online | C0 | 3 | 204800 | 200.00 ± 0.00 | 8.65 ± 0.81 | 8.48 ± 0.58 |
| recurrent | online | C0 | 4 | 281600 | 200.00 ± 0.00 | 9.15 ± 1.02 | 9.29 ± 1.31 |
| recurrent | online | C0 | 5 | 358400 | 200.00 ± 0.00 | 18.23 ± 2.73 | 18.97 ± 4.46 |
| recurrent | online | C0 | 6 | 435200 | 200.00 ± 0.00 | 72.65 ± 12.47 | 73.25 ± 9.88 |
| recurrent | online | C0 | 7 | 512000 | 200.00 ± 0.00 | 79.09 ± 7.25 | 80.72 ± 5.25 |
| recurrent | online | C0 | 8 | 588800 | 200.00 ± 0.00 | 90.23 ± 1.17 | 89.81 ± 1.39 |
| recurrent | online | C0 | 9 | 665600 | 200.00 ± 0.00 | 93.70 ± 1.06 | 93.52 ± 1.15 |
| recurrent | repeat | A0 | 2 | 76800 | 200.00 ± 0.00 | 1.66 ± 0.32 | 1.64 ± 0.34 |
| recurrent | repeat | A0 | 3 | 153600 | 200.00 ± 0.00 | 5.00 ± 3.24 | 4.75 ± 2.56 |
| recurrent | repeat | A0 | 4 | 230400 | 200.00 ± 0.00 | 11.58 ± 4.76 | 12.39 ± 4.56 |
| recurrent | repeat | A0 | 5 | 307200 | 200.00 ± 0.00 | 29.50 ± 12.43 | 33.61 ± 13.85 |
| recurrent | repeat | A0 | 6 | 384000 | 200.00 ± 0.00 | 74.41 ± 8.81 | 74.85 ± 11.35 |
| recurrent | repeat | A1 | 2 | 537600 | 200.00 ± 0.00 | 13.16 ± 1.13 | 15.13 ± 2.40 |
| recurrent | repeat | A1 | 3 | 614400 | 200.00 ± 0.00 | 90.76 ± 0.46 | 91.10 ± 0.79 |
| recurrent | repeat | B0 | 2 | 102400 | 200.00 ± 0.00 | 75.00 ± 0.91 | 74.57 ± 0.32 |
| recurrent | repeat | B0 | 3 | 179200 | 200.00 ± 0.00 | 74.01 ± 0.78 | 73.51 ± 0.41 |
| recurrent | repeat | B0 | 4 | 256000 | 200.00 ± 0.00 | 74.41 ± 1.22 | 73.95 ± 0.42 |
| recurrent | repeat | B0 | 5 | 332800 | 200.00 ± 0.00 | 76.18 ± 1.28 | 75.48 ± 0.69 |
| recurrent | repeat | B0 | 6 | 409600 | 200.00 ± 0.00 | 78.30 ± 0.38 | 78.20 ± 0.15 |
| recurrent | repeat | B0 | 7 | 486400 | 200.00 ± 0.00 | 9.03 ± 0.97 | 9.31 ± 0.62 |
| recurrent | repeat | B0 | 8 | 563200 | 200.00 ± 0.00 | 52.95 ± 7.09 | 53.61 ± 8.84 |
| recurrent | repeat | B0 | 9 | 640000 | 200.00 ± 0.00 | 90.24 ± 0.46 | 90.37 ± 0.67 |
| recurrent | repeat | C0 | 2 | 128000 | 200.00 ± 0.00 | 8.33 ± 0.93 | 8.14 ± 0.83 |
| recurrent | repeat | C0 | 3 | 204800 | 200.00 ± 0.00 | 8.80 ± 1.24 | 8.83 ± 0.66 |
| recurrent | repeat | C0 | 4 | 281600 | 200.00 ± 0.00 | 11.51 ± 0.14 | 11.71 ± 2.74 |
| recurrent | repeat | C0 | 5 | 358400 | 200.00 ± 0.00 | 43.03 ± 19.94 | 42.34 ± 20.76 |
| recurrent | repeat | C0 | 6 | 435200 | 200.00 ± 0.00 | 82.25 ± 5.29 | 82.41 ± 4.18 |
| recurrent | repeat | C0 | 7 | 512000 | 200.00 ± 0.00 | 70.77 ± 11.81 | 73.04 ± 10.63 |
| recurrent | repeat | C0 | 8 | 588800 | 200.00 ± 0.00 | 89.00 ± 0.68 | 88.66 ± 1.10 |
| recurrent | repeat | C0 | 9 | 665600 | 200.00 ± 0.00 | 93.89 ± 1.10 | 93.79 ± 0.85 |
| recurrent | replay | A0 | 2 | 76800 | 200.00 ± 0.00 | 87.11 ± 2.61 | 86.20 ± 2.30 |
| recurrent | replay | A0 | 3 | 153600 | 200.00 ± 0.00 | 93.81 ± 0.73 | 94.03 ± 0.66 |
| recurrent | replay | A0 | 4 | 230400 | 200.00 ± 0.00 | 95.49 ± 0.63 | 95.70 ± 0.26 |
| recurrent | replay | A0 | 5 | 307200 | 200.00 ± 0.00 | 95.94 ± 0.73 | 95.96 ± 0.32 |
| recurrent | replay | A0 | 6 | 384000 | 200.00 ± 0.00 | 95.92 ± 0.52 | 96.02 ± 0.43 |
| recurrent | replay | A1 | 2 | 537600 | 200.00 ± 0.00 | 88.68 ± 0.87 | 88.37 ± 2.04 |
| recurrent | replay | A1 | 3 | 614400 | 200.00 ± 0.00 | 94.43 ± 0.72 | 94.26 ± 0.92 |
| recurrent | replay | B0 | 2 | 102400 | 200.00 ± 0.00 | 85.79 ± 0.45 | 85.63 ± 0.63 |
| recurrent | replay | B0 | 3 | 179200 | 200.00 ± 0.00 | 93.11 ± 0.47 | 93.08 ± 0.09 |
| recurrent | replay | B0 | 4 | 256000 | 200.00 ± 0.00 | 94.22 ± 0.32 | 94.09 ± 0.22 |
| recurrent | replay | B0 | 5 | 332800 | 200.00 ± 0.00 | 95.08 ± 0.64 | 94.80 ± 0.32 |
| recurrent | replay | B0 | 6 | 409600 | 200.00 ± 0.00 | 95.28 ± 0.15 | 95.11 ± 0.18 |
| recurrent | replay | B0 | 7 | 486400 | 200.00 ± 0.00 | 93.25 ± 0.44 | 93.05 ± 0.47 |
| recurrent | replay | B0 | 8 | 563200 | 200.00 ± 0.00 | 95.19 ± 0.38 | 94.69 ± 0.41 |
| recurrent | replay | B0 | 9 | 640000 | 200.00 ± 0.00 | 95.93 ± 0.27 | 95.62 ± 0.23 |
| recurrent | replay | C0 | 2 | 128000 | 200.00 ± 0.00 | 93.40 ± 0.11 | 93.50 ± 0.51 |
| recurrent | replay | C0 | 3 | 204800 | 200.00 ± 0.00 | 95.19 ± 0.43 | 95.22 ± 0.54 |
| recurrent | replay | C0 | 4 | 281600 | 200.00 ± 0.00 | 95.59 ± 0.60 | 95.59 ± 0.55 |
| recurrent | replay | C0 | 5 | 358400 | 200.00 ± 0.00 | 96.07 ± 0.26 | 96.21 ± 0.33 |
| recurrent | replay | C0 | 6 | 435200 | 200.00 ± 0.00 | 96.19 ± 0.26 | 96.48 ± 0.26 |
| recurrent | replay | C0 | 7 | 512000 | 200.00 ± 0.00 | 93.71 ± 0.31 | 93.76 ± 0.29 |
| recurrent | replay | C0 | 8 | 588800 | 200.00 ± 0.00 | 95.18 ± 0.10 | 95.20 ± 0.18 |
| recurrent | replay | C0 | 9 | 665600 | 200.00 ± 0.00 | 95.72 ± 0.19 | 95.61 ± 0.25 |

## Measured budgets and resources

Wall time and peak allocation are descriptive measurements, not matched-compute evidence. Model/buffer sizes and all paired hashes are embedded in summary.json.

| Order | Method | new_examples | optimizer_steps | training_examples | replayed_examples | buffer_bytes | elapsed_seconds | peak_cuda_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blocked | online | 691200.00 ± 0.00 | 2700.00 ± 0.00 | 691200.00 ± 0.00 | 0.00 ± 0.00 | 1343488.00 ± 0.00 | 18.26 ± 0.08 | 30082560.00 ± 0.00 |
| blocked | repeat | 691200.00 ± 0.00 | 5400.00 ± 0.00 | 1382400.00 ± 0.00 | 0.00 ± 0.00 | 1343488.00 ± 0.00 | 22.30 ± 0.18 | 30082560.00 ± 0.00 |
| blocked | replay | 691200.00 ± 0.00 | 5400.00 ± 0.00 | 1382400.00 ± 0.00 | 690944.00 ± 0.00 | 1343488.00 ± 0.00 | 22.09 ± 0.08 | 30122496.00 ± 0.00 |
| recurrent | online | 691200.00 ± 0.00 | 2700.00 ± 0.00 | 691200.00 ± 0.00 | 0.00 ± 0.00 | 1343488.00 ± 0.00 | 18.30 ± 0.11 | 30082560.00 ± 0.00 |
| recurrent | repeat | 691200.00 ± 0.00 | 5400.00 ± 0.00 | 1382400.00 ± 0.00 | 0.00 ± 0.00 | 1343488.00 ± 0.00 | 21.76 ± 0.11 | 30082560.00 ± 0.00 |
| recurrent | replay | 691200.00 ± 0.00 | 5400.00 ± 0.00 | 1382400.00 ± 0.00 | 690944.00 ± 0.00 | 1343488.00 ± 0.00 | 21.98 ± 0.21 | 30122496.00 ± 0.00 |

## Verification

- Each input manifest's full Cartesian product is complete, with no unindexed or duplicate results.
- All six order/method combinations contain the same exactly three independent seed IDs.
- Core configurations, parameter counts, and verified source snapshots match.
- Per-seed teacher/world hashes, initial parameters, and per-condition external streams match.
- Per-seed/order final buffer hashes match across methods.
- Schedules, evaluation chains, checkpoints, return ages, and external/update budgets are complete.

Figures: trajectories.png / trajectories.svg and final_accuracy.png / final_accuracy.svg. The trajectory bands and final error bars show sample SD. The current-rule curve uses A0/B0/C0 before the revision cue, then A1/B0/C0. Its vertical jump at the cue evaluates the new rule with the same pre-update parameters.
