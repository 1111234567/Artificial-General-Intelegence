# Old-A fitting-bank boundary readout

Read-only frozen B000/B008 policies see the same256 original A public windows per seed and the exact original B000-derived numeric anchors. All768 requests are retained. This is an anchored off-policy diagnostic of the fitting bank. It measures action intents on stored histories; it does not measure free behavior, heldout generalization, actual acceptance or completion.

| Group | Endpoint | n | Last WAIT chooses WAIT | First legal exact FINISH | Teacher tick exact FINISH | WAIT then next exact FINISH | Any processing FINISH |
|---|---|---:|---:|---:|---:|---:|---:|
| all | B000 | 768 | 754/768 (98.18%) | 28/768 (3.65%) | 533/768 (69.40%) | 17/768 (2.21%) | 14/768 (1.82%) |
| all | midpoint | 768 | 762/768 (99.22%) | 50/768 (6.51%) | 734/768 (95.57%) | 45/768 (5.86%) | 6/768 (0.78%) |
| all | first | 768 | 342/768 (44.53%) | 697/768 (90.76%) | 644/768 (83.85%) | 308/768 (40.10%) | 430/768 (55.99%) |
| conflict | B000 | 384 | 378/384 (98.44%) | 18/384 (4.69%) | 253/384 (65.89%) | 12/384 (3.12%) | 6/384 (1.56%) |
| conflict | midpoint | 384 | 379/384 (98.70%) | 24/384 (6.25%) | 350/384 (91.15%) | 20/384 (5.21%) | 5/384 (1.30%) |
| conflict | first | 384 | 163/384 (42.45%) | 322/384 (83.85%) | 286/384 (74.48%) | 138/384 (35.94%) | 223/384 (58.07%) |
| other | B000 | 384 | 376/384 (97.92%) | 10/384 (2.60%) | 280/384 (72.92%) | 5/384 (1.30%) | 8/384 (2.08%) |
| other | midpoint | 384 | 383/384 (99.74%) | 26/384 (6.77%) | 384/384 (100.00%) | 25/384 (6.51%) | 1/384 (0.26%) |
| other | first | 384 | 179/384 (46.61%) | 375/384 (97.66%) | 358/384 (93.23%) | 170/384 (44.27%) | 207/384 (53.91%) |

Exact FINISH means the native eleven-way argmax selects the correct version and payload. WAIT then FINISH requires both decisions on the same request and consecutive ticks. Processing FINISH means any FINISH argmax within the independently native-audited eligible processing interval, not an executed early action.

| Seed | Endpoint | Conflict n | Last WAIT correct | First legal exact FINISH | WAIT then next exact FINISH | Any processing FINISH |
|---|---|---:|---:|---:|---:|---:|
| 0 | B000 | 128 | 123 | 13 | 8 | 5 |
| 0 | midpoint | 128 | 127 | 6 | 5 | 1 |
| 0 | first | 128 | 43 | 116 | 40 | 86 |
| 1 | B000 | 128 | 127 | 5 | 4 | 1 |
| 1 | midpoint | 128 | 126 | 6 | 4 | 2 |
| 1 | first | 128 | 61 | 95 | 46 | 68 |
| 2 | B000 | 128 | 128 | 0 | 0 | 0 |
| 2 | midpoint | 128 | 126 | 12 | 11 | 2 |
| 2 | first | 128 | 59 | 111 | 52 | 69 |

Full numerator/denominator counts for all, conflict, other and every raw are in summary.json. Per-request three-boundary logits/actions and every processing tick decision are in seed_NNN/ARM.json; full two-slot logits and independent sequential logits are in ARM_logits.pt. Conditional8 payload/version correctness is retained separately from the eleven-way action choice.

All nine full-sequence forwards were checked against128 separate model.step calls from the same original anchor. The audit independently recomputes all counts using NumPy and checks every saved boundary and processing action. All inputs, parameters, logits and hidden states are finite; model weights and checkpoint/bank hashes remain unchanged.

Compare these fitting-bank metrics with the separately declared independent-teacher A/B readout under data/finish-position/boundary-v1. Histories and anchor construction differ, so bank-to-heldout differences are descriptive, not a controlled effect of one history feature. No diagnostic selection changed the training protocol or selected checkpoints.
