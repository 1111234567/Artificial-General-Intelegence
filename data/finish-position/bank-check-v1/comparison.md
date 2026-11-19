# Fitting-bank and independent-teacher boundary intents

The same two frozen B008 checkpoints are compared under each public history. Counts are pooled over three seeds. The fitting bank is old A only; the independent teacher includes conflicting A and B requests. Histories, request distributions and numeric-anchor construction differ, so differences between histories are descriptive. Both measurements are off-policy action intents, not free-action completion.

| History | Endpoint | n | Last WAIT chooses WAIT | First legal exact FINISH | Old teacher exact FINISH | WAIT then next exact FINISH | Any processing FINISH |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fitting_bank_conflict_A | midpoint | 384 | 379/384 (98.70%) | 24/384 (6.25%) | 350/384 (91.15%) | 20/384 (5.21%) | 5/384 (1.30%) |
| fitting_bank_conflict_A | first | 384 | 163/384 (42.45%) | 322/384 (83.85%) | 286/384 (74.48%) | 138/384 (35.94%) | 223/384 (58.07%) |
| independent_teacher_conflict_A | midpoint | 261 | 253/261 (96.93%) | 15/261 (5.75%) | 126/261 (48.28%) | 8/261 (3.07%) | 8/261 (3.07%) |
| independent_teacher_conflict_A | first | 261 | 66/261 (25.29%) | 103/261 (39.46%) | 106/261 (40.61%) | 25/261 (9.58%) | 197/261 (75.48%) |
| independent_teacher_conflict_B | midpoint | 225 | 223/225 (99.11%) | 9/225 (4.00%) | 71/225 (31.56%) | 7/225 (3.11%) | 2/225 (0.89%) |
| independent_teacher_conflict_B | first | 225 | 214/225 (95.11%) | 21/225 (9.33%) | 135/225 (60.00%) | 10/225 (4.44%) | 11/225 (4.89%) |

On the fitting bank, moving FINISH to first legality raises correct first-legal intent but also makes last-WAIT FINISH intent common; the two-step conjunction remains below40% for conflicting A. The independent-teacher A readout also shows substantially lower last-WAIT accuracy and higher processing-FINISH intent for first. B shows a smaller shift. These measurements alone do not establish which public-history feature causes the differences.

Fitting-bank counts have the independent NumPy and full-vs-step audit in audit.json. Independent-teacher counts are imported from boundary-v1/{midpoint,first}_{0,1,2}.json, with exact source hashes and identical-model checks recorded in comparison.json; their separate audit is owned by the main evaluation.
