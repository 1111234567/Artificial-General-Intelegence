# Existing mixed-rollout request diagnosis

Pooled integer counts below describe these exact three saved streams. Per-seed and per-raw denominators/rates remain in summary.json and by-family-raw.csv. No new actions, model calls or training were performed.

| Arm | Family | Matured | Ordinary | Strict | No BEGIN | BEGIN/no legal window | First legal FINISH correct | Other context | Other wrong | No legal FINISH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | A | 118 | 74 | 8 | 0 | 0 | 49 | 27 | 17 | 25 |
| uniform | B | 117 | 98 | 76 | 0 | 0 | 82 | 19 | 8 | 8 |
| conflict | A | 118 | 50 | 0 | 0 | 0 | 37 | 18 | 10 | 53 |
| conflict | B | 117 | 114 | 83 | 0 | 0 | 98 | 13 | 4 | 2 |
| error | A | 118 | 81 | 2 | 0 | 0 | 51 | 21 | 33 | 13 |
| error | B | 117 | 95 | 60 | 0 | 0 | 72 | 37 | 6 | 2 |
| no_review | A | 118 | 10 | 0 | 0 | 0 | 2 | 89 | 19 | 8 |
| no_review | B | 117 | 116 | 115 | 0 | 0 | 115 | 0 | 1 | 1 |

First active rejection among ordinary-completed but nonstrict requests (exclusive priority in protocol.json):

| Arm | Family | Completed but nonstrict | First-error counts |
|---|---|---:|---|
| uniform | A | 66 | {"duplicate_BEGIN": 1, "early_FINISH": 62, "legal_window_other_context_payload": 3} |
| uniform | B | 22 | {"early_FINISH": 12, "legal_window_other_context_payload": 10} |
| conflict | A | 50 | {"early_FINISH": 49, "legal_window_other_context_payload": 1} |
| conflict | B | 31 | {"early_FINISH": 26, "legal_window_other_context_payload": 4, "legal_window_other_wrong_payload": 1} |
| error | A | 79 | {"early_FINISH": 72, "legal_window_other_context_payload": 6, "legal_window_other_wrong_payload": 1} |
| error | B | 35 | {"early_FINISH": 24, "legal_window_other_context_payload": 10, "legal_window_other_wrong_payload": 1} |
| no_review | A | 10 | {"early_FINISH": 4, "legal_window_other_context_payload": 5, "legal_window_other_wrong_payload": 1} |
| no_review | B | 1 | {"legal_window_other_wrong_payload": 1} |

Conflict-only review, old A: all118 requests started and reached a legal window; first active rejection counts={"early_FINISH": 116, "legal_window_other_context_payload": 1, "wrong_version": 1}. Among completed-but-nonstrict requests whose first error was early FINISH, payload counts={"correct_payload": 21, "other_context_payload": 24, "other_wrong_payload": 4}.

Newly issued eligible NEW or REPLACE commands after fork_tick in evaluation regime, whose score window matures within existing rollout. Exclude inherited warmup requests and right-censored requests. Within this cohort retain conflict raws only.

Observed pre-action active and started state with actual accepted-BEGIN ready_time <= now <= deadline, epsilon1e-6. Timing window is judged before action, independently of chosen version/payload.

For ordinary-completed but nonstrict requests locate earliest rejected non-WAIT action while active, prior to accepted FINISH. WAIT omissions are not rejection errors under native strict metric. Mutually exclusive priority: wrong version > FINISH before BEGIN > duplicate BEGIN > early FINISH > late FINISH > legal-window other-context payload > legal-window other wrong payload > invalid code > other. Also retain overlapping primitive flags, so early plus wrong-content is visible.

Teacher arrays were not saved in mixed artifacts. Reconstruct causal midpoint oracle labels from saved pre-action state for diagnosis only; no labels feed model or world.

Actual legal-window actions support a behavioral diagnosis, not proof of an erased or retained internal association. Correct-vs-other FINISH logit comparison at first legal tick is conditional readout, not completion.
