# Actual selected WAIT positions

This report reads retained actual trainer selections only. It does not rescore policies or change any choice. The three original arms each have 3 seeds x 256 updates x 64 selected requests = 49,152 selected WAIT rows. Physical times use the native world unit (source dt=0.25).

| Arm | Ready distance median | Processing fraction median | Last eligible tick | True FINISH argmax | Early phase <25% | Late phase >=75% |
|---|---:|---:|---:|---:|---:|---:|
| midpoint | 3.491289 | 0.516088 | 0.0000% | 0.0041% | 0.0000% | 0.0000% |
| hard | 0.126440 | 0.982731 | 92.1346% | 10.7279% | 4.5064% | 94.3034% |
| random | 2.860281 | 0.513333 | 5.3772% | 1.6561% | 23.0591% | 25.7751% |

Hard uses maximum FINISH-vs-nonFINISH margin among admissible WAIT ticks. A true FINISH intent means max(FINISH logits)>max(WAIT/BEGIN logits); otherwise the allowed protocol still chooses the largest margin.

| Hard subset | Rows | Ready distance median | Ready distance mean | Last eligible tick |
|---|---:|---:|---:|---:|
| True FINISH > nonFINISH | 5273 | 0.114741 | 0.159043 | 90.3850% |
| No FINISH anywhere in eligible window | 43879 | 0.127476 | 0.542893 | 92.3449% |

| Hard training phase | Rows | Last eligible tick | True FINISH argmax | Ready distance median | Early processing <25% |
|---|---:|---:|---:|---:|---:|
| First 32 review updates | 6144 | 92.4642% | 7.8288% | 0.124959 | 4.9805% |
| Last 32 review updates | 6144 | 92.0573% | 10.5143% | 0.127369 | 4.5898% |

All 768 retained requests have admissible early (<25%) and late (>=75%) processing candidates. Over the full run, hard supervises early processing for 185/768 requests and late processing for 766/768; random covers early/late phases in 768/768 requests. Hard last-eligible coincidences are seed 0: 97.0337%, seed 1: 96.4722%, seed 2: 82.8979%.

These describe the realized training positions and intents. They do not by themselves identify an outcome effect of model-based hardness separately from concentrating feedback near readiness.

Source: `mechanism_diagnostics.json`, SHA256 `24ce69fa56c6648396b9f91928a6c5ff6f71f828814a163602a43bbcd13a54de`. Compact exact values and source log hashes are in `mechanism_summary.json`.
