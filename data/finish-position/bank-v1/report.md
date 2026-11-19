# Fixed-request FINISH position bank

The same three seeds retain exactly the prior 256 old-A requests each. BEGIN, last eligible processing WAIT, all 128 public observations, actual source actions/receipts, numeric anchors and request identities remain unchanged. The sole FINISH positive moves from the original midpoint-teacher tick to first native legality for the same request, version and payload. All other A rows have zero loss weight.

| Seed | Requests | Advance ticks mean / median / range | Advance time mean | New FINISH next to last WAIT | New-position original oracle WAIT |
|---|---:|---:|---:|---:|---:|
| 0 | 256 | 3.992188 / 4 / 3-5 | 0.998047 | 100% | 100% |
| 1 | 256 | 3.933594 / 4 / 3-5 | 0.983398 | 100% | 100% |
| 2 | 256 | 3.980469 / 4 / 3-5 | 0.995117 | 100% | 100% |

All 768 FINISH positions move earlier by 3-5 ticks (dt=0.25); pooled mean 3.968750 ticks / 0.992188 native time units, median 4 ticks / 1 unit. The old FINISH is 4-6 ticks after last WAIT; the new FINISH is exactly one tick after last WAIT for every request. No first and teacher FINISH pair is adjacent or coincident.

Every new position originally has midpoint-oracle WAIT. The first arm deliberately supplies the same correct FINISH label there. The stored original teacher label at the old FINISH position remains available for audit but its A-loss mask is zero in the first arm. This changes positive-target timing and its recurrent context; the experiment must evaluate the resulting policy to determine any effect.

Actual source actions in [first legal FINISH, old teacher FINISH), pooled over 3,048 target-slot ticks:

| Actual target action / native result | Count | Fraction |
|---|---:|---:|
| WAIT:0 | 2901 | 95.1772% |
| FINISH:-1 | 147 | 4.8228% |

No actual source FINISH is accepted in that interval. Exact action-code and receipt distributions for target and other slots, every interval tick and its executed action/result, and per-request ready/deadline/identity records are retained in summary.json and seed_NNN/finish_metadata.json.

Independent audits rebuild all first-legal positions from native frames; replay 12,288 source world ticks and 1,572,864 oracle rows; rebuild all 768 numeric anchors bit-exact; verify real BEGIN receipts, identity/slot/version/payload and original windows; and execute 1,536 proposed correct FINISH actions in cloned native worlds (both positions, every request), all accepted. Every arm/request has exactly one BEGIN, one WAIT and one FINISH loss row.

The bank audit proves the input tensors and public-anchor construction are unchanged. Training separately audits that only obs and anchor enter the policy, while labels and masks are used after the forward pass. No original code or prior bank is modified.
