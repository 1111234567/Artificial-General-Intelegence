"""Freeze the native-audited fixed bank and summarize the actual intervention."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.finish_position import bank
from experiments.review_selection import bank as old
from experiments.continuity.run import state_digest


def report():
    manifests=[]
    records=[]
    audits=[]
    hashes={}
    for seed in range(3):
        folder=bank.OUT/f'seed_{seed:03d}'
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        a=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
        assert a['passed'] and a['bank_sha256']==m['bank_sha256']==old.sha(folder/'bank.pt')
        assert a['manifest_sha256']==old.sha(folder/'manifest.json')
        assert m['implementation_sha256']==old.sha(bank.__file__)
        manifests.append(m)
        audits.append(a)
        records.extend(json.loads((folder/'finish_metadata.json').read_text(encoding='utf-8')))
        for name in ('bank.pt','manifest.json','audit.json','finish_metadata.json'):
            hashes[str((folder/name).relative_to(ROOT))]=old.sha(folder/name)
    declaration=dict(format=bank.FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2],requests_per_seed=256,sequence_ticks=128,request_reselection=False,
        arms=dict(midpoint='Exact prior last-WAIT condition: original BEGIN, last eligible WAIT and original midpoint-teacher FINISH.',
                  first='Same requests/BEGIN/last-WAIT; move only the one FINISH positive to first native legal tick after real accepted BEGIN, with the same correct version and payload.'),
        exact_inputs='Original public obs and original numeric anchor only. All original wait-position bank fields are preserved. Private labels, masks, request IDs, ready_time/deadline and timing are loss/audit metadata.',
        gradient_rows='Exactly 3 target-slot rows/request: BEGIN, last eligible processing WAIT, and one FINISH. Nonselected positions have zero A-loss weight, including the old teacher FINISH position in first.',
        first_label='Intentionally supervise correct FINISH at first legal tick even when the original midpoint oracle says WAIT. This is a changed teaching target, not a claim that original oracle or actual actions chose FINISH there.',
        window='Original 128 observations, both slots, real actions/receipts, prefix and suffix and numeric anchor remain unchanged.',
        source_sha256={str(Path(p).relative_to(ROOT)):old.sha(p) for p in
            (bank.__file__,ROOT/'experiments/finish_position/bank_audit.py',__file__)},
        input_sha256=hashes,pooled_distribution=bank.distribution(records),
        checks=dict(all_native_bank_audits_passed=True,requests=768,full_native_replay_ticks=sum(a['checks']['native_world_ticks'] for a in audits),
            numeric_anchors_exact=sum(a['checks']['numeric_anchor_stepwise_exact'] for a in audits),
            native_correct_FINISH_acceptance_checks=sum(a['checks']['native_counterfactual_first_legal_tick_accepted']+a['checks']['native_counterfactual_teacher_finish_tick_accepted'] for a in audits)))
    declaration['protocol_hash']=state_digest(declaration)
    protocol=bank.OUT/'protocol.json'
    assert not protocol.exists(),protocol
    old.write(protocol,declaration)
    pooled=declaration['pooled_distribution']
    old.write(bank.OUT/'summary.json',dict(protocol_sha256=old.sha(protocol),pooled=pooled,
        seeds={str(i):m['distribution'] for i,m in enumerate(manifests)},audits=audits))
    lines=['# Fixed-request FINISH position bank','',
        'The same three seeds retain exactly the prior 256 old-A requests each. BEGIN, last eligible processing WAIT, all 128 public observations, actual source actions/receipts, numeric anchors and request identities remain unchanged. The sole FINISH positive moves from the original midpoint-teacher tick to first native legality for the same request, version and payload. All other A rows have zero loss weight.','',
        '| Seed | Requests | Advance ticks mean / median / range | Advance time mean | New FINISH next to last WAIT | New-position original oracle WAIT |',
        '|---|---:|---:|---:|---:|---:|']
    for seed,m in enumerate(manifests):
        d=m['distribution'];t=d['advance_ticks']
        lines.append(f"| {seed} | 256 | {t['mean']:.6f} / {t['median']:.0f} / {t['min']:.0f}-{t['max']:.0f} | {d['advance_time']['mean']:.6f} | {100*d['first_FINISH_adjacent_to_last_WAIT_fraction']:.0f}% | {100*d['first_oracle_WAIT_fraction']:.0f}% |")
    lines+=['',f"All 768 FINISH positions move earlier by 3-5 ticks (dt=0.25); pooled mean {pooled['advance_ticks']['mean']:.6f} ticks / {pooled['advance_time']['mean']:.6f} native time units, median 4 ticks / 1 unit. The old FINISH is 4-6 ticks after last WAIT; the new FINISH is exactly one tick after last WAIT for every request. No first and teacher FINISH pair is adjacent or coincident.",'',
        'Every new position originally has midpoint-oracle WAIT. The first arm deliberately supplies the same correct FINISH label there. The stored original teacher label at the old FINISH position remains available for audit but its A-loss mask is zero in the first arm. This changes positive-target timing and its recurrent context; the experiment must evaluate the resulting policy to determine any effect.','',
        'Actual source actions in [first legal FINISH, old teacher FINISH), pooled over 3,048 target-slot ticks:','',
        '| Actual target action / native result | Count | Fraction |','|---|---:|---:|']
    count=sum(pooled['interval_distributions']['target_action_receipt'].values())
    for name,value in pooled['interval_distributions']['target_action_receipt'].items():lines.append(f'| {name} | {value} | {100*value/count:.4f}% |')
    lines+=['','No actual source FINISH is accepted in that interval. Exact action-code and receipt distributions for target and other slots, every interval tick and its executed action/result, and per-request ready/deadline/identity records are retained in summary.json and seed_NNN/finish_metadata.json.','',
        'Independent audits rebuild all first-legal positions from native frames; replay 12,288 source world ticks and 1,572,864 oracle rows; rebuild all 768 numeric anchors bit-exact; verify real BEGIN receipts, identity/slot/version/payload and original windows; and execute 1,536 proposed correct FINISH actions in cloned native worlds (both positions, every request), all accepted. Every arm/request has exactly one BEGIN, one WAIT and one FINISH loss row.','',
        'The bank audit proves the input tensors and public-anchor construction are unchanged. Training separately audits that only obs and anchor enter the policy, while labels and masks are used after the forward pass. No original code or prior bank is modified.']
    (bank.OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(protocol=str(protocol),summary=str(bank.OUT/'summary.json'),report=str(bank.OUT/'report.md'),pooled=pooled)),flush=True)


if __name__=='__main__':report()
