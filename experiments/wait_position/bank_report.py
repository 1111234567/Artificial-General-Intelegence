"""Freeze source-bound bank protocol and summarize actual candidate coverage."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.wait_position import bank
from experiments.review_selection import bank as old
from experiments.continuity.run import state_digest


def freeze():
    path=bank.OUT/'protocol.json'
    if path.exists():
        raise FileExistsError(path)
    seeds={}
    for seed in range(3):
        folder=bank.OUT/f'seed_{seed:03d}'
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
        assert audit['passed']
        assert old.sha(folder/'bank.pt')==manifest['bank_sha256']==audit['bank_sha256']
        assert old.sha(folder/'manifest.json')==audit['manifest_sha256']
        assert old.sha(bank.__file__)==manifest['implementation_sha256']
        seeds[str(seed)]=dict(files={str((folder/name).relative_to(ROOT)):old.sha(folder/name)
            for name in ('bank.pt','manifest.json','audit.json','eligible_metadata.json')},
            original_bank=manifest['original_bank'],original_bank_sha256=manifest['original_bank_sha256'],
            distribution=manifest['distribution'],audit_checks=audit['checks'])
    value=dict(format='wait-position-fixed-A-bank-protocol-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=seeds,samples_per_seed=256,sequence_ticks=128,request_reselection=False,
        original_fields_unchanged='Every field of the original review-selection bank is inherited exactly, including obs,actions,labels,mask,request_mask,anchor,positions,ids,metadata.',
        admissible='Same original request, actor and slot, original128-tick window, strictly after accepted actual BEGIN, native active and started, now<ready_time, midpoint teacher WAIT0. Original midpoint is included; original BEGIN and FINISH excluded.',
        schema='eligible_mask bool[128,256,2]; eligible_time_to_ready float64[128,256,2], physical ready_time-now; eligible_processing_fraction float64[128,256,2], elapsed since accepted BEGIN / duration. Floats are NaN outside mask. eligible_metadata contains all admissible positions and accepted-BEGIN provenance.',
        policy_inputs='Only original public obs and numeric anchor. New mask and timing arrays are private selection/audit metadata and must never enter the GRU, logits, action gates or loss labels.',
        source_sha256={str(Path(p).relative_to(ROOT)):old.sha(p) for p in (bank.__file__,ROOT/'experiments/wait_position/bank_audit.py',__file__)})
    value['protocol_hash']=state_digest(value)
    old.write(path,value)
    old.write(bank.OUT/'summary.json',dict(format=value['format'],protocol_sha256=old.sha(path),
        all_three_native_audits_passed=True,seeds=seeds))
    print(json.dumps(dict(protocol=str(path),protocol_hash=value['protocol_hash'],
        total_candidates={seed:row['distribution']['total_candidates'] for seed,row in seeds.items()})),flush=True)


if __name__=='__main__':
    freeze()
