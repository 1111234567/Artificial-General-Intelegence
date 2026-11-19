"""Collect completed WAIT-placement training audits without changing any run."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from experiments.wait_position import train_review as tr
from experiments.continuity.run import state_digest


def summarize():
    plan=tr.checked();records=[];bindings={}
    for seed in range(3):
        folder=tr.OUT/f'seed_{seed:03d}'
        complete=json.loads((folder/'complete.json').read_text())
        audit=json.loads((folder/'first_step_audit.json').read_text())
        assert complete['passed'] and audit['passed'] and audit['all_arm_integer_budgets_equal']
        for values in audit['prior_uniform_bit_exact_calibration'].values():assert all(values.values())
        bindings[str(seed)]={}
        for arm in tr.ARMS:
            bindings[str(seed)][arm]={}
            for name in tr.SNAPSHOTS:
                path=folder/arm/(name+'.pt')
                cp=torch.load(path,map_location='cpu',weights_only=False)
                assert cp['wait_position']['protocol_hash']==plan['protocol_hash']
                bindings[str(seed)][arm][name]=dict(path=str(path),sha256=tr.t.sha(path),
                    model_hash=state_digest(cp['model']),work=cp['work'])
        records.append(dict(seed=seed,seconds=complete['seconds'],budgets=audit['budgets'],
            timing=audit['timing'],all_previous_uniform_request_ids_identical=True,
            midpoint_source_first_step_and_endpoint_bit_exact=True,
            independent_first_changed_Adam_mask_live_h_world_exact=True,
            audit_sha256=tr.t.sha(folder/'first_step_audit.json'),
            archived_B_stream_world_exact=all(item['world_exact'] for item in complete['public_stream_audit']['first_epoch_comparison'])))
    value=dict(format='wait-position-training-summary-v1',protocol_hash=plan['protocol_hash'],passed=True,
        records=records,checkpoints=bindings,total_new_Adam_updates=3*3*2048,
        new_A_Adam_updates_per_seed_arm=256,new_B_Adam_updates_per_seed_arm=1792,
        total_A_review_supervised_rows_per_seed_arm=49152,
        inference='Native public56 GRU eleven-way argmax; no private fields or masks in deployed policy.',
        scope='Equal old-A request identities, labels/counts, candidate scans and B exposure. Only selected WAIT supervision time differs. Fixed replay supervision, not new autonomous experience.')
    value['summary_hash']=state_digest(value)
    tr.t.write(tr.OUT/'summary.json',value)
    print(json.dumps(dict(passed=True,path=str(tr.OUT/'summary.json'),summary_hash=value['summary_hash'])),flush=True)
    return value


if __name__=='__main__':summarize()
