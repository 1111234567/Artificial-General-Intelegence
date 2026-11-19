"""Read-only deployment/pair/Adam and paired-stream audit of review outputs."""
from __future__ import annotations
import json
import torch
from experiments.learn_remember import symmetric_review as r
from experiments.continuity.run import state_digest


def run():
    torch.set_num_threads(1);r.checked();records=[]
    for seed in range(3):
        folder=r.OUT/f'seed_{seed:03d}';stages=[]
        for stage in ('review_mid','review'):
            dense=torch.load(folder/f'{stage}.pt',map_location='cpu',weights_only=False)
            packed=torch.load(folder/f'{stage}.packed.pt',map_location='cpu',weights_only=False)
            model=r.symmetric.constrain(r.plain.GRUAgent(**dense['model_config']))
            model.load_state_dict(packed['model'],strict=True)
            optimizer=torch.optim.Adam(model.parameters(),lr=dense['config']['lr']);optimizer.load_state_dict(packed['optimizer'])
            moments=r.validate_optimizer(model,optimizer)
            deployed=r.symmetric.deploy(model)
            if any(not torch.equal(value,dense['model'][name]) for name,value in deployed.state_dict().items()):raise AssertionError('packed checkpoint changes deployment')
            for key in ('world','numpy_rng','work'):
                if state_digest(packed[key])!=state_digest(dense[key]):raise AssertionError('paired state mismatch '+key)
            for key in ('hidden','torch_rng'):
                if not torch.equal(packed[key],dense[key]):raise AssertionError('paired state mismatch '+key)
            if state_digest(packed['optimizer'])!=state_digest(dense['optimizer']):raise AssertionError('dense archive changed packed Adam')
            stages.append(dict(stage=stage,Adam=moments,dense_packed_full_state_exact=True,packed_shape=list(model.core.parametrizations.weight_hh_l0.original.shape),
                dense_sha256=r.plain.sha(folder/f'{stage}.pt'),packed_sha256=r.plain.sha(folder/f'{stage}.packed.pt')))
        logs=[json.loads(line) for line in (folder/'review_updates.jsonl').read_text().splitlines()]
        if len(logs)!=12:raise AssertionError('wrong round count')
        for index,row in enumerate(logs):
            if row['fit']['updates']!={'live':224,'review':32} or row['fit']['historical_source_round']!=12+index%4:raise AssertionError('wrong review schedule')
            live=torch.load(folder/f'trajectory_review_{index:03d}.pt',map_location='cpu',weights_only=False)
            old=torch.load(folder/f'trajectory_withdrawal_{index:03d}.pt',map_location='cpu',weights_only=False)
            if not torch.equal(live['teacher_lanes'],old['teacher_lanes']) or live['exposure_after']!=old['exposure_after']:raise AssertionError('intervention schedule changed')
            a=live['obs'].reshape(-1,64,2,28);b=old['obs'].reshape(-1,64,2,28)
            if not torch.equal(a[...,:11],b[...,:11]) or not torch.equal(a[...,24:28],b[...,24:28]):raise AssertionError('external public stream changed')
        native=torch.load(folder/'withdrawal.pt',map_location='cpu',weights_only=False)
        final=torch.load(folder/'review.pt',map_location='cpu',weights_only=False)
        for key in ('scheduler','event_hash'):
            if state_digest(native['world'][key])!=state_digest(final['world'][key]):raise AssertionError('external final state differs '+key)
        records.append(dict(seed=seed,stages=stages,rounds=12,review_updates=384,all_external_teacher_lane_schedule_exact=True,
            final_scheduler_event_hash_exact=True,trainer_audit_sha256=r.plain.sha(folder/'review_audit.json')))
    value=dict(passed=True,format=r.FORMAT,records=records,source_sha256=r.plain.sha(__file__),protocol_sha256=r.plain.sha(r.PLAN))
    r.plain.write_json(r.OUT/'review-independent-audit.json',value)
    print(json.dumps(dict(passed=True,seeds=3)))


if __name__=='__main__':run()
