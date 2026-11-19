"""Independent replay of the first real A choice and Adam step (no trainer helpers)."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.learn_remember.gru_train import GRUAgent
from experiments.continuity.run import state_digest

OUT=ROOT/'data/review-selection/train-v1'


def audit(seed):
    torch.set_num_threads(1)
    bank=torch.load(ROOT/f'data/review-selection/bank-v1/seed_{seed:03d}/bank.pt',map_location='cpu',weights_only=False)
    reports=[]
    for arm in ('uniform','conflict','error'):
        folder=OUT/f'seed_{seed:03d}'/arm
        before=torch.load(folder/'first_review_before.pt',map_location='cpu',weights_only=False)
        after=torch.load(folder/'first_review_after.pt',map_location='cpu',weights_only=False)
        record=torch.load(folder/'first_selection.pt',map_location='cpu',weights_only=False)
        rng=np.random.default_rng(1140000+seed*10000)
        raw=bank['raw'].numpy()
        ids=np.concatenate([rng.choice(np.flatnonzero(raw==r),32,replace=False) for r in range(4)])
        rng.shuffle(ids);ids=torch.from_numpy(ids)
        assert torch.equal(ids,record['candidate_ids'])
        model=GRUAgent(**before['model_config']).cuda();model.load_state_dict(before['model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=before['config']['lr']);optimizer.load_state_dict(deepcopy(before['optimizer']))
        obs=bank['obs'][:,ids].contiguous().cuda();labels=bank['labels'][:,ids].contiguous().cuda()
        mask=bank['mask'][:,ids].contiguous().cuda();anchor=bank['anchor'][ids].contiguous().cuda()
        with torch.no_grad():
            logits,_=model(obs,anchor)
            errors=F.cross_entropy(logits.reshape(-1,11),labels.reshape(-1),reduction='none').reshape_as(labels)
            values=(errors*mask).sum((0,2))/mask.sum((0,2))
        assert torch.equal(values.cpu(),record['scores'])
        if arm=='uniform':
            local=torch.from_numpy(np.random.default_rng(2140000+seed*10000).choice(128,64,replace=False))
        elif arm=='conflict':local=torch.where(bank['conflict'][ids].bool())[0]
        else:local=torch.argsort(values.cpu(),descending=True,stable=True)[:64]
        assert torch.equal(local,record['selected_local']) and torch.equal(ids[local],record['selected_ids'])
        obs=obs[:,local].contiguous();labels=labels[:,local].contiguous();mask=mask[:,local].contiguous();anchor=anchor[local].contiguous()
        logits,_=model(obs,anchor)
        weights=torch.ones(11,device='cuda');weights[0]=before['config']['wait_weight']
        error=F.cross_entropy(logits.reshape(-1,11),labels.reshape(-1),weight=weights,reduction='none').reshape_as(labels)
        loss=(error*mask).sum()/(weights[labels]*mask).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model_exact=all(torch.equal(v.cpu(),after['model'][k]) for k,v in model.state_dict().items())
        adam_exact=state_digest(optimizer.state_dict())==state_digest(after['optimizer'])
        h_exact=torch.equal(before['hidden'],after['hidden'])
        world_exact=state_digest(before['world'])==state_digest(after['world'])
        assert model_exact and adam_exact and h_exact and world_exact
        reports.append(dict(arm=arm,candidate_and_selection_exact=True,scores_exact=True,model_exact=True,Adam_exact=True,
            live_hidden_unchanged=True,world_unchanged=True,loss=float(loss.detach()),supervised_rows=int(mask.sum()),
            before_model_hash=state_digest(before['model']),after_model_hash=state_digest(after['model'])))
    path=OUT/f'seed_{seed:03d}'/'first_step_audit.json'
    path.write_text(json.dumps(dict(seed=seed,passed=True,reports=reports),indent=2)+'\n',encoding='utf-8')
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    print(json.dumps(audit(args.seed)),flush=True)
