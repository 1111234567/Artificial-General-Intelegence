"""Replay public observations, actual decisions and complete causal endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import numpy as np
import torch
from experiments.wait_content import data as d


def audit(seed):
    torch.set_num_threads(1)
    plan=d.checked();folder=d.OUT/f'seed_{seed:03d}';manifest=d.read(folder/'manifest.json')
    if manifest['manifest_hash']!=d.b.state_digest({k:v for k,v in manifest.items() if k!='manifest_hash'}):
        raise AssertionError('manifest digest')
    for path,sha in manifest['files'].items():
        if d.b.old.sha(path)!=sha:raise AssertionError('bound data bytes changed '+path)
    with np.load(folder/'stream.npz') as archive:
        arrays={key:archive[key] for key in archive.files}
    anchors=np.load(folder/'anchors.npy')
    start=torch.load(folder/'start.pt',weights_only=False,map_location='cpu')
    end=torch.load(folder/'endpoint.pt',weights_only=False,map_location='cpu')
    trainer,model=d.load_source(seed,plan)
    if d.record(trainer.world,trainer.h)!=start['record']:raise AssertionError('source fork differs')
    world=d.b.ContextWorld.from_state_dict(start['world'],device='cpu');h=start['h'].clone()
    shapes=dict(obs=(16384,8,56),elapsed=(16384,),teacher_actions=(16384,8,2),actual_actions=(16384,8,2))
    if any(arrays[k].shape!=v for k,v in shapes.items()):raise AssertionError('wrong stream shape')
    if arrays['obs'].dtype!=np.float64 or arrays['elapsed'].dtype!=np.float64:raise AssertionError('FP64 public stream')
    original_hash=d.b.state_digest(model.state_dict());trace=d.read(folder/'trace.json')
    with torch.no_grad():
        for t in range(16384):
            if t%128==0:
                declaration=plan['schedule'][(t//128)%8]
                world.set_family(declaration['family'])
                world.set_regime(duration_range=declaration['duration_range'],gap_range=declaration['gap_range'],
                    name=f"wait-content/block-{t//128:03d}/{declaration['condition']}/f{declaration['family']}",dt=.25)
                if d.record(world,h)!=trace[t//128]['start']:raise AssertionError('block start mismatch')
            obs=world.observe().double();labels=world.oracle('midpoint')
            if not np.array_equal(obs.numpy(),arrays['obs'][t]):raise AssertionError(f'public observations diverge t{t}')
            if not np.array_equal(labels.numpy(),arrays['teacher_actions'][t]):raise AssertionError(f'teacher label differs t{t}')
            if world.elapsed!=float(arrays['elapsed'][t]):raise AssertionError('dt differs')
            for key in d.PRIVATE:
                if not np.array_equal(getattr(world,key).numpy(),arrays[key][t]):raise AssertionError(f'private score metadata differs {key} t{t}')
            logits,h=model.step(obs,h,world.elapsed);actions=logits.argmax(-1)
            if not np.array_equal(actions.numpy(),arrays['actual_actions'][t]):raise AssertionError(f'actual argmax diverges t{t}')
            world.act(actions)
            if (t+1)%128==0 and d.record(world,h)!=trace[t//128]['end']:raise AssertionError('block endpoint differs')
            if (t+1)%4096==0:print('seed',seed,'replayed',t+1,flush=True)
    if d.record(world,h)!=end['record'] or not torch.equal(h,end['h']):raise AssertionError('end h/world diverge')
    if d.b.state_digest(model.state_dict())!=original_hash:raise AssertionError('collector parameters changed')
    if not np.array_equal(h.numpy(),arrays['h_end']) or not np.array_equal(start['h'].numpy(),arrays['h_start']):raise AssertionError('stream state endpoints differ')
    category_counts=Counter();moves=Counter()
    for block,actor,job,rid,end_t,wait_t,target,move in anchors:
        if actor not in range(4) or job not in range(2) or target not in range(8) or move not in (0,1):raise AssertionError('invalid anchor encoding')
        if end_t//128!=block or wait_t//128!=block:raise AssertionError('anchor crosses gradient block')
        if not arrays['request_id'][end_t,actor,job]==arrays['request_id'][wait_t,actor,job]==rid:raise AssertionError('request mismatch')
        if arrays['teacher_actions'][end_t,actor,job]!=3+target or arrays['teacher_actions'][wait_t,actor,job]!=0:raise AssertionError('anchor labels mismatch')
        if not arrays['processing_wait'][wait_t,actor,job] or arrays['phase'][end_t,actor,job]!=3:raise AssertionError('wrong actual phase')
        if not arrays['now'][wait_t]<arrays['ready_time'][wait_t,actor,job]-1e-6:raise AssertionError('not a true processing WAIT')
        if wait_t>=end_t:raise AssertionError('end precedes wait')
        category='short' if arrays['duration'][end_t,actor,job]<=4+1e-6 else 'long'
        category_counts[category]+=1;moves[int(block)]+=int(move)
    for block in range(128):
        count=int((anchors[:,0]==block).sum())
        if count%2 or moves[block]!=count//2:raise AssertionError('nonmatching per-update auxiliary budget')
    result=dict(seed=seed,protocol_hash=plan['protocol_hash'],manifest_hash=manifest['manifest_hash'],
        replay_ticks=16384,public_observations_exact=True,teacher_labels_exact=True,private_metadata_exact=True,
        actual_actions_exact=True,block_endpoints_exact=True,endpoint_h_exact=True,endpoint_world_exact=True,
        frozen_collector_exact=True,anchors=len(anchors),moved=int(anchors[:,-1].sum()),anchor_duration_counts=dict(category_counts),
        per_update_ce_budget_exact=True,paired_target_multisets_exact=True,training_actor_partition_exact=True)
    d.write(folder/'data_audit.json',result)
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=range(3));args=parser.parse_args()
    if args.seed is not None:audit(args.seed);return
    def one(seed):
        with (d.OUT/f'audit_seed_{seed:03d}.log').open('w',encoding='utf-8') as stream:
            p=subprocess.Popen([sys.executable,'-m','experiments.wait_content.data_audit','--seed',str(seed)],cwd=d.ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,returncode=p.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(one,range(3)))
    d.write(d.OUT/'audit_exit_codes.json',codes)
    if any(c['returncode'] for c in codes):raise RuntimeError(codes)


if __name__=='__main__':main()
