"""Matched, recoverable own-policy stream for the WAIT-content experiment.

Only public observations enter the controller. Private world tensors are saved
separately as scoring metadata; midpoint teachers are labels, never actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import torch
from experiments.action_return import readout as b, refit_policy as rp
from experiments.content_timing import natural
from experiments.lifelong.evaluate import CONDITIONS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/wait-content/data-v1'
PLAN = OUT / 'protocol.json'
BLOCKS = 128
BLOCK_SIZE = 128
SCHEDULE = tuple((family, condition) for condition in ('short', 'long_both') for family in range(4))
ANCHOR_COLUMNS = ('block', 'actor', 'job', 'request_id', 'end_t', 'wait_t', 'target', 'move')
PRIVATE = ('request_id', 'active', 'started', 'version', 'payload', 'raw_payload',
           'current_family', 'duration', 'deadline', 'ready_time', 'issued_time',
           'current_eligible', 'current_regime_id', 'event_kind', 'event_regime_id',
           'event_eligible', 'matured_mask', 'matured_request_id', 'matured_completed',
           'matured_clean_completed')
EPS = 1e-6


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalized(value):
    return json.loads(json.dumps(value))


def record(world, h):
    state = world.state_dict()
    return dict(tick=world.tick, time=world.now, elapsed=world.elapsed,
                world_hash=b.state_digest(state), scheduler_hash=b.state_digest(state['scheduler']),
                event_hash=world.event_hash, h_hash=b.state_digest(h),
                family_exposures=world.family_exposures(), actor_ids=list(world.actor_ids),
                case_ids=list(world.case_ids))


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable data protocol already exists')
    parent = read(rp.PLAN)
    if parent['protocol_hash'] != b.state_digest({k: v for k, v in parent.items() if k != 'protocol_hash'}):
        raise ValueError('parent protocol digest')
    sources = {}
    for seed in range(3):
        endpoint = parent['seeds'][str(seed)]['source']['endpoint']
        model = rp.OUT / f'seed_{seed:03d}/model_own_policy.pt'
        result = read(model.parent / 'result.json')
        if b.old.sha(model) != result['model_sha256']['own_policy'] or b.old.sha(endpoint['path']) != endpoint['sha256']:
            raise ValueError('source bytes changed')
        sources[str(seed)] = dict(endpoint=endpoint, model=str(model), model_sha256=b.old.sha(model),
            parent_result=str(model.parent / 'result.json'), parent_result_sha256=b.old.sha(model.parent / 'result.json'))
    source_files = [Path(__file__).resolve(), Path(b.__file__).resolve(), Path(natural.__file__).resolve(),
                    ROOT/'experiments/lifelong/world.py', ROOT/'experiments/continuity/world.py',
                    ROOT/'experiments/undirected/model.py', ROOT/'experiments/action_return/finish_readout.py']
    plan = dict(format='wait-content-shared-own-policy-data-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before collecting new streams or fitting either arm. Source own_policy outcomes were already known. One identical data file is shared by both arms.',
        source_files={str(p): b.old.sha(p) for p in source_files}, sources=sources,
        parent_protocol=dict(path=str(rp.PLAN), sha256=b.old.sha(rp.PLAN)),
        seeds=[0, 1, 2], blocks=BLOCKS, ticks_per_block=BLOCK_SIZE, total_ticks=BLOCKS*BLOCK_SIZE,
        start='Actual historical147456 complete world and h; no initial observation, action, receipt or hidden reset. Full8 actors remain in the world.',
        controller='Saved own_policy eleven-action argmax, frozen throughout; execute every actual action and preserve resulting public receipts. Midpoint teacher is read solely as a supervised label.',
        schedule=[dict(block_mod_8=i, family=f, condition=c, **CONDITIONS[c]) for i, (f,c) in enumerate(SCHEDULE)],
        switching='At each128-tick boundary change future-NEW family and short/long_both regime only. Pending contracts, scheduled consequences, scheduler RNG, receipts, clocks, world and h persist. All8 combinations occur16 times. Current requests retain their actual issued regime and duration in private metadata.',
        partition=dict(fit_lanes=[0,1,2,3], heldout_lanes=[4,5,6,7],
            full_observations='Lanes4..7 are retained only for common public-history evaluation warm starts; excluded from every fitting target and anchor.',
            evaluation='Future requests after the complete stream; report inherited/carryover separately and reject any primary request ID observed during fitting.'),
        stream_schema=dict(obs='float64[T,8,56] raw public observation only', elapsed='float64[T] model dt',
            teacher_actions='int64[T,8,2] midpoint oracle0..10, label only', actual_actions='int64[T,8,2] executed own-policy argmax',
            h_start='float64[8,80] inherited historical147456 state', h_end='float64[8,80] collector endpoint',
            private='Remaining arrays are scoring/anchor metadata only and cannot enter model input.'),
        anchors=dict(file='anchors.npy', dtype='int64[N,8]', columns=list(ANCHOR_COLUMNS),
            end='Earliest midpoint teacher FINISH>=3 in this block for an eligible same actor/job/request.',
            wait='Actual active+started processing observation with midpoint teacher WAIT0, strictly before ready time; nearest processing midpoint within this block, ties earlier. Must be after cue propagation (at least one tick after request issue).',
            matching='Require same request ID, actor, job, target and update block; private labels only select supervised positions and never change trajectory actions.',
            eligibility='Only lanes0..3 and requests first observed strictly after source fork; every complete same-block pair is retained except last identity when block count odd. No outcome or score selection.',
            assignment='Sort pairs by actual duration category, actual family, eight-way target, actor, job, request ID. Alternate move0/1 with block parity; every nonempty block has exactly half moved. End-only arm uses all end_t; mixed uses wait_t iff move1. Identical eight-way label multiset and exact CE count each update.',
            supervision='Only eight FINISH logits receive auxiliary CE. Original action teacher remains WAIT on processing rows, identical in both arms. Actual successful FINISH is never required for eligibility.'),
        persistence='start.pt and endpoint.pt contain world,h,collector_state,record. Every block is atomically saved with full causal endpoint and per-block hashes for exact resume. stream.npz, anchors.npy, private metadata and request lists are hash-bound by manifest.json.',
        limitations=['Externally supervised fixed own-policy data; this is not newly on-policy data under each learning arm.',
                    'Backbone historically trained on all8 actors; only this new fitting holds lanes4..7 out.',
                    'Same-block pairing selects requests with both actual processing and teacher-FINISH opportunities; paired supervision coverage is reported.'])
    plan = normalized(plan)
    plan['protocol_hash'] = b.state_digest(plan)
    write(PLAN, plan)
    return plan


def checked():
    plan = read(PLAN)
    if plan['protocol_hash'] != b.state_digest({k:v for k,v in plan.items() if k != 'protocol_hash'}):
        raise ValueError('data protocol digest')
    for path, sha in plan['source_files'].items():
        if b.old.sha(path) != sha:
            raise ValueError('bound source changed: '+path)
    if b.old.sha(plan['parent_protocol']['path']) != plan['parent_protocol']['sha256']:
        raise ValueError('parent protocol changed')
    for src in plan['sources'].values():
        if b.old.sha(src['model']) != src['model_sha256'] or b.old.sha(src['endpoint']['path']) != src['endpoint']['sha256']:
            raise ValueError('source model/endpoint changed')
    return plan


def load_source(seed, plan=None):
    plan = checked() if plan is None else plan
    src = plan['sources'][str(seed)]
    trainer = b.native.NativeTrainer.from_checkpoint(src['endpoint']['path'], device='cpu')
    if b.state_digest(trainer.semantic_state()) != src['endpoint']['semantic_hash']:
        raise ValueError('historical complete endpoint semantics differ')
    if trainer.world.tick != 147455 or trainer.h.shape != (8,80):
        raise ValueError('wrong historical147456 causal boundary')
    return trainer, natural.load_model(seed, trainer).eval()


def endpoint(world, h, model):
    return dict(world=world.state_dict(), h=h.detach().clone(), collector_state=model.state_dict(),
                record=record(world, h))


def make_block(world, h, model, block):
    family, condition = SCHEDULE[block % len(SCHEDULE)]
    world.set_family(family)
    world.set_regime(**CONDITIONS[condition], name=f'wait-content/block-{block:03d}/{condition}/f{family}', dt=.25)
    start = record(world, h)
    fields = {k: [] for k in ('obs', 'elapsed', 'teacher_actions', 'actual_actions', 'now', 'tick', *PRIVATE)}
    with torch.no_grad():
        for _ in range(BLOCK_SIZE):
            obs = world.observe().double()
            labels = world.oracle('midpoint').clone()
            logits, h = model.step(obs, h, float(world.elapsed))
            actions = logits.argmax(-1)
            fields['obs'].append(obs.clone()); fields['elapsed'].append(world.elapsed)
            fields['teacher_actions'].append(labels); fields['actual_actions'].append(actions.clone())
            fields['now'].append(world.now); fields['tick'].append(world.tick)
            for k in PRIVATE:
                fields[k].append(getattr(world, k).clone())
            world.act(actions)
    arrays = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else
              torch.tensor(v,dtype=torch.int64 if k=='tick' else torch.float64) for k,v in fields.items()}
    feasible=arrays['now'][:,None,None]+arrays['duration']<=arrays['deadline']+EPS
    ready=arrays['now'][:,None,None]>=arrays['ready_time']-EPS
    arrays['phase']=torch.where(~arrays['active'],-1,torch.where(~arrays['started'],
        torch.where(feasible,0,1),torch.where(ready,3,2))).long()
    arrays['processing_wait']=(arrays['phase']==2)&(arrays['teacher_actions']==0)
    return h, dict(block=block, scheduled_family=family, scheduled_condition=condition,
                   start_record=start, arrays=arrays, endpoint=endpoint(world, h, model))


def make_anchors(arrays, source_tick):
    pairs=[]; metadata=[]; coverage=[]
    tmax = len(arrays['obs'])
    for block in range(BLOCKS):
        first, last = block*BLOCK_SIZE, min((block+1)*BLOCK_SIZE, tmax)
        groups = defaultdict(lambda: dict(wait=[], end=[]))
        for t in range(first,last):
            for actor in range(4):
                for job in range(2):
                    if not bool(arrays['current_eligible'][t,actor,job]): continue
                    rid=int(arrays['request_id'][t,actor,job])
                    if rid < 0 or float(arrays['issued_time'][t,actor,job]) <= source_tick*.25+EPS: continue
                    key=(actor,job,rid)
                    target=int(arrays['teacher_actions'][t,actor,job])
                    if target>=3: groups[key]['end'].append(t)
                    processing=bool(arrays['active'][t,actor,job] and arrays['started'][t,actor,job])
                    if processing and target==0 and float(arrays['now'][t])<float(arrays['ready_time'][t,actor,job])-EPS and float(arrays['now'][t])>float(arrays['issued_time'][t,actor,job])+EPS:
                        groups[key]['wait'].append(t)
        eligible=[]
        for (actor,job,rid),g in groups.items():
            if not g['end'] or not g['wait']: continue
            end_t=g['end'][0]
            center=float(arrays['ready_time'][end_t,actor,job])-.5*float(arrays['duration'][end_t,actor,job])
            wait_t=min(g['wait'],key=lambda t:(abs(float(arrays['now'][t])-center),t))
            target=int(arrays['teacher_actions'][end_t,actor,job])-3
            wait_target=4*int(arrays['version'][wait_t,actor,job])+int(arrays['payload'][wait_t,actor,job])
            if target != wait_target: raise AssertionError('same request target changed')
            duration=float(arrays['duration'][end_t,actor,job]);family=int(arrays['current_family'][end_t,actor,job])
            category='short' if duration<=4+EPS else 'long' if duration>=8-EPS else 'inherited_other'
            eligible.append(dict(actor=actor,job=job,request_id=rid,end_t=end_t,wait_t=wait_t,target=target,
                duration=duration,category=category,family=family,regime_id=int(arrays['current_regime_id'][end_t,actor,job]),
                issued_time=float(arrays['issued_time'][end_t,actor,job]),end_time=float(arrays['now'][end_t]),
                wait_time=float(arrays['now'][wait_t]),processing_fraction=(float(arrays['now'][wait_t])-float(arrays['ready_time'][wait_t,actor,job])+duration)/duration))
        eligible.sort(key=lambda v:(v['category'],v['family'],v['target'],v['actor'],v['job'],v['request_id']))
        discarded=[]
        if len(eligible)%2: discarded=[eligible.pop()]
        for k,v in enumerate(eligible):
            move=(k+block)%2
            pairs.append([block,*[v[name] for name in ANCHOR_COLUMNS[1:-1]],move])
            metadata.append(dict(block=block,move=move,**v))
        coverage.append(dict(block=block,paired_retained=len(eligible),moved=len(eligible)//2,
            discarded_odd=len(discarded),discarded=discarded,
            processing_requests=sum(bool(g['wait']) for g in groups.values()),
            teacher_finish_requests=sum(bool(g['end']) for g in groups.values()),
            categories=dict(Counter(v['category'] for v in eligible)),
            moved_categories=dict(Counter(v['category'] for k,v in enumerate(eligible) if (k+block)%2))))
    return np.asarray(pairs,dtype=np.int64).reshape(-1,8),metadata,coverage


def finish(seed, plan, folder):
    blocks=[torch.load(folder/'blocks'/f'block_{k:03d}.pt',map_location='cpu',weights_only=False) for k in range(BLOCKS)]
    arrays={k: torch.cat([v['arrays'][k] for v in blocks]) for k in blocks[0]['arrays']}
    start=torch.load(folder/'start.pt',map_location='cpu',weights_only=False)
    end=blocks[-1]['endpoint']; torch.save(end,folder/'endpoint.pt')
    anchors,metadata,coverage=make_anchors(arrays,start['record']['tick'])
    np.save(folder/'anchors.npy',anchors)
    with (folder/'anchors_metadata.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metadata[0]) if metadata else list(ANCHOR_COLUMNS))
        writer.writeheader();writer.writerows(metadata)
    write(folder/'anchor_coverage.json',coverage)
    public={k:arrays[k].numpy() for k in ('obs','elapsed','teacher_actions','actual_actions')}
    private={k:arrays[k].numpy() for k in arrays if k not in public}
    np.savez_compressed(folder/'stream.npz',**public,h_start=start['h'].numpy(),h_end=end['h'].numpy(),**private)
    request_sets={}
    cases=start['record']['case_ids']
    for name,lanes in [('fit_requests',range(4)),('observed_requests',range(8))]:
        ids=sorted({(cases[i],j,int(r)) for i in lanes for j in range(2) for r in arrays['request_id'][:,i,j].unique().tolist() if r>=0})
        request_sets[name]=[list(v) for v in ids]
    write(folder/'requests.json',request_sets)
    trace=[dict(block=v['block'],scheduled_family=v['scheduled_family'],scheduled_condition=v['scheduled_condition'],
                start=v['start_record'],end=v['endpoint']['record']) for v in blocks]
    write(folder/'trace.json',trace)
    hashes={str(folder/name):b.old.sha(folder/name) for name in ('stream.npz','anchors.npy','anchors_metadata.csv','anchor_coverage.json','requests.json','trace.json','start.pt','endpoint.pt')}
    counts=Counter((v['category'],str(v['family']),str(v['move'])) for v in metadata)
    manifest=dict(format='wait-content-data-manifest-v1',seed=seed,protocol_hash=plan['protocol_hash'],
        files=hashes,start=start['record'],end=end['record'],ticks=len(arrays['obs']),blocks=BLOCKS,
        fit_lanes=[0,1,2,3],heldout_lanes=[4,5,6,7],anchors=len(anchors),moved=int(anchors[:,-1].sum()),
        anchor_counts={':'.join(k):v for k,v in sorted(counts.items())},
        zero_anchor_blocks=[v['block'] for v in coverage if not v['paired_retained']],
        teacher_action_counts=torch.bincount(arrays['teacher_actions'][:,:4].flatten(),minlength=11).tolist(),
        actual_action_counts=torch.bincount(arrays['actual_actions'][:,:4].flatten(),minlength=11).tolist(),
        fit_request_count=len(request_sets['fit_requests']),observed_request_count=len(request_sets['observed_requests']),
        exact_block_continuity=all(blocks[k-1]['endpoint']['record']['h_hash']==blocks[k]['start_record']['h_hash'] for k in range(1,BLOCKS)))
    manifest=normalized(manifest);manifest['manifest_hash']=b.state_digest(manifest)
    write(folder/'manifest.json',manifest)
    return manifest


def collect(seed):
    torch.set_num_threads(1);plan=checked();folder=OUT/f'seed_{seed:03d}'
    folder.mkdir(parents=True,exist_ok=True);(folder/'blocks').mkdir(exist_ok=True)
    trainer,model=load_source(seed,plan)
    world=b.ContextWorld.from_state_dict(trainer.world.state_dict(),device='cpu');h=trainer.h.clone()
    if not (folder/'start.pt').exists(): torch.save(endpoint(world,h,model),folder/'start.pt')
    else:
        saved=torch.load(folder/'start.pt',map_location='cpu',weights_only=False)
        if saved['record']!=record(world,h) or b.state_digest(saved['collector_state'])!=b.state_digest(model.state_dict()):
            raise ValueError('resume start differs')
    files=sorted((folder/'blocks').glob('block_*.pt'))
    if files:
        for k,path in enumerate(files):
            if path.name!=f'block_{k:03d}.pt': raise ValueError('noncontiguous resume blocks')
        previous=torch.load(files[-1],map_location='cpu',weights_only=False)
        world=b.ContextWorld.from_state_dict(previous['endpoint']['world'],device='cpu');h=previous['endpoint']['h'].clone()
        if record(world,h)!=previous['endpoint']['record']: raise ValueError('resume endpoint differs')
    began=time.perf_counter()
    for block in range(len(files),BLOCKS):
        h,value=make_block(world,h,model,block)
        target=folder/'blocks'/f'block_{block:03d}.pt';temp=target.with_suffix('.tmp')
        torch.save(value,temp);temp.replace(target)
        if (block+1)%16==0: print('seed',seed,'blocks',block+1,'tick',world.tick,'seconds',round(time.perf_counter()-began,2),flush=True)
    result=finish(seed,plan,folder);checked()
    print('DONE seed',seed,'anchors',result['anchors'],'moved',result['moved'],flush=True)


def launch():
    checked()
    def one(seed):
        with (OUT/f'seed_{seed:03d}.log').open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-m','experiments.wait_content.data','collect','--seed',str(seed)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            return dict(seed=seed,pid=child.pid,returncode=child.wait())
    with ThreadPoolExecutor(max_workers=3) as pool: values=list(pool.map(one,range(3)))
    write(OUT/'exit_codes.json',values)
    if any(v['returncode'] for v in values): raise RuntimeError(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','collect','launch'));parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='collect': collect(args.seed)
    else: launch()


if __name__=='__main__': main()
