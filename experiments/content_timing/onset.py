"""Fixed one-tick propagation supplement; no original controller is trained."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from experiments.content_timing import matched_flow as m, probe

OUT=m.b.ROOT/'data/content-timing/onset-v1'
PLAN=OUT/'protocol.json'
FORMAT='matched-first-propagation-tick-v1'


def freeze():
    if PLAN.exists():raise FileExistsError('immutable onset protocol exists')
    parent=m.checked();pins={str(Path(__file__).resolve()):m.sha(__file__),str(Path(probe.__file__).resolve()):m.sha(probe.__file__)}
    for seed in range(3):
        folder=m.OUT/f'seed_{seed:03d}'
        for name in ('records.pt','result.json',*[f'background_f{f}.pt' for f in range(4)]):
            path=folder/name;pins[str(path)]=m.sha(path)
    value=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Supplement fixed after seeing the matched-v1 cue invariance and downstream summaries; no onset states or onset probe results were measured before this freeze.',
        parent_protocol=dict(path=str(m.PLAN),sha256=m.sha(m.PLAN),protocol_hash=parent['protocol_hash']),pins=pins,
        population='Exactly the existing576 real anchors: all192 per seed, duration8, four raw payloads; no anchor selection or replacement.',
        replay='Use each saved actual full-world post-observe anchor and continuous pre-cue h. Reconstruct the original eight-treatment batch (duration2 and8 x four raw) in original order so the duration8 cue reproduces exactly. Only duration8 onset data are saved. Apply the same original locked target BEGIN and original non-target background actions, observe exactly one0.25s tick, step the unchanged original controller, measure before taking the next action.',
        matching='Require bitwise equality of every duration8 cue state24 and original11 scores versus saved matched-v1 rows. Require exact public-observation equality across all8 variants at onset and source equality at every non-target slot. BEGIN must actually be accepted.',
        primary='For each seed count how many of192 anchor groups have identical four-raw cue non_input24 states but a nonzero FP64 difference at exactly+0.25. Report exact inequality plus L2 and maximum absolute scales; exactnonzero is propagation detection, not necessarily robust decodability.',
        probe=dict(algorithm='Import unchanged probe.fit_scores: train-fold mean/std, ridge=.001*n on standardized slopes, unpenalized intercept, four-way one-hot least squares.',
            folds='Within each seed and job, train actors4,5 test6,7 and reverse. All four variants of every anchor remain together; pool all4 families as in the original probe. Fit and test only+1tick, no hyperparameter or stage search.',
            labels=['raw','mapped'],null='Identical within-anchor mean test state; balanced four variants force25%null. Source controller and backbone weights are never trained.'),
        outputs='seed_000/records.pt similarly001,002: rows plus x24, scores11, finish_scores8, cue_x24, cue_scores11, h80, model_obs56; all floating tensors FP64. result.json stores exact cue checks, per-anchor propagation and original-fold probe confusion matrices.',
        limits=['This locates first propagation at the requested observation grid, not continuous-time latency within the interval.',
            'A trainable external offline probe does not establish that the inherited controller uses the feature for correct autonomous actions.',
            'Same locked-BEGIN/WAIT controlled-fork interpretation and version-receipt limitation as matched-v1. No additional duration or behavior experiment.'])
    value['protocol_hash']=m.b.state_digest(value);OUT.mkdir(parents=True,exist_ok=True);m.write(PLAN,value)


def checked():
    value=m.read(PLAN);m.checked()
    if value['protocol_hash']!=m.b.state_digest({k:v for k,v in value.items() if k!='protocol_hash'}):raise ValueError('onset protocol digest')
    if m.sha(value['parent_protocol']['path'])!=value['parent_protocol']['sha256']:raise ValueError('parent protocol changed')
    for path,sha in value['pins'].items():
        if m.sha(path)!=sha:raise ValueError('pinned onset input/source changed '+path)
    return value


def probes(anchors,x24):
    states=x24.reshape(len(anchors),4,24).numpy();jobs=np.array([a['job'] for a in anchors]);lanes=np.array([a['actor_lane'] for a in anchors])
    family=np.array([a['family'] for a in anchors]);raw=np.broadcast_to(np.arange(4),(len(anchors),4));mapped=np.array(m.FAMILY_MAPPINGS)[family]
    folds=[]
    for job in range(2):
        for heldout in ((4,5),(6,7)):
            test=(jobs==job)&np.isin(lanes,heldout);train=(jobs==job)&~test
            assert set(family[test])==set(family[train])==set(range(4))
            x=states[train].reshape(-1,24);xt=states[test].reshape(-1,24);centers=states[test].mean(1)
            for label,target in (('raw',raw),('mapped',mapped)):
                scores=probe.fit_scores(x,target[train].ravel(),np.concatenate((xt,centers)))
                y=target[test].ravel();pred=scores[:len(xt)].argmax(-1);null=np.repeat(scores[len(xt):].argmax(-1),4)
                matrix=np.bincount(4*y+pred,minlength=16).reshape(4,4);null_correct=int((null==y).sum())
                assert null_correct*4==len(y)
                folds.append(dict(job=job,test_lanes=list(heldout),train_anchors=int(train.sum()),test_anchors=int(test.sum()),
                    label=label,train_stage='onset1',test_stage='onset1',duration=8.,correct=int(matrix.trace()),total=len(y),
                    null_correct=null_correct,confusion_true_by_pred=matrix.tolist()))
    pooled=[]
    for label in ('raw','mapped'):
        rows=[r for r in folds if r['label']==label];correct=sum(r['correct'] for r in rows);total=sum(r['total'] for r in rows)
        pooled.append(dict(label=label,correct=correct,total=total,accuracy=correct/total,null_correct=sum(r['null_correct'] for r in rows)))
    return dict(folds=folds,pooled=pooled)


def run_seed(seed,plan):
    folder=OUT/f'seed_{seed:03d}';folder.mkdir(exist_ok=False)
    trainer,model=m.load_model(seed,m.checked());model_hash=m.b.state_digest(model.state_dict());start=time.perf_counter()
    old=torch.load(m.OUT/f'seed_{seed:03d}'/'records.pt',weights_only=False,map_location='cpu')
    lookup={(r['anchor_id'],r['raw_payload']):n for n,r in enumerate(old['rows']) if r['duration']==8. and r['stage']=='cue'}
    output={k:[] for k in ('rows','x24','scores11','finish_scores8','cue_x24','cue_scores11','h80','model_obs56')}
    anchors=[];propagation=[]
    with torch.no_grad():
        for family in range(4):
            background=torch.load(m.OUT/f'seed_{seed:03d}'/f'background_f{family}.pt',weights_only=False,map_location='cpu')
            for anchor in background['anchors']:
                i,j=anchor['actor_lane'],anchor['job'];variants=[];worlds=[];obs=[]
                for duration in m.DURATIONS:
                    for raw in range(4):
                        w,o,fingerprint=m.change_command(anchor,raw,duration);worlds.append(w);obs.append(o)
                        variants.append(dict(raw=raw,duration=duration,fingerprint=fingerprint))
                logits,h=model.step(torch.stack(obs).flatten(0,1),anchor['h_before'].repeat(8,1),worlds[0].elapsed)
                cue_logits=logits.reshape(8,8,2,11);cue_h=h.reshape(8,8,80);cue_x=cue_h[4:,i].index_select(-1,model.begin_head.indices).clone()
                for raw in range(4):
                    old_n=lookup[(anchor['anchor_id'],raw)]
                    if not torch.equal(cue_x[raw],old['x24'][old_n]) or not torch.equal(cue_logits[4+raw,i,j],old['scores11'][old_n]):
                        raise ValueError('cue replay not bitwise aligned')
                if not torch.equal(cue_x,cue_x[:1].expand_as(cue_x)):raise ValueError('cue states not invariant')
                locked=background['actions'][anchor['source_index']].clone();locked[i,j]=1+anchor['version']
                for world in worlds:
                    world.act(locked)
                    if int(world.last_result[i,j])!=1:raise ValueError('target BEGIN not accepted')
                obs=torch.stack([w.observe().double() for w in worlds])
                if not torch.equal(obs,obs[:1].expand_as(obs)):raise ValueError('onset receipts differ')
                mask=torch.ones_like(obs[0],dtype=torch.bool);mask[i,j*28:(j+1)*28]=False
                source=background['observations'][anchor['source_index']+1]
                if not all(torch.equal(o[mask],source[mask]) for o in obs):raise ValueError('non-target source diverged')
                logits,h=model.step(obs.flatten(0,1),h,worlds[0].elapsed)
                logits=logits.reshape(8,8,2,11);h=h.reshape(8,8,80);x=h[4:,i].index_select(-1,model.begin_head.indices).clone()
                differences=x[:,None,:]-x[None,:,:];distance=differences.square().sum(-1).sqrt()
                propagation.append(dict(anchor_id=anchor['anchor_id'],family=family,actor_lane=i,job=j,version=anchor['version'],
                    cue_exactly_invariant=True,onset_nonzero=bool(differences.ne(0).any()),
                    onset_all_four_distinct=len({tuple(v.tolist()) for v in x})==4,
                    max_pair_l2=float(distance.max()),min_pair_l2=float(distance[torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)].min()),
                    max_abs_difference=float(differences.abs().max()),cue_bitwise_alignment=True))
                anchors.append(anchor)
                for raw in range(4):
                    world=worlds[4+raw];old_n=lookup[(anchor['anchor_id'],raw)];scores=logits[4+raw,i,j].clone()
                    row=dict(old['rows'][old_n]);row.update(row_id=len(output['rows']),matched_cue_row_id=old_n,stage='onset1',age=.25,
                        tick=world.tick,now=world.now,active=bool(world.active[i,j]),started=bool(world.started[i,j]),ready_time=float(world.ready_time[i,j]),
                        deadline=float(world.deadline[i,j]),model_action=int(scores.argmax()),conditional_choice8=int(scores[3:].argmax()),
                        teacher_action=int(world.oracle('midpoint')[i,j]),legal=m.is_legal(world,i,j),
                        previous_observation_time=anchor['cue_time'],previous_observation_legal=False)
                    choice=row['conditional_choice8'];row.update(content_correct=choice==row['target8'],version_correct=choice//4==row['version'],
                        payload_correct=choice%4==row['mapped_payload'],teacher_finish_expected=False,
                        finish_vs_nonfinish_margin=float(scores[3:].max()-scores[:3].max()))
                    output['rows'].append(row);output['x24'].append(x[raw].clone());output['scores11'].append(scores)
                    output['finish_scores8'].append(scores[3:].clone());output['cue_x24'].append(cue_x[raw].clone())
                    output['cue_scores11'].append(cue_logits[4+raw,i,j].clone());output['h80'].append(h[4+raw,i].clone());output['model_obs56'].append(obs[4+raw,i].clone())
    for key in output:
        if key!='rows':output[key]=torch.stack(output[key])
    output.update(format=FORMAT,protocol_hash=plan['protocol_hash'],non_input_indices=model.begin_head.indices.tolist())
    torch.save(output,folder/'records.pt')
    if m.b.state_digest(model.state_dict())!=model_hash:raise ValueError('original controller changed')
    result=dict(format=FORMAT,seed=seed,protocol_hash=plan['protocol_hash'],anchors=len(anchors),rows=len(output['rows']),
        controller_unchanged=True,cue_bitwise_alignment_all=True,groups_cue_invariant=sum(p['cue_exactly_invariant'] for p in propagation),
        groups_onset_nonzero=sum(p['onset_nonzero'] for p in propagation),groups_onset_four_distinct=sum(p['onset_all_four_distinct'] for p in propagation),
        propagation=propagation,probe=probes(anchors,output['x24']),raw_sha256=m.sha(folder/'records.pt'),seconds=time.perf_counter()-start)
    result['result_hash']=m.b.state_digest(result);m.write(folder/'result.json',result)
    print(seed,result['groups_onset_nonzero'],result['probe']['pooled'],flush=True)
    return result


def run():
    torch.set_num_threads(1);plan=checked();results=[run_seed(seed,plan) for seed in range(3)];checked()
    summary=dict(format=FORMAT,protocol_hash=plan['protocol_hash'],seeds=[{k:r[k] for k in ('seed','anchors','rows','groups_cue_invariant','groups_onset_nonzero','groups_onset_four_distinct','probe','raw_sha256','seconds')} for r in results])
    m.write(OUT/'summary.json',summary)
    lines=['# First propagation tick supplement','',
        'All original576 anchors, duration8, all four raw payloads. Original controller weights remain unchanged. Cue state24 and scores11 reproduced bitwise for every row before taking the same locked BEGIN. Exactly one0.25s tick is then observed and stepped.','',
        '|seed|cue invariant groups|nonzero at+0.25|four distinct at+0.25|raw probe correct/768|mapped probe correct/768|',
        '|---:|---:|---:|---:|---:|---:|']
    for r in results:
        a,b=r['probe']['pooled'];lines.append(f"|{r['seed']}|{r['groups_cue_invariant']}/192|{r['groups_onset_nonzero']}/192|{r['groups_onset_four_distinct']}/192|{a['correct']}/{a['total']}|{b['correct']}/{b['total']}|")
    lines+=['','Probe uses unchanged probe.fit_scores and within-job actor4,5 versus6,7 folds in both directions; all anchor variants remain together. Fixed within-anchor-mean null is192/768=25% for every seed and label. No hyperparameter selection or original-controller training.','',
        'Exact nonzero detects propagation at the observation grid; scale is saved per anchor. Offline decodability is separate from original readout use and autonomous behavior.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run'));args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:run()
