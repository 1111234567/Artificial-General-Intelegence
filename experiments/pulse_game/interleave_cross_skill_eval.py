"""Read six independent private memories from the same graph after seven real contacts."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_capacity_report as ci, interleave_cross_skill as fit
from . import interleave_geometry_eval as geometry, interleave_model_cards as cards, interleave_train as joint
from . import interleave_triangle_train as tri, local_trace_graph as net, revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE


ORDERS=("triangle_first","context_first","interspersed")
MODES=("active","sham","uniform")
METRICS=("all_six","triangle_three","context_both_mark","context_both","mark","number_correct")
LAWS=np.array(list(product((-1,1),repeat=6)),dtype=np.int8)
N_WORLD=96


def make(seed, worlds):
    rng=np.random.default_rng([913711,seed])
    directions=tri.rotation_keys(rng,worlds,-.5)
    scent=rng.normal(size=(worlds,3))
    scent=(1.35*scent/np.linalg.norm(scent,axis=1,keepdims=True)).astype(np.float32)
    scenes=joint.CUE_BANK[rng.permuted(np.tile(np.arange(3),(worlds,1)),axis=1)[:,:2]]
    # Seven support visits: three triangle items, context A/B, and both feeder demos.
    # Six terminal probes ask the first five bindings and a choice between feeder marks.
    objects=np.concatenate((directions,np.repeat(scent[:,None],2,axis=1),
                            np.broadcast_to(joint.MARK_AXIS,(worlds,2,3))),axis=1)
    cues=np.zeros_like(objects);cues[:,3:5]=scenes
    support_orientation=rng.choice((-1,1),(worlds,7)).astype(np.float32)
    support_orientation[:,5]=1;support_orientation[:,6]=-1
    query_orientation=rng.choice((-1,1),(worlds,6)).astype(np.float32)
    s_right=cues+objects*support_orientation[...,None]
    s_left=cues-objects*support_orientation[...,None]
    s_left[:,5:]=s_right[:,5:]
    q_right=cues[:,:6]+objects[:,:6]*query_orientation[...,None]
    q_left=cues[:,:6]-objects[:,:6]*query_orientation[...,None]
    triangles=rng.permuted(np.tile(np.arange(3),(worlds,1)),axis=1)
    contexts=rng.permuted(np.tile(np.arange(3,7),(worlds,1)),axis=1)
    orders=np.stack((np.concatenate((triangles,contexts),axis=1),
                     np.concatenate((contexts,triangles),axis=1),
                     np.column_stack((contexts[:,0],triangles[:,0],contexts[:,1],triangles[:,1],
                                      contexts[:,2],triangles[:,2],contexts[:,3]))),axis=1)
    return dict(left=s_left,right=s_right,study_orientation=support_orientation,
                query_left=q_left,query_right=q_right,query_orientation=query_orientation,
                draws=rng.random((worlds,7,5),dtype=np.float32),
                waits=rng.integers(0,9,size=(worlds,7)),
                q_draws=rng.random((worlds,6),dtype=np.float32),
                q_wait=rng.integers(0,9,size=worlds),orders=orders.astype(np.int64))


@torch.no_grad()
def simulate(g,r,tape,order,mode):
    n=len(tape["left"])
    if mode=="uniform":
        public=np.arange(n)[:,None]
        objects=tape["orders"][:,order,:]
        motor_side=np.where(tape["draws"][public,objects,0]<.5,1,-1).astype(np.int8)
        factor=(LAWS[:,np.minimum(objects,5)].transpose(1,0,2)
                *tape["study_orientation"][public,objects][:,None,:])
        earned=np.where(objects[:,None,:]>=5,factor,factor*motor_side[:,None,:])
        contact=np.stack((np.broadcast_to(motor_side[:,None,:],factor.shape),earned),axis=-1)
        selected=np.where(tape["q_draws"]<.5,1,-1).astype(np.int8)
        return np.broadcast_to(selected[:,None,:],(n,len(LAWS),6)),contact.astype(np.int8)
    lanes=n*len(LAWS)
    world=torch.arange(n).repeat_interleave(len(LAWS))
    laws=torch.from_numpy(np.tile(LAWS,(n,1))).float()
    rows=torch.arange(lanes)
    t={k:torch.from_numpy(v) for k,v in tape.items()}
    rule=r.expand(lanes)
    state=net.start(g,lanes)
    events=[]
    for i in range(7):
        obj=t["orders"][world,order,i]
        if i:state=net.wait(rule,state,t["waits"][world,obj],max_ticks=8)
        left,right=t["left"][world,obj],t["right"][world,obj]
        u=t["draws"][world,obj]
        a=net.act(rule,state,left,right,u[:,0])
        side=a.side
        value=laws[rows,obj.clamp_max(5)]*t["study_orientation"][world,obj]
        taste=torch.where(obj>=5,value,side*value)
        if i==0 and not torch.all(side.reshape(n,64)==side.reshape(n,64)[:,:1]):
            raise AssertionError("First action leaked a private mapping")
        e=net.observe(rule,a,left,right,taste)
        state=net.commit(rule,e,u[:,1:],intervention=mode).state
        events.append(torch.stack((side,taste),1).to(torch.int8))
    state=net.wait(rule,state,t["q_wait"][world],max_ticks=8)
    if mode=="sham":
        for key in tri.revisit.capacity.GRAPH_FIELDS:
            value=getattr(state.graph,key)
            if not torch.equal(value,getattr(g,key).expand_as(value)):
                raise AssertionError("The no-write control changed the graph")
    selected=[]
    for j in range(6):
        u=t["q_draws"][world,j]
        side=motor.act(state,t["query_left"][world,j],t["query_right"][world,j],u).side
        selected.append(side.to(torch.int8))
    return torch.stack(selected,1).numpy().reshape(n,64,6),torch.stack(events,1).numpy().reshape(n,64,7,2)


def metrics(side,truth,condition):
    ok=side==truth
    mask=np.ones(64,dtype=bool)
    if condition=="context_conflict":mask=LAWS[:,3]!=LAWS[:,4]
    if condition=="hard_and_context_conflict":
        mask=(LAWS[:,3]!=LAWS[:,4]) & np.all(LAWS[:,:3]==LAWS[:,:1],axis=1)
    ok=ok[:,mask]
    return np.stack((ok.all(2),ok[:,:,:3].all(2),ok[:,:,3:].all(2),
                     ok[:,:,3:5].all(2),ok[:,:,5],ok.sum(2)),axis=-1).mean(1)


def collect(raw,truth,conditions,rows):
    np.testing.assert_array_equal(raw["truth"],truth)
    for mode in MODES:
        for cond in conditions:
            scored=[]
            for order in ORDERS:
                v=metrics(raw[f"{order}/{mode}/side"],truth,cond)
                rows.setdefault(f"{mode}/{cond}/{order}",[]).append(v)
                scored.append(v)
            rows.setdefault(f"{mode}/{cond}/mean_order",[]).append(np.mean(scored,axis=0))


def run(cohort,step):
    torch.set_num_threads(1)
    original=cohort=="original"
    births=list(range(731,737)) if original else list(range(811,817))
    label="original" if original else "confirmation"
    seed=622819 if original else 622829
    base=make(seed,N_WORLD)
    evaluation_digest=sha256(Path(__file__).read_bytes()).hexdigest()
    truth=LAWS[None,:,:]*base["query_orientation"][:,None,:]
    if len(np.unique(LAWS,axis=0))!=64:raise ValueError("Private maps are not fully crossed")
    folder=DATA/"cross-skill-eval"/cohort/str(step);folder.mkdir(exist_ok=True,parents=True)
    card_spec=json.loads((EVIDENCE/f"model-cards-{label}.json").read_text())
    cells={};card_rows={};hashes={}
    arms=fit.SCHEDULES if step else ("parent",)
    conditions=("all_maps","context_conflict","hard_and_context_conflict")
    for arm in arms:
        results={};ordinary=[]
        for b in births:
            if step:path,tag,stage=fit.DEST/cohort,arm,step
            else:
                path,tag=geometry.sources(cohort,"hard",64000)
                stage=64000
            checkpoint=tri.old_fit.checkpoint_path(path,"odor",tag,b,stage)
            checkpoint_digest=sha256(checkpoint.read_bytes()).hexdigest()
            hashes[checkpoint.relative_to(DATA).as_posix()]=checkpoint_digest
            output=folder/f"{arm}-{b}.npz"
            legacy=False
            if output.exists():
                with np.load(output) as saved:
                    legacy="checkpoint_sha256" not in saved.files
                    if not legacy and (str(saved["checkpoint_sha256"])!=checkpoint_digest
                                       or str(saved["evaluator_sha256"])!=evaluation_digest):
                        raise ValueError(f"Cached array belongs to a different graph or evaluation: {output}")
            if not output.exists() or legacy:
                g,r=tri.old_fit.load_checkpoint(path,"odor",tag,b,stage)
                raw=dict(truth=truth.astype(np.int8))
                for i,order in enumerate(ORDERS):
                    for mode in MODES:
                        sides,receipts=[],[]
                        for offset in range(0,N_WORLD,12):
                            tape={k:v[offset:offset+12] for k,v in base.items()}
                            selected,seen=simulate(g,r,tape,i,mode)
                            sides.append(selected);receipts.append(seen)
                        raw[f"{order}/{mode}/side"]=np.concatenate(sides)
                        raw[f"{order}/{mode}/contact"]=np.concatenate(receipts)
                if not step:
                    source=DATA/"triangle-transfer"/label/f"{b}-64000.npz"
                    identities=json.loads((EVIDENCE/f"triangle-transfer-{label}.json").read_text())["raw_sha256"]
                    if (identities[checkpoint.relative_to(DATA).as_posix()]!=checkpoint_digest
                            or sha256(source.read_bytes()).hexdigest()!=identities[source.name]):
                        raise ValueError("The parent's archived model card does not belong to this graph")
                    with np.load(source) as z:
                        other={k:z["metric/"+k] for k in geometry.CARD_KEYS}
                else:other,_=cards.row(g,r,card_spec["seed"],card_spec["worlds"])
                raw["card"]=np.stack([other[k] for k in geometry.CARD_KEYS],axis=1)
                if legacy:
                    with np.load(output) as old:
                        for k in raw:np.testing.assert_array_equal(raw[k],old[k])
                raw.update(checkpoint_sha256=checkpoint_digest,evaluator_sha256=evaluation_digest)
                tri.revisit._write_raw(output,**raw)
            hashes[output.name]=sha256(output.read_bytes()).hexdigest()
            with np.load(output) as z:
                ordinary.append(z["card"])
                collect(z,truth,conditions,results)
            print(cohort,step,arm,b,flush=True)
        cells[arm]={k:np.stack(v) for k,v in results.items()};card_rows[arm]=np.stack(ordinary)
    if step:
        matched={};parent_cards=[]
        for b in births:
            with np.load(DATA/"cross-skill-eval"/cohort/"0"/f"parent-{b}.npz") as z:
                parent_path,parent_arm=geometry.sources(cohort,"hard",64000)
                key=tri.old_fit.checkpoint_path(parent_path,"odor",parent_arm,b,64000)
                if (str(z["checkpoint_sha256"])!=sha256(key.read_bytes()).hexdigest()
                        or str(z["evaluator_sha256"])!=evaluation_digest):
                    raise ValueError("The comparison parent does not match the deployment evaluation")
                collect(z,truth,conditions,matched)
                parent_cards.append(z["card"])
        cells["parent"]={k:np.stack(v) for k,v in matched.items()}
        card_rows["parent"]=np.stack(parent_cards)
        for a,b in (("alternating","old_then_new"),("alternating","new_then_old"),("old_then_new","new_then_old"),
                    *((arm,"parent") for arm in arms)):
            cells[a+" minus "+b]={k:v-cells[b][k] for k,v in cells[a].items()}
            card_rows[a+" minus "+b]=card_rows[a]-card_rows[b]
    boot=ci.Crossed(len(births),N_WORLD,seed+1,4000)
    card_boot=ci.Crossed(len(births),card_spec["worlds"],card_spec["seed"]+19,4000)
    result=dict(births=births,worlds=N_WORLD,step=step,seed=seed,hashes=hashes,
                notes="Three triangle objects plus A/B context values and one feeder mark make six independent "
                      "laws. Four additional supports comprise two actual context contacts and both marked feeder "
                      "demonstrations. Each of the six terminal actions forks the identical continuous final state. "
                      "All 64 private law assignments are crossed within each world. The separate common card "
                      "contains independent assays on the same inherited model. Paired intervals resample training "
                      "births and their shared test geometries; six terminal choices are not independent units.",
                same_life={a:{k:{name:boot(v[:,:,i]) for i,name in enumerate(METRICS)} for k,v in rows.items()}
                           for a,rows in cells.items()},
                separate_card={a:{k:card_boot(v[:,:,i]) for i,k in enumerate(geometry.CARD_KEYS)}
                               for a,v in card_rows.items()})
    (EVIDENCE/f"cross-skill-{cohort}-{step}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--step",choices=(0,4000,8000),type=int,required=True)
    args=p.parse_args()
    run(args.cohort,args.step)
