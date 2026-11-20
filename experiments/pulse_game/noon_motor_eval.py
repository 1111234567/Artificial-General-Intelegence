"""Which action benefits from a separate native motor port? Test the same physical continuations."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from . import noon_motor_roles as roles, noon_readback as bank
from . import noon_review_world as w, noon_paired_lesson as lesson
from .interleave_capacity_report import Crossed


def load(cohort,b,arm,stage):
    if arm=="native":
        path=w.DATA/"core"/cohort;head=torch.zeros(1,roles.net.NODES+1)
    else:
        path=roles.OUT/cohort
        with np.load(path/arm/f"head-{stage}.npz") as z:
            i=z["births"].tolist().index(b);head=torch.from_numpy(z["weights"][i:i+1])
    g,r=roles.old.load_checkpoint(path,"odor",arm,b,stage)
    return g,r,head,roles.old.checkpoint_path(path,"odor",arm,b,stage)


def all_queries(view,state,t,weight,arm):
    good=[];left=[]
    for i in range(t.count):
        idx=torch.full((t.lanes,),i,dtype=torch.long)
        a,truth=roles.query(view,state,t,idx,weight,arm)
        good.append(a.side==truth);left.append(a.side)
    return torch.stack(good,1),torch.stack(left,1)


@torch.no_grad()
def measure(g,r,head,t,arm):
    view,weight,state,visits=roles.start(g,r,t,head,arm)
    ordinary,ord_side=all_queries(view,state,t,weight,arm)
    choice=roles.select(view,state,t,weight,arm)
    base=roles.select(view,state,t,weight,arm,lesion=True)
    # Reverse which polarity is displayed on both candidate petals without
    # changing their identity, the original state used to execute, or the food.
    flipped=bank.replace(t,menu_orientation=-t.menu_orientation)
    mirror=roles.select(view,state,flipped,weight,arm)
    results=dict(side=choice.side.numpy().astype(np.int8),logit=choice.motor_logit.numpy(),
                 ordinary=ord_side.numpy().astype(np.int8),value=t.law.numpy().astype(np.int8),
                 candidates=t.candidates.numpy(),menu=t.menu_orientation.numpy(),terminal=t.final_item.numpy(),
                 terminal_truth=(t.law*t.final_orientation).numpy().astype(np.int8),
                 support=torch.stack([torch.stack((x.motor.side,x.earned),1) for x in visits],1).numpy().astype(np.int8))
    branches=[];writes=[]
    for side in (-1.,1.):
        use=torch.full_like(t.decision_draw,side)
        per=[]
        for allow in (True,False):
            after=roles.review(view,choice.state,t,use,weight,arm,write=allow)
            good,actual=all_queries(view,after.state,t,weight,arm)
            per.append(good.numpy())
            results[f"{side}/{allow}/final"]=actual.numpy().astype(np.int8)
            results[f"{side}/{allow}/support"]=torch.stack((after.motor.side,after.earned),1).numpy().astype(np.int8)
        np.testing.assert_array_equal(results[f"{side}/True/support"],results[f"{side}/False/support"])
        branches.append(per[0]);writes.append(per[1])
    row=np.arange(t.lanes)
    chosen=choice.side.numpy()>0
    methods={"native":chosen.astype(float),"random":np.full(t.lanes,.5),"reverse_polarity":(mirror.side.numpy()>0).astype(float),
             "disable_review_port":(base.side.numpy()>0).astype(float),
             "redistribute":np.repeat(chosen.reshape(-1,8).mean(1),8)}
    offered_val=-np.take_along_axis(results["value"],results["candidates"],axis=1)*results["menu"]
    is_opposed=offered_val[:,0]!=offered_val[:,1]
    nominal=offered_val[row,chosen.astype(int)]>0
    values=dict(normal_all=ordinary.all(1).numpy(),normal_mean=ordinary.float().mean(1).numpy(),
                right=chosen,polarity_flips=chosen!=(mirror.side.numpy()>0))
    # All eight tested paired histories contain a balanced number of opposed trials.
    values["positive_given_opposed"]=(is_opposed&nominal).astype(float)*2
    for mode,table in (("active",np.stack(branches,1)),("sham",np.stack(writes,1))):
        both=np.take_along_axis(table,np.broadcast_to(results["candidates"][:,None,:],(t.lanes,2,2)),axis=2).all(2)
        scores=dict(mean=table.mean(2),all=table.all(2),both=both,paid=table[row[:,None],np.arange(2)[None],results["terminal"][:,None]])
        for label,z in scores.items():
            left=z[:,0].astype(float);dy=z[:,1].astype(float)-left
            for method,select in methods.items():
                values[f"{mode}/{method}/{label}"]=left+select*dy
            for other in ("random","reverse_polarity","disable_review_port","redistribute"):
                values[f"{mode}/native_minus_{other}/{label}"]=(methods["native"]-methods[other])*dy
    return results,{k:v.reshape(-1,8).mean(1) for k,v in values.items()}


def run(cohort,stage,worlds,arms=None,seed_offset=0,label=None):
    torch.set_num_threads(1)
    arms=arms or ("native",*roles.ARMS)
    ids=list(w.COHORTS[cohort]);seed=(4039591 if cohort=="original" else 4039619)+seed_offset
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    out=w.DATA/"motor-role-eval"/(label or cohort)/str(stage);out.mkdir(parents=True,exist_ok=True)
    code=digest(__file__);hashes={};series={}
    for arm in arms:
        grouped={}
        for b in ids:
            g,r,h,f=load(cohort,b,arm,stage);identity=digest(f)
            extra=roles.OUT/cohort/arm/f"head-{stage}.npz"
            hh=digest(extra) if arm!="native" else "native"
            dest=out/f"{arm}-{b}.npz"
            if not dest.exists():
                raw={}
                for count in (2,4,8):
                    t,_,_=bank.tape(seed,worlds,count);parts=[];readings=[]
                    for j in range(0,worlds,32):
                        idx=torch.arange(8*j,min(8*(j+32),t.lanes));q=lesson.expand(t,idx)
                        data,values=measure(g,r,h,q,arm);parts.append(data);readings.append(values)
                    raw.update({f"{count}/"+k:np.concatenate([z[k] for z in parts]) for k in parts[0]})
                    raw.update({f"metric/{count}/"+k:np.concatenate([z[k] for z in readings]) for k in readings[0]})
                roles.core.archive._write_raw(dest,**raw,body=identity,motor=hh,code=code,worlds=worlds,seed=seed)
            with np.load(dest) as z:
                if (str(z["body"]),str(z["motor"]),str(z["code"]),int(z["worlds"]),int(z["seed"]))!=(identity,hh,code,worlds,seed):
                    raise ValueError("Unmatched motor-role cache")
                for k in z:
                    if k.startswith("metric/"):grouped.setdefault(k[7:],[]).append(z[k])
            hashes[f.relative_to(w.DATA).as_posix()]=identity
            if arm!="native":hashes[extra.relative_to(w.DATA).as_posix()]=hh
            hashes[dest.relative_to(w.DATA).as_posix()]=digest(dest)
            print(cohort,stage,arm,b,flush=True)
        series[arm]={k:np.stack(v) for k,v in grouped.items()}
    for a,b in (("review","shared"),("review","native"),("reserved","native")):
        if a in series and b in series:series[a+" minus "+b]={k:v-series[b][k] for k,v in series[a].items()}
    ci=Crossed(len(ids),worlds,seed+67,3000)
    report=dict(births=ids,seed=seed,worlds=worlds,stage=stage,source=code,hashes=hashes,
        data={a:{k:ci(v) for k,v in vals.items()} for a,vals in series.items()},
        note="Same offers, geometries, native clocks and candidate-private law crossings. Native-vs-random compares "
             "one sampled native route to exact equal allocation over the same two actual experienced outcomes; "
             "donating routes averages other private histories in the same world. The shared-head model uses the "
             "learned head for harvesting as well. The lesion removes the route adjustment only at the decision.")
    roles.core.archive._write_json(w.EVIDENCE/f"motor-{label or cohort}-{stage}.json",report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=w.COHORTS,required=True)
    p.add_argument("--stage",type=int,required=True);p.add_argument("--worlds",type=int,default=256)
    p.add_argument("--arms",nargs="+");p.add_argument("--label");p.add_argument("--seed-offset",type=int,default=0)
    run(**vars(p.parse_args()))
