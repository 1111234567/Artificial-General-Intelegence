"""Check whether an afferent learned signal helps, and whether the native body retained it after fading."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_motor_feedback as fit,noon_market_stats as ci
from . import noon_transfer as transfer


@torch.no_grad()
def read_world(g,r,critic,t,first,arm,step):
    n=t.lanes;row=torch.arange(n);view,s,_,signal,z=fit.begin(g,r,t,first,critic)
    raw=dict(sensory_readings=z.numpy(),feedback_current=signal.numpy(),candidates=t.candidates.numpy(),
             terminal_truth=(t.law*t.final_orientation).numpy().astype(np.int8),u=t.decision_draw.numpy())
    metrics={};saved=None
    amplitude=fit.strength(arm,step)
    cases=[(0.,"off"),(1.,"on")]
    if amplitude not in (0.,1.):cases.append((amplitude,"scheduled"))
    for amp,name in cases:
        a=fit.emit(view,s,t,signal,amp)
        if amp==0:
            reference=fit.w.select(view,s,*fit.w.candidate_menu(t),t.decision_draw)
            torch.testing.assert_close(a.motor_logit,reference.motor_logit,atol=0,rtol=0)
            torch.testing.assert_close(a.state.hidden,reference.state.hidden,atol=0,rtol=0)
        idx=(a.side>0).float();raw[name+"/route"]=a.side.numpy().astype(np.int8);raw[name+"/logit"]=a.motor_logit.numpy()
        # Yoke only the afferent current. The sensory menu and all other activity are recipient-specific.
        donated=[]
        if amp:
            for shift in range(8):
                donor=row//8*8+(row%8+shift)%8
                donated.append((fit.emit(view,s,t,signal[donor],amp).side>0).float())
            donor_side=torch.stack(donated).mean(0)
        else:donor_side=idx
        for mode,write in (("active",True),("sham",False)):
            scored=[];out=[]
            for side in (-1.,1.):
                ss,contact=fit.w.review(view,a.state,t,torch.full_like(a.side,side),write=write)
                acts=[];correct=[]
                for item in range(t.count):
                    indicator=torch.full((n,),item,dtype=torch.long);l,right=fit.w.sensory_for(t,indicator,t.final_orientation[:,item])
                    selected=fit.motor.act(ss,l,right,t.final_draw[:,item]).side
                    acts.append(selected);correct.append(selected==t.law[:,item]*t.final_orientation[:,item])
                good=torch.stack(correct,1)
                scored.append(dict(all=good.all(1).float(),mean=good.float().mean(1),
                                   both=good.gather(1,t.candidates).all(1).float(),paid=good[row,t.final_item].float(),
                                   immediate=(contact.earned>0).float(),net=(contact.earned>0).float()+good[row,t.final_item].float()))
                out.append(torch.stack(acts,1).numpy().astype(np.int8))
                raw[f"{name}/{mode}/{side}/received"]=torch.stack((contact.motor.side,contact.earned),1).numpy().astype(np.int8)
            raw[f"{name}/{mode}/final"]=np.stack(out,1)
            for key in scored[0]:
                l=scored[0][key];delta=scored[1][key]-l
                vals={"native":l+idx*delta,"random":l+.5*delta,"native_minus_random":(idx-.5)*delta,
                      "native_minus_donor_current_action":(idx-donor_side)*delta}
                for k,v in vals.items():metrics[f"{name}/{mode}/{k}/{key}"]=v.reshape(-1,8).mean(1).numpy()
        if name=="off":saved=idx
        elif name=="on":metrics["changed_by_current"]=(saved!=idx).float().reshape(-1,8).mean(1).numpy()
    used="off/" if amplitude==0 else "on/" if amplitude==1 else "scheduled/"
    metrics.update({"deployed/"+k[len(used):]:v for k,v in metrics.items() if k.startswith(used)})
    return raw,metrics


def run(cohort,step,worlds,arms,*,offset=0,label=None):
    torch.set_num_threads(1);critic,head=fit.precursor();ids=list(fit.w.COHORTS[cohort])
    seed=(4219643 if cohort=="original" else 4219661)+offset
    dest=fit.w.DATA/"feedback-eval"/(label or cohort)/str(step);dest.mkdir(parents=True,exist_ok=True)
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    code=sha(__file__);checkpoints={str(head.relative_to(fit.w.DATA)) : sha(head)};series={}
    for arm in arms:
        per={}
        for b in ids:
            g,r=fit.old.load_checkpoint(fit.OUT/cohort,"odor",arm,b,step)
            cp=fit.old.checkpoint_path(fit.OUT/cohort,"odor",arm,b,step);identity=sha(cp)
            f=dest/f"{arm}-{b}.npz"
            if not f.exists():
                raw={}
                for count in ((2,4,8) if step==6000 else (4,)):
                    t,first,_=fit.bank.tape(seed,worlds,count);rs=[];ms=[]
                    for j in range(0,worlds,24):
                        ix=torch.arange(8*j,min(8*(j+24),t.lanes))
                        rr,vv=read_world(g,r,critic,fit.bank.lesson.expand(t,ix),first[ix],arm,step)
                        rs.append(rr);ms.append(vv)
                    raw.update({f"{count}/{k}":np.concatenate([q[k] for q in rs]) for k in rs[0]})
                    raw.update({f"metric/{count}/{k}":np.concatenate([q[k] for q in ms]) for k in ms[0]})
                fit.bank.lesson.storage._write_raw(f,**raw,body=identity,head=sha(head),evaluator=code,worlds=worlds,seed=seed)
            with np.load(f) as z:
                if (str(z["body"]),str(z["head"]),str(z["evaluator"]),int(z["worlds"]),int(z["seed"]))!=(identity,sha(head),code,worlds,seed):
                    raise ValueError("Mismatched native-current scoring cache")
                for k in z:
                    if k.startswith("metric/"):per.setdefault(k[7:],[]).append(z[k])
            checkpoints[cp.relative_to(fit.w.DATA).as_posix()]=identity;checkpoints[f.relative_to(fit.w.DATA).as_posix()]=sha(f)
            print(cohort,step,arm,b,flush=True)
        series[arm]={k:np.stack(v) for k,v in per.items()}
    report=dict(cohort=cohort,step=step,births=ids,worlds=worlds,seed=seed,hashes=checkpoints,
                results=ci.summarize(series,worlds,ids,seed+67,[("attached","unassisted"),("fade","unassisted"),("attached","fade")]),
                definition="off, on are true interventions on the recurrent motor current. deployed uses the "
                           "scheduled stage's amplitude. Donor-current action keeps the recipient's original "
                           "post-menu state and substitutes only the alternative native motor choice.")
    fit.bank.lesson.storage._write_json(fit.w.EVIDENCE/f"feedback-{label or cohort}-{step}.json",report)


def transfer_card(cohort):
    torch.set_num_threads(1);cfg=json.loads(transfer.PLAN.read_text());ids=list(fit.w.COHORTS[cohort])
    seeds=cfg["cohorts"][cohort]["seeds"];worlds=cfg["worlds"]
    directory=fit.w.DATA/"feedback-transfer"/cohort;directory.mkdir(parents=True,exist_ok=True)
    reports={};hashes={};digest=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    for arm in ("parent",*fit.ARMS):
        rows={}
        for b in ids:
            root,role,step=(fit.w.PARENT_DATA/"one-life-repair"/cohort,"mixed",2000) if arm=="parent" else (fit.OUT/cohort,arm,6000)
            cp=fit.old.checkpoint_path(root,"odor",role,b,step);identity=digest(cp)
            g,r=fit.old.load_checkpoint(root,"odor",role,b,step)
            f=directory/f"{arm}-{b}.npz"
            if not f.exists():
                z=transfer.evaluate(g,r,seeds,worlds)
                fit.bank.lesson.storage._write_raw(f,body=identity,**z)
            with np.load(f) as saved:
                if str(saved["body"])!=identity:raise ValueError("Different retention body")
                tab=transfer.score(saved)
            for key,val in tab.items():rows.setdefault(key,[]).append(val)
            hashes[f.relative_to(fit.w.DATA).as_posix()]=digest(f)
            hashes[str(cp.relative_to(fit.w.ROOT))]=identity
        reports[arm]={k:np.stack(v) for k,v in rows.items()}
        print(cohort,arm,"retention",flush=True)
    out={}
    for domain in worlds:
        part={a:{k:v for k,v in rs.items() if k.startswith(domain+"/")} for a,rs in reports.items()}
        out[domain]=ci.summarize(part,worlds[domain],ids,seeds[domain]+71,[("attached","parent"),("fade","parent"),
                         ("unassisted","parent"),("attached","unassisted"),("fade","unassisted")])
    fit.bank.lesson.storage._write_json(fit.w.EVIDENCE/f"feedback-transfer-{cohort}.json",dict(births=ids,worlds=worlds,seeds=seeds,
                hashes=hashes,domains=out,note="No active-review opportunity in these old assays; the learned "
                "current is specific to its review gate and therefore absent in these deployments."))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=fit.w.COHORTS,required=True)
    p.add_argument("--step",type=int,default=6000);p.add_argument("--arms",nargs="+",default=fit.ARMS)
    p.add_argument("--worlds",type=int,default=256);p.add_argument("--offset",type=int,default=0)
    p.add_argument("--label");p.add_argument("--transfer",action="store_true");a=p.parse_args()
    if a.transfer:transfer_card(a.cohort)
    else:run(a.cohort,a.step,a.worlds,a.arms,offset=a.offset,label=a.label)
