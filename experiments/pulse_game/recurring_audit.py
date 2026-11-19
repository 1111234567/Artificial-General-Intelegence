"""Test consolidation against duplicate writes, equal storage and independent physical rollouts."""
import json
import time
import numpy as np
import torch
from . import recurring_memory as memory, recurring_train as selection, recurring_world as task
from . import symmetry_policy as controller, symmetry_policy_train as environments, symmetry_memory_audit as physical_metrics
from . import mixture_flash as flash, symmetry_world as game, research_audit as audit
from .attractor_search import OUT
from .paired_choice import first_contact_pairs
from experiments.seed_evolution.large_runner import _write


SCHEDULE={
 "repeated_anchor":dict(gap=32,distinct=8,query="anchor"),
 "repeated_unannounced":dict(gap=32,distinct=8,query="random"),
 "few_kinds_long":dict(gap=64,distinct=6,query="random"),
 "many_kinds":dict(gap=48,distinct=24,query="random"),
 "rarely_seen":dict(gap=48,distinct=16,query="rare"),
 "all_novel_anchor":dict(gap=32,distinct=8,query="anchor",recurring=False),
 "all_novel_unannounced":dict(gap=32,distinct=8,query="random",recurring=False),
 "noisy_repetition":dict(gap=40,distinct=8,query="random",spread=.16,noise=.24),
 "similar_opposite":dict(gap=40,distinct=12,query="random",conflicts=True,spread=.12),
}


def summarize(w,result,master):
    data=game.metrics(w,result);n=len(data["first"])//2
    a,b=data["first"][:n],data["first"][n:]
    pairs={k:v.tolist() for k,v in first_contact_pairs(w,result).items()}
    return dict(first=((a+b)*.5).tolist(),both=(a*b).tolist(),even=a.tolist(),odd=b.tolist(),
        contact=((data["contact"][:n]+data["contact"][n:])*.5).tolist(),**pairs)


def run(file,seeds,dest):
    f=json.loads(file.read_text());params,body,old,src=selection.parent(f["parent"])
    genes=np.asarray(f["genes"])
    sheet={};parity={}
    for name,args in SCHEDULE.items():
        w,targets=task.make(seeds,**args,nonce=8431993)
        co=memory.Cohort(params,w)
        arms={"selected":(f["mode"],genes),"one_per_contact":(0,genes),"untrained_merge":(1,np.zeros(7))}
        if f["mode"]==2 and name in ("repeated_anchor","repeated_unannounced","noisy_repetition","similar_opposite"):
            arms.update(always_allocate=(3,genes),last_sign=(4,genes),first_sign=(5,genes),wrong_address=(6,genes))
        sheet[name]={}
        for label,(mode,x) in arms.items():
            r,g=co.replacements(body,x,mode=mode,individual=True)[0];sheet[name][label]=summarize(w,r,g)
        if f["mode"]==2 and name in ("repeated_unannounced","all_novel_unannounced","many_kinds"):
            expanded={**params,"capacity":32}
            c2=memory.Cohort(expanded,w)
            for name2,m2 in (("expanded_one_per_contact",0),("expanded_consolidation",2)):
                r,g=c2.replacements(body,genes,mode=m2,individual=True)[0]
                sheet[name][name2]=summarize(w,r,g)
        if name=="repeated_unannounced":
            subset=seeds[:32];test,_=task.make(subset,**args,nonce=8431993)
            chosen=f["mode"];small=memory.Cohort(params,test)
            for label,mode in (("selected",chosen),("one_per_contact",0),("always_allocate",3)):
                r,g=small.replacements(body,genes,mode=mode,individual=True)[0]
                actor=memory.Forager(controller.materialize(params,old,src["settings"]["mode"]),genes,subset+subset,mode)
                expected=flash.rollout(actor,test.menu)
                parity[label]=dict(actions=float((r.contact_side==expected.contact_side).mean()),
                    contact_tick=float((r.contact_tick==expected.contact_tick).mean()),sign=float((r.reward==expected.reward).mean()),
                    master_max=float(np.max(np.abs(g-np.stack(actor.record,1)))),mean_merges=float(actor.merges.mean()),
                    mean_new_addresses=float(actor.count.mean()),max_live_slots=int(np.max((actor.count).clip(max=params["capacity"]))))
    _write(dest/(file.parent.name+".json"),dict(model=f,parent_sha=audit.fingerprint(file),seeds=seeds,cases=sheet,
           parity=parity,pair_choice_protocol="both_first_contacts_required"))
    print("confirmed",file.parent.name,parity["selected"],flush=True)


def main():
    torch.set_num_threads(1);root=OUT/"recurring-write-v1";dest=audit.OUT/"recurring-write-sealed"
    dest.mkdir(exist_ok=True);seeds=environments.evalseeds(327419041,512)
    while True:
        paths=sorted(root.glob("*/frozen.json"))
        for file in paths:
            if not (dest/(file.parent.name+".json")).is_file():run(file,seeds,dest)
        if len(paths)==6:break
        time.sleep(8)
    _write(dest/"complete.json",dict(roots=len(seeds),models=6))


if __name__=="__main__":main()
