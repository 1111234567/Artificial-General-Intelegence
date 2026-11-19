"""A compact, reproducible scorecard for retained rules and continuously evolving physical actors."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from . import research_audit as data,replicate_statistics,replicate_audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def get(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def banked():
    folder=data.OUT/"self-organized-twelve-births"
    base=OUT/"independent-lifelong-births-v1";reg=get(base/"registered.json")
    book=defaultdict(list);shapes={};collisions={};parity=[]
    for seed in reg["fixed_birth_seeds"]:
        row=get(folder/f"birth-{seed}.json");parity.append(row["parity"])
        er=replicate_audit.audit_seeds(base/f"birth-{seed}",{"recipe_seed":seed},row["roots"]+row["long_roots"])
        bad={k:v["intersection"] for k,v in er.items() if v["intersection"]}
        if bad:collisions[str(seed)]=bad
        for situation,metrics in row["cases"].items():
            for arm in ("distributed","local","difference"):
                v=metrics["local"] if arm=="difference" else metrics[arm]
                for key in ("B_after_0","B_after_2","B_after_4","B_after_8","B_after_12","B_after_24","B_after_48","return_A","both","coverage"):
                    if key not in v:continue
                    x=np.asarray(v[key])
                    if arm=="difference":x=x-np.asarray(metrics["distributed"][key])
                    for kind,idx in (("same",[0,3]),("conflict",[1,2])):
                        book[f"{situation}/{arm}/{kind}/{key}"].append(x[idx].mean(0))
        for label,metrics in row["long"].items():
            for regname in ("stable_A","old_B","change_1_1","change_2_2","change_3_4","change_5_8",
                         "change_9_16","return_3_4","whole_after_warmup"):
                v=metrics[regname]
                for step in ("pre","post"):
                    nr=np.asarray(v["count"]);h=np.asarray(v[step])
                    if not np.all(nr):raise ValueError((label,regname))
                    book[f"stream/{label}/{regname}/{step}"].append(h/nr)
    for key in list(book):
        if "/local/" not in key or not key.startswith("stream/"):continue
        for tag in ("distributed","local_no_reopening"):
            other=key.replace("/local/",f"/{tag}/")
            book[key.replace("/local/",f"/local-minus-{tag}/")]=np.asarray(book[key])-np.asarray(book[other])
    return dict(independent_births=reg["fixed_birth_seeds"],seed_intersections=collisions,native_parity=parity,
                definition="12 independent training births; 256 paired roots across four private families; long streams use 96 further roots",
                results={k:replicate_statistics.crossed(v) for k,v in book.items()})


def forks():
    files=sorted((OUT/"native-long-life-v1/retention-fork").glob("cohort-*.json"))
    if len(files)!=8:return {}
    groups=[get(p) for p in files];out={}
    for arm in groups[0]["rows"]:
        out[arm]={}
        for k in ("first_A","first_B","first_both","all","coverage"):
            a=np.concatenate([g["rows"][arm][k] for g in groups])
            out[arm][k]=data.estimates(a)
        if arm!="ongoing":
            x=np.concatenate([np.asarray(g["rows"][arm]["all"])-np.asarray(g["rows"]["ongoing"]["all"]) for g in groups])
            out[arm]["minus_ongoing"]=data.estimates(x)
    return out


def live(name):
    folder=OUT/name
    rows=[get(p) for p in sorted(folder.glob("cohort-*/status.json"))]
    if len(rows)!=3 or any(r["done_chunks"]!=r["total_chunks"] for r in rows):return {"complete":False}
    comparison={}
    for k in rows[0]["counters"]:
        values=np.concatenate([r["counters"][k] for r in rows])
        if not np.all(values[:,0]):continue
        comparison[k]={step:data.estimates(values[:,j]/values[:,0]) for j,step in ((1,"pre"),(2,"post"))}
        comparison[k]["opportunities"]=int(values[:,0].sum())
    return dict(complete=True,reference_birth=get(OUT/"morning-20261119/portable-verification.json")["identity"]["birth_seed"],
                worlds=sum(len(r["seeds"]) for r in rows),requests_per_life=rows[0]["requests_per_life"],
                ticks_per_life=rows[0]["ticks_per_life"],results=comparison,
                continuation_pulses=sum(v["motor_bits"] for r in rows for v in r["checkpoint_continuations"]),
                exact_continuations=sum(len(r["checkpoint_continuations"]) for r in rows),action_chain=[r["action_chain"] for r in rows],
                snapshot_size_range_bytes=[min(r["state_file_bytes"] for r in rows),max(r["state_file_bytes"] for r in rows)])


def auxiliary():
    out={}
    for section,glob,field in (("replay","continuous-replay-sealed","rows"),("A_return","A-return-forks","rows"),
                                ("memory_review","old-experience-sealed","rows"),("context_repair","old-habitat-repair-sealed","rows")):
        book=defaultdict(list)
        for p in sorted((data.OUT/glob).glob("*.json")):
            if p.stem=="complete":continue
            d=get(p)
            if section=="replay":
                for a,vals in d[field].items():
                    for m in ("first","both","coverage"):
                        book[f"{d['geometry']}/{d['pairs']}/{a}/{m}"].append(vals[m])
            elif section=="context_repair":
                for name,v in d[field].items():
                    for m in ("B_after_0","return_A","both"):
                        book[f"{d['kind']}/{d['curriculum']}/{name}/{m}"].append(np.asarray(v[m])[[1,2]].mean(0))
            else:
                for a,vals in d[field].items():
                    for m in (("A","both","sign_before_B","sign_after_B") if section=="A_return" else
                              ("return_A","last_B","both","internal_updates")):
                        book[f"{d['case']}/{a}/{m}"].append(np.asarray(vals[m])[[1,2]].mean(0))
        out[section]={k:data.estimates(np.mean(x,axis=0)) for k,x in book.items()}
    return out


def main():
    folder=OUT/"morning-20261119";folder.mkdir(exist_ok=True)
    _write(folder/"learning-and-retention.json",dict(local_addresses=banked(),forks=forks(),
         native_stream=live("native-long-life-v1"),native_local_stream=live("native-long-life-local-v1"),native_selective_stream=live("native-long-life-selective-v1"),auxiliary=auxiliary()))


if __name__=="__main__":main()

