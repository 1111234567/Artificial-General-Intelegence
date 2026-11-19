"""Publish only after the full large-search audit, never from a partial curve."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics
from experiments.learned_binding.common import write_json

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"data/seed-evolution/2026-11-11-large-v1"
DEST=ROOT/"docs/evidence/seed-evolution-2026-11-11-large"
CONDITIONS=((0,"own"),(64,"own"),(512,"own"),(512,"teacher"),(512,"no_read"),(512,"Bdelay"))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rate(report,group,metric):
    return statistics.mean(s["groups"][group][metric]/s["groups"][group]["n"] for s in report["streams"])

def aggregate(root=DEFAULT,dest=DEST):
    root,dest=Path(root),Path(dest)
    audit=read(root/"audit.json")
    if (audit.get("full_audit_passed") is not True or audit.get("candidates_provenance_checked")!=12096
            or audit.get("heldout_ledgers_recounted")!=72 or audit.get("heldout_policy_replays")!=9):
        raise ValueError("complete independent audit required; no partial conclusions")
    for file,key in (("protocol.json","protocol_sha256"),("search.json","search_sha256"),
                     ("final-freeze.json","final_freeze_sha256"),("test-completion.json","test_completion_sha256")):
        if sha(root/file)!=audit.get(key):raise ValueError(f"audited source changed: {file}")
    expected={}
    completion=read(root/"test-completion.json")
    if len(completion["models"])!=6:raise ValueError("six audited finalists required")
    for row in completion["models"]:
        winner=row["winner"]
        for item in row["conditions"]:
            key=(winner["rep"],winner["arm"],item["name"])
            if key in expected:raise ValueError("duplicate audited test report")
            expected[key]=item["sha256"]
    if len(expected)!=36:raise ValueError("36 audited test reports required")
    search=read(root/"search.json");config=read(root/"protocol.json")["config"]
    curves=[];bindings={};tests=[];comparison=[]
    for h in search["history"]:
        records=search["records"][str(h["rep"])]
        scores=[records[c]["fitness"] for c in h["new_ids"]]
        curves.append(dict(rep=h["rep"],arm=h["arm"],generation=h["generation"],
            new_mean=statistics.mean(scores),new_median=statistics.median(scores),new_min=min(scores),new_max=max(scores),
            best=h["best_fitness"],best_id=h["best_id"],cumulative=len(h["evaluated_ids"])))
    lookup={}
    for r in config["source_seeds"]:
        for arm in ("evolution","random"):
            for step,condition in CONDITIONS:
                p=root/"final"/f"rep_{r}"/arm/"evaluations"/f"{step:03d}_{condition}.json"
                if sha(p)!=expected.get((r,arm,f"{step:03d}_{condition}")):
                    raise ValueError(f"audited test score changed: {p}")
                x=read(p);bindings[str(p.relative_to(ROOT))]=sha(p)
                if sha(p.with_suffix(".pt"))!=x["archives_sha256"]:
                    raise ValueError(f"audited trajectory changed: {p}")
                row=dict(rep=r,arm=arm,step=step,condition=condition,winner=x["winner"],fitness=x["fitness"],
                    all_strict=rate(x,"all","strict"),A_conflict=rate(x,"A_conflict","strict"),
                    B_conflict=rate(x,"B_conflict","strict"),content=rate(x,"all","translation_correct"),
                    idle_per_1000=1000*statistics.mean(s["probe"]["idle_actions"]/s["probe"]["idle_opportunities"] for s in x["streams"]),
                    counts=[s["groups"] for s in x["streams"]])
                tests.append(row);lookup[r,arm,step,condition]=row
    for step,condition in CONDITIONS:
        left=[lookup[r,"evolution",step,condition] for r in config["source_seeds"]]
        right=[lookup[r,"random",step,condition] for r in config["source_seeds"]]
        comparison.append(dict(step=step,condition=condition,evolution_mean=statistics.mean(x["fitness"] for x in left),
            random_mean=statistics.mean(x["fitness"] for x in right),
            paired_differences=[a["fitness"]-b["fitness"] for a,b in zip(left,right)]))
    gains={arm:[lookup[r,arm,512,"own"]["fitness"]-lookup[r,arm,0,"own"]["fitness"] for r in config["source_seeds"]]
           for arm in ("evolution","random")}
    evidence=dict(config=config,search_curves=curves,tests=tests,comparisons=comparison,training_gains_512_minus_0=gains,
        winners=search["winners"],audit_sha256=sha(root/"audit.json"),protocol_sha256=sha(root/"protocol.json"),
        search_sha256=sha(root/"search.json"),report_sources=bindings,analysis_source_sha256=sha(__file__),
        limits=["Three paired searches, not 12096 independent hypothesis tests.",
                "Best-so-far rises by construction. Compare new-candidate distributions and random search.",
                "Step0 tests an evolved prior; 0-to512 tests additional learning. Short64 fitness is only a proxy.",
                "No-read starts before own support and changes its feedback trajectory.",
                "Bdelay evaluates later requests. Across-condition differences are not same-request paired effects."])
    dest.mkdir(parents=True,exist_ok=True);write_json(dest/"results.json",evidence)
    with (dest/"search-curves.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    with (dest/"test-comparison.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["step","condition","evolution_mean_pct","random_mean_pct","paired_delta_pp"])
        for x in comparison:writer.writerow([x["step"],x["condition"],100*x["evolution_mean"],100*x["random_mean"],[100*d for d in x["paired_differences"]]])
    lines=["# 大规模初始化搜索：完整结果","","仅在全量独立审计通过后生成。12,096候选、三对搜索；精英曲线单调上升本身不证明有效。","",
           "| 步数 / 条件 | 进化宏平均严格率 | 随机搜索宏平均严格率 | 三对差值（百分点） |","|---|---:|---:|---|"]
    for x in comparison:lines.append(f"| {x['step']} / {x['condition']} | {100*x['evolution_mean']:.2f}% | {100*x['random_mean']:.2f}% | "+", ".join(f"{100*v:+.2f}" for v in x["paired_differences"])+" |")
    lines+= ["","0步表现与训练增益分别解释；只有三对搜索，不将请求数当成搜索重复数。保持任务难度未变，本轮没有直接检验更容易的课程。","", "完整来源与分母见 [results.json](results.json)。"]
    (dest/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return dict(output=str(dest),comparisons=comparison)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=DEFAULT);p.add_argument("--output",type=Path,default=DEST)
    a=p.parse_args();print(json.dumps(aggregate(a.root,a.output),ensure_ascii=False))
