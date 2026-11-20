"""Verify the material used for the overnight behavioral comparisons against its sealed file index."""

from hashlib import sha256
from pathlib import Path
import platform
import json
import re

import numpy as np
import torch

from ..interleave_interference import DATA, EVIDENCE
from .. import local_trace_train as model_io


def check_exports(state,location,arm,births):
    """Require every companion inference export to match its stage's optimizer snapshot exactly."""
    for index,b in enumerate(births):
        body,rule=model_io.load_checkpoint(location,"odor",arm,b,state["step"])
        for name in ("weights","bias","present"):
            torch.testing.assert_close(getattr(body,name),state["graph"][name][index:index+1],rtol=0,atol=0)
        for name,value in rule.state_dict().items():
            torch.testing.assert_close(value,state["rule"][name][index:index+1],rtol=0,atol=0)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def run():
    unique={}
    verified={}
    def check(report,entries,resolve):
        for k,expected in entries.items():
            p=resolve(k)
            actual=unique.get(p)
            if actual is None:
                actual=sha256(p.read_bytes()).hexdigest()
                unique[p]=actual
            if actual!=expected:
                raise ValueError(f"{report}: checksum mismatch at {p}")
        verified[report]=len(entries)
    trajectories={}
    for c in ("original","independent"):
        key="original" if c=="original" else "confirmation"
        births=list(range(731,737)) if c=="original" else list(range(811,817))
        nc=f"model-cards-{key}"
        check(nc,read(EVIDENCE/f"{nc}.json")["hashes"],lambda k:DATA/"model-cards"/key/k)
        tr=f"triangle-transfer-{key}"
        check(tr,read(EVIDENCE/f"{tr}.json")["raw_sha256"],
              lambda k:DATA/k if "/" in k else DATA/"triangle-transfer"/key/k)
        for stage in (32000,64000):
            nm=f"triangle-orders-long-{c}-{stage}"
            check(nm,read(EVIDENCE/f"{nm}.json")["sha256"],
                  lambda k:DATA/"long-order-verification/order-validation"/("long-"+c)/k)
            nm=f"triangle-components-{c}-{stage}"
            check(nm,read(EVIDENCE/f"{nm}.json")["sha256"],lambda k:DATA/"triangle-components"/c/k)
        for stage in (8000,32000,64000):
            nm=f"geometry-comparison-{c}-{stage}"
            archive=read(EVIDENCE/f"{nm}.json")
            check(nm,archive["raw_sha256"],lambda k:DATA/k)
            for arm in ("hard","orthogonal","interleaved"):
                for b in births:
                    packed=DATA/"geometry-comparison"/c/str(stage)/f"{arm}-{b}.npz"
                    with np.load(packed) as z:
                        if len(str(z["checkpoint_sha256"]))!=64 or len(str(z["evaluator_sha256"]))!=64:
                            raise ValueError("Missing bound source identity")
                        for shape in ("hard","orthogonal","random"):
                            side=z[shape+"/side"]
                            truth=z[shape+"/law"]
                            achieved=(side==truth[None,...,None]).all(3).mean((0,2,3))
                            np.testing.assert_array_equal(achieved,z[shape][:,3])
        for s in (0,4000,8000):
            nm=f"cross-skill-{c}-{s}"
            archive=read(EVIDENCE/f"{nm}.json")
            check(nm,archive["hashes"],lambda k:DATA/k if "/" in k else DATA/"cross-skill-eval"/c/str(s)/k)
            for a in (("parent",) if s==0 else ("old_then_new","new_then_old","alternating")):
                for b in births:
                    packed=DATA/"cross-skill-eval"/c/str(s)/f"{a}-{b}.npz"
                    with np.load(packed) as z:
                        if len(str(z["checkpoint_sha256"]))!=64:
                            raise ValueError("Unbound six-item probe")
                        for order in ("triangle_first","context_first","interspersed"):
                            truth=z["truth"]
                            uniform=z[order+"/uniform/side"]
                            np.testing.assert_array_equal((uniform==truth).all(2).sum(1),np.ones(len(truth)))
        for arm in ("orthogonal","interleaved"):
            for group in ("low","high"):
                d=torch.load(DATA/"geometry-curriculum"/c/arm/group/"resume.pt",weights_only=True,map_location="cpu")
                expected={"hard":32000,"random":16000,"orthogonal":16000} if arm=="interleaved" else {
                    "hard":0,"random":0,"orthogonal":64000}
                if d["step"]!=64000 or d["counts"]!=expected:raise ValueError("Geometry budget mismatch")
                check_exports(d,DATA/"geometry-curriculum"/c,arm,births[:3] if group=="low" else births[3:])
                traj=f"geometry/{c}/{arm}/{group}"
                steps=sorted({int(s["step"]) for s in d["optimizer"]["state"].values()})
                trajectories[traj]=dict(step=d["step"],outer_updates_by_geometry=d["counts"],adam_steps=steps)
        for arm in ("old_then_new","new_then_old","alternating"):
            d=torch.load(DATA/"cross-skill"/c/arm/"resume.pt",weights_only=True,map_location="cpu")
            if (d["step"],d["old"],d["new"])!=(8000,4000,4000):raise ValueError("Transfer budget mismatch")
            check_exports(d,DATA/"cross-skill"/c,arm,births)
            counts=sorted({int(s["step"]) for s in d["optimizer"]["state"].values()})
            trajectories[f"cross-skill/{c}/{arm}"]=dict(step=d["step"],old=d["old"],new=d["new"],adam_steps=counts)
        for stage,split in ((4000,"shared"),(8000,"shared"),(8000,"disjoint")):
            name=f"one-life-{c}-{split}-{stage}"
            table=read(EVIDENCE/f"{name}.json")
            check(name,table["hashes"],lambda k:DATA/k)
            for arm in ("parent","blocked","interspersed"):
                at=64000 if arm=="parent" else stage
                for b in births:
                    f=DATA/"one-life-eval"/c/split/f"{arm}-{b}-{at}.npz"
                    with np.load(f) as z:
                        for order in ("blocked","interspersed"):
                            gained=z["reward/"+order+"/paid"][...,1]
                            success=(gained>0).all(2).mean(1)
                            np.testing.assert_array_equal(success,z["reward/"+order+"/active/all_maps"][:,0])
        for arm in ("blocked","interspersed"):
            d=torch.load(DATA/"one-life"/c/arm/"resume.pt",weights_only=True,map_location="cpu")
            if d["step"]!=8000:raise ValueError("Same-life development budget mismatch")
            check_exports(d,DATA/"one-life"/c,arm,births)
            counts=sorted({int(s["step"]) for s in d["optimizer"]["state"].values()})
            trajectories[f"one-life/{c}/{arm}"]=dict(step=d["step"],adam_steps=counts)
        name=f"one-life-repair-{c}-2000"
        result=read(EVIDENCE/(name+".json"))
        check(name,result["hashes"],lambda k:DATA/k)
        name=f"one-life-repair-confirmation-{c}-2000"
        confirmation=read(EVIDENCE/(name+".json"))
        if (confirmation["ordinary_seed"]==result["ordinary_seed"]
                or confirmation["joint_seed"]==result["joint_seed"]):
            raise ValueError("The independent repair evaluation reused a prior world bank")
        check(name,confirmation["hashes"],lambda k:DATA/k)
        for arm in ("long_only","mixed"):
            d=torch.load(DATA/"one-life-repair"/c/arm/"resume.pt",weights_only=True,map_location="cpu")
            credits=d["trace"][:,:,3].sum(0)
            expected=1000 if arm=="mixed" else 0
            if d["step"]!=2000 or not torch.all(credits==expected):
                raise ValueError("The number of paid short and long training tickets is inconsistent")
            check_exports(d,DATA/"one-life-repair"/c,arm,births)
            trajectories[f"one-life-repair/{c}/{arm}"]=dict(step=2000,short_credits=expected,
                                                              long_credits=2000-expected)
    check("adaptive_gains",read(EVIDENCE/"adaptive-reference.json")["raw_sha256"],lambda k:DATA/"adaptive-reference"/k)
    check("constant_gain_transfer",read(EVIDENCE/"adaptive-deployment.json")["raw_sha256"],
          lambda k:DATA/"adaptive-reference/deployment"/k)
    check("compact_birth_reference",read(EVIDENCE/"compact-reference.json")["raw_sha256"],lambda k:DATA/"compact-reference"/k)
    for name in ("quantized-reference","dynamic-quantized-reference"):
        archive=read(EVIDENCE/(name+".json"))
        check(name,archive["raw_sha256"],lambda k:DATA/name/k)
    check("one-life-body-rule-swap",read(EVIDENCE/"one-life-body-rule-swap.json")["raw_sha256"],
          lambda k:DATA/"one-life-swaps"/k)
    for cohort,part in read(EVIDENCE/"one-life-padding.json")["cohorts"].items():
        check("one-life-padding/"+cohort,part["raw_sha256"],lambda k:DATA/"one-life-padding"/k)
    physical_inventory=DATA/"physical-replay-inventory-final.json"
    physical=read(physical_inventory)
    if (physical["completed_total"],physical["remaining_total"])!=(132,0):
        raise ValueError("The registered secondary physical suites are incomplete")
    check("completed_physical_summaries",physical["main_artifact_sha256"],lambda k:EVIDENCE/k)
    modules=Path(__file__).resolve().parents[1]
    source_files={f.name:sha256(f.read_bytes()).hexdigest() for f in modules.glob("interleave_*.py")}
    source_files.update({f.relative_to(modules).as_posix():sha256(f.read_bytes()).hexdigest()
                         for f in (modules/"reporting").glob("interleaved_*.py")})
    for name in ("self_param_writer.py","local_trace_graph.py","local_trace_train.py","local_trace_eval.py",
                 "edit_receipt_graph.py","edit_receipt_train.py","revisit_choice_graph.py","world.py"):
        source_files[name]=sha256((modules/name).read_bytes()).hexdigest()
    saved_sources={}
    for module,pair in read(EVIDENCE/"checkpoint-publication.json")["transitions"].items():
        archived=DATA/"training-sources"/f"{module}-{pair[0][:12]}.py"
        if sha256(archived.read_bytes()).hexdigest()!=pair[0] or source_files[module+".py"]!=pair[1]:
            raise ValueError(f"Missing as-run or corrected publication source for {module}")
        saved_sources[module]=dict(as_run=pair[0],after_checkpoint_publication_fix=pair[1],
                                   archive=archived.relative_to(DATA).as_posix())
    test_log=DATA/"sealed-tests.log"
    last=test_log.read_text(encoding="utf8").strip().splitlines()[-1]
    outcome=re.fullmatch(r"(\d+) passed(?:, (\d+) warnings?)? in .*",last)
    if not outcome:raise ValueError("The completed pulse-game test suite is missing a passing summary")
    report=dict(clean=True,checked_unique_files=len(unique),checked_bytes=sum(p.stat().st_size for p in unique),
                reports=verified,trained_trajectories=trajectories,
                installed=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__),
                final_source_sha256=source_files,archived_training_sources=saved_sources,
                physical_suites=dict(complete=physical["completed_total"],pending=physical["remaining_total"],
                                     inventory_sha256=sha256(physical_inventory.read_bytes()).hexdigest()),
                tests=dict(passed=int(outcome.group(1)),warnings=int(outcome.group(2) or 0),
                           log_sha256=sha256(test_log.read_bytes()).hexdigest()),
                notes="Checks the selected main model-card, triangle, three-geometry and six-memory studies, "
                      "plus fixed/learned stability studies. The narrow-scope count does not include every file "
                      "produced overnight. Recomputes ordinary three-odor sampled correctness from raw actions "
                      "and the exact all-six uniform law crossing. Other dedicated evidence indices independently "
                      "check field-specific raw data.")
    (EVIDENCE/"final-audit.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")
    print(report["checked_unique_files"],"distinct files verified",flush=True)


if __name__=="__main__":
    run()
