"""Check the controlled observation sequence against the original public world."""
import json
from collections import Counter
import torch

from experiments.content_timing.matched_flow import change_command
from experiments.learn_remember import memory_assay as m


def check_world(seed):
    bg=torch.load(m.source(seed),map_location="cpu",weights_only=False)
    anchors,obs,meta=m.make_branches(bg)
    coverage={}
    for i,a in enumerate(anchors):
        coverage.setdefault((a["job"],a["version"]),i)
    checked=0
    with torch.no_grad():
        for index in coverage.values():
            a=anchors[index]
            lane,j=a["actor_lane"],a["job"]
            for family in range(4):
                for raw in range(4):
                    i=index*16+family*4+raw
                    w,cue,_=change_command(dict(a,family=family),raw,m.DURATION)
                    w.current_family[lane,j]=family
                    w.event_family[lane,j]=family
                    w._assigned_context[lane,j]=family
                    cue[lane,j*28+24:j*28+28]=0
                    cue[lane,j*28+24+family]=1
                    assert torch.equal(cue[lane],obs[0,i])
                    for t in range(meta[i]["teacher_tick"]+1):
                        if t:
                            visible=w.observe()
                            assert torch.equal(visible[lane].double(),obs[t,i])
                        target=int(w.oracle("midpoint")[lane,j])
                        if t==meta[i]["teacher_tick"]:
                            assert target==3+meta[i]["target"]
                        else:
                            assert target<3
                        action=bg["actions"][a["source_index"]+t].clone()
                        action[lane,j]=1+a["version"] if t==0 else 0
                        w.act(action)
                    checked+=1
    return dict(seed=seed,anchor_slot_version_strata=len(coverage),world_branches_checked=checked)


def check_results():
    results=[]
    for p in sorted(m.OUT.glob("*_s[012].json")):
        d=json.loads(p.read_text(encoding="utf-8"))
        stats=Counter()
        for r in d["rows"]:
            assert r["target"]==4*r["version"]+m.FAMILY_MAPPINGS[r["family"]][r["raw"]]
            stats.update(n=1, content_correct=int(r["original_content"]==r["target"]),
                wants_finish=int(r["original_action"]>=3), wants_correct_finish=int(r["original_action"]==3+r["target"]),
                swap_recipient_correct=int(r["swapped_content"]==r["target"]),
                swap_donor_correct=int(r["swapped_content"]==r["donor_target"]))
        assert stats==Counter(d["groups"]["all"])
        assert stats['swap_donor_correct']==stats['content_correct']
        assert d["max_donor_score_error"]<2e-5
        results.append(dict(file=p.name,sha256=m.sha(p),counts=stats))
    return results


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--world",action="store_true");args=p.parse_args()
    torch.set_num_threads(1)
    if args.world:
        path=m.OUT/"public-world-check.json"
        data=[check_world(s) for s in range(3)]
    else:
        path=m.OUT/"readout-check.json"
        data=check_results()
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(path)
