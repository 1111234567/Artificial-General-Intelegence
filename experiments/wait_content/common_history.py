"""Read three actual policies on the untrained collector's identical future histories."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from experiments.wait_content import train
from experiments.wait_content.audit import ARMS, BASE, sha


OUT = BASE/"common-v1"


def load_trace(artifact):
    path = Path(artifact["path"])
    assert sha(path) == artifact["sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def reference_cases(trace):
    def field(k):
        return np.asarray([f[k] for f in trace])[:, 4:8]
    active = field("active") & field("primary_current")
    processing = active & (field("phase")==2)
    labels = field("teacher_actions")
    assert (labels[processing] == 0).all()
    finish = active & (labels>=3)
    first = np.zeros_like(finish)
    ticks, lanes, jobs = np.where(finish)
    identities = np.column_stack((lanes, jobs, field("request_id")[ticks,lanes,jobs]))
    _, where = np.unique(identities, axis=0, return_index=True)
    first[ticks[where], lanes[where], jobs[where]] = True
    return {"processing":processing, "first_teacher_finish":first}, field("target"), field("idle")


@torch.inference_mode()
def run(seed):
    torch.set_num_threads(1)
    folder = BASE/f"eval-v1/seed_{seed:03d}"
    original = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    controllers, initials = {}, {}
    for arm in ARMS:
        if arm == "untrained_anchor":
            controllers[arm], _ = train.load_initial(seed)
        else:
            controllers[arm], _ = train.load_trained(seed,arm)
        warm = torch.load(folder/f"{arm}_warmup.pt", map_location="cpu", weights_only=False)
        initials[arm] = warm["h"]
    records = []
    for baseline in original["records"]:
        if baseline["arm"] != "untrained_anchor":
            continue
        trace = load_trace(baseline["trace_artifact"])
        cases, target, idle = reference_cases(trace)
        original_action = np.array([v["actions"] for v in trace])
        original_choice = np.array([v["conditional_finish_choice"] for v in trace])
        artifact = baseline["observations_artifact"]
        assert sha(artifact["path"]) == artifact["sha256"]
        with np.load(artifact["path"]) as data:
            obs = torch.from_numpy(data["obs"]).double()
            elapsed = data["elapsed"]
        for arm in ARMS:
            h = initials[arm].clone()
            all_scores = []
            for o, dt in zip(obs,elapsed,strict=True):
                scores, h = controllers[arm].step(o,h,float(dt))
                all_scores.append(scores)
            scores = torch.stack(all_scores).numpy()
            act = scores.argmax(-1)
            choice = scores[...,3:].argmax(-1)
            if arm == "untrained_anchor":
                assert np.array_equal(act,original_action) and np.array_equal(choice,original_choice)
            act, choice = act[:,4:8], choice[:,4:8]
            counts = dict(idle_opportunities=int(idle.sum()), idle_actions=int((idle & (act!=0)).sum()))
            for phase, mask in cases.items():
                counts[phase+"_n"] = int(mask.sum())
                counts[phase+"_content_correct"] = int((mask & (choice==target)).sum())
                counts[phase+"_want_finish"] = int((mask & (act>=3)).sum())
                counts[phase+"_want_correct_finish"] = int((mask & (act==target+3)).sum())
            records.append(dict(seed=seed,arm=arm,condition=baseline["condition"],family=baseline["family"],counts=counts))
    pooled = defaultdict(Counter)
    for r in records:
        pooled[r["arm"],r["condition"]].update(r["counts"])
    result = dict(seed=seed, reference_result_sha256=sha(folder/"result.json"), records=records,
        policy_receipts="actual untrained reference actions only; alternative scores are measurements",
        totals=[dict(arm=a,condition=c,counts=v) for (a,c),v in pooled.items()])
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/f"seed_{seed:03d}.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",required=True,type=int,choices=range(3))
    run(parser.parse_args().seed)
