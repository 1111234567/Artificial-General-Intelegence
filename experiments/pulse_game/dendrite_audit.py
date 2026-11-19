"""External tests of two-compartment delayed spiking graphs with equal genetic budgets."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import attractor_audit as probes, attractor_search as base, dendrite_search as design, research_audit as audit
from experiments.seed_evolution.large_runner import _write


def reference(vector, net, task, mode, cut=False):
    inp, offsets, src, weights, project, gene, _ = design.birth(vector, net, mode)
    leak, syn_decay, read_decay, adapt_decay, adapt, mean_rate, retention, gain, add, rec_d = gene
    out = np.empty((*task.goal.shape, base.K), np.float32)
    spike_count = 0
    for life in range(len(task.seeds)):
        h = np.zeros(base.N, np.float32)
        trace, slow, syn, fatigue = (h.copy() for _ in range(4))
        mean = np.full_like(h, .12)
        for j in range(task.cue.shape[1]):
            c, ctx = int(task.cue[life,j]), int(task.ctx[life,j])
            raw = int(task.raw_at[life,j])
            d = int(task.distractor[life,j])
            for tick in range(task.length[life,j]):
                rec = np.zeros_like(h)
                if not cut:
                    for k in range(base.D):
                        rec += weights[:,k] * syn[src[:,k]]
                pulse = np.zeros((base.N, 2), np.float32)
                for port, start in ((0,0), (1,7)):
                    if tick == task.issue[life,j]:
                        pulse[:,port] += inp[:,start+ctx] + (inp[:,start+2+c] if raw == 255 else 0.) + inp[:,start+6]
                    if tick == raw:
                        pulse[:,port] += inp[:,start+2+c]
                    if tick == task.at[life,j]:
                        pulse[:,port] += inp[:,start+d//4] + inp[:,start+2+d%4]
                earlier = np.tanh(slow)
                drive = offsets[:,0] + rec
                drive += add * earlier + pulse[:,0] * (1. + gain * earlier)
                slow = np.clip(retention * slow + pulse[:,1] + rec_d*rec, -12., 12.)
                h = np.maximum(-3., leak*h + drive)
                spike = (h >= offsets[:,1] + adapt*fatigue).astype(np.float32)
                h[spike > 0] = 0.
                fatigue = adapt_decay*fatigue + (1-adapt_decay)*spike
                trace = read_decay*trace + (1-read_decay)*spike
                mean += mean_rate*(trace-mean)
                syn = syn_decay*syn + (1-syn_decay)*spike
                spike_count += int(spike.sum())
                if tick == task.query[life,j]:
                    f = np.zeros(base.K,np.float32)
                    for k in range(base.N):
                        f += project[:,k] * (trace[k]-mean[k])
                    out[life,j] = f
    return out, spike_count/(base.physical_time(task)*base.N)


def parity(folder):
    rng = np.random.default_rng(309063)
    each = []
    for i, mode in enumerate(("basal","add","gain","both")):
        v = rng.uniform(.15,.70, len(design.RANGES))
        t = base.task([819,329,13], "serial_noise", trials=8)
        a, freq, z = design.run([v], 77, t, mode=mode, features=True)
        expected, ref_freq = reference(v, 77, t, mode)
        b = base.cpu_answers(v[:len(base.RANGES)], expected, t)
        each.append(dict(mode=mode, feature_error=float(np.max(np.abs(z[0]-expected))),
                         freq_error=float(abs(freq[0]-ref_freq)), action_equality=float((a[0]==b).mean())))
    _write(folder/"cpu-parity.json", each)
    if any(row["action_equality"]!=1. or row["feature_error"]>.00001 for row in each):
        raise ValueError(each)


def audit_model(info, seeds, graphs, folder):
    name = info.parent.name
    record = json.loads(info.read_text())
    choice = record["best"]
    if len(record["history"]) != 144:
        return False
    protocol = record["protocol"]
    v = choice["vector"]
    mode = protocol["mode"]
    for condition in ("short","long","serial","serial_noise","extrapolate"):
        task = base.task(seeds,condition)
        for birth in graphs:
            path = folder / f"{name}-g{birth}-{condition}.json"
            if path.is_file():
                continue
            variants = dict(native=(mode,False,False,True), cut=(mode,True,False,True),
                            no_outcome=(mode,False,False,False), rewired=(mode,False,True,True))
            if mode!="basal":
                variants["remove_dendrite"] = ("basal",False,False,True)
            if mode == "both":
                variants.update(add_only=("add",False,False,True), gain_only=("gain",False,False,True))
            rows = {}
            for tag, (change,cut,shuf,feed) in variants.items():
                ans, freq, _ = design.run([v],birth,task,mode=change,cut=cut,shuffle=shuf,feedback=feed)
                rows[tag] = {**probes.stats(task,ans[0]), "firing":float(freq[0])}
            _write(path,dict(study=name,mode=mode,search_seed=protocol["seed"],birth=birth,
                             condition=condition,seeds=seeds,rows=rows,selected_generation=choice["generation"],
                             model_vector=v,rule_source=audit.fingerprint(design.__file__)))
    print("checked",name,flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag",default="dendrite-v1")
    args=parser.parse_args()
    torch.set_num_threads(1)
    folder=audit.OUT/(args.tag+"-external")
    folder.mkdir(exist_ok=True)
    parity(folder)
    seeds = [int(np.random.SeedSequence([s,5490271]).generate_state(1)[0]) for s in audit.split("hard_task")[:128]]
    seen=set()
    while len(seen)<12:
        for path in sorted(base.OUT.glob(args.tag+"-*/checkpoint.json")):
            if path not in seen and audit_model(path,seeds,[13,41,71,103],folder):
                seen.add(path)
        if len(seen)<12:
            time.sleep(12)


if __name__=="__main__":
    main()
