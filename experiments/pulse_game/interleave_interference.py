"""Fork two learned objects by geometric overlap, write direction and a real review."""

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import local_trace_graph as graph
from . import local_trace_train as old_fit


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/interleaved-learning-20261120"
EVIDENCE = ROOT / "docs/evidence/interleaved-learning-2026-11-20"
ANGLES = (0, 15, 30, 60, 90)
BRANCHES = ("before_b", "idle_b", "sham_b", "full_b", "undo_opposing", "undo_matched",
            "undo_all_surviving_weights", "review_a", "expose_a", "review_b")


@dataclass
class World:
    a: torch.Tensor
    b: torch.Tensor
    law_a: torch.Tensor
    law_b: torch.Tensor
    orders: torch.Tensor
    draws: torch.Tensor
    shuffle: np.ndarray


def worlds(seed, birth, count, angle):
    rng = np.random.default_rng([9208171, seed, birth])
    a = rng.normal(size=(count, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    tangent = rng.normal(size=a.shape)
    tangent -= (a * tangent).sum(1, keepdims=True) * a
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    b = a * np.cos(np.deg2rad(angle)) + tangent * np.sin(np.deg2rad(angle))
    def shared(v):
        return torch.as_tensor(np.repeat(v, 4, axis=0), dtype=torch.float32)
    return World(shared(1.35 * a), shared(1.35 * b), torch.tensor([-1.,-1.,1.,1.]).repeat(count),
                 torch.tensor([-1.,1.,-1.,1.]).repeat(count), shared(rng.choice((-1,1), (count,5))),
                 shared(rng.random((count,5,5), dtype=np.float32)),
                 np.repeat(rng.random((count,graph.NODES**2)), 4, axis=0))


def experience(rule, state, vector, law, world, t):
    right = vector * world.orders[:, t, None]
    u = world.draws[:, t]
    chosen = graph.act(rule, state, -right, right, u[:, 0])
    taste = chosen.side * world.orders[:, t] * law
    return graph.observe(rule, chosen, -right, right, taste)


def commits(rule, state, vector, law, world, t, mode="active"):
    obs = experience(rule, state, vector, law, world, t)
    return graph.commit(rule, obs, world.draws[:, t, 1:], intervention=mode)


def probe(rule, state, world):
    sides, margins, correct = [], [], []
    for vector, law, t in ((world.a, world.law_a, 3), (world.b, world.law_b, 4)):
        right = vector * world.orders[:, t, None]
        action = graph.act(rule, state, -right, right, world.draws[:, t, 0])
        goal = world.orders[:, t] * law
        sides.append(action.side.to(torch.int8))
        correct.append(action.side == goal)
        margins.append(action.motor_logit * goal)
    return torch.stack(sides,1), torch.stack(correct,1), torch.stack(margins,1)


def stratified_mask(delta, opposing, priorities):
    """Match the number removed within B-update sign and magnitude strata in every life."""
    d = delta.flatten(1).numpy()
    wanted = opposing.flatten(1).numpy()
    match = np.zeros(d.shape, dtype=bool)
    for lane, row in enumerate(d):
        for sign in (-1,1):
            candidates = np.flatnonzero(row * sign > 0)
            ranked = candidates[np.argsort(np.abs(row[candidates]))]
            for strata in np.array_split(ranked, 8):
                k = int(wanted[lane, strata].sum())
                if k:
                    draw = np.argsort(priorities[lane, strata])[:k]
                    match[lane, strata[draw]] = True
    out = torch.from_numpy(match).reshape_as(opposing)
    if not torch.equal(opposing.flatten(1).sum(1), out.flatten(1).sum(1)):
        raise AssertionError("Unmatched per-life rollback count")
    return out


@torch.no_grad()
def evaluate(g, rule, world):
    state = graph.start(g, len(world.a))
    rules = rule.expand(len(world.a))
    done_a = commits(rules, state, world.a, world.law_a, world, 0)
    a = graph.wait(rules, done_a.state, 4)
    exp_b = experience(rules, a, world.b, world.law_b, world, 1)
    u = world.draws[:, 1, 1:]
    b = graph.commit(rules, exp_b, u)
    blank = graph.commit(rules, exp_b, u, intervention="sham")
    g_a = a.graph
    g_ab = b.state.graph
    delta_a = done_a.receipt.weight_delta
    delta_b = b.receipt.weight_delta
    opposing = (delta_a * delta_b < 0)
    reference = stratified_mask(delta_b, opposing, world.shuffle)
    ready = graph.wait(rules, exp_b.state, graph.PRE_WRITE_TICKS)
    def undo(mask):
        wg = torch.where(mask, g_a.weights, g_ab.weights)
        mixed = replace(g_ab, weights=wg)
        updated = graph.wait(rules, replace(ready,graph=mixed), graph.POST_WRITE_TICKS)
        return graph.wait(rules, updated, 4)
    full = graph.wait(rules, b.state, 4)
    after_b = {
        "before_b": a,
        "idle_b": graph.wait(rules, a, 21),
        "sham_b": graph.wait(rules, blank.state, 4),
        "full_b": full,
        "undo_opposing": undo(opposing),
        "undo_matched": undo(reference),
        "undo_all_surviving_weights": undo(delta_b != 0),
    }
    # The fourth arm shares its entire prefix before review; subsequent probes are read-only forks.
    for branch, vector, law, mode in (("review_a",world.a,world.law_a,"active"),
                                      ("expose_a",world.a,world.law_a,"sham"),
                                      ("review_b",world.b,world.law_b,"active")):
        reviewed = commits(rules, full, vector, law, world, 2, mode)
        after_b[branch] = graph.wait(rules, reviewed.state, 4)
    if not torch.equal(undo(torch.zeros_like(opposing)).hidden, full.hidden):
        raise AssertionError("B numerical replay has different elapsed state")
    data = [probe(rules, after_b[branch], world) for branch in BRANCHES]
    flat_a, flat_b = delta_a.flatten(1), delta_b.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=1)
    return dict(side=torch.stack([x[0] for x in data]).numpy(),
                correct=torch.stack([x[1] for x in data]).numpy(),
                margin=torch.stack([x[2] for x in data]).numpy(),
                law_a=world.law_a.numpy(), law_b=world.law_b.numpy(), orders=world.orders.numpy(),
                edit_cosine=cosine.numpy(), opposing_n=opposing.flatten(1).sum(1).numpy(),
                opposing_l1=(delta_b.abs()*opposing).flatten(1).sum(1).numpy(),
                matched_l1=(delta_b.abs()*reference).flatten(1).sum(1).numpy(),
                b_weight_l1=flat_b.abs().sum(1).numpy())


def run(label, load, births, count=128, seed=16503):
    torch.set_num_threads(1)
    raw_dir = DATA / "interference" / label
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary = {"births":births, "per_birth_geometries": count, "seed":seed, "branches":BRANCHES,
               "input_note":"A/B are antipodal within a pair; angles refer to their two canonical positive directions. "
               "Each geometry crosses all four independent A/B label assignments. Identical or near-identical "
               "pairs with opposing laws are an identifiability boundary, not evidence of a solvable memory failure.",
               "results":{}, "raw_sha256":{}}
    for angle in ANGLES:
        all_rows = {"agree":[], "conflict":[]}
        for birth in births:
            g, rule = load(birth)
            w = worlds(seed, birth, count, angle)
            row = evaluate(g, rule, w)
            file = raw_dir / f"{birth}-{angle}.npz"
            np.savez_compressed(file, **row)
            summary["raw_sha256"][file.name] = sha256(file.read_bytes()).hexdigest()
            for name, same in (("agree",True),("conflict",False)):
                mask = (row["law_a"] == row["law_b"]) == same
                entry = {}
                for ix, branch in enumerate(BRANCHES):
                    good = row["correct"][ix,mask]
                    p = 1/(1+np.exp(-row["margin"][ix,mask]))
                    entry[branch] = dict(a=float(good[:,0].mean()), b=float(good[:,1].mean()),
                                         both=float(good.all(1).mean()), pa=float(p[:,0].mean()),
                                         pb=float(p[:,1].mean()))
                entry["write"] = {k:float(row[k][mask].mean()) for k in
                                  ("edit_cosine","opposing_n","opposing_l1","matched_l1","b_weight_l1")}
                all_rows[name].append(entry)
        summary["results"][angle] = all_rows
        print(label, angle, "completed", flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE / f"interference-{label}.json").write_text(json.dumps(summary,indent=1)+"\n",encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="prior-odor")
    p.add_argument("--pairs", type=int, default=128)
    p.add_argument("--births",type=int,nargs="+",default=list(range(731,737)))
    p.add_argument("--seed",type=int,default=16503)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--joint",action="store_true")
    args=p.parse_args()
    if args.joint:
        from . import interleave_train as training
        def loader(birth):
            return training.load_checkpoint(args.run,"local_active",birth,args.step)
    else:
        def loader(birth):
            return old_fit.load_checkpoint(args.run,"odor","active",birth,args.step)
    run(args.label,loader,args.births,args.pairs,args.seed)
