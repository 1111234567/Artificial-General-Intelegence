"""Can the graph infer a feeder mark's value from its own resource receipts?"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from . import revisit_choice_graph as policy
from . import self_param_writer as fw
from .edit_receipt_train import write_arrays


@dataclass
class Scene:
    book: torch.Tensor
    first_mark: torch.Tensor
    right_mark: torch.Tensor
    other_book: torch.Tensor
    uniforms: torch.Tensor


def scenes(seed, count, *, balanced):
    rng = np.random.default_rng(seed)
    if balanced:
        # All sign assignments at each independent random tape; twin worlds differ
        # in the hidden resource law while their motor/edit randomness stays identical.
        signs = np.tile(np.array(list(product((-1., 1.), repeat=4))), (count, 1))
        draws = np.repeat(rng.random((count, 5, 6), dtype=np.float32), 16, axis=0)
    else:
        signs = rng.choice((-1., 1.), (count, 4))
        draws = rng.random((count, 5, 6), dtype=np.float32)
    return Scene(*(torch.from_numpy(v.copy()).float() for v in signs.T), torch.from_numpy(draws))


def marks(sign):
    return torch.stack((1.35 * sign, torch.zeros_like(sign), torch.zeros_like(sign)), 1)


def expand(graph, count):
    return policy.State(fw.Graph(*(getattr(graph, k).repeat_interleave(count, 0)
                                   for k in ("weights", "bias", "present"))),
                        torch.zeros_like(graph.bias).repeat_interleave(count, 0))


def interact(graph, scene, count, *, fixed=False, treatment="own", history="fresh"):
    """Both sides demonstrate the same mark before the graph makes a valued choice."""
    state = expand(graph, count)
    start_graph = state.graph
    book = torch.ones_like(scene.book) if fixed else scene.book
    old_law = -book if history == "switched" else book
    episodes = [] if history == "fresh" else [(scene.first_mark, old_law),
                                               (-scene.first_mark, old_law)]
    episodes.extend(((scene.first_mark, book), (-scene.first_mark, book)))
    scores = []
    for t, (mark, law) in enumerate(episodes):
        icon = marks(mark)
        u = scene.uniforms[:, t]
        action = policy.act(state, icon, icon, u[:, 0])
        # This demonstration delivers the actual resource outcome independently of
        # side, since both available feeders carry the same mark.
        received = mark * law
        seen = received
        if treatment == "blank":
            seen = torch.zeros_like(received)
        elif treatment == "yoked":
            seen = mark * scene.other_book
        experience = policy.taste(action, icon, icon, seen)
        candidates = policy.propose(experience, u[:, 1:5].reshape(-1, 2, 2))
        selected = policy.choose(candidates, u[:, 5])
        result = policy.commit(experience, candidates, selected,
                               intervention="disabled" if treatment == "disable" else "active")
        state = result.state
        scores.append(action.motor_log_prob + candidates.address_log_prob.sum(1)
                      + candidates.op_log_prob.sum(1) + selected.log_prob)
    if treatment == "restore_graph":
        state = policy.State(start_graph, state.hidden)
    if treatment == "zero_h":
        state = policy.State(state.graph, torch.zeros_like(state.hidden))
    right = marks(scene.right_mark)
    action = policy.act(state, -right, right, scene.uniforms[:, -1, 0])
    earned = action.side * scene.right_mark * book
    return action, earned, torch.stack(scores, 1).sum(1)


def fit_group(args):
    root, births, arm, updates, batch, stages = args
    torch.set_num_threads(1)
    base = [fw.spawn(s, 8) for s in births]
    w, b = [torch.nn.Parameter(torch.cat([getattr(g, k) for g in base])) for k in ("weights", "bias")]
    graph = fw.Graph(w, b, torch.cat([g.present for g in base]))
    opt = torch.optim.Adam((w, b), lr=.003)
    def save(step):
        for i, seed in enumerate(births):
            write_arrays(root / arm / f"{seed}-{step}.npz", w=w[i:i+1].detach().numpy(),
                         b=b[i:i+1].detach().numpy(), m=graph.present[i:i+1].numpy())
    save(0)
    control = arm == "fixed"
    track = []
    for step in range(1, updates + 1):
        tapes = [scenes([518911, seed, step], batch, balanced=False) for seed in births]
        scene = Scene(*(torch.cat([getattr(s, name) for s in tapes]) for name in Scene.__annotations__))
        action, earned, prefix_score = interact(graph, scene, batch, fixed=control)
        r = earned.detach().reshape(-1, batch)
        adv = (r - r.mean(1, keepdim=True)) * (batch / (batch - 1))
        # Opposite terminal outcomes on the two sides permit exact averaging of
        # this last action, using the realized side and earned receipt after it acts.
        expected = (2 * action.motor_logit.sigmoid() - 1) * action.side * earned
        loss = -(expected.reshape(-1, batch).mean(1) +
                 (prefix_score.reshape(-1, batch) * adv).mean(1)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            norm = (w.grad.flatten(1).square().sum(1) + b.grad.square().sum(1)).sqrt()
            if not torch.isfinite(norm).all():
                raise ValueError("Nonfinite gradient")
            factor = (2 / norm.clamp_min(1e-10)).clamp_max(1)
            w.grad.mul_(factor[:, None, None])
            b.grad.mul_(factor[:, None])
        opt.step()
        track.append((r.mean(1).numpy() + 1) / 2)
        if step in stages:
            save(step)
            print(arm, births, step, np.mean(track[-100:], axis=0).round(3), flush=True)
    write_arrays(root / arm / f"trace-{births[0]}.npz", births=births, accuracy=np.array(track))


@torch.no_grad()
def evaluate(birth, arm, graph, root, step, pairs):
    torch.set_num_threads(1)
    scene = scenes([795821, birth], pairs, balanced=True)
    raw = {"book": scene.book.numpy(), "first_mark": scene.first_mark.numpy(),
           "right_mark": scene.right_mark.numpy(), "other_book": scene.other_book.numpy()}
    for history in ("fresh", "stable", "switched"):
        treatments = ("own", "blank", "yoked", "disable", "restore_graph", "zero_h") if history == "fresh" else ("own",)
        for treatment in treatments:
            a, r, _ = interact(graph, scene, len(scene.book), treatment=treatment, history=history)
            value = (r > 0).numpy()
            if treatment in ("blank", "yoked") and value.mean() != .5:
                raise AssertionError("Unobserved codebook leaked to the policy")
            key = history + "/" + treatment
            raw[key] = value
            raw[key + "/side"] = a.side.numpy().astype(np.int8)
    write_arrays(root / "evaluation" / arm / f"{birth}-{step}.npz", **raw)
    metrics = {}
    for k, v in raw.items():
        if v.dtype != np.bool_:
            continue
        metrics[k] = float(v.mean())
        if k == "fresh/own":
            for book in (-1, 1):
                metrics[f"fresh/own/book{book}"] = float(v[raw["book"] == book].mean())
    return metrics


def summary(root, births, arms, stages, pairs, loaders):
    result = {}
    for step in stages:
        result[step] = {}
        for arm in arms:
            rows = [evaluate(s, arm, loaders(arm, s, step), root, step, pairs) for s in births]
            values = {k: [r[k] for r in rows] for k in rows[0]}
            result[step][arm] = values
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/revisit-choice-20261119/value"))
    ap.add_argument("--births", type=int, nargs="+", default=list(range(731, 737)))
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=("train", "evaluate", "main-probe"), required=True)
    ap.add_argument("--main-run", type=Path)
    a = ap.parse_args()
    root, births = a.out, a.births
    root.mkdir(parents=True, exist_ok=True)
    stages = sorted({0, min(a.updates, 500), min(a.updates, 2000), a.updates})
    arms = ("fixed", "variable")
    code_hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in
                   map(Path, (policy.__file__, fw.__file__, policy.receipt_graph.__file__, __file__))}
    if a.mode == "evaluate":
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        if (protocol["code_hashes"] != code_hashes or protocol["births"] != births
                or protocol["steps"] != stages or protocol["pairs"] != a.pairs):
            raise ValueError("Evaluation differs from the frozen value-study protocol")
    if a.mode == "train":
        path = root / "protocol.json"
        info = dict(births=births, arms=arms, steps=stages, batch=a.batch, pairs=a.pairs,
                    code_hashes=code_hashes,
                    design="Each life sees both marked feeders separately with opposing unit resource changes, then chooses between marks. Terminal utility is externally defined. Fixed law or independent +/- law per life. Order, test side and random draws are balanced on evaluation.")
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        jobs = [(root,births[i:i+3],arm,a.updates,a.batch,stages)
                for arm in arms for i in range(0,len(births),3)]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(fit_group, jobs))
    elif a.mode == "evaluate":
        def load(arm, birth, step):
            with np.load(root / arm / f"{birth}-{step}.npz") as z:
                return fw.Graph(*(torch.from_numpy(z[k]) for k in ("w","b","m")))
        result = summary(root, births, arms, stages, a.pairs, load)
        (root / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    else:
        from . import revisit_choice_train as training
        manifest = json.loads((a.main_run / "run.json").read_text(encoding="utf-8"))
        final = max(manifest["checkpoints"])
        def load(arm, birth, step):
            return training.load_checkpoint(a.main_run, arm, birth, step)
        result = summary(root, births, ("ranked",), [final], a.pairs, load)
        (root / "main-probe.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
