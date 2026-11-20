"""Audit and measure prefix sharing without changing a fit or its checkpoints."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from platform import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import interleave_triangle_prefix as prefix
from . import interleave_triangle_train as tri
from .interleave_confirm_triangle import TRAIN_SEED as INDEPENDENT_TRAIN_SEED, permuted_tape


COHORTS = {"original": ((731, 732, 733), tri.TRAIN_SEED),
           "independent": ((811, 812, 813), INDEPENDENT_TRAIN_SEED)}
SEED = 8044553
DRAW_COUNT = 10000


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def endpoint(cohort):
    roots, _ = COHORTS[cohort]
    path = tri.DATA / f"triangle-long-{cohort}" / ("group-" + "-".join(map(str, roots))) / "resume.pt"
    fingerprint = digest(path)
    state = torch.load(path, weights_only=True, map_location="cpu")
    if state["step"] != 64000 or state["births"] != list(roots) or digest(path) != fingerprint:
        raise ValueError(f"Need the exact archived 64k group at {path}")
    return state, {"path": path.relative_to(tri.DATA).as_posix(), "sha256": fingerprint}


def clone(state, *, precise=False):
    p = state["graph"]
    graph = tri.native.Graph(nn.Parameter(p["weights"].clone()), nn.Parameter(p["bias"].clone()), p["present"].clone())
    rule = tri.native.Rule(state["births"])
    rule.load_state_dict(state["rule"])
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=.003)
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    if precise:
        return tri.native.Graph(nn.Parameter(graph.weights.detach().double()), nn.Parameter(graph.bias.detach().double()),
                                graph.present), rule.double(), optimizer
    return graph, rule, optimizer


def tape_for(roots, seed, worlds, update, rho=-.5):
    rows = [tri.make_tape(seed, root, worlds, rho, update=update) for root in roots]
    return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}


def vector(tensors, *, gradients=False):
    return torch.cat([(torch.zeros_like(t) if t.grad is None else t.grad).detach().flatten()
                      if gradients else t.detach().flatten() for t in tensors]).clone().numpy()


def step_with_records(model, tape, worlds, accelerated):
    graph, rule, optimizer = model
    support, motor, grad = [], [], []
    original_contact, original_act, original_grad = tri._contact, tri.native.act, tri.revisit._gradients

    def contact(*args, **kwargs):
        a, edit, actual, taste = original_contact(*args, **kwargs)
        support.append(dict(logit=a.motor_logit.detach(), side=a.side.detach(), earned=actual.detach(),
                            taste=taste.detach(), birth=edit.receipt.birth.detach(), retired=edit.receipt.retired.detach()))
        return a, edit, actual, taste

    def act(*args, **kwargs):
        result = original_act(*args, **kwargs)
        motor.append((result.motor_logit.detach(), result.side.detach()))
        return result

    def gradients(parameters, limit):
        raw = vector(parameters, gradients=True)
        norm = original_grad(parameters, limit)
        grad.append((raw, vector(parameters, gradients=True)))
        return norm

    with patch.object(tri.revisit, "_gradients", gradients):
        if accelerated:
            measures = prefix.optimizer_step(graph, rule, optimizer, tape, worlds, allow_approximate=True, trace=support)
        else:
            with patch.object(tri, "_contact", contact), patch.object(tri.native, "act", act):
                measures = tri._step(graph, rule, optimizer, tape, worlds)
            logit, side = motor[-1]
            value = tape["law"].gather(1, tape["query"][:, None]).squeeze(1)
            support.append(dict(logit=logit, side=side, earned=side * tape["query_flip"] * value))
    records = support[-4:]
    assert len(records) == 4 and len(grad)
    packed = {key: torch.stack([r[key] for r in records if key in r]).numpy() for key in records[0]}
    packed.update(raw_gradient=grad[-1][0], clipped_gradient=grad[-1][1], measures=measures.detach().numpy())
    return packed


def distance(first, second):
    a, b = first.astype(float), second.astype(float)
    diff = a - b
    return dict(max_abs=float(np.abs(diff).max()), relative_l2=float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-15)),
                changed=int(np.count_nonzero(first != second)), total=first.size)


def same_state_gradient(state, seed):
    result = {}
    for label, precise in (("fp32", False), ("fp64", True)):
        rows = []
        for offset, rho in enumerate((-.5, 0., -.25, .25)):
            a, ar, _ = clone(state, precise=precise)
            b, br, _ = clone(state, precise=precise)
            tape = tape_for(state["births"], seed, 4, 64001 + offset, rho)
            loss_a, scores_a = tri.training_step(a, ar, tape, 4)
            loss_b, scores_b = prefix.training_step(b, br, tape, 4)
            loss_a.backward()
            loss_b.backward()
            rows.append(dict(rho=rho,
                             loss=distance(loss_a.detach().reshape(1).numpy(), loss_b.detach().reshape(1).numpy()),
                             metrics=distance(scores_a.numpy(), scores_b.numpy()),
                             gradient=distance(vector((a.weights, a.bias, *ar.parameters()), gradients=True),
                                               vector((b.weights, b.bias, *br.parameters()), gradients=True))))
        result[label] = rows
    return result


def trajectory(cohort, state, steps, raw_dir):
    base, compact = clone(state), clone(state)
    seed = COHORTS[cohort][1]
    raw, distances = {}, {}
    first = {}
    for update in range(64001, 64001 + steps):
        tape = tape_for(state["births"], seed, 4, update)
        old = step_with_records(base, tape, 4, False)
        new = step_with_records(compact, tape, 4, True)
        summaries = {name: distance(old[name], new[name]) for name in old}
        d1, d2 = base[2].state_dict()["state"], compact[2].state_dict()["state"]
        assert d1.keys() == d2.keys() and all(float(d1[k]["step"]) == float(d2[k]["step"]) == update for k in d1)
        for name in ("exp_avg", "exp_avg_sq"):
            a, b = (np.concatenate([d[key][name].detach().flatten().numpy() for key in d]) for d in (d1, d2))
            summaries[name] = distance(a, b)
        params = (vector((m[0].weights, m[0].bias, *m[1].parameters())) for m in (base, compact))
        summaries["parameters"] = distance(*params)
        for name, stat in summaries.items():
            if stat["changed"] and name not in first:
                first[name] = update
            distances.setdefault(name, []).append([stat[k] for k in ("max_abs", "relative_l2", "changed", "total")])
        for path, row in (("baseline", old), ("prefix", new)):
            for key in ("logit", "side", "earned", "taste", "birth", "retired", "measures"):
                raw.setdefault(f"{path}/{key}", []).append(row[key])
    array = {name: np.stack(rows) for name, rows in distances.items()}
    archive = raw_dir / f"trajectory-{cohort}.npz"
    tri.revisit._write_raw(archive, **{k: np.stack(v) for k, v in raw.items()},
                           **{f"difference/{name}": value for name, value in array.items()})
    result = {k: dict(first_changed_update=first.get(k), max_absolute_observed=float(column[:, 0].max()),
                      max_relative_l2_observed=float(column[:, 1].max()),
                      final_absolute=float(column[-1, 0]), final_relative_l2=float(column[-1, 1]),
                      differing_entries=int(column[:, 2].sum()), total_entries=int(column[:, 3].sum()))
              for k, column in array.items()}
    final = 64000 + steps
    print(cohort, final, "first differences", first, flush=True)
    return dict(births=state["births"], steps=steps, final_adam_step=final, raw_sha256=digest(archive), results=result)


def check_frozen():
    counts = {"archival_active": 0, "archival_controls_logits": 0, "paired_action_controls": 0, "archive_files": 0}
    for cohort, roots in (("original", range(731, 737)), ("independent", range(811, 817))):
        for stage in (32000, 64000):
            spec = tri.EVIDENCE / f"triangle-orders-long-{cohort}-{stage}.json"
            manifest = json.loads(spec.read_text(encoding="utf8"))
            folder = tri.DATA / "long-order-verification/order-validation" / f"long-{cohort}"
            for birth in roots:
                graph, rule = tri.old_fit.load_checkpoint(tri.DATA / f"triangle-long-{cohort}", "odor", "direct", birth, stage)
                archive = folder / f"{birth}-{stage}.npz"
                if digest(archive) != manifest["sha256"][archive.name]:
                    raise ValueError(f"Frozen archive mismatch at {archive}")
                tape = tri.make_tape(manifest["seed"], birth, manifest["worlds"], -.5)
                with np.load(archive, allow_pickle=False) as expected:
                    for j, order in enumerate(permutations(range(3))):
                        raw = prefix.trials(graph, rule, permuted_tape(tape, order), "active")
                        for key, old in (("logit", "active_logit"), ("selected", "active_selected")):
                            if raw[key].tobytes() != expected[f"{old}_{j}"].tobytes():
                                raise AssertionError(f"Archival {key} differs at {birth} {stage} {j}")
                        counts["archival_active"] += 1
                    for mode in tri.CONTROLS[1:]:
                        value = prefix.trials(graph, rule, tape, mode)["logit"]
                        if value.tobytes() != expected[f"{mode}_logit"].tobytes():
                            raise AssertionError(f"Archival {mode} logit differs at {birth} {stage}")
                        counts["archival_controls_logits"] += 1
                counts["archive_files"] += 1
                fresh = tri.make_tape(SEED, birth, 96 + 96 * (birth % 2), (-.45, -.25, 0., .25)[birth % 4])
                for mode in tri.CONTROLS:
                    baseline, compact = (call(graph, rule, fresh, mode) for call in (tri.trials, prefix.trials))
                    for key in baseline:
                        if baseline[key].dtype != compact[key].dtype or baseline[key].tobytes() != compact[key].tobytes():
                            raise AssertionError(f"Unarchived {mode} {key} differs for root {birth} stage {stage}")
                    counts["paired_action_controls"] += 1
        print(cohort, "frozen bitwise validation done", flush=True)
    return counts


def paired_timings(work, count, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.tile([0, 1], count // 2))
    elapsed = np.empty((count, 2), dtype=np.float64)
    for k, first in enumerate(order):
        for arm in (first, 1 - first):
            t = perf_counter()
            work(k, arm)
            elapsed[k, arm] = perf_counter() - t
    ratio = np.log(elapsed[:, 0] / elapsed[:, 1])
    draws = rng.integers(count, size=(DRAW_COUNT, count))
    lo, hi = np.exp(np.quantile(ratio[draws].mean(1), [.025, .975]))
    return dict(groups=count, baseline_first=int((order == 0).sum()), prefix_first=int((order == 1).sum()),
                geometric_speedup=float(np.exp(ratio.mean())), paired_ci95=[float(lo), float(hi)],
                baseline_seconds_p50=float(np.median(elapsed[:, 0])), prefix_seconds_p50=float(np.median(elapsed[:, 1]))), elapsed, order


def timing(state, groups, block):
    graph, rule, _ = clone(state)

    def train(i, arm):
        a, b, opt = clone(state)
        for i_update in range(block):
            update = 80000 + i * block + i_update
            tape = tape_for(state["births"], SEED + 47, 4, update)
            if arm:
                prefix.optimizer_step(a, b, opt, tape, 4, allow_approximate=True)
            else:
                tri._step(a, b, opt, tape, 4)

    birth = state["births"][0]
    root_graph = tri.native.Graph(graph.weights[:1].detach(), graph.bias[:1].detach(), graph.present[:1])
    root_rule = tri.native.Rule(birth)
    root_rule.load_state_dict({k: v[:1] for k, v in rule.state_dict().items()})

    def evaluate(i, arm):
        tape = tri.make_tape(SEED + i, birth, 128, -.5)
        run = prefix.trials if arm else tri.trials
        for mode in tri.CONTROLS:
            tri._analyze(tape, run(root_graph, root_rule, tape, mode))

    for arm in (0, 1):
        train(-1, arm)
        evaluate(-1, arm)
    train_result, train_raw, train_order = paired_timings(train, groups, SEED + 1)
    eval_result, eval_raw, eval_order = paired_timings(evaluate, groups, SEED + 2)
    return (dict(training=dict(updates_per_group=block, births=3, worlds_per_birth=4, **train_result),
                 frozen=dict(worlds=128, modes=list(tri.CONTROLS), **eval_result)),
            dict(train_seconds=train_raw, train_first=train_order, eval_seconds=eval_raw, eval_first=eval_order))


def run(raw, output, groups, block, steps):
    if groups < 20 or groups % 2 or min(block, steps) < 1:
        raise ValueError("Use an even number of at least twenty timing groups and positive rollout lengths")
    torch.set_num_threads(1)
    report = dict(torch_version=torch.__version__, numpy_version=np.__version__, platform=platform(), torch_threads=1,
                  seed=SEED, baseline_source_sha256=digest(Path(tri.__file__)),
                  prefix_source_sha256=digest(Path(prefix.__file__)), endpoints={}, trajectories={}, same_state_gradients={},
                  timing_note="End to end for fixed local workloads, with order balanced and randomized at the group level. "
                              "Training times include cloning graph and Adam state, making each tape, native lifetime, "
                              "loss, backward, clipping and Adam. Frozen times include tape generation, all four modes, "
                              "three independent two-orientation probes, and all summary statistics. Checkpoint disk I/O "
                              "and process startup are outside both timings.")
    clocks, samples = timing(endpoint("original")[0], groups, block)
    p = raw / "paired-timing.npz"
    tri.revisit._write_raw(p, **samples)
    report.update(benchmark=clocks, timing_sha256=digest(p))
    print("paired timings complete", clocks, flush=True)
    report["bitwise_evaluation"] = check_frozen()
    for cohort, (_, seed) in COHORTS.items():
        state, source = endpoint(cohort)
        report["endpoints"][cohort] = source
        report["same_state_gradients"][cohort] = same_state_gradient(state, seed)
        report["trajectories"][cohort] = trajectory(cohort, state, steps, raw)
    tri.revisit._write_json(output, report)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--updates-per-group", type=int, default=6)
    parser.add_argument("--trajectory-steps", type=int, default=512)
    parser.add_argument("--raw", type=Path, default=tri.DATA / "triangle-prefix-bench")
    parser.add_argument("--output", type=Path, default=tri.EVIDENCE / "triangle-prefix-bench.json")
    args = parser.parse_args()
    run(args.raw, args.output, args.groups, args.updates_per_group, args.trajectory_steps)
