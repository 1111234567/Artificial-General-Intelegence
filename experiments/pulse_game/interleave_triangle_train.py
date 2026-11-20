"""Train and test native memory on fully crossed three-odor assignments."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import batched, product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_binding as binding
from . import interleave_code_probe as probe
from . import interleave_revisit_train as revisit
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import local_trace_train as old_fit


ARMS = ("direct", "morph")
CONTROLS = ("active", "sham", "false_sign", "shuffled_objects")
LABELS = np.array(list(product((-1., 1.), repeat=3)), dtype=np.float32)
GRAM_LEVELS = (-.5, -.45, -.25, 0., .25)
TRAIN_SEED = 3783281
EVAL_SEED = 6244113
MARGIN_THRESHOLD = .05
METRICS = ("terminal_accuracy", "all_same_terminal_accuracy", "mixed_terminal_accuracy",
           "terminal_score_loss", "study_score_loss", "edit_score_loss", "gradient_norm", "float64_recovery")
DATA = revisit.ROOT / "data/interleaved-learning-20261120"
EVIDENCE = revisit.ROOT / "docs/evidence/interleaved-learning-2026-11-20"


def rotation_keys(rng, count, rho):
    """Sample proper rotations of three vectors whose normalized off-diagonal Gram entries are rho."""
    if not -.5 <= rho <= 1.:
        raise ValueError("The three-item equicorrelation Gram matrix must be positive semidefinite")
    rotation, triangular = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    rotation *= np.sign(np.diagonal(triangular, axis1=1, axis2=2))[:, None, :]
    rotation[:, :, -1] *= np.linalg.det(rotation)[:, None]
    projector = np.ones((3, 3)) / 3
    factor = np.sqrt(1 - rho) * (np.eye(3) - projector) + np.sqrt(1 + 2 * rho) * projector
    return (1.35 * (factor @ rotation)).astype(np.float32)


def make_tape(seed, birth, worlds, rho, *, update=None):
    domain = 432811 if update is None else 432810
    streams = np.random.SeedSequence([domain, seed, birth, 0 if update is None else update]).spawn(2)
    geom, process = (np.random.default_rng(stream) for stream in streams)
    right = rotation_keys(geom, worlds, rho)

    def matched(value):
        return torch.from_numpy(np.repeat(value, len(LABELS), axis=0))

    query = process.integers(0, 3, worlds)
    tape = dict(right=matched(right), left=matched(-right), law=torch.from_numpy(np.tile(LABELS, (worlds, 1))),
                order=matched(process.permuted(np.tile(np.arange(3), (worlds, 1)), axis=1)),
                study_flip=matched(prior.signs(process, (worlds, 3))),
                study_u=matched(process.random((worlds, 3, 1 + native.EDIT_DRAWS), dtype=np.float32)),
                probe_u=matched(process.random((worlds, 3), dtype=np.float32)),
                query=matched(query), query_flip=matched(prior.signs(process, worlds)),
                query_u=matched(process.random(worlds, dtype=np.float32)),
                donor_cycle=matched(process.integers(1, 3, worlds)),
                wait=matched(process.integers(0, 9, (worlds, 3))))
    return tape


def rho_for(arm, update, morph_steps):
    if arm == "direct" or update >= morph_steps or morph_steps <= 1:
        return -.5
    return -.5 * (update - 1) / (morph_steps - 1)


def _contact(rule, state, tape, rows, t, mode):
    item, flip = tape["order"][:, t], tape["study_flip"][:, t]
    left, right = binding.present(tape, item, flip)
    draw = tape["study_u"][:, t]
    action = native.act(rule, state, left, right, draw[:, 0])
    earned = action.side * flip * tape["law"][rows, item]
    if mode == "false_sign":
        delivered = -earned
    elif mode == "shuffled_objects":
        donor = (item + tape["donor_cycle"]) % 3
        delivered = action.side * flip * tape["law"][rows, donor]
    else:
        delivered = earned
    experience = native.observe(rule, action, left, right, delivered)
    edit = native.commit(rule, experience, draw[:, 1:], intervention="sham" if mode == "sham" else "active")
    return action, edit, earned, delivered


def training_step(graph, rule, tape, worlds):
    lives = worlds * len(LABELS)
    state = native.start(graph, lives)
    rule = rule.expand(lives)
    rows = torch.arange(len(tape["law"]))
    study, edit = 0., 0.
    for t in range(3):
        action, commit, _, _ = _contact(rule, state, tape, rows, t, "active")
        study = study + action.motor_log_prob
        edit = edit + commit.edit_log_prob
        state = native.wait(rule, commit.state, tape["wait"][:, t], max_ticks=8)
    item, flip = tape["query"], tape["query_flip"]
    left, right = binding.present(tape, item, flip)
    terminal = native.act(rule, state, left, right, tape["query_u"])
    earned = (terminal.side * flip * tape["law"][rows, item]).detach().reshape(-1, worlds, len(LABELS))
    mean = earned.mean(2, keepdim=True)
    advantage = earned - (mean.sum(1, keepdim=True) - mean) / (worlds - 1)
    loss = [old_fit.score_loss(score.reshape_as(earned), advantage) for score in (terminal.motor_log_prob, study, edit)]
    with torch.no_grad():
        correct = (earned > 0).float()
        metrics = torch.stack((correct.mean((1, 2)), correct[..., [0, 7]].mean((1, 2)),
                               correct[..., 1:7].mean((1, 2)), *loss), dim=1)
    return sum(loss).sum(), metrics


def _step(graph, rule, optimizer, tape, worlds):
    parameters = (graph.weights, graph.bias, *rule.parameters())
    for recovery in (False, True):
        optimizer.zero_grad(set_to_none=True)
        if recovery:
            model = native.Graph(nn.Parameter(graph.weights.detach().double()),
                                 nn.Parameter(graph.bias.detach().double()), graph.present)
            coefficients = deepcopy(rule).double()
        else:
            model, coefficients = graph, rule
        gradients = (model.weights, model.bias, *coefficients.parameters())
        loss, metrics = training_step(model, coefficients, tape, worlds)
        if not loss.isfinite():
            continue
        loss.backward()
        norm = revisit._gradients(gradients, 2.0)
        if norm is None:
            continue
        if recovery:
            for parameter, source in zip(parameters, gradients, strict=True):
                parameter.grad = None if source.grad is None else source.grad.to(parameter)
        optimizer.step()
        return torch.cat((metrics.double(), norm[:, None], torch.full_like(norm[:, None], recovery)), dim=1)
    raise FloatingPointError("Non-finite gradient after same-world float64 replay")


def _parents(path, births):
    parents = [old_fit.load_checkpoint(path, "odor", "uniform", birth, 8000) for birth in births]
    values = [torch.cat([getattr(graph, field) for graph, _ in parents]) for field in revisit.capacity.GRAPH_FIELDS]
    graph = native.Graph(nn.Parameter(values[0]), nn.Parameter(values[1]), values[2])
    rule = native.Rule(births)
    rule.load_state_dict({name: torch.cat([parent.state_dict()[name] for _, parent in parents])
                          for name in rule.state_dict()})
    return graph, rule


def fit_group(path, warm, births, target, updates, morph_steps, worlds, manifest_sha, threads):
    torch.set_num_threads(threads)
    for arm in ARMS:
        graph, rule = _parents(warm, births)
        optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=.003)
        folder = revisit._group_dir(path, arm, births)
        resume_file = folder / "resume.pt"
        metrics = torch.empty((len(births), target, len(METRICS)), dtype=torch.float64)
        start, previous_seconds = 0, 0.
        if resume_file.exists():
            checkpoint = torch.load(resume_file, map_location="cpu", weights_only=True)
            if (checkpoint["manifest_sha256"], checkpoint["births"], checkpoint["arm"]) != (manifest_sha, list(births), arm):
                raise ValueError(f"Resume would change the triangle protocol at {resume_file}")
            start, previous_seconds = checkpoint["step"], checkpoint["seconds"]
            if start >= target:
                if all(revisit._checkpoint(path, arm, birth, target).is_file() for birth in births):
                    continue
                if start > target:
                    raise ValueError(f"Missing archived stage {target} before resume update {start}")
            with torch.no_grad():
                for field in revisit.capacity.GRAPH_FIELDS:
                    getattr(graph, field).copy_(checkpoint["graph"][field])
            rule.load_state_dict(checkpoint["rule"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            metrics[:, :start] = checkpoint["metrics"]
        started = perf_counter()

        def save(step, *, stage):
            artifact = dict(arm=arm, births=list(births), step=step, manifest_sha256=manifest_sha,
                            metrics=metrics[:, :step].clone(), optimizer=optimizer.state_dict(), rule=rule.state_dict(),
                            graph={name: getattr(graph, name).detach() for name in revisit.capacity.GRAPH_FIELDS},
                            seconds=previous_seconds + perf_counter() - started)
            if stage:
                for lane, birth in enumerate(births):
                    old_fit.save_checkpoint(path, "odor", arm, birth, step, graph, rule, lane)
                    revisit._write_raw(folder / f"trace-{birth}-{step:06d}.npz", values=metrics[lane, :step].numpy(),
                                       metrics=METRICS)
                    if step:
                        tape = make_tape(TRAIN_SEED, birth, worlds, rho_for(arm, step, morph_steps), update=step)
                        revisit._write_raw(folder / f"tape-{birth}-{step:06d}.npz", **{k: v.numpy() for k, v in tape.items()})
                revisit._atomic(folder / f"optimizer-{step:06d}.pt", lambda temporary: torch.save(artifact, temporary))
            revisit._atomic(resume_file, lambda temporary: torch.save(artifact, temporary))

        if start == target:
            save(target, stage=True)
        for step in range(start + 1, target + 1):
            rho = rho_for(arm, step, morph_steps)
            rows = [make_tape(TRAIN_SEED, birth, worlds, rho, update=step) for birth in births]
            tape = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
            metrics[:, step - 1] = _step(graph, rule, optimizer, tape, worlds)
            if step == target or step % 250 == 0:
                save(step, stage=step == target)
                vals = metrics[:, max(0, step - 100):step]
                rates = vals[..., :3].mean((0, 1)).tolist()
                print(f"triangle {arm} {births} {step}/{updates} rho={rho:.4f} accuracy={rates[0]:.3f} "
                      f"uniform={rates[1]:.3f} mixed={rates[2]:.3f} "
                      f"updates_per_second={(step-start)/(perf_counter() - started):.1f}", flush=True)


@torch.no_grad()
def trials(graph, rule, tape, mode):
    count = len(tape["law"])
    rows = torch.arange(count)
    state = native.start(graph, count)
    rule = rule.expand(count)
    observed, delivered = torch.empty_like(tape["law"]), torch.empty_like(tape["law"])
    for t in range(3):
        item, flip = tape["order"][:, t], tape["study_flip"][:, t]
        action, commit, actual, received = _contact(rule, state, tape, rows, t, mode)
        observed[rows, item] = actual * action.side * flip
        delivered[rows, item] = received * action.side * flip
        state = native.wait(rule, commit.state, 4)
    if not torch.equal(tape["law"], observed):
        raise AssertionError("Public reconstruction of true feedback failed")
    if mode == "sham" and not revisit.capacity._same_graph(graph.repeat(count), state.graph):
        raise AssertionError("Sham support visits changed graph parameters")
    logit, selected = [], []
    for index in range(3):
        item = torch.full((count,), index)
        motor = []
        for orient in (1., -1.):
            left, right = binding.present(tape, item, torch.full((count,), orient))
            draw = tape["probe_u"][:, index]
            motor.append(native.act(rule, state, left, right, draw if orient > 0 else 1 - draw))
        logit.append(torch.stack((motor[0].motor_logit, motor[1].motor_logit), dim=1))
        selected.append(torch.stack((motor[0].side, -motor[1].side), dim=1))
    return dict(logit=torch.stack(logit, dim=1).numpy(), selected=torch.stack(selected, dim=1).numpy().astype(np.int8),
                observed=observed.numpy(), delivered=delivered.numpy())


def _analyze(tape, raw):
    result = probe.analyze(tape, raw, "triangle")
    law = raw["observed"].reshape(-1, 8, 3)
    canonical = raw["selected"].reshape(-1, 8, 3, 2)
    correct = canonical == law[..., None]
    logit = raw["logit"].reshape(-1, 8, 3, 2)
    margin = (logit[..., 0] - logit[..., 1]) * .5 * law
    pass_margin = margin > MARGIN_THRESHOLD
    result.update(per_object_sampled=correct.mean((0, 1, 3)).tolist(),
                  all_sampled_by_assignment=correct.all(2).mean((0, 2)).tolist(),
                  all_margin_by_assignment=pass_margin.all(2).mean(0).tolist(),
                  weakest_object_law=float(min(correct[..., obj, :][law[..., obj] == sign].mean()
                                                 for obj in range(3) for sign in (-1, 1))),
                  delivered_label_sampled_accuracy=float((canonical == raw["delivered"].reshape(-1, 8, 3, 1)).mean()))
    return result


def evaluate(path, evidence, stage, births, worlds):
    torch.set_num_threads(1)
    hashes, roots = {}, {}
    for arm in ARMS:
        roots[arm] = {}
        for birth in births:
            graph, rule = old_fit.load_checkpoint(path, "odor", arm, birth, stage)
            file = revisit._checkpoint(path, arm, birth, stage)
            hashes[file.relative_to(path).as_posix()] = prior.digest(file)
            packed, record = {}, {}
            for rho in GRAM_LEVELS:
                tape = make_tape(EVAL_SEED, birth, worlds, rho)
                modes = CONTROLS if rho == -.5 else CONTROLS[:1]
                responses = {mode: trials(graph, rule, tape, mode) for mode in modes}
                record[str(rho)] = {mode: _analyze(tape, raw) for mode, raw in responses.items()}
                packed.update({f"{rho}/tape/{k}": v.numpy() for k, v in tape.items()})
                packed.update({f"{rho}/{mode}/{k}": v for mode, raw in responses.items() for k, v in raw.items()})
            file = path / "evaluation" / f"{stage:06d}" / f"{arm}-{birth}.npz"
            revisit._write_raw(file, **packed)
            hashes[file.relative_to(path).as_posix()] = prior.digest(file)
            roots[arm][str(birth)] = record
    intervals = prior.BirthIntervals(births, 10_000, EVAL_SEED)
    fields = ("correct_sampled", "all_sampled", "all_greedy_symmetrized", "uniform_all_greedy", "other_all_greedy",
              "both_uniform_histories_pass", "weakest_object_law")

    def get(arm, rho, mode, metric):
        column = np.array([root[str(rho)][mode][metric] for root in roots[arm].values()], dtype=float)
        return column / worlds if metric == "both_uniform_histories_pass" else column

    summary, contrast = {}, {}
    for rho in GRAM_LEVELS:
        modes = CONTROLS if rho == -.5 else CONTROLS[:1]
        summary[str(rho)] = {arm: {mode: {key: intervals.report(get(arm, rho, mode, key)) for key in fields}
                                   for mode in modes} for arm in ARMS}
        contrast[str(rho)] = {key: intervals.report(get("morph", rho, "active", key) - get("direct", rho, "active", key))
                              for key in fields}
    report = dict(stage=stage, births=list(births), seed=EVAL_SEED, worlds_per_birth=worlds,
                  assignments=LABELS.tolist(), both_uniform_histories_margin_threshold=MARGIN_THRESHOLD,
                  signed_margin="law times half the difference of the native logits across opposite motor orientations",
                  controls={"sham": "Suppress graph writes on all three intact sensory contacts.",
                            "false_sign": "Invert each sampled support taste and otherwise permit native writes.",
                            "shuffled_objects": "Apply one matched random three-cycle to the canonical sign assignments, "
                                                "then encode each donor sign using the current object's actual sampled side. "
                                                "The two all-same assignments are invariant to this permutation."},
                  archive_sha256=hashes, roots=roots, summary=summary, morph_minus_direct=contrast,
                  source_manifest_sha256=prior.digest(path / "run.json"))
    name = f"triangle-training-{stage:06d}.json"
    revisit._write_json(path / "evaluation" / name, report)
    if evidence is not None:
        revisit._write_json(evidence / name, report)
    for arm in ARMS:
        row = summary["-0.5"][arm]["active"]
        print(f"triangle eval {stage} {arm} all_sampled={row['all_sampled']['mean']:.3f} "
              f"paired_uniform_margin_pass={row['both_uniform_histories_pass']['mean']:.3f}", flush=True)
    return report


def run(path, warm, evidence, births, updates, morph_steps, train_worlds, eval_worlds, workers, threads, *, until=None):
    if not (1 <= workers <= 2 and threads >= 1 and 1 <= morph_steps <= updates and train_worlds >= 2
            and eval_worlds >= 1 and births and len(set(births)) == len(births)):
        raise ValueError("Invalid triangle schedule or worker count")
    stages = sorted({0, morph_steps, updates, min(1000, updates)})
    if until is not None and until not in stages:
        raise ValueError("Endpoint must be a declared checkpoint stage")
    groups = tuple(batched(births, 3))
    manifest = dict(births=list(births), arms=list(ARMS), updates=updates, morph_steps=morph_steps,
                    splits=dict(train=TRAIN_SEED, evaluation=EVAL_SEED), stages=stages, train_worlds=train_worlds,
                    eval_worlds=eval_worlds, groups=[list(group) for group in groups], geometry_levels=list(GRAM_LEVELS),
                    source_sha256={Path(name).name: prior.digest(Path(name)) for name in
                                   (__file__, binding.__file__, probe.__file__, revisit.__file__, native.__file__)},
                    warm_checkpoint_sha256={str(birth): prior.digest(revisit._checkpoint(warm, "uniform", birth, 8000))
                                            for birth in births},
                    protocol="Exact triangle throughout vs orthogonal through equiangular to exact triangle over the first "
                             "specified phase. Identical triangle tapes thereafter. Three sampled physical studies with "
                             "random order and 0-8 gaps, all eight assignments crossed within rotation. One uniformly "
                             "queried terminal item, physically earned ±1 only; leave-own-rotation-out baseline. "
                             "Both starts use the same uniform-objective graph and rule and fresh Adam moments.")
    file = path / "run.json"
    if file.exists():
        if json.loads(file.read_text(encoding="utf8")) != manifest:
            raise ValueError(f"Triangle protocol differs at {file}")
    else:
        revisit._write_json(file, manifest)
    fingerprint = prior.digest(file)
    with ProcessPoolExecutor(max_workers=min(workers, len(groups))) as pool:
        for stage in (s for s in stages if until is None or s <= until):
            jobs = [pool.submit(fit_group, path, warm, group, stage, updates, morph_steps, train_worlds, fingerprint, threads)
                    for group in groups]
            for job in jobs:
                job.result()
            if not (path / "evaluation" / f"triangle-training-{stage:06d}.json").exists():
                evaluate(path, evidence, stage, births, eval_worlds)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA / "triangle-objective")
    parser.add_argument("--warm", type=Path, default=DATA / "retention-objective")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--births", type=int, nargs="+", default=old_fit.BIRTHS)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--morph-steps", type=int, default=4000)
    parser.add_argument("--train-worlds", type=int, default=4)
    parser.add_argument("--eval-worlds", type=int, default=96)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--until", type=int)
    args = parser.parse_args()
    run(args.out, args.warm, args.evidence, args.births, args.updates, args.morph_steps, args.train_worlds,
        args.eval_worlds, args.workers, args.threads, until=args.until)
