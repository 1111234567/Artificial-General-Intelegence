"""Compare terminal query distributions with identical multi-odor study histories."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import batched
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_capacity as capacity
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import local_trace_train as old_fit
from . import self_param_writer as fw


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("recent", "uniform")
STAGES = (0, 1000, 4000, 8000)
TRAIN_DOMAIN = 0x1E3C1
SEEDS = {"matched": 871831, "replication": 871832}
REVIEW_BRANCHES = ("AB", "review_A", "expose_A", "neutral", "review_B")
METRICS = ("terminal_resource", "accuracy", "negative_law", "positive_law", "both_laws",
           "latest_hits", "latest_trials", "older_hits", "older_trials", "terminal_score_loss",
           "contact_score_loss", "edit_score_loss", "contacts_with_numeric_change", "births", "retirements",
           "gradient_norm", "float64_recovery")


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    pairs: int = 16
    two_object_updates: int = 4000
    world_seed: int = 431983
    min_wait: int = 0
    max_wait: int = 8
    lr: float = .003
    grad_clip: float = 2.0
    autosave: int = 250


@dataclass(frozen=True)
class Tape:
    # These tensors belong to the environment. The graph receives the physical menu and earned taste.
    odor: torch.Tensor
    law: torch.Tensor
    sequence: torch.Tensor
    query: torch.Tensor
    orientation: torch.Tensor
    wait: torch.Tensor
    draw: torch.Tensor


def make_tape(config, births, step):
    """The two arms consume the same deterministic draws and study histories."""
    easy = step <= config.two_object_updates
    contacts = 3 if easy else 5
    pairs = config.pairs
    rows = []
    for birth in births:
        rng = np.random.default_rng([TRAIN_DOMAIN, config.world_seed, birth, step])
        odor = rng.normal(size=(pairs, 4, 3))
        odor = (1.35 * odor / np.linalg.norm(odor, axis=-1, keepdims=True)).astype(np.float32)
        law = prior.signs(rng, (pairs, 4))[:, None, :] * np.array([1, -1], dtype=np.float32)[None, :, None]
        count = np.full(pairs, 2) if easy else rng.integers(2, 5, size=pairs)
        indices = np.arange(contacts)[None, :]
        review = (rng.random((pairs, contacts)) * count[:, None]).astype(np.int64)
        order = rng.permuted(np.tile(np.arange(4), (pairs, 1)), axis=1)
        selected = np.where(indices < count[:, None], indices, review)
        sequence = np.take_along_axis(order, selected, axis=1)
        uniform = order[np.arange(pairs), (rng.random(pairs) * count).astype(np.int64)]
        query = np.stack((sequence[:, -1], uniform), axis=1)
        orientation = prior.signs(rng, (pairs, contacts + 1))
        gap = rng.integers(config.min_wait, config.max_wait + 1, size=(pairs, contacts))
        draws = rng.random((pairs, contacts + 1, 1 + native.EDIT_DRAWS), dtype=np.float32)
        rows.append((odor, law, sequence, query, orientation, gap, draws))
    return Tape(*(torch.from_numpy(np.stack(values)) for values in zip(*rows, strict=True)))


def _study_and_probe(graph, rule, tape, arm, max_wait):
    roots, pairs, contacts = tape.sequence.shape
    lanes = roots * pairs * 2
    state = native.start(graph, pairs * 2)
    rule = rule.expand(pairs * 2)
    query = tape.query[..., ARMS.index(arm)]
    sequence = torch.cat((tape.sequence, query[..., None]), dim=-1)
    geometry = tape.odor.gather(2, sequence[..., None].expand(-1, -1, -1, 3))
    signs = tape.orientation
    menu = (geometry * signs[..., None]).unsqueeze(2).expand(-1, -1, 2, -1, -1).reshape(lanes, contacts + 1, 3)
    truth = (tape.law.gather(3, sequence[:, :, None].expand(-1, -1, 2, -1))
             * signs[:, :, None, :]).reshape(lanes, contacts + 1)
    draws = tape.draw.unsqueeze(2).expand(-1, -1, 2, -1, -1).reshape(lanes, contacts + 1, -1)
    durations = tape.wait.unsqueeze(2).expand(-1, -1, 2, -1).reshape(lanes, contacts)
    motor_score, edit_score = 0., 0.
    numeric = torch.zeros(lanes, device=graph.bias.device, dtype=graph.bias.dtype)
    births, retirements = torch.zeros_like(numeric), torch.zeros_like(numeric)
    for c in range(contacts):
        left, right = menu[:, c], -menu[:, c]
        action = native.act(rule, state, left, right, draws[:, c, 0])
        taste = action.side * truth[:, c]
        experience = native.observe(rule, action, left, right, taste)
        committed = native.commit(rule, experience, draws[:, c, 1:])
        motor_score = motor_score + action.motor_log_prob
        edit_score = edit_score + committed.edit_log_prob
        with torch.no_grad():
            receipt = committed.receipt
            numeric += receipt.weight_delta.flatten(1).ne(0).any(1) | receipt.bias_delta.ne(0).any(1)
            births += receipt.birth >= 0
            retirements += receipt.retired >= 0
        state = native.wait(rule, committed.state, durations[:, c], max_ticks=max_wait)
    terminal = native.act(rule, state, menu[:, -1], -menu[:, -1], draws[:, -1, 0])
    reward = (terminal.side * truth[:, -1]).detach()
    return reward, terminal.motor_log_prob, motor_score, edit_score, numeric, births, retirements


def training_step(graph, rule, tape, arm, max_wait):
    roots, pairs, _ = tape.sequence.shape
    reward, terminal, inspected, edits, numeric, born, retired = (
        value.reshape(roots, pairs, 2) for value in _study_and_probe(graph, rule, tape, arm, max_wait)
    )
    advantage = old_fit.leave_pair_out_advantage(reward)
    losses = tuple(old_fit.score_loss(score, advantage) for score in (terminal, inspected, edits))
    with torch.no_grad():
        correct = reward > 0
        query = tape.query[..., ARMS.index(arm)]
        legal_law = tape.law.gather(3, query[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
        newer = (query == tape.sequence[..., -1])[:, :, None].expand_as(correct)
        values = torch.stack((reward.mean((1, 2)), correct.float().mean((1, 2)),
                              (correct & (legal_law < 0)).sum((1, 2)) / pairs,
                              (correct & (legal_law > 0)).sum((1, 2)) / pairs,
                              correct.all(2).float().mean(1), (correct & newer).sum((1, 2)), newer.sum((1, 2)),
                              (correct & ~newer).sum((1, 2)), (~newer).sum((1, 2)), *losses,
                              *(value.mean((1, 2)) for value in (numeric, born, retired))), dim=1)
    return sum(losses).sum(), values


def _gradients(parameters, limit):
    with torch.no_grad():
        gradient = [p.grad for p in parameters if p.grad is not None]
        # Squaring large finite float32 gradients can overflow before clipping.
        norms = sum(g.flatten(1).double().square().sum(1) for g in gradient).sqrt()
        if not norms.isfinite().all():
            return None
        factors = (limit / (norms + 1e-6)).clamp(max=1)
        for grad in gradient:
            grad.mul_(factors.reshape((-1,) + (1,) * (grad.ndim - 1)))
        return norms


def _fit_step(graph, rule, optimizer, tape, arm, config):
    parameters = (graph.weights, graph.bias, *rule.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = training_step(graph, rule, tape, arm, config.max_wait)
    if loss.isfinite():
        loss.backward()
        norms = _gradients(parameters, config.grad_clip)
    else:
        norms = None
    recovered = norms is None
    if recovered:
        # Replay the same world and uniforms from the unchanged pre-step optimizer state.
        precise_graph = native.Graph(nn.Parameter(graph.weights.detach().double()),
                                    nn.Parameter(graph.bias.detach().double()), graph.present)
        precise_rule = deepcopy(rule).double()
        loss, metrics = training_step(precise_graph, precise_rule, tape, arm, config.max_wait)
        if not loss.isfinite():
            raise FloatingPointError("Non-finite score after same-world float64 replay")
        loss.backward()
        precise = (precise_graph.weights, precise_graph.bias, *precise_rule.parameters())
        norms = _gradients(precise, config.grad_clip)
        if norms is None:
            raise FloatingPointError("Non-finite gradient after same-world float64 replay")
        for param, source in zip(parameters, precise, strict=True):
            param.grad = None if source.grad is None else source.grad.to(param.dtype)
    optimizer.step()
    return torch.cat((metrics.double(), norms[:, None], torch.full_like(norms[:, None], float(recovered))), dim=1)


def _atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    _atomic(path, lambda tmp: tmp.write_text(payload, encoding="utf8"))


def _write_raw(path, **arrays):
    def save(temporary):
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    _atomic(path, save)


def _warm_graph(warm, births):
    parents = [old_fit.load_checkpoint(warm, "odor", "active", birth, 8000) for birth in births]
    values = [torch.cat([getattr(graph, field) for graph, _ in parents]) for field in capacity.GRAPH_FIELDS]
    graph = native.Graph(nn.Parameter(values[0]), nn.Parameter(values[1]), values[2])
    rule = native.Rule(births)
    rule.load_state_dict({name: torch.cat([parent.state_dict()[name] for _, parent in parents])
                          for name in rule.state_dict()})
    return graph, rule


def _group_dir(path, arm, births):
    return path / "training" / arm / ("births-" + "-".join(map(str, births)))


def _checkpoint(path, arm, birth, step):
    return old_fit.checkpoint_path(path, "odor", arm, birth, step)


def _fit_group(path, warm, config, births, target, manifest_hash, progress, threads):
    torch.set_num_threads(threads)
    for arm in ARMS:
        graph, rule = _warm_graph(warm, births)
        optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=config.lr)
        folder = _group_dir(path, arm, births)
        blob = folder / "resume.pt"
        metrics = torch.empty((len(births), target, len(METRICS)), dtype=torch.float64)
        start_step, previous_seconds = 0, 0.
        if blob.exists():
            state = torch.load(blob, map_location="cpu", weights_only=True)
            if (state["arm"], state["births"], state["manifest_sha256"]) != (arm, list(births), manifest_hash):
                raise ValueError(f"Resume would change the declared fit at {blob}")
            start_step, previous_seconds = state["step"], state["seconds"]
            if start_step > target:
                if not all(_checkpoint(path, arm, birth, target).exists() for birth in births):
                    raise ValueError(f"Missing stage {target} before resume step {start_step}")
                continue
            with torch.no_grad():
                for field in capacity.GRAPH_FIELDS:
                    getattr(graph, field).copy_(state["graph"][field])
            rule.load_state_dict(state["rule"])
            optimizer.load_state_dict(state["optimizer"])
            metrics[:, :start_step] = state["metrics"]

        started = perf_counter()

        def save(step, *, stage):
            seconds = previous_seconds + perf_counter() - started
            state = dict(arm=arm, births=list(births), manifest_sha256=manifest_hash, step=step, seconds=seconds,
                         graph={field: getattr(graph, field).detach() for field in capacity.GRAPH_FIELDS},
                         rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics[:, :step].clone())
            if stage:
                for lane, birth in enumerate(births):
                    old_fit.save_checkpoint(path, "odor", arm, birth, step, graph, rule, lane)
                    _write_raw(folder / f"trace-{birth}-{step:06d}.npz", metrics=METRICS,
                               values=metrics[lane, :step].numpy())
                _atomic(folder / f"optimizer-{step:06d}.pt", lambda tmp: torch.save(state, tmp))
            _atomic(blob, lambda tmp: torch.save(state, tmp))

        if start_step == target:
            if not all(_checkpoint(path, arm, birth, target).exists() for birth in births):
                save(target, stage=True)
            continue
        for step in range(start_step + 1, target + 1):
            tape = make_tape(config, births, step)
            metrics[:, step - 1] = _fit_step(graph, rule, optimizer, tape, arm, config)
            if step == target or step % config.autosave == 0:
                save(step, stage=step == target)
            if step == target or (progress and step % progress == 0):
                trailing = metrics[:, max(0, step - 100):step]
                older_hits, older_trials = trailing[..., 7].sum().item(), trailing[..., 8].sum().item()
                older = f"{older_hits / older_trials:.3f}" if older_trials else "na"
                norm, recovery = trailing[..., -2].max().item(), int(trailing[..., -1].sum().item())
                print(f"{arm} births={births} step={step}/{config.updates} accuracy={trailing[..., 1].mean():.3f} "
                      f"older={older} max_grad={norm:.3g} recovered={recovery} "
                      f"updates_per_second={(step - start_step) / (perf_counter() - started):.1f}", flush=True)


def _review_tape(tape, obj):
    # Slot C contributes independent position and sampling draws, then A/B supplies the physical geometry and law.
    left, right, value = (getattr(tape, key).clone() for key in ("left", "right", "right_value"))
    original = tape.directions[:, 2].repeat_interleave(4, dim=0)
    direction = tape.directions[:, obj].repeat_interleave(4, dim=0)
    orientation = (left[:, 2, 0] * original).sum(dim=1).sign()
    left[:, 2, 0] = direction * orientation[:, None]
    right[:, 2, 0] = -left[:, 2, 0]
    value[:, 2, 0] = orientation * tape.law[:, obj]
    return replace(tape, left=left, right=right, right_value=value)


@torch.no_grad()
def evaluate_reviews(graph, rule, tape):
    """Test a real review against equal-clock exposure and neutral waits on the capacity tape."""
    initial = native.start(graph, tape.lanes)
    rule = prior.birth_rule(rule)
    a, contact_a = capacity._study(rule, initial, tape, 0, write=True)
    ab, contact_b = capacity._study(rule, a, tape, 1, write=True)
    review_a, review_b = (_review_tape(tape, obj) for obj in (0, 1))
    a_real, contact_real_a = capacity._study(rule, ab, review_a, 2, write=True)
    a_sham, contact_sham_a = capacity._study(rule, ab, review_a, 2, write=False)
    b_real, contact_real_b = capacity._study(rule, ab, review_b, 2, write=True)
    neutral = capacity._wait(rule, ab, capacity.CONTACT_TICKS)
    if not (torch.equal(contact_real_a, contact_sham_a) and capacity._same_graph(ab.graph, a_sham.graph)):
        raise AssertionError("Review interventions must share their experience through the write")
    rows = []
    for state in (ab, a_real, a_sham, neutral, b_real):
        snapshot = capacity._wait(rule, state, 4)
        actions = [capacity._probe(rule, snapshot, tape, obj) for obj in (0, 1)]
        rows.append(torch.stack([torch.stack((action.side, action.motor_logit)) for action in actions]))
    choices = torch.stack(rows).numpy()
    return {"side": choices[:, :, 0, :].astype(np.int8), "logit": choices[:, :, 1, :],
            "law": tape.law[:, :2].numpy().T, "truth": tape.right_value[:, :2, -1].numpy().T,
            "contact": torch.stack((contact_a, contact_b, contact_real_a, contact_sham_a, contact_real_b)).numpy()}


def _summarize_reviews(rows, intervals):
    def metrics(row, branch):
        a, b = row["side"][branch] == row["truth"]
        return [a.mean(), b.mean(), (a & b).mean(), (a[::2] & a[1::2]).mean(),
                min(a[row["law"][0] < 0].mean(), a[row["law"][0] > 0].mean()),
                min(b[row["law"][1] < 0].mean(), b[row["law"][1] > 0].mean())]

    names = ("A", "B", "A_and_B", "A_both_laws", "A_weaker_law", "B_weaker_law")
    values = {branch: np.array([metrics(row, b) for row in rows]).T for b, branch in enumerate(REVIEW_BRANCHES)}

    def estimates(rates):
        return {key: intervals.report(value) for key, value in zip(names, rates, strict=True)}

    return {"branches": {branch: estimates(v) for branch, v in values.items()},
            "review_A_minus_control": {branch: estimates(values["review_A"] - values[branch])
                                       for branch in REVIEW_BRANCHES if branch != "review_A"}}


def evaluate_stage(path, evidence, births, stage, *, split="matched", worlds=192):
    seed = SEEDS[split]
    tape = capacity.make_tape("odor", seed, worlds, max(capacity.CAPACITIES))
    intervals = prior.BirthIntervals(births, 10_000, seed)
    files = {"raw_sha256": {}, "checkpoint_sha256": {}}
    cap_results, review_results = {}, {}
    for arm in ARMS:
        cap_rows, review_rows = [], []
        for birth in births:
            graph, rule = old_fit.load_checkpoint(path, "odor", arm, birth, stage)
            checkpoint = _checkpoint(path, arm, birth, stage)
            files["checkpoint_sha256"][checkpoint.relative_to(path).as_posix()] = prior.digest(checkpoint)
            cap_result = capacity.evaluate(graph, rule, tape)
            review_result = evaluate_reviews(graph, rule, tape)
            for test, result in (("capacity", cap_result), ("review", review_result)):
                name = f"evaluation/{split}/{stage:06d}/{test}-{arm}-{birth}.npz"
                _write_raw(path / name, **result)
                files["raw_sha256"][name] = prior.digest(path / name)
            cap_rows.append(cap_result)
            review_rows.append(review_result)
        cap_results[arm] = capacity.summarize(cap_rows, intervals)
        review_results[arm] = _summarize_reviews(review_rows, intervals)
    arm_difference = prior.compare_rates(cap_results["uniform"], cap_results["recent"], intervals,
                                        tuple((branch, branch) for branch in capacity.BRANCHES))
    report = dict(stage=stage, split=split, seed=seed, independent_worlds_per_birth=worlds,
                  births=list(births), branches=list(capacity.BRANCHES), review_branches=list(REVIEW_BRANCHES),
                  interval_unit="equal-weight birth, paired differences before resampling", **files,
                  training_manifest_sha256=prior.digest(path / "run.json"),
                  capacity=cap_results, review=review_results, capacity_uniform_minus_recent=arm_difference)
    name = f"retention-objective-{split}-{stage:06d}.json"
    _write_json(path / "evaluation" / name, report)
    if evidence is not None:
        _write_json(evidence / name, report)
    for arm in ARMS:
        k1 = cap_results[arm]["1"]["branches"]["write"]
        k4 = cap_results[arm]["4"]["branches"]["write"]["A"]["mean"]
        review = review_results[arm]["review_A_minus_control"]["expose_A"]["A_and_B"]["mean"]
        print(f"eval {split} stage={stage} {arm} K1_A={k1['A']['mean']:.3f} K1_both={k1['A_and_B']['mean']:.3f} "
              f"K4_A={k4:.3f} review_joint_lift={review:+.3f}", flush=True)
    return report


def _plan(path, warm, config, births, groups):
    warm_hashes = {str(birth): prior.digest(_checkpoint(warm, "active", birth, 8000)) for birth in births}
    sources = (__file__, native.__file__, old_fit.__file__, fw.__file__, capacity.__file__)
    plan = dict(config=asdict(config), births=list(births), groups=[list(group) for group in groups], arms=list(ARMS),
                checkpoints=sorted({s for s in STAGES if s <= config.updates} | {config.updates}),
                warm_task="odor", warm_stage=8000, warm_checkpoint_sha256=warm_hashes,
                training_domain=TRAIN_DOMAIN, evaluation_seeds=SEEDS, metrics=METRICS,
                source_sha256={Path(name).name: prior.digest(Path(name)) for name in sources},
                contact_budget_per_birth=2 * config.pairs * (3 * min(config.updates, config.two_object_updates)
                                                            + 5 * max(0, config.updates - config.two_object_updates)),
                protocol={"study": "Variable private laws per world and item, with complementary-law twins. Two introductions and "
                                   "one real review in the first phase. In the later phase, sample two to four distinct odors and "
                                   "fill five slots with introductions followed by reviews uniformly drawn from seen items.",
                          "target": "The recent arm queries the latest actual contact. The uniform arm queries an item uniformly "
                                    "among those seen. Both arms share the study tape and terminal side orientation/draw.",
                          "credit": "Use the physically earned terminal ±1 alone for leave-pair-out REINFORCE over terminal, "
                                    "study/review motor and topology-sampling scores. Continuous state and parameters carry through.",
                          "recovery": "Measure squared gradient norms in float64, clip per birth. A nonfinite float32 step is "
                                      "replayed on the same tape from unchanged graph/optimizer in float64 and logged.",
                          "review": "Independent A/B probes after A→B. For the review branches, use independent orientation/draws and "
                                    "equal extra time for a real review of A, sham exposure to A, neutral ticks, or real review of B."})
    manifest = path / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf8")) != json.loads(json.dumps(plan)):
            raise ValueError(f"Immutable run protocol differs at {manifest}")
    else:
        _write_json(manifest, plan)
    return plan, prior.digest(manifest)


def run(path, warm, config, births, *, workers=2, group_size=3, threads=1, progress=250,
        worlds=192, evidence=None, until=None, evaluate=True):
    if not (0 <= config.two_object_updates and config.updates >= 1 and config.pairs >= 2
            and 0 <= config.min_wait <= config.max_wait and config.lr > 0 and config.grad_clip > 0
            and min(workers, group_size, threads, worlds, config.autosave) > 0 and len(births) == len(set(births))
            and births and progress >= 0):
        raise ValueError("Invalid revisit fit configuration")
    path, warm = Path(path), Path(warm)
    groups = tuple(batched(births, group_size))
    plan, manifest_hash = _plan(path, warm, config, births, groups)
    end = config.updates if until is None else until
    if end not in plan["checkpoints"]:
        raise ValueError("The requested endpoint must be a declared checkpoint stage")
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=min(workers, len(groups))) as pool:
        for stage in (value for value in plan["checkpoints"] if value <= end):
            jobs = [pool.submit(_fit_group, path, warm, config, group, stage, manifest_hash, progress, threads)
                    for group in groups]
            for job in jobs:
                job.result()
            if stage > 0:
                for step in (1 if stage <= 1000 else max(1, stage - config.autosave + 1), stage):
                    for birth in births:
                        tape = make_tape(config, (birth,), step)
                        _write_raw(path / "tapes" / f"birth-{birth}-update-{step:06d}.npz",
                                   **{key: value.numpy() for key, value in vars(tape).items()})
            if evaluate and not (path / "evaluation" / f"retention-objective-matched-{stage:06d}.json").exists():
                evaluate_stage(path, evidence, births, stage, worlds=worlds)
    if evaluate and end == config.updates:
        target = path / "evaluation" / f"retention-objective-replication-{end:06d}.json"
        if not target.exists():
            evaluate_stage(path, evidence, births, end, split="replication", worlds=worlds)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/interleaved-learning-20261120/retention-objective")
    parser.add_argument("--warm", type=Path, default=ROOT / "data/local-trace-20261119/local")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--births", type=int, nargs="+", default=old_fit.BIRTHS)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--until", type=int)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--two-object-updates", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--worlds", type=int, default=192)
    parser.add_argument("--progress", type=int, default=250)
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    config = Config(updates=args.updates, pairs=args.pairs, two_object_updates=args.two_object_updates)
    run(args.out, args.warm, config, args.births, workers=args.workers, group_size=args.group_size,
        threads=args.threads, progress=args.progress, worlds=args.worlds, evidence=args.evidence,
        until=args.until, evaluate=not args.no_evaluate)


if __name__ == "__main__":
    main()
