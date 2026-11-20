"""Factor terminal query recency and exposure count on shared four-odor histories."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import batched, product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity as cap
from . import interleave_revisit_train as revisit
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import local_trace_train as old_fit


DATA = revisit.ROOT / "data/interleaved-learning-20261120"
EVIDENCE = revisit.ROOT / "docs/evidence/interleaved-learning-2026-11-20"
ARMS = ("late_twice", "late_mixed", "both_twice", "both_mixed")
CELLS = ("early_once", "early_twice", "late_once", "late_twice")
SUPPORT = 6
TRAIN_DOMAIN = 713028
EVAL_DOMAIN = 713029
DIAGNOSTIC_SEED = 6807871
TRAIN_SEED = 431987
HOLDOUT_SEED = 9284531
CAP_SEED = 871831
LABELS = np.array(list(product((-1., 1.), repeat=4)), dtype=np.float32)


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    pairs: int = 16
    world_seed: int = TRAIN_SEED
    min_wait: int = 0
    max_wait: int = 8
    lr: float = .003
    grad_clip: float = 2.
    autosave: int = 250


@dataclass(frozen=True)
class Tape:
    physical: revisit.Tape
    roles: torch.Tensor


def make_tape(config, births, step, *, full_laws=False):
    tapes, roles = [], []
    for birth in births:
        rng = np.random.default_rng([EVAL_DOMAIN if full_laws else TRAIN_DOMAIN, config.world_seed, birth, step])
        pairs = config.pairs
        geometry = rng.normal(size=(pairs, 4, 3))
        geometry = (1.35 * geometry / np.linalg.norm(geometry, axis=-1, keepdims=True)).astype(np.float32)
        role = rng.permuted(np.tile(np.arange(4), (pairs, 1)), axis=1)
        order = np.concatenate([rng.permuted(np.tile(pair, (pairs, 1)), axis=1)
                                for pair in ([1, 3], [0, 1], [2, 3])], axis=1)
        sequence = np.take_along_axis(role, order, axis=1)
        use_late = rng.integers(0, 2, pairs)
        use_twice = rng.integers(0, 2, pairs)
        query_roles = np.stack((np.full(pairs, 3), 2 + use_twice, 2 * use_late + 1, 2 * use_late + use_twice), axis=1)
        query = np.take_along_axis(role, query_roles, axis=1)
        mirrored = prior.signs(rng, (pairs, 4))[:, None] * np.array([1., -1.], dtype=np.float32)[None, :, None]
        laws = np.broadcast_to(LABELS, (pairs, *LABELS.shape)) if full_laws else mirrored
        waits = rng.integers(config.min_wait, config.max_wait + 1, size=(pairs, SUPPORT))
        orientation = prior.signs(rng, (pairs, SUPPORT + 1))
        draws = rng.random((pairs, SUPPORT + 1, 1 + native.EDIT_DRAWS), dtype=np.float32)
        tapes.append((geometry, laws, sequence, query, orientation, waits, draws))
        roles.append(role)
    physical = revisit.Tape(*(torch.from_numpy(np.stack(values)) for values in zip(*tapes, strict=True)))
    return Tape(physical, torch.from_numpy(np.stack(roles)))


def _for_arm(tape, arm):
    index = ARMS.index(arm)
    return replace(tape.physical, query=tape.physical.query[..., index:index + 1])


@torch.no_grad()
def evaluate_cells(graph, rule, tape):
    physical = tape.physical
    _, worlds, laws, _ = physical.law.shape
    count = worlds * laws
    row = torch.arange(count)
    state = native.start(graph, count)
    rule = rule.expand(count)

    def lanes(value):
        return value.squeeze(0).repeat_interleave(laws, dim=0)

    orientations, draws = lanes(physical.orientation), lanes(physical.draw)
    geometry, sequence = lanes(physical.odor), lanes(physical.sequence)
    law = physical.law.reshape(count, 4)
    slots = lanes(tape.roles)
    contacts = []
    for t in range(SUPPORT):
        item, flip = sequence[:, t], orientations[:, t]
        left = geometry[row, item] * flip[:, None]
        motor = native.act(rule, state, left, -left, draws[:, t, 0])
        earned = motor.side * flip * law[row, item]
        experience = native.observe(rule, motor, left, -left, earned)
        edit = native.commit(rule, experience, draws[:, t, 1:])
        state = native.wait(rule, edit.state, lanes(physical.wait[:, :, t]), max_ticks=8)
        contacts.append(torch.stack((motor.side, earned)))
    # Every role is queried independently from this same snapshot, with the same physical orientation and motor draw.
    answers, logits, truths, last = [], [], [], []
    final_flip = orientations[:, -1]
    for role in range(len(CELLS)):
        item = slots[:, role]
        left = geometry[row, item] * final_flip[:, None]
        motor = native.act(rule, state, left, -left, draws[:, -1, 0])
        answers.append(motor.side)
        logits.append(motor.motor_logit)
        truths.append(final_flip * law[row, item])
        last.append(item == sequence[:, -1])
    raw = {key: value.numpy() for key, value in vars(physical).items()}
    raw.update(roles=tape.roles.numpy(), contact=torch.stack(contacts).numpy(),
               side=torch.stack(answers).numpy().astype(np.int8), logit=torch.stack(logits).numpy(),
               truth=torch.stack(truths).numpy().astype(np.int8), actual_last=torch.stack(last).numpy(),
               law=law.numpy())
    return raw


def _cell_metrics(raw):
    result = {}
    for index, cell in enumerate(CELLS):
        side, truth = raw["side"][index], raw["truth"][index]
        actual = side == truth
        slots = raw["roles"].reshape(-1, 4)[:, index]
        lane_slots = np.repeat(slots, raw["law"].shape[0] // len(slots))
        law = raw["law"][np.arange(len(lane_slots)), lane_slots]
        result[cell] = dict(accuracy=float(actual.mean()), weaker_law=min(float(actual[law > 0].mean()),
                                                                        float(actual[law < 0].mean())))
        if cell.startswith("late"):
            is_last = raw["actual_last"][index]
            result[cell].update(actual_last=float(actual[is_last].mean()), earlier_of_two=float(actual[~is_last].mean()))
    result["all_four"] = {"accuracy": float((raw["side"] == raw["truth"]).all(0).mean())}
    return result


def _summarize_cells(rows, intervals):
    return {cell: {key: intervals.report([row[cell][key] for row in rows]) for key in fields}
            for cell, fields in rows[0].items()}


def diagnose(out, evidence, sources, *, worlds=64):
    """Read original models on the four-cell protocol before any new model is fitted."""
    torch.set_num_threads(1)
    report = dict(seed=DIAGNOSTIC_SEED, worlds_per_birth=worlds, private_assignments_per_world=len(LABELS),
                  cells=list(CELLS), cohorts={}, raw_sha256={}, checkpoint_sha256={})
    for cohort, (source, births) in sources.items():
        intervals = prior.BirthIntervals(births, 10_000, DIAGNOSTIC_SEED)
        rows = {arm: [] for arm in revisit.ARMS}
        for birth in births:
            tape = make_tape(Config(pairs=worlds, world_seed=DIAGNOSTIC_SEED), (birth,), 0, full_laws=True)
            for arm in revisit.ARMS:
                graph, rule = old_fit.load_checkpoint(source, "odor", arm, birth, 8000)
                checkpoint = revisit._checkpoint(source, arm, birth, 8000)
                report["checkpoint_sha256"][f"{cohort}/{arm}/{birth}"] = prior.digest(checkpoint)
                raw = evaluate_cells(graph, rule, tape)
                raw_name = f"diagnostic/raw/{cohort}-{arm}-{birth}.npz"
                revisit._write_raw(out / raw_name, **raw)
                report["raw_sha256"][raw_name] = prior.digest(out / raw_name)
                rows[arm].append(_cell_metrics(raw))
        report["cohorts"][cohort] = {arm: _summarize_cells(results, intervals) for arm, results in rows.items()}
        for arm in revisit.ARMS:
            rates = report["cohorts"][cohort][arm]
            print(f"diagnostic {cohort} {arm} " + " ".join(f"{c}={rates[c]['accuracy']['mean']:.3f}" for c in CELLS),
                  flush=True)
    name = "retention-factorial-original-diagnostic.json"
    revisit._write_json(out / "diagnostic" / name, report)
    revisit._write_json(evidence / name, report)
    return report


def fit_group(path, warm, config, births, arm, target, manifest_sha, threads):
    torch.set_num_threads(threads)
    graph, rule = revisit._warm_graph(warm, births)
    optimizer = torch.optim.Adam((graph.weights, graph.bias, *rule.parameters()), lr=config.lr)
    folder = revisit._group_dir(path, arm, births)
    resume_file = folder / "resume.pt"
    metrics = torch.empty((len(births), target, len(revisit.METRICS)), dtype=torch.float64)
    start, previous_seconds = 0, 0.
    if resume_file.exists():
        state = torch.load(resume_file, map_location="cpu", weights_only=True)
        if (state["manifest_sha256"], state["births"], state["arm"]) != (manifest_sha, list(births), arm):
            raise ValueError(f"Resume would change factorial protocol at {resume_file}")
        start, previous_seconds = state["step"], state["seconds"]
        if start >= target and all(revisit._checkpoint(path, arm, birth, target).is_file() for birth in births):
            return
        if start > target:
            raise ValueError(f"Missing archived checkpoint at {target} before resume update {start}")
        with torch.no_grad():
            for field in cap.GRAPH_FIELDS:
                getattr(graph, field).copy_(state["graph"][field])
        rule.load_state_dict(state["rule"])
        optimizer.load_state_dict(state["optimizer"])
        metrics[:, :start] = state["metrics"]
    started = perf_counter()

    def save(step, *, stage):
        artifact = dict(arm=arm, births=list(births), step=step, manifest_sha256=manifest_sha,
                        graph={field: getattr(graph, field).detach() for field in cap.GRAPH_FIELDS},
                        rule=rule.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics[:, :step].clone(),
                        seconds=previous_seconds + perf_counter() - started)
        if stage:
            for lane, birth in enumerate(births):
                old_fit.save_checkpoint(path, "odor", arm, birth, step, graph, rule, lane)
                revisit._write_raw(folder / f"trace-{birth}-{step:06d}.npz", values=metrics[lane, :step].numpy(),
                                   metrics=revisit.METRICS)
            revisit._atomic(folder / f"optimizer-{step:06d}.pt", lambda temporary: torch.save(artifact, temporary))
        revisit._atomic(resume_file, lambda temporary: torch.save(artifact, temporary))

    if start == target:
        save(target, stage=True)
    for step in range(start + 1, target + 1):
        tape = make_tape(config, births, step)
        # The shared trainer reads the selected query from column zero; the full factorial arm stays environment-only.
        metrics[:, step - 1] = revisit._fit_step(graph, rule, optimizer, _for_arm(tape, arm), "recent", config)
        if step == target or step % config.autosave == 0:
            save(step, stage=step == target)
            trailing = metrics[:, max(0, step - 100):step, 1].mean()
            print(f"factorial {arm} {births} {step}/{config.updates} terminal={trailing:.3f} "
                  f"updates_per_second={(step - start) / (perf_counter() - started):.1f}", flush=True)


def _factors(rates, intervals):
    vectors = {arm: np.array([rates[arm]["birth_values"][str(birth)] for birth in intervals.births]) for arm in ARMS}
    late_twice, late_mixed, both_twice, both_mixed = (vectors[arm] for arm in ARMS)
    return {"allow_early_query": intervals.report(.5 * ((both_mixed + both_twice) - (late_mixed + late_twice))),
            "allow_once_query": intervals.report(.5 * ((both_mixed + late_mixed) - (both_twice + late_twice))),
            "interaction": intervals.report((both_mixed - both_twice) - (late_mixed - late_twice))}


def evaluate(path, evidence, config, births, stage, split, seed, capacity_seed, worlds, label):
    torch.set_num_threads(1)
    intervals = prior.BirthIntervals(births, 10_000, seed)
    cap_tape = cap.make_tape("odor", capacity_seed, 192, 4)
    cell_eval, capacity_eval = {}, {}
    checks = {}
    for arm in ARMS:
        cell_rows, cap_rows = [], []
        for birth in births:
            tape = make_tape(replace(config, pairs=worlds, world_seed=seed), (birth,), 0, full_laws=True)
            graph, rule = old_fit.load_checkpoint(path, "odor", arm, birth, stage)
            trained = revisit._checkpoint(path, arm, birth, stage)
            checks[trained.relative_to(path).as_posix()] = prior.digest(trained)
            raw_cells = evaluate_cells(graph, rule, tape)
            raw_cap = cap.evaluate(graph, rule, cap_tape, (0, 1, 4))
            cell_rows.append(_cell_metrics(raw_cells))
            cap_rows.append(raw_cap)
            for kind, raw in (("cells", raw_cells), ("capacity", raw_cap)):
                name = f"evaluation/{split}/{stage:06d}/{kind}-{arm}-{birth}.npz"
                revisit._write_raw(path / name, **raw)
                checks[name] = prior.digest(path / name)
        cell_eval[arm] = _summarize_cells(cell_rows, intervals)
        capacity_eval[arm] = cap.summarize(cap_rows, intervals)
    cell_factors = {cell: {metric: _factors({arm: cell_eval[arm][cell][metric] for arm in ARMS}, intervals)
                           for metric in fields} for cell, fields in cell_eval[ARMS[0]].items()}
    cap_factors = {}
    for k, block in capacity_eval[ARMS[0]].items():
        cap_factors[k] = {branch: {metric: _factors({arm: capacity_eval[arm][k]["branches"][branch][metric]
                                                    for arm in ARMS}, intervals) for metric in fields}
                          for branch, fields in block["branches"].items()}
    report = dict(stage=stage, split=split, seed=seed, capacity_seed=capacity_seed, cells=list(CELLS),
                  births=list(births), worlds_per_birth=worlds, private_assignments_per_world=len(LABELS),
                  source_manifest_sha256=prior.digest(path / "run.json"), cells_by_arm=cell_eval,
                  capacity_by_arm=capacity_eval, cell_factors=cell_factors, capacity_factors=cap_factors,
                  artifact_sha256=checks)
    name = f"{label}-{split}-{stage:06d}.json"
    revisit._write_json(path / "evaluation" / name, report)
    revisit._write_json(evidence / name, report)
    for arm in ARMS:
        result = capacity_eval[arm]["1"]["branches"]["write"]["A_and_B"]["mean"]
        print(f"factorial eval {label} {split} {stage} {arm} K1_both={result:.3f}", flush=True)
    return report


def run(path, warm, evidence, label, config, births, worlds, holdout_seed, workers, threads):
    if not (births and len(set(births)) == len(births) and 1 <= workers <= 3 and threads >= 1
            and worlds >= 8 and config.pairs >= 2 and config.updates >= 1):
        raise ValueError("Use distinct births, positive budgets, and one to three workers")
    stages = sorted({s for s in revisit.STAGES if s <= config.updates} | {config.updates})
    groups = tuple(batched(births, 3))
    manifest = dict(config=asdict(config), births=list(births), arms=list(ARMS), cells=list(CELLS), stages=stages,
                    groups=[list(group) for group in groups], supports_per_life=SUPPORT,
                    support_contacts_per_birth=2 * config.pairs * SUPPORT * config.updates,
                    holdout_seed=holdout_seed, matched_capacity_seed=CAP_SEED, eval_worlds=worlds,
                    warm_checkpoint_sha256={str(b): prior.digest(revisit._checkpoint(warm, "active", b, 8000)) for b in births},
                    source_sha256={Path(name).name: prior.digest(Path(name)) for name in
                                   (__file__, revisit.__file__, cap.__file__, old_fit.__file__, native.__file__)},
                    protocol="Four independently drawn odor pairs. First show both twice-exposed objects in random order. "
                             "Early and late blocks each present one once-exposed object and the second visit to one "
                             "twice-exposed object, in random order. Physical items are independently assigned to these roles. "
                             "Cross late-only vs half early and half late with twice-only vs half once and half twice. "
                             "Late-only names the final two-contact window, not the last item. Shared prefix and terminal "
                             "motor draws, no object labels to the graph. Single earned terminal ±1 credits all sampled "
                             "motor/structural decisions, matched Adam budget, fresh Adam moments from frozen warm graphs.")
    file = path / "run.json"
    if file.exists():
        if json.loads(file.read_text(encoding="utf8")) != manifest:
            raise ValueError(f"Factorial protocol differs at {file}")
    else:
        revisit._write_json(file, manifest)
    signature = prior.digest(file)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for stage in stages:
            work = [pool.submit(fit_group, path, warm, config, group, arm, stage, signature, threads)
                    for group in groups for arm in ARMS]
            for job in work:
                job.result()
            if stage:
                for birth in births:
                    tape = make_tape(config, (birth,), stage)
                    revisit._write_raw(path / "tapes" / f"birth-{birth}-update-{stage:06d}.npz", roles=tape.roles.numpy(),
                                       **{key: value.numpy() for key, value in vars(tape.physical).items()})
    for split, seed, cap_seed, stages_to_score in (("matched", DIAGNOSTIC_SEED, CAP_SEED, (0, config.updates)),
                                                   ("sealed", holdout_seed, holdout_seed, (0, config.updates))):
        for stage in stages_to_score:
            if not (path / "evaluation" / f"{label}-{split}-{stage:06d}.json").exists():
                evaluate(path, evidence, config, births, stage, split, seed, cap_seed, worlds, label)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA / "retention-factorial")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--label", default="retention-factorial")
    parser.add_argument("--warm", type=Path, default=revisit.ROOT / "data/local-trace-20261119/local")
    parser.add_argument("--births", type=int, nargs="+", default=old_fit.BIRTHS)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--world-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--holdout-seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    if args.diagnose:
        sources = {"original": (DATA / "retention-objective", tuple(range(731, 737))),
                   "confirmation": (DATA / "retention-objective-confirmation", tuple(range(811, 817)))}
        diagnose(args.out, args.evidence, sources, worlds=args.worlds)
    else:
        config = Config(updates=args.updates, pairs=args.pairs, world_seed=args.world_seed)
        run(args.out, args.warm, args.evidence, args.label, config, args.births, args.worlds, args.holdout_seed,
            args.workers, args.threads)
