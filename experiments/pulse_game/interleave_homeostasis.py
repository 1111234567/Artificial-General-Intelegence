"""Test relaxation of inherited graph parameters during frozen physical lifetimes."""

from argparse import ArgumentParser
from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_train as train, local_trace_graph as net, local_trace_train as old
from .interleave_development import estimate


ROOT = Path(__file__).resolve().parents[2]
DATA = train.DEFAULT_OUTPUT.parent
PLAN = ROOT / "docs/evidence/interleaved-learning-2026-11-20/homeostasis-plan.json"
OUT = DATA / "homeostasis"
AGES = (0, 4, 16, 64, 128)
SELECTION_AGES = (16, 64, 128)
ECOLOGIES = ("recurring", "absent_a")
REFERENCES = ("birth", "zero", "prewrite")
METRICS = ("a_before_x", "a", "b", "x", "all_three")
CAL_SEED, CONFIRM_SEED = 730561, 730589
CAL_DOMAIN, CONFIRM_DOMAIN = 2201171, 2201173
CAL_BIRTHS, CONFIRM_BIRTHS = tuple(range(731, 737)), tuple(range(811, 817))
WAIT = 4


@dataclass(frozen=True)
class Tape:
    objects: torch.Tensor
    laws: torch.Tensor
    visits: torch.Tensor
    orientation: torch.Tensor
    draws: torch.Tensor
    x_orientation: torch.Tensor
    x_draws: torch.Tensor
    probe_orientation: torch.Tensor
    probe_draws: torch.Tensor

    @property
    def pairs(self):
        return len(self.objects)


def _sign(rng, shape):
    return (2 * rng.integers(2, size=shape, dtype=np.int8) - 1).astype(np.float32)


def make_tape(pairs, seed, domain, max_age):
    """Share objects, visits to B, and all motor and write draws across ecologies."""
    rng = np.random.default_rng([domain, seed])
    shapes = (pairs, max_age + 3)
    objects = rng.normal(size=(*shapes, 3)).astype(np.float32)
    objects *= 1.35 / np.linalg.norm(objects, axis=-1, keepdims=True)
    laws = _sign(rng, shapes)
    columns = np.arange(pairs)[None, :]
    intro = rng.integers(2, size=pairs)
    anchors = np.stack((intro, 1 - intro), axis=0)
    distractors = np.broadcast_to(np.arange(3, max_age + 3)[:, None], (max_age, pairs))
    visits = np.repeat(np.concatenate((anchors, distractors), axis=0)[None, ...], len(ECOLOGIES), axis=0)
    blocks = np.arange(max_age // 4)[:, None]
    slots = 2 + 4 * blocks + rng.integers(4, size=(len(blocks), pairs))
    recurrence = (blocks + rng.integers(2, size=pairs)) % 2
    visits[0, slots, columns] = recurrence
    visits[1, slots, columns] = np.where(recurrence == 1, 1, slots + 1)
    return Tape(*(torch.from_numpy(value) for value in (
        objects, laws, visits, _sign(rng, (max_age + 2, pairs)),
        rng.random((max_age + 2, pairs, 5), dtype=np.float32), _sign(rng, pairs),
        rng.random((pairs, 5), dtype=np.float32), _sign(rng, (pairs, 3)),
        rng.random((pairs, 3), dtype=np.float32),
    )))


def interpolate(rule, experience, committed, template, ancestral, reference, gamma):
    """Keep the native support result and replay the exact native write timing."""
    post = committed.state.graph
    surviving = ancestral & post.present
    if not gamma:
        return committed.state, surviving
    if reference == "birth":
        w, b = template.weights, template.bias
    elif reference == "zero":
        w, b = torch.zeros_like(post.weights), torch.zeros_like(post.bias)
    elif reference == "prewrite":
        w, b = experience.state.graph.weights, experience.state.graph.bias
    else:
        raise ValueError(f"Unknown relaxation reference {reference!r}")
    next_weight = net._nonzero(post.weights.lerp(w, gamma))
    weights = torch.where(surviving, next_weight, post.weights)
    amended = replace(post, weights=weights, bias=post.bias.lerp(b, gamma))
    ready = net.wait(rule, experience.state, net.PRE_WRITE_TICKS)
    return net.wait(rule, replace(ready, graph=amended), net.POST_WRITE_TICKS), surviving


def _contact(rule, state, template, ancestral, right, factor, draws, reference, gamma):
    action = net.act(rule, state, -right, right, draws[:, 0])
    earned = action.side * factor
    experience = net.observe(rule, action, -right, right, earned)
    committed = net.commit(rule, experience, draws[:, 1:])
    after, surviving = interpolate(rule, experience, committed, template, ancestral, reference, gamma)
    return net.wait(rule, after, WAIT), surviving, action.side, earned


def _append(record, **values):
    for key, value in values.items():
        record.setdefault(key, []).append(value.numpy())


@torch.no_grad()
def replay(graph, rule, tape, ecology, ages, reference, gamma):
    """Replay matched lifetime prefixes, with both X laws at each frozen cut."""
    identity = tape.visits[ECOLOGIES.index(ecology), :max(ages) + 2]
    pairs = tape.pairs
    cols = torch.arange(pairs)[None, :]
    history_sign = torch.tensor((1., -1.)).repeat(pairs)
    x_law = torch.tensor((1., -1., 1., -1.)).repeat(pairs)
    history_menu = (tape.objects[cols, identity] * tape.orientation[:len(identity), :, None]).repeat_interleave(2, dim=1)
    history_factor = ((tape.laws[cols, identity] * tape.orientation[:len(identity)]).repeat_interleave(2, dim=1)
                      * history_sign)
    history_draw = tape.draws[:len(identity)].repeat_interleave(2, dim=1)
    template2, template4 = graph.repeat(2 * pairs), graph.repeat(4 * pairs)
    state = net.start(template2)
    ancestral = state.graph.present
    rule2, rule4 = rule.expand(2 * pairs), rule.expand(4 * pairs)
    twins = torch.arange(2 * pairs).repeat_interleave(2)
    x_right = (tape.objects[:, 2] * tape.x_orientation[:, None]).repeat_interleave(4, 0)
    x_factor = tape.x_orientation.repeat_interleave(4) * x_law
    x_draw = tape.x_draws.repeat_interleave(4, 0)
    probe_right = (tape.objects[:, :3] * tape.probe_orientation[..., None]).repeat_interleave(4, 0)
    probe_laws = torch.cat((tape.laws[:, :2].repeat_interleave(4, 0) * history_sign.repeat_interleave(2)[:, None],
                            x_law[:, None]), dim=1)
    probe_target = probe_laws * tape.probe_orientation.repeat_interleave(4, 0)
    probe_draw = tape.probe_draws.repeat_interleave(4, 0)
    record, history = {}, {}

    def shape(value):
        return value.reshape(pairs, 2, 2, *value.shape[1:])

    for index, (right, factor, draws) in enumerate(zip(history_menu, history_factor, history_draw, strict=True), 1):
        state, ancestral, chosen, earned = _contact(
            rule2, state, template2, ancestral, right, factor, draws, reference, gamma)
        _append(history, history_side=chosen.reshape(pairs, 2).to(torch.int8),
                history_earned=earned.reshape(pairs, 2).to(torch.int8))
        if index - 2 not in ages:
            continue
        snapshot = state.subset(twins)
        a_menu = probe_right[:, 0]
        before_side = net.act(rule4, snapshot, -a_menu, a_menu, probe_draw[:, 0]).side
        after, survived, _, x_earned = _contact(rule4, snapshot, template4, ancestral[twins], x_right,
                                                x_factor, x_draw, reference, gamma)
        queried = [net.act(rule4, after, -right, right, draw).side
                   for right, draw in zip(probe_right.unbind(dim=1), probe_draw.unbind(dim=1), strict=True)]
        side = torch.stack(queried, dim=-1)
        _append(record, side=shape(side).to(torch.int8), target=shape(probe_target).to(torch.int8),
                a_before_side=shape(before_side).to(torch.int8), x_earned=shape(x_earned).to(torch.int8),
                ancestral_after_x=shape(survived.sum((1, 2))), support_after_x=shape(after.graph.present.sum((1, 2))),
                weight_l1_from_birth=shape((after.graph.weights - template4.weights).abs().sum((1, 2))),
                bias_l1_from_birth=shape((after.graph.bias - template4.bias).abs().sum(1)))
    return {"visits": identity.numpy(), **{key: np.stack(value) for key, value in {**record, **history}.items()}}


def correctness(raw):
    success = raw["side"] == raw["target"]
    a_before = raw["a_before_side"] == raw["target"][..., 0]
    return np.concatenate((a_before[..., None], success, success.all(-1)[..., None]), axis=-1)


def _label(reference, gain):
    return "unchanged" if gain == 0 else f"{reference}-{gain:g}"


def _save_raw(path, arrays):
    def write(temporary):
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    train._atomic_save(path, write)


def _run_split(output, source, births, tape, variants, ages, step):
    output.mkdir(parents=True, exist_ok=True)
    arrays = {field.name: getattr(tape, field.name).numpy() for field in fields(tape)}
    world_file = output / "worlds.npz"
    _save_raw(world_file, arrays)
    results = {}
    fingerprints = {world_file.name: sha256(world_file.read_bytes()).hexdigest()}
    for birth in births:
        graph, rule = old.load_checkpoint(source, "odor", "uniform", birth, step)
        for ecology in ECOLOGIES:
            for reference, gain in variants:
                label = _label(reference, gain)
                raw = replay(graph, rule, tape, ecology, ages, reference, gain)
                file = output / f"{birth}-{ecology}-{label}.npz"
                _save_raw(file, raw)
                fingerprints[file.name] = sha256(file.read_bytes()).hexdigest()
                results[birth, ecology, label] = correctness(raw).mean(axis=(2, 3))
        print(f"{output.name} birth={birth} variants={len(variants)}", flush=True)
    return results, fingerprints


def select(results, births, ages, grid):
    old_ages = [ages.index(age) for age in SELECTION_AGES]
    report = {}
    for reference in REFERENCES:
        ranked = {}
        for gain in grid:
            label = _label(reference, gain)
            values = [results[birth, ecology, label][old_ages, :, -1]
                      for birth in births for ecology in ECOLOGIES]
            ranked[gain] = float(np.mean(values))
        best = min(ranked, key=lambda gain: (-ranked[gain], gain))
        report[reference] = {"selected_gain": best, "calibration_mean": ranked[best], "grid": ranked}
    return report


def _confirm(results, births, ages, variants, pairs, bootstrap):
    rng = np.random.default_rng([CONFIRM_DOMAIN, CONFIRM_SEED, 711])
    boot_birth = rng.integers(len(births), size=(bootstrap, len(births)))
    boot_pair = rng.integers(pairs, size=(bootstrap, pairs))
    tables, contrasts = {}, {}
    for ecology in ECOLOGIES:
        control = np.stack([results[birth, ecology, "unchanged"] for birth in births])
        for reference, gain in variants:
            label = _label(reference, gain)
            values = np.stack([results[birth, ecology, label] for birth in births])
            key = f"{ecology}/{reference}" if reference != "unchanged" else f"{ecology}/unchanged"

            def summarize(array):
                return {str(age): {metric: estimate(array[:, i, :, j], boot_birth, boot_pair)
                                   for j, metric in enumerate(METRICS)} for i, age in enumerate(ages)}

            tables[key] = summarize(values)
            if reference != "unchanged":
                contrasts[key] = summarize(values - control)
    return tables, contrasts


def _lock(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf8") != payload:
            raise ValueError(f"Immutable homeostasis protocol differs at {path}")
    else:
        train._atomic_save(path, lambda file: file.write_text(payload, encoding="utf8"))
    return sha256(payload.encode("utf8")).hexdigest()


def run(path, plan_path, cal_source, confirm_source, cal_pairs, confirm_pairs, bootstrap, step):
    torch.set_num_threads(1)
    grid = (0., *json.loads(plan_path.read_text(encoding="utf8"))["gain_grid"])
    if tuple(sorted(set(grid))) != grid or grid[-1] >= 1:
        raise ValueError("Use a sorted distinct gain grid with gains in [0, 1)")
    manifest = dict(plan_sha256=train._source_digest(plan_path), ages=AGES, selection_ages=SELECTION_AGES,
                    ecologies=ECOLOGIES, grid=grid, bootstrap=bootstrap, metrics=METRICS, quiet_ticks=WAIT,
                    calibration=dict(births=CAL_BIRTHS, pairs=cal_pairs, seed=CAL_SEED, domain=CAL_DOMAIN),
                    confirmation=dict(births=CONFIRM_BIRTHS, pairs=confirm_pairs, seed=CONFIRM_SEED, domain=CONFIRM_DOMAIN),
                    source_sha256={Path(f).name: train._source_digest(f) for f in (__file__, net.__file__, old.__file__)},
                    checkpoint_sha256={label: {str(birth): train._source_digest(old.checkpoint_path(
                        directory, "odor", "uniform", birth, step)) for birth in births}
                                       for label, directory, births in (("calibration", cal_source, CAL_BIRTHS),
                                                                         ("confirmation", confirm_source, CONFIRM_BIRTHS))},
                    checkpoint_step=step,
                    native_timing="Ordinary support, taste and structural commit run on every contact. Interpolate "
                                  "continuously surviving ancestral weights and all biases after the native write, "
                                  "then recompute exact native post-write ticks from the same precommit activity. "
                                  "The native graph support is used verbatim. Native nonzero weight floor applies.",
                    all_three="Actual sampled correct A, B and X motor choices from separate silent forks of the "
                              "same post-X state, with independent X laws on identical prefixes. A-before-X is a "
                              "silent query from the earlier prefix with the same A menu and draw.",
                    schedule="Random order of initial A and B, then one scheduled revisit slot per four contacts "
                             "alternating A and B in random initial order. B slots match exactly. The absent-A "
                             "ecology replaces every later A slot with a distinct new object. X never enters the prefix.",
                    external_cost=dict(dense_fp32_template_bytes_per_birth=4 * (net.NODES**2 + net.NODES),
                                       ancestry_bool_bytes_per_lane=net.NODES**2,
                                       note="Birth-reference uses a fixed stored numeric template from the starting "
                                            "uniform graph. The other references require no extra fixed numeric "
                                            "template. Prewrite uses the just-prior graph values. This intervention "
                                            "is evaluated externally with no learned consolidation component."))
    manifest_hash = _lock(path / "run.json", manifest)
    cal_variants = [("unchanged", 0.)] + [(reference, gain) for reference in REFERENCES for gain in grid[1:]]
    cal_tape = make_tape(cal_pairs, CAL_SEED, CAL_DOMAIN, max(AGES))
    cal, cal_fingerprints = _run_split(path / "calibration", cal_source, CAL_BIRTHS, cal_tape, cal_variants, AGES, step)
    chosen = select(cal, CAL_BIRTHS, AGES, grid)
    selection_hash = _lock(path / "selected-gains.json", {"manifest_sha256": manifest_hash, "selection": chosen,
                                                          "calibration_raw_sha256": cal_fingerprints})
    print("Frozen gains:", {reference: row["selected_gain"] for reference, row in chosen.items()}, flush=True)
    confirm_variants = [("unchanged", 0.)] + [(reference, row["selected_gain"]) for reference, row in chosen.items()]
    applied = [item for item in confirm_variants if item[1] or item[0] == "unchanged"]
    confirm_tape = make_tape(confirm_pairs, CONFIRM_SEED, CONFIRM_DOMAIN, max(AGES))
    learned, fingerprints = _run_split(path / "confirmation", confirm_source, CONFIRM_BIRTHS,
                                        confirm_tape, applied, AGES, step)
    results, paired = _confirm(learned, CONFIRM_BIRTHS, AGES, confirm_variants, confirm_pairs, bootstrap)
    report = dict(manifest_sha256=manifest_hash, selection_sha256=selection_hash, selection=chosen,
                  confirmation_raw_sha256=fingerprints, by_ecology=results, native_relative_contrasts=paired)
    _lock(path / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--calibration-source", type=Path, default=DATA / "retention-objective")
    parser.add_argument("--confirmation-source", type=Path, default=DATA / "retention-objective-confirmation")
    parser.add_argument("--calibration-pairs", type=int, default=96)
    parser.add_argument("--confirmation-pairs", type=int, default=192)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--checkpoint-step", type=int, default=8000)
    options = parser.parse_args()
    if min(options.calibration_pairs, options.confirmation_pairs, options.bootstrap) < 2:
        parser.error("Use at least two pairs and two bootstrap draws")
    run(options.out, options.plan, options.calibration_source, options.confirmation_source,
        options.calibration_pairs, options.confirmation_pairs, options.bootstrap, options.checkpoint_step)
