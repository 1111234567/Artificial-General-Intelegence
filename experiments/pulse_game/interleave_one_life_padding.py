"""Freeze one-life models and separate contact history from elapsed ticks."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_capacity_report as ci
from . import interleave_geometry_eval as geometry, interleave_one_life as study
from . import interleave_train as joint, interleave_triangle_train as tri, local_trace_eval as prior


net = tri.native
SEED = 8196581
BRANCHES = ("short", "pre_neutral", "pre_sham", "pre_write", "post_neutral", "post_sham", "post_write",
            "pre_neutral_replay", "pre_sham_replay", "pre_write_replay", "interleaved_sham", "interleaved_write")
METRICS = ("A", "B", "both", "AB_study_side_mismatch", "AB_study_taste_mismatch")
COHORTS = {"original": (list(range(731, 737)), "original"),
           "independent": (list(range(811, 817)), "confirmation")}
MODELS = ("parent", *study.ARM)
GEOMETRIES = ("ordinary", "triangle_pair")
EXTRA = 5
PROBE_WAIT = 4


def paired_inputs(seed, worlds):
    ordinary = cap.make_tape("odor", seed, worlds, 1)
    rng = np.random.default_rng([724183, seed])
    triad = tri.rotation_keys(rng, worlds, -.5)
    axis = torch.from_numpy(triad[:, :2])
    old = ordinary.directions.repeat_interleave(4, dim=0)
    sign = (ordinary.left * old[:, :, None]).sum(dim=-1).sign()
    left = sign[..., None] * axis.repeat_interleave(4, dim=0)[:, :, None]
    triangular = replace(ordinary, directions=axis, left=left, right=-left)

    scents = prior.directions(rng, 2 * worlds).reshape(worlds, 2, 3)
    cue_idx = rng.permuted(np.tile(np.arange(3), (worlds, 1)), axis=1)[:, :2]
    cue = np.zeros((worlds, EXTRA, 3), dtype=np.float32)
    cue[:, 1:3] = joint.CUE_BANK[cue_idx]
    vectors = np.concatenate((scents[:, :1], np.repeat(scents[:, 1:2], 2, axis=1),
                              np.broadcast_to(joint.MARK_AXIS, (worlds, 2, 3))), axis=1)
    orientation = prior.signs(rng, (worlds, EXTRA))
    orientation[:, 3:] = (1, -1)
    orient_vec = vectors * orientation[..., None]
    filler_right, filler_left = cue + orient_vec, cue - orient_vec
    filler_left[:, 3:] = filler_right[:, 3:]
    hidden = prior.signs(rng, (worlds, 4))[:, [0, 1, 2, 3, 3]] * orientation
    const, factor = np.zeros_like(hidden), hidden.copy()
    const[:, 3:], factor[:, 3:] = hidden[:, 3:], 0
    shuffled = rng.permuted(np.tile(np.arange(EXTRA), (worlds, 1)), axis=1)
    values = dict(left=filler_left, right=filler_right, constant=const, factor=factor,
                  draw=rng.random((worlds, EXTRA, 1 + net.EDIT_DRAWS), dtype=np.float32),
                  wait=rng.integers(0, 9, size=(worlds, EXTRA)), object=np.broadcast_to(np.arange(EXTRA), (worlds, EXTRA)))
    filler = {k: torch.from_numpy(v[np.arange(worlds)[:, None], shuffled].repeat(4, axis=0).copy())
              for k, v in values.items()}
    return {"ordinary": ordinary, "triangle_pair": triangular}, filler


def contact(rule, start, left, right, draw, edit, factor, const, *, write, force=None):
    act = net.act(rule, start, left, right, draw, side_override=force)
    taste = act.side * factor + const
    result = net.commit(rule, net.observe(rule, act, left, right, taste), edit,
                        intervention="active" if write else "sham")
    return result.state, torch.stack((act.side, taste, act.motor_logit))


def focal(rule, state, tape, index, force=None):
    if force is None:
        next_state, logs = cap._study(rule, state, tape, index, write=True)
        return next_state, logs[:, 0]
    return contact(rule, state, tape.left[:, index, 0], tape.right[:, index, 0], tape.motor_draw[:, index, 0],
                   tape.edit_draw[:, index, 0], tape.right_value[:, index, 0], tape.constant_value[:, index, 0],
                   write=True, force=force)


def foreign(rule, state, filler, index, write):
    s = net.wait(rule, state, filler["wait"][:, index], max_ticks=8)
    u = filler["draw"][:, index]
    return contact(rule, s, filler["left"][:, index], filler["right"][:, index], u[:, 0], u[:, 1:],
                   filler["factor"][:, index], filler["constant"][:, index], write=write)


def read(rule, state, tape):
    snapshot = net.wait(rule, state, PROBE_WAIT)
    probes = [cap._probe(rule, snapshot, tape, j) for j in (0, 1)]
    return torch.stack([a.side for a in probes]), torch.stack([a.motor_logit for a in probes])


@torch.no_grad()
def short(graph, rule, tape):
    initial = net.start(graph, tape.lanes)
    a, first = focal(rule, initial, tape, 0)
    ab, second = focal(rule, a, tape, 1)
    return ab, torch.stack((first, second))


@torch.no_grad()
def evaluate(graph, rule, tape, filler):
    rule = prior.birth_rule(rule)
    cold = net.start(graph, tape.lanes)
    ab, visits = short(graph, rule, tape)
    duration = cap.CONTACT_TICKS * EXTRA + filler["wait"].sum(1)
    filler_empty = visits.new_full((EXTRA, visits.shape[1], visits.shape[2]), torch.nan)
    states = {"short": (ab, visits, filler_empty)}

    def extend(state, mode):
        if mode == "neutral":
            return net.wait(rule, state, duration, max_ticks=EXTRA * (cap.CONTACT_TICKS + 8)), filler_empty
        start = state
        log = []
        for j in range(EXTRA):
            state, seen = foreign(rule, state, filler, j, write=mode == "write")
            log.append(seen)
        if mode == "sham" and not cap._same_graph(state.graph, start.graph):
            raise AssertionError("Write-disabled filler changed the parameters")
        return state, torch.stack(log)

    for mode in ("neutral", "sham", "write"):
        pre, pre_log = extend(cold, mode)
        post, post_log = extend(ab, mode)
        states[f"post_{mode}"] = post, visits, post_log
        for replay in (False, True):
            a, r1 = focal(rule, pre, tape, 0, force=visits[0, 0] if replay else None)
            b, r2 = focal(rule, a, tape, 1, force=visits[1, 0] if replay else None)
            pair = torch.stack((r1, r2))
            if replay and not torch.equal(pair[:, :2], visits[:, :2]):
                raise AssertionError("Replayed study must recreate both sides and tastes")
            states[f"pre_{mode}" + ("_replay" if replay else "")] = b, pair, pre_log

    for write in (False, True):
        state, fillers, pair = cold, [], []
        for label, slot in (('x', 0), ('x', 1), ('ab', 0), ('x', 2), ('x', 3), ('ab', 1), ('x', 4)):
            if label == 'ab':
                state, obs = focal(rule, state, tape, slot)
                pair.append(obs)
            else:
                state, obs = foreign(rule, state, filler, slot, write=write)
                fillers.append(obs)
        states["interleaved_" + ("write" if write else "sham")] = state, torch.stack(pair), torch.stack(fillers)

    raw = {"truth": tape.right_value[:, :, -1].T.numpy(), "extra_ticks": duration.numpy()}
    values = [read(rule, states[name][0], tape) for name in BRANCHES]
    raw.update(side=torch.stack([value[0] for value in values]).numpy().astype(np.int8),
               logit=torch.stack([value[1] for value in values]).numpy(),
               AB_contact=torch.stack([states[name][1] for name in BRANCHES]).numpy(),
               filler_contact=torch.stack([states[name][2] for name in BRANCHES]).numpy())
    return raw


def per_world(raw, worlds):
    ok = raw["side"] == raw["truth"][None]
    changes = raw["AB_contact"][:, :, :2] != raw["AB_contact"][:1, :, :2]
    values = (ok[:, 0], ok[:, 1], ok.all(1), changes[:, :, 0].any(1), changes[:, :, 1].any(1))
    return {name: val.reshape(len(BRANCHES), worlds, 4).mean(2) for name, val in zip(METRICS, values, strict=True)}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load(cohort, model, birth):
    if model == "parent":
        path, arm = geometry.sources(cohort, "hard", 64000)
        step = 64000
    else:
        path, arm, step = study.DEST / cohort, model, study.STEPS
    return (tri.old_fit.load_checkpoint(path, "odor", arm, birth, step),
            tri.old_fit.checkpoint_path(path, "odor", arm, birth, step), step)


@torch.no_grad()
def audit_production(graph, rule, cohort, model, birth, step, archived, archived_seed, worlds):
    tape = cap.make_tape("odor", archived_seed, worlds, 1)
    baseline = cap.evaluate(graph, rule, tape, (0, 1), snapshot_worlds=0)
    birth_rule = prior.birth_rule(rule)
    state, contacts = short(graph, birth_rule, tape)
    side, logit = read(birth_rule, state, tape)
    for new, old in ((side.numpy().astype(np.int8), baseline["side"][0, 1, :2]),
                     (logit.numpy(), baseline["logit"][0, 1, :2]),
                     (contacts.numpy(), baseline["contact"][0, :2, :, 0])):
        if new.dtype != old.dtype or new.tobytes() != old.tobytes():
            raise AssertionError("The short branch changed production actions, tastes, or logits")
    score = (side.numpy() == tape.right_value[:, :, -1].T.numpy()).all(0).reshape(worlds, 4).mean(1)
    raw = tri.DATA / "one-life-eval" / cohort / "shared" / f"{model}-{birth}-{step}.npz"
    if digest(raw) != archived["hashes"][raw.relative_to(tri.DATA).as_posix()]:
        raise ValueError(f"Production evidence changed for {raw}")
    with np.load(raw, allow_pickle=False) as bank:
        np.testing.assert_array_equal(score, bank["card"][:, geometry.CARD_KEYS.index("all_2")])


def run(worlds, raw, summary):
    torch.set_num_threads(1)
    tapes, filler = paired_inputs(SEED, worlds)
    duration = (cap.CONTACT_TICKS * EXTRA + filler["wait"][::4].sum(1)).numpy()
    base_ticks = 2 * cap.CONTACT_TICKS + PROBE_WAIT
    archive = dict(seed=SEED, worlds_per_birth=worlds, lanes_per_world=4, branches=list(BRANCHES),
                   bootstrap="4000 paired crossed resamples of independent births and shared geometries; four private lanes averaged first.",
                   timing=dict(contact_ticks=cap.CONTACT_TICKS, probe_wait=PROBE_WAIT,
                               short_total=base_ticks,
                               padding_min=int(duration.min()), padding_max=int(duration.max()),
                               padding_mean=float(duration.mean()), padded_total_min=int(duration.min()) + base_ticks,
                               padded_total_max=int(duration.max()) + base_ticks),
                   protocol="A/B keep the ordinary capacity menus, private laws, motor and edit draws, and four-tick terminal wait. "
                            "The triangle-pair variant changes the A/B vectors only. Five independently valued fillers use an "
                            "unrelated odor, one unrelated odor with two visible cues, and two feeder demonstrations. "
                            "Matched neutral branches advance the same ticks as actual filler contacts including each 0–8 gap. "
                            "Native supports use their own actions; the pre-replay arms force only A/B to the sampled short-branch "
                            "side and thus expose the exact same tastes while still writing normally. Native A/B probes fork the "
                            "same endpoint and receive no taste or edit. This is exploratory frozen testing after training.",
                   source_sha256=digest(Path(__file__)), cohorts={})
    for label, (births, card_tag) in COHORTS.items():
        boot = ci.Crossed(len(births), worlds, SEED + 3, 4000)
        old_card = json.loads((tri.EVIDENCE / f"model-cards-{card_tag}.json").read_text(encoding="utf8"))
        prod = json.loads((tri.EVIDENCE / f"one-life-{label}-shared-8000.json").read_text(encoding="utf8"))
        arrays = {geom: {} for geom in GEOMETRIES}
        proof, hashes = {}, {}
        for model in MODELS:
            cells = {geom: [] for geom in GEOMETRIES}
            for birth in births:
                (graph, rule), checkpoint, step = load(label, model, birth)
                content = digest(checkpoint)
                if content != prod["hashes"][checkpoint.relative_to(tri.DATA).as_posix()]:
                    raise ValueError("The selected checkpoint differs from the production trial")
                audit_production(graph, rule, label, model, birth, step, prod, old_card["seed"], old_card["worlds"])
                proof[f"{model}/{birth}"] = content
                for geom, tape in tapes.items():
                    result = evaluate(graph, rule, tape, filler)
                    cells[geom].append(per_world(result, worlds))
                    fname = f"{label}/{geom}/{model}-{birth}.npz"
                    observed = {**result, **{"AB/" + k: v.numpy() for k, v in vars(tape).items() if torch.is_tensor(v)},
                                **{"filler/" + k: v.numpy() for k, v in filler.items()}}
                    tri.revisit._write_raw(raw / fname, **observed)
                    hashes[fname] = digest(raw / fname)
                print(label, model, birth, "production bitwise verified", flush=True)
            for geom in GEOMETRIES:
                arrays[geom][model] = {name: np.stack([row[name] for row in cells[geom]]) for name in METRICS}
        measured = {}
        conds = {name: i for i, name in enumerate(BRANCHES)}
        contrasts = [(arm, "short") for arm in BRANCHES[1:]] + [
            (f"{pos}_{a}", f"{pos}_{b}") for pos in ("pre", "post") for a, b in (("write", "sham"), ("sham", "neutral"))]
        for geom, models in arrays.items():
            cells = {m: {arm: {key: boot(v[:, i]) for key, v in row.items()} for arm, i in conds.items()}
                     for m, row in models.items()}
            stats = {m: {f"{left} minus {right}": {key: boot(v[:, conds[left]] - v[:, conds[right]])
                                                   for key, v in row.items()} for left, right in contrasts}
                     for m, row in models.items()}
            parents = {f"{m} minus parent": {arm: boot(models[m]["both"][:, i] - models["parent"]["both"][:, i])
                                               for arm, i in conds.items()} for m in study.ARM}
            measured[geom] = dict(cells=cells, within_model=stats, joint_minus_parent=parents)
        archive["cohorts"][label] = dict(births=births, production_bitwise=proof, raw_sha256=hashes, results=measured)
        tri.revisit._write_json(summary, archive)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=192)
    parser.add_argument("--raw", type=Path, default=tri.DATA / "one-life-padding")
    parser.add_argument("--output", type=Path, default=tri.EVIDENCE / "one-life-padding.json")
    args = parser.parse_args()
    run(args.worlds, args.raw, args.output)
