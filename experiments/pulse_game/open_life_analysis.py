"""Verify registered physical lives and summarize paired capability endpoints."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import numpy as np

from . import joint_live_model as joint, joint_live_report as report
from . import open_life_study as study, ring_logic_study as io, ring_logic_world as sensory, world
from .free_graph_model import strongly_connected


def verify_arm(data, arm):
    field = lambda name: data[f"{arm}/{name}"]
    physical, edits, records = field("physical"), field("edits"), field("event_records")
    masks, raw, end = field("final_mask"), field("final_raw"), field("final_newborn_records")
    observed = physical[..., 2] != 0
    if np.any(raw[~masks] != 0) or not np.isfinite(raw[masks]).all():
        raise RuntimeError("An arc has an inconsistent coefficient")
    for name in ("weight_step", "input_step") + (() if arm == "legacy" else ("fast_step",)):
        if np.any(field(name)[~observed] != 0) or not np.isfinite(field(name)).all():
            raise RuntimeError(f"Non-physical or non-finite update to {name}")
    if not np.array_equal(np.isfinite(records).all(axis=-1), edits >= 0):
        raise RuntimeError("Edit IDs and records disagree")
    if np.any((edits >= 0).any(axis=-1) & ~observed):
        raise RuntimeError("Structure changed without own physical contact")
    if arm in ("head_only", "effective") and np.any(field("fast_step")[..., 1] != 0):
        raise RuntimeError("A contextual ablation used contextual memory")
    if arm == "effective" and np.any(field("fast_step")[..., 0] != 0):
        raise RuntimeError("The effective comparator changed its head coefficient")
    retired, surviving = [], []
    for g, rule, r in np.ndindex(edits.shape[:3]):
        at = g, rule, r
        head = g * len(sensory.RULES) + rule
        mask = data["initial_mask"][head].copy()
        receipts = observed[at].cumsum()
        generations = {}
        for t in np.flatnonzero((edits[at] >= 0).any(axis=-1)):
            birth, retirement = edits[at][t]
            if birth >= 0:
                target, source = divmod(int(birth), joint.WIDTH)
                if source == target or mask.flat[birth] or birth == retirement:
                    raise RuntimeError("Illegal growth coordinates")
                mask.flat[birth] = True
                event = records[at][t, 0]
                expected_degree = mask[target].sum()
                if arm == "legacy" and retirement >= 0 and retirement // joint.WIDTH == target:
                    expected_degree -= 1
                if event[0] != t or event[1] != receipts[t] or event[3] != expected_degree:
                    raise RuntimeError("Growth record disagrees with the physical edit")
                report.verify_generation(event, event, receipts[t])
                generations[birth] = event
            if retirement >= 0:
                if not mask.flat[retirement]:
                    raise RuntimeError("Retirement addresses an inactive arc")
                event = records[at][t, 1]
                if event[5] != mask[retirement // joint.WIDTH].sum():
                    raise RuntimeError("Retirement used the wrong degree")
                mask.flat[retirement] = False
                if not strongly_connected(joint.WIDTH, zip(*np.nonzero(mask.T), strict=True)):
                    raise RuntimeError("Retirement broke strong connectivity")
                if retirement in generations:
                    report.verify_generation(event, generations.pop(retirement), receipts[t])
                    retired.append(event)
        np.testing.assert_array_equal(mask, masks[at])
        saved = end[at].reshape(-1, len(joint.BIRTH_FIELDS))
        active = np.array(sorted(generations), dtype=int)
        np.testing.assert_array_equal(active, np.flatnonzero(np.isfinite(saved).all(axis=1)))
        np.testing.assert_array_equal(saved[active, 0], field("final_newborn_trial")[at].ravel()[active])
        np.testing.assert_array_equal(saved[active, 2], field("final_newborn_raw")[at].ravel()[active])
        np.testing.assert_array_equal(saved[active, 4], raw[at].ravel()[active])
        for arc, event in generations.items():
            terminal = saved[arc]
            if terminal[5] != mask[arc // joint.WIDTH].sum():
                raise RuntimeError("Terminal incident degree differs from edge record")
            report.verify_generation(terminal, event, receipts[-1])
            surviving.append(terminal)
    retired = np.asarray(retired).reshape(-1, len(joint.BIRTH_FIELDS))
    surviving = np.asarray(surviving).reshape(-1, len(joint.BIRTH_FIELDS))
    summary = report.summarize_newborns(retired, surviving)
    if summary["born"] != np.count_nonzero(edits[..., 0] >= 0):
        raise RuntimeError("An edge did not appear in generation statistics")
    # In the four effective-weight arms raw already represents synaptic efficacy.
    if arm != "legacy":
        del summary["abs_effective_delta_quantiles"]
    return summary


def arm_summary(data, arm, condition, outcome):
    prefix = f"{arm}/"
    field = lambda name: data[prefix + name]
    physical = field("physical")
    final = field("final_mask").sum(axis=(-2, -1))
    replays = joint_windows(physical, data["flip_trial"]) if condition == "reversal" else {}
    replays |= outcome
    summary = dict(
        accuracy={window: float(values.mean()) for window, values in replays.items()},
        accuracy_by_rule={rule: {key: float(value[:, i].mean()) for key, value in replays.items()}
                          for i, rule in enumerate(sensory.RULES)},
        timeout=float((physical[..., 0] < 0).mean()),
        grown=int(np.count_nonzero(field("edits")[..., 0] >= 0)),
        retired=int(np.count_nonzero(field("edits")[..., 1] >= 0)),
        final_edges=[int(final.min()), float(final.mean()), int(final.max())],
        weight_writes=dict(zip(joint.WEIGHT_NAMES, np.count_nonzero(field("weight_step"), axis=(0, 1, 2, 3)).tolist(), strict=True)),
        newborns=verify_arm(data, arm))
    if arm != "legacy":
        summary["fast"] = dict(
            writes=dict(zip(study.model.FAST_FIELDS[:2], np.count_nonzero(field("fast_step")[..., :2], axis=(0, 1, 2, 3)).tolist(), strict=True)),
            end_head_quantiles=report.quantiles(field("final_head")),
            late_context_mass_quantiles=report.quantiles(field("fast_step")[..., -20:, 5][physical[..., -20:, 2] != 0]))
    return summary


def joint_windows(physical, flip):
    correct = physical[..., 2] > 0
    return {label: np.stack([correct[..., i, t+start:t+end].mean(axis=-1)
                            for i, t in enumerate(flip)], axis=-1)
            for label, (start, end) in {"before": (-20, 0), "first20": (0, 20)}.items()}


def verify_physical(physical, right):
    side, tick, taste = np.moveaxis(physical, -1, 0)
    expected = np.where(side < 0, 0, np.where(side == right, 1, -1))
    np.testing.assert_array_equal(taste, expected)
    if np.any(~np.isin(side, (-1, 0, 1))) or np.any(tick[side < 0] != -1):
        raise RuntimeError("A physical contact record has an invalid side or timeout")
    if np.any((tick[side >= 0] < 0) | (tick[side >= 0] >= world.TICKS)):
        raise RuntimeError("A physical contact occurred outside the allowed ticks")


def assert_common_prefix(left, right, offset):
    for name in ("initial_mask", "initial_raw", "pairs", "placebo_sign", "flip_trial"):
        np.testing.assert_array_equal(left[name], right[name])
    for i, cut in enumerate(left["flip_trial"] + offset):
        for arm in study.ARMS:
            names = ("physical", "edits", "event_records", "weight_step", "input_step")
            if arm != "legacy":
                names += ("fast_step",)
            for name in names:
                key = f"{arm}/{name}"
                np.testing.assert_array_equal(left[key][:, :, i, :cut], right[key][:, :, i, :cut])


def analyze(out, evidence):
    config = study.sealed_recipe(out)
    confirmation = io.read(out / "confirmation.json")
    if confirmation["heads_sha256"] != io.sha(out / "confirmed-heads.json"):
        raise RuntimeError("Confirmation head manifest changed")
    heads = io.read(out / "confirmed-heads.json")
    g, r = len(study.TEST_GROUPS), len(study.TEST_ROOTS)
    rng = np.random.default_rng(2026111903)
    by_group = rng.multinomial(g, np.full(g, 1/g), size=study.BOOTSTRAP) / g
    by_root = rng.multinomial(r, np.full(r, 1/r), size=study.BOOTSTRAP) / r
    alpha = .05 / (2 * 25)
    snapshots = {}
    physical = {}
    for case in study.CONDITIONS:
        path = out / f"{case}.npz"
        if confirmation["files"][case] != io.sha(path):
            raise RuntimeError("A registered physical archive was modified")
        with np.load(path, allow_pickle=False) as saved:
            data = dict(saved)
        tape_parts, change = study.tapes(study.TEST_ROOTS, heads["rows"], case, "confirmation")
        tape = tape_parts[0]
        np.testing.assert_array_equal(data["pairs"], tape_parts[3])
        np.testing.assert_array_equal(data["flip_trial"], tape_parts[4])
        np.testing.assert_array_equal(data["placebo_sign"], tape_parts[5])
        np.testing.assert_array_equal(data["changed"], change)
        for arm in study.ARMS:
            physical_shape = data[f"{arm}/physical"].shape[:-1]
            verify_physical(data[f"{arm}/physical"], (tape.good_id != tape.left_id).reshape(physical_shape))
        values = {arm: study.views(data[f"{arm}/physical"], case, data["changed"]) for arm in study.ARMS}
        compare = {}
        for window in values["adaptive"]:
            primary = {arm: values["adaptive"][window] - values[arm][window] for arm in study.ARMS[1:]}
            primary["chance"] = values["adaptive"][window] - .5
            compare[window] = {}
            for arm, difference in primary.items():
                paired = difference.mean(axis=1)
                draws = ((by_group @ paired) * by_root).sum(axis=-1)
                compare[window][arm] = dict(delta=float(paired.mean()),
                                            simultaneous_95pct=np.quantile(draws, (alpha, 1-alpha)).tolist())
        summaries = {arm: arm_summary(data, arm, case, values[arm]) for arm in study.ARMS}
        snapshots[case] = dict(arms=summaries, contrasts=compare)
        if case != "partial":
            physical[case] = {name: value for name, value in data.items() if not name.split("/")[-1].startswith("final_")}
        print("verified", case, flush=True)
    assert_common_prefix(physical["stable"], physical["reversal"], 0)
    assert_common_prefix(physical["reversal"], physical["return"], 40)
    decision = {
        "stable_retention": snapshots["stable"]["contrasts"]["late"]["legacy"]["simultaneous_95pct"][0] > 0,
        "learns_reversal": snapshots["reversal"]["contrasts"]["late"]["chance"]["simultaneous_95pct"][0] > 0,
        "successful_return": snapshots["return"]["contrasts"]["late"]["legacy"]["simultaneous_95pct"][0] > 0,
        "conditional_adaptation": snapshots["partial"]["contrasts"]["changed"]["head_only"]["simultaneous_95pct"][0] > 0,
    }
    output = dict(config=config, cases=snapshots, decision=decision, physical_lives_per_case_per_arm=g * r * len(sensory.RULES),
                  bootstrap_draws=study.BOOTSTRAP, matched_prefixes_verified=True, confirmation_sha256=io.sha(out / "confirmation.json"))
    io.write(out / "summary.json", output)
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("registered", "development", "recipe", "validation", "confirmed-heads", "confirmation", "summary"):
        filename = f"{name}.json"
        copyfile(out / filename, evidence / filename)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE)
    args = parser.parse_args()
    analyze(args.out, args.evidence)


if __name__ == "__main__":
    main()
