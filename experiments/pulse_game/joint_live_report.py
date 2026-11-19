"""Verify all physical edit archives, then summarize the paired factorial and controls."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from . import joint_live_integrated as integrated
from . import joint_live_model as lab, joint_live_study as study
from . import live_module_edits_study as original, ring_logic_study as io, ring_logic_world as sensory


FIELDS = {name: i for i, name in enumerate(lab.BIRTH_FIELDS)}
BOOTSTRAP = 20000


def quantiles(values):
    return np.quantile(values, [0., .25, .5, .75, 1.]).tolist() if len(values) else []


def summarize_newborns(retired, surviving):
    combined = np.concatenate((retired, surviving))
    values = {field: combined[:, i] for field, i in FIELDS.items()}
    before, after = values["initial_raw"], values["end_raw"]
    seen = values["later_receipts"] > 0
    learned = values["changed_receipts"] > 0
    return dict(born=len(combined), surviving=len(surviving), retired=len(retired), had_later_receipt=int(seen.sum()),
                changed=int(learned.sum()), surviving_changed=int(np.count_nonzero(surviving[:, FIELDS["changed_receipts"]])),
                retired_changed=int(np.count_nonzero(retired[:, FIELDS["changed_receipts"]])),
                sign_flips=int(np.count_nonzero(before * after < 0)),
                signed_raw_birth_quantiles=quantiles(before), signed_raw_end_quantiles=quantiles(after),
                abs_raw_birth_quantiles=quantiles(abs(before)), abs_raw_delta_quantiles=quantiles(abs(after-before)),
                abs_effective_delta_quantiles=quantiles(abs(after / np.sqrt(values["end_indegree"])
                                                             - before / np.sqrt(values["initial_indegree"]))),
                path_raw_quantiles=quantiles(values["path_raw"]), later_receipts_quantiles=quantiles(values["later_receipts"]),
                changed_receipts_quantiles=quantiles(values["changed_receipts"]),
                initial_indegree_quantiles=quantiles(values["initial_indegree"]), end_indegree_quantiles=quantiles(values["end_indegree"]),
                first_changed_after_trials_quantiles=quantiles(values["first_change"][learned] - values["trial"][learned]))


def verify_generation(record, birth, clock):
    trial, receipt, initial, degree, terminal, end_degree, age, changed, first, path = record
    np.testing.assert_array_equal(record[:4], birth[:4])
    if age != clock - receipt or not 0 <= changed <= age or end_degree < 1:
        raise RuntimeError("An edge generation has inconsistent age or degree")
    if changed:
        if first <= trial or path <= 0 or path < abs(terminal-initial) - 2e-5:
            raise RuntimeError("A newborn write predates birth or has an inconsistent path")
    elif first != -1 or path != 0 or terminal != initial:
        raise RuntimeError("An unmodified raw coefficient differs from its birth value")


def verify_arm(data, arm):
    field = lambda name: data[f"{arm}/{name}"]
    physical, edits = field("physical"), field("edits")
    records, terminal = field("event_records"), field("final_newborn_records")
    masks, weights = field("final_mask"), field("final_raw")
    first_masks, first_raw = data["initial_mask"], data["initial_raw"]
    if np.any(weights[~masks] != 0) or not np.isfinite(weights[masks]).all():
        raise RuntimeError("An inactive arc has a weight, or an active weight is non-finite")
    if np.any(field("weight_step")[physical[..., 2] == 0] != 0) or np.any(field("input_step")[physical[..., 2] == 0] != 0):
        raise RuntimeError("A weight changed without an actual signed receipt")
    if not np.array_equal(np.isfinite(records).all(axis=-1), edits >= 0):
        raise RuntimeError("Edit IDs and edit records disagree")
    retired = []
    surviving = []
    for i, j, k in np.ndindex(edits.shape[:3]):
        at = i, j, k
        head = i * len(sensory.RULES) + j
        mask = first_masks[head].copy()
        prior = first_raw[head].sum(axis=-1) / mask.sum(axis=-1)
        receipts = (physical[at][..., 2] != 0).cumsum()
        generations = {}
        events = edits[at]
        for t in np.flatnonzero((events >= 0).any(axis=-1)):
            add, remove = events[t]
            if add >= 0:
                target, source = divmod(int(add), lab.WIDTH)
                if source == target or mask.flat[add] or add == remove:
                    raise RuntimeError("Invalid newly grown arc")
                mask.flat[add] = True
                saved = records[at][t, 0]
                if not np.isclose(saved[2], prior[target]) or saved[0] != t or saved[1] != receipts[t]:
                    raise RuntimeError("A birth record differs from the registered initialization or clock")
                if saved[3] != mask[target].sum() - int(remove >= 0 and remove // lab.WIDTH == target):
                    raise RuntimeError("A birth used the wrong incident normalization")
                verify_generation(saved, saved, receipts[t])
                generations[add] = saved
            if remove >= 0:
                if not mask.flat[remove]:
                    raise RuntimeError("Retirement does not address an active arc")
                saved = records[at][t, 1]
                if saved[FIELDS["end_indegree"]] != mask[remove // lab.WIDTH].sum():
                    raise RuntimeError("A retirement used the wrong incident normalization")
                mask.flat[remove] = False
                if not original.lab.strongly_connected(lab.WIDTH, zip(*np.nonzero(mask.T), strict=True)):
                    raise RuntimeError("A retirement broke strong connectivity")
                if remove in generations:
                    verify_generation(saved, generations.pop(remove), receipts[t])
                    retired.append(saved)
        np.testing.assert_array_equal(mask, masks[at])
        recorded = terminal[at].reshape(-1, len(lab.BIRTH_FIELDS))
        active = np.array(sorted(generations), dtype=int)
        np.testing.assert_array_equal(active, np.flatnonzero(np.isfinite(recorded).all(axis=1)))
        np.testing.assert_array_equal(recorded[active, FIELDS["trial"]], field_for_lane(data, arm, "final_newborn_trial", at)[active])
        np.testing.assert_array_equal(recorded[active, FIELDS["initial_raw"]], field_for_lane(data, arm, "final_newborn_raw", at)[active])
        np.testing.assert_array_equal(recorded[active, FIELDS["end_raw"]], weights[at].ravel()[active])
        for arc, birth in generations.items():
            saved = recorded[arc]
            if saved[FIELDS["end_indegree"]] != mask[arc // lab.WIDTH].sum():
                raise RuntimeError("An endpoint used the wrong incident normalization")
            verify_generation(saved, birth, receipts[-1])
            surviving.append(saved)
    retired = np.asarray(retired).reshape(-1, len(lab.BIRTH_FIELDS))
    surviving = np.asarray(surviving).reshape(-1, len(lab.BIRTH_FIELDS))
    summary = summarize_newborns(retired, surviving)
    if summary["born"] != np.count_nonzero(edits[..., 0] >= 0):
        raise RuntimeError("Some newborn generations are missing")
    if arm in ("edge_only", "joint_pin_newborn") and summary["changed"]:
        raise RuntimeError("Pinned raw coefficients changed")
    return summary


def field_for_lane(data, arm, name, at):
    return data[f"{arm}/{name}"][at].ravel()


def contrasts(views, by_group, by_root, family, alpha):
    samples = {name: (views[hi] - views[lo]).mean(axis=1) for name, (hi, lo) in family.items()}
    if family == study.FACTORIAL:
        samples["interaction"] = samples["edits_given_weights"] - samples["edits_alone"]
    report = {}
    for name, sample in samples.items():
        draws = ((by_group @ sample) * by_root).sum(axis=-1)
        report[name] = dict(delta=float(sample.mean()), simultaneous_95pct=np.quantile(draws, (alpha, 1-alpha)).tolist(),
                            delta_by_group=sample.mean(axis=1).tolist())
    return report


def report(out, *, integrated_choice=False):
    source = integrated if integrated_choice else study
    recipe = source.sealed_recipe(out)
    registration = io.read(out / "registered.json")
    heads = io.read(out / "confirmed-heads.json")
    if heads["recipe_sha256"] != io.sha(out / "recipe.json"):
        raise RuntimeError("Scored starting heads are tied to a different recipe")
    rng = np.random.default_rng(2549413)
    g, r = len(registration["test_groups"]), len(registration["test_roots"])
    by_group = rng.multinomial(g, np.full(g, 1/g), size=BOOTSTRAP) / g
    by_root = rng.multinomial(r, np.full(r, 1/r), size=BOOTSTRAP) / r
    primary_count = sum(map(len, study.WINDOWS.values()))
    fields = ("physical", "edits", "event_records", "weight_step", "input_step")
    controls = ("neither", "joint_independent_sign")
    stores = {}
    result = {}
    for condition, keys in study.WINDOWS.items():
        with np.load(out / f"{condition}.npz", allow_pickle=False) as zipped:
            data = dict(zipped)
        shared, donors = {}, {}
        flips = data["flip_trial"]
        values = {arm: study.windows(data[f"{arm}/physical"], flips) for arm in study.ARMS}
        for arm in study.ARMS:
            steps = data[f"{arm}/weight_step"]
            input_steps = data[f"{arm}/input_step"]
            events = data[f"{arm}/edits"] >= 0
            final = data[f"{arm}/final_mask"].sum(axis=(-2, -1))
            physical = data[f"{arm}/physical"]
            if arm in ("neither", "edge_only") and (steps.any() or input_steps.any()):
                raise RuntimeError("A non-weight-learning arm changed an existing coefficient")
            shared[arm] = dict(accuracy={key: float(values[arm][key].mean()) for key in keys},
                accuracy_by_rule={rule: {key: float(values[arm][key][:, j].mean()) for key in keys}
                                  for j, rule in enumerate(sensory.RULES)},
                mean_grown=float(events[..., 0].sum(axis=-1).mean()), mean_retired=float(events[..., 1].sum(axis=-1).mean()),
                final_arcs=[int(final.min()), float(final.mean()), int(final.max())], timeout=float((physical[..., 0] < 0).mean()),
                changed_receipts_by_family=dict(zip(lab.WEIGHT_NAMES, np.count_nonzero(steps, axis=(0, 1, 2, 3)).tolist(), strict=True)),
                changed_receipts_by_input=dict(zip(("odor0", "odor1", "odor2", "marker", "movement"),
                    np.count_nonzero(input_steps, axis=(0, 1, 2, 3)).tolist(), strict=True)),
                newborn=verify_arm(data, arm))
            donors[arm] = events
        result[condition] = dict(arms=shared, windows={}, random_event_matching=dict(
            matched=int(np.count_nonzero(donors["joint"] & donors["joint_random"])),
            missed=int(np.count_nonzero(donors["joint"] & ~donors["joint_random"])),
            extra=int(np.count_nonzero(~donors["joint"] & donors["joint_random"]))))
        for key in keys:
            views = {arm: values[arm][key] for arm in study.ARMS}
            result[condition]["windows"][key] = dict(
                factorial=contrasts(views, by_group, by_root, study.FACTORIAL, .05 / (2 * primary_count * (len(study.FACTORIAL) + 1))),
                controls=contrasts(views, by_group, by_root, study.CONTROLS, .05 / (2 * primary_count * len(study.CONTROLS))))
        stores[condition] = {key: value for key, value in data.items() if key in ("pairs", "flip_trial", "initial_mask", "initial_raw", "placebo_sign")
                             or key.rsplit("/", 1)[-1] in fields or key.split("/", 1)[0] in controls}
    for case, ref, offset in (("reversal", "stable", 0), ("return", "reversal", 40)):
        x, y = stores[case], stores[ref]
        for key in ("pairs", "flip_trial", "initial_mask", "initial_raw", "placebo_sign"):
            np.testing.assert_array_equal(x[key], y[key])
        for root, turn in enumerate(x["flip_trial"]):
            for arm in study.ARMS:
                for field in fields:
                    key = f"{arm}/{field}"
                    np.testing.assert_array_equal(x[key][:, :, root, :turn+offset], y[key][:, :, root, :turn+offset])
        for key in x:
            if key.split("/", 1)[0] in controls:
                focus = (..., slice(None, 2)) if key.endswith("/physical") else (...,)
                np.testing.assert_array_equal(x[key][focus], y[key][focus])
    global_heads = original.verify_global_heads(dict(dev=io.read(out / "development.json")["checkpoints"], heldout=heads["checkpoints"]))
    summary = dict(recipe=recipe, results=result, groups=registration["test_groups"], roots=registration["test_roots"],
        scores_sha256={case: io.sha(out / f"{case}.npz") for case in study.WINDOWS},
        verification=dict(reference_neither_and_edge_only_equal_at_confirmation=True, every_edit_and_final_support_reconstructed=True,
            every_generation_clock_and_raw_write_audited=True, no_weight_writes_without_actual_receipt=True,
            frozen_and_placebo_behaviors_independent_of_hidden_changes=True, all_prechange_histories_equal=True,
            global_birth_heads=global_heads))
    io.write(out / "summary.json", summary)


def package(out, evidence):
    evidence.mkdir(parents=True, exist_ok=True)
    stage_files = ("registered.json", "development.json", "recipe.json", "confirmed-heads.json", "summary.json")
    with ZipFile(evidence / "raw.zip", "w", compression=ZIP_STORED) as compressed:
        for name in (*stage_files, *(f"{case}.npz" for case in study.WINDOWS)):
            compressed.write(out / name, name)
    for name in stage_files:
        copyfile(out / name, evidence / name)
    io.write(evidence / "provenance.json", {name: io.sha(evidence / name) for name in (*stage_files, "raw.zip")})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=integrated.OUT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--weight-selected", action="store_true")
    args = parser.parse_args()
    report(args.out, integrated_choice=not args.weight_selected)
    if args.evidence is not None:
        package(args.out, args.evidence)


if __name__ == "__main__":
    main()
