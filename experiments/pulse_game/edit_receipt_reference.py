"""Score public-observation memory rules and short native deployments on frozen worlds."""

from argparse import ArgumentParser
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_eval as ev
from . import edit_receipt_graph as native
from . import edit_receipt_train as train


REFERENCES = {"empirical_all": None, "empirical_recent3": 3}
MODE_FIELDS = ("side", "taste", "correct", "p_correct", "applied", "accepted")


class TasteHistory:
    """An exact observed odor pair indexes signed evidence from past contacts."""

    def __init__(self, limit):
        self.limit = limit
        self.evidence = {}

    def act(self, left, right, uniform):
        a, b = tuple(left), tuple(right)
        if b < a:
            key, orientation = (b, a), -1
        else:
            key, orientation = (a, b), 1
        score = sum(self.evidence.get(key, ()))
        if score == 0:
            side = 1 if uniform < .5 else -1
        else:
            side = orientation if score > 0 else -orientation
        return key, orientation, side

    def receive(self, key, orientation, side, taste):
        if key not in self.evidence:
            self.evidence[key] = deque(maxlen=self.limit)
        self.evidence[key].append(int(orientation * side * taste))


def empirical_deploy(tape, memories, *, feedback=True):
    """The policy receives ordered odor vectors and its own realized taste only."""
    left = tape.left.numpy()
    truth, outcome = tape.truth.numpy(), tape.outcome.numpy()
    uniform = tape.draws[:, :, 2].numpy()
    sides, tastes = np.empty_like(truth, dtype=np.int8), np.empty_like(truth, dtype=np.int8)
    for world, memory in enumerate(memories):
        for contact in range(tape.length):
            odor = left[world, contact]
            key, orientation, side = memory.act(odor, -odor, uniform[world, contact])
            taste = int(side * outcome[world, contact])
            sides[world, contact], tastes[world, contact] = side, taste
            if feedback:
                memory.receive(key, orientation, side, taste)
    return {"side": sides, "taste": tastes, "correct": sides == truth}


def new_memories(tape, limit):
    return [TasteHistory(limit) for _ in range(len(tape.truth))]


def load_worlds(root, info):
    path = root / info["file"]
    if ev.digest(path) != info["sha256"]:
        raise ValueError(f"Frozen world file failed its digest check: {path}")
    worlds = {}
    splits = {"deployment": ev.SCENARIOS, "fork_holdout": ev.SCENARIOS,
              "return": ("A_first", "C_first", "B", "A_return"), "fork_validation": ("mixed",)}
    with np.load(path, allow_pickle=False) as saved:
        for split, names in splits.items():
            worlds[split] = {}
            for name in names:
                prefix = f"{split}/{name}"
                fields = (torch.from_numpy(saved[f"{prefix}/{field}"])
                          for field in ("left", "truth", "outcome", "draws", "cue"))
                worlds[split][name] = ev.Tape(*fields)
    return worlds


def empirical_records(worlds, settings):
    data = {}
    for name, limit in REFERENCES.items():
        for split in ("deployment", "fork_holdout", "fork_validation"):
            for condition, tape in worlds[split].items():
                result = empirical_deploy(tape, new_memories(tape, limit))
                ev.prefix(data, f"{split}/{condition}/{name}", result)
        phases = worlds["return"]
        return_tape = phases["A_return"]
        for history, first in (("ABA", "A_first"), ("CBA", "C_first"), ("B_only", None)):
            memories = new_memories(return_tape, limit)
            if first is not None:
                empirical_deploy(phases[first], memories)
            empirical_deploy(phases["B"], memories)
            early = empirical_deploy(return_tape.window(0, settings.early), memories, feedback=False)
            actual = empirical_deploy(return_tape, memories)
            ev.prefix(data, f"return/{history}/{name}", actual)
            ev.prefix(data, f"return_pre_feedback/{history}/{name}", early)
    return data


@torch.no_grad()
def native_records(run_dir, tape, birth, arm, checkpoints, config):
    records = {}
    policy = ev.policy_settings(arm, config)
    for step in checkpoints:
        graph = train.load_checkpoint(run_dir, arm, birth, step)
        initial = native.start(graph, len(tape.truth))
        for mode, intervention in ev.WRITE_MODES.items():
            _, observation, _ = ev.deploy(initial, tape, policy, intervention=intervention)
            ev.prefix(records, f"short_native/{arm}/{step}/{mode}",
                      {key: observation[key] for key in MODE_FIELDS})
    return records


def fit_short(run_dir, evaluation, out, protocol, birth, arms, checkpoints):
    torch.set_num_threads(1)
    settings = ev.Settings(**protocol["settings"])
    worlds = load_worlds(evaluation, protocol["tapes"][str(birth)])
    records = empirical_records(worlds, settings)
    for arm in arms:
        records.update(native_records(run_dir, worlds["fork_validation"]["mixed"], birth, arm,
                                      checkpoints, protocol["training_config"]))
    path = out / f"birth_{birth}" / "reference-records.npz"
    train.write_arrays(path, **records)
    print(f"scored reference and short deployments for birth={birth}", flush=True)
    return {"file": path.relative_to(out).as_posix(), "sha256": ev.digest(path)}


def reference_scores(read, protocol, births):
    settings = ev.Settings(**protocol["settings"])
    indices = {"deployment": ev.boot_pairs(len(births), settings.worlds, settings.bootstrap_draws, settings.seed),
               "fork_holdout": ev.boot_pairs(len(births), settings.holdout_worlds,
                                             settings.bootstrap_draws, settings.seed + 1),
               "fork_validation": ev.boot_pairs(len(births), settings.holdout_worlds,
                                                settings.bootstrap_draws, settings.seed + 2)}
    contacts = protocol["training_config"]["contacts"]
    short_windows = {"contact_1": (0, 1), "first_half": (0, contacts // 2),
                     "last_half": (contacts // 2, contacts), "all": (0, contacts)}
    past_return = {"contact_1": (0, 1), "early": (0, settings.early),
                   "late": (settings.phase_contacts - settings.late, settings.phase_contacts)}
    tables = {}
    for split, conditions in (("deployment", ev.SCENARIOS), ("fork_holdout", ev.SCENARIOS),
                              ("fork_validation", ("mixed",))):
        tables[split] = {}
        for condition in conditions:
            accuracy = {name: read(f"{split}/{condition}/{name}/correct") for name in REFERENCES}
            accuracy["recent3_minus_all"] = accuracy["empirical_recent3"] - accuracy["empirical_all"]
            windows = short_windows if split == "fork_validation" else ev.windows(settings, condition)
            tables[split][condition] = ev.paired_accuracy_scores(accuracy, windows, indices[split])
    for group in ("return", "return_pre_feedback"):
        tables[group] = {}
        windows = past_return if group == "return" else {name: past_return[name] for name in ("contact_1", "early")}
        for name in REFERENCES:
            accuracy = {history: read(f"{group}/{history}/{name}/correct") for history in ev.HISTORIES}
            accuracy.update({f"ABA_minus_{control}": accuracy["ABA"] - accuracy[control]
                             for control in ("B_only", "CBA")})
            tables[group][name] = ev.paired_accuracy_scores(accuracy, windows, indices["deployment"])
    return tables, short_windows, indices


def fit_scores(read, evaluation, protocol, births, arms, checkpoints, windows, short_indices, long_indices):
    settings = ev.Settings(**protocol["settings"])
    references = {name: read(f"fork_validation/mixed/{name}/correct") for name in REFERENCES}
    by_step, comparators = {}, {}
    initial = {(arm, mode): read(f"short_native/{arm}/0/{mode}/correct")
               for arm in arms for mode in ev.WRITE_MODES}
    for step in checkpoints:
        by_step[str(step)] = {}
        for arm in arms:
            short = {}
            for metric in ("correct", "taste", "p_correct", "side", "applied"):
                active, disabled = (read(f"short_native/{arm}/{step}/{mode}/{metric}")
                                    for mode in ("full", "no_write"))
                if metric == "taste":
                    active, disabled = (active > 0).astype(float), (disabled > 0).astype(float)
                if metric == "side":
                    short["sampled_action_disagreement"] = ev.paired_accuracy_scores(
                        {"full_vs_disabled": active != disabled}, windows, short_indices)
                    continue
                outcomes = {"full": active, "disabled": disabled, "full_minus_disabled": active - disabled}
                if metric == "correct":
                    outcomes.update({f"full_minus_{name}": active - scores for name, scores in references.items()})
                    outcomes["full_minus_birth"] = active - initial[arm, "full"]
                    outcomes["disabled_minus_birth"] = disabled - initial[arm, "no_write"]
                    outcomes["write_effect_minus_birth"] = (outcomes["full_minus_birth"]
                                                           - outcomes["disabled_minus_birth"])
                    outcomes["full_minus_chance"] = active - .5
                short[metric] = ev.paired_accuracy_scores(outcomes, windows, short_indices)
            by_step[str(step)][arm] = short

    # Compare the mature evaluator and empirical scores within the same worlds.
    # The full long-lifetime model scores are already in evaluation/summary.json.
    final_step = max(checkpoints)
    for arm in arms:
        comparators[arm] = {}
        for condition in ev.SCENARIOS:
            actual = []
            for birth in births:
                path = ev.fit_file(evaluation, arm, birth, final_step)
                with np.load(path, allow_pickle=False) as raw:
                    actual.append(raw[f"deployment/{condition}/full/correct"])
            full = np.stack(actual).astype(float)
            gaps = {f"full_minus_{name}": full - read(f"deployment/{condition}/{name}/correct")
                    for name in REFERENCES}
            comparators[arm][condition] = ev.paired_accuracy_scores(
                gaps, ev.windows(settings, condition), long_indices)
    return by_step, {str(final_step): comparators}


def run(run_dir, evaluation, out, *, births=None, arms=None, steps=None):
    run_dir, evaluation, out = map(Path, (run_dir, evaluation, out))
    if out.resolve() in {run_dir.resolve(), evaluation.resolve()}:
        raise ValueError("Use a separate reference output directory")
    protocol_path = evaluation / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    training_hash = ev.digest(run_dir / "run.json")
    if protocol["training_manifest_sha256"] != training_hash:
        raise ValueError("Evaluation and reference must use the same training manifest")
    births = ev.select(births, protocol["births"], "births")
    arms = ev.select(arms, protocol["arms"], "arms")
    steps = ev.select(steps, protocol["checkpoints"], "checkpoints")
    if not births or not arms or not steps:
        raise ValueError("Choose at least one evaluated birth, arm and checkpoint")
    protocol_hash = ev.digest(protocol_path)
    evaluation_summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    if evaluation_summary["protocol_sha256"] != protocol_hash:
        raise ValueError("Evaluation protocol changed after its results were scored")
    fit_index = {(item["birth"], item["arm"], item["step"]): item for item in evaluation_summary["fits"]}
    final_step = max(steps)
    used_fits = {}
    for birth in births:
        for arm in arms:
            for step in sorted({0, *steps}):
                digest = ev.digest(train.checkpoint_path(run_dir, arm, birth, step))
                key = (birth, arm, step)
                if key in fit_index and digest != fit_index[key]["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed since evaluation: {key!r}")
                if step == final_step:
                    source = fit_index[key]
                    if ev.digest(evaluation / source["file"]) != source["sha256"]:
                        raise ValueError(f"Evaluation records changed since scoring: {source['file']}")
                used_fits[f"{birth}/{arm}/{step}"] = digest
    out.mkdir(parents=True, exist_ok=True)
    files = {str(birth): fit_short(run_dir, evaluation, out, protocol, birth, arms, sorted({0, *steps}))
             for birth in births}
    arrays = [np.load(out / files[str(birth)]["file"], allow_pickle=False) for birth in births]
    try:
        def read(key):
            return np.stack([file[key] for file in arrays]).astype(float)

        references, short_windows, indices = reference_scores(read, protocol, births)
        short, gaps = fit_scores(read, evaluation, protocol, births, arms, steps, short_windows,
                                 indices["fork_validation"], indices["deployment"])
    finally:
        for file in arrays:
            file.close()
    ev.write_json(out / "summary.json", {
        "schema_version": 1,
        "source_sha256": ev.digest(Path(__file__)),
        "evaluation_protocol_sha256": protocol_hash,
        "training_manifest_sha256": training_hash,
        "births": births, "arms": arms, "steps": steps,
        "worlds_per_birth": {"deployment": protocol["settings"]["worlds"],
                             "holdout": protocol["settings"]["holdout_worlds"]},
        "short_windows_zero_based_half_open": short_windows,
        "tapes": {str(birth): protocol["tapes"][str(birth)] for birth in births},
        "frozen_checkpoint_sha256": used_fits,
        "records": files,
        "methods": {
            "empirical": "Canonicalize the two observed odor vectors lexicographically. Decide the side from "
                         "the sign of the cumulative canonical taste evidence and use the shared motor uniform "
                         "for ties. Update the corresponding odor pair with actual taste times action and "
                         "orientation. Recent3 uses the last three observations of that pair. Both rules carry "
                         "memory across phases and receive no cue IDs, noise flags, or clean answers.",
            "short_native": f"Replay each frozen graph on the disjoint {protocol['training_config']['contacts']}"
                            "-contact validation split "
                            "with the native gate and receipt path. Both branches use the same worlds and "
                            "uniforms. Disabled blocks actual writes and still delivers its native receipt.",
            "statistics": "All means weight births equally. Intervals are paired, pointwise 95% hierarchical "
                          "bootstraps over births and worlds. Accuracy labels are clean-side correctness.",
        },
        "empirical": references, "short_native": short, "long_native_minus_empirical": gaps,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ev.DATA)
    parser.add_argument("--evaluation", type=Path, help="Defaults to RUN/evaluation.")
    parser.add_argument("--out", type=Path, help="Defaults to RUN/reference.")
    parser.add_argument("--births", type=int, nargs="+")
    parser.add_argument("--arms", nargs="+", choices=train.ARMS)
    parser.add_argument("--steps", type=int, nargs="+")
    args = parser.parse_args()
    run(args.run, args.evaluation or args.run / "evaluation", args.out or args.run / "reference",
        births=args.births, arms=args.arms, steps=args.steps)


if __name__ == "__main__":
    main()
