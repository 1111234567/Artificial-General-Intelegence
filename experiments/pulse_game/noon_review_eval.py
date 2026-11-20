"""Replay one frozen physical history with matched action, content, and write interventions."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import noon_review_train as train
from . import noon_review_world as world


@dataclass(frozen=True)
class Intervention:
    base: str = "native"
    echo: int = 1
    content: int = 1
    write: bool = True
    contact: bool = True


BRANCHES = {
    "native": Intervention(),
    "flip_action": Intervention(echo=-1, content=-1),
    "swap_content": Intervention(content=-1),
    "swap_echo": Intervention(echo=-1),
    "native_sham": Intervention(write=False),
    "flip_action_sham": Intervention(echo=-1, content=-1, write=False),
    "swap_content_sham": Intervention(content=-1, write=False),
    "swap_echo_sham": Intervention(echo=-1, write=False),
    "random": Intervention(base="random"),
    "random_sham": Intervention(base="random", write=False),
    "uncertain": Intervention(base="uncertain"),
    "uncertain_sham": Intervention(base="uncertain", write=False),
    "quiet_after_menu": Intervention(write=False, contact=False),
}
SHAM_PAIRS = {name.removesuffix("_sham"): name for name in BRANCHES if name.endswith("_sham")}
MEASURES = ("accuracy", "all_correct", "sampled_target_accuracy", "both_laws_accuracy",
            "expected_accuracy", "reviewed_accuracy", "unreviewed_accuracy")


def to_np(value):
    return value.detach().cpu().numpy()


@torch.no_grad()
def evaluate(graph, rule, tape, uncertain, confidence):
    coeff = rule.expand(tape.lanes)
    state, introductions = world.introduce(coeff, native.start(graph, tape.lanes), tape)
    left, right = world.candidate_menu(tape)
    choice = world.select(coeff, state, left, right, tape.decision_draw)
    mirrored = world.select(coeff, state, right, left, 1 - tape.decision_draw)
    policies = {"native": choice.side, "random": world.random_side(tape), "uncertain": uncertain}

    raw = dict(candidate=tape.candidates, confidence=confidence, mirrored_side=mirrored.side,
               mirrored_logit=mirrored.motor_logit, menu_motor_side=choice.side, menu_motor_logit=choice.motor_logit,
               study_taste=torch.stack([contact.earned for contact in introductions], dim=1),
               truth=tape.law * tape.final_orientation, law=tape.law, sampled_target=tape.final_item)

    def probes(name, snapshot):
        actions = [world.final_probe(coeff, snapshot, tape, item)[0] for item in range(tape.count)]
        raw[f"{name}/final_side"] = torch.stack([action.side for action in actions], dim=1)
        raw[f"{name}/final_logit"] = torch.stack([action.motor_logit for action in actions], dim=1)

    probes("ordinary", state)
    total_review_time = world.MODE_TICKS + native.REACTION_TICKS + world.ECHO_TICKS + world.CONTACT_TICKS + world.GAP
    probes("quiet_without_menu", native.wait(coeff, state, total_review_time))

    for name, condition in BRANCHES.items():
        policy = policies[condition.base]
        executed, content = condition.echo * policy, condition.content * policy
        snapshot, contact = world.review(coeff, choice.state, tape, executed, content_side=content,
                                         write=condition.write, contact=condition.contact)
        probes(name, snapshot)
        raw[f"{name}/executed_side"] = executed
        if contact is not None:
            raw[f"{name}/object"] = world.selected_item(tape, content)
            raw[f"{name}/contact_side"] = contact.motor.side
            raw[f"{name}/taste"] = contact.earned
            receipt = contact.edit.receipt
            raw[f"{name}/graph_distance"] = torch.stack((receipt.weight_delta.abs().flatten(1).sum(1),
                                                         receipt.bias_delta.abs().sum(1), (receipt.birth >= 0).float(),
                                                         (receipt.retired >= 0).float()), dim=1)
        if not condition.write and not all(torch.equal(getattr(snapshot.graph, field), getattr(state.graph, field))
                                           for field in ("weights", "bias", "present")):
            raise AssertionError(f"{name} mutated the pre-menu graph")

    for active, sham in SHAM_PAIRS.items():
        for key in ("object", "executed_side", "contact_side", "taste"):
            if not torch.equal(raw[f"{active}/{key}"], raw[f"{sham}/{key}"]):
                raise AssertionError(f"{active} and {sham} diverged earlier than the write")
    if not (torch.equal(raw["native/object"], raw["swap_echo/object"])
            and torch.equal(raw["native/executed_side"], raw["swap_content/executed_side"])
            and (raw["native/object"] != raw["flip_action/object"]).all()
            and torch.equal(raw["flip_action/object"], raw["swap_content/object"])):
        raise AssertionError("Mismatched action/content factorial")
    return {key: to_np(value) for key, value in raw.items()}


def read_metrics(raw, branch):
    target = raw["truth"]
    correct = raw[f"{branch}/final_side"] == target
    margin = raw[f"{branch}/final_logit"] * target
    rows = np.arange(len(correct))
    choice = raw.get(f"{branch}/object")
    if choice is None:
        choice = raw["native/object"]
    chosen = correct[rows, choice]
    return np.asarray((correct.mean(), correct.all(1).mean(), correct[rows, raw["sampled_target"]].mean(),
                       (correct[::2] & correct[1::2]).mean(), (1 / (1 + np.exp(-margin))).mean(),
                       chosen.mean(), ((correct.sum(1) - chosen) / (correct.shape[1] - 1)).mean()))


def selector_diagnostics(raw):
    side, logit = raw["menu_motor_side"], raw["menu_motor_logit"]
    natural_item = raw["native/object"]
    remapped = raw["mirrored_side"] == -side
    native_oldest = natural_item == raw["candidate"].min(1)
    importances = (1 / (1 + np.exp(-logit)))[::2] - (1 / (1 + np.exp(-logit)))[1::2]
    aligned_choice = (raw["native/object"] == raw["uncertain/object"]).mean()
    return dict(right_fraction=float((side > 0).mean()), oldest_fraction=float(native_oldest.mean()),
                mirrored_same_item_fraction=float(remapped.mean()),
                private_history_side_disagreement=float((side[::2] != side[1::2]).mean()),
                private_history_mean_policy_probability_change=float(np.abs(importances).mean()),
                agreement_parent_uncertainty=float(aligned_choice))


def _estimates(values, intervals):
    return {key: intervals.report(column) for key, column in zip(MEASURES, values.T, strict=True)}


def summarize(array, diagnostics, births, seed):
    intervals = prior.BirthIntervals(list(births), 20_000, seed)
    stats = {arm: {name: _estimates(values, intervals) for name, values in branches.items()}
             for arm, branches in array.items()}
    same_body = array["native"]
    contrasts = {f"same_body_native_minus_{branch}": same_body["native"] - values
                 for branch, values in same_body.items() if branch != "native"}
    contrasts["native_write_gain_minus_random_write_gain"] = (
        (same_body["native"] - same_body["native_sham"]) - (same_body["random"] - same_body["random_sham"]))
    contrasts["native_write_gain_minus_other_object_write_gain"] = (
        (same_body["native"] - same_body["native_sham"])
        - (same_body["swap_content"] - same_body["swap_content_sham"]))
    contrasts["executed_side_marginal"] = .5 * (same_body["native"] + same_body["swap_content"]
                                                - same_body["swap_echo"] - same_body["flip_action"])
    contrasts["actual_object_marginal"] = .5 * (same_body["native"] + same_body["swap_echo"]
                                                - same_body["swap_content"] - same_body["flip_action"])
    for arm, branches in array.items():
        if arm != "native":
            contrasts[f"native_training_minus_{arm}_training"] = same_body["native"] - branches[arm]
            contrasts[f"ordinary_native_training_minus_{arm}_training"] = same_body["ordinary"] - branches["ordinary"]
    diagnostic_means = {arm: {key: intervals.report([value[key] for value in values]) for key in values[0]}
                        for arm, values in diagnostics.items()}
    return dict(rates=stats, differences={name: _estimates(value, intervals) for name, value in contrasts.items()},
                selector_diagnostics=diagnostic_means)


def evaluate_cohort(path, cohort, stage, split, counts, worlds):
    torch.set_num_threads(1)
    seed = world.EVAL_SEEDS[split][cohort]
    births = world.COHORTS[cohort]
    arms = world.ARMS if stage else ("native",)
    directory = path / "evaluation" / split / f"{stage:06d}" / cohort / f"n{worlds}"
    all_counts, raw_sha256, checkpoint_sha256 = {}, {}, {}
    for count in counts:
        tape = world.make_tape(seed, worlds, count)
        file = directory / f"tape-{count}.npz"
        archive._write_raw(file, **{key: to_np(value) for key, value in vars(tape).items()})
        raw_sha256[file.relative_to(path).as_posix()] = prior.digest(file)
        arrays = {arm: {} for arm in arms}
        diagnostics = {arm: [] for arm in arms}
        for birth in births:
            parent_graph, parent_rule = world.load_parent(cohort, birth)
            with torch.no_grad():
                view = parent_rule.expand(tape.lanes)
                state, _ = world.introduce(view, native.start(parent_graph, tape.lanes), tape)
                uncertain, confidence = world.parent_uncertainty(view, state, tape)
            for arm in arms:
                checkpoint = train.checkpoint_path(path, cohort, arm, birth, stage)
                checkpoint_sha256[checkpoint.relative_to(path).as_posix()] = prior.digest(checkpoint)
                graph, rule = world.load_frozen(cohort, arm, birth, stage, path=path)
                raw = evaluate(graph, rule, tape, uncertain, confidence)
                file = directory / f"{arm}-{birth}-{count}.npz"
                archive._write_raw(file, **raw)
                raw_sha256[file.relative_to(path).as_posix()] = prior.digest(file)
                diagnostics[arm].append(selector_diagnostics(raw))
                for branch in (*BRANCHES, "ordinary", "quiet_without_menu"):
                    arrays[arm].setdefault(branch, []).append(read_metrics(raw, branch))
        values = {arm: {branch: np.array(rows) for branch, rows in branches.items()}
                  for arm, branches in arrays.items()}
        all_counts[str(count)] = summarize(values, diagnostics, births, seed)
        a = all_counts[str(count)]["rates"]["native"]["native"]["accuracy"]["mean"]
        b = all_counts[str(count)]["differences"]["same_body_native_minus_random"]["accuracy"]["mean"]
        print(f"{split} {cohort} {stage} n={count} native {a:.5f} minus same-body random {b:+.5f}", flush=True)
    report = dict(stage=stage, split=split, seed=seed, birth_cohort=cohort, births=births,
                  independent_worlds_per_birth=worlds, heldout_protocol=(split == "heldout"),
                  arms=arms, branches=tuple(BRANCHES), conditional_measures=MEASURES, loads=all_counts,
                  checkpoint_sha256=checkpoint_sha256, raw_sha256=raw_sha256,
                  source_sha256={Path(name).name: prior.digest(Path(name)) for name in (__file__, world.__file__)},
                  manifest_sha256=prior.digest(path / "run.json"))
    filename = f"noon-review-{split}-{stage:06d}-{cohort}-n{worlds}-{'-'.join(map(str, counts))}.json"
    archive._write_json(directory / filename, report)
    return filename, report


def run(path, stage, split, counts, worlds, *, cohorts=tuple(world.COHORTS), evidence=None, workers=2):
    if len(cohorts) == 1:
        reports = [evaluate_cohort(path, cohorts[0], stage, split, counts, worlds)]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(cohorts))) as pool:
            futures = [pool.submit(evaluate_cohort, path, cohort, stage, split, counts, worlds) for cohort in cohorts]
            reports = [future.result() for future in futures]
    if evidence is not None:
        for name, report in reports:
            archive._write_json(evidence / name, report)
    return reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=world.DATA / "core")
    parser.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    parser.add_argument("--stage", type=int, choices=world.CHECKPOINTS, required=True)
    parser.add_argument("--split", choices=world.EVAL_SEEDS, required=True)
    parser.add_argument("--cohorts", nargs="+", choices=world.COHORTS, default=list(world.COHORTS))
    parser.add_argument("--counts", nargs="+", choices=(2, 4, 8), type=int, default=[4])
    parser.add_argument("--worlds", type=int, default=384)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.path, args.stage, args.split, args.counts, args.worlds, cohorts=args.cohorts,
        evidence=args.evidence, workers=args.workers)


if __name__ == "__main__":
    main()
