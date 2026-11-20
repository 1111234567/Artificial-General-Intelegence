"""Fair-clock test of a physical, continuously carried candidate preview."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import noon_review_world as world
from . import self_param_writer as body


DOMAIN = 0x61A124
FROZEN_SEEDS = {"original": 76202441, "independent": 76202442}
FIT_SEEDS = {"original": 76202461, "independent": 76202462}
FIT_ARMS = ("informative", "distractor")
PREVIEW_CUE_TICKS = 2
FIT_PATH = world.DATA / "preview-fit"
SCORING = ("accuracy", "all_correct", "sampled_target")


@dataclass(frozen=True)
class PreviewTape:
    order: torch.Tensor
    sign: torch.Tensor
    distractor: torch.Tensor


def make_preview(seed, worlds, *, births=(), step=None):
    if bool(births) != (step is not None):
        raise ValueError("Pass both preview training roots and step or neither")
    roots = ([DOMAIN, seed, birth, step] for birth in births) if births else ([DOMAIN, seed],)

    def generate(entropy):
        rng = np.random.default_rng(entropy)
        a = rng.integers(0, 2, worlds)
        order = np.stack((a, 1 - a), axis=1)
        sign = (rng.integers(0, 2, (worlds, 2)) * 2 - 1).astype(np.float32)
        directions = rng.normal(size=(worlds, 2, 3))
        directions = (1.35 * directions / np.linalg.norm(directions, axis=-1, keepdims=True)).astype(np.float32)
        return tuple(np.repeat(array, 2, 0) for array in (order, sign, directions))

    return PreviewTape(*(torch.from_numpy(np.concatenate(arrays)) for arrays in
                         zip(*(generate(root) for root in roots), strict=True)))


def sweep(view, state, tape, preview, informative):
    """Return the physically advanced state and the probe logits, without submitting writes."""
    cue = state.graph.bias.new_tensor((-.85, 0., 0.))[None].expand(tape.lanes, -1)
    state = native._advance(view, state, body.sensory(state.graph, cue, cue), PREVIEW_CUE_TICKS)
    rows = torch.arange(tape.lanes, device=tape.odor.device)[:, None]
    candidates = tape.odor[rows, tape.candidates[rows, preview.order]] if informative else preview.distractor
    stimulus = candidates * preview.sign[..., None]
    logits = []
    for slot in range(2):
        signal = stimulus[:, slot]
        action = native.act(view, state, signal, -signal, stochastic=False)
        logits.append(action.motor_logit)
        state = action.state
    return state, torch.stack(logits, dim=1).gather(1, preview.order)


def readout(view, state, tape):
    final = [world.final_probe(view, state, tape, item) for item in range(tape.count)]
    return torch.stack([action.side == truth for action, truth in final], dim=1)


@torch.no_grad()
def evaluate(graph, rule, tape, preview):
    view = rule.expand(tape.lanes)
    initial = native.start(graph, tape.lanes)
    state, _ = world.introduce(view, initial, tape)
    pair = world.candidate_menu(tape)
    choices, scores, values = {}, {}, {}
    for condition in FIT_ARMS:
        after_preview, logit = sweep(view, state, tape, preview, condition == "informative")
        action = world.select(view, after_preview, *pair, tape.decision_draw)
        choices[condition] = action.side
        scores[condition] = logit
        candidates = []
        for side in (-1, 1):
            after_review, _ = world.review(view, action.state, tape, torch.full_like(action.side, side))
            candidates.append(readout(view, after_review, tape))
        values[condition] = torch.stack(candidates, dim=1)

    raw = dict(candidate=tape.candidates, preview_order=preview.order, final_item=tape.final_item)
    metrics = {}

    def reads(accuracy):
        rows = torch.arange(tape.lanes)
        return torch.stack((accuracy.float().mean(1), accuracy.all(1).float(), accuracy[rows, tape.final_item].float()))

    for condition in FIT_ARMS:
        actual = values[condition]
        left, right = actual.unbind(dim=1)
        raw[f"{condition}/all_options"] = actual
        raw[f"{condition}/emitted_side"] = choices[condition]
        raw[f"{condition}/preview_logit"] = scores[condition]
        shortlisted = scores[condition].abs()
        external = torch.where(shortlisted[:, 0] == shortlisted[:, 1], world.random_side(tape),
                               torch.where(shortlisted[:, 0] < shortlisted[:, 1], -1., 1.))
        for decision, side in (*choices.items(), ("measured_uncertainty", external)):
            measured = torch.where((side > 0)[:, None], right, left)
            metrics[f"{condition}_state_{decision}_side"] = reads(measured).mean(1)
        metrics[f"{condition}_state_random_side"] = .5 * (reads(left).mean(1) + reads(right).mean(1))
    metrics["ordinary_before_preview"] = reads(readout(view, state, tape)).mean(1)
    metrics["informative_minus_distractor_native"] = (metrics["informative_state_informative_side"]
                                                      - metrics["distractor_state_distractor_side"])
    metrics["informative_native_minus_same_state_random"] = (metrics["informative_state_informative_side"]
                                                              - metrics["informative_state_random_side"])
    metrics["informative_native_minus_same_state_distractor_side"] = (metrics["informative_state_informative_side"]
                                                                      - metrics["informative_state_distractor_side"])
    metrics["distractor_native_minus_same_state_random"] = (metrics["distractor_state_distractor_side"]
                                                            - metrics["distractor_state_random_side"])
    return {name: value.numpy() for name, value in metrics.items()}, {name: value.numpy() for name, value in raw.items()}


def evaluate_cohort(cohort, stages, worlds, path, evidence, fitted):
    torch.set_num_threads(1)
    seeds = FIT_SEEDS if fitted else FROZEN_SEEDS
    seed = seeds[cohort]
    births = world.COHORTS[cohort]
    tape, preview = world.make_tape(seed, worlds, 4), make_preview(seed, worlds)
    intervals = prior.BirthIntervals(list(births), 20_000, seed)
    fit_arms = FIT_ARMS if fitted else ("native",)
    stages_report, hashes = {}, {}
    for stage in stages:
        output = {arm: {} for arm in fit_arms}
        for birth in births:
            for arm, stats in output.items():
                graph, rule = world.load_frozen(cohort, arm, birth, stage, path=path)
                metrics, raw = evaluate(graph, rule, tape, preview)
                for name, values in metrics.items():
                    stats.setdefault(name, []).append(values)
                rawfile = path / "evaluation" / "preview" / f"{stage:06d}" / cohort / f"{arm}-{birth}-n{worlds}.npz"
                archive._write_raw(rawfile, **raw)
                hashes[rawfile.relative_to(path).as_posix()] = prior.digest(rawfile)
        rates = {arm: {name: {key: intervals.report(column) for key, column in
                             zip(SCORING, np.array(values).T, strict=True)} for name, values in stats.items()}
                 for arm, stats in output.items()}
        transfer = {}
        if fitted:
            for name in output["informative"]:
                informative = np.array(output["informative"][name])
                distractor = np.array(output["distractor"][name])
                transfer[name] = {key: intervals.report(column) for key, column in
                                  zip(SCORING, (informative - distractor).T, strict=True)}
        stages_report[str(stage)] = dict(arms=rates, informative_training_minus_distractor_training=transfer)
        print(cohort, stage, {arm: rates[arm]["informative_native_minus_same_state_random"]["accuracy"]["mean"]
                             for arm in fit_arms}, flush=True)
    label = "fitted" if fitted else "frozen"
    result = dict(cohort=cohort, seed=seed, worlds=worlds, births=births, mode=label, stages=stages_report,
                  raw_sha256=hashes, script_sha256=prior.digest(Path(__file__)))
    name = f"noon-review-preview-{label}-{cohort}-n{worlds}-{'-'.join(map(str, stages))}.json"
    archive._write_json(evidence / name, result)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--stages", type=int, nargs="+", required=True)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--path", type=Path)
    p.add_argument("--fitted", action="store_true")
    p.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    p.add_argument("--cohorts", nargs="+", choices=world.COHORTS, default=list(world.COHORTS))
    a = p.parse_args()
    path = a.path or (FIT_PATH if a.fitted else world.DATA / "core")
    with ProcessPoolExecutor(max_workers=len(a.cohorts)) as pool:
        jobs = [pool.submit(evaluate_cohort, name, a.stages, a.worlds, path, a.evidence, a.fitted) for name in a.cohorts]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
