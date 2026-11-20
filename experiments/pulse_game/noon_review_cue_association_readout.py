"""Vary only the two explicit association cells while the recipient body stays fixed."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from . import noon_review_cue_association as assay
from . import noon_horizon as env


PLAN = env.EVIDENCE / "noon-review-cue-association-readout-plan.md"
SEEDS = {"original": 76202671, "independent": 76202672}
CONDITIONS = ("write", "calibration_sham")


@torch.no_grad()
def replay(graph, rule, final, demos, order, head):
    base = final.tape.lanes
    rows = torch.arange(base).repeat_interleave(2)
    mapping = torch.tensor((-1., 1.)).repeat(base)
    test = final.subset(rows)
    history = [lesson.subset(rows) for lesson in demos]
    prefix_long = order[rows]
    expanded = torch.arange(test.tape.lanes).repeat_interleave(2)
    trial = test.subset(expanded)
    cue = torch.tensor((-1., 1.)).repeat(test.tape.lanes)
    actual_long = (cue * mapping[expanded]) > 0
    other = torch.arange(trial.tape.lanes) ^ 2
    assert torch.equal(cue, cue[other])
    view = rule.expand(test.tape.lanes)
    decision_view = rule.expand(trial.tape.lanes)
    output = {}
    for condition in CONDITIONS:
        state = env.net.start(graph, test.tape.lanes)
        memory = torch.zeros(test.tape.lanes, 2)
        for lesson, length in zip(history, (prefix_long, ~prefix_long), strict=True):
            symbol = (2 * length.float() - 1) * mapping
            state, _, observed_tail = assay.closed_calibration(rule, state, lesson, symbol, length,
                                                                 write=condition == "write")
            memory, _ = assay.observe_mapping(memory, symbol, observed_tail)
        state, _ = assay.study(view, state, test, forced=True, write=True)
        state, previews = assay.learned.previews(view, state, test.tape)
        delta = (previews[:, 1] - previews[:, 0])[expanded]
        audition, features = assay.learned.route_basis(decision_view, state.subset(expanded), trial.tape, cue, delta)
        slot = (cue > 0).long()[:, None]
        own = memory[expanded].gather(1, slot).squeeze(1)
        wrong = memory[expanded][other].gather(1, slot).squeeze(1)
        assert torch.equal(own, -wrong)
        own_logits = assay.adjusted_head(head, audition.motor_logit, features, own)
        wrong_logits = assay.adjusted_head(head, audition.motor_logit, features, wrong)
        draw = trial.tape.decision_draw
        own_side = assay.actuate(own_logits, draw).gt(0).long()
        wrong_side = assay.actuate(wrong_logits, draw).gt(0).long()
        # These two paid routes are evaluated on the recipient's exact same post-menu body.
        executed = assay.physical_options(rule, audition.state, trial, actual_long)
        indices = torch.arange(len(cue))
        right_unknown = trial.tape.candidates[:, 1] == trial.missing
        prefix = condition + "/"
        output[prefix + "preview_recognition"] = torch.where(delta == 0, .5, ((delta < 0) == right_unknown).float())
        output[prefix + "preview_delta_abs"] = delta.abs()
        output[prefix + "pulse_abs"] = features[:, 2].abs()
        output[prefix + "own"] = executed[indices, own_side, :, 1]
        output[prefix + "swapped_cells"] = executed[indices, wrong_side, :, 1]
        output[prefix + "changed_side"] = own_side != wrong_side
    return {key: value.numpy() for key, value in output.items()}


def summarize(raw, n):
    metrics = {}
    for condition in CONDITIONS:
        prefix = condition + "/"
        own, alternate = (raw[prefix + k] for k in ("own", "swapped_cells"))
        diff = own - alternate
        named = {key: raw[prefix + key] for key in ("preview_recognition", "preview_delta_abs", "pulse_abs", "changed_side")}
        named.update(own_resource=own.sum(1), swapped_cells_resource=alternate.sum(1),
                     own_minus_swapped_total=diff.sum(1), own_minus_swapped_review=diff[:, 0],
                     own_minus_swapped_tail=diff[:, 1:].sum(1))
        for key, values in named.items():
            metrics[prefix + key] = values.reshape(n, 16).mean(1)
    for label in ("preview_recognition", "preview_delta_abs", "pulse_abs", "own_minus_swapped_total"):
        metrics["write_minus_sham/" + label] = metrics["write/" + label] - metrics["calibration_sham/" + label]
    return metrics


def run(cohort, worlds, batch):
    torch.set_num_threads(1)
    seed = SEEDS[cohort]
    births = env.COHORTS[cohort]
    seal = json.loads(assay.SEAL.read_text(encoding="utf8"))
    final, demos, order = assay.generate(seed, worlds)
    grouped, raw_files, ancestry = [], {}, {}
    for birth in births:
        body = env.checkpoint(cohort, birth)
        head_file = assay.learned.ROOT / cohort / f"head-{birth}.npz"
        for path, identity in ((body, "body_unchanged"), (head_file, "training_and_deployment_files")):
            relative = path.relative_to(env.world.ROOT).as_posix()
            ancestry[relative] = env.digest(path)
            assert ancestry[relative] == seal[identity][relative]
        graph, rule = env.load(cohort, birth)
        with np.load(head_file, allow_pickle=False) as raw:
            head = torch.from_numpy(raw["informative"])
        pieces = []
        for offset in range(0, final.tape.lanes, 4 * batch):
            ids = torch.arange(offset, min(offset + 4 * batch, final.tape.lanes))
            pieces.append(replay(graph, rule, final.subset(ids), [p.subset(ids) for p in demos], order[ids], head))
        raw = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        path = assay.OUT / "readout-ablation" / cohort / f"{birth}-n{worlds}.npz"
        env.storage._write_raw(path, **raw)
        raw_files[path.relative_to(assay.OUT).as_posix()] = env.digest(path)
        grouped.append(summarize(raw, worlds))
    values = {name: np.stack([g[name] for g in grouped]) for name in grouped[0]}
    results = env.intervals.PairedCrossed(len(births), worlds, seed + 31).summary(values)
    report = dict(cohort=cohort, worlds=worlds, seed=seed, births=births, results=results, ancestry=ancestry,
                  raw_sha256=raw_files, plan_sha256=env.digest(PLAN), main_source_sha256=env.digest(assay.__file__),
                  source_sha256=env.digest(Path(__file__)))
    env.storage._write_json(env.EVIDENCE / f"noon-review-cue-association-readout-{cohort}-n{worlds}.json", report)
    keys = ("write/preview_recognition", "calibration_sham/preview_recognition", "write/pulse_abs",
            "calibration_sham/pulse_abs", "write/own_minus_swapped_total", "calibration_sham/own_minus_swapped_total")
    print(cohort, {key: results[key] for key in keys}, flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--batch", type=int, default=20)
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(run, cohort, a.worlds, a.batch) for cohort in env.COHORTS]
        for result in results:
            result.result()
