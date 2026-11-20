"""Test whether a fixed physical cue acquires a new meaning inside one continuous life."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from . import noon_horizon as env
from . import noon_horizon_anchor as anchors
from . import noon_horizon_anchor_preview as learned


OUT = env.world.DATA / "cue-association"
PLAN = env.EVIDENCE / "noon-review-cue-association-plan.md"
SEAL = env.EVIDENCE / "horizon-anchor-seal.json"
SEED = {"main": {"original": 76202631, "independent": 76202632},
        "replicate": {"original": 76202641, "independent": 76202642},
        "smoke": {"original": 76202651, "independent": 76202652}}
CONDITIONS = ("write", "calibration_sham", "neutral")
HORIZON = 4
END_TICKS = 3
COLLECT_TICKS = env.world.CONTACT_TICKS + env.world.GAP
ROUTE_TICKS = 3 * env.net.REACTION_TICKS + env.world.MODE_TICKS + env.CUE_TICKS
CALIBRATION_TICKS = (3 + HORIZON) * COLLECT_TICKS + ROUTE_TICKS + env.world.ECHO_TICKS + END_TICKS
ASSOCIATION_RATE = .85


@dataclass(frozen=True)
class Scenario:
    tape: env.world.Tape
    missing: torch.Tensor
    future: env.Opportunity
    tail_edits: torch.Tensor

    def subset(self, index):
        return Scenario(env.lesson.expand(self.tape, index), self.missing[index], self.future.subset(index),
                        self.tail_edits[index])


def generate(seed, worlds):
    def scene(key):
        entropy = int(np.random.SeedSequence([seed, 246113, key]).generate_state(1)[0])
        tape, missing = env.lesson.prepare(entropy, (), 0, worlds, 2, omissions=True, heldout=True)
        rng = np.random.default_rng([entropy, 315189])
        draws = rng.random((worlds, HORIZON, env.net.EDIT_DRAWS), dtype=np.float32)
        return Scenario(tape, missing, env.opportunities(entropy, (), 0, worlds, HORIZON),
                        torch.from_numpy(np.repeat(draws, 4, axis=0)))

    rng = np.random.default_rng([seed, 331906])
    order = torch.from_numpy(np.repeat(rng.integers(0, 2, worlds).astype(bool), 4))
    return scene(0), (scene(1), scene(2)), order


def fair(draw):
    return torch.where(draw < .5, 1., -1.)


def signed_contact(view, state, tape, item, sign, draw, *, write, forced=False, omission=None):
    left, right = env.world.sensory_for(tape, item, sign)
    truth = tape.law[torch.arange(tape.lanes), item] * sign
    if omission is not None:
        truth = torch.where(omission, 0., truth)
    if not forced:
        result = env.world.collect(view, state, left, right, truth, draw, write=write)
        return result.state, result.earned
    action = env.net.act(view, state, left, right, side_override=fair(draw[:, 0]))
    earned = truth * action.side
    observed = env.net.observe(view, action, left, right, earned)
    learned_state = env.net.commit(view, observed, draw[:, 1:], intervention="active" if write else "sham").state
    return env.net.wait(view, learned_state, env.world.GAP), earned


def study(view, state, scenario, *, forced, write):
    tape = scenario.tape
    taste = []
    for item in range(2):
        state, earned = signed_contact(view, state, tape, item, tape.study_orientation[:, item],
                                      tape.study_draw[:, item], write=write, forced=forced,
                                      omission=scenario.missing == item)
        taste.append(earned)
    return state, torch.stack(taste, dim=1)


def merge(base, indices, changes):
    graph = env.net.Graph(*(getattr(base.graph, key).index_copy(0, indices, getattr(changes.graph, key))
                            for key in ("weights", "bias", "present")))
    values = [getattr(base, key).index_copy(0, indices, getattr(changes, key))
              for key in ("hidden", "sender_trace", "receiver_trace")]
    return env.net.State(graph, *values)


def boundary(view, state):
    marker = state.graph.bias.new_tensor((0., .85, 0.))[None].expand(len(state.hidden), -1)
    return env.net._advance(view, state, env.net.native.sensory(state.graph, marker, marker), END_TICKS)


def closed_calibration(rule, state, scenario, cue, longer, *, write):
    """Use the same actually enacted sides so the two histories differ only in the cue."""
    tape = scenario.tape
    view = rule.expand(tape.lanes)
    state, prefix = study(view, state, scenario, forced=True, write=write)
    previewed, pre = learned.previews(view, state, tape)
    action, _ = learned.route_basis(view, previewed, tape, cue, pre[:, 1] - pre[:, 0])
    chosen = fair(tape.external_draw)
    state = env.world.echo_choice(view, action.state, chosen)
    selected = env.world.selected_item(tape, chosen)
    state, review = signed_contact(view, state, tape, selected, tape.review_orientation, tape.review_draw,
                                   write=write, forced=True)
    active = torch.where(longer)[0]
    idle = torch.where(~longer)[0]
    events = []
    if len(active):
        active_state = state.subset(active)
        physical = scenario.subset(active)
        subview = rule.expand(len(active))
        for idx in range(HORIZON):
            future = physical.future
            draws = torch.cat((future.motor[:, idx, None], physical.tail_edits[:, idx]), dim=1)
            active_state, taste = signed_contact(subview, active_state, physical.tape, future.target[:, idx],
                                                future.orientation[:, idx], draws, write=write, forced=True)
            events.append(review.new_zeros(tape.lanes).index_copy(0, active, taste))
        state = merge(state, active, active_state)
    if len(idle):
        asleep = env.net.wait(rule.expand(len(idle)), state.subset(idle), HORIZON * COLLECT_TICKS)
        state = merge(state, idle, asleep)
    state = boundary(view, state)
    observed_tail = torch.stack(events, dim=1) if events else review.new_zeros((tape.lanes, HORIZON))
    return state, torch.column_stack((prefix, review, observed_tail)), observed_tail


def observe_mapping(memory, cue, observed_tail):
    """Two explicit experimental cells receive only a witnessed cue and completed events."""
    slot = (cue > 0).long()[:, None]
    occurrence = (observed_tail != 0).any(1).float()
    signal = 2 * occurrence - 1
    active = memory.gather(1, slot).squeeze(1)
    return memory.scatter(1, slot, (active + ASSOCIATION_RATE * (signal - active))[:, None]), signal


def actuate(logit, draw):
    return 2 * (draw < logit.sigmoid()).float() - 1


def adjusted_head(head, native, feature, semantic):
    pulse = semantic * feature[:, 2].abs()
    response = torch.column_stack((feature[:, :2], pulse, feature[:, 1] * pulse))
    return native + response @ head


def physical_options(rule, state, scenario, longer):
    """Replay the two possible actual routes. Policy formation never reads these branches."""
    results = []
    for side in (-1., 1.):
        entries = state.hidden.new_zeros((scenario.tape.lanes, HORIZON + 1, 2))
        for length, mask in ((0, ~longer), (HORIZON, longer)):
            index = torch.where(mask)[0]
            branch = scenario.subset(index)
            action = entries.new_full((len(index),), side)
            view = rule.expand(len(index))
            actual, finished = anchors.execute(view, state.subset(index), branch.tape, branch.future, action, length)
            boundary(view, finished)
            entries.index_copy_(0, index, actual)
        results.append(entries)
    return torch.stack(results, dim=1)


@torch.no_grad()
def measure(graph, rule, final, calibrations, order, head):
    base = final.tape.lanes
    rows = torch.arange(base).repeat_interleave(2)
    phase = torch.tensor((-1., 1.)).repeat(base)
    long_first = order[rows]
    training = [scenario.subset(rows) for scenario in calibrations]
    test = final.subset(rows)
    tested = torch.arange(test.tape.lanes).repeat_interleave(2)
    trial = test.subset(tested)
    pulse = torch.tensor((-1., 1.)).repeat(test.tape.lanes)
    real_long = pulse * phase[tested] > 0
    donors = torch.arange(trial.tape.lanes) ^ 2
    assert torch.equal(pulse, pulse[donors]) and torch.equal(real_long, ~real_long[donors])
    initial = env.net.start(graph, test.tape.lanes)
    full_view = rule.expand(test.tape.lanes)
    observations = {}
    for condition in CONDITIONS:
        state = initial
        memory = torch.zeros(test.tape.lanes, 2)
        evidence = torch.zeros_like(memory)
        seen = []
        if condition != "neutral":
            for scenario, longer in zip(training, (long_first, ~long_first), strict=True):
                cue = (2 * longer.float() - 1) * phase
                state, earned, observed = closed_calibration(rule, state, scenario, cue, longer,
                                                              write=condition == "write")
                if not torch.equal(earned[::2], earned[1::2]):
                    raise AssertionError("Opposite mappings altered a calibration receipt")
                memory, labeled = observe_mapping(memory, cue, observed)
                evidence = evidence.scatter(1, (cue > 0).long()[:, None], labeled[:, None])
                seen.append(torch.column_stack((cue, earned)))
            if condition == "calibration_sham" and not all(torch.equal(getattr(state.graph, key), getattr(initial.graph, key))
                                                            for key in ("weights", "bias", "present")):
                raise AssertionError("The calibration-only sham changed the graph")
        else:
            for _ in training:
                state = boundary(full_view, env.net.wait(full_view, state, CALIBRATION_TICKS - END_TICKS))
        observations[f"{condition}/graph_change"] = (
            (state.graph.weights - initial.graph.weights).abs().flatten(1).sum(1)[tested])
        state, two_tastes = study(full_view, state, test, forced=True, write=True)
        if not torch.equal(two_tastes[::2], two_tastes[1::2]):
            raise AssertionError("Opposite mappings altered an initial test receipt")
        state, preview = learned.previews(full_view, state, test.tape)
        view = rule.expand(trial.tape.lanes)
        delta = (preview[:, 1] - preview[:, 0])[tested]
        action, feature = learned.route_basis(view, state.subset(tested), trial.tape, pulse, delta)
        raw_head = action.motor_logit + feature @ head
        native = action.motor_logit
        draw = trial.tape.decision_draw
        choices = {"head": actuate(raw_head, draw), "native": action.side}
        choices["head_donor"] = choices["head"][donors]
        choices["native_donor"] = choices["native"][donors]
        if condition != "neutral":
            slot = (pulse > 0).long()[:, None]
            public = evidence[tested].gather(1, slot).squeeze(1)
            sensed = memory[tested].gather(1, slot).squeeze(1)
            if not torch.equal(public > 0, real_long):
                raise AssertionError("Public observed events disagreed with the scheduled test mapping")
            choices["public"] = actuate(adjusted_head(head, native, feature, public), draw)
            choices["association"] = actuate(adjusted_head(head, native, feature, sensed), draw)
            choices["association_donor"] = choices["association"][donors]
            observations[f"{condition}/calibration"] = torch.stack(seen, dim=1)[tested]
            observations[f"{condition}/association_trace"] = memory[tested]
        outcomes = physical_options(rule, action.state, trial, real_long)
        columns = torch.arange(trial.tape.lanes)
        observations[f"{condition}/outcomes"] = outcomes
        observations[f"{condition}/support_taste"] = two_tastes[tested]
        observations[f"{condition}/head_logit"] = raw_head
        observations[f"{condition}/native_logit"] = native
        observations[f"{condition}/unknown_right"] = trial.tape.candidates[:, 1] == trial.missing
        for label, side in choices.items():
            observations[f"{condition}/{label}/side"] = side
            observations[f"{condition}/{label}/events"] = outcomes[columns, (side > 0).long()]
    observations.update(cue=pulse, long=real_long, standard_mapping=phase[tested] > 0)
    return {key: value.numpy() for key, value in observations.items()}


def rates(raw, worlds):
    result = {}

    def score(name, value):
        result[name] = np.asarray(value).reshape(worlds, -1).mean(1)

    long = raw["long"].reshape(worlds, 4, 2, 2)
    for condition in CONDITIONS:
        scores, select = {}, {}
        for key in raw:
            if not (key.startswith(condition + "/") and key.endswith("/side")):
                continue
            label = key.split("/")[1]
            side = (raw[key] > 0).reshape(worlds, 4, 2, 2)
            unknown = raw[f"{condition}/unknown_right"].reshape(worlds, 4, 2, 2)
            picked = (side == unknown).astype(float)
            events = raw[f"{condition}/{label}/events"][..., 1].reshape(worlds, 4, 2, 2, -1)
            paid = events.sum(-1)
            scores[label] = paid
            select[label] = picked
            tag = f"{condition}/{label}"
            score(tag + "/long_minus_short_unknown", picked[long] - picked[~long])
            score(tag + "/reversed_mapping_long_minus_short_unknown", picked[:, :, 0, 0] - picked[:, :, 0, 1])
            score(tag + "/fixed_mapping_long_minus_short_unknown", picked[:, :, 1, 1] - picked[:, :, 1, 0])
            score(tag + "/unknown_same_cue_long_minus_short", (2 * long[:, :, 1] - 1) * (picked[:, :, 1] - picked[:, :, 0]))
            score(tag + "/same_cue_route_switched", side[:, :, 0] != side[:, :, 1])
            score(tag + "/total", paid)
            score(tag + "/fixed_total", paid[:, :, 1])
            score(tag + "/reversed_total", paid[:, :, 0])
            score(tag + "/review", events[..., 0])
            score(tag + "/tail", events[..., 1:].sum(-1))
        for arm, other in (("head", "head_donor"), ("native", "native_donor"), ("head", "native"),
                           ("association", "association_donor"), ("association", "head"), ("public", "head")):
            if arm not in scores or other not in scores:
                continue
            delta = scores[arm] - scores[other]
            score(f"{condition}/{arm}_minus_{other}/total", delta)
            score(f"{condition}/{arm}_minus_{other}/fixed_total", delta[:, :, 1])
            score(f"{condition}/{arm}_minus_{other}/reversed_total", delta[:, :, 0])
        score(condition + "/calibration_weight_change", raw[f"{condition}/graph_change"])
    for model in ("head", "native"):
        for metric in ("long_minus_short_unknown", "same_cue_route_switched", "total"):
            result[f"write_minus_calibration_sham/{model}/{metric}"] = (result[f"write/{model}/{metric}"]
                                                                          - result[f"calibration_sham/{model}/{metric}"])
    return result


def cohort(cohort_name, split, worlds, batch):
    torch.set_num_threads(1)
    seed = SEED[split][cohort_name]
    births = env.COHORTS[cohort_name]
    origin = json.loads(SEAL.read_text(encoding="utf8"))
    final, demos, order = generate(seed, worlds)
    history, hashes, sources = [], {}, {}
    for birth in births:
        body = env.checkpoint(cohort_name, birth)
        head_file = learned.ROOT / cohort_name / f"head-{birth}.npz"
        for path, addresses in ((body, "body_unchanged"), (head_file, "training_and_deployment_files")):
            key = path.relative_to(env.world.ROOT).as_posix()
            sources[key] = env.digest(path)
            if sources[key] != origin[addresses][key]:
                raise ValueError(f"Source changed after parent seal: {path}")
        with np.load(head_file, allow_pickle=False) as store:
            if str(store["body_sha256"]) != sources[body.relative_to(env.world.ROOT).as_posix()]:
                raise ValueError("Four coefficients do not match their native body")
            head = torch.from_numpy(store["informative"])
            assert head.shape == (4,) and head.isfinite().all()
        graph, rule = env.load(cohort_name, birth)
        parts = []
        for offset in range(0, final.tape.lanes, 4 * batch):
            index = torch.arange(offset, min(offset + 4 * batch, final.tape.lanes))
            parts.append(measure(graph, rule, final.subset(index), [d.subset(index) for d in demos], order[index], head))
        raw = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        dest = OUT / split / cohort_name / f"{birth}-n{worlds}.npz"
        env.storage._write_raw(dest, **raw)
        hashes[dest.relative_to(OUT).as_posix()] = env.digest(dest)
        values = rates(raw, worlds)
        history.append(values)
        print(split, cohort_name, birth, {name: round(values[name].mean(), 4) for name in (
            "write/head/long_minus_short_unknown", "write/head_minus_head_donor/total",
            "write/association_minus_head/total", "write/association_minus_association_donor/total")}, flush=True)
    paired = {name: np.stack([result[name] for result in history]) for name in history[0]}
    values = env.intervals.PairedCrossed(len(births), worlds, seed + 271).summary(paired)
    report = dict(cohort=cohort_name, split=split, seed=seed, worlds=worlds, batch=batch, births=births,
                  calibration_ticks_per_episode=CALIBRATION_TICKS, local_association_rate=ASSOCIATION_RATE,
                  plan_sha256=env.digest(PLAN), seal_sha256=env.digest(SEAL), sources=sources,
                  code_sha256={Path(path).name: env.digest(path) for path in (__file__, learned.__file__, anchors.__file__,
                                                                            env.__file__, env.net.__file__)},
                  raw_sha256=hashes, results=values)
    env.storage._write_json(env.EVIDENCE / f"noon-review-cue-association-{split}-{cohort_name}-n{worlds}.json", report)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=SEED, required=True)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--cohorts", choices=env.COHORTS, nargs="+", default=list(env.COHORTS))
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=len(a.cohorts)) as pool:
        jobs = [pool.submit(cohort, c, a.split, a.worlds, a.batch) for c in a.cohorts]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
