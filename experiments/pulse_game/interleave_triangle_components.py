"""Locate the memory used to answer three independently valued cues on a zero-sum triangle."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
import json

import numpy as np
import torch

from . import interleave_confirm_triangle as verification
from . import interleave_triangle_train as tri
from . import local_trace_graph as graph
from . import revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE


WRITES = ("active", "weights_and_bias", "weights_only", "bias_only", "topology_only", "sham")
SWAPS = ("opposite_graph", "opposite_activity", "opposite_all")
METRICS = ("strict_per_order", "strict_all_orders", "all_maps_all_orders",
           "hard_actual_three", "mixed_actual_three", "growth_per_life", "retirements_per_life")


def learn(body, rule, tape, mode):
    n = len(tape["law"])
    rows = torch.arange(n)
    state = graph.start(body, n)
    events = torch.zeros(n, 2)
    for stage in range(3):
        item, orientation = tape["order"][:, stage], tape["study_flip"][:, stage]
        left, right = tri.binding.present(tape, item, orientation)
        draws = tape["study_u"][:, stage]
        action = graph.act(rule, state, left, right, draws[:, 0])
        taste = action.side * orientation * tape["law"][rows, item]
        observed = graph.observe(rule, action, left, right, taste)
        result = graph.commit(rule, observed, draws[:, 1:], intervention=mode)
        events[:, 0] += result.receipt.birth >= 0
        events[:, 1] += result.receipt.retired >= 0
        state = graph.wait(rule, result.state, 4)
    return state, events.numpy()


def answers(state, tape):
    n = len(tape["law"])
    probability, actual = [], []
    for object_id in range(3):
        logits, choices = [], []
        for sign in (1., -1.):
            left, right = tri.binding.present(tape, torch.full((n,), object_id), torch.full((n,), sign))
            draw = tape["probe_u"][:, object_id]
            # Terminal siblings never reuse activity or local traces. The same native
            # five-tick motor path can therefore skip updating both sets of traces.
            action = motor.act(state, left, right, draw if sign > 0 else 1 - draw)
            logits.append(action.motor_logit)
            choices.append(action.side * sign)
        probability.append(torch.stack(logits, 1))
        actual.append(torch.stack(choices, 1))
    return torch.stack(probability, 1).numpy(), torch.stack(actual, 1).numpy().astype(np.int8)


@torch.no_grad()
def trial(body, inherited, base, archive):
    n = len(base["law"])
    rule = inherited.expand(n)
    opposite = torch.arange(n) ^ 7
    raw = {}
    for j, order in enumerate(permutations(range(3))):
        tape = verification.permuted_tape(base, order)
        for mode in WRITES:
            state, events = learn(body, rule, tape, mode)
            states = {mode: state}
            if mode == "active":
                other = state.subset(opposite)
                states.update(opposite_graph=graph.swap(state, other, "graph"),
                              opposite_activity=graph.swap(state, other, ("hidden", "trace")),
                              opposite_all=other)
            for label, snapshot in states.items():
                logits, sampled = answers(snapshot, tape)
                if label == "active":
                    np.testing.assert_array_equal(logits, archive[f"active_logit_{j}"])
                    np.testing.assert_array_equal(sampled, archive[f"active_selected_{j}"])
                elif label == "opposite_all":
                    np.testing.assert_array_equal(logits, archive[f"active_logit_{j}"][opposite.numpy()])
                raw[f"{j}/{label}/logit"] = logits
                raw[f"{j}/{label}/action"] = sampled
                raw[f"{j}/{label}/events"] = events
    raw["law"] = base["law"].numpy()
    return raw


def measure(raw, mode):
    laws = raw["law"].reshape(-1, 8, 3)
    strict, all_maps, hard_actual, mixed_actual, events = [], [], [], [], []
    for j in range(6):
        prefix = f"{j}/{mode}/"
        z = raw[prefix + "logit"].reshape(-1, 8, 3, 2)
        correct_margins = (z[..., 0] - z[..., 1]) * laws / 2 > tri.MARGIN_THRESHOLD
        passes = correct_margins.all(2)
        strict.append(passes[:, [0, 7]].all(1))
        all_maps.append(passes.all(1))
        actions = raw[prefix + "action"].reshape(z.shape)
        all_three = (actions == laws[..., None]).all(2)
        hard_actual.append(all_three[:, [0, 7]].mean((1, 2)))
        mixed_actual.append(all_three[:, 1:7].mean((1, 2)))
        events.append(raw[prefix + "events"].reshape(-1, 8, 2).mean(1))
    changes = np.mean(events, axis=0)
    return np.stack((np.mean(strict, axis=0), np.all(strict, axis=0), np.all(all_maps, axis=0),
                     np.mean(hard_actual, axis=0), np.mean(mixed_actual, axis=0), *changes.T), axis=-1)


def run(cohort, step):
    torch.set_num_threads(1)
    births = list(range(731, 737)) if cohort == "original" else list(range(811, 817))
    path = DATA / ("triangle-long-" + cohort)
    dest = DATA / "triangle-components" / cohort
    dest.mkdir(parents=True, exist_ok=True)
    existing = json.loads((EVIDENCE / f"triangle-orders-long-{cohort}-{step}.json").read_text())
    seed, worlds = existing["seed"], existing["worlds"]
    rng = np.random.default_rng(4265811)
    root_draws = rng.integers(len(births), size=(4000, len(births)))
    world_weights = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=(len(births), len(root_draws))) / worlds
    matrices = {k: [] for k in (*WRITES, *SWAPS)}
    checks = {}
    for birth in births:
        g, r = tri.old_fit.load_checkpoint(path, "odor", "direct", birth, step)
        tape = tri.make_tape(seed, birth, worlds, -.5)
        file = DATA / "long-order-verification/order-validation" / ("long-" + cohort) / f"{birth}-{step}.npz"
        if sha256(file.read_bytes()).hexdigest() != existing["sha256"][file.name]:
            raise ValueError("Ordinary stored actions changed")
        with np.load(file) as prior:
            raw = trial(g, r, tape, prior)
        for mode, rows in matrices.items():
            rows.append(measure(raw, mode))
        f = dest / f"{birth}-{step}.npz"
        np.savez_compressed(f, **raw)
        checks[f.name] = sha256(f.read_bytes()).hexdigest()
        print(cohort, step, birth, flush=True)

    def summary(values):
        boots = world_weights @ values
        paired = boots[root_draws, np.arange(len(root_draws))[:, None]].mean(1)
        return {name: dict(mean=float(values[:, :, j].mean()), by_birth=values[:, :, j].mean(1).tolist(),
                           ci95=np.quantile(paired[:, j], [.025, .975]).tolist())
                for j, name in enumerate(METRICS)}

    result = dict(cohort=cohort, step=step, births=births, worlds_per_birth=worlds, seed=seed,
                  notes="After training with all write permissions, intervene only in held-out lives. Keep every "
                        "support contact, actual taste, six orders and eight laws. Same final-state probes before "
                        "any subsequent taste. Opposite donor has all three private answers inverted with the same "
                        "objects, order and random draws. Its actual study actions can differ. Event counts refer to "
                        "the three support contacts, shared by terminal-state interventions. The active branch must "
                        "reproduce every archived motor logit and action bitwise. Intervals resample training births "
                        "and independent held-out geometries within them.",
                  roots_and_worlds=len(births)*worlds, sha256=checks)
    means = {k: np.stack(v) for k, v in matrices.items()}
    result["scores"] = {k: summary(v) for k, v in means.items()}
    result["paired_against_active"] = {k: summary(v - means["active"]) for k, v in means.items() if k != "active"}
    (EVIDENCE / f"triangle-components-{cohort}-{step}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=("original", "independent"), required=True)
    p.add_argument("--step", choices=(32000, 48000, 64000), type=int, required=True)
    args = p.parse_args()
    run(args.cohort, args.step)
