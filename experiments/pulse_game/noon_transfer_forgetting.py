"""Compare old A retention while replaying the same physical receipts."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity_report as stats, interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive
from . import noon_transfer_custody as custody


PLAN = card.OUT / "transfer-retention-plan.json"
MODES = ("write", "sham", "idle")


def support_contact(rule, state, tape, item, side, intervention):
    left, right = tape.left[:, item, 0], tape.right[:, item, 0]
    action = card.native.act(rule, state, left, right, tape.motor_draw[:, item, 0], side_override=side)
    earned = action.side * tape.right_value[:, item, 0]
    experience = card.native.observe(rule, action, left, right, earned)
    state = card.native.commit(rule, experience, tape.edit_draw[:, item, 0], intervention=intervention).state
    return state, action.side, earned


def select_support(parent, rule, tape):
    state = card.native.start(parent, tape.lanes)
    rule = rule.expand(tape.lanes)
    sides, receipts = [], []
    for item in range(tape.max_k + 1):
        state, side, taste = support_contact(rule, state, tape, item, None, "active")
        sides.append(side)
        receipts.append(taste)
    return torch.stack(sides), torch.stack(receipts)


def replay(graph, rule, tape, support, counts):
    rule = rule.expand(tape.lanes)
    state = card.native.start(graph, tape.lanes)
    a, _, observed = support_contact(rule, state, tape, 0, support[0][0], "active")
    torch.testing.assert_close(support[1][0], observed, rtol=0, atol=0)
    branches = dict.fromkeys(MODES, a)
    probes = {mode: [] for mode in MODES}
    target = tape.right_value[:, 0, -1].numpy()
    for item in range(tape.max_k + 1):
        if item:
            for mode, intervention in (("write", "active"), ("sham", "sham")):
                changed, chosen, taste = support_contact(rule, branches[mode], tape, item, support[0][item], intervention)
                torch.testing.assert_close(chosen, support[0][item], rtol=0, atol=0)
                torch.testing.assert_close(taste, support[1][item], rtol=0, atol=0)
                branches[mode] = changed
            branches["idle"] = card.native.wait(rule, branches["idle"], card.capacity.CONTACT_TICKS)
            if not card.capacity._same_graph(a.graph, branches["sham"].graph):
                raise AssertionError("A later no-write contact changed the installed graph")
        if item + 1 in counts:
            for mode, s in branches.items():
                state = card.native.wait(rule, s, 4)
                probes[mode].append(card.capacity._probe(rule, state, tape, 0).side.numpy())
    return {mode: np.stack(p) for mode, p in probes.items()}, target


def cells(saved, counts):
    target = saved["truth"]
    values = {}
    for mode in MODES:
        result = (saved[mode] == target).reshape(len(counts), -1, 4).mean(2)
        values.update({f"{mode}/{count}": row for count, row in zip(counts, result, strict=True)})
        values.update({f"decrement/{mode}/{count}": row - result[0]
                       for count, row in zip(counts[1:], result[1:], strict=True)})
    for count in counts[1:]:
        values[f"write_minus_sham/{count}"] = values[f"write/{count}"] - values[f"sham/{count}"]
        values[f"sham_minus_idle/{count}"] = values[f"sham/{count}"] - values[f"idle/{count}"]
    return values


@torch.no_grad()
def run(stage):
    torch.set_num_threads(1)
    plan = archive.read(PLAN)
    counts = plan["counts"]
    proof = archive.read(card.OUT / f"transfer-stage-{stage}-custody.json")
    previous = archive.read(archive.REPORT)["cohorts"]
    combinations = {"uniform": ("uniform",), "native": ("native",), "random": ("random",),
                    "uncertain": ("uncertain",), "native_body_uniform_rule": ("native", "uniform"),
                    "uniform_body_native_rule": ("uniform", "native")}
    result = {}
    for cohort, seed in plan["cohorts"].items():
        tape = card.capacity.make_tape("odor", seed, plan["worlds"], max(counts)-1)
        births = card.swap.COHORTS[cohort]
        interval = stats.Crossed(len(births), tape.worlds, seed+41, 4000)
        by_model = {name: [] for name in combinations}
        identity = {}
        for birth in births:
            sources = {"uniform": previous[cohort]["experts"][f"uniform/{birth}"],
                       **{arm: group["models"][str(birth)] for arm, group in proof["checkpoints"][cohort].items()}}
            for ref in sources.values():
                archive.verify(card.ROOT / ref["path"], ref["sha256"])
            data = card.DATA / "transfer" / cohort
            data.mkdir(parents=True, exist_ok=True)
            args = archive.model(cohort, "uniform")
            parent = card.checkpoint.load_checkpoint(args[0], "odor", args[1], birth, args[2])
            originals = {"uniform": parent,
                         **{arm: card.checkpoint.load_checkpoint(card.DATA / "core" / cohort, "odor", arm, birth, stage)
                            for arm in proof["checkpoints"][cohort]}}
            actions = select_support(*parent, tape)
            for name, components in combinations.items():
                graph, rule = originals[components[0]][0], originals[components[-1]][1]
                raw, target = replay(graph, rule, tape, actions, counts)
                arrays = dict(raw, truth=target, forced_side=actions[0].numpy(), earned=actions[1].numpy())
                path = data / f"retention-{stage}-{name}-{birth}.npz"
                io._write_raw(path, **arrays)
                identity[f"{name}/{birth}"] = dict(trace=custody.record(path), sources=[sources[k] for k in components])
                by_model[name].append(cells(arrays, counts))
            print(cohort, stage, birth, flush=True)
        bucket = {name: {k: np.stack([v[k] for v in results]) for k in results[0]} for name, results in by_model.items()}
        pairs = (("native", "uniform"), ("random", "uniform"), ("uncertain", "uniform"),
                 ("native", "random"), ("native", "uncertain"), ("native_body_uniform_rule", "uniform"),
                 ("uniform_body_native_rule", "uniform"), ("native", "native_body_uniform_rule"),
                 ("native", "uniform_body_native_rule"))
        interaction = {
            k: interval(v - bucket["native_body_uniform_rule"][k] -
                        bucket["uniform_body_native_rule"][k] + bucket["uniform"][k])
            for k, v in bucket["native"].items()
        }
        result[cohort] = dict(births=births, models={name: {k: interval(v) for k, v in rows.items()}
                                                  for name, rows in bucket.items()},
                              contrasts={f"{a} minus {b}": {k: interval(v - bucket[b][k]) for k, v in bucket[a].items()}
                                         for a, b in pairs}, body_rule_interaction=interaction, hashes=identity)
    io._write_json(card.OUT / f"transfer-retention-{stage}.json",
                   dict(stage=stage, worlds=plan["worlds"], source=custody.record(Path(__file__)),
                        plan=custody.record(PLAN), cohorts=result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int)
    args = parser.parse_args()
    run(args.stage)
