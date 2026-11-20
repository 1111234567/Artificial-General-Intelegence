"""Recheck frozen exact-triangle behavior after matched neutral waits."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_triangle_train as triangle
from .interleave_confirm_triangle import permuted_tape
from .interleave_induction_report import Nested
from .interleave_interference import DATA, EVIDENCE


ORDERS = tuple(permutations(range(3)))
WAITS = {"fixed_4": (4, 4, 4), "random_0_8": None, "fixed_16": (16, 16, 16),
         "fixed_64": (64, 64, 64), "last_64": (4, 4, 64)}
SEED = 781351
THRESHOLD = .05


@torch.no_grad()
def trials(graph, rule, tape, schedule):
    net = triangle.native
    count = len(tape["law"])
    rule = rule.expand(count)
    rows = torch.arange(count)
    state = net.start(graph, count)
    contacts = []
    for i in range(3):
        action, commit, earned, _ = triangle._contact(rule, state, tape, rows, i, "active")
        state = commit.state
        contacts.append(torch.stack((action.side, earned), dim=1))
        if schedule is None:
            state = net.wait(rule, state, tape["wait"][:, i], max_ticks=8)
        else:
            state = net.wait(rule, state, schedule[i])
    logits, chosen = [], []
    # All six action reads leave the pre-probe state untouched.
    for item in range(3):
        actions = []
        for orientation in (1., -1.):
            left, right = triangle.binding.present(tape, torch.full((count,), item), torch.full((count,), orientation))
            draw = tape["probe_u"][:, item]
            actions.append(net.act(rule, state, left, right, draw if orientation > 0 else 1 - draw))
        logits.append(torch.stack((actions[0].motor_logit, actions[1].motor_logit), dim=1))
        chosen.append(torch.stack((actions[0].side, -actions[1].side), dim=1))
    return (torch.stack(logits, dim=1).numpy(), torch.stack(chosen, dim=1).numpy().astype(np.int8),
            torch.stack(contacts, dim=1).numpy().astype(np.int8))


def indicators(logit, side, law):
    """Keep order, geometry, law and both motor orientations aligned."""
    logit = logit.reshape(len(ORDERS), *law.shape, 2)
    side = side.reshape(logit.shape)
    margin = .5 * (logit[..., 0] - logit[..., 1]) * law[None]
    margin_all_three = (margin > THRESHOLD).all(3)
    hard_pair = margin_all_three[:, :, [0, 7]].all(2)
    actual_all_three = (side == law[None, ..., None]).all(3)
    return dict(hard_pair_per_order=hard_pair,
                hard_pair_all_orders=hard_pair.all(0)[None],
                all_eight_maps_all_orders=margin_all_three.all((0, 2))[None],
                hard_margin_all_three=margin_all_three[:, :, [0, 7]],
                hard_actual_all_three=actual_all_three[:, :, [0, 7]],
                all_negative_actual_all_three=actual_all_three[:, :, 0],
                all_positive_actual_all_three=actual_all_three[:, :, 7],
                mixed_margin_all_three=margin_all_three[:, :, 1:7],
                mixed_actual_all_three=actual_all_three[:, :, 1:7])


def evaluate(cohort, step, raw_dir):
    ref_path = EVIDENCE / f"triangle-orders-long-{cohort}-{step}.json"
    manifest = json.loads(ref_path.read_text(encoding="utf8"))
    if not manifest["all_orders"] or manifest["step"] != step:
        raise ValueError(f"Need the archived six-order validation at {ref_path}")
    births, worlds, seed = (manifest[k] for k in ("births", "worlds", "seed"))
    checkpoint_dir = DATA / f"triangle-long-{cohort}"
    base_dir = DATA / "long-order-verification/order-validation" / f"long-{cohort}"
    metrics = {mode: [] for mode in WAITS}
    checks = {}
    for birth in births:
        checkpoint = triangle.old_fit.checkpoint_path(checkpoint_dir, "odor", "direct", birth, step)
        checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
        graph, rule = triangle.old_fit.load_checkpoint(checkpoint_dir, "odor", "direct", birth, step)
        if sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
            raise ValueError(f"Checkpoint changed while loading {checkpoint}")
        base = triangle.make_tape(seed, birth, worlds, -.5)
        law = base["law"].numpy().reshape(worlds, 8, 3)
        tapes = [permuted_tape(base, pair) for pair in ORDERS]
        raw = {"law": law, "order": np.array(ORDERS), "right": base["right"].numpy()[::8],
               "study_flip": np.stack([t["study_flip"].numpy()[::8] for t in tapes]),
               "study_u": np.stack([t["study_u"].numpy()[::8] for t in tapes]),
               "probe_u": base["probe_u"].numpy()[::8],
               "object_bound_wait": np.stack([t["wait"].numpy()[::8] for t in tapes])}
        canonical = [t["wait"].numpy()[::8][:, np.argsort(order)] for t, order in zip(tapes, ORDERS, strict=True)]
        if not all(np.array_equal(w, canonical[0]) for w in canonical):
            raise AssertionError("Random waits must follow the associated object through all six orders")
        for name, schedule in WAITS.items():
            logit, side, contacts = (np.stack(rows) for rows in zip(*(trials(graph, rule, t, schedule) for t in tapes),
                                                                  strict=True))
            raw.update({f"{name}/logit": logit, f"{name}/side": side, f"{name}/contact": contacts})
            if not np.array_equal(contacts[:, :, 0], raw["fixed_4/contact"][:, :, 0]):
                raise AssertionError("Timing branches differed before the first wait")
            if name == "last_64" and not np.array_equal(contacts, raw["fixed_4/contact"]):
                raise AssertionError("Only changing the final wait must preserve all study actions and tastes")
            metrics[name].append(indicators(logit, side, law))
        archive = base_dir / f"{birth}-{step}.npz"
        baseline_hash = sha256(archive.read_bytes()).hexdigest()
        if baseline_hash != manifest["sha256"][archive.name]:
            raise ValueError(f"Recorded baseline hash differs for {archive}")
        with np.load(archive, allow_pickle=False) as expected:
            np.testing.assert_array_equal(law, expected["laws"])
            for j in range(len(ORDERS)):
                # tobytes also rejects changes of sign on stored zeros.
                for output, key in (("logit", "active_logit"), ("side", "active_selected")):
                    a, b = raw[f"fixed_4/{output}"][j], expected[f"{key}_{j}"]
                    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                        raise AssertionError(f"Baseline mismatch for {cohort} {birth} {step} order {j} {output}")
            np.testing.assert_array_equal(metrics["fixed_4"][-1]["hard_pair_per_order"], expected["paired"])
        destination = raw_dir / cohort / f"{birth}-{step}.npz"
        triangle.revisit._write_raw(destination, **raw)
        checks[str(birth)] = {"checkpoint": checkpoint_hash, "original_baseline": baseline_hash,
                              "raw": sha256(destination.read_bytes()).hexdigest(),
                              "raw_file": destination.relative_to(raw_dir).as_posix()}
        print(f"triangle wait {cohort} {step} {birth}: fixed-4 bitwise verified", flush=True)

    bs = Nested(len(births), worlds, SEED, 3000)
    modes, by_mode = {}, {}
    for mode, records in metrics.items():
        by_mode[mode] = {}
        modes[mode] = {}
        for metric in records[0]:
            rows = [record[metric] for record in records]
            whole = np.stack(rows)
            # World is the second axis of each per-birth array. Keep all order, law and orientation outcomes
            # for the same geometry together before the within-birth resample.
            world = np.moveaxis(whole, 2, 1).reshape(len(births), worlds, -1).mean(2)
            by_mode[mode][metric] = world
            modes[mode][metric] = dict(**bs.interval(world), successes=int(whole.sum()), trials=whole.size,
                                       counts_by_birth={str(birth): int(row.sum()) for birth, row in zip(births, rows)})
    diff = {mode: {metric: bs.interval(values - by_mode["fixed_4"][metric]) for metric, values in row.items()}
            for mode, row in by_mode.items() if mode != "fixed_4"}
    return dict(births=births, worlds_per_birth=worlds, seed=seed, baseline_logit_and_choice_bitwise=True,
                baseline_report_sha256=sha256(ref_path.read_bytes()).hexdigest(), archive_sha256=checks,
                results=modes, differences_from_fixed_4=diff)


def write_note(path, summary):
    def row(metric):
        return f"{metric['mean']:.2%}（{metric['successes']}/{metric['trials']}）"

    lines = ["# 三角形任务的等待时长压力测试", "",
             "冻结图保留每个物体的气味、私有规律、呈现方向和行动及编辑随机数。"
             "每次真实接触后只改变中性等待。0–8 拍沿用验证带已有的训练同分布随机间隔，随物体移动。"
             "只延长最后一次等待时，三次学习动作及实际味觉逐位相同。固定 4 拍的动作及原始 logit 均与存档逐位相同。", "",
             "严格联合要求 +++ 与 --- 两条独立规律历史的六个对称化余量全部大于 0.05。"
             "每顺序把同一几何的两条规律历史合成一项。实际全对按单历史、单运动方向的三个各自分叉动作计分。"
             "先保留同一几何的六顺序、两方向及所有规律配对，再在出生内重采样几何。"
             "此处的时间单位是原生神经时钟。", "",
             "| 出生组与训练步数 | 三次接触后等待 | 严格联合，每顺序 | 严格联合，六顺序均通过 | 困难两图实际全对 | 其他六图实际全对 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    captions = {"fixed_4": "4、4、4", "random_0_8": "按物体随机 0–8", "fixed_16": "16、16、16",
                "fixed_64": "64、64、64", "last_64": "4、4、64"}
    for cohort, name in (("original", "原出生"), ("independent", "独立出生")):
        for step, part in sorted(summary["cohorts"].get(cohort, {}).items(), key=lambda pair: int(pair[0])):
            for mode, scores in part["results"].items():
                selected = [row(scores[key]) for key in ("hard_pair_per_order", "hard_pair_all_orders",
                                                          "hard_actual_all_three", "mixed_actual_all_three")]
                lines.append("| " + " | ".join([f"{name} {int(step)//1000}k", captions[mode], *selected]) + " |")
    lines.extend(("", "[配对区间、逐出生分母与文件指纹](triangle-wait-stress.json) · "
                       "[复现代码](../../../experiments/pulse_game/interleave_triangle_wait.py)"))
    triangle.revisit._atomic(path, lambda tmp: tmp.write_text("\n".join(lines) + "\n", encoding="utf8"))


def run(steps, cohorts, raw_dir, evidence):
    torch.set_num_threads(1)
    file = evidence / "triangle-wait-stress.json"
    source = sha256(Path(__file__).read_bytes()).hexdigest()
    summary = dict(source_sha256=source, threshold=THRESHOLD, schedule=WAITS,
                   interval="3,000 paired birth-then-independent-world bootstrap resamples. "
                            "All orders, eight laws and both motor orientations are retained within geometry.",
                   random_wait="Use the existing uniform 0–8 validation tape gaps, which match the training distribution. "
                               "The waits are moved with each object when its order changes.",
                   raw_axis={"logit/side": ["order", "world*law", "object", "orientation"],
                             "contact": ["order", "world*law", "chronological visit", "side/earned taste"],
                             "object_bound_wait": ["order", "world", "chronological visit"]}, cohorts={})
    if file.is_file():
        first = json.loads(file.read_text(encoding="utf8"))
        if first["source_sha256"] != source:
            raise ValueError(f"Different wait evaluator already populated {file}")
        summary["cohorts"] = first["cohorts"]
    for cohort in cohorts:
        for step in steps:
            scoped = summary["cohorts"].setdefault(cohort, {})
            scoped[str(step)] = evaluate(cohort, step, raw_dir)
            triangle.revisit._write_json(file, summary)
            write_note(evidence / "triangle-wait-stress.md", summary)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=(32000, 64000))
    parser.add_argument("--cohorts", choices=("original", "independent"), nargs="+", default=("original", "independent"))
    parser.add_argument("--raw", type=Path, default=DATA / "triangle-wait-stress")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    run(args.steps, args.cohorts, args.raw, args.evidence)
