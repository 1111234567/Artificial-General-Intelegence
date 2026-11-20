"""Trace two frozen triangle graphs on their prespecified first validation geometry."""

from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_triangle_train as tri, self_param_writer as writer
from .local_trace_eval import digest


ROOT = tri.revisit.ROOT
STEP = 64000
WORLD_INDEX = 0
ORDERINGS = tuple(permutations(range(3)))
COHORTS = (("original", 731), ("independent", 811))
HISTORIES = (("+++", 7), ("---", 0))
ITEMS = "ABC"
NAME = "triangle-lived-example-64000"


def _side(value):
    return "右" if value > 0 else "左"


def _reaction(rule, state, left, right, native_action):
    stimulus = writer.sensory(state.graph, left, right)
    ticks = []
    for _ in range(tri.native.REACTION_TICKS):
        state = tri.native._advance(rule, state, stimulus, 1)
        ticks.append(state.hidden[:, (10, 11)])
    pair = torch.stack(ticks, dim=1)
    difference = pair[:, :, 1] - pair[:, :, 0]
    torch.testing.assert_close(pair[:, -1], native_action.state.hidden[:, (10, 11)], rtol=0, atol=0)
    torch.testing.assert_close(2.5 * difference[:, -1], native_action.motor_logit, rtol=0, atol=0)
    return pair, difference


def _record(lane, item, flip, draw, action, trace):
    pair, difference = trace
    side = float(action.side[lane])
    return dict(item=ITEMS[item], orientation=int(flip), draw=float(draw), side=_side(side),
                probability_right=float(action.motor_logit[lane].sigmoid()),
                probability_selected=float((side * action.motor_logit[lane]).sigmoid()),
                logit_right=float(action.motor_logit[lane]),
                motor_node_10=pair[lane, :, 0].tolist(), motor_node_11=pair[lane, :, 1].tolist(),
                motor_difference=difference[lane].tolist())


def _relative(file):
    return file.relative_to(ROOT).as_posix()


@torch.no_grad()
def run_example(cohort, birth):
    validation_file = tri.EVIDENCE / f"triangle-orders-long-{cohort}-{STEP}.json"
    validation = json.loads(validation_file.read_text(encoding="utf8"))
    if validation["step"] != STEP or birth not in validation["births"] or not validation["all_orders"]:
        raise ValueError("Expected the fixed 64k all-orders validation")
    source = tri.DATA / f"triangle-long-{cohort}"
    check = tri.old_fit.checkpoint_path(source, "odor", "direct", birth, STEP)
    graph, rule = tri.old_fit.load_checkpoint(source, "odor", "direct", birth, STEP)

    whole = tri.make_tape(validation["seed"], birth, validation["worlds"], -.5)
    count = len(tri.LABELS)
    begin = WORLD_INDEX * count
    tape = {name: value[begin:begin + count] for name, value in whole.items()}
    order = tape["order"][0].tolist()
    archived_order = ORDERINGS.index(tuple(order))
    archive = (tri.DATA / "long-order-verification" / "order-validation" /
               f"long-{cohort}" / f"{birth}-{STEP}.npz")
    archive_sha = digest(archive)
    if archive_sha != validation["sha256"][archive.name]:
        raise AssertionError(f"Archived validation fingerprint differs for birth {birth}")

    reference = tri.trials(graph, rule, tape, "active")
    with np.load(archive, allow_pickle=False) as raw:
        archived_logits = raw[f"active_logit_{archived_order}"][begin:begin + count]
        archived_choices = raw[f"active_selected_{archived_order}"][begin:begin + count]
    np.testing.assert_array_equal(reference["logit"], archived_logits)
    np.testing.assert_array_equal(reference["selected"], archived_choices)

    state = tri.native.start(graph, count)
    view = rule.expand(count)
    lanes = torch.arange(count)
    histories = {label: {"study": [], "revisit": []} for label, _ in HISTORIES}
    for contact in range(3):
        item, flip = tape["order"][:, contact], tape["study_flip"][:, contact]
        left, right = tri.binding.present(tape, item, flip)
        action, commit, earned, received = tri._contact(view, state, tape, lanes, contact, "active")
        torch.testing.assert_close(earned, received, rtol=0, atol=0)
        trace = _reaction(view, state, left, right, action)
        for label, lane in HISTORIES:
            entry = _record(lane, int(item[lane]), flip[lane], tape["study_u"][lane, contact, 0], action, trace)
            histories[label]["study"].append(dict(entry, taste=int(received[lane]), gap_after_ticks=4))
        state = tri.native.wait(view, commit.state, 4)

    for item in range(3):
        flip = torch.ones(count)
        item_tensor = torch.full((count,), item)
        left, right = tri.binding.present(tape, item_tensor, flip)
        draw = tape["probe_u"][:, item]
        action = tri.native.act(view, state, left, right, draw)
        other_left, other_right = tri.binding.present(tape, item_tensor, -flip)
        mirrored = tri.native.act(view, state, other_left, other_right, 1 - draw)
        np.testing.assert_array_equal(action.motor_logit.numpy(), reference["logit"][:, item, 0])
        np.testing.assert_array_equal(action.side.numpy(), reference["selected"][:, item, 0])
        np.testing.assert_array_equal(mirrored.motor_logit.numpy(), reference["logit"][:, item, 1])
        np.testing.assert_array_equal(-mirrored.side.numpy(), reference["selected"][:, item, 1])
        trace = _reaction(view, state, left, right, action)
        other_trace = _reaction(view, state, other_left, other_right, mirrored)
        for label, lane in HISTORIES:
            goal = float(tape["law"][lane, item])
            entry = _record(lane, item, flip[lane], draw[lane], action, trace)
            mirror_entry = _record(lane, item, -flip[lane], 1 - draw[lane], mirrored, other_trace)
            mirror_entry.update(taste=None, target_side=_side(-goal), correct=float(mirrored.side[lane]) == -goal)
            histories[label]["revisit"].append(
                dict(entry, taste=None, target_side=_side(goal), correct=float(action.side[lane]) == goal,
                     mirrored_orientation=mirror_entry))

    for history in histories.values():
        history["correct_revisits"] = sum(out["correct"] for out in history["revisit"])
        history["correct_mirrored_revisits"] = sum(out["mirrored_orientation"]["correct"]
                                                   for out in history["revisit"])
    return dict(cohort=cohort, birth=birth, checkpoint={"path": _relative(check), "sha256": digest(check)},
                archived_validation={"path": _relative(archive), "sha256": archive_sha, "order_index": archived_order},
                seed=validation["seed"], validation_worlds=validation["worlds"], geometry_index=WORLD_INDEX, rho=-.5,
                right_vectors=tape["right"][0].tolist(), sampled_order="".join(ITEMS[i] for i in order),
                histories=histories)


def markdown(report):
    lines = ["# 64k 三角形模型的一次固定回放", "",
             "出生 731 和 811 各取原验证轨迹的第一个几何，保留轨迹抽中的呈现顺序与动作随机数。"
             "每个出生的两段经历共享身体、几何和随机数，环境分别给 A、B、C 赋值 `+++` 与 `---`。"
             "实际收到的 taste 由当次朝向、选侧和环境赋值共同决定。", "",
             "五个运动差按时间排列，均为右运动节点 `h[11]` 减左运动节点 `h[10]`。"
             "右选概率为末 tick 差值乘 2.5 后取 sigmoid。三次回访各自从同一个学习后状态分叉，"
             "表格保留验证中的正向抽样，表后也列出同态镜像检查及实际失手。", ""]
    for example in report["examples"]:
        lines += [f"## 出生 {example['birth']}", "",
                  f"呈现顺序 **{example['sampled_order']}**。"
                  f"模型 `{example['checkpoint']['path']}`。", "",
                  f"SHA-256 `{example['checkpoint']['sha256']}`。", ""]
        for label, history in example["histories"].items():
            lines += [f"### `{label}` 实际经历", "",
                      f"回访答对 **{history['correct_revisits']}/3**。", "",
                      "| 阶段 | 时序物体 | 选侧 | taste | 右选概率 | 抽样数 | 五个 tick 的运动差 | 回访结果 |",
                      "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
            for phase, key in (("经历", "study"), ("回访", "revisit")):
                for action in history[key]:
                    delta = "，".join(f"{value:+.3f}" for value in action["motor_difference"])
                    taste = "—" if action["taste"] is None else f"{action['taste']:+d}"
                    status = "—" if key == "study" else f"{'正确' if action['correct'] else '失手'}，目标{action['target_side']}"
                    lines.append(f"| {phase} | {action['item']} | {action['side']} | {taste} | "
                                 f"{action['probability_right']:.4f} | {action['draw']:.4f} | {delta} | {status} |")
            mirrored = [action["mirrored_orientation"] for action in history["revisit"]]
            readings = [
                f"{action['item']} 选{action['side']}，右选概率 {action['probability_right']:.4f}"
                f"（{'正确' if action['correct'] else '失手，目标' + action['target_side']}）"
                for action in mirrored
            ]
            lines += ["", f"镜像回访答对 **{history['correct_mirrored_revisits']}/3**。"
                      + "；".join(readings) + "。", ""]
    lines += ["回访 logits 与原生 `trials` 函数、归档验证回放逐值相同。"
              "JSON 保留全精度节点读数以及验证文件路径和指纹。", "",
              "从仓库根目录复现", "", "```powershell",
              r".\.venv\Scripts\python.exe -m experiments.pulse_game.interleave_triangle_example", "```", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tri.EVIDENCE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if tri.native.REACTION_TICKS != 5:
        raise AssertionError("Expected the fixed five reaction ticks")
    report = dict(checkpoint_step=STEP, prespecified_births=[birth for _, birth in COHORTS],
                  geometry_index=WORLD_INDEX, chosen_revisit_orientation=1,
                  examples=[run_example(cohort, birth) for cohort, birth in COHORTS])
    tri.revisit._write_json(args.out / f"{NAME}.json", report)
    rendered = markdown(report)
    tri.revisit._atomic(args.out / f"{NAME}.md", lambda file: file.write_text(rendered, encoding="utf8"))
    print(rendered)


if __name__ == "__main__":
    main()
