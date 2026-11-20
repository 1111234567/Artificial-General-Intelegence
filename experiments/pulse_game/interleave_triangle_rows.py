"""Locate exact-triangle retrieval effects with held-out receiver-row grafts."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from . import interleave_triangle_train as tri
from . import interleave_triangle_components as components
from .interleave_confirm_triangle import permuted_tape
from .interleave_induction_report import Nested
from .interleave_interference import DATA, EVIDENCE
from .interleave_triangle_wait import ORDERS


CAL_SEED = 6210631
TEST_SEED = 6210673
STEP = 64000
REGIONS = {"sensory_rows": tuple(range(10)), "motor_rows": (10, 11), "other_rows": tuple(range(12, 20))}
ARMS = ("intact", "top_1", "matched_1", "top_3", "matched_3", "opposite_graph", "opposite_activity",
        "opposite_all", *REGIONS)


def hard_tape(seed, birth, worlds):
    tape = tri.make_tape(seed, birth, worlds, -.5)
    rows = (torch.arange(worlds)[:, None] * 8 + torch.tensor([0, 7])).flatten()
    return {key: value[rows] for key, value in tape.items()}


def repeat_lanes(values, variants):
    return values.repeat((variants,) + (1,) * (values.ndim - 1))


def graft(state, donor, select):
    """Replace the full incoming receiver row while retaining the recipient's live state."""
    n = len(select)
    graph = tri.native.Graph(
        torch.where(select[..., None], donor.graph.weights, state.graph.weights).flatten(0, 1),
        torch.where(select, donor.graph.bias, state.graph.bias).flatten(0, 1),
        torch.where(select[..., None], donor.graph.present, state.graph.present).flatten(0, 1),
    )
    return tri.native.State(graph, *(repeat_lanes(getattr(state, key), n) for key in
                                    ("hidden", "sender_trace", "receiver_trace")))


def batch_answers(state, donor, mask, tape):
    expanded = {key: repeat_lanes(tape[key], len(mask)) for key in ("left", "right", "law", "probe_u")}
    combined = graft(state, donor, mask)
    raw = components.answers(combined, expanded)
    return tuple(val.reshape(len(mask), len(tape["law"]) // 2, 2, 3, 2) for val in raw), combined.graph.present


def stats(logit, action, law):
    reward = action == law[None, ..., None]
    margin = .5 * (logit[..., 0] - logit[..., 1]) * law[None]
    return dict(own_three=reward.all(3), donor_three=(~reward).all(3), own_each=reward,
                both_hard_margin=(margin > .05).all((2, 3)), mean_signed_margin=margin)


def row_dose(state, donor):
    return torch.stack(((state.graph.weights - donor.graph.weights).abs().sum(2),
                        (state.graph.bias - donor.graph.bias).abs(),
                        (state.graph.present != donor.graph.present).sum(2)), dim=-1).numpy()


def mask_rows(lanes, width, rows):
    mask = torch.zeros((lanes, width), dtype=torch.bool)
    mask[:, list(rows)] = True
    return mask


def connected(adjacency):
    """Check reachability from one node both along and against directed connections."""
    return _reachable(adjacency) & _reachable(adjacency.transpose(1, 2))


def _reachable(adjacency):
    found = torch.zeros_like(adjacency[:, 0])
    found[:, 0] = True
    for _ in range(adjacency.shape[1] - 1):
        grown = found | (adjacency & found[:, None, :]).any(dim=2)
        if torch.equal(grown, found):
            break
        found = grown
    return found.all(1)


def match_rows(dose, target, eligible_sets, floor):
    """Find the nearest changed mass before probing. Match the same rows in each law pair."""
    if not np.array_equal(dose[::2], dose[1::2]):
        raise AssertionError("Opposite laws must have symmetric absolute graph differences")
    fixed = dose[::2]
    desired = fixed[:, target, :].sum(1)
    candidates = fixed[:, eligible_sets, :].sum(2)
    discrepancy = ((candidates - desired[:, None, :]) / (desired[:, None, :] + floor)) ** 2
    rows = eligible_sets[discrepancy.sum(-1).argmin(1)]
    mask = np.zeros(dose.shape[:2], dtype=bool)
    lane = np.arange(len(mask))[:, None]
    mask[lane, rows.repeat(2, axis=0)] = True
    return torch.from_numpy(mask)


@torch.no_grad()
def calibrate(graph, inherited, base):
    n, width = len(base["law"]), graph.width
    donor_indices = torch.arange(n) ^ 1
    rule = inherited.expand(n)
    masks = torch.cat((torch.zeros((1, n, width), dtype=torch.bool),
                       torch.eye(width, dtype=torch.bool)[:, None].expand(-1, n, -1)))
    rows, doses, raw = [], [], {}
    for order in ORDERS:
        tape = permuted_tape(base, order)
        state, _ = components.learn(graph, rule, tape, "active")
        donor = state.subset(donor_indices)
        (logit, action), _ = batch_answers(state, donor, masks, tape)
        law = tape["law"].numpy().reshape(-1, 2, 3)
        rows.append(stats(logit, action, law)["own_three"].mean((1, 2, 3)))
        doses.append(row_dose(state, donor))
        raw["cal/" + "".join(map(str, order)) + "/logit"] = logit
        raw["cal/" + "".join(map(str, order)) + "/side"] = action
    means = np.mean(rows, axis=0)
    loss = means[0] - means[1:]
    rank = np.argsort(-loss, kind="stable")
    floor = np.mean(doses, axis=(0, 1, 2)).clip(1e-5)
    floor[-1] = 1.
    raw["cal/dose"] = np.stack(doses)
    return rank, loss, floor, raw


@torch.no_grad()
def assess(graph, inherited, base, rank, floor, seed):
    n, width = len(base["law"]), graph.width
    rule, donor_indices = inherited.expand(n), torch.arange(n) ^ 1
    rng = np.random.default_rng(seed)
    remaining = lambda k: [i for i in range(width) if i not in rank[:k]]
    candidates = [rng.permutation(np.array(list(combinations(remaining(k), k)))) for k in (1, 3)]
    traces, measurements, dose_report, topology = {}, [], [], []
    for order in ORDERS:
        tape = permuted_tape(base, order)
        state, _ = components.learn(graph, rule, tape, "active")
        donor = state.subset(donor_indices)
        dose = row_dose(state, donor)
        masks = torch.stack((mask_rows(n, width, ()), mask_rows(n, width, rank[:1]),
                             match_rows(dose, rank[:1], candidates[0], floor), mask_rows(n, width, rank[:3]),
                             match_rows(dose, rank[:3], candidates[1], floor), mask_rows(n, width, range(width)),
                             *(mask_rows(n, width, ids) for ids in REGIONS.values())))
        if (masks[1] & masks[2]).any() or (masks[3] & masks[4]).any():
            raise AssertionError("A matched control used a nominated row")
        if not ((masks[2].sum(1) == 1).all() and (masks[4].sum(1) == 3).all()):
            raise AssertionError("Wrong matched row count")
        (logit, action), presence = batch_answers(state, donor, masks, tape)
        h_only = tri.native.swap(state, donor, ("hidden", "trace"))
        hs = [components.answers(view, tape) for view in (h_only, donor)]
        hs_logit, hs_side = (np.stack([row[i].reshape(-1, 2, 3, 2) for row in hs]) for i in range(2))
        np.testing.assert_allclose(hs_logit[1], logit[0, :, ::-1], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(hs_side[1], action[0, :, ::-1])
        logit = np.concatenate((logit[:6], hs_logit, logit[6:]))
        action = np.concatenate((action[:6], hs_side, action[6:]))
        raw_connected = connected(presence).numpy().reshape(len(masks), -1, 2)
        raw_connected = np.concatenate((raw_connected[:6], raw_connected[[0, 5]], raw_connected[6:]))
        applied = (dose[None] * masks.numpy()[..., None]).sum(2).reshape(len(masks), -1, 2, 3)
        applied = np.concatenate((applied[:6], applied[[0, 5]], applied[6:]))
        raw_masks = masks.numpy()
        raw_masks = np.concatenate((raw_masks[:6], raw_masks[[0, 5]], raw_masks[6:]))
        law = tape["law"].numpy().reshape(-1, 2, 3)
        measurements.append(stats(logit, action, law))
        dose_report.append(applied)
        topology.append(raw_connected)
        tag = "test/" + "".join(map(str, order))
        traces.update({f"{tag}/logit": logit, f"{tag}/side": action, f"{tag}/mask": raw_masks,
                       f"{tag}/dose": applied, f"{tag}/connected": raw_connected})
    def per_world(value):
        values = np.stack(value, axis=1)  # arm, order, world, ...
        return np.moveaxis(values, 2, 1).reshape(len(ARMS), -1, values.shape[1] * int(np.prod(values.shape[3:]))).mean(2)
    scores = {key: per_world([m[key] for m in measurements]) for key in measurements[0]}
    scores["invalid_topology"] = per_world([~v for v in topology])
    for i, score in enumerate(("weights_l1", "bias_l1", "support_bits")):
        scores[score] = per_world([x[..., i] for x in dose_report])
    return scores, traces


def summarize(cohort, births, matrices, worlds):
    samples = {name: np.stack([row[name] for row in matrices]) for name in matrices[0]}
    bs = Nested(len(births), worlds, TEST_SEED, 3000)
    outcome = {arm: {key: bs.interval(array[:, i]) for key, array in samples.items()}
               for i, arm in enumerate(ARMS)}
    comparisons = (("top_1", "matched_1"), ("top_3", "matched_3"), ("top_1", "intact"), ("top_3", "intact"),
                   ("opposite_graph", "intact"), ("opposite_activity", "intact"), ("opposite_all", "intact"))
    contrast = {}
    for left, right in comparisons:
        a, b = ARMS.index(left), ARMS.index(right)
        contrast[left + " minus " + right] = {key: bs.interval(v[:, a] - v[:, b]) for key, v in samples.items()}
    return dict(cohort=cohort, births=births, outcomes=outcome, contrasts=contrast)


def dose_overlap(raw, floor):
    """Post-hoc check on unchanged topology and closely balanced numeric changes."""
    law = raw["test/tape/law"].reshape(-1, 2, 3)
    rows = {}
    for left, right in (("top_1", "matched_1"), ("top_3", "matched_3")):
        i, j = ARMS.index(left), ARMS.index(right)
        outcomes = []
        for order in ORDERS:
            tag = "test/" + "".join(map(str, order))
            target, control = raw[f"{tag}/dose"][[i, j], :, 0].astype(float)
            near = (np.abs(target[:, :2] - control[:, :2]) <= .10 * (target[:, :2] + floor[:2])).all(1)
            keep = near & (target[:, 2] == 0) & (control[:, 2] == 0)
            actions = raw[f"{tag}/side"][[i, j]]
            own = (actions == law[None, ..., None]).all(3).mean((2, 3))
            donor = (actions == -law[None, ..., None]).all(3).mean((2, 3))
            outcomes.append(np.stack((keep, keep * (own[0] - own[1]), keep * (donor[0] - donor[1]))))
        rows[left + " minus " + right] = np.sum(outcomes, axis=0)
    return rows


def summarize_overlap(rows, births, worlds):
    bs = Nested(len(births), worlds, TEST_SEED, 3000)
    summary = {}
    for k in rows[0]:
        packed = np.stack([row[k] for row in rows])
        count = packed[:, 0]
        totals = count.sum(1)
        sampled_count = count[bs.roots[:, :, None], bs.worlds].sum(2)
        good_draws = (sampled_count > 0).all(1)
        entry = dict(eligible_pairs_by_birth={str(birth): int(n) for birth, n in zip(births, totals)},
                     possible_pairs_per_birth=len(ORDERS) * worlds)
        for i, metric in enumerate(("own_three", "donor_three"), start=1):
            sum_by_world = packed[:, i]
            by_birth = [float(num / den) if den else None for num, den in zip(sum_by_world.sum(1), totals)]
            result = {"by_birth": by_birth}
            if totals.all() and good_draws.any():
                sampled = sum_by_world[bs.roots[:, :, None], bs.worlds].sum(2)[good_draws]
                draws = (sampled / sampled_count[good_draws]).mean(1)
                result.update(mean=float(np.mean(by_birth)), ci95=np.quantile(draws, [.025, .975]).tolist(),
                              accepted_bootstrap_draws=int(good_draws.sum()))
            entry[metric] = result
        summary[k] = entry
    return summary


def write_note(path, report):
    def percent(value):
        return f"{value['mean']:.2%}"

    names = dict(intact="原状态", top_1="校准前 1 行", matched_1="匹配 1 行", top_3="校准前 3 行",
                 matched_3="匹配 3 行", opposite_graph="整图换供体", opposite_activity="活动和迹换供体",
                 opposite_all="整状态换供体", sensory_rows="全部感觉行", motor_rows="全部运动行",
                 other_rows="其余全部行")
    msg = ["# 三角形困难映射的接收行移植", "",
           "从同一个出生图经历相反私有价值，学习期间共享几何、六顺序、行动与编辑随机数。"
           "接收者只换指定接收节点的整行入边权重、偏置与支撑位，保留其自身活动及迹。"
           "三次查询均由同一末状态独立分叉，不再观察味觉或提交编辑。", "",
           f"每个出生用 {report['calibration_worlds']} 个校准几何按单行移植导致的实际三项全对降幅排名。"
           f"{report['test_worlds']} 个新几何上仅依干预前总边权差、偏置差和支撑变更数匹配其他行。"
           "集合级结果报告实际改变剂量及部分移植后有无破坏强连通。三类节点整组移植的行数各异。", "",
           "| 组别 | 干预 | 受体三项实际全对 | 供体三项实际全对 | 两种困难历史严格联合 |",
           "| --- | --- | ---: | ---: | ---: |"]
    for cohort, label in (("original", "原出生"), ("independent", "独立出生")):
        for arm, stats in report["cohorts"][cohort]["outcomes"].items():
            row = [percent(stats[k]) for k in ("own_three", "donor_three", "both_hard_margin")]
            msg.append("| " + " | ".join([label, names[arm], *row]) + " |")
    msg.extend(("", "事后剂量敏感性筛去两臂有支撑位改变的几何×顺序，并要求边权与偏置改变量接近。"
                    "前 1 行的差异在两组均保留。前 3 行有出生缺少重叠，整组效应仍包含改变量差异。", "",
                    "[完整区间、每出生选行、剂量与指纹](triangle-row-graft.json) · "
                     "[复现代码](../../../experiments/pulse_game/interleave_triangle_rows.py)"))
    tri.revisit._atomic(path, lambda p: p.write_text("\n".join(msg) + "\n", encoding="utf8"))


def run(raw, evidence, calibration_worlds=96, test_worlds=128):
    torch.set_num_threads(1)
    report = dict(step=STEP, calibration_seed=CAL_SEED, test_seed=TEST_SEED, calibration_worlds=calibration_worlds,
                  test_worlds=test_worlds, orders=list(ORDERS), law_histories=[[-1, -1, -1], [1, 1, 1]],
                  scoring="Own/donor three actions are taken from the same state, separately for each layout. Strict criterion "
                          "requires all six own-law-signed symmetrized margins > .05. Selection ranks calibration single-row "
                          "loss of actual own-layout joint accuracy, breaking exact ties by receiver index. Holdout controls "
                          "exclude nominated rows and minimize squared relative weight/bias L1 and topology mismatch; no "
                          "probe outcome enters matching. All orders, laws and orientation decisions stay paired within geometry.",
                  graft="Transplant receiver row of weights, bias and present bits from opposite-law full history. Keep "
                        "recipient h and traces. An arbitrary row transplant can change strong connectivity; report the "
                        "violation rate without filtering these trials. Grouped dose differences are descriptive.",
                  thresholds=dict(strict_margin=.05, bootstrap_draws=3000), arm_order=ARMS, row_regions=REGIONS,
                  posthoc_dose_overlap="Holdout geometry/order pairs need no support-bit change in either arm and absolute "
                                       "weight and bias L1 mismatch at most 10% of target plus calibration scale. Reweight "
                                       "within each birth to its eligible pairs. If any birth has zero overlap, retain all "
                                       "reported counts and give no cohort estimate.",
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), cohorts={})
    all_selected, counts = [], {}
    for cohort, roots in (("original", range(731, 737)), ("independent", range(811, 817))):
        artifacts, selected, outcomes, overlap = {}, {}, [], []
        for birth in roots:
            source = DATA / f"triangle-long-{cohort}"
            checkpoint = tri.old_fit.checkpoint_path(source, "odor", "direct", birth, STEP)
            graph, rule = tri.old_fit.load_checkpoint(source, "odor", "direct", birth, STEP)
            cal, holdout = hard_tape(CAL_SEED, birth, calibration_worlds), hard_tape(TEST_SEED, birth, test_worlds)
            cal_geometries = {v.tobytes() for v in cal["right"][::2].numpy()}
            if any(v.tobytes() in cal_geometries for v in holdout["right"][::2].numpy()):
                raise AssertionError("Calibration and holdout tapes must be distinct")
            rank, loss, floor, cal_raw = calibrate(graph, rule, cal)
            measured, test_raw = assess(graph, rule, holdout, rank, floor, [TEST_SEED, birth])
            for prefix, tape in (("cal", cal), ("test", holdout)):
                test_raw.update({prefix + "/tape/" + k: v.numpy() for k, v in tape.items()})
            overlap.append(dose_overlap(test_raw, floor))
            file = raw / cohort / f"{birth}.npz"
            tri.revisit._write_raw(file, **cal_raw, **test_raw, selected_rank=rank, calibration_loss=loss, dose_floor=floor)
            selected[str(birth)] = dict(top_1=int(rank[0]), top_3=rank[:3].tolist(), calibration_drop_by_row=loss.tolist(),
                                        holdout_top_1_minus_matched_1=float((measured["own_three"][1]-measured["own_three"][2]).mean()),
                                        holdout_top_3_minus_matched_3=float((measured["own_three"][3]-measured["own_three"][4]).mean()))
            all_selected.append(rank[:3])
            for arm, val in (("top_1", selected[str(birth)]["holdout_top_1_minus_matched_1"]),
                             ("top_3", selected[str(birth)]["holdout_top_3_minus_matched_3"])):
                counts.setdefault(arm, []).append(val < 0)
            outcomes.append(measured)
            artifacts[str(birth)] = dict(raw_file=file.relative_to(raw).as_posix(), raw_sha256=sha256(file.read_bytes()).hexdigest(),
                                         checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest())
            print(f"triangle row graft {cohort} {birth}: {rank[:3].tolist()} verified", flush=True)
        row = summarize(cohort, list(roots), outcomes, test_worlds)
        report["cohorts"][cohort] = {**row, "selected": selected, "artifacts": artifacts,
                                     "posthoc_dose_overlap": summarize_overlap(overlap, list(roots), test_worlds)}
    report["cross_births"] = dict(top_1_hits=dict(Counter(int(x[0]) for x in all_selected)),
                                 top_3_hits=dict(Counter(int(y) for x in all_selected for y in x)),
                                 lower_own_accuracy_than_matched={key: int(sum(vals)) for key, vals in counts.items()},
                                 births=12)
    file = evidence / "triangle-row-graft.json"
    tri.revisit._write_json(file, report)
    write_note(evidence / "triangle-row-graft.md", report)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DATA / "triangle-row-graft")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    run(args.raw, args.evidence)
