"""Read a frozen meta-evidence agent's latent regime information without affecting its actions."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_credit_train as credit
from . import interleave_eval as storage, interleave_meta_evidence as task, local_trace_graph as net


CAL_DOMAIN, TEST_DOMAIN, PERMUTATION_DOMAIN, BOOTSTRAP_DOMAIN = range(0x7EA510, 0x7EA514)
CAL_SEED, TEST_SEED = 514771, 902011
STAGES = ("before_X", "after_X3")
FAMILIES = ("hidden", "traces", "graph", "signed_all", "energy_only", "combined")
N_PERMUTATIONS = 32
RIDGE = .1
BOOTSTRAP = 8000
BRANCHES = ((8, "yoked"), (16, "yoked"), (8, "natural"), (8, "blank_prefix"), (0, "no_prefix"))


@dataclass(frozen=True)
class Replay:
    snapshots: dict
    side: torch.Tensor
    reward: torch.Tensor
    history: torch.Tensor


def _bank(worlds, n, split):
    seed, domain = (CAL_SEED, CAL_DOMAIN) if split == "cal" else (TEST_SEED, TEST_DOMAIN)
    cfg = task.Config(worlds=worlds, seed=seed)
    # One common independent tape bank is replayed by every birth and both arms.
    source_n = n or 8
    tape = task.batch(cfg, [0], 6000, source_n, "direct", domain=domain, conditioning=(task.P, task.H))
    if n:
        return tape
    trajectories = ("right", "factor", "u", "gap", "canonical", "orient", "ids")
    return {name: values[:, 2 * source_n:] if name in trajectories else values for name, values in tape.items()}


def _features(state, graph):
    h = state.hidden.numpy()
    sender, receiver = state.sender_trace.numpy(), state.receiver_trace.numpy()
    current = (state.graph.weights * state.graph.present).numpy()
    bias = state.graph.bias.numpy()
    base = (graph.weights * graph.present).numpy()
    change = current - base
    bias_change = bias - graph.bias.numpy()
    raw_trace = np.concatenate((sender.reshape(len(h), -1), receiver.reshape(len(h), -1)), axis=1)
    raw_graph = np.concatenate((current.reshape(len(h), -1), bias), axis=1)
    h_magnitude = np.abs(h)
    trace_energy = np.concatenate(((sender ** 2).sum(axis=2), (receiver ** 2).sum(axis=2)), axis=1)
    graph_energy = np.concatenate((np.abs(change).sum(axis=2), (change ** 2).sum(axis=2), np.abs(bias_change),
                                   state.graph.present.numpy().sum(axis=2, dtype=np.float32)), axis=1)
    signed = np.concatenate((h, raw_trace, raw_graph), axis=1)
    energy = np.concatenate((h_magnitude, trace_energy, graph_energy), axis=1)
    return dict(hidden=np.concatenate((h, h_magnitude), axis=1),
                traces=np.concatenate((raw_trace, trace_energy), axis=1),
                graph=np.concatenate((raw_graph, graph_energy), axis=1), signed_all=signed,
                energy_only=energy, combined=np.concatenate((signed, energy), axis=1))


@torch.no_grad()
def observe(graph, rule, tape, n, branch):
    """Capture two states in the native episode while keeping the animal's final choice intact."""
    if branch not in {name for _, name in BRANCHES}:
        raise ValueError(f"Unknown replay branch {branch!r}")
    lanes = len(tape["regime"])
    if lanes % 8:
        raise ValueError("Each world must carry all eight regime and sign lanes")
    state, view = net.start(graph, lanes), rule.expand(lanes)
    same_x = branch != "natural"
    stable = torch.arange(lanes) & ~4
    before = 2 * n
    snapshots, history = {}, []
    for index in range(before + 4):
        if index and index % 4 == 0:
            state = credit.boundary(state, "all")
        if index:
            state = net.wait(view, state, tape["gap"][:, index], max_ticks=8)
        if index == before:
            snapshots[STAGES[0]] = _features(state, graph)
        if index == before + 3:
            snapshots[STAGES[1]] = _features(state, graph)
        right, draws = tape["right"][:, index], tape["u"][:, index]
        act = net.act(view, state, -right, right, draws[:, 0])
        if same_x and before <= index < before + 3:
            act = net.act(view, state, -right, right, draws[:, 0], side_override=act.side[stable])
        reward = act.side * tape["factor"][:, index]
        if index < before + 3:
            taste = torch.zeros_like(reward) if branch == "blank_prefix" and index < before else reward
            experience = net.observe(view, act, -right, right, taste)
            state = net.commit(view, experience, draws[:, 1:]).state
            history.append(torch.stack((act.side, taste), dim=1))
    seen = torch.stack(history, dim=1)
    if same_x:
        tastes = seen[:, before:].reshape(-1, 2, 4, 3, 2)
        if not torch.equal(tastes[:, 0], tastes[:, 1]):
            raise AssertionError("The X contact and taste must match literally across regimes")
    if branch in ("no_prefix", "blank_prefix"):
        for snapshot in snapshots.values():
            pool = snapshot["combined"].reshape(-1, 2, 4, snapshot["combined"].shape[-1])
            if not np.array_equal(pool[:, 0], pool[:, 1]):
                raise AssertionError("Private regime entered the state without signed prior experience")
    return Replay(snapshots, act.side, reward, seen)


def _targets(tape):
    regime = tape["regime"].numpy().reshape(-1, 8)
    if not (np.all(regime[:, :4] == 0) and np.all(regime[:, 4:] == 1)):
        raise ValueError("Each world must retain both ordered regimes and all their sign twins")
    for name in ("mirror", "x_sign"):
        values = tape[name].numpy().reshape(-1, 2, 4)
        if not np.array_equal(values[:, 0], values[:, 1]):
            raise ValueError("The same four sign twins must be available in both regimes")
    return (1 - 2 * regime).astype(np.float64)


def _shuffled_y(labels, n):
    rng = np.random.default_rng([PERMUTATION_DOMAIN, n, len(labels)])
    flips = rng.choice((-1., 1.), size=(len(labels), N_PERMUTATIONS))
    null = labels[:, :, None] * flips[:, None, :]
    return np.column_stack((labels.ravel(), null.reshape(-1, N_PERMUTATIONS)))


def ridge_scores(cal, test, labels):
    """Use only calibration moments, with one solve for true and whole-world shuffled labels."""
    cal, test = cal.astype(np.float64), test.astype(np.float64)
    mean, scale = cal.mean(axis=0), cal.std(axis=0)
    keep = scale > 1e-7
    if not keep.any():
        return np.zeros((len(test), labels.shape[1]), dtype=np.float64)
    x = torch.from_numpy((cal[:, keep] - mean[keep]) / scale[keep])
    z = torch.from_numpy((test[:, keep] - mean[keep]) / scale[keep])
    y = torch.from_numpy(labels)
    x_count, features = x.shape
    cov = x.T @ x / x_count + RIDGE * torch.eye(features, dtype=x.dtype)
    coef = torch.linalg.solve(cov, x.T @ (y - y.mean(dim=0)) / x_count)
    return (z @ coef + y.mean(dim=0)).numpy()


def _accuracy(scores, label):
    signed = scores * label
    return (signed > 1e-8) + .5 * (np.abs(signed) <= 1e-8)


def _matched_order(scores):
    s = scores.reshape(-1, 2, 4)
    delta = s[:, 0] - s[:, 1]
    return (delta > 1e-8).mean(axis=1) + .5 * (np.abs(delta) <= 1e-8).mean(axis=1)


def _readout(cal, test, labels, observed, branch, name, stage):
    pred = ridge_scores(cal.snapshots[stage][name], test.snapshots[stage][name], labels)
    positive = _accuracy(pred, observed.reshape(-1, 1)).reshape(*observed.shape, -1).mean(axis=1)
    score = positive[:, 0]
    null = positive[:, 1:]
    if branch in ("no_prefix", "blank_prefix") and (not np.all(score == .5) or not np.all(null == .5)):
        raise AssertionError("Chance heldout classification is exact on the negative controls")
    return {"accuracy": score, "shuffled": null.mean(axis=1), "vs_shuffled": score - null.mean(axis=1),
            "paired_order": _matched_order(pred[:, 0])}, pred.astype(np.float32)


def _behavior(tape, snapshots, n, branch):
    targets = _targets(tape)
    world_shape = targets.shape
    earned = snapshots.reward.numpy().reshape(world_shape) > 0
    majority = (tape["x_sign"] * tape["orient"][:, -1]).numpy()
    follow = (snapshots.side.numpy() == majority).reshape(world_shape)
    preserve = np.where(targets > 0, follow, ~follow)
    result = {"native_earned": earned.mean(1), "native_expected_policy": preserve.mean(1),
              "native_majority_gap": follow[:, :4].mean(1) - follow[:, 4:].mean(1)}
    raw = dict(final_side=snapshots.side.numpy(), final_reward=snapshots.reward.numpy(),
               history=snapshots.history.numpy(), labels=targets, majority_side=majority)
    if branch != "blank_prefix":
        public_side, llr = task.reference(tape, snapshots.history, n, task.P, task.H)
        payoff = tape["factor"][:, -1].numpy()
        result.update(public_class=_accuracy(llr.reshape(world_shape), targets).mean(1),
                      public_earned=(public_side * payoff > 0).reshape(world_shape).mean(1))
        raw.update(public_side=public_side, public_log_likelihood=llr)
    return result, raw


def _statistics(data, *, primary=False):
    values = np.stack(data)
    births, worlds = values.shape
    rng = np.random.default_rng([BOOTSTRAP_DOMAIN, births, worlds])
    ib = rng.integers(births, size=(BOOTSTRAP, births))
    iw = rng.integers(worlds, size=(BOOTSTRAP, worlds))
    boot = np.concatenate([values[ib[i:i + 250, :, None], iw[i:i + 250, None, :]].mean(axis=(1, 2))
                           for i in range(0, BOOTSTRAP, 250)])
    result = dict(mean=float(values.mean()), pointwise_95=np.quantile(boot, [.025, .975]).tolist(),
                  by_birth=values.mean(axis=1).tolist())
    if primary:
        result["simultaneous_95_two_stages"] = np.quantile(boot, [.0125, .9875]).tolist()
    return result


def check_checkpoints(run, arms, step):
    return {arm: age.ckpath(run, arm, step) for arm in arms
            if age.ckpath(run, arm, step).is_file()}


def run_arm(run, arm, step, births, cal_worlds, test_worlds):
    torch.set_num_threads(1)
    checkpoint = age.ckpath(run, arm, step)
    target = run / "state-readout" / f"{arm}-{step}"
    collected, digests = {}, {}
    banks = {(n, split): _bank(count, n, split) for n in (0, 8, 16)
             for split, count in (("cal", cal_worlds), ("test", test_worlds))}
    for birth in births:
        graph, rule = age.load_checkpoint(run, arm, birth, step)
        raw = {}
        for n, branch in BRANCHES:
            label = f"{n}/{branch}"
            train_tape, test_tape = (banks[n, split] for split in ("cal", "test"))
            cal = observe(graph, rule, train_tape, n, branch)
            test = observe(graph, rule, test_tape, n, branch)
            y_cal, expected = _shuffled_y(_targets(train_tape), n), _targets(test_tape)
            behavioral, evidence = _behavior(test_tape, test, n, branch)
            for key, values in behavioral.items():
                collected.setdefault(f"{label}/behavior/{key}", []).append(values)
            raw.update({f"{label}/behavior/{key}": values for key, values in evidence.items()})
            families = FAMILIES if branch == "yoked" else ("combined",)
            for stage in STAGES:
                for family in families:
                    metrics, scores = _readout(cal, test, y_cal, expected, branch, family, stage)
                    path = f"{label}/{stage}/{family}"
                    raw[f"{path}/scores"] = scores
                    for key, values in metrics.items():
                        collected.setdefault(f"{path}/{key}", []).append(values)
        path = target / f"{birth}.npz"
        storage._atomic_npz(path, raw)
        digests[str(birth)] = sha256(path.read_bytes()).hexdigest()
        print(f"{arm} {birth} readout complete", flush=True)
    results = {key: _statistics(values, primary=key.startswith("8/yoked/") and "/combined/" in key)
               for key, values in collected.items()}
    for n in (8, 16):
        for family in FAMILIES:
            for metric in ("accuracy", "paired_order"):
                before, after = (collected[f"{n}/yoked/{time}/{family}/{metric}"] for time in STAGES)
                results[f"{n}/yoked/after_minus_before/{family}/{metric}"] = _statistics(
                    [a - b for a, b in zip(after, before, strict=True)])
    raw_path = target.relative_to(storage.ROOT) if target.is_relative_to(storage.ROOT) else target
    report = dict(arm=arm, step=step, births=births, calibration_worlds=cal_worlds, test_worlds=test_worlds,
                  seeds=dict(calibration=[CAL_DOMAIN, CAL_SEED], test=[TEST_DOMAIN, TEST_SEED],
                             permutation=PERMUTATION_DOMAIN, bootstrap=BOOTSTRAP_DOMAIN),
                  checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(), payload_sha256=digests,
                  source_sha256={Path(f).name: sha256(Path(f).read_bytes()).hexdigest()
                                 for f in (__file__, task.__file__, net.__file__)},
                  results=results, raw_path=str(raw_path),
                  ridge=RIDGE, label_shuffles=N_PERMUTATIONS, bootstrap=BOOTSTRAP,
                  notes="Regime labels are visible to the offline readout only. At all times the graph retains its own "
                        "hidden units, node traces, weights and bias. The readout uses one state and fixed birth "
                        "parameters for graph energy, standardized on an independent calibration bank. The primary "
                        "pair is combined features at the two matched-X snapshots with 8 earlier visits per "
                        "object, and two-stage Bonferroni intervals. Secondary modes reuse the same diagnosis. "
                        "External readout output never affects a motor or a graph write.")
    storage._atomic_json(storage.EVIDENCE / f"meta-evidence-state-readout-{arm}-{step}.json", report)


def _percent(value):
    return f"{100*value:.1f}%"


def _interval(row, primary=False):
    lo, hi = row["simultaneous_95_two_stages" if primary else "pointwise_95"]
    return f"{_percent(row['mean'])} [{100*lo:.1f}, {100*hi:.1f}]"


def write_summary(step):
    docs = storage.EVIDENCE
    reports = {}
    for arm in task.ARMS:
        path = docs / f"meta-evidence-state-readout-{arm}-{step}.json"
        if path.exists():
            reports[arm] = json.loads(path.read_text(encoding="utf-8"))
    if not reports:
        return
    baseline = next(iter(reports.values()))
    lines = ["# 旧物经历能从图状态读出来吗", "",
             f"每种训练方式使用 {len(baseline['births'])} 个封存出生，各自用 {baseline['calibration_worlds']} 个世界校准，"
             f"再到 {baseline['test_worlds']} 个独立世界测验。每个世界保留两类规律和全部四种符号孪生。拟合依赖一个时刻"
             "的图状态，标准化只用校准世界。X 的三次实际触侧与味觉逐世界跨类别匹配。表中两时点联合读出的区间对这两个"
             "时点校正，其余区间逐项计算。", "",
             "## 主要条件", "",
             "每个旧物先接触八次。外部读出的评分是按单个状态分辨稳定有噪和可靠翻转。身体得分是最后真实动作的味觉，"
             "按照任务本身的随机性计分。", "",
             "| 固定检查点 | 闻到 X 前 | X 三味后 | 公开历史识别类别 | 原生自然动作获益 |",
             "|---|---:|---:|---:|---:|"]
    for arm, report in reports.items():
        data = report["results"]
        values = [_interval(data[f"8/yoked/{stage}/combined/accuracy"], primary=True) for stage in STAGES]
        values.extend(_interval(data[path]) for path in ("8/yoked/behavior/public_class", "8/natural/behavior/native_earned"))
        lines.append("| " + " | ".join((f"{arm} {step}", *values)) + " |")
    lines.extend(["", "| 固定检查点 | 读出减乱标，闻到 X 前 | 读出减乱标，X 三味后 | 无前史 / 有旧物无味觉 |",
                  "|---|---:|---:|---:|"])
    for arm, report in reports.items():
        data = report["results"]
        values = []
        for stage in STAGES:
            row = data[f"8/yoked/{stage}/combined/vs_shuffled"]
            lo, hi = row["simultaneous_95_two_stages"]
            values.append(f"{100*row['mean']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        controls = [_percent(data[f"{n}/{branch}/after_X3/combined/accuracy"]["mean"])
                    for n, branch in ((0, "no_prefix"), (8, "blank_prefix"))]
        lines.append("| " + " | ".join((f"{arm} {step}", *values, " / ".join(controls))) + " |")
    lines.extend(["", "## 分量与行为", "",
                  "下表给出各分量各自拟合的单状态准确率。活动使用有符号值和绝对值，痕迹加每节点能量。连接使用现有"
                  "有效权值和偏置，另加入相对固定出生图的入边变化量及入度。", "",
                  "| 固定检查点 | 时点 | 活动 h | 局部痕迹 | 权值与偏置 | 仅幅度和能量 |",
                  "|---|---|---:|---:|---:|---:|"])
    for arm, report in reports.items():
        data = report["results"]
        for stage in STAGES:
            values = [_percent(data[f"8/yoked/{stage}/{name}/accuracy"]["mean"])
                      for name in ("hidden", "traces", "graph", "energy_only")]
            lines.append("| " + " | ".join((f"{arm} {step}", "闻到 X 前" if stage == STAGES[0] else "X 三味后", *values)) + " |")
    lines.extend(("", "两种读出用途分开。外部分类器只测单状态有多少可读信息，动物一直保留自己的原生动作。"
                  "给定这组读出和样本，区间含机会水平意味着结果不足以确定可读性。", ""))
    links = [f"[{arm} 逐出生统计](meta-evidence-state-readout-{arm}-{step}.json)" for arm in reports]
    lines.append(" · ".join((*links, "[读出与复现代码](../../../experiments/pulse_game/interleave_meta_readout.py)")))
    output = docs / "meta-evidence-state-readout.md"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=storage.RAW / "meta-evidence")
    parser.add_argument("--arms", choices=task.ARMS, nargs="+", default=task.ARMS)
    parser.add_argument("--step", default=6000, type=int)
    parser.add_argument("--births", nargs="+", type=int, default=list(range(731, 737)))
    parser.add_argument("--cal", default=192, type=int)
    parser.add_argument("--test", default=256, type=int)
    args = parser.parse_args(argv)
    available = check_checkpoints(args.run, args.arms, args.step)
    if len(available) != len(args.arms):
        raise FileNotFoundError(f"Missing fixed checkpoint for {set(args.arms) - available.keys()}")
    for arm in args.arms:
        run_arm(args.run.resolve(), arm, args.step, args.births, args.cal, args.test)
    write_summary(args.step)


if __name__ == "__main__":
    main()
