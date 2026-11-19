"""Aggregate paired training births, retaining individual contrasts and denominators."""

from argparse import ArgumentParser
import itertools
import json
from pathlib import Path
from hashlib import sha256

import numpy as np

from .edit_receipt_graph import GATE_GAIN
from .self_param_writer import BODY

def interval(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": None, "n": 0}
    mean = float(x.mean())
    out = {"mean": mean, "n": len(x), "by_birth": x.tolist()}
    # Two-sided t(.975) for paired independent birth means; never treat actions as births.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    if len(x) in critical:
        margin = critical[len(x)] * x.std(ddof=1) / np.sqrt(len(x))
        flips = np.array(list(itertools.product((-1, 1), repeat=len(x))))
        randomization = np.abs(flips @ x / len(x))
        out.update(ci95=[mean - margin, mean + margin],
                   sign_randomization_p=float(np.mean(randomization >= abs(mean) - 1e-12)))
    return out


def auc(score, positive):
    score, positive = np.asarray(score), np.asarray(positive)
    n, p = (~positive).sum(), positive.sum()
    if not n or not p:
        return np.nan
    sorted_neg = np.sort(score[~positive])
    smaller = np.searchsorted(sorted_neg, score[positive], side="left")
    le = np.searchsorted(sorted_neg, score[positive], side="right")
    return float((smaller + le).mean() / (2 * n))


def rank_gain(score, target):
    """Post-hoc single-edit gain over random acceptance at the same 50% quota."""
    if not len(score):
        return np.nan
    cutoff = np.median(score)
    high = (score > cutoff).astype(float)
    tie = score == cutoff
    if tie.any():
        high[tie] = (.5 * len(score) - high.sum()) / tie.sum()
    return np.mean((high - .5) * target)


def collect(path, width):
    with np.load(path, allow_pickle=False) as z:
        result = {}
        s = lambda key: z["deployment/stable_noise/" + key + "/correct"]
        r = lambda key: z["deployment/reversal/" + key + "/correct"]
        phase = z["transplant/fresh_writer/full/correct"].shape[1]
        modes = ["full", "no_write"]
        if "deployment/stable_noise/yoked/correct" in z:
            modes.append("yoked")
        for mode in modes:
            stable, reversal = s(mode), r(mode)
            half = reversal.shape[1] // 2
            result.update({
                f"stable_late_{mode}": stable[:, -8:].mean(),
                f"stable_first_{mode}": stable[:, 0].mean(),
                f"stable_early_{mode}": stable[:, 1:4].mean(),
                f"reversal_first4_{mode}": reversal[:, half:half+4].mean(),
                f"reversal_late_{mode}": reversal[:, -8:].mean(),
            })
        for name, start in (("ABA", 2 * phase), ("CBA", 2 * phase), ("B_only", phase)):
            trace = z[f"return/{name}/correct"]
            result[f"return_{name}"] = trace[:, start].mean()
            result[f"return_{name}_early"] = trace[:, start+1:start+4].mean()
            result[f"silent_{name}"] = z[f"return_silent/{name}/correct"][:, 1:4].mean()
            result[f"prefeedback_{name}"] = z[f"return_pre_feedback/{name}/correct"].mean()
        for name in ("A_writer", "C_writer", "B_writer", "fresh_writer"):
            trace = z[f"transplant/{name}/full/correct"]
            result[f"graft_{name}_first"] = trace[:, 0].mean()
            result[f"graft_{name}_early"] = trace[:, 1:4].mean()
            result[f"graft_{name}_valid"] = z[f"transplant/{name}/valid"].mean()
        for name in ("full", "no_write"):
            trace = z[f"checkpoint_writer/stable_noise/{name}/correct"]
            result[f"cold_writer_late_{name}"] = trace[:, -8:].mean()
        result["stable_write_difference"] = result["stable_late_full"] - result["stable_late_no_write"]
        result["return_history"] = result["return_ABA"] - result["return_CBA"]
        result["return_over_B"] = result["return_ABA"] - result["return_B_only"]
        result["graft_history"] = result["graft_A_writer_first"] - result["graft_C_writer_first"]
        result["stable_life_gain"] = result["stable_late_full"] - result["stable_first_full"]
        op = z["deployment/stable_noise/full/op"]
        applied = z["deployment/stable_noise/full/applied"]
        nonempty = op != 0
        pred_all = z["deployment/stable_noise/full/predicted_benefit"][nonempty]
        result["proposal_rate"] = nonempty.mean()
        result["applied_rate"] = applied.mean()
        result["mean_absolute_prediction"] = np.abs(pred_all).mean() if len(pred_all) else np.nan
        result["accepted_given_proposed"] = (z["deployment/stable_noise/full/accepted"][nonempty].mean()
                                             if len(pred_all) else np.nan)
        for code, name in ((1, "adjust"), (2, "birth"), (3, "retire")):
            result[f"applied_{name}"] = ((op == code) & applied).sum() / op.size
        bias = z["deployment/stable_noise/full/address"] >= width ** 2
        result["applied_bias"] = ((op == 1) & applied & bias).mean()
        result["applied_existing_weight"] = ((op == 1) & applied & ~bias).mean()
        for split in ("fork_holdout/stable_noise", "fork_holdout/reversal", "fork_validation/mixed"):
            mask = z[split + "/applied"]
            pred = z[split + "/predicted_benefit"][mask]
            target = z[split + "/delta_positive_taste"][mask]
            changed = z[split + "/action_changed"][mask].any(-1)
            nonzero = target != 0
            gate = 1 / (1 + np.exp(-GATE_GAIN * pred))
            short = split.replace("fork_holdout/", "fork_").replace("fork_validation/mixed", "fork_matched")
            for name, value in (
                ("opportunities", mask.size), ("applied", len(pred)), ("nonzero", nonzero.sum()),
                ("positive", (target > 0).sum()), ("negative", (target < 0).sum()),
                ("action_changed", changed.sum()), ("mean_outcome", target.mean() if len(pred) else np.nan),
                ("mse_model", np.square(pred - target).mean() if len(pred) else np.nan),
                ("mse_zero", np.square(target).mean() if len(pred) else np.nan),
                ("direction_auc", auc(pred[nonzero], target[nonzero] > 0)),
                ("gate_near_half", (np.abs(gate - .5) < .05).mean() if len(pred) else np.nan),
                ("gate_on_good", gate[target > 0].mean() if np.any(target > 0) else np.nan),
                ("gate_on_bad", gate[target < 0].mean() if np.any(target < 0) else np.nan),
                ("ranked_half_gain", rank_gain(pred, target)),
                ("hindsight_half_gain", rank_gain(target, target)),
            ):
                result[f"{short}_{name}"] = value
    return {k: float(v) for k, v in result.items()}


def channels(paths):
    output = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            applied = z["applied"].ravel()
            # [fork,channel,world,delay] -> [channel,fork*world,delay]
            def unpack(key):
                v = z[key]
                return v.transpose(1, 0, 2, 3).reshape(4, -1, v.shape[-1])
            choice, correct = unpack("choice"), unpack("correct").astype(float)
            samples = {}
            for scope, mask in (("applied", applied), ("unchanged", ~applied)):
                examples = {"count": int(mask.sum())}
                for name, i, j in (("total", 3, 0), ("parameters_same_h", 2, 0),
                                   ("activity_same_p", 1, 0)):
                    horizon = choice.shape[-1]
                    changed = (choice[i, mask] != choice[j, mask])
                    first = np.where(changed.any(1), 1 + changed.argmax(1), horizon + 1)
                    net = correct[i, mask] - correct[j, mask]
                    examples[name] = {str(k): {
                        "changed": int((first <= k).sum()),
                        "net_correct": int(net[:, :k].sum()),
                        "positive_lives": int((net[:, :k].sum(1) > 0).sum()),
                        "negative_lives": int((net[:, :k].sum(1) < 0).sum())}
                        for k in (1, 3, 8, 16, 24) if k <= horizon}
                samples[scope] = examples
            output[p.stem] = samples
    return output


def figure(stages, arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = {"baseline": "新增感觉关闭", "sensing": "自身感觉", "causal": "自身感觉＋配对后果",
              "shuffled": "自身感觉＋打乱后果"}
    colors = ["#687888", "#207e84", "#a64a29", "#8d65a5"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    steps = sorted(stages)
    for ax, metric, title in zip(axes, ("stable_late_full", "reversal_first4_full", "return_history"),
                                  ("稳定世界的末八次选择", "反转后的前四次选择", "回访首答的历史优势"), strict=True):
        if metric == "return_history":
            ax.axhline(0, color="#b5b9bc", linewidth=1)
        for i, arm in enumerate(arms):
            values = np.array([stages[t]["arms"][arm][metric]["mean"] for t in steps])
            ax.plot(range(len(steps)), 100 * values, color=colors[i], marker="o", markersize=4,
                    linewidth=1.8, label=labels[arm])
            if arm == "causal":
                bounds = np.array([stages[t]["arms"][arm][metric]["ci95"] for t in steps]) * 100
                ax.fill_between(range(len(steps)), bounds[:, 0], bounds[:, 1], color=colors[i], alpha=.12)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(steps)), steps)
        ax.set_xlabel("Adam 更新次数（各测点等距）")
        ax.set_ylabel("ABA 减 CBA（百分点）" if metric == "return_history" else "实际正确率（%）")
        if metric != "return_history":
            ax.set_ylim(45, 55)
            ax.axhline(50, linewidth=.8, color="#b5b9bc", zorder=0)
        ax.grid(axis="y", alpha=.25)
        ax.spines[["right", "top"]].set_visible(False)
    handles, text = axes[0].get_legend_handles_labels()
    fig.legend(handles, text, loc="outside lower center", ncols=4, frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def verify_artifacts(run_dir, eval_dir, plan):
    from .edit_receipt_train import CHECKPOINT, METRICS
    records = {}
    roots, arms = plan["births"], plan["arms"]
    for root in roots:
        birth = None
        for arm in arms:
            for step in plan["checkpoints"]:
                rel = CHECKPOINT.format(birth=root, arm=arm, step=step)
                path = run_dir / rel
                with np.load(path, allow_pickle=False) as g:
                    w, b, m = (g[key] for key in ("weights", "bias", "present"))
                    if not np.isfinite(w).all() or not np.isfinite(b).all() or np.any(w[~m] != 0):
                        raise AssertionError(f"Invalid birth-graph numeric state: {rel}")
                    if step == 0:
                        if birth is None:
                            birth = (w, b, m)
                        elif not all(np.array_equal(a, v) for a, v in zip(birth, (w, b, m), strict=True)):
                            raise AssertionError(f"Paired initial graph mismatch: {rel}")
                with np.load(eval_dir / rel, allow_pickle=False) as z:
                    for env in ("stable_noise", "reversal"):
                        live = z[f"deployment/{env}/full/side"][:, 0]
                        frozen = z[f"deployment/{env}/no_write/side"][:, 0]
                        if not np.array_equal(live, frozen):
                            raise AssertionError(f"Pre-feedback outcome differs in write control: {rel}")
                records[rel] = {"graph": sha256(path.read_bytes()).hexdigest(),
                                "evaluation": sha256((eval_dir / rel).read_bytes()).hexdigest()}
            logs = run_dir / f"birth_{root}/{arm}/training.npz"
            with np.load(logs, allow_pickle=False) as t:
                if (list(t["metrics"]) != list(METRICS)
                        or t["values"].shape != (plan["config"]["updates"], len(METRICS))
                        or not np.isfinite(t["values"]).all()):
                    raise AssertionError(f"Incomplete or invalid training trace: {logs}")
            records[f"birth_{root}/{arm}/training.npz"] = {"training": sha256(logs.read_bytes()).hexdigest()}
    protocol = json.loads((eval_dir / "protocol.json").read_text())
    if protocol["source_mismatch_with_training_manifest"]:
        raise AssertionError("Source used for evaluation differs from training")
    return {"fits": len(roots) * len(arms), "stages": len(plan["checkpoints"]),
            "training_sources": plan["sources_sha256"], "evaluation_sources": protocol["sources_sha256"],
            "verified": ["paired bit-identical birth parameters", "finite complete traces",
                         "zero coefficients at absent edges", "matched full/no-write first action"],
            "sha256": records}


def run(run_dir, eval_dir, output):
    plan = json.loads((run_dir / "run.json").read_text())
    roots, arms = plan["births"], plan["arms"]
    stages = {}
    for stage in plan["checkpoints"]:
        per_arm = {}
        for arm in arms:
            batch = []
            for seed in roots:
                path = eval_dir / f"birth_{seed}" / arm / f"checkpoint_{stage}.npz"
                batch.append(collect(path, BODY + plan["config"]["writers"]))
            per_arm[arm] = {k: [r[k] for r in batch] for k in batch[0]}
        means = {a: {k: interval(v) for k, v in m.items()} for a, m in per_arm.items()}
        contrasts = {}
        for a, b in (("sensing", "baseline"), ("causal", "sensing"), ("causal", "shuffled")):
            contrasts[a + "-" + b] = {
                k: interval(np.array(v) - per_arm[b][k]) for k, v in per_arm[a].items() if k in per_arm[b]
            }
        stages[stage] = {"arms": means, "paired_contrasts": contrasts}
    changes = {}
    final = max(stages)
    for earlier in sorted(set(stages) - {final}):
        changes[f"{final}-{earlier}"] = {}
        for arm in arms:
            early = stages[earlier]["arms"][arm]
            late = stages[final]["arms"][arm]
            keys = ("stable_late_full", "reversal_first4_full", "return_history",
                    "stable_write_difference", "fork_matched_mse_model", "applied_rate")
            changes[f"{final}-{earlier}"][arm] = {
                k: interval(np.array(late[k]["by_birth"]) - early[k]["by_birth"]) for k in keys
                if early[k]["n"] == late[k]["n"] == len(roots)
            }
    result = {"births": roots, "stages": stages, "longitudinal_paired": changes,
              "channels": channels(sorted((run_dir / "channels").glob("*.npz")))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    figure(stages, arms, output.with_name("learning.png"))
    verified = verify_artifacts(run_dir, eval_dir, plan)
    output.with_name("verification.json").write_text(json.dumps(verified, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.run, args.eval, args.out)
