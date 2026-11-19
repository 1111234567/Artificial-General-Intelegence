"""Merge the frozen capacity panels and report paired learning and retention."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..ring_alt_habitats import summarize
from ..spike_analysis import estimates


def merge(root):
    def load(name):
        return json.loads((root / f"ring-alt-habitats-{name}.json").read_text(encoding="utf-8"))

    plan, low, high = (load(n) for n in ("frozen", "result-low", "result-high"))
    for key in ("world_seeds", "sensory_separation", "hidden_map_codes"):
        if low[key] != high[key]:
            raise ValueError(f"World generators disagree on {key}")
    common = ("start", "worlds", "births", "intro", "rehearsal", "probe", "horizon", "panel", "sources")
    for key in common:
        if low["protocol"][key] != high["protocol"][key] or low["protocol"][key] != plan[key]:
            raise ValueError(f"Capacity shard differs on {key}")
    rows = low["rows"] + high["rows"]
    actual = {(r["habitats"], r["capacity"], r["rule"], r["birth"], r["schedule"]) for r in rows}
    expected = {(*setup, b, label) for setup in plan["conditions"] for b in plan["births"]
                for label in ("blocked", "spaced")}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("A frozen condition is missing or duplicated")
    return plan, rows, low["sensory_alias_control"], low["hidden_map_codes"], low["sensory_separation"]


def select(rows, k, capacity, when, rule="age"):
    return [r for r in rows if (r["habitats"], r["capacity"], r["rule"], r["schedule"]) == (k, capacity, rule, when)]


def quiet(rows, k, capacity, when, rule="age"):
    return np.mean([r["quiet_after_rehearsal"] for r in select(rows, k, capacity, when, rule)], axis=0)


def common_01(rows, k, when):
    values = []
    for row in select(rows, k, 32, when):
        view = np.asarray(row["quiet_after_rehearsal"])
        ids = np.asarray(row["orders"])
        actual = np.empty_like(view)
        np.put_along_axis(actual, ids, view, axis=1)
        values.append(actual[:, :2].mean(1))
    return np.mean(values, axis=0)


def compact(rows, alias, private_codes, separation):
    summary = summarize(rows)
    paired = {schedule: {} for schedule in ("blocked", "spaced")}
    for schedule, entry in paired.items():
        for label, capacity, control in (("16_minus_8", 16, 8), ("32_minus_16", 32, 16), ("32_minus_8", 32, 8)):
            diff = quiet(rows, 8, capacity, schedule) - quiet(rows, 8, control, schedule)
            entry[label] = {"all": estimates(diff.mean(1)), "old": estimates(diff[:, :-1].mean(1)),
                            "last": estimates(diff[:, -1])}
        for number in (2, 4, 8):
            value = common_01(rows, number, schedule)
            entry[f"common_0_1_despite_K{number}"] = estimates(value)
            if number != 2:
                entry[f"common_0_1_K{number}_minus_K2"] = estimates(value - common_01(rows, 2, schedule))
        diff = quiet(rows, 8, 32, schedule) - quiet(rows, 8, 32, schedule, rule="ordinary")
        entry["age_minus_ordinary"] = {"all": estimates(diff.mean(1)), "old": estimates(diff[:, :-1].mean(1)),
                                       "last": estimates(diff[:, -1])}
    unique_at_eight = np.array([len(set(row[:8])) == 8 for row in private_codes])
    replay = {}
    for capacity in (8, 16, 32):
        a = quiet(rows, 8, capacity, "blocked").mean(1)[unique_at_eight]
        b = quiet(rows, 8, capacity, "spaced").mean(1)[unique_at_eight]
        replay[str(capacity)] = {"blocked": estimates(a), "spaced": estimates(b), "spaced_minus_blocked": estimates(b - a)}
    storage = {}
    for row in rows:
        key = str(row["capacity"])
        shape = row["resource_after_rehearsal"]["scope_sizes"]
        total = sum(shape.values())
        storage[key] = {**shape, "total_scoped_float32_states": total, "scoped_bytes_per_life": 4 * total}
    scope = {"minimum_within_habitat_distances": [min(x["within"]) for x in separation],
             "minimum_between_habitat_distances": [min(x["cross"]) for x in separation],
             "repeat_full_discrete_map_values": {k: sum(len(set(row[:k])) < k for row in private_codes) for k in (2, 4, 8)},
             "exploratory_no_repeat_map_value_worlds": int(unique_at_eight.sum()), "exploratory_no_repeat_map_K8": replay,
             "four_adaptive_storage_components": storage}
    return {"absolute": summary, "paired": paired, "identical_sensory_control": alias, "scope_audit": scope}


def chart(report, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {8: "#aa604a", 16: "#8165a9", 32: "#207266"}
    table = report["absolute"]
    ax = axes[0, 0]
    for n, color in colors.items():
        for s, marker, ls in (("blocked", "o", "--"), ("spaced", "s", "-")):
            x = [k for k in (2, 4, 8) if f"{k}/{n}/age" in table]
            samples = [table[f"{k}/{n}/age"][s]["all"]["mean"] for k in x]
            ax.plot(x, samples, marker=marker, ls=ls, ms=4, color=color, label=f"{n} slots, {s}")
    ax.set_title("A   Quiet retention across habitat load", loc="left", fontweight="bold")
    ax.set_xlabel("Separate habitats")
    ax.set_xticks([2, 4, 8])
    ax.legend(loc="best", frameon=True, edgecolor="none", fontsize=8)
    ax.set_ylabel("Correct first bite")

    ax = axes[0, 1]
    for offset, (conditions, marker, label) in enumerate((("old", "o", "Older habitats"), ("most_recent", "s", "Last habitat"))):
        for i, (n, s) in enumerate((n, s) for n in colors for s in ("blocked", "spaced")):
            v = table[f"8/{n}/age"][s][conditions]
            mid, lo, hi = v["mean"], *v["ci95"]
            ax.errorbar(i + .12 * (2 * offset - 1), mid, yerr=[[mid-lo], [hi-mid]], marker=marker,
                        ms=4, capsize=2, color=colors[n], label=label if i == 0 else None)
    ax.set_xticks(range(6), [f"{c}\n{s}" for c in colors for s in ("blocked", "spaced")], fontsize=8)
    ax.set_title("B   Eight habitats, by retention rank", loc="left", fontweight="bold")
    ax.set_xlabel("Slots per sensory bank and review order")
    ax.legend(loc="best", frameon=True, edgecolor="none", fontsize=8)

    ax = axes[1, 0]
    for i, (n, s) in enumerate((n, s) for n in colors for s in ("blocked", "spaced")):
        c = table[f"8/{n}/age"][s]
        for j, (role, mark, label) in enumerate((("cue_retired", "o", "Cue bank"), ("fruit_retired", "s", "Fruit bank"))):
            v = c[role]
            mid, lo, hi = v["mean"], *v["ci95"]
            ax.errorbar(i + .12 * (2 * j - 1), mid + 1, yerr=[[mid-lo], [hi-mid]], marker=mark,
                        ms=4, capsize=2, color=colors[n], label=label if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(6), [f"{c}\n{s}" for c in colors for s in ("blocked", "spaced")], fontsize=8)
    ax.set_ylabel("1 + mean branch retirements (log scale)")
    ax.set_title("C   Turnover at the same eight-habitat load", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, edgecolor="none", fontsize=8)

    ax = axes[1, 1]
    for i, (s, marker) in enumerate((("blocked", "o"), ("spaced", "s"))):
        for k in (2, 4, 8):
            d = report["paired"][s][f"common_0_1_despite_K{k}"]
            mid, lo, hi = d["mean"], *d["ci95"]
            ax.errorbar(k + .12 * (2 * i - 1), mid, yerr=[[mid-lo], [hi-mid]], marker=marker,
                        ms=5, capsize=3, color=("#8165a9", "#207266")[i], label=s if k == 2 else None)
    ax.set_xticks((2, 4, 8))
    ax.set_xlabel("Total learned habitats")
    ax.set_ylabel("Quiet accuracy for the same two habitats")
    ax.set_title("D   Load effect at fixed 32-slot capacity", loc="left", fontweight="bold")
    ax.legend(loc="best", frameon=True, edgecolor="none", fontsize=8)

    for panel in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        panel.set_ylim(.16, 1.)
        panel.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
        panel.axhline(.25, linewidth=.8, color="#999999", ls=":")
    for panel in axes.ravel():
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", color="#ddd", lw=.65)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    plan, rows, alias, hidden, separated = merge(args.data)
    report = {"protocol": plan, **compact(rows, alias, hidden, separated)}
    (args.data / "ring-alt-habitats-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    chart(report, args.data / "ring-alt-habitats-v1.png")


if __name__ == "__main__":
    main()
