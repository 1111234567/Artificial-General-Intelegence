"""Summarize the completed, independently audited learned-binding experiment.

This program never trains a model, opens a world, or runs a policy.  Ordinary
execution requires all 39 probes, 384 B blocks, nine native replays and six
independently recomputed PG updates to have passed the existing audit.  Run
``--self-test`` for an entirely temporary synthetic-artifact test instead.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/learned-binding/v1"
DEFAULT_OUTPUT = ROOT / "docs/evidence/learned-binding-2026-11-07"
SEEDS = (0, 1, 2)
KINDS = ("manual_hard", "manual_soft", "learned")
BASE_PROBES = ("own_immediate", "own_after_B_frozen", "teacher_probe")
EXTRA_PROBES = ("own_after_B_outcome", "teacher_clear_memory", "teacher_clear_encoder", "teacher_no_read")
SOURCE_GROUPS = ("all", "A", "B", "A_conflict", "B_conflict", "A_shared", "B_shared")
GROUPS = SOURCE_GROUPS + ("conflict", "shared")
METRICS = ("strict", "ordinary", "translation_correct")
COPIES = ("protocol.json", "training-freeze.json", "data-freeze.json", "audit.json")
FIRST_DEFINITION = ("First eligible request within this prospective probe for each actor/family/raw; "
                    "pooled across its two jobs. It may follow other A feedback and is not a no-A-feedback test.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


class Sources:
    def __init__(self, repository):
        self.repository = Path(repository).resolve()
        self.files = {}
        self.payloads = {}

    def name(self, path):
        path = Path(path).resolve()
        try:
            return path.relative_to(self.repository).as_posix()
        except ValueError:
            return str(path)

    def record(self, path, expected=None):
        path = Path(path)
        require(path.is_file(), f"missing source: {path}")
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f"source SHA changed: {path}")
        name = self.name(path)
        if name in self.files:
            require(self.files[name] == digest, f"source changed during summary: {path}")
        self.files[name] = digest
        return dict(path=name, sha256=digest)

    def read(self, path):
        source = self.record(path)
        data = Path(path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == source["sha256"], f"source changed while reading: {path}")
        return json.loads(data.decode("utf-8")), source

    def bundle(self, prefix, report):
        # Payloads were fully rehashed/replayed by audit.py.  Their audited
        # hashes are retained, while this summary reads only their JSON views.
        for suffix, field in ((".npz", "arrays_sha256"), (".pt", "states_sha256")):
            path = Path(prefix).with_suffix(suffix)
            require(path.is_file(), f"missing audited payload: {path}")
            digest = report[field]
            require(isinstance(digest, str) and len(digest) == 64, f"invalid payload SHA: {path}")
            self.payloads[self.name(path)] = digest


def check_audit(audit):
    require(audit.get("format") == "learned-binding-independent-audit-v1", "unrecognized audit format")
    for name, expected in (("probe_count", 39), ("B_block_count", 384),
                           ("native_replay_count", 9), ("independently_recomputed_pg_updates", 6)):
        require(type(audit.get(name)) is int and audit[name] == expected,
                f"audit {name} must equal {expected}")
    for key in ("all_world_actions_and_public_rewards_replayed",
                "all_controller_live_states_preserved_across_optimizer_updates",
                "all_source_executors_unchanged"):
        require(audit.get(key) is True, "audit flag failed or absent: " + key)
    require(not audit.get("sources", {}).get("temporary_fixture"), "temporary audit cannot authorize formal results")

    def visit(value, label="audit"):
        if isinstance(value, bool):
            require(value is True, "audit contains a false check: " + label)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, label + "." + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, label + f"[{index}]")
    visit(audit)
    require(len(audit.get("probes", [])) == 39, "audit probe detail count")
    require(len(audit.get("native_replays", [])) == 9, "audit replay detail count")
    for key in ("only_oracle_call_is_explicit_teacher_support", "own_driver_has_no_oracle_call"):
        require(audit.get("oracle_paths", {}).get(key) is True, "audit oracle-path check: " + key)
    for probe in audit["probes"]:
        require(probe.get("world_replayed") is True, "audit probe world replay missing")
    for replay in audit["native_replays"]:
        for key in ("arrays_bitwise", "full_final_state_bitwise", "oracle_disabled"):
            require(replay.get(key) is True, "audit native replay check: " + key)
    phases = audit.get("B_phases", [])
    require(sum(p["blocks"] for p in phases) == 384, "audit B phase detail count")
    require(sum(len(p["pg_checks"]) for p in phases) == 6, "audit PG detail count")
    for phase in phases:
        require(phase.get("clocks_pending_and_live_state_chain_verified") is True, "audit B state-chain check")
        for check in phase["pg_checks"]:
            for key in ("model_and_adam_bitwise_equal", "public_rewards_recomputed", "joint_actor_objective"):
                require(check.get(key) is True, "audit PG recomputation check: " + key)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def equal_seed_mean(values):
    # Missing denominators must not silently change the equal-three-seed rule.
    return sum(values) / len(SEEDS) if len(values) == len(SEEDS) and all(v is not None for v in values) else None


def group_counts(report, scope, group):
    if group in ("conflict", "shared"):
        rows = [report[scope][family + "_" + group] for family in ("A", "B")]
        counts = {key: sum(row[key] for row in rows) for key in ("n",) + METRICS}
    else:
        counts = {key: report[scope][group][key] for key in ("n",) + METRICS}
    require(all(type(value) is int and value >= 0 for value in counts.values()), "invalid request counts")
    require(counts["strict"] <= counts["ordinary"] <= counts["n"], "strict/ordinary request bounds")
    require(counts["translation_correct"] <= counts["n"], "translation count bounds")
    return counts


def summarize_group(reports, scope, group):
    rows = {}
    for seed, report in zip(SEEDS, reports):
        counts = group_counts(report, scope, group)
        rows[str(seed)] = dict(**counts, **{metric + "_rate": ratio(counts[metric], counts["n"])
                                           for metric in METRICS})
    pooled = {key: sum(row[key] for row in rows.values()) for key in ("n",) + METRICS}
    pooled.update({metric + "_rate": ratio(pooled[metric], pooled["n"]) for metric in METRICS})
    return dict(per_seed=rows, equal_seed_mean={metric + "_rate": equal_seed_mean(
        [rows[str(seed)][metric + "_rate"] for seed in SEEDS]) for metric in METRICS}, pooled=pooled)


def summarize_probe(kind, name, reports, refs):
    scopes = {scope: {group: summarize_group(reports, scope, group) for group in GROUPS}
              for scope in ("groups", "first_groups")}
    idle = {str(seed): dict(actions=r["idle_actions"], opportunities=r["idle_opportunities"],
        per1000=None if not r["idle_opportunities"] else 1000 * r["idle_actions"] / r["idle_opportunities"])
        for seed, r in zip(SEEDS, reports)}
    pooled_idle = dict(actions=sum(r["actions"] for r in idle.values()),
                       opportunities=sum(r["opportunities"] for r in idle.values()))
    pooled_idle["per1000"] = (1000 * pooled_idle["actions"] / pooled_idle["opportunities"]
                                if pooled_idle["opportunities"] else None)
    gates = {}
    for category in ("all", "success_finish", "command", "other"):
        per_seed = {str(seed): r["gate_statistics"][category] for seed, r in zip(SEEDS, reports)}
        total, count = sum(r["sum"] for r in per_seed.values()), sum(r["count"] for r in per_seed.values())
        gates[category] = dict(per_seed=per_seed,
            equal_seed_mean=equal_seed_mean([r["mean"] for r in per_seed.values()]),
            pooled=dict(sum=total, count=count, mean=ratio(total, count)))
    return dict(kind=kind, name=name, source_reports=refs, **scopes,
        idle=dict(per_seed=idle, equal_seed_mean_per1000=equal_seed_mean([r["per1000"] for r in idle.values()]),
                  pooled=pooled_idle), gate_statistics=gates,
        memory_norm=dict(per_seed={str(seed): r["memory_norm"] for seed, r in zip(SEEDS, reports)},
                         equal_seed_mean=equal_seed_mean([r["memory_norm"] for r in reports])),
        parameters_per_seed={str(seed): r["parameters"] for seed, r in zip(SEEDS, reports)},
        boundaries_per_seed={str(seed): dict(start_tick=r["start_tick"], end_tick=r["end_tick"],
            right_censored=len(r["right_censored"]), inherited_matured=len(r["left_boundary_matured"]),
            first_right_censored=len(r["first_right_censored"]), start_work=r["start_work"],
            final_work=r["final_work"], command_translations=r["command_translations"],
            correct_translations=r["correct_translations"])
            for seed, r in zip(SEEDS, reports)})


def learning_curve(root, kind, phase, settings, sources):
    per_seed = {}
    denominator = settings["pg_ticks"] * settings["actors"]
    for seed in SEEDS:
        folder = root / f"seed_{seed:03d}/behavior" / kind / phase
        completion, completion_source = sources.read(folder / "completion.json")
        require(completion["blocks"] == 32 and completion["learn"] is (phase == "outcome_B"), "B completion flags")
        records, updates = [], []
        for index in range(32):
            prefix = folder / f"block_{index:03d}"
            report, source = sources.read(prefix.with_suffix(".json"))
            require(report["index"] == index and report["learn"] is (phase == "outcome_B"), "B block identity")
            sources.bundle(prefix, report)
            update = report["update"]
            updates.append(update)
            require(math.isclose(update["reward_sum"], update["positive_finishes"] - .1 * update["rejected"],
                                 rel_tol=1e-5, abs_tol=1e-3), "B reward count inconsistency")
            records.append(dict(block=index + 1, source=source, update=update,
                reward_per_actor_tick=update["reward_sum"] / denominator,
                finishes_per_actor_tick=update["positive_finishes"] / denominator,
                rejected_per_actor_tick=update["rejected"] / denominator))
        require(completion["records"] == updates, "B completion/block records changed")
        windows = []
        for offset in range(0, 32, 8):
            selected = records[offset:offset + 8]
            window = dict(first_block=offset + 1, last_block=offset + 8,
                positive_finishes=sum(r["update"]["positive_finishes"] for r in selected),
                rejected=sum(r["update"]["rejected"] for r in selected),
                reward_sum=sum(r["update"]["reward_sum"] for r in selected),
                actor_ticks=8 * denominator)
            for raw, field in (("reward_sum", "reward"), ("positive_finishes", "finishes"), ("rejected", "rejected")):
                window[field + "_per_actor_tick"] = window[raw] / window["actor_ticks"]
            windows.append(window)
        per_seed[str(seed)] = dict(blocks=records, windows=windows, completion_source=completion_source,
            phase_ticks=completion["phase_ticks"], closure_ticks_excluded=completion["counted_extra_closure_ticks"])
    means = []
    for index in range(4):
        selected = [per_seed[str(seed)]["windows"][index] for seed in SEEDS]
        means.append(dict(first_block=selected[0]["first_block"], last_block=selected[0]["last_block"],
            **{metric + "_per_actor_tick": equal_seed_mean([r[metric + "_per_actor_tick"] for r in selected])
               for metric in ("reward", "finishes", "rejected")},
            pooled={key: sum(r[key] for r in selected)
                    for key in ("positive_finishes", "rejected", "reward_sum", "actor_ticks")}))
    return dict(kind=kind, phase=phase, per_seed=per_seed, equal_seed_windows=means,
                normalization="First normalize each block by 512 ticks * 32 actors; average eight consecutive blocks. Both jobs contribute to actor reward.")


def write_csv(path, probes):
    fields = ["kind", "probe", "scope", "group"]
    for seed in SEEDS:
        fields += [f"seed{seed}_n"] + [f"seed{seed}_{m}_rate" for m in METRICS]
    fields += ["equal_seed_" + m + "_rate" for m in METRICS]
    fields += ["pooled_n"] + ["pooled_" + m for m in METRICS] + ["pooled_" + m + "_rate" for m in METRICS]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for probe in probes:
            for scope in ("groups", "first_groups"):
                for group in GROUPS:
                    item = probe[scope][group]
                    row = dict(kind=probe["kind"], probe=probe["name"], scope=scope, group=group)
                    for seed in SEEDS:
                        values = item["per_seed"][str(seed)]
                        row[f"seed{seed}_n"] = values["n"]
                        for metric in METRICS:
                            row[f"seed{seed}_{metric}_rate"] = values[metric + "_rate"]
                    row.update({"equal_seed_" + key: value for key, value in item["equal_seed_mean"].items()})
                    row.update({"pooled_" + key: value for key, value in item["pooled"].items()})
                    writer.writerow(row)


def plot_summary(path, probes, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(p["kind"], p["name"]): p for p in probes}
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6))
    colors = dict(manual_hard="#52606d", manual_soft="#e09f3e", learned="#1976a3", outcome="#b44968")
    labels = dict(manual_hard="Manual hard", manual_soft="Manual soft", learned="Native frozen")

    def percent(kind, name, group="all"):
        value = lookup[kind, name]["groups"][group]["equal_seed_mean"]["strict_rate"]
        return float("nan") if value is None else 100 * value

    for kind in KINDS:
        axes[0].plot([0, 1], [percent(kind, "own_immediate", "A_conflict"),
            percent(kind, "own_after_B_frozen", "A_conflict")], "o-", color=colors[kind], label=labels[kind])
    axes[0].plot([0, 1], [percent("learned", "own_immediate", "A_conflict"),
        percent("learned", "own_after_B_outcome", "A_conflict")], "o--", color=colors["outcome"], label="Native PG")
    axes[0].set(xticks=[0, 1], xticklabels=["After own AB support", "After B experience"],
                ylim=(0, 100), ylabel="A conflict strict completion (%)", title="Old A: prospective behavior probes")
    axes[0].legend(fontsize=8, loc="best")
    positions = np.arange(3)
    for offset, name, label, color in ((-.18, "own_immediate", "Own-action support", "#768da3"),
                                       (.18, "teacher_probe", "Teacher support", "#4c9f70")):
        axes[1].bar(positions + offset, [percent(kind, name) for kind in KINDS], .34, color=color, label=label)
        for index, kind in enumerate(KINDS):
            seed_values = lookup[kind, name]["groups"]["all"]["per_seed"]
            rates = [seed_values[str(seed)]["strict_rate"] for seed in SEEDS]
            axes[1].scatter(positions[index] + offset + np.linspace(-.06, .06, 3),
                           [float("nan") if r is None else 100 * r for r in rates], s=15, color="black", zorder=3)
    axes[1].set(xticks=positions, xticklabels=["Manual hard", "Manual soft", "Native"], ylim=(0, 100),
                ylabel="All requests: strict completion (%)", title="Own experience vs actual teacher support")
    axes[1].legend(fontsize=8)
    native_curves = {c["phase"]: c for c in curves if c["kind"] == "learned"}
    plotted = []
    for phase, label, color in (("frozen_B", "Native frozen", colors["learned"]),
                                ("outcome_B", "Native PG", colors["outcome"])):
        curve = native_curves[phase]
        mean = curve["equal_seed_windows"]
        x = [r["last_block"] for r in mean]
        y = [r["reward_per_actor_tick"] for r in mean]
        plotted.extend(y)
        for seed in SEEDS:
            seed_y = [r["reward_per_actor_tick"] for r in curve["per_seed"][str(seed)]["windows"]]
            plotted.extend(seed_y)
            axes[2].plot(x, seed_y, color=color, alpha=.23, linewidth=1)
        axes[2].plot(x, y, "o-", color=color, linewidth=2, label=label)
    lower, upper = min([0.] + plotted), max([0.] + plotted)
    span = max(upper - lower, .001)
    axes[2].set(ylim=(lower - .08 * span if lower < 0 else 0., upper + .12 * span),
                xlabel="B block (512 ticks each; 8-block windows)", ylabel="Signed public reward / actor / tick",
                title="Native B experience: observed reward")
    axes[2].axhline(0, color="#555555", linewidth=.8)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Learned binding — three seeds weighted equally", fontsize=14)
    fig.text(.02, .025, "Each probe scores a new 2,048-tick request set; points do not follow the same requests. "
             "Manual and native training histories differ.\nNative frozen/PG branches share the same interaction budget. "
             "Thin reward curves / black dots show individual seeds.", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize(root=DEFAULT_INPUT, output=DEFAULT_OUTPUT, *, repository=ROOT):
    root, output, repository = Path(root), Path(output), Path(repository)
    sources = Sources(repository)
    audit, audit_ref = sources.read(root / "audit.json")
    check_audit(audit)
    protocol, protocol_ref = sources.read(root / "protocol.json")
    freeze, freeze_ref = sources.read(root / "training-freeze.json")
    _, data_ref = sources.read(root / "data-freeze.json")
    require(freeze.get("all_complete_before_scoring") is True, "training freeze absent")
    settings = protocol["settings"]
    require(settings["seeds"] == list(SEEDS), "expected exactly three fixed seeds")
    require(settings["pg_blocks"] == 32 and settings["pg_ticks"] == 512 and settings["actors"] == 32,
            "unexpected B interaction budget")
    for name, digest in protocol["sources"].items():
        sources.record(repository / name, digest)
    origins = {name: sources.record(repository / row["path"], row["sha256"])
               for name, row in protocol["origins"].items()}
    require(len(freeze["checkpoints"]) == 3, "native checkpoint freeze count")
    for name, digest in freeze["checkpoints"].items():
        sources.record(repository / name, digest)
    audit_script = sources.record(Path(__file__).with_name("audit.py"))
    generator = sources.record(Path(__file__))
    audit_identities = {(p["seed"], p["kind"], p["name"]): p for p in audit["probes"]}
    require(len(audit_identities) == 39, "duplicate audited probe identities")
    probes = []
    for kind in KINDS:
        for name in BASE_PROBES + (EXTRA_PROBES if kind == "learned" else ()):
            reports, refs = [], []
            for seed in SEEDS:
                folder = root / f"seed_{seed:03d}/behavior" / kind
                manifest, _ = sources.read(folder / "completion.json")
                require(manifest["seed"] == seed and manifest["kind"] == kind, "behavior identity")
                prefix = folder / name
                report, ref = sources.read(prefix.with_suffix(".json"))
                require(ref["sha256"] == manifest["files"][name + ".json"], "probe manifest SHA")
                require((report["seed"], report["kind"], report["name"]) == (seed, kind, name), "probe identity")
                require((seed, kind, name) in audit_identities, "probe absent from audit")
                require(report["groups"] == audit_identities[seed, kind, name]["groups"], "audited probe groups changed")
                sources.bundle(prefix, report)
                reports.append(report)
                refs.append(ref)
            probes.append(summarize_probe(kind, name, reports, refs))
    training = {}
    for seed in SEEDS:
        folder = root / f"seed_{seed:03d}/pretrain"
        metadata, ref = sources.read(folder / "training.json")
        require(metadata["seed"] == seed and metadata["executor_unchanged"] is True, "CE metadata identity/freeze")
        checkpoint = sources.record(folder / "final.pt", metadata["checkpoint_sha256"])
        require(metadata["settings"] == settings, "CE settings differ from protocol")
        training[str(seed)] = dict(metadata_source=ref, checkpoint_source=checkpoint, metadata=metadata)
    curves = [learning_curve(root, kind, "frozen_B", settings, sources) for kind in KINDS]
    curves.append(learning_curve(root, "learned", "outcome_B", settings, sources))
    require(sum(len(seed["blocks"]) for curve in curves for seed in curve["per_seed"].values()) == 384,
            "incomplete B curve sources")
    result = dict(format="learned-binding-audited-results-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        settings=settings, audit_gate=dict(probe_count=39, B_block_count=384, native_replay_count=9,
            independently_recomputed_pg_updates=6, all_flags_true=True, report_source=audit_ref),
        definitions=dict(rates="Counts divided by all issued eligible requests whose native deadlines matured in that probe.",
            strict="Completed successfully with no rejected action while active before success; idle mistakes are separate.",
            ordinary="Successful completion regardless of prior rejected active actions.",
            translation_correct="Predicted target at the issuing command, on the same matured eligible denominator.",
            equal_seed_mean="Arithmetic mean of the three per-seed rates; null if any seed denominator is zero.",
            pooled="Sum of counts across seeds, with its own count-weighted ratio; not the reported equal-seed mean.",
            first_groups=FIRST_DEFINITION,
            idle="Non-WAIT actions per 1,000 actual idle job opportunities.",
            gates="Observed learned/provided gates only; success and command categories may overlap.",
            prospective="Each probe contains newly issued requests; different probe ticks are not the same request cohort.",
            interventions="Clear-M/clear-z are one-time interventions followed by normal state updates; no-read uses uniform payload probabilities."),
        comparability=dict(manual_native_training_equal=False,
            manual_native_note="Manual hard/soft reuse a different historical training budget and parameterization from native; do not infer an equal-training causal comparison.",
            native_frozen_pg_interaction_equal=True,
            native_branch_note="Native frozen/PG share support endpoint and B interaction budget; their actions and receipts can subsequently differ.",
            teacher_note="Teacher-supported probes are a separate diagnostic, never pooled with the own-action main line."),
        probes=probes, training=training, B_learning_curves=curves,
        provenance=dict(protocol_source=protocol_ref, training_freeze_source=freeze_ref, data_freeze_source=data_ref,
            source_parameters=origins, audit_calculation_script_current=audit_script,
            audit_script_binding_from_report={key: value for key, value in audit.items()
                                              if "source" in key and key != "sources"},
            results_generator=generator, files_sha256=sources.files,
            audited_payload_sha256=sources.payloads,
            payload_note="Payload hashes are inherited from JSON bundles already rehashed and replayed by audit.py; this summarizer does not rerun policies."))
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("results.json", "summary.csv", "summary.png") + COPIES:
        require(not (output / filename).exists(), f"refusing to mix with existing evidence: {output / filename}")
    write_json(output / "results.json", result)
    write_csv(output / "summary.csv", probes)
    plot_summary(output / "summary.png", probes, curves)
    for filename in COPIES:
        shutil.copyfile(root / filename, output / filename)
        require(sha(output / filename) == sources.files[sources.name(root / filename)], "evidence copy SHA")
    return dict(output=str(output.resolve()), variants=len(probes), probes=39, B_blocks=384,
                csv_rows=len(probes) * 2 * len(GROUPS),
                files={name: dict(sha256=sha(output / name), bytes=(output / name).stat().st_size)
                       for name in ("results.json", "summary.csv", "summary.png") + COPIES})


def self_test():
    """Exercise all output paths on disposable mock summaries, never formal data."""
    with tempfile.TemporaryDirectory(prefix="learned-binding-results-") as temporary:
        repository = Path(temporary)
        root, output = repository / "data/mock", repository / "evidence"
        root.mkdir(parents=True)
        settings = dict(seeds=list(SEEDS), pg_blocks=32, pg_ticks=512, actors=32)
        source = repository / "fixture_source.py"
        source.write_text("# temporary synthetic fixture\n", encoding="utf-8")
        origins, checkpoints = {}, {}
        for seed in SEEDS:
            checkpoint = root / f"seed_{seed:03d}/pretrain/final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic checkpoint")
            name = checkpoint.relative_to(repository).as_posix()
            checkpoints[name] = sha(checkpoint)
            origins[f"{seed}/executor"] = dict(path=name, sha256=sha(checkpoint))
            write_json(checkpoint.with_name("training.json"), dict(seed=seed, executor_unchanged=True,
                checkpoint_sha256=sha(checkpoint), settings=settings,
                parameter_report=dict(trainable_parameters=8210, state_scalars=320),
                credit_check=dict(executor_gradients_absent=True), log=[]))
        write_json(root / "protocol.json", dict(settings=settings,
            sources={source.relative_to(repository).as_posix(): sha(source)}, origins=origins))
        write_json(root / "training-freeze.json", dict(all_complete_before_scoring=True, checkpoints=checkpoints))
        write_json(root / "data-freeze.json", dict(streams=[]))
        audited, phases = [], []

        def bundle(prefix, report):
            for suffix, key in ((".npz", "arrays_sha256"), (".pt", "states_sha256")):
                prefix.with_suffix(suffix).write_bytes(b"synthetic audited payload")
                report[key] = sha(prefix.with_suffix(suffix))
            write_json(prefix.with_suffix(".json"), report)

        def split_counts(row):
            first = {key: value // 2 for key, value in row.items()}
            return first, {key: row[key] - first[key] for key in row}

        for seed in SEEDS:
            n, strict = ((10, 1), (20, 4), (40, 16))[seed]
            all_counts = dict(n=n, strict=strict, ordinary=min(n, 2 * strict), translation_correct=strict)
            a, b = split_counts(all_counts)
            ac, ash = split_counts(a)
            bc, bsh = split_counts(b)
            groups = dict(all=all_counts, A=a, B=b, A_conflict=ac, B_conflict=bc, A_shared=ash, B_shared=bsh)
            for kind in KINDS:
                folder = root / f"seed_{seed:03d}/behavior" / kind
                folder.mkdir(parents=True)
                manifest = {}
                for name in BASE_PROBES + (EXTRA_PROBES if kind == "learned" else ()):
                    report = dict(seed=seed, kind=kind, name=name, groups=groups, first_groups=groups,
                        idle_actions=seed + 1, idle_opportunities=100, memory_norm=1. + seed,
                        parameters=dict(state_scalars=320 if kind == "learned" else 264),
                        gate_statistics={category: dict(sum=5., count=10, mean=.5)
                                         for category in ("all", "success_finish", "command", "other")},
                        start_tick=4095, end_tick=6143, start_work=dict(ticks=4096), final_work=dict(ticks=6144),
                        right_censored=[], left_boundary_matured=[], first_right_censored=[],
                        command_translations=n, correct_translations=strict)
                    bundle(folder / name, report)
                    manifest[name + ".json"] = sha(folder / (name + ".json"))
                    audited.append(dict(seed=seed, kind=kind, name=name, groups=groups, world_replayed=True))
                write_json(folder / "completion.json", dict(seed=seed, kind=kind, files=manifest))
                for phase in ("frozen_B", "outcome_B") if kind == "learned" else ("frozen_B",):
                    phase_folder = folder / phase
                    phase_folder.mkdir()
                    records = []
                    for index in range(32):
                        update = dict(positive_finishes=index + seed + 1, rejected=index * 2,
                                      reward_sum=index + seed + 1 - .1 * index * 2)
                        records.append(update)
                        bundle(phase_folder / f"block_{index:03d}",
                               dict(index=index, learn=phase == "outcome_B", update=update))
                    write_json(phase_folder / "completion.json", dict(blocks=32, learn=phase == "outcome_B",
                        records=records, phase_ticks=16385, counted_extra_closure_ticks=1))
                    phases.append(dict(blocks=32, clocks_pending_and_live_state_chain_verified=True,
                        pg_checks=[dict(model_and_adam_bitwise_equal=True, public_rewards_recomputed=True,
                                        joint_actor_objective=True)] * 2 if phase == "outcome_B" else []))
        audit = dict(format="learned-binding-independent-audit-v1", sources=dict(source_files=1),
            probe_count=39, B_block_count=384, native_replay_count=9, independently_recomputed_pg_updates=6,
            all_world_actions_and_public_rewards_replayed=True,
            all_controller_live_states_preserved_across_optimizer_updates=True, all_source_executors_unchanged=True,
            oracle_paths=dict(only_oracle_call_is_explicit_teacher_support=True, own_driver_has_no_oracle_call=True),
            probes=audited, native_replays=[dict(arrays_bitwise=True, full_final_state_bitwise=True,
                                                oracle_disabled=True)] * 9, B_phases=phases)
        write_json(root / "audit.json", audit)
        for failed in (dict(audit, probe_count=38), dict(audit, all_source_executors_unchanged=False)):
            try:
                check_audit(failed)
            except ValueError:
                pass
            else:
                raise AssertionError("audit gate accepted a failing fixture")
        metadata = summarize(root, output, repository=repository)
        result = json.loads((output / "results.json").read_text(encoding="utf-8"))
        group = result["probes"][0]["groups"]["all"]
        require(math.isclose(group["equal_seed_mean"]["strict_rate"], (.1 + .2 + .4) / 3), "equal-seed average")
        require(math.isclose(group["pooled"]["strict_rate"], 21 / 70), "pooled ratio")
        window = result["B_learning_curves"][0]["per_seed"]["0"]["windows"][0]
        require(window["positive_finishes"] == 36 and window["actor_ticks"] == 8 * 512 * 32, "curve count/normalization")
        require(metadata["csv_rows"] == 234, "CSV dimensions")
        require((output / "summary.png").stat().st_size > 10000, "rendered figure missing")
        return dict(self_test="passed", probes=39, B_blocks=384, csv_rows=234,
                    equal_seed_rate=group["equal_seed_mean"]["strict_rate"], pooled_rate=group["pooled"]["strict_rate"],
                    temporary_fixture_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else summarize(args.root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
