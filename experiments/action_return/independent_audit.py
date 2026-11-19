"""Independent arithmetic and decoder-artifact audit; never trains or acts.

This reader deliberately recomputes counts/rates and softmax objectives instead
of calling either study's summarizer. It is not part of either frozen executor.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/action-return"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(actual, expected, label, tol=1e-10):
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual} != {expected}")
    elif abs(actual - expected) > tol:
        raise AssertionError(f"{label}: {actual} != {expected}")


def ratio(counts, numerator, denominator, scale=1.):
    d = counts.get(denominator, 0)
    return scale * counts.get(numerator, 0) / d if d else None


def audit_readout(folder):
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    assert sha(plan["source"]["path"]) == plan["source"]["sha256"]
    for path, expected in plan["original_source_pins"].items():
        assert sha(path) == expected
    torch.set_num_threads(1)
    bindings, fits_verified, rates_verified, matched_cases = [], [], 0, 0
    for seed in range(3):
        out = folder / f"seed_{seed:03d}"
        raw, info = read(out / "result.json"), read(out / "fit.json")
        assert raw["seed"] == seed and raw["source_unchanged"]
        assert raw["protocol_sha256"] == sha(folder / "protocol.json")
        assert raw["fits_sha256"] == sha(out / "fits.pt")
        assert raw["fit_json_sha256"] == sha(out / "fit.json")
        assert info["train_dataset_sha256"] == sha(out / "train_dataset.pt.gz")
        saved = torch.load(out / "fits.pt", map_location="cpu", weights_only=False)
        data = torch.load(io.BytesIO(gzip.decompress((out / "train_dataset.pt.gz").read_bytes())),
                          map_location="cpu", weights_only=False)
        reconstruction = data["reconstruction"]
        assert set(reconstruction["train_actors"]).isdisjoint(reconstruction["test_actors"])
        assert reconstruction["actual_h_observations_labels_actions_and_world_endpoints_exact"]
        for arm, jobs in saved["fits"].items():
            for job, fit in enumerate(jobs):
                rows = data["groups"][job]
                x = rows["obs"] if arm == "public_obs56" else rows["h"]
                if arm == "non_input24":
                    x = x[:, data["non_input"]]
                labels = rows["label"]
                target = torch.where(labels <= 2, labels, 0)
                weights = torch.ones(len(labels), dtype=torch.float64)
                weights[labels == 0] = .1
                mean = x.mean(0)
                std = (x - mean).square().mean(0).sqrt()
                assert torch.allclose(mean, fit["mean"], atol=1e-13, rtol=0)
                assert torch.allclose(torch.where(std != 0, std, 1.), fit["std"], atol=1e-13, rtol=0)
                assert not bool(fit["coeff"][:, std == 0].ne(0).any())
                z = (x - fit["mean"]) / fit["std"]
                logits = torch.cat((torch.zeros((len(x), 1), dtype=torch.float64),
                                    z @ fit["coeff"].T + fit["bias"]), dim=1)
                objective = (weights * F.cross_entropy(logits, target, reduction="none")).sum() / weights.sum()
                objective += .001 * fit["coeff"].square().sum()
                recorded = info["fit"][arm][job]
                equal(float(objective), recorded["final_objective"], "decoder objective")
                assert recorded["positions"] == len(x) == 64 * 128 * 4
                assert recorded["original_action_counts"] == torch.bincount(labels, minlength=11).tolist()
                assert recorded["class_counts"] == torch.bincount(target, minlength=3).tolist()
                fits_verified.append(dict(seed=seed, arm=arm, job=job, positions=len(x),
                                          objective=float(objective), original_counts=recorded["original_action_counts"]))
        records = raw["records"]
        assert len(records) == 32
        archived = read(plan["sources"][str(seed)]["native_evaluation"]["path"])["records"]
        for record in records:
            if record["arm"] == "native":
                previous = next(r for r in archived if r["family"] == record["family"] and r["condition"] == record["condition"])
                assert record["record"] == previous, "native fork must equal its full archived original"
            all_counts = record["shadow"]["all"]["counts"]
            first, second = (record["shadow"][g]["counts"] for g in ("heldout", "overlap"))
            for key in set(all_counts) | set(first) | set(second):
                assert all_counts.get(key, 0) == first.get(key, 0) + second.get(key, 0), key
            for key, field in (("primary_matured", "matured"), ("primary_completed", "completed"),
                               ("primary_clean_completed", "clean_completed")):
                assert all_counts.get(key, 0) == record["record"]["primary"][field]
            native = next(r for r in records if r["arm"] == "native" and r["condition"] == record["condition"]
                          and r["family"] == record["family"])
            for group in ("heldout", "overlap", "all"):
                assert record["shadow"][group]["primary_request_count"] == record["shadow"][group]["counts"].get("primary_matured", 0)
                assert record["shadow"][group]["primary_request_set_sha256"] == native["shadow"][group]["primary_request_set_sha256"]
                matched_cases += 1
        for row in (r for r in summary["rows"] if r["seed"] == seed):
            counts = Counter()
            for r in records:
                if r["arm"] == row["arm"] and r["condition"] == row["condition"]:
                    counts.update(r["shadow"][row["actor_group"]]["counts"])
            assert dict(counts) == row["counts"]
            specs = {"began_matured_request_rate": ("primary_began", "primary_matured", 1),
                     "feasible_began_matured_request_rate": ("primary_feasible_began", "primary_matured", 1),
                     "ordinary_completion_rate": ("primary_completed", "primary_matured", 1),
                     "strict_completion_rate": ("primary_clean_completed", "primary_matured", 1),
                     "idle_false_begin_per1000": ("idle_false_begin", "idle_opportunities", 1000),
                     "correct_begin_sensitivity": ("correct_begin_at_teacher_position", "teacher_begin_positions", 1),
                     "any_begin_sensitivity": ("any_begin_at_teacher_position", "teacher_begin_positions", 1),
                     "begin_teacher_precision": ("correct_begin_at_teacher_position", "begin_attempts", 1)}
            for field, args in specs.items():
                equal(row[field], ratio(counts, *args), f"readout {field}")
                rates_verified += 1
        bindings.append(dict(path=str(out / "result.json"), sha256=sha(out / "result.json")))
    return dict(audit="passed", independent_seeds=3, decoder_fits=fits_verified,
                rates_verified=rates_verified, matched_group_case_hashes=matched_cases,
                result_bindings=bindings,
                scope="Saved artifacts and counts; source was independently reviewed. No new behavior or training.")


def native_rates(r):
    p, o = r["primary"], r["opportunities"]
    return dict(strict_pp=100 * p["clean_completed"] / p["matured"],
                ordinary_pp=100 * p["completed"] / p["matured"],
                idle_per1000=1000 * o["inactive_actions"] / o["inactive_opportunities"])


def audit_absence(folder):
    plan = read(folder / "protocol.json")
    inherited_pins = {}
    for previous in (ROOT / "data/coverage-exit/retirement-v1/protocol.json",
                     ROOT / "data/coverage-exit/magnitude-v1/protocol.json"):
        pins = read(previous)["execution_files"]
        for path, expected in pins.items():
            assert sha(path) == expected
        inherited_pins[str(previous)] = dict(sha256=sha(previous), execution_files_verified=len(pins))
    for path, expected in plan["execution_files"].items():
        assert sha(path) == expected
    report = read(folder / "report.json")
    results, bindings, rows_checked, paired_checked = {}, [], 0, 0
    for seed in range(3):
        for arm in ("magnitude", "signed_full"):
            path = folder / f"seed_{seed:03d}" / arm / "result.json"
            v = read(path)
            assert v["source_unchanged"] and v["seed"] == seed and v["arm"] == arm
            results[seed, arm] = v
            s = v["snapshots"]
            assert [s[p]["tick"] for p in ("initial", "withdrawal", "return")] == [147456, 163840, 172032]
            for phase, previous, updates in (("withdrawal", "initial", 128), ("return", "withdrawal", 64)):
                a, b = s[previous], s[phase]
                assert b["updates"] - a["updates"] == updates
                for key in ("eligible_pairs", "nonzero_pairs", "cumulative_births", "cumulative_deaths", "probe_calls"):
                    assert b["operator"][key] == a["operator"][key], key
                assert b["work"]["optimizer_steps"] - a["work"]["optimizer_steps"] == updates
                assert b["work"]["structural_probe_calls"] == a["work"]["structural_probe_calls"]
            for phase, records in v["records"].items():
                assert len(records) == 8
                for r in records:
                    row = next(x for x in report["counts"] if x["seed"] == seed and x["arm"] == arm
                               and x["phase"] == phase and x["family"] == r["family"] and x["condition"] == r["condition"])
                    for key, value in native_rates(r).items():
                        equal(row[key], value, f"absence {key}")
                    assert row["matured"] == r["primary"]["matured"]
                    assert row["strict_completed"] == r["primary"]["clean_completed"]
                    assert row["ordinary_completed"] == r["primary"]["completed"]
                    rows_checked += 1
            bindings.append(dict(path=str(path), sha256=sha(path)))
    def find(v, phase, family, condition):
        return next(r for r in v["records"][phase] if r["family"] == family and r["condition"] == condition)
    for row in report["paired"]:
        seed, phase, family, condition = (row[k] for k in ("seed", "phase", "family", "condition"))
        m, c = (results[seed, a] for a in ("magnitude", "signed_full"))
        a, b, ai, bi = (native_rates(find(v, p, family, condition))
                         for v, p in ((m, phase), (c, phase), (m, "initial"), (c, "initial")))
        previous = "withdrawal" if phase == "return" else "initial"
        ap, bp = (native_rates(find(v, previous, family, condition)) for v in (m, c))
        mr, cr = find(m, phase, family, condition), find(c, phase, family, condition)
        for key in ("case_ids", "actor_ids", "fork_tick", "start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash"):
            assert mr[key] == cr[key], key
        assert mr["primary"]["request_fingerprint"] == cr["primary"]["request_fingerprint"]
        for key in a:
            equal(row[f"raw_{key}"], a[key] - b[key], "paired raw")
            equal(row[f"difference_in_changes_{key}"], (a[key] - ai[key]) - (b[key] - bi[key]), "paired change")
            equal(row[f"difference_in_phase_changes_{key}"], (a[key] - ap[key]) - (b[key] - bp[key]), "paired phase change")
            paired_checked += 3
    for row in report["aggregate"]:
        selected = [r for r in report["paired"] if all(r[k] == row[k] for k in ("phase", "family", "condition"))]
        assert len(selected) == 3
        for key, value in row.items():
            if key.startswith(("raw_", "difference_")):
                equal(value["mean"], statistics.mean(r[key] for r in selected), "paired aggregate mean")
                equal(value["sample_sd"], statistics.stdev(r[key] for r in selected), "paired aggregate SD")
    return dict(audit="passed", independent_seeds=3, endpoint_count_rows=rows_checked,
                inherited_execution_provenance=inherited_pins,
                paired_values_checked=paired_checked, result_bindings=bindings,
                family0_short_acquired_counts=[{k:r[k] for k in ("seed", "arm", "matured", "strict_completed", "ordinary_completed", "strict_pp")}
                                               for r in report["counts"] if r["phase"] == "initial" and r["family"] == 0 and r["condition"] == "short"],
                scope="Raw per-fork count arithmetic, paired cases, update/support budgets, change contrasts, seed mean/sample SD.")


def audit_finish(folder):
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    assert sha(ROOT / "experiments/action_return/finish_readout.py") == plan["source_sha256"]
    assert sha(BASE / "readout-v1/protocol.json") == plan["parent_protocol_sha256"]
    torch.set_num_threads(1)
    from experiments.action_return.finish_readout import replace_other
    generator = torch.Generator().manual_seed(8042026)
    decisions = 0
    for scale in (0., 1e-9, 1., 1e9):
        original = torch.randn((2000, 2, 11), generator=generator, dtype=torch.float64) * scale
        offsets = torch.randn((2000, 2, 8), generator=generator, dtype=torch.float64) * scale
        inherited = original.argmax(-1)
        conditional = torch.cat((torch.zeros_like(offsets[..., :1]), offsets), -1).argmax(-1)
        conditional = torch.where(conditional == 0, 0, conditional + 2)
        expected = torch.where((inherited == 1) | (inherited == 2), inherited, conditional)
        assert torch.equal(replace_other(original, offsets).argmax(-1), expected)
        decisions += inherited.numel()
    objectives, bindings, matched_cases = [], [], 0
    for path, expected in plan["bindings"].items():
        assert sha(path) == expected
    for seed in range(3):
        root = BASE / "readout-v1" / f"seed_{seed:03d}"
        out = folder / f"seed_{seed:03d}"
        raw = read(out / "result.json")
        assert raw["protocol_sha256"] == sha(folder / "protocol.json")
        assert raw["model_sha256"] == sha(out / "model.pt")
        model = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
        tables = torch.load(io.BytesIO(gzip.decompress((root / "train_dataset.pt.gz").read_bytes())),
                            map_location="cpu", weights_only=False)
        control = read(root / "result.json")
        for job in range(2):
            rows = tables["groups"][job]
            selected = (rows["label"] == 0) | (rows["label"] >= 3)
            actions = rows["label"][selected]
            x = rows["h"][selected][:, tables["non_input"]]
            labels = torch.where(actions == 0, 0, actions - 2)
            weights = torch.ones(len(labels), dtype=torch.float64)
            weights[labels == 0] = .1
            mean, std, coef, bias = (model[f"j{job}_{k}"] for k in ("mean", "std", "coef", "bias"))
            recomputed_std = (x - x.mean(0)).square().mean(0).sqrt()
            assert torch.allclose(mean, x.mean(0), atol=1e-13, rtol=0)
            assert torch.allclose(std, torch.where(recomputed_std != 0, recomputed_std, 1.), atol=1e-13, rtol=0)
            logits = torch.cat((torch.zeros((len(x), 1), dtype=torch.float64),
                                ((x - mean) / std) @ coef.T + bias), dim=-1)
            objective = (weights * F.cross_entropy(logits, labels, reduction="none")).sum() / weights.sum()
            objective += .001 * coef.square().sum()
            equal(float(objective), raw["diagnostics"][job]["final"], "conditional finish objective")
            assert torch.bincount(labels, minlength=9).tolist() == raw["diagnostics"][job]["class_counts"]
            objectives.append(dict(seed=seed, job=job, positions=len(x), objective=float(objective)))
        for condition in ("short", "long_both"):
            totals = {name: Counter() for name in ("begin_only", "begin_and_finish")}
            responses = Counter()
            for family in range(4):
                a = next(r for r in control["records"] if r["arm"] == "non_input24" and r["condition"] == condition and r["family"] == family)
                b = next(r for r in raw["records"] if r["condition"] == condition and r["family"] == family)
                for key in ("start_world_state_hash", "start_hidden_hash", "start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash"):
                    assert a["record"][key] == b["record"][key]
                for group in ("heldout", "overlap", "all"):
                    assert a["shadow"][group]["primary_request_set_sha256"] == b["shadow"][group]["primary_request_set_sha256"]
                    matched_cases += 1
                response = b["heldout_finish_responses"]
                assert response.get("finish_labels", 0) == b["shadow"]["heldout"]["counts"].get("teacher_finish_labels", 0)
                assert response.get("finish_labels", 0) == sum(response.get(k, 0) for k in ("chose_begin", "chose_wait", "chose_finish"))
                assert response.get("correct_finish", 0) <= response.get("chose_finish", 0)
                responses.update(response)
                for name, record in (("begin_only", a), ("begin_and_finish", b)):
                    totals[name].update(record["shadow"]["heldout"]["counts"])
            for policy, counts in totals.items():
                row = next(r for r in summary["heldout"] if r["seed"] == seed and r["condition"] == condition and r["policy"] == policy)
                assert row["counts"] == dict(counts)
                if policy == "begin_and_finish":
                    assert row["finish_responses"] == dict(responses)
        bindings.append(dict(path=str(out / "result.json"), sha256=sha(out / "result.json")))
    return dict(audit="passed", independent_seeds=3, conditional_objectives=objectives,
                same_state_policy_decisions_verified=decisions,
                matched_group_case_hashes=matched_cases, result_bindings=bindings,
                scope="Conditional fit objectives, original row weights, fixed feature standardization and heldout matched-fork counts.")


def audit_swap(folder):
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    assert sha(plan["source"]["path"]) == plan["source"]["sha256"]
    assert sha(plan["main_protocol"]["path"]) == plan["main_protocol"]["sha256"]
    originals_verified = pulse_partitions = rates_verified = 0
    bindings = []
    for seed in range(3):
        for pin in plan["sources"][str(seed)].values():
            assert sha(pin["path"]) == pin["sha256"]
        main = read(plan["sources"][str(seed)]["result"]["path"])
        path = folder / f"seed_{seed:03d}/result.json"
        raw = read(path)
        assert raw["source_unchanged"] and len(raw["records"]) == 16
        assert raw["protocol_sha256"] == sha(folder / "protocol.json")
        for r in raw["records"]:
            base = next(x for x in main["records"] if x["arm"] == "non_input24" and x["condition"] == r["condition"] and x["family"] == r["family"])
            if r["arm"] == "original_non_input24":
                assert r["record"] == base["record"]
                for group in r["shadow"]:
                    assert {k:v for k,v in r["shadow"][group].items() if k != "pulse_counts"} == base["shadow"][group]
                originals_verified += 1
            else:
                gate = r["record"]["model"]
                assert gate["same_h_gate_checks"] == 2048 * 8 * 2
                assert gate["unique_selected_begin_reversals"] + gate["tied_selected_begin"] == r["shadow"]["all"]["counts"].get("begin_attempts", 0)
            for key in ("start_world_state_hash", "start_hidden_hash", "end_scheduler_hash", "end_event_hash", "actor_ids", "case_ids"):
                assert r["record"][key] == base["record"][key]
            for group in r["shadow"]:
                a = r["shadow"][group]
                assert a["primary_request_set_sha256"] == base["shadow"][group]["primary_request_set_sha256"]
                e, c = a["pulse_counts"], a["counts"]
                assert c.get("primary_feasible_began", 0) == e.get("primary_feasible_began_at_cue", 0) + e.get("primary_feasible_began_without_pulse", 0)
                pulse_partitions += 1
        for row in (r for r in summary["rows"] if r["seed"] == seed):
            counts, pulse = Counter(), Counter()
            for r in raw["records"]:
                if r["arm"] == row["arm"] and r["condition"] == row["condition"]:
                    counts.update(r["shadow"][row["actor_group"]]["counts"])
                    pulse.update(r["shadow"][row["actor_group"]]["pulse_counts"])
            assert dict(counts) == row["counts"] and dict(pulse) == row["pulse_counts"]
            equal(row["feasible_began_rate"], ratio(counts, "primary_feasible_began", "primary_matured"), "swap began")
            equal(row["idle_actions_per1000"], ratio(counts, "idle_actions", "idle_opportunities", 1000), "swap idle")
            rates_verified += 2
        bindings.append(dict(path=str(path), sha256=sha(path)))
    return dict(audit="passed", independent_seeds=3, exact_original_forks=originals_verified,
                pulse_partitions_verified=pulse_partitions, rates_verified=rates_verified, result_bindings=bindings,
                scope="Posthoc fixed score-swap; full original repeat equality, paired external streams, matured request IDs, pulse count partition and summary rates.")


def audit_crossed(folder):
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    assert sha(ROOT / "experiments/action_return/crossed_readout.py") == plan["source_sha256"]
    for path, expected in plan["inputs"].items():
        assert sha(path) == expected
    assert summary["additional_forks"] == 24
    matched_cases = count_rows = 0
    bindings = []
    for seed in range(3):
        base = read(BASE / f"readout-v1/seed_{seed:03d}/result.json")["records"]
        finish = read(BASE / f"finish-readout-v1/seed_{seed:03d}/result.json")["records"]
        path = folder / f"seed_{seed:03d}.json"
        raw = read(path)
        assert raw["protocol_sha256"] == sha(folder / "protocol.json")
        assert len(raw["records"]) == 8
        for record in raw["records"]:
            comps = [next(r for r in base if r["arm"] == arm and r["condition"] == record["condition"] and r["family"] == record["family"])
                     for arm in ("non_input24", "public_obs56")]
            comps.append(next(r for r in finish if r["condition"] == record["condition"] and r["family"] == record["family"]))
            for comp in comps:
                for key in ("start_world_state_hash", "start_hidden_hash", "start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash", "actor_ids", "case_ids"):
                    assert record["record"][key] == comp["record"][key]
                for group in ("heldout", "overlap", "all"):
                    assert record["shadow"][group]["primary_request_set_sha256"] == comp["shadow"][group]["primary_request_set_sha256"]
                    matched_cases += 1
            response = record["heldout_finish_responses"]
            assert response.get("finish_labels", 0) == sum(response.get(k, 0) for k in ("chose_begin", "chose_wait", "chose_finish"))
            assert response.get("finish_labels", 0) == record["shadow"]["heldout"]["counts"].get("teacher_finish_labels", 0)
        cells = {("non_input24", "native"): [r for r in base if r["arm"] == "non_input24"],
                 ("public_obs56", "native"): [r for r in base if r["arm"] == "public_obs56"],
                 ("non_input24", "non_input24"): finish,
                 ("public_obs56", "non_input24"): raw["records"]}
        for row in (r for r in summary["heldout"] if r["seed"] == seed):
            c = Counter()
            selected = [r for r in cells[row["begin"], row["finish"]] if r["condition"] == row["condition"]]
            assert len(selected) == 4
            for r in selected:
                c.update(r["shadow"]["heldout"]["counts"])
            assert dict(c) == row["counts"]
            count_rows += 1
        bindings.append(dict(path=str(path), sha256=sha(path)))
    return dict(audit="passed", independent_seeds=3, new_forks=24, summary_count_rows=count_rows,
                matched_group_case_hashes=matched_cases, result_bindings=bindings,
                scope="Frozen source/input hashes, source-reviewed direct coefficient reuse, all four cells' matched cases and raw heldout counts.")


def refit_rates(c):
    return dict(strict_pp=ratio(c, "primary_clean_completed", "primary_matured", 100),
                ordinary_pp=ratio(c, "primary_completed", "primary_matured", 100),
                feasible_begin_pp=ratio(c, "primary_feasible_began", "primary_matured", 100),
                idle_actions_per1000=ratio(c, "idle_actions", "idle_opportunities", 1000),
                idle_begin_per1000=ratio(c, "idle_false_begin", "idle_opportunities", 1000),
                begin_attempts_per_matured=ratio(c, "begin_attempts", "primary_matured"),
                finish_attempts_per_matured=ratio(c, "finish_attempts", "primary_matured"))


def audit_refit(folder):
    from experiments.continuity.run import state_digest
    from experiments.action_return import readout as begin, finish_readout as finish
    from experiments.undirected.model import PortGraphModel
    plan, summary = read(folder / "protocol.json"), read(folder / "summary.json")
    torch.set_num_threads(1)
    for group in ("source_files", "input_files", "previous_protocols"):
        for path, expected in plan[group].items():
            assert sha(path) == expected
    objectives, bindings, replay_ticks, paired_values = [], [], 0, 0
    for seed in range(3):
        src = plan["seeds"][str(seed)]
        path = folder / f"seed_{seed:03d}/result.json"
        raw, anchor = read(path), read(src["anchor"])
        assert raw["source_unchanged"] and raw["protocol_hash"] == plan["protocol_hash"]
        assert raw["result_hash"] == state_digest({k:v for k,v in raw.items() if k != "result_hash"})
        for file, expected in raw["collection_files"].items():
            assert sha(file) == expected
        tables = torch.load(io.BytesIO(gzip.decompress(Path(src["dataset"]).read_bytes())), map_location="cpu", weights_only=False)
        indices = tables["non_input"]
        own = torch.load(path.parent / "own_collection.pt", map_location="cpu", weights_only=False)
        archive = torch.load(path.parent / "archive_collection.pt", map_location="cpu", weights_only=False)
        trajectory = own["trajectory"]
        assert trajectory["observations"].shape == (8192, 8, 56)
        assert trajectory["actual_actions"].shape == trajectory["labels"].shape == (8192, 8, 2)
        assert state_digest(trajectory["start_h"]) == raw["collection"]["midpoint"]["frozen_h_hash"]
        assert raw["collection"]["midpoint"]["tick"] == 139263
        for flag in ("original_assisted_actual_endpoint_verified", "original_dataset_reconstruction_exact", "original_collector_state_exact", "matched_external_streams"):
            assert raw["collection"][flag]
        for a, b in zip(archive["trace"], trajectory["trace"]):
            for key in ("scheduler_hash", "event_hash", "family_exposures", "tick", "case_ids", "actor_ids"):
                assert a[key] == b[key]
        prefix_actions = [torch.zeros(11, dtype=torch.long) for _ in range(2)]
        for file in src["audits"][:64]:
            audit = torch.load(io.BytesIO(gzip.decompress(Path(file).read_bytes())), map_location="cpu", weights_only=False)
            actions = audit["actions"]["actual_actions"]
            for job in range(2):
                prefix_actions[job] += torch.bincount(actions[:, :4, job].reshape(-1), minlength=11)
        # Replay every recorded self-policy model step from its actual saved h.
        # Labels remain analysis-only and do not enter this model invocation.
        original_fit = torch.load(src["fits"], map_location="cpu", weights_only=False)["fits"]["non_input24"]
        state = torch.load(src["collector_model"], map_location="cpu", weights_only=False)
        collector = finish.FinishController(begin.BeginController(PortGraphModel().double(), "non_input24", original_fit, indices),
            [{k:state[f"j{j}_{k}"] for k in ("mean", "std", "coef", "bias")} for j in range(2)])
        collector.load_state_dict(state, strict=True)
        h = trajectory["start_h"].clone()
        expected_features = torch.stack([own["rows"][j]["x"].reshape(8192, 4, 24) for j in range(2)], dim=2)
        assert torch.equal(expected_features[:, :, 0], expected_features[:, :, 1])
        with torch.no_grad():
            for t, obs in enumerate(trajectory["observations"]):
                logits, h = collector.step(obs, h, .25)
                assert torch.equal(logits.argmax(-1), trajectory["actual_actions"][t]), "self-policy action replay"
                assert torch.equal(h[:4, indices], expected_features[t, :, 0]), "self-policy continuous feature replay"
                replay_ticks += 1
        assert torch.equal(h, trajectory["end_h"])
        for job in range(2):
            assert torch.equal(own["rows"][job]["label"], trajectory["labels"][:, :4, job].reshape(-1))
            assert torch.equal(own["rows"][job]["action"], trajectory["actual_actions"][:, :4, job].reshape(-1))
            assert own["rows"][job]["request"] == archive["rows"][job]["request"]
        for arm, added in (("assisted_archive", archive), ("own_policy", own)):
            model_file = path.parent / f"model_{arm}.pt"
            assert sha(model_file) == raw["model_sha256"][arm]
            fitted = torch.load(model_file, map_location="cpu", weights_only=False)
            for job in range(2):
                x = torch.cat((tables["groups"][job]["h"][:, indices], added["rows"][job]["x"]))
                y = torch.cat((tables["groups"][job]["label"], added["rows"][job]["label"]))
                assert len(x) == len(y) == 65536
                info = raw["fit"][arm][job]
                assert info["positions"] == 65536
                assert info["teacher_label_counts"] == torch.bincount(y, minlength=11).tolist()
                assert info["executed_action_counts"] == (prefix_actions[job] + torch.bincount(added["rows"][job]["action"], minlength=11)).tolist()
                for kind in ("begin", "finish"):
                    selected = torch.ones(len(y), dtype=torch.bool) if kind == "begin" else (y == 0) | (y >= 3)
                    xx, yy = x[selected], y[selected]
                    target = torch.where(yy <= 2, yy, 0) if kind == "begin" else torch.where(yy == 0, 0, yy - 2)
                    prefix = f"begin_head.j{job}_" if kind == "begin" else f"j{job}_"
                    mean, std, coef, bias = (fitted[prefix + k] for k in ("mean", "std", "coeff" if kind == "begin" else "coef", "bias"))
                    sigma = (xx - xx.mean(0)).square().mean(0).sqrt()
                    assert torch.allclose(mean, xx.mean(0), atol=1e-13, rtol=0)
                    assert torch.allclose(std, torch.where(sigma != 0, sigma, 1.), atol=1e-13, rtol=0)
                    weights = torch.ones(len(yy), dtype=torch.float64)
                    weights[yy == 0] = .1
                    logits = torch.cat((torch.zeros((len(xx), 1), dtype=torch.float64), ((xx - mean) / std) @ coef.T + bias), -1)
                    objective = (weights * F.cross_entropy(logits, target, reduction="none")).sum() / weights.sum() + .001 * coef.square().sum()
                    equal(float(objective), info[kind]["final_objective" if kind == "begin" else "final"], "refit objective")
                    objectives.append(dict(seed=seed, arm=arm, job=job, head=kind, positions=len(xx), objective=float(objective)))
        records = raw["records"]
        assert len(records) == 16
        for condition in ("short", "long_both"):
            totals = {arm:Counter() for arm in ("assisted_archive", "own_policy", "unrefit_anchor")}
            for family in range(4):
                cases = [next(r for r in records if r["arm"] == arm and r["condition"] == condition and r["family"] == family) for arm in ("assisted_archive", "own_policy")]
                cases.append(next(r for r in anchor["records"] if r["condition"] == condition and r["family"] == family))
                for record in cases[1:]:
                    for key in ("start_world_state_hash", "start_hidden_hash", "start_scheduler_hash", "end_scheduler_hash", "start_event_hash", "end_event_hash"):
                        assert record["record"][key] == cases[0]["record"][key]
                    for group in ("heldout", "overlap", "all"):
                        assert record["shadow"][group]["primary_request_set_sha256"] == cases[0]["shadow"][group]["primary_request_set_sha256"]
                for arm, record in zip(totals, cases):
                    totals[arm].update(record["shadow"]["heldout"]["counts"])
            rates = {}
            for arm, c in totals.items():
                row = next(r for r in summary["heldout"] if r["seed"] == seed and r["condition"] == condition and r["arm"] == arm)
                assert row["counts"] == dict(c)
                rates[arm] = refit_rates(c)
                for k, v in rates[arm].items():
                    equal(row["rates"][k], v, "refit rates")
            paired = next(r for r in summary["paired"] if r["seed"] == seed and r["condition"] == condition)
            for k in rates["own_policy"]:
                equal(paired[k], rates["own_policy"][k] - rates["assisted_archive"][k], "refit paired")
                paired_values += 1
        bindings.append(dict(path=str(path), sha256=sha(path)))
    for condition, values in summary["aggregate"].items():
        selected = [r for r in summary["paired"] if r["condition"] == condition]
        for metric, pair in values.items():
            equal(pair["mean"], statistics.mean(r[metric] for r in selected), "refit paired mean")
            equal(pair["sample_sd"], statistics.stdev(r[metric] for r in selected), "refit paired SD")
    return dict(audit="passed", independent_seeds=3, fitted_heads=objectives,
                own_policy_model_replay_ticks=replay_ticks, paired_values_verified=paired_values, result_bindings=bindings,
                scope="All input/source SHA; exact32768+32768-row budgets,24 head objectives, every own-policy action and continuous feature from saved obs/h, matched cases and aggregate rates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("readout", "absence", "finish", "swap", "crossed", "refit", "all"), default="all")
    args = parser.parse_args()
    value = dict(format="action-return-independent-audit-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                 verifier=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), studies={},
                 interpretation_limits=[
                     "Decoder-fitting heldout lanes were used by earlier backbone training; this is not a wholly unseen-backbone population.",
                     "BEGIN policies use all public ticks including current command cues. Success does not isolate delayed-memory use.",
                     "No-current-pulse BEGINs can still use public action/result receipts, including information from rejected trial actions.",
                     "External trained action scores change actual choices but do not establish native recurrent-port learning or synaptic repair.",
                     "Same-h gate invariance in conditional FINISH and swap controls does not imply identical gates on later divergent trajectories.",
                     "Withdrawal compares inherited strategies with different learned runtime states; it does not isolate deletion from all prior plasticity.",
                     "Family0 external NEW withdrawal retains pending requests and substantial old-family historical supervision.",
                     "Short family0 has acquired strict behavior; the long-task completion floor cannot demonstrate retention.",
                     "Three independent seeds provide descriptive differences and sample SD, not a population-scale efficacy claim.",
                 ])
    if args.study in ("readout", "all"):
        value["studies"]["readout"] = audit_readout(BASE / "readout-v1")
    if args.study in ("absence", "all"):
        value["studies"]["absence"] = audit_absence(BASE / "absence-v1")
    if args.study in ("finish", "all"):
        value["studies"]["finish"] = audit_finish(BASE / "finish-readout-v1")
    if args.study in ("swap", "all"):
        value["studies"]["swap"] = audit_swap(BASE / "readout-swap-v1")
    if args.study in ("crossed", "all"):
        value["studies"]["crossed"] = audit_crossed(BASE / "crossed-readout-v1")
    if args.study in ("refit", "all"):
        value["studies"]["refit"] = audit_refit(BASE / "refit-policy-v1")
    path = BASE / f"independent_audit_{args.study}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(audit="passed", output=str(path), studies=list(value["studies"]))))


if __name__ == "__main__":
    main()
