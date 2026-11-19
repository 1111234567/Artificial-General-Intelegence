"""Read-only audit of exact-coordinate ablations; never evaluates or trains.

Uses raw checkpoint tensors, the canonical digest utility, and Python arithmetic.
Does not import edge_identity or call any study summarizer/intervention helper.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from experiments.continuity.run import state_digest

torch.set_num_threads(1)
BASE = ROOT / "data/action-return/edge-identity-v1"
SEMANTIC_FIELDS = {
    "format", "config", "model_config", "model", "optimizer", "parameter_layout",
    "hidden", "hidden_time", "world", "schedule", "schedule_history", "course_history",
    "replay", "operator", "credit", "rng", "stats", "auxiliary_stats", "work", "ticks",
    "updates", "hidden_resets", "birth", "action_hash", "data_hash", "trace", "last_chunk",
    "last_auxiliary", "last_live_family_tick", "last_aux_family_tick",
    "live_supervised_family_positions", "retirement_origin",
}
METRICS = ("strict_pp", "ordinary_pp", "idle_per1000")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal(actual, expected, label):
    if actual is None or expected is None:
        assert actual is expected, label
    else:
        assert math.isclose(actual, expected, abs_tol=1e-11, rel_tol=1e-13), (label, actual, expected)


def loaded(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def source_state(bound):
    checkpoint = loaded(bound["path"])
    if "native" in checkpoint:
        assert checkpoint["state_hash"] == state_digest({k: v for k, v in checkpoint.items() if k != "state_hash"})
        semantic = {k: checkpoint[k] for k in ("format", "adoption", "native")}
        native = checkpoint["native"]
    else:
        semantic = {k: checkpoint[k] for k in SEMANTIC_FIELDS}
        native = semantic
        assert checkpoint["state_hashes"]["semantic"] == state_digest(semantic)
        for key, value in checkpoint["state_hashes"].items():
            if key != "semantic":
                assert value == state_digest(checkpoint[key]), key
    assert bound["semantic_hash"] == state_digest(semantic)
    return native


def count_record(record):
    """Check all request-cohort denominator partitions and return raw counts."""
    counts = {}
    for cohort in ("primary", "inherited", "carryover_contract"):
        row = record[cohort]
        assert 0 <= row["clean_completed"] <= row["completed"] <= row["matured"]
        for field in ("matured", "completed", "clean_completed"):
            assert sum(row["per_actor"][field]) == row[field]
        for rate, numerator in (("completion_rate", "completed"), ("clean_completion_rate", "clean_completed")):
            equal(row[rate], row[numerator] / row["matured"] if row["matured"] else None, cohort + rate)
        issued = record["initial_eligible_pending"] if cohort == "inherited" else record[cohort + "_eligible_issued"]
        assert issued == row["matured"] + record["censored"][cohort]["count"]
        counts[cohort] = {
            key: row[key] for key in ("matured", "completed", "clean_completed", "request_fingerprint", "per_actor")
        }
        counts[cohort].update(eligible_issued=issued, censored=record["censored"][cohort])
    opportunity = record["opportunities"]
    assert opportunity["opportunities"] == opportunity["active_opportunities"] + opportunity["inactive_opportunities"]
    assert opportunity["opportunities"] == record["steps"] * record["batch_size"] * record["active_jobs"]
    assert opportunity["inactive_actions"] == opportunity["inactive_begin_actions"] + opportunity["inactive_finish_actions"]
    assert 0 <= opportunity["inactive_actions"] <= opportunity["inactive_opportunities"]
    for field, scale, denominator in (
        ("inactive_actions_per_idle_opportunity", 1, "inactive_opportunities"),
        ("inactive_actions_per_1000_idle_opportunities", 1000, "inactive_opportunities"),
        ("inactive_actions_per_1000_total_opportunities", 1000, "opportunities"),
    ):
        equal(record[field], scale * opportunity["inactive_actions"] / opportunity[denominator], field)
    counts["opportunities"] = opportunity
    return counts


def rates(record):
    primary, opportunity = record["primary"], record["opportunities"]
    return {
        "strict_pp": 100 * primary["clean_completed"] / primary["matured"],
        "ordinary_pp": 100 * primary["completed"] / primary["matured"],
        "idle_per1000": 1000 * opportunity["inactive_actions"] / opportunity["inactive_opportunities"],
    }


def main():
    plan = read(BASE / "protocol.json")
    summary = read(BASE / "summary.json")
    bound_files = {}

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        current = sha(path)
        if expected is not None:
            assert current == expected, path
        if path in bound_files:
            assert bound_files[path] == current
        bound_files[path] = current
        return current

    plan_hash = bind(BASE / "protocol.json")
    assert summary["protocol_sha256"] == plan_hash
    bind(plan["source"]["path"], plan["source"]["sha256"])
    bind(plan["absence_protocol"]["path"], plan["absence_protocol"]["sha256"])
    absence_plan = read(plan["absence_protocol"]["path"])
    for path, expected in absence_plan["execution_files"].items():
        bind(path, expected)
    assert plan["seeds"] == [0, 1, 2]
    assert plan["phases"] == ["initial", "withdrawal", "return"]
    assert plan["ticks"] == [147456, 163840, 172032]
    assert plan["families"] == [0, 1, 2, 3]
    assert plan["condition"] == "short" and plan["steps"] == 2048
    assert plan["posthoc"] and summary["posthoc"]
    assert plan["new_forks"] == summary["new_forks"] == 36
    assert plan["reused_baseline_forks"] == summary["reused_forks"] == 36
    identities, weights, rows, full_counts, result_bindings = [], [], [], [], []

    for seed, expected_count in enumerate((20, 8, 36)):
        src = next(row for row in plan["sources"] if row["seed"] == seed)
        for source in (src["magnitude_source"], src["signed_source"]):
            for key in ("checkpoint", "result"):
                bind(source[key], source[key + "_sha256"])
        mag_source = src["magnitude_source"]
        magnitude = source_state({"path": mag_source["checkpoint"], "semantic_hash": mag_source["endpoint_semantic_hash"]})
        operator = magnitude["operator"]
        retired_from_masks = (operator["initial_mask"] & ~operator["mask"]).nonzero().flatten().tolist()
        retirement_events = [row["event"] for row in operator["event_history"]]
        all_retired = [int(index) for row in retirement_events for index in row["actual_retire_indices"]]
        ids = sorted(all_retired)
        assert len(ids) == len(set(ids)) == operator["cumulative_deaths"] == expected_count
        assert ids == retired_from_masks == src["pair_indices"]
        assert operator["cumulative_births"] == 0
        assert not any(row["actual_birth_indices"] for row in retirement_events)
        assert not bool(magnitude["model"]["edge_shared"][ids].ne(0).any())
        assert int(operator["initial_mask"].sum()) == 640
        assert int(operator["mask"].sum()) == 640 - expected_count
        assert src["retired_count"] == expected_count
        assert magnitude["model"]["pair_index"][:, ids].T.tolist() == src["node_pairs"]
        identities.append(dict(seed=seed, count=expected_count, pair_indices=ids, node_pairs=src["node_pairs"],
                               ledger_mask_checkpoint_exact=True, initial_support=640, remaining_support=640-expected_count))
        bind(src["baseline_result"]["path"], src["baseline_result"]["sha256"])
        original_baselines = read(src["baseline_result"]["path"])
        path = BASE / f"seed_{seed:03d}/result.json"
        result_hash = bind(path)
        raw = read(path)
        assert raw["seed"] == seed and raw["protocol_sha256"] == plan_hash and raw["source_unchanged"]
        assert raw["new_forks"] == raw["reused_forks"] == len(raw["records"]) == 12
        assert len(raw["weights"]) == 3
        assert len({(row["phase"], row["family"]) for row in raw["records"]}) == 12
        result_bindings.append(dict(path=str(path), sha256=result_hash))

        for stage in src["stages"]:
            phase, tick, cp = stage["phase"], stage["tick"], stage["checkpoint"]
            bind(cp["path"], cp["sha256"])
            state = source_state(cp)
            assert state["ticks"] == tick and state["model_config"]["mode"] == "undirected"
            assert state["config"]["dtype"] == "float64"
            model = state["model"]
            model_hash = state_digest(model)
            shared = model["edge_shared"]
            assert shared.dtype == torch.float64 and int(shared.ne(0).sum()) == 640
            width = state["model_config"]["width"]
            assert torch.equal(model["pair_index"], torch.triu_indices(width, width, offset=1))
            assert model["pair_index"][:, ids].T.tolist() == src["node_pairs"]
            eligible = state["operator"].get("origin", state["operator"])["mask"]
            assert int(eligible.sum()) == 640 and bool(eligible[ids].all())
            changed_model = {key: tensor.clone() for key, tensor in model.items()}
            changed_model["edge_shared"][ids] = 0
            changed_hash = state_digest(changed_model)
            selected = shared[ids]
            assert int(selected.ne(0).sum()) == expected_count
            old_matrix = torch.zeros((width, width), dtype=torch.float64)
            new_matrix = old_matrix.clone()
            u, v = model["pair_index"]
            old_matrix[u, v] = old_matrix[v, u] = shared * (1 / math.sqrt(2))
            new_matrix[u, v] = new_matrix[v, u] = changed_model["edge_shared"] * (1 / math.sqrt(2))
            delta = new_matrix - old_matrix
            expected_delta = torch.zeros_like(delta)
            su, sv = model["pair_index"][:, ids]
            expected_delta[su, sv] = expected_delta[sv, su] = -selected * (1 / math.sqrt(2))
            assert torch.equal(delta, expected_delta)
            assert int(delta.ne(0).sum()) == 2 * expected_count
            assert torch.equal(delta, delta.T) and not bool(delta.diag().ne(0).any())
            stat = next(row for row in raw["weights"] if row["phase"] == phase)
            assert stat["seed"] == seed and stat["tick"] == tick
            assert stat["pair_indices"] == ids and stat["node_pairs"] == src["node_pairs"]
            assert stat["count"] == stat["source_nonzero_selected"] == expected_count
            assert stat["source_model_sha256"] == model_hash and stat["new_model_sha256"] == changed_hash
            assert stat["shared_values"] == selected.tolist()
            assert stat["physical_Wuv_values"] == old_matrix[su, sv].tolist()
            abs_values = selected.abs()
            recomputed = dict(abs_min=float(abs_values.min()), abs_median=float(torch.quantile(abs_values, .5)),
                              abs_mean=float(abs_values.mean()), abs_max=float(abs_values.max()),
                              rms=float(selected.square().mean().sqrt()), l1=float(abs_values.sum()),
                              l2=float(selected.norm()), removed_bidirectional_matrix_frobenius=float(delta.norm()),
                              source_complete_shared_l2=float(shared.norm()),
                              selected_l2_fraction=float(selected.norm() / shared.norm()))
            for key, value in recomputed.items():
                equal(stat[key], value, "weight " + key)
            weight_check = dict(seed=seed, phase=phase, tick=tick, pair_indices=ids, **recomputed,
                                shared_values=selected.tolist(), physical_Wuv_values=old_matrix[su, sv].tolist(),
                                delta_Wuv_values=delta[su, sv].tolist(), changed_oriented_couplings=int(delta.ne(0).sum()),
                                delta_W_hash=state_digest(delta), source_model_hash=model_hash, ablated_model_hash=changed_hash)
            weights.append(weight_check)
            hashes = dict(start_world_state_hash=state_digest(state["world"]), start_hidden_hash=state_digest({"hidden": state["hidden"]}),
                          start_scheduler_hash=state_digest(state["world"]["scheduler"]), start_event_hash=state["world"]["event_hash"])
            for family in range(4):
                row = next(r for r in raw["records"] if r["phase"] == phase and r["family"] == family)
                assert row["seed"] == seed and row["tick"] == tick
                baseline, ablated = row["baseline"], row["ablated"]
                archived = next(r for r in original_baselines["records"][phase] if r["family"] == family and r["condition"] == "short")
                assert baseline == archived, "baseline must equal entire archived record"
                for arm, record, expected_model in (("baseline", baseline, model), ("ablated", ablated, changed_model)):
                    assert record["family"] == family and record["condition"] == "short"
                    assert record["steps"] == 2048 and record["fork_tick"] == state["world"]["tick"]
                    assert record["end_tick"] == record["fork_tick"] + 2048
                    assert record["model"]["n_actions"] == 11 and record["action_policy"] == "model" and record["intervention"] == "none"
                    assert all(record[key] == 0 for key in ("world_resets", "hidden_resets", "parameter_updates"))
                    assert record["action_hash"] == record["model_action_hash"]
                    assert record["opportunities"]["model_action_changes"] == 0
                    assert record["parameter_hash"] == state_digest(expected_model)
                    assert record["source_continuation_hash"] == state_digest(dict(model=expected_model, world=state["world"], hidden=state["hidden"]))
                    for key, expected in hashes.items():
                        assert record[key] == expected, key
                    for path, expected in record["source_hashes"].items():
                        bind(ROOT / path, expected)
                    full_counts.append(dict(seed=seed, phase=phase, tick=tick, world_fork_tick=record["fork_tick"],
                                            family=family, arm=arm, **count_record(record)))
                for key in ("case_ids", "actor_ids", "initial_eligible_pending", "end_scheduler_hash", "end_event_hash"):
                    assert baseline[key] == ablated[key], key
                for cohort in ("primary", "inherited", "carryover_contract"):
                    for key in ("request_fingerprint", "matured"):
                        assert baseline[cohort][key] == ablated[cohort][key], (cohort, key)
                baseline_rates, ablated_rates = rates(baseline), rates(ablated)
                delta_rates = {key: ablated_rates[key] - baseline_rates[key] for key in METRICS}
                counts = {}
                for arm, record in (("baseline", baseline), ("ablated", ablated)):
                    p, o = record["primary"], record["opportunities"]
                    counts[arm] = dict(matured=p["matured"], strict_completed=p["clean_completed"], ordinary_completed=p["completed"],
                                       idle_actions=o["inactive_actions"], idle_opportunities=o["inactive_opportunities"])
                independent_row = dict(seed=seed, phase=phase, tick=tick, family=family, delta=delta_rates,
                                       baseline_rates=baseline_rates, ablated_rates=ablated_rates, counts=counts)
                expected_row = next(r for r in summary["rows"] if r["seed"] == seed and r["phase"] == phase and r["family"] == family)
                assert independent_row == expected_row
                for key in METRICS:
                    equal(row["delta"][key], delta_rates[key], "raw delta " + key)
                    equal(row["baseline_rates"][key], baseline_rates[key], "raw baseline " + key)
                    equal(row["ablated_rates"][key], ablated_rates[key], "raw ablated " + key)
                rows.append(independent_row)

    assert len(summary["rows"]) == len(rows) == 36
    assert len(summary["weights"]) == len(weights) == 9
    for binding in summary["result_bindings"]:
        bind(binding["path"], binding["sha256"])
    assert summary["result_bindings"] == result_bindings
    for recorded in summary["weights"]:
        original = read(BASE / f"seed_{recorded['seed']:03d}/result.json")
        assert recorded == next(row for row in original["weights"] if row["phase"] == recorded["phase"])
    aggregates = []
    for phase in plan["phases"]:
        for scope in ("family0", "all4_equal_family"):
            per_seed = []
            for seed in range(3):
                selected = [row for row in rows if row["seed"] == seed and row["phase"] == phase and (scope != "family0" or row["family"] == 0)]
                assert len(selected) == (1 if scope == "family0" else 4)
                per_seed.append(dict(seed=seed, delta={key: statistics.mean(row["delta"][key] for row in selected) for key in METRICS}))
            aggregate = dict(phase=phase, scope=scope, per_seed=per_seed,
                             delta={key: dict(mean=statistics.mean(row["delta"][key] for row in per_seed),
                                               sample_sd=statistics.stdev(row["delta"][key] for row in per_seed)) for key in METRICS})
            expected = next(row for row in summary["aggregates"] if row["phase"] == phase and row["scope"] == scope)
            assert aggregate == expected
            aggregates.append(aggregate)
    assert len(summary["aggregates"]) == 6
    with (BASE / "paired.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 36
    for recorded in csv_rows:
        selected = next(row for row in rows if all(str(row[key]) == recorded[key] for key in ("seed", "phase", "tick", "family")))
        for arm in ("baseline", "ablated"):
            for key, value in selected["counts"][arm].items():
                assert int(recorded[arm + "_" + key]) == value
        for key, value in selected["delta"].items():
            equal(float(recorded["delta_" + key]), value, "CSV delta " + key)
    bind(BASE / "summary.json")
    bind(BASE / "paired.csv")
    for path, expected in bound_files.items():
        assert sha(path) == expected, "artifact changed during read-only audit: " + path
    value = dict(format="action-return-independent-exact-edge-audit-v1", audit="passed",
                 created_utc=datetime.now(timezone.utc).isoformat(), verifier=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                 independent_seeds=3, protocol_sha256=plan_hash, result_bindings=result_bindings,
                 checked=dict(retirement_identities=3, source_timepoints=9, paired_forks=36, full_count_records=72,
                              independent_summary_rows=36, independently_recomputed_aggregates=6, csv_rows=36,
                              behavior_or_training_runs=0, torch_cpu_threads=torch.get_num_threads()),
                 identities=identities, weights=weights, paired=rows, full_denominators=full_counts, aggregates=aggregates,
                 bindings=bound_files,
                 scope="Raw checkpoint retirement ledgers/masks; exact selected pair identities; independent model-copy hashes and physical delta-W; original world/h and external streams; complete raw request/idle denominators; raw, summary, CSV and seed mean/sample SD arithmetic. No behavior rerun or new experiment.",
                 limitations=[
                     "Audits saved artifacts and their pinned executor; does not independently replay the behavioral trajectories because per-tick edge-ablation traces are not saved.",
                     "Posthoc instantaneous frozen ablation keeps historical hidden state. It measures current use of these control-model coordinates; it is not an online retirement trial or a longitudinal attribution of magnitude-versus-control differences.",
                     "Identities were selected from the already observed magnitude retirement ledger, not sampled randomly. Three seeds supply descriptive contrasts and sample SD only.",
                     "Idle opportunity denominators depend on each fork's own completion policy; both actual denominators are retained. Equal-family aggregate is the mean of family rates within each seed, not a pooled request-rate contrast.",
                 ])
    output = ROOT / "data/action-return/independent_audit_edge_identity.json"
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(audit="passed", output=str(output), checked=value["checked"])))


if __name__ == "__main__":
    main()
