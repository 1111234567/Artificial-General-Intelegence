"""Read-only, single-seed endpoint subset; never load a checkpoint or run a fork.

Run from F:\\agi with the project Python. Only the four declared seed_000
recorded512/fixed/warmup variants are read. The frozen reuse validator checks
the complete evaluation JSON; reported counts are only tick 196608's 8 forks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.undirected import queue, study


FOLDER = Path(__file__).resolve().parent
STEM = FOLDER / "interim-interfaces-seed0-2026-11-01"
PROTOCOL = FOLDER / "protocol-v1.json"
TICK = 196608
CONDITIONS = ("short", "long_both")
EVALUATOR_SOURCES = ("experiments/lifelong/evaluate.py", "experiments/lifelong/world.py",
                     "experiments/integration/diagnose.py", "experiments/continuity/world.py",
                     "experiments/continuity/evaluate.py")


def require(value, label):
    if not value:
        raise ValueError(label)


def read(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def same(actual, expected, label):
    require(actual is None if expected is None else type(actual) in (int, float)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), label)


def command_manifest(kind, protocol, sha):
    path = FOLDER / f"{kind}-commands-v1.json"
    manifest, jobs = queue._load_plan(kind, path)
    require(manifest["cwd"] == str(ROOT) and manifest["protocol"] == str(PROTOCOL), "command identity")
    require(manifest["file_hashes"] == {**protocol["sources"], str(PROTOCOL): sha}, "declared source inventory")
    require([job["branch"] for job in jobs] == protocol["branches"], "complete manifest branch inventory")
    return path, manifest, {job["id"]: job for job in jobs}


def queue_snapshot(kind, manifest_path, manifest):
    path = ROOT / f"data/undirected/main-v1/{kind}-execution.json"
    state, sha = read(path)
    require((state["format"], state["kind"], state["plan"], state["plan_digest"]) ==
            (queue.QUEUE_FORMAT, kind, str(manifest_path), queue.digest(manifest)), "queue snapshot binding")
    return state, dict(path=str(path), sha256=sha, note="Mutable queue file: hash binds these captured bytes only.")


def check_job(job, item, kind, protocol_sha):
    out = Path(item["output"])
    prefix = [str(study.PYTHON.resolve()), "-m", "experiments.undirected.study"]
    tail = ["--protocol", str(PROTOCOL), "--branch-id", item["id"]]
    if kind == "training":
        command, outputs = prefix + ["train"] + tail, [str(out / "checkpoint_latest.pt"), str(out / "terminal.json")]
    else:
        destination = ROOT / "data/undirected/main-v1/evaluation" / f"{item['id']}.json"
        command = prefix + ["evaluate"] + tail + ["--checkpoint", str(out / "checkpoint_latest.pt"), "--output", str(destination)]
        outputs = [str(destination)]
    require(job["command"] == command and job["outputs"] == outputs, "exact command/output")
    require((job["id"], job["branch_id"], job["branch"], job["protocol_sha256"], job["terminal"]) ==
            (item["id"], item["id"], item, protocol_sha, str(out / "terminal.json")), "job branch binding")


def completed_row(job, state, expected_status="success"):
    row = state["jobs"].get(job["id"], {})
    require(row.get("returncode") == 0 and row.get("process_returncode") == 0, "job not completed successfully")
    require(row.get("command_digest") == queue.digest(job["command"]), "saved command digest")
    require(row.get("terminal_status") == expected_status and row.get("endpoint_comparable") is True, "saved terminal status")
    require(set(row.get("output_hashes", {})) == set(job["outputs"]), "saved output hash inventory")
    for path, sha in row["output_hashes"].items():
        require(queue.file_sha256(path) == sha, "actual output bytes differ from saved queue record")
    return row


def counts(record, expected_sources, steps):
    require(record["source_hashes"] == expected_sources, "actual evaluator source binding")
    primary, opportunities = record["primary"], record["opportunities"]
    result = {name: primary[name] for name in ("matured", "completed", "clean_completed")}
    for name in result:
        require(type(result[name]) is int and result[name] >= 0, "raw count type")
        require(sum(primary["per_actor"][name]) == result[name], "raw per-actor count sum")
        require(sum(row["primary"][name] for row in record["family_cohorts"].values()) == result[name], "raw family count sum")
    require(result["clean_completed"] <= result["completed"] <= result["matured"], "cohort count ordering")
    require(result["matured"] + record["censored"]["primary"]["count"] == record["primary_eligible_issued"], "matured/right-censored bookkeeping")
    result.update({name: opportunities[name] for name in ("opportunities", "active_opportunities", "inactive_opportunities", "inactive_actions")})
    require(all(type(result[name]) is int and result[name] >= 0 for name in result), "opportunity count type")
    require(result["opportunities"] == steps * record["batch_size"] * record["active_jobs"], "actual total opportunity count")
    require(result["active_opportunities"] + result["inactive_opportunities"] == result["opportunities"], "opportunity partition")
    require(result["inactive_actions"] <= result["inactive_opportunities"], "idle action count")
    require(opportunities["inactive_begin_actions"] + opportunities["inactive_finish_actions"] == result["inactive_actions"], "idle action partition")
    result["completion_rate"] = ratio(result["completed"], result["matured"])
    result["clean_completion_rate"] = ratio(result["clean_completed"], result["matured"])
    result["idle_actions_per_1000_actual_idle_opportunities"] = ratio(1000 * result["inactive_actions"], result["inactive_opportunities"])
    same(primary["completion_rate"], result["completion_rate"], "stored completion rate")
    same(primary["clean_completion_rate"], result["clean_completion_rate"], "stored clean completion rate")
    same(record["inactive_actions_per_1000_idle_opportunities"], result["idle_actions_per_1000_actual_idle_opportunities"], "stored idle rate")
    if record["probe"]["condition"] == "long_both":
        diagnosis = record["request_diagnosis"]["primary"]
        names = ("requests", "completed", "clean_completed", "began", "with_observed_legal_opportunity",
                 "active_finish_attempts", "legally_timed_finish_attempts", "first_observed_legal_content_correct",
                 "first_observed_legal_count")
        diagnostic_counts = {name: diagnosis[name] for name in names}
        require(all(type(count) is int and count >= 0 for count in diagnostic_counts.values()), "diagnosis count type")
        require(diagnosis["requests"] == result["matured"] and diagnosis["completed"] == result["completed"]
                and diagnosis["clean_completed"] == result["clean_completed"], "diagnosis primary denominator/count binding")
        require(diagnosis["began"] <= diagnosis["requests"] and diagnosis["with_observed_legal_opportunity"] <= diagnosis["requests"]
                and diagnosis["first_observed_legal_content_correct"] <= diagnosis["first_observed_legal_count"]
                <= diagnosis["with_observed_legal_opportunity"]
                and diagnosis["legally_timed_finish_attempts"] <= diagnosis["active_finish_attempts"], "diagnosis count bounds")
        result["request_diagnosis_primary"] = diagnostic_counts
    return result


def extract(item, plan, sha, training_job, evaluation_job, training_state, evaluation_state):
    training = completed_row(training_job, training_state)
    evaluation = completed_row(evaluation_job, evaluation_state)
    terminal, terminal_sha = read(training_job["terminal"])
    require(training["output_hashes"][training_job["terminal"]] == terminal_sha, "actual terminal queue SHA")
    require(terminal["status"] == "success" and terminal["branch"] == item and terminal["protocol_sha256"] == sha,
            "actual training terminal identity")
    require(terminal["last_tick"] == terminal["target_ticks"] == plan["target_ticks"] and terminal["completed"] is True
            and terminal["endpoint_comparable"] is True, "actual completed training lifetime")
    require(training["output_hashes"][terminal["checkpoint_path"]] == terminal["checkpoint_sha256"], "training final checkpoint queue SHA")
    value, evaluation_sha = read(evaluation_job["outputs"][0])
    require(evaluation["output_hashes"][evaluation_job["outputs"][0]] == evaluation_sha, "actual evaluation queue SHA")
    # Frozen validator validates content hash, complete declared specs, terminal,
    # saved CP/layout and fork-state bindings. No load or evaluation is allowed.
    with patch.object(study.torch, "load", side_effect=AssertionError("checkpoint loading forbidden")), \
         patch.object(study, "evaluate_fork", side_effect=AssertionError("reevaluation forbidden")):
        study._checked_evaluation(value, plan, item, sha, terminal)
    checkpoint = terminal["checkpoints"][str(TICK)]
    require(queue.file_sha256(checkpoint["path"]) == checkpoint["sha256"], "actual main-endpoint checkpoint bytes")
    expected_sources = {name: plan["sources"][str(ROOT / name)] for name in EVALUATOR_SOURCES}
    selected = [record for record in value["records"] if record["probe"]["tick"] == TICK]
    expected_specs = [dict(tick=TICK, family=family, condition=condition, steps=plan["evaluation_steps"])
                      for family in range(4) for condition in CONDITIONS]
    require([record["probe"] for record in selected] == expected_specs, "exact eight main endpoint forks")
    rows = [dict(family=record["probe"]["family"], condition=record["probe"]["condition"],
                 **counts(record, expected_sources, plan["evaluation_steps"])) for record in selected]
    metrics = ("completion_rate", "clean_completion_rate", "idle_actions_per_1000_actual_idle_opportunities")
    means = {}
    for condition in CONDITIONS:
        family_rows = [row for row in rows if row["condition"] == condition]
        require(len(family_rows) == 4 and all(row[name] is not None for row in family_rows for name in metrics), "equal-family mean requires all four denominators")
        means[condition] = {name: sum(row[name] for row in family_rows) / 4 for name in metrics}
    diagnosis_rows = [row["request_diagnosis_primary"] for row in rows if row["condition"] == "long_both"]
    diagnosis_totals = {name: sum(row[name] for row in diagnosis_rows) for name in diagnosis_rows[0]}
    return dict(id=item["id"], interface=item["interface"], mode=item["mode"], seed=0, trained_seed_count=1,
                tick=TICK, counts=rows, family_equal_means=means, long_both_diagnosis_totals=diagnosis_totals, provenance=dict(
                    training_command=training_job["command"], training_command_digest=training["command_digest"],
                    training_queue_row=training, terminal_path=training_job["terminal"], terminal_sha256=terminal_sha,
                    evaluation_command=evaluation_job["command"], evaluation_command_digest=evaluation["command_digest"],
                    evaluation_queue_row=evaluation, evaluation_path=evaluation_job["outputs"][0], evaluation_sha256=evaluation_sha,
                    evaluation_result_hash=value["result_hash"], checkpoint_at_reported_tick=checkpoint,
                    evaluation_state_at_reported_tick=value["states"][str(TICK)], actual_evaluator_sources=expected_sources,
                    frozen_validator="experiments.undirected.study._checked_evaluation", frozen_validator_passed=True,
                    checkpoint_files_hashed_only=True, checkpoint_loads=0, evaluation_reruns=0))


def main():
    plan, sha = read(PROTOCOL)
    queue.verify_files(plan["sources"])
    items = [item for item in plan["branches"] if item["seed"] == 0 and item["credit"] == "recorded512"
             and item["structure"] == "fixed" and item["course"] == "warmup"]
    require(len(items) == 4 and {(item["interface"], item["mode"]) for item in items} ==
            {(interface, mode) for interface in ("core", "ports") for mode in ("undirected", "untied")}, "exact requested subset")
    manifests, states, provenance = {}, {}, {}
    for kind in ("training", "evaluation"):
        path, manifest, jobs = command_manifest(kind, plan, sha)
        state, snapshot = queue_snapshot(kind, path, manifest)
        manifests[kind], states[kind] = jobs, state
        provenance[kind] = dict(command_manifest=dict(path=str(path), sha256=queue.file_sha256(path)), execution_snapshot=snapshot)
        for item in items:
            check_job(jobs[item["id"]], item, kind, sha)
    branches, pending = [], []
    for item in items:
        identifier = item["id"]
        evaluation_job = manifests["evaluation"][identifier]
        row = states["evaluation"]["jobs"].get(identifier, {})
        if row.get("returncode") != 0 or not Path(evaluation_job["outputs"][0]).is_file():
            pending.append(dict(id=identifier, status="awaiting_completed_frozen_evaluation", process_returncode=row.get("returncode")))
            continue
        branches.append(extract(item, plan, sha, manifests["training"][identifier], evaluation_job, states["training"], states["evaluation"]))
    value = dict(format="undirected-interim-interfaces-seed0-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                 scope="Only seed_000, recorded512/fixed/warmup, interface x direction, main endpoint196608, four families short/long_both. Descriptive subset only; not the all-42 audit.",
                 complete_requested_subset=len(branches) == 4, requested_branches=[item["id"] for item in items],
                 available_branches=len(branches), pending=pending, branches=branches,
                 aggregation="Arithmetic mean of four family-specific rates per condition. One trained seed per variant; no across-seed mean/SD/CI or winner declaration.",
                 protocol=dict(path=str(PROTOCOL), sha256=sha), declared_source_hashes=plan["sources"],
                 queue_provenance=provenance, extractor=dict(path=str(Path(__file__).resolve()), sha256=queue.file_sha256(__file__)),
                 limitations=["Frozen validator checks JSON provenance, not an independent replay of behavior.",
                              "Checkpoint bytes were stream-hashed, not loaded; no model/world/training/evaluation was run.",
                              "Main endpoint196608 is before withdrawal/return; results do not measure final retention.",
                              "Family-average rates are not pooled request rates. Idle denominator is actual inactive opportunities.",
                              "First-legal content readout uses the existing eight FINISH logits on the actual state, independently of the executed eleven-way argmax; it is not a state-information upper bound or overall behavior."])
    queue.atomic_json(STEM.with_suffix(".json"), value)
    lines = ["# seed_000 四种接口/方向的中期行为子集", "", f"生成时间 UTC：{value['created_utc']}；已完成 {len(branches)}/4 分支。", "",
             "仅限 recorded512 / fixed / warmup，主训练终点 tick196608，每族 short 与 long_both 各2048步冻结续跑。每种配置只有一个训练 seed；下表是四个 family 各自比率的等权算术均值，不是跨 seed 均值，没有跨 seed 方差、置信区间或最终胜者结论。主终点早于撤回/回归，不代表最终保持率；本子集独立于42分支的完整审计。", "",
             "|接口|方向|条件|完成率 %|干净完成率 %|每千次实际闲置机会的动作数|", "|---|---|---|---:|---:|---:|"]
    for branch in branches:
        for condition, means in branch["family_equal_means"].items():
            lines.append(f"|{branch['interface']}|{branch['mode']}|{condition}|{100*means['completion_rate']:.4f}|{100*means['clean_completion_rate']:.4f}|{means['idle_actions_per_1000_actual_idle_opportunities']:.4f}|")
    lines += ["", "## long_both 的已成熟 primary 请求诊断", "",
              "下表是四族的原始计数之和；requests / completed / clean_completed 已逐族核对主分母。FINISH 尝试计数可以对同一请求累计多次。首次合法机会的内容读出是现有八个 FINISH logits 的条件 argmax，实际行动仍由完整十一个动作的 argmax 决定；该读出不等于状态信息上限，也不等于整体完成行为。", "",
              "|接口|方向|requests|completed / clean|已 BEGIN|观察到合法机会|活跃期 FINISH 尝试|合法时 FINISH 尝试|首次合法内容正确 / 读出数|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for branch in branches:
        d = branch["long_both_diagnosis_totals"]
        lines.append(f"|{branch['interface']}|{branch['mode']}|{d['requests']}|{d['completed']} / {d['clean_completed']}|{d['began']}|{d['with_observed_legal_opportunity']}|{d['active_finish_attempts']}|{d['legally_timed_finish_attempts']}|{d['first_observed_legal_content_correct']} / {d['first_observed_legal_count']}|")
    lines += ["", "逐族 matured / completed / clean_completed、实际机会与闲置动作原始计数、重算比率、训练及评估命令、队列完成记录和输出 SHA256 均保存在同名 JSON。对读取的评估执行冻结的 `_checked_evaluation` 校验，并核对自身内容哈希、声明 spec、终态和 checkpoint 绑定；实际训练输出及所报主终点 checkpoint 只流式哈希，没有加载。源码指纹按冻结协议和两个命令清单核验。未重跑评估、模型、世界或训练。", ""]
    if pending:
        lines += ["待完成："] + [f"- `{item['id']}`" for item in pending] + [""]
    STEM.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(available=len(branches), pending=[item["id"] for item in pending],
                          means={branch["interface"] + "/" + branch["mode"]: branch["family_equal_means"] for branch in branches},
                          long_both_diagnosis={branch["interface"] + "/" + branch["mode"]: branch["long_both_diagnosis_totals"] for branch in branches},
                          output=str(STEM.with_suffix(".json")), sha256=queue.file_sha256(STEM.with_suffix(".json"))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
