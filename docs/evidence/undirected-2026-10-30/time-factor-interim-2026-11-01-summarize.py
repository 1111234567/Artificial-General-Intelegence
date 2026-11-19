"""Recover source-declared JSON types and verify every existing pre-JSON digest.

This repair only decodes existing records. It performs no model forward, teacher
call, checkpoint load, or evaluation rerun, and never rewrites a raw fork.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("time-factor-interim-2026-11-01.py")
spec = importlib.util.spec_from_file_location("time_factor_original_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
original_check = runner.checked_record


def restore_declared_types(value):
    result = deepcopy(value)
    record = result["record"]
    assert set(record["phase_codes"]) == {"0", "1", "2", "3"}
    record["phase_codes"] = {int(key): item for key, item in record["phase_codes"].items()}
    for name in ("duration_range", "gap_range", "window_range"):
        assert isinstance(record["regime"][name], list) and len(record["regime"][name]) == 2
        record["regime"][name] = tuple(record["regime"][name])
    # The inverse changes container/key types only, never a scientific value.
    assert json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False)) == value
    return result


def checked_record(value, specification, model, protocol_sha):
    restored = restore_declared_types(value)
    original_check(restored, specification, model, protocol_sha)
    return value["record"]


def main():
    protocol = runner.load_protocol()
    protocol_sha = runner.file_sha256(runner.PROTOCOL)
    rows, files = [], {}
    for model in protocol["models"]:
        assert runner.file_sha256(model["checkpoint"]["path"]) == model["checkpoint"]["sha256"]
        assert runner.file_sha256(model["snapshot_index"]) == model["snapshot_index_sha256"]
        for specification in (item for item in protocol["grid"] if item["branch_id"] == model["branch"]["id"]):
            path = runner.fork_path(specification)
            value = runner.read(path)
            restored = restore_declared_types(value)
            original_check(restored, specification, model, protocol_sha)
            actual = runner.state_digest({key: item for key, item in restored.items() if key != "result_hash"})
            assert actual == value["result_hash"]
            files[str(path.resolve())] = runner.file_sha256(path)
            rows.append(dict(path=str(path.resolve()), sha256=files[str(path.resolve())],
                             existing_pre_json_digest=value["result_hash"], recovered_pre_json_digest=actual, status="exact_match"))
    assert len(rows) == 32
    sources = {str(path.resolve()): runner.file_sha256(path) for path in
               (Path(__file__), SCRIPT, runner.ROOT / "experiments/lifelong/evaluate.py",
                runner.ROOT / "experiments/continuity/world.py", runner.ROOT / "experiments/continuity/run.py")}
    audit = dict(format="undirected-interim-json-type-recovery-v1", status="passed", records=rows, record_count=32,
        generated_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_sha, sources=sources,
        inverse_mapping={"record.phase_codes": "String keys restored to int, from evaluate.py dict(enumerate(PHASES)).",
                         "record.regime.duration_range": "List restored to tuple, from asdict(Regime) and declared tuple field.",
                         "record.regime.gap_range": "List restored to tuple, from asdict(Regime) and declared tuple field.",
                         "record.regime.window_range": "List restored to tuple, from asdict(Regime) and declared tuple field."},
        evidence="All 32 original pre-publication result_hash values match frozen state_digest after exactly these source-declared type inversions. Current raw bytes are additionally SHA-bound; no raw file was rewritten.",
        evaluator_reruns=0, checkpoint_loads=0, scientific_fields_changed=0)
    audit_path = runner.OUTPUT / "json-type-recovery-audit.json"
    runner.atomic_json(audit_path, audit)
    runner.checked_record = checked_record
    with patch.object(runner, "evaluate", side_effect=AssertionError("postprocessing cannot rerun evaluation")), \
         patch.object(runner.native.NativeTrainer, "from_checkpoint", side_effect=AssertionError("postprocessing cannot reload a checkpoint")):
        runner.summarize()
    runner.verify_files({**files, **sources})
    binding = dict(path=str(audit_path), sha256=runner.file_sha256(audit_path), existing_pre_json_digests_exactly_verified=32)
    for path in (runner.OUTPUT / "summary.json", runner.SUMMARY):
        value = runner.read(path)
        value["raw_record_integrity"] = binding
        runner.atomic_json(path, value)
    markdown = runner.MARKDOWN.read_text(encoding="utf-8")
    markdown = markdown.replace("新增分叉：32/32。未出现的正式比较项仍待其原队列完成。", "新增分叉：32/32。四个正式评估文件的 short/long_both 比较项均已完成并绑定。")
    markdown = markdown.replace("BEGIN、内容、时机诊断属于已成熟 primary 请求；", "失败诊断列统计普通完成口径的未完成请求；严格完成率单列于上表。BEGIN、内容、时机诊断属于已成熟 primary 请求；")
    runner.MARKDOWN.write_text(markdown, encoding="utf-8")
    with runner.MARKDOWN.open("a", encoding="utf-8") as stream:
        stream.write("\n运行记录：第一次未发表分叉被补充脚本自身的时钟断言拒绝。冻结实现以完成更新数 196608 保存原世界 tick 196607；修正补充校验的这项对应关系后执行既定网格，原脚本、原协议和错误日志保留在 operational-recovery-v1。正式执行源码、快照与网格未改动，这次操作错误未记作模型失败。\n")
        stream.write("\n读取校验说明：原分叉的写前内容摘要区分 tuple/list 和整数/字符串键，JSON 会规范化这些类型。依据冻结评价源码，只恢复 phase_codes 的整数键和 regime 三个范围的 tuple 类型后，32/32 原有写前摘要逐一精确复现；原分叉文件未改写，另记录其现有字节 SHA。[恢复审计](../../../data/undirected/interim-time-factor-v1/json-type-recovery-audit.json)。未重跑分叉或重新加载快照。\n")
    print(json.dumps(dict(status="passed", original_pre_json_digests_verified=32, raw_files_unchanged=True,
                          recovery_audit=str(audit_path), summary=str(runner.SUMMARY), markdown=str(runner.MARKDOWN))), flush=True)


if __name__ == "__main__":
    main()
