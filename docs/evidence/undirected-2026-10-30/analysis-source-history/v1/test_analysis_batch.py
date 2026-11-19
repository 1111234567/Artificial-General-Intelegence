"""Small metadata/fake-process scheduler tests; no learner or JVP worker runs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.undirected import analysis_batch as batch
from experiments.undirected.queue import atomic_json, file_sha256


def fixture(folder):
    root = Path(folder)
    protocol = root / "protocol.json"
    branches = [dict(id=f"branch_{index}", credit=method, output=str(root / f"training_{index}"),
                     native_config=dict(credit=method)) for index, method in enumerate(("rtrl", "tbptt128", "rtrl"))]
    sources = {str((batch.ROOT / "experiments" / name).resolve()): file_sha256(batch.ROOT / "experiments" / name) for name in batch.RUNTIME}
    atomic_json(protocol, dict(format="undirected-protocol-v1", mechanical_test=True, branches=branches,
        target_ticks=2, sources=sources, python_executable=str((batch.ROOT / ".venv/Scripts/python.exe").resolve())))
    path = batch.prepare(protocol, root / "analysis", mechanical=True)
    plan = batch._load_plan(path)
    atomic_json(plan["limits"], dict(max_workers=1, min_available_bytes=0))
    return path, plan


def terminal(job, plan, *, empty=False):
    out = Path(job["branch"]["output"])
    folder = out / "live_records"
    folder.mkdir(parents=True)
    binding, manifest = folder / "binding.json", folder / "manifest.jsonl"
    binding.write_text("mechanical scheduler fixture only", encoding="utf-8")
    records = 0 if empty else 1
    if records:
        block = folder / "block_000001.pt.gz"
        block.write_bytes(b"not a numerical audit; fake-process scheduler fixture")
        manifest.write_text(json.dumps(dict(file=block.name, sha256=file_sha256(block))) + "\n", encoding="utf-8")
    else:
        manifest.write_text("", encoding="utf-8")
    checkpoint = out / ("checkpoint_last_complete.pt" if empty else "checkpoint_latest.pt")
    checkpoint.write_bytes(b"not a model; scheduler metadata fixture")
    value = dict(format="undirected-branch-terminal-v1", branch=job["branch"], protocol_sha256=plan["protocol_sha256"],
        status="numerical_failure" if empty else "success", target_ticks=2, last_tick=0 if empty else 2,
        completed=not empty, endpoint_comparable=not empty, checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=file_sha256(checkpoint),
        checkpoints={str(0 if empty else 2): dict(path=str(checkpoint.resolve()), sha256=file_sha256(checkpoint))},
        ledger=dict(records=records, last_tick=0 if empty else 2, binding_hash="a" * 64,
                    binding_path=str(binding.resolve()), binding_sha256=file_sha256(binding),
                    manifest_path=str(manifest.resolve()), manifest_sha256=file_sha256(manifest)))
    atomic_json(job["terminal"], value)
    return value


def audit_artifact(job, plan):
    end = batch._terminal(job, plan)
    records = end["ledger"]["records"]
    method = job["branch"]["credit"]
    calibrated = bool(records and method.startswith("rtrl"))
    sources = ({str((batch.ROOT / "experiments/undirected" / name).resolve()):
                file_sha256(batch.ROOT / "experiments/undirected" / name)
                for name in ("sensitivity_audit.py", "ledger.py", "model.py")} if records else {})
    fingerprints = {job["terminal"]: file_sha256(job["terminal"]), end["checkpoint_path"]: end["checkpoint_sha256"],
                    end["ledger"]["binding_path"]: end["ledger"]["binding_sha256"],
                    end["ledger"]["manifest_path"]: end["ledger"]["manifest_sha256"], **sources}
    if records:
        fingerprints.update({source: plan["file_hashes"][source] for source in plan["runtime_source_paths"]})
        for line in Path(end["ledger"]["manifest_path"]).read_text().splitlines():
            row = json.loads(line)
            fingerprints[str((Path(end["ledger"]["manifest_path"]).parent / row["file"]).resolve())] = row["sha256"]
    result = dict(format=batch.AUDIT_FORMAT, terminal_path=job["terminal"], terminal_sha256=file_sha256(job["terminal"]),
        terminal_status=end["status"], endpoint_comparable=end["endpoint_comparable"], method=method, record_count=records,
        binding_hash=end["ledger"]["binding_hash"], manifest_sha256=end["ledger"]["manifest_sha256"],
        directions=2, seed=731, atol=1e-9, rtol=1e-8, sensitivity_calibrated=calibrated,
        status="passed" if records else "not_applicable_no_successful_live_blocks",
        validation_scope=("primal_history_and_projected_saved_S" if calibrated else
                          "primal_history_only_no_saved_S_comparison" if records else "no_successful_live_history"),
        input_fingerprints=fingerprints, analysis_sources=sources, records=[{}] * records,
        work=dict(jvp_calls=2 * records, model_step_calls=2 * end["last_tick"], credit_engine_jacobian_calls=0, optimizer_steps=0, world_steps=0))
    atomic_json(job["output"], result)


class AnalysisBatchTests(unittest.TestCase):
    def launcher(self, plan, code=0, delay=0, environments=None):
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.pid = 999999999
                self.returncode = code
                self.polls = 0
                if environments is not None:
                    environments.append(kwargs["env"])
                destination = command[command.index("--output") + 1]
                job = next(job for job in plan["jobs"] if job["output"] == destination)
                if code == 0:
                    audit_artifact(job, plan)

            def poll(self):
                self.polls += 1
                return self.returncode if self.polls > delay else None
        return FakeProcess

    def test_prepare_waits_for_terminal_without_loading_torch_or_launching_a_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = fixture(folder)
            self.assertTrue(plan["analysis_only"])
            self.assertEqual(len(plan["jobs"]), 3)
            with patch.object(batch.subprocess, "Popen", side_effect=AssertionError("not ready")):
                self.assertEqual(batch.execute(path, poll_interval=.001), 2)
            state = batch._json(Path(plan["output_root"]) / "analysis-execution-v1.json")
            self.assertFalse(state["complete"])
            self.assertTrue(all(row["status"] == "waiting_terminal" for row in state["jobs"].values()))

    def test_memory_limits_resume_and_scope_preserve_all_declared_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = fixture(folder)
            for index, job in enumerate(plan["jobs"]):
                terminal(job, plan, empty=index == 2)
            atomic_json(plan["limits"], dict(max_workers=1, min_available_bytes=1000))
            with patch.object(batch, "available_memory", return_value=1), patch.object(batch.subprocess, "Popen", side_effect=AssertionError("RAM gated")):
                self.assertEqual(batch.execute(path, poll_interval=.001), 2)
            atomic_json(plan["limits"], dict(max_workers=2, min_available_bytes=0))
            with patch.object(batch.subprocess, "Popen", self.launcher(plan)):
                self.assertEqual(batch.execute(path, poll_interval=.001), 0)
            state = batch._json(Path(plan["output_root"]) / "analysis-execution-v1.json")
            self.assertTrue(state["complete"])
            self.assertTrue(state["jobs"]["branch_0"]["sensitivity_calibrated"])
            self.assertFalse(state["jobs"]["branch_1"]["sensitivity_calibrated"])
            self.assertFalse(state["jobs"]["branch_2"]["sensitivity_calibrated"])
            self.assertEqual(state["jobs"]["branch_2"]["terminal_status"], "numerical_failure")
            mapping = batch._json(Path(plan["output_root"]) / "sensitivity-results-v1.json")
            self.assertEqual(set(mapping), {"branch_0", "branch_1", "branch_2"})
            with patch.object(batch.subprocess, "Popen", side_effect=AssertionError("completed analyses restarted")):
                self.assertEqual(batch.execute(path, poll_interval=.001), 0)

    def test_execution_failure_stays_failure_and_can_resume_without_replacing_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = fixture(folder)
            terminal(plan["jobs"][0], plan)
            with patch.object(batch.subprocess, "Popen", self.launcher(plan, code=4)):
                self.assertEqual(batch.execute(path, poll_interval=.001), 1)
            self.assertFalse(batch._json(Path(plan["output_root"]) / "analysis-execution-v1.json")["complete"])
            with patch.object(batch.subprocess, "Popen", self.launcher(plan)):
                self.assertEqual(batch.execute(path, poll_interval=.001), 2)
            self.assertEqual(batch._json(Path(plan["output_root"]) / "analysis-execution-v1.json")["jobs"]["branch_0"]["status"], "complete")

    def test_foreign_source_artifact_and_false_no_s_calibration_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = fixture(folder)
            job = plan["jobs"][1]
            end = terminal(job, plan)
            audit_artifact(job, plan)
            result = batch._json(job["output"])
            result["sensitivity_calibrated"] = True
            atomic_json(job["output"], result)
            with self.assertRaisesRegex(ValueError, "scope"):
                batch.validate_result(job, plan, end)
            damaged = deepcopy(plan)
            damaged["file_hashes"][next(iter(damaged["file_hashes"]))] = "f" * 64
            atomic_json(path, damaged)
            with self.assertRaises(ValueError):
                batch.execute(path, poll_interval=.001)

    def test_waiting_terminal_cache_avoids_rehashing_large_checkpoints_each_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = fixture(folder)
            for index, job in enumerate(plan["jobs"]):
                terminal(job, plan, empty=index == 2)
            reads, environments = {}, []
            original = batch._terminal

            def counted(job, plan, **kwargs):
                if kwargs.get("verify_bytes", True):
                    reads[job["id"]] = reads.get(job["id"], 0) + 1
                return original(job, plan, **kwargs)

            with patch.object(batch, "_terminal", counted), patch.object(batch.subprocess, "Popen", self.launcher(plan, delay=3, environments=environments)):
                self.assertEqual(batch.execute(path, watch=True, poll_interval=.001), 0)
            # Dispatch + collection + once-only final validation; the fake
            # worker's metadata construction adds one. No per-poll CP hashes.
            self.assertTrue(all(count == 4 for count in reads.values()))
            self.assertEqual(len(environments), 3)
            for environment in environments:
                for name, expected in batch.WORKER_ENVIRONMENT.items():
                    self.assertEqual(environment[name], expected)

    def test_once_only_final_validation_catches_inputs_changed_after_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = fixture(folder)
            for index, job in enumerate(plan["jobs"]):
                terminal(job, plan, empty=index == 2)
            validate, collected = batch.validate_result, []

            def tamper_after_collection(job, plan, end):
                value = validate(job, plan, end)
                collected.append(job["id"])
                if len(collected) == 3:
                    first = batch._json(plan["jobs"][0]["terminal"])
                    Path(first["checkpoint_path"]).write_bytes(b"changed after its output was collected")
                return value

            with patch.object(batch, "validate_result", tamper_after_collection), patch.object(batch.subprocess, "Popen", self.launcher(plan)):
                with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
                    batch.execute(path, poll_interval=.001)
            state = batch._json(Path(plan["output_root"]) / "analysis-execution-v1.json")
            self.assertFalse(state["complete"])
            self.assertIn("coordinator_error", state)


if __name__ == "__main__":
    unittest.main()
