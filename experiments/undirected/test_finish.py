"""Operations failure boundaries; no learner or formal behavioral data is used."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.undirected import finish, queue


class FinishTests(unittest.TestCase):
    def test_reuse_of_verified_audits_rechecks_the_complete_bound_inventory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, plan_path = root / "protocol.json", root / "plan.json"
            result = root / "sample-audit.json"
            files = [result, root / "sensitivity-results-v1.json", root / "analysis-execution-v1.json"]
            for path in [protocol, plan_path, *files]:
                path.write_text("{}", encoding="utf-8")
            plan = dict(output_root=str(root), jobs=[dict(output=str(result))])
            artifacts = {str(path): queue.file_sha256(path) for path in files}
            stage = dict(status="verified", branch_count=1, artifacts=artifacts)
            binding = dict(protocol=str(protocol), protocol_sha256=queue.file_sha256(protocol),
                           analysis_plan=str(plan_path), analysis_plan_sha256=queue.file_sha256(plan_path))
            prior = root / "old-status.json"
            def write():
                prior.write_text(json.dumps(dict(binding=binding, stages=dict(analysis=stage))), encoding="utf-8")
            write()
            self.assertEqual(finish.reuse_analysis(prior, protocol, plan_path, plan)["status"], "revalidated")
            result.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaises(ValueError):
                finish.reuse_analysis(prior, protocol, plan_path, plan)
            del artifacts[str(result)]
            write()
            with self.assertRaisesRegex(ValueError, "inventory"):
                finish.reuse_analysis(prior, protocol, plan_path, plan)

    def test_dead_producer_cannot_leave_watch_waiting_or_claim_completion(self):
        incomplete = {kind: dict(complete=False) for kind in queue.KINDS}
        dead = SimpleNamespace(alive=lambda: False)
        with self.assertRaisesRegex(RuntimeError, "producer died"):
            finish.require_producer(incomplete, dead, lambda: incomplete)
        complete = {kind: dict(complete=True) for kind in queue.KINDS}
        self.assertTrue(finish.require_producer(incomplete, dead, lambda: complete))
        self.assertTrue(finish.require_producer(complete, None, lambda: self.fail("unnecessary reread")))
        self.assertFalse(finish.require_producer(incomplete, SimpleNamespace(alive=lambda: True), lambda: self.fail("live producer")))

    def test_declared_complete_requires_all_jobs_and_valid_byte_bindings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"mechanical artifact, not a behavioral result")
            plans, states = {}, {}
            for kind in queue.KINDS:
                path = root / f"{kind}-plan.json"
                plan = dict(kind=kind)
                jobs = [] if kind == "baseline" else [dict(id="mechanical", command=["unused"], outputs=[str(artifact)])]
                plans[kind] = path, plan, jobs
                row = dict(returncode=0, terminal_status="success", command_digest=queue.digest(["unused"]),
                           output_hashes={str(artifact): queue.file_sha256(artifact)})
                states[kind] = dict(format=queue.QUEUE_FORMAT, kind=kind, plan=str(path), plan_digest=queue.digest(plan),
                    jobs={} if not jobs else {"mechanical": row}, complete=True,
                    terminal_counts=dict(success=len(jobs), numerical_failure=0))

            def write_states(values):
                for kind, value in values.items():
                    (root / f"{kind}-execution.json").write_text(json.dumps(value), encoding="utf-8")

            write_states(states)
            with patch.object(queue, "terminal", return_value=dict(status="success")):
                self.assertTrue(all(row["complete"] for row in finish.queue_state(root, plans, verify_outputs=True).values()))
                artifact.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "output/command"):
                    finish.queue_state(root, plans, verify_outputs=True)
            damaged = deepcopy(states)
            damaged["training"]["jobs"] = {}
            write_states(damaged)
            with self.assertRaisesRegex(ValueError, "falsely claims"):
                finish.queue_state(root, plans)
            damaged = deepcopy(states)
            damaged["evaluation"].update(complete=False, coordinator_error="execution interrupted")
            write_states(damaged)
            with self.assertRaisesRegex(RuntimeError, "producer execution failed"):
                finish.queue_state(root, plans)

    @unittest.skipUnless(os.name == "nt", "Windows kernel file locking")
    def test_existing_lock_file_is_reusable_but_concurrent_writer_is_refused(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "finish.lock"
            with finish.sole_writer(path):
                with self.assertRaisesRegex(RuntimeError, "another finish coordinator"):
                    with finish.sole_writer(path):
                        self.fail("duplicate coordinator entered")
            with finish.sole_writer(path):
                pass


if __name__ == "__main__":
    unittest.main()
