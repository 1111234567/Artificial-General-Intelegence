"""Targeted synthetic CPU measurement and CUDA continuation references."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from experiments.continuity.run import state_digest
from experiments.credit import cycle as frozen_cycle
from experiments.credit import cycle_cpu
from experiments.credit import evaluate as frozen_evaluate
from experiments.credit.run import CreditTrainer, file_sha256
from experiments.credit.test_run import source_trainer, config


def make_fixture(root, device):
    trainer = CreditTrainer(source_trainer(device=device), config("diag_random_replay"))
    trainer.run_updates(1)
    source = root/"main.pt"
    trainer.save_checkpoint(source)
    cycles = dict(modes=["diag_random_replay"], seeds=[173], start_stage_ticks=4,
        sequence=["A","B"], segment_ticks=8, milestones=[4,8],
        regimes={"A":[[1,4],[.5,2]],"B":[[4,8],[4,12]]}, eval_conditions=["iid","long_sparse"], eval_steps=4)
    protocol = root/"protocol.json"
    protocol.write_text(json.dumps({"cycles":cycles}))
    reuse = root/"v1"
    revision = root/"revision.json"
    revision.write_text(json.dumps(dict(format="credit-cycle-measurement-revision-v2",revision_id="test",
        base_protocol_sha256=file_sha256(protocol),evaluation_device="cpu",training_device=device,
        eval_conditions=cycles["eval_conditions"],eval_steps=4,dt=.25,reuse_root=str(reuse.resolve()),
        adapter_source_sha256=file_sha256(cycle_cpu.__file__),frozen_evaluator_sha256=file_sha256(frozen_evaluate.__file__),
        frozen_cycle_sha256=file_sha256(frozen_cycle.__file__))))
    return trainer,source,protocol,revision,reuse


class CPUCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cpu_facade_delegates_every_training_object_and_preserves_state(self):
        with TemporaryDirectory() as temp:
            trainer,_,_,revision,_ = make_fixture(Path(temp),"cpu")
            facade = cycle_cpu.CPUFacade(trainer)
            for name in ("model","world","h","config"):
                self.assertIs(getattr(facade,name),getattr(trainer,name))
            before = state_digest(trainer.semantic_state())
            result = cycle_cpu.cpu_evaluate(trainer,revision,conditions=("iid",),steps=4)
            self.assertEqual(before,state_digest(trainer.semantic_state()))
            self.assertEqual(result["training_state_hash_before"],before)
            self.assertEqual(result["training_state_hash_after"],before)
            self.assertEqual(result["measurement_backend"],"cpu")

    def _prefix_reference(self,device):
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root=Path(temp)
            _,source,protocol,revision,reuse=make_fixture(root,device)
            old_branch=reuse/"seed_173"/"diag_random_replay"
            # Stop the synthetic v1 run immediately AFTER its first complete row.
            original=frozen_cycle.evaluate_trainer
            calls=0
            def stop_after_prefix(trainer,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise RuntimeError("synthetic stopped v1 worker")
                return original(trainer,**kwargs)
            frozen_cycle.evaluate_trainer=stop_after_prefix
            try:
                with self.assertRaisesRegex(RuntimeError,"synthetic stopped"):
                    frozen_cycle.run_cycles(source,old_branch,protocol,device)
            finally:
                frozen_cycle.evaluate_trainer=original
            old_hashes={str(p):file_sha256(p) for p in old_branch.rglob("*") if p.is_file()}
            output=root/"v2"/"branch"
            result=cycle_cpu.run_cpu_cycles(source,output,protocol,revision,reuse,device)
            self.assertEqual(result["retained_v1_milestones"],1)
            self.assertEqual(result["completed_milestones"],4)
            self.assertEqual(result["measurement_forks"],8)
            self.assertEqual(old_hashes,{str(p):file_sha256(p) for p in old_branch.rglob("*") if p.is_file()})
            actual=CreditTrainer.from_checkpoint(result["final_checkpoint"],device=device)
            expected=CreditTrainer.from_checkpoint(source,device=device)
            for segment,name in enumerate(("A","B")):
                duration,gap=json.loads(protocol.read_text())["cycles"]["regimes"][name]
                expected.change_regime(duration,gap,f"revisit/{segment:02d}/{name}")
                expected.run_updates(2)
            self.assertEqual(state_digest(actual.semantic_state()),state_digest(expected.semantic_state()))
            # Also compare with all original frozen CUDA/CPU callbacks, not
            # only a manual schedule with observations omitted.
            reference_folder=root/"original_callback"
            frozen_cycle.run_cycles(source,reference_folder,protocol,device)
            reference=json.loads((reference_folder/"complete.json").read_text())
            reference_trainer=CreditTrainer.from_checkpoint(reference["completed"][-1]["checkpoint"],device=device)
            self.assertEqual(state_digest(actual.semantic_state()),state_digest(reference_trainer.semantic_state()))
            # Idempotent v2 resume does not retrain or alter the semantic result.
            repeated=cycle_cpu.run_cpu_cycles(source,output,protocol,revision,reuse,device)
            self.assertEqual(repeated["final_training_state_hash"],result["final_training_state_hash"])
            self.assertIs(frozen_cycle.evaluate_trainer,original)

    def test_bootstrap_resume_matches_uninterrupted_cpu_training(self):
        self._prefix_reference("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_real_cuda_training_prefix_cpu_measurements_and_resume_exact(self):
        self._prefix_reference("cuda")


if __name__ == "__main__":
    unittest.main()
