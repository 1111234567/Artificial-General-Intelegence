"""Independent JVP checks of changed-parameter histories and saved S."""
from copy import deepcopy
import gzip
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.continuity.run import state_digest
from experiments.undirected.credit import CreditConfig, TemporalCreditEngine, snapshot_parameters
from experiments.undirected.model import UndirectedModel, PortGraphModel
from experiments.undirected import ledger
from experiments.undirected import run as native
from experiments.undirected import sensitivity_audit as audit


def _tiny(interface="ports", mode="undirected"):
    if interface == "ports":
        model = PortGraphModel(input_size=3, n_jobs=1, n_actions=2, width=6,
                               input_scale=(1., 1., 1.), mode=mode).double()
    else:
        model = UndirectedModel(input_size=3, n_jobs=1, n_actions=2, width=6,
                                input_scale=(1., 1., 1.), initialization_seed=821, mode=mode).double()
    generator = torch.Generator().manual_seed(823)
    with torch.no_grad():
        model.edge_shared.copy_(torch.randn(model.edge_shared.shape, dtype=torch.double, generator=generator) * .15)
        if mode == "untied":
            model.edge_skew.copy_(torch.randn(model.edge_skew.shape, dtype=torch.double, generator=generator) * .03)
    return model


def _fixture(folder, *, method="rtrl", interface="ports", mode="undirected", binding=None):
    folder = Path(folder)
    binding = dict(branch_id="mechanical-jvp", protocol_sha256="a" * 64) if binding is None else binding
    writer = ledger.LiveLedger(folder / "live_records", binding)
    model = _tiny(interface, mode)
    engine = TemporalCreditEngine(model, CreditConfig(method, update_ticks=2))
    source_hashes, dependency_hashes = native.source_hashes(), native.source_hashes(native.dependency_snapshots())
    h, trace = model.initial_state(2), []
    checkpoints = {}

    def checkpoint(tick, age):
        path = folder / f"checkpoint_{tick:06d}.pt"
        torch.save(dict(ticks=tick, updates=age, hidden=h.clone(), config=dict(credit=method),
                        model_config=model.config(), credit=engine.state_dict()), path)
        checkpoints[tick] = path

    checkpoint(0, 0)
    for index in range(3):
        generator = torch.Generator().manual_seed(827 + index)
        observations = torch.randn(2, 2, 3, generator=generator, dtype=torch.double) * .1
        targets = torch.randint(2, (2, 2, 1), generator=generator)
        elapsed, outputs = (.25, .375), []
        start_h, parameters = h.clone(), snapshot_parameters(model)
        with torch.no_grad():
            for observation, dt in zip(observations, elapsed):
                output, h = model(observation, h, dt)
                outputs.append(output)
        actions = torch.stack(outputs).argmax(-1)
        contexts = torch.full_like(targets, index % 4)
        chunk = dict(observations=observations, targets=targets, elapsed=elapsed, start_h=start_h, end_h=h.clone(),
                     start_tick=2 * index, end_tick=2 * (index + 1), actions=actions,
                     family_context=contexts, family_counts={family: 0 for family in range(4)})
        _, metrics = engine.gradients(chunk, parameters)
        causal = {name: chunk[name] for name in ("observations", "targets", "elapsed", "start_tick", "end_tick", "family_context")}
        trace.append(dict(tick=chunk["end_tick"], update=index + 1, action_chunk_hash=state_digest(actions),
                          data_chunk_hash=state_digest(causal), live_hidden_hash=state_digest(h), credit=metrics))
        trainer = SimpleNamespace(model=model, credit=engine, last_chunk=chunk, trace=trace, h=h,
                                  updates=index + 1, ticks=chunk["end_tick"], source_hashes=source_hashes,
                                  dependency_hashes=dependency_hashes)
        writer(trainer)
        if index == 0:
            # Re-open the manifest between real blocks, as a resumed writer does.
            writer = ledger.LiveLedger(folder / "live_records", binding)
        with torch.no_grad():
            if index == 0:
                model.edge_shared[0] = 0.  # Physical retirement is exogenous.
            elif index == 1:
                model.edge_shared[0] = .12  # Later birth does not reset tangents.
            if interface == "core":
                model.encoder.weight.add_(.005)
            if mode == "untied":
                model.edge_skew.add_(.002)
    checkpoint(6, 3)
    return binding, checkpoints, engine


class SensitivityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_changed_parameter_blocks_match_saved_full_S_for_both_interfaces(self):
        for interface in ("ports", "core"):
            for mode in ("undirected", "untied"):
                with self.subTest(interface=interface, mode=mode), tempfile.TemporaryDirectory() as folder:
                    binding, checkpoints, _ = _fixture(folder, interface=interface, mode=mode)
                    with patch.object(TemporalCreditEngine, "_block_jacobians", side_effect=AssertionError("audit reused RTRL")), \
                         patch.object(TemporalCreditEngine, "gradients", side_effect=AssertionError("audit reused training gradients")):
                        result = audit.audit_ledger(Path(folder) / "live_records", checkpoints, expected_binding=binding)
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(result["record_count"], 3)
                    self.assertEqual(result["sensitivity_checked_ticks"], [6])
                    self.assertTrue(result["sensitivity_calibrated"])
                    self.assertEqual(result["directions"], 2)
                    self.assertEqual(result["work"]["jvp_calls"], 6)
                    self.assertEqual(result["work"]["credit_engine_jacobian_calls"], 0)
                    self.assertEqual(result["work"]["optimizer_steps"], 0)
                    self.assertTrue(all(not row["tangent_reset"] for row in result["records"]))

    def test_reset_control_resets_each_block_and_other_methods_explicitly_skip_S(self):
        for method in ("rtrl_reset", "recorded512", "tbptt128"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                binding, checkpoints, _ = _fixture(folder, method=method)
                result = audit.audit_ledger(Path(folder) / "live_records", checkpoints, expected_binding=binding)
                if method == "rtrl_reset":
                    self.assertEqual(result["sensitivity_checked_ticks"], [6])
                    self.assertTrue(all(row["tangent_reset"] for row in result["records"]))
                else:
                    self.assertEqual(result["sensitivity_checked_ticks"], [])
                    self.assertEqual(result["sensitivity_skipped_ticks"], [6])
                    self.assertFalse(result["sensitivity_calibrated"])
                    self.assertEqual(result["validation_scope"], "primal_history_only_no_saved_S_comparison")

    def test_private_audit_rng_and_repeated_results_do_not_modify_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, checkpoints, _ = _fixture(folder)
            torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
            first = audit.audit_ledger(Path(folder) / "live_records", checkpoints, expected_binding=binding)
            second = audit.audit_ledger(Path(folder) / "live_records", checkpoints, expected_binding=binding)
            self.assertEqual(first, second)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            self.assertEqual(state_digest(np.random.get_state()), state_digest(numpy_rng))
            self.assertEqual(first["atol"], 1e-9)
            self.assertEqual(first["rtol"], 1e-8)

    def test_tampered_saved_S_is_rejected_by_the_independent_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, checkpoints, _ = _fixture(folder)
            value = torch.load(checkpoints[6], map_location="cpu", weights_only=False)
            value["credit"]["sensitivity"][0, 0, 0] += .1
            torch.save(value, checkpoints[6])
            # Path-only references deliberately exercise numerical comparison,
            # rather than merely the externally supplied checkpoint byte hash.
            with self.assertRaisesRegex(ValueError, "saved S projection"):
                audit.audit_ledger(Path(folder) / "live_records", checkpoints, expected_binding=binding)

    def test_tampered_data_or_canonical_checkpoint_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, checkpoints, _ = _fixture(folder)
            canonical = {tick: dict(path=str(path), sha256=ledger.file_sha256(path)) for tick, path in checkpoints.items()}
            audit.audit_ledger(Path(folder) / "live_records", canonical, expected_binding=binding)
            with checkpoints[6].open("ab") as stream:
                stream.write(b"foreign")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                audit.audit_ledger(Path(folder) / "live_records", canonical, expected_binding=binding)
        with tempfile.TemporaryDirectory() as folder:
            binding, checkpoints, _ = _fixture(folder)
            ledger_folder = Path(folder) / "live_records"
            inventory = ledger.read_ledger(ledger_folder)
            path = ledger_folder / inventory["entries"][1]["file"]
            value = torch.load(io.BytesIO(gzip.decompress(path.read_bytes())), weights_only=False)
            value["record"]["observations"].add_(.2)
            stream = io.BytesIO()
            torch.save(value, stream)
            path.write_bytes(gzip.compress(stream.getvalue(), mtime=0))
            with self.assertRaises(ValueError):
                audit.audit_ledger(ledger_folder, checkpoints, expected_binding=binding)

    def test_wrong_method_layout_missing_boundary_and_foreign_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, checkpoints, _ = _fixture(folder)
            ledger_folder = Path(folder) / "live_records"
            with self.assertRaises(ValueError):
                audit.audit_ledger(ledger_folder, checkpoints, expected_binding={"branch_id": "foreign"})
            with self.assertRaisesRegex(ValueError, "boundary"):
                audit.audit_ledger(ledger_folder, {3: checkpoints[6]}, expected_binding=binding)
            value = torch.load(checkpoints[6], map_location="cpu", weights_only=False)
            value["config"]["credit"] = "rtrl_reset"
            torch.save(value, checkpoints[6])
            with self.assertRaisesRegex(ValueError, "method"):
                audit.audit_ledger(ledger_folder, checkpoints, expected_binding=binding)

    def test_norms_and_tolerances_use_safe_scales(self):
        huge = torch.full((4,), 1e308, dtype=torch.float64)
        summary = audit._norm_summary(huge)
        self.assertIsNone(summary["l2"])
        self.assertTrue(math.isfinite(summary["log10_l2"]))
        comparison = audit._compare(huge * (1. + 1e-10), huge, "large finite values")
        self.assertLess(comparison["relative_l2_error"], 1e-8)

    def test_terminal_cli_binds_the_ledger_summary_and_method(self):
        with tempfile.TemporaryDirectory() as folder:
            branch = dict(id="mechanical-jvp", credit="rtrl", native_config=dict(credit="rtrl", chunk_ticks=2))
            binding = dict(protocol_sha256="a" * 64, branch=branch,
                           recording_convention="committed-live-block-pre-parameter-trajectory-v1")
            _, checkpoints, _ = _fixture(folder, binding=binding)
            writer = ledger.LiveLedger(Path(folder) / "live_records", binding)
            value = dict(format="undirected-branch-terminal-v1", branch=branch, protocol_sha256="a" * 64,
                status="success", completed=True, endpoint_comparable=True, last_tick=6, target_ticks=6,
                checkpoint_path=str(checkpoints[6]), checkpoint_sha256=ledger.file_sha256(checkpoints[6]),
                checkpoints={str(tick): dict(path=str(path), sha256=ledger.file_sha256(path)) for tick, path in checkpoints.items()},
                ledger=writer.summary(updates=3, require_exact=True))
            terminal = Path(folder) / "terminal.json"
            terminal.write_text(json.dumps(value), encoding="utf-8")
            result = audit.audit_terminal(terminal)
            self.assertTrue(result["expected_binding_verified"])
            self.assertTrue(result["sensitivity_calibrated"])
            output = Path(folder) / "audit.json"
            audit.main(["--terminal", str(terminal), "--output", str(output)])
            self.assertEqual(json.loads(output.read_text())["terminal_sha256"], ledger.file_sha256(terminal))
            value["ledger"]["manifest_sha256"] = "f" * 64
            terminal.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary fingerprint"):
                audit.audit_terminal(terminal)


if __name__ == "__main__":
    unittest.main()
