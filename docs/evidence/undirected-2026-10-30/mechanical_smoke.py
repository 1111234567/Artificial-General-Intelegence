"""Full-size implementation smoke; no task-score selection."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def worker(spec):
    import torch
    from experiments.undirected.run import NativeConfig, NativeTrainer, train
    from experiments.undirected.ledger import LiveLedger
    from experiments.continuity.run import state_digest
    torch.set_num_threads(1)
    label, changes = spec
    config = NativeConfig(seed=91, device="cpu", **changes)
    folder = ROOT / "data/undirected/smoke-v1" / label
    if folder.exists():
        raise FileExistsError(folder)
    binding = {"purpose": "mechanical-full-size-smoke", "case": label}
    ledger = LiveLedger(folder / "live_records", binding)
    # The writer creates its own directory; the native new-run guard correctly
    # rejects nonempty output, so its output lives in a separate child.
    trainer = train(config, folder / "training", stop_at=4096, on_update=ledger)
    saved = trainer.state_dict()
    trainer.train_update()
    expected = state_digest(trainer.semantic_state())
    restored = NativeTrainer.from_state_dict(saved, device="cpu")
    restored.train_update()
    assert state_digest(restored.semantic_state()) == expected
    matrix = restored.model.recurrent_matrix().detach()
    if config.mode == "undirected":
        assert torch.equal(matrix, matrix.T)
    assert torch.isfinite(restored.h).all()
    if restored.credit.sensitivity is not None:
        assert torch.isfinite(restored.credit.sensitivity).all()
    return {"case": label, "config": asdict(config), "saved_ticks": saved["ticks"],
        "resumed_tick": restored.ticks, "exact_next_update": True,
        "credit_path_ticks": restored.credit.last_metrics["credit_path_ticks"],
        "credit_hard_cap_ticks": restored.credit.last_metrics["credit_hard_cap_ticks"],
        "primal_error_max": max(point["credit"]["primal_max_abs_error"] for point in restored.trace),
        "sensitivity_finite": True, "matrix_antisymmetry_max": float((matrix-matrix.T).abs().max()),
        "source_hashes": saved["source_hashes"], "dependency_hashes": saved["dependency_hashes"],
        "purpose": "Numerical continuity/ledger/structure smoke only; behavioral scores not inspected or used."}


if __name__ == "__main__":
    cases = [("ports_shared_rtrl", {}),
        ("ports_shared_rtrl_both", {"structure": "both"}),
        ("ports_untied_rtrl", {"mode": "untied"}),
        ("core_shared_recorded", {"interface": "core", "credit": "recorded512"}),
        ("ports_shared_rtrl_wide", {"course": "wide"})]
    with ProcessPoolExecutor(max_workers=5) as pool:
        values = []
        for value in pool.map(worker, cases):
            values.append(value)
            print(json.dumps({k:value[k] for k in ("case","saved_ticks","resumed_tick","credit_path_ticks","exact_next_update")}),flush=True)
    output = Path(__file__).with_name("mechanical-smoke-v1.json")
    output.write_text(json.dumps({"format":"undirected-mechanical-smoke-v1","cases":values},indent=2),encoding="utf-8")
