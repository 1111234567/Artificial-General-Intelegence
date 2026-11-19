"""Final-source warmup, pending recovery and independent full-history JVP smoke."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    import torch
    from experiments.undirected.run import NativeConfig, NativeTrainer, train
    from experiments.undirected.ledger import LiveLedger, file_sha256
    from experiments.undirected.sensitivity_audit import audit_ledger
    from experiments.continuity.run import state_digest
    torch.set_num_threads(1)
    folder = ROOT / "data/undirected/smoke-v3"
    if folder.exists():
        raise FileExistsError(folder)
    config = NativeConfig(seed=91, interface="ports", mode="undirected", credit="rtrl",
                          structure="both", course="warmup", device="cpu")
    binding = {"purpose": "final-source mechanical smoke", "case": "warmup_rtrl_both"}
    writer = LiveLedger(folder / "live_records", binding)
    trainer = train(config, folder / "training", stop_at=4096, on_update=writer)
    saved = trainer.state_dict()
    saved_digest = state_digest(saved)
    trainer.train_update()
    expected = state_digest(trainer.semantic_state())
    restored = NativeTrainer.from_state_dict(saved)
    restored.train_update()
    assert state_digest(restored.semantic_state()) == expected
    assert state_digest(saved) == saved_digest
    assert restored.credit.last_metrics["credit_path_ticks"] == 4224
    checkpoint = folder / "training/checkpoint_latest.pt"
    calibration = audit_ledger(folder / "live_records", {"4096":{
        "path":str(checkpoint),"sha256":file_sha256(checkpoint)}}, expected_binding=binding)
    result = dict(format="undirected-final-source-smoke-v3", saved_ticks=4096,
        resumed_ticks=4224, exact_next_update=True, input_state_immutable=True,
        snapshot_sha256=file_sha256(checkpoint), calibration=calibration,
        source_hashes=saved["source_hashes"], dependency_hashes=saved["dependency_hashes"],
        note="Production shape, genuine warmup stream, no behavioral-score selection; no runtime change after these sources.")
    output = Path(__file__).with_name("mechanical-smoke-v3.json")
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"saved_ticks":4096,"resumed_ticks":4224,"exact_next_update":True,
                      "input_state_immutable":True,"calibration_keys":list(calibration)}))
