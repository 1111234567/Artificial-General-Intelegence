"""Independent actual-update audit of the sole class-normalization control.

Reuses the earlier read-only auditor's plain nn.GRU/Linear, comparisons and
archive checks. The new objective is derived here directly with PyTorch CE;
the balanced training module and its loss/model/learner are never imported.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.action_split import train_audit as a
from experiments.continuity.run import state_digest

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/action-split/balanced-v1"
PARENT = ROOT / "data/action-split/train-v1"
OUT = ROOT / "data/action-split/balanced-audit-v1"
ORIGINAL_ROWS = a.independent_rows


def independent_rows(stream, start, stop):
    counts = ORIGINAL_ROWS(stream, start, stop)
    assert (stop - start) % 128 == 0
    assert all(counts[key] > 0 for key in ("mode_WAIT_rows", "mode_BEGIN_rows", "mode_FINISH_rows"))
    counts["mode_weight_sum"] = float(3 * ((stop - start) // 128))
    return counts


def reconstruct_update(folder, stream, step, device):
    before = a.load(folder / f"step{step:03d}_before.pt")
    after = a.load(folder / f"step{step:03d}_after.pt")
    saved_grad = a.load(folder / f"step{step:03d}_gradients.pt")
    assert before["format"] == after["format"] == "action-split-balanced-training-v1"
    model = a.IndependentPolicy(before, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(before["optimizer"])
    start, stop = before["stream_cursor"], before["stream_cursor"] + 128
    assert start == 512 + 128 * (step - 1)
    values = {name: torch.from_numpy(stream[name][start:stop]).to(device) for name in a.FIELDS}
    model.train()
    scores, hidden = model(values["obs"], before["hidden"].to(device))
    flat = scores.reshape(-1, 13)
    modes = values["mode_target"].reshape(-1)
    class_masks = (modes.eq(0), modes.eq(1), modes.eq(2))
    assert all(bool(mask.any()) for mask in class_masks)
    # Independent direct formula. Keep the declared floating-point summation
    # order for the required bitwise comparison of actual optimizer updates.
    wait_ce = F.cross_entropy(flat[class_masks[0], :3], modes[class_masks[0]])
    begin_mode_ce = F.cross_entropy(flat[class_masks[1], :3], modes[class_masks[1]])
    finish_mode_ce = F.cross_entropy(flat[class_masks[2], :3], modes[class_masks[2]])
    mode = (wait_ce + begin_mode_ce + finish_mode_ce) / 3.
    begin_mask, active = class_masks[1], values["active"].reshape(-1)
    begin_labels = values["begin_target"].reshape(-1)
    finish_labels = values["finish_target"].reshape(-1)
    begin = F.cross_entropy(flat[begin_mask, 3:5], begin_labels[begin_mask])
    finish = (F.cross_entropy(flat[active, 5:], finish_labels[active])
        if bool(active.any()) else flat[:, 5:].sum() * 0.)
    total = mode + begin + finish
    total.backward()
    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()}
    a.exact(gradients, saved_grad["preclip"], "balanced preclip gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    clipped = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()}
    a.exact(clipped, saved_grad["clipped"], "balanced clipped gradients")
    assert float(norm) == saved_grad["preclip_grad_norm"]
    a.exact(hidden, saved_grad["next_hidden"], "balanced gradient artifact h")
    a.exact(hidden, after["hidden"], "balanced carried preupdate h")
    optimizer.step()
    a.exact(model.state_dict(), after["model"], "balanced actual postupdate parameters")
    a.exact(optimizer.state_dict(), after["optimizer"], "balanced actual postupdate Adam")
    assert after["stream_cursor"] == stop and after["step"] == step
    core = [g for name, g in gradients.items() if name.startswith("core.")]
    if step == 1:
        if before["arm"] == "full":
            assert all(g is not None and not bool(g.any()) for g in core)
        else:
            assert all(g is None for g in core)
    if step == 2 and before["arm"] == "full":
        assert any(g is not None and bool(g.any()) for g in core)
    return dict(step=step, passed=True, tensor_comparison="bitwise exact",
        independent_mode_formula="(meanCEWAIT + meanCEBEGIN + meanCEFINISH)/3",
        perclass_mode_CE=[float(v.detach()) for v in (wait_ce, begin_mode_ce, finish_mode_ce)],
        loss={key: float(value.detach()) for key, value in
            (("total", total), ("mode", mode), ("begin", begin), ("finish", finish))},
        preclip_grad_norm=float(norm), start=start, stop=stop,
        before_sha256=a.sha(folder / f"step{step:03d}_before.pt"),
        after_sha256=a.sha(folder / f"step{step:03d}_after.pt"),
        gradients_sha256=a.sha(folder / f"step{step:03d}_gradients.pt"))


def matched_parent(seed, plan):
    results = {}
    for arm in ("frozen", "full"):
        folder, parent = TRAIN / f"seed_{seed:03d}" / arm, PARENT / f"seed_{seed:03d}" / arm
        initial, old = a.load(folder / "step000.pt"), a.load(parent / "step000.pt")
        for key in ("model", "model_config", "optimizer", "hidden", "stream_cursor", "settings", "work"):
            a.exact(initial[key], old[key], "matched original initialization/warm/Adam/zerohead: " + key)
        rows = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        old_rows = [json.loads(line) for line in (parent / "updates.jsonl").read_text().splitlines()]
        for now, old_row in zip(rows, old_rows):
            for key in ("step", "tick_start", "tick_stop", "input_sha256", "mode_rows", "mode_WAIT_rows",
                "mode_BEGIN_rows", "mode_FINISH_rows", "begin_rows", "finish_rows"):
                assert now[key] == old_row[key], (seed, arm, key)
            assert now["mode_weight_sum"] == 3.
        final, old_final = a.load(folder / "step256.pt"), a.load(parent / "step256.pt")
        for key in old_final["work"]:
            if key != "mode_weight_sum":
                assert final["work"][key] == old_final["work"][key], (seed, arm, key)
        assert final["work"]["mode_weight_sum"] == 768.
        assert initial["bindings"] == old["bindings"] == plan["input_sha256"][str(seed)]
        results[arm] = dict(passed=True, original_initial_weights_zerohead_warm_h_Adam_bit_exact=True,
            all256_input_identities_and_raw_row_counts_equal_original=True,
            all_raw_work_counters_equal_original=True, mode_reduction_denominator_only_changed_to768=True)
    return results


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    plan = a.read(TRAIN / "protocol.json")
    assert plan["format"] == "action-split-balanced-training-v1"
    assert state_digest({key: value for key, value in plan.items() if key != "protocol_hash"}) == plan["protocol_hash"]
    for section in ("source_sha256", "upstream_sha256"):
        for path, expected in plan[section].items():
            assert a.sha(path) == expected, path
    assert a.sha(ROOT / "data/action-split/streams-v1/protocol.json") == plan["data_protocol_sha256"]
    for items in plan["input_sha256"].values():
        for path, expected in items.items():
            assert a.sha(path) == expected, path
    freeze = a.read(TRAIN / "globalfreeze.json")
    assert state_digest({key: value for key, value in freeze.items() if key != "freeze_hash"}) == freeze["freeze_hash"]
    assert freeze["protocol_hash"] == plan["protocol_hash"]
    assert len(freeze["endpoints"]) == 6
    for endpoint in freeze["endpoints"].values():
        assert a.sha(endpoint["path"]) == endpoint["sha256"]
        assert len(endpoint["snapshots"]) == 3
        for snapshot in endpoint["snapshots"].values():
            assert a.sha(snapshot["path"]) == snapshot["sha256"]
            assert a.sha(snapshot["metadata_path"]) == snapshot["metadata_sha256"]
    # Bind only the read-only auditor. No training/model/loss module is called.
    saved = (a.TRAIN, a.independent_rows, a.reconstruct_update)
    a.TRAIN, a.independent_rows, a.reconstruct_update = TRAIN, independent_rows, reconstruct_update
    try:
        seeds = [dict(**a.audit_seed(seed, plan, "cuda"), matched_parent=matched_parent(seed, plan))
            for seed in (0, 1, 2)]
    finally:
        a.TRAIN, a.independent_rows, a.reconstruct_update = saved
    result = dict(format="action-split-balanced-independent-audit-v1", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), device="cuda",
        training_protocol_hash=plan["protocol_hash"], freeze_hash=freeze["freeze_hash"],
        auditor_sha256=a.sha(__file__), reused_readonly_auditor_sha256=a.sha(a.__file__),
        scope="All6 initial/warm/endpoints and256 update data identities/counts, all18 snapshots; independentfirst2 actual CUDAoptimizersteps perarm using directthreeclassmeans; originaltraining rawbudgets/init/h match; no balanced/model/loss/learner calls",
        seeds=seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "audit.json").open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(passed=True, path=str(OUT / "audit.json"), sha256=a.sha(OUT / "audit.json"))))


if __name__ == "__main__":
    main()
