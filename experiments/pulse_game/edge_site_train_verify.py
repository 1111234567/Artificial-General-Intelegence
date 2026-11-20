"""Audit receipt matching and the sampled-rule score derivative."""

from itertools import combinations

import numpy as np
import torch

from . import edge_site_train as gate
from . import edge_site_train_fit as fit


def check_ledger(g, rule):
    combined = fit.DATA / "training-heads.npz"
    if not combined.exists():
        return None
    all_events = set()
    replayed = 0
    with np.load(combined) as heads:
        assert int(heads["final_step"]) == fit.STEPS
        for arm in fit.ARMS:
            for replica, seed in enumerate(fit.SEEDS):
                root = fit.folder(replica, arm)
                with np.load(root / f"step-{fit.STEPS:04d}.npz") as row, np.load(root / "ledger.npz") as ledger:
                    np.testing.assert_array_equal(row["coefficients"], heads[arm][replica])
                    np.testing.assert_array_equal(ledger["steps"], np.arange(1, fit.STEPS + 1))
                    events = set(str(sha) for sha in ledger["event_sha256"])
                    assert len(events) == fit.STEPS and not events.intersection(all_events)
                    all_events.update(events)
                    weights = torch.nn.Parameter(torch.zeros(len(gate.FEATURES)))
                    optimizer = torch.optim.Adam([weights], lr=fit.LR)
                    stats, stamp = fit.one_update(g, rule, weights, optimizer, seed, 1, arm, fit.WORLDS)
                    np.testing.assert_array_equal(stats, ledger["metrics"][0])
                    assert stamp == str(ledger["event_sha256"][0])
                    replayed += int(ledger["routes_per_step"])
    return {"published_heads_sha256": fit.digest(combined), "unique_batch_receipt_digests": len(all_events),
            "exactly_replayed_first_batch_routes": replayed}


@torch.no_grad()
def check():
    torch.set_num_threads(1)
    g, rule = gate.noon_ticket.inherited("original")
    world, rng = fit.training_worlds(47289211, 1, "full", 128)
    prefix = gate.prepare(g, rule, world)
    group = prefix.candidates
    draw = torch.from_numpy(rng.random(group.valid.shape, dtype=np.float32))
    priority = torch.from_numpy(rng.random(group.address.shape, dtype=np.float32))
    choices = {
        "first": torch.zeros_like(group.valid, dtype=torch.long),
        "last": torch.full_like(group.valid, gate.GROUP_SIZE - 1, dtype=torch.long),
        "random": gate.choose(prefix, mode="random", draw=draw),
        "fixed": gate.choose(prefix, mode="fixed", priority=priority),
    }
    states = {key: gate.apply(prefix, action) for key, action in choices.items()}
    states["uniform"] = gate.apply(prefix, mode="uniform")
    native = gate.apply(prefix, mode="native")
    none = gate.apply(prefix, mode="no_numeric")
    before = prefix.before_b.graph
    native_change = prefix.native_b.receipt.weight_delta
    present_changes = gate._edge(native_change, group.address)[group.valid]
    assert (present_changes.sign() == group.amount[group.valid, None].sign()).all()
    torch.testing.assert_close(
        present_changes.abs().min(1).values * gate.DEFER_FRACTION, group.amount[group.valid].abs(),
        rtol=0, atol=0)
    torch.testing.assert_close(gate.realized_delta(prefix, native), native_change, rtol=0, atol=0)
    assert not gate.realized_delta(prefix, none).count_nonzero()

    budgets = {}
    for name, state in states.items():
        assert torch.equal(state.graph.present, native.graph.present)
        assert torch.equal(state.graph.bias, native.graph.bias)
        # Edges that are not common to the two supports follow the same structural receipt.
        common = before.present & native.graph.present
        assert torch.equal(state.graph.weights[~common], native.graph.weights[~common])
        change = gate.realized_delta(prefix, state)
        assert torch.equal(change.sign(), native_change.sign())
        assert (change.abs() <= native_change.abs()).all()
        flat = change.flatten(1).double()
        budgets[name] = torch.stack((flat.clamp_min(0).sum(1), flat.clamp_max(0).sum(1)), dim=1)
    maximum = torch.stack([(budgets[left] - budgets[right]).abs().amax(0)
                           for left, right in combinations(states, 2)]).amax(0)
    assert float(maximum.max()) <= 2e-6
    empty = gate.group_proposals(torch.zeros((2, 20, 20)))
    assert empty.address.shape == (2, 0, 4)
    assert empty.amount.shape == empty.valid.shape == (2, 0)

    coefficients = torch.from_numpy(rng.normal(0, 0.1, len(gate.FEATURES)).astype(np.float32))
    chosen = gate.choose(prefix, coefficients, draw=draw, gradient=True)
    advantage = torch.from_numpy(rng.normal(0, 1, len(world.a)).astype(np.float32))
    analytical = (chosen.score_gradient * advantage[:, None]).mean(0)
    with torch.enable_grad():
        differentiable = coefficients.double().requires_grad_()
        log_probability = (chosen.features.double() @ differentiable).log_softmax(-1)
        realized = log_probability.gather(-1, chosen.index[..., None]).squeeze(-1)
        contrast = ((realized * group.valid).sum(1) * advantage).mean()
        (gradient,) = torch.autograd.grad(contrast, (differentiable,))
    torch.testing.assert_close(gradient, analytical.double(), atol=2e-6, rtol=0)
    empty_coefficients = torch.zeros_like(coefficients)
    blank = gate.choose(prefix, empty_coefficients, mode="blank", draw=draw)
    shuffle = gate.choose(prefix, empty_coefficients, mode="shuffled", draw=draw, priority=priority)
    assert torch.equal(blank.index, shuffle.index)
    assert torch.equal(blank.log_prob, shuffle.log_prob)
    return {
        "status": "pass", "independent_audit_seed": 47289211, "routes": len(world.a),
        "address_groups": int(group.valid.sum()),
        "max_absolute_dose_difference": {"positive": float(maximum[0]), "negative": float(maximum[1])},
        "max_score_gradient_difference": float((gradient - analytical).abs().max()),
        "parameters_with_gradient": len(gate.FEATURES),
        "equal_support_bias_and_birth_coefficients": True, "equal_numeric_write_count_and_sign": True,
        "native_has_original_receipt": True, "no_numeric_applies_only_bias_and_structure": True,
        "relative_deferred_dose": float(group.amount.abs().sum() / native_change.abs().sum()),
        "ledger": check_ledger(g, rule),
        "verification_source_sha256": fit.digest(__file__),
        "registered_protocol_sha256": fit.register(),
    }


if __name__ == "__main__":
    result = check()
    fit.storage._write_json(fit.EVIDENCE / "training-numerical-audit.json", result)
    print(result, flush=True)
