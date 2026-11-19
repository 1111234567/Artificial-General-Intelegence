"""Marginalize native birth/retirement sampling around a legacy bootstrap contact."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import self_write_core as core
from .self_write_core_bootstrap import calculate, pair_examples, tensor, training_trial


EVIDENCE = Path("docs/evidence/self-edit-dose-2026-11-19/gradient")
ADDRESSES = np.r_[-1, np.flatnonzero(~core.DIAGONAL)]


def branch_graphs(graph):
    """Use native apply once on all distinct admissible command pairs."""
    visible = graph.mask[0].ravel()
    born, retired = np.broadcast_arrays(ADDRESSES[:, None], ADDRESSES[None, :])
    accepted_birth = np.where((born >= 0) & ~visible[born], born, -1)
    attempted_retire = np.where((retired >= 0) & (visible[retired] | (retired == accepted_birth)), retired, -1)
    coordinates = np.stack((accepted_birth.ravel(), attempted_retire.ravel()), axis=1)
    attempts, compress_attempt = np.unique(coordinates, axis=0, return_inverse=True)

    replicas = core.repeated_graph(graph, len(attempts))
    commands = core.Proposal(np.zeros_like(replicas.w), np.zeros_like(replicas.b), attempts[:, 0],
                             np.ones(len(attempts), dtype=graph.w.dtype), attempts[:, 1], None, None)
    actual, receipt = core.apply(replicas, commands)
    np.testing.assert_array_equal(receipt.birth_status == 1, attempts[:, 0] >= 0)
    applied_retire = np.where(receipt.retire_status == 1, attempts[:, 1], -1)
    applied_birth = attempts[:, 0]
    cancelled = (applied_birth >= 0) & (applied_birth == applied_retire)
    canonical = np.stack((np.where(cancelled, -1, applied_birth), np.where(cancelled, -1, applied_retire)), axis=1)
    effective, first, compress_result = np.unique(canonical, axis=0, return_index=True, return_inverse=True)
    inverse = compress_result[compress_attempt].reshape(len(ADDRESSES), len(ADDRESSES))
    blocked = (receipt.retire_status == 4)[compress_attempt].reshape(inverse.shape)
    np.testing.assert_array_equal(actual.mask, actual.mask[first][compress_result])
    return effective, actual.mask[first], inverse, blocked


def legacy_contact(graph, example):
    left, right, _, taste_by_side = example
    choice = core.advance(graph, core.rest(graph), core.sense(left, right), core.MOTOR_TICKS)
    side = np.where(core.motor(choice) > 0, 1, -1)
    taste = taste_by_side[np.arange(len(side)), (side > 0).astype(int)]
    pre, post = core.contact_states(graph, choice, left, right, side, taste)
    local, gate, edge, bias = core.write_fields(pre, post, *core.view_arrays(graph, "own", None))
    return core.motor(choice).item(), side.item(), taste.item(), pre, post, local, gate, edge, bias


def probabilities(post, local, graph):
    w, b, mask = core.view_arrays(graph, "own", None)
    readback = post[:, 13, None, None] * (np.tanh(w) + .25 * mask + .15 * np.tanh(b[:, :, None]))
    scores = (3 * local + readback).ravel()[ADDRESSES[1:]]
    gates = np.clip(post[0, [14, 15]], 0, 1) * np.array([.16, .20])
    result = []
    for logits, gate in zip((scores, -scores), gates, strict=True):
        densities = np.exp(logits - logits.max())
        result.append(np.r_[1 - gate, gate * (densities / densities.sum())])
    return np.outer(*result), gates


def losses_after_writes(w, b, masks, example):
    left, right, target, _ = example
    graph = core.Graph(w, b, masks)
    rest = core.rest(graph)
    margins = []
    for sign, pair in ((1, (left, right)), (-1, (right, left))):
        activity = core.advance(graph, rest, core.sense(*pair), core.MOTOR_TICKS)
        margins.append(-5 * target * sign * core.motor(activity))
    return np.logaddexp(0, margins).mean(axis=0)


def evaluate(graph, example, mapping):
    operations, masks, inverse, blocked = mapping
    motor, side, taste, pre, post, local, gate, edge, bias = legacy_contact(graph, example)
    distribution, gates = probabilities(post, local, graph)
    edge_allowed = graph.mask & (np.abs(edge) >= core.EPSILON)
    bias_allowed = np.abs(bias) >= core.EPSILON
    raw_w = graph.w + edge * edge_allowed
    raw_b = graph.b + bias * bias_allowed
    w = np.broadcast_to(np.clip(raw_w, -core.LIMIT, core.LIMIT), masks.shape).copy()
    b = np.clip(raw_b, -core.LIMIT, core.LIMIT)
    w *= masks
    alive = np.flatnonzero(operations[:, 0] >= 0)
    row, col = np.divmod(operations[alive, 0], core.N)
    w[alive, row, col] = core.BIRTH_STEP * np.tanh(2 * (local[0, row, col] + gate[0]))
    losses = losses_after_writes(w, b, masks, example)
    p = np.bincount(inverse.ravel(), weights=distribution.ravel(), minlength=len(operations))
    np.testing.assert_allclose(p.sum(), 1, atol=5e-14, rtol=0)
    return {"loss": losses, "probability": p, "expected": p @ losses,
            "motor": motor, "side": side, "taste": taste, "gates": gates,
            "post_gates": post[0, [14, 15]], "edges_allowed": edge_allowed, "bias_allowed": bias_allowed,
            "edges_clipped": np.abs(raw_w) > core.LIMIT, "bias_clipped": np.abs(raw_b) > core.LIMIT,
            "blocked_mass": distribution[blocked].sum()}


def score_derivative(graph, example, raw_losses):
    """Differentiate only the marginalized sampling probabilities with branch losses fixed."""
    left, right, _, taste_by_side = example
    w = tensor(graph.w.copy()).requires_grad_()
    b = tensor(graph.b.copy()).requires_grad_()
    mask = tensor(graph.mask)
    h = torch.zeros_like(b)
    choice = calculate(w, b, h, tensor(core.sense(left, right)), core.MOTOR_TICKS)
    side = np.where(core.motor(choice.detach().numpy()) > 0, 1, -1)
    taste = taste_by_side[np.arange(len(side)), (side > 0).astype(int)]
    pre = calculate(w, b, choice, tensor(core.sense(left, right, side)), core.CONTACT_TICKS)
    post = calculate(w, b, pre, tensor(core.sense(left, right, side, taste)), core.TASTE_TICKS)
    local, _, _, _ = core.write_fields(pre, post, w, b, mask, torch.tanh)
    readback = post[:, 13, None, None] * (w.tanh() + .25 * mask + .15 * b[:, :, None].tanh())
    scores = (3 * local + readback).flatten()[tensor(ADDRESSES[1:])]
    birth = .16 * post[0, 14].clamp(0, 1)
    retire = .20 * post[0, 15].clamp(0, 1)
    def marginal(birth_gate, retire_gate, address_scores):
        birth_prob = torch.cat(((1 - birth_gate)[None], birth_gate * address_scores.softmax(dim=0)))
        retire_prob = torch.cat(((1 - retire_gate)[None], retire_gate * (-address_scores).softmax(dim=0)))
        return birth_prob @ tensor(raw_losses) @ retire_prob

    expected = marginal(birth, retire, scores)
    gate_only = marginal(birth, retire, scores.detach())
    address_only = marginal(birth.detach(), retire.detach(), scores)
    components = {"total": expected, "gate": gate_only, "address": address_only}
    gradients = {
        name: tuple(gradient.detach().numpy() for gradient in torch.autograd.grad(
            loss, (w, b), retain_graph=(name != "address")))
        for name, loss in components.items()
    }
    return expected.item(), gradients


def key_record(graph, example, key, mapping):
    operations, _, _, _ = mapping
    left, right, _, taste_by_side = example
    native, _, side, taste, edit, receipt = core.step(
        graph, core.rest(graph), left, right, taste_by_side, key, view="own",
    )
    retire = int(edit.retire[0]) if receipt.retire_status[0] == 1 else -1
    birth = int(edit.birth[0]) if receipt.birth_status[0] == 1 else -1
    if birth == retire:
        birth = retire = -1
    branch = np.flatnonzero(np.all(operations == [birth, retire], axis=1)).item()
    measured = losses_after_writes(native.w, native.b, native.mask, example).item()
    return {"key": key, "applied_birth": birth, "applied_retire": retire, "branch": branch,
            "chosen_side": int(side[0]), "taste": float(taste[0]), "native_loss": measured}


def keyed_autograd(graph, example, key):
    """Use the original training_trial without substituting a sampling policy."""
    w = tensor(graph.w[0].copy()).requires_grad_()
    b = tensor(graph.b[0].copy()).requires_grad_()
    loss, _ = training_trial(w, b, graph.mask[0].copy(), example, key, "own")
    dw, db = torch.autograd.grad(loss, (w, b))
    return loss.item(), (dw[None].numpy(), db[None].numpy())


def direction_vectors(graph, seed):
    generator = np.random.default_rng(seed)
    writer = np.zeros_like(graph.w)
    writer[:, core.WRITER] = generator.standard_normal(writer[:, core.WRITER].shape)
    writer *= graph.mask
    w = writer / np.linalg.norm(writer)
    absent_w = np.zeros_like(w)
    absent_b = np.zeros_like(graph.b)
    b14 = absent_b.copy()
    b15 = absent_b.copy()
    b14[0, 14] = b15[0, 15] = 1
    return {"existing_writer_inputs": (w, absent_b), "bias_14": (absent_w, b14), "bias_15": (absent_w, b15)}


def dot(gradients, vectors):
    return sum(float(np.sum(x * y)) for x, y in zip(gradients, vectors, strict=True))


PIECEWISE_BRANCH_FIELDS = ("side", "edges_allowed", "bias_allowed", "edges_clipped", "bias_clipped")


def same_local_regime(base, *perturbed):
    for label in PIECEWISE_BRANCH_FIELDS:
        if any(np.any(point[label] != base[label]) for point in perturbed):
            return False
    regimes = [np.searchsorted([0, 1], entry["post_gates"]) for entry in (base, *perturbed)]
    return all(np.array_equal(regimes[0], point) for point in regimes[1:])


def run_case(spec, plan):
    checkpoint = spec.get("checkpoint")
    if checkpoint is None:
        reviewed = core.spawn_graph(spec["spawn_root"])
        checkpoint_hash = None
    else:
        path = Path(checkpoint)
        checkpoint_hash = sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as stored:
            reviewed = core.Graph(stored["w"], stored["b"], stored["mask"])
    example = pair_examples(np.random.default_rng(spec["example_root"]), 1)
    graph = core.Graph(reviewed.w.astype(np.float64), reviewed.b.astype(np.float64), reviewed.mask.copy())
    mapping = branch_graphs(reviewed)
    outcomes, _, inverse, _ = mapping
    initial = evaluate(graph, example, mapping)
    probability_loss, probability_grad = score_derivative(graph, example, initial["loss"][inverse])
    np.testing.assert_allclose(initial["expected"], probability_loss, atol=1e-12, rtol=0)

    samples = [key_record(reviewed, example, spec["native_key"], mapping)]
    pending = {op for op in ("applied_birth", "applied_retire") if samples[0][op] < 0}
    for key in range(spec["native_key"] + 1, spec["native_key"] + 257):
        if not pending:
            break
        additional = key_record(reviewed, example, key, mapping)
        observed = {op for op in pending if additional[op] >= 0}
        if observed:
            samples.append(additional)
            pending.difference_update(observed)
    torch_gradients = []
    for sample in samples:
        loss, gradients = keyed_autograd(reviewed, example, sample["key"])
        sample["historical_autograd_loss"] = loss
        sample["enumerated_branch_loss"] = float(initial["loss"][sample["branch"]])
        np.testing.assert_allclose([sample["native_loss"], sample["enumerated_branch_loss"]], loss, atol=2e-6, rtol=0)
        torch_gradients.append(gradients)

    report = {"spec": spec, "checkpoint_sha256": checkpoint_hash, "expected": float(initial["expected"]),
              "raw_pairs": int(inverse.size), "distinct_applied_graphs": len(outcomes),
              "positive_applied_probabilities": int(np.count_nonzero(initial["probability"])),
              "birth_retire_trigger_probabilities": initial["gates"].tolist(),
              "birth_retire_post_activities": initial["post_gates"].tolist(),
              "retire_rejected_by_scc_probability": float(initial["blocked_mass"]),
              "base_side": initial["side"], "base_motor": initial["motor"], "base_taste": initial["taste"],
              "samples": samples, "directions": {}}
    for name, vector in direction_vectors(graph, plan["direction_root"]).items():
        dw, db = vector
        branch_direction = [dot(gradient, vector) for gradient in torch_gradients]
        probability_components = {key: dot(gradient, vector) for key, gradient in probability_grad.items()}
        score = probability_components["total"]
        np.testing.assert_allclose(sum(probability_components[part] for part in ("gate", "address")),
                                   score, atol=2e-12, rtol=0)
        finite_steps = []
        for step in plan["central_steps"]:
            lower = evaluate(core.Graph(graph.w - step * dw, graph.b - step * db, graph.mask), example, mapping)
            upper = evaluate(core.Graph(graph.w + step * dw, graph.b + step * db, graph.mask), example, mapping)
            d_loss = (upper["loss"] - lower["loss"]) / (2 * step)
            d_probability = (upper["probability"] - lower["probability"]) / (2 * step)
            total = (upper["expected"] - lower["expected"]) / (2 * step)
            path = initial["probability"] @ d_loss
            selection = d_probability @ initial["loss"]
            observed = [float(d_loss[sample["branch"]]) for sample in samples]
            same_regime = same_local_regime(initial, lower, upper)
            if same_regime:
                np.testing.assert_allclose(total, path + selection, atol=2e-7, rtol=1e-5)
                np.testing.assert_allclose(selection, score, atol=2e-7, rtol=1e-5)
                np.testing.assert_allclose(observed, branch_direction, atol=6e-6, rtol=1e-4)
            finite_steps.append({"step": step, "local_regime_unchanged": same_regime,
                                 "expected_loss_derivative": float(total), "expected_fixed_path_derivative": float(path),
                                 "score_probability_derivative": float(selection),
                                 "decomposition_residual": float(total - path - selection),
                                 "native_branch_derivative": observed})
        assert finite_steps[-1]["local_regime_unchanged"]
        report["directions"][name] = {"analytic_score_probability_derivative": score,
                                      "analytic_score_probability_components": probability_components,
                                      "historical_native_sampled_derivative": branch_direction,
                                      "central_differences": finite_steps}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = (args.evidence / "protocol.json").read_bytes()
    plan = json.loads(protocol)
    cases = []
    for case in plan["cases"]:
        report = run_case(case, plan)
        cases.append(report)
        print(f"{case['label']}: {report['raw_pairs']} proposal pairs, {report['distinct_applied_graphs']} outcomes",
              flush=True)
    source = Path(__file__)
    hashes = {name: sha256(source.with_name(name).read_bytes()).hexdigest()
              for name in (source.name, "self_write_core.py", "self_write_core_bootstrap.py")}
    report = {"protocol_sha256": sha256(protocol).hexdigest(), "sources": hashes, "cases": cases}
    (args.evidence / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
