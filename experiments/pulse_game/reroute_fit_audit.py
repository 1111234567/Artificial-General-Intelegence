"""Independently reconstruct route, teacher, and map evidence from fixed fits."""

from argparse import ArgumentParser
from itertools import combinations

import numpy as np
import torch

from . import reroute_fit as fit
from . import reroute_fit_transfer as transfer


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def exact(observed, expected, message):
    require(torch.equal(observed, expected), message)


def reconstructed(module, birth):
    tables, tapes = [], []
    for step in range(1, module.STEPS + 1):
        scene, q_draw, a_draw = module.episode(birth, step)
        tables.append(scene)
        tapes.append(torch.stack((q_draw, a_draw), dim=-1))
    fields = ("door_rooms", "prior_endpoints", "inspected_room", "observed_endpoint", "visit_order", "goals")
    rooms, prior, inspected, newly_observed, order, goals = [
        torch.stack([getattr(scene, field) for scene in tables]) for field in fields
    ]

    current = prior.clone()
    current.scatter_(-1, inspected.unsqueeze(-1), newly_observed.unsqueeze(-1))
    reached = current.gather(-1, rooms)
    payoffs = goals.gather(-1, reached)
    old_payoffs = goals.gather(-1, prior.gather(-1, rooms))
    entered_changed = current.ne(prior)
    tied = reached[..., 0] == reached[..., 1]
    inverted = old_payoffs.argmax(-1).ne(payoffs.argmax(-1))
    cases = torch.where(tied, 3, torch.where(~entered_changed.any(-1), 2, torch.where(inverted, 0, 1)))

    def map_counts(matrix):
        values, counts = np.unique(matrix.reshape(-1, 2).numpy(), axis=0, return_counts=True)
        return {f"{left},{right}": int(count) for (left, right), count in zip(values, counts, strict=True)}

    priors = map_counts(prior)
    allowed = {"0,1", "1,2", "2,0"} if module is transfer else {"0,1", "0,2", "1,0", "1,2", "2,0", "2,1"}
    require(set(priors) == allowed, f"Missing allowed prior maps or leakage into {birth}: {priors}")
    require(bool(order[..., 0].ne(order[..., 1]).all()), f"Missing the second prior door at {birth}")
    counts = {
        "prior_room_endpoints": priors, "current_room_endpoints": map_counts(current),
        "easy_cases": torch.bincount(cases[:module.EASY_STEPS].flatten(), minlength=4).tolist(),
        "formal_cases": torch.bincount(cases[module.EASY_STEPS:].flatten(), minlength=4).tolist(),
    }
    return rooms, reached, payoffs, cases, torch.stack(tapes), counts


def payoff_at(matrix, action):
    return matrix.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)


def route_audit(kind, saved, rooms, reached, payoff, cases, tape):
    ledger, curve = saved["ledger"], saved["curve"]
    routes = ledger.shape[:2]
    require(ledger.shape == (*routes, len(fit.TRACE)) and torch.isfinite(ledger).all(), "Nonfinite route ledger")
    require(curve.shape == (routes[0], len(fit.METRICS)) and torch.isfinite(curve).all(), "Broken curve")
    row = dict(zip(fit.TRACE, ledger.unbind(-1), strict=True))
    q, a = row["query"], row["action"]
    exact(torch.stack((row["query_draw"], row["action_draw"]), dim=-1), tape, "Exploration tape mismatch")
    for decision, alpha in (("query", fit.QUERY_EXPLORATION), ("action", fit.ACTION_EXPLORATION)):
        probability = row[f"{decision}_right_probability"]
        expected = (1 - alpha) / (1 + (-row[f"{decision}_logit"]).exp()) + alpha / 2
        torch.testing.assert_close(probability, expected, rtol=1e-6, atol=1e-7)
        exact(row[decision], row[f"{decision}_draw"].lt(probability), f"Invalid {decision} selection")
    for value, matrix in (("actual_room", rooms), ("actual_endpoint", reached), ("paid", payoff)):
        exact(row[value], payoff_at(matrix, a), f"Recorded {value} disagrees with the single action")

    matches = q.eq(a)
    choice_p = row["action_right_probability"]
    propensity = torch.where(a.bool(), choice_p, 1 - choice_p)
    weight = matches / propensity
    exact(row["actual_route_weight"], weight, "Incorrect causal eligibility or inverse propensity")
    donors = row["room_teacher_source"].long().numpy()
    targets = row["room_teacher_target"].numpy()
    room = row["actual_room"].numpy()
    for step, eligible in enumerate(matches.numpy()):
        active = eligible.nonzero()[0]
        expected_donors = np.full(routes[1], -1)
        expected_targets = np.zeros(routes[1])
        if len(active) > 1 and kind != "terminal":
            if kind == "next_room":
                expected_donors[active] = active
            else:
                expected_donors[active] = np.r_[active[1:], active[:1]]
            expected_targets[active] = room[step, expected_donors[active]] * 2 - 1
        if not np.array_equal(donors[step], expected_donors) or not np.array_equal(targets[step], expected_targets):
            raise AssertionError(f"Invalid room donor or target in {kind}, step {step + 1}")

    def mean_weighted(value, weights):
        return (value * weights).sum(-1) / weights.sum(-1).clamp_min(1.)

    teacher_weight = weight * row["room_teacher_source"].ge(0.)
    room_prediction = row["forecast_room"]
    paid = row["paid"]
    score_q = torch.where(q.bool(), row["query_right_probability"], 1 - row["query_right_probability"])
    advantage = (paid - paid.mean(-1, keepdim=True)) * routes[1] / (routes[1] - 1)
    values = {
        "paid": paid.mean(-1), "query_action_match": matches.float().mean(-1),
        "weighted_terminal_mse": mean_weighted((row["forecast_pay"] - paid).square(), weight),
        "weighted_real_room_mse": mean_weighted((room_prediction - (2 * row["actual_room"] - 1)).square(), weight),
        "weighted_room_teacher_mse": mean_weighted((room_prediction - row["room_teacher_target"]).square(),
                                                  teacher_weight),
        "weighted_real_room_acc": mean_weighted(room_prediction.ge(0).eq(row["actual_room"].bool()), weight),
        "policy_loss": -((score_q.log() + propensity.log()) * advantage).mean(-1),
        "query_right": q.mean(-1), "action_right": a.mean(-1),
        "used_room_target_count": row["room_teacher_source"].ge(0.).sum(-1).float(),
    }
    for name, expected in values.items():
        torch.testing.assert_close(curve[:, fit.METRICS.index(name)], expected, rtol=5e-6, atol=2e-7)
    return q, a, {
        "actual_routes": q.numel(), "matched_routes": matches.sum().item(),
        "matched_case_counts": torch.bincount(cases[matches], minlength=4).tolist(),
        "room_targets_applied": int(teacher_weight.ne(0.).sum()),
    }


def replay_start(module, kind, birth, saved):
    model = fit.make_graph(birth)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=fit.LR, foreach=False, fused=False)
    curve, ledger, gradients = fit.update(model, optimizer, kind, *module.episode(birth, 1), probe_gradients=True)
    exact(curve, saved["curve"][0], "First curve did not replay bitwise")
    exact(ledger, saved["ledger"][0], "First route did not replay bitwise")
    require(gradients == saved["through_history_gradient_norms"]["1"], "First gradient witness did not replay")
    exact(saved["state"]["mask"], initial["mask"], "Birth topology changed")
    require(state_count(saved) == len(list(model.parameters())), "Missing final optimizer parameter")
    return {key: int(initial[key].ne(value).sum()) for key, value in saved["state"].items()}


def state_count(saved):
    steps = [int(value["step"]) for value in saved["optimizer"]["state"].values()]
    require(all(step == saved["step"] for step in steps), "Incomplete optimizer progress")
    return len(steps)


def summarize(module, births):
    fit.configure("cpu")
    is_transfer = module is transfer
    identity = module.register()
    result = {"audit_source_sha256": fit.digest(__file__), "protocol_sha256": identity, "births": births,
              "steps": module.STEPS, "batch": module.BATCH, "cohorts": {}, "arms": {},
              "case_order": ["flip", "stay", "unchanged", "tie"]}
    windows = ("first_256", "first_formal_256", "last_256")
    ranges = (slice(0, module.EASY_STEPS), slice(module.EASY_STEPS, 2 * module.EASY_STEPS),
              slice(-module.EASY_STEPS, None))
    for birth in births:
        rooms, reached, payoff, cases, tape, task_counts = reconstructed(module, birth)
        record = dict(worlds=task_counts, arms={}, paired_actions={})
        movements, births_seen, initial_routes = {}, set(), {}
        common = [index for index, field in enumerate(fit.TRACE) if not field.startswith("room_teacher_")]
        for kind in module.KINDS:
            _, saved, sha = module.load_final(kind, birth)
            require(saved["ledger"].shape == (module.STEPS, module.BATCH, len(fit.TRACE)), "Incomplete route shape")
            q, a, checks = route_audit(kind, saved, rooms, reached, payoff, cases, tape)
            movements[kind] = (q, a)
            initial_routes[kind] = saved["ledger"][0, :, common].clone()
            changed = replay_start(module, kind, birth, saved)
            births_seen.add(saved["birth_sha256"])
            record["arms"][kind] = {
                "checkpoint": module.destination(kind, birth).relative_to(fit.ROOT).as_posix(),
                "sha256": sha, "birth_sha256": saved["birth_sha256"], "training_seconds": saved["seconds"],
                "first_step_replayed_bitwise": True, "changed_entries": changed,
                "through_history_gradient_norms": saved["through_history_gradient_norms"], **checks,
                **{name: dict(zip(fit.METRICS, saved["curve"][span].mean(0).tolist(), strict=True))
                   for name, span in zip(windows, ranges, strict=True)},
            }
            print(birth, kind, checks, flush=True)
        require(len(births_seen) == 1, "Paired arms did not start with the same 40-node birth")
        for left, right in combinations(module.KINDS, 2):
            first_q, first_a = movements[left]
            second_q, second_a = movements[right]
            same_q, same_a = first_q.eq(second_q), first_a.eq(second_a)
            exact(initial_routes[left], initial_routes[right], "Initially paired bodies had different physical paths")
            record["paired_actions"][f"{left}/{right}"] = {
                "common_query_draws": first_q.numel(), "common_action_draws": first_a.numel(),
                "same_query": int(same_q.sum()), "same_action": int(same_a.sum()), "same_both": int((same_q & same_a).sum()),
                "identical_initial_update_routes": module.BATCH,
            }
        result["cohorts"][str(birth)] = record
    for kind in module.KINDS:
        groups = [record["arms"][kind] for record in result["cohorts"].values()]
        result["arms"][kind] = {
            "actual_routes": sum(row["actual_routes"] for row in groups),
            "matched_routes": sum(row["matched_routes"] for row in groups),
            "applied_room_targets": sum(row["room_targets_applied"] for row in groups),
            **{name: {key: np.mean([row[name][key] for row in groups]).item() for key in fit.METRICS}
               for name in windows},
        }
    name = "fit-transfer-results.json" if is_transfer else "fit-results.json"
    if tuple(births) == module.BIRTHS:
        fit.storage._write_json(fit.EVIDENCE / name, result)
    print(result["arms"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS, action="append")
    args = parser.parse_args()
    module = transfer if args.transfer else fit
    summarize(module, tuple(args.birth) if args.birth else module.BIRTHS)


if __name__ == "__main__":
    main()
