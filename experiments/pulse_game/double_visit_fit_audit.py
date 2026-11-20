"""Independently verify the inherited exposure, route ledgers and life-long gradients."""

from argparse import ArgumentParser
from unittest.mock import patch

import numpy as np
import torch

from . import double_visit_fit as fit
from . import double_visit_world as world
from . import reroute_fit_audit as prior_audit


require = prior_audit.require
exact = prior_audit.exact
take = prior_audit.payoff_at


def inheritance(birth):
    inherited = fit.inherited
    prior = fit.prior_fit
    _, record, sha = inherited.load_final("next_room", birth)
    require(record["ledger"].shape == (inherited.STEPS, inherited.BATCH, len(inherited.TRACE)),
            "Inherited checkpoint has an incomplete route ledger")
    rooms, reached, paid, cases, draws, counts = prior_audit.reconstructed(inherited, birth)
    _, _, audit = prior_audit.route_audit("next_room", record, rooms, reached, paid, cases, draws)
    prior_audit.replay_start(inherited, "next_room", birth, record)
    return {
        "file": fit.source_weight_file(birth).relative_to(fit.ROOT).as_posix(),
        "file_sha256": sha, "weights_sha256": fit.state_digest(record["state"]),
        "birth_weights_sha256": prior.state_digest(prior.make_graph(birth).state_dict()),
        "prior_map_counts": counts["prior_room_endpoints"],
        "heldout_map_rows": 0, "initial_update_replayed_bitwise": True, **audit,
    }


def independent_world_tables(birth):
    scenes, tapes = [], []
    for step in range(1, fit.STEPS + 1):
        scene = world.scenes([fit.DOMAIN, birth, step, 0], fit.BATCH, easy=step <= fit.EASY_STEPS)
        draws = np.random.default_rng([fit.DOMAIN, birth, step, 1]).random((2, fit.BATCH), dtype=np.float32)
        scenes.append(torch.cat((scene.door_rooms, scene.prior_endpoints, scene.inspected_rooms,
                                 scene.observed_endpoints, scene.visit_order, scene.goals), dim=1).to(torch.int8))
        tapes.append(torch.from_numpy(draws.T.copy()))
    worlds = torch.stack(scenes)
    doors, prior, inspected, observed, order = [worlds[..., offset:offset + 2].long()
                                               for offset in range(0, 10, 2)]
    goals = worlds[..., 10:].float()
    require((doors.sum(-1) == 1).all() and (order.sum(-1) == 1).all(), "Incomplete door history")
    exact(goals.sort(dim=-1).values, torch.tensor([-1., 0., 1.]).expand_as(goals), "Invalid payout assignment")
    maps = torch.cat((prior, 3 - prior.sum(-1, keepdim=True)), dim=-1).flatten(0, 1)
    map_ids, counts = torch.unique(maps, dim=0, return_counts=True)
    found = {tuple(row): n for row, n in zip(map_ids.tolist(), counts.tolist(), strict=True)}
    require(set(found) == set(fit.TRAINING_MAPS), f"Wrong map support for {birth}: {found}")

    # Evaluate the two observations in order for each physical room.
    touched_first = inspected[..., :1] == torch.arange(2)
    touched_second = inspected[..., 1:] == torch.arange(2)
    current = torch.where(touched_second, observed[..., 1:], torch.where(touched_first, observed[..., :1], prior))
    reached = current.gather(-1, doors)
    paid = goals.gather(-1, reached)
    counts = {
        "old_maps": {",".join(map(str, key)): value for key, value in found.items()},
        "odd_old_map_rows": 0, "same_room_revisited": int(inspected[..., 0].eq(inspected[..., 1]).sum()),
        "different_rooms_inspected": int(inspected[..., 0].ne(inspected[..., 1]).sum()),
        "current_route_endpoint_tie": int(reached[..., 0].eq(reached[..., 1]).sum()),
    }
    return worlds, torch.stack(tapes), doors, reached, paid, counts


def audit_routes(condition, record, tables):
    worlds, tape, rooms, reached, pay, _ = tables
    ledger = record["trace"]
    curve = record["curve"]
    require(ledger.shape == (fit.STEPS, fit.BATCH, len(fit.LEDGER_TRACE)), "Incomplete route ledger")
    require(curve.shape == (fit.STEPS, len(fit.METRICS)), "Incomplete curve")
    require(torch.isfinite(ledger).all() and torch.isfinite(curve).all(), "Nonfinite fit row")
    exact(record["worlds"], worlds, "Recorded worlds differ from the registered world tape")
    row = dict(zip(fit.LEDGER_TRACE, ledger.unbind(-1), strict=True))
    exact(torch.stack((row["query_draw"], row["action_draw"]), dim=-1), tape, "Motor tape mismatch")
    for key, exploration in (("query", fit.QUERY_EXPLORATION), ("action", fit.ACTION_EXPLORATION)):
        probability = row[f"{key}_right_probability"]
        predicted = row[f"{key}_logit"].sigmoid() * (1 - exploration) + exploration / 2
        torch.testing.assert_close(probability, predicted, rtol=1e-6, atol=1e-7)
        exact(row[key], row[f"{key}_draw"].lt(probability), f"Invalid sampled {key}")

    query, action = row["query"], row["action"]
    for label, matrix in (("actual_room", rooms), ("actual_endpoint", reached), ("paid", pay)):
        exact(row[label], take(matrix, action), f"Incorrect real-route {label}")
    matches = query.eq(action)
    exact(row["eligible"], matches, "Incorrect calibration eligibility")
    propensity = torch.where(action.bool(), row["action_right_probability"], 1. - row["action_right_probability"])
    weights = matches / propensity
    exact(row["calibration_weight"], weights, "Calibration used the wrong propensity")
    targets = row["actual_endpoint"].masked_fill(~matches, -1)
    if condition == "baseline":
        targets.fill_(-1)
    exact(row["endpoint_teacher_target"], targets, "Unexpected endpoint teaching target")

    endpoint_logits = torch.stack([row[f"endpoint_logit{i}"] for i in range(3)], dim=-1)
    identity_probability = endpoint_logits.log_softmax(dim=-1)
    identity_nll = -take(identity_probability, row["actual_endpoint"])
    policy_probs = torch.where(query.bool(), row["query_right_probability"], 1. - row["query_right_probability"])
    reward = row["paid"]
    advantage = (reward - reward.mean(dim=-1, keepdim=True)) * fit.BATCH / (fit.BATCH - 1)

    def average(values):
        return (weights * values).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.)

    independent = (
        reward.mean(dim=-1), matches.float().mean(dim=-1), average((row["predicted_payoff"] - reward).square()),
        average((row["predicted_room"] - (2 * row["actual_room"] - 1)).square()),
        average(row["predicted_room"].ge(0).eq(row["actual_room"].bool())), average(identity_nll),
        average(endpoint_logits.argmax(dim=-1).eq(row["actual_endpoint"])),
        -((policy_probs.log() + propensity.log()) * advantage).mean(dim=-1),
    )
    torch.testing.assert_close(curve[:, :-1], torch.column_stack(independent), rtol=5e-6, atol=2e-7)
    require(curve[:, -1].gt(0.).all(), "Missing training gradient")
    return row, {"worlds": query.numel(), "real_routes": query.numel(), "eligible_routes": int(matches.sum()),
                 "identity_targets": int(targets.ge(0).sum()),
                 "identity_target_counts": targets[targets.ge(0)].long().bincount(minlength=3).tolist()}


def replay_life(birth, condition, record, final_state):
    model = fit.initial_model(birth)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=fit.LR, foreach=False, fused=False)
    captured, differentiable = [], []
    native_tick = model.tick
    native_witness = fit.gradient_evidence
    physical_periods = (world.VISIT_TICKS, 2 * world.VISIT_TICKS,
                        2 * world.VISIT_TICKS + world.INSPECTION_TICKS, world.HISTORY_TICKS)
    trans_history = {}

    def tick(*args, **kwargs):
        result = native_tick(*args, **kwargs)
        captured.append(result[0])
        differentiable.append(torch.is_grad_enabled())
        return result

    def deeper(shared, trial, objectives):
        before_settlement = world.HISTORY_TICKS + world.GOAL_TICKS + world.PREPARATION_TICKS + world.TRIAL_TICKS
        require(differentiable == [True] * before_settlement + [False] * world.COMMIT_TICKS,
                "The full recurrent prefix must be differentiable and final settlement gradient-free")
        require(shared is captured[world.HISTORY_TICKS - 1], "Full history missing from shared state")
        for during in physical_periods:
            trans_history[str(during)] = {name: fit.prior_fit.witness(loss, captured[during - 1])
                                          for name, loss in objectives.items()}
        return native_witness(shared, trial, objectives)

    with patch.object(model, "tick", side_effect=tick), patch.object(fit, "gradient_evidence", side_effect=deeper):
        curve, ledger, gradients = fit.update(model, optimizer, condition, *fit.episode(birth, 1), check_grad=True)
    expected_ticks = (world.HISTORY_TICKS + world.GOAL_TICKS + world.PREPARATION_TICKS
                      + world.TRIAL_TICKS + world.COMMIT_TICKS)
    require(len(captured) == expected_ticks, "Settlement used the wrong tick count")
    exact(curve, record["curve"][0], "First optimizer update curve failed bitwise replay")
    exact(ledger, record["trace"][0], "First physical route failed bitwise replay")
    require(gradients == record["gradients"]["1"], "First gradient evidence failed replay")
    exact(initial["mask"], final_state["mask"], "Native graph topology changed")
    require(fit.state_digest(initial) == record["initial_sha256"], "Wrong shared initial weights")
    for update, witness in record["gradients"].items():
        objectives = {"payoff", "room", "policy"}
        if condition == "identity":
            objectives.add("identity")
            norms = np.array(witness["identity_direct_neuron_norms"])
            local = list(range(world.ENDPOINT_READOUT.start, world.ENDPOINT_READOUT.stop))
            require(np.all(norms[local] > 0.) and np.flatnonzero(norms).tolist() == local,
                    f"Step {update} differentiated the endpoint loss outside the original primary readout")
        for objective in objectives:
            norms = witness[objective]
            require(len(norms) == 4 and np.isfinite(norms).all() and sum(norms) > 0.,
                    f"Step {update} is missing a complete history witness for {objective}")
    require(set(record["gradients"]) == {"1", str(fit.EASY_STEPS + 1)}, "Gradient probe checkpoints missing")
    require(prior_audit.state_count(record) == len(list(model.parameters())), "Incomplete Adam history")
    return {"first_update_replayed_bitwise": True, "life_ticks": expected_ticks,
            "through_history_gradient_norms": record["gradients"],
            "first_update_gradients_by_physical_tick": trans_history,
            "changed_parameters": {key: int(value.ne(final_state[key]).sum()) for key, value in initial.items()}}


def summarize(births, *, inherited_only=False):
    fit.prior_fit.configure("cpu")
    identity = fit.register()
    windows = {"first_256": slice(0, fit.EASY_STEPS), "last_256": slice(-fit.EASY_STEPS, None)}
    result = {"audit_source_sha256": fit.digest(__file__), "fit_protocol_sha256": identity,
              "births": list(births), "updates": fit.STEPS, "worlds_per_update": fit.BATCH, "pairs": {}}
    for birth in births:
        inherited = inheritance(birth)
        record = {"inherited": inherited}
        result["pairs"][str(birth)] = record
        if inherited_only:
            print(birth, inherited["prior_map_counts"], flush=True)
            continue
        tables = independent_world_tables(birth)
        record["world_counts"] = tables[-1]
        record["conditions"] = {}
        actions = {}
        for condition in fit.CONDITIONS:
            model, final, file_digest = fit.load_final(birth, condition)
            checkpoint = fit.ledger_path(birth, condition)
            checkpoint_digest = fit.digest(checkpoint)
            require(checkpoint_digest == final["ledger_sha256"], "Ledger checksum mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(saved["signature"] == final["signature"] and saved["step"] == fit.STEPS,
                    "Ledger identity mismatch")
            require(fit.state_digest(saved["state"]) == final["state_sha256"], "Final weights differ from ledger")
            require(final["initial_sha256"] == inherited["weights_sha256"], "Unexpected initialization weights")
            rows, route_counts = audit_routes(condition, saved, tables)
            parameters = replay_life(birth, condition, saved, model.state_dict())
            record["conditions"][condition] = {
                "weights_path": fit.destination(birth, condition).relative_to(fit.ROOT).as_posix(),
                "weights_sha256": file_digest, "ledger_path": checkpoint.relative_to(fit.ROOT).as_posix(),
                "ledger_sha256": checkpoint_digest, "seconds": saved["seconds"],
                **route_counts, **parameters,
                **{name: dict(zip(fit.METRICS, saved["curve"][section].mean(0).tolist(), strict=True))
                   for name, section in windows.items()},
            }
            actions[condition] = (
                rows["query"].clone(), rows["action"].clone(),
                torch.column_stack([value[0] for key, value in rows.items() if key != "endpoint_teacher_target"]),
            )
        a, b = (actions[name] for name in fit.CONDITIONS)
        exact(a[2], b[2], "Paired states or physical paths differed before the first gradient update")
        same_query, same_action = a[0].eq(b[0]), a[1].eq(b[1])
        record["paired"] = {"identical_world_and_draw_rows": a[0].numel(),
                            "identical_initial_life_rows": len(a[2]), "same_query": int(same_query.sum()),
                            "same_action": int(same_action.sum()), "same_both": int((same_query & same_action).sum())}
        print(birth, record["world_counts"], record["paired"], flush=True)
    if tuple(births) == fit.BIRTHS:
        target = "fit-inheritance-audit.json" if inherited_only else "fit-audit.json"
        fit.prior_fit.storage._write_json(fit.EVIDENCE / target, result)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS, action="append")
    parser.add_argument("--inherited-only", action="store_true")
    args = parser.parse_args()
    summarize(tuple(args.birth) if args.birth else fit.BIRTHS, inherited_only=args.inherited_only)


if __name__ == "__main__":
    main()
