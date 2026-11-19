"""Read the main-port FINISH margins along the intact CUDA BF16 trajectory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import uncapped_bf16_backend as backend
from .evolved_plasticity import INPUTS
from .evolved_plasticity_world import FIRST_FINISH, NEW, N_ACTIONS, N_SYMBOLS, _receipt_table
from .uncapped_bf16_run import configure
from .uncapped_plasticity_run import OUT


SAVE = OUT.with_name("2026-11-16-uncapped-full-circuit") / "version_trace.json"
SEEDS = range(113_020_260, 113_020_268)
RAILS = ((3, 63), (4, 62))
MAIN_OUTPUT = slice(INPUTS, INPUTS + N_ACTIONS)
CUE_NODES = (3, 4)


class RecordingPolicy(backend._BF16Policy):
    def __init__(self, birth, actors, device):
        super().__init__(birth, actors, device)
        self.frames = []
        self.cue_frames = []

    def act(self, obs):
        before = self.h[0, :, MAIN_OUTPUT].clone()
        cue_before = self.h[0, :, CUE_NODES]
        activity = self.activity[0]
        action = super().act(obs)
        self.frames.append((before, activity, self.matrix[0, :, MAIN_OUTPUT].clone(),
                            self.h[0, :, MAIN_OUTPUT].clone(), action[0, :, 0]))
        self.cue_frames.append((cue_before, self.matrix[0, :, CUE_NODES, :],
                                obs[0, :, CUE_NODES], self.h[0, :, CUE_NODES]))
        return action


def projected_h(prior, activity, weights, bias):
    # Preserve the same BF16 casts and fused updates as act(). Rows here cover
    # the main port only and full replay equality is checked below.
    shape = prior.shape
    current = torch.bmm(weights.flatten(0, 1), activity.flatten(0, 1).unsqueeze(-1))
    return prior.mul(.75).add_(current.reshape(shape) + bias, alpha=.25)


def zero_rails(weights):
    weights = weights.clone()
    for source, target in RAILS:
        weights[..., target - INPUTS, source] = 0
    return weights


@torch.no_grad()
def record(genome, seed, actors, visits, device):
    actors, device, _, birth, tape = backend._inputs([genome], seed, actors, device, visits, "full")
    birth = {key: value.to(device) for key, value in birth.items()}
    policy = RecordingPolicy(birth, actors, device)
    metrics = backend._rollout(policy, backend._device_tape(tape, device), _receipt_table(device).bfloat16())
    prior, activity, weights, logits, action = (torch.stack(field) for field in zip(*policy.frames, strict=True))
    cue_prior, cue_weights, cue_obs, cue_logits = (torch.stack(field) for field in zip(*policy.cue_frames, strict=True))
    bias = policy.bias[0, :, MAIN_OUTPUT]
    reconstructed = projected_h(prior, activity, weights, bias)
    if not torch.equal(reconstructed, logits) or not torch.equal(action, logits.argmax(-1)):
        raise AssertionError("One update from recorded BF16 state did not match the actual readout")
    no_rails = projected_h(prior, activity, zero_rails(weights), bias)
    cue_bias = policy.bias[0, :, CUE_NODES]
    cue_reconstructed = projected_h(cue_prior, activity, cue_weights, cue_bias).add_(cue_obs, alpha=.25)
    if not torch.equal(cue_reconstructed, cue_logits):
        raise AssertionError("One update from recorded BF16 state did not match the actual cue state")
    cue_parts = {
        "state_after_observation": cue_logits.float(),
        "state_before_observation": cue_prior.float(),
        "previous_cue_logit": .75 * cue_prior.float(),
        "cue_input_pulse": .25 * cue_obs.float(),
        "cue_bias": torch.broadcast_to(.25 * cue_bias.float(), cue_prior.shape),
        "recurrent_graph": (.25 * cue_weights.float() * activity.float().unsqueeze(-2)).sum(-1),
    }
    terms = weights.float() * activity.float().unsqueeze(-2) * .25
    rail_terms = torch.zeros_like(prior, dtype=torch.float32)
    for source, target in RAILS:
        rail_terms[..., target - INPUTS] = terms[..., target - INPUTS, source]
    components = {
        "previous_logits": .75 * prior.float(),
        "bias": torch.broadcast_to(.25 * bias.float(), logits.shape),
        "opponent_rails": rail_terms,
        "other_version_input_edges": terms[..., 3:5].sum(-1) - rail_terms,
        "announce_wait_node_9": terms[..., 9],
        "new_node_0_and_ready_node_10": terms[..., 0] + terms[..., 10],
        "raw_and_context_inputs": terms[..., 5:9].sum(-1) + terms[..., 24:28].sum(-1),
        "main_receipt_inputs": terms[..., 11:24].sum(-1),
    }
    components["other_graph"] = terms.sum(-1) - sum(components[name] for name in list(components)[2:])
    components["bf16_rounding"] = logits.float() - sum(components.values())

    requests, ticks, _ = tape.stage.shape
    issue = tape.public[..., 0, NEW].bool().numpy().argmax(1)
    stage = tape.stage.numpy()
    first_ready = (stage == 2).argmax(1)
    version = (tape.target.numpy() - FIRST_FINISH) // N_SYMBOLS
    age = np.arange(ticks)[None, :, None] - issue[:, None, :]
    action = action.cpu().numpy().reshape(requests, ticks, actors)
    legal = (stage == 2) & (action >= FIRST_FINISH)
    first_legal = legal & (legal.cumsum(axis=1) == 1)
    version_right = ((action - FIRST_FINISH) // N_SYMBOLS == version[:, None, :]) & first_legal
    if not np.isclose(first_legal.any(axis=1).mean(), metrics["ready_coverage"].item()) or not np.isclose(
        version_right.sum() / max(1, first_legal.sum()), metrics["version_accuracy"].item()
    ):
        raise AssertionError("Recorded first legal FINISH did not reproduce the sealed world metrics")

    def as_world(value):
        return value.float().cpu().numpy().reshape(requests, ticks, actors, -1)

    parts = {name: as_world(value) for name, value in components.items()}
    per_source_margin = as_world(terms[..., 62 - INPUTS, :] - terms[..., 63 - INPUTS, :])
    new_request = (np.arange(requests)[:, None], issue, np.arange(actors)[None, :])
    within_request = np.zeros_like(parts["opponent_rails"])
    for tick in range(1, ticks):
        within_request[:, tick] = np.where(age[:, tick, :, None] > 0,
                                           .75 * within_request[:, tick - 1] + parts["opponent_rails"][:, tick], 0.)
    ready_by_age = (stage == 2) & (age == (first_ready - issue)[:, None, :])
    all_request_ticks = np.broadcast_to(np.arange(ticks)[None, :, None], action.shape)
    return {
        "metrics": {name: value.item() for name, value in metrics.items()},
        "version": np.broadcast_to(version[:, None, :], action.shape),
        "logits": as_world(logits),
        "no_rails": as_world(no_rails),
        "components": parts,
        "rails_since_new": within_request,
        "fixed_margin_per_source": per_source_margin,
        "fixed_margin_per_source_since_new": per_source_margin - per_source_margin[new_request][:, None, :, :],
        "rail_activity": as_world(activity[..., [3, 4]]),
        "rail_weights": as_world(torch.stack([weights[..., x - INPUTS, u] for u, x in RAILS], dim=-1)),
        "cue_parts": {name: as_world(value) for name, value in cue_parts.items()},
        "prior_cue_parts": {name: as_world(torch.cat((value.new_zeros((1, *value.shape[1:])), value[:-1]), dim=0))
                            for name, value in cue_parts.items()},
        "target_symbol": np.broadcast_to(((tape.target.numpy() - FIRST_FINISH) % N_SYMBOLS)[:, None, :], action.shape),
        "wait": np.broadcast_to((first_ready - issue)[:, None, :], action.shape),
        "masks": {"pre_new_in_request": age == -1, "new": age == 0,
                  "delay_1": (stage == 1) & (age == 1), "delay_2": (stage == 1) & (age == 2),
                  "first_ready": ready_by_age, "first_legal_finish": first_legal,
                  "first_legal_on_second_ready": first_legal & ~ready_by_age,
                  "second_ready_without_first_legal": (stage == 2) & ~ready_by_age & (
                      all_request_ticks <= np.where(first_legal.any(axis=1), first_legal.argmax(axis=1), ticks)[:, None, :]
                  )},
    }


def describe(values):
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "mean_abs": float(np.abs(values).mean()), "positive_fraction": float((values > 0).mean()),
            "negative_fraction": float((values < 0).mean())}


def readout(logits, version):
    banks = logits[:, FIRST_FINISH:].reshape(-1, 2, N_SYMBOLS)
    maxima = banks.argmax(-1) + np.arange(2) * N_SYMBOLS + FIRST_FINISH
    rows = np.arange(version.size)
    own, opponent = maxima[rows, version], maxima[rows, 1 - version]
    chosen = logits.argmax(-1)
    finish_version = (chosen - FIRST_FINISH) // N_SYMBOLS
    bank_right = banks.max(axis=-1).argmax(-1) == version
    return maxima, own, opponent, chosen, bank_right, finish_version == version


def measure(records, phase, version_filter=None, wait_filter=None):
    selected = [r["masks"][phase].copy() for r in records]
    for row, mask in zip(records, selected, strict=True):
        if version_filter is not None:
            mask &= row["version"] == version_filter
        if wait_filter is not None:
            mask &= row["wait"] == wait_filter

    def gather(key, component=None):
        return np.concatenate([(r[key] if component is None else r[key][component])[mask]
                               for r, mask in zip(records, selected, strict=True)])

    version = gather("version")
    if not version.size:
        return {"count": 0}
    logits, no_rails = gather("logits"), gather("no_rails")
    maxima, own, opponent, chosen, bank_right, action_right = readout(logits, version)
    _, _, _, chosen_without, bank_right_without, action_right_without = readout(no_rails, version)
    rows = np.arange(version.size)
    rails_w, rails_a = gather("rail_weights"), gather("rail_activity")
    comparison_62_63 = np.where(version[:, None] == 0, (62 - INPUTS, 63 - INPUTS), (63 - INPUTS, 62 - INPUTS))

    def margin(values, left, right):
        return values[rows, left] - values[rows, right]

    def comparison(left, right):
        exact = margin(logits, left, right)
        terms = {term: margin(gather("components", term), left, right) for term in records[0]["components"]}
        rail = terms["opponent_rails"]
        graph_without_rails = sum(value for name, value in terms.items() if name not in (
            "previous_logits", "bias", "opponent_rails", "bf16_rounding"))
        return {"margin": describe(exact), "per_update_terms": {term: describe(x) for term, x in terms.items()},
                "other_graph_all": describe(graph_without_rails),
                "direct_rails_carried_since_new": describe(margin(gather("rails_since_new"), left, right)),
                "conditional_exact_rail_margin_effect": describe(exact - margin(no_rails, left, right)),
                "rail_larger_than_carry_count": int((abs(rail) > abs(terms["previous_logits"])).sum())}

    is_finish = chosen >= FIRST_FINISH
    creates_wait = (chosen_without == 0) & action_right
    bank0_62, bank1_63 = maxima[:, 0] == (62 - INPUTS), maxima[:, 1] == (63 - INPUTS)
    rail_parts = -.25 * rails_w * rails_a
    best_finish = logits[:, FIRST_FINISH:].argmax(-1) + FIRST_FINISH
    per_source = gather("fixed_margin_per_source")
    per_source_delta = gather("fixed_margin_per_source_since_new")
    sign = (1 - 2 * version)[:, None]
    source_effect = (per_source_delta * sign).mean(0)
    largest_sources = np.argsort(-abs(source_effect), kind="stable")[:8]
    target_symbol = gather("target_symbol")
    return {
        "count": int(version.size),
        "bank0_winner_counts_by_node": {str(node): int((maxima[:, 0] == node - INPUTS).sum()) for node in range(59, 63)},
        "bank1_winner_counts_by_node": {str(node): int((maxima[:, 1] == node - INPUTS).sum()) for node in range(63, 67)},
        "both_62_and_63_win_their_bank_count": int((bank0_62 & bank1_63).sum()),
        "main_argmax_counts_by_node": {str(node): int((chosen == node - INPUTS).sum()) for node in range(56, 67)},
        "finish_bank_version_correct_count": int(bank_right.sum()),
        "main_legal_finish_argmax_count": int(is_finish.sum()),
        "main_finish_version_correct_count": int(action_right.sum()),
        "bank_correct_to_wrong_without_rails": int((bank_right & ~bank_right_without).sum()),
        "bank_wrong_to_correct_without_rails": int((~bank_right & bank_right_without).sum()),
        "correct_finish_to_other_argmax_without_rails": int((action_right & ~action_right_without).sum()),
        "other_to_correct_finish_argmax_without_rails": int((~action_right & action_right_without).sum()),
        "correct_finish_to_wait_without_rails": int(creates_wait.sum()),
        "correct_finish_to_wrong_finish_without_rails": int((action_right & (chosen_without >= FIRST_FINISH)
                                                              & ~action_right_without).sum()),
        "target_symbol_counts_by_true_version": [[int(((version == v) & (target_symbol == symbol)).sum())
                                                   for symbol in range(N_SYMBOLS)] for v in (0, 1)],
        "rail_live_forward_weights": {f"{u}-{i}": describe(rails_w[:, k]) for k, (u, i) in enumerate(RAILS)},
        "rail_source_prior_activities": {str(u): describe(rails_a[:, k]) for k, (u, _) in enumerate(RAILS)},
        "cue_update_at_this_tick": {
            str(node): {name: describe(gather("cue_parts", name)[:, i]) for name in records[0]["cue_parts"]}
            for i, node in enumerate(CUE_NODES)
        },
        "cue_update_used_for_this_action": {
            str(node): {name: describe(gather("prior_cue_parts", name)[:, i]) for name in records[0]["prior_cue_parts"]}
            for i, node in enumerate(CUE_NODES)
        },
        "rail_contribution_aligned_to_fixed_margin": {
            f"{u}-{i}": describe(rail_parts[:, k] * (1 - 2 * version) * (1 - 2 * k))
            for k, (u, i) in enumerate(RAILS)
        },
        "leading_sources_by_delta_from_new_into_fixed_margin": [
            {"node": int(i), "mean_aligned_contribution": float((per_source[:, i] * sign[:, 0]).mean()),
             "mean_aligned_delta_from_new": float(source_effect[i])}
            for i in largest_sources if source_effect[i] != 0
        ],
        "both_banks_max_margin": comparison(own, opponent),
        "target_aligned_62_vs_63_margin": comparison(*comparison_62_63.T),
        "best_finish_vs_wait_margin": comparison(best_finish, np.zeros_like(best_finish)),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=SAVE)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--visits", type=int, default=8)
    args = parser.parse_args()
    configure()
    device = torch.device("cuda")
    checkpoint = args.run_dir / "champion.pt"
    champion = torch.load(checkpoint, map_location="cpu", weights_only=True)
    genome = champion["genome"]
    if champion["row"]["id"] != "g1023-i06-c0096":
        raise ValueError("Unexpected champion for this circuit readout")
    pair_u, pair_v = torch.triu_indices(genome["node_bias"].numel(), genome["node_bias"].numel(), 1)
    rails_at_birth = {f"{u}-{v}": genome["birth_w"][(pair_u == u) & (pair_v == v)].item() for u, v in RAILS}
    records = [record(genome, seed, args.actors, args.visits, device) for seed in args.seeds]
    reference = backend.evaluate_continuous_eager([genome], seed=args.seeds[0], actors=args.actors,
                                                 device=device, visits=args.visits)
    if any(reference[key].item() != value for key, value in records[0]["metrics"].items()):
        raise AssertionError("Traced and uninstrumented eager rollouts differ")
    phases = records[0]["masks"]
    _write(args.out, {
        "candidate": champion["row"]["id"], "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seeds": list(args.seeds), "actors_per_seed": args.actors, "visits": args.visits,
        "precision": "sealed CUDA BF16 eager rollout; local additive accounting in FP32 with a BF16 rounding residual",
        "rail_birth_weights_fp32": rails_at_birth,
        "definitions": {
            "phase": "Version cue and READY enter only input logits during the current tick and can affect outputs via the next tick. Delay 1/2 excludes READY. Pre-NEW here includes only issue-offset-1 contracts.",
            "margin": "correct version bank maximum minus other version bank maximum; both selected nodes held fixed when decomposing. The 62 versus 63 margin uses those two nodes even when others win their banks. BF16 ties use the lowest node index.",
            "conditional_update": "Zero both reciprocal rails only for the conditional readout of the main-output rows, with this tick's other weights and the intact pre-tick activity and logits held fixed; this is not a new closed-loop trajectory.",
            "carried_rails": "Positive/negative net direct terms from the two rails after NEW, fractionally retained with .75 each tick. Holds the observed trajectory and omits the direct rail term at NEW, which was generated from the previous request's state.",
            "verification": "Every recorded main-output logit and version-input state was bit-identical to the unedited BF16 reconstruction. First legal FINISH coverage and version accuracy matched the rollout every seed, and all first-seed metrics matched the uninstrumented evaluator exactly.",
        },
        "phase": {phase: measure(records, phase) for phase in phases},
        "by_version": {str(v): {phase: measure(records, phase, version_filter=v) for phase in phases} for v in range(2)},
        "by_announced_wait": {str(wait): {phase: measure(records, phase, wait_filter=wait)
                                         for phase in ("first_ready", "first_legal_finish")} for wait in (1, 2, 3)},
        "by_seed": {str(seed): {"sealed_metrics": row["metrics"],
                                "first_ready": measure([row], "first_ready"),
                                "first_legal_finish": measure([row], "first_legal_finish")}
                    for seed, row in zip(args.seeds, records, strict=True)},
    })
    print(args.out)


if __name__ == "__main__":
    main()
