"""Test concrete directed champion circuits in paired, held-out closed loops."""

from __future__ import annotations

from functools import partial
import hashlib
import random
from statistics import fmean, stdev

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from . import uncapped_directed_reward as backend
from .evolved_plasticity import WEIGHT_LIMIT
from .uncapped_directed_world_fused import rollout_with_table
from .uncapped_directed_reward_run import OUT, TIMING
from .uncapped_ready_signal_run import schedule


IDS = ("g2047-i03-c0763", "g2047-i00-c0263")
VERSION_RAILS = ((3, 59), (3, 63), (4, 65))
RECEIPT_WAIT = ((22, 56), (23, 56))
REJECT_RELAY = ((23, 4),)
RAW_CONTEXT_RELAYS = ((5, 95), (6, 32), (7, 32), (24, 68), (25, 31))
WATCHED = (*VERSION_RAILS, *RECEIPT_WAIT, *REJECT_RELAY, *RAW_CONTEXT_RELAYS,
           (4, 56), (56, 65), (68, 63), (32, 66), (95, 63))
FIELDS = ("strict", "first_finish_correct_first_return", "symbol_accuracy", "version_accuracy", "version_coverage",
          "ready_coverage", "premature_request_rate", "idle", "mean_edges", "births", "deaths", "reward_events",
          "changed_writes", "weight_change", *(f"edge_{src}_{dst}_mean_change" for src, dst in WATCHED))


def matched_cuts(genome, loci, paths, seed):
    """Same destination, same sign, and similar absolute birth weight."""
    rng = random.Random(seed)
    weights, active = genome["birth_w"], genome["birth_mask"]
    used = set()
    for source, target in paths:
        weight = float(weights[loci[source, target]])
        candidates = [(abs(float(weights[index]) - weight), edge) for edge, index in loci.items()
                      if edge[1] == target and edge[0] not in (3, 4, 22, 23) and edge not in used
                      and active[index] and (bool(weights[index] >= 0) == (weight >= 0))]
        used.add(rng.choice(sorted(candidates)[:3])[1])
    return tuple(sorted(used))


def paired_difference(samples):
    center = fmean(samples)
    critical_t = 2.13145  # Two-sided 95% interval across the 16 paired worlds.
    radius = critical_t * stdev(samples) / len(samples) ** .5
    return {"delta": center, "interval95": [center - radius, center + radius]}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = OUT / "island_champions.pt"
    bank = torch.load(source, map_location="cpu", weights_only=True)["genomes"]
    u, v = torch.triu_indices(96, 96, 1)
    src, dst = torch.cat((v, u)), torch.cat((u, v))
    loci = {edge: index for index, edge in enumerate(zip(src.tolist(), dst.tolist(), strict=True))}
    first = bank[IDS[0]]
    cases = {
        "island3_full": (IDS[0], (), "all"),
        "island0_full": (IDS[1], (), "all"),
        "cut_version_rails": (IDS[0], VERSION_RAILS, "all"),
        "cut_receipt_to_wait": (IDS[0], RECEIPT_WAIT, "all"),
        "cut_reject_to_version1": (IDS[0], REJECT_RELAY, "all"),
        "cut_raw_context_relays": (IDS[0], RAW_CONTEXT_RELAYS, "all"),
        "writes_to_primary_actions_only": (IDS[0], (), "primary_only"),
        "writes_excluding_primary_actions": (IDS[0], (), "outside_primary"),
        "no_reward_writes": (IDS[0], (), "none"),
    }
    for i in range(4):
        cases[f"matched_version_cut_{i}"] = (IDS[0], matched_cuts(first, loci, VERSION_RAILS, 231 + i), "all")
        cases[f"matched_wait_cut_{i}"] = (IDS[0], matched_cuts(first, loci, RECEIPT_WAIT, 531 + i), "all")
    primary = (dst >= 56) & (dst <= 66)
    allowed = {"all": torch.ones_like(primary), "primary_only": primary, "outside_primary": ~primary,
               "none": torch.zeros_like(primary)}
    allowed_stack = torch.stack([allowed[rule] for _, _, rule in cases.values()])
    positions = [(lane, a, b) for lane, (_, cuts, _) in enumerate(cases.values()) for a, b in cuts]
    lanes, sources, targets = zip(*positions, strict=True)
    base_policy, base_rollout = backend._DirectedRewardPolicy, backend._rollout
    base_world = backend._rollout_with_preloaded_receipts

    class CircuitPolicy(base_policy):
        def __init__(self, birth, actors, device, mode):
            self.cut_lanes = torch.tensor(lanes, device=device)
            self.cut_sources = torch.tensor(sources, device=device)
            self.cut_targets = torch.tensor(targets, device=device)
            self.cut_zero = torch.zeros((len(lanes), actors), device=device, dtype=torch.float16)
            self.watched = torch.tensor([loci[arc] for arc in WATCHED], device=device)
            self.allowed = allowed_stack.to(device).view(len(cases), 1, 2, -1)
            super().__init__(birth, actors, device, mode)

        def _update_matrix(self):
            super()._update_matrix()
            self.matrix[self.cut_lanes, :, self.cut_targets, self.cut_sources] = self.cut_zero

        def _change_weights(self, coactivity, event):
            self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)
            enabled = event[..., None, None] & self.allowed
            self.attempted_writes.add_((self.eligibility.ne(0) & enabled).sum((-2, -1)))
            scale = (self.reward_rate[:, None] * self.reward)[..., None, None]
            proposal = torch.addcmul(self.w, self.eligibility, scale, out=self.proposal).clamp_(-WEIGHT_LIMIT, WEIGHT_LIMIT)
            torch.where(enabled, proposal, self.w, out=proposal)
            self.weight_updates.add_(proposal.ne(self.w).sum((-2, -1)))
            self.weight_change.add_((proposal - self.w).abs_().sum((-2, -1), dtype=torch.float32))
            self.w.copy_(proposal)

    def measure(policy, tape, table):
        before = policy.w.flatten(-2).index_select(-1, policy.watched).float()
        metrics = base_rollout(policy, tape, table)
        final = policy.w.flatten(-2).index_select(-1, policy.watched).float()
        signed = (final - before).mean(1)
        return {**metrics, **{f"edge_{a}_{b}_mean_change": signed[:, i] for i, (a, b) in enumerate(WATCHED)}}

    backend._DirectedRewardPolicy = CircuitPolicy
    backend._rollout = measure
    backend._rollout_with_preloaded_receipts = rollout_with_table
    seeds = list(range(97_410_100, 97_410_116))
    genomes = [bank[ident] for ident, _, _ in cases.values()]
    rows = []
    try:
        evaluator = backend.RewardEvaluation(partial(schedule, timing=TIMING))
        for i, seed in enumerate(seeds):
            result = evaluator(genomes, seed=seed, actors=16, device="cuda", visits=8)
            values = {key: result[key].cpu().tolist() for key in FIELDS}
            rows.append({"seed": seed, **{name: {key: column[j] for key, column in values.items()}
                                        for j, name in enumerate(cases)}})
            if (i + 1) % 4 == 0:
                print(f"Fresh circuit worlds {i + 1}/{len(seeds)}", flush=True)
    finally:
        backend._DirectedRewardPolicy, backend._rollout = base_policy, base_rollout
        backend._rollout_with_preloaded_receipts = base_world

    means = {name: {key: fmean(row[name][key] for row in rows) for key in FIELDS} for name in cases}
    differences = {name: {key: paired_difference([row[name][key] - row["island3_full"][key] for row in rows])
                          for key in FIELDS} for name in cases if not name.startswith("island")}
    for group, cut in (("version", "cut_version_rails"), ("wait", "cut_receipt_to_wait")):
        differences[f"{cut}_minus_matched"] = {
            key: paired_difference([row[cut][key] - fmean(row[f"matched_{group}_cut_{i}"][key] for i in range(4))
                                    for row in rows]) for key in FIELDS}
    _write(OUT / "audits" / "2048-directed-champion-circuit.json", {
        "completed_utc": _utc(), "champion_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seeds": seeds, "actors": 16, "visits": 8, "timing": TIMING, "device": torch.cuda.get_device_name(),
        "intervention": "Cuts affect forward transmission; write masks affect the reward increment. Public receipts remain real.",
        "variant_specs": {name: {"genome": ident, "cut_arcs": [
            {"source": a, "target": b, "birth_weight": float(first["birth_w"][loci[a, b]])} for a, b in cuts],
            "write_rule": rule, "birth_arcs_in_write_rule": int((bank[ident]["birth_mask"] & allowed[rule]).sum())}
            for name, (ident, cuts, rule) in cases.items()},
        "means": means, "paired_deltas_vs_island3_full_or_named_control": differences, "paired": rows,
    })
    for name, row in means.items():
        metrics = ("strict", "first_finish_correct_first_return", "symbol_accuracy", "version_accuracy",
                   "ready_coverage", "premature_request_rate", "idle")
        print(name, {key: round(row[key] * 100, 3) for key in metrics},
              "writes/receipt", round(row["changed_writes"] / max(1., row["reward_events"]), 2), flush=True)


if __name__ == "__main__":
    main()
