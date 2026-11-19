"""Measure what online weight and support changes contribute to version routing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics

import torch

from experiments.seed_evolution.large_runner import _write
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected.evolved_plasticity import RULE_FEATURES, WEIGHT_LIMIT
from experiments.undirected.evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, N_SYMBOLS, WAIT, _receipt_table
from experiments.undirected.plasticity_continuous_world import TICKS_PER_REQUEST, VISITS
from experiments.undirected.uncapped_bf16_run import configure
from experiments.undirected.uncapped_plasticity_run import OUT


SAVE = OUT.with_name("2026-11-16-uncapped-full-circuit") / "version_plasticity.json"
SEEDS = tuple(range(119_673_720, 119_673_728))
ACTORS = 16
RAILS = ((3, 63), (4, 62))
TRACKED = (*RAILS, (3, 56), (4, 56), (38, 56), (23, 56), (23, 66))
METRICS = (
    "version_accuracy", "version_coverage", "ready_coverage", "symbol_accuracy",
    "completion", "strict", "strict_return", "off_ready_rate", "mean_edges",
    "births", "deaths",
)


class LifeProbe(backend._BF16Policy):
    def __init__(self, birth, actors, tracked, weight_mask, support_changes, *, collect_drives=False):
        super().__init__(birth, actors, birth["birth_w"].device)
        self.weight_mask = None if weight_mask is None else weight_mask[:, None, :]
        self.support_changes = support_changes
        self.tracked = tracked
        self.collect_drives = collect_drives
        self.visit_ticks = N_SYMBOLS * TICKS_PER_REQUEST
        self.main_actions = []
        self.visit_weights = [self.w.index_select(-1, tracked)]
        self.rail_drives = []

    def _change_weights(self, local_activity, receipt):
        previous = self.w
        if self.weight_mask is None:
            super()._change_weights(local_activity, receipt)
        else:
            updates_before = self.weight_updates.clone()
            change_before = self.weight_change.clone()
            super()._change_weights(local_activity, receipt)
            self.w = torch.where(self.weight_mask, self.w, previous)
            actual = (self.w - previous).abs_()
            self.weight_updates.copy_(updates_before + actual.ne(0).sum(-1))
            self.weight_change.copy_(change_before + actual.sum(-1))

        if self.collect_drives:
            rails = self.tracked[:len(RAILS)]
            first = local_activity.index_select(-1, self.u[rails])
            second = local_activity.index_select(-1, self.v[rails])
            eligible = self.eligibility.index_select(-1, rails).float()
            coefficients = self.rule[:, 0]
            receipt_term = (receipt * coefficients[:, None, 5:]).sum(-1, keepdim=True).float()
            bias, weight, coactivity, trace, distance = coefficients[:, None, :5].float().unbind(-1)
            self.rail_drives.append(torch.stack((
                bias[..., None].expand_as(eligible),
                weight[..., None] * self.forward_w.index_select(-1, rails).float() / WEIGHT_LIMIT,
                coactivity[..., None] * (first * second).float(),
                trace[..., None] * eligible,
                distance[..., None] * (first - second).abs_().float(),
                receipt_term * eligible,
                (self.w - previous).index_select(-1, rails),
            )))

    def _change_support(self, local_activity, receipt):
        if self.support_changes:
            super()._change_support(local_activity, receipt)

    def act(self, obs):
        actions = super().act(obs)
        self.main_actions.append(actions[..., 0])
        if self.step % self.visit_ticks == 0:
            self.visit_weights.append(self.w.index_select(-1, self.tracked))
        return actions


def groups(genome):
    u, v = torch.triu_indices(genome["node_bias"].numel(), genome["node_bias"].numel(), offset=1)
    pairs = list(zip(u.tolist(), v.tolist(), strict=True))
    indices = {pair: index for index, pair in enumerate(pairs)}
    tracked = torch.tensor([indices[edge] for edge in TRACKED])
    birth = genome["birth_mask"]
    opponent = torch.zeros_like(birth)
    opponent[tracked[:len(RAILS)]] = True
    cue = u.eq(3) | u.eq(4) | v.eq(3) | v.eq(4)
    main_u = (u >= 56) & (u <= 66)
    main_v = (v >= 56) & (v <= 66)
    main = main_u | main_v
    direct = ((u.eq(3) | u.eq(4)) & main_v)
    selected = {
        "opponent_edges": opponent & birth,
        "version_cue_edges": cue & birth,
        "main_output_edges": main & birth,
        "version_to_main_edges": direct & birth,
    }
    if not birth[tracked].all():
        raise ValueError("Expected tracked edges in the inherited support")
    return pairs, tracked, selected


def visits_from_actions(actions, tape, metrics):
    actions = torch.stack(actions).unflatten(0, (-1, TICKS_PER_REQUEST))
    ready = tape.stage.unsqueeze(2).eq(2)
    target = tape.target[:, None, None, :]
    accepted = ready & actions.eq(target)
    incomplete = (accepted.cumsum(1) - accepted.long()).eq(0)
    active = (ready | tape.stage.unsqueeze(2).eq(1)) & incomplete
    legal = ready & incomplete & actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS)
    first_finish = legal & legal.cumsum(1).eq(1)
    version_right = (first_finish & ((actions - FIRST_FINISH) // N_SYMBOLS).eq(
        (target - FIRST_FINISH) // N_SYMBOLS)).any(1)
    covered = legal.any(1)
    complete = accepted.any(1)
    rejected = active & actions.ne(WAIT) & ~accepted
    strict = complete & ~rejected.any(1)
    torch.testing.assert_close(covered.float().mean((0, 2)), metrics["ready_coverage"])
    torch.testing.assert_close(version_right.float().mean((0, 2)), metrics["version_coverage"])
    torch.testing.assert_close(strict.float().mean((0, 2)), metrics["strict"])
    per_visit = torch.stack((covered, version_right, complete, strict)).unflatten(1, (-1, N_SYMBOLS))
    covered, version, completion, strict = per_visit.float().mean((2, 4)).unbind(0)
    return {
        "ready_coverage": covered.T.cpu().tolist(),
        "version_coverage": version.T.cpu().tolist(),
        "version_accuracy": torch.where(covered > 0, version / covered, 0.).T.cpu().tolist(),
        "completion": completion.T.cpu().tolist(),
        "strict": strict.T.cpu().tolist(),
    }


def verify_weights(graph, start):
    if graph.weight_mask is not None and not graph.support_changes:
        frozen = ~graph.weight_mask
        changed = graph.w.ne(start) & frozen
        if changed.any() or graph.births.any() or graph.deaths.any():
            raise ValueError("A frozen weight or support changed")


def summarize(values):
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "positive": sum(value > 0 for value in values), "negative": sum(value < 0 for value in values)}


def run_group(genome, tape, names, weight_mask, support_changes, tracked, selected, table, *, zero_rule_fields=None):
    _, birth = backend._stack_genomes([genome] * len(names))
    if zero_rule_fields:
        for name, fields in zero_rule_fields.items():
            birth["rule"][names.index(name), 0, list(fields)] = 0.
    birth = {key: value.to(table.device) for key, value in birth.items()}
    graph = LifeProbe(birth, ACTORS, tracked, weight_mask, support_changes, collect_drives=names == ["full"])
    start = graph.w.clone()
    metrics = backend._rollout(graph, tape, table)
    verify_weights(graph, start)
    per_visit_metrics = visits_from_actions(graph.main_actions, tape, metrics)
    metric_columns = {key: metrics[key].cpu().tolist() for key in METRICS}
    weight_trajectory = torch.stack(graph.visit_weights, dim=1).cpu()
    endpoints = graph.w.index_select(-1, tracked).cpu()
    abs_change = (graph.w - start).abs()
    birth_mask = birth["birth_mask"][0]
    mean_absolute_changes = {
        key: abs_change[..., mask].mean((1, 2)).cpu().tolist()
        for key, mask in {"all_birth_edges": birth_mask, **selected}.items()
    }
    conditions = {}
    for i, name in enumerate(names):
        tracked_values = {}
        for j, (u, v) in enumerate(TRACKED):
            weights = endpoints[i, :, j]
            tracked_values[f"{u}-{v}"] = {
                "actor_mean_by_visit_with_birth": weight_trajectory[i, :, :, j].mean(-1).tolist(),
                "final_by_actor": weights.tolist(),
                "at_negative_limit": weights.eq(-WEIGHT_LIMIT).sum().item(),
                "at_positive_limit": weights.eq(WEIGHT_LIMIT).sum().item(),
            }
        conditions[name] = {
            "metrics": {key: values[i] for key, values in metric_columns.items()},
            "by_visit": {key: values[i] for key, values in per_visit_metrics.items()},
            "tracked_weights": tracked_values,
            "mean_absolute_birth_weight_change": {key: values[i] for key, values in mean_absolute_changes.items()},
        }
    if graph.rail_drives:
        drives = torch.stack(graph.rail_drives).mean((0, 2, 3)).cpu().T.tolist()
        features = ("bias", "normalized_weight", "coactivity", "eligibility", "activity_distance",
                    "receipt_times_eligibility", "applied_weight_delta")
        conditions["full"]["rail_drive_mean"] = {
            f"{u}-{v}": dict(zip(features, values, strict=True))
            for (u, v), values in zip(RAILS, drives, strict=True)
        }
    return conditions


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=SAVE)
    args = parser.parse_args()
    configure()
    device = torch.device("cuda")
    checkpoint = args.run_dir / "champion.pt"
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    genome = record["genome"]
    pairs, tracked, selected = groups(genome)
    birth_mask = genome["birth_mask"]
    conditions = {"frozen": torch.zeros_like(birth_mask), "weights_only": birth_mask}
    comparisons = {
        "full_minus_frozen": ("full", "frozen"),
        "full_minus_weights_only": ("full", "weights_only"),
        "full_minus_support_only": ("full", "support_only"),
        "support_only_minus_frozen": ("support_only", "frozen"),
        "weights_only_minus_frozen": ("weights_only", "frozen"),
    }
    for label, mask in selected.items():
        only, except_group = f"only_{label}", f"except_{label}"
        conditions[only] = mask
        conditions[except_group] = birth_mask & ~mask
        comparisons[f"{only}_minus_frozen"] = (only, "frozen")
        comparisons[f"weights_only_minus_{except_group}"] = ("weights_only", except_group)
    zero_rule_fields = {
        "without_receipt_weight_modulation": tuple(range(5, len(RULE_FEATURES))),
        "without_weight_restoring_term": (1,),
    }
    for name in zero_rule_fields:
        comparisons[f"{name}_minus_weights_only"] = (name, "weights_only")
    weight_mask = torch.stack(list(conditions.values())).to(device)
    rule_names = list(zero_rule_fields)
    rule_weight_mask = birth_mask.to(device).expand(len(rule_names), -1)
    tracked = tracked.to(device)
    selected_device = {key: value.to(device) for key, value in selected.items()}
    table = _receipt_table(device).bfloat16()
    rows = []
    for seed in SEEDS:
        _, _, _, _, tape = backend._inputs([genome], seed, ACTORS, device, VISITS, "full")
        tape = backend._device_tape(tape, device)
        full = run_group(genome, tape, ["full"], None, True, tracked, selected_device, table)
        support_only = run_group(genome, tape, ["support_only"], torch.zeros_like(weight_mask[:1]),
                                 True, tracked, selected_device, table)
        weights = run_group(genome, tape, list(conditions), weight_mask, False, tracked, selected_device, table)
        rule = run_group(genome, tape, rule_names, rule_weight_mask, False, tracked, selected_device, table,
                         zero_rule_fields=zero_rule_fields)
        rows.append({"seed": seed, **full, **support_only, **weights, **rule})
        print(f"seed={seed} complete", flush=True)

    names = ["full", "support_only", *conditions, *rule_names]
    means = {name: {key: statistics.fmean(row[name]["metrics"][key] for row in rows) for key in METRICS}
             for name in names}
    differences = {name: {key: summarize([row[first]["metrics"][key] - row[second]["metrics"][key] for row in rows])
                          for key in METRICS} for name, (first, second) in comparisons.items()}
    _write(args.out, {
        "candidate": record["row"]["id"], "champion_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seed_count": len(SEEDS), "actors_per_seed": ACTORS, "visits": VISITS, "precision": "BF16 reference eager",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "seeds": list(SEEDS), "means": means, "paired_differences": differences, "paired": rows,
        "conditions": {
            "full": "Every ordinary weight update and support transition enabled.",
            "support_only": "Support transitions and initial weights of new edges enabled. No ordinary weight updates.",
            "frozen": "Birth weights and support fixed. Recurrent activity and receipts evolve normally.",
            "weights_only": "Ordinary weight updates enabled on all birth edges. Support fixed.",
            "only_or_except_group": "Support fixed. Enable ordinary weight updates only in the named birth-edge group, or only in its complement.",
            "rule_term_interventions": "Support fixed. Each listed term is set to zero in the ordinary weight rule. Receipts still reach the graph as observations.",
        },
        "zero_rule_terms": {name: [RULE_FEATURES[index] for index in fields] for name, fields in zero_rule_fields.items()},
        "birth_edge_groups": {key: [pair for pair, include in zip(pairs, mask.tolist(), strict=True) if include]
                              for key, mask in selected.items()},
        "rail_drive_mean_definition": "Additive drive terms expressed in FP32 from the BF16 inputs just before tanh and step; the applied delta includes tanh and clamping but excludes support transitions. Terms are descriptive, measured on the full trajectory.",
    })
    for name, metrics in means.items():
        print(name, {key: round(metrics[key] * 100, 3) for key in METRICS[:7]})


if __name__ == "__main__":
    main()
