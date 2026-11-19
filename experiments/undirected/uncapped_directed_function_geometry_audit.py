"""Use frozen generation-2048 birth genomes to probe two functional geometries."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _utc, _write
from . import uncapped_directed_reward as backend
from .evolved_plasticity import ACTIONS, INPUTS, JOBS, WEIGHT_LIMIT
from .uncapped_directed_breeding import _crossover, validate_genome
from .uncapped_directed_compact_execution_run import CompactEvaluation
from .uncapped_directed_reward_run import NODES, OUT, TIMING, rank
from .uncapped_ready_signal_run import schedule


PORTS = INPUTS + JOBS * ACTIONS
U, V = torch.triu_indices(NODES, NODES, 1)
DST, SRC = torch.cat((U, V)), torch.cat((V, U))
HIDDEN_ARC = (DST >= PORTS) | (SRC >= PORTS)
TRACE_SEEDS = tuple(range(97_419_000, 97_419_004))
DISCOVERY_SEEDS = tuple(range(97_419_100, 97_419_112))
HOLDOUT_SEEDS = tuple(range(97_419_200, 97_419_224))
TRACE_TICKS = (0, 3, 7, 10, 14, 17)
TICKS_PER_REQUEST = TIMING["max_wait"] + TIMING["finish_window"] + 2
ACTORS, VISITS = 16, 8
CROSS_REPLICATES, DISCOVERY_DIRECTIONS, RANDOM_DIRECTIONS = 4, 16, 12
RADIUS, HEADROOM = .40, .06
FIELDS = ("first_finish_correct_first_return", "strict", "ready_coverage", "wrong_answer_rate",
          "premature_request_rate", "idle")
FIRST = FIELDS[0]
DEFAULT_OUTPUT = OUT / "audits" / "2048-directed-function-geometry-2026-11-19.json"


def load_parents(branch=OUT, generation=2048, ranking=rank):
    champ_path = branch / "island_champions.pt"
    elite_path = branch / "generations" / f"{generation:04d}" / "elites.pt"
    hashes = {}

    def load(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    champion_bank = load(champ_path)["genomes"]
    elite_bank = load(elite_path)
    parents, meta = {}, {}
    for i, island in enumerate(elite_bank["islands"]):
        champion, runner_up = sorted(island, key=ranking, reverse=True)[:2]
        for label, row in ((f"champion_{i}", champion), (f"runner_up_{i}", runner_up)):
            genome = elite_bank["genomes"][row["id"]]
            validate_genome(genome, NODES)
            if not bool(genome["directed"]):
                raise ValueError("This audit expects directed archived parents")
            if label.startswith("champion") and any(
                not torch.equal(value, champion_bank[row["id"]][key]) for key, value in genome.items()
            ):
                raise ValueError("Champion checkpoint and export disagree")
            parents[label] = genome
            meta[label] = {k: row[k] for k in ("id", "island", "lineage", "selection_score")}
        if all(torch.equal(parents[f"champion_{i}"][key], parents[f"runner_up_{i}"][key]) for key in genome):
            raise ValueError("The two selected local parents have the same birth genome")
    return parents, meta, hashes


def matrix(values):
    result = torch.zeros((NODES, NODES), dtype=values.dtype)
    result[DST, SRC] = values
    return result


def relabel(genome, new_to_old):
    if not torch.equal(new_to_old.sort().values, torch.arange(NODES)) or (new_to_old[:PORTS] != torch.arange(PORTS)).any():
        raise ValueError("Only hidden nodes may be permuted")
    return {**genome, "node_bias": genome["node_bias"][new_to_old],
            **{key: matrix(genome[key])[new_to_old[DST], new_to_old[SRC]] for key in ("birth_w", "birth_mask")}}


def random_relabel(seed):
    return torch.cat((torch.arange(PORTS), PORTS + torch.randperm(NODES - PORTS, generator=generator(seed))))


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def min_assignment(cost):
    """Return the minimum-cost square assignment via shortest augmenting paths."""
    values = cost.tolist()
    n = len(values)
    u, v, matching, previous = ([0] * (n + 1) for _ in range(4))
    for i in range(1, n + 1):
        matching[0], column = i, 0
        used, slack = [False] * (n + 1), [float("inf")] * (n + 1)
        while True:
            used[column] = True
            row = matching[column]
            step, successor = float("inf"), 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                gap = values[row - 1][j - 1] - u[row] - v[j]
                if gap < slack[j]:
                    slack[j], previous[j] = gap, column
                if slack[j] < step:
                    step, successor = slack[j], j
            for j in range(n + 1):
                if used[j]:
                    u[matching[j]] += step
                    v[j] -= step
                else:
                    slack[j] -= step
            column = successor
            if matching[column] == 0:
                break
        while column:
            before = previous[column]
            matching[column] = matching[before]
            column = before
    result = torch.empty(n, dtype=torch.long)
    for column, row in enumerate(matching[1:]):
        result[row - 1] = column
    return result


def port_signature(genome):
    w = matrix(genome["birth_w"])
    return torch.cat((w[PORTS:, :PORTS], w[:PORTS, PORTS:].T,
                      3. * genome["node_bias"][PORTS:, None]), dim=1)


def squared_dist(a, b):
    return torch.cdist(a, b, compute_mode="use_mm_for_euclid_dist").square()


def align(primary, donor, primary_trace, donor_trace):
    wired = squared_dist(port_signature(primary), port_signature(donor))
    active = squared_dist(primary_trace, donor_trace)
    cost = wired / wired.mean().clamp_min(1e-12) + active / active.mean().clamp_min(1e-12)
    assignment = min_assignment(cost)
    row = torch.arange(NODES - PORTS)
    order = torch.cat((torch.arange(PORTS), PORTS + assignment))
    return order, {"identity_cost": float(cost.diag().mean()),
                   "aligned_cost": float(cost[row, assignment].mean()),
                   "identity_trace_cost": float(active.diag().mean()),
                   "aligned_trace_cost": float(active[row, assignment].mean()),
                   "identity_port_cost": float(wired.diag().mean()),
                   "aligned_port_cost": float(wired[row, assignment].mean()),
                   "hidden_nodes_reassigned": int(assignment.ne(row).sum())}


@torch.no_grad()
def record_traces(parents, relabelled, orders, device, *, evaluator=None, seeds=TRACE_SEEDS,
                  actors=ACTORS, visits=VISITS):
    """Capture closed-loop activity and public action equality on discovery worlds."""
    original = backend._DirectedRewardPolicy
    instances = []

    class TracePolicy(original):
        def __init__(self, *args):
            super().__init__(*args)
            self.traces, self.actions = [], []
            instances.append(self)

        def act(self, obs):
            actions = super().act(obs)
            if (self.step - 1) % TICKS_PER_REQUEST in TRACE_TICKS:
                self.traces.append(self.activity[..., PORTS:].clone())
            self.actions.append(actions)
            return actions

    traces, witness = [], []
    cases = {**parents, **relabelled}
    labels = tuple(cases)
    if evaluator is None:
        evaluator = backend.RewardEvaluation(partial(schedule, timing=TIMING))
    backend._DirectedRewardPolicy = TracePolicy
    try:
        for seed in seeds:
            evaluator(list(cases.values()), seed=seed, actors=actors, device=device, visits=visits, capture=False)
            policy = instances.pop()
            activity = torch.stack(policy.traces, dim=2).cpu()
            actions = torch.stack(policy.actions, dim=2).cpu()
            traces.append(activity.transpose(1, 3).flatten(2).float())
            for i in range(4):
                name = f"champion_{i}"
                a, b = labels.index(name), labels.index(f"self_relabel_{i}")
                restored = activity[b, ..., orders[i].argsort()[PORTS:] - PORTS]
                witness.append({"seed": seed, "parent": name,
                                "comparisons": actions[a].numel(),
                                "different_actions": int(actions[a].ne(actions[b]).sum()),
                                "sampled_hidden_values": activity[a].numel(),
                                "different_sampled_hidden_values": int(restored.ne(activity[a]).sum()),
                                "max_sampled_hidden_abs_gap": float((restored - activity[a]).abs().max())})
    finally:
        backend._DirectedRewardPolicy = original
    return dict(zip(labels, torch.cat(traces, dim=-1), strict=True)), witness


def cross(primary, donor, seed):
    child = _crossover(primary, donor, seed, directed=True, donor_rule_row=None)
    # Isolate joint mask and weight replacement from the production rate mutation.
    return {**child, "reward_rate": primary["reward_rate"]}


def cross_delta(primary, child):
    delta = child["birth_w"] - primary["birth_w"]
    flipped = child["birth_mask"] != primary["birth_mask"]
    return {"weight_l2": float(delta.norm()), "hidden_incident_l2": float(delta[HIDDEN_ARC].norm()),
            "fixed_port_to_port_l2": float(delta[~HIDDEN_ARC].norm()), "flipped_mask": int(flipped.sum()),
            "flipped_hidden_incident_mask": int(flipped[HIDDEN_ARC].sum()),
            "changed_birth_weights": int(delta.count_nonzero())}


def crossover_cases(parents, relabelled, orders, traces):
    cases = {**parents, **relabelled}
    specs, alignments, self_checks = {}, {}, []
    for i in range(4):
        name = f"champion_{i}"
        parent = parents[name]
        pseudo = f"self_relabel_{i}"
        recovery, costs = align(parent, relabelled[pseudo], traces[name], traces[pseudo])
        exact = relabel(relabelled[pseudo], orders[i].argsort())
        matched = relabel(relabelled[pseudo], recovery)
        if any(not torch.equal(exact[key], parent[key]) for key in parent):
            raise ValueError("Hidden permutation followed by its inverse changed the birth genome")
        self_checks.append({"parent": name, **costs, "relabel_order": orders[i].tolist(),
                            "aligned_order": recovery.tolist(),
                            "matched_genome_equal": all(torch.equal(matched[key], parent[key]) for key in parent)})
        donors = [("self", pseudo), ("intra", f"runner_up_{i}")]
        donors += [("inter", f"champion_{j}") for j in range(4) if j != i]
        for j, (pair_type, donor_name) in enumerate(donors):
            donor = cases[donor_name]
            order, distances = align(parent, donor, traces[name], traces[donor_name])
            alignments[f"{name}/{donor_name}"] = {**distances, "new_to_old": order.tolist()}
            aligned = relabel(donor, order)
            for k in range(CROSS_REPLICATES):
                seed = 971_000 + 100 * i + 10 * j + k
                permutation = random_relabel(seed + 50_000)
                variants = {"raw": cross(parent, donor, seed), "aligned": cross(parent, aligned, seed)}
                if pair_type != "self":
                    variants["random_relabel"] = cross(parent, relabel(donor, permutation), seed)
                    # One of four paired copies supplies a production donor rule row.
                    variants["production_raw"] = _crossover(parent, donor, seed, directed=True,
                                                              donor_rule_row=(i + j) % 2 if k == 0 else None)
                for variant, genome in variants.items():
                    key = f"{name}/{donor_name}/{k}/{variant}"
                    cases[key] = genome
                    specs[key] = {"primary": name, "donor": donor_name, "stratum": pair_type,
                                  "variant": variant, "crossover_seed": seed, "relabel_seed": seed + 50_000,
                                  "production_rule_row": (i + j) % 2 if variant == "production_raw" and k == 0 else None,
                                  **cross_delta(parent, genome)}
    return cases, specs, alignments, self_checks


class Evaluator:
    def __init__(self, device, batch_size, *, actors=ACTORS, visits=VISITS, evaluator=None, fields=FIELDS):
        self.device, self.batch_size = device, batch_size
        self.actors, self.visits, self.fields = actors, visits, fields
        world = partial(schedule, timing=TIMING)
        if evaluator is None:
            evaluator = CompactEvaluation(world) if device == "cuda" else backend.RewardEvaluation(world)
        self.evaluator = evaluator

    def score_rows(self, values, count):
        tensor = torch.stack([values[field][:count] for field in self.fields], dim=-1).cpu().tolist()
        return [dict(zip(self.fields, scores, strict=True)) for scores in tensor]

    def measure(self, cases, seeds, label):
        names, genomes = tuple(cases), tuple(cases.values())
        rows = {seed: {} for seed in seeds}
        for start in range(0, len(names), self.batch_size):
            stop = start + self.batch_size
            group = list(genomes[start:stop])
            real = len(group)
            if real < self.batch_size and self.device == "cuda":
                group += [group[-1]] * (self.batch_size - real)
            for seed in seeds:
                values = self.evaluator(group, seed=seed, actors=self.actors, device=self.device, visits=self.visits,
                                        capture=self.device == "cuda")
                rows[seed].update(zip(names[start:stop], self.score_rows(values, real), strict=True))
            print(f"{label}: {min(stop, len(names))}/{len(names)} candidates across {len(seeds)} worlds", flush=True)
        return rows


def symmetric_variants(parent, direction, scope):
    direction = direction / direction.norm() * RADIUS
    w = parent["birth_w"]
    variants, norms = {}, {}
    for sign in (1, -1):
        sign_name = "+" if sign == 1 else "-"
        weights = w.clone()
        weights[scope] += sign * direction
        if ((weights - w)[~scope] != 0).any() or weights[scope].abs().max() > WEIGHT_LIMIT + 1e-7:
            raise ValueError("Perturbation escaped its scope or would require clipping")
        effective = weights.half().float() - w.half().float()
        if abs(float(effective.norm()) - RADIUS) > .002 * RADIUS:
            raise ValueError("The evaluated FP16 radius differs by more than 0.2%")
        variants[sign_name] = {**parent, "birth_w": weights}
        norms[sign_name] = {"birth_l2": float((weights - w).norm()), "evaluated_fp16_l2": float(effective.norm()),
                            "evaluated_fp16_changed_arcs": int(effective.count_nonzero()),
                            "max_coordinate_step": float(direction.abs().max())}
    return variants, norms


def perturbation_cases(parents, evaluator, *, discovery_seeds=DISCOVERY_SEEDS, endpoint=FIRST):
    discovery, basis, scopes = {}, {}, {}
    discovery_norms = {}
    for i in range(4):
        name = f"champion_{i}"
        parent = parents[name]
        scope = parent["birth_mask"] & parent["birth_w"].abs().le(WEIGHT_LIMIT - HEADROOM)
        scopes[name] = scope
        choices = torch.randn((DISCOVERY_DIRECTIONS, int(scope.sum())), generator=generator(160_000 + i))
        basis[name] = choices / choices.norm(dim=1, keepdim=True)
        for j, direction in enumerate(basis[name]):
            variants, norms = symmetric_variants(parent, direction, scope)
            discovery.update({f"{name}/basis_{j}/{sign}": genome for sign, genome in variants.items()})
            discovery_norms[f"{name}/basis_{j}"] = norms
    discovery_rows = evaluator.measure(discovery, discovery_seeds, "perturbation discovery")
    holdout, meta = {}, {}
    for i, (name, items) in enumerate(basis.items()):
        slopes = torch.tensor([np.mean([row[f"{name}/basis_{j}/+"][endpoint] - row[f"{name}/basis_{j}/-"][endpoint]
                                       for row in discovery_rows.values()]) for j in range(DISCOVERY_DIRECTIONS)])
        combined = (slopes[:, None] * items).sum(0)
        if combined.norm() == 0:
            raise ValueError("Discovery produced no directional signal")
        g = generator(170_000 + i)
        controls = torch.randn((RANDOM_DIRECTIONS, items.size(1)), generator=g)
        directions = {"coordinated": combined, **{f"random_{j}": v for j, v in enumerate(controls)}}
        parent, scope = parents[name], scopes[name]
        holdout[name] = parent
        meta[name] = {"scope_arc_count": int(scope.sum()), "basis_seed": 160_000 + i,
                      "random_control_seed": 170_000 + i, "discovery_plus_minus": slopes.tolist(), "norms": {}}
        for label, direction in directions.items():
            variants, norms = symmetric_variants(parent, direction, scope)
            holdout.update({f"{name}/{label}/{sign}": genome for sign, genome in variants.items()})
            meta[name]["norms"][label] = norms
    return holdout, meta, discovery_rows, discovery_norms


def interval(values, draws):
    values = np.asarray(values)
    return {"mean": float(values.mean()),
            "interval95": np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}


def summarize(rows, contrasts, fields=FIELDS):
    n = len(rows)
    draws = np.random.default_rng(971_900).integers(n, size=(8192, n))
    values = list(rows.values())
    return {label: {field: interval([contrast(row, field) for row in values], draws) for field in fields}
            for label, contrast in contrasts.items()}


def summarize_cross(rows, specs, fields=FIELDS):
    contrasts = {}
    for i in (None, *range(4)):
        parent = None if i is None else f"champion_{i}"
        suffix = "" if i is None else f"/{parent}"
        for stratum in ("self", "intra", "inter"):
            for variant in ("raw", "aligned", "random_relabel", "production_raw"):
                subset = {key: s for key, s in specs.items() if s["stratum"] == stratum and s["variant"] == variant
                          and (parent is None or s["primary"] == parent)}
                if not subset:
                    continue
                base = {key: s["primary"] for key, s in subset.items()}
                if variant != "raw":
                    raw = {key: key.rsplit("/", 1)[0] + "/raw" for key in subset}
                    contrasts[f"{stratum}/{variant}_minus_raw{suffix}"] = (
                        lambda row, field, raw=raw: np.mean([row[k][field] - row[v][field] for k, v in raw.items()]))
                contrasts[f"{stratum}/{variant}_minus_primary{suffix}"] = (
                    lambda row, field, base=base: np.mean([row[k][field] - row[v][field] for k, v in base.items()]))
    contrasts["self/relabelled_parent_minus_primary"] = (
        lambda row, field: np.mean([row[f"self_relabel_{i}"][field] - row[f"champion_{i}"][field] for i in range(4)]))
    result = summarize(rows, contrasts, fields)
    for variant in ("raw", "aligned", "production_raw"):
        keys = [[key for key, spec in specs.items() if spec["stratum"] == s and spec["variant"] == variant]
                for s in ("inter", "intra")]
        def damage(row, field, group):
            return np.mean([row[k][field] - row[specs[k]["primary"]][field] for k in group])
        result.update(summarize(rows, {f"inter_minus_intra/{variant}":
                                      lambda row, field: damage(row, field, keys[0]) - damage(row, field, keys[1])}, fields))
    return result


def summarize_perturbations(rows, fields=FIELDS):
    contrasts = {}
    def effects(row, field, name):
        baseline = row[name][field]
        picks = [row[f"{name}/coordinated/{sign}"][field] - baseline for sign in ("+", "-")]
        random = [[row[f"{name}/random_{j}/{sign}"][field] - baseline for sign in ("+", "-")]
                  for j in range(RANDOM_DIRECTIONS)]
        return {"plus_minus_parent": picks[0], "minus_minus_parent": picks[1],
                "plus_minus_minus": picks[0] - picks[1],
                "random_plus_minus_parent": np.mean([plus for plus, _ in random]),
                "random_minus_minus_parent": np.mean([minus for _, minus in random]),
                "random_plus_minus_minus": np.mean([plus - minus for plus, minus in random]),
                "plus_minus_random_plus": picks[0] - np.mean([plus for plus, _ in random]),
                "directional_advantage_vs_random": picks[0] - picks[1] - np.mean([plus - minus for plus, minus in random]),
                "symmetric_effect_vs_random": np.mean(picks) - np.mean(random)}

    parents = [f"champion_{i}" for i in range(4)]
    first_world = next(iter(rows.values()))
    for key in effects(first_world, fields[0], parents[0]):
        for group in (parents, *([parent] for parent in parents)):
            suffix = "" if len(group) == 4 else f"/{group[0]}"
            contrasts[f"{key}{suffix}"] = (
                lambda row, field, key=key, group=group: np.mean([effects(row, field, parent)[key] for parent in group]))
    return summarize(rows, contrasts, fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.exists():
        parser.error("Require a positive batch size and a fresh output path")
    torch.set_num_threads(4)
    parents, parent_meta, hashes = load_parents()
    orders = [random_relabel(870_000 + i) for i in range(4)]
    relabelled = {f"self_relabel_{i}": relabel(parents[f"champion_{i}"], order) for i, order in enumerate(orders)}
    traces, witness = record_traces(parents, relabelled, orders, args.device)
    print("Recorded hidden trajectories and relabeling witness", flush=True)
    evaluator = Evaluator(args.device, args.batch_size)
    crosses, cross_specs, alignments, self_checks = crossover_cases(parents, relabelled, orders, traces)
    # Commit every perturbation direction before inspecting any held-out scores.
    perturbations, perturb_specs, discovery, discovery_norms = perturbation_cases(parents, evaluator)
    crossover_rows = evaluator.measure(crosses, HOLDOUT_SEEDS, "crossover holdout")
    perturbation_rows = evaluator.measure(perturbations, HOLDOUT_SEEDS, "perturbation holdout")
    crossover_contrasts = summarize_cross(crossover_rows, cross_specs)
    perturbation_contrasts = summarize_perturbations(perturbation_rows)
    report = {"schema": 1, "completed_utc": _utc(), "source_archive_sha256": hashes,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": args.device, "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
              "torch_version": torch.__version__, "batch_size": args.batch_size, "actors": ACTORS, "visits": VISITS,
              "timing": TIMING, "parents": parent_meta, "primary_endpoint": FIRST,
              "intervals": "95% percentile bootstrap over paired held-out world seeds, fixed parents and operations. "
                           "Nominal intervals reuse worlds and are not multiplicity-adjusted.",
              "trace_discovery": {"world_seeds": TRACE_SEEDS, "ticks_in_request": TRACE_TICKS,
                                  "hidden_nodes": list(range(PORTS, NODES)), "permutation_witness": witness},
              "crossover": {"world_seeds": HOLDOUT_SEEDS, "alignments": alignments, "self_checks": self_checks,
                            "candidate_specs": cross_specs, "paired_scores": crossover_rows,
                            "paired_contrasts": crossover_contrasts},
              "perturbation": {"discovery_world_seeds": DISCOVERY_SEEDS, "holdout_world_seeds": HOLDOUT_SEEDS,
                               "birth_weight_l2": RADIUS, "headroom": HEADROOM,
                               "discovery_direction_count": DISCOVERY_DIRECTIONS, "random_control_count": RANDOM_DIRECTIONS,
                               "parent_specs": perturb_specs, "discovery_norms": discovery_norms,
                               "discovery_paired_scores": discovery, "holdout_paired_scores": perturbation_rows,
                               "holdout_paired_contrasts": perturbation_contrasts}}
    _write(args.output, report)
    for group, keys in ((crossover_contrasts, ("self/relabelled_parent_minus_primary", "self/raw_minus_primary",
                        "self/aligned_minus_primary", "inter/raw_minus_primary", "intra/raw_minus_primary",
                        "inter/aligned_minus_raw", "intra/aligned_minus_raw", "inter_minus_intra/raw")),
                        (perturbation_contrasts, ("plus_minus_parent", "minus_minus_parent", "plus_minus_minus",
                        "plus_minus_random_plus", "directional_advantage_vs_random"))):
        for key in keys:
            print(key, group[key][FIRST], flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
