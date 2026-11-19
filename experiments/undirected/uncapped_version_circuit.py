"""Trace and isolate the final champion's version decision in fresh closed loops."""
from __future__ import annotations

import statistics

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected.evolved_plasticity_world import N_SYMBOLS, N_ACTIONS, FIRST_FINISH, _receipt_table
from experiments.undirected.plasticity_continuous_world import TICKS_PER_REQUEST
from experiments.undirected.uncapped_plasticity_run import OUT
from experiments.undirected.uncapped_read_weights import VERSION_RAILS, WAIT_RAILS, FrozenBirth, weight_matrix


DESTINATION = OUT.with_name("2026-11-16-uncapped-full-circuit") / "version_subgraph.json"
OUTCOMES = ("strict", "completion", "ready_coverage", "version_accuracy",
            "version_coverage", "strict_first", "strict_return", "symbol_accuracy")


class FixedSupport(backend._BF16Policy):
    def _change_support(self, local_activity, receipt):
        pass


class ReportDecisions:
    def __init__(self, graph, tape, cues):
        self.graph = graph
        self.population = graph.population
        self.tape = tape
        self.cues = torch.tensor(cues, device=graph.device).unsqueeze(-1)
        self.tick = 0
        # First and second READY, separately for actual version zero and one.
        self.counts = torch.zeros((5, self.population, 2, 2), dtype=torch.int32, device=graph.device)

    def act(self, obs):
        request, tick = divmod(self.tick, TICKS_PER_REQUEST)
        ready_now = obs[..., 10].ne(0)
        new = obs[..., 0].ne(0)
        drop = ((self.cues == 1) & ~new) | ((self.cues == 2) & ~ready_now) | (self.cues == 3)
        obs[..., 3:5].masked_fill_(drop.unsqueeze(-1), 0.)
        action = self.graph.act(obs)
        readiness = self.tape.stage[request, tick].eq(2)
        previous = self.tape.stage[request, tick - 1].eq(2) if tick else torch.zeros_like(readiness)
        target_version = ((self.tape.target[request] - FIRST_FINISH) // N_SYMBOLS)
        logits = self.graph.h[..., 59:67].reshape(self.population, self.graph.actors, 2, N_SYMBOLS)
        predicted = logits.amax(-1).argmax(-1)
        candidates = logits.argmax(-1)
        used_pair = candidates[..., 0].eq(3) & candidates[..., 1].eq(0)
        attempted = (action[..., 0] >= FIRST_FINISH) & (action[..., 0] < N_ACTIONS)
        correct_action_version = (action[..., 0] - FIRST_FINISH) // N_SYMBOLS == target_version
        for window, opportunity in enumerate((readiness & ~previous, readiness & previous)):
            for version in range(2):
                valid = ready_now & opportunity & target_version.eq(version)
                n, correct, finish, finish_correct, pair = self.counts[:, :, window, version].unbind()
                n.add_(valid.sum(-1))
                correct.add_((valid & predicted.eq(version)).sum(-1))
                finish.add_((valid & attempted).sum(-1))
                finish_correct.add_((valid & attempted & correct_action_version).sum(-1))
                pair.add_((valid & used_pair).sum(-1))
        self.tick += 1
        return action


def base_genome():
    return torch.load(OUT / "champion.pt", weights_only=True, map_location="cpu")


def induced_mask(nodes, u, v):
    nodes = torch.as_tensor(sorted(nodes), dtype=torch.long)
    return torch.isin(u, nodes) & torch.isin(v, nodes)


def reduced_trials(genome):
    matrix = weight_matrix(genome)
    u, v = torch.triu_indices(len(matrix), len(matrix), 1)
    version = {3, 4, 62, 63}
    gate = version | {56}
    core = gate | {0, 9, 10, 22, 23, 38, 60, 64, 66, 61}
    main = set(range(28)) | set(range(56, 67)) | {78, 79}
    broader = core | {5, 6, 7, 8, 24, 25, 59, 65}
    paths = (matrix[:, [3, 4, 10, 23]].abs().amax(-1)
             * matrix[:, [56, 60, 62, 63, 64, 66]].abs().amax(-1))
    local = broader | set(paths.argsort(descending=True)[:18].tolist())
    paths_in = (matrix[:, [3, 4, 10, 23]].abs().amax(-1)
                * matrix[:, [62, 63]].abs().amax(-1))
    controls = set(paths_in.argsort(descending=True)[:28].tolist())
    known_pairs = {*VERSION_RAILS, *WAIT_RAILS, (10, 60), (10, 64), (10, 38),
                   (38, 56), (23, 56), (23, 66), (23, 61), (10, 63)}
    explicit = torch.tensor([(a, b) in known_pairs for a, b in zip(u.tolist(), v.tolist(), strict=True)])
    masks = {"inhibitory_pair": induced_mask(version, u, v), "with_wait": induced_mask(gate, u, v),
             "explicit_routes": explicit, "named_nodes": induced_mask(core, u, v),
             "main_port_only": induced_mask(main, u, v), "named_relays": induced_mask(local, u, v),
             "version_relays": induced_mask(broader | controls, u, v)}
    result = {"whole_fixed_support": genome}
    for name, mask in masks.items():
        mask &= genome["birth_mask"]
        result[name] = {**genome, "birth_mask": mask, "birth_w": genome["birth_w"] * mask}
    return result


def data_for(results, counts):
    result = {key: results[key].detach().cpu().tolist() for key in OUTCOMES}
    result["window_counts"] = counts.permute(1, 2, 3, 0).cpu().tolist()
    return result


@torch.no_grad()
def eval_many(genomes, policy, cues, seed):
    actors, device, _, birth, tape = backend._inputs(genomes, seed, 16, "cuda", 8, "full")
    graph = policy({key: value.to(device) for key, value in birth.items()}, actors, device)
    table = _receipt_table(device).bfloat16()
    probe = ReportDecisions(graph, backend._device_tape(tape, device), cues)
    rollout = backend._rollout_with_preloaded_receipts(table)
    return data_for(rollout(probe, probe.tape, len(genomes)), probe.counts)


def unpack(result, names):
    return {name: {key: value[i] for key, value in result.items()} for i, name in enumerate(names)}


def mean_metrics(rows, name):
    result = {k: statistics.fmean(row[name][k] for row in rows) for k in OUTCOMES}
    totals = torch.as_tensor([row[name]["window_counts"] for row in rows]).sum(0)
    n, correct, attempts, attempts_correct, both_pair = totals.unbind(-1)
    acc = lambda numerator, denominator: (numerator.float() / denominator.clamp_min(1)).tolist()
    result["window_prediction_by_version"] = acc(correct, n)
    result["window_prediction"] = acc(correct.sum(-1), n.sum(-1))
    result["window_attempts"] = acc(attempts.sum(-1), n.sum(-1))
    result["window_attempt_version_accuracy"] = acc(attempts_correct.sum(-1), attempts.sum(-1))
    result["window_two_opponents_are_bank_maxima"] = acc(both_pair.sum(-1), n.sum(-1))
    result["window_counts"] = totals.tolist()
    return result


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = base_genome()
    genome = record["genome"]
    cue_names = ("whole_native", "version_at_new_only", "version_at_ready_only", "version_absent")
    reduced = reduced_trials(genome)
    rows = []
    for seed in range(61_405_080, 61_405_092):
        full = eval_many([genome] * len(cue_names), backend._BF16Policy, range(len(cue_names)), seed)
        cut = eval_many(list(reduced.values()), FixedSupport, [0] * len(reduced), seed)
        frozen = eval_many([genome, reduced["explicit_routes"]], FrozenBirth, [0, 0], seed)
        rows.append({"seed": seed, **unpack(full, cue_names), **unpack(cut, reduced),
                     **unpack(frozen, ("whole_frozen", "explicit_routes_frozen"))})
    names = (*cue_names, *reduced, "whole_frozen", "explicit_routes_frozen")
    means = {name: mean_metrics(rows, name) for name in names}
    _write(DESTINATION, {"candidate": record["row"]["id"], "created_utc": _utc(),
                         "edge_counts": {name: int(g["birth_mask"].sum()) for name, g in reduced.items()},
                         "means": means, "paired": rows})
    for name, row in means.items():
        print(name, "edges", (int(reduced[name]["birth_mask"].sum()) if name in reduced else "full"),
              {k: round(100*row[k], 2) for k in OUTCOMES},
              {k: row[k] for k in ("window_prediction_by_version", "window_attempts",
                                    "window_two_opponents_are_bank_maxima")}, flush=True)


if __name__ == "__main__":
    main()
