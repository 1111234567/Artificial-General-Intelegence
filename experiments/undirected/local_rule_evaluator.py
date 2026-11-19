"""Matched closed-loop evaluation of the original, local and modulated controllers."""

from dataclasses import replace

import torch

from . import local_rule_world as world
from . import uncapped_private_symbol_run as prior
from . import uncapped_private_symbol_span as span
from . import uncapped_private_symbol_world as private
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, RAW, READY, WAIT, _receipt_table
from .uncapped_answer_feedback import _RecordPrimaryActions
from .uncapped_directed_reward import _DirectedRewardPolicy, _stack_directed_genomes
from .uncapped_private_symbol_post_answer import post_answer_actions
from .uncapped_private_symbol_span_run import MainSpanEvaluation
from .uncapped_reward_precision import _matmul_precision


def score(row, requests=512):
    """Each strict pair earns one extra strict point at unchanged per-request scale."""
    a, b = row["symbol_late_a"], row["symbol_late_b"]
    result = {**row, "selection_score": (
        row["strict"] + .1 * (row["completion"] - row["strict"])
        + (row["pairs_both_strict"] - .1 * row["post_answer_actions"]) / requests
    ), "first_return_worst_context": min(a, b), "first_return_mean": (a + b) / 2}
    if min(a, b) > .25:
        result["sparsity_bonus"] = 1 - row["mean_edges"] / (96 * 95)
    return result


def trace_statistics(actions, tape, pair_indices):
    stage = tape.stage[:, :, None]
    active = stage.eq(1) | stage.eq(2)
    legal = stage.eq(2) & actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS)
    accepted = legal & (private.normalize_actions(actions) == tape.target[:, None, None])
    attempted, first = legal.max(1)
    initial = accepted.gather(1, first[:, None]).squeeze(1) & attempted
    moved, first_move = (active & actions.ne(WAIT)).max(1)
    strict = moved & accepted.gather(1, first_move[:, None]).squeeze(1)
    pair_count = pair_indices.ge(0).sum()
    b_strict = strict.gather(0, pair_indices.clamp_min(0)[:, None].expand_as(strict))
    both = (strict & b_strict & pair_indices[:, None].ge(0)).sum((0, 2))
    late = slice(len(stage) // 2, None)
    metrics = {}
    for ctx, label in enumerate(("a", "b")):
        use = tape.context[late].eq(ctx)[:, None]
        metrics[f"symbol_late_{label}"] = (initial[late] & use).sum((0, 2)) / use.sum().clamp_min(1)
    early_finish = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & stage.eq(1)
    return {**metrics, "strict": strict.float().mean((0, 2)),
            "completion": accepted.any(1).float().mean((0, 2)),
            "first_finish_correct": initial.float().mean((0, 2)),
            "ready_coverage": attempted.float().mean((0, 2)),
            "premature_request_rate": early_finish.any(1).float().mean((0, 2)),
            "post_answer_actions": post_answer_actions(actions, tape),
            "pairs_both_strict": both, "eligible_pairs": pair_count.expand_as(both),
            "pair_strict_rate": both / pair_count.clamp_min(1)}


def control_rollout(policy, tape, table):
    metrics, actions = span.observed_rollout(policy, tape, table)
    # The archived capture only copies public, target, stage and context each generation.
    raw = tape.public[..., 0, RAW].sum(1).argmax(-1)
    return {**metrics, **trace_statistics(actions, tape, world.partners(raw, tape.context))}


def _convert(tape, device):
    fields = {key: getattr(tape, key).to(device) for key in ("raw", "stage", "target", "context", "maps")}
    return replace(tape, public=tape.public.to(device=device, dtype=torch.float16), **fields)


def stack_birth(genomes, device, arm):
    _, birth = _stack_directed_genomes(genomes)
    if arm != "control":
        from .local_rule_policy import default_plastic
        birth["plastic"] = torch.stack([g["plastic"] if "plastic" in g else default_plastic() for g in genomes])
    return {k: v.to(device) for k, v in birth.items()}


def make_policy(genomes, actors, device, arm):
    birth = stack_birth(genomes, device, arm)
    if arm == "control":
        return _DirectedRewardPolicy(birth, actors, device, "full")
    from .local_rule_policy import LocalRulePolicy
    return LocalRulePolicy(birth, actors, device, arm)


class _RequestReplay:
    """One captured 18-tick block, continuous policy state and two-port receipts across blocks."""

    def __init__(self, genomes, device, arm, tape):
        self.policy = make_policy(genomes, tape.raw.shape[1], device, arm)
        ticks, actors = tape.stage.shape[1:]
        pop = len(genomes)
        n = pop * actors
        self.frame = torch.empty_like(tape.public[0])
        self.stage = torch.empty_like(tape.stage[0])
        self.target = torch.empty_like(tape.target[0])
        self.receipts = torch.zeros((pop, actors, 2), dtype=torch.long, device=device)
        self.flags = torch.zeros((6, pop, actors), dtype=torch.bool, device=device)
        self.counts = torch.zeros((8, pop, actors), dtype=torch.int32, device=device)
        outcomes = torch.empty((6, 1, pop, actors), dtype=torch.bool, device=device)
        self.trace = torch.empty((ticks, pop, actors), dtype=torch.long, device=device)
        table = _receipt_table(device).half()
        kernels = private._kernels(device)
        obs = torch.empty((pop, actors, 56), dtype=torch.float16, device=device)
        self.frame.copy_(tape.public[0])
        self.stage.copy_(tape.stage[0])
        self.target.copy_(tape.target[0])

        def request():
            for tick in range(ticks):
                kernels["directed_world_observation"](
                    ((n * 56 + 127) // 128,), (128,), self.frame[tick], table,
                    self.receipts, self.flags, obs, actors, n, int(tick == 0))
                action = self.policy.act(obs)
                self.trace[tick].copy_(action[..., 0])
                kernels["directed_world_action"](
                    ((n + 127) // 128,), (128,), action, self.stage[tick], self.target,
                    self.counts, self.receipts, self.flags, outcomes,
                    n, actors, 1, 0, int(tick == 0), int(tick == ticks - 1), *action.stride())
        request()
        self.graph = torch.cuda.CUDAGraph()
        # The kernels retain raw pointers to these allocations across every graph replay.
        self.scratch = (outcomes, table, obs)
        with torch.cuda.graph(self.graph):
            request()

    def run(self, birth, tape, pairs):
        self.policy.reset(birth)
        self.receipts.zero_()
        self.flags.zero_()
        self.counts.zero_()
        trace = self.trace.new_empty((len(tape.raw), *self.trace.shape))
        for idx in range(len(tape.raw)):
            self.frame.copy_(tape.public[idx])
            self.stage.copy_(tape.stage[idx])
            self.target.copy_(tape.target[idx])
            self.graph.replay()
            trace[idx].copy_(self.trace)
        counts = self.counts.sum(-1)
        return {**self.policy.summary(), "idle": counts[0] / counts[1].clamp_min(1),
                **trace_statistics(trace, tape, pairs)}


class Evaluation:
    def __init__(self, device, split):
        if split not in ("training", "audit"):
            raise ValueError(split)
        self.training = split == "training"
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.control = None
        self.replays = {}
        self.schedule_key = None

    def clear_cache(self):
        if self.control is not None:
            self.control.clear_cache()
        self.replays.clear()
        self.schedule_key = None

    close = clear_cache

    @torch.no_grad()
    def __call__(self, genomes, *, arm, seed, actors=8, visits=16):
        arm = "control" if arm == "original" else arm
        if arm not in ("control", "local", "modulated") or not genomes:
            raise ValueError("expected genomes and a known arm")
        cpu = world.tape_on_cpu(seed, actors, visits, self.training)
        if arm == "control" and self.training and self.device.type == "cuda":
            if self.control is None:
                # The old span engine selects its verified training path by schedule identity.
                saved = prior.world.training_schedule
                prior.world.training_schedule = world.training_schedule
                try:
                    self.control = MainSpanEvaluation(world.training_schedule, rollout=control_rollout)
                finally:
                    prior.world.training_schedule = saved
            metrics = self.control(genomes, seed=seed, actors=actors, visits=visits, device=self.device, capture=True)
        else:
            key = (seed, actors, visits)
            if self.schedule_key != key:
                self.tape = _convert(cpu, self.device)
                self.pairs = world.partners(cpu.raw, cpu.context).to(self.device)
                self.schedule_key = key
            tape = self.tape
            with _matmul_precision(torch.float16, False):
                if arm != "control" and self.device.type == "cuda":
                    shape = arm, len(genomes), actors, tape.stage.size(1)
                    if shape not in self.replays:
                        self.replays[shape] = _RequestReplay(genomes, self.device, arm, tape)
                    metrics = self.replays[shape].run(stack_birth(genomes, self.device, arm), tape, self.pairs)
                else:
                    policy = make_policy(genomes, actors, self.device, arm)
                    observer = _RecordPrimaryActions(policy, tape, len(genomes))
                    base = private.rollout_with_table(_receipt_table(self.device).half())(observer, tape, len(genomes))
                    metrics = {**policy.summary(), "idle": base["idle"],
                               **trace_statistics(observer.actions, tape, self.pairs)}
        # Preserve integer event counts. Training+audits consume the same schema.
        columns = {k: v.detach().cpu().tolist() for k, v in metrics.items()}
        rows = []
        for i in range(len(genomes)):
            row = {k: vals[i] for k, vals in columns.items()}
            row["symbol_late_mean"] = (row["symbol_late_a"] + row["symbol_late_b"]) / 2
            rows.append(score(row, len(cpu.raw) * actors))
        return rows
