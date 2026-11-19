"""Compare founder and terminal IID champions on common worlds and a same-state context fork."""

from dataclasses import replace
from functools import lru_cache, partial

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import uncapped_private_symbol_iid_run as run
from . import uncapped_private_symbol_iid_world as world
from . import uncapped_private_symbol_world as private
from .evolved_plasticity_world import CONTEXT, FIRST_FINISH, N_ACTIONS, NEW, READY
from .uncapped_private_symbol_audit import _ForkStateObserver


QUESTION = 48
METRICS = ("strict", "completion", "post_answer_actions", "first_finish_correct", "symbol_late_mean",
           "symbol_late_a", "symbol_late_b", "ready_coverage", "premature_request_rate")


def fork_schedule(*, seed, actors, device, visits):
    seed, context = divmod(seed, 2)
    tape = world.schedule(seed=seed, actors=actors, device=device, visits=visits)
    fields = {k: getattr(tape, k)[:QUESTION + 1].clone() for k in ("raw", "context", "public", "stage", "target")}
    port = fields["public"][-1, ..., 0, :]
    port[..., CONTEXT].zero_()
    port[..., CONTEXT.start + context] = port[..., NEW]
    fields["context"][-1].fill_(context)
    fields["target"][-1] = FIRST_FINISH + tape.maps[torch.arange(actors, device=device), context, fields["raw"][-1]]
    return replace(tape, **fields)


def fork_rollout(policy, tape, table):
    observer = _ForkStateObserver(policy, QUESTION * tape.stage.shape[1])
    _, raw = private.observed_rollout(observer, tape, table)
    actions = private.normalize_actions(raw[-1])
    eligible = tape.public[-1, ..., 0, READY].bool() & tape.stage[-1].eq(2)
    legal = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & eligible.unsqueeze(1)
    covered, tick = legal.max(0)
    answer = torch.where(covered, actions.gather(0, tick.unsqueeze(0)).squeeze(0), -1)
    return {**observer.prefix_state, "answer": answer, "correct": covered & answer.eq(tape.target[-1]),
            "target": tape.target[-1].expand_as(answer)}


def evaluate(generations=(1, 1024)):
    torch.set_num_threads(1)
    branch = run.OUT / "pair-01" / "downhill"
    read = run.previous_run.runner.base.read
    seeds = list(range(96_900_000, 96_900_032))
    settings = dict(device="cuda", capture=True, actors=run.ACTORS, visits=run.VISITS)
    evaluator = run.previous_run.PrivateEvaluation(lru_cache(maxsize=32)(world.schedule),
                                                  rollout=partial(run._rollout, training=False))
    fork = run.previous_run.PrivateEvaluation(fork_schedule, rollout=fork_rollout)
    result = {}
    try:
        for generation in generations:
            audit = read(branch / "audits" / f"{generation:04d}.json")
            bank = torch.load(branch / "generations" / f"{generation:04d}" / "elites.pt",
                              map_location="cpu", weights_only=True)["genomes"]
            ids = [audit["champions"][str(i)] for i in range(4)]
            genomes = [bank[ident] for ident in ids]
            rows = {mode: [] for mode in ("full", "no_reward_write", "frozen")}
            for mode, draws in rows.items():
                for idx, seed in enumerate(seeds):
                    m = evaluator(genomes, mode=mode, seed=seed, **settings)
                    cell = {key: m[key].cpu().tolist() for key in METRICS}
                    if generation == 1024 and mode == "full":
                        assert audit["seeds"][idx] == seed
                        native = audit["conditions"]["regular_native"]["modes"]["full"]["islands"]
                        for key in METRICS:
                            assert cell[key] == [native[str(i)]["paired_scores"][idx][key] for i in range(4)]
                    draws.append({"seed": seed, **cell})
            split = []
            for seed in seeds:
                left, right = (fork(genomes, mode="full", seed=2 * seed + ctx, **settings) for ctx in (0, 1))
                state_keys = left.keys() - {"answer", "correct", "target"}
                assert state_keys and all(torch.equal(left[k], right[k]) for k in state_keys)
                both = left["answer"].ge(0) & right["answer"].ge(0)
                swapped_a = left["answer"].eq(right["target"])
                swapped_b = right["answer"].eq(left["target"])
                split.append({"seed": seed, "covered": both.sum(-1).cpu().tolist(),
                              "switched": (both & left["answer"].ne(right["answer"])).sum(-1).cpu().tolist(),
                              "correct_answers": (left["correct"].sum(-1) + right["correct"].sum(-1)).cpu().tolist(),
                              "swapped_target_correct_answers": (swapped_a.sum(-1) + swapped_b.sum(-1)).cpu().tolist(),
                              "dual_swapped_correct": (swapped_a & swapped_b).sum(-1).cpu().tolist(),
                              "both_correct": (left["correct"] & right["correct"]).sum(-1).cpu().tolist()})
            result[str(generation)] = dict(champions=ids, modes=rows, fork=split)
            print(generation, {key: float(np.mean([x[key] for x in rows['full']])) for key in METRICS}, flush=True)
    finally:
        evaluator.clear_cache()
        fork.clear_cache()
    _write(run.OUT / "audits" / "iid-founder-terminal-mechanism.json", {
        "seeds": seeds, "actors": run.ACTORS, "visits": run.VISITS, "query_after_requests": QUESTION,
        "generations": result})


if __name__ == "__main__":
    evaluate()
