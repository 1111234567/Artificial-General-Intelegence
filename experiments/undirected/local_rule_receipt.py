"""Same-state single-receipt intervention, with identical later input in both branches."""

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import local_rule_world as world
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, READY, _receipt_table
from .local_rule_evaluator import _convert, make_policy
from .uncapped_private_symbol_receipt_probe import clear_receipts, snapshot as original_snapshot, restore as original_restore
from .uncapped_private_symbol_receipt_world import native_inputs, probe_frame
from .uncapped_reward_precision import _ACK, _REJ, _matmul_precision


SOURCE = 32
SEEDS = 102_910_000


def _snapshot(policy):
    if hasattr(policy, "state_tensors"):
        return {key: v.clone() for key, v in policy.state_tensors().items()}
    return original_snapshot(policy)


def _restore(policy, state):
    if isinstance(state, dict):
        for key, v in policy.state_tensors().items():
            v.copy_(state[key])
    else:
        original_restore(policy, state)


def _query(policy, template, keys):
    saved = _snapshot(policy)
    answers = []
    for raw, context in keys:
        _restore(policy, saved)
        first = torch.full_like(policy.pending, -1, dtype=torch.long)
        for step in template:
            obs = probe_frame(step, raw, context)
            action = policy.act(obs.flatten(-2)[None].expand(policy.population, -1, -1))[..., 0]
            legal = obs[..., 0, READY].bool() & action.ge(FIRST_FINISH) & action.lt(N_ACTIONS)
            first = torch.where(legal & first.lt(0), (action - FIRST_FINISH) % 4, first)
        answers.append(first)
    _restore(policy, saved)
    return answers


def _batch(genomes, seeds, device, arm):
    tapes = [world.schedule(seed=s, actors=8, visits=16) for s in seeds]
    dims = dict(public=2, stage=2, raw=1, context=1, target=1, maps=0)
    tape = _convert(replace(tapes[0], **{k: torch.cat([getattr(t, k) for t in tapes], dim=d)
                                        for k, d in dims.items()}), device)
    policy = make_policy(genomes, tape.raw.size(1), device, arm)
    saved = []
    obs, actions = native_inputs(policy, tape, _receipt_table(device).half(), SOURCE,
                                 lambda: saved.append(_snapshot(policy)))
    covered, first = (actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & obs[..., READY].bool()).max(1)
    pulse_tick = first[0] + 1
    eligible = covered[0] & pulse_tick.lt(obs.shape[1])
    previous = (actions[0].gather(0, first[0][None])[0] - FIRST_FINISH) % 4
    raw, ctx = tape.raw[SOURCE], tape.context[SOURCE]
    keys = [(raw, ctx), (raw, 1 - ctx), ((raw + 1) % 4, ctx)]
    idx = torch.arange(raw.numel(), device=device)
    targets = [tape.maps[idx, c, r] for r, c in keys]
    truth = previous.eq(targets[0])
    pulse_at = pulse_tick.clamp_max(obs.size(1) - 1)[None]
    native_ack = obs[0, ..., _ACK].gather(0, pulse_at)[0].bool()
    native_rej = obs[0, ..., _REJ].gather(0, pulse_at)[0].bool()
    if ((~(native_ack ^ native_rej) | (native_ack != truth)) & eligible).any():
        raise RuntimeError("Teaching action does not match native feedback")
    clear_receipts(obs[1, 0], torch.ones_like(eligible))
    ticks = torch.arange(obs.size(1), device=device)[None, :, None, None]
    following_write = covered[:, None] & ticks.eq(first[:, None])
    answers = {}
    for ack in (False, True):
        _restore(policy, saved[0])
        errors = torch.zeros_like(eligible)
        for tick, original in enumerate(obs[0]):
            step = original.clone()
            pulse = eligible & pulse_tick.eq(tick)
            after = eligible & pulse_tick.lt(tick)
            step[..., READY].masked_fill_(pulse | after, 0)
            clear_receipts(step, after)
            step[..., _ACK].masked_fill_(pulse, ack)
            step[..., _REJ].masked_fill_(pulse, not ack)
            errors |= pulse & ~policy.pending
            trial = policy.act(step)[..., 0]
            errors |= (~eligible | (tick < pulse_tick)) & trial.ne(actions[0, tick])
        if errors.any():
            raise RuntimeError("Factual actions differ before the intervened receipt")
        for delay in range(9):
            if delay in (0, 1, 8):
                answers[ack, delay] = _query(policy, tape.public[SOURCE + 9], keys)
            if delay < 8:
                for tick, frame in enumerate(obs[delay + 1]):
                    if tick:
                        policy.pending.copy_(following_write[delay + 1, tick - 1])
                    elif delay:
                        policy.pending.copy_(following_write[delay, -1])
                    else:
                        policy.pending.zero_()
                    policy.act(frame)
    rows = []
    for delay in (0, 1, 8):
        yes, no = answers[True, delay], answers[False, delay]
        genuine = [torch.where(truth, y, n) for y, n in zip(yes, no)]
        inverted = [torch.where(truth, n, y) for y, n in zip(yes, no)]
        values = {"eligible": eligible,
                  "joint_true": (genuine[0] == targets[0]) & (genuine[1] == targets[1]),
                  "joint_inverted": (inverted[0] == targets[0]) & (inverted[1] == targets[1])}
        for j, name in enumerate(("same", "other_context", "other_raw")):
            values.update({name + "_ack_repeat": yes[j].eq(previous), name + "_rej_repeat": no[j].eq(previous),
                           name + "_truthful_correct": genuine[j].eq(targets[j]),
                           name + "_inverted_correct": inverted[j].eq(targets[j]),
                           name + "_both_answer": yes[j].ge(0) & no[j].ge(0)})
        host = {k: (v & eligible).reshape(len(genomes), len(seeds), 8).sum(-1).cpu().tolist()
                for k, v in values.items()}
        for j, seed in enumerate(seeds):
            for g in range(len(genomes)):
                rows.append(dict(seed=seed, champion_index=g, delay=delay, **{k: v[g][j] for k, v in host.items()}))
    return rows


def summarize(rows):
    result = {}
    for delay in (0, 1, 8):
        groups = defaultdict(list)
        for row in rows:
            if row["delay"] == delay:
                groups[row["seed"]].append(row)
        effects = []
        for group in groups.values():
            count = sum(r["eligible"] for r in group)
            if not count:
                continue
            def rate(k):
                return sum(r[k] for r in group) / count
            contrasts = {"joint_correct_truthful": rate("joint_true"),
                         "joint_truthful_minus_inverted": rate("joint_true") - rate("joint_inverted")}
            for key in ("same", "other_context", "other_raw"):
                contrasts[key + "_ack_minus_rej_repeat"] = rate(key + "_ack_repeat") - rate(key + "_rej_repeat")
                contrasts[key + "_truthful_correct_gain"] = rate(key + "_truthful_correct") - rate(key + "_inverted_correct")
            contrasts["repeat_specificity_context"] = (contrasts["same_ack_minus_rej_repeat"]
                                                       - contrasts["other_context_ack_minus_rej_repeat"])
            effects.append(contrasts)
        summary = {"worlds_with_eligible_events": len(effects)}
        if effects:
            draws = np.random.default_rng(0).integers(len(effects), size=(4096, len(effects)))
            for key in effects[0]:
                vals = np.array([r[key] for r in effects])
                summary[key] = {"mean": float(vals.mean()), "ci95": np.quantile(vals[draws].mean(1), [.025, .975]).tolist()
                                if len(effects) > 1 else None}
        result[str(delay)] = summary
    return result


@torch.no_grad()
def audit_receipts(out, generation, arm, genomes_by_id, seed_count, device):
    if len(genomes_by_id) != 4:
        raise ValueError("receipt audit requires four island champions")
    path = Path(out) / "audits" / f"receipt-{generation:04d}-{arm}.json"
    ids = list(genomes_by_id)
    seeds = list(range(SEEDS, SEEDS + seed_count))
    meta = dict(schema=1, generation=generation, arm=arm, champions=ids, seeds=seeds)
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        if any(report.get(k) != v for k, v in meta.items()) or len(report["rows"]) != 12 * seed_count:
            raise ValueError(f"Incompatible receipt checkpoint {path}")
        return report
    device = torch.device("cuda:0" if str(device) == "cuda" else device)
    rows = []
    with _matmul_precision(torch.float16, False):
        for idx in range(0, seed_count, 4):
            rows.extend(_batch(list(genomes_by_id.values()), seeds[idx:idx + 4], device, arm))
    report = dict(**meta, opportunities=seed_count * 8 * 4, events=sum(r["eligible"] for r in rows if r["delay"] == 0),
                  rows=rows, delays=summarize(rows),
                  intervention="Only the sign of the first legal teaching receipt differs; matched retirement, "
                               "yoked later inputs and first-write opportunities, three unreinforced probe keys.")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, report)
    return report
