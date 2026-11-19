"""Probe answer feedback and the ready cue on saved private-symbol champions."""

import argparse
from functools import lru_cache
from statistics import fmean

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from .evolved_plasticity import GATE_INTERVAL
from .evolved_plasticity_world import FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, N_SYMBOLS, READY, RECEIPT
from .uncapped_private_symbol_run import OUT, PrivateEvaluation, world
from .uncapped_reward_precision import _ACK, _REJ


_TREATMENTS = 8
_HORIZON = 3


def _finish(action):
    return (action >= FIRST_FINISH) & (action < N_ACTIONS)


def _symbol(action):
    return (action - FIRST_FINISH) % N_SYMBOLS


class _FeedbackProbe:
    def __init__(self, policy, tape):
        self.policy = policy
        self.stage = tape.stage.flatten(0, 1)
        self.ticks = tape.stage.size(1)
        self.target = tape.target
        self.total = len(self.stage)
        batch = policy.pending.shape
        device = policy.h.device
        lane = torch.arange(batch[0], device=device)[:, None] // (batch[0] // _TREATMENTS)
        self.reject = (lane & 1).bool()
        self.open = (lane & 2).bool()
        self.no_write = (lane & 4).bool()
        # Spread the first eligible probe through later requests across actors.
        self.start_request = 8 + (torch.arange(batch[1], device=device) * 5) % 16
        self.step = torch.full(batch, -_HORIZON, device=device, dtype=torch.long)
        self.previous = torch.zeros_like(self.step)
        self.first = torch.full_like(self.step, -1)
        self.answer = torch.full_like(self.step, -1)
        self.accepted = torch.zeros_like(self.step, dtype=torch.bool)
        self.responses = torch.full((*batch, _HORIZON), -1, device=device, dtype=torch.long)

    def act(self, obs):
        step = self.policy.step
        request, tick = divmod(step, self.ticks)
        if request < 8 or tick < 9:
            action = self.policy.act(obs)
            self.previous.copy_(action[..., 0])
            return action

        eligible = (step + 1 < self.total and tick + 1 < self.ticks
                    and 0 < step % GATE_INTERVAL <= GATE_INTERVAL - _HORIZON)
        trigger = ((self.step < 0) & self.policy.pending & (request >= self.start_request)
                   & self.stage[step + 1].eq(2)) if eligible else torch.zeros_like(self.accepted)
        self.first.copy_(torch.where(trigger, self.previous, self.first))
        self.answer.copy_(torch.where(trigger, self.target[request], self.answer))
        self.accepted.logical_or_(trigger & obs[..., _ACK].bool())
        self.step.masked_fill_(trigger, step)
        lag = step - self.step
        active = (lag >= 0) & (lag < _HORIZON) & self.stage[step].eq(2)
        suffix = active & ~trigger

        adjusted = obs.clone()
        adjusted[..., READY].copy_(torch.where(active, self.open.to(obs.dtype), obs[..., READY]))
        for offset in (0, FEATURES_PER_SLOT):
            receipt = slice(offset + RECEIPT.start, offset + RECEIPT.stop)
            adjusted[..., receipt].masked_fill_(suffix.unsqueeze(-1), 0.)
        adjusted[..., _ACK].copy_(torch.where(trigger, (~self.reject).to(obs.dtype), adjusted[..., _ACK]))
        adjusted[..., _REJ].copy_(torch.where(trigger, self.reject.to(obs.dtype), adjusted[..., _REJ]))
        self.policy.pending.logical_and_(~(trigger & self.no_write))

        action = self.policy.act(adjusted)
        primary = action[..., 0]
        for i in range(_HORIZON):
            self.responses[..., i].copy_(torch.where(active & (lag == i), primary, self.responses[..., i]))
        self.previous.copy_(primary)
        return action


def _probe_rollout(policy, tape, table):
    probe = _FeedbackProbe(policy, tape)
    world.rollout_with_table(table)(probe, tape, policy.population)
    return {key: getattr(probe, key) for key in ("step", "first", "answer", "accepted", "responses")}


def _natural_rollout(policy, tape, table):
    metrics, trace = world.observed_rollout(policy, tape, table)
    legal = _finish(trace) & tape.stage.eq(2).unsqueeze(2)
    attempted, first_tick = legal.max(dim=1)
    first = trace.gather(1, first_tick.unsqueeze(1)).squeeze(1)
    target = tape.target.unsqueeze(1)
    correct = _symbol(first).eq(_symbol(target))
    ticks = torch.arange(trace.size(1), device=trace.device)[None, :, None, None]
    remainder = (ticks > first_tick.unsqueeze(1)) & tape.stage.eq(2).unsqueeze(2)
    eligible = attempted & remainder.any(1)
    wrong = eligible & ~correct
    right = eligible & correct
    retry = legal & remainder
    repaired = retry & _symbol(trace).eq(_symbol(target).unsqueeze(1))
    return {"wrong_with_time": wrong.sum((0, 2)), "right_with_time": right.sum((0, 2)),
            "wrong_retried": (wrong & retry.any(1)).sum((0, 2)),
            "right_retried": (right & retry.any(1)).sum((0, 2)),
            "wrong_repaired": (wrong & repaired.any(1)).sum((0, 2)),
            **{key: metrics[key] for key in ("strict", "completion", "first_finish_correct")}}


def _read_probes(fields, identifiers, branches, seed):
    count = len(identifiers)
    meta = {key: value.cpu().reshape(_TREATMENTS, count, *value.shape[1:]) for key, value in fields.items()}
    for key in ("step", "first", "answer", "accepted"):
        if any(not torch.equal(meta[key][0], alternate) for alternate in meta[key][1:]):
            raise RuntimeError(f"Probe prefixes differ across treatments for {key}")
    responses = meta["responses"]
    if any(not torch.equal(responses[4, ..., 0], alternate[..., 0]) for alternate in responses[5:]):
        raise RuntimeError("Sensory-only intervention changed the pulse-tick action")
    rows = []
    for i, (ident, branch) in enumerate(zip(identifiers, branches, strict=True)):
        for actor, step in enumerate(meta["step"][0, i].tolist()):
            if step < 0:
                continue
            rows.append({"branch": branch, "id": ident, "seed": seed, "actor": actor, "pulse_step": step,
                         "previous": int(meta["first"][0, i, actor]), "target": int(meta["answer"][0, i, actor]),
                         "previous_correct": bool(meta["accepted"][0, i, actor]),
                         "responses": responses[:, i, actor].tolist()})
    return rows


def _outcomes(row, lane, start):
    actions = [action for action in row["responses"][lane][start:] if FIRST_FINISH <= action < N_ACTIONS]
    previous = _symbol(row["previous"])
    target = _symbol(row["target"])
    return (bool(actions), any(_symbol(action) != previous for action in actions),
            any(_symbol(action) == target for action in actions))


def _contrasts(rows):
    result = {}
    for previous_correct in (False, True):
        subjects = [row for row in rows if row["previous_correct"] == previous_correct]
        if not subjects:
            continue
        key = "after_right" if previous_correct else "after_wrong"
        result[key] = {"states": len(subjects)}
        for name, start, lanes in (("all_channels", 0, range(4)), ("sensory_only", 1, range(4, 8))):
            values = {}
            for lane in lanes:
                values[str(lane)] = dict(zip(("answers_again", "changes_answer", "gives_correct_answer"),
                                            map(fmean, zip(*(_outcomes(row, lane, start) for row in subjects))),
                                            strict=True))
            result[key][name] = values
    return result


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", nargs="+", type=int, default=[320, 448])
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    device = json.loads((OUT / "protocol.json").read_text())["device"]
    schedule = lru_cache(world.schedule)
    natural = PrivateEvaluation(schedule, rollout=_natural_rollout)
    probe = PrivateEvaluation(schedule, rollout=_probe_rollout)
    report = {"conditions": {str(lane): {"receipt": "REJ" if lane & 1 else "ACK",
                                        "ready": bool(lane & 2), "reward_write": not bool(lane & 4)}
                             for lane in range(_TREATMENTS)}, "generations": {}}
    for generation in args.generations:
        samples, natural_rows = [], []
        for pair in range(1, 5):
            for arm in ("downhill", "nondecreasing"):
                branch = f"{pair:02d}/{arm}"
                path = OUT / f"pair-{pair:02d}" / arm
                audit = json.loads((path / "audits" / f"{generation:04d}.json").read_text())
                checkpoint = torch.load(path / "generations" / f"{generation:04d}" / "elites.pt",
                                        map_location="cpu", weights_only=True)
                samples.append((audit["candidate"], branch, checkpoint["genomes"][audit["candidate"]]))
                del checkpoint
        probe_rows = []
        for offset in range(0, len(samples), 4):
            ids, branches, genomes = zip(*samples[offset:offset+4], strict=True)
            for seed in range(96600000, 96600008):
                settings = dict(seed=seed, actors=16, visits=8, device=device, capture=(device == "cuda"))
                actual = natural(genomes, **settings)
                host = {key: values.cpu().tolist() for key, values in actual.items()}
                for index, (ident, branch) in enumerate(zip(ids, branches, strict=True)):
                    natural_rows.append({"branch": branch, "id": ident, "seed": seed,
                                         **{key: values[index] for key, values in host.items()}})
                intervention = probe(genomes * _TREATMENTS, **settings)
                probe_rows.extend(_read_probes(intervention, ids, branches, seed))
            print(f"generation {generation}: verified {offset+4} of {len(samples)} matched champions", flush=True)
        report["generations"][generation] = {"natural": natural_rows, "probes": probe_rows,
                                             "contrasts": _contrasts(probe_rows)}
    report["finished_utc"] = _utc()
    natural.clear_cache()
    probe.clear_cache()
    target = OUT / "audits" / f"retry-feedback-{args.generations[0]:04d}-{args.generations[-1]:04d}.json"
    _write(target, report)
    print(target, flush=True)


if __name__ == "__main__":
    main()
