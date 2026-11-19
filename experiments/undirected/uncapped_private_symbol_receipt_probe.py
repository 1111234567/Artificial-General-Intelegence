"""Fork the feedback to one native first answer, then retest three keys."""

import argparse
from dataclasses import replace
import json

import torch

from experiments.seed_evolution.large_runner import _write
from . import uncapped_private_symbol_iid_run as run
from . import uncapped_private_symbol_iid_world as world
from .evolved_plasticity import GATE_INTERVAL
from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, N_SYMBOLS, READY, RECEIPT, FEATURES_PER_SLOT, _receipt_table
from .uncapped_directed_reward import _DirectedRewardPolicy, _stack_directed_genomes
from .uncapped_private_symbol_audit import _STATE_FIELDS
from .uncapped_private_symbol_receipt_world import native_inputs, probe_frame
from .uncapped_private_symbol_receipt_stats import summarize
from .uncapped_reward_precision import _ACK, _REJ, _matmul_precision


SOURCE = 32  # zero-based, after a normal 32-request history
MODES = ("ack", "rej", "ack_no_write", "rej_no_write")
DELAYS = (0, 1, 8)
SEED = 97_120_000


def snapshot(policy):
    return (policy.step, policy.edge_span_start,
            {name: getattr(policy, name).clone() for name in (*_STATE_FIELDS, "matrix")})


def restore(policy, state):
    policy.step, policy.edge_span_start, values = state
    for name, value in values.items():
        getattr(policy, name).copy_(value)


def batch_tape(seeds, device):
    tapes = [world.schedule(seed=s, actors=run.ACTORS, visits=run.VISITS, device="cpu") for s in seeds]
    dims = dict(public=2, stage=2, target=1, raw=1, context=1, maps=0)
    fields = {k: torch.cat([getattr(t, k) for t in tapes], dim=d).to(device) for k, d in dims.items()}
    fields["public"] = fields["public"].half()
    return replace(tapes[0], **fields)


def clear_receipts(frame, mask):
    for offset in (0, FEATURES_PER_SLOT):
        frame[..., offset + RECEIPT.start:offset + RECEIPT.stop].masked_fill_(mask[..., None], 0)


def teaching(policy, frames, reference_actions, pulse_tick, eligible, mode, native_tail):
    changed = torch.zeros_like(policy.weight_updates)
    invalid = torch.zeros_like(eligible)
    for tick, original in enumerate(frames):
        pulse = eligible & pulse_tick.eq(tick)
        after = eligible & pulse_tick.lt(tick)
        frame = original.clone()
        if not native_tail:
            frame[..., READY].masked_fill_(pulse | after, 0)
            clear_receipts(frame, after)
        # Keep the receipt's one-hot copy of the action which actually elicited it.
        frame[..., _ACK].masked_fill_(pulse, mode.startswith("ack"))
        frame[..., _REJ].masked_fill_(pulse, mode.startswith("rej"))
        invalid |= pulse & ~policy.pending
        if mode.endswith("_no_write"):
            policy.pending.logical_and_(~pulse)
        before = policy.weight_updates.clone()
        act = policy.act(frame)[..., 0]
        changed += (policy.weight_updates - before) * pulse
        before_receipt = ~eligible | (tick < pulse_tick)
        invalid |= before_receipt & act.ne(reference_actions[tick])
        expected_reward = (0 if mode.endswith("_no_write") else (1 if mode == "ack" else -1))
        invalid |= pulse & policy.reward.ne(expected_reward)
    if invalid.any().item():
        raise RuntimeError("native prefix, receipt timing or intervention parity failed")
    if mode.endswith("_no_write") and changed.any().item():
        raise RuntimeError("the disabled event changed weights")
    return changed


def test_keys(policy, template_frames, keys):
    before_query = snapshot(policy)
    results = {}
    for label, (raw, context) in keys.items():
        restore(policy, before_query)
        answer = torch.full_like(policy.weight_updates, -1)
        for template in template_frames:
            frame = probe_frame(template, raw, context)
            action = policy.act(frame.flatten(1)[None].expand(policy.population, -1, -1))[..., 0]
            legal = frame[:, 0, READY].bool() & action.ge(FIRST_FINISH) & action.lt(N_ACTIONS)
            answer = torch.where(answer.lt(0) & legal, (action - FIRST_FINISH) % N_SYMBOLS, answer)
        results[label] = answer.cpu()
    restore(policy, before_query)
    return results


def one_batch(genomes, ids, seeds, device, source, native_tail):
    tape = batch_tape(seeds, device)
    _, birth = _stack_directed_genomes(genomes)
    policy = _DirectedRewardPolicy({k: v.to(device) for k, v in birth.items()}, tape.raw.size(1), device, "full")
    snapshots = []
    obs, actions = native_inputs(policy, tape, _receipt_table(device).half(), source,
                                 lambda: snapshots.append(snapshot(policy)))
    (before_teaching,) = snapshots
    finish = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & obs[..., READY].bool()
    native_covered, first = finish.max(1)
    pulse_tick = first[0] + 1
    eligible = native_covered[0] & pulse_tick.lt(obs.size(1))
    previous = ((actions[0].gather(0, first[0][None])[0] - FIRST_FINISH) % N_SYMBOLS)
    raw, context = tape.raw[source], tape.context[source]
    keys = dict(same=(raw, context), other_context=(raw, 1 - context), other_raw=((raw + 1) % N_SYMBOLS, context))
    actor_ids = torch.arange(tape.raw.size(1), device=device)
    targets = {k: tape.maps[actor_ids, ctx, sym] for k, (sym, ctx) in keys.items()}
    if not ((targets["same"] == tape.target[source] - FIRST_FINISH).all()
            and (targets["same"] != targets["other_context"]).all()
            and (targets["same"] != targets["other_raw"]).all()):
        raise RuntimeError("query key or target mismatch")
    receipt_at = pulse_tick.clamp_max(obs.size(1) - 1)[None]
    ack = obs[0, ..., _ACK].gather(0, receipt_at)[0].bool()
    rej = obs[0, ..., _REJ].gather(0, receipt_at)[0].bool()
    if ((~(ack ^ rej) | (ack != previous.eq(targets["same"]))) & eligible).any():
        raise RuntimeError("native receipt inconsistent with eliciting action")

    # All four arms hear the SAME prerecorded distractor observations, including both ports.
    # Associate their signed writes with the factual actor's first legal answers on these requests.
    ticks = torch.arange(obs.size(1), device=device)[None, :, None, None]
    factual_write_after = native_covered[:, None] & ticks.eq(first[:, None])
    if not native_tail:
        obs[1, 0, ..., RECEIPT] = 0
        obs[1, 0, ..., FEATURES_PER_SLOT + RECEIPT.start:FEATURES_PER_SLOT + RECEIPT.stop] = 0
    outcomes, changes = {}, {}
    for mode in MODES:
        restore(policy, before_teaching)
        changes[mode] = teaching(policy, obs[0], actions[0], pulse_tick, eligible, mode, native_tail).cpu()
        for delay in range(9):
            if delay in DELAYS:
                outcomes[mode, delay] = test_keys(policy, tape.public[source + 9], keys)
            if delay < 8:
                for tick, frame in enumerate(obs[delay + 1]):
                    if tick:
                        policy.pending.copy_(factual_write_after[delay + 1, tick - 1])
                    elif delay:
                        policy.pending.copy_(factual_write_after[delay, -1])
                    else:
                        policy.pending.zero_()
                    policy.act(frame)

    rows, pulses = [], []
    prev, ctx, raw_key, pulse = (v.cpu() for v in (previous, context, raw, pulse_tick))
    lookup = {k: v.cpu() for k, v in targets.items()}
    for i, a in eligible.nonzero().tolist():
        world_idx, actor = divmod(a, run.ACTORS)
        participant = dict(seed=seeds[world_idx], champion=ids[i], actor=actor, previous=int(prev[i, a]))
        pulses.append({**participant, "context": int(ctx[a]), "raw": int(raw_key[a]), "tick": int(pulse[i, a]),
                       "support_tick": (source * obs.size(1) + int(pulse[i, a])) % GATE_INTERVAL == 0,
                       "changed_weights": {m: int(changes[m][i, a]) for m in MODES}})
        for delay in DELAYS:
            for key in keys:
                rows.append({**participant, "delay": delay, "key": key, "target": int(lookup[key][a]),
                             "answers": {m: int(outcomes[m, delay][key][i, a]) for m in MODES}})
    return rows, pulses, int(native_covered[0].sum())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--batch-seeds", type=int, default=8)
    parser.add_argument("--source-request", type=int, default=SOURCE + 1)
    parser.add_argument("--native-tail", action="store_true",
                        help="Keep the factual READY and observations through the end of the teaching request.")
    args = parser.parse_args()
    if min(args.seeds, args.batch_seeds) < 1:
        parser.error("use a positive seed count and batch size")
    if not 1 <= args.source_request <= run.VISITS * N_SYMBOLS - 9:
        parser.error("source request must leave eight distractors and a query template")
    torch.set_num_threads(1)
    branch = run.OUT / "pair-01" / "downhill"
    generation = 1024
    audit = json.loads((branch / "audits" / f"{generation:04d}.json").read_text())
    ids = [audit["champions"][str(i)] for i in range(4)]
    bank = torch.load(branch / "generations" / f"{generation:04d}" / "elites.pt",
                      map_location="cpu", weights_only=True)["genomes"]
    genomes = [bank[k] for k in ids]
    del bank
    rows, pulses, covered = [], [], 0
    seeds = list(range(SEED, SEED + args.seeds))
    device = torch.device("cuda:0")
    with _matmul_precision(torch.float16, False):
        for idx in range(0, len(seeds), args.batch_seeds):
            block = seeds[idx:idx + args.batch_seeds]
            cells, events, count = one_batch(genomes, ids, block, device, args.source_request - 1, args.native_tail)
            rows.extend(cells)
            pulses.extend(events)
            covered += count
            print(f"{idx + len(block)}/{len(seeds)} seeds, {len(pulses)} matched events", flush=True)
    report = dict(generation=generation, champions=ids, seeds=seeds, actors_per_seed=run.ACTORS,
                  source_request=args.source_request, native_tail=args.native_tail,
                  opportunities=len(seeds) * run.ACTORS * len(ids),
                  source_covered=covered, analyzed_pulses=len(pulses), summary=summarize(rows),
                  pulses=pulses, rows=rows,
                  protocol=("Native prefix and factual teaching tail. " if args.native_tail else
                            "Native prefix and same immediate retirement for ACK and REJ. ") + "One pulse. "
                  "All later distractor observations and signed-write opportunities yoked to the native actor. "
                  "Distractor raw excludes both tested raw symbols. Same timing template for all unreinforced queries.")
    suffix = f"-q{args.source_request}" if args.source_request != SOURCE + 1 else ""
    suffix += "-native-tail" if args.native_tail else ""
    path = run.OUT / "audits" / f"iid-receipt-binding-{args.seeds}seeds{suffix}.json"
    _write(path, report)
    print(path, flush=True)


if __name__ == "__main__":
    main()
