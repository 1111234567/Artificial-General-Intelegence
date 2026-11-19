"""Second, vectorized reconstruction of the autonomous evaluation evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from experiments.learn_remember import evaluate_gru as e
from experiments.learn_remember.gru_train import load_checkpoint


def vector_counts(frames, initial, regime, fork_tick, raw_group=(0, 1, 2, 3)):
    # Maturation is emitted before this tick's NEW/REPLACE. The preceding
    # retained slot metadata therefore belongs to the matured identity.
    previous_regime = np.concatenate([e.array(initial['tensors']['current_regime_id'])[None],
                                     frames['current_regime_id'][:-1]])
    previous_raw = np.concatenate([e.array(initial['context']['tensors']['raw_payload'])[None],
                                  frames['raw_payload'][:-1]])
    primary = frames['matured_mask'] & (frames['matured_request_id'] >= 0) & (
        frames['matured_request_id'] // initial['spec']['n_jobs'] > fork_tick) & (previous_regime == regime)
    primary &= np.isin(previous_raw, raw_group)
    enabled = np.arange(initial['spec']['n_jobs'])[None, None] < initial['spec']['active_jobs']
    idle = enabled & ~frames['active'] & np.isin(frames['raw_payload'], raw_group)
    actions = frames['actions']
    return dict(matured=int(primary.sum()),
                completed=int((primary & frames['matured_completed']).sum()),
                clean_completed=int((primary & frames['matured_clean_completed']).sum()),
                idle_opportunities=int(idle.sum()), idle_actions=int((idle & (actions != 0)).sum()),
                idle_begin=int((idle & (actions > 0) & (actions < 3)).sum()),
                idle_finish=int((idle & (actions >= 3)).sum()))


@torch.inference_mode()
def audit(*, require_complete=False, replay=True):
    torch.set_num_threads(1)
    first = e.audit()
    plan = e.checked()
    checkpoints = {}
    replayed = []
    seen = set()
    independent_forks = 0
    for path in sorted(e.OUT.glob('seed_*/*_family_?.json')):
        record = e.read(path)
        if record['source'] != 'common':
            continue
        seen.add((record['seed'], record['stage'], record['condition'], record['family']))
        states = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        with np.load(path.with_suffix('.npz')) as archive:
            frames = {key: archive[key] for key in archive.files}
        args = (frames, states['initial_world'], record['evaluation_regime_id'], record['fork_tick'])
        for group, claimed in [((0, 1, 2, 3), record['counts']), ((1, 3), record['unique_raw_1_3'])] + [
            ((raw,), record['by_raw'][str(raw)]) for raw in range(4)]:
            for key, value in vector_counts(*args, raw_group=group).items():
                assert claimed.get(key, 0) == value, (path, group, key)
        common = e.common_stream(record['seed'])
        assert e.state_digest(states['initial_world']) == common['world_hash']
        spec = states['initial_world']['spec']
        assert spec['seed'] == 100000 + record['seed'] and spec['actor_ids'] == tuple(range(10000, 10008))
        checkpoint_path = record['checkpoint_path']
        if checkpoint_path not in checkpoints:
            model, world, hidden, checkpoint = load_checkpoint(checkpoint_path)
            model.eval()
            assert checkpoint['config']['seed'] == record['seed']
            assert world.seed != spec['seed'] and set(world.actor_ids).isdisjoint(spec['actor_ids'])
            warmed = e.warmup(model, common)
            checkpoints[checkpoint_path] = model, warmed
        model, warmed = checkpoints[checkpoint_path]
        assert torch.equal(warmed, states['initial_h'])
        if replay and record['family'] == 3:
            state = states['initial_h'].clone()
            for index, obs in enumerate(frames['obs']):
                logits, state = model.step(torch.from_numpy(obs), state, .25)
                assert np.array_equal(e.array(logits.argmax(-1)), frames['actions'][index]), (path, index)
            assert torch.equal(state, states['end_h']), path
            replayed.append(str(path))
        independent_forks += 1
    expected = {(seed, stage, condition, family) for seed in plan['seeds'] for stage in plan['stages']
                for condition in plan['conditions'] for family in plan['families']}
    missing = sorted(expected - seen)
    if require_complete:
        assert not missing, missing
    result = dict(format='learn-remember-independent-array-audit-v1', protocol_hash=plan['protocol_hash'],
        first_audit=first, common_forks=independent_forks, expected_common_forks=len(expected), missing=missing,
        vectorized_counts_verified=True, checkpoint_seed_and_heldout_world_disjointness_verified=True,
        matched_common_hidden_replay_verified=True, model_action_replays=replayed,
        complete=not missing, sources={str(Path(__file__).resolve()): e.sha(__file__)})
    e.write(e.OUT / 'independent-audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--no-replay', action='store_true')
    args = parser.parse_args()
    result = audit(require_complete=args.require_complete, replay=not args.no_replay)
    print(json.dumps({key: result[key] for key in ('common_forks', 'expected_common_forks', 'complete')}))


if __name__ == '__main__':
    main()
