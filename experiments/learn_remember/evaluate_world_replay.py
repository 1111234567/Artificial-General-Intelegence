"""Re-execute saved actions in the unmodified world, independent of the model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.learn_remember import evaluate_gru as e
from experiments.lifelong.evaluate import CONDITIONS
from experiments.lifelong.world import ContextWorld


def replay():
    torch.set_num_threads(1)
    e.checked()
    verified = []
    # Acquisition end, withdrawal end and matched review end; both evaluation
    # ranges and every seed. This audit subset is selected after evaluation by
    # stage meaning, not by success, score, raw symbol, error type or trajectory.
    for seed in range(3):
        for stage in ('wide', 'withdrawal', 'review'):
            for condition in ('short', 'long_both'):
                path = e.OUT / f'seed_{seed:03d}' / f'{stage}_common_{condition}_family_3.json'
                row = e.read(path)
                states = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
                world = ContextWorld.from_state_dict(states['initial_world'])
                world.set_family(3)
                world.set_regime(**CONDITIONS[condition], name=f'evaluation/3/{condition}', dt=.25)
                with np.load(path.with_suffix('.npz')) as frames:
                    for tick in range(row['steps']):
                        observed = world.observe()
                        assert np.array_equal(e.array(observed), frames['obs'][tick]), (path, tick, 'obs')
                        for field in ('active', 'request_id', 'current_regime_id', 'raw_payload', 'current_family',
                                      'issued_time', 'duration', 'event_kind', 'event_eligible', 'matured_mask',
                                      'matured_request_id', 'matured_completed', 'matured_clean_completed'):
                            assert np.array_equal(e.array(getattr(world, field)), frames[field][tick]), (path, tick, field)
                        world.act(torch.from_numpy(frames['actions'][tick]))
                        assert np.array_equal(e.array(world.last_result), frames['result'][tick]), (path, tick, 'result')
                assert e.state_digest(world.state_dict()) == row['end_world_hash']
                verified.append(dict(path=str(path), record_sha256=e.sha(path), steps=row['steps']))
    result = dict(format='learn-remember-independent-world-replay-v1', forks=len(verified),
                  ticks=sum(row['steps'] for row in verified), verified=True, records=verified,
                  scope='All three seeds; family3 short/long; acquisition end, withdrawal end, review end.',
                  source_sha256=e.sha(__file__))
    e.write(e.OUT / 'world-replay-audit.json', result)
    print(json.dumps({key: result[key] for key in ('forks', 'ticks', 'verified')}))
    return result


if __name__ == '__main__':
    replay()
