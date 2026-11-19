"""Execute the already frozen evaluation as atomic checkpoints become ready."""
import argparse
import json
import time

import torch

from experiments.experience_transfer import eval_transfer as e


def run(seed):
    torch.set_num_threads(1)
    pending = [(arm, stage) for arm in e.CURVE_ARMS for stage in e.SNAPSHOTS]
    pending.append(('short_origin', 'B000'))
    while pending:
        progressed = False
        for arm, stage in list(pending):
            direct = arm == 'short_origin'
            checkpoint = e.TRAIN / f'seed_{seed:03d}' / ('direct' if direct else arm) / ('origin.pt' if direct else stage+'.pt')
            table_arm = 'related' if arm in ('short_origin','direct') else arm
            scenarios = ('B','history','mixed') if arm in e.ARMS and stage in ('B000','B008') else ('B',)
            if all((e.OUT / f'seed_{seed:03d}' / f'{arm}_{stage}_{scenario}.json').exists() for scenario in scenarios):
                pending.remove((arm,stage))
                continue
            if not checkpoint.with_suffix('.json').exists() or not all((e.OUT / f'reference_seed_{seed:03d}_{table_arm}_{scenario}.json').exists() for scenario in scenarios):
                continue
            e.run_checkpoint(checkpoint, seed, arm, stage, direct=direct)
            pending.remove((arm,stage))
            progressed = True
        if not progressed and pending:
            time.sleep(5)
    print(json.dumps(dict(event='seed_complete',seed=seed)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=range(3),required=True)
    run(p.parse_args().seed)
