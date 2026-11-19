"""Finalize the complete preregistered evidence; no outcome-based stopping."""
import json
import time

import torch

from experiments.experience_transfer import eval_transfer as e, eval_report


def expected_paths():
    for seed in range(3):
        for arm in e.CURVE_ARMS:
            for stage in e.SNAPSHOTS:
                scenarios=('B','history','mixed') if arm in e.ARMS and stage in ('B000','B008') else ('B',)
                for scenario in scenarios:
                    yield e.OUT/f'seed_{seed:03d}'/f'{arm}_{stage}_{scenario}.json'
        yield e.OUT/f'seed_{seed:03d}'/'short_origin_B000_B.json'


def finalize():
    torch.set_num_threads(1)
    paths=list(expected_paths())
    assert len(paths)==90
    while not all(p.exists() for p in paths):
        time.sleep(5)
    binding=[]
    for path in paths:
        row=e.read(path)
        cp=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=False)
        assert cp['transfer']['seed']==row['seed']
        assert cp['transfer']['arm']==('direct' if row['arm']=='short_origin' else row['arm'])
        assert cp['mapping_table']==row['mapping_table']
        assert e.state_digest(cp['model'])==row['model_hash']
        assert cp['work']==row['checkpoint_work']
        binding.append(dict(result=str(path),checkpoint_sha256=row['checkpoint_sha256'],
                            arm=row['arm'],seed=row['seed'],snapshot=row['snapshot']))
    summary=e.summarize()
    assert summary['forks']==90
    eval_report.run()
    audit=e.audit()
    assert audit['verified'] and audit['forks']==90 and audit['matched_groups']==9
    value=dict(complete=True,forks=90,protocol_hash=e.checked()['protocol_hash'],
        checkpoint_bindings=binding,finalizer_sha256=e.sha(__file__),
        report_source_sha256=e.sha(eval_report.__file__))
    e.write(e.OUT/'completion.json',value)
    print(json.dumps(dict(event='evaluation_complete',forks=90,protocol_hash=value['protocol_hash'])),flush=True)


if __name__=='__main__':
    finalize()
