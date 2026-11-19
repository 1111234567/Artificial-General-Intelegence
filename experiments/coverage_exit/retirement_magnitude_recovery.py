"""Offline next-update restore audit after actual formal magnitude retirements.

These extra updates are diagnostic clones only. They neither replace the frozen
128-update endpoints nor supply any training/evaluation score to the study.
"""
import json
from pathlib import Path
import torch
from experiments.continuity.run import state_digest, restore_rng_state
from experiments.undirected.run import _atomic_json, file_sha256
from experiments.coverage_exit.retirement_magnitude import NativeTrainer
from experiments.coverage_exit.retirement_magnitude_study import OUT, read_plan
from experiments.coverage_exit.fast_window import install_window


def audit():
    torch.set_num_threads(1);plan=read_plan();records=[]
    for seed in range(3):
        result_path=OUT/f'seed_{seed:03d}/result.json'
        result=json.loads(result_path.read_text());cp=Path(result['checkpoint'])
        cp_sha=file_sha256(cp)
        if cp_sha!=result['checkpoint_sha256']:raise ValueError('endpoint bytes changed')
        continuing=NativeTrainer.from_checkpoint(cp,device='cpu');install_window(continuing)
        original=continuing.semantic_state();origin_hash=state_digest(original)
        if origin_hash!=result['endpoint_semantic_hash']:raise ValueError('actual retirement endpoint differs')
        if continuing.operator.cumulative_deaths<=0:raise ValueError('audit requires actual retired lifetime history')
        saved=continuing.state_dict()
        restored=NativeTrainer.from_state_dict(saved);install_window(restored)
        if state_digest(restored.semantic_state())!=origin_hash:raise ValueError('full restore differs')
        first=continuing.train_update();first_hash=state_digest(continuing.semantic_state())
        restore_rng_state(saved['rng'])
        second=restored.train_update();second_hash=state_digest(restored.semantic_state())
        if state_digest(first)!=state_digest(second) or first_hash!=second_hash:raise ValueError('next real update differs after restore')
        if file_sha256(cp)!=cp_sha:raise AssertionError('offline audit changed formal checkpoint')
        records.append(dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=cp_sha,source_semantic_hash=origin_hash,
            science_endpoint_update=original['updates'],science_endpoint_deaths=original['operator']['cumulative_deaths'],
            audit_update=continuing.updates,audit_next_update_deaths=first['update_metrics']['structure']['deaths'],
            bitwise_next_point_equal=True,bitwise_next_complete_semantic_equal=True,
            next_point_hash=state_digest(first),next_semantic_hash=first_hash,source_unchanged=True))
    read_plan()
    value=dict(format='magnitude-actual-retirement-offline-recovery-v1',audit='passed',protocol_hash=plan['protocol_hash'],
        source_code=dict(path=str(Path(__file__).resolve()),sha256=file_sha256(__file__)),records=records,
        scope='Three independently loaded actual retirement endpoints; compare one continuing cloned update with save/restore of that full state. Includes execution of pending proposal16. Extra updates excluded from science endpoints, budgets, behavior and effect summaries; original checkpoints unchanged.')
    value['audit_hash']=state_digest(value);_atomic_json(OUT/'actual_retirement_recovery.json',value);return value


if __name__=='__main__':print(json.dumps(audit(),indent=2))
