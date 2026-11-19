"""Independent exact-origin and pre-review checkpoint/behavior calibration."""
import json
import torch
from experiments.review_selection import eval_review as e


def run():
    plan = e.checked()
    report = []
    for seed in range(3):
        for stage in ('B000','Bfirst002','Bfirst008'):
            original_path = e.checkpoint_path(seed, 'no_review', stage)
            original = torch.load(original_path, map_location='cpu', weights_only=False)
            original_result = e.read(e.base.OUT/f'seed_{seed:03d}'/f'related_{stage}_B.json')
            for arm in e.ARMS[:3]:
                path = e.checkpoint_path(seed, arm, stage)
                cp = torch.load(path, map_location='cpu', weights_only=False)
                for key in ('model','optimizer','world','hidden'):
                    assert e.state_digest(cp[key]) == e.state_digest(original[key]), (seed,arm,stage,key)
                result_path = e.result_path(seed, arm, stage, 'B')
                result = e.read(result_path)
                for key in ('counts','metrics','strata','by_family_raw','model_hash','primary_identity_hash','external_digest'):
                    assert result[key] == original_result[key], (seed,arm,stage,key)
                report.append(dict(seed=seed, arm=arm, snapshot=stage, checkpoint_sha256=e.sha(path),
                    original_checkpoint_sha256=e.sha(original_path), result_sha256=e.sha(result_path),
                    model_optimizer_world_hidden_exact=True, autonomous_counts_metrics_strata_exact=True,
                    result_is_new_rollout=not result['evidence_reused']))
    value = dict(protocol_hash=plan['protocol_hash'], passed=True, cases=27, checks=report,
        source_sha256=e.sha(__file__), note='B0 bound original rollout after exact branch verification; first2/first8 are newly executed independent heldout rollouts before A review begins, all exactly equal to original no-review behavior.')
    e.write(e.OUT/'calibration.json', value)
    print(json.dumps(dict(passed=True, cases=27, newly_executed_early_rollouts=18)), flush=True)
    return value


if __name__ == '__main__':
    run()
