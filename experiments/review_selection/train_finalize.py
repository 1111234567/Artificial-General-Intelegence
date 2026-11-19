"""Finalize complete training with the separately corrected arithmetic-exact audit."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from experiments.review_selection import train_review as training
from experiments.review_selection import train_replay_check as replay
from experiments.review_selection import train_summary


def main():
    plan=training.checked()
    correction=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_hash=plan['protocol_hash'],
        type='Independent audit arithmetic correction; training, source and frozen protocol unchanged',
        original_auditor=str(ROOT/'experiments/review_selection/train_audit.py'),
        original_auditor_sha256=training.t.sha(ROOT/'experiments/review_selection/train_audit.py'),
        corrected_auditor=str(Path(replay.__file__).resolve()),corrected_auditor_sha256=training.t.sha(replay.__file__),
        correction='Original independent auditor divides summed NLL by a per-example GPU int tensor equal to3, while frozen trainer divides by scalar3. GPU float32 division differs by up to1ULP on seed0(max1.1920928955078125e-7). Corrected auditor uses scalar3, then requires bitwise scores, choices, model and Adam equality. No score, selection, training data, gradient or checkpoint is changed.',
        bank_qualification='Eligibility requires true in-process WAIT and first legal midpoint teacher FINISH in actual old own-action trajectories. Requests already autonomously completed before midpoint are excluded; qualification is action-dependent. Bank is not an unbiased uniform sample of all historical requests.')
    training.t.write(training.OUT/'audit_correction.json',correction)
    for seed in range(3):
        folder=training.OUT/f'seed_{seed:03d}'
        stream=json.loads((folder/'stream_audit.json').read_text(encoding='utf-8'))
        assert len(stream['rounds'])==8
        work={}
        for arm in training.ARMS:
            for name in training.SNAPSHOTS:
                path=folder/arm/(name+'.pt')
                assert path.is_file()
                metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                assert metadata['sha256']==training.t.sha(path)
            checkpoint=torch.load(folder/arm/'B008.pt',map_location='cpu',weights_only=False)
            assert checkpoint['review_selection']['protocol_hash']==plan['protocol_hash']
            assert checkpoint['round']==8 and checkpoint['review_selection']['review_updates']==256
            assert checkpoint['work']['A_gradient_updates']==256 and checkpoint['work']['B_gradient_updates']==1792
            work[arm]=checkpoint['work']
        reports=replay.audit(seed)
        training.t.write(folder/'complete.json',dict(passed=True,seed=seed,protocol_hash=plan['protocol_hash'],
            completion_method='Full B008 checkpoint/hash/budget verification plus corrected independent first-step replay',
            audit_correction_sha256=training.t.sha(training.OUT/'audit_correction.json'),
            work=work,first_step_audit=reports,public_stream_audit=stream))
    training.checked()
    train_summary.main()


if __name__=='__main__':main()
