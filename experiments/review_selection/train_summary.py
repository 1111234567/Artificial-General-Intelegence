"""Audit all logged choices and summarize fixed-budget review training."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.experience_transfer.train_transfer import sha

OUT=ROOT/'data/review-selection/train-v1'


def main():
    reports=[];audits=[]
    for seed in range(3):
        folder=OUT/f'seed_{seed:03d}'
        complete=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        assert complete['passed']
        bank=torch.load(ROOT/f'data/review-selection/bank-v1/seed_{seed:03d}/bank.pt',map_location='cpu',weights_only=False)
        all_rows={arm:[json.loads(line) for line in (folder/arm/'selections.jsonl').read_text().splitlines()]
                  for arm in ('uniform','conflict','error')}
        for arm,rows in all_rows.items():
            counts=torch.zeros(256,dtype=torch.long)
            for index,row in enumerate(rows):
                ids=torch.tensor(row['candidates']);selected=torch.tensor(row['selected'])
                assert len(ids)==128 and len(torch.unique(ids))==128
                assert torch.equal(torch.bincount(bank['raw'][ids],minlength=4),torch.full((4,),32))
                assert row['candidates']==all_rows['uniform'][index]['candidates']
                rng=np.random.default_rng(1140000+seed*10000+index)
                expected=np.concatenate([rng.choice(torch.where(bank['raw']==r)[0].numpy(),32,replace=False) for r in range(4)])
                rng.shuffle(expected);assert ids.tolist()==expected.tolist()
                if arm=='uniform':
                    local=np.random.default_rng(2140000+seed*10000+index).choice(128,64,replace=False)
                    chosen=ids[local]
                elif arm=='conflict':chosen=ids[bank['conflict'][ids].bool()]
                else:chosen=ids[torch.argsort(torch.tensor(row['scores']),descending=True,stable=True)[:64]]
                assert torch.equal(chosen,selected)
                assert row['reused_requests']==int((counts[selected]>0).sum())
                counts[selected]+=1
                assert row['unique_so_far']==int((counts>0).sum())
                assert row['candidate_count']==128 and row['scoring_rows']==384
                assert row['gradient_rows']==192 and row['gradient_nonwait_rows']==128
                assert row['gradient_tensor']==[128,64,2,11]
                assert int(bank['mask'][:,selected].sum())==192
                assert row['raw_counts']==torch.bincount(bank['raw'][selected],minlength=4).tolist()
                assert row['conflict_count']==int(bank['conflict'][selected].sum())
            assert len(rows)==256 and int(counts.sum())==16384
            cp=torch.load(folder/arm/'B008.pt',map_location='cpu',weights_only=False)
            assert torch.equal(cp['review_selection']['selection_counts'],counts)
            original=torch.load(ROOT/f'data/experience-transfer/training-v1/seed_{seed:03d}/related/B000.pt',map_location='cpu',weights_only=False)
            baseline=torch.load(ROOT/f'data/experience-transfer/training-v1/seed_{seed:03d}/related/B008.pt',map_location='cpu',weights_only=False)
            work=cp['work'];assert work['gradient_updates']-original['work']['gradient_updates']==2048
            assert work['B_gradient_updates']==1792 and work['A_gradient_updates']==256
            assert work['A_gradient_label_positions']==49152
            assert work['A_score_label_comparisons']==98304 and work['A_score_forward_lane_ticks']==4194304
            assert baseline['work']['gradient_updates']-original['work']['gradient_updates']==2048
            conflict=bank['conflict'].bool();components=np.asarray([r['selected_component_mean'] for r in rows])
            candidate_components=np.asarray([r['candidate_component_mean'] for r in rows])
            row=dict(seed=seed,arm=arm,total_updates=2048,B_updates=1792,A_updates=256,
                baseline_B_updates=2048,review_gradient_rows=49152,B_gradient_rows=work['B_gradient_label_positions'],
                baseline_B_gradient_rows=baseline['work']['B_gradient_label_positions'],
                scoring_label_comparisons=98304,scoring_forward_lane_ticks=4194304,
                selected_requests=16384,unique_requests=int((counts>0).sum()),unique_conflict=int(((counts>0)&conflict).sum()),
                unique_shared=int(((counts>0)&~conflict).sum()),bank_size=256,
                conflict_selected_fraction=float(counts[conflict].sum()/counts.sum()),
                raw_selected_counts=[int(counts[bank['raw']==r].sum()) for r in range(4)],
                reused_fraction=sum(r['reused_requests'] for r in rows)/16384,max_request_uses=int(counts.max()),
                selected_mean_CE=float(np.mean([r['selected_mean_score'] for r in rows])),
                candidate_mean_CE=float(np.mean([r['candidate_mean_score'] for r in rows])),
                selected_BEGIN_WAIT_FINISH_CE=components.mean(0).tolist(),candidate_BEGIN_WAIT_FINISH_CE=candidate_components.mean(0).tolist(),
                actual_B_NEW=work['B_actual_NEW_including_boundary'],original_work_B_NEW=work['B_unique_new_commands'],
                checkpoint_sha256=sha(folder/arm/'B008.pt'),selection_log_sha256=sha(folder/arm/'selections.jsonl'))
            reports.append(row)
        assert len({r['B_gradient_rows'] for r in reports if r['seed']==seed})==1
        early=complete['public_stream_audit']['first_epoch_comparison']
        audits.append(dict(seed=seed,all256choices_per_arm_exact=True,candidates_shared_exact=True,all_feedback_counts_exact=True,
            total_updates_equal_baseline=True,first_gradient_replay_exact=complete['first_step_audit'],early_snapshots=early))
    output=dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),reports=reports,audits=audits,
        qualification='Equal total updates across baseline/review; equal candidate scoring, tensor shapes and192 A supervision rows among review arms. Baseline has256 B updates per round versus224 B plus32 A; B supervised row counts therefore differ.')
    (OUT/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,path=str(OUT/'summary.json'),reports=reports)),flush=True)


if __name__=='__main__':main()
