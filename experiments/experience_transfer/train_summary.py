"""Collect audited training doses without interpreting heldout outcomes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import torch
from experiments.experience_transfer import train_transfer as t


def main():
    plan=t.checked();out=dict(passed=True,protocol_path=str(t.PLAN),protocol_sha256=t.sha(t.PLAN),
        source_sha256=t.sha(__file__),seeds={},comparison='related versus disjoint is matched total training and exact B inputs/feedback. Direct is a time-aligned familiar-D no-history-update reference with fewer total updates.',
        limits=['B training executes common teacher demonstrations; ability is assessed separately with free actions.',
                'Zero mapping overlap does not imply statistical independence.',
                'Direct history teacher_lanes arrays are unused shared RNG draws; labels=None and zero updates are authoritative.',
                'All worlds and numeric h continue. A/C B inputs are exact; direct can briefly differ in old pending non-B slot receipts.',
                'B NEW work counts exclude the separately reported unlabeled WAIT-boundary step.'])
    snapshots=('B000','Bfirst002','Bfirst008','B001','B002','B004','B008')
    for seed in t.SEEDS:
        folder=t.OUT/f'seed_{seed:03d}'
        audit=json.loads((folder/'training_independent_audit.json').read_text(encoding='utf-8'))
        command=json.loads((folder/'B_command_accounting.json').read_text(encoding='utf-8'))
        assert audit['passed'] and command['passed']
        audit_source=ROOT/'experiments/experience_transfer/train_audit.py'
        if t.sha(audit_source)!=audit['source_sha256']:
            audit_source=t.OUT/'audit-source-snapshots/train_audit-before-command-accounting.py'
        assert t.sha(audit_source)==audit['source_sha256'],'audit source provenance missing'
        spec=t.SEEDS[seed];cue=spec['B_cue'];arms={}
        for arm in t.ARMS:
            af=folder/arm;rows=[]
            B0=torch.load(af/'B000.pt',map_location='cpu',weights_only=False)
            for name in snapshots:
                cp=torch.load(af/(name+'.pt'),map_location='cpu',weights_only=False)
                rows.append(dict(snapshot=name,path=str(af/(name+'.pt')),sha256=t.sha(af/(name+'.pt')),
                    real_B_environment_ticks=cp['work']['environment_ticks']-B0['work']['environment_ticks'],
                    B_gradient_updates=cp['work']['gradient_updates']-B0['work']['gradient_updates'],
                    total_added_gradient_updates=cp['work']['gradient_updates'],
                    B_unique_feedback_positions=cp['work']['B_unique_label_positions'],
                    B_unique_nonwait_feedback=cp['work']['B_unique_nonwait_labels'],
                    trained_sequence_B_NEW=cp['work']['B_unique_new_commands'],
                    trained_sequence_B_REPLACE=cp['work']['B_unique_replace_commands'],
                    world_cumulative_B_NEW=int(cp['world']['context']['tensors']['family_new_counts'][...,cue].sum()),
                    world_cumulative_B_REPLACE=int(cp['world']['context']['tensors']['family_replace_counts'][...,cue].sum())))
            a=torch.load(af/'trajectory_history_006.pt',map_location='cpu',weights_only=False)['initial_world']['counts']
            b=torch.load(af/'history012.pt',map_location='cpu',weights_only=False)['world']['counts']
            counts={k:int(b[k].sum()-a[k].sum()) for k in ('matured_eligible','completed_matured','clean_completed_matured')}
            arms[arm]=dict(snapshots=rows,actual_native_last6_history_rounds_counts=counts,
                          command_accounting=command['arms'][arm],audit=audit['arms'][arm])
        assert audit['pairing']['disjoint']['B_public_input_unequal_float_positions']==0
        out['seeds'][str(seed)]=dict(spec=spec,arms=arms,pairing=audit['pairing'],
              online_pairing=json.loads((folder/'B_pairing_audit.json').read_text(encoding='utf-8')),
              independent_audit_path=str(folder/'training_independent_audit.json'),
              independent_audit_sha256=t.sha(folder/'training_independent_audit.json'),
              independent_audit_source=str(audit_source),independent_audit_source_sha256=t.sha(audit_source))
    out['exit_codes']=json.loads((t.OUT/'exit_codes.json').read_text(encoding='utf-8'))
    assert len(out['exit_codes'])==3 and all(v['exit_code']==0 for v in out['exit_codes'])
    t.write(t.OUT/'summary.json',out)
    print(json.dumps(dict(passed=True,path=str(t.OUT/'summary.json'),sha256=t.sha(t.OUT/'summary.json'))))


if __name__=='__main__':main()
