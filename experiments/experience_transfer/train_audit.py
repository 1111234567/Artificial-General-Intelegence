"""Independent archive replay and budget checks for the paired history run."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from experiments.experience_transfer import train_transfer as t
from experiments.continuity.run import state_digest


def same(a,b):
    return state_digest(a)==state_digest(b)


def replay(path,table,B_cue=None):
    d=torch.load(path,map_location='cpu',weights_only=False)
    with t.mapping_scope(table):
        w=t.worlds.ContextWorld.from_state_dict(d['initial_world'],device='cpu')
        for tick in range(len(d['obs'])):
            obs=w.observe()
            assert torch.equal(obs,d['obs'][tick]),f'public observation replay mismatch {path}/{tick}'
            if d['labels'] is not None:
                labels=w.oracle('midpoint')
                if B_cue is not None:
                    mask=w.current_family==B_cue
                    assert torch.equal(mask,d['mask'][tick])
                    labels=torch.where(mask,labels,torch.zeros_like(labels))
                    assert torch.equal(labels,d['actions'][tick]),'B action is not common valid teacher'
                assert torch.equal(labels,d['labels'][tick]),f'label provenance mismatch {path}/{tick}'
            w.act(d['actions'][tick])
        return w.state_dict()


def actual_h(model,observations,h):
    policy=deepcopy(model).cpu().eval()
    with torch.no_grad():
        for ob in observations:_,h=policy.step(ob,h)
    return h


def reproduce_early_updates(folder,trajectory,boundary):
    """Recompute the actual online h and first8 Adam updates from B0."""
    model,_,_,cp=t.load_checkpoint(folder/'B000.pt','cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=cp['config']['lr'])
    optimizer.load_state_dict(deepcopy(cp['optimizer']))
    h=boundary['hidden'].clone()
    checks={}
    for block in range(8):
        sl=slice(block*128,(block+1)*128)
        obs,labels,mask=(trajectory[k][sl] for k in ('obs','labels','mask'))
        anchor=h.clone();h=actual_h(model,obs,h)
        logits,_=model(obs.cuda(),anchor.cuda());labels,mask=labels.cuda(),mask.cuda()
        weights=torch.ones(11,device='cuda');weights[0]=cp['config']['wait_weight']
        losses=F.cross_entropy(logits.flatten(0,2),labels.flatten(),weight=weights,reduction='none').reshape_as(labels)
        loss=(losses*mask).sum()/(weights[labels]*mask).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if block+1 in (2,8):
            saved=torch.load(folder/f'Bfirst{block+1:03d}.pt',map_location='cpu',weights_only=False)
            exact_h=torch.equal(h,saved['hidden'])
            exact_model=same(model.state_dict(),saved['model'])
            exact_optimizer=same(optimizer.state_dict(),saved['optimizer'])
            assert exact_h and exact_model and exact_optimizer,'first online Adam/public h replay differs'
            checks[str(block+1)]=dict(hidden_exact=True,model_exact=True,Adam_exact=True)
    return checks


def command_accounting(seed):
    """Separate unlabeled WAIT-boundary commands from trained-sequence cases."""
    folder=t.OUT/f'seed_{seed:03d}';cue=t.SEEDS[seed]['B_cue']
    boundary=torch.load(folder/'B_boundary.pt',map_location='cpu',weights_only=False)
    rows={}
    for arm in t.ARMS:
        b0=torch.load(folder/arm/'B000.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/arm/'B008.pt',map_location='cpu',weights_only=False)
        count=lambda cp,key:int(cp['context']['tensors'][key][...,cue].sum())
        initial={key:count(b0['world'],key) for key in ('family_new_counts','family_replace_counts')}
        wait_new=count(boundary[arm]['world'],'family_new_counts')-initial['family_new_counts']
        wait_replace=count(boundary[arm]['world'],'family_replace_counts')-initial['family_replace_counts']
        total_new=count(last['world'],'family_new_counts')-initial['family_new_counts']
        total_replace=count(last['world'],'family_replace_counts')-initial['family_replace_counts']
        work_new=last['work']['B_unique_new_commands'];work_replace=last['work']['B_unique_replace_commands']
        assert total_new-work_new==wait_new and total_replace-work_replace==wait_replace
        rows[arm]=dict(B0_world_B_new=initial['family_new_counts'],B0_world_B_replace=initial['family_replace_counts'],
                      unlabeled_boundary_B_new=wait_new,unlabeled_boundary_B_replace=wait_replace,
                      endpoint_world_B_new=total_new,endpoint_world_B_replace=total_replace,
                      trained_sequence_work_B_new=work_new,trained_sequence_work_B_replace=work_replace,
                      endpoint_minus_work_B_new=total_new-work_new,endpoint_minus_work_B_replace=total_replace-work_replace)
    assert all(v==rows['related'] for v in rows.values()),'three-arm command accounting mismatch'
    result=dict(passed=True,seed=seed,B_cue=cue,arms=rows,source_sha256=t.sha(__file__),protocol_sha256=t.sha(t.PLAN),
                axis='work B_unique_new_commands counts only NEW commands inside trained sequences; boundary WAIT observation has no loss and is reported separately')
    t.write(folder/'B_command_accounting.json',result)
    return result


def audit(seed):
    torch.set_num_threads(1)
    plan=t.checked();folder=t.OUT/f'seed_{seed:03d}'
    origin=torch.load(plan['origins'][str(seed)]['path'],map_location='cpu',weights_only=False)
    result=dict(seed=seed,passed=True,origin=plan['origins'][str(seed)],arms={},pairing={})
    histories={};Bs={}
    for arm in t.ARMS:
        af=folder/arm;table=t.mapping_table(seed,arm)
        first=torch.load(af/'origin.pt',map_location='cpu',weights_only=False)
        assert same(first['model'],origin['model']) and same(first['optimizer'],origin['optimizer'])
        assert torch.equal(first['hidden'],origin['hidden'])
        source_world=origin['world'].copy();source_world['context']=dict(source_world['context'],mappings=table)
        assert same(first['world'],source_world),'initial world differs beyond never-issued mapping rows'
        history=[];end=None
        for index in range(t.HISTORY_ROUNDS):
            p=af/f'trajectory_history_{index:03d}.pt';d=torch.load(p,map_location='cpu',weights_only=False)
            if end is not None:assert same(end,d['initial_world']),'history real-world continuity broken'
            end=replay(p,table)
            if arm!='direct':
                teacher=d['teacher_lanes'][None,:,None].expand_as(d['labels'])
                assert torch.equal(d['actions'][teacher],d['labels'][teacher])
            else:assert d['labels'] is None
            history.append(d)
        h_checks=[]
        # Known frozen weights before rounds1/5/9 independently recover actual
        # carried h into rounds2/6/10. Direct has unmodified weights throughout.
        for index in (range(11) if arm=='direct' else (0,4,8)):
            cp_path=af/'origin.pt' if index==0 or arm=='direct' else af/f'history{index:03d}.pt'
            hm,_,_,_=t.load_checkpoint(cp_path,'cpu')
            hn=actual_h(hm,history[index]['obs'],history[index]['anchor'])
            assert torch.equal(hn,history[index+1]['anchor']),'numeric h reset/reconstruction between history rounds'
            h_checks.append(index+1)
        final_history=torch.load(af/'history012.pt',map_location='cpu',weights_only=False)
        B0=torch.load(af/'B000.pt',map_location='cpu',weights_only=False)
        assert same(end,final_history['world']) and same(end,B0['world'])
        assert same(B0['optimizer'],final_history['optimizer']) and same(B0['model'],final_history['model'])
        assert torch.equal(B0['hidden'],final_history['hidden'])
        if arm=='direct':
            assert same(B0['model'],origin['model']) and same(B0['optimizer'],origin['optimizer'])
            assert B0['work']['gradient_updates']==0
        else:assert B0['work']['gradient_updates']==3072
        boundary=torch.load(folder/'B_boundary.pt',map_location='cpu',weights_only=False)[arm]
        with t.mapping_scope(table):
            bw=t.worlds.ContextWorld.from_state_dict(B0['world'],device='cpu')
            bw.set_family(t.SEEDS[seed]['B_cue'])
            bw.set_regime(duration_range=(1.,16.),gap_range=(.5,32.),name='experience-transfer/B')
            ob=bw.observe();bw.act(torch.zeros((64,2),dtype=torch.long))
            assert same(bw.state_dict(),boundary['world'])
        model,_,h,_=t.load_checkpoint(af/'B000.pt','cpu')
        with torch.no_grad():_,expected_h=model.step(ob,h)
        assert torch.equal(expected_h,boundary['hidden']),'boundary h not actual public WAIT continuation'
        B=[];end=boundary['world']
        for index in range(t.B_ROUNDS):
            p=af/f'trajectory_B_{index:03d}.pt';d=torch.load(p,map_location='cpu',weights_only=False)
            assert same(end,d['initial_world']),'B world continuity broken'
            end=replay(p,table,t.SEEDS[seed]['B_cue'])
            B.append(d)
        final=torch.load(af/'B008.pt',map_location='cpu',weights_only=False)
        assert same(end,final['world'])
        nlabels=sum(int(d['mask'].sum()) for d in B)
        nnonwait=sum(int((d['labels']!=0).sum()) for d in B)
        assert final['work']['B_unique_label_positions']==nlabels
        assert final['work']['B_unique_nonwait_labels']==nnonwait
        assert final['work']['B_gradient_label_positions']==nlabels*32
        assert final['work']['old_history_replay_positions']==0
        assert final['work']['gradient_updates']-B0['work']['gradient_updates']==2048
        for name,nticks,nupdates in [('Bfirst002',256,2),('Bfirst008',1024,8),('B001',1024,256),('B002',2048,512),('B004',4096,1024),('B008',8192,2048)]:
            cp=torch.load(af/(name+'.pt'),map_location='cpu',weights_only=False)
            assert cp['work']['environment_ticks']-B0['work']['environment_ticks']==nticks+1
            assert cp['work']['gradient_updates']-B0['work']['gradient_updates']==nupdates
            assert cp['world']['context']['mappings']==table
            with t.mapping_scope(table):
                restored=t.worlds.ContextWorld.from_state_dict(cp['world'],device='cpu')
                assert same(restored.state_dict(),cp['world'])
        histories[arm]=history;Bs[arm]=B
        result['arms'][arm]=dict(origin_weight_optimizer_hidden_world_exact=True,all_public_observations_and_labels_replayed=True,
            full_world_continuity_exact=True,boundary_hidden_recomputed_exact=True,feedback_positions=nlabels,
            nonwait_feedback_positions=nnonwait,B_updates=2048,final_work=final['work'],
            history_h_continuity_independently_checked_after_rounds=h_checks,
            first_B_online_updates_reproduced=reproduce_early_updates(af,B[0],boundary))
    for arm in t.ARMS[1:]:
        for a,b in zip(histories['related'],histories[arm]):
            ao=a['obs'].reshape(-1,64,2,28);bo=b['obs'].reshape(-1,64,2,28)
            assert torch.equal(ao[...,:11],bo[...,:11]),'external cases not matched during history alignment'
            if arm=='disjoint':
                assert torch.equal(ao[...,24:],bo[...,24:])
                assert torch.equal(a['teacher_lanes'],b['teacher_lanes'])
        unequal=0
        for a,b in zip(Bs['related'],Bs[arm]):
            for key in ('labels','actions','mask'):assert torch.equal(a[key],b[key]),'B training budget/teacher stream mismatched'
            unequal+=int((a['obs']!=b['obs']).sum())
        result['pairing'][arm]=dict(all_history_external_cases_exact=True,all_B_labels_masks_actions_exact=True,
                                  B_public_input_unequal_float_positions=unequal)
    result['B_command_accounting']=command_accounting(seed)
    result['source_sha256']=t.sha(__file__);result['protocol_sha256']=t.sha(t.PLAN)
    t.write(folder/'training_independent_audit.json',result)
    print(result,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);audit(p.parse_args().seed)
