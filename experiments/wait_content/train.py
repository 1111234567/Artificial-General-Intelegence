"""Matched recurrent learning of content at END versus processing WAIT.

Only the timing of a fixed auxiliary eight-class content CE differs.  The
shared observations, all-position action targets, graph support, trainable
parameters, fresh optimizer and continuous 128-tick TBPTT budget are identical.
The inherited group-first BEGIN/conditional-FINISH policy remains the deployed
policy; its exact-tie repair is deliberately absent from the gradient path.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.action_return import readout as b, finish_readout as f, refit_policy as rp
from experiments.continuity import run as continuity

ROOT = b.ROOT
OUT = ROOT / 'data/wait-content/train-v1'
PLAN = OUT / 'protocol.json'
DATA = ROOT / 'data/wait-content/data-v1'
ARMS = ('end_only', 'wait_mixed')
BLOCK = 128
UPDATES = 128
FORMAT = 'matched-wait-content-recurrent-train-v1'


def normalized(value):
    return json.loads(json.dumps(value))


def source_info(seed):
    parent = b.old.read(rp.PLAN)
    source = parent['seeds'][str(seed)]['source']['endpoint']
    return dict(endpoint=source, controller=dict(path=str(rp.OUT / f'seed_{seed:03d}/model_own_policy.pt'),
                sha256=b.old.sha(rp.OUT / f'seed_{seed:03d}/model_own_policy.pt')))


def load_initial(seed):
    """Return the exact inherited controller and its actual full147456 anchor."""
    src = source_info(seed)
    trainer = b.native.NativeTrainer.from_checkpoint(src['endpoint']['path'], device='cpu')
    if b.state_digest(trainer.semantic_state()) != src['endpoint']['semantic_hash']:
        raise ValueError('source native endpoint semantic hash')
    saved = torch.load(src['controller']['path'], map_location='cpu', weights_only=False)
    begin_fits = [{k: saved[f'begin_head.j{j}_{k}'] for k in ('mean', 'std', 'coeff', 'bias')} for j in range(2)]
    finish_fits = [{k: saved[f'j{j}_{k}'] for k in ('mean', 'std', 'coef', 'bias')} for j in range(2)]
    model = f.FinishController(b.BeginController(trainer.model, 'non_input24', begin_fits,
                                               saved['begin_head.indices'].tolist()), finish_fits)
    model.load_state_dict(saved, strict=True)
    if b.state_digest(saved) != b.state_digest(model.state_dict()):
        raise ValueError('inherited controller state differs')
    return model, trainer


def trainable(model, mask):
    """Keep fixed ports, BEGIN decoder and normalization; learn supported edges/FINISH."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = model.begin_head.base
    if base.mode != 'undirected' or base.edge_shared.dtype != torch.float64:
        raise ValueError('requires FP64 undirected shared-coordinate model')
    base.edge_shared.requires_grad_(True)
    for j in range(2):
        for key in ('coef', 'bias'):
            name = f'j{j}_{key}'
            value = getattr(model, name).detach().clone()
            delattr(model, name)
            model.register_parameter(name, nn.Parameter(value))
    mask = mask.bool().clone()
    base.validate_eligibility(mask)
    base.edge_shared.register_hook(lambda grad: grad * mask.to(grad))
    return [value for value in model.parameters() if value.requires_grad]


def raw_offsets(model, hidden):
    x = hidden.index_select(-1, model.begin_head.indices)
    begins, finishes = [], []
    for j in range(model.n_jobs):
        z = (x-getattr(model.begin_head, f'j{j}_mean')) / getattr(model.begin_head, f'j{j}_std')
        begins.append(z @ getattr(model.begin_head, f'j{j}_coeff').T + getattr(model.begin_head, f'j{j}_bias'))
        z = (x-getattr(model, f'j{j}_mean')) / getattr(model, f'j{j}_std')
        finishes.append(z @ getattr(model, f'j{j}_coef').T + getattr(model, f'j{j}_bias'))
    return torch.stack(begins, -2), torch.stack(finishes, -2)


def rollout(model, observations, hidden, elapsed):
    """The verified fast_window Heun arithmetic, matrix expanded once per update."""
    base = model.begin_head.base
    matrix = base.recurrent_matrix()
    states = []
    for obs, dt in zip(observations, elapsed):
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('invalid elapsed')
        steps = max(1, math.ceil(dt / base.max_step))
        delta = dt / steps
        for _ in range(steps):
            first = -hidden + F.linear(hidden.tanh(), matrix)
            predictor = hidden + delta * first
            second = -predictor + F.linear(predictor.tanh(), matrix)
            hidden = hidden + .5 * delta * (first + second)
        hidden = hidden + base.observation_pulse(obs)
        states.append(hidden)
    states = torch.stack(states)
    begin, finish = raw_offsets(model, states)
    return begin, finish, hidden, states


def selected_anchors(anchors, block, arm):
    chosen = anchors[anchors[:, 0] == block]
    if len(chosen):
        if len(chosen) % 2 or int(chosen[:, 7].sum()) != len(chosen)//2:
            raise ValueError('each update needs exactly half the fixed requests moved')
        global_t = chosen[:, 4].copy()
        if arm == 'wait_mixed':
            global_t[chosen[:, 7] == 1] = chosen[chosen[:, 7] == 1, 5]
        return chosen, torch.as_tensor(np.column_stack((global_t-block*BLOCK, chosen[:, 1], chosen[:, 2])), dtype=torch.long)
    return chosen, torch.empty((0, 3), dtype=torch.long)


def objective(begin, finish, actions, chosen, positions):
    """Proper branch CEs aligned with the inherited independent argmax decisions.

    OTHER/BEGINv and WAIT/eight-FINISH targets receive all original teacher
    positions. Every teacher FINISH contributes its specific content identically
    in both arms. No FINISH class-count logsumexp is used as an action gate.
    """
    group_target = torch.where(actions.le(2), actions, 0)
    weights = torch.ones_like(actions, dtype=torch.float64)
    weights[actions.eq(0)] = .1
    denominator = weights.sum()
    group_logits = torch.cat((begin.new_zeros(*begin.shape[:-1], 1), begin), -1)
    group = (F.cross_entropy(group_logits.flatten(0, -2), group_target.flatten(), reduction='none').reshape_as(weights)*weights).sum()/denominator
    keep = actions.eq(0) | actions.ge(3)
    conditional_logits = torch.cat((finish.new_zeros(*finish.shape[:-1], 1), finish), -1)
    conditional_target = torch.where(actions.eq(0), 0, actions-2)
    conditional = (F.cross_entropy(conditional_logits[keep], conditional_target[keep], reduction='none')*weights[keep]).sum()/denominator
    if len(chosen):
        target = torch.as_tensor(chosen[:, 6], dtype=torch.long)
        auxiliary = F.cross_entropy(finish[positions[:, 0], positions[:, 1], positions[:, 2]], target)
    else:
        auxiliary = finish.sum()*0
    return group+conditional+auxiliary, dict(action=group+conditional, group=group,
        conditional=conditional, extra=auxiliary), dict(action_denominator=float(denominator),
        conditional_weight_sum=float(weights[keep].sum()), action_positions=actions.numel(),
        action_class_counts=torch.bincount(actions.flatten(), minlength=11).tolist(),
        auxiliary_count=len(chosen), auxiliary_denominator=len(chosen),
        auxiliary_class_counts=np.bincount(chosen[:, 6], minlength=8).tolist())


def load_data(seed):
    folder = DATA / f'seed_{seed:03d}'
    with np.load(folder/'stream.npz', allow_pickle=False) as bundle:
        data = {key: bundle[key].copy() for key in bundle.files}
    anchors = np.load(folder/'anchors.npy', allow_pickle=False)
    if data['obs'].shape != (UPDATES*BLOCK, 8, 56) or anchors.ndim != 2 or anchors.shape[1] != 8:
        raise ValueError('unexpected fixed stream or anchor shape')
    if anchors.dtype != np.int64:
        raise ValueError('anchor dtype')
    elapsed = data.get('elapsed', data.get('dt'))
    if elapsed.ndim == 2:
        if not np.array_equal(elapsed, np.repeat(elapsed[:, :1], 8, axis=1)):
            raise ValueError('per-lane elapsed differs')
        elapsed = elapsed[:, 0]
    data['elapsed'] = elapsed
    for block in range(UPDATES):
        chosen, positions = selected_anchors(anchors, block, 'wait_mixed')
        if len(chosen):
            if np.any(chosen[:, 1] >= 4) or np.any(chosen[:, 2] >= 2) or np.any(chosen[:, 1:4] < 0) or np.any((chosen[:, 6] < 0) | (chosen[:, 6] >= 8)) or np.any((chosen[:,7] < 0) | (chosen[:,7] > 1)):
                raise ValueError('invalid anchor lane/job/request/target')
            if np.any(chosen[:, 4]//BLOCK != block) or np.any(chosen[:, 5]//BLOCK != block):
                raise ValueError('cross-update auxiliary candidate pair')
            e,a,j = chosen[:,4],chosen[:,1],chosen[:,2]
            w = chosen[:,5]
            if not np.array_equal(data['teacher_actions'][e,a,j], chosen[:,6]+3) or np.any(data['teacher_actions'][w,a,j] != 0):
                raise ValueError('end/processing action targets differ from declared candidates')
            if not np.array_equal(data['request_id'][e,a,j],chosen[:,3]) or not np.array_equal(data['request_id'][w,a,j], chosen[:,3]):
                raise ValueError('paired request identity differs')
            if 'processing_wait' in data and not data['processing_wait'][w,a,j].all():
                raise ValueError('processing candidate is not a real processing WAIT')
            if 'finish_target' in data and (not np.array_equal(data['finish_target'][e,a,j],chosen[:,6]) or not np.array_equal(data['finish_target'][w,a,j],chosen[:,6])):
                raise ValueError('paired content differs')
    return data, anchors


def files_for_seed(seed):
    return [DATA/f'seed_{seed:03d}'/name for name in ('stream.npz','anchors.npy','start.pt','endpoint.pt','requests.json')]


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable training protocol exists')
    sources = {str(seed): source_info(seed) for seed in range(3)}
    inputs = {str(rp.PLAN): b.old.sha(rp.PLAN)}
    for src in sources.values():
        for item in src.values():
            inputs[item['path']] = b.old.sha(item['path'])
    for seed in range(3):
        load_data(seed)
        for path in files_for_seed(seed):
            inputs[str(path)] = b.old.sha(path)
    data_plan = DATA/'protocol.json'
    inputs[str(data_plan)] = b.old.sha(data_plan)
    inputs[str(OUT/'preflight.json')] = b.old.sha(OUT/'preflight.json')
    paths = [Path(__file__).resolve(), Path(b.__file__).resolve(), Path(f.__file__).resolve(),
             Path(__file__).resolve().with_name('train_tests.py'),
             ROOT/'experiments/coverage_exit/fast_window.py', ROOT/'experiments/undirected/model.py']
    value = dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),seeds=[0,1,2],arms=list(ARMS),
        declaration='Frozen after source own_policy results and content timing diagnostics, before this recurrent training or its free evaluations. No selection by new training outcomes.',
        source_files={str(p):b.old.sha(p) for p in paths},input_files=inputs,sources=sources,
        data=dict(root=str(DATA),all_ticks=UPDATES*BLOCK,block_ticks=BLOCK,actors_fit=[0,1,2,3],actors_heldout=[4,5,6,7],
            rule='One common frozen own_policy autonomous public stream per seed; teachers supply labels only. Full8-lane data retained for separate evaluation warmup; training reads lanes0..3 only.',
            anchors_columns=['block','actor','job','request_id','end_t','wait_t','target0to7','move'],
            matching='Identical requests, targets, eight-class counts and quantity at every update. end_only supervises every fixed anchor at END; wait_mixed moves exactly half to the same-request processing-WAIT candidate within the same update. Original WAIT target remains WAIT.'),
        train=dict(updates=UPDATES,ticks_per_update=BLOCK,dtype='float64',device='cpu',threads_per_process=1,
            state='Start actual147456 inherited h on fit lanes; continuously carry complete h through every update. Detach graph only at128-tick optimizer boundaries; no hidden reset or request boundary warmup.',
            parameters='Original support of undirected edge_shared plus existing conditional FINISH coef/bias for two jobs; preserve exact support and project inactive coordinates0 after every step. Fixed input/output ports, BEGIN affine coefficients, all normalization means/std and graph topology remain frozen.',
            action_loss='All tick/job positions. Weighted OTHER/BEGIN1/BEGIN2 CE plus weighted WAIT/eightFINISH conditional9 CE at teacher nonBEGIN rows; both numerators divide by original all-position weights.sum. WAIT0 weight0.1; all others1. Every teacher FINISH retains its specific eight-class target in both arms.',
            extra_loss='Additional conditional8 CE on selected eightFINISH offsets only, reduction mean, weight1. Same labels/counts/denominator per update; only temporal positions differ. No WAIT action is relabeled FINISH.',
            optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,initialization='Fresh identical zero-state Adam in both arms; source backbone Adam from its original147456 training is retained as provenance, not resumed.'),
            clipping=dict(kind='global_l2',max_norm=1.),auxiliary_weight=1.,regularization='none',
            initialization='Same exact own_policy controller, source support and source h in both arms; conditional FINISH buffers promoted to trainable parameters without numerical change.'),
        inference='Unmodified inherited FinishController.step: group-first BEGIN choice from two fixed affine offsets; independent conditional WAIT/eightFINISH choice from0 and eight learned offsets. Exact tie repair runs only at inference, never in gradient path. No private state gates or teacher inputs.',
        persistence='Per-arm source binding, initial state, final all-state checkpoint and model, each update losses/budget/data/hidden/model hashes. Seed0 first auxiliary-moved update saves complete before/after model, Adam and h for independent reconstruction.',
        evaluation='Separate prospective evaluation by wait_eval; primary actual long task completion and idle action frequency. Learned states during fixed-data replay are not mistaken for an on-policy deployment world.',
        runtime=dict(workers=3,threads=1,dtype='float64'),
        limits=['Externally labeled fixed autonomous experience, not new self-generated experience during recurrent optimization.',
                'BEGIN decoder coefficients are fixed but trained recurrent weights may change its inputs and decisions.',
                'Actors4..7 are held out only from this fit; original backbone birth training saw all8 actors.'])
    value=normalized(value);value['protocol_hash']=b.state_digest(value)
    OUT.mkdir(parents=True,exist_ok=True);b.old.write(PLAN,value)
    return value


def checked():
    plan = b.old.read(PLAN)
    if plan['protocol_hash'] != b.state_digest({k:v for k,v in plan.items() if k!='protocol_hash'}):
        raise ValueError('training protocol hash')
    for group in ('source_files','input_files'):
        for path,sha in plan[group].items():
            if b.old.sha(path) != sha:
                raise ValueError('training source/input changed '+path)
    return plan


def frozen_buffers(model):
    return {key:value.detach().clone() for key,value in model.state_dict().items()
            if key!='begin_head.base.edge_shared' and key not in ('j0_coef','j0_bias','j1_coef','j1_bias')}


def snapshot(model, optimizer, h, mask, *, seed, arm, block, plan):
    state = dict(format=FORMAT,seed=seed,arm=arm,block=block,ticks=block*BLOCK,
        protocol_hash=plan['protocol_hash'],model=continuity.cpu_tree(model.state_dict()),
        optimizer=continuity.cpu_tree(optimizer.state_dict()),hidden=h.detach().cpu().clone(),
        mask=mask.clone(),rng=continuity.rng_state(),sources=plan['sources'][str(seed)],
        input_files=plan['input_files'],trainable_names=[name for name,p in model.named_parameters() if p.requires_grad],
        interpretation='All state of this fixed-stream recurrent optimizer. World and scheduler are immutable shared collection artifacts, not a learner-acted world; bound start.pt/endpoint.pt preserve their full state.')
    state['state_hash']=b.state_digest(state)
    return state


def atomic_save(value, path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def load_trained(seed, arm, path=None):
    """Evaluation API: deployed eleven-class model, complete training checkpoint."""
    path=Path(path) if path else OUT/f'seed_{seed:03d}'/arm/'checkpoint.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['state_hash'] != b.state_digest({k:v for k,v in saved.items() if k!='state_hash'}):
        raise ValueError('training checkpoint state hash')
    model,trainer=load_initial(seed)
    trainable(model,saved['mask'])
    model.load_state_dict(saved['model'],strict=True)
    model.begin_head.base.validate_eligibility(saved['mask'])
    return model,saved


def preflight(seed=0):
    """Real128-tick numerical parity and gradient/mask checks before formal training."""
    torch.set_num_threads(1)
    data,anchors=load_data(seed)
    model,trainer=load_initial(seed)
    start=torch.load(DATA/f'seed_{seed:03d}/start.pt',map_location='cpu',weights_only=False)
    if not torch.equal(start['h'],trainer.h) or b.state_digest(start['collector_state'])!=b.state_digest(model.state_dict()):
        raise ValueError('collection did not start at inherited h/controller')
    original=deepcopy(model)
    mask=trainer.operator.mask.clone()
    parameters=trainable(model,mask)
    before=b.state_digest(original.state_dict())
    if b.state_digest(model.state_dict())!=before:
        raise ValueError('promotion changed initial scores')
    block=next((k for k in range(UPDATES) if np.any(anchors[:,0]==k)),0)
    # All earlier observations remain present so this selected block begins at
    # the actual continuously replayed h under unmodified initial parameters.
    h=trainer.h[:4].clone()
    observations=torch.as_tensor(data['obs'][:(block+1)*BLOCK,:4],dtype=torch.float64)
    elapsed=data['elapsed'][:(block+1)*BLOCK]
    with torch.no_grad():
        h_old=h.clone();states=[]
        for obs,dt in zip(observations,elapsed):
            _,h_old=original.step(obs,h_old,float(dt));states.append(h_old.clone())
        _,_,h_fast,fast_states=rollout(model,observations,h,elapsed)
        exact=torch.equal(torch.stack(states),fast_states)
        error=float((torch.stack(states)-fast_states).abs().max())
        if not exact:
            raise ValueError(f'fast_window numerical parity {error}')
        h_start=h if block==0 else states[block*BLOCK-1]
    begin,finish,h_end,_=rollout(model,observations[-BLOCK:],h_start.detach(),elapsed[-BLOCK:])
    actions=torch.as_tensor(data['teacher_actions'][block*BLOCK:(block+1)*BLOCK,:4],dtype=torch.long)
    chosen,pos=selected_anchors(anchors,block,'wait_mixed')
    loss,parts,budget=objective(begin,finish,actions,chosen,pos)
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError('missing/nonfinite trainable gradient')
    if model.begin_head.base.edge_shared.grad[~mask].ne(0).any():
        raise ValueError('inactive edge gradient')
    record=dict(seed=seed,block=block,initial_controller_exact=True,fast_window_states_bit_exact=exact,
        maximum_state_error=error,backward_finite=True,inactive_gradient_exact_zero=True,
        first_auxiliary_block=block,budget=budget,parts={k:float(v.detach()) for k,v in parts.items()},
        source_model_hash=before,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        supported_edges=int(mask.sum()),potential_edges=len(mask))
    OUT.mkdir(parents=True,exist_ok=True)
    b.old.write(OUT/'preflight.json',record)
    return record


def run(seed, arm):
    torch.set_num_threads(1)
    if arm not in ARMS:
        raise ValueError(arm)
    plan=checked();data,anchors=load_data(seed)
    folder=OUT/f'seed_{seed:03d}'/arm
    folder.mkdir(parents=True,exist_ok=False)
    model,trainer=load_initial(seed)
    source_semantic=b.state_digest(trainer.semantic_state())
    mask=trainer.operator.mask.clone()
    inherited_optimizer_hash=b.state_digest(trainer.optimizer.state_dict())
    parameters=trainable(model,mask)
    optimizer=torch.optim.Adam(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
    h=trainer.h[:4].clone().detach()
    initial=snapshot(model,optimizer,h,mask,seed=seed,arm=arm,block=0,plan=plan)
    atomic_save(initial,folder/'initial.pt')
    fixed=frozen_buffers(model)
    initial_hash=b.state_digest(model.state_dict())
    first_moved=next((k for k in range(UPDATES) if np.any((anchors[:,0]==k)&(anchors[:,7]==1))),None)
    log=[];started=time.perf_counter()
    with (folder/'updates.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for block in range(UPDATES):
            t=block*BLOCK
            obs=torch.as_tensor(data['obs'][t:t+BLOCK,:4],dtype=torch.float64)
            actions=torch.as_tensor(data['teacher_actions'][t:t+BLOCK,:4],dtype=torch.long)
            elapsed=data['elapsed'][t:t+BLOCK]
            chosen,positions=selected_anchors(anchors,block,arm)
            optimizer.zero_grad(set_to_none=True)
            h_start=h.detach().clone()
            if seed==0 and block==first_moved:
                atomic_save(snapshot(model,optimizer,h,mask,seed=seed,arm=arm,block=block,plan=plan),folder/'audit_before.pt')
            begin,finish,h_end,_=rollout(model,obs,h,elapsed)
            loss,parts,budget=objective(begin,finish,actions,chosen,positions)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite objective')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.begin_head.base.edge_shared.masked_fill_(~mask,0.)
            model.begin_head.base.validate_eligibility(mask)
            h=h_end.detach()
            if seed==0 and block==first_moved:
                atomic_save(snapshot(model,optimizer,h,mask,seed=seed,arm=arm,block=block+1,plan=plan),folder/'audit_after.pt')
                atomic_save(dict(block=block,h_start=h_start,h_end=h,observations=obs,elapsed=elapsed,
                    actions=actions,chosen_anchors=torch.as_tensor(chosen),positions=positions),folder/'audit_batch.pt')
            entry=dict(block=block,update=block+1,tick_start=t,tick_stop=t+BLOCK,
                loss=float(loss.detach()),losses={k:float(v.detach()) for k,v in parts.items()},
                gradient_norm_before_clip=float(norm),**budget,
                moved_positions=int(chosen[:,7].sum()) if arm=='wait_mixed' else 0,
                requested_moving_half=int(chosen[:,7].sum()),
                auxiliary_targets=chosen[:,6].tolist(),auxiliary_positions=positions.tolist(),
                auxiliary_request_keys=chosen[:,1:4].tolist(),
                pair_hash=b.state_digest(torch.as_tensor(chosen)),
                action_data_hash=b.state_digest(dict(observations=obs,elapsed=elapsed.tolist(),targets=actions)),
                h_start_hash=b.state_digest(h_start),h_end_hash=b.state_digest(h),
                model_hash=b.state_digest(model.state_dict()),optimizer_hash=b.state_digest(optimizer.state_dict()),
                active_edges=int(mask.sum()),inactive_exact_zero=True)
            stream.write(json.dumps(entry,separators=(',',':'))+'\n');stream.flush();log.append(entry)
            if (block+1)%16==0:
                print(seed,arm,'updates',block+1,'loss',entry['loss'],'aux',len(chosen),'seconds',round(time.perf_counter()-started,1),flush=True)
    if any(not torch.equal(value,model.state_dict()[key]) for key,value in fixed.items()):
        raise ValueError('frozen interface/buffer changed')
    if source_semantic!=b.state_digest(trainer.semantic_state()):
        raise ValueError('native source trainer changed')
    final=snapshot(model,optimizer,h,mask,seed=seed,arm=arm,block=UPDATES,plan=plan)
    atomic_save(final,folder/'checkpoint.pt');atomic_save(model.state_dict(),folder/'model.pt')
    counts=np.sum([entry['auxiliary_class_counts'] for entry in log],axis=0).tolist()
    result=dict(format=FORMAT,seed=seed,arm=arm,protocol_hash=plan['protocol_hash'],updates=UPDATES,ticks=UPDATES*BLOCK,
        seconds=time.perf_counter()-started,initial_model_hash=initial_hash,final_model_hash=b.state_digest(model.state_dict()),
        checkpoint_sha256=b.old.sha(folder/'checkpoint.pt'),model_sha256=b.old.sha(folder/'model.pt'),
        updates_sha256=b.old.sha(folder/'updates.jsonl'),source_semantic_hash=source_semantic,
        source_optimizer_hash=inherited_optimizer_hash,new_optimizer_initially_empty=True,
        trainable_names=final['trainable_names'],effective_trainable_scalar_count=int(mask.sum())+sum(getattr(model,f'j{j}_{key}').numel() for j in range(2) for key in ('coef','bias')),
        supported_edges=int(mask.sum()),mask_hash=b.state_digest(mask),topology_changed=False,hidden_resets=0,
        source_unchanged=True,frozen_interfaces_exact=True,all_position_action_class_counts=np.sum([e['action_class_counts'] for e in log],axis=0).tolist(),
        extra_class_counts=counts,extra_positions=sum(e['auxiliary_count'] for e in log),
        moved_positions=sum(e['moved_positions'] for e in log),source_backbone_preexisting_updates=trainer.updates)
    result=normalized(result);result['result_hash']=b.state_digest(result)
    b.old.write(folder/'result.json',result)
    checked()
    return result


def launch():
    checked()
    if not (OUT/'preflight.json').exists():
        raise ValueError('preflight required before formal training')
    def one(job):
        seed,arm=job
        with (OUT/f'seed_{seed:03d}_{arm}.log').open('w') as stream:
            proc=subprocess.Popen([sys.executable,'-m','experiments.wait_content.train','run','--seed',str(seed),'--arm',arm],cwd=ROOT,
                env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},'CUDA_VISIBLE_DEVICES':'-1'},stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,arm=arm,pid=proc.pid,returncode=proc.wait())
    # Three single-thread workers; seed/arm order is fixed and never selected by loss.
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes=list(pool.map(one,[(seed,arm) for seed in range(3) for arm in ARMS]))
    b.old.write(OUT/'exit_codes.json',codes)
    if any(item['returncode'] for item in codes):
        raise RuntimeError(codes)
    verify_budget()


def verify_budget():
    checked();records=[]
    for seed in range(3):
        logs=[]
        for arm in ARMS:
            folder=OUT/f'seed_{seed:03d}'/arm
            result=b.old.read(folder/'result.json')
            if result['result_hash'] != b.state_digest({k:v for k,v in result.items() if k!='result_hash'}):
                raise ValueError('result digest')
            entries=[json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
            if len(entries)!=UPDATES:
                raise ValueError('update count')
            logs.append(entries)
        for left,right in zip(*logs):
            for key in ('block','action_positions','action_denominator','action_class_counts','conditional_weight_sum',
                        'auxiliary_count','auxiliary_denominator','auxiliary_class_counts','auxiliary_targets',
                        'auxiliary_request_keys','pair_hash','action_data_hash','active_edges'):
                if left[key]!=right[key]:
                    raise ValueError('unequal update budget '+key)
            if right['moved_positions']*2 != right['auxiliary_count']:
                raise ValueError('exact moving half')
        records.append(dict(seed=seed,updates_exact=UPDATES,all_update_data_labels_denominators_equal=True,
            all_update_extra_content_classes_requests_counts_equal=True,exact_half_moved=True,
            auxiliary_count=sum(e['auxiliary_count'] for e in logs[0])))
    b.old.write(OUT/'budget_verification.json',dict(format=FORMAT,records=records,passed=True))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','preflight','run','launch','verify'))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--arm',choices=ARMS,default=ARMS[0])
    args=parser.parse_args()
    if args.command=='prepare': value=prepare()
    elif args.command=='preflight': value=preflight(args.seed)
    elif args.command=='run': value=run(args.seed,args.arm)
    elif args.command=='launch': value=launch()
    else: value=verify_budget()
    if args.command in ('preflight','verify'):
        print(json.dumps(value,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
