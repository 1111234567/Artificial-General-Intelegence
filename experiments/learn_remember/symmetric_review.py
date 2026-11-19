"""Transfer the fixed plain-GRU 1/8 historical review to unique shared pairs.

No source trainer is changed. Restoring a dense deployment checkpoint first
reconstitutes the true packed pair parameter and its existing packed Adam.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from experiments.learn_remember import gru_train as plain,gru_review as review,symmetric_gru as symmetric
from experiments.lifelong.world import ContextWorld

OUT=symmetric.OUT
PLAN=OUT/'review_protocol.json'
FORMAT='symmetric-gru-fixed-one-eighth-review-v1'


def declare():
    if PLAN.exists():raise FileExistsError(PLAN)
    files=[Path(__file__).resolve(),Path(plain.__file__).resolve(),Path(review.__file__).resolve(),Path(symmetric.__file__).resolve()]
    value=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),seeds=[0,1,2],
        declaration='Last fixed transfer factor, declared before any symmetric review run. All3seeds, no additional experiments or outcome-dependent choice.',
        source_files={str(path):plain.sha(path) for path in files},parent_protocol_sha256=plain.sha(OUT/'protocol.json'),
        source='Each corresponding symmetric-v1/seed_XXX/wide.pt complete model+packed Adam+world+h+work+NumPy/torch/CUDA RNG. Individual file SHAs are fixed in per-seed review_plan.json before collection/fit.',
        restoration='Use SharedPairs.right_inverse on symmetric dense weight_hh_l0 to restore exactly3 unique upper-triangle tables. Load original packed Adam with parameter-order and per-parameter moment-shape checks; require dense deployment equality on actual next public observation and h. Do not run a dense unconstrained optimizer.',
        intervention='Exactly ordinary gru_review.ReviewTrainer.fit_review/run_review:32 epochs/round; every eighth epoch historical, other28 live.12withdrawal rounds,1024ticks/round,128tickTBPTT,64lanes. Same clipping,WAITweight,learning rate and Adam continuation as no-review. No extra updates;1/8 live fit budget replaced.',
        historical_rounds=list(review.HISTORICAL_ROUNDS),history_schedule='12+(withdrawal_round modulo4), same4entire last-wide sequences and numeric collection anchors, without success selection.',
        matching='Wait for corresponding no-review withdrawal.pt and all12 withdrawal trajectories. Reuse SharedTrainer.collect and ReviewTrainer.run_review to check identical teacher-lane masks, family exposures, public external event pulses, and raw public family context. Own actions and receipts can differ.',
        historical_audit='Reuse historical_metadata, replay wide rounds8..15 from wide_mid actual world, verify every public observation and midpoint label under recorded executed actions. Report family3 slots/nonWAIT labels for selected12..15.',
        persistence='Separate review_mid.pt and review.pt native dense deployment checkpoints with existing packed Adam, plus review_mid.packed.pt and review.packed.pt containing true parametrized trainable state. Source wide/no-review files never overwritten.',
        evaluation='Parent uses same heldout free-behavior and memory evaluator as other stages; this worker only trains and audits fixed-budget restoration and matching.',
        limitations=['Explicit supervised historical review, not autonomous memory discovery.','Symmetric recurrent matrices in3GRU gates; directed input/output interfaces and node-dependent gates need not imply reciprocal effective influences.','Original training mixed teacher/own lanes as recorded; never describe teacher collection as autonomous.'])
    plain.write_json(PLAN,value)
    return value


def checked():
    value=json.loads(PLAN.read_text(encoding='utf-8'))
    replacements={}
    amendment=OUT/'review-implementation-amendment.json'
    if amendment.exists():
        for item in json.loads(amendment.read_text(encoding='utf-8'))['changed_sources']:
            if value['source_files'][item['path']]!=item['original_sha256'] or plain.sha(item['original_snapshot'])!=item['original_sha256']:raise ValueError('frozen original not preserved')
            replacements[item['path']]=item['amended_sha256']
    for path,expected in value['source_files'].items():
        expected=replacements.get(path,expected)
        if plain.sha(path)!=expected:raise ValueError('frozen review source changed '+path)
    if plain.sha(OUT/'protocol.json')!=value['parent_protocol_sha256']:raise ValueError('parent protocol changed')
    return value


def validate_optimizer(model,optimizer):
    packed=model.core.parametrizations.weight_hh_l0.original
    if packed.shape!=(3,model.width*(model.width+1)//2):raise AssertionError('not unique unordered pairs')
    matrices=model.core.weight_hh_l0.reshape(3,model.width,model.width)
    if not torch.equal(matrices,matrices.transpose(-1,-2)):raise AssertionError('nonreciprocal recurrent matrix')
    rows=[]
    for name,parameter in model.named_parameters():
        state=optimizer.state.get(parameter)
        if state is None:raise AssertionError('missing inherited Adam '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=parameter.shape:raise AssertionError('wrong Adam shape '+name+' '+key)
            if not torch.isfinite(state[key]).all():raise AssertionError('nonfinite inherited Adam')
        rows.append(dict(name=name,shape=list(parameter.shape),Adam_step=int(state['step'])))
    if sum(p is packed for group in optimizer.param_groups for p in group['params'])!=1:raise AssertionError('packed parameter optimizer multiplicity')
    return rows


class SymmetricReviewTrainer(symmetric.SharedTrainer,review.ReviewTrainer):
    def __init__(self,seed):
        checked();folder=OUT/f'seed_{seed:03d}'
        if (folder/'review.pt').exists() or (folder/'review_updates.jsonl').exists():raise FileExistsError('review branch exists')
        required=[folder/'wide.pt',folder/'wide_mid.pt',folder/'withdrawal.pt']
        required += [folder/f'trajectory_withdrawal_{i:03d}.pt' for i in range(12)]
        required += [folder/f'trajectory_wide_{i:03d}.pt' for i in range(8,16)]
        if any(not path.exists() for path in required):raise FileNotFoundError('matching no-review withdrawal is not complete')
        checkpoint=torch.load(folder/'wide.pt',map_location='cpu',weights_only=False)
        self.cfg=plain.Config(**checkpoint['config']);self.folder=folder
        if (self.cfg.epochs,self.cfg.batch,self.cfg.block,self.cfg.ticks_per_round,self.cfg.withdrawal_rounds)!=(32,64,128,1024,12):raise ValueError('ordinary GRU budget mismatch')
        dense=checkpoint['model']['core.weight_hh_l0'].reshape(3,self.cfg.width,self.cfg.width)
        if not torch.equal(dense,dense.transpose(-1,-2)):raise AssertionError('wide checkpoint not exactly symmetric')
        self.model=symmetric.constrain(plain.GRUAgent(**checkpoint['model_config']).to(self.cfg.device))
        native=self.model.state_dict();rule=self.model.core.parametrizations.weight_hh_l0[0]
        for name,value in checkpoint['model'].items():
            if name=='core.weight_hh_l0':native['core.parametrizations.weight_hh_l0.original']=rule.right_inverse(value.to(self.cfg.device))
            else:native[name]=value.to(self.cfg.device)
        self.model.load_state_dict(native,strict=True)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.cfg.lr)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        restored_optimizer=validate_optimizer(self.model,self.optimizer)
        deployed=symmetric.deploy(self.model).to(self.cfg.device)
        if any(not torch.equal(value.cpu(),checkpoint['model'][name]) for name,value in deployed.state_dict().items()):raise AssertionError('restored deployment weights differ')
        self.world=ContextWorld.from_state_dict(checkpoint['world'],device='cpu');self.hidden=checkpoint['hidden'].clone()
        # Actual next observation is read on a detached world solely for parity.
        probe=ContextWorld.from_state_dict(checkpoint['world'],device='cpu');obs=probe.observe().to(self.cfg.device)
        with torch.no_grad():
            packed_logits,packed_h=self.model.step(obs,self.hidden.to(self.cfg.device))
            dense_logits,dense_h=deployed.step(obs,self.hidden.to(self.cfg.device))
        if not torch.equal(packed_logits,dense_logits) or not torch.equal(packed_h,dense_h):raise AssertionError('packed/deployed execution differs')
        self.work=copy.deepcopy(checkpoint['work']);self.work.update(review_updates=0,review_gradient_lane_ticks=0,
            review_family3_gradient_tick_slots=0,review_family3_nonwait_labels=0)
        self.stage,self.round=checkpoint['stage'],checkpoint['round'];self.withdrawal_started=False
        self.metadata,self.sources=review.historical_metadata(folder)
        self.history={index:torch.load(folder/f'trajectory_wide_{index:03d}.pt',map_location='cpu',weights_only=False) for index in review.HISTORICAL_ROUNDS}
        self.rng=np.random.default_rng();self.rng.bit_generator.state=copy.deepcopy(checkpoint['numpy_rng'])
        # Historical parity and temporary model construction consume random
        # draws; restore exact source RNG last, before any new collection.
        torch.set_rng_state(checkpoint['torch_rng'])
        if checkpoint['cuda_rng']:torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        self.bindings={str(path):plain.sha(path) for path in required}
        self.restoration=dict(packed_shape=list(self.model.core.parametrizations.weight_hh_l0.original.shape),
            unique_recurrent_coordinates=self.model.core.parametrizations.weight_hh_l0.original.numel(),
            trainable_parameters=sum(p.numel() for p in self.model.parameters()),Adam=restored_optimizer,
            exact_dense_weights=True,actual_next_logits_hidden_bit_exact=True,
            numpy_rng_exact=self.rng.bit_generator.state==checkpoint['numpy_rng'],
            torch_rng_exact=torch.equal(torch.get_rng_state(),checkpoint['torch_rng']))
        plain.write_json(folder/'review_plan.json',dict(format=FORMAT,global_protocol_sha256=plain.sha(PLAN),
            source_files=checked()['source_files'],bindings=self.bindings,historical_rounds=list(review.HISTORICAL_ROUNDS),
            historical_stats=self.metadata,historical_sources=self.sources,restoration=self.restoration,
            allocation='Inherited ordinary1/8 review exactly; no new updates or current-round reweighting.'))

    def save(self,name):
        validate_optimizer(self.model,self.optimizer)
        symmetric.SharedTrainer.save(self,name)
        native=dict(format=FORMAT,stage=name,model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            optimizer=copy.deepcopy(self.optimizer.state_dict()),world=self.world.state_dict(),hidden=self.hidden.clone(),
            numpy_rng=copy.deepcopy(self.rng.bit_generator.state),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],work=copy.deepcopy(self.work),
            deployment_sha256=plain.sha(self.folder/f'{name}.pt'),protocol_sha256=plain.sha(PLAN))
        torch.save(native,self.folder/f'{name}.packed.pt')

    def run_review(self):
        review.ReviewTrainer.run_review(self)
        for path,expected in self.bindings.items():
            if plain.sha(path)!=expected:raise AssertionError('frozen source modified '+path)
        final=validate_optimizer(self.model,self.optimizer)
        rows=[json.loads(line) for line in (self.folder/'review_updates.jsonl').read_text().splitlines()]
        if len(rows)!=12 or any(row['fit']['updates']!={'live':224,'review':32} for row in rows):raise AssertionError('review update allocation mismatch')
        source=torch.load(self.folder/'wide.pt',map_location='cpu',weights_only=False)
        native_end=torch.load(self.folder/'withdrawal.pt',map_location='cpu',weights_only=False)
        for key in ('gradient_updates','gradient_lane_ticks','environment_ticks','lane_ticks'):
            if self.work[key]-source['work'][key]!=native_end['work'][key]-source['work'][key]:raise AssertionError('no-review budget mismatch '+key)
        plain.write_json(self.folder/'review_audit.json',dict(format=FORMAT,passed=True,restoration=self.restoration,final_Adam=final,
            historical_stats=self.metadata,all12external_stream_teacher_lanes_exact=True,source_files_unchanged=True,
            review_updates=self.work['review_updates'],review_gradient_lane_ticks=self.work['review_gradient_lane_ticks'],
            work=self.work,matched_no_review_budget=True,source_bindings=self.bindings,
            outputs={str(self.folder/name):plain.sha(self.folder/name) for name in ('review_mid.pt','review.pt','review_mid.packed.pt','review.packed.pt','review_updates.jsonl','review_plan.json')},
            source_sha256=plain.sha(__file__),protocol_sha256=plain.sha(PLAN)))


def launch():
    checked()
    def one(seed):
        with (OUT/f'review_{seed}.log').open('w',encoding='utf-8') as stream:
            proc=subprocess.Popen([sys.executable,'-m','experiments.learn_remember.symmetric_review','run','--seed',str(seed)],cwd=plain.ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            return dict(seed=seed,returncode=proc.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,range(3)))
    plain.write_json(OUT/'review_exit_codes.json',results)
    if any(row['returncode'] for row in results):raise RuntimeError(results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('declare','run','launch'));parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.command=='declare':declare()
    elif args.command=='run':SymmetricReviewTrainer(args.seed).run_review()
    else:launch()
