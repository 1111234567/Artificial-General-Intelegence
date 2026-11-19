"""Frozen paired A/C history experiment from the plain GRU short snapshots.

The original world and recurrent learner sources are imported unchanged.  A
process-local mapping scope is required whenever a custom world is restored or
advanced.  Oracle labels are archived separately from executed actions.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.learn_remember import gru_train as plain
from experiments.lifelong import world as worlds

OUT = ROOT / 'data/experience-transfer/training-v1'
PLAN = OUT / 'protocol.json'
NATIVE = tuple(tuple(v) for v in worlds.FAMILY_MAPPINGS)
SEEDS = {
    0: dict(history_cue=0, B_cue=1, A=[0,1,2,3], B=[0,3,2,1], C=[2,1,0,3], conflict_raw=[1,3], shared_raw=[0,2]),
    1: dict(history_cue=1, B_cue=2, A=[0,1,2,3], B=[2,1,0,3], C=[0,3,2,1], conflict_raw=[0,2], shared_raw=[1,3]),
    2: dict(history_cue=2, B_cue=0, A=[0,3,1,2], B=[3,0,1,2], C=[0,3,2,1], conflict_raw=[0,1], shared_raw=[2,3]),
}
ARMS = ('related', 'disjoint', 'direct')
HISTORY_ROUNDS, B_ROUNDS, TICKS, BLOCK, EPOCHS = 12, 8, 1024, 128, 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def mapping_scope(table):
    """Temporarily use an explicitly archived rule table, including strict load."""
    previous = worlds.FAMILY_MAPPINGS
    table = tuple(tuple(int(x) for x in row) for row in table)
    assert len(table) == 4 and all(sorted(row) == [0,1,2,3] for row in table)
    worlds.FAMILY_MAPPINGS = table
    try:
        yield
    finally:
        worlds.FAMILY_MAPPINGS = previous


def mapping_table(seed, arm):
    spec = SEEDS[int(seed)]
    rows = [list(row) for row in NATIVE]
    rows[spec['history_cue']] = spec['C'] if arm == 'disjoint' else spec['A']
    rows[spec['B_cue']] = spec['B']
    return rows


def load_checkpoint(path, device='cpu'):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    model = plain.GRUAgent(**cp['model_config']).to(device)
    model.load_state_dict(cp['model'])
    table = cp.get('mapping_table', cp['world']['context']['mappings'])
    with mapping_scope(table):
        world = worlds.ContextWorld.from_state_dict(cp['world'], device='cpu')
    return model, world, cp['hidden'].to(device), cp


def source_files():
    return [Path(__file__).resolve(), Path(plain.__file__).resolve(), Path(worlds.__file__).resolve(),
            ROOT/'experiments/continuity/world.py']


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    origin = {str(seed): dict(path=str(plain.OUT/f'seed_{seed:03d}/short.pt'),
              sha256=sha(plain.OUT/f'seed_{seed:03d}/short.pt')) for seed in SEEDS}
    for seed, spec in SEEDS.items():
        a,b,c = spec['A'],spec['B'],spec['C']
        eq = lambda x,y: sum(i==j for i,j in zip(x,y))
        assert eq(a,b)==2 and eq(c,b)==0
        assert eq(a,NATIVE[3])==eq(b,NATIVE[3])==eq(c,NATIVE[3])==0
        assert all(a[i]==c[i] and a[i]!=b[i] for i in spec['conflict_raw'])
        cp = torch.load(origin[str(seed)]['path'], map_location='cpu', weights_only=False)
        assert cp['stage']=='short' and cp['world']['context']['family_allowed']==[3]
        assert cp['world']['context']['tensors']['family_new_counts'][...,:3].sum()==0
    plan = dict(format='experience-transfer-training-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=SEEDS, arms=ARMS, origins=origin, history_rounds=HISTORY_ROUNDS, B_rounds=B_ROUNDS,
        ticks_per_round=TICKS, block=BLOCK, epochs=EPOCHS, device='cuda',
        source_files={str(p):sha(p) for p in source_files()},
        history='Related A and disjoint C each12rounds of1024ticks,64actors,two jobs,wide duration1..16 and gap.5..32;32epochs each. Same short weight/Adam/h/world branch. Teacher probability max(0,1-round/6), shared lane draws; latter6rounds genuine own argmax. No replay across rounds.',
        direct='Time-aligned no-history-update reference: short weights and Adam never update during12rounds. Existing policy freely acts on familiar cue3 D using the same wide regime/exogenous schedule. Its h/world continue. This is not equal total gradient budget and not an input-free history.',
        B='All three real worlds change future NEW to the same new B cue and wide regime. Archive B000 before B exposure. One shared WAIT boundary step has no loss. Thereafter common midpoint-oracle B demonstrations execute in all worlds; old pending non-B slots execute WAIT with zero loss. Require identical B masks and labels at every position. Inputs can differ for inherited receipts; measured explicitly. No world/h replacement. All trained B cases/labels/masks/action counts exactly paired.',
        B_update_order='For every B round collect one128tick block and do one gradient update immediately,8blocks=epoch1. Then31 temporal passes over that round only, from its actual initial h. Save Bfirst002 after256 real B ticks/2updates and Bfirst008 after1024ticks/8updates in round1, before remaining31passes. B001/B002/B004/B008 follow complete rounds; no previous-round or old-history replay.',
        B_snapshots=['B000','Bfirst002','Bfirst008','B001','B002','B004','B008'],
        continuity='Numeric h continues from actual public history across collection/updates/stages; updates do not regenerate or reset the actual subject h. Truncated BPTT uses detached contiguous128tick blocks. Per-round replay is current data only; replay h starts at its actual collection anchor.',
        limits=['Supervised behavioral prototype; B training is teacher-controlled, not free-action acquisition.',
                'Disjoint means zero raw-to-target overlap, not statistical independence; all rules are permutations.',
                'Main causal comparison related versus disjoint has matched training/feedback budgets. Direct reference has fewer gradient updates.',
                'Three fixed seeds; mappings/cues rotate, but this does not cover all permutations.'])
    write(PLAN, plan)
    print(json.dumps(dict(event='declared',path=str(PLAN),sha256=sha(PLAN))),flush=True)


def checked():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    for p,h in plan['source_files'].items():
        assert sha(p)==h, 'frozen source changed: '+p
    for spec in plan['origins'].values():
        assert sha(spec['path'])==spec['sha256']
    return plan


class Learner:
    def __init__(self, seed, arm, plan):
        self.seed,self.arm,self.plan=seed,arm,plan
        self.spec=SEEDS[seed]
        self.table=mapping_table(seed,arm)
        self.folder=OUT/f'seed_{seed:03d}'/arm
        self.folder.mkdir(parents=True,exist_ok=False)
        self.model,self.world,self.hidden,self.origin=load_checkpoint(plan['origins'][str(seed)]['path'],'cuda')
        self.cfg=plain.Config(**self.origin['config'])
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.cfg.lr)
        self.optimizer.load_state_dict(deepcopy(self.origin['optimizer']))
        self.hidden=self.hidden.cpu()
        self.rng=np.random.default_rng(950000+seed)
        self.stage,self.round='origin',0
        self.work=dict(environment_ticks=0,lane_ticks=0,gradient_updates=0,gradient_lane_ticks=0,
                       B_gradient_label_positions=0,B_unique_label_positions=0,B_unique_nonwait_labels=0,
                       B_unique_new_commands=0,B_unique_replace_commands=0,old_history_replay_positions=0)
        self.binding=dict(origin_sha256=plan['origins'][str(seed)]['sha256'],protocol_sha256=sha(PLAN))
        with mapping_scope(self.table):
            state=self.world.state_dict()
        # Only never-issued context rows differ. Existing cue3 targets remain exact.
        with mapping_scope(self.table):
            self.world=worlds.ContextWorld.from_state_dict(state,device='cpu')
        self.save('origin')

    @classmethod
    def from_checkpoint(cls,path,folder,table=None):
        """Restore a complete B0 branch; optional new B rule must be unissued."""
        self=cls.__new__(cls)
        self.model,self.world,self.hidden,cp=load_checkpoint(path,'cuda')
        self.hidden=self.hidden.cpu();self.seed=cp['transfer']['seed'];self.arm=cp['transfer']['arm']
        self.spec=deepcopy(cp['transfer']['spec']);self.plan=checked()
        self.table=deepcopy(cp['mapping_table'] if table is None else table)
        self.cfg=plain.Config(**cp['config']);self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.cfg.lr)
        self.optimizer.load_state_dict(deepcopy(cp['optimizer']))
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.rng=np.random.default_rng();self.rng.bit_generator.state=deepcopy(cp['numpy_rng'])
        torch.set_rng_state(cp['torch_rng'])
        if cp.get('cuda_rng'):torch.cuda.set_rng_state_all(cp['cuda_rng'])
        self.stage,self.round=cp['stage'],cp['round'];self.work=deepcopy(cp['work']);self.origin=cp
        self.binding=dict(origin_sha256=cp['transfer']['origin_sha256'],protocol_sha256=cp['transfer']['protocol_sha256'],
                          branch_checkpoint=str(Path(path).resolve()),branch_checkpoint_sha256=sha(path))
        state=self.world_state()
        with mapping_scope(self.table):self.world=worlds.ContextWorld.from_state_dict(state,device='cpu')
        return self

    def world_state(self):
        with mapping_scope(self.table):return self.world.state_dict()

    def save(self,name):
        with mapping_scope(self.table):
            state=self.world.state_dict()
        cp=dict(format=plain.FORMAT,model_config=self.model.config(),
                model={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                optimizer=deepcopy(self.optimizer.state_dict()),world=state,hidden=self.hidden.detach().cpu().clone(),
                config=asdict(self.cfg),stage=self.stage,round=self.round,work=deepcopy(self.work),
                numpy_rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                mapping_table=self.table,transfer=dict(seed=self.seed,arm=self.arm,B_round=self.round if self.stage=='B' else None,
                history_rule=self.spec['C'] if self.arm=='disjoint' else self.spec['A'],spec=self.spec,**self.binding),
                source_sha256=sha(__file__),saved_time=time.time())
        target=self.folder/(name+'.pt')
        temporary=target.with_suffix('.pt.tmp')
        torch.save(cp,temporary); temporary.replace(target)
        write(self.folder/(name+'.json'),dict(path=str(target),sha256=sha(target),stage=self.stage,
              round=self.round,work=self.work,hidden_norm=float(self.hidden.norm()),world_tick=self.world.tick,**self.binding))
        print(json.dumps(dict(event='checkpoint',seed=self.seed,arm=self.arm,name=name,work=self.work)),flush=True)

    def observe_step(self,policy,mode,teacher_lanes=None):
        with mapping_scope(self.table):
            obs=self.world.observe()
            labels=self.world.oracle('midpoint') if mode!='direct' else None
        with torch.no_grad():
            logits,self.hidden=policy.step(obs,self.hidden)
        chosen=logits.argmax(-1)
        actions=chosen if mode=='direct' else torch.where(teacher_lanes[:,None],labels,chosen)
        with mapping_scope(self.table):
            self.world.act(actions)
        return obs,labels,actions

    def collect_history(self,index):
        beta=max(0.,1.-index/6)
        lanes=torch.from_numpy(self.rng.random(self.cfg.batch)<beta)
        policy=deepcopy(self.model).cpu().eval()
        anchor=self.hidden.clone(); rows=[]
        initial=self.world_state()
        for _ in range(TICKS):
            rows.append(self.observe_step(policy,'direct' if self.arm=='direct' else 'history',lanes))
        self.work['environment_ticks']+=TICKS; self.work['lane_ticks']+=TICKS*self.cfg.batch
        data=dict(obs=torch.stack([r[0] for r in rows]),actions=torch.stack([r[2] for r in rows]),
                  anchor=anchor,teacher_probability=beta if self.arm!='direct' else 0.,teacher_lanes=lanes,
                  initial_world=initial,label_policy='oracle midpoint' if self.arm!='direct' else 'none',
                  labels=torch.stack([r[1] for r in rows]) if self.arm!='direct' else None)
        torch.save(data,self.folder/f'trajectory_history_{index:03d}.pt')
        return data

    def update_block(self,obs,labels,mask,h):
        logits,h=self.model(obs.cuda(),h.cuda())
        labels,mask=labels.cuda(),mask.cuda()
        weights=torch.ones(11,device='cuda'); weights[0]=self.cfg.wait_weight
        losses=F.cross_entropy(logits.flatten(0,2),labels.flatten(),weight=weights,reduction='none').reshape_as(labels)
        denominator=(weights[labels]*mask).sum()
        if denominator>0:
            loss=(losses*mask).sum()/denominator
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            self.optimizer.step()
            value=float(loss.detach())
        else:
            value=0.
        self.work['gradient_updates']+=1
        self.work['gradient_lane_ticks']+=len(obs)*self.cfg.batch
        if self.stage=='B': self.work['B_gradient_label_positions']+=int(mask.sum())
        return h.detach(),value

    def fit(self,data,epochs):
        losses=[]
        for _ in range(epochs):
            h=data['anchor']
            for start in range(0,len(data['obs']),BLOCK):
                mask=data.get('mask',torch.ones_like(data['labels'],dtype=torch.bool))[start:start+BLOCK]
                h,loss=self.update_block(data['obs'][start:start+BLOCK],data['labels'][start:start+BLOCK],mask,h)
                losses.append(loss)
        return sum(losses)/len(losses)

    def run_history(self):
        self.stage='history'
        with mapping_scope(self.table):
            self.world.set_family(3 if self.arm=='direct' else self.spec['history_cue'])
            self.world.set_regime(duration_range=(1.,16.),gap_range=(.5,32.),name='experience-transfer/history')
        for index in range(HISTORY_ROUNDS):
            self.round=index+1; start=time.perf_counter()
            data=self.collect_history(index)
            loss=None if self.arm=='direct' else self.fit(data,EPOCHS)
            row=dict(event='history_round',seed=self.seed,arm=self.arm,round=self.round,loss=loss,
                     seconds=time.perf_counter()-start,work=deepcopy(self.work))
            self.log(row)
            if self.round in (4,8,12): self.save(f'history{self.round:03d}')
        if self.arm=='direct':
            assert all(torch.equal(self.model.state_dict()[k].cpu(),v) for k,v in self.origin['model'].items())

    def log(self,row):
        with (self.folder/'updates.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)


def B_training(learners,rounds=B_ROUNDS):
    seed=learners[0].seed; cue=learners[0].spec['B_cue']
    report=dict(seed=seed,B_feedback_position_exact=True,B_actions_exact=True,
                obs_unequal_positions=0,obs_unequal_B_slot_positions=0,obs_unequal_nonB_slot_positions=0,
                obs_unequal_ticks=0,first_unequal_tick=None,last_unequal_tick=None,
                compared_to='related',by_arm={v.arm:dict(unequal_positions=0) for v in learners[1:]})
    for v in learners:
        v.stage='B';v.round=0;v.save('B000')
        with mapping_scope(v.table):
            v.world.set_family(cue)
            v.world.set_regime(duration_range=(1.,16.),gap_range=(.5,32.),name='experience-transfer/B')
        # No oracle call or gradient feedback at this common boundary observation.
        policy=deepcopy(v.model).cpu().eval()
        with mapping_scope(v.table): obs=v.world.observe()
        with torch.no_grad(): _,v.hidden=policy.step(obs,v.hidden)
        with mapping_scope(v.table): v.world.act(torch.zeros((v.cfg.batch,2),dtype=torch.long))
        v.work['environment_ticks']+=1;v.work['lane_ticks']+=v.cfg.batch
    target_folder=learners[0].folder.parent
    boundary_states={v.arm:dict(world=v.world_state(),hidden=v.hidden.clone()) for v in learners}
    torch.save(boundary_states,target_folder/'B_boundary.pt')
    for round_index in range(rounds):
        starttime=time.perf_counter(); collected={v.arm:[] for v in learners}
        anchors={v.arm:v.hidden.clone() for v in learners}
        initial={v.arm:v.world_state() for v in learners}
        losses={v.arm:[] for v in learners}
        for v in learners: v.round=round_index+1
        for block in range(TICKS//BLOCK):
            policies={v.arm:deepcopy(v.model).cpu().eval() for v in learners}
            block_rows={v.arm:[] for v in learners}
            block_anchors={v.arm:v.hidden.clone() for v in learners}
            for t in range(BLOCK):
                rows=[]
                for v in learners:
                    with mapping_scope(v.table):
                        obs=v.world.observe(); target=v.world.oracle('midpoint')
                    mask=v.world.current_family==cue
                    labels=torch.where(mask,target,torch.zeros_like(target))
                    rows.append((obs,labels,mask))
                ref=rows[0]
                for v,row in zip(learners[1:],rows[1:]):
                    assert torch.equal(row[1],ref[1]) and torch.equal(row[2],ref[2]), 'B feedback mismatch'
                    unequal=row[0]!=ref[0]
                    n=int(unequal.sum());report['by_arm'][v.arm]['unequal_positions']+=n
                    report['obs_unequal_positions']+=n
                    slot_unequal=unequal.reshape(v.cfg.batch,2,28)
                    report['obs_unequal_B_slot_positions']+=int((slot_unequal*ref[2][...,None]).sum())
                    report['obs_unequal_nonB_slot_positions']+=int((slot_unequal*(~ref[2])[...,None]).sum())
                    if n:
                        tick=round_index*TICKS+block*BLOCK+t
                        report['obs_unequal_ticks']+=1
                        if report['first_unequal_tick'] is None:report['first_unequal_tick']=tick
                        report['last_unequal_tick']=tick
                # Common real demonstration actions; old non-B pending slots WAIT.
                actions=ref[1]
                for v,row in zip(learners,rows):
                    with torch.no_grad(): _,v.hidden=policies[v.arm].step(row[0],v.hidden)
                    with mapping_scope(v.table):
                        new=((v.world.event_kind==worlds.NEW)&(v.world.event_family==cue))
                        replace=((v.world.event_kind==worlds.REPLACE)&(v.world.event_family==cue))
                        v.work['B_unique_new_commands']+=int(new.sum())
                        v.work['B_unique_replace_commands']+=int(replace.sum())
                        v.world.act(actions)
                    v.work['B_unique_label_positions']+=int(row[2].sum())
                    v.work['B_unique_nonwait_labels']+=int((row[1]!=0).sum())
                    block_rows[v.arm].append((row[0],row[1],actions.clone(),row[2]))
            for v in learners:
                v.work['environment_ticks']+=BLOCK;v.work['lane_ticks']+=BLOCK*v.cfg.batch
                rows=block_rows[v.arm]
                data=dict(obs=torch.stack([r[0] for r in rows]),labels=torch.stack([r[1] for r in rows]),
                          actions=torch.stack([r[2] for r in rows]),mask=torch.stack([r[3] for r in rows]),anchor=block_anchors[v.arm])
                _,loss=v.update_block(data['obs'],data['labels'],data['mask'],data['anchor'])
                losses[v.arm].append(loss);collected[v.arm].append(data)
                if round_index==0 and block+1 in (2,8):v.save(f'Bfirst{block+1:03d}')
        for v in learners:
            data={key:torch.cat([d[key] for d in collected[v.arm]]) for key in ('obs','labels','actions','mask')}
            data.update(anchor=anchors[v.arm],initial_world=initial[v.arm],actual_action_policy='common B midpoint demonstration; nonB WAIT',
                        label_policy='B midpoint only; nonB masked; no prior history replay',first_epoch='online128tick blocks')
            torch.save(data,v.folder/f'trajectory_B_{round_index:03d}.pt')
            loss=v.fit(data,EPOCHS-1)
            v.log(dict(event='B_round',seed=seed,arm=v.arm,round=round_index+1,loss=loss,
                       first_epoch_block_loss=losses[v.arm],seconds=time.perf_counter()-starttime,work=deepcopy(v.work)))
            if round_index+1 in (1,2,4,8):v.save(f'B{round_index+1:03d}')
        write(target_folder/'B_pairing_audit.json',report)
    return report


def run(seed):
    torch.set_num_threads(1);torch.manual_seed(900000+seed);np.random.seed(900000+seed)
    assert torch.cuda.is_available()
    plan=checked()
    learners=[Learner(seed,arm,plan) for arm in ARMS]
    for v in learners:v.run_history()
    report=B_training(learners)
    checked()
    write(OUT/f'seed_{seed:03d}'/'complete.json',dict(passed=True,seed=seed,B_pairing=report,
          protocol_sha256=sha(PLAN),work={v.arm:v.work for v in learners}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('declare','run','all'))
    parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=='declare':declare()
    elif args.command=='run':run(args.seed)
    else:
        results=[]
        for seed in SEEDS:
            log=OUT/f'seed_{seed:03d}.log'
            with log.open('w',encoding='utf-8') as stream:
                proc=subprocess.run([sys.executable,'-u','-m','experiments.experience_transfer.train_transfer','run','--seed',str(seed)],
                                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            results.append(dict(seed=seed,exit_code=proc.returncode))
            write(OUT/'exit_codes.json',results)
            if proc.returncode:raise RuntimeError(f'seed{seed} failed; see {log}')


if __name__=='__main__':main()
