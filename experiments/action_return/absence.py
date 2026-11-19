"""Versioned, inherited-support continuation of already trained native systems.

The native update is reused unchanged. Only its future schedule and pair
operator are replaced at an explicit, archived optimizer boundary. Historical
trace hashes are never rewritten. The outstanding proposal is retained as
evidence, permanently canceled, and cannot execute or resume in this protocol.
"""
from copy import copy, deepcopy
from pathlib import Path
import torch

from experiments.continuity import run as continuity
from experiments.undirected import run as native
from experiments.lifelong.schedule import Schedule
from experiments.lifelong.world import ContextWorld
from experiments.coverage_exit import retirement_run, retirement_magnitude
from experiments.coverage_exit.retirement import RetirementOperator
from experiments.coverage_exit.retirement_magnitude import MagnitudeOperator
from experiments.coverage_exit.fast_window import install_window

FORMAT = 'action-return-absence-continuation-v1'
OP_FORMAT = 'action-return-fixed-inherited-operator-v1'
START, WITHDRAW_END, RETURN_END = 147456, 163840, 172032


def source_trainer(path, arm):
    cls = retirement_magnitude.NativeTrainer if arm == 'magnitude' else retirement_run.NativeTrainer
    if arm not in ('magnitude', 'signed_full'):
        raise ValueError('unknown inherited system')
    return cls.from_checkpoint(path, device='cpu')


class FutureSchedule:
    def __init__(self, original):
        self.original, self.config = original, original.config

    def mode_for_tick(self, tick):
        if tick < START:
            return self.original.mode_for_tick(tick)
        if tick > RETURN_END:
            raise ValueError('future-only continuation complete')
        phase = 'absence_withdrawal' if tick < WITHDRAW_END else 'absence_return'
        return dict(phase=phase, name=phase, future_family_allowed=(1, 2, 3) if tick < WITHDRAW_END else (0,),
                    macrocycle=None, family_block=None)

    def state_dict(self):
        return dict(format='action-return-future-only-schedule-v1', original=self.original.state_dict(),
                    start=START, withdrawal_end=WITHDRAW_END, return_end=RETURN_END)

    @classmethod
    def from_state(cls, state):
        result = cls(Schedule.from_state_dict(state['original']))
        if continuity.state_digest(result.state_dict()) != continuity.state_digest(state):
            raise ValueError('future schedule schema/boundaries changed')
        return result


class FixedMixin:
    @torch.no_grad()
    def after_optimizer(self, optimizer, raw_gradients, clip_scale, probe=None):
        # Validate native Adam ages and all potential-coordinate gradients,
        # retaining inactive moments exactly as the original optimizer does.
        self._runtime(optimizer, raw_gradients, clip_scale)
        self.project_and_validate()
        if not torch.equal(self.mask, self.frozen_origin['mask']) or not torch.equal(self._support(), self.frozen_origin['support_mask']):
            raise ValueError('inherited eligibility or numeric support changed')
        self.updates += 1
        self.update_history.append(dict(update=self.updates, birth_indices=[], retire_indices=[],
                                        numeric_birth_indices=[], numeric_death_indices=[]))
        return {**self.summary(), 'update': self.updates, 'births': 0, 'deaths': 0, 'probe': None,
                'pending_generated': None, 'structural_numeric_births': 0, 'structural_numeric_deaths': 0}

    def summary(self, model=None):
        result = super().summary(model)
        result.update(structure='fixed_inherited', mode='fixed_inherited',
                      pending_status='archived and permanently canceled at adoption',
                      frozen_since_update=self.frozen_origin['updates'])
        return result

    def state_dict(self):
        result = dict(format=OP_FORMAT, arm=self.frozen_arm, origin=self.frozen_origin,
                      origin_edge_shared=self.frozen_edge_shared, updates=self.updates,
                      suffix_history=deepcopy(self.update_history[self.frozen_origin['updates']:]))
        result = continuity.cpu_tree(result)
        result['state_hash'] = continuity.state_digest(result)
        return result

    def validate_frozen(self):
        age = self.frozen_origin['updates']
        if (self.updates < age or len(self.update_history) != self.updates
                or continuity.state_digest(self.update_history[:age]) != continuity.state_digest(self.frozen_origin['update_history'])):
            raise ValueError('fixed operator lost historical prefix')
        expected = [dict(update=i, birth_indices=[], retire_indices=[], numeric_birth_indices=[], numeric_death_indices=[])
                    for i in range(age + 1, self.updates + 1)]
        if self.update_history[age:] != expected or not torch.equal(self.mask, self.frozen_origin['mask']) or not torch.equal(self._support(), self.frozen_origin['support_mask']):
            raise ValueError('fixed operator suffix/support invalid')
        self.model.validate_eligibility(self.mask)


class FixedMagnitude(FixedMixin, MagnitudeOperator):
    pass


class FixedSigned(FixedMixin, RetirementOperator):
    pass


def adopt_operator(operator, arm):
    operator.frozen_origin = operator.state_dict()
    operator.frozen_edge_shared = operator.model.edge_shared.detach().cpu().clone()
    operator.frozen_arm = arm
    operator.__class__ = FixedMagnitude if arm == 'magnitude' else FixedSigned
    operator.validate_frozen()
    return operator


def restore_operator(model, state):
    if state['format'] != OP_FORMAT or state['state_hash'] != continuity.state_digest({k:v for k,v in state.items() if k != 'state_hash'}):
        raise ValueError('fixed operator digest/schema')
    # Revalidate the complete historical operator using its actual historical
    # physical edge weights; do not validate old proposals against new weights.
    historical_model = deepcopy(model)
    with torch.no_grad():
        historical_model.edge_shared.copy_(state['origin_edge_shared'])
    cls = MagnitudeOperator if state['arm'] == 'magnitude' else RetirementOperator
    result = cls.from_state(historical_model, state['origin'])
    adopt_operator(result, state['arm'])
    result.model = model
    result.updates = state['updates']
    result.update_history += deepcopy(state['suffix_history'])
    result.validate_frozen()
    if continuity.state_digest(result.state_dict()) != continuity.state_digest(state):
        raise ValueError('fixed operator restore differs')
    return result


class AbsenceContinuation:
    def __init__(self, path, arm):
        self.path, self.arm = str(Path(path).resolve()), arm
        self.trainer = source_trainer(path, arm)
        before = self.trainer.semantic_state()
        if before['ticks'] != START or before['updates'] != START // self.trainer.config.chunk_ticks:
            raise ValueError('wrong adoption boundary')
        self.adoption = dict(format=FORMAT, source=self.path, source_sha256=native.file_sha256(path), arm=arm,
                             source_semantic_hash=continuity.state_digest(before),
                             source_field_hashes={k:continuity.state_digest(v) for k,v in before.items()},
                             source_trace_prefix_hash=continuity.state_digest(before['trace']),
                             original_schedule=before['schedule'], original_operator=before['operator'],
                             original_main_boundary=self.trainer.schedule.main_ticks,
                             pending_rule='archive entire outstanding proposal; permanently cancel; no resume',
                             schedule_rule='only future NEW policy changes; pending world events and old replay retained')
        adopt_operator(self.trainer.operator, arm)
        self.trainer.schedule = FutureSchedule(self.trainer.schedule)
        after = self.trainer.semantic_state()
        changed = {k for k in before if continuity.state_digest(before[k]) != continuity.state_digest(after[k])}
        if changed != {'operator', 'schedule'}:
            raise ValueError(f'adoption changed continuing state: {changed}')
        install_window(self.trainer)
        self.validate()

    def train_update(self):
        if self.trainer.ticks >= RETURN_END:
            raise ValueError('continuation complete')
        return self.trainer.train_update()

    def validate(self):
        t, a = self.trainer, self.adoption
        if native.file_sha256(self.path) != a['source_sha256']:
            raise ValueError('original checkpoint changed')
        if continuity.state_digest(t.trace[:START // t.config.chunk_ticks]) != a['source_trace_prefix_hash']:
            raise ValueError('rewritten historical trace')
        if continuity.state_digest(t.operator.frozen_origin) != a['source_field_hashes']['operator']:
            raise ValueError('operator archive differs from original boundary')
        t.operator.validate_frozen()
        # The unmodified native validator checks causal block bindings, original
        # trace compositions, credit, hidden/world/Adam age and latest boundary.
        # Exactly at adoption, use a shallow historical view because the final
        # old trace correctly commits the original operator and schedule.
        view = copy(t)
        if t.ticks == START:
            cls = MagnitudeOperator if self.arm == 'magnitude' else RetirementOperator
            view.operator = cls.from_state(t.model, a['original_operator'])
            view.schedule = Schedule.from_state_dict(a['original_schedule'])
        native.NativeTrainer._validate_continuation(view)

    def semantic_state(self):
        return dict(format=FORMAT, adoption=deepcopy(self.adoption), native=self.trainer.semantic_state())

    def state_dict(self):
        self.validate()
        value = self.semantic_state()
        value.update(source_files={str(Path(__file__).resolve()):native.file_sha256(__file__)},
                     original_sources=deepcopy(self.trainer.source_hashes),
                     original_dependencies=deepcopy(self.trainer.dependency_hashes))
        value['state_hash'] = continuity.state_digest(value)
        return value

    def save_checkpoint(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.tmp');torch.save(self.state_dict(),tmp);tmp.replace(path)

    @classmethod
    def from_state(cls, state):
        if state['format'] != FORMAT or state['state_hash'] != continuity.state_digest({k:v for k,v in state.items() if k != 'state_hash'}):
            raise ValueError('outer checkpoint digest/schema')
        for path, sha in state['source_files'].items():
            if native.file_sha256(path) != sha:
                raise ValueError('continuation runtime source mismatch')
        a, s = state['adoption'], state['native']
        result = cls(a['source'], a['arm'])
        t = result.trainer
        if continuity.state_digest(result.adoption) != continuity.state_digest(a) or t.source_hashes != state['original_sources'] or t.dependency_hashes != state['original_dependencies']:
            raise ValueError('original adoption/source dependency mismatch')
        for name, expected in (('config', continuity.cpu_tree(t.semantic_state()['config'])),
                               ('model_config',t.model.config()),('parameter_layout',native._layout(t.model)),
                               ('retirement_origin',t.retirement_origin)):
            if continuity.state_digest(s[name]) != continuity.state_digest(expected):
                raise ValueError(f'immutable native field changed: {name}')
        native._validate_tensor_map(s['model'],t.model.state_dict(),'model')
        t.model.load_state_dict(s['model'],strict=True)
        native._validate_optimizer(s['optimizer'],t.optimizer,t.model,s['updates'])
        t.optimizer.load_state_dict(deepcopy(s['optimizer']))
        t.h = s['hidden'].detach().clone()
        t.world = ContextWorld.from_state_dict(s['world'],device=t.config.world_device)
        t.schedule = FutureSchedule.from_state(s['schedule'])
        t.replay.load_state_dict(s['replay'])
        t.operator = restore_operator(t.model,s['operator'])
        t.credit.load_state_dict(s['credit'])
        native._validate_tensor_map(s['stats'],t.stats.state_dict(),'statistics')
        t.stats.load_state_dict(s['stats'])
        custom = {'format','config','model_config','parameter_layout','model','optimizer','hidden','world',
                  'schedule','replay','operator','credit','stats','rng','retirement_origin'}
        for name in s.keys()-custom:
            setattr(t,name,continuity.cpu_tree(s[name]))
        native._validate_rng(s['rng']);continuity.restore_rng_state(s['rng'])
        result.validate()
        if continuity.state_digest(result.semantic_state()) != continuity.state_digest({k:state[k] for k in ('format','adoption','native')}):
            raise ValueError('complete continuation restore differs')
        install_window(t)
        return result

    @classmethod
    def from_checkpoint(cls,path):
        return cls.from_state(torch.load(path,map_location='cpu',weights_only=False))
