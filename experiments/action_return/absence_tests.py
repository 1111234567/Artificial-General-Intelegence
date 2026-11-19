"""Mechanical-only source adoption, frozen update, restore and rejection audit."""
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from experiments.action_return.absence import AbsenceContinuation, FutureSchedule, START, WITHDRAW_END, RETURN_END
from experiments.action_return.absence_study import OUT, sources
from experiments.continuity.run import state_digest, restore_rng_state
from experiments.undirected.run import _atomic_json, file_sha256


def validate():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);records=[];started=time.perf_counter()
    for src in sources():
        if src['seed']!=0:continue
        run=AbsenceContinuation(src['checkpoint'],src['arm']);t=run.trainer
        before=run.state_dict();mask=t.operator.mask.clone();edge=t.model.edge_shared.detach().clone()
        if t.operator.pending is None or t.operator.pending['execute_update']!=t.updates+1:raise ValueError('test must archive an actually pending proposal')
        first=run.train_update();run.validate();after=run.state_dict()
        if not torch.equal(t.operator.mask,mask) or t.operator.probe_calls!=before['native']['operator']['origin']['probe_calls']:raise ValueError('pending proposal executed')
        if torch.equal(t.model.edge_shared,edge):raise ValueError('weight learning did not continue')
        restored=AbsenceContinuation.from_state(after)
        if state_digest(restored.semantic_state())!=state_digest(run.semantic_state()):raise ValueError('restore mismatch')
        saved_rng=deepcopy(after['native']['rng']);a=run.train_update();a_hash=state_digest(run.semantic_state())
        restore_rng_state(saved_rng);b=restored.train_update();b_hash=state_digest(restored.semantic_state())
        if state_digest(a)!=state_digest(b) or a_hash!=b_hash:raise ValueError('next actual native update mismatch')
        bad=deepcopy(after);bad['native']['operator']['origin']['mask'][0]=~bad['native']['operator']['origin']['mask'][0]
        rejected=False
        try:AbsenceContinuation.from_state(bad)
        except ValueError:rejected=True
        if not rejected:raise ValueError('tamper accepted')
        schedule=FutureSchedule.from_state(after['native']['schedule'])
        if schedule.mode_for_tick(START-1)!=schedule.original.mode_for_tick(START-1):raise ValueError('past schedule changed')
        if schedule.mode_for_tick(START)['future_family_allowed']!=(1,2,3) or schedule.mode_for_tick(WITHDRAW_END)['future_family_allowed']!=(0,):raise ValueError('phase boundaries')
        if schedule.original.main_ticks!=196608 or RETURN_END>=schedule.original.main_ticks:raise ValueError('future overlay failed to precede original withdrawal')
        records.append(dict(arm=src['arm'],seed=0,source_sha256=src['checkpoint_sha256'],adoption_preserves_all_except_schedule_operator=True,
                            source_tick=START,source_pending_archived=True,no_commit_or_probe=True,weights_continue_learning=True,
                            complete_restore_equal=True,next_actual_update_bitwise_equal=True,tamper_rejected=True,
                            historical_schedule_unchanged=True,original_main_boundary=196608,
                            next_point_hash=state_digest(a),next_semantic_hash=a_hash))
    value=dict(format='absence-mechanical-validation-v1',audit='passed',records=records,seconds=time.perf_counter()-started,
               source_files={str(Path(__file__).with_name(n)):file_sha256(Path(__file__).with_name(n)) for n in ('absence.py','absence_study.py','absence_tests.py')},
               scope='cloned seed0 mechanical updates only, no behavior scores; excluded from scientific budgets and endpoints')
    value['validation_hash']=state_digest(value);_atomic_json(OUT/'validation.json',value);return value


if __name__=='__main__':print(json.dumps(validate(),indent=2))
