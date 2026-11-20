from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import double_visit_localize as probe
from experiments.pulse_game import double_visit_world as world


def test_direct_room_diagnostic_does_not_read_the_exit_or_future_goal():
    model=world.DoubleVisitGraph(731).requires_grad_(False)
    e=world.scenes(13257,6)
    state=world.prepare(model,e)
    q=torch.arange(len(e))%2
    # Change labels held by the experimenter after forking the actual live body.
    hidden=replace(e,observed_endpoints=(e.observed_endpoints+1)%3,goals=e.goals.roll(1,1))
    for tag in ("room_cue","door_cue"):
        x=probe.external_probe(model,state,e,q,tag)
        y=probe.external_probe(model,state,hidden,q,tag)
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_readout_exchange_preserves_same_body_physical_states():
    model=world.DoubleVisitGraph(733).requires_grad_(False)
    e=world.scenes(98813,8)
    tr=world.try_side(model,world.prepare(model,e),torch.arange(len(e))%2)
    ix=[14,16,17,18]
    returned=probe.present_after_readout(model,tr,given=tr.states[4].h[:,ix],units=ix)
    for field in ("h","fast","trace","candidate"):
        torch.testing.assert_close(getattr(returned,field),getattr(tr.final,field),rtol=0,atol=0)


def test_external_linear_calibration_uses_previously_assigned_physical_labels_only():
    labels=np.tile(np.arange(3),4).reshape(6,2)
    profile=np.eye(3)[labels]
    y=probe.readout(profile,labels,profile[::-1])
    np.testing.assert_array_equal(y.argmax(2),labels[::-1])
