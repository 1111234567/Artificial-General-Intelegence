from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_age_train as age, interleave_reliability as testbed


def test_public_baseline_reasoning_uses_visible_keys_and_actual_tastes_only():
    cfg=testbed.Config(worlds=12)
    tape=testbed.batch(cfg,[731],4,"clean",testbed.DOMAIN+99)
    expected=testbed.reference(tape,"clean")
    np.testing.assert_array_equal(expected[:,2:],tape["clean"][:,2:].numpy())
    changed={**tape,"obj":-tape["obj"],"clean":-tape["clean"],"high_vol":~tape["high_vol"]}
    np.testing.assert_array_equal(testbed.reference(changed,"clean"),expected)


def test_graph_choices_are_invariant_to_scorer_only_hidden_truth_and_volatility():
    cfg=testbed.Config(worlds=2,contacts=6)
    tape=testbed.batch(cfg,[731],6,"mixed",testbed.DOMAIN+17)
    g,r=age.init([731],"cold")
    with torch.no_grad():
        full=testbed.run_life(g,r,tape,cfg)
        changed=testbed.run_life(g,r,{**tape,"clean":-tape["clean"],"high_vol":~tape["high_vol"]},cfg)
    for a,b in zip(full,changed,strict=True):assert torch.equal(a,b)


def test_regime_worlds_share_inputs_and_randomness():
    cfg=testbed.Config(worlds=5,contacts=10)
    a=testbed.make(cfg,731,9,"clean")
    b=testbed.make(cfg,731,9,"mixed")
    for k in ("right","draw","wait","orient","obj"):
        np.testing.assert_array_equal(a[k],b[k])
