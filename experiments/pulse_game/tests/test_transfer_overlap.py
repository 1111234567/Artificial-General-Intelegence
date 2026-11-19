import numpy as np

from experiments.pulse_game import relation_world as world, transfer_overlap as sweep


def test_angle_changes_the_old_map_and_preserves_paired_controls():
    roots=[9851,9863,9871]
    base=world.make(roots,partition="develop",history_pairs=4,probes=2,doses=(0,2,4),nonce=73)
    old=base.history["related"]
    for angle in (0,30,60,90):
        variant=sweep.prior_at_angle(base,angle)
        assert variant.teaching is base.teaching
        assert variant.probes["new"] is base.probes["new"]
        new_old=variant.history["related"]
        np.testing.assert_array_equal(old.menu.chemicals[:,1::3],new_old.menu.chemicals[:,1::3])
        np.testing.assert_array_equal(old.menu.realized,new_old.menu.realized)
        np.testing.assert_array_equal(old.menu.body.wind,new_old.menu.body.wind)
        for s in roots:
            source,target=sweep.planes(s,base.nonce,angle)
            np.testing.assert_allclose(np.linalg.svd(source.T@target,compute_uv=False),
                                       [np.cos(np.deg2rad(angle))]*2,atol=1e-12)
        if angle==0:
            np.testing.assert_array_equal(old.menu.chemicals,new_old.menu.chemicals)
            np.testing.assert_array_equal(base.probes["old_related"].menu.chemicals,variant.probes["old_related"].menu.chemicals)
        else:
            assert np.any(old.menu.chemicals[:,2::3] != new_old.menu.chemicals[:,2::3])
    donors=sweep.sign_twins(base)
    np.testing.assert_array_equal(base.parity[:,0],base.parity[donors,0])
    np.testing.assert_array_equal(base.parity[:,1],-base.parity[donors,1])
    np.testing.assert_array_equal(np.array(base.seeds),np.array(base.seeds)[donors])


def test_gate_replay_preserves_missing_receipts_and_initial_history():
    donors=np.array([1,0,3,2])
    w=np.array([.1,.6,.3,np.nan])
    trace={2:(w,None,np.isfinite(w)),5:(w,None,np.isfinite(w))}
    result,missing=sweep.sibling_schedule(trace,3,donors)
    assert set(result)=={5}
    np.testing.assert_allclose(result[5],[.6,.1,.3,0.])
    assert missing==1
