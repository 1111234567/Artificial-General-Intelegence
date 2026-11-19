import numpy as np

from experiments.pulse_game import open_graph_remap as study


def test_remap_preserves_sensory_life_and_changes_balanced_half_only():
    roots=[11,12,20]
    rows=[(0,0,1,rule,"unused") for rule in range(3)]
    parts=study.design(roots,rows)
    first=study.scoring_tape(parts,rows,0)
    second=study.scoring_tape(parts,rows,1)
    assert first[1] is second[1]
    assert np.all(parts[-1].sum(axis=-1)==10)
    baseline=study.sensory.truth()[:,None,:]
    assert np.all((parts[-1]&baseline).sum(axis=-1)==5)
    for lane,idx in enumerate(parts[2]):
        a,b=(t[0].good_id[lane] for t in (first,second))
        start=parts[4][idx]
        np.testing.assert_array_equal(a[:start],b[:start])
        np.testing.assert_array_equal(a[start:start+60],1-b[start:start+60])
        np.testing.assert_array_equal(a[start+60:],b[start+60:])
        for pos in range(start,start+60,20):
            assert np.unique(parts[3][idx,pos:pos+20]).size==20
