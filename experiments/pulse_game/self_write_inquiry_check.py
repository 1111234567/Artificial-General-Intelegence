"""Verify the native inquiry assay, including what its own contact can reveal."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import self_write_inquiry as inquiry, world


def physical_reference(sides,roots):
    tape=world.make_tape(tuple(map(int,roots)),trials=1,distance_jitter=.18,wind_scale=.065)
    body=inquiry.Body(tape,np.zeros((len(roots),2),dtype=np.int8))
    pulses=np.column_stack((sides<0,sides>0))
    for _ in range(world.TICKS):
        body.step(pulses)
    actual=body.result().contact_side[:,0]
    return np.where(actual<0,0,2*actual-1)


def main():
    source=inquiry.EVIDENCE
    summary=inquiry.io.read(source/"summary.json")
    zip_path=source/"raw.zip"
    assert summary["archive_sha256"]==inquiry.io.sha(zip_path)
    assert summary["source_sha256"]==inquiry.io.sha(Path(inquiry.__file__))
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
        with np.load(BytesIO(z.read("outcomes.npz")),allow_pickle=False) as archived:
            vals=dict(archived)
    for idx,birth in enumerate(inquiry.REG["births"]):
        live=inquiry.evaluate(birth)
        assert set(live)==set(vals)
        for key,x in live.items():np.testing.assert_array_equal(x,vals[key][idx])
    refs={}
    for route in ("sentinel","decoy"):
        np.testing.assert_array_equal(vals[f"{route}/0/side"],vals[f"{route}/1/side"])
        tastes0,tastes1=(vals[f"{route}/{k}/taste"] for k in (0,1))
        np.testing.assert_array_equal(tastes0,-tastes1 if route=="sentinel" else tastes1)
        # The decoder remembers only which encountered odor was tasted and its own
        # signed outcome. Present the same odor to the right and its inverse to the
        # left on a new silent probe, with a fair side permutation. Decoy tastes
        # carry no evidence about either hidden target. The action choice uses no target.
        sign=vals["sentinel_sign"].astype(np.int8)
        rng=np.random.default_rng(353900)
        flip=rng.choice((-1,1),size=sign.shape)
        totals=[]
        for twin,taste in enumerate((tastes0,tastes1)):
            decision=np.where(taste>0,flip,-flip) if route=="sentinel" else np.ones_like(flip)
            reached=physical_reference(decision.ravel(),np.arange(decision.size)+27870300).reshape(decision.shape)
            correct=flip*sign*(1 if twin==0 else -1)
            totals.append(float((reached==correct).mean()))
        refs[route]=float(np.mean(totals))
    result=dict(frozen_births_replayed=len(vals["layout"]),archived_arrays=len(vals),
                exact_replay=True, public_item_match_reference=refs,
                replay_core_sha256=inquiry.io.sha(Path(inquiry.core.__file__)),
                original_core_sha256=summary["core_sha256"],
                source_sha256=inquiry.io.sha(Path(__file__)))
    inquiry.io.write(source/"verification.json",result)
    print(result)


if __name__=="__main__":
    main()
