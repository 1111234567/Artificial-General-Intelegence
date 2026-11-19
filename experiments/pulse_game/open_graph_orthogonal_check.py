"""Recalculate transplant direction and replay one whole pair cohort from its birth."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import open_graph_orthogonal as study


def main():
    root=study.EVIDENCE
    summary=study.io.read(root/"summary.json")
    path=root/"physical.zip"
    assert study.io.sha(path)==summary["archive_sha256"]
    births=summary["births"];recipe=study.net.Recipe(**summary["recipe"])
    # The full score was regenerated with all silent probes on independent state copies.
    assert study.io.sha(Path(study.__file__))==summary["source_sha256"]
    checked=0
    with ZipFile(path) as z:
        assert z.testzip() is None
        def arr(name):return np.load(BytesIO(z.read(name+".npy")),allow_pickle=False)
        for a in ("joint","placebo"):
            fresh=study.simulate(births[0],a,recipe)
            for k,x in fresh.items():
                old=arr(a+"/"+k)[0].reshape(x.shape)
                np.testing.assert_array_equal(x,old)
            for stage,label in enumerate(study.REG["stages"]):
                for field in ("teach","silent"):
                    result=arr(f"{a}/{label}/{field}")
                    assert np.array_equal(result[...,0]<0,result[...,2]==0)
                    checked+=result[...,0].size
            if a=="placebo":
                for field in ("teach","silent"):
                    val=arr("placebo/B/"+field)
                    np.testing.assert_array_equal(val[:,0,:,:,:2],val[:,1,:,:,:2])
        intact=arr("joint/B/silent"); swap=arr("joint/B/synapses")
        activity=arr("joint/B/activity")
        checked+=swap[...,0].size+activity[...,0].size
        np.testing.assert_array_equal(intact[:,0,:,:,3],1-intact[:,1,:,:,3])
        own=swap[...,0]==swap[...,3]
        donor_truth=intact[:,::-1,:,:,3]
        report=dict(physical_branch_opportunities=checked,checked_one_birth_replay_all_arrays=True,
                    contacts=float((swap[...,0]>=0).mean()),recipient_accuracy=float(own.mean()),
                    donor_accuracy=float((swap[...,0]==donor_truth).mean()),
                    donor_side_reproduced=float((swap[...,0]==intact[:,::-1,:,:,0]).mean()),
                    donor_tick_reproduced=float((swap[...,1]==intact[:,::-1,:,:,1]).mean()),
                    per_birth_first_b_contact_identical=True, code_sha256=study.io.sha(Path(__file__)))
        b=arr("joint/B/teach")
        np.testing.assert_array_equal(b[:,0,:,0,:2],b[:,1,:,0,:2])
    study.io.write(root/"checks.json",report)
    print(report)


if __name__=="__main__":main()
