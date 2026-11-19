"""Independently replay one continuous-state fit and verify all archived outputs."""
from io import BytesIO
from pathlib import Path
import tempfile
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from . import self_dose_continuous as s


def main():
    scores=trials=0
    for archive_name, manifest, member_key in (
            ("traces-and-checkpoints.zip","summary.json","archive"),
            ("narrow-traces-and-checkpoints.zip","capacity.json","archive_sha256")):
        archive=s.EVIDENCE/archive_name
        assert s.io.sha(archive)==s.io.read(s.EVIDENCE/manifest)[member_key]
        with ZipFile(archive) as zip_file:
            assert zip_file.testzip() is None
            for name in zip_file.namelist():
                with np.load(BytesIO(zip_file.read(name)),allow_pickle=False) as data:
                    if name.endswith(".scores.npz"):
                        for env in ("short","clean","noise","change"):
                            started=data[env+"/live"]
                            assert np.isin(started,(-1,1)).all()
                            np.testing.assert_array_equal(started[::2,:,1],-started[1::2,:,1])
                            for arm in ("live","disabled"):
                                key=env+"/"+arm
                                commands=data[key+"/commands"]
                                statuses=data[key+"/status"]
                                assert len(commands)==96 and commands.shape[1]==statuses.shape[1]
                                if arm=="disabled":assert (statuses!=1).all()
                                trials+=statuses.size
                        scores+=1
                    else:
                        assert np.count_nonzero(data["w"][~data["mask"]])==0
                        assert np.isfinite(data["curve"]).all()
                        assert data["contacts"].item()==69120
    example=(s.REG["births"][0],"fresh","all")
    old=s.paths(*example)
    with tempfile.TemporaryDirectory(prefix="agi-self-dose-replay-") as directory:
        with patch.object(s,"DATA",Path(directory)):
            s.train(example);s.evaluate(example)
            other=s.paths(*example)
            with np.load(old) as original,np.load(other) as replay:
                for key in ("w","b","mask","curve","seen","contacts"):
                    np.testing.assert_array_equal(original[key],replay[key])
            with np.load(old.with_suffix(".scores.npz")) as original,np.load(other.with_suffix(".scores.npz")) as replay:
                assert original.files==replay.files
                for key in replay.files:np.testing.assert_array_equal(original[key],replay[key])
    out=s.EVIDENCE
    fork=s.io.read(out/"fork-results.json")
    assert s.io.sha(out/"fork-traces.zip")==fork["raw_sha256"]
    s.io.write(out/"verification.json",dict(checkpoints_and_score_files=scores,checked_motor_opportunities=trials,
          independent_fit_and_all_holdouts_exact=list(example),sources=s.source_files(),
          verifier_sha256=s.io.sha(Path(__file__))))
    print(scores,trials,"verified")


if __name__=="__main__":main()
