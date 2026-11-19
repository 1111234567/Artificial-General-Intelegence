"""Restore the original selected recipe path and verify the switch experiment from its archive."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np

from . import sparse_pair_switch as study


def prepare():
    source=study.EVIDENCE.parent/"alternative"
    selected=source/"development.json"
    if study.io.read(selected)["registered_sha256"]!=study.io.sha(source/"registered.json"):
        raise RuntimeError("Alternative selection does not match its registration")
    dest=study.io.REPO/"data/pulse-game/sparse-native-20261119-alternative/development.json"
    if dest.exists() and study.io.sha(dest)!=study.io.sha(selected):
        raise RuntimeError("Existing local alternative selection differs")
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(selected.read_bytes())
    chosen=study.io.read(study.MAIN/"development.json")
    if chosen["registered_sha256"]!=study.io.sha(study.MAIN/"registered.json"):
        raise RuntimeError("Primary selection does not match its registration")


def verify():
    prepare()
    root=study.EVIDENCE
    summary=study.io.read(root/"summary.json")
    assert summary["protocol_sha256"]==study.io.sha(root/"protocol.json")
    assert summary["code_sha256"]==study.io.sha(Path(study.__file__))
    assert summary["raw_sha256"]==study.io.sha(root/"raw.zip")
    modes=study.REG["modes"]
    matrix_checks=0
    with zipfile.ZipFile(root/"raw.zip") as z:
        assert z.testzip() is None and len(z.namelist())==len(modes)*(len(study.REG["births"])//2)
        for mode in modes:
            for i in range(0,len(study.REG["births"]),2):
                with np.load(BytesIO(z.read(f"{mode}-{i}.npz")),allow_pickle=False) as x:
                    for key in x.files:
                        if "/count" in key:continue
                        a=x[key];side,tick,sign,truth,shown=np.moveaxis(a,-1,0)
                        assert np.array_equal(side<0,tick<0) and np.array_equal(side<0,sign==0)
                        assert np.isin(truth,[0,1]).all()
                        if "/training" in key:
                            np.testing.assert_array_equal(shown,sign)
                        else:assert not shown.any()
                        matrix_checks+=side.size
                    for d in study.REG["B_doses"]:
                        np.testing.assert_array_equal(x[f"0/B/{d}"][...,3],1-x[f"1/B/{d}"][...,3])
                    np.testing.assert_array_equal(x["0/B/0"][...,:2],x["1/B/0"][...,:2])
                    if i==0:
                        fresh=study.simulate(study.REG["births"][:2],mode)
                        assert set(x.files)==set(fresh)
                        for key,arr in fresh.items():np.testing.assert_array_equal(arr,x[key])
    result=dict(physical_branch_rows=matrix_checks,first_two_births_replay_exact_all_modes=True,
                complementary_B_truth_and_zero_dose_action=True, recipes_checked=True,
                verifier_sha256=study.io.sha(Path(__file__)))
    study.io.write(root/"checks.json",result)
    print(result)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("phase",choices=("prepare","verify"))
    if p.parse_args().phase=="verify":verify()
    else:prepare()
