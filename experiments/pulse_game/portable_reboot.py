"""Prove exported full-state continuation across terminated and fresh Python processes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


CHILD="""
import builtins,hashlib,json,sys
sys.path.insert(0,sys.argv[1])
raw_import=builtins.__import__
def prohibit(name,*a,**k):
    if name.split('.')[0]=='torch':raise RuntimeError('Torch imported')
    return raw_import(name,*a,**k)
builtins.__import__=prohibit
import numpy as np
from lifelong import lifelong_agent as ap,unified_world as uw,habitat_world as hw
folder,stage,style,snapshot=sys.argv[1:5]
seeds=[7131,7159,7207,7219,7253,7283,7307,7349]
ep=uw.make(seeds,initial=12,depths=(0,2,4,8,12),span=2,noise=.05,nonce=751391)
w=hw.Body(ep.arena,silent_probes=True)
cut=31
if stage=='resume':
    a=ap.restore_life(snapshot)
    w.trial_index=cut;w.tick_index=0;w._load_trial()
else:
    a=ap.new_lives(seeds,address=style);a.reset(len(seeds))
obs=w.observe();actions=hashlib.sha256();ticks=0
while obs is not None:
    if stage=='stop' and w.trial_index==cut and w.tick_index==0:
        ap.save_life(snapshot,a);print(json.dumps({'saved_requests':a.round+1,'saved_ticks':a.it}));sys.exit(0)
    p=a.act(obs)
    if w.trial_index>=cut:actions.update(p.tobytes());ticks+=1
    obs=w.step(p)
res=w.result();state=hashlib.sha256()
for k in ap.STATE_ARRAYS:state.update(np.asarray(getattr(a,k)).tobytes())
if style.startswith('local'):
    for k in ('prototypes','counts','current','overflows'):state.update(getattr(a.places,k).tobytes())
if style=='local-selective':
    for k in ap.READER_ARRAYS:state.update(getattr(a,k).tobytes())
state.update(json.dumps([x.bit_generator.state for x in a.rng],sort_keys=True).encode())
print(json.dumps(dict(action_sha=actions.hexdigest(),receipt_sha=hashlib.sha256(res.observed_reward[:,cut:].tobytes()).hexdigest(),
    touch_sha=hashlib.sha256(res.contact_tick[:,cut:].tobytes()+res.contact_side[:,cut:].tobytes()).hexdigest(),
    state_sha=state.hexdigest(),actor_ticks=a.it,compared_motor_bits=ticks*2*len(seeds))))
"""


def main():
    folder=OUT/"morning-20261119";bundle=folder/"continuous-learning-cpu.zip";checks={}
    with tempfile.TemporaryDirectory(prefix="portable_agi_restart_") as tmp:
        path=Path(tmp)
        with zipfile.ZipFile(bundle) as z:z.extractall(path)
        for style in ("distributed","local","local-selective"):
            snapshot=path/(style+".npz");records={}
            for stage in ("full","stop","resume"):
                cmd=[sys.executable,"-I","-c",CHILD,str(path),stage,style,str(snapshot)]
                run=subprocess.run(cmd,capture_output=True,text=True,cwd=tmp,check=True,timeout=120)
                records[stage]=json.loads(run.stdout)
            if records["full"]!=records["resume"]:raise AssertionError("Process restart differs: "+style)
            checks[style]=dict(exact=True,split=records["stop"],trace=records["resume"],snapshot_bytes=snapshot.stat().st_size)
    _write(folder/"fresh-process-restart.json",dict(archive_sha=hashlib.sha256(bundle.read_bytes()).hexdigest(),
          note="Source process exited. A new process rebuilt the physical request boundary and loaded only the portable agent state.",
          checks=checks))


if __name__=="__main__":main()
