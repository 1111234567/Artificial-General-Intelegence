"""Learn an association update rule on independent public-example streams."""
from __future__ import annotations
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.associative_state.model import AssociationMemory

OUT = ROOT / "data/associative-state/v1"
ARMS = ("add", "delta", "recurrent")
SETTINGS = dict(updates=384, batch=64, events=48, lr=.003, key_dim=16,
                online_lr=.003, support_ticks=1536, delay_ticks=4096,
                eval_ticks=2048, actors=32, dtype="float32")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def declare():
    path = OUT / "protocol.json"
    if path.exists():
        raise FileExistsError(path)
    names = ("model.py", "bridge.py", "train.py", "evaluate.py", "launch.py", "analysis-plan.md")
    sources = {str((Path(__file__).parent/n).relative_to(ROOT)): sha(Path(__file__).parent/n) for n in names}
    for name in ("experiments/learn_remember/gru_train.py", "experiments/lifelong/world.py",
                 "experiments/continuity/world.py", "experiments/experience_transfer/train_transfer.py",
                 "experiments/coverage_retention/train.py", "experiments/teacher_timing/data.py",
                 "experiments/teacher_timing/train.py"):
        sources[name] = sha(ROOT/name)
    from experiments.coverage_retention.train import source_checkpoint
    origins = {str(s): dict(path=str(source_checkpoint(s).relative_to(ROOT)), sha256=sha(source_checkpoint(s))) for s in range(3)}
    write_json(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), settings=SETTINGS,
                         arms=ARMS, seeds=[0,1,2], sources=sources, origins=origins))


def check_protocol():
    p = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert p["settings"] == SETTINGS
    for n,h in p["sources"].items():
        assert sha(ROOT/n) == h, n
    return p


def sample(rng, batch=64, events=48):
    """Each query targets the most recent *observed* value, never hidden changes."""
    tables = np.argsort(rng.random((batch,2,4)), axis=-1).reshape(batch,8)
    revised = tables.copy()
    # Two A associations change; old observations remain authoritative until taught.
    revised[:,0], revised[:,1] = tables[:,1], tables[:,0]
    seen = np.zeros((batch,8), bool)
    known = np.zeros((batch,8), np.int64)
    order = np.argsort(rng.random((batch,8)), axis=-1)
    rows = np.arange(batch)
    keys, values, queries, targets = [], [], [], []
    for t in range(events):
        k = order[:,t] if t<8 else rng.integers(0,8,batch)
        table = revised if t>=events//2 else tables
        v = table[rows,k]
        known[rows,k], seen[rows,k] = v, True
        q = np.argmax(np.where(seen, rng.random((batch,8)), -1.), axis=-1)
        keys.append(k); values.append(v); queries.append(q); targets.append(known[rows,q].copy())
    return tuple(torch.from_numpy(np.stack(x)) for x in (keys,values,queries,targets))


def train(seed, arm, device="cpu", pilot=False, steps=None):
    if not pilot:
        check_protocol()
    torch.set_num_threads(1)
    torch.manual_seed(51000+seed)
    np.random.seed(61000+seed)
    model=AssociationMemory(kind=arm, key_dim=SETTINGS["key_dim"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=SETTINGS["lr"])
    rng=np.random.default_rng(71000+seed)
    count=steps if pilot and steps is not None else SETTINGS["updates"]
    folder=OUT/f"seed_{seed:03d}"/arm
    if not pilot:
        folder.mkdir(parents=True,exist_ok=False)
        torch.save(model.state_dict(),folder/"initial.pt")
    log=[]; timer=time.perf_counter(); digest=hashlib.sha256()
    for update in range(count):
        batch=sample(rng,SETTINGS["batch"],SETTINGS["events"])
        for x in batch: digest.update(x.numpy().tobytes())
        keys,values,queries,targets=(x.to(device) for x in batch)
        state=model.initial_state(SETTINGS["batch"], device=device)
        losses=[]; correct=[]
        for k,v,q,target in zip(keys,values,queries,targets):
            state=model.write(state,k,v)
            logits=model.read(state,q)
            losses.append(F.cross_entropy(logits,target))
            correct.append((logits.argmax(-1)==target).float().mean().detach())
        loss=torch.stack(losses).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
        optimizer.step()
        if update==0 or (update+1)%64==0 or update+1==count:
            row=dict(update=update+1,loss=float(loss.detach()),accuracy=float(torch.stack(correct).mean()),grad_norm=grad,
                     seconds=time.perf_counter()-timer)
            log.append(row)
            print(json.dumps(dict(event="train",seed=seed,arm=arm,**row)),flush=True)
    result=dict(seed=seed,arm=arm,settings=SETTINGS,updates=count,model_config=model.config(),
                parameters=model.parameter_report(),train_sample_sha256=digest.hexdigest(),seconds=time.perf_counter()-timer,
                log=log,device=device,torch_version=torch.__version__)
    if not pilot:
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(),metadata=result),folder/"final.pt")
        result["checkpoint_sha256"]=sha(folder/"final.pt")
        write_json(folder/"training.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("declare","train","pilot"))
    p.add_argument("--seed",type=int,default=0);p.add_argument("--arm",choices=ARMS,default="delta")
    p.add_argument("--device",default="cpu");p.add_argument("--steps",type=int,default=4)
    a=p.parse_args()
    if a.command=="declare":declare()
    else:train(a.seed,a.arm,a.device,pilot=a.command=="pilot",steps=a.steps)
