from pathlib import Path
import hashlib
import json
import torch
from experiments.learn_remember.gru_train import GRUAgent
from experiments.experience_transfer.train_transfer import mapping_table, SEEDS
from experiments.coverage_retention.train import source_checkpoint
from experiments.learned_binding.model import NativeMemoryPolicy

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/learned-binding/v1"
SETTINGS=dict(seeds=[0,1,2],training_streams=16,train_batch=32,train_ticks=1024,
    ce_updates=512,dense_updates=128,ce_lr=.001,wait_weight=.25,
    actors=32,support_ticks=4096,probe_ticks=2048,pg_blocks=32,pg_ticks=512,
    pg_lr=.0001,gamma=.995,reject_penalty=.1,dtype="float32",
    support_duration=[8.,16.],support_gap=[16.,32.])
KINDS=("manual_hard","manual_soft","learned")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_tree_equal(a,b,path="state"):
    import numpy as np
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:assert_tree_equal(a[k],b[k],f"{path}.{k}")
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):assert_tree_equal(x,y,f"{path}.{i}")
    else:assert a==b,path


def executor(seed):
    cp=torch.load(source_checkpoint(seed),map_location="cpu",weights_only=False)
    model=GRUAgent(**cp["model_config"])
    model.load_state_dict({k:v for k,v in cp["model"].items() if k.startswith(("core.","head."))})
    return model.eval().requires_grad_(False)


def new_native(seed,device="cpu"):
    torch.manual_seed(372000+seed)
    return NativeMemoryPolicy(executor=executor(seed),original_mappings=mapping_table(seed,"related")).to(device)


def native_checkpoint(seed):return OUT/f"seed_{seed:03d}"/"pretrain"/"final.pt"


def load_native(seed,device="cpu"):
    cp=torch.load(native_checkpoint(seed),map_location="cpu",weights_only=False)
    model=NativeMemoryPolicy(**cp["model_config"])
    model.load_state_dict(cp["model"])
    return model.to(device).eval()


def manual_checkpoint(seed):
    return ROOT/f"data/associative-state/shared-v2/seed_{seed:03d}/delta/final.pt"


def check_protocol():
    p=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert p["settings"]==SETTINGS
    for name,h in p["sources"].items():assert sha(ROOT/name)==h,name
    for row in p["origins"].values():assert sha(ROOT/row["path"])==row["sha256"]
    return p


def declare():
    from datetime import datetime,timezone
    path=OUT/"protocol.json"
    if path.exists():raise FileExistsError(path)
    names=("common.py","model.py","data.py","pretrain.py","interaction.py","run.py","analysis-plan.md")
    sources={str((Path(__file__).parent/n).relative_to(ROOT)):sha(Path(__file__).parent/n) for n in names}
    for name in ("experiments/associative_state/model.py","experiments/associative_state/shared.py",
        "experiments/associative_state/bridge.py","experiments/learn_remember/gru_train.py",
        "experiments/experience_transfer/train_transfer.py",
        "experiments/lifelong/world.py","experiments/continuity/world.py"):
        sources[name]=sha(ROOT/name)
    origins={}
    for seed in range(3):
        for label,p in (("executor",source_checkpoint(seed)),("manual",manual_checkpoint(seed))):
            origins[f"{seed}/{label}"]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    write_json(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),settings=SETTINGS,sources=sources,
        origins=origins,primary="Self-action acquisition and B-only learning, with frozen-parameter native control and unchanged manual hard/soft references.",
        teacher_diagnostic="Same actual teacher stream, separate from primary own-action acquisition.",
        training="Native action CE, followed by own sampled actions and only scalar public outcome PG. No payload/binding/gate auxiliary loss."))
