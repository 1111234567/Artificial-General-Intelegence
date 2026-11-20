"""Run the prespecified full-menu contextual replication from independent warm births."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import time

from . import interleave_context_time as task, local_trace_train as prior
from .interleave_interference import DATA


BIRTHS=tuple(range(811,817))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=DATA/"retention-objective-confirmation")
    p.add_argument("--out",type=Path,default=DATA/"context-timing-confirmation")
    p.add_argument("--follow",action="store_true")
    a=p.parse_args()
    parents=[prior.checkpoint_path(a.source,"odor","uniform",b,8000) for b in BIRTHS]
    while not all(x.is_file() for x in parents):
        if not a.follow:raise FileNotFoundError("Uniform-query parents are incomplete")
        time.sleep(15)
    cfg=task.Config(seed=915743)
    plan=dict(config=task.asdict(cfg),births=BIRTHS,arms=["full_direct"],source=str(a.source),
              source_step=8000,code_sha256=sha256(Path(task.__file__).read_bytes()).hexdigest(),
              parent_sha256={str(b):sha256(x.read_bytes()).hexdigest() for b,x in zip(BIRTHS,parents,strict=True)},
              selection="Replicate fixed full_direct 8000; no model chosen using confirmation performance.")
    a.out.mkdir(parents=True,exist_ok=True)
    task.common._atomic_save(a.out/"run.json",lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    task.fit(a.out,a.source,8000,cfg,BIRTHS,"full_direct",8000)
    task.evaluate(a.out,cfg,BIRTHS,["full_direct"],8000,9124367,192,"independent-full-direct-8000")


if __name__=="__main__":main()
