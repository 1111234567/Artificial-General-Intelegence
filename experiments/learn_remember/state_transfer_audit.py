"""Independently replay a state transplant with the original lifelong evaluator."""
import argparse
import json
import torch

from experiments.learn_remember import state_transfer as s
from experiments.lifelong.evaluate import evaluate
from experiments.continuity.run import state_digest


def verify(seed):
    torch.set_num_threads(1)
    checks=[]
    for r in json.loads((s.OUT/f"seed_{seed:03d}.json").read_text(encoding="utf-8")):
        p=s.m.BASE/("gru-v1" if r["arch"]=="plain" else "symmetric-v1")/f"seed_{seed:03d}"/"wide.pt"
        assert r["checkpoint_sha"]==s.m.sha(p)
        model,_,saved_h,_=s.load_checkpoint(p)
        donor=saved_h[:8].clone()
        world=s.ContextWorld.from_state_dict(s.e.common_stream(seed)["world"],device="cpu")
        old=evaluate(model.eval(),world,donor,family=3,condition="long_both",steps=2048,device="cpu")
        for k in ("matured","completed","clean_completed"):
            assert r["counts_transplant"][k]==old["primary"][k]
        assert r["counts_transplant"]["idle_actions"]==old["opportunities"]["inactive_actions"]
        assert r["counts_transplant"]["idle_opportunities"]==old["opportunities"]["inactive_opportunities"]
        # The two evaluators bind the same tensor with different key containers.
        assert r["donor_h_hash"]==state_digest(donor)
        assert old["start_hidden_hash"]==state_digest({"hidden":donor})
        assert r["world_hash"]==old["start_world_state_hash"]
        checks.append(dict(arch=r["arch"],counts=r["counts_transplant"],action_sha=old["action_hash"],
                           end_h_sha=old["end_hidden_hash"],checkpoint_sha=s.m.sha(p)))
    (s.OUT/f"native-check-{seed}.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=range(3),required=True)
    verify(p.parse_args().seed)
