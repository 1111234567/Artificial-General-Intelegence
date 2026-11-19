"""Can the carried training state account for the deployment gap on identical cases?"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from experiments.learn_remember import evaluate_gru as e, memory_assay as m
from experiments.learn_remember.gru_train import load_checkpoint
from experiments.lifelong.world import ContextWorld


OUT=m.BASE/"state-transfer-v1"
ARCHES=("plain","symmetric")


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"protocol.json").open("x",encoding="utf-8") as f:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),script=m.sha(__file__),
            reason="Posthoc: the shared-pair wide checkpoint performed far better from the saved training endpoint than from the heldout comparison worlds. Those endpoints changed both world and hidden state.",
            intervention="Wide checkpoint only, both architectures, all3 seeds. Keep the standard independent WAIT-warmed eight-actor heldout world and exact future stream. Replace only its initial h with the trained agent's first eight real native endpoint lane states, in their original order; do not select or alter them. The donor actors and worlds are unrelated to the new requests.",
            sample="family3 long_both for2048 ticks; ordinary,strict and idle on the standard cases, and an additional suffix cohort issued only in the second1024 ticks and matured by the end. Compare against exact same stage's WAIT baseline. No training, online teacher or subsequent resets.",
            limit="Artificial state transplant, evidence about sensitivity to entry state; not a new deployment success criterion or proof of an attractor."),f,indent=2)


@torch.inference_mode()
def run(seed):
    torch.set_num_threads(1)
    plan=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert plan["script"]==m.sha(__file__)
    common=e.common_stream(seed)
    results=[]
    for arch in ARCHES:
        folder=m.BASE/("gru-v1" if arch=="plain" else "symmetric-v1")/f"seed_{seed:03d}"
        model,native,h,cp=load_checkpoint(folder/"wide.pt")
        world=ContextWorld.from_state_dict(common["world"],device="cpu")
        assert world.seed!=native.seed and set(world.actor_ids).isdisjoint(native.actor_ids)
        actual,frames,_=e.rollout(model.eval(),world,h[:8].clone(),family=3,condition="long_both",steps=2048)
        if arch=="plain":
            stem=e.OUT/f"seed_{seed:03d}/wide_common_long_both_family_3"
            base=e.read(stem.with_suffix(".json"))
            base["primary_requests"]=e.read(stem.with_suffix(".requests.json"))
            facts=stem.with_suffix(".npz")
        else:
            base=next(r for r in e.read(folder/"behavior_wide.json")["records"] if r["family"]==3 and r["condition"]=="long_both")
            facts=folder/"wide_long_both_f3.facts.npz"
        for key in ("start_world_hash","end_scheduler_hash","end_event_hash","primary_identity_hash"):
            assert base[key]==actual[key]
        def suffix(scores,actions,active):
            rows=[r for r in scores["primary_requests"] if r["request_id"]//2>scores["fork_tick"]+1024]
            idle=~active[1024:]
            return dict(matured=len(rows),completed=sum(r["completed"] for r in rows),
                clean_completed=sum(r["clean_completed"] for r in rows),idle_opportunities=int(idle.sum()),
                idle_actions=int((idle & (actions[1024:]!=0)).sum()),
                identities=sorted([r["actor"],r["job"],r["request_id"]] for r in rows))
        with np.load(facts) as old:
            original=suffix(base,old["actions"],old["active"])
        transplant=suffix(actual,frames["actions"],frames["active"])
        assert original["identities"]==transplant["identities"]
        results.append(dict(arch=arch,seed=seed,checkpoint_sha=m.sha(folder/"wide.pt"),donor_h_shape=list(h[:8].shape),
            counts_original=base["counts"],counts_transplant=actual["counts"],original_suffix=original,transplant_suffix=transplant,
            from_h_hash=base["start_h_hash"],donor_h_hash=actual["start_h_hash"],world_hash=actual["start_world_hash"],
            request_digest=actual["primary_identity_hash"]))
    with (OUT/f"seed_{seed:03d}.json").open("x",encoding="utf-8") as f:
        json.dump(results,f,indent=2)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("declare","run"));p.add_argument("--seed",type=int,choices=range(3))
    a=p.parse_args(); declare() if a.command=="declare" else run(a.seed)
