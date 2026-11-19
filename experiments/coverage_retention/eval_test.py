"""Native retained-world checks before any new endpoint evaluation."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import torch

from experiments.action_split.evaluate import FACT_FIELDS,summarize_facts
from experiments.continuity.run import state_digest
from experiments.experience_transfer.train_transfer import mapping_scope,mapping_table,SEEDS
from experiments.lifelong.world import ContextWorld
from experiments.coverage_retention import evaluate,live_eval

ROOT=Path(__file__).resolve().parents[2]


def source_world():
    path=ROOT/"data/teacher-timing/streams-v1/seed_000/midpoint/world_states.pt"
    return torch.load(path,map_location="cpu",weights_only=False)["final"]


class ContinuationEvaluation(unittest.TestCase):
    def test_fresh_world_accepts_distinct_new_actor_ids(self):
        with mapping_scope(mapping_table(0,"related")):
            for condition in evaluate.CONDITIONS:
                world=ContextWorld(**evaluate.config(0,condition))
                self.assertEqual(world.actor_ids,tuple(range(100000,100032)))
                self.assertEqual(tuple(world.observe().shape),(32,56))

    def test_restore_retains_source_h_and_world_numerics(self):
        cp_path=ROOT/"data/teacher-timing/train-v1/midpoint/seed_000/action/step256.pt"
        world_path=ROOT/"data/teacher-timing/streams-v1/seed_000/midpoint/world_states.pt"
        cp=torch.load(cp_path,map_location="cpu",weights_only=False)
        with mapping_scope(mapping_table(0,"related")),patch.object(live_eval,"checkpoint",return_value=cp_path),patch.object(live_eval,"world_path",return_value=world_path):
            model,h,world,before=live_eval.restore(0,"control")
            self.assertTrue(torch.equal(h,cp["hidden"]))
            self.assertEqual(state_digest(model.state_dict()),state_digest(cp["model"]))
            after=world.state_dict()
            before=deepcopy(before);after=deepcopy(after)
            for state in (before,after):
                for key in ("family_allowed","future_family","history"):state["context"].pop(key)
            self.assertEqual(state_digest(before),state_digest(after))
            self.assertEqual(world.tick,33279)
            self.assertTrue(world._acted)

    def test_inherited_maturity_excluded_from_new_request_denominator(self):
        rows={k:[] for k in (*FACT_FIELDS,"actions","receipts","now")}
        with mapping_scope(mapping_table(0,"related")):
            world=ContextWorld.from_state_dict(source_world(),device="cpu")
            for _ in range(640):
                world.observe();actions=world.oracle("midpoint")
                for key,attr in FACT_FIELDS.items():rows[key].append(getattr(world,attr).numpy().copy())
                rows["actions"].append(actions.numpy().copy());rows["now"].append(world.now)
                world.act(actions);rows["receipts"].append(world.last_result.numpy().copy())
        facts={k:np.stack(v) for k,v in rows.items()};facts["actor_ids"]=np.asarray(world.actor_ids)
        new_ids={(int(l),int(j),int(facts["event_id"][t,l,j]))
                 for t,l,j in zip(*np.where(np.isin(facts["event_kind"],(1,2))))}
        maturities=[(int(t),int(l),int(j),int(facts["mature_id"][t,l,j])) for t,l,j in zip(*np.where(facts["mature"]))]
        self.assertGreater(sum((l,j,rid) not in new_ids for t,l,j,rid in maturities),0)
        result=summarize_facts(facts,0,warm=128)
        self.assertGreater(result["groups"]["all"]["matured"],50)
        self.assertEqual(result["groups"]["all"]["strict"],result["groups"]["all"]["matured"])
        self.assertTrue(all(r["issued"]>=128 for r in result["requests"]))
        self.assertTrue(all(r["rid"]>=2*(33280+128) for r in result["requests"]))


if __name__=="__main__":unittest.main()
