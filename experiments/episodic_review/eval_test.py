"""Actual native-world checks for the two prospective retention stages."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from experiments.continuity.run import state_digest
from experiments.experience_transfer.train_transfer import mapping_scope,mapping_table,SEEDS
from experiments.lifelong.world import ContextWorld
from experiments.episodic_review import evaluate as e


def without_policy(state):
    result=deepcopy(state)
    for name in ("family_allowed","future_family","history"):result["context"].pop(name)
    return result


class TwoStageWorlds(unittest.TestCase):
    def test_live_sources_are_common_within_stage_and_unreset(self):
        for seed in range(3):
            with mapping_scope(mapping_table(seed,"related")):
                for stage,last in ((256,66047),(320,74239)):
                    paths=[e.world_path(seed,a+str(stage)) for a in e.ARMS]
                    self.assertEqual(len(set(paths)),1)
                    initial=torch.load(paths[0],map_location="cpu",weights_only=False)["final"]
                    fingerprint=state_digest(initial)
                    world=ContextWorld.from_state_dict(initial,device="cpu")
                    self.assertEqual(world.tick,last)
                    self.assertTrue(world._acted)
                    world.set_family_policy((SEEDS[seed]["history_cue"],SEEDS[seed]["B_cue"]))
                    self.assertEqual(state_digest(without_policy(initial)),state_digest(without_policy(world.state_dict())))
                    obs=world.observe().reshape(32,2,28)
                    self.assertEqual(world.tick,last+1)
                    self.assertEqual(world.now,(last+1)*.25)
                    ack=initial["tensors"]["_ack_action"].numpy()
                    receipt=initial["tensors"]["_ack_result"].numpy()
                    expected=np.eye(11,dtype=np.float32)[ack]*(ack!=0)[...,None]
                    np.testing.assert_array_equal(obs[...,11:22].numpy(),expected)
                    np.testing.assert_array_equal(obs[...,22].numpy(),receipt==1)
                    np.testing.assert_array_equal(obs[...,23].numpy(),receipt==-1)
                    self.assertEqual(state_digest(initial),fingerprint)

    def test_fresh_reference_is_a_distinct_native_origin(self):
        with mapping_scope(mapping_table(0,"related")):
            for condition in e.CONDITIONS:
                world=ContextWorld(**e.config(0,condition))
                self.assertEqual(world.tick,-1)
                self.assertEqual(world.actor_ids,tuple(range(101000,101032)))
                self.assertEqual(tuple(world.observe().shape),(32,56))
                self.assertEqual(world.now,0)


if __name__=="__main__":unittest.main()
