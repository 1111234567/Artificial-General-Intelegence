"""Behavioral denominator and whole-window tests on actual native worlds."""
import unittest
import numpy as np
import torch

from experiments.action_split.evaluate import FACT_FIELDS, summarize_facts
from experiments.experience_transfer.train_transfer import mapping_scope,mapping_table,SEEDS
from experiments.lifelong.world import ContextWorld


def facts_for(policy):
    data={k:[] for k in (*FACT_FIELDS,"actions","receipts","now")}
    with mapping_scope(mapping_table(0,"related")):
        world=ContextWorld(seed=5516,batch_size=4,n_jobs=2,active_jobs=2,
            actor_ids=(19,20,21,22),family_allowed=(0,1),dt=.25,
            duration_range=(1.,2.),gap_range=(.5,1.),cancel=.2,replace=.3)
        for _ in range(240):
            world.observe()
            a=world.oracle("earliest")
            if policy == "wait":a.zero_()
            elif policy == "never_finish":a=torch.where(a>=3,0,a)
            elif policy == "early":
                a=torch.where(world.active & world.started,3+4*world.version+world.payload,a)
            elif policy == "late_begin":
                a=torch.where(world.active & ~world.started & (world.now+world.duration>world.deadline+1e-6),1+world.version,0)
            elif policy != "teacher":raise ValueError(policy)
            for k,attr in FACT_FIELDS.items():data[k].append(getattr(world,attr).numpy().copy())
            data["now"].append(world.now)
            data["actions"].append(a.numpy().copy())
            world.act(a)
            data["receipts"].append(world.last_result.numpy().copy())
    out={k:np.stack(v) for k,v in data.items()}
    out["actor_ids"]=np.asarray((19,20,21,22))
    return out


class ActualWorldTests(unittest.TestCase):
    def test_wait_failures_keep_full_denominator(self):
        r=summarize_facts(facts_for("wait"),0,warm=16)
        a=r["groups"]["all"]
        self.assertGreater(a["matured"],30)
        self.assertEqual(a["no_begin"],a["matured"])
        self.assertEqual(a["ordinary"],0)
        self.assertEqual(a["legal_no_finish"],0)
        self.assertEqual(r["idle_actions"],0)

    def test_success_and_early_retries_are_different(self):
        clean=summarize_facts(facts_for("teacher"),0,warm=16)["groups"]["all"]
        dirty=summarize_facts(facts_for("early"),0,warm=16)["groups"]["all"]
        self.assertEqual(clean["strict"],clean["matured"])
        self.assertEqual(dirty["ordinary"],dirty["matured"])
        self.assertEqual(dirty["strict"],0)
        self.assertEqual(dirty["first_error_early"],dirty["matured"])

    def test_legal_window_misses_separate_from_missing_window(self):
        wait=summarize_facts(facts_for("never_finish"),0,warm=16)["groups"]["all"]
        late=summarize_facts(facts_for("late_begin"),0,warm=16)["groups"]["all"]
        self.assertEqual(wait["legal_no_finish"],wait["matured"])
        self.assertEqual(wait["started_no_window"],0)
        self.assertEqual(late["started_no_window"],late["matured"])
        self.assertEqual(late["legal_no_finish"],0)

    def test_corrupted_receipt_cannot_pass(self):
        f=facts_for("teacher")
        i=tuple(np.argwhere(f["receipts"]==1)[0])
        f["receipts"][i]=-1
        with self.assertRaises(AssertionError):summarize_facts(f,0,warm=16)


if __name__=="__main__":unittest.main()
