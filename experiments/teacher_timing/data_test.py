"""Actual native teacher intervention, label coverage and immutable anchoring."""
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np
import torch

from experiments.teacher_timing import data as d


class TeacherTimingDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = d.config(0)
        cfg.update(seed=1001, batch_size=2, actor_ids=(95000,95001),
                   duration_range=(1.,2.),gap_range=(.5,1.))
        cls.facts, cls.context = {}, {}
        for teacher in d.TEACHERS:
            cls.facts[teacher], cls.context[teacher] = d.collect_native(
                0,teacher,total=768,world_config=cfg)

    def test_exact_physical_configuration_and_budget(self):
        self.assertEqual((d.WARM,d.TRAIN_STEPS,d.TOTAL,d.BLOCK,d.BATCH,d.UPDATES),
                         (512,32768,33280,128,32,256))
        for seed in d.SEEDS:
            self.assertEqual(d.config(seed),d.original.config(seed))
            self.assertEqual(d.config(seed)["seed"],850000+seed)
            self.assertEqual(d.config(seed)["actor_ids"],tuple(range(95000,95032)))
            self.assertEqual(d.config(seed)["duration_range"],(8.,16.))
            self.assertEqual(d.config(seed)["gap_range"],(16.,32.))

    def test_earliest_delegates_original_artifacts_without_recalculation(self):
        obs, action = np.zeros((1,1,56),np.float32), np.zeros((1,1,2),np.int64)
        value, mask = np.zeros((1,1,2),np.float32), np.ones((1,1,2),bool)
        with patch.object(d.original,"load_stream",return_value=dict(obs=obs,actions=action)) as raw:
            with patch.object(d.original_targets,"load_targets",return_value=dict(time_target=value,time_mask=mask)) as aux:
                result=d.load_stream(0,"earliest")
        raw.assert_called_once_with(0,("obs","actions"))
        aux.assert_called_once_with(0,d.original_targets.TARGET_KEYS)
        for key, source in (("obs",obs),("actions",action),("time_target",value),("time_mask",mask)):
            self.assertIs(result[key],source)
        self.assertEqual(d.raw_path(0,"earliest"),d.original.stream_path(0))

    def test_both_native_policies_receipts_and_full_replay(self):
        for teacher in d.TEACHERS:
            f=self.facts[teacher]
            report=d.verify_facts(f,teacher)
            self.assertEqual(report["matured_strict_completion"],1.)
            with d.mapping_scope(d.mapping_table(0,"related")):
                world=d.ContextWorld.from_state_dict(self.context[teacher]["initial_world"],device="cpu")
                for tick in range(len(f["actions"])):
                    np.testing.assert_array_equal(world.observe().numpy(),f["obs"][tick])
                    np.testing.assert_array_equal(world.oracle(teacher).numpy(),f["actions"][tick])
                    world.act(torch.from_numpy(f["actions"][tick]))
                    np.testing.assert_array_equal(world.last_result.numpy(),f["receipts"][tick])
                self.assertEqual(world.metrics(),self.context[teacher]["metrics"])

    def test_same_exogenous_request_identity_but_real_observation_intervention(self):
        e,m=self.facts["earliest"],self.facts["midpoint"]
        pair=d.compare_pair(e,m)
        self.assertGreater(pair["changed_public_action_receipt_scalars"],0)
        self.assertEqual(pair["changed_observation_scalars"],pair["changed_public_action_receipt_scalars"])
        self.assertGreater(pair["action_different_slot_rows"],0)
        self.assertEqual(self.context["earliest"]["events_sha256"],self.context["midpoint"]["events_sha256"])
        self.assertGreater(float(d.finish_records(m)["latency_seconds"].mean()),
                           float(d.finish_records(e)["latency_seconds"].mean())+.5)

    def test_auxiliary_rows_larger_and_negative_coverage_no_eligibility_filter(self):
        ec=d.counts(self.facts["earliest"])
        mc=d.counts(self.facts["midpoint"])
        self.assertEqual(ec["action_rows"],mc["action_rows"])
        self.assertGreater(mc["time_rows"],ec["time_rows"])
        self.assertEqual(ec["negative_seconds_025_to_1"],0)
        self.assertEqual(ec["negative_seconds_lt_1"],0)
        self.assertGreater(mc["negative_seconds_025_to_1"],0)
        self.assertGreater(mc["negative_seconds_lt_1"],0)
        self.assertGreater(mc["time_ineligible_diagnostic"],0)
        for teacher in d.TEACHERS:
            f=self.facts[teacher]
            np.testing.assert_array_equal(f["time_mask"],f["active"] & f["started"] & np.isfinite(f["ready"]))
            self.assertTrue((f["time_target"][~f["time_mask"]]==0).all())

    def test_signed_physical_boundary_bins_and_block_counts(self):
        f={key:value.copy() for key,value in self.facts["midpoint"].items()}
        full=d.counts(f)
        blocks=[d.counts(f,i,i+128) for i in range(0,768,128)]
        for key,value in full.items():
            self.assertEqual(sum(b[key] for b in blocks),value,key)
        f={k:v[:1].copy() for k,v in f.items() if k!="actor_ids"}
        f["now"][:]=1.
        f["ready"][:]=np.asarray([[1.,.75],[0.,-.01]])
        f["active"][:]=f["started"][:]=True
        f["time_target"],f["time_mask"]=d.signed_targets(f["ready"],f["now"],f["active"],f["started"])
        bins=d.counts(f)
        self.assertEqual((bins["time_zero"],bins["negative_seconds_0_to_025"],
                          bins["negative_seconds_025_to_1"],bins["negative_seconds_lt_1"]),(1,1,1,1))

    def test_independent_accepted_begin_auditor_detects_corrupt_ready_and_labels(self):
        spec=importlib.util.spec_from_file_location("teacher_timing_raw_auditor",d.OUT/"audit_raw.py")
        auditor=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        for teacher in d.TEACHERS:
            facts=self.facts[teacher]
            report=auditor.accepted_begin_ledger(facts)
            self.assertFalse(report["eligibility_used"])
            self.assertEqual(report["time_rows"],int(facts["time_mask"].sum()))
            self.assertEqual(auditor.independent_counts(facts,0,768),d.counts(facts))
            row=tuple(np.argwhere(facts["time_mask"])[0])
            for key in ("ready","time_target"):
                broken={**facts,key:facts[key].copy()}
                broken[key][row]+=.125
                with self.assertRaises(AssertionError):
                    auditor.accepted_begin_ledger(broken)


if __name__=="__main__":
    unittest.main()
