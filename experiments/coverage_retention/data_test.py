"""Native source continuity, retained pending work, and independent labels."""
import importlib.util
from copy import deepcopy
import unittest

import numpy as np
import torch

from experiments.coverage_retention import data as d


class CoverageContinuationDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=d.source.config(0)
        cfg.update(batch_size=2,actor_ids=(95000,95001))
        preview,_=d.source.collect_native(0,"midpoint",total=512,world_config=cfg)
        replacement=np.nonzero(np.any((preview["event_kind"]==2)&(preview["event_family"]==0),axis=(1,2)))[0]
        cls.cursor=int(replacement[0])-2
        cls.prefix,context=d.source.collect_native(0,"midpoint",total=cls.cursor,world_config=cfg)
        cls.source_state=context["final_world"]
        cls.facts={}; cls.context={}
        for arm in d.ARMS:
            cls.facts[arm],cls.context[arm]=d.collect_native(0,arm,total=768,initial_state=cls.source_state)
        spec=importlib.util.spec_from_file_location("coverage_raw_audit",d.OUT/"audit_raw.py")
        cls.auditor=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.auditor)

    def test_exact_protocol_budget(self):
        self.assertEqual((d.WARM,d.TOTAL,d.BLOCK,d.BATCH,d.UPDATES),(0,32768,128,32,256))
        for seed in d.SEEDS:
            self.assertEqual(d.family_policy(seed,"withdraw"),(d.SEED_SPECS[seed]["B_cue"],))
            self.assertEqual(d.regime_change(seed,"wide")["duration_range"],(8.,32.))
            self.assertEqual(d.regime_change(seed,"control")["duration_range"],(8.,16.))

    def test_every_native_boundary_leaf_preserved(self):
        for arm in d.ARMS:
            context=self.context[arm]
            self.assertEqual(d.state_digest(context["before_change"]),d.state_digest(self.source_state))
            d.boundary_audit(context["before_change"],context["after_change"],0,arm)
            corrupted=deepcopy(context["after_change"])
            corrupted["scheduler"]["_rng_state"][0,0]+=1
            with self.assertRaises(AssertionError): d.boundary_audit(self.source_state,corrupted,0,arm)
            for key in ("scheduler","tensors","counts"):
                self.assertEqual(d.state_digest(context["before_change"][key]),d.state_digest(context["after_change"][key]))

    def test_global_clock_native_teacher_and_receipt_boundary(self):
        for arm in d.ARMS:
            f=self.facts[arm]
            d.verify_facts(f,0,arm,self.source_state)
            self.assertEqual(f["world_tick"][0],self.cursor)
            self.assertEqual(f["tick"][0],0)
            self.assertEqual(f["now"][0],self.cursor*.25)

    def test_control_withdraw_physical_raw_events_and_rng_match(self):
        a,b=self.facts["control"],self.facts["withdraw"]
        for key in d.PAIR_FIELDS:
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        finala,finalb=self.context["control"]["final"],self.context["withdraw"]["final"]
        self.assertEqual(d.state_digest(finala["scheduler"]),d.state_digest(finalb["scheduler"]))
        self.assertTrue(torch.equal(finala["context"]["rng_state"],finalb["context"]["rng_state"]))
        self.assertTrue((a["family"]!=b["family"]).any())
        self.assertTrue((a["duration"]!=self.facts["wide"]["duration"]).any())

    def test_withdrawal_no_new_A_and_preserves_inherited_A_labels(self):
        f=self.facts["withdraw"]
        self.assertFalse(((f["event_kind"]==1)&(f["event_family"]==d.SEED_SPECS[0]["history_cue"])).any())
        inherited=(f["family"]==d.SEED_SPECS[0]["history_cue"])&(f["cycle_id"]<2*self.cursor)
        self.assertTrue((inherited&f["active"]).any())
        self.assertTrue((inherited&f["time_mask"]).any())
        replacement=(f["event_kind"]==2)&(f["event_family"]==0)&(f["event_cycle_id"]<2*self.cursor)
        self.assertTrue(replacement.any())
        self.assertTrue((f["event_id"][replacement]>=2*self.cursor).all())

    def test_all_labels_and_counts_independently_reconstructed(self):
        for arm in d.ARMS:
            f=self.facts[arm]
            report=self.auditor.accepted_begin_ledger(f,0,self.prefix)
            self.assertEqual(report["time_rows"],int(f["time_mask"].sum()))
            self.assertFalse(report["eligibility_used"])
            total=d.counts(f,0,source_cursor=self.cursor)
            self.assertEqual(total,self.auditor.independent_counts(f,0,0,768,cutoff=self.cursor))
            blocks=[d.counts(f,0,i,i+128,source_cursor=self.cursor) for i in range(0,768,128)]
            for key,value in total.items(): self.assertEqual(sum(b[key] for b in blocks),value,key)
            row=tuple(np.argwhere(f["time_mask"])[0])
            for key in ("ready","time_target"):
                corrupt={**f,key:f[key].copy()}; corrupt[key][row]+=.125
                with self.assertRaises(AssertionError): self.auditor.accepted_begin_ledger(corrupt,0,self.prefix)


if __name__=="__main__": unittest.main()
