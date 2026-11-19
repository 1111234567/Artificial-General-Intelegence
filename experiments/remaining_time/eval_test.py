"""Checks for read-only time metrics and retained whole-request denominators."""
import unittest
import numpy as np
from experiments.action_split.eval_test import facts_for
from experiments.action_split.evaluate import summarize_facts
from experiments.remaining_time.evaluate import time_statistics


def sample(policy="teacher"):
    f=facts_for(policy)
    valid=f["active"]&f["started"]&np.isfinite(f["ready"])
    t=np.zeros_like(f["ready"],dtype=np.float64)
    np.subtract(f["ready"],f["now"][:,None,None],out=t,where=valid)
    logits=np.zeros((*t.shape,11),dtype=np.float32)
    target=3+4*f["version"]+f["payload"]
    np.put_along_axis(logits,target[...,None],1,axis=-1)
    return f,t/16,logits,summarize_facts(f,0,warm=16)


class TimeMetrics(unittest.TestCase):
    def test_exact_target_units_and_boundary(self):
        f,p,l,r=sample()
        x=time_statistics(f,p,l,r,warm=16)
        self.assertEqual(x["groups"]["all_valid"]["mae_physical"],0)
        self.assertGreater(x["boundary"]["all"]["pairs"],20)
        self.assertEqual(x["boundary"]["all"]["both_sign_correct"],x["boundary"]["all"]["pairs"])
        self.assertEqual(x["boundary"]["all"]["first_content_correct"],x["boundary"]["all"]["pairs"])

    def test_normalized_error_converts_to_physical_and_ticks(self):
        f,p,l,r=sample()
        x=time_statistics(f,p+.125,l,r,warm=16)["groups"]["all_valid"]
        self.assertEqual(x["mae_physical"],2)
        self.assertEqual(x["rmse_physical"],2)
        self.assertEqual(x["mae_ticks"],8)

    def test_missing_started_rows_not_zero_error_success(self):
        f,p,l,r=sample("wait")
        x=time_statistics(f,p,l,r,warm=16)
        self.assertGreater(x["mature_request_denominator"],20)
        self.assertEqual(x["paired_requests"],0)
        self.assertEqual(x["groups"]["all_valid"]["rows"],0)
        self.assertIsNone(x["groups"]["all_valid"]["mae_physical"])

    def test_zero_predictions_have_no_successful_boundary_pair(self):
        f,p,l,r=sample()
        x=time_statistics(f,np.zeros_like(p),l,r,warm=16)
        b=x["boundary"]["all"]
        self.assertEqual(b["both_sign_correct"],0)
        self.assertEqual(b["false_early"],b["pairs"])
        self.assertEqual(b["direction_tie"],b["pairs"])

    def test_eligibility_does_not_filter_current_time_rows(self):
        f,p,l,r=sample()
        a=time_statistics(f,p,l,r,warm=16)
        f["eligible"]=~f["eligible"]
        b=time_statistics(f,p,l,r,warm=16)
        self.assertEqual(a,b)


if __name__=="__main__":unittest.main()
