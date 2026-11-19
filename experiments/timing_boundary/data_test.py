"""Boundary selection must preserve cohorts, causal phase, and split isolation."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.timing_boundary import data as d


def fixture():
    ticks=16; shape=(ticks,1,2)
    f={k:np.zeros(shape,dtype=bool if k in ("active","started","eligible","event_eligible","mature","mature_done","mature_strict") else np.int64)
       for k in ("rid","active","started","eligible","event_eligible","mature","mature_id","mature_done","mature_strict", "event_kind","event_id","family","raw","version","payload")}
    f.update(now=np.arange(ticks)*d.DT,actor_ids=np.array([91000]),
             issued_time=np.zeros(shape),duration=np.ones(shape),ready=np.full(shape,1.4),deadline=np.full(shape,2.5))
    f["rid"][:]=-1; f["mature_id"][:]=-1
    f["event_kind"][1,0,0]=1; f["event_id"][1,0,0]=2
    f["rid"][1:,0,0]=2; f["event_eligible"][1,0,0]=True
    f["active"][1:9,0,0]=True; f["started"][2:9,0,0]=True
    f["mature"][11,0,0]=True; f["mature_id"][11,0,0]=2
    f["mature_done"][11,0,0]=True; f["mature_strict"][11,0,0]=True
    return f


class TimingDataTest(unittest.TestCase):
    def test_disjoint_splits_and_deterministic_request_keys(self):
        groups=[set(d.actors(s)) for s in d.SPLITS]
        self.assertTrue(all(not groups[i]&groups[j] for i in range(3) for j in range(i)))
        keys={d.request_key(seed,split,actor,job,rid) for seed in range(3) for split in d.SPLITS
              for actor in d.actors(split) for job in range(2) for rid in (0,2,9999)}
        self.assertEqual(len(keys),3*3*32*2*3)

    def test_boundary_phase_and_partial_offsets(self):
        rows,r=d.request_table(fixture(),0,"train",warm=0)
        pair=r["requests"][0]
        self.assertTrue(pair["included"])
        endpoints={x["offset"]:x for x in rows if x["offset"] in (-1,0)}
        self.assertEqual(endpoints[-1]["tick"]+1,endpoints[0]["tick"])
        self.assertFalse(endpoints[-1]["legal"]);self.assertTrue(endpoints[0]["legal"])
        self.assertGreater(endpoints[-1]["remaining_ticks"],0)
        self.assertLessEqual(endpoints[0]["remaining_ticks"],0)
        self.assertIn(-8,pair["missing_offsets"])
        self.assertEqual(r["counts"]["included_pairs"],1)

    def test_maturity_warm_eligibility_filters_do_not_change_denominator(self):
        f=fixture()
        _,warm=d.request_table(f,0,"train",warm=2)
        self.assertEqual(warm["exclusions"],{"issued_beforewarm":1})
        f["mature"][:]=False
        _,pending=d.request_table(f,0,"train",warm=0)
        self.assertEqual(pending["exclusions"],{"not_matured_at_horizon":1})
        f["event_eligible"][1,0,0]=False
        _,ineligible=d.request_table(f,0,"train",warm=0)
        self.assertEqual(ineligible["exclusions"],{"ineligible":1})
        self.assertEqual(ineligible["counts"]["commands"],1)

    def test_no_boundary_is_reported_not_silently_dropped(self):
        f=fixture();f["active"][6:,0,0]=False
        rows,r=d.request_table(f,0,"train",warm=0)
        self.assertFalse(rows)
        self.assertEqual(r["counts"]["matured_eligible_afterwarm"],1)
        self.assertEqual(r["exclusions"],{"no_active_started_legal":1})

    def test_native_small_world_reconstruction(self):
        # A real successful teacher stream verifies receipt, eligibility and
        # maturation arithmetic rather than merely reproducing row selection.
        saved={k:[] for k in (*d.FACT_FIELDS,"obs","actions","receipts","now","tick")}
        with d.mapping_scope(d.mapping_table(0,"related")):
            w=d.ContextWorld(seed=1001,batch_size=1,n_jobs=2,active_jobs=2,actor_ids=(91000,),
                family_allowed=(0,1),dt=.25,duration_range=(1.,2.),gap_range=(.5,1.))
            for _ in range(160):
                obs=w.observe();a=w.oracle("midpoint")
                for key,attr in d.FACT_FIELDS.items():saved[key].append(getattr(w,attr).numpy().copy())
                for key,value in (("obs",obs.numpy().copy()),("actions",a.numpy().copy()),("now",w.now),("tick",w.tick)):
                    saved[key].append(value)
                w.act(a);saved["receipts"].append(w.last_result.numpy().copy())
        f={k:np.stack(v) for k,v in saved.items()};f["actor_ids"]=np.array([91000])
        _,r=d.request_table(f,0,"train",warm=0)
        result=d.verify_native_facts(f,r["requests"])
        self.assertEqual(result["receipts"],"exact")
        self.assertGreater(r["counts"]["matured_eligible_afterwarm"],0)

    def test_random_state_and_shape_are_reproducible(self):
        a=d.load_model(0,"random");b=d.load_model(0,"random");c=d.load_model(1,"random")
        self.assertEqual(d.state_sha(a),d.state_sha(b))
        self.assertNotEqual(d.state_sha(a),d.state_sha(c))
        self.assertEqual(a.width,192)


if __name__ == "__main__":unittest.main()
