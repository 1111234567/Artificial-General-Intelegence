"""Fixed-phase extraction never silently drops a report request."""
from copy import deepcopy
import unittest

import numpy as np

from experiments.teacher_timing import phase_readout as phase


def tiny():
    shape = (10, 1, 1)
    facts = {key: np.zeros(shape, dtype=np.int64) for key in ("rid", "version", "payload", "actions",
        "receipts", "event_kind", "event_id", "mature_id", "event_version", "event_payload")}
    facts.update({key: np.zeros(shape, dtype=bool) for key in ("active", "started", "event_eligible", "mature")})
    facts.update(now=np.arange(10)*.25, ready=np.full(shape, .7), deadline=np.full(shape, 2.), actor_ids=np.asarray([99000]))
    facts["rid"][:, 0, 0] = 9
    facts["active"][1:7] = True
    facts["started"][2:7] = True
    facts["version"][:, 0, 0], facts["payload"][:, 0, 0] = 1, 2
    facts["actions"][6, 0, 0], facts["receipts"][6, 0, 0] = 9, 1
    facts["event_kind"][1, 0, 0], facts["event_id"][1, 0, 0] = 1, 9
    facts["event_eligible"][1, 0, 0] = True
    facts["event_version"][1, 0, 0], facts["event_payload"][1, 0, 0] = 1, 2
    facts["mature"][8, 0, 0], facts["mature_id"][8, 0, 0] = True, 9
    report = dict(requests=[dict(actor=99000, lane=0, job=0, rid=9, issued=1, mature_tick=8)])
    return facts, report


class FixedPhaseTest(unittest.TestCase):
    def test_selects_firstlegal_and_actual_accepted_finish_for_every_request(self):
        facts, report = tiny()
        selected = phase.select_positions(facts, report, warm=1, expected=1)
        np.testing.assert_array_equal(selected["ticks"], [[3, 6]])
        np.testing.assert_array_equal(selected["target_content"], [6])
        # Tested-model correctness/completion fields never select the request.
        report["requests"][0].update(ordinary=False, strict=False)
        np.testing.assert_array_equal(phase.select_positions(facts, report, warm=1)["ticks"], [[3, 6]])

    def test_missing_phase_or_changed_request_fails_instead_of_filtering(self):
        facts, report = tiny()
        broken = []
        x = deepcopy(facts); x["receipts"][6] = 0; broken.append(x)
        x = deepcopy(facts); x["payload"][4] = 3; broken.append(x)
        x = deepcopy(facts); x["active"][4] = False; broken.append(x)
        x = deepcopy(facts); x["mature_id"][8] = 10; broken.append(x)
        x = deepcopy(facts); x["event_eligible"][1] = False; broken.append(x)
        for item in broken:
            with self.assertRaises(AssertionError):
                phase.select_positions(item, report, warm=1)
        with self.assertRaises(AssertionError):
            phase.select_positions(facts, report, warm=1, expected=2)

    def test_native_and_conditional_readouts_and_all_four_pair_outcomes(self):
        targets = np.asarray([6, 6, 6, 6])
        scores = np.full((4, 2, 11), -10., dtype=np.float32)
        # Conditional correctness: wrong->correct, correct->wrong, both, neither.
        content = np.asarray([[0, 6], [6, 0], [6, 6], [0, 0]])
        for request in range(4):
            for position in range(2):
                scores[request, position, content[request, position]+3] = 2.
        scores[:, 0, 0] = 3.  # Native policy WAITs, regardless of content ranking.
        summary, details = phase.summarize_logits(scores, targets)
        self.assertEqual(summary["paired_conditional8_content"], dict(firstwrong_latercorrect=1,
            firstcorrect_laterwrong=1, bothcorrect=1, neither=1))
        self.assertEqual(summary["phases"]["first_legal"]["native11_WAIT"], 4)
        self.assertEqual(summary["phases"]["first_legal"]["conditional8_content_correct"], 2)
        self.assertEqual(summary["phases"]["teacher_FINISH"]["native11_correct_FINISH"], 2)
        self.assertEqual(summary["phases"]["teacher_FINISH"]["native11_wrong_FINISH"], 2)
        self.assertEqual(summary["paired_native11_correct_FINISH"]["firstwrong_latercorrect"], 2)
        np.testing.assert_array_equal(details["conditional8_argmax"], content)


if __name__ == "__main__":
    unittest.main()
