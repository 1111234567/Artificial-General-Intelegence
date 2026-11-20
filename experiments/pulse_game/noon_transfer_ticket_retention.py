"""Separate ticket deployment scores from fixed-action retention interventions."""

from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive
from . import noon_transfer_custody as custody, noon_transfer_forgetting as retention


MODELS = ("uniform", "ticket_native", "ticket_coin")
PAIRS = (("ticket_native", "uniform"), ("ticket_coin", "uniform"), ("ticket_native", "ticket_coin"))


def ordinary_pair(graph, rule, tape, support):
    rule = rule.expand(tape.lanes)
    start = card.native.start(graph, tape.lanes)
    a, _, _ = retention.support_contact(rule, start, tape, 0, support[0][0], "active")
    ab, _, _ = retention.support_contact(rule, a, tape, 1, support[0][1], "active")
    snapshot = card.native.wait(rule, ab, 4)
    pair = np.stack([card.capacity._probe(rule, snapshot, tape, item).side.numpy() for item in (0, 1)])
    b_only, _, _ = retention.support_contact(rule, start, tape, 1, support[0][1], "active")
    b = card.capacity._probe(rule, card.native.wait(rule, b_only, 4), tape, 1).side.numpy()
    return {"pair_side": pair, "pair_truth": tape.right_value[:, :2, -1].numpy().T, "B_alone": b}


def measures(arrays, count):
    scores = retention.cells(arrays, count)
    truth = arrays["pair_truth"]
    a, b = arrays["pair_side"] == truth
    scores["pair/A"] = a.reshape(-1, 4).mean(1)
    scores["pair/B"] = b.reshape(-1, 4).mean(1)
    scores["pair/both"] = (a & b).reshape(-1, 4).mean(1)
    scores["pair/B_alone"] = (arrays["B_alone"] == truth[1]).reshape(-1, 4).mean(1)
    scores["decrement/B_given_A"] = scores["pair/B"] - scores["pair/B_alone"]
    return scores


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    plan = archive.read(retention.PLAN)
    archives = archive.read(archive.REPORT)["cohorts"]
    counts, n = plan["counts"], plan["worlds"]
    result = {}
    for cohort, seed in plan["cohorts"].items():
        natural_report = card.OUT / f"transfer-ticket-{cohort}-8000.json"
        standard = archive.read(natural_report)
        births = standard["births"]
        tape = card.capacity.make_tape("odor", seed, n, max(counts) - 1)
        interval = card.stats.Crossed(len(births), n, seed + 41, 4000)
        arrays = {model: [] for model in MODELS}
        files = {}
        natural = {model: {k: v for k, v in standard["models"][model].items()
                           if k in ("objects/1/no_review/A", "objects/B_alone/B", "objects/2/no_review/A",
                                    "objects/2/no_review/B", "objects/2/no_review/all")}
                   for model in MODELS}
        for birth in births:
            b = str(birth)
            folder, tag, step = archive.model(cohort, "uniform")
            baseline = card.checkpoint.load_checkpoint(folder, "odor", tag, birth, step)
            same_actions = retention.select_support(*baseline, tape)
            specimens = {"uniform": (baseline, archives[cohort]["experts"][f"uniform/{birth}"])}
            for arm in ("native", "coin"):
                model = f"ticket_{arm}"
                entry = standard["model_custody"][model]["models"][b]
                graph_rule = card.checkpoint.load_checkpoint(card.DATA / "tickets" / cohort, "odor", arm, birth, 8000)
                specimens[model] = graph_rule, entry
            for model, ((graph, rule), entry) in specimens.items():
                archive.verify(card.ROOT / entry["path"], entry["sha256"])
                record, truth = retention.replay(graph, rule, tape, same_actions, counts)
                pair = ordinary_pair(graph, rule, tape, same_actions)
                np.testing.assert_array_equal(record["write"][1], pair["pair_side"][0])
                np.testing.assert_array_equal(pair["pair_truth"][0], truth)
                np.testing.assert_array_equal(record["write"][0], record["sham"][0])
                np.testing.assert_array_equal(record["write"][0], record["idle"][0])
                raw = dict(record, truth=truth, **pair, forced_side=same_actions[0].numpy(), earned=same_actions[1].numpy())
                out = card.DATA / "transfer" / cohort / f"ticket-retention-{model}-{birth}.npz"
                io._write_raw(out, **raw)
                files[f"{model}/{birth}"] = {"trace": custody.record(out), "model": entry}
                with np.load(out) as saved:
                    np.testing.assert_array_equal(saved["earned"], saved["forced_side"] * tape.right_value[:, :, 0].numpy().T)
                    arrays[model].append(measures(saved, counts))
            print(cohort, birth, flush=True)
        cells = {name: {k: np.stack([row[k] for row in rows]) for k in rows[0]} for name, rows in arrays.items()}
        result[cohort] = {
            "births": births,
            "natural_reference": natural,
            "natural_reference_report": custody.record(natural_report),
            "fixed_receipts": {name: {k: interval(v) for k, v in values.items()} for name, values in cells.items()},
            "paired_fixed_receipts": {f"{a} minus {b}": {k: interval(v - cells[b][k]) for k, v in cells[a].items()}
                                      for a, b in PAIRS},
            "hashes": files,
        }
    artifact = {"method": "Deploy naturally in the parent; replay the same physical selection, tastes, menus, and edit "
                "draws on all models. The initial state for the subsequent-write, sham-write and idle branches comes "
                "from the A contact. The frozen ticket comparisons without a forced collection are in natural_reference.",
                "plan": custody.record(retention.PLAN), "source": custody.record(Path(__file__)), "cohorts": result}
    io._write_json(card.OUT / "transfer-ticket-retention-8000.json", artifact)


if __name__ == "__main__":
    run()
