"""Evaluate the fixed repair models on the newly registered world splits."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_one_life_repair_eval as audit
from .interleave_interference import DATA, EVIDENCE


PLAN = EVIDENCE / "one-life-repair-confirmation-plan.json"
STEP = 2000


@torch.no_grad()
def run(cohort):
    torch.set_num_threads(1)
    repair, paid, swap = audit.repair, audit.paid, audit.swap
    plan = audit.read(PLAN)
    seed = plan["seeds"][cohort]
    n = plan["worlds"]
    births, _ = repair.setup(cohort)
    checked = audit.read(EVIDENCE / f"one-life-repair-{cohort}-{STEP}.json")
    old_tape = swap.capacity.make_tape("odor", seed["ordinary"], n["ordinary"], 1)
    joint_tape = paid.six.make(seed["joint"], n["joint"])
    old_interval = audit.stats.Crossed(len(births), n["ordinary"], seed["ordinary"] + 19, 4000)
    joint_interval = audit.stats.Crossed(len(births), n["joint"], seed["joint"] + 1, 4000)
    folder = DATA / "one-life-repair-confirmation" / cohort
    folder.mkdir(parents=True, exist_ok=True)
    files = (Path(__file__), Path(audit.__file__), Path(paid.__file__), Path(swap.__file__),
             Path(paid.six.__file__), Path(swap.capacity.__file__), Path(repair.net.__file__),
             Path(repair.motor.__file__), Path(repair.prior.__file__), Path(repair.net.native.__file__))
    code = {p.name: repair.digest(p) for p in files}
    plan_key = repair.digest(PLAN)
    signature = json.dumps((plan_key, code), sort_keys=True)
    old = repair.prior.tri.old_fit
    measures, hashes = {}, {}
    for arm in ("parent", *repair.ARMS):
        rows = []
        for birth in births:
            if arm == "parent":
                location, tag, step = repair.prior.DEST / cohort, "interspersed", 8000
            else:
                location, tag, step = repair.DEST / cohort, arm, STEP
            checkpoint = old.checkpoint_path(location, "odor", tag, birth, step)
            name = checkpoint.relative_to(DATA).as_posix()
            identity = repair.digest(checkpoint)
            if identity != checked["hashes"][name]:
                raise ValueError("Checkpoint changed after the fixed repair evaluation")
            hashes[name] = identity
            file = folder / f"{arm}-{birth}.npz"
            if not file.exists():
                graph, rule = old.load_checkpoint(location, "odor", tag, birth, step)
                isolated = swap.old_trial(graph, rule, old_tape)
                joint = swap.new_trial(graph, rule, joint_tape)
                consecutive = paid.consecutive(graph, rule, seed["joint"] + 71, n["joint"])
                raw = {**joint, **{f"ordinary/{k}": v for k, v in isolated.items()},
                       **{f"reward/{k}": v for k, v in consecutive.items()},
                       "checkpoint_sha256": identity, "evaluator_signature": signature}
                repair.prior.tri.revisit._write_raw(file, **raw)
            with np.load(file) as raw:
                if str(raw["checkpoint_sha256"]) != identity or str(raw["evaluator_signature"]) != signature:
                    raise ValueError("Confirmation cache belongs to a different graph or protocol")
                rewards = {k[7:]: raw[k] for k in raw.files if k.startswith("reward/")}
                rows.append(audit.measures(raw["ordinary/per_world"], raw, rewards))
            hashes[file.relative_to(DATA).as_posix()] = repair.digest(file)
            print(cohort, arm, birth, flush=True)
        measures[arm] = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    for a, b in (("mixed", "long_only"), ("mixed", "parent"), ("long_only", "parent")):
        measures[f"{a} minus {b}"] = {k: v - measures[b][k] for k, v in measures[a].items()}
    result = {}
    for arm, bank in measures.items():
        result[arm] = {}
        for key, values in bank.items():
            if key == "ordinary":
                labels, interval = swap.SPEC["old"], old_interval
            else:
                labels = paid.PAID if key.startswith("paid/") else paid.six.METRICS
                interval = joint_interval
            result[arm][key] = {m: interval(values[:, :, i]) for i, m in enumerate(labels)}
    output = dict(cohort=cohort, step=STEP, births=births, plan_sha256=plan_key, source_sha256=code,
                  numpy=np.__version__, torch=str(torch.__version__), hashes=hashes,
                  ordinary_worlds=n["ordinary"], ordinary_seed=seed["ordinary"],
                  joint_worlds=n["joint"], joint_seed=seed["joint"], results=result)
    repair.prior.tri.revisit._write_json(EVIDENCE / f"one-life-repair-confirmation-{cohort}-2000.json", output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=audit.repair.COHORTS, required=True)
    args = parser.parse_args()
    run(args.cohort)
