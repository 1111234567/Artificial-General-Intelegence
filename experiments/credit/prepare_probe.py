"""Freeze a finite independent-readout study before any source is evaluated."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

from experiments.continuity.run import atomic_json
from experiments.credit.probe import ProbeConfig

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    evidence = ROOT/"docs/evidence/credit-2026-10-27"
    output = evidence/"probe-protocol-v1.json"
    if output.exists():
        raise FileExistsError(output)
    files = [ROOT/"experiments/credit/probe.py", ROOT/"experiments/credit/test_probe.py",
             ROOT/"experiments/credit/batch.py", Path(__file__)]
    files += sorted((ROOT/"experiments/continuity").glob("*.py"))
    files += sorted((ROOT/"experiments/integration").glob("*.py"))
    hashes = {str(path): sha(path) for path in files}
    jobs = []
    arms = ("frozen", "bp64", "replay64", "bp128", "coact_mix")
    for seed in range(3):
        for arm in arms:
            source = ROOT/f"data/integration/main-v1/seed_{seed:03d}/{arm}/checkpoint.pt"
            hashes[str(source)] = sha(source)
            folder = ROOT/f"data/credit/probe-v1/seed_{seed:03d}_{arm}"
            jobs.append(dict(id=f"seed_{seed:03d}_{arm}",
                command=[sys.executable, "-m", "experiments.credit.probe", "--checkpoint", str(source),
                         "--output", str(folder/"probe.json"), "--dataset-dir", str(folder/"datasets"),
                         "--steps", "1024", "--device", "cpu", "--threads", "1"],
                log=str(folder/"process.log"), outputs=[str(folder/"probe.json")]))
    protocol = dict(format="credit-probe-protocol-v1", config=asdict(ProbeConfig()), seeds=[0,1,2],
        modes=list(arms), own_action_forks=75, conditions=["iid","long_wait","longer","sparse","long_sparse"],
        features=["raw_h","exact_model_tanh_h","actual_obs96"],
        targets=["payload","version","joint","active","raw_started","active_started"],
        controls=["none","request_consistent_random_joint_labels","within_condition_job_split_feature_shuffle"],
        meaning="Observational independent readouts of existing frozen complete B continuations; no controller update.",
        transfer="Train/validate on four 1--8 conditions, test longer 8--16 on heldout actors; within-longer separate.",
        sampling="Private request/tick hash caps; actor40/12/12 split; unknown inherited/carryover content masked.",
        fitting="Train-only mean/std; per-job one-hot ridge; alpha selected on validation only.",
        file_hashes=hashes, cwd=str(ROOT), jobs=jobs)
    atomic_json(output, protocol)
    print(output)
