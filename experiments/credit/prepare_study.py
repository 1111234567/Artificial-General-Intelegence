"""Freeze training, evaluation and familiar-regime revisitation before main runs."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

from experiments.continuity.run import atomic_json
from experiments.credit.run import CreditConfig, MODES

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"docs/evidence/credit-2026-10-27"
DATA = ROOT/"data/credit"
SELECTED = ("readout_only", "bp64", "bp_replay", "diag_sym", "diag_sym_replay")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(module, *args):
    return [sys.executable, "-m", f"experiments.credit.{module}", *map(str, args)]


if __name__ == "__main__":
    protocol_path = EVIDENCE/"study-protocol-v1.json"
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    files = [ROOT/f"experiments/credit/{name}.py" for name in
             ("run", "local", "evaluate", "cycle", "batch", "launch", "prepare_study", "test_run", "test_local", "test_evaluate")]
    files += sorted((ROOT/"experiments/continuity").glob("*.py"))
    files += sorted((ROOT/"experiments/integration").glob("*.py"))
    hashes = {str(path): sha(path) for path in files}
    pilot = ROOT/"data/credit/pilot-benchmark-v1/summary.json"
    hashes[str(pilot)] = sha(pilot)
    origins = []
    for seed in range(3):
        path = ROOT/f"data/continuity/main-20261027-v1/seed_{seed:03d}/online128/checkpoint.pt"
        hashes[str(path)] = sha(path)
        origins.append(dict(seed=seed, checkpoint=str(path), sha256=hashes[str(path)]))
    protocol = dict(format="credit-study-protocol-v1", frozen_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2], config=asdict(CreditConfig()), device="cuda",
        modes={name: asdict(mode) for name, mode in MODES.items()}, origins=origins,
        train_ticks=65536, dt=.25, batch_size=64, n_jobs=4, width=128,
        updated_parameters=dict(readout_only=5676, frozen=0, other=34348),
        learning_rate=.001, gradient_clip=1., optimizer="inherited Adam; one step per64live ticks",
        pilot="Independent A seed101 for numerical, continuation and resource checks; no main test-driven tuning.",
        replay=dict(source="immutable inherited A reservoir32x128ticks", warmup_ticks=64, loss_ticks=64,
                    current_weight=.5, old_weight=.5, preupdate_weights=True,
                    old_bp_graph_ticks=128, old_eligibility_initial="zero then full128tick evolution",
                    live_state_overwrite=False),
        evaluation=dict(conditions=["iid","long_wait","longer","sparse","long_sparse"], steps=2048,
                        primary_forks=165, extra_interventions=["finish_content_oracle","timing_oracle"],
                        intervention_modes=list(SELECTED), intervention_conditions=["longer","long_sparse"],
                        extra_forks=60, total_forks=225),
        cycles=dict(modes=list(SELECTED), seeds=[0,1,2], start_stage_ticks=65536,
                    sequence=["A","B"]*4, segment_ticks=4096, milestones=[1024,4096],
                    regimes={"A":[[1.,4.],[.5,2.]],"B":[[4.,8.],[4.,12.]]},
                    eval_conditions=["iid","long_sparse"], eval_steps=512, total_forks=480,
                    meaning="Familiar-regime reacquisition and retention, not new-task plasticity proof."),
        constraints=["No live h/world/receipt/clock/Adam reset.",
                     "Trace reset controls reset only added E/F; every learned arm still has analytic readout gradient.",
                     "Diagonal sensitivity approximation drops cross-neuron derivative paths and is stale after updates.",
                     "Same optimizer-step budget, not equal computational work; old source passes reported separately.",
                     "Readout reservoir is pretrained, not random spiking LSM; supervised labels, no self-evaluator."],
        file_hashes=hashes)
    atomic_json(protocol_path, protocol)
    frozen = {**hashes, str(protocol_path): sha(protocol_path)}
    train_jobs, eval_jobs, cycle_jobs = [], [], []
    for origin in origins:
        seed=origin["seed"]
        for mode in MODES:
            name=f"seed_{seed:03d}_{mode}"
            folder=DATA/f"main-v1/seed_{seed:03d}/{mode}"
            cp=folder/"checkpoint.pt"
            train_jobs.append(dict(id=name, command=command("launch", "--from-checkpoint", origin["checkpoint"],
                "--mode", mode, "--output", folder, "--ticks", 65536, "--device", "cuda", "--threads",1,
                "--checkpoint-every",256), log=str(DATA/f"main-v1/logs/{name}.log"),
                outputs=[str(cp),str(folder/"report.json"),str(folder/"launch-complete.json")]))
            output=DATA/f"evaluation-v1/{name}.json"
            arguments=["--checkpoint",cp,"--output",output,"--device","cpu","--threads",1,"--steps",2048]
            if mode in SELECTED:
                arguments += ["--interventions", "finish_content_oracle", "timing_oracle"]
            eval_jobs.append(dict(id=name, command=command("evaluate",*arguments),
                log=str(DATA/f"evaluation-v1/logs/{name}.log"), outputs=[str(output)]))
            if mode in SELECTED:
                destination=DATA/f"cycles-v1/seed_{seed:03d}/{mode}"
                cycle_jobs.append(dict(id=name, command=command("cycle", "--checkpoint",cp,
                    "--output",destination,"--protocol",protocol_path,"--device","cuda","--threads",1),
                    log=str(DATA/f"cycles-v1/logs/{name}.log"), outputs=[str(destination/"complete.json")]))
    for label,jobs in (("main",train_jobs),("evaluation",eval_jobs),("cycles",cycle_jobs)):
        atomic_json(EVIDENCE/f"{label}-commands-v1.json", dict(format="credit-commands-v1", cwd=str(ROOT),
            protocol=str(protocol_path), protocol_sha256=sha(protocol_path), file_hashes=frozen, jobs=jobs))
    print(protocol_path)
