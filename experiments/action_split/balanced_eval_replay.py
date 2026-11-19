"""Independent replay audit of the single declared balanced-mode control.

The original replay source remains frozen.  Its protocol/hash helpers are
reused with isolated paths; this wrapper changes the fresh-case identity checks
and launches its own module.  Native actions are independently reconstructed
with plain PyTorch GRU/linear modules and explicit conditional argmax.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from experiments.action_split import eval_replay as base

ROOT = base.ROOT
INPUT = ROOT / "data/action-split/balanced-eval-v1"
OUT = ROOT / "data/action-split/balanced-eval-audit-v1"
PLAN = OUT / "replay_plan.json"
ORIGINAL_SOURCE_FILES = base.source_files


def source_files():
    # Pin this wrapper as well as the unchanged replay implementation and every
    # source frozen by the balanced evaluation protocol.
    sources = ORIGINAL_SOURCE_FILES() + [Path(__file__)]
    sources += [ROOT / name for name in base.read(INPUT / "protocol.json")["sources"]]
    return list(dict.fromkeys(sources))


def bind():
    base.INPUT, base.OUT, base.PLAN = INPUT, OUT, PLAN
    base.source_files = source_files


def declare():
    bind()
    protocol = base.read(INPUT / "protocol.json")
    assert protocol["world_seed"] == "852000+seed"
    assert protocol["actor_ids"] == list(range(97000, 97032))
    for name, digest in protocol["sources"].items():
        assert base.sha(ROOT / name) == digest, name
    frozen = base.read(INPUT / "freeze.json")
    for name, digest in frozen["checkpoints"].items():
        assert base.sha(ROOT / name) == digest, name
    base.declare()


@base.torch.inference_mode()
def audit(index):
    """Same replay checks, with the balanced protocol's fresh-case identities.

    The parent audit hardcodes the first experiment's worldseed/actors.  A
    separate function here preserves its source and its completed results.
    """
    bind()
    base.torch.set_num_threads(1)
    plan = base.checked()
    row = plan["runs"][index]
    for name in ("metadata", "facts", "checkpoint"):
        assert base.sha(ROOT / row[name]) == row[name+"_sha256"], row[name]
    metadata = base.read(ROOT / row["metadata"])
    cp = base.torch.load(ROOT / row["checkpoint"], weights_only=False, map_location="cpu")
    cfg = cp["model_config"]
    assert cfg["input_size"] == 56 and cfg["n_jobs"] == 2
    n_actions = int(cfg["n_actions"])
    assert n_actions == (11 if row["policy"] == "original" else 13)
    core = base.nn.GRU(cfg["input_size"], cfg["width"], num_layers=1).eval().requires_grad_(False)
    head = base.nn.Linear(cfg["width"], cfg["n_jobs"]*n_actions).eval().requires_grad_(False)
    core.load_state_dict({k[len("core."):]: v for k,v in cp["model"].items() if k.startswith("core.")})
    head.load_state_dict({k[len("head."):]: v for k,v in cp["model"].items() if k.startswith("head.")})
    with base.np.load(ROOT / row["facts"], allow_pickle=False) as archive:
        facts = {name: archive[name] for name in archive.files}
    assert facts["obs"].dtype == base.np.float32 and facts["obs"].shape == (4608, 32, 56)
    assert facts["actions"].shape == (4608, 32, 2)
    expected_hashes = {name: base.hashlib.sha256(base.np.ascontiguousarray(facts[name])).hexdigest()
                       for name in base.FACT_FIELDS}
    replay_hashes = {name: base.hashlib.sha256() for name in base.FACT_FIELDS}
    kwargs = metadata["world_config"]
    assert kwargs["seed"] == 852000+row["seed"]
    assert kwargs["actor_ids"] == list(range(97000, 97032))
    assert kwargs["dt"] == .25
    assert kwargs["duration_range"] == ([8., 16.] if row["condition"] == "long" else [16., 32.])
    state = base.torch.zeros(1, 32, cfg["width"], dtype=head.weight.dtype)
    with base.mapping_scope(base.mapping_table(row["seed"], "related")):
        world = base.ContextWorld(**kwargs)
        for tick in range(4608):
            obs = world.observe()
            assert base.np.array_equal(obs.numpy(), facts["obs"][tick]), (index, tick, "obs")
            for name, attr in base.FACT_FIELDS.items():
                replay_hashes[name].update(getattr(world, attr).numpy())
            assert world.now == facts["now"][tick] == tick*.25
            values, state = core(obs[None], state)
            scores = head(values[0]).reshape(32, 2, n_actions)
            if n_actions == 11:
                chosen = scores.argmax(-1)
            else:
                mode = scores[..., :3].argmax(-1)
                begin = 1+scores[..., 3:5].argmax(-1)
                finish = 3+scores[..., 5:13].argmax(-1)
                chosen = base.torch.zeros_like(mode)
                chosen[mode == 1] = begin[mode == 1]
                chosen[mode == 2] = finish[mode == 2]
            assert base.np.array_equal(chosen.numpy(), facts["actions"][tick]), (index, tick, "policy action")
            world.act(base.torch.from_numpy(facts["actions"][tick]))
            assert base.np.array_equal(world.last_result.numpy(), facts["receipts"][tick]), (index, tick, "receipt")
        assert world.event_hash == metadata["external_hash"]
    for name in base.FACT_FIELDS:
        assert replay_hashes[name].hexdigest() == expected_hashes[name], (index, name)
    assert base.np.array_equal(state[0].numpy(), facts["final_h"]), (index, "final_h")
    result = dict(index=index, seed=row["seed"], policy=row["policy"], condition=row["condition"],
        passed=True, ticks=4608, opportunities=4608*32*2, public_observations="bitwise exact",
        private_fields={name: "bitwise exact" for name in base.FACT_FIELDS},
        actions="independent plainGRU+linear and explicit decoder; all bitwise exact",
        receipts="native exact everytick", final_h="bitwise exact", clock="exact everytick",
        external_hash=world.event_hash, input_files=row, plan_sha256=base.sha(PLAN),
        auditor_sha256=base.sha(__file__), parent_auditor_sha256=base.sha(base.__file__))
    base.write_new(OUT / "replay_runs" / f"run_{index:03d}.json", result)
    print(base.json.dumps(dict(index=index, seed=row["seed"], policy=row["policy"],
                               condition=row["condition"], passed=True)), flush=True)


def launch(workers=6):
    bind()
    plan = base.checked()
    assert 1 <= workers <= 6
    logs = OUT / "replay_logs"
    logs.mkdir(parents=True, exist_ok=True)
    def job(index):
        path = logs / f"run_{index:03d}.log"
        with path.open("x", encoding="utf-8") as handle:
            done = subprocess.run([sys.executable, "-m", "experiments.action_split.balanced_eval_replay",
                "run", "--index", str(index)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if base.os.name == "nt" else 0)
        record = dict(index=index, exit_code=done.returncode, log=str(path.relative_to(ROOT)))
        print(base.json.dumps(record), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=workers) as pool:
        workers_done = list(pool.map(job, range(len(plan["runs"]))))
    assert all(row["exit_code"] == 0 for row in workers_done), workers_done
    results = [base.read(OUT / "replay_runs" / f"run_{index:03d}.json") for index in range(30)]
    base.write_new(OUT / "replay.json", dict(passed=True, runs=30,
        ticks_compared=sum(row["ticks"] for row in results),
        action_opportunities_compared=sum(row["opportunities"] for row in results),
        native_observations_receipts_privatefacts="all bitwise exact",
        independent_policy_actions_finalh="all bitwise exact", posthoc_control=True,
        plan_sha256=base.sha(PLAN), auditor_sha256=base.sha(__file__),
        parent_auditor_sha256=base.sha(base.__file__), workers=workers_done, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "all"))
    parser.add_argument("--index", type=int, choices=range(30))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "run":
        audit(args.index)
    else:
        launch(args.workers)
