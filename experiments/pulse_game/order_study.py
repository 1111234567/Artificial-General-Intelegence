"""Register, train, freeze and score the pulse-game temporal-order comparison."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from . import order_model as model, order_world as env


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "data" / "pulse-game" / "learning-transfer-20261119" / "order"
TOPOLOGIES = ("graph", "self")
DEV_BIRTHS = (126293, 689241)
BIRTHS = (403154, 693025, 762880, 118340, 297468, 130176,
          715492, 545305, 864034, 462128, 973017, 358313)
RATES = (.008, .016)
STEPS = (220, 500, 900)
BATCH_SIZE = 48
TRAIN = tuple(range(40135000, 40135512))
DEV = tuple(range(61352400, 61352496))
TEST = tuple(range(87281900, 87282092))
TRAIN_REQUESTS = 6
DEV_REQUESTS = 6
TEST_REQUESTS = 12
FORKS = ("swap", "first_only", "first_only_flip", "second_only", "second_only_flip",
         "timing_only", "timing_only_flip")
SOURCES = ("order_world.py", "order_model.py", "order_study.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def register(out):
    assert not (set(TRAIN) & set(DEV) or set(TRAIN) & set(TEST) or set(DEV) & set(TEST))
    assert not set(DEV_BIRTHS) & set(BIRTHS)
    inputs = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": "The directed neighbor edge improves physical contact accuracy on separated ordered odors over an independently trained self edge.",
        "primary": "Mean correct physical contacts per opportunity, long profile, graph minus self. Every timeout counts as incorrect.",
        "secondary": "Standard profile, contact coverage, fork reversals, fixed-weight interventions, and control floors.",
        "interpretation": "Report a positive difference when its 95% interval excludes zero. A 95% interval within +/-3 pp supports practical similarity for this task; an interval crossing both boundaries is inconclusive.",
        "balance": "Distinct ordered pair uniformly drawn from 3 odors. Chronological successor vs reverse yields opposite true sides. Any isolated odor or unordered pair has chance accuracy. Timings, motor gate, distractors, geometry, and layouts do not encode true side.",
        "paired_forks": "Run identical prior 11 requests and fork the 12th. Swapping the odor identities preserves timing, carrier and unrelated observations. Single-odor and timing-only twin pairs have bit-identical public histories and opposite private grades.",
        "candidate_models": "32 leaky cells; each receives the same class of learned input and leak projections. Neighbor ring candidate edge versus additional self candidate edge; both contain the same intrinsic self retention. Same tensors at birth, width, number of learned coefficients, readout, batch order, and Adam.",
        "reward_grounding": "Independent exogenous exploration chooses a random physical side and sends identical full motor pulses to both training arms. Chosen-action value targets use only actual delivered signed taste after contact. The sensory state sees neither private targets nor texture or reward labels; training uses receipt as supervision.",
        "epoch_definition": "Each optimizer step samples 48 lives with no replacement and two consecutive public requests. Burn in on first request with frozen gradients, update on available receipts for second request across all movement ticks; alternate timing profiles. Persistent at validation and test. This is action-outcome supervision with offline Adam.",
        "development_selection": "For every rate and step limit, mean physical accuracy over both dev births, both profiles, both architectures; highest wins, then fewer steps, then lower rate. Choose one shared recipe. All primary births are independently trained from scratch after recipe lock.",
        "paired_comparison": "Holdout roots and births are crossed. Resample matched birth pairs and world roots independently for percentile 95% bootstrap CIs. Use bootstraps on paired differences, not overlap of marginal intervals.",
        "diagnostics": ["chronological oracle on public odors", "random physical explorer",
                        "forks with single cue or timing only", "drop candidate recurrent edge",
                        "reset before the second target", "no recurrence"],
        "width": model.WIDTH,
        "batch_size": BATCH_SIZE,
        "training_roots": TRAIN, "dev_roots": DEV, "holdout_roots": TEST,
        "training_requests": TRAIN_REQUESTS, "dev_requests": DEV_REQUESTS, "holdout_requests": TEST_REQUESTS,
        "dev_births": DEV_BIRTHS, "confirm_births": BIRTHS,
        "rates": RATES, "checkpoints": STEPS,
        "timing": {key: vars(value) for key, value in env.TIMING.items()},
        "source_sha256_at_registration": source_hashes(),
    }
    if (out / "registered.json").exists():
        raise RuntimeError("Registration already exists")
    write(out / "registered.json", inputs)


def budgets():
    return {topology: {key: value.numel() for key, value in model.Circuit(0, topology).named_parameters()}
            for topology in TOPOLOGIES}


def pools():
    return [model.pool(env.make(TRAIN, TRAIN_REQUESTS, kind)) for kind in env.TIMING]


def sampler_seed(birth):
    return int(np.random.SeedSequence([birth, 559603]).generate_state(1)[0])


def development(out, rate):
    read(out / "registered.json")
    if rate not in RATES:
        raise ValueError(f"Unregistered rate {rate}")
    if (out / "recipe.json").exists():
        raise RuntimeError("Development is closed after recipe lock")
    training = pools()
    worlds = {kind: env.make(DEV, DEV_REQUESTS, kind) for kind in env.TIMING}
    for birth in DEV_BIRTHS:
        for topology in TOPOLOGIES:
            dest = out / "development" / f"rate-{rate:.3f}-birth-{birth}-{topology}.json"
            if dest.exists() and read(dest)["scores"][-1]["step"] == STEPS[-1]:
                continue
            net = model.Circuit(birth, topology)
            scores = []

            def record(circuit, step, history, optimizer):
                score = {"step": step, "physical": {}}
                for profile, tape in worlds.items():
                    result = model.run(circuit, tape)
                    score["physical"][profile] = {"accuracy": result.accuracy,
                                                   "contact_rate": result.contact_rate}
                scores.append(score)
                write(dest, {"rate": rate, "birth": birth, "topology": topology,
                             "scores": scores, "history": history})
                print(rate, birth, topology, score, flush=True)

            model.train(net, training, seed=sampler_seed(birth), rate=rate, checkpoints=STEPS,
                        batch_size=BATCH_SIZE, steps_per_step_record=20, validate=record)


def freeze_recipe(out):
    if (out / "recipe.json").exists():
        raise RuntimeError("Recipe already locked")
    trials = []
    for rate in RATES:
        rows = [read(out / "development" / f"rate-{rate:.3f}-birth-{birth}-{arm}.json")
                for birth in DEV_BIRTHS for arm in TOPOLOGIES]
        for i, step in enumerate(STEPS):
            scores = [row["scores"][i]["physical"][profile]["accuracy"]
                      for row in rows for profile in env.TIMING]
            trials.append({"rate": rate, "steps": step, "score": float(np.mean(scores))})
    selected = max(trials, key=lambda row: (row["score"], -row["steps"], -row["rate"]))
    write(out / "recipe.json", {"selected": selected, "development": trials, "budget": budgets(),
                                 "registered_sha256": digest(out / "registered.json"),
                                 "source_sha256": source_hashes(),
                                 "platform": platform.platform(), "python": platform.python_version(),
                                 "torch": torch.__version__, "numpy": np.__version__})


def train_birth(out, birth):
    if birth not in BIRTHS:
        raise ValueError(f"Unregistered confirmation birth {birth}")
    recipe = read(out / "recipe.json")["selected"]
    training = pools()
    receipt_counts = {kind: {"opportunities": values[2].size, "delivered": int((values[2] != 0).sum()),
                             "positive": int((values[2] > 0).sum()), "negative": int((values[2] < 0).sum())}
                      for kind, values in zip(env.TIMING, training, strict=True)}
    dest = out / "births" / str(birth)
    dest.mkdir(parents=True, exist_ok=True)
    for topology in TOPOLOGIES:
        weights = dest / f"{topology}.pt"
        metrics = dest / f"{topology}-training.json"
        if weights.exists() and metrics.exists():
            continue
        net = model.Circuit(birth, topology)
        begin = time.perf_counter()

        def save(circuit, step, history, optimizer):
            payload = {"birth": birth, "topology": topology, "state": circuit.state_dict(), "step": step,
                       "recipe_sha256": digest(out / "recipe.json")}
            temporary = weights.with_suffix(".writing")
            torch.save(payload, temporary)
            temporary.replace(weights)
            write(metrics, {"history": history, "training_seconds": time.perf_counter() - begin,
                            "receipt_counts": receipt_counts, "example_order_seed": sampler_seed(birth),
                            "checkpoint_sha256": digest(weights)})
            print("trained", birth, topology, step, flush=True)

        model.train(net, training, seed=sampler_seed(birth), rate=recipe["rate"], checkpoints=(recipe["steps"],),
                    batch_size=BATCH_SIZE, steps_per_step_record=20, validate=save)


def frozen(out):
    file = out / "frozen.json"
    if file.exists():
        raise RuntimeError("Checkpoints already frozen")
    checkpoints = {}
    for birth in BIRTHS:
        for topology in TOPOLOGIES:
            path = out / "births" / str(birth) / f"{topology}.pt"
            train_log = path.with_name(f"{topology}-training.json")
            checkpoint_hash = digest(path)
            if read(train_log)["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError(f"Training log mismatch for {path}")
            weights = torch.load(path, weights_only=True, map_location="cpu")
            if weights["recipe_sha256"] != digest(out / "recipe.json"):
                raise RuntimeError(f"Training recipe mismatch for {path}")
            checkpoints[path.relative_to(out).as_posix()] = checkpoint_hash
    write(file, {"checkpoints": checkpoints, "recipe_sha256": digest(out / "recipe.json"),
                 "registered_sha256": digest(out / "registered.json"), "source_sha256": source_hashes(),
                 "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def load_net(out, birth, topology, manifest):
    path = out / "births" / str(birth) / f"{topology}.pt"
    if digest(path) != manifest["checkpoints"][path.relative_to(out).as_posix()]:
        raise RuntimeError(f"Checkpoint hash mismatch for {path}")
    net = model.Circuit(birth, topology)
    net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["state"])
    return net


def outcomes(result, last=False):
    values = [result.correct, result.contact_side, result.contact_tick, result.reward]
    if last:
        values = [value[:, -1] for value in values]
    return np.stack(values, axis=-1, dtype=np.int8)


def confirm(out):
    manifest = read(out / "frozen.json")
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("Source hash mismatch after freeze")
    for profile in env.TIMING:
        tape = env.make(TEST, TEST_REQUESTS, profile)
        forks = {kind: env.fork(tape, kind) for kind in FORKS}
        diagnostic = {
            "oracle": outcomes(env.chronological_oracle(tape)),
            "explorer": outcomes(env.explore(tape)),
            "oracle_swap": outcomes(env.chronological_oracle(forks["swap"]), last=True),
        }
        goal = env._sides(tape.target_symbols)
        for kind in FORKS:
            counterfactual = forks[kind]
            if not np.array_equal(counterfactual.observations[:-1], tape.observations[:-1]):
                raise RuntimeError("Fork changed common-prefix observations")
            if kind == "swap" or kind.endswith("_flip"):
                if not (env._sides(counterfactual.target_symbols)[:, -1] == 1 - goal[:, -1]).all():
                    raise RuntimeError(f"Fork {kind} failed to invert expected side")
            if kind.endswith("_flip"):
                if not np.array_equal(counterfactual.observations, forks[kind[:-5]].observations):
                    raise RuntimeError(f"Fork {kind} leaked information through observations")
        np.savez_compressed(out / f"{profile}-worlds.npz", roots=TEST, goal_side=goal,
                            target_ticks=tape.target_ticks, target_symbols=tape.target_symbols,
                            distances=tape.body.distances, **diagnostic)

        for birth in BIRTHS:
            path = out / "births" / str(birth) / f"{profile}.npz"
            if path.exists():
                continue
            values = {}
            for topology in TOPOLOGIES:
                net = load_net(out, birth, topology, manifest)
                intact = model.run(net, tape)
                values[topology + "_intact"] = outcomes(intact)
                for name, mode in (("drop_candidate_edge", 1), ("erase_history", 2), ("no_recurrence", 3)):
                    values[topology + "_" + name] = outcomes(model.run(net, tape, mode))
                for kind, counterfactual in forks.items():
                    result = model.run(net, counterfactual)
                    if not np.array_equal(result.contact_side[:, :-1], intact.contact_side[:, :-1]):
                        raise RuntimeError(f"Fork {kind} changed common-prefix physical actions")
                    values[topology + "_" + kind] = outcomes(result, last=True)
                for kind in ("first_only", "second_only", "timing_only"):
                    first, second = values[topology + "_" + kind], values[topology + "_" + kind + "_flip"]
                    if not np.array_equal(first[:, 1:3], second[:, 1:3]):
                        raise RuntimeError(f"Fork {kind} leaked answer into physical choices")
            np.savez_compressed(path, **values)
            print("heldout", profile, birth, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "development", "recipe", "train", "freeze", "confirm"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--births", nargs="+", type=int, default=BIRTHS)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    if opts.stage == "register":
        register(opts.out)
    elif opts.stage == "development":
        development(opts.out, opts.rate)
    elif opts.stage == "recipe":
        freeze_recipe(opts.out)
    elif opts.stage == "train":
        for birth in opts.births:
            train_birth(opts.out, birth)
    elif opts.stage == "freeze":
        frozen(opts.out)
    elif opts.stage == "confirm":
        confirm(opts.out)


if __name__ == "__main__":
    main()
