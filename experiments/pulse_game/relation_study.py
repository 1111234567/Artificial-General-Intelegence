"""Fit a lifetime update rule on reflection games and test it on sealed rotations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/learning-transfer-2026-11-19"
DATA = ROOT / "data/pulse-game/learning-transfer-20261119/relation"
REG_PATH = EVIDENCE / "relation-protocol.json"
REG = json.loads(REG_PATH.read_text(encoding="utf-8"))
BIRTHS = REG["births"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def body_path(seed):
    return DATA / "bodies" / f"birth-{seed}.npz"

def fit_path(seed):
    # Development scores are valid only for the exact rule, game and selection code.
    folder = Path(__file__).parent
    ingredients = [REG_PATH, Path(__file__), folder/"relation_world.py", folder/"relation_learner.py",
                   body_path(seed), DATA/"bodies/readout-options.json"]
    key = hashlib.sha256("".join(digest(p) for p in ingredients).encode()).hexdigest()[:16]
    return DATA / "fit" / key / f"{seed}.json"


def prepare():
    """Use the released bodies if present. The original run also supports export."""
    from . import lifelong_agent
    folder = DATA / "bodies"
    folder.mkdir(parents=True, exist_ok=True)
    names = [body_path(seed).name for seed in BIRTHS] + ["readout-options.json"]
    bundle = EVIDENCE / "relation-births.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as source:
            if set(source.namelist()) != set(names):
                raise ValueError("Unexpected body archive members")
            for name in names:
                (folder / name).write_bytes(source.read(name))
        return
    from . import independent_births, research_audit, unified_physical
    bank = ROOT / "data/pulse-game/independent-lifelong-births-v1"
    registry = read(bank / "registered.json")
    assert registry["fixed_birth_seeds"][:len(BIRTHS)] == BIRTHS
    (folder / names[-1]).write_bytes(lifelong_agent.DEFAULT.with_name(names[-1]).read_bytes())
    for seed in BIRTHS:
        parent = bank / f"birth-{seed}"
        frozen, = parent.glob("*--full-*/frozen.json")
        f, r, motor = independent_births.resources(parent)
        f, r, motor = unified_physical.materialize(f, r, motor, np.asarray(read(frozen)["best"]["genome"]), "full")
        r["release"] = registry["reopening_rule"]
        identity = dict(birth_seed=seed, graph_sha256=research_audit.fingerprint(parent / "receptor.pt"),
                        selection_sha256=research_audit.fingerprint(frozen))
        lifelong_agent.save(body_path(seed), f, r, motor, identity)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for name in names:
            out.write(folder / name, name)


def roots(rng, count, lower, upper):
    return (lower + rng.choice(upper - lower, count, replace=False)).tolist()


def new_factory(seed, vec, mode):
    from .relation_learner import Rule, birth
    rule = Rule.from_vector(Rule.default_vector() if mode == "initial" else np.asarray(vec))
    actual_mode = "adaptive" if mode == "initial" else mode
    def make(life_seeds):
        return birth(life_seeds, rule, mode=actual_mode, checkpoint=body_path(seed))
    return make


def score(seed, vec, protocol):
    from .relation_world import evaluate
    runs = evaluate(protocol, new_factory(seed, vec, "adaptive"), rollback=False, shuffle=False)
    components = []
    for history in ("related", "unrelated"):
        results = runs[history]["intact"]
        components.append([float(np.mean(results[str(d)][metric]))
                           for d, metric in ((2,"new"), (4,"new"), (16,"new"), (16,"old"))])
    return float(np.mean(components))


def fit(seed):
    from .relation_world import make
    from .relation_learner import Rule
    out = fit_path(seed)
    if out.exists():
        return
    start = perf_counter()
    generator = np.random.default_rng([seed, 417108])
    default = np.asarray(Rule.default_vector(), dtype=float)
    sigmas = np.asarray(Rule.mutation_scales(), dtype=float)
    if default.shape != sigmas.shape or not np.all(sigmas > 0):
        raise ValueError("Invalid parameter search scales")
    center = default.copy()
    finalists = [center]
    training = []
    def dev(ids):
        return make(ids, partition="develop", history_pairs=REG["history_pairs"], probes=1,
                    doses=(0,2,4,16), counterbalance=True)
    for index, scale in enumerate(REG["perturbation_scales"]):
        ids = roots(generator, REG["training_worlds_per_round"], 100_000 + index*55_000_000, 55_000_000 + index*55_000_000)
        game = dev(ids)
        choices = [center]
        for _ in range(REG["directions_per_round"]):
            step = generator.normal(size=len(center)) * scale * sigmas
            choices.extend([center + step, center - step])
        scores = [score(seed, v, game) for v in choices]
        winner = int(np.argmax(scores))
        center = choices[winner]
        finalists.append(center)
        training.append(dict(roots=ids, scores=scores, winner=winner, vectors=[v.tolist() for v in choices]))
        print(f"birth {seed} generation {index+1} reward {scores[winner]:.3f}", flush=True)
    validation_roots = roots(generator, REG["validation_worlds"], 180_000_000, 260_000_000)
    validation = dev(validation_roots)
    scores = [score(seed, v, validation) for v in finalists]
    i = int(np.argmax(scores))
    write(out, dict(seed=seed, selected=finalists[i].tolist(), selected_index=i, validation_roots=validation_roots,
                    validation_scores=scores, training=training, body_sha256=digest(body_path(seed)),
                    elapsed_seconds=perf_counter()-start))


def seal():
    folder = Path(__file__).parent
    paths = [REG_PATH, EVIDENCE / "relation-births.zip",
             *(folder / name for name in ("relation_study.py", "relation_world.py", "relation_learner.py",
                                          "habitat_world.py", "world.py", "lifelong_agent.py", "content_routing.py"))]
    payload = dict(files={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
                   selected={str(s):read(fit_path(s))["selected"] for s in BIRTHS})
    original = EVIDENCE / "relation-original-seal.json"
    if original.exists():
        first = read(original)
        if first["selected"] != payload["selected"]:
            raise RuntimeError("Output-only repair must preserve all selected parameters")
        changes = [p for p in payload["files"] if payload["files"][p] != first["files"][p]]
        if changes != ["experiments/pulse_game/relation_study.py"]:
            raise RuntimeError(f"Unexpected change to original sealed inputs: {changes}")
        payload["original_seal_sha256"] = digest(original)
        payload["amendment"] = "The first holdout invocation stopped at chunk two because _seed and _parity had been added to the accumulated output keys only. No case file or metric was emitted. Compare schemas after adding both keys. Same environments, seeds, fits and selected vectors; no new selection."
    path = EVIDENCE / "relation-seal.json"
    if path.exists() and read(path) != payload:
        raise RuntimeError("Sealed settings changed")
    if not path.exists(): write(path, payload)
    return payload


def target(ids):
    from .relation_world import make
    return make(ids, partition="heldout", history_pairs=REG["history_pairs"], probes=REG["probes_per_dose"],
                doses=tuple(REG["doses"]), counterbalance=True)


def test_roots():
    return roots(np.random.default_rng(REG["test_seed"]), REG["test_worlds"], 300_000_000, 470_000_000)


def pack_rows(nested):
    arrays = {}
    for history, variants in nested.items():
        for variant, doses in variants.items():
            for dose, row in doses.items():
                for key, v in row.items():
                    x = np.asarray(v)
                    if x.dtype.kind not in "iufb" or not np.isfinite(x).all():
                        raise ValueError((history, variant, dose, key, x.dtype))
                    arrays[f"{history}/{variant}/{dose}/{key}"] = x
    return arrays


def evaluate_case(args):
    from .relation_world import evaluate, public_oracle, physical_oracle
    seed, arm = args
    out = DATA / "test" / f"{seed}-{arm}.npz"
    if out.exists(): return
    start = perf_counter()
    ids = test_roots()
    chunks = {}
    for offset in range(0, len(ids), 32):
        game = target(ids[offset:offset+32])
        if arm in ("public_oracle", "physical_oracle"):
            rows = (public_oracle if arm == "public_oracle" else physical_oracle)(game)
        else:
            frozen = read(EVIDENCE / "relation-seal.json")["selected"][str(seed)]
            rows = evaluate(game, new_factory(seed, frozen, arm), rollback=arm=="adaptive", shuffle=arm=="adaptive")
        arrays = pack_rows(rows)
        n = len(game.seeds)
        if n != len(ids[offset:offset+32]) * 4: raise ValueError("Counterbalance missing")
        # Store row identity explicitly; the world's row order need not group parities by root.
        arrays["_seed"] = np.asarray(game.seeds)
        arrays["_parity"] = np.asarray(game.parity)
        if chunks and set(chunks) != set(arrays): raise ValueError("Output schema changed across chunks")
        for k, x in arrays.items():
            if x.shape[0] != n: raise ValueError((k, x.shape, n))
            chunks.setdefault(k, []).append(x)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.npz")
    np.savez_compressed(partial, **{k:np.concatenate(v) for k,v in chunks.items()})
    partial.replace(out)
    print(f"sealed {seed} {arm} in {perf_counter()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "fit", "seal", "test", "report"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "prepare": prepare()
    elif args.phase == "seal": seal()
    elif args.phase == "report":
        from .relation_analysis import report
        report()
    else:
        if args.phase == "fit":
            worker, jobs = fit, BIRTHS
        else:
            seal()
            worker = evaluate_case
            jobs = [(s,a) for a in ("adaptive","local-selective","local-only","initial","frozen") for s in BIRTHS]
            jobs += [(0,"public_oracle"),(0,"physical_oracle")]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, jobs))


if __name__ == "__main__":
    main()
