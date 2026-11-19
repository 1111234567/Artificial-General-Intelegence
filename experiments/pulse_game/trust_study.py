"""Fit, seal and evaluate the contact-conditioned covariance gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np

from . import lifelong_agent, relation_study, relation_world, trust_replay
from .relation_learner import RelationLearner, Rule
from .trust_gate import Gate, TrustLearner


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/adaptive-trust-2026-11-19"
DATA = ROOT / "data/pulse-game/adaptive-trust-20261119"
PROTOCOL = EVIDENCE / "protocol.json"
BASE = relation_study.EVIDENCE / "relation-seal.json"
BUNDLE = relation_study.EVIDENCE / "relation-births.zip"
CHECKPOINT = EVIDENCE / "checkpoint.json"
REG = relation_study.read(PROTOCOL)
BIRTHS = REG["births"]
ARMS = ("always", "local", "constant", "explicit", "learned", "swapped")


@lru_cache(maxsize=6)
def resources(birth_id):
    """Read only the six previously released receptor, actuator and rule sets."""
    name = f"birth-{birth_id}.npz"
    from io import BytesIO
    with zipfile.ZipFile(BUNDLE) as archive:
        front, _, motion, _ = lifelong_agent.load(BytesIO(archive.read(name)))
    rule = Rule(relation_study.read(BASE)["selected"][str(birth_id)])
    return front, motion, rule


def factory(seeds, birth_id, mode, *, schedule=None):
    """Build an observation-only actor suitable for any ordinary relation world."""
    front, motion, rule = resources(birth_id)
    if mode in ("learned", "constant"):
        selected = relation_study.read(CHECKPOINT)
        weights = selected["learned"] if mode == "learned" else [selected["constant"]]
    else:
        weights = ()
    return TrustLearner(front, seeds, motion=motion, rule=rule, gate=Gate(mode, tuple(weights)),
                        schedule=schedule)


def root_ids(spec):
    return relation_study.roots(np.random.default_rng(spec["seed"]), spec["count"], spec["lower"], spec["upper"])


def make(ids, partition):
    probes = REG["development_probes"] if partition == "develop" else REG["holdout_probes"]
    return relation_world.make(ids, partition=partition, history_pairs=12, probes=probes,
                               doses=tuple(REG["doses"]), counterbalance=True, nonce=REG["nonce"])


def score(results):
    weights = {2: .4, 4: .4, 8: .1, 16: .1}
    return float(np.mean([
        sum(weight * probes[dose]["new"]["correct"].mean() for dose, weight in weights.items())
        for probes, _, _ in results.values()
    ]))


def score_candidates(job):
    birth_id, ids, gates = job
    start = perf_counter()
    front, motion, rule = resources(birth_id)
    protocol = make(ids, "develop")
    prepared = trust_replay.prepare(protocol, front, motion, rule, old=False, initial=False)
    scores = [score(trust_replay.evaluate(prepared, rule, motion, gate)) for gate in gates]
    print(f"reflection {birth_id} ({len(ids)} worlds, {len(gates)} candidates) {perf_counter() - start:.1f}s",
          flush=True)
    return scores


def candidate_pool():
    for center in REG["fit"]["initial_centers"]:
        for slope in REG["fit"]["initial_slopes"]:
            # Generic behavioral search starts with several rising response curves.
            yield Gate("learned", (slope * (.8 - center), .6 * slope, 0., 0., 0., 0.,
                                  .4 * slope, 0., 0.))
    yield Gate("learned", (0., 0., 0., 0., 0., 0., 0., 0., 0.))


def fit(workers):
    if CHECKPOINT.exists():
        raise RuntimeError("The checkpoint already exists; use an independent study for another fit")
    start = perf_counter()
    config = REG["fit"]
    rng = np.random.default_rng(config["seed"])
    initial = list(candidate_pool())
    candidates, finalists, rounds = initial, initial.copy(), []
    scale = np.array([2., 7., 2., 1., 1., 1.5, 6., 2., 1.5])

    def search(pool, ids, gates):
        scores = np.asarray(list(pool.map(score_candidates, [(b, ids, gates) for b in BIRTHS])))
        return scores, np.argsort(-scores.mean(axis=0), kind="stable")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for spec, size, anneal in zip(REG["roots"]["training"], config["perturbations_by_round"],
                                      config["perturbation_scales"], strict=True):
            ids = root_ids(spec)
            perturbed = []
            for i in range(size // 2):
                center = np.asarray(candidates[i % len(candidates)].weights)
                step = rng.normal(size=len(center)) * scale * anneal
                for sign in (1, -1):
                    perturbed.append(Gate("learned", tuple((center + sign * step).tolist())))
            choices = candidates + perturbed
            scores, ranking = search(pool, ids, choices)
            winners = ranking[:config["elite_per_round"]]
            candidates = [choices[i] for i in winners]
            finalists.extend(candidates)
            rounds.append(dict(roots=ids, candidates=[g.weights for g in choices], scores_by_front=scores.tolist(),
                               elite=winners.tolist()))
        ids = root_ids(REG["roots"]["validation"])
        finalists = list(dict.fromkeys(g.weights for g in finalists))
        learned = [Gate("learned", w) for w in finalists]
        constants = [Gate("constant", (w,)) for w in config["constants"]]
        choices = learned + constants + [Gate("explicit")]
        scores, _ = search(pool, ids, choices)
    means = scores.mean(axis=0)
    selected = int(np.argmax(means[:len(learned)]))
    best_constant = int(np.argmax(means[len(learned):-1])) + len(learned)
    provenance = dict(training=rounds, validation=dict(roots=ids, candidates=[
        {"mode": g.mode, "weights": g.weights} for g in choices], scores_by_front=scores.tolist(),
        learned_selected=selected, constant_selected=best_constant), elapsed_seconds=perf_counter() - start)
    relation_study.write(EVIDENCE / "dev-fit.json", provenance)
    report = dict(learned=list(choices[selected].weights), constant=choices[best_constant].weights[0],
                  development_objectives=dict(learned=float(means[selected]), constant=float(means[best_constant]),
                                              explicit=float(means[-1])))
    relation_study.write(CHECKPOINT, report)
    print(json.dumps(report), flush=True)


def source_fingerprints():
    base = relation_study.read(BASE)
    for name, value in base["files"].items():
        if relation_study.digest(ROOT / name) != value:
            raise RuntimeError(f"Previous sealed dependency changed: {name}")
    local = [PROTOCOL, BASE, CHECKPOINT, EVIDENCE / "dev-fit.json"]
    local += [Path(__file__).parent / name for name in ("trust_gate.py", "trust_replay.py",
        "trust_study.py", "trust_analysis.py", "habitat_deploy.py", "context_local.py", "mixture_world.py",
        "mixture_flash.py", "tests/test_trust_study.py")]
    return {**base["files"], **{p.relative_to(ROOT).as_posix(): relation_study.digest(p) for p in local}}


def seal(*, create=False):
    path = EVIDENCE / "seal.json"
    if not create and not path.exists():
        raise RuntimeError("Freeze the choices and source with the seal phase before opening holdout")
    payload = dict(files=source_fingerprints(), selected=relation_study.read(CHECKPOINT))
    if path.exists():
        if relation_study.read(path) != payload:
            raise RuntimeError("Sealed source or checkpoint changed")
    elif create:
        from datetime import datetime, timezone
        from platform import python_version
        relation_study.write(path, payload)
        relation_study.write(EVIDENCE / "sealed-at.json", dict(utc=datetime.now(timezone.utc).isoformat(),
                              python=python_version(), numpy=np.__version__))
    return payload["selected"]


def verify_closure(protocol, front, motion, rule, learner, cached, *, replay_base=False, schedules=None):
    """Compare every silent probe and eligible write with the original tick loop."""
    probes, writes = 0, 0
    for history, (doses, traces, exposure) in cached.items():
        schedule = None
        if schedules is not None:
            start = protocol.history[history].menu.body.trials
            schedule = {start + index: arr for index, arr in enumerate(schedules[history].T)}
        if replay_base:
            actor = RelationLearner(front, protocol.seeds, motion=motion, rule=rule,
                                    share_prior=learner.mode == "always")
        else:
            actor = TrustLearner(front, protocol.seeds, motion=motion, rule=rule, gate=learner, schedule=schedule)
        actor.reset(len(protocol.seeds))

        def compare_stage(stage, prefix, span):
            actual = relation_world._play(actor, stage)
            for key, data in (("side", actual.contact_side), ("tick", actual.contact_tick),
                              ("taste", actual.observed_reward.sum(axis=2))):
                np.testing.assert_array_equal(data, exposure[f"{prefix}_{key}"][:, span])

        compare_stage(protocol.history[history], "history", slice(None))
        last = 0
        for dose, branches in doses.items():
            if dose:
                span = slice(3 * last // 2, 3 * dose // 2)
                compare_stage(protocol.teaching.slice(span.start, span.stop), "teaching", span)
            for key, outcomes in branches.items():
                stage = protocol.probes[key if key == "new" else "old_" + history]
                reference = relation_world._probe(actor, stage)
                for a, b in (("correct", "success"), ("choice", "side"), ("tick", "tick"), ("anchor", "anchor")):
                    np.testing.assert_array_equal(outcomes[a], reference[b])
                probes += outcomes["correct"].size
            last = dose
        if not replay_base:
            n = protocol.history[history].menu.body.trials
            for index, expected in enumerate(traces["gate"].T):
                actual = actor.gate_trace.get(n + index)
                expected_mask = np.isfinite(expected)
                if actual is None:
                    if expected_mask.any():
                        raise ValueError("A cached gate write did not happen in closure")
                    continue
                amount, compatibility, mask = actual
                np.testing.assert_array_equal(expected_mask, mask)
                np.testing.assert_allclose(expected, amount, atol=2e-6, rtol=0.)
                np.testing.assert_allclose(traces["evidence"][:, index, 0], compatibility, atol=2e-6, rtol=0.)
                writes += int(mask.sum())
    return dict(probes=probes, writes=writes)


def pack(protocol, outcomes):
    data = dict(root=np.asarray(protocol.seeds), parity=protocol.parity)
    for arm, histories in outcomes.items():
        for history, (probes, traces, exposure) in histories.items():
            for key, values in exposure.items():
                k = f"{history}/{key}"
                if k in data and not np.array_equal(data[k], values):
                    raise ValueError(f"Arm changes the teaching contacts: {k}")
                data[k] = values
            for dose, metrics in probes.items():
                for metric, obs in metrics.items():
                    for key, values in obs.items():
                        data[f"{arm}/{history}/{dose}/{metric}/{key}"] = values
            for key, values in traces.items():
                data[f"{arm}/{history}/{key}"] = values
    return data


def test_birth(job):
    birth_id, selected, ids = job
    out = DATA / "holdout" / f"{birth_id}.npz"
    validation = DATA / "holdout" / f"{birth_id}-closure.json"
    if out.exists() and validation.exists():
        return
    start = perf_counter()
    front, motion, rule = resources(birth_id)
    arrays = {}
    for offset in range(0, len(ids), 32):
        protocol = make(ids[offset:offset + 32], "heldout")
        prepared = trust_replay.prepare(protocol, front, motion, rule, old=True)
        arms = {}
        gates = [Gate("always"), Gate("local"), Gate("constant", (selected["constant"],)),
                 Gate("explicit"), Gate("learned", tuple(selected["learned"]))]
        for gate in gates:
            arms[gate.mode] = trust_replay.evaluate(prepared, rule, motion, gate, record=gate.mode == "learned")
        schedules = {a: arms["learned"][b][1]["gate"]
                     for a, b in (("related", "unrelated"), ("unrelated", "related"))}
        arms["swapped"] = trust_replay.evaluate(prepared, rule, motion, gates[-1],
                                                 overrides=schedules, record=True)
        if offset == 0:
            checks = {gate.mode: verify_closure(protocol, front, motion, rule, gate, arms[gate.mode],
                                               replay_base=gate.mode in ("always", "local"))
                      for gate in (gates[0], gates[1], gates[-1])}
            checks["swapped"] = verify_closure(protocol, front, motion, rule, gates[-1], arms["swapped"],
                                                schedules=schedules)
            relation_study.write(validation, checks)
        for key, value in pack(protocol, arms).items():
            arrays.setdefault(key, []).append(value)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **{k: np.concatenate(v) for k, v in arrays.items()})
    temporary.replace(out)
    print(f"quarter-turn {birth_id} ({len(ids)} worlds, {len(ARMS)} arms) {perf_counter() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "seal", "test", "report"))
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    if args.phase == "fit":
        fit(args.workers)
    elif args.phase == "seal":
        seal(create=True)
    elif args.phase == "test":
        selected = seal()
        ids = root_ids(REG["roots"]["holdout"])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(test_birth, [(birth, selected, ids) for birth in BIRTHS]))
    else:
        seal()
        from .trust_analysis import report
        report()


if __name__ == "__main__":
    main()
