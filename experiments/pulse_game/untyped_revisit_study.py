"""Select role features by actual physical receipts and audit independent worlds."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from . import adaptive_local, adaptive_local_gpu as gpu, research_audit as audit
from . import symmetry_world as geometry, untyped_revisit as model
from . import untyped_senses as sensors, untyped_senses_study as v1, unified_confirm
from .attractor_search import OUT


HOME = audit.OUT / "untyped_revisit"
BIRTHS = OUT / "independent-lifelong-births-v1"
FEATURES = "short", "negative_squared_distance", "slow", "prototype"
CONDITIONS = ("standard", "prolonged", "noise", "side_perturbed", "side_heavy", "matched_jitter", "unidentifiable")
EXTERNAL_WORLD_SEED = 613087219
EXTERNAL_NONCE = 583617279


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


def roots(seed, n):
    return (np.random.default_rng(seed).choice(465_000_000, size=n, replace=False) + 7_000_000).tolist()


def episode(seeds, label, nonce):
    standard = "standard" if label.startswith("side_") or label == "matched_jitter" else label
    ep, packets = v1.episode(seeds, standard, nonce)
    if label == "matched_jitter":
        changed = packets.values.copy()
        # Independent public receptor errors of the same scale in both packets.
        for i, seed in enumerate(ep.arena.menu.body.seeds):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 129777481]))
            changed[i] = np.clip(changed[i] + .20 * rng.normal(size=changed[i].shape), -1., 1.)
        return ep, replace(packets, values=changed)
    if not label.startswith("side_"):
        return ep, packets
    noise = {"side_perturbed": .18, "side_heavy": .40}[label]
    changed = packets.values.copy()
    j = np.arange(changed.shape[1])
    for i, seed in enumerate(ep.arena.menu.body.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 803607211]))
        ix = packets.truth[i]
        changed[i, j, ix] = geometry.unit(changed[i, j, ix] + noise * rng.normal(size=changed.shape[1:2] + (2, 4)))
    return ep, replace(packets, values=changed)


def play(ep, raw, phenotype, alpha, weights, *, oracle="none", dot=False):
    f, route, motion, body, _ = phenotype
    view, roles, slots, overflow = model.prepare(ep, raw, weights, oracle=oracle, dot=dot)
    real = gpu.Cohort(view, f, route, motion).play(body, alpha)
    return real, roles, slots, overflow


def candidates():
    return {
        "fast": [[0, 1, 0, 0], [1, 0, 0, 0], *[[1, w, 0, 0] for w in (.25, .5, 1, 2, 4, 8)]],
        "slow": [[1, d, s, 0] for d in (.5, 1, 2, 4) for s in (1, 4)] + [[0, 2, 1, 0], [0, 4, 1, 0]],
        "proto": [[1, d, 0, p] for d in (.5, 1, 2, 4) for p in (.3, 1)] + [[0, 2, 0, 1], [0, 4, 0, 1]],
        "both": [[1, d, s, p] for d in (1, 2, 4) for s, p in ((1, .3), (2, .6), (4, 1.2))],
    }


def select(out, registrations, models, args):
    alpha = registrations["reopening_rule"]
    pool = candidates()
    search = [(key, a) for key, w in pool.items() for a in w]
    screens = []
    dev = [("standard", 17904755), ("side_heavy", 146388736)]
    for j, (name, se) in enumerate(dev):
        ep, pk = episode(roots(se, args.dev), name, 426097764)
        vals = [v1.correct_actual(play(ep, pk, models[j], alpha, w)[0], ep) for _, w in search]
        screens.append(vals)
        print("screen", name, round(float(max(vals)), 4), flush=True)
    aggregate = np.mean(screens, axis=0)
    shortlist = []
    for family in pool:
        choices = [i for i, (key, _) in enumerate(search) if key == family]
        shortlist.extend(sorted(choices, key=lambda i: (-aggregate[i], i))[:3])
    rerank = []
    dev2 = [("standard", 293072712), ("side_perturbed", 160547761), ("side_heavy", 192321285),
            ("prolonged", 218067693), ("matched_jitter", 138449457)]
    for j, (name, se) in enumerate(dev2):
        ep, pk = episode(roots(se, args.rerank), name, 157838707)
        vals = [v1.correct_actual(play(ep, pk, models[j % len(models)], alpha, search[ix][1])[0], ep)
                for ix in shortlist]
        rerank.append(vals)
        print("rerank", name, round(float(max(vals)), 4), flush=True)
    aggregate = np.mean(rerank, axis=0)
    rank = sorted(range(len(shortlist)), key=lambda i: (-aggregate[i], shortlist[i]))
    best = shortlist[rank[0]]
    winners = {k: next(search[shortlist[i]][1] for i in rank if search[shortlist[i]][0] == k) for k in pool}
    result = {"features": FEATURES, "screen_config": dev, "rerank_config": dev2, "screen_roots": args.dev,
              "rerank_roots": args.rerank, "search": search, "screen_actual_rewards": screens,
              "shortlist": shortlist, "rerank_actual_rewards": rerank, "winners": winners,
              "selected": search[best], "source_sha256": audit.fingerprint(Path(model.__file__)),
              "score": "v1.correct_actual actual nonzero reached route rewards, equal condition weight",
              "development_births": registrations["fixed_birth_seeds"][:3]}
    save(out / "selected.json", result)
    print("selected", search[best], flush=True)
    return result


def role_rows(ep, packets, outputs, n):
    real, roles, slots, overflow = outputs
    actual = real.reward
    got = roles == packets.truth
    a = ep.checks["return_A"]
    b = ep.checks[f"B_after_{ep.steps[-1]}"]
    start = ep.arena.start_test
    counts = (slots[:, :start, None] == np.arange(6)[None, None, :]).sum(1)
    preferred = counts.argmax(1)
    data = unified_confirm.data(ep, real, n)
    fields = {
        "all_roles": got.mean(1), "initial_A_training_roles": got[:, :start].mean(1),
        "first_four_roles": got[:, :4].mean(1), "middle_B_roles": got[:, start:a-1].mean(1),
        "return_A_anchor_role": got[:, a-1], "return_A_query_role": got[:, a],
        "return_A_both_roles": got[:, a-1] & got[:, a], "return_A_anchor_matches_initial_slot": slots[:, a-1] == preferred,
        "return_A_query_matches_initial_slot": slots[:, a] == preferred, "initial_A_most_used_slot_share": counts.max(1) / start,
        "B_final_query_role": got[:, b], "actual_return_A_positive": actual[:, a] > 0,
        "actual_return_A_contact": real.contact_side[:, a] >= 0, "slot_overflow": overflow,
        "actual_B_final_positive": actual[:, b] > 0, "actual_both_positive": (actual[:, a] > 0) & (actual[:, b] > 0),
    }
    data.update({k: v.reshape(4, n).tolist() for k, v in fields.items()})
    return data


def verify_native(seeds, name, f, weights, alpha):
    ep, pk = episode(seeds, name, EXTERNAL_NONCE)
    exp, roles, _, _ = play(ep, pk, f, alpha, weights)
    front, route, motion = f[-1]
    underlying = adaptive_local.LocalAdaptive(front, {**route, "release": alpha}, seeds * 4, motor=motion)
    agent = model.Live(underlying, weights)
    agent.reset(len(seeds) * 4)
    body = sensors.RawBody(ep, pk)
    obs = body.observe()
    while obs is not None:
        if obs[:, :2].any():
            raise ValueError("Typed context in the raw live body")
        obs = body.step(agent.act(obs))
    native = body.result()
    return {key: {"different": int((x != y).sum()), "elements": int(x.size)} for key, x, y in
            (("packet_roles", np.stack(agent.role, 1), roles), ("side", native.contact_side, exp.contact_side),
             ("tick", native.contact_tick, exp.contact_tick), ("reward", native.reward, exp.reward))}


def confirm(out, registrations, models, chosen, args):
    if chosen["source_sha256"] != audit.fingerprint(Path(model.__file__)):
        raise ValueError("The receptor source changed since selection")
    alpha = registrations["reopening_rule"]
    n = args.worlds
    seeds = roots(EXTERNAL_WORLD_SEED, n)
    methods = {
        "v1_short": ([1, 0, 0, 0], "none", False), "v1_dot": ([0, 1, 0, 0], "none", True),
        "squared_only": ([0, 1, 0, 0], "none", False),
        "selected": (chosen["selected"][1], "none", False),
        "known_roles": ([0, 0, 0, 0], "all", False),
    }
    for name, weight in chosen["winners"].items():
        methods[f"fit_{name}"] = (weight, "none", False)
    for key in ("initial_A", "return_anchor", "return_query", "return_pair"):
        methods[f"v1_fix_{key}"] = ([1, 0, 0, 0], key, False)
    for birth, phenotype in models.items():
        for case in CONDITIONS:
            ep, pk = episode(seeds, case, EXTERNAL_NONCE)
            rows = {}
            cached = {}
            for method, (w, oracle, dot) in methods.items():
                if oracle not in {"none", "all"} and case not in {"standard", "side_heavy"}:
                    continue
                ident = tuple(w), oracle, dot
                if ident not in cached:
                    cached[ident] = role_rows(ep, pk, play(ep, pk, phenotype, alpha, w, oracle=oracle, dot=dot), n)
                rows[method] = cached[ident]
            parity = None
            if case in ("standard", "side_heavy"):
                parity = verify_native(seeds[:8], case, phenotype, chosen["selected"][1], alpha)
            item = {"seeds": seeds, "birth": birth, "case": case, "rows": rows, "real_port_parity": parity,
                    "method_config": methods, "nonce": EXTERNAL_NONCE}
            save(out / f"external-{birth}-{case}.json", item)
            print(birth, case, {i: round(float(np.asarray(rows[i]["return_A"])[[1, 2]].mean()), 3)
                                for i in ("v1_short", "selected", "known_roles")}, flush=True)
    save(out / "complete.json", {"births": list(models), "worlds": n, "conditions": CONDITIONS})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=["discover", "confirm"], required=True)
    p.add_argument("--out", type=Path, default=HOME)
    p.add_argument("--dev", type=int, default=32)
    p.add_argument("--rerank", type=int, default=48)
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--birth-indices", type=int, nargs="+", default=[0, 1, 3, 4])
    args = p.parse_args()
    torch.set_num_threads(1)
    reg = json.loads((BIRTHS / "registered.json").read_text())
    s = reg["fixed_birth_seeds"]
    if args.part == "discover":
        parents = [v1.organism(BIRTHS / f"birth-{i}") for i in s[:3]]
        select(args.out, reg, parents, args)
    else:
        picked = json.loads((args.out / "selected.json").read_text())
        parents = {s[i]: v1.organism(BIRTHS / f"birth-{s[i]}") for i in args.birth_indices}
        confirm(args.out, reg, parents, picked, args)


if __name__ == "__main__":
    main()
