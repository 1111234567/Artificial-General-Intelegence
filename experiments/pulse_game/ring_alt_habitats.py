"""Long-life memory capacity with native bites in independently scented habitats."""

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .ring_alt_chemical import ChemSpec
from .ring_alt_chemical_forward import CohortForager
from .ring_alt_chemical_shift import Continue
from .ring_alt_run import preference_code
from .ring_world import CUE, DERANGED, FRUIT, NEW, PERMS, TASTED, make_tape, rollout
from .spike_analysis import estimates


WITHIN = .50
BETWEEN = .261
MAX_HABITATS = 8


def scents(seed, sense):
    """Draw arbitrary new mixtures, conditioning on resolvability alone."""
    rng = np.random.default_rng(np.random.SeedSequence((seed, sense, 4497351)))
    all_odor = np.empty((MAX_HABITATS, 4, 4), dtype=np.float32)
    flat = all_odor.reshape(-1, 4)
    for j in range(len(flat)):
        group = (j // 4) * 4
        for _ in range(24):
            trial = rng.gamma(.83, 1., (192, 4)).astype(np.float32)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            if j:
                distance = np.square(trial[:, None] - flat[None, :j]).sum(axis=2)
                radii = np.full(j, BETWEEN**2, dtype=np.float32)
                radii[group:] = WITHIN**2
                acceptable = (distance >= radii).all(axis=1)
                fits = np.flatnonzero(acceptable)
            else:
                fits = np.array([0])
            if len(fits):
                flat[j] = trial[fits[0]]
                break
        else:
            raise RuntimeError(f"Could not resolve 32 odors for seed {seed}")
    return all_odor[rng.permutation(MAX_HABITATS)]


def palette_collection(seeds):
    bundle = tuple(np.stack([scents(s, sense) for s in seeds]) for sense in (1, 2))
    audit = []
    group = np.arange(4 * MAX_HABITATS) // 4
    same = group[:, None] == group[None, :]
    compare = np.triu(np.ones(same.shape, dtype=bool), k=1)
    for matrix in bundle:
        f = matrix.reshape(-1, 4 * MAX_HABITATS, 4)
        dist = np.linalg.norm(f[:, :, None] - f[:, None, :], axis=-1)
        within = dist[:, same & compare].min(axis=1)
        cross = dist[:, ~same & compare].min(axis=1)
        assert min(within) >= WITHIN - 1e-6
        assert min(cross) >= BETWEEN - 1e-6
        audit.append({"within": within.tolist(), "cross": cross.tolist()})
    return bundle, audit


def _map_family(panel):
    maps = np.array([np.stack((a, a[d])) for a in PERMS for d in DERANGED])
    rows = np.random.default_rng(584421).permutation(len(maps))
    half = len(rows) // 2
    return maps[rows[:half] if panel == "development" else rows[half:]]


def hidden_preferences(seeds, panel):
    maps = _map_family(panel)
    chosen = [np.random.default_rng(np.random.SeedSequence((s, 67511))).integers(len(maps), size=MAX_HABITATS)
              for s in seeds]
    return maps[np.asarray(chosen)]


def permutations(seeds, n):
    return np.stack([np.random.default_rng(np.random.SeedSequence((s, 62651))).permutation(n) for s in seeds])


def catalog(seeds, length, world_seed_offset, preferences):
    """One fixed ordinary physical tape per life and habitat."""
    local = [make_tape([int(s) + world_seed_offset + h * 10_000_019 for s in seeds], length)
             for h in range(preferences.shape[1])]
    ctx, cue, pos, layout = [np.stack([getattr(t, k) for t in local], axis=1)
                             for k in ("ctx", "cue", "positions", "layout")]
    life = np.arange(len(seeds))[:, None, None]
    hab = np.arange(len(local))[None, :, None]
    goals = preferences[life, hab, ctx, cue]
    return (ctx, cue, pos, layout, goals)


def gather(seeds, maps, cat, order, *, offset):
    """Permute identical habitat-tagged physical trials without changing content."""
    slots = np.zeros_like(order)
    for h in range(cat[0].shape[1]):
        mask = order == h
        slots[mask] = ((mask.cumsum(axis=1) - 1)[mask] + offset)
    bodies = np.arange(len(seeds))[:, None]
    fields = [column[bodies, order, slots] for column in cat]
    # The simulator uses goal for scoring. The multi-habitat map is held
    # outside the actor, and this unused per-life field keeps the native shape.
    base = make_tape(seeds, 1)
    return replace(base, ctx=fields[0], cue=fields[1], positions=fields[2], layout=fields[3],
                   preferences=maps[:, 0], goal=fields[4])


class Exposure:
    def __init__(self, agent, mixtures, habitats, *, resume=False):
        self.agent, self.mixtures, self.habitats = agent, mixtures, habitats
        self.resume = resume

    def reset(self, n):
        self.reader = Continue(self.agent) if self.resume else self.agent
        self.reader.reset(n)
        self.t = -1
        self.rows = np.arange(n)

    def act(self, obs):
        if obs[0, NEW]:
            self.t += 1
            current = self.habitats[:, self.t]
            self.cue_palette = self.mixtures[0][self.rows, current]
            self.fruit_palette = self.mixtures[1][self.rows, current]
        x = obs.copy()
        for sel, bank in ((CUE, self.cue_palette), (FRUIT, self.fruit_palette), (TASTED, self.fruit_palette)):
            x[:, sel] = np.einsum("bi,bij->bj", obs[:, sel], bank, optimize=False)
        return self.reader.act(x)


def new_actor(cfg, seeds, birth, horizon, rule):
    action = [np.random.SeedSequence((s, 0x4348454D)) for s in seeds]
    return CohortForager(cfg, birth_seed=birth, action_seeds=action, rule=rule, horizon=horizon)


def quiet_test(actor, tape, scents_, order):
    copy_ = copy.deepcopy(actor)
    copy_.adapt_association = copy_.adapt_receptors = False
    result = rollout(Exposure(copy_, scents_, order, resume=True), tape, feedback="none")
    return result.first_correct, result.bites > 0


def by_rank(hit, order, ranks):
    return np.stack([(hit & (order == ranks[:, i, None])).sum(axis=1) /
                     (order == ranks[:, i, None]).sum(axis=1) for i in range(ranks.shape[1])], axis=1)


def resource_trace(actor):
    state = actor.stats()
    return {"fruit_alive": state["fruit"]["active"], "cue_alive": state["cue"]["active"],
            "fruit_retired": state["fruit"]["retirements"], "cue_retired": state["cue"]["retirements"],
            "fruit_support": state["fruit"]["participation"], "cue_support": state["cue"]["participation"],
            "weight_writes": state["weight_updates"], "scope_sizes": {
                "prior_covariance": actor.p[0].size if actor.p is not None else 0,
                "post_covariance": actor.fruit_p[0].size if actor.fruit_p is not None else 0,
                "association_weights": actor.weights[0].size,
                "odor_centers": actor.fruit.prototype[0].size + actor.cue.prototype[0].size}}


def one_config(seeds, cfg, palettes, preferences, catalog_train, catalog_probe, protocol, k, capacity, birth, rule):
    intro, practice, checks = (protocol[x] for x in ("intro", "rehearsal", "probe"))
    active = palettes[0][:, :k], palettes[1][:, :k]
    order = permutations(seeds, k)
    intro_order = np.repeat(order, intro, axis=1)
    intro_tape = gather(seeds, preferences, catalog_train, intro_order, offset=0)
    probe_order = np.tile(order[:, ::-1], (1, checks))
    probe_tape = gather(seeds, preferences, catalog_probe, probe_order, offset=0)
    spec = replace(cfg, capacity=capacity, cue_capacity=capacity)
    individual = new_actor(spec, seeds, birth, protocol["horizon"], rule)
    started = perf_counter()
    base = rollout(Exposure(individual, active, intro_order), intro_tape)
    initiation = base.first_correct.reshape(len(seeds), k, intro)
    old, seen = quiet_test(individual, probe_tape, active, probe_order)
    initial = by_rank(old, probe_order, order)
    phase_resource = resource_trace(individual)
    result = []
    for label, review in (("blocked", np.repeat(order, practice, axis=1)), ("spaced", np.tile(order, (1, practice)))):
        model = copy.deepcopy(individual)
        exposure = gather(seeds, preferences, catalog_train, review, offset=intro)
        live = rollout(Exposure(model, active, review, resume=True), exposure)
        final, touched = quiet_test(model, probe_tape, active, probe_order)
        done = by_rank(final, probe_order, order)
        test_coverage = by_rank(touched, probe_order, order)
        result.append({"schedule": label, "birth": birth, "rule": rule, "capacity": capacity, "habitats": k,
                       "intro_first8": initiation[:, :, :8].mean(2).tolist(), "intro_last8": initiation[:, :, -8:].mean(2).tolist(),
                       "quiet_before_rehearsal": initial.tolist(), "quiet_after_rehearsal": done.tolist(),
                       "quiet_coverage": test_coverage.tolist(), "initial_coverage": by_rank(seen, probe_order, order).tolist(),
                       "review_actual": by_rank(live.first_correct, review, order).tolist(),
                       "resource_before_rehearsal": phase_resource, "resource_after_rehearsal": resource_trace(model),
                       "orders": order.tolist()})
        print(f"{birth:3d} K{k} C{capacity:2d} {rule:8s} {label:7s} quiet {initial.mean():.3f}→{done.mean():.3f}; "
              f"old {done[:,:-1].mean():.3f}, recent {done[:,-1].mean():.3f}; {perf_counter()-started:.1f}s", flush=True)
    return result


def summarize(rows):
    results = {}
    for k, capacity, rule in dict.fromkeys((r["habitats"], r["capacity"], r["rule"]) for r in rows):
        segment = [r for r in rows if r["habitats"] == k and r["capacity"] == capacity and r["rule"] == rule]
        a, b = [np.mean([r["quiet_after_rehearsal"] for r in segment if r["schedule"] == s], axis=0)
                for s in ("blocked", "spaced")]
        before = np.mean([r["quiet_before_rehearsal"] for r in segment if r["schedule"] == "blocked"], axis=0)
        out = {"quiet_baseline_all": estimates(before.mean(1)),
               "intro_first8": estimates(np.mean([r["intro_first8"] for r in segment], axis=(0, 2))),
               "intro_last8": estimates(np.mean([r["intro_last8"] for r in segment], axis=(0, 2)))}
        for name, v in (("blocked", a), ("spaced", b), ("spaced_minus_blocked", b - a)):
            out[name] = {"all": estimates(v.mean(1)), "old": estimates(v[:, :-1].mean(1)),
                         "most_recent": estimates(v[:, -1]), "by_rank": [estimates(v[:, j]) for j in range(k)]}
            if name != "spaced_minus_blocked":
                out[name]["retention_gain_from_first_assay"] = estimates((v - before).mean(1))
                group = [r["resource_after_rehearsal"] for r in segment if r["schedule"] == name]
                for key in ("fruit_alive", "cue_alive", "fruit_retired", "cue_retired", "fruit_support"):
                    out[name][key] = estimates(np.mean([row[key] for row in group], axis=0))
        results[f"{k}/{capacity}/{rule}"] = out
    return results


def sensory_alias_control(cfg, seeds, palettes, maps, protocol, births):
    """Alter only the two private outcomes under an identical quiet tape."""
    copies = tuple(np.repeat(bank[:, :1], 2, axis=1) for bank in palettes)
    conflicts = np.stack((maps[:, 0], (maps[:, 0] + 1) % 4), axis=1)
    length = protocol["intro"] + protocol["rehearsal"]
    ct = catalog(seeds, length, 691013, conflicts)
    nh = len(seeds)
    unknown = np.stack([np.random.default_rng(np.random.SeedSequence((s, 34918))).permutation(
        np.repeat(np.arange(2), length)) for s in seeds])
    train = gather(seeds, conflicts, ct, unknown, offset=0)
    shared = make_tape([s + 691019021 for s in seeds], protocol["probe"] * 2)
    goals = [conflicts[:, h][np.arange(nh)[:, None], shared.ctx, shared.cue] for h in range(2)]
    selected = []
    for birth in births:
        actor = new_actor(replace(cfg, capacity=8, cue_capacity=8), seeds, birth, protocol["horizon"], "age")
        rollout(Exposure(actor, copies, unknown), train)
        records = []
        for h in range(2):
            held = copy.deepcopy(actor)
            held.adapt_association = held.adapt_receptors = False
            alias = np.full_like(shared.cue, h)
            records.append(rollout(Exposure(held, copies, alias, resume=True), replace(shared, goal=goals[h]), feedback="none"))
        np.testing.assert_array_equal(records[0].first, records[1].first)
        np.testing.assert_array_equal(records[0].first_tick, records[1].first_tick)
        np.testing.assert_array_equal(records[0].motion, records[1].motion)
        score = .5 * (records[0].first_correct.astype(np.float32) + records[1].first_correct)
        assert np.max(score) <= .5
        selected.append({"birth": birth, "both_average": score.mean(1).tolist(),
                         "each": [r.first_correct.mean(1).tolist() for r in records]})
    return {"alternate_hidden_targets_changed_only": True, "physical_trajectories_identical": True, "bound": .5,
            "rows": selected, "grade": estimates(np.mean([r["both_average"] for r in selected], axis=0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frozen", type=Path)
    p.add_argument("protocol", type=Path)
    p.add_argument("artifact", type=Path)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = ChemSpec(**json.loads(args.frozen.read_text(encoding="utf-8"))["spec"])
    seeds = tuple(range(protocol["start"], protocol["start"] + protocol["worlds"]))
    palettes, separation = palette_collection(seeds)
    hidden = hidden_preferences(seeds, protocol["panel"])
    all_trials = protocol["intro"] + protocol["rehearsal"]
    train = catalog(seeds, all_trials, 568219, hidden)
    probe = catalog(seeds, protocol["probe"], 317906843, hidden)
    output = {"protocol": protocol, "world_seeds": seeds, "sensory_separation": separation,
              "hidden_map_codes": preference_code(hidden.reshape(-1, 2, 4)).reshape(len(seeds), -1).tolist(), "rows": []}
    for k, capacity, rule in protocol["conditions"]:
        for birth in protocol["births"]:
            rows = one_config(seeds, cfg, palettes, hidden, train, probe, protocol, k, capacity, birth, rule)
            output["rows"].extend(rows)
        # Durable aggregate after each complete condition.
        output["summary"] = summarize(output["rows"])
        args.artifact.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    if protocol.get("run_alias", True):
        output["sensory_alias_control"] = sensory_alias_control(cfg, seeds, palettes, hidden, protocol, protocol["births"])
    args.artifact.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
