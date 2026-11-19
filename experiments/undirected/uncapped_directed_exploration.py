"""Continue directed evolution with reserved, same-world exploration contests."""

from __future__ import annotations

from collections import Counter, deque
import math
import random
from statistics import fmean

import torch

from . import uncapped_directed_breeding as breeding
from . import uncapped_island_recombination as recombination
from .uncapped_directed_reward_run import rank


MUTANTS = 3
CONTEST_WIDTH = 1 + MUTANTS
TEMPERATURE = .01
REFRESH_PERIOD = 4
REFRESH_PROBES = 16
candidate = breeding.candidate


def protocol(population, islands):
    """Describe the fixed offspring tradeoff in a serializable run protocol."""
    if islands != breeding.ISLANDS or population < 16 * islands or population % islands:
        raise ValueError("Exploration requires four equal islands with at least 16 candidates each")
    width = population // islands
    keep = width // 8
    slots = max(1, keep // 8)
    ordinary_kinds = Counter(recombination._schedule(width))
    ordinary_kinds["local"] -= slots * CONTEST_WIDTH
    return {
        "slots_per_island": slots,
        "ordinary_parents_per_island": keep - slots,
        "ordinary_lineage_cap": max(1, keep // 4),
        "ordinary_offspring_per_island": dict(ordinary_kinds),
        "reserved_evaluations_per_island": slots * CONTEST_WIDTH,
        "exact_retests_per_slot": 1,
        "local_mutants_per_slot": MUTANTS,
        "selection_temperature": TEMPERATURE,
        "selection_rule": "sample one mutant uniformly; on the current development tape accept with "
                          "min(1, exp((mutant_score - exact_parent_score) / temperature))",
        "initial_branch_rule": "take the strongest remaining candidate from each founder family "
                               "in rank order before giving any family another slot",
        "ordinary_breeding_rule": "all scored offspring are eligible for ordinary selection; "
                                  "rank-eligible protected parents may also mate in the ordinary pool, "
                                  "but their exact retest is reserved",
        "minimum_rounds_before_refresh": REFRESH_PERIOD,
        "refresh_period": REFRESH_PERIOD,
        "refresh_shortlist_per_island": 2 * keep,
        "refresh_structure_probes": REFRESH_PROBES,
        "refresh_rule": "stagger by island and slot; prefer an unrepresented founder family in the "
                        "rank shortlist, then maximize birth-arc mask distance from the outgoing branch; "
                        "defer if no probe has a new mask",
    }


def _seed(config, purpose, *parts):
    return random.Random(":".join(map(str, ("directed-exploration", purpose, config["seed"], *parts))))


def _diverse_start(ranked, size):
    families = {}
    for row in ranked:
        families.setdefault(row["lineage"], deque()).append(row)
    diverse = []
    while families and len(diverse) < size:
        for lineage, family in list(families.items()):
            diverse.append(family.popleft())
            if not family:
                del families[lineage]
            if len(diverse) == size:
                break
    return diverse


def _parents(config, generation, previous, settings):
    slots = settings["slots_per_island"]
    pools = previous["islands"]
    if len(pools) != config["islands"] or not all(pools):
        raise ValueError("Previous checkpoint must have a nonempty elite pool for every island")
    stored = previous.get("exploration")
    if stored is not None and len(stored) != len(pools):
        raise ValueError("Expected exploration state for each island")
    exact, mating, contests = [], [], []
    for island, pool in enumerate(pools):
        by_id = {row["id"]: row for row in pool}
        ranked = sorted(pool, key=rank, reverse=True)
        if stored is not None:
            branch = stored[island]
        else:
            # The reciprocal comparison checkpoint has no exploration state yet.
            seeds = _diverse_start(ranked, slots)
            branch = [{"slot": slot, "id": row["id"], "opened_generation": generation}
                      for slot, row in enumerate(seeds)]
        if (len(branch) != slots or [state["slot"] for state in branch] != list(range(slots))
                or len({state["id"] for state in branch}) != slots
                or any(state["id"] not in by_id or not 1 <= state["opened_generation"] <= generation
                       for state in branch)):
            raise ValueError(f"Invalid exploration branches in island {island}")
        contests.append([(state, by_id[state["id"]]) for state in branch])
        protected = {state["id"] for state in branch}
        exact.append([row for row in by_id.values() if row["id"] not in protected])
        if not exact[-1]:
            raise ValueError(f"No ordinary breeding parents in island {island}")
        ordinary_rank = _ordinary_elites(ranked, (), settings["ordinary_parents_per_island"],
                                         settings["ordinary_lineage_cap"])
        mating.append(exact[-1] + [row for row in ordinary_rank if row["id"] in protected])
    return exact, mating, contests


def plan(config, generation, previous, warm_founders=()):
    """Keep the candidate budget fixed and permit all islands to exchange donors."""
    settings = protocol(config["population"], config["islands"])
    islands = config["islands"]
    width = config["population"] // islands
    if generation < 1 or (generation != 1 and not previous):
        raise ValueError("A previous checkpoint is required after generation one")
    if not previous:
        # Reuse only the historic warm-slot mapping. Other founders draw independent seeds.
        entries = breeding.plan(config, generation, {}, warm_founders=warm_founders)
        seeds = _seed(config, "offspring", generation).sample(range(1, 2**31), len(entries))
        for row, seed in zip(entries, seeds, strict=True):
            row.update(directed=True, seed=seed)
            if row["kind"] == "founder":
                row.pop("paired_founder")
                row["lineage"] = row["id"]
        return entries

    ordinary_width = width - settings["reserved_evaluations_per_island"]
    schedules = []
    for island in range(islands):
        schedule = [kind for kind, count in settings["ordinary_offspring_per_island"].items()
                    for _ in range(count)]
        _seed(config, "schedule", generation, island).shuffle(schedule)
        schedules.append(schedule)
    exact, mating, contests = _parents(config, generation, previous, settings)
    exact_count = [0] * islands
    crosses = [{"intra": [], "inter": []} for _ in range(islands)]
    mates = [[other for other in range(islands) if other != island] for island in range(islands)]
    seeds = iter(_seed(config, "offspring", generation).sample(range(1, 2**31), config["population"]))
    entries = []
    for lane in range(width):
        for island in range(islands):
            identifier = f"g{generation-1:03d}-i{island:02d}-c{lane:04d}"
            seed = next(seeds)
            parent = donor = branch = donor_island = None
            if lane >= ordinary_width:
                slot, proposal = divmod(lane - ordinary_width, CONTEST_WIDTH)
                state, parent = contests[island][slot]
                kind = "elite" if proposal == 0 else "local"
                branch = {"slot": slot, "proposal": proposal, "opened_generation": state["opened_generation"]}
            else:
                kind = schedules[island][lane]
                pool = mating[island]
                chooser = random.Random(seed)
                if kind == "elite":
                    parent = exact[island][exact_count[island] % len(exact[island])]
                    exact_count[island] += 1
                elif kind == "intra" and len(pool) > 1:
                    parent, donor = chooser.sample(pool, 2)
                    donor_island = island
                elif kind != "founder":
                    parent = chooser.choice(pool)
                    if kind == "inter":
                        donor_island = chooser.choice(mates[island])
                        donor = chooser.choice(mating[donor_island])
                    elif kind == "intra":
                        kind = "local"
            entry = {
                "id": identifier, "island": island, "pair": island // 2, "kind": kind, "seed": seed,
                "directed": True, "lineage": parent["lineage"] if parent else identifier,
                "parent": parent["id"] if parent else None, "parent_island": island if parent else None,
                "donor": donor["id"] if donor else None, "donor_island": donor_island,
                "donor_lineage": donor["lineage"] if donor else None,
            }
            if branch is not None:
                entry["exploration"] = branch
            if donor is not None:
                entry["donor_rule_row"] = None
                crosses[island][kind].append(entry)
            entries.append(entry)

    for island, kinds in enumerate(crosses):
        for kind, group in kinds.items():
            rule_rng = _seed(config, "rule-crossover", generation, island, kind)
            for entry in rule_rng.sample(group, int(len(group) * breeding.RULE_DONOR_FRACTION)):
                entry["donor_rule_row"] = rule_rng.randrange(len(breeding.RULE_ROW_NAMES))
    return entries


def _admit(config, generation, island, slot, rows):
    contest = sorted(rows, key=lambda row: row["exploration"]["proposal"])
    if ([row["exploration"]["proposal"] for row in contest] != list(range(CONTEST_WIDTH))
            or [row["kind"] for row in contest] != ["elite"] + MUTANTS * ["local"]
            or len({(row["parent"], row["lineage"], row["exploration"]["opened_generation"])
                    for row in contest}) != 1):
        raise ValueError(f"Incomplete or mismatched exploration contest in island {island} slot {slot}")
    incumbent = contest[0]
    rng = _seed(config, "admit", generation, island, slot)
    proposal = contest[rng.randrange(1, CONTEST_WIDTH)]
    delta = proposal["selection_score"] - incumbent["selection_score"]
    if not math.isfinite(delta):
        raise ValueError("Exploration requires finite development scores")
    probability = math.exp(min(0., delta / TEMPERATURE))
    draw = rng.random()
    accepted = draw < probability
    return {
        "slot": slot, "id": (proposal if accepted else incumbent)["id"],
        "opened_generation": incumbent["exploration"]["opened_generation"],
        "parent": incumbent["parent"], "incumbent": incumbent["id"], "proposal": proposal["id"],
        "incumbent_score": incumbent["selection_score"], "proposal_score": proposal["selection_score"],
        "delta": delta, "probability": probability, "draw": draw,
        "accepted": accepted, "accepted_downhill": accepted and delta < 0.,
    }


def _refresh(states, ranked, island, generation, settings, genome_for):
    by_id = {row["id"]: row for row in ranked}
    shortlist = ranked[:settings["refresh_shortlist_per_island"]]
    for state in states:
        slot = state["slot"]
        if (generation - state["opened_generation"] + 1 < REFRESH_PERIOD
                or (generation - slot - island) % REFRESH_PERIOD):
            continue
        active = {item["id"] for item in states}
        families = {by_id[identifier]["lineage"] for identifier in active}
        outgoing = state["id"]
        active_masks = [genome_for(by_id[identifier])["birth_mask"] for identifier in sorted(active)]
        old_mask = genome_for(by_id[outgoing])["birth_mask"]
        probes, seen = [], set()
        for row in shortlist:
            if row["id"] not in active and row["lineage"] not in families and row["lineage"] not in seen:
                probes.append(row)
                seen.add(row["lineage"])
                if len(probes) == REFRESH_PROBES:
                    break
        # A collapsed lineage distribution can still contain competing birth graphs.
        chosen = None
        for group in (probes, [row for row in shortlist if row["id"] not in active][:REFRESH_PROBES]):
            best_distance = 0
            for row in group:
                mask = genome_for(row)["birth_mask"]
                if any(torch.equal(mask, other) for other in active_masks):
                    continue
                distance = int(torch.count_nonzero(mask != old_mask))
                if distance > best_distance:
                    chosen, best_distance = row, distance
            if chosen is not None:
                break
        if chosen is None:
            state["refresh"] = {"outgoing": outgoing, "incoming": None}
        else:
            state["refresh"] = {
                "outgoing": outgoing, "incoming": chosen["id"], "mask_distance": best_distance,
                "new_family": chosen["lineage"] not in families, "selection_score": chosen["selection_score"],
            }
            state["id"], state["opened_generation"] = chosen["id"], generation + 1


def _ordinary_elites(ranked, protected, keep, lineage_cap):
    protected_ids = {row["id"] for row in protected}
    exact_parents = {row["parent"] for row in protected if row["kind"] == "elite"}
    primary, deferred, counts = [], [], Counter()
    for row in ranked:
        if row["id"] in protected_ids:
            continue
        if row["kind"] == "elite":
            # Repeated exact retests on the same tape do not consume multiple breeding slots.
            if row["parent"] in exact_parents:
                continue
            exact_parents.add(row["parent"])
        if counts[row["lineage"]] < lineage_cap:
            primary.append(row)
            counts[row["lineage"]] += 1
        else:
            deferred.append(row)
        if len(primary) == keep:
            break
    selected = primary + deferred[:keep - len(primary)]
    if len(selected) != keep:
        raise ValueError("Not enough distinct candidates to fill ordinary parent slots")
    return selected


def elites(config, rows, previous, *, warm_genomes=None):
    """Retain ranked parents plus persistent explore branches and their decision evidence."""
    settings = protocol(config["population"], config["islands"])
    slots = settings["slots_per_island"]
    if len(rows) != config["population"]:
        raise ValueError("Selection requires the complete development generation")
    generation = int(rows[0]["id"].split("-", 1)[0][1:]) + 1
    pools, genomes, exploration = [], {}, []
    for island in range(config["islands"]):
        ranked = sorted((row for row in rows if row["island"] == island), key=rank, reverse=True)
        by_id = {row["id"]: row for row in ranked}
        cache = {}

        def genome_for(row):
            identifier = row["id"]
            if identifier not in cache:
                cache[identifier] = candidate(row, previous, config["nodes"], warm_genomes)
            return cache[identifier]

        if previous:
            groups = [[] for _ in range(slots)]
            for row in ranked:
                if "exploration" in row:
                    groups[row["exploration"]["slot"]].append(row)
            states = [_admit(config, generation, island, slot, group) for slot, group in enumerate(groups)]
            _refresh(states, ranked, island, generation, settings, genome_for)
        else:
            states = [{"slot": slot, "id": row["id"], "opened_generation": generation + 1}
                      for slot, row in enumerate(_diverse_start(ranked, slots))]
        protected = [by_id[state["id"]] for state in states]
        ordinary = _ordinary_elites(ranked, protected, settings["ordinary_parents_per_island"],
                                    settings["ordinary_lineage_cap"])
        pools.append(ordinary + protected)
        genomes.update({row["id"]: genome_for(row) for row in pools[-1]})
        exploration.append(states)

    records = [state for island in exploration for state in island]
    trials = [state for state in records if "accepted" in state]
    refreshes = [state["refresh"] for state in records if "refresh" in state]
    return {
        "islands": pools, "genomes": genomes, "exploration": exploration,
        "exploration_summary": {
            "slots": len(records), "contests": len(trials),
            "accepted": sum(state["accepted"] for state in trials),
            "accepted_downhill": sum(state["accepted_downhill"] for state in trials),
            "retained_downhill": sum(state["accepted_downhill"] and state["id"] == state["proposal"]
                                     for state in trials),
            "declined": sum(not state["accepted"] for state in trials),
            "mean_proposal_delta": fmean(state["delta"] for state in trials) if trials else None,
            "refreshed": sum(state["incoming"] is not None for state in refreshes),
            "refreshed_new_family": sum(state.get("new_family", False) for state in refreshes),
            "refresh_deferred": sum(state["incoming"] is None for state in refreshes),
        },
    }
