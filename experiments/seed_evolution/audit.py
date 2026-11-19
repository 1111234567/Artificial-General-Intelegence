"""Independent provenance, selection, and replay checks for seed evolution."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import math
from numbers import Integral
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments.learned_binding import audit as binding_audit, common, data
from experiments.learned_binding.common import SETTINGS as HISTORICAL_CE
from experiments.learned_binding.frozen_trace import FrozenTracePolicy, TRAINABLE_NAMES
from experiments.learned_binding.model import ENCODER_WIDTH


require = binding_audit.require
same_tree = binding_audit.same_tree
ARMS = ("evolution", "random")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/seed-evolution/2026-11-11-v1"
ARTIFACTS = {"initial.pt", "inner64.pt", "score.json", "training.json"}
CONDITIONS = ((0, "own"), (64, "own"), (512, "own"), (512, "teacher"),
              (512, "no_read"), (512, "Bdelay"))
EXOGENOUS_FIELDS = ("world_start_hash", "world_support_end_hash", "world_end_hash",
                    "support_exogenous_hash", "probe_exogenous_hash")


def conflict_fitness(streams):
    """Equal weight for A/B conflicts within each world and for every world."""
    require(bool(streams), "no scored worlds")
    require(len({stream["stream_id"] for stream in streams}) == len(streams),
            "repeated scored world")
    rates = []
    for stream in streams:
        family_rates = []
        for group in ("A_conflict", "B_conflict"):
            counts = stream["groups"][group]
            n, strict = counts["n"], counts["strict"]
            require(isinstance(n, Integral) and isinstance(strict, Integral)
                    and n > 0 and 0 <= strict <= n, "invalid " + group + " counts")
            family_rates.append(strict / n)
        fitness = sum(family_rates) / 2
        require(math.isclose(stream["fitness"], fitness, abs_tol=1e-12, rel_tol=0.0),
                "fitness does not match the world ledger")
        rates.append(fitness)
    return sum(rates) / len(rates)


def historical_stream_at(completed_updates):
    """The next stream after a given number of completed official CE updates."""
    dense = HISTORICAL_CE["dense_updates"]
    streams = HISTORICAL_CE["training_streams"]
    return (completed_updates % (streams // 2) if completed_updates < dense
            else (completed_updates - dense) % streams)


def trainable_parameters(model):
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    require(set(trainable) == TRAINABLE_NAMES, "wrong trainable parameter set")
    require(sum(parameter.numel() for parameter in trainable.values()) == 2258,
            "wrong trainable parameter count")
    return trainable


def make_adam(trainable, saved_optimizer=None, completed_updates=0):
    optimizer = torch.optim.Adam(trainable.values(), lr=HISTORICAL_CE["ce_lr"])
    if saved_optimizer is not None:
        optimizer.load_state_dict(deepcopy(saved_optimizer))
    require(len(optimizer.param_groups) == 1, "wrong Adam parameter groups")
    require(optimizer.param_groups[0]["lr"] == HISTORICAL_CE["ce_lr"], "wrong Adam learning rate")
    if completed_updates:
        require(len(optimizer.state) == len(trainable), "wrong Adam state coverage")
        require(all(float(value["step"]) == completed_updates
                    for value in optimizer.state.values()), "wrong Adam update counter")
    else:
        require(not optimizer.state, "initial Adam already has state")
    return optimizer


def independent_ce_update(model, optimizer, trainable, stream):
    """Run the original weighted action CE and clipped Adam from a clean state."""
    device = next(model.parameters()).device
    observations = stream["obs"].to(device)
    actions = stream["actions"].to(device)
    logits, _ = model(observations, model.initial_state(observations.shape[1]))
    weights = logits.new_ones(11)
    weights[0] = HISTORICAL_CE["wait_weight"]
    loss = F.cross_entropy(logits.reshape(-1, 11), actions.reshape(-1), weight=weights)
    require(bool(torch.isfinite(loss)), "nonfinite replay loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, parameter in model.named_parameters():
        if name in trainable:
            require(parameter.grad is not None and bool(parameter.grad.isfinite().all()),
                    "missing or nonfinite gradient: " + name)
        else:
            require(parameter.grad is None, "gradient reached frozen parameter: " + name)
    norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm)}


def checkpoint_model(checkpoint, device="cpu"):
    model = FrozenTracePolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def replay_checkpoint_update(before, after, stream, device="cpu"):
    step = before["updates"]
    require(after["updates"] == step + 1, "checkpoints are not consecutive")
    require(before["source_seed"] == after["source_seed"], "executor source changed")
    same_tree(before["model_config"], after["model_config"], "checkpoint model configuration")
    model = checkpoint_model(before, device).train()
    trainable = trainable_parameters(model)
    optimizer = make_adam(trainable, before["optimizer"], step)
    metrics = independent_ce_update(model, optimizer, trainable, stream)
    same_tree(model.state_dict(), after["model"], "independent CE/Adam parameters")
    same_tree(optimizer.state_dict(), after["optimizer"], "independent CE/Adam state")
    return dict(step=step + 1, model_and_adam_bitwise_equal=True,
                frozen_gradients_absent=True, **metrics)


def rank_ids(ids, candidates):
    """Rebuild the declared maximum-fitness, ascending-ID ranking."""
    return sorted(ids, key=lambda cid: (-candidates[cid]["fitness"], cid))


def audit_generation(row, prior_ids, new_ids, candidates, arm):
    prior_top_two = rank_ids(prior_ids, candidates)[:2]
    require(row["new_ids"] == new_ids, "unexpected or repeated new candidate IDs")
    require(row["top2_before"] == prior_top_two, "parent pool differs from prior top two")
    cumulative = [*prior_ids, *new_ids]
    require(row["evaluated_ids"] == cumulative, "wrong cumulative evaluation budget")
    for cid in new_ids:
        candidate = candidates[cid]
        if arm == "evolution" and prior_top_two:
            require(candidate["parent_id"] in prior_top_two, "child used ineligible parent")
        else:
            require(candidate["parent_id"] is None, "independent restart has a parent")
    winner = rank_ids(cumulative, candidates)[0]
    require(row["best_id"] == winner, "wrong cumulative winner")
    require(row["best_fitness"] == candidates[winner]["fitness"], "wrong winner fitness")
    return cumulative, winner


def checked_file(root, path, expected_hash):
    root = Path(root).resolve()
    file = (root / path).resolve()
    require(file.is_relative_to(root) and file.is_file(), "missing or out-of-root artifact: " + str(path))
    with file.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    require(actual_hash == expected_hash, "artifact SHA-256 changed: " + str(path))
    return file


def verify_sources(protocol):
    require(bool(protocol["source_hashes"]), "missing protocol source bindings")
    for path, digest in protocol["source_hashes"].items():
        checked_file(ROOT, path, digest)
    require(set(protocol["origins"]) == {"0", "1", "2"}, "executor origin coverage")
    for seed, reference in protocol["origins"].items():
        require((ROOT / reference["path"]).resolve() == common.source_checkpoint(int(seed)).resolve(),
                "unexpected frozen executor origin")
        checked_file(ROOT, reference["path"], reference["sha256"])
    reference = protocol["data_freeze_ref"]
    manifest_path = checked_file(ROOT, reference["path"], reference["sha256"])
    require(manifest_path == (ROOT / "data/learned-binding/v1/data-freeze.json").resolve(),
            "unexpected frozen CE data manifest")
    manifest = binding_audit.read_json(manifest_path)
    streams = {(row["seed"], row["stream"]): row for row in manifest["streams"]}
    expected = {(seed, index) for seed in HISTORICAL_CE["seeds"]
                for index in range(HISTORICAL_CE["training_streams"])}
    require(set(streams) == expected and len(manifest["streams"]) == len(expected),
            "incomplete or duplicate official training streams")
    for row in streams.values():
        checked_file(ROOT, row["path"], row["sha256"])
    return streams


def load_source_streams(streams, source_seed, stream_indices):
    loaded = {}
    for index in stream_indices:
        reference = streams[source_seed, index]
        source = binding_audit.load_pt(ROOT / reference["path"])
        identity = source["metadata"]
        require(identity["source_seed"] == source_seed and identity["stream_id"] == index
                and identity["split"] == "train", "training stream identity differs from manifest")
        loaded[index] = {name: source[name] for name in ("obs", "actions")}
    return loaded


def verify_scoring(score, source_seed, stream_ids, expected_model_hash, config, condition="own"):
    require(score["source_seed"] == source_seed, "evaluation used a different source executor")
    require(score["parameter_hash"] == expected_model_hash, "evaluation used different parameters")
    require([item["stream_id"] for item in score["streams"]] == list(stream_ids),
            "evaluation used different or reordered streams")
    for name in ("actors", "support_ticks", "probe_ticks", "random_seed"):
        require(score[name] == config[name], "evaluation setting changed: " + name)
    require(score["read_enabled"] is (condition != "no_read"), "read intervention changed")
    require(score["write_enabled"] is True, "writes disabled during evaluation")
    require(score["teacher_support"] is (condition == "teacher"), "teacher intervention changed")
    require(score["b_ticks"] == (config["b_ticks"] if condition == "Bdelay" else 0),
            "B delay changed")
    for stream, identity in zip(score["streams"], world_identities(
            source_seed, stream_ids, config["random_seed"], config["actors"])):
        require(stream["actor_ids"] == list(identity["actor_ids"]), "evaluation actors differ")
        require(stream["contexts"] == list(identity["contexts"]), "evaluation contexts differ")
        require(stream["private_mapping_hash"] == common.tensor_hash(identity["mappings"]),
                "evaluation private mapping differs")
        stream_seed = np.random.SeedSequence(
            [0x53454544, source_seed, config["random_seed"], stream["stream_id"]])
        require(stream["sampling_seed"] == int(stream_seed.generate_state(1, dtype=np.uint64)[0]),
                "evaluation sampling seed differs")
    fitness = conflict_fitness(score["streams"])
    require(math.isclose(score["fitness"], fitness, abs_tol=1e-12, rel_tol=0.0),
            "overall fitness does not match stream ledgers")
    return fitness


def world_identities(source_seed, stream_ids, random_seed, actors):
    identities = [data._identity(source_seed, stream_id, random_seed, "eval", actors)
                  for stream_id in stream_ids]
    worlds = {identity["world_seed"] for identity in identities}
    actor_ids = {actor for identity in identities for actor in identity["actor_ids"]}
    require(len(worlds) == len(stream_ids) and len(actor_ids) == len(stream_ids) * actors,
            "duplicate world seeds or actor identities")
    return identities


def candidate_id(arm, generation, index):
    return f"{arm}_g{generation:02d}_i{index:02d}"


def candidate_directory(root, rep, cid):
    return root / "search" / f"rep_{rep}" / cid


def regenerate_genome(source_seed, seed, executor):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = FrozenTracePolicy(executor=executor,
                                  original_mappings=common.mapping_table(source_seed, "related"))
    return {name: tensor for name, tensor in model.state_dict().items()
            if not name.startswith("executor.")}


def mutate_initial_genome(parent, mutation_seed, sigma):
    """Recreate each random coordinate and keep the historical priors intact."""
    child = {name: tensor.clone() for name, tensor in parent.items()}
    generator = torch.Generator(device="cpu").manual_seed(mutation_seed)
    for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        tensor = child["encoder." + name]
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        if name.startswith("bias"):
            noise[ENCODER_WIDTH:2 * ENCODER_WIDTH].zero_()
        tensor.add_(noise, alpha=sigma * (1 / math.sqrt(3 * ENCODER_WIDTH)))
    for head in ("query", "key", "value", "gate"):
        tensor = child[head + ".weight"]
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        tensor.add_(noise, alpha=sigma * .02)
    return child


def genome_hash(genome):
    return binding_audit.parameter_hash(dict(sorted(genome.items())))


def audit_training_report(training, seed, updates, genome, executor):
    meta = training["metadata"]
    encoder = {name.removeprefix("encoder."): tensor for name, tensor in genome.items()
               if name.startswith("encoder.")}
    expected = dict(source_seed=seed, steps=updates, device="cpu", genome_hash=genome_hash(genome),
                    executor_hash=binding_audit.parameter_hash(executor),
                    encoder_hash=binding_audit.parameter_hash(encoder),
                    stream_indices=[historical_stream_at(step) for step in range(updates)],
                    initial_state=dict(h=0, z=0, memory=0), reset_every_stream=True,
                    optimizer="Adam", fresh_optimizer=True, learning_rate=HISTORICAL_CE["ce_lr"],
                    wait_weight=HISTORICAL_CE["wait_weight"], grad_clip_norm=5.,
                    trainable_parameters=2258, encoder_unchanged=True, executor_unchanged=True,
                    tf32=False)
    same_tree(meta, expected, "official CE settings and schedule")
    log = training["log"]
    updates_logged = [row["update"] for row in log]
    require(updates_logged == sorted(set(updates_logged)) and updates_logged[0] == 1
            and updates_logged[-1] == updates, "incomplete or repeated training log steps")
    for row in log:
        require(row["stream"] == historical_stream_at(row["update"] - 1), "logged CE stream differs")
        require(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]), "nonfinite CE metrics")


def audit_frozen_checkpoint(checkpoint, seed, updates, genome, executor):
    require(checkpoint["source_seed"] == seed and checkpoint["updates"] == updates,
            "checkpoint source or update budget differs")
    state = checkpoint["model"]
    fixed_policy = {name: tensor for name, tensor in genome.items()
                    if name.startswith("encoder.") or name == "original_mappings" or not updates}
    same_tree({name: state[name] for name in fixed_policy}, fixed_policy, "frozen policy state")
    same_tree({name.removeprefix("executor."): tensor for name, tensor in state.items()
               if name.startswith("executor.")}, executor, "frozen source executor")
    optimizer = checkpoint["optimizer"]
    require(len(optimizer["param_groups"]) == 1
            and len(optimizer["param_groups"][0]["params"]) == len(TRAINABLE_NAMES),
            "optimizer parameter coverage differs")
    if updates:
        require(set(optimizer["state"]) == set(optimizer["param_groups"][0]["params"]),
                "optimizer state differs from parameter group")
        require(all(float(state["step"]) == updates for state in optimizer["state"].values()),
                "optimizer update count differs")
    else:
        require(not optimizer["state"], "optimizer is already trained at initialization")


def validate_protocol(protocol):
    config = protocol["config"]
    expected = dict(repeats=3, generations=4, population=4, parents=2, sigma=.15, inner_steps=64,
                    final_steps=512, source_seeds=[0, 1, 2], dev_streams=[1100, 1101],
                    test_streams=[2100, 2101], actors=16, support_ticks=2048, probe_ticks=2048,
                    random_seed=78111, b_ticks=4096, search_device="cpu", final_device="cpu",
                    dtype="float32", threads_per_worker=1)
    same_tree(config, expected, "predeclared experiment configuration")
    require(protocol["unique_candidates"] == 84 and protocol["budget_per_arm_per_repeat"] == 16
            and protocol["shared_first_generation"] is True, "predeclared search budget")
    sample = [dict(rep=rep, id=candidate_id(arm, generation, 0))
              for rep in config["source_seeds"]
              for arm, generation in (("shared", 0), ("evolution", 3), ("random", 3))]
    same_tree(protocol["audit_candidates"], sample, "predeclared candidate replay sample")
    return config


def audit_candidate(root, record, config, source_executor, genomes):
    rep, arm, generation, index = (record[key] for key in ("rep", "arm", "generation", "index"))
    cid = candidate_id(arm, generation, index)
    folder = candidate_directory(root, rep, cid)
    require(record["id"] == cid and record["source_seed"] == rep
            and record["sigma"] == config["sigma"], "candidate identity or mutation scale differs")
    same_tree(binding_audit.read_json(folder / "candidate.json"), record, "candidate ledger")
    require(set(record["artifacts"]) == ARTIFACTS, "incomplete candidate artifact bindings")
    files = {name: checked_file(folder, name, digest) for name, digest in record["artifacts"].items()}
    genome = binding_audit.load_pt(files["initial.pt"])
    require(genome_hash(genome) == record["genome_hash"], "candidate genome hash differs")
    if arm == "evolution":
        parent = record["parent_id"]
        require(parent in genomes, "parent is unavailable or from the same generation")
        expected_seed = 810000 + rep * 10000 + generation * 100 + index
        require(record["mutation_seed"] == expected_seed and record["init_seed"] is None,
                "mutation seed differs")
        expected_genome = mutate_initial_genome(genomes[parent], expected_seed, config["sigma"])
    else:
        expected_seed = 710000 + rep * 10000 + generation * 100 + index
        require(record["init_seed"] == expected_seed and record["mutation_seed"] is None
                and record["parent_selection_seed"] is None, "restart seed or parent differs")
        expected_genome = regenerate_genome(rep, expected_seed, source_executor)
    same_tree(genome, expected_genome, "candidate genome from initial parent or restart")
    checkpoint = binding_audit.load_pt(files["inner64.pt"])
    executor = source_executor.state_dict()
    audit_frozen_checkpoint(checkpoint, rep, config["inner_steps"], genome, executor)
    trained_hash = binding_audit.parameter_hash(checkpoint["model"])
    require(trained_hash == record["trained_hash"], "candidate trained model hash differs")
    audit_training_report(binding_audit.read_json(files["training.json"]), rep, config["inner_steps"],
                          genome, executor)
    score = binding_audit.read_json(files["score.json"])
    verify_scoring(score, rep, config["dev_streams"], trained_hash, config)
    require(score["fitness"] == record["fitness"], "candidate fitness differs from score")
    genomes[cid] = genome
    return [{key: stream[key] for key in EXOGENOUS_FIELDS} for stream in score["streams"]]


def audit_search(root, protocol, config, executors):
    search = binding_audit.read_json(root / "search.json")
    repeats = config["source_seeds"]
    require(set(search["records"]) == {str(rep) for rep in repeats}, "search repeat coverage")
    require(search["unique_candidates"] == protocol["unique_candidates"], "search candidate budget")
    history = {(row["rep"], row["arm"], row["generation"]): row for row in search["history"]}
    expected_history = {(rep, arm, g) for rep in repeats for arm in ARMS
                        for g in range(config["generations"])}
    require(set(history) == expected_history and len(search["history"]) == len(expected_history),
            "generation history is incomplete or duplicated")
    winners = []
    for rep in repeats:
        records = search["records"][str(rep)]
        genomes = {}
        dev_schedule = None
        cumulative = {arm: [] for arm in ARMS}
        for generation in range(config["generations"]):
            shared = generation == 0
            for arm in ARMS:
                new_ids = [candidate_id("shared" if shared else arm, generation, index)
                           for index in range(config["population"])]
                row = history[rep, arm, generation]
                cumulative[arm], best = audit_generation(row, cumulative[arm], new_ids, records, arm)
                parent_seed = 650000 + rep * 100 + generation
                selector = random.Random(parent_seed)
                for index, cid in enumerate(new_ids):
                    record = records[cid]
                    require((record["rep"], record["arm"], record["generation"], record["index"])
                            == (rep, "shared" if shared else arm, generation, index),
                            "candidate metadata differs from schedule")
                    if arm == "evolution" and not shared:
                        require(record["parent_selection_seed"] == parent_seed
                                and record["parent_id"] == selector.choice(row["top2_before"]),
                                "parent does not replay from the declared selection seed")
                    if cid not in genomes:
                        schedule = audit_candidate(root, record, config, executors[rep], genomes)
                        if dev_schedule is None:
                            dev_schedule = schedule
                        else:
                            same_tree(dev_schedule, schedule, "unpaired development exogenous schedule")
                if generation == config["generations"] - 1:
                    require(len(cumulative[arm]) == protocol["budget_per_arm_per_repeat"],
                            "unequal candidate budgets")
                    winners.append(dict(rep=rep, arm=arm, id=best))
        require(set(genomes) == set(records), "unused or missing candidates")
        folders = {path.name for path in (root / "search" / f"rep_{rep}").iterdir() if path.is_dir()}
        require(folders == set(genomes), "candidate folders differ from the ledger")
        require(len({row["genome_hash"] for row in records.values()}) == len(records),
                "duplicate initial genomes")
    same_tree(winners, search["winners"], "locked search winners")
    for generation in range(config["generations"]):
        saved = binding_audit.read_json(root / f"generation_{generation}.json")
        expected_rows = [row for row in search["history"] if row["generation"] <= generation]
        expected_records = {
            rep: {cid: record for cid, record in rows.items() if record["generation"] <= generation}
            for rep, rows in search["records"].items()
        }
        same_tree(saved, dict(generation=generation, history=expected_rows, records=expected_records),
                  "saved generation boundary")
    return winners


def scoring_options(config, stream_key):
    return dict(stream_ids=tuple(config[stream_key]), actors=config["actors"],
                support_ticks=config["support_ticks"], probe_ticks=config["probe_ticks"],
                random_seed=config["random_seed"])


def replay_candidate(job):
    """Use an independent training loop and repeat closed-loop development scoring."""
    from experiments.seed_evolution import core

    root, rep, cid, streams, config, retrain = job
    torch.set_num_threads(1)
    folder = candidate_directory(root, rep, cid)
    checkpoint = binding_audit.load_pt(folder / "inner64.pt")
    if retrain:
        genome = binding_audit.load_pt(folder / "initial.pt")
        model = core.model_from_genome(rep, genome).train()
        trainable = trainable_parameters(model)
        optimizer = make_adam(trainable)
        indices = [historical_stream_at(step) for step in range(config["inner_steps"])]
        source = load_source_streams(streams, rep, sorted(set(indices)))
        for index in indices:
            independent_ce_update(model, optimizer, trainable, source[index])
        same_tree(model.state_dict(), checkpoint["model"], "independent candidate CE parameters")
        same_tree(optimizer.state_dict(), checkpoint["optimizer"], "independent candidate Adam")
    replay = core.score_model(rep, checkpoint["model"], **scoring_options(config, "dev_streams"),
                              archive=retrain)
    if retrain:
        traces = replay.pop("archives")
        for stream in replay["streams"]:
            trace = traces[stream["stream_id"]]["probe"]
            recount = binding_audit.recount_probe(trace["report"], trace["arrays"], trace)
            same_tree(recount["groups"], stream["groups"], "candidate replay ledgers")
    same_tree(replay, binding_audit.read_json(folder / "score.json"), "development score replay")
    return dict(rep=rep, id=cid, independent_ce_updates=config["inner_steps"] if retrain else 0,
                dev_worlds=len(replay["streams"]), fitness=replay["fitness"])


def audit_support(block, read_enabled, teacher, model_hash, policy_replay):
    start, final = block["start"], block["final"]
    require(start["parameter_hash"] == final["parameter_hash"] == model_hash,
            "support used or changed unexpected parameters")
    ticks, actors, jobs = block["actions"].shape
    require(jobs == 2 and final["work"]["ticks"] - start["work"]["ticks"] == ticks,
            "support action or tick count differs")
    for key, active in (("sampled_action_rows", not teacher), ("teacher_action_rows", teacher)):
        require(final["work"][key] - start["work"][key] == (ticks * actors * jobs if active else 0),
                "support policy counts differ: " + key)
    if not read_enabled:
        require(bool(torch.all(block["read_probs"] == .25)), "support no-read was not enforced")
    if policy_replay:
        arrays = {name: block[name].numpy() for name in ("obs", "actions")}
        binding_audit.replay_world(arrays, block, teacher=teacher)
        if not teacher:
            with patch.object(binding_audit, "NativeMemoryPolicy", FrozenTracePolicy):
                subject = binding_audit.subject_from_snapshot(start)
                replay = subject.support(ticks, sample=True, read_enabled=read_enabled)
            same_tree(replay, block, "sampled support replay")


def audit_evaluation_archive(score, archives, config, policy_streams):
    expected_ids = config["test_streams"]
    require(set(archives) == set(expected_ids), "held-out archive coverage")
    condition = score["condition"]
    read_enabled = condition != "no_read"
    model_hash = score["parameter_hash"]
    identities = world_identities(score["source_seed"], expected_ids, config["random_seed"], config["actors"])
    replay_count = 0
    for stream, identity in zip(score["streams"], identities):
        stream_id = stream["stream_id"]
        trace = archives[stream_id]
        policy_replay = stream_id in policy_streams
        keys = {"support", "probe", "b_support"} if condition == "Bdelay" else {"support", "probe"}
        require(set(trace) == keys, "held-out phase coverage")
        support, probe = trace["support"], trace["probe"]
        start_world = support["start"]["world"]
        require(start_world["spec"]["seed"] == identity["world_seed"], "held-out world seed differs")
        require(list(start_world["spec"]["actor_ids"]) == stream["actor_ids"], "held-out actors differ")
        same_tree(start_world["random_mapping"]["mappings"], identity["mappings"], "held-out mappings")
        require(start_world["tick"] == -1 and support["start"]["pending_obs"] is None
                and all(not bool(tensor.any()) for tensor in support["start"]["state"].values()),
                "held-out support did not start fresh")
        require(len(support["actions"]) == config["support_ticks"], "wrong held-out support budget")
        require(len(probe["arrays"]["actions"]) == config["probe_ticks"], "wrong held-out probe budget")
        expected_schedule = dict(
            world_start_hash=start_world["event_hash"],
            world_support_end_hash=support["final"]["world"]["event_hash"],
            world_end_hash=probe["final"]["world"]["event_hash"],
            support_exogenous_hash=binding_audit._exogenous_hash(support["obs"].numpy()),
            probe_exogenous_hash=binding_audit._exogenous_hash(probe["arrays"]["obs"]),
        )
        same_tree({key: stream[key] for key in EXOGENOUS_FIELDS}, expected_schedule,
                  "held-out exogenous schedule record")
        audit_support(support, read_enabled, condition == "teacher", model_hash, policy_replay)
        if condition == "Bdelay":
            b_support = trace["b_support"]
            require(len(b_support["actions"]) == config["b_ticks"], "wrong B exposure budget")
            next_start = dict(support["final"], world=binding_audit._after_policy(
                support["final"]["world"], (stream["contexts"][1],)))
            same_tree(next_start, b_support["start"], "B exposure interrupted the live state")
            audit_support(b_support, read_enabled, False, model_hash, policy_replay)
            probe_start = dict(b_support["final"], world=binding_audit._after_policy(
                b_support["final"]["world"], stream["contexts"]))
        else:
            probe_start = support["final"]
        same_tree(probe_start, probe["start"], "held-out probe continuity")
        require(probe["report"]["read_enabled"] is read_enabled
                and probe["report"]["write_enabled"] is True
                and probe["final"]["parameter_hash"] == model_hash, "wrong probe intervention or parameters")
        if not read_enabled:
            require(bool(np.all(probe["arrays"]["read_probs"] == .25)), "probe no-read was not enforced")
        recount = binding_audit.recount_probe(probe["report"], probe["arrays"], probe,
                                             world_replay=policy_replay)
        same_tree(recount["groups"], stream["groups"], "held-out compact conflict ledger")
        for name in ("first_groups", "idle_opportunities", "idle_actions", "command_translations",
                     "correct_translations"):
            same_tree(probe["report"][name], stream["probe"][name], "held-out compact " + name)
        if policy_replay:
            with patch.object(binding_audit, "NativeMemoryPolicy", FrozenTracePolicy):
                binding_audit.replay_probe(probe["report"], probe["arrays"], probe)
            replay_count += 1
    return replay_count


def audit_final(job):
    root, row, completed, config, stream_manifest = job
    torch.set_num_threads(1)
    winner = row["winner"]
    rep, arm, cid = (winner[key] for key in ("rep", "arm", "id"))
    checkpoint_path = checked_file(ROOT, row["path"], row["sha256"])
    folder = root / "final" / f"rep_{rep}" / arm
    require(checkpoint_path == (folder / "training.pt").resolve(), "unexpected final training path")
    trained = binding_audit.load_pt(checkpoint_path)
    training_record = binding_audit.read_json(folder / "training.json")
    same_tree(training_record, dict(winner=winner, sha256=row["sha256"],
                                    metadata=trained["metadata"], log=trained["log"]),
              "final training record")
    final_freeze_time = (root / "final-freeze.json").stat().st_mtime_ns
    search_freeze_time = (root / "search.json").stat().st_mtime_ns
    require(search_freeze_time <= checkpoint_path.stat().st_mtime_ns <= final_freeze_time,
            "final training order differs from search and test barriers")
    candidate_folder = candidate_directory(root, rep, cid)
    genome = binding_audit.load_pt(candidate_folder / "initial.pt")
    executor = common.executor(rep).state_dict()
    audit_training_report(trained, rep, config["final_steps"], genome, executor)
    checkpoints = trained["checkpoints"]
    require(set(checkpoints) == {0, 1, 64, 511, 512}, "missing final replay checkpoints")
    for step, checkpoint in checkpoints.items():
        audit_frozen_checkpoint(checkpoint, rep, step, genome, executor)
    same_tree(checkpoints[64], binding_audit.load_pt(candidate_folder / "inner64.pt"),
              "search-to-final checkpoint at 64 updates")
    updates = (0, 511)
    sources = load_source_streams(stream_manifest, rep, [historical_stream_at(step) for step in updates])
    replayed_updates = []
    for step in updates:
        metrics = replay_checkpoint_update(checkpoints[step], checkpoints[step + 1],
                                           sources[historical_stream_at(step)])
        logged = next(row for row in trained["log"] if row["update"] == step + 1)
        for name in ("loss", "grad_norm"):
            require(metrics[name] == logged[name], "final replay metric differs: " + name)
        replayed_updates.append(metrics)

    evaluation_dir = folder / "evaluations"
    same_tree(binding_audit.read_json(evaluation_dir / "completion.json"), completed,
              "held-out completion record")
    names = [f"{step:03d}_{condition}" for step, condition in CONDITIONS]
    require([item["name"] for item in completed["conditions"]] == names,
            "held-out condition coverage differs")
    recounts = 0
    policy_replays = 0
    model_hashes = {}
    for (step, condition), item in zip(CONDITIONS, completed["conditions"]):
        score_path = checked_file(evaluation_dir, item["name"] + ".json", item["sha256"])
        score = binding_audit.read_json(score_path)
        archive_path = checked_file(evaluation_dir, item["name"] + ".pt", score["archives_sha256"])
        require(min(score_path.stat().st_mtime_ns, archive_path.stat().st_mtime_ns) >= final_freeze_time,
                "held-out artifacts predate final training freeze")
        require(score["winner"] == winner and score["step"] == step and score["condition"] == condition,
                "held-out artifact identity differs")
        model_hash = binding_audit.parameter_hash(checkpoints[step]["model"])
        model_hashes[step] = model_hash
        verify_scoring(score, rep, config["test_streams"], model_hash, config, condition)
        require(score["fitness"] == item["fitness"], "held-out summary fitness differs")
        archives = binding_audit.load_pt(archive_path)
        select_replay = step == 512 and (condition == "own" or (rep == 0 and arm == "evolution"))
        replay_streams = config["test_streams"][:1] if select_replay else []
        policy_replays += audit_evaluation_archive(score, archives, config, replay_streams)
        recounts += len(archives)
    return dict(winner=winner, replayed_ce_updates=replayed_updates,
                heldout_ledgers_recounted=recounts, heldout_policy_replays=policy_replays,
                model_hashes=model_hashes, fitness={item["name"]: item["fitness"]
                                                    for item in completed["conditions"]})


def audit_all(root=OUT, workers=9):
    root = Path(root).resolve()
    torch.set_num_threads(1)
    protocol_path = root / "protocol.json"
    protocol = binding_audit.read_json(protocol_path)
    config = validate_protocol(protocol)
    sources = verify_sources(protocol)
    executors = {rep: common.executor(rep) for rep in config["source_seeds"]}
    winners = audit_search(root, protocol, config, executors)
    freeze = binding_audit.read_json(root / "final-freeze.json")
    require(freeze["all_final_training_complete_before_test"] is True, "final training barrier absent")
    identities = [identity for rep in config["source_seeds"] for identity in world_identities(
        rep, [*config["dev_streams"], *config["test_streams"]], config["random_seed"], config["actors"])]
    require(len({identity["world_seed"] for identity in identities}) == len(identities),
            "development and held-out worlds overlap")
    require(len({actor for identity in identities for actor in identity["actor_ids"]})
            == len(identities) * config["actors"], "development and held-out actors overlap")
    completion = binding_audit.read_json(root / "test-completion.json")
    same_tree([row["winner"] for row in freeze["models"]], winners, "final trained winners")
    same_tree([row["winner"] for row in completion["models"]], winners, "held-out evaluated winners")

    sample = {(row["rep"], row["id"]) for row in protocol["audit_candidates"]}
    selected = {(row["rep"], row["id"]) for row in winners}
    candidate_jobs = [(root, rep, cid, sources, config, (rep, cid) in sample)
                      for rep, cid in sorted(sample | selected)]
    final_jobs = [(root, row, result, config, sources)
                  for row, result in zip(freeze["models"], completion["models"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        candidates = list(pool.map(replay_candidate, candidate_jobs))
        finals = list(pool.map(audit_final, final_jobs))

    shared = []
    for left, right in zip(finals[::2], finals[1::2]):
        if left["winner"]["id"] == right["winner"]["id"]:
            same_tree(left["model_hashes"], right["model_hashes"], "shared winner trained models")
            same_tree(left["fitness"], right["fitness"], "shared winner held-out scores")
            shared.append(left["winner"]["rep"])
    return dict(protocol_sha256=common.sha(protocol_path), audit_sha256=common.sha(__file__),
                candidates_provenance_checked=protocol["unique_candidates"],
                fixed_ce_streams=len(sources), development_replays=candidates,
                final_checks=finals, shared_winner_repeats=shared,
                heldout_ledgers_recounted=sum(row["heldout_ledgers_recounted"] for row in finals),
                heldout_policy_replays=sum(row["heldout_policy_replays"] for row in finals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    report = audit_all(args.root, workers=args.workers)
    common.write_json(args.root / "audit.json", report)
    print(f"Audited {report['candidates_provenance_checked']} candidates and "
          f"{report['heldout_ledgers_recounted']} held-out ledgers.")
