"""Register, refit, and score the 96-node archived logic transfer."""

from argparse import ArgumentParser
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import archival_logic_model as model, ring_logic_study as shared, ring_logic_world as env


REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "data" / "pulse-game" / "archival-logic-20261119"
ARCHIVE = REPO / "data" / "undirected" / "2026-11-17-plasticity-directed-reward-v1" / "champion.pt"
ARCHIVE_SHA = "a963dd3f6c050c0a38718e75d784cbe16cc4e8c591de90b2e0caa80724eb0212"
ARCHIVE_ID = "g2047-i01-c0910"
WIDTH, EDGES = 96, 1218
STARTS = 12
ROOTS = {"train": range(67252000, 67252192), "standard": range(67254000, 67254128),
         "long": range(67255000, 67255128)}
STEPS = (240, 1200)
RATE = .024
WARMUP = 7
BOOTSTRAP_DRAWS = 50_000
SOURCES = ("archival_logic_model.py", "archival_logic_study.py", "archival_logic_analysis.py",
           "ring_logic_model.py", "ring_logic_study.py", "ring_logic_world.py", "world.py")


def source_hashes():
    return {name: shared.sha(Path(__file__).with_name(name)) for name in SOURCES}


def edges_hash(support):
    return hashlib.sha256(support.astype(np.uint8).tobytes()).hexdigest()


def load_support(path=ARCHIVE):
    if shared.sha(path) != ARCHIVE_SHA:
        raise RuntimeError("The archived winner changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    genome = saved["genome"]
    if saved["row"]["id"] != ARCHIVE_ID or not bool(genome["directed"]):
        raise RuntimeError("Unexpected archived graph identity or representation")
    u, v = torch.triu_indices(WIDTH, WIDTH, 1)
    source, target = torch.cat((v, u)), torch.cat((u, v))
    active = genome["birth_mask"] & genome["birth_w"].ne(0)
    support = np.zeros((WIDTH, WIDTH), dtype=bool)
    support[target[active].numpy(), source[active].numpy()] = True
    if int(support.sum()) != EDGES or not strongly_connected(support):
        raise RuntimeError("Unexpected archived graph invariants")
    return support


def strongly_connected(support):
    """Two reachability searches certify a single SCC including every node."""
    for graph in (support, support.T):
        seen = {0}
        todo = [0]
        while todo:
            node = todo.pop()
            neighbors = set(np.flatnonzero(graph[node])) - seen
            seen.update(neighbors)
            todo.extend(neighbors)
        if len(seen) != len(graph):
            return False
    return True


def switched(support, seed):
    """Shuffle destinations of pairs of arcs in a symmetric proposal chain."""
    rng = random.Random(int(seed))
    original = set(zip(*np.nonzero(support), strict=True))
    pool = sorted(original)
    occupied = set(pool)
    arc_count = len(pool)
    attempts = 40 * arc_count
    accepted = 0
    overlap_trace = []
    for trial in range(1, attempts + 1):
        i, j = rng.randrange(arc_count), rng.randrange(arc_count)
        edge_a, edge_b = pool[i], pool[j]
        target_a, source_a = edge_a
        target_b, source_b = edge_b
        new_a, new_b = (target_b, source_a), (target_a, source_b)
        if source_a != target_b and source_b != target_a and new_a not in occupied and new_b not in occupied:
            occupied.remove(edge_a)
            occupied.remove(edge_b)
            occupied.update((new_a, new_b))
            pool[i], pool[j] = new_a, new_b
            accepted += 1
        if trial % (10 * arc_count) == 0:
            overlap_trace.append({"proposals": trial, "accepted": accepted,
                                  "common_with_archive": len(occupied & original)})
    new = np.zeros_like(support)
    target, source = zip(*pool, strict=True)
    new[target, source] = True
    if accepted < 20 * arc_count or not strongly_connected(new):
        raise RuntimeError(f"Control did not satisfy mixing budget or SCC: {seed}")
    for axis in (0, 1):
        np.testing.assert_array_equal(new.sum(axis=axis), support.sum(axis=axis))
    if new.trace() != support.trace():
        raise RuntimeError("Unexpected change in self arcs")
    return new, overlap_trace


def graph_record(support):
    return {"sha256_uint8_target_source": edges_hash(support),
            "sources_by_target": [np.flatnonzero(row).tolist() for row in support],
            "nodes": len(support), "arcs": int(support.sum()), "self_arcs": int(support.trace()),
            "strong_components": 1, "reciprocal_arcs": int((support & support.T).sum()),
            "in_degree": support.sum(axis=1).tolist(), "out_degree": support.sum(axis=0).tolist()}


def decode_graph(record):
    support = np.zeros((record["nodes"], record["nodes"]), dtype=bool)
    for target, sources in enumerate(record["sources_by_target"]):
        support[target, sources] = True
    if edges_hash(support) != record["sha256_uint8_target_source"]:
        raise RuntimeError("Stored support hash disagrees with its membership")
    return support


def register(out):
    if (out / "registered.json").exists():
        raise RuntimeError("The experiment is already registered")
    archived = load_support()
    rng = np.random.default_rng(661773931)
    births, control_seeds = rng.choice(2 ** 31, size=(2, STARTS), replace=False)
    controls = []
    switched_supports = []
    for birth, control in zip(births, control_seeds, strict=True):
        randomized, trace = switched(archived, control)
        switched_supports.append(randomized)
        overlap = int((archived & randomized).sum())
        controls.append({"birth": int(birth), "switch_seed": int(control), "trace": trace,
                         "common_with_archive": overlap, "fraction_archive_retained": overlap / EDGES,
                         **graph_record(randomized)})
    graph_path = out / "graphs.json"
    shared.write(graph_path, {"archive": graph_record(archived), "controls": controls,
                             "control_pair_common_arcs": [[int((a & b).sum()) for b in switched_supports]
                                                          for a in switched_supports]})
    bank = model.Bank([int(births[0])], [archived])
    parameters = {name: int(value.numel()) for name, value in bank.named_parameters()}
    all_roots = {key: list(value) for key, value in ROOTS.items()}
    if sum(map(len, all_roots.values())) != len(set().union(*ROOTS.values())):
        raise RuntimeError("Training and evaluation roots overlap")
    shared.write(out / "registered.json", {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": {"path": str(ARCHIVE.relative_to(REPO).as_posix()), "id": ARCHIVE_ID,
                    "sha256": ARCHIVE_SHA, "support": "birth_mask AND (birth_w != 0)",
                    "edge_representation": "u,v = torch.triu_indices(96, 96, 1); src=cat(v,u); dst=cat(u,v)"},
        "graphs_sha256": shared.sha(graph_path), "source_sha256": source_hashes(),
        "arms": model.ARMS, "rules": env.RULES, "pairs": env.PAIRS.tolist(),
        "truth": env.truth().astype(int).tolist(), "timing": env.TIMING,
        "world_roots": all_roots, "births": births.tolist(), "control_seeds": control_seeds.tolist(),
        "parameters_by_name": parameters, "parameters_per_fitted_model": sum(parameters.values()),
        "raw_parameter_match": "Paired drive, candidate/leak bias, readout, and offset are identical at initialization. The incoming weight multiset matches for each target, with independent N(0, .82^2/in_degree) draws in sorted-source order. Each active edge is independently trainable. Off-support arcs always contribute zero.",
        "bias_and_readout": "New full nodewise tanh candidate and leaky gate per node, global linear right-left readout. None of the original weights, bias, plasticity rules, port bindings or motors used.",
        "controls": "12 independent symmetric directed 2-switch walks of 40 proposals per arc, rejecting self and duplicate arcs. At least 20 accepted switches per arc. Exact in/out degree for every indexed node and all 96 nodes in one SCC at each scored endpoint.",
        "training": {"optimizer": "Adam", "rate": RATE, "updates": STEPS, "batch_size": len(env.PAIRS),
                     "warmup": WARMUP, "random_stream_seed": 2997728, "train_world_action_seed": ROOTS["train"].start ^ 0xD1530B,
                     "objective": "Shared ring-logic objective with a random disclosed physical taste, common roots along all pairs/rules within each start, gradients through query only. All starts retained; fixed rate from the prior ring logic study.",
                     "accelerator_math": "float32 tensors; TF32 allowed for CUDA matmul"},
        "confirmation": "Both budgets, standard and independent long-gap worlds, every start and all 20 ordered pairs. Accuracy = positive first physical contact; timeouts wrong. Primary contrasts are archived minus switched accuracy. Two-way paired percentile bootstrap resamples starts and roots as crossed blocks at each profile, with 95% simultaneous Bonferroni intervals over all 12 test cells.",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    })
    print("registered", out, "parameters/model", sum(parameters.values()), flush=True)


def registered(out):
    info = shared.read(out / "registered.json")
    if info["source_sha256"] != source_hashes():
        raise RuntimeError("The code changed after registration")
    if info["graphs_sha256"] != shared.sha(out / "graphs.json") or shared.sha(ARCHIVE) != ARCHIVE_SHA:
        raise RuntimeError("A registered support artifact changed")
    return info


def bank_for(out, info, device):
    graphs = shared.read(out / "graphs.json")
    parent = decode_graph(graphs["archive"])
    control = [decode_graph(record) for record in graphs["controls"]]
    rows = [(start, 0, birth, rule, arm) for start, birth in enumerate(info["births"])
            for rule in range(len(env.RULES)) for arm in model.ARMS]
    supports = [parent if row[4] == "archive" else control[row[0]] for row in rows]
    bank = model.Bank([row[2] for row in rows], supports).to(device)
    return bank, rows


def train(out, device):
    info = registered(out)
    if (out / "frozen.json").exists():
        raise RuntimeError("The archived transfer checkpoints are already frozen")
    train_roots = ROOTS["train"]
    prefix, query, _ = env.sensory(train_roots, "standard")
    contacted, taste = env.exploration(env.body_tape(train_roots), info["training"]["train_world_action_seed"])
    data = [torch.from_numpy(array).to(device) for array in (prefix, query, contacted, taste)]
    bank, rows = bank_for(out, info, device)
    targets = torch.tensor([row[3] for row in rows], device=device)
    groups = torch.arange(STARTS, device=device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=RATE, fused=device.type == "cuda",
                                 capturable=device.type == "cuda")
    torch.manual_seed(info["training"]["random_stream_seed"])

    def update():
        return model.train_step(bank, *data, targets, groups, env.TIMING["standard"][0], 1, optimizer)

    start = time.perf_counter()
    count = 0
    if device.type == "cuda":
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(WARMUP):
                loss = update()
        torch.cuda.current_stream(device).wait_stream(stream)
        count = WARMUP
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = update()
        print("captured", len(rows), "models", round(time.perf_counter() - start, 2), "s", flush=True)
    artifacts = []
    for target in STEPS:
        for _ in range(count, target):
            if device.type == "cuda":
                graph.replay()
            else:
                loss = update()
        count = target
        completed = int(optimizer.state[bank.drive]["step"].item())
        if completed != target or not torch.isfinite(loss):
            raise RuntimeError(f"Expected {target} updates and finite loss; got {completed}")
        dest = out / f"step-{target}.pt"
        partial = dest.with_suffix(".writing.pt")
        torch.save({name: parameter.detach().cpu() for name, parameter in bank.named_parameters()}, partial)
        partial.replace(dest)
        record = {"path": dest.name, "sha256": shared.sha(dest), "updates": target,
                  "mean_loss": float(loss) / len(rows), "seconds": time.perf_counter() - start}
        artifacts.append(record)
        print("fitted", record, flush=True)
    runtime = shared.runtime(device)
    runtime["matmul_tf32"] = device.type == "cuda"
    shared.write(out / "frozen.json", {"registered_sha256": shared.sha(out / "registered.json"),
                                      "runtime": runtime, "checkpoints": artifacts})


def frozen(out):
    info = registered(out)
    verification = shared.read(out / "frozen.json")
    if verification["registered_sha256"] != shared.sha(out / "registered.json"):
        raise RuntimeError("Registration changed after freezing")
    for item in verification["checkpoints"]:
        if shared.sha(out / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Checkpoint changed after freezing: {item['path']}")
    if tuple(item["updates"] for item in verification["checkpoints"]) != STEPS:
        raise RuntimeError("Checkpoint budgets differ from registered budgets")
    return info, verification


def confirm(out, device):
    info, verification = frozen(out)
    bank, rows = bank_for(out, info, device)
    split = {key: (env.body_tape(ROOTS[key]), *env.sensory(ROOTS[key], key), key)
             for key in ("standard", "long")}
    oracles = {}
    for profile, (tape, _, query, _, _) in split.items():
        oracles[profile] = [env.oracle(tape, query, rule).reshape(-1, len(env.PAIRS), 3)
                            for rule in range(len(env.RULES))]
    scores = {f"oracle-{name}": outcome for name, outcome in oracles.items()}
    for item in verification["checkpoints"]:
        weights = torch.load(out / item["path"], map_location="cpu", weights_only=True)
        bank.load_state_dict({**weights, "positions": bank.positions})
        for profile, episode in split.items():
            results = shared.score(bank, rows, episode, batch_roots=4)
            key = f"{item['updates']}-{profile}"
            scores[key] = results.reshape(STARTS, len(env.RULES), len(model.ARMS), *results.shape[1:])
            print("confirmed", key, (results[..., 2] > 0).mean(), flush=True)
    path = out / "scores.npz"
    tmp = path.with_suffix(".writing.npz")
    np.savez_compressed(tmp, **scores)
    tmp.replace(path)
    shared.write(out / "scored.json", {"frozen_sha256": shared.sha(out / "frozen.json"),
                                      "scores_sha256": shared.sha(path)})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "train", "confirm"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="cuda")
    options = parser.parse_args()
    torch.set_num_threads(1)
    dev = torch.device(options.device)
    torch.set_float32_matmul_precision("high" if dev.type == "cuda" else "highest")
    if options.stage == "register":
        register(options.out)
    elif options.stage == "train":
        train(options.out, dev)
    else:
        confirm(options.out, dev)


if __name__ == "__main__":
    main()
