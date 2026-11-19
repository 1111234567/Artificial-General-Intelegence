"""Register, develop, freeze, and score the matched ring logic study."""

from argparse import ArgumentParser
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from . import ring_logic_model as model, ring_logic_world as env, world


REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "data" / "pulse-game" / "ring-logic-20261119"
SOURCES = ("ring_logic_model.py", "ring_logic_world.py", "ring_logic_study.py",
           "ring_logic_analysis.py", "world.py")
ROOTS = {"dev_train": (38592000, 128), "development": (39243000, 40),
         "main_train": (55731000, 192), "calibration": (57842000, 48),
         "confirmation": (88647000, 128), "long": (89647000, 128)}
BIRTH_ROOT = 31698230
DEV_ROOT = 91694030
MAIN_BIRTHS = 16
DEV_BIRTHS = 4
RESTARTS = 2
RATES = (.012, .024)
STEPS = (240, 1200)
REPETITIONS = 1
SHARD = 8
WARMUP = 5
INTERVENTIONS = ("drop", "reset", "first_only", "second_only", "timing_only")
PRIMARY_ARMS = ("self", "directed", "multiloop")
BOOTSTRAP_DRAWS = 20_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCES}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def roots(stage):
    start, length = ROOTS[stage]
    return range(start, start + length)


def births(stage):
    base, count = (DEV_ROOT, DEV_BIRTHS) if stage == "development" else (BIRTH_ROOT, MAIN_BIRTHS)
    return np.random.default_rng(base).choice(2 ** 31, size=(count, RESTARTS), replace=False)


def specs(birth_matrix, start=0):
    return [
        (group, restart, int(seed), rule, arm)
        for group, group_births in enumerate(birth_matrix, start)
        for restart, seed in enumerate(group_births)
        for rule in range(len(env.RULES))
        for arm in model.ARMS
    ]


def bank_for(rows, device):
    return model.Bank([r[2] for r in rows], [r[4] for r in rows]).to(device)


def topology(seed, arm):
    n = model.WIDTH
    targets = np.arange(n)
    sources = {"directed": [(targets - 1) % n], "self": [targets],
               "undirected": [(targets - 1) % n, (targets + 1) % n]}
    if arm == "multiloop":
        segments = [(targets - 1) % n, model.chord_sources(seed).numpy()]
    else:
        segments = sources[arm]
    adjacency = np.zeros((n, n), dtype=np.uint8)
    for block in segments:
        adjacency[targets, block] = 1
    walks = adjacency.astype(bool)
    reach = walks.copy()
    first_return = np.zeros(n, dtype=np.int16)
    for length in range(1, n + 1):
        np.copyto(first_return, length, where=walks.diagonal() & (first_return == 0))
        walks = (walks.astype(np.int32) @ adjacency) != 0
        reach |= walks
    return {"sources_per_target": np.stack(segments, axis=-1).tolist(),
            "adjacency_sha256_uint8_row_major": hashlib.sha256(adjacency.tobytes()).hexdigest(),
            "in_degree": adjacency.sum(axis=1).tolist(),
            "out_degree": adjacency.sum(axis=0).tolist(),
            "self_edges": int(adjacency.trace()),
            "shortest_directed_return": first_return.tolist(),
            "strong_components": len({tuple(np.flatnonzero(reach[i] & reach[:, i]))
                                      for i in range(n)}),
            "reciprocal_arcs": int((adjacency & adjacency.T).sum() - adjacency.trace())}


def runtime(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "device": str(device), "cuda_runtime": torch.version.cuda,
            "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu_threads": torch.get_num_threads(), "autocast": False,
            "matmul_tf32": False, "cuda_graph": device.type == "cuda"}


def register(out, device):
    target = out / "registered.json"
    if target.exists():
        raise RuntimeError("Registration already exists")
    draw = {key: list(roots(key)) for key in ROOTS}
    if sum(map(len, draw.values())) != len({root for vals in draw.values() for root in vals}):
        raise RuntimeError("World splits overlap")
    b = {s: births(s).tolist() for s in ("development", "confirmation")}
    if len(set(np.asarray(b["development"]).flatten()) & set(np.asarray(b["confirmation"]).flatten())):
        raise RuntimeError("Birth splits overlap")
    shapes = {key: list(value.shape[1:]) for key, value in
              model.Bank([0], ["directed"]).named_parameters()}
    write(target, {
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary": "Three simple pairwise contrasts among self, directed and multiloop for each of the three rules at the long optimizer budget on standard timing, after independent calibration chooses one of two restart checkpoints per birth group. Score all 20 pairs physically with timeouts incorrect. Family of 9 Bonferroni simultaneous 95% percentile intervals resampling birth groups and episode roots independently.",
        "secondary": "Short optimizer budget, each individual restart, all-cell success, per-cell minimum, swap correctness, long odor-gap out-of-distribution sensitivity, drop candidate and reset diagnostics, one-or-zero-cue control, shared vs distinct simple-ring directionality, CPU inference timing with the active arc implementation.",
        "development": "Choose one Adam rate using only development births and roots, by mean physical accuracy at the 1200-update checkpoint over all rules, arms and restart draws. Exact ties choose the smaller rate. Confirm models begin at new from-scratch births, using independent training worlds. After each optimizer budget the higher calibration score selects a restart separately by group, rule and topology. Exact ties choose restart 0. Calibration choices and checkpoint hashes are frozen before opening confirmation and long profile roots.",
        "rule_names": env.RULES, "truth": env.truth().astype(int).tolist(),
        "pairs": env.PAIRS.tolist(), "odor_codes": env.ODORS.tolist(), "timing": env.TIMING,
        "controls": list(INTERVENTIONS), "chance": "Every rule has 2 positive and 2 negative per first or second odor. Swapped labels flip for all 10 unordered orientation pairs, for no adjacency pairs, and for 4 lookup pairs. Missing identity histories do not carry their hidden label. Same phase, carriers, distractors, background, prefixes, physical layout and wind across each 20-way fork.",
        "objective": "Use data from public signed tastes of a random forced side that ran through PulseWorld. Sample one root with replacement for each of 20 cells each update; match row and RNG across arms and rules within each restart. Different restart groups draw independent minibatch indices. Prior request is fully experienced without gradients; gradients reach the whole final sensory phase and all 15 movement predictions. Models are independent across bank rows. Adam, no schedules.",
        "topology": "32 cells with identical intrinsic gated self retention. 32 recurrent gains; directed g_i from predecessor, self g_i from self; symmetric bonds (g_i on i-1 and g_{i+1} on i+1) / sqrt(2); multiloop g_i / sqrt(2) on each incoming predecessor and permutation-chord edge. Chord sources generated with a separate transformed birth seed and reject own and predecessor indices; every closed multiloop is 2-in 2-out and shares its strongly connected backbone. Unweighted support and candidate matrix Frobenius scale are both reported; the topology-specific nonlinear update Jacobian need not be symmetric.",
        "parameter_shapes": shapes,
        "parameter_count": sum(int(np.prod(value)) for value in shapes.values()),
        "topologies": model.ARMS, "primary_arms": PRIMARY_ARMS, "world_roots": draw,
        "births": b, "restarts": RESTARTS, "rates": RATES, "checkpoints": STEPS,
        "batch_size": len(env.PAIRS) * REPETITIONS, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "runtime": runtime(device), "source_sha256": source_hashes(),
    })
    graph_rows = []
    for split, matrix in b.items():
        for group, restart, seed, rule, arm in specs(matrix):
            if rule == 0:
                graph_rows.append({"split": split, "group": group, "restart": restart, "seed": seed,
                                   "arm": arm, **topology(seed, arm)})
    write(out / "graphs.json", graph_rows)


def require_registered(out):
    registration = read(out / "registered.json")
    if registration["source_sha256"] != source_hashes():
        raise RuntimeError("Program differs from its registration")
    return registration


def training_data(stage, device):
    train_roots = roots(stage)
    prefix, query, _ = env.sensory(train_roots, "standard")
    tape = env.body_tape(train_roots)
    seed = int(train_roots.start) ^ 0xD1530B
    chosen, reward = env.exploration(tape, seed)
    return [torch.from_numpy(array).to(device) for array in (prefix, query, chosen, reward)]


def fit(bank, rows, data, device, *, rate, seed):
    targets = torch.tensor([row[3] for row in rows], device=device)
    groups = torch.arange(len(rows) // (len(env.RULES) * len(model.ARMS)), device=device)
    opt = torch.optim.Adam(bank.parameters(), lr=rate, fused=device.type == "cuda",
                           capturable=device.type == "cuda")
    torch.manual_seed(seed)

    def step():
        return model.train_step(bank, *data, targets, groups, env.TIMING["standard"][0],
                                REPETITIONS, opt)

    begin = time.perf_counter()
    count = 0
    if device.type == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP):
                loss = step()
        torch.cuda.current_stream().wait_stream(stream)
        count = WARMUP
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()

    for target in STEPS:
        for _ in range(count, target):
            if device.type == "cuda":
                graph.replay()
            else:
                loss = step()
        count = target
        completed = int(opt.state[bank.drive]["step"].item())
        if completed != target or not torch.isfinite(loss):
            raise RuntimeError(f"Expected {target} Adam updates and finite loss, got {completed}")
        yield target, float(loss) / len(rows), time.perf_counter() - begin


def episode(stage, profile):
    world_roots = roots(stage)
    prefix, query, target = env.sensory(world_roots, profile)
    return env.body_tape(world_roots), prefix, query, target, profile


@torch.no_grad()
def score(bank, rows, episode_data, intervention=None, *, path=0, batch_roots=8):
    tape, prefix, query, target, profile = episode_data
    sensory_ticks = env.TIMING[profile][0]
    n, cells = query.shape[:2]
    right_chunks = []
    device = bank.drive.device
    for begin in range(0, n, batch_roots):
        p, q, positions = prefix[begin:begin + batch_roots], query[begin:begin + batch_roots], target[begin:begin + batch_roots]
        if intervention in ("first_only", "second_only", "timing_only"):
            q = q.copy()
            idx = np.arange(len(p))
            if intervention != "first_only":
                q[idx, :, positions[:, 0], :3] = 0
            if intervention != "second_only":
                q[idx, :, positions[:, 1], :3] = 0
        drop = intervention == "drop"
        _, state = bank(torch.as_tensor(p, device=device)[None], bank.start(len(p)), drop=drop,
                        path=path)
        state = state.repeat_interleave(cells, dim=1)
        reset = None
        if intervention == "reset":
            mask = np.arange(q.shape[2])[None, :] == positions[:, 1, None]
            reset = torch.as_tensor(np.repeat(mask, cells, axis=0), device=device)[None]
        x = torch.as_tensor(q.reshape(-1, *q.shape[-2:]), device=device)[None]
        prediction, _ = bank(x, state, retain=sensory_ticks, drop=drop, reset=reset, path=path)
        right_chunks.append((prediction[:, :, :world.TICKS] >= 0).cpu().numpy())
    choices = np.concatenate(right_chunks, axis=1).transpose(0, 2, 1)
    return np.stack([env.contacts(tape, right, row[3]).reshape(n, cells, 3)
                     for row, right in zip(rows, choices, strict=True)])


def save_checkpoint(dest, bank, rows):
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = {key: value.detach().cpu().numpy() for key, value in bank.named_parameters()}
    temporary = dest.with_suffix(".writing.npz")
    np.savez_compressed(temporary, **source,
                        specs=np.array([[a, b, c, d, model.ARMS.index(e)] for a, b, c, d, e in rows]))
    temporary.replace(dest)


def load_checkpoint(source, device, pick=None):
    with np.load(source, allow_pickle=False) as archive:
        ids = np.arange(len(archive["specs"])) if pick is None else np.asarray(pick)
        rows = [(int(a), int(b), int(c), int(d), model.ARMS[e]) for a, b, c, d, e in archive["specs"][ids]]
        bank = bank_for(rows, device)
        bank.load_state_dict({key: torch.from_numpy(archive[key][ids]) for key in model.PARAMETERS})
    return bank, rows


def develop(out, device):
    require_registered(out)
    if (out / "recipe.json").exists():
        raise RuntimeError("Development is closed")
    data = training_data("dev_train", device)
    evaluation = episode("development", "standard")
    matrix = births("development")
    rows = specs(matrix)
    candidates = []
    seed = DEV_ROOT ^ 78139
    for rate in RATES:
        bank = bank_for(rows, device)
        observations = []
        for steps, loss, elapsed in fit(bank, rows, data, device, rate=rate, seed=seed):
            rewards = score(bank, rows, evaluation)[..., 2]
            results = (rewards > 0).mean(axis=(1, 2)).tolist()
            observations.append({"step": steps, "loss": loss, "seconds": elapsed,
                                 "per_candidate_accuracy": results, "mean_accuracy": float(np.mean(results))})
            print("development", rate, steps, loss, elapsed, np.mean(results), flush=True)
        candidates.append({"rate": rate, "results": observations})
        write(out / "development.json", {"rows": rows, "trials": candidates})
    rate = max(candidates, key=lambda item: (item["results"][-1]["mean_accuracy"], -item["rate"]))["rate"]
    write(out / "recipe.json", {"rate": rate, "development_sha256": sha(out / "development.json"),
                                "runtime": runtime(device)})


def fit_main(out, device):
    require_registered(out)
    if (out / "frozen.json").exists():
        raise RuntimeError("Main training checkpoints are already frozen")
    recipe = read(out / "recipe.json")
    if recipe["development_sha256"] != sha(out / "development.json"):
        raise RuntimeError("Development scoring changed after recipe selection")
    data = training_data("main_train", device)
    evaluation = episode("calibration", "standard")
    matrix = births("confirmation")
    artifacts = []
    for start in range(0, len(matrix), SHARD):
        rows = specs(matrix[start:start + SHARD], start)
        bank = bank_for(rows, device)
        seed = BIRTH_ROOT ^ (start + 78139)
        for steps, loss, elapsed in fit(bank, rows, data, device, rate=recipe["rate"], seed=seed):
            dest = out / "checkpoints" / f"birth-{start:02d}-step-{steps}.npz"
            save_checkpoint(dest, bank, rows)
            rewards = score(bank, rows, evaluation)[..., 2]
            values = (rewards > 0).mean(axis=(1, 2)).reshape(-1, RESTARTS, len(env.RULES), len(model.ARMS))
            selection = values.argmax(axis=1)
            artifacts.append({"path": dest.relative_to(out).as_posix(), "sha256": sha(dest),
                              "updates": steps, "first_group": start, "seconds": elapsed,
                              "mean_batch_loss": loss, "calibration": values.tolist(),
                              "calibration_selected": selection.tolist()})
            print("training", start, steps, loss, elapsed, float(values.mean()), flush=True)
        write(out / "training-progress.json", artifacts)
    write(out / "frozen.json", {"recipe_sha256": sha(out / "recipe.json"),
                                "registered_sha256": sha(out / "registered.json"),
                                "graphs_sha256": sha(out / "graphs.json"), "files": artifacts})


def freeze_guard(out):
    frozen = read(out / "frozen.json")
    for file in ("registered", "recipe", "graphs"):
        if sha(out / f"{file}.json") != frozen[f"{file}_sha256"]:
            raise RuntimeError(f"{file} changed after freezing")
    for record in frozen["files"]:
        if sha(out / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Checkpoint altered: {record['path']}")
    return frozen


def _selection(artifact):
    restarts = np.asarray(artifact["calibration_selected"])
    groups, rule, arm = np.indices(restarts.shape)
    shape = (len(groups), RESTARTS, len(env.RULES), len(model.ARMS))
    return np.ravel_multi_index((groups, restarts, rule, arm), shape).reshape(-1)


def confirm(out, device):
    require_registered(out)
    frozen = freeze_guard(out)
    suites = {"standard": episode("confirmation", "standard"), "long": episode("long", "long")}
    references = {}
    for profile, evaluation in suites.items():
        tape, _, query, _, _ = evaluation
        references[profile] = [env.oracle(tape, query, rule).reshape(-1, len(env.PAIRS), 3)
                               for rule in range(len(env.RULES))]
    np.savez_compressed(out / "oracles.npz", **references)
    for artifact in frozen["files"]:
        start, step = artifact["first_group"], artifact["updates"]
        dest = out / "scores" / f"birth-{start:02d}-step-{step}.npz"
        if dest.exists():
            continue
        bank, rows = load_checkpoint(out / artifact["path"], device)
        scores = {profile: score(bank, rows, evaluation) for profile, evaluation in suites.items()}
        del bank
        if step == STEPS[-1]:
            selected, selected_rows = load_checkpoint(out / artifact["path"], device, _selection(artifact))
            for intervention in INTERVENTIONS:
                scores[intervention] = score(selected, selected_rows, suites["standard"], intervention)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".writing.npz")
        np.savez_compressed(temp, **scores)
        temp.replace(dest)
        print("confirmed", start, step, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "develop", "train", "confirm"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    stages = {"register": register, "develop": develop, "train": fit_main, "confirm": confirm}
    stages[args.stage](args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
