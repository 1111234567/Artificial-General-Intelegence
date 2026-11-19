"""Register, freeze, and physically score the fixed free-degree graph experiment."""

from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np
import torch

from . import free_graph_model as model, ring_logic_model as ring, ring_logic_world as env
from . import ring_logic_study as original


DEFAULT = original.REPO / "data" / "pulse-game" / "free-graph-20261119"
SOURCES = ("free_graph_model.py", "free_graph_study.py", "free_graph_analysis.py",
           "ring_logic_model.py", "ring_logic_world.py", "ring_logic_study.py",
           "ring_logic_analysis.py", "world.py")
ROOTS = {"train": (67252000, 192), "calibration": (67253000, 48),
         "confirmation": (67254000, 128), "long": (67255000, 128)}
BIRTH_SEED = 78322409
GROUPS = 16
RESTARTS = 2
STEPS = (240, 1200)
RATE = .024
SHARD = 8
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 260629382
DESIGNS = {"tied": model.ARMS, "independent": ("regular", "fully_free")}


def source_hashes():
    return {name: original.sha(Path(__file__).with_name(name)) for name in SOURCES}


def roots(stage):
    start, length = ROOTS[stage]
    return range(start, start + length)


def births():
    return np.random.default_rng(BIRTH_SEED).choice(2 ** 31, size=(GROUPS, RESTARTS), replace=False)


def specs(matrix, design, start):
    return [(group, restart, int(seed), rule, arm)
            for group, group_births in enumerate(matrix, start)
            for restart, seed in enumerate(group_births)
            for rule in range(len(env.RULES))
            for arm in DESIGNS[design]]


def make_bank(rows, design, device):
    return model.Bank([row[2] for row in rows], [row[4] for row in rows],
                      independent=design == "independent").to(device)


def register(out, device):
    target = out / "registered.json"
    if target.exists():
        raise RuntimeError("Registration already exists")
    world_roots = {stage: list(roots(stage)) for stage in ROOTS}
    current = [root for values in world_roots.values() for root in values]
    previous = {root for stage in original.ROOTS for root in original.roots(stage)}
    matrix = births()
    previous_births = {seed for stage in ("development", "confirmation")
                       for seed in original.births(stage).ravel()}
    if len(set(current)) != len(current) or previous.intersection(current):
        raise RuntimeError("World splits must be disjoint and unseen by the previous logic study")
    if previous_births.intersection(matrix.ravel()):
        raise RuntimeError("Births overlap the previous logic study")
    graph_rows = [{"group": group, "restart": restart, "seed": seed, "arm": arm,
                   **model.graph_record(seed, arm)}
                  for group, restart, seed, rule, arm in specs(matrix, "tied", 0) if rule == 0]
    original.write(out / "graphs.json", graph_rows)
    shapes = {}
    for design in DESIGNS:
        bank = make_bank(specs(matrix[:1], design, 0)[:1], design, "cpu")
        shapes[design] = {key: list(value.shape[1:]) for key, value in bank.named_parameters()}
    original.write(target, {
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does relaxing 2-in/2-out degree and then the common ordered Hamiltonian backbone change trainability in the two-odor physical task?",
        "primary": "Physical accuracy at 1200 updates on standard timing. Three paired graph contrasts for each of three rules, calibration-selected. Report a family of 9 Bonferroni simultaneous 95% two-sided percentile intervals crossing independent paired resampling of birth group and physical world root. All 20 forks of a root stay intact. Timeout contributes an incorrect cell.",
        "secondary": "Analogous family of 9 simultaneous intervals for the all-20-cells-correct world probability. The independent edge-weight comparison is supplementary with a separate family of 3 simultaneous intervals for each accuracy outcome. The change in the graph contrast from tied to independent gains uses one more supplementary family of 3 for each outcome. Early 240-update and unseen long-gap suites are descriptive with individual intervals. Per-restart, best observed restart, 95% success counts, all-20 successes and per-cell outcomes are descriptive.",
        "topologies": {
            "regular": "Ordered directed cycle plus a seeded permutation of chord sources, exactly 2-in/2-out and identical support to the previous multiloop arm for any same seed.",
            "free_chords": "Same ordered directed cycle plus a uniformly sampled 32-edge subset of the remaining 960 non-self ordered pairs, with no constraints on per-node degrees beyond the backbone.",
            "fully_free": "Uniform labelled simple digraph conditional on 32 nodes, 64 non-self arcs and strong connectivity. Independently draw two uniform covering endpoint sequences using exact integer completion counts. Reject paired sequences yielding a self arc, duplicate arc or failure of strong connectivity. Strong connectivity entails both sequences cover all nodes, and exactly 64! accepted sequences encode each valid support. No mandatory backbone or per-vertex degree constraint. Realizations are sampled once per restart and held fixed for all rules, then listed in the registration's graph file.",
        },
        "circuit": "Identical 32 leaky gated state cells and 64 non-self arcs. The tied variant learns one gain per receiver, multiplying each incoming state by the receiver gain divided by sqrt(actual indegree); matrix row norm in magnitude equals that receiver's gain magnitude. The independent variant learns one raw gain per edge, each initialized to the corresponding tied receiver gain, divided by the same fixed degree factor. All other initial tensors and formulas are identical across arms and parameter designs. A common dense recurrent kernel is used for every arm. Intrinsic gated retention is identical everywhere.",
        "training": "Unseen births and training world roots. Use public actual physical contact tastes under shared forced random exploration. Each update samples one root with replacement for each of 20 ordered odor cells and each restart, paired across rules, topologies and parameter designs. Match original Adam rate and the two update budgets. Keep every checkpoint. Within each birth group, model arm, rule and budget, select a restart by physical accuracy on independent calibration roots, exact ties selecting restart 0. Freeze parameters, selection and graph records before preparing any confirmation or long world.",
        "rule_names": env.RULES, "truth": env.truth().astype(int).tolist(),
        "pairs": env.PAIRS.tolist(), "odor_codes": env.ODORS.tolist(), "timing": env.TIMING,
        "width": ring.WIDTH, "arc_count": model.EDGE_COUNT, "births": matrix.tolist(),
        "world_roots": world_roots, "arms": DESIGNS, "parameter_shapes": shapes,
        "parameter_counts": {key: sum(int(np.prod(v)) for v in values.values())
                             for key, values in shapes.items()},
        "optimizer": "Adam", "rate": RATE, "updates": STEPS, "batch_size": len(env.PAIRS),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime": original.runtime(device), "graphs_sha256": original.sha(out / "graphs.json"),
        "source_sha256": source_hashes(),
    })


def require_registered(out):
    registration = original.read(out / "registered.json")
    if registration["source_sha256"] != source_hashes():
        raise RuntimeError("Program differs from its registration")
    if registration["graphs_sha256"] != original.sha(out / "graphs.json"):
        raise RuntimeError("Graph records differ from registration")
    return registration


def training_data(device):
    train_roots = roots("train")
    prefix, query, _ = env.sensory(train_roots, "standard")
    chosen, reward = env.exploration(env.body_tape(train_roots), train_roots.start ^ 0xD1530B)
    return [torch.from_numpy(array).to(device) for array in (prefix, query, chosen, reward)]


def scoring_data(world_roots, profile):
    prefix, query, target = env.sensory(world_roots, profile)
    return env.body_tape(world_roots), prefix, query, target, profile


def fit(bank, rows, design, data, device, seed):
    targets = torch.tensor([row[3] for row in rows], device=device)
    groups = torch.arange(len(rows) // (len(env.RULES) * len(DESIGNS[design])), device=device)
    cuda = device.type == "cuda"
    opt = torch.optim.Adam(bank.parameters(), lr=RATE, fused=cuda, capturable=cuda)
    torch.manual_seed(seed)

    def step():
        return ring.train_step(bank, *data, targets, groups, env.TIMING["standard"][0], 1, opt)

    start = time.perf_counter()
    count = 0
    if cuda:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(original.WARMUP):
                loss = step()
        torch.cuda.current_stream().wait_stream(stream)
        count = original.WARMUP
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()
    for budget in STEPS:
        for _ in range(count, budget):
            if cuda:
                graph.replay()
            else:
                loss = step()
        count = budget
        completed = int(opt.state[bank.drive]["step"].item())
        if completed != budget or not torch.isfinite(loss):
            raise RuntimeError(f"Expected {budget} Adam updates and finite loss, got {completed}")
        yield budget, float(loss) / len(rows), time.perf_counter() - start


def write_arrays(dest, values):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".writing.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(dest)


def save_checkpoint(dest, bank, rows, design):
    values = {name: param.detach().cpu().numpy() for name, param in bank.named_parameters()}
    values["specs"] = np.array([[g, r, seed, rule, model.ARMS.index(arm)]
                                for g, r, seed, rule, arm in rows])
    values["design"] = np.asarray(design)
    write_arrays(dest, values)


def load_checkpoint(source, device, pick=None):
    with np.load(source, allow_pickle=False) as archive:
        ids = np.arange(len(archive["specs"])) if pick is None else np.asarray(pick)
        rows = [(int(g), int(r), int(seed), int(rule), model.ARMS[a])
                for g, r, seed, rule, a in archive["specs"][ids]]
        bank = make_bank(rows, archive["design"].item(), device)
        bank.load_state_dict({name: torch.from_numpy(archive[name][ids]) for name in ring.PARAMETERS})
    return bank, rows


def train(out, device):
    registration = require_registered(out)
    if (out / "frozen.json").exists():
        raise RuntimeError("Training checkpoints are already frozen")
    if registration["runtime"] != original.runtime(device):
        raise RuntimeError("Training runtime differs from registration")
    data = training_data(device)
    calibration = scoring_data(roots("calibration"), "standard")
    matrix = births()
    progress = out / "training-progress.json"
    records = original.read(progress) if progress.exists() else []
    for start in range(0, len(matrix), SHARD):
        for design, arms in DESIGNS.items():
            if any(item["first_group"] == start and item["design"] == design for item in records):
                continue
            rows = specs(matrix[start:start + SHARD], design, start)
            bank = make_bank(rows, design, device)
            for budget, loss, elapsed in fit(bank, rows, design, data, device, BIRTH_SEED ^ (start + 78139)):
                dest = out / "checkpoints" / f"{design}-birth-{start:02d}-step-{budget}.npz"
                save_checkpoint(dest, bank, rows, design)
                score = original.score(bank, rows, calibration, batch_roots=4)
                accurate = (score[..., 2] > 0).mean(axis=(1, 2))
                values = accurate.reshape(-1, RESTARTS, len(env.RULES), len(arms))
                records.append({"path": dest.relative_to(out).as_posix(), "sha256": original.sha(dest),
                                "design": design, "first_group": start, "updates": budget,
                                "elapsed_seconds": elapsed, "mean_batch_loss": loss,
                                "calibration_accuracy": values.tolist(),
                                "calibration_selected": values.argmax(axis=1).tolist()})
                print("training", design, start, budget, elapsed, float(accurate.mean()), flush=True)
            original.write(progress, records)
    expected = {(start, design, budget) for start in range(0, len(matrix), SHARD)
                for design in DESIGNS for budget in STEPS}
    if {(i["first_group"], i["design"], i["updates"]) for i in records} != expected:
        raise RuntimeError("Training is missing registered checkpoints")
    if any(original.sha(out / item["path"]) != item["sha256"] for item in records):
        raise RuntimeError("Checkpoint changed before freezing")
    original.write(out / "frozen.json", {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registered_sha256": original.sha(out / "registered.json"), "files": records,
    })


def freeze_guard(out):
    require_registered(out)
    frozen = original.read(out / "frozen.json")
    if original.sha(out / "registered.json") != frozen["registered_sha256"]:
        raise RuntimeError("Registration differs from freeze")
    for record in frozen["files"]:
        if original.sha(out / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Checkpoint altered: {record['path']}")
    return frozen


def selection_indices(artifact):
    selected = np.asarray(artifact["calibration_selected"])
    group, rule, arm = np.indices(selected.shape)
    shape = (len(group), RESTARTS, len(env.RULES), len(DESIGNS[artifact["design"]]))
    return np.ravel_multi_index((group, selected, rule, arm), shape).ravel()


def score_path(out, artifact):
    return out / "scores" / Path(artifact["path"]).name


def confirm(out, device):
    frozen = freeze_guard(out)
    suites = {"standard": scoring_data(roots("confirmation"), "standard"),
              "long": scoring_data(roots("long"), "long")}
    references = {}
    for profile, (tape, _, query, _, _) in suites.items():
        references[profile] = np.stack([env.oracle(tape, query, rule).reshape(-1, len(env.PAIRS), 3)
                                        for rule in range(len(env.RULES))])
    write_arrays(out / "oracles.npz", references)
    for artifact in frozen["files"]:
        dest = score_path(out, artifact)
        if dest.exists():
            continue
        bank, rows = load_checkpoint(out / artifact["path"], device)
        write_arrays(dest, {profile: original.score(bank, rows, evaluation, batch_roots=4)
                            for profile, evaluation in suites.items()})
        print("confirmed", dest.name, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "train", "confirm"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    stages = {"register": register, "train": train, "confirm": confirm}
    stages[args.stage](args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
