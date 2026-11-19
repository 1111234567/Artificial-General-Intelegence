"""Hold the closed multiloop graph fixed and vary the two incoming gains."""

from argparse import ArgumentParser
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import ring_logic_analysis as analysis, ring_logic_model as model
from . import ring_logic_study as study, ring_logic_world as env


DEFAULT = study.DEFAULT.with_name("ring-routing-20261119")
ARMS = ("equal", "angle", "free")
BIRTH_SEED = 84290317
GROUPS = 8
ROOTS = {"train": (66104200, 192), "calibration": (68031300, 48),
         "confirmation": (69072600, 128)}
UPDATES = 1200


class RoutingBank(model.Bank):
    def __init__(self, rows):
        super().__init__([row[2] for row in rows], ["multiloop"] * len(rows))
        angle = torch.tensor([row[4] == "angle" for row in rows])[:, None]
        free = torch.tensor([i for i, row in enumerate(rows) if row[4] == "free"], dtype=torch.long)
        self.register_buffer("angle", angle, persistent=False)
        self.register_buffer("free", free, persistent=False)
        self.register_buffer("birth_gain", self.gain.detach().clone(), persistent=False)
        self.chord_gain = nn.Parameter(self.gain.detach()[free].clone())
        with torch.no_grad():
            self.gain.masked_fill_(angle, 0.)

    def coefficients(self, *, drop=False):
        gain = self.gain
        unit = gain.new_ones(()) / math.sqrt(2.)
        theta = torch.where(self.angle, gain, 0.)
        radius = self.birth_gain * unit
        left = torch.where(self.angle, radius * (theta.cos() - theta.sin()), gain * unit)
        right_gain = gain.index_copy(0, self.free, self.chord_gain)
        right = torch.where(self.angle, radius * (theta.cos() + theta.sin()), right_gain * unit)
        if drop:
            left, right = torch.zeros_like(left), torch.zeros_like(right)
        return left, right, torch.zeros_like(gain)


def _roots(stage):
    first, count = ROOTS[stage]
    return range(first, first + count)


def _rows(births):
    return [(group, restart, seed, rule, arm)
            for group, pair in enumerate(births) for restart, seed in enumerate(pair)
            for rule in range(len(env.RULES)) for arm in ARMS]


def _hashes():
    return {**study.source_hashes(), Path(__file__).name: study.sha(Path(__file__))}


def register(out, device):
    destination = out / "registered.json"
    if destination.exists():
        raise RuntimeError("Routing study is already registered")
    recipe = study.DEFAULT / "recipe.json"
    prior = study.require_registered(study.DEFAULT)
    births = np.random.default_rng(BIRTH_SEED).choice(2 ** 31, size=(GROUPS, study.RESTARTS),
                                                    replace=False).tolist()
    prior_births = {seed for matrix in prior["births"].values() for row in matrix for seed in row}
    prior_roots = {root for values in prior["world_roots"].values() for root in values}
    roots = [root for stage in ROOTS for root in _roots(stage)]
    if prior_births.intersection(np.ravel(births)) or prior_roots.intersection(roots) or \
            len(set(roots)) != len(roots):
        raise RuntimeError("The prior and routing study splits must be disjoint")
    study.write(destination, {
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary": "Angle minus equal and free minus equal physical accuracy at 1200 Adam updates, for each of three rules. Report six simultaneous 95% percentile intervals, resampling paired birth groups and episode roots independently with Bonferroni tails.",
        "controls": "Every birth, restart, and rule keeps the same 2-in/2-out closed backbone-plus-chord graph and all five input channels, leak, readout, birth candidate coefficients and input stream. Every univariate arm learns 32 scalars on these edges. The diagnostic free arm learns 64, with a fixed 1/sqrt(2) factor on each raw gain.",
        "parameters": {"equal": 449, "angle": 449, "free": 481},
        "coefficients": {"equal": "g/sqrt(2) on both arcs, signed g learns",
                         "angle": "g_birth/sqrt(2) times (cos(theta)-sin(theta), cos(theta)+sin(theta)), theta starts at 0 and learns; the L2 norm per receiving row is fixed at birth",
                         "free": "(g_left,g_chord)/sqrt(2), independent signed coordinates start at the same g_birth"},
        "training": "Exact original signed public contact receipts, minibatch logic, Adam configuration and motion, with an independently drawn train population. Fit from scratch. Calibrate each arm/rule/birth by physical accuracy across its two restart checkpoints, ties select restart 0. Only inspect confirmation after all selections and checkpoints are frozen.",
        "world_roots": {stage: list(_roots(stage)) for stage in ROOTS}, "births": births,
        "chords_by_birth": {str(seed): model.chord_sources(seed).tolist() for seed in np.ravel(births)},
        "rows": _rows(births), "rules": env.RULES, "pair_table": env.PAIRS.tolist(),
        "updates": UPDATES, "training_rng": BIRTH_SEED ^ 78139, "rate": study.read(recipe)["rate"],
        "main_development_recipe_sha256": study.sha(recipe),
        "parameter_count_by_arm": [449, 449, 481], "source_sha256": _hashes(),
        "runtime": study.runtime(device),
    })


def _registered(out):
    record = study.read(out / "registered.json")
    if record["source_sha256"] != _hashes():
        raise RuntimeError("Routing sources have changed since registration")
    return record


def _episode(stage):
    roots = _roots(stage)
    prefix, query, target = env.sensory(roots, "standard")
    return env.body_tape(roots), prefix, query, target, "standard"


def _fit(bank, rows, data, device, registration):
    targets = torch.tensor([row[3] for row in rows], device=device)
    groups = torch.arange(GROUPS * study.RESTARTS, device=device)
    use_cuda = device.type == "cuda"
    optimizer = torch.optim.Adam(bank.parameters(), lr=registration["rate"], fused=use_cuda,
                                 capturable=use_cuda)
    torch.manual_seed(registration["training_rng"])

    def step():
        return model.train_step(bank, *data, targets, groups, env.TIMING["standard"][0],
                                study.REPETITIONS, optimizer)

    completed = 0
    if use_cuda:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(study.WARMUP):
                loss = step()
        torch.cuda.current_stream().wait_stream(stream)
        completed = study.WARMUP
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()
    for _ in range(completed, UPDATES):
        if use_cuda:
            graph.replay()
        else:
            loss = step()
    if int(optimizer.state[bank.gain]["step"].item()) != UPDATES or not loss.isfinite():
        raise RuntimeError("Training did not reach its registered budget with finite loss")
    return float(loss) / len(rows)


def train(out, device):
    registration = _registered(out)
    frozen = out / "frozen.json"
    checkpoint = out / "checkpoint.npz"
    if frozen.exists() or checkpoint.exists():
        raise RuntimeError("Routing study training outputs already exist")
    rows = registration["rows"]
    bank = RoutingBank(rows).to(device)
    tape, prefix, query, _, _ = _episode("train")
    chosen, reward = env.exploration(tape, _roots("train").start ^ 0xD1530B)
    data = [torch.from_numpy(array).to(device) for array in (prefix, query, chosen, reward)]
    begin = time.perf_counter()
    loss = _fit(bank, rows, data, device, registration)
    elapsed = time.perf_counter() - begin
    arrays = {key: value.detach().cpu().numpy() for key, value in bank.named_parameters()}
    temporary = checkpoint.with_suffix(".writing.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(checkpoint)
    calibration = study.score(bank, rows, _episode("calibration"))
    accuracy = (calibration[..., 2] > 0).mean(axis=(1, 2))
    accuracy = accuracy.reshape(GROUPS, study.RESTARTS, len(env.RULES), len(ARMS))
    study.write(frozen, {"checkpoint_sha256": study.sha(checkpoint),
                         "registration_sha256": study.sha(out / "registered.json"),
                         "calibration_accuracy": accuracy.tolist(),
                         "calibration_selected": accuracy.argmax(axis=1).tolist(),
                         "updates": UPDATES, "mean_last_batch_loss": loss, "training_seconds": elapsed})
    print(f"frozen at {UPDATES} updates in {elapsed:.1f} seconds", flush=True)


def confirm(out, device):
    registration = _registered(out)
    frozen = study.read(out / "frozen.json")
    checkpoint = out / "checkpoint.npz"
    if frozen["checkpoint_sha256"] != study.sha(checkpoint) or \
            frozen["registration_sha256"] != study.sha(out / "registered.json"):
        raise RuntimeError("Frozen study inputs changed")
    score_file = out / "confirmation.npz"
    if score_file.exists():
        raise RuntimeError("Routing confirmation is already scored")
    rows = registration["rows"]
    bank = RoutingBank(rows).to(device)
    with np.load(checkpoint, allow_pickle=False) as saved:
        bank.load_state_dict({key: torch.from_numpy(saved[key]) for key in saved.files})
    scores = study.score(bank, rows, _episode("confirmation"))
    scores = scores.reshape(GROUPS, study.RESTARTS, len(env.RULES), len(ARMS), *scores.shape[1:])
    selection = np.asarray(frozen["calibration_selected"], dtype=np.int64)
    birth, rule, arm = np.indices(selection.shape)
    selected = scores[birth, selection, rule, arm]
    np.savez_compressed(score_file, scores=scores, selected=selected, restarts=selection)
    weights = analysis._weights(GROUPS, len(_roots("confirmation")))
    alpha = .05 / (2 * (len(ARMS) - 1) * len(env.RULES))
    summary = {"scores_sha256": study.sha(score_file), "birth_groups": GROUPS,
               "episode_roots": len(_roots("confirmation")), "pairs_per_root": len(env.PAIRS),
               "interval_method": "simultaneous 95% across six contrasts; paired birth/world bootstrap",
               "rules": {}}
    for r, name in enumerate(env.RULES):
        values = selected[:, r, ..., 2]
        good = values > 0
        summary["rules"][name] = {
            "accuracy": dict(zip(ARMS, good.mean(axis=(0, 2, 3)).tolist(), strict=True)),
            "timeouts": dict(zip(ARMS, (values == 0).mean(axis=(0, 2, 3)).tolist(), strict=True)),
            "accuracy_by_birth": dict(zip(ARMS, good.mean(axis=(2, 3)).T.tolist(), strict=True)),
            "accuracy_all_restarts": (scores[:, :, r, ..., 2] > 0).mean(axis=(3, 4)).tolist(),
            "paired_contrasts": {arm: analysis._estimate(good[:, i], good[:, 0], weights, alpha)
                                 for i, arm in enumerate(ARMS) if i},
        }
    study.write(out / "summary.json", summary)


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
