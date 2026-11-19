"""Read current local morphology into the native 16-node recurrent writer."""

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import self_write_core as core


DIRECTORY = Path("docs/evidence/self-writing-2026-11-19/look")
PORT = (11, 9)
READ_NODES = slice(6, 9)
SIGNAL_NODES = slice(12, 14)
INSPECTION_TICKS = 3
SETTLE_TICKS = 1
VIEWS = ("own", "bias_only", "sham", "yoked")
BOOT_SEEDS = tuple(range(36811, 36816))
TEST_SEEDS = tuple(range(73900, 73912))
TEST_BATCH = 128
BOOT_STEPS = 48
BOOT_BATCH = 80
HEAD_SIZE = 8
DOSE = np.float32(0.17)


@dataclass(frozen=True)
class Encounter:
    graph: core.Graph
    odor: np.ndarray
    donor: np.ndarray


def encounter(seed, batch):
    """Native randomized bodies with a calibrated two-odor motor fixture."""
    rng = np.random.default_rng(seed)
    bodies = [core.spawn_graph(int(subseed)) for subseed in rng.integers(1 << 50, size=batch)]
    body = core.Graph(np.concatenate([item.w for item in bodies]) * 0.42,
                      np.concatenate([item.b for item in bodies]) * 0.36,
                      np.concatenate([item.mask for item in bodies]))
    present = rng.random(batch) < 0.65
    body.mask[:, PORT[0], PORT[1]] = present
    body.w[:, PORT[0], PORT[1]] = rng.uniform(-0.72, 0.72, batch) * present
    body.b[:, PORT[0]] = rng.uniform(-0.41, 0.41, batch)
    body.mask[:, 10, 0] = body.mask[:, 11, 3] = True
    body.w[:, 10, 0] = body.w[:, 11, 3] = 0.46

    odor = rng.uniform(0, 0.12, (batch, 2, core.N)).astype(np.float32)
    odor[:, :, 6:] = 0
    odor[:, 0, 0] = rng.uniform(0.78, 1.12, batch)
    odor[:, 1, 3] = rng.uniform(0.78, 1.12, batch)
    odor[:, :, PORT[1]] = 0.93
    perm = rng.permutation(batch)
    donor = np.empty(batch, dtype=np.int64)
    donor[perm] = np.roll(perm, 1)
    return Encounter(body, odor, donor)


def with_head(body, head):
    """Install eight native scalars on read-port afferents and writer biases."""
    result = body.clone()
    result.w[:, SIGNAL_NODES, READ_NODES] = head[:6].reshape(2, 3)
    result.b[:, SIGNAL_NODES] = head[6:]
    return result


def read_stimulus(body, view, donor):
    target, source = PORT
    stimulus = np.zeros_like(body.b)
    if view == "sham":
        return stimulus
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}")
    sensed = body if view != "yoked" else body.lanes(donor)
    stimulus[:, 8] = sensed.b[:, target]
    if view != "bias_only":
        stimulus[:, 6] = sensed.mask[:, target, source]
        stimulus[:, 7] = sensed.w[:, target, source] * sensed.mask[:, target, source]
    return stimulus


def inspect(body, stimulus):
    h = core.advance(body, core.rest(body), stimulus, INSPECTION_TICKS)
    return core.advance(body, h, np.zeros_like(stimulus), SETTLE_TICKS)


def author(h):
    """The actuator decodes only activities from two native writer nodes."""
    index = np.clip(np.rint(4 * (h[:, 13] - h[:, 12])), -2, 2)
    amount = (DOSE * index).astype(np.float32)
    bias_delta = np.zeros_like(h)
    bias_delta[:, PORT[0]] = amount
    empty_address = np.full(len(h), -1, dtype=np.int64)
    empty_score = np.zeros(len(h), dtype=np.float32)
    return core.Proposal(np.zeros((len(h), core.N, core.N), dtype=np.float32), bias_delta,
                         empty_address, empty_score, empty_address, empty_score, empty_score)


def contact(body, odor):
    """Both contacts start from rest on the same body after the inspection phase."""
    left = core.advance(body, core.rest(body), odor[:, 0], core.MOTOR_TICKS)
    right = core.advance(body, core.rest(body), odor[:, 1], core.MOTOR_TICKS)
    margins = np.column_stack((core.motor(left), core.motor(right)))
    choices = margins > 0
    rewards = choices == np.array([False, True])
    return margins, choices, rewards


def execute(body, h, odor, *, allow=True):
    proposed = author(h)
    edited, receipt = core.apply(body, proposed, arm="active" if allow else "none")
    margins, choices, rewards = contact(edited, odor)
    return {"h": h, "proposal": proposed.bias_delta[:, PORT[0]], "write": receipt.bias_delta[:, PORT[0]],
            "margins": margins, "choices": choices, "rewards": rewards}


def run_view(case, head, view):
    body = with_head(case.graph, head)
    h = inspect(body, read_stimulus(body, view, case.donor))
    return execute(body, h, case.odor)


def objective(case, head, view):
    record = run_view(case, head, view)
    return record["rewards"].mean() - 0.012 * (record["write"] != 0).mean()


def train(seed, view):
    """Black-box birth search on realized contacts, never on a target edit."""
    random = np.random.default_rng(seed)
    head = np.zeros(HEAD_SIZE, dtype=np.float32)
    finalists = [head]
    for step in range(BOOT_STEPS):
        case = encounter(int(random.integers(1 << 55)), BOOT_BATCH)
        scale = 0.36 - 0.24 * step / BOOT_STEPS
        direction = random.normal(0, scale, (6, HEAD_SIZE)).astype(np.float32)
        candidates = np.clip(np.vstack((head, head + direction, head - direction)), -2.0, 2.0)
        scores = [objective(case, candidate, view) for candidate in candidates]
        head = candidates[np.argmax(scores)]
        if (step + 1) % 6 == 0:
            finalists.append(head)
    validate = encounter(int(random.integers(1 << 55)), 768)
    scores = [objective(validate, candidate, view) for candidate in finalists]
    return finalists[np.argmax(scores)], float(max(scores))


def arrays(cases):
    return {"w": np.stack([case.graph.w for case in cases]), "b": np.stack([case.graph.b for case in cases]),
            "mask": np.stack([case.graph.mask for case in cases]), "odor": np.stack([case.odor for case in cases]),
            "donor": np.stack([case.donor for case in cases])}


def case_digest(cases):
    digest = hashlib.sha256()
    for name, array in arrays(cases).items():
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_cases():
    return [encounter(seed, TEST_BATCH) for seed in TEST_SEEDS]


def seal(directory):
    directory.mkdir(parents=True, exist_ok=True)
    record = {"body_digest": case_digest(test_cases()), "test_seeds": TEST_SEEDS,
              "test_batch": TEST_BATCH, "boot_seeds": BOOT_SEEDS}
    with (directory / "seal.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(record, indent=2) + "\n")


def evaluate(case, heads):
    results = {view: run_view(case, heads[view], view) for view in VIEWS}
    body = with_head(case.graph, heads["own"])
    for view in VIEWS[1:]:
        h = inspect(body, read_stimulus(body, view, case.donor))
        results[f"own_head_{view}"] = execute(body, h, case.odor)
    for view in VIEWS:
        halted_body = with_head(case.graph, heads[view])
        results[f"{view}_writes_blocked"] = execute(halted_body, results[view]["h"], case.odor, allow=False)
    sham = results["own_head_sham"]["h"].copy()
    sham[:, SIGNAL_NODES] = results["own"]["h"][:, SIGNAL_NODES]
    results["own_writer_state_transplant"] = execute(body, sham, case.odor)
    return results


def metrics(record):
    reward = record["rewards"]
    written = record["write"]
    return {"choices_correct": float(reward.mean()), "left_correct": float(reward[..., 0].mean()),
            "right_correct": float(reward[..., 1].mean()), "both_odors_correct": float(reward.all(axis=-1).mean()),
            "write_rate": float((written != 0).mean()), "positive_writes": float((written > 0).mean()),
            "negative_writes": float((written < 0).mean())}


def paired(first, other):
    delta = (first.astype(float) - other).mean(axis=tuple(range(2, first.ndim)))
    rng = np.random.default_rng(807130)
    roots, blocks = delta.shape
    seed_counts = rng.multinomial(roots, np.full(roots, 1 / roots), size=2000)
    block_counts = rng.multinomial(blocks, np.full(blocks, 1 / blocks), size=2000)
    estimates = np.einsum("bs,sg,bg->b", seed_counts, delta, block_counts) / (roots * blocks)
    return {"difference": float(delta.mean()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist()}


def run(directory):
    if (directory / "summary.json").exists() or (directory / "run.zip").exists():
        raise FileExistsError("Look evidence already exists")
    training = [{view: train(seed, view) for view in VIEWS} for seed in BOOT_SEEDS]
    heads = [{view: value[0] for view, value in row.items()} for row in training]
    pending = test_cases()
    seal_record = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    if case_digest(pending) != seal_record["body_digest"]:
        raise ValueError("Held-out body digest mismatch")
    data = [[evaluate(case, head) for case in pending] for head in heads]
    records = {view: {field: np.asarray([[block[view][field] for block in row] for row in data])
                      for field in record} for view, record in data[0][0].items()}
    own = records["own"]
    contrasts = {}
    for view in ("sham", "bias_only", "yoked", "own_head_sham", "own_head_bias_only", "own_head_yoked",
                 "own_writes_blocked", "own_writer_state_transplant"):
        compared = records[view]
        contrasts[view] = {"all_state_rms": float(np.sqrt(np.mean((own["h"] - compared["h"]) ** 2))),
                           "writer_state_rms": float(np.sqrt(np.mean((own["h"][..., SIGNAL_NODES]
                                                                     - compared["h"][..., SIGNAL_NODES]) ** 2))),
                           "proposal_disagree": float((own["proposal"] != compared["proposal"]).mean()),
                           "actual_write_disagree": float((own["write"] != compared["write"]).mean()),
                           "physical_choice_disagree": float((own["choices"] != compared["choices"]).mean()),
                           "physical_accuracy_gain": paired(own["rewards"], compared["rewards"])}
    restored = records["own_writer_state_transplant"]
    for field in ("proposal", "write", "margins", "choices", "rewards"):
        np.testing.assert_array_equal(own[field], restored[field])
    report = {"heldout_cases_per_root": TEST_BATCH * len(TEST_SEEDS), "roots": BOOT_SEEDS,
              "views": {name: metrics(record) for name, record in records.items()},
              "contrasts_to_own": contrasts,
              "selected_dev_reward_less_write_cost": [{view: value[1] for view, value in row.items()} for row in training]}
    archive_data = {f"{view}__{field}": array for view, record in records.items() for field, array in record.items()}
    payload = BytesIO()
    head_array = np.asarray([[head[view] for view in VIEWS] for head in heads])
    np.savez_compressed(payload, **arrays(pending), heads=head_array, **archive_data)
    with ZipFile(directory / "run.zip", "x") as zipped:
        zipped.writestr("records.npz", payload.getvalue())
    report["archive_sha256"] = hashlib.sha256((directory / "run.zip").read_bytes()).hexdigest()
    report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), Path(core.__file__))}
    with (directory / "summary.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "run"))
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    (seal if args.phase == "seal" else run)(args.directory)
