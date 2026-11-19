"""Test whether an activity pulse predicts the sign of a real local graph write."""

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn

from . import self_write_view as graph
from .self_write_view_study import paired_interval


DIRECTORY = Path("docs/evidence/self-writing-2026-11-19/view")
MODES = ("sham", "current_structure", "true_pulse", "yoked_pulse")
PROBE = 0.0625
TICKS = 4
DEVELOPMENT_SEEDS = tuple(range(286141, 286147))
TEST_SEEDS = tuple(range(715314, 715326))
TEST_BATCH = 270
DEVELOPMENT_BATCH = 189
DEVELOPMENT_ROUNDS = 320
RECEIVER = np.concatenate((graph.RECEIVERS, np.arange(graph.WIDTH)))
SENDER = np.concatenate((graph.SENDERS, np.zeros(graph.WIDTH, dtype=np.int64)))
EDGE_PORTS = np.flatnonzero(graph.RECEIVERS != 0)
BIAS_PORTS = graph.EDGES + np.tile(np.arange(1, graph.WIDTH), graph.WIDTH - 1)
PORTS = np.concatenate((EDGE_PORTS, EDGE_PORTS, BIAS_PORTS))
KINDS = np.repeat((0, 1, 2), len(EDGE_PORTS))
FEATURES = 3 * graph.SLOTS + graph.WIDTH * (1 + 2 * TICKS)


@dataclass(frozen=True)
class Case:
    body: graph.Body
    port: np.ndarray
    start: np.ndarray
    kind: np.ndarray
    donor: np.ndarray


def birth(seed, batch):
    if batch < 2 * len(PORTS):
        raise ValueError("Each candidate type and port needs at least two bodies")
    rng = np.random.default_rng(seed)
    choices = rng.permutation(np.resize(np.arange(len(PORTS)), batch))
    port = PORTS[choices]
    kind = KINDS[choices]
    rows = np.arange(batch)
    exists = rng.random((batch, graph.SLOTS)) < 0.55
    exists[:, graph.EDGES:] = True
    exists[rows, port] = kind != 1
    value = rng.uniform(-0.8, 0.8, size=exists.shape).astype(np.float32) * exists
    start = rng.uniform(-1, 1, size=(batch, graph.WIDTH)).astype(np.float32)
    donor = rows.copy()
    for slot in np.unique(port):
        for subtype in np.unique(kind[port == slot]):
            group = rows[(port == slot) & (kind == subtype)]
            order = rng.permutation(group)
            donor[order] = np.roll(order, 1)
    return Case(graph.Body(exists, value), port, start, kind, donor)


def advance(body, start, pulse):
    matrix = np.zeros((len(start), graph.WIDTH, graph.WIDTH), dtype=np.float32)
    matrix[:, graph.RECEIVERS, graph.SENDERS] = body.values[:, :graph.EDGES] * body.exists[:, :graph.EDGES]
    bias = body.values[:, graph.EDGES:]
    state = start
    path = []
    for tick in range(TICKS):
        drive = np.einsum("brs,bs->br", matrix, state) + bias
        if tick == 0:
            drive += pulse
        state = np.tanh(drive)
        path.append(state)
    return np.stack(path, axis=1)


def observations(case):
    rows = np.arange(len(case.port))
    zero = np.zeros_like(case.start)
    pulse = zero.copy()
    # Test a positive write on the first transition only. For an edge, its
    # effective dose depends on the same sender activity a real edge would see.
    contact = np.where(case.port >= graph.EDGES, 1, case.start[rows, SENDER[case.port]])
    pulse[rows, RECEIVER[case.port]] = PROBE * contact
    baseline = advance(case.body, case.start, zero)
    sham = advance(case.body, case.start, zero)
    tested = advance(case.body, case.start, pulse)
    address = np.eye(graph.SLOTS, dtype=np.float32)[case.port]
    public = np.concatenate((address, case.start, baseline.reshape(len(rows), -1)), axis=1)
    structure = np.concatenate((case.body.exists, case.body.values), axis=1)
    sham_response = ((sham - baseline) / PROBE).reshape(len(rows), -1)
    response = ((tested - baseline) / PROBE).reshape(len(rows), -1)
    blank = np.zeros_like(structure)
    return {"sham": np.concatenate((public, sham_response, blank), axis=1),
            "current_structure": np.concatenate((public, sham_response, structure), axis=1),
            "true_pulse": np.concatenate((public, response, blank), axis=1),
            "yoked_pulse": np.concatenate((public, response[case.donor], blank), axis=1)}, baseline, response


def actual_edit(case, positive):
    target_present = case.body.exists[np.arange(len(positive)), case.port]
    code = np.where(positive, 2, 1) + 4 * ~target_present
    receipt = graph.apply(case.body, case.port * graph.STRIDE + code)
    if not receipt.changed.all():
        raise AssertionError("Both predeclared signed actions must execute")
    return advance(receipt.body, case.start, np.zeros_like(case.start))


def motor(path):
    return path[:, -2:, 0].mean(axis=1)


def writer():
    return nn.Sequential(nn.Linear(FEATURES, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 2))


def train(seed):
    models = {}
    optimizers = {}
    for mode in MODES:
        torch.manual_seed(seed)
        models[mode] = writer()
        optimizers[mode] = torch.optim.Adam(models[mode].parameters(), lr=0.002)
    rng = np.random.default_rng(seed)
    for _ in range(DEVELOPMENT_ROUNDS):
        case = birth(int(rng.integers(1 << 62)), DEVELOPMENT_BATCH)
        signal, baseline, _ = observations(case)
        before = motor(baseline)
        actions = rng.integers(2, size=len(before))
        rewards = torch.from_numpy(6 * (motor(actual_edit(case, actions)) - before))
        for mode in MODES:
            output = models[mode](torch.from_numpy(signal[mode]))
            selected = output.gather(1, torch.from_numpy(actions[:, None]))[:, 0]
            loss = (selected - rewards).square().mean()
            optimizers[mode].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[mode].step()
    return models


def evaluate(models, case):
    signal, baseline, response = observations(case)
    before = motor(baseline)
    size = len(before)
    plus = motor(actual_edit(case, np.ones(size, dtype=bool))) - before
    minus = motor(actual_edit(case, np.zeros(size, dtype=bool))) - before
    choices = {"fixed_plus": np.ones(size, dtype=bool), "fixed_minus": np.zeros(size, dtype=bool),
               "pulse_rule": motor(response.reshape(size, TICKS, graph.WIDTH)) > 0,
               "oracle_sign": plus > minus}
    with torch.inference_mode():
        for mode, model in models.items():
            choices[mode] = model(torch.from_numpy(signal[mode])).argmax(dim=1).numpy() == 1
        choices["same_model_yoked_pulse"] = models["true_pulse"](
            torch.from_numpy(signal["yoked_pulse"])).argmax(dim=1).numpy() == 1
    effects = {mode: motor(actual_edit(case, choose)) - before for mode, choose in choices.items()}
    return {"plus": plus, "minus": minus, "choices": choices, "effects": effects}


def cases():
    return [birth(seed, TEST_BATCH) for seed in TEST_SEEDS]


def arrays(parts):
    return {name: np.stack([value for value in values]) for name, values in (
        ("exists", (case.body.exists for case in parts)), ("values", (case.body.values for case in parts)),
        ("port", (case.port for case in parts)), ("start", (case.start for case in parts)),
        ("kind", (case.kind for case in parts)), ("donor", (case.donor for case in parts)))}


def digest(parts):
    sha = hashlib.sha256()
    for name, data in arrays(parts).items():
        sha.update(name.encode())
        sha.update(data.tobytes())
    return sha.hexdigest()


def seal(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "probe-seal.json").open("x", encoding="utf-8") as fp:
        json.dump({"case_sha256": digest(cases()), "heldout_block_seeds": TEST_SEEDS,
                   "development_seeds": DEVELOPMENT_SEEDS, "heldout_batch": TEST_BATCH}, fp, indent=2)
        fp.write("\n")


def run(directory):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if (directory / "probe-results.json").exists() or (directory / "probe-run.zip").exists():
        raise FileExistsError("Probe outputs already exist")
    models = [train(seed) for seed in DEVELOPMENT_SEEDS]
    heldout = cases()
    if digest(heldout) != json.loads((directory / "probe-seal.json").read_text())["case_sha256"]:
        raise ValueError("Held-out body digest mismatch")
    panel = [[evaluate(editors, case) for case in heldout] for editors in models]
    effects = {mode: np.asarray([[row["effects"][mode] for row in block] for block in panel])
               for mode in panel[0][0]["effects"]}
    choices = {mode: np.asarray([[row["choices"][mode] for row in block] for block in panel])
               for mode in panel[0][0]["choices"]}
    plus = np.stack([result["plus"] for result in panel[0]])
    minus = np.stack([result["minus"] for result in panel[0]])
    best = plus > minus
    decisive = np.abs(plus - minus) > 1e-5
    all_cases = np.ones_like(decisive)
    kind = arrays(heldout)["kind"]
    report = {"cases_per_seed": int(best.size), "decisive_cases_per_seed": int(decisive.sum()),
              "positive_edit_improved": float((plus > 1e-5).mean()),
              "positive_edit_harmed": float((plus < -1e-5).mean()),
              "mode_metrics": {}, "paired_true_pulse_gains": {}}
    sign_ok = {mode: sign == best for mode, sign in choices.items()}
    for mode, effect in effects.items():
        report["mode_metrics"][mode] = {"sign_accuracy_on_decisive": float(sign_ok[mode][:, decisive].mean()),
                                        "motor_change_mean": float(effect.mean()),
                                        "improved_rate": float((effect > 1e-5).mean()),
                                        "harm_rate": float((effect < -1e-5).mean()),
                                        "sign_accuracy_by_type": {name: float(sign_ok[mode][:, decisive & (kind == i)].mean())
                                                                  for i, name in enumerate(("edge_weight", "add_edge", "bias"))}}
    for mode in ("sham", "current_structure", "yoked_pulse", "same_model_yoked_pulse"):
        report["paired_true_pulse_gains"][mode] = {"sign_accuracy": paired_interval(sign_ok["true_pulse"], sign_ok[mode], decisive),
                                                 "motor_change": paired_interval(effects["true_pulse"], effects[mode], all_cases)}
    raw = BytesIO()
    np.savez_compressed(raw, **arrays(heldout), plus=plus, minus=minus,
                        **{f"effect__{mode}": value for mode, value in effects.items()},
                        **{f"choice__{mode}": value for mode, value in choices.items()})
    weights = BytesIO()
    np.savez_compressed(weights, **{f"{seed}__{mode}__{name}": tensor.detach().numpy()
                                     for seed, group in zip(DEVELOPMENT_SEEDS, models, strict=True)
                                     for mode, module in group.items() for name, tensor in module.state_dict().items()})
    with ZipFile(directory / "probe-run.zip", "x") as archive:
        archive.writestr("rounds.npz", raw.getvalue())
        archive.writestr("weights.npz", weights.getvalue())
    report["archive_sha256"] = hashlib.sha256((directory / "probe-run.zip").read_bytes()).hexdigest()
    with (directory / "probe-results.json").open("x", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "run"))
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    options = parser.parse_args()
    (seal if options.phase == "seal" else run)(options.directory)
