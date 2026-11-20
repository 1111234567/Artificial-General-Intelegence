"""Measure actual old-object current and intervene in subsequent motor behavior."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import native_rehearsal_graph as live
from . import native_rehearsal_reactivation_stats as stats


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/native-rehearsal-20261120"
EVIDENCE = ROOT / "docs/evidence/native-rehearsal-20261120"
PROTOCOL = EVIDENCE / "reactivation-protocol.md"
KINDS = ("single", "copy", "permuted", "guide_a", "guide_c")
BIRTHS = (731, 733, 735)
SEED = 451023793
ROOTS = 128
FINAL = 1536
MODES = ("live", "own", "swap_a", "swap_c", "rotated", "cut")
STATE_FIELDS = ("h", "fast", "trace")
LANES = 16


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fingerprint(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def motor(model, state, episode):
    emitted, payouts, margins = [], [], []
    for order in (0, 1):
        frame = replace(episode, first_query=torch.full_like(episode.first_query, order))
        side, _, pays, logit = live.queries(model, state, frame)
        object_orientation = episode.orientation[:, 3:5].flip(1) if order else episode.orientation[:, 3:5]
        target = episode.laws[:, (0, 2)] * object_orientation
        if not torch.equal(side * target, pays):
            raise AssertionError("Actual query payout exceeds the environment choice record")
        emitted.append(side.numpy().astype(np.int8))
        payouts.append(pays.numpy().astype(np.int8))
        margins.append(logit.numpy())
    return np.stack(emitted), np.stack(payouts), np.stack(margins)


def b_contact_current(model, state, seed_episode, expected, visit):
    """Trace a duplicate of the actual last B contact and verify its physical state."""
    _, right, recorded_side, earned = visit
    zero = right.new_zeros((len(right), 4))
    menu = torch.cat((-right, right, zero), dim=1)
    currents = []

    def tick(state, stimuli, n):
        for _ in range(n):
            state, back = model.tick(state, stimuli)
            currents.append(back)
        return state

    state = tick(state, menu, 3)
    side = torch.where(seed_episode.draws[:, 2] < model.logit(state).sigmoid(), 1.0, -1.0)
    if not torch.equal(side, recorded_side):
        raise AssertionError("Instrumented B contact changed the actual choice")
    touch = menu.clone()
    touch[:, 6], touch[:, 7] = side.lt(0), side.gt(0)
    state = tick(state, touch, 2)
    taste = touch.clone()
    taste[:, 8], taste[:, 9] = earned.relu(), (-earned).relu()
    state = tick(state, taste, 2)
    for field in STATE_FIELDS:
        torch.testing.assert_close(getattr(state, field), getattr(expected, field), rtol=0, atol=0)
    return torch.stack(currents)


def sibling_indices(e):
    flat = torch.arange(len(e.laws))
    for odd, bit in ((0, 4), (1, 2), (2, 1)):
        sibling = e.laws[flat ^ bit]
        expected = e.laws.clone()
        expected[:, odd] *= -1
        torch.testing.assert_close(sibling, expected, rtol=0, atol=0)
        torch.testing.assert_close(e.vectors[flat ^ bit], e.vectors, rtol=0, atol=0)
    if not torch.equal(e.laws[flat ^ 8], e.laws):
        raise AssertionError("A mirrors must have the same taste assignments")
    return flat ^ 4, flat ^ 2


def equal_dose_rotation(original, donor):
    """Rotate the replacement orthogonally to its baseline and to the displaced component."""
    rng = np.random.default_rng([SEED, 889394])
    stock = original.double()
    delta = donor.double() - stock
    axis = torch.nn.functional.normalize(stock, dim=-1, eps=1e-14)
    along = (delta * axis).sum(dim=-1, keepdim=True) * axis
    residual = delta - along
    response = torch.nn.functional.normalize(residual, dim=-1, eps=1e-14)
    new = torch.from_numpy(rng.standard_normal(stock.shape))
    new = new - (new * axis).sum(dim=-1, keepdim=True) * axis
    new = new - (new * response).sum(dim=-1, keepdim=True) * response
    new = torch.nn.functional.normalize(new, dim=-1, eps=1e-14)
    return (stock + along + new * residual.norm(dim=-1, keepdim=True)).to(original.dtype)


def energy_control(original, swap_a, swap_c, rotated):
    own = original.double()
    rms = {}
    for name, counterfactual in (("swap_a", swap_a), ("swap_c", swap_c), ("rotated", rotated)):
        replacement = counterfactual.double()
        rms[name] = (replacement - own).square().mean(dim=-1).sqrt().numpy()
        if name != "rotated":
            shape = (len(own), -1, LANES, own.shape[-1])
            torch.testing.assert_close(own.reshape(shape).square().sum(dim=(2, 3)),
                                       replacement.reshape(shape).square().sum(dim=(2, 3)), rtol=1e-12, atol=1e-12)
    no_a = swap_a.double()
    other = rotated.double()
    errors = ((other.norm(dim=-1) - no_a.norm(dim=-1)).abs().max().item(),
              ((other - own).norm(dim=-1) - (no_a - own).norm(dim=-1)).abs().max().item())
    if max(errors) > 5e-6:
        raise AssertionError(f"The rotated replacement lost its dose match: {errors}")
    return np.stack(list(rms.values())), np.asarray(errors)


@torch.no_grad()
def evaluate(model, e):
    before_b, last_b, _, visits = live.support(model, e)
    natural, _, currents = live.silence(model, last_b)
    favored = torch.stack([live.recorded_favorable(visits, obj) for obj in range(3)], dim=1)
    torch.testing.assert_close(favored, e.vectors * e.laws[..., None], rtol=0, atol=0)
    boards = {"live": natural}
    result = {}
    if currents is not None:
        flip_a, flip_c = sibling_indices(e)
        b_current = b_contact_current(model, before_b, e, last_b, visits[-1])
        swap_a, swap_c = currents[:, flip_a], currents[:, flip_c]
        rotated = equal_dose_rotation(currents, swap_a)
        dose_rms, match = energy_control(currents, swap_a, swap_c, rotated)
        for name, override in (("own", currents), ("swap_a", swap_a), ("swap_c", swap_c), ("rotated", rotated)):
            boards[name], _, _ = live.silence(model, last_b, override=override)
        boards["cut"], _, _ = live.silence(model, last_b, cut=True)
        self_override_error = [(getattr(natural, key) - getattr(boards["own"], key)).abs().max().item()
                               for key in STATE_FIELDS]
        result = dict(currents=currents.numpy(), b_contact_current=b_current.numpy(),
                      dose_rms=dose_rms, rotation_max_error=match, own_state_error=np.asarray(self_override_error))
    sides, pay, margins = (np.stack(values) for values in zip(*(motor(model, state, e) for state in boards.values())))
    if currents is not None and not np.array_equal(sides[0], sides[1]):
        raise AssertionError("Own-current clamp changed the control's actual choices")
    receipt_log = dict(study_obj=torch.stack([x[0] for x in visits], dim=1).numpy(),
                       study_right=torch.stack([x[1] for x in visits], dim=1).numpy(),
                       study_side=torch.stack([x[2] for x in visits], dim=1).numpy(),
                       study_taste=torch.stack([x[3] for x in visits], dim=1).numpy())
    fields = {key: getattr(e, key).numpy() for key in ("vectors", "laws", "orientation", "draws", "first_old", "angle")}
    return dict(**fields, **receipt_log, **result, favorable=favored.numpy(),
                modes=np.asarray(list(boards)), sides=sides, pay=pay, margins=margins)


def run_one(cell, *, roots, step, folder):
    torch.set_num_threads(1)
    kind, birth = cell
    checkpoint = DATA / kind / f"seed-{birth}.pt"
    seal = fingerprint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["kind"], saved["birth"], saved["step"]) != (kind, birth, step) or fingerprint(checkpoint) != seal:
        raise ValueError(f"Expected immutable final checkpoint {kind}/{birth}/{step}")
    model = live.NativeGraph(birth, saved["layout"])
    model.load_state_dict(saved["state"])
    path = folder / f"{kind}-{birth}.npz"
    array = evaluate(model, live.scenes(SEED, roots, evaluation=True))
    np.savez_compressed(path, **array, kind=kind, birth=birth, step=step, checkpoint_sha256=seal["sha256"])
    print(f"reactivation {kind}/{birth} step {step} complete", flush=True)
    return dict(kind=kind, birth=birth, checkpoint=seal, raw=fingerprint(path))


def summarize(label, manifest):
    rows = {}
    expected = live.scenes(manifest["seed"], manifest["roots"], evaluation=True)
    for record in manifest["cells"]:
        path = ROOT / record["raw"]["path"]
        if fingerprint(path) != record["raw"]:
            raise ValueError(f"Raw record changed: {path}")
        with np.load(path, allow_pickle=False) as row:
            if (str(row["kind"]), int(row["birth"]), int(row["step"]), str(row["checkpoint_sha256"])) != (
                    record["kind"], record["birth"], manifest["step"], record["checkpoint"]["sha256"]):
                raise ValueError(f"Raw provenance mismatch: {path}")
            for field in ("vectors", "laws", "orientation", "draws", "first_old", "angle"):
                np.testing.assert_array_equal(row[field], getattr(expected, field).numpy())
            rows[record["kind"], record["birth"]] = stats.measure(row, manifest["roots"])
    result = stats.report(rows, manifest)
    write_json(EVIDENCE / f"reactivation-{label}.json", dict(manifest=manifest, **result))


def run(label, roots, step, workers, summarize_only):
    folder = DATA / f"reactivation-{label}"
    index = folder / "manifest.json"
    if summarize_only:
        summarize(label, json.loads(index.read_text(encoding="utf-8")))
        return
    folder.mkdir(parents=True, exist_ok=True)
    order = [(kind, birth) for kind in KINDS for birth in BIRTHS]
    measure = partial(run_one, roots=roots, step=step, folder=folder)
    if workers == 1:
        cells = list(map(measure, order))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cells = list(pool.map(measure, order))
    files = (PROTOCOL, Path(__file__), Path(stats.__file__), Path(live.__file__))
    manifest = dict(roots=roots, step=step, seed=SEED, kinds=KINDS, births=BIRTHS,
                    source=[fingerprint(path) for path in files], cells=cells)
    write_json(index, manifest)
    summarize(label, manifest)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="heldout")
    parser.add_argument("--roots", type=int, default=ROOTS)
    parser.add_argument("--step", type=int, default=FINAL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    run(options.label, options.roots, options.step, options.workers, options.summarize_only)
