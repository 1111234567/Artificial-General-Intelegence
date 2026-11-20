"""Quantize only the fixed birth template in the four-study absent-A assay."""

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity_report as stats, interleave_homeostasis as home
from . import interleave_slow_reference as slow, interleave_train as common
from . import local_trace_graph as net, local_trace_train as old, revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE, ROOT


LABEL = "quantized-reference"
OUT = DATA / LABEL
REPORT = EVIDENCE / f"{LABEL}.json"
AGES = (16, 64, 128)
FIELDS = slow.FIELDS[:6]
PRIMARY = ("new_X", "all_three")
GAMMA = .12
MARGIN = .02
DRAWS = 4000
STEP = 8000
FORMATS = ("q2", "q3", "q4", "q6", "q8", "fp16", "exact", "sparse40")
STORAGE = dict(zip(FORMATS, (185, 238, 290, 395, 500, 840, 1680, 290), strict=True))
WIDTH = net.NODES
SHAPE = (WIDTH, WIDTH + 1)
COUNT = WIDTH * (WIDTH + 1)
SPARSE_COUNT = 40
COHORTS = {
    "original": dict(source="retention-objective", births=home.CAL_BIRTHS, seed=7341941, domain=738881),
    "confirmation": dict(source="retention-objective-confirmation", births=home.CONFIRM_BIRTHS,
                         seed=7341983, domain=738889),
}


@dataclass(frozen=True)
class Reference:
    values: torch.Tensor
    eligible: torch.Tensor


def pack_words(words, width):
    words = np.asarray(words, dtype=np.uint16).reshape(-1)
    if np.any(words >= 1 << width):
        raise ValueError("Word exceeds its packed width")
    shifts = np.arange(width, dtype=np.uint16)
    bits = ((words[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    return np.packbits(bits, bitorder="little").tobytes()


def unpack_words(payload, width, count):
    if len(payload) != (width * count + 7) // 8:
        raise ValueError("Incorrect packed word count")
    bits = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder="little")
    if bits[width * count:].any():
        raise ValueError("Nonzero padding")
    shifts = np.arange(width, dtype=np.uint16)
    return (bits[:width * count].reshape(count, width).astype(np.uint16) << shifts).sum(
        axis=1, dtype=np.uint16)


def birth_values(graph):
    valid = (graph.present & graph.weights.ne(0))[0].numpy()
    values = np.concatenate((np.where(valid, graph.weights[0].detach().numpy(), 0.),
                             graph.bias[0].detach().numpy()[:, None]), axis=1)
    if values.dtype != np.float32 or values.shape != SHAPE or not np.isfinite(values).all():
        raise ValueError("Expected finite fp32 parameters in one 20-node birth")
    return values, valid


def encode(values, valid, mode):
    available = np.c_[valid, np.ones(WIDTH, dtype=bool)]
    if mode.startswith("q"):
        bits = int(mode[1:])
        limit = (1 << (bits - 1)) - 1
        scale = (np.max(np.abs(values), axis=1) / limit).astype("<f4")
        normalized = np.divide(values, scale[:, None], out=np.zeros_like(values), where=scale[:, None] != 0)
        signed = np.rint(normalized).clip(-limit, limit).astype(np.int16)
        words = np.where(available, signed + limit, (1 << bits) - 1)
        return scale.tobytes() + pack_words(words, bits)
    if mode in ("fp16", "exact"):
        dtype, word_type, sentinel = {"fp16": ("<f2", "<u2", 0x7E00),
                                      "exact": ("<f4", "<u4", 0x7FC00000)}[mode]
        numbers = values.astype(dtype)
        if not np.isfinite(numbers).all():
            raise ValueError("Source values exceed the floating reference range")
        words = numbers.view(word_type)
        words[~available] = sentinel
        return words.tobytes()
    if mode != "sparse40":
        raise ValueError(f"Unsupported reference {mode!r}")
    weights = values[:, :-1].reshape(-1)
    dense = np.flatnonzero(valid)
    if len(dense) < SPARSE_COUNT:
        raise ValueError("Too few eligible weights for the fixed-size sparse comparator")
    largest = np.lexsort((dense, -np.abs(weights[dense])))[:SPARSE_COUNT]
    chosen = np.sort(dense[largest])
    return (values[:, -1].astype("<f4").tobytes() + pack_words(chosen, 9)
            + weights[chosen].astype("<f4").tobytes() + bytes(5))


def decode(payload, mode):
    if len(payload) != STORAGE[mode]:
        raise ValueError(f"Unexpected reference storage for {mode}")
    if mode.startswith("q"):
        bits = int(mode[1:])
        limit = (1 << (bits - 1)) - 1
        scale = np.frombuffer(payload[:80], "<f4")
        words = unpack_words(payload[80:], bits, COUNT).reshape(SHAPE)
        available = words != (1 << bits) - 1
        number = (words.astype(np.int16) - limit).astype(np.float32) * scale[:, None]
    elif mode in ("fp16", "exact"):
        dtype, word_type, sentinel = {"fp16": ("<f2", "<u2", 0x7E00),
                                      "exact": ("<f4", "<u4", 0x7FC00000)}[mode]
        number = np.frombuffer(payload, dtype).astype(np.float32).reshape(SHAPE)
        available = np.frombuffer(payload, word_type).reshape(SHAPE) != sentinel
        if not np.isfinite(number[available]).all():
            raise ValueError("Invalid floating reference word")
    else:
        chosen = unpack_words(payload[80:125], 9, SPARSE_COUNT)
        if np.any(chosen >= WIDTH**2) or len(np.unique(chosen)) != SPARSE_COUNT or any(payload[285:]):
            raise ValueError("Invalid sparse reference addresses or padding")
        weights = np.zeros(WIDTH**2, dtype=np.float32)
        weights[chosen] = np.frombuffer(payload[125:285], "<f4")
        available_w = np.zeros(WIDTH**2, dtype=bool)
        available_w[chosen] = True
        number = np.c_[weights.reshape(WIDTH, WIDTH), np.frombuffer(payload[:80], "<f4")]
        available = np.c_[available_w.reshape(WIDTH, WIDTH), np.ones(WIDTH, dtype=bool)]
    if not available[:, -1].all():
        raise ValueError("Every bias is an addressed coordinate")
    return Reference(torch.from_numpy(np.where(available, number, 0.)),
                     torch.from_numpy(available[:, :-1].copy()))


def remap(post, reference, gamma):
    """Decode zero as a numeric target at an eligible, currently expressed address."""
    rows = torch.cat((post.weights * post.present, post.bias.unsqueeze(-1)), dim=-1)
    active = torch.cat((reference.eligible & post.present,
                        torch.ones_like(rows[..., -1:], dtype=torch.bool)), dim=-1)
    amended = torch.where(active, rows.lerp(reference.values, gamma), rows)
    return replace(post, weights=torch.where(post.present, net._nonzero(amended[..., :-1]), post.weights),
                   bias=amended[..., -1])


def contact(rule, state, right, factor, draw, reference, gamma):
    action = net.act(rule, state, -right, right, draw[:, 0])
    taste = action.side * factor
    sensed = net.observe(rule, action, -right, right, taste)
    committed = net.commit(rule, sensed, draw[:, 1:])
    adjusted = remap(committed.state.graph, reference, gamma)
    ready = net.wait(rule, sensed.state, net.PRE_WRITE_TICKS)
    written = net.wait(rule, replace(ready, graph=adjusted), net.POST_WRITE_TICKS)
    return net.wait(rule, written, home.WAIT), taste


@dataclass(frozen=True)
class Physical:
    right: torch.Tensor
    factor: torch.Tensor
    draw: torch.Tensor
    x: torch.Tensor
    x_factor: torch.Tensor
    x_draw: torch.Tensor
    menus: torch.Tensor
    choices: torch.Tensor
    target: torch.Tensor
    worlds: int


def physical(tapes):
    tape, extra, extra_orientation, extra_draws = tapes
    visits = tape.visits[home.ECOLOGIES.index("absent_a")]
    ids = torch.cat((visits[:2], extra, visits[2:]))
    orientation = torch.cat((tape.orientation[:2], extra_orientation, tape.orientation[2:]))
    draw = torch.cat((tape.draws[:2], extra_draws, tape.draws[2:]))
    n = tape.pairs
    rows = torch.arange(n)[None, :]
    right = (tape.objects[rows, ids] * orientation[..., None]).repeat_interleave(4, 1)
    ab_laws = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]).repeat(n, 1)
    sign = torch.where(ids.repeat_interleave(4, 1) == 0, ab_laws[:, 0], ab_laws[:, 1])
    factor = (tape.laws[rows, ids] * orientation).repeat_interleave(4, 1) * sign
    xlaw = torch.tensor([-1., 1.]).repeat(4 * n)
    X = (tape.objects[:, 2] * tape.x_orientation[:, None]).repeat_interleave(8, 0)
    x_factor = tape.x_orientation.repeat_interleave(8) * xlaw
    menus = (tape.objects[:, :3] * tape.probe_orientation[..., None]).repeat_interleave(8, 0)
    target = torch.cat(((tape.laws[:, :2].repeat_interleave(4, 0) * ab_laws).repeat_interleave(2, 0),
                        xlaw[:, None]), dim=1) * tape.probe_orientation.repeat_interleave(8, 0)
    return Physical(right, factor, draw.repeat_interleave(4, 1), X, x_factor, tape.x_draws.repeat_interleave(8, 0),
                    menus, tape.probe_draws.repeat_interleave(8, 0), target, n)


@torch.no_grad()
def replay(graph, rule, tape, reference, ages=AGES):
    lanes = tape.worlds * 4
    twins = torch.arange(lanes).repeat_interleave(2)
    state = net.start(graph, lanes)
    view, expanded = rule.expand(lanes), rule.expand(2 * lanes)
    gamma = graph.bias.new_tensor(GAMMA)
    sides, before, receipt = [], [], []
    for contact_index in range(8 + max(ages)):
        state, _ = contact(view, state, tape.right[contact_index], tape.factor[contact_index],
                           tape.draw[contact_index], reference, gamma)
        if contact_index - 7 not in ages:
            continue
        snapshot = state.subset(twins)
        old_side = motor.act(snapshot, -tape.menus[:, 0], tape.menus[:, 0], tape.choices[:, 0]).side
        acquired, earned = contact(expanded, snapshot, tape.x, tape.x_factor, tape.x_draw, reference, gamma)
        chosen = torch.stack([motor.act(acquired, -tape.menus[:, i], tape.menus[:, i],
                                        tape.choices[:, i]).side for i in range(3)], 1)
        sides.append(chosen.reshape(tape.worlds, 4, 2, 3).to(torch.int8).numpy())
        before.append(old_side.reshape(tape.worlds, 4, 2).to(torch.int8).numpy())
        receipt.append(earned.reshape(tape.worlds, 4, 2).to(torch.int8).numpy())
    side = np.stack(sides)
    old = np.stack(before)
    target = tape.target.reshape(tape.worlds, 4, 2, 3).to(torch.int8).numpy()
    correct = side == target
    bits = np.concatenate(((old == target[..., 0])[..., None], correct, correct.all(axis=-1)[..., None],
                           (correct[..., 0] & correct[..., 2])[..., None]), axis=-1)
    return dict(side=side, old_before_side=old, X_taste=np.stack(receipt), correct=bits, target=target)


def _hash(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def protocol(out, worlds):
    cohort_paths = {}
    for label, key in COHORTS.items():
        directory = DATA / key["source"]
        paths = [old.checkpoint_path(directory, "odor", "uniform", birth, STEP) for birth in key["births"]]
        cohort_paths[label] = {**key, "parent_checkpoints": {
            path.relative_to(ROOT).as_posix(): _hash(path) for path in paths}}
    plan = dict(cohorts=cohort_paths, worlds=worlds, ages=AGES, metrics=FIELDS, equivalence_margin=MARGIN,
                step=STEP, gamma=GAMMA, bootstrap=DRAWS, primary=PRIMARY, formats=FORMATS, stored_bytes=STORAGE,
                dependencies={Path(path).name: _hash(path) for path in
                              (__file__, home.__file__, slow.__file__, net.__file__, stats.__file__)},
                decision="In each of two independent cohorts, average 16/64/128 equally within world. "
                         "For both X and A/B/X all correct, require the full paired crossed 95% interval versus "
                         "exact to lie strictly within [-.02,+.02]. Smallest qualifying tested dense payload wins. "
                         "Per-age and per-object results are also shown; they retain their own intervals.",
                physical="Four A and four B studies, then A absent with B repetition matching the published "
                         "compact-reference assay. Four crossed A/B laws and two X laws are averaged within each "
                         "geometry for resampling. Live parameters, traces, activities and native edits stay fp32. "
                         "After the native commit, current eligible addresses and every bias move .12 toward the "
                         "fixed decoded template and exact native post-write ticks are replayed.",
                addresses="Fixed row-major grid has 20 incoming source slots and one bias per receiver. "
                          "Only a nonzero original birth edge whose address is currently present is eligible "
                          "in dense formats. Later regrowth at a birth address restores that address's eligibility. "
                          "Store one reserved code or NaN sentinel for an absent birth address. "
                          "A zero produced by quantization is a legitimate target on an eligible address.",
                packing="Dense q formats store 20 fp32 symmetric max-absolute scales (80 B), then 420 little-endian "
                        "bit-packed words across all rows with zero end padding. Codes use round-to-nearest-even "
                        "and symmetric signed levels ±(2^(b-1)-1). Binary16/binary32 store 420 little-endian words "
                        "with unused NaN sentinels at absent weights. Mode is selected externally.",
                sparse="FP32 biases (80 B), forty lowest-index-tie-broken largest-magnitude eligible weights "
                       "(160 B), sorted 9-bit dense addresses (45 B), and 5 padding bytes. Only selected "
                       "weights receive the reference pull, with all others retaining native writes. "
                       "The sparse comparator occupies the same 290 B as q4.",
                memory_accounting="Actual .bin payload bytes. The live graph, temporary fp32 decode tensors, "
                                  "evaluation activities and container filesystem overhead are additional.")
    return home._lock(out / "run.json", plan)


def _save_pack(file, data):
    if file.exists():
        if file.read_bytes() != data:
            raise ValueError(f"Fixed template changed at {file}")
    else:
        common._atomic_save(file, lambda temporary: temporary.write_bytes(data))


def run_cohort(out, cohort, worlds):
    plan_hash = protocol(out, worlds)
    info = COHORTS[cohort]
    tapes = slow.public_tapes(worlds, info["seed"], info["domain"])
    inputs = physical(tapes)
    physical_archive = {name: value.numpy() for name, value in vars(tapes[0]).items()}
    physical_archive.update(extra_object=tapes[1].numpy(), extra_orientation=tapes[2].numpy(), extra_draw=tapes[3].numpy())
    home._save_raw(out / cohort / "worlds.npz", physical_archive)
    for birth in info["births"]:
        destination = out / cohort / f"birth-{birth}"
        raw = destination / "replay.npz"
        graph, rule = old.load_checkpoint(DATA / info["source"], "odor", "uniform", birth, STEP)
        values, valid = birth_values(graph)
        packs = {mode: encode(values, valid, mode) for mode in FORMATS}
        hashes, rounded, errors, addresses = [], [], [], []
        result = {}
        for mode, payload in packs.items():
            template_file = destination / f"{mode}.bin"
            _save_pack(template_file, payload)
            reference = decode(payload, mode)
            if mode != "sparse40":
                np.testing.assert_array_equal(reference.eligible.numpy(), valid)
            hashes.append(_hash(template_file))
            addresses.append(int(reference.eligible.sum()))
            rounded.append(int((reference.eligible & reference.values[:, :-1].eq(0)).sum()))
            represented = np.c_[reference.eligible.numpy(), np.ones(WIDTH, bool)]
            errors.append(float(np.abs(reference.values.numpy() - values)[represented].mean()))
            if raw.exists():
                continue
            started = perf_counter()
            row = replay(graph, rule, inputs, reference)
            for key, value in row.items():
                result.setdefault(key, []).append(value)
            print(f"{cohort} birth={birth} {mode} {len(payload)}B {perf_counter()-started:.1f}s", flush=True)
        if not raw.exists():
            home._save_raw(raw, dict(**{key: np.stack(value) for key, value in result.items()},
                                     formats=FORMATS, ages=AGES, protocol_sha256=plan_hash, packet_sha256=hashes,
                                     eligible_weights_rounded_to_zero=rounded, numeric_mae=errors,
                                     eligible_weight_addresses=addresses))
        else:
            with np.load(raw, allow_pickle=False) as saved:
                if saved["protocol_sha256"].item() != plan_hash or list(saved["packet_sha256"]) != hashes:
                    raise ValueError(f"Existing replay is incompatible at {raw}")


def _inside(bounds):
    return -MARGIN < bounds[0] and bounds[1] < MARGIN


def summarize(out, evidence, worlds):
    plan_hash = protocol(out, worlds)
    audit = {}
    results = {}
    exact = FORMATS.index("exact")
    for label, info in COHORTS.items():
        cohort, rows = out / label, []
        world_path = cohort / "worlds.npz"
        audit[world_path.relative_to(out).as_posix()] = _hash(world_path)
        diagnostics = {}
        for birth in info["births"]:
            dirpath = cohort / f"birth-{birth}"
            raw = dirpath / "replay.npz"
            with np.load(raw, allow_pickle=False) as data:
                if data["protocol_sha256"].item() != plan_hash or tuple(data["formats"]) != FORMATS:
                    raise ValueError(f"Replay protocol differs at {raw}")
                for mode, packet_sha in zip(FORMATS, data["packet_sha256"], strict=True):
                    path = dirpath / f"{mode}.bin"
                    if _hash(path) != packet_sha or path.stat().st_size != STORAGE[mode]:
                        raise ValueError(f"Packed payload failed verification at {path}")
                    audit[path.relative_to(out).as_posix()] = packet_sha
                success = data["side"] == data["target"][:, None]
                base = data["old_before_side"] == data["target"][:, None, ..., 0]
                checked = np.concatenate((base[..., None], success, success.all(-1)[..., None],
                                          (success[..., 0] & success[..., 2])[..., None]), axis=-1)
                np.testing.assert_array_equal(data["correct"], checked)
                if not np.all(np.abs(data["X_taste"]) == 1):
                    raise ValueError(f"Expected signed physically earned X taste at {raw}")
                rows.append(checked.mean((3, 4)))
                diagnostics[str(birth)] = {
                    mode: {"eligible_weight_addresses": int(addresses),
                           "eligible_weights_quantized_zero": int(zeros), "represented_address_mae": float(mae)}
                    for mode, addresses, zeros, mae in zip(
                        FORMATS, data["eligible_weight_addresses"], data["eligible_weights_rounded_to_zero"],
                        data["numeric_mae"], strict=True)}
            audit[raw.relative_to(out).as_posix()] = _hash(raw)
        scores = np.stack(rows)
        ci = stats.Crossed(len(info["births"]), worlds, info["seed"] + 559, DRAWS)
        summary = {}
        for mode_index, mode in enumerate(FORMATS):
            cell = {}
            for age, values in [("pooled", scores.mean(axis=2)), *[(str(age), scores[:, :, i]) for i, age in enumerate(AGES)]]:
                u, v = values[:, mode_index], values[:, exact]
                cell[age] = {field: {"score": ci(u[..., i]), "minus_exact": ci((u - v)[..., i])}
                             for i, field in enumerate(FIELDS)}
                for value in cell[age].values():
                    value["within_margin"] = _inside(value["minus_exact"]["ci95"])
            cell["pooled_primary_equivalent"] = all(cell["pooled"][field]["within_margin"] for field in PRIMARY)
            summary[mode] = cell
        results[label] = {"outcomes": summary, "template_diagnostics": diagnostics}
    equivalent = [mode for mode in FORMATS[:-1]
                  if all(data["outcomes"][mode]["pooled_primary_equivalent"] for data in results.values())]
    raw = out.resolve()
    report = dict(protocol_sha256=plan_hash, plan=json.loads((out / "run.json").read_text(encoding="utf8")),
                  raw_path=str(raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw), raw_sha256=audit, results=results,
                  dense_modes_passing_both_cohorts=equivalent,
                  minimum_tested_payload=min(equivalent, key=STORAGE.__getitem__))
    home._lock(out / "summary.json", report)
    if evidence is not None:
        home._lock(evidence, report)
    print("Both-cohort equivalence:", equivalent, "minimum:", report["minimum_tested_payload"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "replay", "report"))
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--evidence", type=Path, default=REPORT)
    parser.add_argument("--worlds", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worlds < 2:
        parser.error("Use at least two independent geometries per cohort")
    if args.phase == "plan":
        print(protocol(args.out, args.worlds), flush=True)
    elif args.phase == "replay":
        if args.cohort is None:
            parser.error("Replay requires a fixed cohort")
        run_cohort(args.out, args.cohort, args.worlds)
    else:
        summarize(args.out, args.evidence, args.worlds)


if __name__ == "__main__":
    main()
