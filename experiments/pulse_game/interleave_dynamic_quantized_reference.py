"""Replay fixed-gain mutable reference precision with paired opposite-A swaps."""

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity_report as stats, interleave_homeostasis as home
from . import interleave_quantized_reference as frozen, interleave_slow_reference as slow
from . import interleave_train as common, local_trace_graph as net, local_trace_train as old
from . import revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE, ROOT


OUT = DATA / "dynamic-quantized-reference"
REPORT = EVIDENCE / "dynamic-quantized-reference.json"
AGES = (16, 64, 128)
ECOLOGIES = ("absent_a", "recurring")
GAMMA, BETA = .12, .02
MARGIN = .02
DRAWS = 4000
MODE_NAMES = ("fp32_slow", "q4_static", "q4_nearest", "q4_stochastic", "q8_nearest")
CHANNELS = ("weights", "biases")
COUNTS = ("eligible", "continuous_change", "outside_range", "at_range", "stored_change")
PHASES = ("prestudy", "intervening", "X")
BRANCHES = ("own", "opposite_A")
FIELDS = frozen.FIELDS
QUERY = ("A", "B", "X")
COHORTS = {
    "original": dict(source="retention-objective", births=home.CAL_BIRTHS, seed=7393001, domain=741113),
    "confirmation": dict(source="retention-objective-confirmation", births=home.CONFIRM_BIRTHS,
                         seed=7393019, domain=741127),
}


@dataclass(frozen=True)
class Mode:
    bits: int | None
    moving: bool
    stochastic: bool = False

    @property
    def maximum(self):
        return 2 ** (self.bits - 1) - 1 if self.bits else None


MODES = dict(zip(MODE_NAMES, (Mode(None, True), Mode(4, False), Mode(4, True), Mode(4, True, True),
                             Mode(8, True)), strict=True))


@dataclass(frozen=True)
class Memory:
    live: net.State
    ref: torch.Tensor
    ancestry: torch.Tensor

    def subset(self, lanes):
        return Memory(self.live.subset(lanes), self.ref[lanes], self.ancestry[lanes])


def storage(mode):
    bits = 32 if mode.bits is None else mode.bits
    values = (frozen.COUNT * bits + 7) // 8
    return {"values": values, "row_scales": net.NODES * 4, "packed_ancestry": (net.NODES**2 + 7) // 8,
            "total": values + net.NODES * 4 + (net.NODES**2 + 7) // 8}


def step_size(extent, mode):
    return extent / mode.maximum if mode.bits else extent


def code(values, extent, mode, uniform=None):
    if mode.bits is None:
        return values
    step = step_size(extent, mode)
    norm = (values / torch.where(step > 0, step, 1.)).clamp(-mode.maximum, mode.maximum)
    if uniform is not None:
        lower = norm.floor()
        return (lower + (uniform < (norm - lower))).to(torch.int8)
    return norm.round().to(torch.int8)


def decode(words, extent, mode):
    return words if mode.bits is None else words * step_size(extent, mode)


def pack_snapshot(memory, extent, mode, lane=0):
    """Serialize one frozen post-history reference and its continuous-survival mask."""
    raw = memory.ref[lane].numpy()
    if mode.bits is None:
        values = raw.astype("<f4").tobytes()
    else:
        values = frozen.pack_words(raw.astype(np.int16) + mode.maximum, mode.bits)
        recovered = frozen.unpack_words(values, mode.bits, frozen.COUNT).reshape(frozen.SHAPE)
        np.testing.assert_array_equal(recovered.astype(np.int16) - mode.maximum, raw)
    scale = extent.numpy().astype("<f4").tobytes()
    alive = frozen.pack_words(memory.ancestry[lane].numpy(), 1)
    np.testing.assert_array_equal(frozen.unpack_words(alive, 1, net.NODES**2).reshape(net.NODES, net.NODES),
                                  memory.ancestry[lane].numpy())
    packed = values + scale + alive
    if len(packed) != storage(mode)["total"]:
        raise AssertionError("Packed dynamic memory differs from storage declaration")
    return packed


def _initial(graph, lanes, mode):
    rows, valid = frozen.birth_values(graph)
    rows = torch.from_numpy(rows)
    extent = rows.abs().amax(1, keepdim=True)
    reference = code(rows, extent, mode)
    return Memory(net.start(graph, lanes), reference.expand(lanes, -1, -1),
                  torch.from_numpy(valid).expand(lanes, -1, -1)), extent


def _columns(mask):
    return torch.stack((mask[..., :-1].sum((1, 2)), mask[..., -1].sum(1)), dim=1)


def update_reference(state, fast, alive, extent, mode, uniform=None):
    prior = decode(state.ref, extent, mode)
    valid = torch.cat((alive, torch.ones_like(alive[..., :1])), dim=-1)
    count = _columns(valid)
    if not mode.moving:
        zeros = torch.zeros_like(count)
        at_range = state.ref.abs().eq(mode.maximum) if mode.bits else prior.abs().eq(extent)
        return state.ref, torch.stack((count, zeros, zeros, _columns(valid & at_range), zeros), -1)
    values = torch.cat((fast.weights, fast.bias.unsqueeze(-1)), dim=-1)
    proposal = prior.lerp(values, BETA)
    bounded = torch.clamp(proposal, min=-extent, max=extent)
    words = code(bounded, extent, mode, uniform)
    stored = torch.where(valid, words, state.ref)
    at_range = stored.abs().eq(mode.maximum) if mode.bits else stored.abs().eq(extent)
    events = (valid & proposal.ne(prior), valid & proposal.abs().gt(extent), valid & at_range,
              valid & stored.ne(state.ref))
    return stored, torch.stack((count, *(_columns(event) for event in events)), -1)


def _reference_graph(post, reference, ancestry, *, gain):
    original = torch.cat((post.weights * post.present, post.bias.unsqueeze(-1)), dim=-1)
    valid = torch.cat((ancestry & post.present, torch.ones_like(ancestry[..., :1])), dim=-1)
    amended = torch.where(valid, original.lerp(reference, gain), original)
    return replace(post, weights=torch.where(post.present, net._nonzero(amended[..., :-1]), post.weights),
                   bias=amended[..., -1])


def contact(rule, state, right, factor, draws, extent, mode, uniform=None):
    action = net.act(rule, state.live, -right, right, draws[:, 0])
    taste = action.side * factor
    learned = net.observe(rule, action, -right, right, taste)
    committed = net.commit(rule, learned, draws[:, 1:])
    post = committed.state.graph
    alive = state.ancestry & post.present
    fast = _reference_graph(post, decode(state.ref, extent, mode), alive, gain=GAMMA)
    ready = net.wait(rule, learned.state, net.PRE_WRITE_TICKS)
    live = net.wait(rule, replace(ready, graph=fast), net.POST_WRITE_TICKS)
    memory, traffic = update_reference(state, fast, alive, extent, mode, uniform)
    return Memory(net.wait(rule, live, home.WAIT), memory, alive), taste, traffic


def swapped(memory, indices):
    donor = memory.subset(indices)
    both = memory.ancestry & donor.ancestry
    mask = torch.cat((both, torch.ones_like(both[..., :1])), dim=-1)
    different = _columns(mask & memory.ref.ne(donor.ref))
    return replace(memory, ref=torch.where(mask, donor.ref, memory.ref)), different


def _uniform(info, stage, visit, worlds, lane_copies, mode):
    if not mode.stochastic:
        return None
    rng = np.random.default_rng([info["domain"], info["seed"], stage, visit, 8433])
    values = rng.random((worlds, *frozen.SHAPE), dtype=np.float32)
    return torch.from_numpy(values).repeat_interleave(lane_copies, dim=0)


def _physical(tapes, ecology):
    if ecology == "absent_a":
        return frozen.physical(tapes)
    first, *rest = tapes
    recurring = first.visits[home.ECOLOGIES.index("recurring")]
    return frozen.physical((replace(first, visits=recurring.expand_as(first.visits)), *rest))


def _probe(state, tape):
    picks = [motor.act(state, -right, right, draw) for right, draw in
             zip(tape.menus.unbind(1), tape.choices.unbind(1), strict=True)]
    side = torch.stack([action.side for action in picks], dim=1)
    score = torch.stack([action.motor_logit for action in picks], dim=1)
    return side, (score * tape.target).sigmoid()


@torch.no_grad()
def replay(graph, rule, tape, model, info, ages=AGES):
    mode = MODES[model]
    worlds = tape.worlds
    lanes = 4 * worlds
    twin = torch.arange(lanes).repeat_interleave(2)
    opposite_a = (torch.arange(lanes) ^ 2).repeat_interleave(2) * 2 + torch.arange(2).repeat(lanes)
    state, extent = _initial(graph, lanes, mode)
    view, expanded = rule.expand(lanes), rule.expand(2 * lanes)
    pre = torch.zeros((lanes, len(CHANNELS), len(COUNTS)), dtype=torch.int32)
    middle = torch.zeros_like(pre)
    readings = {}

    def store(**values):
        for name, array in values.items():
            readings.setdefault(name, []).append(array.numpy())

    def by_world(value):
        return value.reshape(worlds, 4, 2, *value.shape[1:])

    for i in range(8 + max(ages)):
        state, _, count = contact(view, state, tape.right[i], tape.factor[i], tape.draw[i], extent, mode,
                                  _uniform(info, 0, i, worlds, 4, mode))
        group = pre if i < 8 else middle
        group += count
        age = i - 7
        if age not in ages:
            continue
        snapshot = state.subset(twin)
        transplant, changed = swapped(snapshot, opposite_a)
        old = motor.act(snapshot.live, -tape.menus[:, 0], tape.menus[:, 0], tape.choices[:, 0]).side
        own, shared = ([], [], []), ([], [], [])
        earned = None
        rounding = _uniform(info, 1, age, worlds, 8, mode)
        for index, branch in enumerate((snapshot, transplant)):
            read_graph = _reference_graph(branch.live.graph, decode(branch.ref, extent, mode),
                                          branch.ancestry, gain=1.)
            read_side, read_prob = _probe(replace(branch.live, graph=read_graph), tape)
            acquired, taste, traffic = contact(expanded, branch, tape.x, tape.x_factor, tape.x_draw, extent, mode,
                                               rounding)
            side, prob = _probe(acquired.live, tape)
            good = side == tape.target
            success = torch.stack((old == tape.target[:, 0], *good.unbind(1),
                                   good.all(-1), good[:, 0] & good[:, 2]), dim=1)
            if index == 0:
                earned = taste
                store(x_traffic=by_world(traffic), x_taste=by_world(taste).to(torch.int8))
            elif not torch.equal(earned, taste):
                raise AssertionError("The hidden reference changed the X action before its first write")
            for cells, values in ((own, (side.to(torch.int8), prob, success)),
                                  (shared, (read_side.to(torch.int8), read_prob, read_side == tape.target))):
                for container, value in zip(cells, values, strict=True):
                    container.append(by_world(value))
        store(side=torch.stack(own[0]), probability_correct=torch.stack(own[1]), correct=torch.stack(own[2]),
              slow_read_side=torch.stack(shared[0]), slow_read_probability_correct=torch.stack(shared[1]),
              slow_read_correct=torch.stack(shared[2]), changed_by_swap=by_world(changed),
              ancestry=state.ancestry.sum((1, 2)).reshape(worlds, 4))
    packed = pack_snapshot(state, extent, mode)
    result = {key: np.stack(value) for key, value in readings.items()}
    return dict(result, target=by_world(tape.target).to(torch.int8).numpy(),
                history_traffic=torch.stack((pre, middle)).reshape(2, worlds, 4, len(CHANNELS), len(COUNTS)).numpy()), packed


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def protocol(out, worlds):
    sources = {}
    for name, info in COHORTS.items():
        paths = [old.checkpoint_path(DATA / info["source"], "odor", "uniform", birth, 8000)
                 for birth in info["births"]]
        sources[name] = {**info, "checkpoints": {p.relative_to(ROOT).as_posix(): _hash(p) for p in paths}}
    plan = dict(cohorts=sources, worlds=worlds, ecologies=ECOLOGIES, primary_ecology="absent_a",
                ages=AGES, gamma=GAMMA, beta=BETA, modes=MODE_NAMES, bootstrap=DRAWS,
                counts=COUNTS, count_channels=CHANNELS, count_phases=PHASES,
                metrics=FIELDS, queries=QUERY, branches=BRANCHES, equivalence_margin=MARGIN,
                stored_bytes={model: storage(mode) for model, mode in MODES.items()},
                source_sha256={Path(p).name: _hash(p) for p in
                               (__file__, frozen.__file__, slow.__file__, home.__file__, net.__file__, stats.__file__)},
                protocol="Four physical studies of each A and B, then 16/64/128 intervening contacts. "
                         "Main worlds omit A; secondary worlds allow A to recur. B timing matches. Each probe "
                         "starts from the same post-X state. Heldout geometry and all four A/B laws and two X "
                         "laws cross. Physical choices and edits retain native fp32 throughout.",
                range="All five reference variants share the same fixed per-receiver max-absolute birth "
                      "radius including bias, the same continuously surviving original nonzero address mask, "
                      "and always-eligible biases. FP32 proposals also clip to that row interval. "
                      "After each native write, the fast graph receives .12 interpolation toward the prior "
                      "reference and exact post-write ticks. Mutable references then move .02 toward this "
                      "corrected fast graph, clip, and store only the new reference words. Static beta is zero.",
                stochastic="Independently keyed uniform per address and contact, repeated across matched law "
                           "branches of the same geometry. Stochastically choose adjacent codes by fractional "
                           "distance, unbiased conditional on the clipped proposal. No residual or fp32 shadow "
                           "is carried. The external evaluation random stream is separate from payload bytes.",
                counts_def="Denominator is each surviving eligible coordinate update, reported separately "
                           "for weights and biases. continuous_change means the fp32 .02 proposal differs "
                           "from the previous decoded value. outside_range counts the unclipped proposal "
                           "beyond the birth radius. at_range counts resulting endpoint codes/values. "
                           "stored_change counts actual word changes. Static proposes no change.",
                swap="At each age, transplant common-surviving numeric reference coordinates and bias from "
                     "the opposite-A matched world sharing geometry and B law. Recipient live graph, activity, "
                     "traces and ancestry stay in place. Compare a reference-only graph read and behavior "
                     "after one matched actual X contact. The X choice and taste are asserted equal between "
                     "swap branches. Own minus opposite A correctness is the label-specific readout.",
                decision="Report pooled equal-age and per-age paired 95% crossed intervals for A/B/X and "
                         "all three. Compare each mode against FP32 and static 4 bits. Primary behavior "
                         "equivalence to FP32 requires X and all-three paired intervals fully within ±2pp "
                         "in both cohorts on pooled absent-A worlds. The swap's direct A and native A after X "
                         "are primary memory diagnostics, with B/X as specificity controls.",
                packed_snapshot="Actual bytes for one world-0 lane before the final X fork: reference words, "
                                "20 little-endian fp32 row radii, then packed 400-bit continuous-survival mask. "
                                "The ordinary live graph and transient decode memory are additional.")
    return home._lock(out / "run.json", plan)


def replay_cohort(out, cohort, ecology, worlds):
    signature = protocol(out, worlds)
    info = COHORTS[cohort]
    tapes = slow.public_tapes(worlds, info["seed"], info["domain"])
    inputs = _physical(tapes, ecology)
    archive = {name: value.numpy() for name, value in vars(tapes[0]).items()}
    archive.update(extra_object=tapes[1].numpy(), extra_orientation=tapes[2].numpy(), extra_draw=tapes[3].numpy())
    home._save_raw(out / cohort / "worlds.npz", archive)
    for birth in info["births"]:
        g, r = old.load_checkpoint(DATA / info["source"], "odor", "uniform", birth, 8000)
        folder = out / cohort / ecology / f"birth-{birth}"
        for model in MODE_NAMES:
            path = folder / f"{model}.npz"
            packet = folder / f"{model}.bin"
            if path.exists():
                with np.load(path, allow_pickle=False) as a:
                    if a["protocol_sha256"].item() != signature or a["packet_sha256"].item() != _hash(packet):
                        raise ValueError(f"Saved result differs from current protocol at {path}")
                continue
            start = perf_counter()
            raw, final_memory = replay(g, r, inputs, model, info)
            common._atomic_save(packet, lambda dest: dest.write_bytes(final_memory))
            home._save_raw(path, dict(**raw, ages=AGES, protocol_sha256=signature, packet_sha256=_hash(packet)))
            print(cohort, ecology, birth, model, f"{perf_counter()-start:.1f}s", flush=True)


def _ratio(array, numerator):
    total = array[..., 0].sum()
    return None if total == 0 else float(array[..., numerator].sum() / total)


def _load_cohort(out, cohort, ecology, signature, worlds):
    data, audit, traffic = {}, {}, {}
    for model, mode in MODES.items():
        rows, old_counts, x_counts = [], [], []
        for birth in COHORTS[cohort]["births"]:
            folder = out / cohort / ecology / f"birth-{birth}"
            path, packet = folder / f"{model}.npz", folder / f"{model}.bin"
            with np.load(path, allow_pickle=False) as raw:
                if (raw["protocol_sha256"].item() != signature or raw["packet_sha256"].item() != _hash(packet)
                        or packet.stat().st_size != storage(mode)["total"]):
                    raise ValueError(f"Artifact fingerprint or storage mismatch at {path}")
                correct = raw["side"] == raw["target"]
                np.testing.assert_array_equal(correct, raw["correct"][..., 1:4])
                np.testing.assert_array_equal(correct.all(-1), raw["correct"][..., 4])
                np.testing.assert_array_equal(raw["slow_read_side"] == raw["target"], raw["slow_read_correct"])
                if correct.shape != (len(AGES), 2, worlds, 4, 2, 3) or not np.all(np.abs(raw["x_taste"]) == 1):
                    raise ValueError(f"Incorrect physical samples at {path}")
                rows.append({key: raw[key].mean((3, 4)) for key in (
                    "correct", "probability_correct", "slow_read_correct", "slow_read_probability_correct")})
                old_counts.append(raw["history_traffic"])
                x_counts.append(raw["x_traffic"])
            audit[path.relative_to(out).as_posix()] = _hash(path)
            audit[packet.relative_to(out).as_posix()] = _hash(packet)
        data[model] = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        stats_rows = (np.stack(old_counts)[:, 0], np.stack(old_counts)[:, 1], np.stack(x_counts))
        traffic[model] = {phase: {name: {
            **{key: _ratio(count[..., i, :], idx) for idx, key in enumerate(COUNTS[1:], 1)},
            "eligible_updates": int(count[..., i, 0].sum()),
        } for i, name in enumerate(CHANNELS)} for phase, count in zip(PHASES, stats_rows, strict=True)}
    return data, audit, traffic


def summarize(out, evidence, worlds, ecologies):
    signature = protocol(out, worlds)
    report = dict(plan=json.loads((out / "run.json").read_text(encoding="utf8")), protocol_sha256=signature,
                  evaluated_ecologies=ecologies, raw_sha256={}, results={})
    for cohort, info in COHORTS.items():
        report["raw_sha256"][f"{cohort}/worlds.npz"] = _hash(out / cohort / "worlds.npz")
        ci = stats.Crossed(len(info["births"]), worlds, info["seed"] + 8909, DRAWS)
        outputs = {}
        for ecology in ecologies:
            data, audit, traffic = _load_cohort(out, cohort, ecology, signature, worlds)
            report["raw_sha256"].update(audit)
            controls = {}
            for model, batch in data.items():
                def stats_for(values, names):
                    slices = {"pooled": values.mean(1), **{str(age): values[:, i] for i, age in enumerate(AGES)}}
                    return {age: {name: ci(v[..., j]) for j, name in enumerate(names)} for age, v in slices.items()}

                native = batch["correct"]
                direct = batch["slow_read_correct"]
                direct_prob = batch["slow_read_probability_correct"]
                outcome = {"own": stats_for(native[:, :, 0], FIELDS),
                           "native_own_minus_swap": stats_for(native[:, :, 0] - native[:, :, 1], FIELDS),
                           "slow_read_own_minus_swap": stats_for(direct[:, :, 0] - direct[:, :, 1], QUERY),
                           "slow_read_probability_own_minus_swap": stats_for(direct_prob[:, :, 0] - direct_prob[:, :, 1], QUERY),
                           "minus_fp32": stats_for(native[:, :, 0] - data["fp32_slow"]["correct"][:, :, 0], FIELDS),
                           "minus_static4": stats_for(native[:, :, 0] - data["q4_static"]["correct"][:, :, 0], FIELDS)}
                outcome["equivalent_to_fp32_pooled"] = all(
                    -MARGIN < outcome["minus_fp32"]["pooled"][field]["ci95"][0]
                    <= outcome["minus_fp32"]["pooled"][field]["ci95"][1] < MARGIN
                    for field in ("new_X", "all_three"))
                controls[model] = outcome
            outputs[ecology] = {"outcomes": controls, "traffic": traffic}
        report["results"][cohort] = outputs
    raw = out.resolve()
    report["raw_path"] = str(raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw)
    home._lock(out / "summary.json", report)
    home._lock(evidence, report)
    print(evidence, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "replay", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--evidence", type=Path, default=REPORT)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--ecologies", nargs="+", choices=ECOLOGIES, default=("absent_a",))
    parser.add_argument("--worlds", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worlds < 2 or len(set(args.ecologies)) != len(args.ecologies):
        parser.error("Use unique listed ecologies and at least two independent worlds")
    if args.phase == "plan":
        print(protocol(args.out, args.worlds), flush=True)
    elif args.phase == "replay":
        if args.cohort is None:
            parser.error("Replay needs a cohort")
        for ecology in args.ecologies:
            replay_cohort(args.out, args.cohort, ecology, args.worlds)
    else:
        summarize(args.out, args.evidence, args.worlds, args.ecologies)


if __name__ == "__main__":
    main()
