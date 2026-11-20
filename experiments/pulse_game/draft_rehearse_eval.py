"""Read the frozen single-draft selectors on new geometries and bodies."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
import json

import numpy as np
import torch

from . import draft_rehearse as draft
from . import draft_rehearse_eval_stats as stats
from . import edge_site_train as site
from . import edge_site_train_fit as ancestors
from . import local_trace_graph as net
from .edge_site_eval import DOSE_NAMES, fingerprint
from .interleave_interference import worlds


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/draft-rehearsal-20261120"
EVIDENCE = ROOT / "docs/evidence/draft-rehearsal-20261120"
PROTOCOL = EVIDENCE / "eval-protocol.md"
FIT_PROTOCOL = EVIDENCE / "fit-protocol.json"
SEED = 63190477
TRAIN_DOMAIN = 73111943
BIRTHS = (811, 812, 813, 814, 815, 816)
TRAIN_BIRTHS = (731, 732, 733, 734, 735, 736)
ANGLES = (15, 30, 90)
ROOTS = 256
MODES = ("post", "pre", "wrong", "post_wrong", "pre_wrong", "random", "native")
CONDITIONS = {"post": ("post", "post"), "pre": ("pre", "pre"), "wrong": ("wrong", "wrong"),
              "post_wrong": ("wrong", "post"), "pre_wrong": ("pre_wrong", "pre")}
HEADS = ("post", "pre", "wrong")
READS = ("live_4", "common_4", "live_32")
ORDERS = (0, 1)
STATE_FIELDS = ("hidden", "sender_trace", "receiver_trace")
SLOTS = net.NODES ** 2 // site.GROUP_SIZE


def repeated(values):
    return torch.from_numpy(np.repeat(values, 4, axis=0))


def rng(*fields):
    return np.random.default_rng([SEED, *fields])


def prepare_world(world, visit):
    if visit == 1:
        return world
    count = len(world.a) // 4
    event = rng(26479, visit)
    order = world.orders.clone()
    order[:, 1] = repeated(event.choice((-1, 1), count))
    draws = world.draws.clone()
    draws[:, 1] = repeated(event.random((count, 5), dtype=np.float32))
    return replace(world, orders=order, draws=draws)


def read(rule, state, world, wait, truth):
    sides = []
    for order in ORDERS:
        replay = draft.probe(rule, state, world, wait=wait, order=order)
        if not torch.equal(replay.correct, replay.side == truth):
            raise AssertionError("Actual payoff and environmental answer disagree")
        sides.append(replay.side.to(torch.int8).numpy())
    return np.stack(sides)


def selecting(prefix, rehearsals, heads, replica, mode, draws):
    if mode == "native":
        return None
    draws = draws[:, :prefix.candidates.valid.shape[1]]
    if mode == "random":
        sample = site.choose(prefix, mode="random", draw=draws)
        if not torch.equal(sample.index, (draws * site.GROUP_SIZE).long()):
            raise AssertionError("Uniform address control drifted from its presampled choices")
        return sample
    rehearsal, arm = CONDITIONS[mode]
    return draft.choose(prefix, rehearsals[rehearsal], heads[arm][replica], draw=draws)


def address_pad(address):
    return np.pad(address, ((0, 0), (0, SLOTS - address.shape[1])), constant_values=-1)


def candidate_fields(prefix):
    group = prefix.candidates
    where = group.address.numpy().astype(np.int16)
    valid = group.valid.numpy()
    return (np.pad(np.where(valid[..., None], where, -1),
                   ((0, 0), (0, SLOTS - where.shape[1]), (0, 0)), constant_values=-1),
            np.pad(group.amount.numpy(), ((0, 0), (0, SLOTS - where.shape[1]))))


@torch.no_grad()
def evaluate(graph, rule, world, heads, repeat_b):
    size = len(world.a)
    prepare_seed = (SEED, 42713)
    prefix = draft.prepare(graph, rule, world, seed=prepare_seed, lanes_per_geometry=4)
    rehearsals = {name: draft.rehearse(prefix, name) for name in draft.MODES}
    native = draft.apply(prefix, mode="native")
    truth = torch.stack((world.orders[:, 3] * world.law_a, world.orders[:, 4] * world.law_b), dim=1)
    n_replicas = len(heads["post"])
    read_names = (*READS, "b_three_4") if repeat_b else READS
    side = np.empty((len(read_names), n_replicas, len(MODES), len(ORDERS), size, 2), dtype=np.int8)
    address = np.empty((n_replicas, len(MODES), size, SLOTS), dtype=np.int16)
    dose = np.empty((n_replicas, len(MODES), size, len(DOSE_NAMES)), dtype=np.float64)
    transient_error = np.empty((n_replicas, len(MODES), len(STATE_FIELDS)))
    later_worlds = {visit: prepare_world(world, visit) for visit in (2, 3)} if repeat_b else {}
    later_dose = np.empty((len(later_worlds), *dose.shape), dtype=dose.dtype)
    field = np.stack([[rng(20023, visit, replica).random((size // 4, SLOTS), dtype=np.float32)
                       for replica in range(n_replicas)] for visit in range(1, 4 if repeat_b else 2)])
    candidate, plan = candidate_fields(prefix)

    for replica in range(n_replicas):
        draws = repeated(field[0, replica])
        for index, mode in enumerate(MODES):
            selected = selecting(prefix, rehearsals, heads, replica, mode, draws)
            write = "native" if mode == "native" else "selected"
            state = draft.apply(prefix, selected, mode=write)
            dose[replica, index], location = draft.physical_dose(prefix, state, selected, write)
            address[replica, index] = address_pad(location)
            transient_error[replica, index] = [float((getattr(state, name) - getattr(native, name)).abs().max())
                                               for name in STATE_FIELDS]
            common = net.swap(state, native, ("hidden", "trace"))
            for target, start, wait in ((0, state, 4), (1, common, 4), (2, state, 32)):
                side[target, replica, index] = read(prefix.rule, start, world, wait, truth)

            episode = prefix
            for visit, now in later_worlds.items():
                episode = draft.prepare_next_b(episode, state, now, seed=(*prepare_seed, visit),
                                               lanes_per_geometry=4)
                available = {} if mode not in CONDITIONS else {
                    CONDITIONS[mode][0]: draft.rehearse(episode, CONDITIONS[mode][0])}
                selected = selecting(episode, available, heads, replica, mode, repeated(field[visit - 1, replica]))
                state = draft.apply(episode, selected, mode=write)
                later_dose[visit - 2, replica, index], _ = draft.physical_dose(episode, state, selected, write)
            if repeat_b:
                side[3, replica, index] = read(episode.rule, state, world, 4, truth)

    archives = (prefix.study_c.archived, prefix.study_a.archived, prefix.study_b.archived)
    return dict(side=side, truth=truth.to(torch.int8).numpy(),
                laws=torch.stack((world.law_a, world.law_b), dim=1).to(torch.int8).numpy(),
                geometry=torch.stack((world.a, world.b), dim=1).numpy(),
                unrelated_c=prefix.c.numpy(), receipt_menus=torch.stack([r.right for r in archives], dim=1).numpy(),
                receipt_side=torch.stack([r.side for r in archives], dim=1).numpy(),
                receipt_taste=torch.stack([r.taste for r in archives], dim=1).numpy(),
                rehearsal_retained_first=prefix.retained_first.numpy(), rehearsal_draw=prefix.motor_draw.numpy(),
                orders=world.orders.numpy(), draws=world.draws.numpy(),
                candidate=candidate, plan=plan, selector_draw=field, address=address, dose=dose,
                later_dose=later_dose, transient_error=transient_error, reads=np.asarray(read_names))


def read_heads(path):
    with np.load(path, allow_pickle=False) as bundle:
        if tuple(bundle["feature_names"]) != draft.FEATURES:
            raise ValueError("Frozen feature order differs from the physical selector")
        replicas = bundle["replica_seeds"].tolist()
        if len(replicas) != 2 or len(set(replicas)) != len(replicas):
            raise ValueError("Expected two independent frozen training replicates")
        heads = {mode: bundle[mode] for mode in HEADS}
        for mode, head in heads.items():
            if head.shape != (len(replicas), len(draft.FEATURES)) or not np.all(np.isfinite(head)):
                raise ValueError(f"Invalid frozen {mode} head")
        metadata = dict(final_step=int(bundle["final_step"]), replica_seeds=replicas,
                        fit_protocol_sha256=str(bundle["protocol_sha256"]))
    return heads, metadata


def check_fit_lineage(metadata):
    registered = json.loads(FIT_PROTOCOL.read_text(encoding="utf-8"))
    actual = (metadata["fit_protocol_sha256"], metadata["final_step"], metadata["replica_seeds"])
    expected = (fingerprint(FIT_PROTOCOL)["sha256"], registered["steps"], registered["fit_seeds"])
    if actual != expected:
        raise ValueError("Evaluation requires the prespecified final training heads")
    if (tuple(registered["training_births"]), tuple(registered["heldout_births"]), registered["geometry_domain"]) != (
            TRAIN_BIRTHS, BIRTHS, TRAIN_DOMAIN) or SEED == TRAIN_DOMAIN:
        raise ValueError("Evaluation and training geometry or bodies are not independent")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_birth(birth, *, label, raw, heads, count, repeat_b, identity, frozen, threads):
    torch.set_num_threads(threads)
    graph, rule = draft.load_parent(birth)
    files = {}
    for angle in ANGLES:
        world = worlds(SEED, 0, count, angle)
        rows = evaluate(graph, rule, world, heads, repeat_b)
        path = raw / f"{birth}-{angle}.npz"
        np.savez_compressed(path, **rows, birth=birth, angle=angle, seed=SEED,
                            protocol_sha256=identity, heads_sha256=frozen)
        files[path.name] = fingerprint(path)
        print(f"{label}: {birth} at {angle}° complete", flush=True)
    return files


def collect(label, head_path, count, births, repeat_b, workers, threads):
    heads, metadata = read_heads(head_path)
    check_fit_lineage(metadata)
    if any(birth not in BIRTHS for birth in births) or len(set(births)) != len(births):
        raise ValueError("A test batch must use distinct held-out bodies")
    raw = DATA / f"eval-{label}"
    raw.mkdir(parents=True, exist_ok=True)
    parents = ancestors.parent_paths()
    inputs = {path.name: fingerprint(path) for path in
              (PROTOCOL, FIT_PROTOCOL, Path(__file__), Path(stats.__file__), Path(draft.__file__))}
    manifest = dict(**metadata, heads=fingerprint(head_path), inputs=inputs, seed=SEED,
                    training_geometry_domain=TRAIN_DOMAIN, training_births=TRAIN_BIRTHS,
                    births=births, geometry_roots=count, angles=ANGLES, modes=MODES,
                    orders=ORDERS, dose_columns=DOSE_NAMES, repeat_b=repeat_b,
                    parent_sha256={str(birth): fingerprint(parents[str(birth)]) for birth in births}, files={})
    run = partial(run_birth, label=label, raw=raw, heads=heads, count=count, repeat_b=repeat_b, threads=threads,
                  identity=inputs[PROTOCOL.name]["sha256"], frozen=manifest["heads"]["sha256"])
    if workers == 1:
        for files in map(run, births):
            manifest["files"].update(files)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for files in pool.map(run, births):
                manifest["files"].update(files)
    write_json(raw / "manifest.json", manifest)
    return manifest


def summarize(label, manifest):
    raw = DATA / f"eval-{label}"
    panels = {}
    audit = []
    expected_world = {angle: worlds(manifest["seed"], 0, manifest["geometry_roots"], angle)
                      for angle in manifest["angles"]}
    for filename, recorded in manifest["files"].items():
        path = raw / filename
        if fingerprint(path) != recorded:
            raise ValueError(f"Archived row hash changed: {path}")
        with np.load(path, allow_pickle=False) as row:
            birth, angle = (int(row[key]) for key in ("birth", "angle"))
            if (int(row["seed"]), str(row["protocol_sha256"]), str(row["heads_sha256"])) != (
                    manifest["seed"], manifest["inputs"][PROTOCOL.name]["sha256"], manifest["heads"]["sha256"]):
                raise ValueError(f"Inconsistent sealed inputs: {path}")
            world = expected_world[angle]
            np.testing.assert_array_equal(row["geometry"], torch.stack((world.a, world.b), dim=1).numpy())
            np.testing.assert_array_equal(row["orders"], world.orders.numpy())
            np.testing.assert_array_equal(row["draws"], world.draws.numpy())
            panels[birth, angle], result = stats.validate(row, manifest)
            audit.append(dict(birth=birth, angle=angle, **result))
    result = stats.analyze(panels, manifest)
    result.update(audit=audit, manifest=manifest,
                  confirms_registered_holdout=tuple(manifest["births"]) == BIRTHS and manifest["geometry_roots"] == ROOTS)
    output = EVIDENCE / f"eval-{label}.json"
    write_json(output, result)
    print(output, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, default=DATA / "fit-heads.npz")
    parser.add_argument("--label", default="heldout")
    parser.add_argument("--roots", type=int, default=ROOTS)
    parser.add_argument("--births", type=int, nargs="+", default=BIRTHS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeat-b", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    config = parser.parse_args()
    torch.set_num_threads(config.threads)
    if config.summarize_only:
        manifest = json.loads((DATA / f"eval-{config.label}" / "manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = collect(config.label, config.heads, config.roots, config.births, config.repeat_b,
                           config.workers, config.threads)
    summarize(config.label, manifest)


if __name__ == "__main__":
    main()
