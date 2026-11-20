"""Evaluate frozen edge-site selectors on fresh directions and independent bodies."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import edge_site_eval_stats as stats
from . import edge_site_train as site
from . import edge_site_train_fit as fit
from . import local_trace_graph as net
from .interleave_interference import worlds


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/edge-site-20261120"
EVIDENCE = ROOT / "docs/evidence/edge-site-20261120"
SEED = 93780531
BIRTHS = tuple(range(811, 817))
ANGLES = (15, 30, 90)
FINAL_STEP = 1024
ROOTS = 256
PRIMARY_READ = (1, 4, 0)
FINAL_READS = (PRIMARY_READ, (1, 32, 0), (1, 4, 1), (3, 4, 0), (3, 32, 0))
MODES = ("full", "native", "no_numeric", "sham", "random", "no_old", "blank", "shuffled", "fixed")
PLAIN = ("native", "no_numeric", "sham")
DOSE_NAMES = ("written_count", "positive_count", "negative_count", "positive_l1", "negative_l1", "l1",
              "deferred_count", "deferred_positive_l1", "deferred_negative_l1", "deferred_l1",
              "plan_positive_l1", "plan_negative_l1", "positive_bias_l1", "negative_bias_l1",
              "edge_births", "edge_retirements", "selected_count", "max_unselected_change")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fingerprint(path):
    resolved = path.resolve()
    location = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)
    return dict(path=location, sha256=digest(resolved))


def intermediate_heads(step):
    artifact = dict(final_step=step, replica_seeds=np.asarray(fit.SEEDS))
    sources = {}
    training_id = digest(fit.PROTOCOL)
    for arm in fit.ARMS:
        coefficients = []
        for replica, seed in enumerate(fit.SEEDS):
            path = fit.folder(replica, arm) / f"step-{step:04d}.npz"
            with np.load(path, allow_pickle=False) as row:
                metadata = (int(row["step"]), int(row["seed"]), int(row["replica"]), str(row["arm"]), str(row["protocol"]))
                if metadata != (step, seed, replica, arm, training_id):
                    raise ValueError(f"Inconsistent training checkpoint {path}")
                coefficients.append(row["coefficients"])
            sources[f"{arm}/{replica}"] = fingerprint(path)
        artifact[arm] = np.stack(coefficients)
    return artifact, sources


def balance(tensor):
    flat = tensor.flatten(1)
    pos = flat.clamp_min(0).sum(1)
    neg = (-flat.clamp_max(0)).sum(1)
    return pos, neg, pos + neg


def lane_counts(mask):
    return mask.flatten(1).sum(1)


def physical_dose(prefix, state, selection, mode):
    before = prefix.experience_b.state.graph
    native = prefix.native_b.state.graph
    actual = state.graph
    if mode == "sham":
        for name in ("weights", "bias", "present"):
            if not torch.equal(getattr(actual, name), getattr(before, name)):
                raise AssertionError(f"A complete B sham changed {name}")
    elif not torch.equal(actual.present, native.present) or not torch.equal(actual.bias, native.bias):
        raise AssertionError(f"{mode} changed native B support or bias")
    born = actual.present & ~before.present
    retired = ~actual.present & before.present
    common = actual.present & before.present
    if mode != "sham" and not torch.equal(actual.weights[born], native.weights[born]):
        raise AssertionError(f"{mode} changed native birth payloads")
    realized = torch.where(common, actual.weights.double() - before.weights.double(), 0.)
    deferred = torch.where(common, native.weights.double() - actual.weights.double(), 0.)
    bias = actual.bias.double() - before.bias.double()
    if mode in {"sham", "no_numeric"} and (realized != 0).any():
        raise AssertionError(f"{mode} changed surviving edge numbers")
    if mode == "native" and (deferred != 0).any():
        raise AssertionError("Native arm changed the native numeric payload")

    valid = prefix.candidates.valid
    address = torch.full_like(valid, -1, dtype=torch.long)
    planned = prefix.candidates.amount.new_zeros(valid.shape, dtype=torch.float64)
    outside = planned.new_zeros(len(valid))
    if selection is not None:
        address = prefix.candidates.address.gather(2, selection.index.clamp(0, 3).long()[..., None]).squeeze(-1)
        address = torch.where(valid, address, -1)
        if ((address < 0) & valid).any():
            raise AssertionError(f"{mode} omitted a valid candidate group")
        planned = torch.where(valid, prefix.candidates.amount.double(), 0.)
        changed_locations = deferred.flatten(1)
        selected_locations = torch.zeros_like(changed_locations, dtype=torch.int32)
        selected_locations.scatter_add_(1, address.clamp_min(0), valid.int())
        outside = changed_locations.masked_fill(selected_locations != 0, 0.).abs().amax(dim=1)
        if (outside > 0).any() or (selected_locations > 1).any():
            raise AssertionError(f"{mode} lost unique address locality")
    pos, neg, total = balance(realized)
    late_pos, late_neg, late_total = balance(deferred)
    plan_pos, plan_neg, _ = balance(planned)
    bias_pos, bias_neg, _ = balance(bias)
    doses = torch.stack((lane_counts(realized != 0), lane_counts(realized > 0), lane_counts(realized < 0),
                         pos, neg, total, lane_counts(deferred != 0), late_pos, late_neg, late_total,
                         plan_pos, plan_neg, bias_pos, bias_neg, lane_counts(born), lane_counts(retired),
                         (address >= 0).sum(1), outside), dim=-1)
    return doses.numpy(), address.numpy().astype(np.int16)


def paired_random(world, n_groups, visit):
    roots = len(world.a) // 4
    rng = np.random.default_rng([SEED, 86179, visit])
    draw = np.repeat(rng.random((roots, n_groups), dtype=np.float32), 4, axis=0)
    priority = np.repeat(rng.random((roots, n_groups, 4), dtype=np.float32), 4, axis=0)
    return torch.from_numpy(draw), torch.from_numpy(priority)


def repeat_world(world, visit):
    if visit == 1:
        return world
    rng = np.random.default_rng([SEED, 423161, visit])
    roots = len(world.a) // 4
    orders = world.orders.clone()
    draws = world.draws.clone()
    orders[:, 1] = torch.from_numpy(np.repeat(rng.choice((-1, 1), size=roots), 4))
    draws[:, 1] = torch.from_numpy(np.repeat(rng.random((roots, 5), dtype=np.float32), 4, axis=0))
    return replace(world, orders=orders, draws=draws)


def select(prefix, world, visit, mode, full, no_old):
    if mode in PLAIN:
        return None
    draws, priority = paired_random(world, prefix.candidates.valid.shape[1], visit)
    weights = no_old if mode == "no_old" else full
    return site.choose(prefix, weights, mode=mode, draw=draws, priority=priority)


def apply(prefix, selection, mode):
    if mode == "sham":
        return net.wait(prefix.rule, prefix.experience_b.state, net.PRE_WRITE_TICKS + net.POST_WRITE_TICKS)
    return site.apply(prefix, selection, mode=mode if mode in PLAIN else "selected")


@torch.no_grad()
def evaluate(graph, rule, world, full_heads, no_old_heads, reads):
    if tuple(sorted(reads, key=lambda read: read[0])) != reads:
        raise ValueError("Readouts must be grouped in chronological visit order")
    visits = max(read[0] for read in reads)
    prefix_a = site.prepare(graph, rule, world)
    replicate_rows = []
    for full, no_old in zip(full_heads, no_old_heads, strict=True):
        mode_rows = []
        for mode in MODES:
            initial = prefix_a
            traces = {key: [] for key in ("side", "correct", "payout", "logit", "dose", "address", "b_study_side", "b_taste")}
            for visit in range(1, visits + 1):
                op_world = repeat_world(world, visit)
                if visit > 1:
                    initial = site.prepare_b(prefix_a.rule, state, prefix_a.old, op_world)
                selection = select(initial, op_world, visit, mode, full, no_old)
                state = apply(initial, selection, mode)
                dose, addresses = physical_dose(initial, state, selection, mode)
                traces["dose"].append(dose)
                if visit == 1:
                    traces["address"].append(addresses)
                traces["b_study_side"].append(initial.action_b.side.numpy().astype(np.int8))
                traces["b_taste"].append(initial.experience_b.taste.numpy().astype(np.int8))
                for count, wait, order in reads:
                    if count != visit:
                        continue
                    direction = torch.full((len(world.a),), order, dtype=torch.long)
                    probe = site.probe(prefix_a.rule, state, world, wait=wait, order=direction)
                    traces["side"].append(probe.side.numpy().astype(np.int8))
                    traces["correct"].append(probe.correct.numpy())
                    traces["payout"].append(probe.payout.numpy().astype(np.int8))
                    traces["logit"].append(probe.logit.numpy())
            mode_rows.append({key: np.stack(value) for key, value in traces.items()})
        replicate_rows.append({key: np.stack([row[key] for row in mode_rows]) for key in mode_rows[0]})
    raw = {key: np.stack([row[key] for row in replicate_rows]) for key in replicate_rows[0]}
    raw.update({key: getattr(world, key).numpy() for key in ("a", "b", "law_a", "law_b", "orders", "draws")})
    raw["a_study_side"] = prefix_a.action_a.side.numpy().astype(np.int8)
    raw["candidate_address"] = prefix_a.candidates.address.numpy().astype(np.int16)
    raw["candidate_amount"] = prefix_a.candidates.amount.numpy()
    raw["candidate_valid"] = prefix_a.candidates.valid.numpy()
    goals = np.stack((raw["law_a"] * raw["orders"][:, 3], raw["law_b"] * raw["orders"][:, 4]), axis=-1)
    if not np.array_equal(raw["side"] == goals, raw["correct"]):
        raise AssertionError("Scored law or serial endpoint differs from actual chosen side")
    return raw


def audit_matched(rows, relation=None):
    full = MODES.index("full")
    random = MODES.index("random")
    dose_rows = []
    groups = shared = 0
    for row in rows:
        lanes = slice(None) if relation is None else (row["law_a"] == row["law_b"]) == relation
        dose_rows.append(row["dose"][:, :, 0, lanes])
        address = row["address"][:, :, 0, lanes]
        selected = address[:, full] >= 0
        groups += int(selected.sum())
        shared += int((selected & (address[:, full] == address[:, random])).sum())
    dose = np.concatenate(dose_rows, axis=2)
    ranges = {}
    for index, name in enumerate(DOSE_NAMES):
        difference = dose[:, full, :, index] - dose[:, random, :, index]
        absolute = np.abs(difference)
        ranges[name] = dict(full_mean=float(dose[:, full, :, index].mean()),
                            random_mean=float(dose[:, random, :, index].mean()),
                            absolute_difference_mean=float(absolute.mean()), absolute_difference_max=float(absolute.max()),
                            exact_equal_fraction=float((absolute == 0).mean()))
    return dict(per_life_dose=ranges, active_groups=groups, same_address=shared,
                choice_overlap=shared / groups if groups else None,
                denominator=int(np.prod(dose[:, full, :, 0].shape)))


def regime(rows_by_angle, angles, read, reads, relation, *, replica=None):
    read_ix = reads.index(read)
    per_angle = []
    for angle in angles:
        rows = rows_by_angle[angle]
        per_birth = []
        for row in rows:
            mask = ((row["law_a"] == row["law_b"]) == relation).reshape(-1, 4)
            if not np.all(mask.sum(axis=1) == 2):
                raise ValueError("Every geometry root must have two laws of each relation")
            good = row["correct"][:, :, read_ix].reshape(-1, len(MODES), len(mask), 4, 2)
            if replica is not None:
                good = good[replica:replica + 1]
            good = good[:, :, mask, :].reshape(len(good), len(MODES), len(mask), 2, 2)
            good = good.transpose(2, 0, 3, 1, 4).reshape(len(mask), -1, len(MODES), 2)
            per_birth.append(stats.score(good))
        per_angle.append(np.stack(per_birth))
    return np.concatenate(per_angle, axis=2)


def report(rows_by_angle, reads, count, n_replicas, *, final):
    boot = stats.CrossedBootstrap(len(next(iter(rows_by_angle.values()))), count)
    active_rows = [row for angle in (15, 30) for row in rows_by_angle[angle]]
    primary = regime(rows_by_angle, (15, 30), PRIMARY_READ, reads, False)
    pool = stats.summarize(primary, MODES, boot)
    pool["registered_gates"] = stats.primary_gates(pool)
    pool["is_preregistered_final"] = final
    by_replica = [stats.summarize(regime(rows_by_angle, (15, 30), PRIMARY_READ, reads, False, replica=replica), MODES, boot)
                  for replica in range(n_replicas)]
    strata = {}
    for angle in rows_by_angle:
        for read in reads:
            for same in (False, True):
                visits, delay, order = read
                key = f"{angle}/{('agree' if same else 'conflict')}/B{visits}/wait{delay}/{('BA' if order else 'AB')}"
                strata[key] = stats.summarize(regime(rows_by_angle, (angle,), read, reads, same), MODES, boot)
    return dict(primary=pool, primary_per_training_replica=by_replica, strata=strata,
                matched_primary_b1=audit_matched(active_rows, False),
                matched_b1_by_angle={angle: audit_matched(rows) for angle, rows in rows_by_angle.items()})


def run(artifact, label, *, count=ROOTS, births=BIRTHS, reads=FINAL_READS):
    torch.set_num_threads(2)
    step = int(artifact["final_step"])
    raw_dir = DATA / f"evaluation-{label}-{step:04d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    full_heads, no_old_heads = (torch.from_numpy(artifact[name]).float() for name in ("full", "no_old"))
    replica_seeds = artifact["replica_seeds"].tolist()
    if (full_heads.shape != no_old_heads.shape or full_heads.shape != (len(replica_seeds), len(site.FEATURES))
            or not full_heads.isfinite().all() or not no_old_heads.isfinite().all()):
        raise ValueError("Unexpected complete and no-old head formats")
    rows_by_angle = {angle: [] for angle in ANGLES}
    paths = {}
    for birth in births:
        graph, rule = site.load_parent(birth)
        for angle in ANGLES:
            world = worlds(SEED, 0, count, angle)
            raw = evaluate(graph, rule, world, full_heads, no_old_heads, reads)
            path = raw_dir / f"{birth}-{angle}.npz"
            np.savez_compressed(path, **raw)
            rows_by_angle[angle].append(raw)
            paths[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
            print(f"{label} step {step} birth {birth}, {angle}° complete", flush=True)
    final = count == ROOTS and tuple(births) == BIRTHS and step == FINAL_STEP and tuple(replica_seeds) == fit.SEEDS
    details = report(rows_by_angle, reads, count, len(replica_seeds), final=final)
    return dict(step=step, registered_final_step=FINAL_STEP, eval_seed=SEED, geometry_roots=count,
                bootstrap_resamples=stats.N_BOOT, bootstrap_seed=stats.BOOT_SEED,
                births=list(births), angles=ANGLES, modes=MODES, read_keys=reads, dose_columns=DOSE_NAMES,
                training_replica_seeds=replica_seeds, raw_sha256=paths, **details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--heads", type=Path)
    source.add_argument("--checkpoint-step", type=int, choices=fit.CHECKPOINTS)
    parser.add_argument("--label", default="heldout")
    parser.add_argument("--roots", type=int, default=ROOTS)
    parser.add_argument("--births", type=int, nargs="+", default=BIRTHS)
    parser.add_argument("--primary-only", action="store_true")
    cfg = parser.parse_args()
    reads = (PRIMARY_READ,) if cfg.primary_only else FINAL_READS
    if cfg.heads is not None:
        with np.load(cfg.heads, allow_pickle=False) as artifact:
            if "feature_names" in artifact and tuple(artifact["feature_names"]) != site.FEATURES:
                raise ValueError("Stored head feature order does not match the evaluation runtime")
            result = run(artifact, cfg.label, count=cfg.roots, births=cfg.births, reads=reads)
        sources = dict(published=fingerprint(cfg.heads))
    else:
        artifact, sources = intermediate_heads(cfg.checkpoint_step)
        result = run(artifact, cfg.label, count=cfg.roots, births=cfg.births, reads=reads)
    result["heads"] = sources
    parent_paths = fit.parent_paths()
    result["parent_sha256"] = {str(birth): fingerprint(parent_paths[str(birth)]) for birth in cfg.births}
    result["source_sha256"] = {
        path.name: digest(path) for path in (Path(__file__), Path(stats.__file__), Path(site.__file__))
    }
    result["protocol_sha256"] = {
        path.name: digest(path) for path in (EVIDENCE / "evaluation-protocol.md", EVIDENCE / "evaluation-implementation.md",
                                             fit.PROTOCOL)
    }
    output = EVIDENCE / f"evaluation-{cfg.label}-{result['step']:04d}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
