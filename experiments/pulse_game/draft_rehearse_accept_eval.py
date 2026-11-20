"""Evaluate accepting a B draft with two independent new panels of paired physical lives."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import draft_rehearse as draft
from . import draft_rehearse_accept as gate
from . import draft_rehearse_fit as address_fit
from . import edge_site_eval_stats as paired
from . import interleave_revisit_train as storage
from . import noon_ticket as parents
from .interleave_interference import worlds


WORLD_SEEDS = (95340681, 95340707)
BIRTHS = tuple(parents.births("independent"))
ROOTS = 64
ANGLES = (15, 30)
METHODS = ("context", "constant", "matched_random", "native", "selected")
LAWS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def load_gate_heads(path, *, smoke=False):
    file_hash = address_fit.digest(path)
    n_arms, n_policies, n_features = len(address_fit.ARMS), len(gate.POLICIES), len(gate.REPLAY_READS)
    if smoke:
        return np.zeros((len(gate.SEEDS), n_arms, n_policies, n_features), dtype=np.float32), {}, []
    coefficients, inputs, training = [], {}, []
    identity = (file_hash, address_fit.digest(gate.PLAN), address_fit.digest(Path(gate.__file__)))
    for replica, seed in enumerate(gate.SEEDS):
        source = gate.DATA / f"accept-fit-{replica}.npz"
        inputs[source.name] = address_fit.digest(source)
        with np.load(source, allow_pickle=False) as rows:
            stamp = tuple(str(rows[key]) for key in ("address_heads_sha256", "plan_sha256", "source_sha256"))
            if stamp != identity or int(rows["seed"]) != seed or int(rows["replica"]) != replica:
                raise ValueError(f"Gate was fit with different sealed inputs: {source}")
            if not np.array_equal(rows["step"], np.arange(1, gate.STEPS + 1)):
                raise ValueError(f"Unfinished accept-gate fit: {source}")
            values = rows["coefficient"][:, :, -1]
            if values.shape != (n_arms, n_policies, n_features) or not np.isfinite(values).all():
                raise ValueError(f"Invalid accept-gate coefficients: {source}")
            if np.any(values[:, 1, 1:] != 0):
                raise ValueError(f"Constant gate consumed rehearsal data: {source}")
            coefficients.append(values)
            events = rows["actions"]
            last = events[:, :, -64:]
            correct = last[..., 1:] > 0
            training.append(dict(replica=replica, lives_per_policy=int(np.prod(events.shape[2:4])),
                                 acceptance_last_64=last[..., 0].mean(axis=(2, 3)).tolist(),
                                 real_last_64=paired.score(correct).mean(axis=(2, 3)).tolist()))
    return np.stack(coefficients), inputs, training


def matched_acceptance(gated, first, rng):
    """Randomly reassign the same number of accepts within each law pair and revisit order."""
    shuffled = np.zeros_like(gated)
    for law in range(len(LAWS)):
        for order in (0, 1):
            lane = np.flatnonzero(first[law::len(LAWS)] == order) * len(LAWS) + law
            count = int(gated[lane].sum())
            shuffled[rng.permutation(lane)[:count]] = True
            if shuffled[lane].sum() != count:
                raise AssertionError("Same-rate random gate changed the acceptance count")
    return shuffled


@torch.no_grad()
def panel(graph, rule, world, seed, birth, angle, address_heads, weights, count):
    prefix = draft.prepare(graph, rule, world, seed=(seed, 783315), lanes_per_geometry=len(LAWS))
    truth = torch.stack((world.orders[:, 3] * world.law_a, world.orders[:, 4] * world.law_b), dim=1).numpy()
    order_rng = np.random.default_rng([seed, 112640, angle])
    first = np.repeat(order_rng.permutation(count) >= count // 2, len(LAWS)).astype(np.int64)
    sides = np.empty((*weights.shape[:2], len(METHODS), len(first), 2), dtype=np.int8)
    accepted = np.empty(sides.shape[:-1], dtype=bool)
    doses = np.empty((*accepted.shape, 3), dtype=np.float32)
    features = []
    native = draft.apply(prefix, mode="native")
    for i, arm in enumerate(address_fit.ARMS):
        read = draft.rehearse(prefix, arm)
        context = gate.read_rehearsal(read)
        features.append(context.numpy())
        for replica in range(len(gate.SEEDS)):
            rng = np.random.default_rng([seed, 29736, angle, replica])
            groups = rng.random((count, address_fit.MAX_GROUPS), dtype=np.float32)
            address_draw = torch.from_numpy(np.repeat(groups[:, :prefix.candidates.valid.shape[1]], len(LAWS), axis=0))
            gate_draw = torch.from_numpy(np.repeat(rng.random(count, dtype=np.float32), len(LAWS)))
            choice = draft.choose(prefix, read, address_heads[arm][replica], draw=address_draw)
            contextual, constant = (torch.from_numpy(row) for row in weights[replica, i])
            use_context = gate_draw < (context @ contextual).sigmoid()
            use_constant = gate_draw < constant[0].sigmoid()
            shuffled = matched_acceptance(use_context.numpy(), first,
                                           np.random.default_rng([seed, 31401, birth, angle, replica, i]))
            masks = (use_context, use_constant, torch.from_numpy(shuffled),
                     torch.ones_like(use_context), torch.zeros_like(use_context))
            for j, mask in enumerate(masks):
                state = draft.apply(prefix, choice, accept=mask)
                for name in ("hidden", "sender_trace", "receiver_trace"):
                    if not torch.equal(getattr(state, name), getattr(native, name)):
                        raise AssertionError("Gate changed the common live state")
                signed = (native.graph.weights - state.graph.weights).flatten(1)
                changed = signed.ne(0).sum(1)
                if not torch.equal(changed, prefix.candidates.valid.sum(1).masked_fill(mask, 0)):
                    raise AssertionError("Gate did not implement the sampled all-native or selected write")
                doses[replica, i, j] = torch.stack((changed, signed.clamp_min(0).sum(1),
                                                   -signed.clamp_max(0).sum(1)), dim=1).numpy()
                accepted[replica, i, j] = mask.numpy()
                probe = draft.probe(prefix.rule, state, world, wait=address_fit.REVISIT_WAIT, order=first)
                if not np.array_equal(probe.correct.numpy(), probe.side.numpy() == truth):
                    raise AssertionError("Physical action disagrees with scorable private target")
                sides[replica, i, j] = probe.side.numpy()
    return dict(sides=sides, truth=truth.astype(np.int8), accepted=accepted, doses=doses,
                first=first, gate_inputs=np.stack(features), worlds=count, seed=seed, birth=birth, angle=angle)


def summarize(results, count, births):
    bootstrap = paired.CrossedBootstrap(len(births), count)
    output = {}
    for agree, laws in ((False, [1, 2]), (True, [0, 3])):
        label = "agree" if agree else "conflict"
        output[label] = {}
        for i, arm in enumerate(address_fit.ARMS):
            by_birth, doses, accepts = [], [], []
            for birth in births:
                angles = []
                for angle in ANGLES:
                    rows = results[birth, angle]
                    good = paired.score(rows["sides"][:, i] == rows["truth"])
                    selected = good.reshape(len(gate.SEEDS), len(METHODS), count, len(LAWS), -1)[:, :, :, laws]
                    angles.append(selected.transpose(2, 0, 3, 1, 4).reshape(count, -1, len(METHODS), len(paired.METRICS)))
                    dose = rows["doses"][:, i].reshape(len(gate.SEEDS), len(METHODS), count, len(LAWS), 3)
                    accepted = rows["accepted"][:, i].reshape(len(gate.SEEDS), len(METHODS), count, len(LAWS))
                    doses.append(dose[:, :, :, laws].mean(axis=(0, 2, 3)))
                    accepts.append(accepted[:, :, :, laws].mean(axis=(0, 2, 3)))
                by_birth.append(np.concatenate(angles, axis=1))
            result = paired.summarize(np.stack(by_birth), ("full", *METHODS[1:]), bootstrap)
            result["rates"]["context"] = result["rates"].pop("full")
            result["contrasts"] = {key.replace("full_", "context_"): value for key, value in result["contrasts"].items()}
            result["acceptance"] = dict(zip(METHODS, np.mean(accepts, axis=0).tolist(), strict=True))
            result["deferred_dose_per_life"] = dict(zip(METHODS, np.mean(doses, axis=0).tolist(), strict=True))
            output[label][arm] = result
    return output


def run(head_path, *, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    address_heads = gate.load_address_heads(head_path, require_final=not smoke)
    weights, file_hash, training = load_gate_heads(head_path, smoke=smoke)
    if smoke:
        seeds, count, births = WORLD_SEEDS[:1], 4, BIRTHS[:1]
    else:
        seeds, count, births = WORLD_SEEDS, ROOTS, BIRTHS
    evidence = dict(seeds=seeds, geometry_roots=count, births=births, angles=ANGLES,
                    arms=address_fit.ARMS, gate_policies=gate.POLICIES, methods=METHODS,
                    dose_columns=("addresses", "positive_l1", "negative_l1"),
                    training=training, coefficients=weights.tolist(), fit_files=file_hash,
                    heads_sha256=address_fit.digest(head_path), source_sha256=address_fit.digest(Path(__file__)),
                    bootstrap_seed=paired.BOOT_SEED, bootstrap_resamples=paired.N_BOOT, panels={})
    if not smoke:
        raw = gate.DATA / "accept-eval"
        raw.mkdir(parents=True, exist_ok=True)
        evidence["plan_sha256"] = address_fit.digest(gate.PLAN)
    for seed in seeds:
        results, checksums = {}, {}
        for birth in births:
            graph, rule = draft.load_parent(birth)
            for angle in ANGLES:
                rows = panel(graph, rule, worlds(seed, 0, count, angle), seed, birth, angle, address_heads, weights, count)
                results[birth, angle] = rows
                if not smoke:
                    path = raw / f"{seed}-{birth}-{angle}.npz"
                    storage._write_raw(path, **rows)
                    checksums[path.name] = address_fit.digest(path)
            print(f"accept evaluation {seed}, birth {birth} complete", flush=True)
        evidence["panels"][seed] = dict(raw_sha256=checksums, **summarize(results, count, births))
    if not smoke:
        storage._write_json(gate.EVIDENCE / "accept-results.json", evidence)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, default=gate.DATA / "fit-heads.npz")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.heads, smoke=args.smoke)


if __name__ == "__main__":
    main()
