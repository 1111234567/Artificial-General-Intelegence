"""Compare autonomous edits with an offline gradient for the preceding motor choice."""

from argparse import ArgumentParser
from dataclasses import dataclass, fields
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import revisit_choice_eval as holdout
from . import revisit_choice_graph as native
from . import revisit_choice_train as train
from . import self_param_writer as fw


DOMAIN = 0x4D721
DEFAULT_SEED = 76403
EVIDENCE = Path(__file__).resolve().parents[2] / "docs/evidence/revisit-choice-2026-11-19/gradient-probe.json"


@dataclass(frozen=True)
class World:
    first_left: torch.Tensor
    return_left: torch.Tensor
    first_outcome: torch.Tensor
    return_outcome: torch.Tensor
    first_draw: torch.Tensor
    proposal_draws: torch.Tensor
    gate_draw: torch.Tensor
    return_draw: torch.Tensor
    wait: torch.Tensor

    def subset(self, indices):
        return World(*(getattr(self, field.name)[indices] for field in fields(self)))


def make_world(seed, birth, pairs, min_wait, max_wait):
    tape_seed = int(np.random.SeedSequence([seed, DOMAIN, 0]).generate_state(1)[0])
    tape = holdout.make_holdout(tape_seed, birth, pairs)
    keep = torch.arange(tape.lanes) % 4 < 2

    def both(values):
        return torch.cat((values[keep], values[keep]))

    first_outcome = torch.tensor([-1., 1.]).repeat_interleave(2 * pairs)
    wait_rng = np.random.default_rng([seed, DOMAIN, birth, 1])
    waits = torch.from_numpy(wait_rng.integers(min_wait, max_wait + 1, 2 * pairs)).repeat(2)
    return World(both(tape.first_left), both(tape.second_left("same")), first_outcome,
                 first_outcome * (1 - 2 * both(tape.swap).float()), both(tape.first_draw),
                 both(tape.proposal_draws), both(tape.gate_draw), both(tape.second_draw), waits)


def returned(state, world, max_wait):
    delayed = native.wait(state, world.wait, max_ticks=max_wait)
    action = native.act(delayed, world.return_left, -world.return_left, world.return_draw)
    return action.side * world.return_outcome, (action.motor_logit * world.return_outcome).sigmoid()


def motor_alignment(start, post, command, first_left, desired, original_logit, delta, keep_connected):
    graph = start.graph
    address = command.address
    edge, receiver, sender = fw.decode(address, graph.width)
    rows = torch.arange(len(address), device=address.device)
    with torch.enable_grad():
        weight = graph.weights.detach().requires_grad_()
        bias = graph.bias.detach().requires_grad_()
        before = native.State(fw.Graph(weight, bias, graph.present), start.hidden)
        _, logit = native.motor(before, first_left, -first_left)
        grads = torch.autograd.grad((logit * desired).sum(), (weight, bias))
        slope = torch.where(edge, grads[0][rows, receiver, sender], grads[1][rows, receiver])
    with torch.no_grad():
        linear = torch.where(command.op == fw.Op.ADJUST, slope * delta, torch.nan)
        _, after = native.motor(native.State(post.graph, start.hidden), first_left, -first_left)
        finite = (after - original_logit) * desired
        # Structural commands have no corresponding signed continuous null.
        adjust = command.op == fw.Op.ADJUST
        mirror = fw.Command(address, torch.where(adjust, command.op, fw.Op.NONE), -delta)
        mirror, mirrored_receipt = fw.commit(graph, mirror, keep_connected=keep_connected, value_limit=None)
        _, reversed_logit = native.motor(native.State(mirror, start.hidden), first_left, -first_left)
        reversed_applied = adjust & (mirrored_receipt.status == fw.Status.APPLIED)
        reversed_finite = torch.where(reversed_applied, (reversed_logit - original_logit) * desired, torch.nan)
    return edge, linear, finite, reversed_finite


def measure_chunk(graph, world, max_wait, keep_connected):
    with torch.no_grad():
        start = native.start(graph, len(world.first_outcome))
        first = native.act(start, world.first_left, -world.first_left, world.first_draw)
        received = first.side * world.first_outcome
        experience = native.taste(first, world.first_left, -world.first_left, received)
        proposals = native.propose(experience, world.proposal_draws)
        decision = native.choose(proposals, world.gate_draw)
        result = native.commit(experience, proposals, decision, keep_connected=keep_connected)
        applied = result.raw_receipt.status == fw.Status.APPLIED
        if not applied.any():
            return received.numpy(), None
        indices = torch.where(applied)[0]
        sub = world.subset(indices)
        initial = start.subset(indices)
        post = result.state.subset(indices)
        command = result.command
        command = fw.Command(command.address[indices], command.op[indices], command.payload[indices])
        raw = result.raw_receipt
        delta = raw.value_after[indices] - raw.value_before[indices]
        real_taste, real_expected = returned(post, sub, max_wait)
        # Replay the same selected candidate with its physical edit suppressed;
        # both branches start the delay from the exact applied branch activity.
        fixed_h = native.State(initial.graph, post.hidden)
        fixed_taste, fixed_expected = returned(fixed_h, sub, max_wait)
        ex = experience.subset(indices)
        candidates = proposals.subset(indices)
        forced = native.choose(candidates, override=decision.index[indices])
        disabled = native.commit(ex, candidates, forced, intervention="disabled",
                                 keep_connected=keep_connected)
        disabled_taste, disabled_expected = returned(disabled.state, sub, max_wait)
        desired = first.side[indices] * received[indices]

    edge, linear, finite, reversed_finite = motor_alignment(
        initial, post, command, sub.first_left, desired, first.motor_logit[indices], delta, keep_connected,
    )
    data = {
        "edge": edge, "op": command.op, "first_taste": received[indices], "wait": sub.wait,
        "linear": linear, "finite": finite, "reversed_finite": reversed_finite,
        "revisit_fixed_h_actual": (real_taste - fixed_taste) / 2,
        "revisit_fixed_h_expected": real_expected - fixed_expected,
        "revisit_disabled_actual": (real_taste - disabled_taste) / 2,
        "revisit_disabled_expected": real_expected - disabled_expected,
    }
    return received.numpy(), {key: value.numpy() for key, value in data.items()}


def counts(values, eps):
    return {"positive": int((values > eps).sum()), "negative": int((values < -eps).sum()),
            "near_zero": int((np.isfinite(values) & (np.abs(values) <= eps)).sum())}


def describe(data, mask, eps):
    values = {key: array[mask] for key, array in data.items()}
    linear, finite, mirror = (values[key] for key in ("linear", "finite", "reversed_finite"))
    # For structural commands use the exact finite replay: no matching continuous gradient exists.
    alignment = np.where(np.isfinite(linear), linear, finite)
    fixed_actual = values["revisit_fixed_h_actual"]

    def mean(array):
        value = holdout.mean(array)
        return value if np.isfinite(value) else None

    return {
        "applied": len(finite), "mean_wait": mean(values["wait"]),
        "first_linear": counts(linear, eps), "first_finite": counts(finite, eps),
        "first_finite_reversed": counts(mirror, eps),
        "first_finite_minus_reversed": counts(finite - mirror, eps),
        "mean_first_linear": mean(linear), "mean_first_finite": mean(finite),
        "mean_first_finite_minus_reversed": mean(finite - mirror),
        "mean_revisit_fixed_h_actual": mean(fixed_actual),
        "mean_revisit_fixed_h_expected": mean(values["revisit_fixed_h_expected"]),
        "mean_revisit_disabled_actual": mean(values["revisit_disabled_actual"]),
        "mean_revisit_disabled_expected": mean(values["revisit_disabled_expected"]),
        "revisit_fixed_h_actual": counts(fixed_actual, 0),
        "revisit_disabled_actual": counts(values["revisit_disabled_actual"], 0),
        "revisit_fixed_h_given_first_alignment": {
            "positive": counts(fixed_actual[alignment > eps], 0),
            "negative": counts(fixed_actual[alignment < -eps], 0),
            "near_zero": counts(fixed_actual[np.abs(alignment) <= eps], 0),
        },
    }


def breakdown(data, eps):
    op, edge, taste = (data[key] for key in ("op", "edge", "first_taste"))
    operations = {"all": np.ones(len(op), dtype=bool), "adjust_bias": (op == fw.Op.ADJUST) & ~edge,
                  "adjust_weight": (op == fw.Op.ADJUST) & edge, "birth": op == fw.Op.BIRTH,
                  "retire": op == fw.Op.RETIRE}
    return {name: {"both": describe(data, mask, eps),
                   "positive_taste": describe(data, mask & (taste > 0), eps),
                   "negative_taste": describe(data, mask & (taste < 0), eps)}
            for name, mask in operations.items()}


def join(chunks):
    if not chunks:
        return {name: np.empty(0, dtype=bool if name == "edge" else float) for name in (
            "edge", "op", "first_taste", "wait", "linear", "finite", "reversed_finite",
            "revisit_fixed_h_actual", "revisit_fixed_h_expected",
            "revisit_disabled_actual", "revisit_disabled_expected",
        )}
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}


def summarize(run_dir, *, stage, pairs, chunk_size, seed, threshold):
    plan = json.loads((run_dir / "run.json").read_text(encoding="utf8"))
    options = plan["config"]
    min_wait, max_wait = options["min_wait"], options["max_wait"]
    births = plan["births"]
    arm = "ranked"
    by_birth, all_data = {}, []
    for birth in births:
        filename = train.storage.checkpoint_path(run_dir, arm, birth, stage)
        graph = train.load_checkpoint(run_dir, arm, birth, stage)
        world = make_world(seed, birth, pairs, min_wait, max_wait)
        observed, evidence = [], []
        for start in range(0, len(world.wait), chunk_size):
            sample = world.subset(slice(start, start + chunk_size))
            taste, edits = measure_chunk(graph, sample, max_wait, options["keep_connected"])
            observed.append(taste)
            if edits is not None:
                evidence.append(edits)
        joined = join(evidence)
        all_data.append(joined)
        by_birth[str(birth)] = {
            "checkpoint_sha256": sha256(filename.read_bytes()).hexdigest(),
            "first_tastes": counts(np.concatenate(observed), 0), "edits": breakdown(joined, threshold),
        }
    bootstrap = holdout.BirthBootstrap(births, seed, 20000)

    def fraction(record, name):
        counts = record[name]
        nonzero = counts["positive"] + counts["negative"]
        return counts["positive"] / nonzero if nonzero else np.nan

    metrics = ("mean_first_linear", "mean_first_finite", "mean_first_finite_minus_reversed",
               "mean_revisit_fixed_h_actual", "mean_revisit_fixed_h_expected",
               "mean_revisit_disabled_actual", "mean_revisit_disabled_expected")
    rates = ("first_linear", "first_finite", "first_finite_reversed", "first_finite_minus_reversed")
    aggregate = breakdown(join(all_data), threshold)
    uncertainty = {}
    for operation, signs in aggregate.items():
        uncertainty[operation] = {}
        for sign in signs:
            groups = [by_birth[str(birth)]["edits"][operation][sign] for birth in births]
            uncertainty[operation][sign] = {
                **{name: bootstrap([group[name] for group in groups]) for name in metrics},
                **{name + "_positive_given_nonzero": bootstrap([fraction(group, name) for group in groups])
                   for name in rates},
            }
    return {
        "schema_version": 1, "stage": stage, "arm": arm, "births": births,
        "seed": seed, "domain": DOMAIN, "fresh_base_odors_per_birth": pairs,
        "visits_per_birth": 4 * pairs, "delays_inclusive": [min_wait, max_wait], "near_zero_signed_logit": threshold,
        "objective": "External gradient of desired-side signed first motor logit at the exact pre-first activity. Desired side equals the actual first side times the experienced taste.",
        "linear_operator": "Derivative at the self-selected address times the actual signed scalar change; structural edits use the exact finite difference.",
        "finite_reversal": "Signed motor effect of negating the applied scalar change at the same address. Only applied continuous reversals enter its paired comparison.",
        "return_fixed_h": "Exact post-commit activity held fixed, with the selected physical edit either present or suppressed; same wait and same actual return draw.",
        "return_disabled": "Same selected command with physical commit disabled and natural corresponding receipt/activity.",
        "sampling": "First and return side sampled autonomously. Both stable latent taste assignments and both revisit side orders are crossed for each novel odor with common draws.",
        "inference_limit": "Alignment with one externally chosen gradient objective does not identify the internal algorithm.",
        "aggregate_counts": aggregate, "paired_birth_bootstrap": uncertainty, "by_birth": by_birth,
        "sources_sha256": {**plan["sources_sha256"],
                           **{source.name: sha256(source.read_bytes()).hexdigest()
                              for source in (Path(__file__), Path(holdout.__file__))}},
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=train.DEFAULT_OUTPUT)
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    parser.add_argument("--stage", type=int, default=8000)
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=1e-6)
    args = parser.parse_args()
    if args.pairs < 1 or args.chunk < 1 or args.seed < 0 or args.threshold < 0:
        parser.error("Require positive pairs and chunk size, nonnegative seed and threshold")
    torch.set_num_threads(1)
    result = summarize(args.run, stage=args.stage, pairs=args.pairs, chunk_size=args.chunk,
                       seed=args.seed, threshold=args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(f"{args.out}: {result['aggregate_counts']['all']['both']['applied']} applied edits", flush=True)


if __name__ == "__main__":
    main()
