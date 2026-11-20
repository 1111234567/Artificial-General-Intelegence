"""Fit one continuously writing 20-node policy on interleaved physical contacts."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from itertools import combinations, groupby
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import local_trace_graph as local
from . import local_trace_train as previous
from . import revisit_choice_graph as legacy
from . import self_param_writer as fw
from .edit_receipt_train import clip_birth_gradients


ARMS = ("local_active", "local_independent", "legacy_active")
BIRTHS = previous.BIRTHS
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data/interleaved-learning-20261120/cold"
TRAIN_DOMAIN = 2201137
DEVELOPMENT_DOMAIN = 2201139
CONTACTS = 14
ODOR, MARK = 0, 1
MARK_CONTEXT = 3
MARK_AXIS = previous.MARK_AXIS.copy()
CUE_BANK = np.array(((0., 1.1, 0.), (0., -.55, .95262794), (0., -.55, -.95262794)),
                    dtype=np.float32)
METRICS = (
    "joint_resource", "joint_correct", "odor_correct", "mark_correct", "a_correct", "b_correct",
    "c_correct", "conflict_a_correct", "conflict_b_correct", "agree_a_correct", "agree_b_correct",
    "mark_first_correct", "mark_second_correct", "motor_loss", "edit_loss", "contacts_numeric",
    "edges_born_per_life", "edges_retired_per_life", "grad_norm",
)


def modality_layouts(max_run):
    """All seven-plus-seven interleavings with the designated maximum run length."""
    return np.array([indices for indices in combinations(range(CONTACTS), CONTACTS // 2)
                     if max(len(list(run)) for _, run in groupby(i in indices for i in range(CONTACTS))) <= max_run])


TRAIN_LAYOUTS = modality_layouts(3)


@dataclass(frozen=True)
class Config:
    updates: int = 8000
    pairs: int = 16
    min_wait: int = 0
    max_wait: int = 8
    world_seed: int = 43197
    lr: float = .003
    grad_clip: float = 2.
    checkpoint_every: int = 500


@dataclass(frozen=True)
class Tape:
    """World data in birth, independent pair, opposite-law twin, and contact order."""

    left: torch.Tensor
    right: torch.Tensor
    draws: torch.Tensor
    waits: torch.Tensor
    right_factor: torch.Tensor
    fixed_factor: torch.Tensor
    donor_right_factor: torch.Tensor
    donor_fixed_factor: torch.Tensor
    context: torch.Tensor
    probe: torch.Tensor
    objective_weight: torch.Tensor
    law: torch.Tensor

    @property
    def batch_shape(self):
        return self.left.shape[:3]


def _signs(rng, shape):
    return (rng.integers(2, size=shape, dtype=np.int8) * 2 - 1).astype(np.float32)


def _odor(rng, pairs):
    values = rng.normal(size=(pairs, 3)).astype(np.float32)
    return 1.35 * values / np.linalg.norm(values, axis=1, keepdims=True)


def _sample_lives(config, birth, step, domain, layouts):
    key = (domain, config.world_seed, birth, step)
    rng, donor_rng = (np.random.default_rng([*key, stream]) for stream in (0, 1))
    pairs = config.pairs
    rows = np.arange(pairs)[:, None]
    odor_at = layouts[rng.integers(len(layouts), size=pairs)]
    mask = np.zeros((pairs, CONTACTS), dtype=bool)
    mask[rows, odor_at] = True
    mark_at = np.where(~mask)[1].reshape(pairs, CONTACTS // 2)

    context = np.full((pairs, CONTACTS), MARK_CONTEXT, dtype=np.int64)
    introduced = rng.permuted(np.broadcast_to(np.arange(3), (pairs, 3)), axis=1)
    reviewed = rng.permuted(np.c_[introduced, rng.integers(3, size=pairs)], axis=1)
    context[rows, odor_at] = np.concatenate((introduced, reviewed), axis=1)
    cue_order = rng.permuted(np.broadcast_to(np.arange(3), (pairs, 3)), axis=1)
    odor_context = context[rows, odor_at]
    cue = CUE_BANK[cue_order[rows, odor_context]]
    shared, other = _odor(rng, pairs), _odor(rng, pairs)
    odor = np.where(odor_context[..., None] == 2, other[:, None], shared[:, None])
    odor_sign = _signs(rng, (pairs, 7))
    odor_right = odor * odor_sign[..., None]

    left = np.empty((pairs, CONTACTS, 3), dtype=np.float32)
    right = np.empty_like(left)
    left[rows, odor_at] = cue - odor_right
    right[rows, odor_at] = cue + odor_right

    # Both marks are demonstrated before either mark probe. The extra mark is a
    # randomly placed demonstration and may arrive even after the second probe.
    first, first_right, second, second_right, extra = _signs(rng, (5, pairs))
    mark_chain = np.stack((first, -first, first_right, second, -second, second_right, extra), axis=1)
    extra_at = rng.integers(7, size=(pairs, 1))
    column = np.arange(7)[None, :]
    mark_index = np.where(column == extra_at, 6, column - (column > extra_at))
    mark_sign = mark_chain[rows, mark_index]
    is_mark_probe = (mark_index == 2) | (mark_index == 5)
    mark = mark_sign[..., None] * MARK_AXIS
    right[rows, mark_at] = mark
    left[rows, mark_at] = np.where(is_mark_probe[..., None], -mark, mark)

    # Probe identifiers are scorer-only fields; the last two values mark the
    # order of two opposed-mark choices. Non-probes carry zero objective weight.
    probe = np.full((pairs, CONTACTS), -1, dtype=np.int64)
    probe[rows, odor_at[:, 3:]] = reviewed
    probe[rows, mark_at] = np.where(mark_index == 2, 3, np.where(mark_index == 5, 4, -1))
    weight = np.zeros((pairs, CONTACTS), dtype=np.float32)
    multiplicity = (reviewed[..., None] == reviewed[:, None, :]).sum(axis=-1)
    weight[rows, odor_at[:, 3:]] = 1 / (6 * multiplicity)
    weight[rows, mark_at] = is_mark_probe * .25

    side_scale = np.zeros((pairs, CONTACTS), dtype=np.float32)
    side_scale[rows, odor_at] = odor_sign
    side_scale[rows, mark_at] = mark_sign * is_mark_probe
    fixed_scale = np.zeros_like(side_scale)
    fixed_scale[rows, mark_at] = mark_sign * ~is_mark_probe
    law = _signs(rng, (pairs, 4))
    donor = _signs(donor_rng, (pairs, 4))

    def opposite(values):
        return np.stack((values, -values), axis=1)

    def common(values):
        return np.repeat(values[:, None], 2, axis=1)

    true_context, donor_context = law[rows, context], donor[rows, context]
    draws = rng.random((pairs, CONTACTS, 6), dtype=np.float32)
    waits = rng.integers(config.min_wait, config.max_wait + 1, size=(pairs, CONTACTS))
    return (common(left), common(right), common(draws), common(waits), opposite(side_scale * true_context),
            opposite(fixed_scale * true_context), opposite(side_scale * donor_context),
            opposite(fixed_scale * donor_context), common(context), common(probe), common(weight), opposite(law))


def make_tape(births, config, step, *, domain=TRAIN_DOMAIN, layouts=TRAIN_LAYOUTS):
    """Sample complete physical lives with independently addressed world and donor streams."""
    lives = [_sample_lives(config, birth, step, domain, layouts) for birth in births]
    return Tape(*(torch.from_numpy(np.stack(field)) for field in zip(*lives, strict=True)))


@dataclass(frozen=True)
class Rollout:
    resource: torch.Tensor
    observed: torch.Tensor
    motor_score: torch.Tensor
    edit_score: torch.Tensor
    side: torch.Tensor
    numeric_contacts: torch.Tensor
    births: torch.Tensor
    retirements: torch.Tensor
    final_state: local.State | legacy.State


def rollout(graph, rule, tape, arm, max_wait=None, *, intervention="active"):
    """Sample physical motors and writes, carrying all resulting state within every life."""
    if arm not in ARMS or (rule is None) != (arm == "legacy_active"):
        raise ValueError("Arm and controller disagree")
    lives = tape.batch_shape[1] * tape.batch_shape[2]
    controller = (previous.LegacyController(graph, rule, lives) if rule is None
                  else previous.LocalController(graph, rule, lives))
    left, right, draws, waits, earned_side, earned_fixed = (
        value.flatten(0, 2).transpose(0, 1) for value in
        (tape.left, tape.right, tape.draws, tape.waits, tape.right_factor, tape.fixed_factor)
    )
    independent = arm == "local_independent"
    observed_side, observed_fixed = (value.flatten(0, 2).transpose(0, 1) for value in
                                    (tape.donor_right_factor, tape.donor_fixed_factor)) if independent else (
                                        earned_side, earned_fixed)
    resource, observed, motor_scores, edit_scores, sides = [], [], [], [], []
    numeric = graph.weights.new_zeros(len(controller.state.graph.bias))
    births, retirements = torch.zeros_like(numeric), torch.zeros_like(numeric)
    max_wait = int(tape.waits.max()) if max_wait is None else max_wait
    for contact, (a, b, u) in enumerate(zip(left, right, draws, strict=True)):
        if max_wait and contact:
            controller.wait(waits[contact], max_wait)
        action = controller.act(a, b, u)
        earned = earned_side[contact] * action.side + earned_fixed[contact]
        tasted = observed_side[contact] * action.side + observed_fixed[contact]
        before = controller.state.graph
        if intervention == "active":
            edit_score = controller.contact(action, a, b, tasted, u)
        elif rule is None:
            experience = legacy.taste(action, a, b, tasted)
            proposals = legacy.propose(experience, u[:, 1:5].reshape(-1, 2, 2))
            decision = legacy.choose(proposals, u[:, 5])
            controller.state = legacy.commit(experience, proposals, decision, intervention=intervention).state
            edit_score = (proposals.address_log_prob + proposals.op_log_prob).sum(1) + decision.log_prob
        else:
            experience = local.observe(controller.rule, action, a, b, tasted)
            result = local.commit(controller.rule, experience, u[:, 1:5], intervention=intervention)
            controller.state, edit_score = result.state, result.edit_log_prob
        with torch.no_grad():
            after = controller.state.graph
            shared = before.present & after.present
            numeric += ((before.weights.ne(after.weights) & shared).flatten(1).any(1)
                        | (before.bias != after.bias).any(1))
            births += (~before.present & after.present).sum((1, 2))
            retirements += (before.present & ~after.present).sum((1, 2))
        resource.append(earned)
        observed.append(tasted)
        motor_scores.append(action.motor_log_prob)
        edit_scores.append(edit_score)
        sides.append(action.side)

    def by_contact(values):
        return torch.stack(values, dim=1).reshape(*tape.batch_shape, CONTACTS)

    return Rollout(*(by_contact(values) for values in (resource, observed, motor_scores, edit_scores, sides)),
                   *(values.reshape(tape.batch_shape) for values in (numeric, births, retirements)),
                   controller.state)


def causal_advantage(resource, objective_weight):
    """Reward-to-go with a baseline that excludes both twins of the current pair."""
    weighted = (resource * objective_weight).detach()
    mean_pair = weighted.mean(dim=2, keepdim=True)
    baseline = (mean_pair.sum(dim=1, keepdim=True) - mean_pair) / (weighted.shape[1] - 1)
    return (weighted - baseline).flip(-1).cumsum(dim=-1).flip(-1)


def score_and_measure(rolled, tape):
    advantage = causal_advantage(rolled.resource, tape.objective_weight)
    motor = -(rolled.motor_score * advantage).sum(dim=-1).mean((1, 2))
    edit = -(rolled.edit_score[..., :-1] * advantage[..., 1:]).sum(dim=-1).mean((1, 2))
    with torch.no_grad():
        probe = tape.probe
        positive = rolled.resource > 0
        conflicting = tape.law[..., 0] != tape.law[..., 1]
        conflict = conflicting[..., None]

        def fraction(mask):
            return (positive & mask).sum((1, 2, 3)) / mask.sum((1, 2, 3))

        def weighted_success(selection):
            value = (positive * tape.objective_weight * selection).sum(dim=-1)
            return 2 * value.mean((1, 2))

        joint = (rolled.resource * tape.objective_weight).sum(dim=-1).mean((1, 2))
        metrics = torch.stack((joint, .5 + .5 * joint, weighted_success(probe < 3),
                               weighted_success(probe >= 3), *(fraction(probe == k) for k in range(3)),
                               fraction((probe == 0) & conflict), fraction((probe == 1) & conflict),
                               fraction((probe == 0) & ~conflict), fraction((probe == 1) & ~conflict),
                               fraction(probe == 3), fraction(probe == 4), motor, edit,
                               *(value.mean((1, 2)) for value in
                                 (rolled.numeric_contacts, rolled.births, rolled.retirements))), dim=1)
    return (motor + edit).sum(), metrics


def checkpoint_path(path, arm, step):
    return Path(path) / arm / f"step-{step:06d}.pt"


def load_checkpoint(path, arm, birth, step):
    """Load one frozen birth for evaluation, independent of optimizer and tape RNG."""
    payload = torch.load(checkpoint_path(path, arm, step), map_location="cpu", weights_only=True)
    if (payload["arm"], payload["step"]) != (arm, step):
        raise ValueError(f"Checkpoint identity mismatch for {arm} at step {step}")
    index = payload["births"].index(birth)
    lane = slice(index, index + 1)
    graph = fw.Graph(*(payload["graph"][field.name][lane] for field in fields(fw.Graph)))
    rule = None
    if payload["rule"] is not None:
        rule = local.Rule(birth)
        rule.load_state_dict({name: value[lane] for name, value in payload["rule"].items()})
        rule.eval().requires_grad_(False)
    return graph, rule


def _atomic_save(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _birth(births, arm, warm):
    controller = "legacy" if arm == "legacy_active" else "local"
    rule = local.Rule(births) if controller == "local" else None
    if warm is None:
        graph = previous.birth_graph(births, controller)
    else:
        source, task, step = warm
        parents = [previous.load_checkpoint(Path(source) / controller, task, "active", birth, step)
                   for birth in births]
        graph = fw.Graph(*(torch.cat([getattr(parent, field.name) for parent, _ in parents])
                           for field in fields(fw.Graph)))
        if rule is not None:
            states = [parent.state_dict() for _, parent in parents]
            rule.load_state_dict({key: torch.cat([state[key] for state in states]) for key in states[0]})
    return fw.Graph(nn.Parameter(graph.weights), nn.Parameter(graph.bias), graph.present), rule


def fit_arm(path, config, births, arm, warm, plan_hash, progress):
    torch.set_num_threads(1)
    graph, rule = _birth(births, arm, warm)
    parameters = (graph.weights, graph.bias, *(rule.parameters() if rule is not None else ()))
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    metrics = torch.empty((len(births), config.updates, len(METRICS)), dtype=torch.float32)
    checkpoints = sorted((path / arm).glob("step-*.pt"))
    step, previous_seconds = 0, 0.
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        if (state["arm"], state["births"], state["plan_hash"]) != (arm, list(births), plan_hash):
            raise ValueError(f"Resume would change the declared fit at {checkpoints[-1]}")
        step, previous_seconds = state["step"], state["seconds"]
        if checkpoints[-1] != checkpoint_path(path, arm, step) or state["world_seed"] != config.world_seed:
            raise ValueError(f"Invalid resume step or seed at {checkpoints[-1]}")
        with torch.no_grad():
            for field in fields(fw.Graph):
                getattr(graph, field.name).copy_(state["graph"][field.name])
        if rule is not None:
            rule.load_state_dict(state["rule"])
        optimizer.load_state_dict(state["optimizer"])
        metrics[:, :step] = state["metrics"]

    started = perf_counter()

    def elapsed():
        return previous_seconds + perf_counter() - started

    def save(index):
        state = dict(arm=arm, births=list(births), step=index, plan_hash=plan_hash,
                     world_seed=config.world_seed, world_domain=TRAIN_DOMAIN,
                     graph={field.name: getattr(graph, field.name).detach() for field in fields(fw.Graph)},
                     rule=rule.state_dict() if rule is not None else None,
                     optimizer=optimizer.state_dict(), metrics=metrics[:, :index].clone(), seconds=elapsed())
        _atomic_save(checkpoint_path(path, arm, index), lambda temporary: torch.save(state, temporary))

    if not checkpoints:
        save(0)
    print(f"arm={arm} start={step} updates={config.updates} births={','.join(map(str, births))}", flush=True)
    for index in range(step + 1, config.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        tape = make_tape(births, config, index)
        rolled = rollout(graph, rule, tape, arm, config.max_wait)
        loss, values = score_and_measure(rolled, tape)
        loss.backward()
        norm = clip_birth_gradients(tuple(p for p in parameters if p.grad is not None), config.grad_clip)
        if not torch.isfinite(norm).all():
            raise FloatingPointError(f"Non-finite gradient at {arm} step {index}")
        optimizer.step()
        metrics[:, index - 1, :-1] = values
        metrics[:, index - 1, -1] = norm
        del rolled, loss
        saved = index % config.checkpoint_every == 0 or index == config.updates
        if saved:
            save(index)
        if index == 1 or saved or (progress and index % progress == 0):
            recent = metrics[:, max(0, index - 100):index, 1:4].mean(1).mean(0)
            joint, odor, mark = recent.tolist()
            print(f"arm={arm} step={index}/{config.updates} joint={joint:.3f} odor={odor:.3f} "
                  f"marks={mark:.3f} seconds={elapsed():.1f}", flush=True)


def _source_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run(path, config, births=BIRTHS, arms=ARMS, *, warm=None, workers=3, progress=100):
    births, arms = tuple(births), tuple(arms)
    if (config.updates < 1 or config.pairs < 2 or config.min_wait < 0 or config.max_wait < config.min_wait
            or config.lr <= 0 or config.grad_clip <= 0 or config.checkpoint_every < 1 or config.world_seed < 0
            or workers < 1 or progress < 0 or not births or min(births) < 0 or len(set(births)) != len(births)
            or not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS)):
        raise ValueError("Invalid continuous fit configuration")
    path = Path(path)
    sources = [module.__file__ for module in (local, legacy, fw, previous)] + [__file__]
    contact_ticks = local.REACTION_TICKS + local.CONTACT_TICKS + local.TASTE_TICKS + local.PRE_WRITE_TICKS + local.POST_WRITE_TICKS
    warm_hashes = None
    if warm is not None:
        source, task, step = warm
        warm = (str(Path(source).resolve()), task, step)
        ctrl = {"legacy" if arm == "legacy_active" else "local" for arm in arms}
        warm_hashes = {f"{controller}/{birth}": _source_digest(previous.checkpoint_path(
            Path(source) / controller, task, "active", birth, step)) for controller in ctrl for birth in births}
    plan = dict(config=asdict(config), births=list(births), arms=list(arms), warm=warm, warm_sha256=warm_hashes,
                training_domain=TRAIN_DOMAIN, development_domain=DEVELOPMENT_DOMAIN,
                lives_per_birth=config.updates * config.pairs * 2,
                contacts_per_life=CONTACTS, active_ticks_per_contact=contact_ticks, possible_layouts=len(TRAIN_LAYOUTS),
                metrics=METRICS, source_sha256={Path(source).name: _source_digest(source) for source in sources},
                protocol={"physical_sequence": "Uniform random seven odor plus seven mark contacts with at most "
                                                "three consecutive contacts per modality. Every contact carries hidden, "
                                                "trace where present, and the edited graph. Random waits before contacts.",
                          "odor": "A/B use the same novel odor and independently sampled private laws; C has a "
                                  "different novel odor and independent law. Ambient bilateral cues are permuted "
                                  "independently of laws. All three first exposures precede all four probes. "
                                  "One randomly chosen context gets a second return.",
                          "mark": "Fixed familiar marks and independently sampled law. Both signs are shown on "
                                  "separate side-invariant contacts before each scored choice, with one extra demo.",
                          "objective": "Equal weights for each task, equal within-task weights for each odor context "
                                       "and each mark probe. Score sampled motor with earned reward-to-go including the "
                                       "current contact. Credit sampled edits only with strictly later earned reward. "
                                       "Independent pairs supply the score baseline.",
                          "independent": "Donor taste uses an independently sampled set of private laws with "
                                         "identical sensory inputs, physical action contingencies, timing and optimizer "
                                         "objective. All sampled physical actions are scored on true laws.",
                          "holdout": "World/twin draws are addressed by (domain, world_seed, birth, update, stream). "
                                     "Independent evaluation uses fresh domains, odors, and layouts with longer runs."})
    manifest = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    manifest_path = path / "run.json"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf8") != manifest:
        raise ValueError(f"Immutable run protocol differs at {manifest_path}")
    if not manifest_path.exists():
        _atomic_save(manifest_path, lambda temporary: temporary.write_text(manifest, encoding="utf8"))
    digest = sha256(manifest.encode("utf8")).hexdigest()
    jobs = [(path, config, births, arm, warm, digest, progress) for arm in arms]
    if workers == 1:
        for job in jobs:
            fit_arm(*job)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for result in [pool.submit(fit_arm, *job) for job in jobs]:
                result.result()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--births", type=int, nargs="+", default=BIRTHS)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--world-seed", type=int, default=43197)
    parser.add_argument("--min-wait", type=int, default=0)
    parser.add_argument("--max-wait", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--grad-clip", type=float, default=2.)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--warm-root", type=Path, default=previous.DEFAULT_OUTPUT)
    parser.add_argument("--warm-task", choices=previous.TASKS)
    parser.add_argument("--warm-step", type=int, default=8000)
    args = parser.parse_args()
    fit_config = Config(args.updates, args.pairs, args.min_wait, args.max_wait, args.world_seed,
                        args.lr, args.grad_clip, args.checkpoint_every)
    warm_start = (args.warm_root, args.warm_task, args.warm_step) if args.warm_task else None
    run(args.out, fit_config, args.births, args.arms, warm=warm_start, workers=args.workers, progress=args.progress)
