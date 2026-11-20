"""Try one native B write, rehearse two real receipts, and revise existing edges once."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from . import edge_site_train as site
from . import local_trace_graph as net


MODES = ("post", "pre", "wrong", "pre_wrong")
CONTACT_TICKS = net.REACTION_TICKS + net.CONTACT_TICKS + net.TASTE_TICKS
REHEARSAL_TICKS = 2 * CONTACT_TICKS
PREPARE_GAP = 4
REPLAY_FEATURES = (
    "sender_activity", "receiver_activity", "endpoint_activity", "receiver_virtual_taste_response",
    "sender_to_receiver_response", "signed_sender_to_receiver_response", "virtual_taste_endpoint_activity",
    "signed_virtual_taste_endpoint_activity",
)
FEATURES = site.CURRENT_FEATURES + tuple(
    f"{kind}_{feature}" for kind in ("retained", "current") for feature in REPLAY_FEATURES
)
load_parent = site.load_parent
probe = site.probe


@dataclass(frozen=True)
class Recall:
    right: torch.Tensor
    side: torch.Tensor
    taste: torch.Tensor


@dataclass(frozen=True)
class Study:
    before: net.State
    action: net.Action
    experience: net.Experience
    commit: net.Commit
    archived: Recall


@dataclass(frozen=True, kw_only=True)
class Prefix(site.Prefix):
    c: torch.Tensor
    study_c: Study
    study_a: Study
    study_b: Study
    ready_b: net.State
    live: net.State
    retained_first: torch.Tensor
    motor_draw: torch.Tensor


@dataclass(frozen=True)
class Replayed:
    archived: Recall
    action: net.Action
    experience: net.Experience


@dataclass(frozen=True)
class Rehearsal:
    retained: Replayed
    current: Replayed
    start: net.State
    end: net.State


def _rng(seed, stream):
    if isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, stream), pool_size=seed.pool_size)
    else:
        seed = np.random.SeedSequence(seed, spawn_key=(stream,))
    return np.random.default_rng(seed)


def _repeat(values, lanes_per_geometry, like):
    return torch.as_tensor(np.repeat(values, lanes_per_geometry, axis=0), device=like.device)


def _unrelated(world, seed, lanes_per_geometry):
    rng = _rng(seed, 174387)
    count = len(world.a) // lanes_per_geometry
    fields = _repeat(rng.integers(0, 2, (count, 2), dtype=np.int8), lanes_per_geometry, world.a)
    order, law = (2 * fields.to(world.a.dtype) - 1).unbind(1)
    draws = _repeat(rng.random((count, 1 + net.EDIT_DRAWS), dtype=np.float32), lanes_per_geometry, world.a)
    # Use a perpendicular third cue. For collinear A/B, a fresh random tangent serves the same role.
    c = torch.linalg.cross(world.a, world.b, dim=1)
    other = _repeat(rng.standard_normal((count, 3), dtype=np.float32), lanes_per_geometry, world.a)
    tangent = torch.linalg.cross(world.a, other, dim=1)
    c = torch.where(c.norm(dim=1, keepdim=True) > 1e-6, c, tangent)
    c = c * (world.a.norm(dim=1, keepdim=True) / c.norm(dim=1, keepdim=True))
    return c, order, law, draws


def _replay_tape(seed, like, births, lanes_per_geometry):
    rng = _rng(seed, 571708)
    count = len(like) // (births * lanes_per_geometry)
    order = np.empty((births, count), dtype=bool)
    for lane in order:
        first_count = (count + int(rng.integers(0, 2))) // 2
        lane[:] = rng.permutation(count) < first_count
    draws = rng.random((births * count, 2), dtype=np.float32)
    return (_repeat(order.ravel(), lanes_per_geometry, like), _repeat(draws, lanes_per_geometry, like))


def _study(rule, state, vector, law, orientation, draws):
    right = vector * orientation[:, None]
    action = net.act(rule, state, -right, right, draws[:, 0])
    taste = action.side * orientation * law
    experience = net.observe(rule, action, -right, right, taste)
    commit = net.commit(rule, experience, draws[:, 1:])
    return Study(state, action, experience, commit, Recall(right, action.side, experience.taste))


def _draft(rule, state, world, c, study_c, study_a, *, seed, births, lanes_per_geometry):
    before_b = net.wait(rule, state, PREPARE_GAP)
    study_b = _study(rule, before_b, world.b, world.law_b, world.orders[:, 1], world.draws[:, 1])
    ready_b = net.wait(rule, study_b.experience.state, net.PRE_WRITE_TICKS)
    # The live body spends this time on its old graph. The draft and projected responses are private scratch.
    live = net.wait(rule, ready_b, net.POST_WRITE_TICKS + REHEARSAL_TICKS)
    retained_first, motor_draw = _replay_tape(seed, world.a, births, lanes_per_geometry)
    old_activity = study_a.action.state.hidden
    old = site.OldTrace(study_a.commit.receipt.weight_delta, old_activity, old_activity)
    return Prefix(rule, old, before_b, study_b.action, study_b.experience, study_b.commit,
                  site.group_proposals(study_b.commit.receipt.weight_delta),
                  study_a.before, study_a.action, study_a.experience, study_a.commit,
                  c=c, study_c=study_c, study_a=study_a, study_b=study_b, ready_b=ready_b, live=live,
                  retained_first=retained_first, motor_draw=motor_draw)


def _check_geometry_lanes(lanes, births, lanes_per_geometry):
    if lanes == 0 or lanes_per_geometry < 1 or lanes % (births * lanes_per_geometry):
        raise ValueError("Expected equal numbers of complete geometry groups per body")


@torch.no_grad()
def prepare(graph, rule, world, *, seed, lanes_per_geometry=1):
    """Study C, A and B through real contacts; produce one private native B draft."""
    births = len(graph.bias)
    _check_geometry_lanes(len(world.a), births, lanes_per_geometry)
    lanes_per_birth = len(world.a) // births
    expanded = rule.expand(lanes_per_birth)
    initial = net.start(graph, lanes_per_birth)
    c, orientation, law, draws = _unrelated(world, seed, lanes_per_geometry)
    study_c = _study(expanded, initial, c, law, orientation, draws)
    before_a = net.wait(expanded, study_c.commit.state, PREPARE_GAP)
    study_a = _study(expanded, before_a, world.a, world.law_a, world.orders[:, 0], world.draws[:, 0])
    return _draft(expanded, study_a.commit.state, world, c, study_c, study_a,
                  seed=seed, births=births, lanes_per_geometry=lanes_per_geometry)


@torch.no_grad()
def prepare_next_b(prefix, state, world, *, seed, lanes_per_geometry=1):
    """Continue a life with another B study, keeping the original C and A receipts."""
    if world.b.shape != prefix.study_b.archived.right.shape or state.hidden.shape != prefix.live.hidden.shape:
        raise ValueError("Expected the original batch of living lanes")
    _check_geometry_lanes(len(world.b), 1, lanes_per_geometry)
    return _draft(prefix.rule, state, world, prefix.c, prefix.study_c, prefix.study_a,
                  seed=seed, births=1, lanes_per_geometry=lanes_per_geometry)


def _stage(rule, state, receipt, draw):
    right = receipt.right
    action = net.act(rule, state, -right, right, draw)
    # Opposite sides always have opposite tastes in this task. The forecast comes from the archived contact.
    virtual_taste = receipt.taste * receipt.side * action.side
    experience = net.observe(rule, action, -right, right, virtual_taste)
    return Replayed(receipt, action, experience)


def _pick(mask, yes, no):
    mask = mask.reshape((len(mask),) + (1,) * (yes.ndim - 1))
    return torch.where(mask, yes, no)


def _pick_receipt(mask, yes, no):
    return Recall(*(_pick(mask, getattr(yes, name), getattr(no, name)) for name in ("right", "side", "taste")))


def _pick_state(mask, yes, no):
    return net.State(yes.graph, *(_pick(mask, getattr(yes, name), getattr(no, name))
                                  for name in ("hidden", "sender_trace", "receiver_trace")))


def _by_object(mask, first, second, archived):
    a, b = first.action, second.action
    action = net.Action(_pick_state(mask, a.state, b.state), *(
        _pick(mask, getattr(a, name), getattr(b, name)) for name in ("side", "motor_logit", "motor_log_prob")))
    a, b = first.experience, second.experience
    experience = net.Experience(_pick_state(mask, a.state, b.state), *(
        _pick(mask, getattr(a, name), getattr(b, name)) for name in ("side", "taste", "feedback")))
    return Replayed(archived, action, experience)


@torch.no_grad()
def rehearse(prefix, mode="post"):
    """Run each receipt once on the same projected graph, in a balanced serial order."""
    if mode not in MODES:
        raise ValueError(f"Unknown rehearsal {mode!r}")
    graph = prefix.ready_b.graph if mode in {"pre", "pre_wrong"} else prefix.native_b.state.graph
    start = net.wait(prefix.rule, replace(prefix.ready_b, graph=graph), net.POST_WRITE_TICKS)
    retained = (prefix.study_c if mode in {"wrong", "pre_wrong"} else prefix.study_a).archived
    current = prefix.study_b.archived
    mask = prefix.retained_first
    old_draw, new_draw = prefix.motor_draw.unbind(1)
    first = _stage(prefix.rule, start, _pick_receipt(mask, retained, current), _pick(mask, old_draw, new_draw))
    second = _stage(prefix.rule, first.experience.state, _pick_receipt(mask, current, retained),
                    _pick(mask, new_draw, old_draw))
    return Rehearsal(_by_object(mask, first, second, retained), _by_object(~mask, first, second, current),
                     start, second.experience.state)


def edge_features(prefix, rehearsal):
    """Use only native B proposal and two endpoints' activity and feedback for each edge."""
    current, _ = site._feature_components(prefix, None, None)
    addresses = prefix.candidates.address
    receiver, sender = addresses // net.NODES, addresses % net.NODES
    sign = prefix.candidates.amount.sign()[..., None]

    def features(read):
        activity = read.action.state.hidden
        send, receive = site._node(activity, sender), site._node(activity, receiver)
        response = site._node(read.experience.feedback[:, :, 2], receiver)
        virtual_taste = read.experience.taste[:, None, None]
        endpoint = send * receive
        product = send * response
        return (send, receive, endpoint, response, product, sign * product,
                virtual_taste * endpoint, sign * virtual_taste * endpoint)

    return torch.stack((*current, *features(rehearsal.retained), *features(rehearsal.current)), dim=-1)


@torch.no_grad()
def choose(prefix, rehearsal, weights, *, draw=None, gradient=False):
    """Sample one address per group using a single coefficient row shared by every edge."""
    valid = prefix.candidates.valid
    if draw is not None and draw.shape != valid.shape:
        raise ValueError("Expected one motor-independent draw per lane and group")
    if gradient and draw is None:
        raise ValueError("A score gradient requires a sampled action")
    features = edge_features(prefix, rehearsal)
    weights = torch.as_tensor(weights, dtype=features.dtype, device=features.device)
    if weights.shape != (len(FEATURES),):
        raise ValueError("Expected one coefficient per shared local feature")
    logits = features @ weights
    log_prob = logits.log_softmax(-1)
    probabilities = log_prob.exp()
    index = logits.argmax(-1) if draw is None else (probabilities.cumsum(-1)[..., :-1] <= draw[..., None]).sum(-1)
    chosen = log_prob.gather(-1, index[..., None]).squeeze(-1)
    score = None
    if gradient:
        chosen_features = features.gather(-2, index[..., None, None].expand(-1, -1, 1, len(FEATURES))).squeeze(-2)
        expected = (probabilities[..., None] * features).sum(-2)
        score = ((chosen_features - expected) * valid[..., None]).sum(1)
    return site.Selection(index, torch.where(valid, chosen, 0), features, score)


@torch.no_grad()
def apply(prefix, selection=None, *, mode="selected", accept=None):
    """Attach one final graph to the common live activity without advancing time or replaying a choice."""
    graph = prefix.native_b.state.graph
    if mode == "native":
        return replace(prefix.live, graph=graph)
    group = prefix.candidates
    if mode == "selected":
        index = selection.index if isinstance(selection, site.Selection) else selection
        if index is None or index.shape != group.valid.shape:
            raise ValueError("Expected one index per lane and candidate group")
        addresses = group.address.gather(2, index[..., None]).squeeze(-1)
        amount = group.amount
        if accept is not None:
            if accept.shape != (len(amount),) or accept.dtype != torch.bool:
                raise ValueError("Expected a boolean native-draft acceptance flag per lane")
            amount = amount.masked_fill(accept[:, None], 0)
    elif mode == "uniform":
        addresses = group.address.flatten(1)
        amount = (group.amount[..., None] / site.GROUP_SIZE).expand_as(group.address).flatten(1)
    else:
        raise ValueError(f"Unknown draft revision {mode!r}")
    weights = graph.weights.flatten(1).scatter_add(1, addresses, -amount).reshape_as(graph.weights)
    return replace(prefix.live, graph=replace(graph, weights=weights))


def physical_dose(prefix, state, selection, mode):
    """Check the state isolation and reuse the established per-lane receipt audit."""
    from .edge_site_eval import physical_dose as audit

    for name in ("hidden", "sender_trace", "receiver_trace"):
        if not torch.equal(getattr(prefix.live, name), getattr(state, name)):
            raise AssertionError(f"Draft revision changed the live {name}")
    if selection is not None and not isinstance(selection, site.Selection):
        selection = site.Selection(selection, None)
    return audit(prefix, state, selection, mode)
