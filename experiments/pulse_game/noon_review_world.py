"""Physical menus and native contacts for the 20-node review experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import local_trace_graph as native
from . import local_trace_train as checkpoints
from . import self_param_writer as body


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/active-review-20261120"
EVIDENCE = ROOT / "docs/evidence/active-review-20261120"
PARENT_DATA = ROOT / "data/interleaved-learning-20261120"
COHORTS = {"original": tuple(range(731, 737)), "independent": tuple(range(811, 817))}
PARENTS = {"original": "retention-objective", "independent": "retention-objective-confirmation"}
ARMS = ("native", "random", "uncertain")
PARENT_STAGE = 8000
TRAIN_DOMAIN = 0x61A120
EVAL_DOMAIN = 0x61A121
TRAIN_SEED = 76202117
EVAL_SEEDS = {"screen": {"original": 76202211, "independent": 76202212},
              "heldout": {"original": 76202261, "independent": 76202262}}
CHECKPOINTS = (0, 128, 256, 512, 1000, 2000, 4000, 8000, 16000)
MODE_TICKS = 2
ECHO_TICKS = 3
GAP = 4
CONTACT_TICKS = sum((native.REACTION_TICKS, native.CONTACT_TICKS, native.TASTE_TICKS,
                     native.PRE_WRITE_TICKS, native.POST_WRITE_TICKS))


@dataclass(frozen=True)
class Tape:
    odor: torch.Tensor
    law: torch.Tensor
    study_orientation: torch.Tensor
    study_draw: torch.Tensor
    candidates: torch.Tensor
    menu_orientation: torch.Tensor
    decision_draw: torch.Tensor
    external_draw: torch.Tensor
    review_orientation: torch.Tensor
    review_draw: torch.Tensor
    final_orientation: torch.Tensor
    final_draw: torch.Tensor
    final_item: torch.Tensor

    @property
    def lanes(self):
        return len(self.odor)

    @property
    def count(self):
        return self.odor.shape[1]


def make_tape(seed, worlds, count, *, births=(), update=None):
    """Generate root-major complementary histories and common public randomness."""
    if not (count >= 2 and worlds >= 1 and bool(births) == (update is not None)):
        raise ValueError("Tape requires two or more objects and paired training birth/update arguments")
    roots = ([TRAIN_DOMAIN, seed, root, update, count] for root in births) if births else (
        [EVAL_DOMAIN, seed, count],)

    def generate(entropy):
        geometry, private, menus, draws = (np.random.default_rng(part) for part in
                                            np.random.SeedSequence(entropy).spawn(4))

        def signs(rng, shape):
            return (2 * rng.integers(0, 2, shape) - 1).astype(np.float32)

        points = geometry.normal(size=(worlds, count, 3))
        points = (1.35 * points / np.linalg.norm(points, axis=2, keepdims=True)).astype(np.float32)
        law = signs(private, (worlds, count))
        assignments = np.stack((law, -law), axis=1).reshape(worlds * 2, count)
        candidates = menus.permuted(np.broadcast_to(np.arange(count), (worlds, count)), axis=1)[:, :2]
        public = (points, signs(menus, (worlds, count)),
                  draws.random((worlds, count, native.EDIT_DRAWS + 1), dtype=np.float32),
                  candidates, signs(menus, (worlds, 2)), draws.random(worlds, dtype=np.float32),
                  draws.random(worlds, dtype=np.float32), signs(menus, worlds),
                  draws.random((worlds, native.EDIT_DRAWS + 1), dtype=np.float32),
                  signs(menus, (worlds, count)), draws.random((worlds, count), dtype=np.float32),
                  menus.integers(0, count, worlds))
        odor, *rest = (np.repeat(value, 2, axis=0) for value in public)
        return odor, assignments, *rest

    return Tape(*(torch.from_numpy(np.concatenate(blocks, axis=0)) for blocks in
                  zip(*(generate(entropy) for entropy in roots), strict=True)))


def parent_path(cohort):
    return PARENT_DATA / PARENTS[cohort]


def load_parent(cohort, birth):
    return checkpoints.load_checkpoint(parent_path(cohort), "odor", "uniform", birth, PARENT_STAGE)


def load_frozen(cohort, arm, birth, step, *, path=DATA / "core"):
    return checkpoints.load_checkpoint(path / cohort, "odor", arm, birth, step)


def _rows(tape):
    return torch.arange(tape.lanes, device=tape.odor.device)


def sensory_for(tape, item, orientation):
    """Public collection menu only. Truth is accessed separately at contact."""
    left = tape.odor[_rows(tape), item] * orientation[:, None]
    return left, -left


@dataclass(frozen=True)
class Contact:
    state: native.State
    motor: native.Action
    earned: torch.Tensor
    edit: native.Commit


def collect(rule, state, left, right, truth, draw, *, write=True):
    motor = native.act(rule, state, left, right, draw[:, 0])
    earned = truth * motor.side
    experience = native.observe(rule, motor, left, right, earned)
    commit = native.commit(rule, experience, draw[:, 1:], intervention="active" if write else "sham")
    return Contact(native.wait(rule, commit.state, GAP), motor, earned, commit)


def introduce(rule, state, tape, *, write=True):
    """Ordinary contacts have the original antisymmetric same-object physical menu."""
    contacts = []
    for item in range(tape.count):
        sign = tape.study_orientation[:, item]
        left, right = sensory_for(tape, item, sign)
        contact = collect(rule, state, left, right, tape.law[:, item] * sign,
                          tape.study_draw[:, item], write=write)
        state = contact.state
        contacts.append(contact)
    return state, contacts


def candidate_menu(tape):
    """The selector receives two observable odors and no other tape fields."""
    odors = tape.odor[_rows(tape)[:, None], tape.candidates] * tape.menu_orientation[..., None]
    return odors[:, 0], odors[:, 1]


def select(rule, state, left, right, draw):
    """Use the native motor ports after a public, taste-free review mode pulse."""
    mode = state.graph.bias.new_tensor((0., 0., .85))[None].expand_as(left)
    state = native._advance(rule, state, body.sensory(state.graph, mode, mode), MODE_TICKS)
    return native.act(rule, state, left, right, draw)


def selected_item(tape, side):
    return tape.candidates[_rows(tape), (side > 0).long()]


def echo_choice(rule, state, side):
    neutral = state.graph.bias.new_zeros((len(side), 3))
    stimulus = body.sensory(state.graph, neutral, neutral, side=side)
    return native._advance(rule, state, stimulus, ECHO_TICKS)


def review(rule, post_menu, tape, side, *, content_side=None, write=True, contact=True):
    """The environment executes the side and only then reveals earned taste."""
    state = echo_choice(rule, post_menu, side)
    if not contact:
        return native.wait(rule, state, CONTACT_TICKS + GAP), None
    item = selected_item(tape, side if content_side is None else content_side)
    sign = tape.review_orientation
    left, right = sensory_for(tape, item, sign)
    result = collect(rule, state, left, right, tape.law[_rows(tape), item] * sign,
                     tape.review_draw, write=write)
    return result.state, result


def final_probe(rule, state, tape, item):
    rows = _rows(tape)
    sign = tape.final_orientation[rows, item]
    left, right = sensory_for(tape, item, sign)
    action = native.act(rule, state, left, right, tape.final_draw[rows, item])
    return action, tape.law[rows, item] * sign


def random_side(tape):
    return 2 * (tape.external_draw < .5).to(tape.odor.dtype) - 1


@torch.no_grad()
def parent_uncertainty(rule, parent_state, tape):
    """A comparison policy using the frozen parent's read-only two-orientation confidence."""
    confidence = []
    for index in range(2):
        candidate = tape.candidates[:, index]
        sign = tape.menu_orientation[:, index]
        left, right = sensory_for(tape, candidate, sign)
        forward = native.act(rule, parent_state, left, right, stochastic=False).motor_logit
        reverse = native.act(rule, parent_state, right, left, stochastic=False).motor_logit
        confidence.append((forward - reverse).abs() * .5)
    left, right = confidence
    fair = random_side(tape)
    side = torch.where(left == right, fair, torch.where(left < right, -torch.ones_like(fair), torch.ones_like(fair)))
    return side, torch.stack(confidence, dim=1)
