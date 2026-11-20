"""Fork one post-contact state into cue-only A and B queries with a common motor draw."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_train as train
from .interleave_development import estimate


DOMAIN = 2201147
ARM = "local_active"
OUT = train.DEFAULT_OUTPUT.parent / "context-fork"
BRANCHES = ("pre", "post", "neutral17", "target_write_disabled", "post_graph_only")
METRICS = ("a_correct", "b_correct", "both_correct", "different_sides", "correct_probability_a",
           "correct_probability_b", "older_correct", "newer_correct", "correct_minus_wrong_split")
CONTACT_TICKS = (train.local.REACTION_TICKS + train.local.CONTACT_TICKS + train.local.TASTE_TICKS
                 + train.local.PRE_WRITE_TICKS + train.local.POST_WRITE_TICKS)


def make_bank(config, pairs, batch):
    """Both arms and all births replay the exact same disjoint worlds."""
    return tuple(train.make_tape((0,), replace(config, pairs=min(batch, pairs - start)), start,
                                 domain=DOMAIN) for start in range(0, pairs, batch))


def _cue_queries(tape):
    context, probe, law = (value.flatten(0, 2) for value in (tape.context, tape.probe, tape.law))
    lanes = torch.arange(len(context))
    introduction = tuple((context == item).to(torch.int8).argmax(dim=1) for item in (0, 1))
    cutoff = torch.maximum(*introduction)
    phase = (probe == 0).to(torch.int8).argmax(dim=1)
    left, right = (value.flatten(0, 2) for value in (tape.left, tape.right))
    odor = (right[lanes, phase] - left[lanes, phase]) / 2
    cues = torch.stack(tuple((left[lanes, at] + right[lanes, at]) / 2 for at in introduction), dim=1)
    target_a = tape.right_factor.flatten(0, 2)[lanes, phase]
    target_b = target_a * law[:, 0] * law[:, 1]
    draw = tape.draws.flatten(0, 2)[lanes, phase, 0]
    newer_is_b = introduction[1] > introduction[0]
    return cutoff, cues, odor, draw, torch.stack((target_a, target_b), dim=1), newer_is_b


def _subset_rule(rule, lanes):
    return replace(rule, **{field.name: getattr(rule, field.name)[lanes] for field in fields(rule)})


def _probe(rule, state, cues, odor, draw):
    probes = [train.local.act(rule, state, cue - odor, cue + odor, draw) for cue in cues.unbind(1)]
    return tuple(torch.stack([getattr(result, field) for result in probes], dim=1)
                 for field in ("side", "motor_logit"))


@torch.no_grad()
def replay(graph, rule, tape, max_wait):
    """Observe the second A/B introduction once and branch only the frozen query state."""
    if tape.batch_shape[0] != 1:
        raise ValueError("Use one shared batch of cue-fork worlds")
    cutoff, cues, odor, draw, target, newer_is_b = _cue_queries(tape)
    lanes = len(cutoff)
    controller = train.previous.LocalController(graph, rule, lanes)
    left, right, draws, waits, right_factor, fixed_factor = (
        value.flatten(0, 2).transpose(0, 1) for value in
        (tape.left, tape.right, tape.draws, tape.waits, tape.right_factor, tape.fixed_factor)
    )
    side = torch.empty((len(BRANCHES), lanes, 2))
    logit = torch.empty_like(side)
    for contact in range(int(cutoff.max()) + 1):
        if contact and max_wait:
            controller.wait(waits[contact], max_wait)
        a, b, u = left[contact], right[contact], draws[contact]
        pre = controller.state
        action = controller.act(a, b, u)
        observed = right_factor[contact] * action.side + fixed_factor[contact]
        experience = train.local.observe(controller.rule, action, a, b, observed)
        commit = train.local.commit(controller.rule, experience, u[:, 1:5])
        controller.state = commit.state
        picks = (cutoff == contact).nonzero(as_tuple=True)[0]
        if not len(picks):
            continue
        coeff = _subset_rule(controller.rule, picks)
        initial, written = pre.subset(picks), commit.state.subset(picks)
        blank = train.local.wait(coeff, initial, CONTACT_TICKS)
        subset = replace(experience, state=experience.state.subset(picks), side=experience.side[picks],
                         taste=experience.taste[picks], feedback=experience.feedback[picks])
        no_write = train.local.commit(coeff, subset, u[picks, 1:5], intervention="disabled").state
        graph_only = train.local.start(written.graph)
        for branch, state in enumerate((initial, written, blank, no_write, graph_only)):
            side[branch, picks], logit[branch, picks] = _probe(coeff, state, cues[picks], odor[picks], draw[picks])
    shape = (len(BRANCHES), *tape.batch_shape[1:], 2)
    return (side.reshape(shape).numpy(), logit.reshape(shape).numpy(), target.reshape(shape[1:]).numpy(),
            newer_is_b.reshape(tape.batch_shape[1:]).numpy())


def _measure(sides, logits, target, newer_is_b, agreeing):
    correct = sides == target
    a, b = correct[..., 0], correct[..., 1]
    different = sides[..., 0] != sides[..., 1]
    p_a, p_b = np.moveaxis(torch.from_numpy(logits * target).sigmoid().numpy(), -1, 0)
    older, newer = np.where(newer_is_b, a, b), np.where(newer_is_b, b, a)
    aligned_split = different * (a.astype(np.int8) + b.astype(np.int8) - 1)
    values = np.stack((a, b, a & b, different, p_a, p_b, older, newer, aligned_split), axis=-1)
    strata = np.equal(target[..., 0], target[..., 1]) == agreeing
    values = np.where(strata[..., None], values, np.nan)
    return values.mean(axis=-2)


def run(path, checkpoints, births, pairs, batch, seed, bootstrap):
    torch.set_num_threads(1)
    config = train.Config(world_seed=seed)
    bank = make_bank(config, pairs, batch)
    rng = np.random.default_rng([DOMAIN, seed, 300])
    boot_birth = rng.integers(len(births), size=(bootstrap, len(births)))
    boot_pair = rng.integers(pairs, size=(bootstrap, pairs))
    result = dict(births=births, pairs_per_birth=pairs, domain=DOMAIN, seed=seed, bootstrap=bootstrap,
                  branches=BRANCHES, metrics=METRICS, checkpoint_sha256={}, results={},
                  source_sha256={Path(file).name: sha256(Path(file).read_bytes()).hexdigest()
                                 for file in (__file__, train.__file__, train.local.__file__)},
                  protocol="A and B choose from the same antipodal odor pair with identical orientation "
                           "and motor draw. The bilateral ambient vector alone switches. Every probe forks "
                           "one immutable state and carries no feedback. Pre probes immediately precede the "
                           "second A/B introduction; post probes follow its actual contact, taste and write. "
                           "Neutral17 advances equal time without the contact. Target-write-disabled executes "
                           "the same contact with the same sampled motor, taste and disabled write. Post-graph-only "
                           "clears the hidden state and traces after the actual contact. The later cue and "
                           "independent true laws vary across held-out worlds. Both-correct and different-sides "
                           "use a shared uniform draw by design.")
    raw = {}

    def summarize(values):
        return {metric: estimate(values[..., index], boot_birth, boot_pair) for index, metric in enumerate(METRICS)}

    for label, source, step in checkpoints:
        file = train.checkpoint_path(source, ARM, step)
        result["checkpoint_sha256"][label] = sha256(file.read_bytes()).hexdigest()
        state = []
        for birth in births:
            graph, rule = train.load_checkpoint(source, ARM, birth, step)
            blocks = [replay(graph, rule, tape, config.max_wait) for tape in bank]
            state.append(tuple(np.concatenate(sequence, axis=axis) for sequence, axis in
                               zip(zip(*blocks), (1, 1, 0, 0), strict=True)))
        side, logit, target, newer = (np.stack(field) for field in zip(*state, strict=True))
        raw.update({f"{label}/side": side, f"{label}/logit": logit, f"{label}/target": target,
                    f"{label}/newer_is_b": newer})
        tables = {}
        for stratum, agreeing in (("agree", True), ("conflict", False)):
            metrics = _measure(side, logit, target[:, None], newer[:, None], agreeing)
            estimates = {branch: summarize(metrics[:, i]) for i, branch in enumerate(BRANCHES)}
            estimates.update({f"post_minus_{branch}": summarize(metrics[:, 1] - metrics[:, i])
                              for i, branch in enumerate(BRANCHES) if branch != "post"})
            tables[stratum] = estimates
        result["results"][label] = tables
        print(f"model={label} step={step} conflict_both={tables['conflict']['post']['both_correct']['mean']:.3f}", flush=True)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "by-pair.npz", **raw)
    (path / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs=3, metavar=("LABEL", "RUN", "STEP"), action="append", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--births", type=int, nargs="+", default=train.BIRTHS)
    parser.add_argument("--pairs", type=int, default=192)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=26921)
    parser.add_argument("--bootstrap", type=int, default=2000)
    options = parser.parse_args()
    checkpoints = [(label, Path(path), int(step)) for label, path, step in options.checkpoint]
    if (len({checkpoint[0] for checkpoint in checkpoints}) != len(checkpoints)
            or min(options.pairs, options.batch, options.bootstrap) < 2):
        parser.error("Use distinct checkpoint labels and at least two pairs, draws and worlds per batch")
    run(options.out, checkpoints, options.births, options.pairs, options.batch, options.seed, options.bootstrap)
