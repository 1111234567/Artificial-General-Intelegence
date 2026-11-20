"""Test what spatial information is necessary to keep a graph plastic through a long life."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_capacity_report as stats, interleave_homeostasis as home
from . import interleave_slow_reference as slow, local_trace_graph as net
from . import local_trace_train as fit, revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE


PLAN = EVIDENCE / "compact-reference-plan.json"
ARMS = ("global_radius", "row_radius", "signed_rows", "indexed_exact")
PRIMARY = [slow.AGES.index(age) for age in (16, 64, 128)]
FIELDS = slow.FIELDS[:6]


def signed_template(values):
    """Actually pack four ternary signs per byte and decode the deployment template."""
    signs = values.sign().numpy().astype(np.int8)
    codes = (signs.ravel() + 1).astype(np.uint8).reshape(-1, 4)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    packed = np.bitwise_or.reduce(codes << shifts, axis=1)
    decoded = ((packed[:, None] >> shifts) & 3).astype(np.int8).reshape(signs.shape) - 1
    np.testing.assert_array_equal(decoded, signs)
    amplitudes = (values.square().sum(-1, keepdim=True) /
                  values.ne(0).sum(-1, keepdim=True).clamp_min(1)).sqrt()
    storage = packed.nbytes + amplitudes.numel() * amplitudes.element_size()
    return torch.from_numpy(decoded).to(values.dtype) * amplitudes, storage


def reference(body, mode):
    row = torch.cat((body.weights * body.present, body.bias.unsqueeze(-1)), dim=-1)
    if mode == "global_radius":
        return row.norm(), 4
    if mode == "row_radius":
        return row.norm(dim=-1, keepdim=True), body.width * 4
    if mode == "signed_rows":
        return signed_template(row)
    if mode == "indexed_exact":
        return row, row.numel() * row.element_size()
    if mode == "native":
        return None, 0
    raise ValueError(mode)


def apply_numeric(post, ref, gain, mode):
    rows = torch.cat((post.weights * post.present, post.bias.unsqueeze(-1)), dim=-1)
    if mode in ("global_radius", "row_radius"):
        axes = (-2, -1) if mode == "global_radius" else (-1,)
        radius = torch.linalg.vector_norm(rows, dim=axes, keepdim=True)
        ratio = torch.where(radius > 0, ref / radius.clamp_min(1e-12), torch.ones_like(radius))
        amended = rows * (1 + gain * (ratio - 1))
        # An outer-trained birth can already exceed the local write limits.
        # Preserve those values when the gain is zero, exactly as commit does.
        w = net._bounded_update(post.weights, amended[..., :-1] - rows[..., :-1], net.WEIGHT_LIMIT)
        b = net._bounded_update(post.bias, amended[..., -1] - post.bias, net.BIAS_LIMIT)
    else:
        eligible = torch.cat(((ref[..., :-1] != 0) & post.present, torch.ones_like(rows[..., -1:], dtype=torch.bool)), -1)
        amended = torch.where(eligible, rows.lerp(ref, gain), rows)
        w, b = amended[..., :-1], amended[..., -1]
    return replace(post, weights=torch.where(post.present, net._nonzero(w), post.weights), bias=b)


def contact(rule, state, right, factor, u, ref, gain, mode):
    a = net.act(rule, state, -right, right, u[:, 0])
    e = net.observe(rule, a, -right, right, a.side * factor)
    result = net.commit(rule, e, u[:, 1:])
    after = result.state
    if mode != "native":
        amended = apply_numeric(after.graph, ref, gain, mode)
        ready = net.wait(rule, e.state, net.PRE_WRITE_TICKS)
        after = net.wait(rule, replace(ready, graph=amended), net.POST_WRITE_TICKS)
    return net.wait(rule, after, home.WAIT)


@torch.no_grad()
def evaluate(g, r, tapes, ecology, exposures, mode, gains):
    tape, extra_ids, extra_side, extra_draws = tapes
    n, v = tape.pairs, len(gains)
    lanes = n * v * 4
    state = net.start(g, lanes)
    view, double = r.expand(lanes), r.expand(2 * lanes)
    ref, storage = reference(g, mode)
    gamma = torch.tensor(gains).repeat_interleave(4*n).reshape(-1, 1, 1)
    def copies(x):
        return x.repeat((v,) + (1,) * (x.ndim - 1))
    rows = torch.arange(n)
    laws = torch.tensor([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]]).repeat(n, 1)
    def visit(state, ids, orientation, u):
        right = (tape.objects[rows, ids] * orientation[:, None]).repeat_interleave(4, 0)
        sign = torch.where(ids.repeat_interleave(4) == 0, laws[:, 0], laws[:, 1])
        factor = (tape.laws[rows, ids] * orientation).repeat_interleave(4) * sign
        return contact(view, state, copies(right), copies(factor), copies(u.repeat_interleave(4, 0)), ref, gamma, mode)
    order = tape.visits[home.ECOLOGIES.index(ecology)]
    for t in range(2):
        state = visit(state, order[t], tape.orientation[t], tape.draws[t])
    if exposures == 4:
        for t in range(6):
            state = visit(state, extra_ids[t], extra_side[t], extra_draws[t])
    twin = torch.arange(lanes).repeat_interleave(2)
    x = copies((tape.objects[:, 2] * tape.x_orientation[:, None]).repeat_interleave(8, 0))
    xlaw = torch.tensor([-1., 1.]).repeat(4*n)
    factor = copies(tape.x_orientation.repeat_interleave(8) * xlaw)
    u = copies(tape.x_draws.repeat_interleave(8, 0))
    menus = copies((tape.objects[:, :3] * tape.probe_orientation[..., None]).repeat_interleave(8, 0))
    ab = (tape.laws[:, :2].repeat_interleave(4, 0) * laws).repeat_interleave(2, 0)
    truth = copies(torch.cat((ab, xlaw[:, None]), 1) * tape.probe_orientation.repeat_interleave(8, 0))
    draw = copies(tape.probe_draws.repeat_interleave(8, 0))
    answers, outcomes = [], []
    for k in range(max(slow.AGES) + 1):
        if k in slow.AGES:
            snapshot = state.subset(twin)
            a = motor.act(snapshot, -menus[:, 0], menus[:, 0], draw[:, 0]).side
            learned = contact(double, snapshot, x, factor, u, ref, gamma[twin], mode)
            side = torch.stack([motor.act(learned, -menus[:, i], menus[:, i], draw[:, i]).side for i in range(3)], 1)
            correct = side == truth
            bits = torch.stack((a == truth[:, 0], *correct.unbind(1), correct.all(1), correct[:, 0] & correct[:, 2]), 1)
            outcomes.append(bits.reshape(v, n, 4, 2, len(FIELDS)).numpy())
            answers.append(side.reshape(v, n, 4, 2, 3).to(torch.int8).numpy())
        if k < max(slow.AGES):
            t = k + 2
            state = visit(state, order[t], tape.orientation[t], tape.draws[t])
    return np.stack(outcomes, axis=1), np.stack(answers, axis=1), storage


def run():
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    dest = DATA / "compact-reference"
    dest.mkdir(parents=True, exist_ok=True)
    home._lock(dest/"run.json", dict(plan_sha256=sha256(PLAN.read_bytes()).hexdigest(),
                                     code_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    selection, hashes, cost = {}, {}, {}
    for stage, path in (("calibration", "retention-objective"), ("confirmation", "retention-objective-confirmation")):
        spec = plan[stage]
        tapes = slow.public_tapes(spec["worlds"], spec["seed"], spec["domain"])
        evals = {}
        for b in spec["births"]:
            g, r = fit.load_checkpoint(DATA/path, "odor", "uniform", b, 8000)
            for eco, reps in ([("absent_a", 4)] if stage == "calibration" else
                              [(eco, rep) for eco in home.ECOLOGIES for rep in (1,4)]):
                for mode in (ARMS if stage == "calibration" else ("native", *ARMS)):
                    gains = spec["gamma_grid"] if stage == "calibration" else [selection[mode] if mode != "native" else 0.]
                    f = dest / f"{stage}-{b}-{eco}-{reps}-{mode}.npz"
                    if not f.exists():
                        bits, sides, nbytes = evaluate(g, r, tapes, eco, reps, mode, gains)
                        home._save_raw(f, dict(correct=bits, sides=sides, gains=gains, template_bytes=nbytes))
                    with np.load(f) as z:
                        np.testing.assert_array_equal(z["gains"], gains)
                        bits = z["correct"]
                        cost[mode] = int(z["template_bytes"])
                    if stage == "confirmation" and mode == "native":
                        archive = DATA/"slow-reference/confirmation"/f"{b}-{eco}-{reps}-native.npz"
                        with np.load(archive) as z:
                            np.testing.assert_array_equal(bits[0], z["correct"][..., :len(FIELDS)])
                    hashes[f.name] = sha256(f.read_bytes()).hexdigest()
                    evals.setdefault(f"{eco}/{reps}/{mode}", []).append(bits[:, PRIMARY].mean((1, 3, 4)))
            print(stage, b, flush=True)
        if stage == "calibration":
            for mode in ARMS:
                curves = np.stack(evals["absent_a/4/"+mode]).mean((0, 2))
                selection[mode] = spec["gamma_grid"][int(curves[:, 4].argmax())]
            home._lock(dest/"selection.json", dict(gains=selection, calibration_sha256=dict(hashes)))
        else:
            ci = stats.Crossed(len(spec["births"]), spec["worlds"], 617701, 4000)
            data = {k:np.stack(v)[:,0] for k,v in evals.items()}
            for eco in home.ECOLOGIES:
                for rep in (1,4):
                    start = f"{eco}/{rep}/"
                    for old in ("static", "selected"):
                        rows = []
                        for b in spec["births"]:
                            with np.load(DATA/"slow-reference/confirmation"/f"{b}-{eco}-{rep}-{old}.npz") as z:
                                rows.append(z["correct"][PRIMARY, ..., :len(FIELDS)].mean((0, 2, 3)))
                        data[start+old] = np.stack(rows)
                    for a,b in (("row_radius","native"),("global_radius","native"),("row_radius","global_radius"),
                                ("signed_rows","row_radius"),("indexed_exact","signed_rows"),
                                ("signed_rows","static"),("row_radius","static"),("signed_rows","selected")):
                        data[start+a+" minus "+b] = data[start+a] - data[start+b]
            result = {k:{m:ci(v[:,:,i]) for i,m in enumerate(FIELDS)} for k,v in data.items()}
            report = dict(selected_gamma=selection, stored_reference_bytes=cost, outcomes=result, raw_sha256=hashes,
                          note="Paired crossed bootstrap over six independent births and shared geometry. Age and "
                               "opposing private outcomes are retained together. All model actions are scored before "
                               "the next feedback. Baseline bits reproduce the archived native and asymmetric contrasts "
                               "are indicated as first minus second.")
            (EVIDENCE/"compact-reference.json").write_text(json.dumps(report,indent=1)+"\n",encoding="utf8")


if __name__ == "__main__":
    run()
