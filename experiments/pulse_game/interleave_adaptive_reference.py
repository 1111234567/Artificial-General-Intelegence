"""Fit a receiver-local stabilization gain from the resource of a single actual terminal choice."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import interleave_homeostasis as home, interleave_slow_reference as slow
from . import local_trace_graph as net, local_trace_train as old, revisit_choice_graph as motor
from .interleave_interference import DATA, EVIDENCE


DEST = DATA / "adaptive-reference"
PLAN = EVIDENCE / "adaptive-reference-plan.json"
STEPS = 2048
SEEDS = (612371, 612379, 612391)
FITS = ("constant", "adaptive")
READS = ("old_before", "old_after", "B", "new_X", "all_three", "old_and_X")
GAIN0 = float(np.log(.12 / .88))


@torch.no_grad()
def contact(rule, memory, right, law, draw, coefficients, location, shuffle):
    action = net.act(rule, memory.live, -right, right, draw[:, 0])
    experience = net.observe(rule, action, -right, right, action.side * law)
    committed = net.commit(rule, experience, draw[:, 1:])
    post = committed.state.graph
    surviving = memory.ancestry & post.present
    old = memory.live.graph
    divergence = ((old.weights - memory.slow_w).abs() * memory.ancestry).sum(-1)
    divergence = divergence / memory.ancestry.sum(-1).clamp_min(1)
    feature = torch.stack((experience.feedback[:, :, 1], experience.feedback[:, :, 2], divergence), dim=2)
    logits = coefficients[:, :1] + (feature * coefficients[:, None, 1:]).sum(2)
    gain = logits.sigmoid()
    if location == "permuted":
        gain = gain[:, shuffle]
    elif location == "mean":
        gain = gain.mean(1, keepdim=True)
    w = torch.where(surviving, net._nonzero(post.weights.lerp(memory.slow_w, gain[:, :, None])), post.weights)
    b = post.bias.lerp(memory.slow_b, gain)
    ready = net.wait(rule, experience.state, net.PRE_WRITE_TICKS)
    fast = net.wait(rule, replace(ready, graph=replace(post, weights=w, bias=b)), net.POST_WRITE_TICKS)
    next_w = torch.where(surviving, memory.slow_w.lerp(w, .02), memory.slow_w)
    next_b = memory.slow_b.lerp(b, .02)
    return slow.Memory(net.wait(rule, fast, home.WAIT), next_w, next_b, surviving)


@torch.no_grad()
def life(g, r, tape_set, coefficients, stages, ecology, exposures, *, query=None, location="local", shuffle=None):
    """Use four paired histories until X, introduce X's two laws only when X appears."""
    tape, extra, extra_side, extra_u = tape_set
    n, variants = tape.pairs, len(coefficients)
    count = 4 * n * variants
    def copy(tensor):
        return tensor.repeat((variants,) + (1,) * (tensor.ndim - 1))
    w = torch.as_tensor(coefficients, dtype=g.bias.dtype).repeat_interleave(4*n, dim=0)
    rows = torch.arange(n)
    view, double = r.expand(count), r.expand(2*count)
    start = net.start(g, count)
    m = slow.Memory(start, start.graph.weights, start.graph.bias, start.graph.present)
    laws = torch.tensor([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]]).repeat(n, 1)
    def event(mem, ids, ori, draw):
        cue = (tape.objects[rows, ids] * ori[:, None]).repeat_interleave(4, 0)
        sign = torch.where(ids.repeat_interleave(4) == 0, laws[:, 0], laws[:, 1])
        factor = (tape.laws[rows, ids]*ori).repeat_interleave(4) * sign
        return contact(view, mem, copy(cue), copy(factor), copy(draw.repeat_interleave(4, 0)), w, location, shuffle)
    visits = tape.visits[home.ECOLOGIES.index(ecology)]
    for i in range(2):
        m = event(m, visits[i], tape.orientation[i], tape.draws[i])
    if exposures == 4:
        for i in range(6):
            m = event(m, extra[i], extra_side[i], extra_u[i])
    twin = torch.arange(count).repeat_interleave(2)
    x = copy((tape.objects[:, 2] * tape.x_orientation[:, None]).repeat_interleave(8, 0))
    sign_x = torch.tensor([-1., 1.]).repeat(4*n)
    fx = copy(tape.x_orientation.repeat_interleave(8) * sign_x)
    ux = copy(tape.x_draws.repeat_interleave(8, 0))
    menus = copy((tape.objects[:, :3] * tape.probe_orientation[:, :, None]).repeat_interleave(8, 0))
    ab = (tape.laws[:, :2].repeat_interleave(4, 0) * laws).repeat_interleave(2, 0)
    values = copy(torch.cat((ab, sign_x[:, None]), 1) * tape.probe_orientation.repeat_interleave(8, 0))
    uniform = copy(tape.probe_draws.repeat_interleave(8, 0))
    outcomes = []
    for k in range(max(stages)+1):
        if k in stages:
            snap = m.subset(twin)
            after = contact(double, snap, x, fx, ux, w[twin], location, shuffle)
            if query is not None:
                # Only this action is taken in a training life. Its realized resource drives the outer update.
                idx = copy(torch.as_tensor(query).repeat_interleave(8))
                lane = torch.arange(len(idx))
                menu = menus[lane, idx]
                a = motor.act(after.live, -menu, menu, uniform[lane, idx])
                reward = a.side * values[lane, idx]
                return reward.reshape(variants, n, 8).numpy()
            first = motor.act(snap.live, -menus[:, 0], menus[:, 0], uniform[:, 0]).side
            choices = torch.stack([motor.act(after.live, -menus[:, i], menus[:, i], uniform[:, i]).side
                                   for i in range(3)], 1)
            good = choices == values
            bits = torch.stack((first == values[:, 0], *good.unbind(1), good.all(1), good[:, 0]&good[:, 2]), 1)
            outcomes.append(bits.reshape(variants,n,4,2,len(READS)).numpy())
        if k < max(stages):
            i = 2+k
            m = event(m, visits[i], tape.orientation[i], tape.draws[i])
    return np.stack(outcomes, axis=1)


def fitted(seed):
    return DEST / str(seed) / "fit.npz"


def fit(seed, stop=STEPS):
    torch.set_num_threads(1)
    dst = fitted(seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    plan = dict(seed=seed, steps=STEPS, worlds=6, perturbation_pairs=4, lr=.025, sigma=.4,
                plan_sha256=sha256(PLAN.read_bytes()).hexdigest(), code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    home._lock(dst.parent/"run.json", plan)
    parents = [old.load_checkpoint(DATA/"retention-objective","odor","uniform",b,8000) for b in home.CAL_BIRTHS]
    theta = np.array([[GAIN0, 0., 0., 0.]] * 2, dtype=np.float64)
    first, m, v, trace = 0, np.zeros_like(theta), np.zeros_like(theta), []
    if dst.is_file():
        with np.load(dst) as z:
            first=int(z["step"]);theta=z["theta"];m=z["m"];v=z["v"];trace=z["trace"].tolist()
    t0 = perf_counter()
    for step in range(first+1, stop+1):
        rng = np.random.default_rng([seed, step, 1771])
        noise = rng.normal(size=(plan["perturbation_pairs"],4))
        direction = np.broadcast_to(noise[None], (2,*noise.shape)).copy()
        direction[0,:,1:] = 0
        weights = theta[:,None,None,:] + plan["sigma"]*direction[:,:,None,:]*np.array([1.,-1.])[None,None,:,None]
        g,r = parents[int(rng.integers(len(parents)))]
        age = int(rng.choice((4,8,16,32)))
        doses = int(rng.choice((1,4)))
        eco = home.ECOLOGIES[int(rng.integers(2))]
        tape = slow.public_tapes(plan["worlds"], int(rng.integers(2**31)), 844759)
        # Each balanced group of three worlds asks one question about each of A, B, X.
        query = rng.permutation(np.tile(np.arange(3), plan["worlds"]//3))
        returns = life(g,r,tape,weights.reshape(-1,4),[age],eco,doses,query=query)
        value = returns.mean((1,2)).reshape(2, len(noise), 2)
        grad = ((value[:,:,0]-value[:,:,1])[:,:,None]*direction).mean(1)/(2*plan["sigma"])
        m = .9*m+.1*grad
        v = .999*v+.001*grad*grad
        theta += plan["lr"]*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        trace.append(np.concatenate((value.mean((1,2)),theta.ravel())).tolist())
        if step%64 == 0 or step == stop:
            home._save_raw(dst,dict(step=step,theta=theta,m=m,v=v,trace=trace))
            if step in (512,1024,STEPS):
                home._save_raw(dst.with_name(f"stage-{step}.npz"),dict(step=step,theta=theta))
            print(seed,step,round((perf_counter()-t0)/(step-first),3),theta.round(2),flush=True)


def verify(seed):
    torch.set_num_threads(1)
    path = fitted(seed)
    with np.load(path) as z:
        if int(z["step"]) != STEPS:
            raise ValueError("The terminal rule has not been fitted")
        theta = z["theta"]
    protocol = json.loads((DATA/"slow-reference/run.json").read_text())
    n = protocol["test_worlds"]
    tapes = slow.public_tapes(n,6341879,732193)
    destination = path.parent/"validation"
    destination.mkdir(exist_ok=True)
    for b in home.CONFIRM_BIRTHS:
        g,r = old.load_checkpoint(DATA/"retention-objective-confirmation","odor","uniform",b,8000)
        perm = torch.as_tensor(np.random.default_rng([seed,b,9317]).permutation(net.NODES))
        for eco in home.ECOLOGIES:
            for reps in (1,4):
                f = destination/f"{b}-{eco}-{reps}.npz"
                if f.exists():continue
                # The nearest fp32 sigmoid(logit(.12)) differs from the old scalar
                # by one ULP. Replay the historical literal scalar with its native
                # entry point; both fitted arms still use their actual learned logits.
                control = slow.live(g,r,tapes,eco,reps,.12,.02)["correct"][...,:len(READS)]
                raw = life(g,r,tapes,theta,slow.AGES,eco,reps)
                with np.load(DATA/"slow-reference/confirmation"/f"{b}-{eco}-{reps}-selected.npz") as z:
                    np.testing.assert_array_equal(control,z["correct"][...,:len(READS)])
                result = dict(fixed=control,constant=raw[0],adaptive=raw[1])
                for loc in ("permuted","mean"):
                    result[loc] = life(g,r,tapes,theta[1:],slow.AGES,eco,reps,location=loc,shuffle=perm)[0]
                home._save_raw(f,result)
        print("verified",seed,b,flush=True)


def resume_fit(seed,stop):
    """Keep the exact fitting source associated with a saved optimizer."""
    from importlib.util import module_from_spec, spec_from_file_location
    source=Path(__file__).read_bytes()
    digest=sha256(source).hexdigest()
    manifest=fitted(seed).parent/"run.json"
    previous=json.loads(manifest.read_text())["code_sha256"] if manifest.exists() else digest
    archived=DEST/f"fit-source-{previous[:12]}.py"
    if previous==digest:
        DEST.mkdir(parents=True,exist_ok=True)
        if not archived.exists():archived.write_bytes(source)
        return fit(seed,stop)
    saved=archived.read_bytes()
    boundary=b"\ndef verify("
    if sha256(saved).hexdigest()!=previous or saved.split(boundary)[0]!=source.split(boundary)[0]:
        raise ValueError("The fitting implementation differs from the frozen optimizer")
    spec=spec_from_file_location(__package__+"._saved_adaptive_fit",archived)
    frozen=module_from_spec(spec)
    spec.loader.exec_module(frozen)
    return frozen.fit(seed,stop)


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--stop",type=int,default=STEPS)
    p.add_argument("--mode",choices=("fit","verify"),default="fit")
    args=p.parse_args()
    if args.mode=="fit":
        if not 1<=args.stop<=STEPS:raise ValueError("Invalid training endpoint")
        resume_fit(args.seed,args.stop)
    else:verify(args.seed)
