"""Select the location of a fixed B write by rehearsing the two actual support receipts."""

import argparse
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import edge_site_train as sites
from . import interleave_interference as interference
from . import interleave_revisit_train as storage
from . import local_trace_graph as net
from . import noon_ticket as parents


ROOT = parents.ROOT
DATA = ROOT / "data/edge-site-20261120/replay"
EVIDENCE = ROOT / "docs/evidence/edge-site-20261120"
SEED = 49017073
ANGLES = (15, 30, 90)
POLICIES = ("replay", "b_only", "wrong_old", "random", "uniform", "native")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def advance(graph, hidden, stimulus, ticks):
    return net.native.advance(graph, hidden, stimulus, ticks)


def response(graph, hidden, menu):
    stimulus = net.native.sensory(graph, -menu, menu)
    new = advance(graph, hidden, stimulus, net.REACTION_TICKS)
    return new, 2.5*(new[:,11]-new[:,10])


def expected_scores(state, receipts):
    graph = state.graph
    hidden = advance(graph,state.hidden,torch.zeros_like(graph.bias),4)
    readings = []
    for menu, remembered_side in receipts:
        # Both queries are read-only restagings of one previously tasted cue.
        _,plus = response(graph, hidden, menu)
        _,minus = response(graph, hidden, -menu)
        readings.append(((plus*remembered_side).sigmoid()+(-minus*remembered_side).sigmoid())*.5)
    return torch.stack(readings, dim=-1)


def choose(prefix, receipts, random_addresses):
    """Only the body and the menu/side reconstructed from each previous taste are available here."""
    reference = sites.apply(prefix, random_addresses)
    base = expected_scores(reference, receipts)
    laws = ((1., (0.,.5,1.,2.)), (-1., (.5,1.,2.)))
    n, groups = random_addresses.shape
    reference_scores = base[:,None,:].expand(n,4,2)
    sweeps = {key:random_addresses.clone() for law,coeffs in laws for key in ((law,w) for w in coeffs)}
    for group in range(groups):
        scores = reference_scores.clone()
        for site in range(4):
            candidate = random_addresses.clone()
            candidate[:,group] = site
            scores[:,site] = expected_scores(sites.apply(prefix,candidate),receipts)
        log_b = scores[:,:,1].clamp_min(1e-12).log()
        for law, coeffs in laws:
            confidence = scores[:,:,0] if law>0 else 1-scores[:,:,0]
            for weight in coeffs:
                objective = log_b if weight==0 else log_b + weight*confidence.clamp_min(1e-12).log()
                sweeps[(law,weight)][:,group] = objective.argmax(1)

    completed = {key:expected_scores(sites.apply(prefix,address),receipts) for key,address in sweeps.items()}
    selected = {}
    for law, coeffs in laws:
        keys = [(1.,0.) if weight==0 else (law,weight) for weight in ((0.,.5,1.,2.))]
        options = torch.stack([random_addresses,*(sweeps[key] for key in keys)],1)
        predictions = torch.stack([base,*(completed[key] for key in keys)],1)
        known_a = predictions[:,:,0] if law>0 else 1-predictions[:,:,0]
        target = known_a*predictions[:,:,1]
        target = target.masked_fill(predictions[:,:,1]<base[:,1,None]-.01,-torch.inf)
        lane = torch.arange(n)
        selected["replay" if law>0 else "wrong_old"] = options[lane,target.argmax(1)]
    better_b = completed[(1.,0.)][:,1]>base[:,1]
    selected["b_only"] = torch.where(better_b[:,None],sweeps[(1.,0.)],random_addresses)
    selected["random"] = random_addresses
    return selected


def test_world(world, final_draws):
    # Draws and orientations for the actual downstream questions are generated
    # in their own stream, and never passed to the address search.
    orient,first,uniforms = final_draws
    orders = world.orders.clone()
    draws = world.draws.clone()
    orders[:,3:5] = orient
    draws[:,3:5,0] = uniforms
    return replace(world, orders=orders, draws=draws), first


def random_fields(worlds, angle, groups):
    rng = np.random.default_rng([SEED,angle,196])
    def public(x):
        return torch.from_numpy(np.repeat(x,4,axis=0))
    choice = public(rng.integers(4,size=(worlds,groups)))
    rng = np.random.default_rng([SEED,angle,197])
    orient = public(rng.choice(np.array([-1.,1.],dtype=np.float32),(worlds,2)))
    first = public(rng.integers(2,size=worlds))
    draw = public(rng.random((worlds,2),dtype=np.float32))
    return choice,(orient,first,draw)


@torch.no_grad()
def run_birth(birth, worlds=128):
    torch.set_num_threads(1)
    cohort = "original" if birth<800 else "independent"
    g,r = sites.load_parent(birth)
    archive = {}
    for angle in ANGLES:
        world = interference.worlds(SEED,0,worlds,angle)
        prefix = sites.prepare(g,r,world)
        gs = prefix.candidates.address.shape[1]
        random,final_draws = random_fields(worlds,angle,gs)
        _,_,test_draws = final_draws
        downstream,first = test_world(world,final_draws)
        receipts = (
            (world.a*world.orders[:,0,None],prefix.action_a.side*prefix.experience_a.taste),
            (world.b*world.orders[:,1,None],prefix.action_b.side*prefix.experience_b.taste),
        )
        picks = choose(prefix,receipts,random)
        full = sites.apply(prefix,random,mode="native")
        for policy in POLICIES:
            addresses = picks.get(policy,random)
            mode = policy if policy in ("native","uniform") else "selected"
            state = sites.apply(prefix,addresses,mode=mode)
            path = f"{angle}/{policy}"
            if policy in picks:
                archive[path+"/addresses"] = addresses.numpy().astype(np.int8)
            difference = (full.graph.weights-state.graph.weights).flatten(1)
            archive[path+"/positive"] = difference.clamp_min(0).sum(1).numpy()
            archive[path+"/negative"] = (-difference).clamp_min(0).sum(1).numpy()
            archive[path+"/edges"] = difference.ne(0).sum(1).numpy()
            for delay in (4,32):
                result = sites.probe(prefix.rule,state,downstream,wait=delay,order=first)
                archive[path+f"/{delay}/action"] = result.side.to(torch.int8).numpy()
                target = torch.stack((world.law_a,world.law_b),1)*final_draws[0]
                assert torch.equal(result.correct, result.side==target)
                archive[f"{angle}/target"] = target.to(torch.int8).numpy()
        archive[f"{angle}/groups"] = prefix.candidates.valid.sum(1).numpy()
        archive[f"{angle}/final_draws"] = test_draws.numpy()
        archive[f"{angle}/laws"] = torch.stack((world.law_a,world.law_b),1).numpy()
    path = parents.old.checkpoint_path(parents.source(cohort),"odor","uniform",birth,8000)
    output = DATA if worlds==128 else DATA/"smoke"/str(worlds)
    used = (sites.apply,sites.probe,sites.prepare,sites.prepare_b,sites.group_proposals,sites._contact)
    physics = hashlib.sha256("".join(inspect.getsource(f) for f in used).encode()).hexdigest()
    storage._write_raw(output/f"birth-{birth}.npz", **archive, birth=birth, worlds=worlds, seed=SEED,
                       body=digest(path), source=digest(__file__), physics=physics,
                       plan=digest(EVIDENCE/"replay-plan.md"))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--birth",type=int,required=True)
    ap.add_argument("--worlds",type=int,default=128)
    args=ap.parse_args()
    run_birth(args.birth,args.worlds)
