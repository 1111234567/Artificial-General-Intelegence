"""Add a small physically sensed route residual to a frozen native planning body."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import noon_horizon as env, noon_horizon_eval as metrics


PLAN = env.EVIDENCE / "horizon-anchor-plan.json"
ROOT = env.DATA / "anchored"
LABELS = ("linear_informative", "linear_uninformative", "gated_informative", "gated_uninformative")
WIDTH = env.net.NODES + 1
LENGTH = 4


def settings():
    plan = json.loads(PLAN.read_text(encoding="utf8"))
    seal = env.EVIDENCE / "horizon-seal.json"
    if env.digest(seal) != plan["parent_experiment_sha256"]:
        raise ValueError("The previously sealed planning ecology changed")
    return plan, json.loads(seal.read_text(encoding="utf8"))["source_checkpoints"]


def observe_route(rule, state, tape, sign):
    """Perform the same physical mode, cue and menu as the original body."""
    left, right = env.world.candidate_menu(tape)
    pulse = state.graph.bias.new_tensor((0., 0., .85))[None].expand_as(left)
    prior = env.net._advance(rule, state, env.net.native.sensory(state.graph, pulse, pulse), env.world.MODE_TICKS)
    sensory = pulse.clone()
    sensory[:, 0] = sign * 1.35
    state = env.net._advance(rule, prior, env.net.native.sensory(state.graph, sensory, sensory), env.CUE_TICKS)
    response = (state.hidden[:, (0, 3)] - prior.hidden[:, (0, 3)]).mean(1, keepdim=True)
    candidate = env.net.act(rule, state, left, right, tape.decision_draw)
    values = torch.column_stack((torch.ones_like(candidate.side), candidate.state.hidden))
    return candidate, torch.cat((values,values*response), dim=1)


def load_head(path, label, body_sha256):
    """Read only the declared residual and bind it to its original physical body."""
    if label not in LABELS:
        raise ValueError("Unknown route residual")
    with np.load(path, allow_pickle=False) as raw:
        if str(raw["body_sha256"]) != body_sha256 or str(raw["plan_sha256"]) != env.digest(PLAN):
            raise ValueError("Route residual has a different physical parent or protocol")
        params = torch.from_numpy(raw[label])
    size = WIDTH if label.startswith("linear_") else 2*WIDTH
    if tuple(params.shape) != (size,) or not params.isfinite().all():
        raise ValueError("Invalid route residual")
    return params


def route(rule, state, tape, cue, coefficients):
    """Use the observable native route state and choose one side before reward."""
    action, raw = observe_route(rule,state,tape,cue)
    logit = action.motor_logit
    if coefficients is not None:
        logit = logit + (raw[:, :len(coefficients)] * coefficients[None]).sum(1)
    side = 2 * (tape.decision_draw < logit.sigmoid()).to(logit.dtype) - 1
    return action.state, side, logit


def execute(rule, state, tape, future, side, length):
    """Deploy one route on the recipient body, preserving native paid actions."""
    after, contact = env.world.review(rule,state,tape,side)
    tail, final = env.paid_tail(rule,after,tape,future,length)
    review = torch.column_stack((contact.motor.side,contact.earned))[:,None]
    return torch.cat((review,tail),dim=1), final


def deploy(graph, rule, tape, missing, future, cue, length, coefficients=None):
    """Single-path deployment with public cue input and no competing route replay."""
    state, _ = env.initial(graph,rule,tape,missing)
    coeff = rule.expand(tape.lanes)
    post, selected, logit = route(coeff,state,tape,cue,coefficients)
    events, _ = execute(coeff,post,tape,future,selected,length)
    return selected, logit, events


def cross_cues(rule,state,tape,future):
    lanes = tape.lanes
    index = torch.arange(lanes).repeat_interleave(2)
    branch = env.lesson.expand(tape,index)
    sign = state.hidden.new_tensor((-1., 1.)).repeat(lanes)
    action, features = observe_route(rule.expand(2*lanes),state.subset(index),branch,sign)
    return branch, future.subset(index), action, features


@torch.no_grad()
def stage_bank(graph, rule, tape, missing, future):
    state, history = env.initial(graph,rule,tape,missing)
    lanes = tape.lanes
    phase = state.hidden.new_tensor((-1., 1.)).repeat(lanes)
    idx = torch.arange(lanes).repeat_interleave(2)
    branch, future_twice, action, feature = cross_cues(rule,state,tape,future)
    exact = env.signal(rule.expand(2*lanes),state.subset(idx),branch,phase)
    torch.testing.assert_close(exact.motor_logit,action.motor_logit,rtol=0,atol=0)
    expected, _ = env.branches(graph,rule,action.state,branch,future_twice,LENGTH)
    return dict(basis=feature.reshape(lanes,2,2*WIDTH).numpy(),
                native=action.motor_logit.reshape(lanes,2).numpy(),
                events=expected.reshape(lanes,2,2,LENGTH+1,2).to(torch.int8).numpy(),
                phase=future.cue_noise.numpy(),
                candidate=tape.candidates.numpy(), no_taste=(tape.candidates[:,1]==missing).long().numpy(),
                known_history=history.to(torch.int8).numpy(), queries=future.target.numpy())


def matrices(raw):
    lane = torch.arange(len(raw["events"]))[:,None]
    phase = torch.from_numpy(raw["phase"])[:,None] > 0
    horizon = torch.arange(2)[None].expand(len(lane),-1)
    c = (horizon, torch.where(phase,horizon,1-horizon))
    earned = torch.from_numpy(raw["events"][...,1]).float()
    delta = (earned[:,:,1]-earned[:,:,0]).cumsum(-1)
    rows = []
    for k in (0,1,0,1):
        basis = torch.from_numpy(raw["basis"])[lane,c[k]].reshape(-1,8,2*WIDTH)
        native = torch.from_numpy(raw["native"])[lane,c[k]].reshape(-1,8)
        payoff = delta[lane,c[k],horizon*LENGTH].reshape(-1,8)
        rows.append((basis,native,payoff))
    return tuple(torch.stack(parts) for parts in zip(*rows,strict=True))


def fit(raw, plan, birth):
    basis, native, delta = matrices(raw)
    anchor = torch.nn.Parameter(torch.zeros((len(LABELS),2*WIDTH)))
    permitted = torch.ones_like(anchor)
    permitted[:2,WIDTH:] = 0
    step = torch.optim.Adam((anchor,),lr=plan["lr"])
    draw = torch.Generator().manual_seed(plan["optimizer_seed"]+birth)
    trace = []
    for i in range(plan["updates"]):
        take = torch.randint(basis.shape[1],(plan["geometries_per_update"],),generator=draw)
        residual = torch.einsum("atnf,af->atn",basis[:,take],anchor*permitted)
        probability = (native[:,take]+residual).sigmoid()
        advantage = (probability*delta[:,take]).mean((1,2))
        step.zero_grad(set_to_none=True)
        (-advantage.sum()).backward()
        step.step()
        if (i+1)%250==0:
            trace.append(advantage.detach())
    if not anchor.isfinite().all():
        raise FloatingPointError("Nonfinite route residual")
    w = anchor.detach().numpy()
    stored = {name:w[k,:WIDTH] if k<2 else w[k] for k,name in enumerate(LABELS)}
    return stored, torch.stack(trace).numpy()


@torch.no_grad()
def deployment_table(graph, rule, tape, missing, future, heads):
    state, _ = env.initial(graph,rule,tape,missing)
    raw = stage_bank(graph,rule,tape,missing,future)
    truth = torch.where(tape.candidates[:,1]==missing,-1.,1.)  # side whose early contact yielded observed taste
    coeff = rule.expand(tape.lanes)
    for label, values in (("native",None),*heads.items(),("always_known",None)):
        chosen, readings, paid = [], [], []
        for j,horizon in enumerate((0,LENGTH)):
            cue = state.hidden.new_full((tape.lanes,),2*j-1)
            if label=="always_known":
                post, _, _ = route(coeff,state,tape,cue,None)
                side = truth
                logit = side * float("inf")
                earned,_ = execute(coeff,post,tape,future,side,horizon)
            else:
                side, logit, earned = deploy(graph,rule,tape,missing,future,cue,horizon,values)
            column = (side > 0).long().numpy()
            factual = raw["events"][np.arange(tape.lanes),j,column]
            observed = earned.to(torch.int8).numpy()
            np.testing.assert_array_equal(factual[:,:horizon+1],observed[:,:horizon+1])
            if horizon==0:
                assert np.count_nonzero(observed[:,1:])==0
            chosen.append(side.to(torch.int8).numpy())
            readings.append(logit.numpy())
            paid.append(observed)
        raw[f"{label}/action"] = np.stack(chosen,axis=1)
        raw[f"{label}/motor_logit"] = np.stack(readings,axis=1)
        raw[f"{label}/direct_events"] = np.stack(paid,axis=1)
    return raw


def grouped(raw, worlds):
    common = {key:raw[key] for key in ("candidate","no_taste","events")}
    actual = {}
    for label in ("native",*LABELS,"always_known"):
        numbers = {"action":raw[f"{label}/action"],"motor_logit":raw[f"{label}/motor_logit"],**common}
        actual[label]=metrics.measure(numbers,worlds)
        direct = raw[f"{label}/direct_events"][...,1].sum(-1).reshape(worlds,4,2).mean(1)
        for j,key in enumerate(("short","long")):
            np.testing.assert_array_equal(actual[label][f"{key}/native/total"],direct[:,j])
    scores = {f"{label}/{key}":val for label,table in actual.items() for key,val in table.items()}
    for head in LABELS:
        for other in ("native","always_known"):
            for key,val in actual[head].items():
                scores[f"{head}_minus_{other}/{key}"] = val-actual[other][key]
    for kind in ("linear","gated"):
        for key,val in actual[f"{kind}_informative"].items():
            scores[f"{kind}_informative_minus_uninformative/{key}"] = val-actual[f"{kind}_uninformative"][key]
    return scores


def replay(graph, rule, seed, worlds, function):
    tape, missing = env.lesson.prepare(seed,(),0,worlds,2,omissions=True,heldout=True)
    future = env.opportunities(seed,(),0,worlds,LENGTH)
    blocks=[]
    for begin in range(0,tape.lanes,256):
        ids=torch.arange(begin,min(tape.lanes,begin+256))
        blocks.append(function(graph,rule,env.lesson.expand(tape,ids),missing[ids],future.subset(ids)))
    return {key:np.concatenate([b[key] for b in blocks]) for key in blocks[0]}


def run(cohort):
    plan, parent = settings()
    body_sources = {}
    archive = {}
    code={Path(p).name:env.digest(p) for p in (__file__,env.__file__,metrics.__file__,env.lesson.__file__,env.net.__file__)}
    plan_sha = env.digest(PLAN)
    train_dir, readout_dir, holdout_dir = (ROOT/folder/cohort for folder in ("bank","checkpoints","holdout"))
    for directory in (train_dir,readout_dir,holdout_dir):directory.mkdir(parents=True,exist_ok=True)
    ages = env.COHORTS[cohort]
    training, evaluation = plan["training_seeds"][cohort], plan["holdout_seeds"][cohort]
    worlds, held = plan["worlds_per_bank"],plan["holdout_worlds"]
    rows=[]
    for b in ages:
        checkpoint = env.checkpoint(cohort,b)
        key = checkpoint.relative_to(env.world.ROOT).as_posix()
        signature = env.digest(checkpoint)
        if parent[key]!=signature:raise ValueError("Parent body differs from the previous seal")
        graph,rule=env.load(cohort,b)
        body_sources[key]=signature
        protocol = repr((code,plan_sha,signature,b))
        first = perf_counter()
        file = train_dir/f"{worlds}-{b}.npz"
        if not file.exists():
            raw = replay(graph,rule,training,worlds,stage_bank)
            env.storage._write_raw(file,**raw,protocol=protocol,seed=training)
        head_file = readout_dir/f"birth-{b}.npz"
        if not head_file.exists():
            with np.load(file,allow_pickle=False) as bank:
                if str(bank["protocol"])!=protocol or int(bank["seed"])!=training:
                    raise ValueError("The recorded training choices use a different protocol")
                heads,trace = fit(bank,plan,b)
            env.storage._write_raw(head_file,**heads,trace=trace,body_sha256=signature,plan_sha256=plan_sha,
                                   protocol=protocol,updates=plan["updates"],training_bank_sha256=env.digest(file))
        loaded = {label:load_head(head_file,label,signature) for label in LABELS}
        exact = holdout_dir/f"{held}-{b}.npz"
        if not exact.exists():
            test = replay(graph,rule,evaluation,held,lambda *args:deployment_table(*args,loaded))
            env.storage._write_raw(exact,**test,protocol=protocol,seed=evaluation,head_sha256=env.digest(head_file))
        with np.load(exact,allow_pickle=False) as observed:
            if str(observed["protocol"])!=protocol or int(observed["seed"])!=evaluation:
                raise ValueError("The held-out route choices use a different protocol")
            if str(observed["head_sha256"])!=env.digest(head_file):raise ValueError("Evaluated head differs")
            rows.append(grouped(observed,held))
        for path in (file,head_file,exact):archive[path.relative_to(env.world.ROOT).as_posix()]=env.digest(path)
        print(cohort,b,round(perf_counter()-first,2),flush=True)
    scores={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    estimate=env.intervals.PairedCrossed(len(ages),held,evaluation+331).summary(scores)
    output=dict(cohort=cohort,births=list(ages),worlds=held,training_worlds=worlds,
                training_seed=training,holdout_seed=evaluation,plan_sha256=plan_sha,source_code=code,
                body_sources=body_sources,files=archive,metrics=estimate)
    env.storage._write_json(env.EVIDENCE/f"horizon-anchor-{cohort}.json",output)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",choices=env.COHORTS,required=True)
    args=cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.cohort)
