"""Conditional WAIT/FINISH readout on the same 24 states as the BEGIN study."""
import argparse
from collections import Counter
import gzip
import io
from pathlib import Path
from unittest.mock import patch

from experiments.action_return import readout as begin
import torch
from torch import nn
from torch.nn import functional as F


ROOT = begin.ROOT / "data/action-return/finish-readout-v1"
PLAN = ROOT / "protocol.json"


def declare():
    parent = begin.check()
    if PLAN.exists():
        raise FileExistsError(PLAN)
    bindings = {}
    for seed in parent["seeds"]:
        folder = begin.OUT / f"seed_{seed:03d}"
        fit_record = begin.old.read(folder / "fit.json")
        if fit_record["protocol_sha256"] != begin.old.sha(begin.PLAN):
            raise ValueError("foreign BEGIN fitting inputs")
        for file, key in (("train_dataset.pt.gz","train_dataset_sha256"),("fits.pt","fits_sha256")):
            if begin.old.sha(folder/file) != fit_record[key]:
                raise ValueError("declared fitting artifact changed")
            bindings[str(folder/file)] = fit_record[key]
        bindings[str(folder/"fit.json")] = begin.old.sha(folder/"fit.json")
    value = dict(format="conditional-finish-readout-v1", parent_protocol_sha256=begin.old.sha(begin.PLAN),
        source_sha256=begin.old.sha(__file__), bindings=bindings, seeds=parent["seeds"],
        design="Supplement to the primary study, declared before fitting or evaluating this conditional controller. Partial BEGIN-only results were available. Reuse frozen non_input24 BEGIN head and its exact same-state gate and version. Within the non-BEGIN group learn WAIT plus eight FINISH choices. No teacher at inference.",
        fitting="Same assisted training features, original labels and actor/time slice as BEGIN; fit only teacher non-BEGIN positions. Per seed/job, WAIT-reference 9-way affine softmax, native WAIT weight 0.1 and FINISH weight 1, train-only standardization, L2=.001 on slopes, same LBFGS settings. No threshold search.",
        evaluation="Same 4 families, short and long_both, 2048 free ticks from each actual endpoint. Primary actors4..7, count matured/began/completed/strict, legal and executed finish attempts, all idle actions per actual idle opportunity. Paired control is the same trained non_input24 BEGIN controller with native WAIT/FINISH.",
        limits="Conditional score replacement is external to the frozen recurrent field. Its new actions change later receipts and hidden states. Actors4..7 are held out only from readout fitting.")
    ROOT.mkdir(parents=True,exist_ok=True)
    begin.old.write(PLAN,value)


def checked():
    plan = begin.old.read(PLAN)
    parent = begin.check()
    if plan["source_sha256"] != begin.old.sha(__file__) or plan["parent_protocol_sha256"] != begin.old.sha(begin.PLAN):
        raise ValueError("supplement source or BEGIN protocol changed")
    for path, expected in plan["bindings"].items():
        if begin.old.sha(path) != expected:
            raise ValueError(f"changed training artifact {path}")
    return plan,parent


def train_finish(features, actions, options):
    keep = (actions==0) | (actions>=3)
    x = features[keep].double()
    y = torch.where(actions[keep]==0,0,actions[keep]-2).long()
    counts = torch.bincount(y,minlength=9)
    if len(counts)!=9 or counts.min()==0:
        raise ValueError("unobserved conditional action class")
    mean = x.mean(0)
    scale = (x-mean).square().mean(0).sqrt()
    active = scale>0
    z = (x[:,active]-mean[active])/scale[active]
    weights = torch.ones_like(y,dtype=torch.float64)
    weights[y==0] = .1
    coef = nn.Parameter(z.new_zeros(8,z.shape[1]))
    bias = nn.Parameter(z.new_zeros(8))
    optimizer = torch.optim.LBFGS([coef,bias],**{k:v for k,v in options.items() if k!="name"})
    def objective():
        optimizer.zero_grad()
        raw = torch.cat((z.new_zeros(len(z),1),z@coef.T+bias),dim=-1)
        loss = (weights*F.cross_entropy(raw,y,reduction="none")).sum()/weights.sum() + .001*coef.square().sum()
        loss.backward()
        return loss
    initial = float(objective().detach())
    optimizer.step(objective)
    final = float(objective().detach())
    if not 0 <= final <= initial:
        raise ArithmeticError("conditional objective failed to descend")
    restored = x.new_zeros(8,x.shape[1])
    restored[:,active] = coef.detach()
    fit = dict(mean=mean,std=torch.where(active,scale,torch.ones_like(scale)),coef=restored,bias=bias.detach().clone())
    diagnostics = dict(class_counts=counts.tolist(),initial=initial,final=final,
        iterations=optimizer.state[coef]["n_iter"],max_abs_gradient=max(float(q.grad.abs().max()) for q in (coef,bias)))
    return fit,diagnostics


def replace_other(native_begin, finish_offsets):
    """Preserve the inherited discrete BEGIN choice even if two groups tie."""
    choices = torch.cat((torch.zeros_like(finish_offsets[...,:1]),finish_offsets),-1)
    original = native_begin.argmax(-1)
    best_other = torch.cat((native_begin[...,:1],native_begin[...,3:]),-1).amax(-1,keepdim=True)
    centered = best_other + (choices - choices.amax(-1,keepdim=True))
    result = torch.cat((centered[...,:1],native_begin[...,1:3],centered[...,1:]),-1)
    winner = choices.argmax(-1)
    winner = torch.where(winner==0,0,winner+2)
    intended = torch.where((original==1)|(original==2),original,winner)
    # Only an exact tie (including a tie from rounding after centering) can
    # violate the declared group-first tie rule. Resolve it by one FP64 ulp.
    mismatch = result.argmax(-1) != intended
    score = result.gather(-1,intended.unsqueeze(-1))
    maximum = result.amax(-1,keepdim=True)
    chosen = torch.where(mismatch.unsqueeze(-1),torch.nextafter(maximum,torch.full_like(maximum,float("inf"))),score)
    return result.scatter(-1,intended.unsqueeze(-1),chosen)


class FinishController(nn.Module):
    def __init__(self, begin_head, fits):
        super().__init__()
        self.begin_head = begin_head
        for field in ("width","input_size","n_jobs","n_actions"):
            setattr(self,field,getattr(begin_head,field))
        for job,item in enumerate(fits):
            for key,value in item.items():
                self.register_buffer(f"j{job}_{key}",value.clone())

    def step(self, obs, h, dt, max_step=None):
        original, h = self.begin_head.step(obs,h,dt,max_step)
        x = h.index_select(-1,self.begin_head.indices)
        raw=[]
        for j in range(self.n_jobs):
            z=(x-getattr(self,f"j{j}_mean"))/getattr(self,f"j{j}_std")
            raw.append(z@getattr(self,f"j{j}_coef").T+getattr(self,f"j{j}_bias"))
        return replace_other(original,torch.stack(raw,-2)),h

    def config(self):
        return dict(kind="external_conditional_finish_v1",begin=self.begin_head.config(),features=24)


def evaluate_conditional(model,trainer,family,condition,requests):
    """Count responses to the teacher's FINISH positions on this policy's trajectory."""
    counts=Counter()
    original=begin.ContextWorld.act
    def observed(world,actions):
        wanted=world.oracle("midpoint")[4:]
        choice=actions[4:]
        valid=wanted>=3
        counts["finish_labels"] += int(valid.sum())
        counts["chose_begin"] += int((valid & ((choice==1)|(choice==2))).sum())
        counts["chose_wait"] += int((valid & (choice==0)).sum())
        counts["chose_finish"] += int((valid & (choice>=3)).sum())
        counts["correct_finish"] += int((valid & (choice==wanted)).sum())
        return original(world,actions)
    with patch.object(begin.ContextWorld,"act",observed):
        value=begin.evaluate_shadow(model,trainer,family=family,condition=condition,requests=requests)
    expected=value["shadow"]["heldout"]["counts"].get("teacher_finish_labels",0)
    if expected!=counts["finish_labels"] or expected!=sum(counts[k] for k in ("chose_wait","chose_begin","chose_finish")):
        raise AssertionError("inconsistent FINISH label counts")
    value["heldout_finish_responses"]=dict(counts)
    return value


def run(seed):
    plan,parent = checked()
    torch.set_num_threads(1)
    folder=begin.OUT/f"seed_{seed:03d}"
    tables = torch.load(io.BytesIO(gzip.decompress((folder/"train_dataset.pt.gz").read_bytes())),weights_only=False,map_location="cpu")
    fits = torch.load(folder/"fits.pt",map_location="cpu",weights_only=False)["fits"]["non_input24"]
    src=parent["sources"][str(seed)]["endpoint"]
    trainer=begin.native.NativeTrainer.from_checkpoint(src["path"],device="cpu")
    before=begin.state_digest(trainer.semantic_state())
    if before!=src["semantic_hash"]:
        raise ValueError("wrong complete endpoint")
    out=ROOT/f"seed_{seed:03d}"
    out.mkdir(exist_ok=False)
    indices=tables["non_input"]
    requests=[]; learned=[]; diagnostics=[]
    for j in range(trainer.model.n_jobs):
        group=tables["groups"][j]
        fit,info=train_finish(group["h"][:,indices],group["label"],parent["train"]["optimizer"])
        learned.append(fit);diagnostics.append(info);requests.extend(group["request"])
    model=FinishController(begin.BeginController(trainer.model,"non_input24",fits,indices),learned)
    torch.save(model.state_dict(),out/"model.pt")
    records=[]
    for condition in parent["conditions"]:
        for family in parent["families"]:
            value=evaluate_conditional(model,trainer,family=family,condition=condition,requests=requests)
            value.update(condition=condition,family=family)
            records.append(value)
            print(seed,condition,family,flush=True)
    if before!=begin.state_digest(trainer.semantic_state()):
        raise AssertionError("source endpoint changed")
    result=dict(seed=seed,protocol_sha256=begin.old.sha(PLAN),model_sha256=begin.old.sha(out/"model.pt"),
        diagnostics=diagnostics,source_semantic_hash=before,records=records)
    begin.old.write(out/"result.json",result)


def summarize():
    plan,parent=checked()
    rows=[]; inputs={}
    for seed in plan["seeds"]:
        primary=begin.OUT/f"seed_{seed:03d}"/"result.json"
        added=ROOT/f"seed_{seed:03d}"/"result.json"
        a,b=begin.old.read(primary),begin.old.read(added)
        if a["protocol_sha256"]!=plan["parent_protocol_sha256"] or b["protocol_sha256"]!=begin.old.sha(PLAN):
            raise ValueError("wrong completed readout branch")
        if b["model_sha256"]!=begin.old.sha(added.parent/"model.pt"):
            raise ValueError("conditional action model bytes changed")
        inputs.update({str(primary):begin.old.sha(primary),str(added):begin.old.sha(added)})
        for condition in parent["conditions"]:
            totals={name:Counter() for name in ("begin_only","begin_and_finish")}
            responses=Counter()
            for family in parent["families"]:
                control=next(r for r in a["records"] if (r["arm"],r["condition"],r["family"])==("non_input24",condition,family))
                test=next(r for r in b["records"] if (r["condition"],r["family"])==(condition,family))
                for field in ("start_world_state_hash","start_hidden_hash","end_scheduler_hash","end_event_hash"):
                    if control["record"][field]!=test["record"][field]:
                        raise AssertionError(f"unmatched frozen fork {field}")
                left,right=(r["shadow"]["heldout"] for r in (control,test))
                if left["primary_request_set_sha256"]!=right["primary_request_set_sha256"]:
                    raise AssertionError("matured primary request identities differ")
                responses.update(test["heldout_finish_responses"])
                for name,group in zip(totals,(left,right)):
                    totals[name].update(group["counts"])
            for name,counts in totals.items():
                rows.append(dict(seed=seed,condition=condition,policy=name,counts=dict(counts),
                                 finish_responses=dict(responses) if name=="begin_and_finish" else None))
    result=dict(completed=True,additional_forks=24,reused_forks=24,protocol_sha256=begin.old.sha(PLAN),inputs=inputs,heldout=rows)
    begin.old.write(ROOT/"summary.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","run","summarize"))
    parser.add_argument("--seed",type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=="declare": declare()
    elif args.command=="run": run(args.seed)
    else: summarize()
