"""Resume an organism in another process with only its own live state and fresh sensory inputs."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from . import native_rehearsal_eval as evaluation
from . import native_rehearsal_fit as fit
from . import native_rehearsal_graph as live


@torch.no_grad()
def continuation(request,output):
    torch.set_num_threads(1)
    saved=torch.load(request,map_location="cpu",weights_only=True)
    model=live.NativeGraph(saved["birth"],saved["layout"])
    model.load_state_dict(saved["body"])
    state=live.State(**saved["life"])
    zero=state.h.new_zeros((len(state.h),10))
    for _ in range(saved["wait"]):
        state,_=model.tick(state,zero)
    end=state
    answers=[]
    # No law or payoff enters this process. The motor actually controls the touched side.
    for menu,uniform in zip(saved["upcoming_menus"],saved["upcoming_draws"],strict=True):
        sensory=torch.cat((-menu,menu,menu.new_zeros((len(menu),4))),1)
        state=model.advance(state,sensory,3)
        choice=torch.where(uniform<model.logit(state).sigmoid(),1.,-1.)
        answers.append(choice)
        sensory[:,6]=(choice<0).float();sensory[:,7]=(choice>0).float()
        state=model.advance(state,sensory,4)
    torch.save(dict(side=torch.stack(answers,1),h=end.h,fast=end.fast,trace=end.trace),output)


@torch.no_grad()
def check():
    torch.set_num_threads(1)
    result={}
    with TemporaryDirectory(prefix="agi-neural-state-") as temporary:
        folder=Path(temporary)
        for kind,birth in (("copy",731),("guide_a",733),("single",735)):
            body,_=evaluation.load(kind,birth)
            e=live.scenes(719443,16,evaluation=True)
            _,begin,_,_=live.support(body,e)
            quiet,_,_=live.silence(body,begin)
            action,_,_,_=live.queries(body,quiet,e)
            rows=torch.arange(len(e.laws))
            seq=torch.stack((e.first_query,1-e.first_query),1)
            menus=[e.vectors[rows,seq[:,i]*2]*e.orientation[:,i+3,None] for i in (0,1)]
            request=folder/"input.pt";answer=folder/"output.pt"
            torch.save(dict(life=vars(begin),body=body.state_dict(),layout="single" if kind=="single" else "copy",
                            birth=birth,wait=12,upcoming_menus=menus,upcoming_draws=[e.draws[:,3],e.draws[:,4]]),request)
            subprocess.run([sys.executable,"-B","-m","experiments.pulse_game.native_rehearsal_stream",
                            "--request",str(request),"--output",str(answer)],check=True,cwd=fit.ROOT)
            restored=torch.load(answer,map_location="cpu",weights_only=True)
            for field in ("h","fast","trace"):
                torch.testing.assert_close(restored[field],getattr(quiet,field),rtol=0,atol=0)
            torch.testing.assert_close(restored["side"],action.gather(1,seq),rtol=0,atol=0)
            result[kind]=dict(birth=birth,lives=len(rows),real_actions=2*len(rows),weight_edges=int(body.mask.sum()),
                              dynamic_fields_per_lane=[list(v.shape[1:]) for v in vars(begin).values()],
                              heldout_answer_and_state_equal=True)
    fit.storage._write_json(fit.EVIDENCE/"stream-resume.json",result)
    print("fresh processes reproduced",sum(v["real_actions"] for v in result.values()),"actions and all quiet-window state")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--request",type=Path);p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.request:continuation(args.request,args.output)
    else:check()
