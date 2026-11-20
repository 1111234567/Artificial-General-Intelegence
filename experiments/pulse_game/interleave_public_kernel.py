"""Fixed-size, public-sensory references for the specificity/transfer tradeoff."""

from argparse import ArgumentParser
from math import factorial
import json
from pathlib import Path

import numpy as np

from . import interleave_capacity as cap, interleave_induction as induction, interleave_triangle_train as triangle
from .interleave_interference import EVIDENCE


DEGREES=(1,3,5,7)


def basis(x,degree):
    """Feature inner product equals (cos(angle))**degree for normalized directions."""
    u=x.astype(np.float64)
    u=u/np.linalg.norm(u,axis=-1,keepdims=True)
    terms=[]
    for a in range(degree+1):
        for b in range(degree-a+1):
            c=degree-a-b
            scale=np.sqrt(factorial(degree)/(factorial(a)*factorial(b)*factorial(c)))
            terms.append(scale*u[...,0]**a*u[...,1]**b*u[...,2]**c)
    return np.stack(terms,-1)


def select(mem,left,right,u,degree):
    score=((basis(right,degree)-basis(left,degree))*mem).sum(-1)
    return np.where(score>1e-10,1,np.where(score< -1e-10,-1,np.where(u<.5,1,-1)))


def learn(mem,left,right,side,earned,degree):
    physical=np.where(side[:,None]>0,right,left)
    return mem+earned[:,None]*basis(physical,degree)


def memory_size(d):
    return (d+1)*(d+2)//2


def capacity(t,d):
    left=t.left.numpy();right=t.right.numpy()
    factor=t.right_value.numpy();fixed=t.constant_value.numpy();u=t.motor_draw.numpy()
    mem=np.zeros((t.lanes,memory_size(d)));cells={}
    for k in range(t.max_k+1):
        for c in range(t.contacts):
            a=select(mem,left[:,k,c],right[:,k,c],u[:,k,c],d)
            r=a*factor[:,k,c]+fixed[:,k,c]
            mem=learn(mem,left[:,k,c],right[:,k,c],a,r,d)
        if k not in cap.CAPACITIES:continue
        good=np.stack([select(mem,left[:,j,-1],right[:,j,-1],u[:,j,-1],d)==factor[:,j,-1] for j in range(k+1)],1)
        cells[k]=dict(A=float(good[:,0].mean()),latest=float(good[:,-1].mean()),
                      both=float((good[:,0]&good[:,-1]).mean()),all=float(good.all(1).mean()),
                      mean_correct_number=float(good.sum(1).mean()))
    return cells


def transfer(w,d):
    x=w["x"].numpy();o=w["orient"].numpy();u=w["u"].numpy();law=w["law"].numpy();end=w["max_k"]
    factor=w["x_sign"].numpy()>0
    mem=np.zeros((len(x),memory_size(d)));cells={}
    for k in range(end+1):
        if k in induction.CUTS:
            right=x[:,end]*o[:,end,None];goal=law[:,end]*o[:,end]
            a=select(mem,-right,right,u[:,end,0],d)
            alt=learn(mem,-right,right,a,a*goal,d)
            xr=x[:,end]*o[:,end+2,None];yr=x[:,end+1]*o[:,end+1,None]
            xgood=select(alt,-xr,xr,u[:,end+2,0],d)==law[:,end]*o[:,end+2]
            ygood=select(alt,-yr,yr,u[:,end+1,0],d)==law[:,end+1]*o[:,end+1]
            cells[k]=dict(novel=float((a==goal)[factor].mean()),
                          exception=float(xgood[~factor].mean()),second_after_exception=float(ygood[~factor].mean()),
                          both_after_exception=float((xgood&ygood)[~factor].mean()))
        if k==end:break
        right=x[:,k]*o[:,k,None]
        side=select(mem,-right,right,u[:,k,0],d)
        mem=learn(mem,-right,right,side,side*law[:,k]*o[:,k],d)
    return cells


def triangle_case(seed,worlds,d):
    t=triangle.make_tape(seed,731,worlds,-.5)
    x=t["right"].numpy();law=t["law"].numpy();order=t["order"].numpy()
    ori=t["study_flip"].numpy();u=t["study_u"].numpy();rows=np.arange(len(x))
    mem=np.zeros((len(x),memory_size(d)))
    for i in range(3):
        obj=order[:,i];right=x[rows,obj]*ori[:,i,None]
        side=select(mem,-right,right,u[:,i,0],d)
        earned=side*law[rows,obj]*ori[:,i]
        mem=learn(mem,-right,right,side,earned,d)
    margin=(basis(x,d)*mem[:,None]).sum(-1)*law
    ok=(margin.reshape(worlds,8,3)>.05).all(2)
    return dict(strict_pair=float(ok[:,[0,7]].all(1).mean()),all_eight=float(ok.all(1).mean()),
                mixed=float(ok[:,1:7].mean()),minimum_signed_margin=float(margin.min()))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=EVIDENCE/"public-kernels.json")
    args=p.parse_args()
    tape=cap.make_tape("odor",871831,192,16)
    out=dict(model="Choose the higher current value, break ties with the public motor uniform. Learn only from "
                   "the tasted resource and the selected sensory vector. The finite symmetric tensor has "
                   "(d+1)(d+2)/2 real components. It is a hand-designed reference, not a trained graph.",degrees={})
    for d in DEGREES:
        cells={}
        for geom in ("linear","cubic"):
            w=induction.world(311285,731,1024,geom,0.,64)
            cells[geom]=transfer(w,d)
        out["degrees"][d]=dict(scalar_slots=memory_size(d),capacity=capacity(tape,d),
                               transfer=cells,triangle=triangle_case(184701,256,d))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":main()
