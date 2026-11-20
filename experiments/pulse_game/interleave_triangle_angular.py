"""Read the angular response of the trained recurrent motor circuit after opposite taste histories."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_triangle_train as triangle, local_trace_graph as net
from .interleave_interference import DATA, EVIDENCE


ANGLE_COUNT = 72
OFFSETS = (0, 5, 10, 20, 30)
WORLDS = 128
SEED = 6139481


def angular_bias(curve):
    """Antisymmetrize the sensory orientation, then contrast all-positive and all-negative histories."""
    odd = (curve - np.roll(curve,ANGLE_COUNT//2,axis=-1))*.5
    return (odd[:,1] - odd[:,0])*.5


def projection(curve, harmonic):
    angle = 2*np.pi*harmonic*np.arange(ANGLE_COUNT)/ANGLE_COUNT
    a = 2*(curve*np.cos(angle)).mean(-1)
    b = 2*(curve*np.sin(angle)).mean(-1)
    reconstructed = a[...,None]*np.cos(angle) + b[...,None]*np.sin(angle)
    return a,b,reconstructed


@torch.no_grad()
def response(g, r, tape):
    n = len(tape["law"])//8
    ids = torch.arange(8*n).reshape(n,8)[:,[0,7]].flatten()
    core = {k:v[ids] for k,v in tape.items()}
    lanes = 2*n
    state = net.start(g, lanes)
    rule = r.expand(lanes)
    for t in range(3):
        _, result, _, _ = triangle._contact(rule,state,core,torch.arange(lanes),t,"active")
        state = net.wait(rule,result.state,4)
    x = core["right"][:,0]
    y = (core["right"][:,1] + .5*x)/(np.sqrt(3)/2)
    angles = torch.arange(ANGLE_COUNT,dtype=x.dtype)*(2*np.pi/ANGLE_COUNT)
    vectors = x[:,None,:]*angles.cos()[None,:,None] + y[:,None,:]*angles.sin()[None,:,None]
    step = ANGLE_COUNT//3
    for i in range(3):
        vectors[:,i*step] = core["right"][:,i]
        vectors[:,(i*step+ANGLE_COUNT//2)%ANGLE_COUNT] = -core["right"][:,i]
    curves, last_drive = [], []
    for chunk in vectors.split(6,dim=1):
        k = chunk.shape[1]
        s = state.subset(torch.arange(lanes).repeat_interleave(k))
        v = chunk.reshape(lanes*k,3)
        drive = s.graph.bias + net.native.sensory(s.graph,-v,v)
        weights = s.graph.weights*s.graph.present
        h=s.hidden
        signals=[]
        for tick in range(net.REACTION_TICKS):
            before = torch.bmm(weights,h.unsqueeze(2)).squeeze(2) + drive
            h = (1-net.native.LEAK)*h + net.native.LEAK*before.tanh()
            signals.append((2.5*(h[:,11]-h[:,10])).reshape(n,2,k))
        curves.append(torch.stack(signals,1))
        last_drive.append(before[:,[10,11]].reshape(n,2,k,2))
    curves=torch.cat(curves,dim=-1).numpy()  # world, tick, history, angle.
    z=torch.cat(last_drive,dim=2).numpy().transpose(0,1,3,2)
    ordinary=triangle.trials(g,r,tape,"active")
    logits=ordinary["logit"].reshape(n,8,3,2)[:,[0,7]]
    np.testing.assert_array_equal(curves[:,-1,:,::step],logits[...,0])
    np.testing.assert_array_equal(curves[:,-1,:,(np.arange(3)*step+ANGLE_COUNT//2)%ANGLE_COUNT].transpose(1,2,0),
                                  logits[...,1])
    # Location on the curve is physical direction relative to the first learned odor.
    angular=np.stack([angular_bias(curves[:,i]) for i in range(net.REACTION_TICKS)],axis=1)
    final=angular[:,-1]
    three=(final+np.roll(final,step,axis=-1)+np.roll(final,2*step,axis=-1))/3
    _,_,three_fit=projection(final,3)
    _,_,one_fit=projection(final,1)
    one_drive=z.mean(-1,keepdims=True) + projection(z,1)[2]
    nonlinear=np.tanh(one_drive)
    from_fundamental=angular_bias(2.5*net.native.LEAK*(nonlinear[:,:,1]-nonlinear[:,:,0]))
    leak=(1-net.native.LEAK)*angular[:,-2]
    residual=final-leak-from_fundamental
    u=core["probe_u"].numpy().reshape(n,2,3)
    law=core["law"].numpy().reshape(n,2,3)
    signs=[]
    sites=np.arange(3)*step
    for offset in OFFSETS:
        choices=[]
        for rot in (-1,1):
            for orientation in (-1,1):
                turned=(sites+rot*(offset//5)+(orientation<0)*(ANGLE_COUNT//2))%ANGLE_COUNT
                logits=curves[:,-1,:,turned].transpose(1,2,0)
                physical=np.where((u if orientation>0 else 1-u)<torch.from_numpy(logits).sigmoid().numpy(),1,-1)
                choices.append(physical*orientation==law)
        signs.append(np.stack(choices,axis=1))
    good=np.stack(signs,axis=1)   # world, offset, orientation/sign combination, history, item.
    actual=ordinary["selected"].reshape(n,8,3,2)[:,[0,7]]==law[...,None]
    np.testing.assert_array_equal(good[:,0].all(-1).mean((1,2)),actual.all(2).mean((1,2)))
    return dict(tick_curve=angular,actual=good,energy_total=(final**2).mean(-1),
                energy_fundamental=(one_fit**2).mean(-1),energy_three=(three_fit**2).mean(-1),
                energy_threefold=(three**2).mean(-1),threefold_at_items=three[:,0],
                third_from_final_first_order=projection(from_fundamental,3)[0],
                third_from_leak=projection(leak,3)[0],third_from_other_drive=projection(residual,3)[0],
                aligned_third=projection(final,3)[0])


def intervals(matrix):
    # The angular protocol uses distinct geometries per birth. Bootstrap worlds
    # within birth and keep all rotations/histories for that world together.
    rng=np.random.default_rng(419713)
    count,worlds=matrix.shape[:2]
    root=rng.integers(count,size=(4000,count))
    samples=[]
    for row in matrix:
        w=rng.multinomial(worlds,np.full(worlds,1/worlds),4000)/worlds
        samples.append(w@row)
    boot=np.stack(samples)[root,np.arange(4000)[:,None]].mean(1)
    return dict(mean=matrix.mean((0,1)).tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),
                by_birth=matrix.mean(1).tolist())


def run(cohort):
    torch.set_num_threads(1)
    independent=cohort=="independent"
    parents=DATA/("triangle-confirmation" if independent else "triangle-objective")
    long=DATA/("triangle-long-"+cohort)
    births=list(range(811,817)) if independent else list(range(731,737))
    dest=DATA/"triangle-angular"/cohort
    dest.mkdir(parents=True,exist_ok=True)
    hashes={};results={};stages={}
    for step,source in ((0,parents),(8000,parents),(32000,long),(64000,long)):
        roots=[]
        for b in births:
            f=dest/f"{b}-{step}.npz"
            if not f.exists():
                g,r=triangle.old_fit.load_checkpoint(source,"odor","direct",b,step)
                raw=response(g,r,triangle.make_tape(SEED,b,WORLDS,-.5))
                np.savez_compressed(f,**raw)
            with np.load(f) as z:
                d={k:z[k] for k in z.files}
            hashes[f.name]=sha256(f.read_bytes()).hexdigest()
            d["rotated_actual_all_three"]=d["actual"].all(-1).mean((2,3))
            d["tick_third"]=projection(d["tick_curve"],3)[0]
            d["final_curve"]=d["tick_curve"][:,-1]
            roots.append(d)
        m={k:np.stack([d[k] for d in roots]) for k in roots[0] if k not in ("tick_curve","actual")}
        stages[step]=m
        results[step]={k:intervals(v) for k,v in m.items()}
        results[step]["third_fraction_of_threefold_energy"]=float(m["energy_three"].mean()/m["energy_threefold"].mean())
        results[step]["third_fraction_of_all_energy"]=float(m["energy_three"].mean()/m["energy_total"].mean())
        np.testing.assert_allclose(m["third_from_leak"]+m["third_from_final_first_order"]+m["third_from_other_drive"],
                                   m["aligned_third"],atol=3e-6)
        print(cohort,step,flush=True)
    contrasts={k:intervals(stages[64000][k]-stages[0][k]) for k in
               ("aligned_third","threefold_at_items","rotated_actual_all_three","tick_third")}
    summary=dict(births=births,seed=SEED,worlds_per_birth=WORLDS,degree_spacing=5,probe_offsets=OFFSETS,
                 source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),raw_sha256=hashes,
                 definition="Each birth has independent rotated triples. After the actual +++ and --- support "
                            "histories, sample motor logits in their common input plane at 72 angles. Contrast the "
                            "odd response under opposite histories. Average across 0,120,240 degrees to isolate "
                            "multiples of three; first harmonic cancels exactly. Third and higher terms refer to "
                            "discrete angular harmonics. The last-tick decomposition projects the motor preactivation "
                            "onto DC plus first harmonic before applying tanh; it is a signal analysis, not a lesion. "
                            "At nonzero probe offsets the diagnostic asks whether the physical action assigned to "
                            "the nearest trained odor is retained. Left-right actions use the original random draws.",
                 results=results,paired_64000_minus_parent=contrasts)
    (EVIDENCE/f"triangle-angular-{cohort}.json").write_text(json.dumps(summary,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=("original","independent"),required=True)
    run(parser.parse_args().cohort)
