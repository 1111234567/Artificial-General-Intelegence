"""Does a bootstrapped graph choose an informative contact and use its own write?"""
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from . import self_write_core as core, self_write_core_bootstrap as boot, world
from . import ring_logic_study as io


EVIDENCE=io.REPO/"docs/evidence/self-writing-2026-11-19/inquiry"
REG=io.read(EVIDENCE/"protocol.json")
RUNS=io.REPO/"docs/evidence/self-writing-2026-11-19/core"
ARMS=("own","no_write","swapped","blind","stale","corrupt")


class Body(world.PulseWorld):
    """Only actually contacted tastes are available at the ordinary next-tick port."""
    def __init__(self,tape,values):
        super().__init__(tape)
        self.values=values

    def _move(self,pulses):
        super()._move(pulses)
        trial=self.trial_index
        hit=np.flatnonzero(self._contact_tick[:,trial]==self.tick_index)
        self._reward[hit,trial]=self.values[hit,self._contact_side[hit,trial]]


@dataclass
class Trial:
    left: np.ndarray
    right: np.ndarray
    key: np.ndarray
    tastes: np.ndarray
    tape: world.Tape


def make_trial(roots,vectors,axes,signs,values,nonce):
    # axes and signs are specified by [world, side]. Only physical concentrations go to core.
    lanes=np.arange(len(roots))
    odors=vectors[lanes[:,None],axes]*signs[:,:,None]
    tape_roots=tuple(int(np.random.SeedSequence([root,nonce,911]).generate_state(1)[0]) for root in roots)
    tape=world.make_tape(tape_roots,trials=1,distance_jitter=.18,wind_scale=.065)
    return Trial(odors[:,0].copy(),odors[:,1].copy(),np.asarray(axes),np.asarray(values,dtype=np.int8),tape)


def contact(g,h,tr,forced_side=None):
    # Native motor decision has its normal integration window. Its sign is held as
    # a motor pulse command while the same physical body moves until first contact.
    command_state=core.advance(g,h,core.sense(tr.left,tr.right),core.MOTOR_TICKS)
    command=np.where(core.motor(command_state)>0,1,-1)
    if forced_side is not None:command=forced_side
    body=Body(tr.tape,tr.tastes)
    pulses=np.column_stack((command<0,command>0))
    for _ in range(world.TICKS):body.step(pulses)
    r=body.result()
    side=np.where(r.contact_side[:,0]<0,0,2*r.contact_side[:,0]-1)
    taste=r.observed_reward[:,0].sum(axis=-1,dtype=np.int16)
    assert np.array_equal(taste,r.reward[:,0])
    pre,post=core.contact_states(g,command_state,tr.left,tr.right,side,taste)
    return pre,post,side,taste,r.contact_tick[:,0]


def issue(g,h,tr,key):
    pre,post,side,taste,_=contact(g,h,tr)
    if (side==0).any():raise RuntimeError("Uncredited exposure: adapt masking before using this world")
    cmd=core.proposal(g,pre,post,key,view="own")
    ng,receipt=core.apply(g,cmd,arm="active")
    return ng,post,side,taste,receipt


def probe(g,h,trial,sign):
    _,_,side,_,_=contact(g,h,trial)
    # Best odor in this private rule is the signed endpoint.
    return side==sign


def evaluate(birth):
    start,count=REG["worlds"];roots=np.arange(start,start+count)
    rngs=[np.random.default_rng([int(r),566127]) for r in roots]
    # Both available sentinel valences in each physical root; no mean-utility edge.
    sym=np.tile(np.array([1,-1]),count)
    ids=np.repeat(roots,2);n=len(ids);idx=np.arange(n)
    vectors=np.repeat(np.array([1.35*np.linalg.qr(rng.normal(size=(3,3)))[0].T for rng in rngs],
                              dtype=np.float32),2,axis=0)
    def draw_sides():return np.repeat([r.choice([-1,1]) for r in rngs],2)
    def two_sides(a):return np.column_stack((a,-a))
    def att(k):return int(np.random.SeedSequence([birth,k,655693]).generate_state(1)[0])
    raw=boot.load_checkpoint(RUNS/f"bootstrap-own-{birth}.npz")
    g0=core.repeated_graph(raw,n);g=g0.clone();h=core.rest(g)
    for block in range(4):
        order=np.repeat([r.permutation(3) for r in rngs],2,axis=0)
        for j in range(3):
            axis=order[:,j]
            signs=two_sides(draw_sides())
            fair=draw_sides()
            values=np.where(axis[:,None]==2,two_sides(fair),signs)
            trial=make_trial(ids,vectors,np.repeat(axis[:,None],2,1),signs,values,10+3*block+j)
            g,h,_,_,_=issue(g,h,trial,att(10+3*block+j))
    same=np.ones((n,2),dtype=int)
    bad=make_trial(ids,vectors,np.zeros_like(same),same,-same,50)
    gp,hp,*_=issue(g,h,bad,att(50))
    good=replace(bad,tastes=same)
    neutral,hn,*_=issue(g,h,good,att(50))
    layout=draw_sides() # +1 iff B is on the right.
    sentinel_right=layout==1
    axes=np.column_stack((np.where(sentinel_right,2,1),np.where(sentinel_right,1,2)))
    distractor_sign=draw_sides()
    a_sign=np.where(axes==1,sym[:,None],distractor_sign[:,None])
    fair=draw_sides()
    q=make_trial(ids,vectors,axes,a_sign,np.where(axes==1,sym[:,None],fair[:,None]),60)
    qs=replace(q,tastes=np.where(axes==1,-sym[:,None],fair[:,None]))
    future_sign=np.repeat([r.choice([-1,1]) for r in rngs],2)
    future=make_trial(ids,vectors,np.ones_like(same),two_sides(future_sign),np.zeros_like(same),70)
    learn_ref=probe(g,h,future,-future_sign)
    _,_,ordinary,_,_=contact(gp,hp,q)
    _,_,under_positive,_,_=contact(neutral,hn,q)
    out=dict(info_chosen=(ordinary==layout),positive_info=(under_positive==layout),layout=layout,
             sentinel_sign=sym,pre_B=learn_ref,uncredited=(ordinary==0))
    for route,force in (("choice",None),("sentinel",layout),("decoy",-layout)):
        leaves=[contact(gp,hp,t,force) for t in (q,qs)]
        assert np.array_equal(leaves[0][2],leaves[1][2])
        commands=[core.proposal(gp,l[0],l[1],att(60),view="own") for l in leaves]
        for k,l in enumerate(leaves):
            pre,post,side,taste,tick=l
            if (side==0).any():raise RuntimeError("Unexpected missing probe contact")
            used={"own":commands[k]}
            if route=="sentinel":
                used.update(swapped=commands[1-k],no_write=commands[k],
                            blind=core.proposal(gp,pre,post,att(60),view="blind"),
                            stale=core.proposal(gp,pre,post,att(60),view="own",view_graph=g0),
                            corrupt=core.proposal(gp,pre,post,att(60),view="corrupt"))
            out[f"{route}/{k}/taste"]=taste;out[f"{route}/{k}/side"]=side
            for arm,cmd in used.items():
                updated,receipt=core.apply(gp,cmd,arm="none" if arm=="no_write" else "active")
                target=(-future_sign if k==0 else future_sign)
                out[f"{route}/{k}/{arm}/carry"]=probe(updated,post,future,target)
                out[f"{route}/{k}/{arm}/rest"]=probe(updated,core.rest(updated),future,target)
                if route=="sentinel":
                    base,rr=core.apply(gp,commands[k])
                    tag=f"effect/{k}/{arm}/"
                    out[tag+"delta_w_rms"]=np.sqrt(np.mean((base.w-updated.w)**2,axis=(1,2)))
                    out[tag+"delta_b_rms"]=np.sqrt(np.mean((base.b-updated.b)**2,axis=1))
                    out[tag+"different_support"]=(base.mask!=updated.mask).any(axis=(1,2))
                    for op,status in (("birth",receipt.birth_status),("retire",receipt.retire_status)):
                        out[tag+op+"_asked"]=getattr(cmd,op)>=0
                        out[tag+op+"_applied"]=status==1
                        out[tag+op+"_invalid"]=status==3
                        out[tag+op+"_guard"]=status==4
    return {k:np.asarray(v).reshape(count,2) for k,v in out.items()}


def report():
    rows=[evaluate(s) for s in REG["births"]]
    full={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    bank={k:x.mean(-1) for k,x in full.items()}
    b,m=bank["info_chosen"].shape
    rng=np.random.default_rng(53102)
    wb=rng.multinomial(b,np.ones(b)/b,20000)/b
    wm=rng.multinomial(m,np.ones(m)/m,20000)/m
    def est(v,alpha=.025):
        draws=((wb@v)*wm).sum(1)
        return dict(mean=float(v.mean()),ci=np.quantile(draws,[alpha,1-alpha]).tolist(),per_birth=v.mean(1).tolist())
    def tip(route,arm,mode="carry"):return (bank[f"{route}/0/{arm}/{mode}"]+bank[f"{route}/1/{arm}/{mode}"])/2
    axes={"information_preference":bank["info_chosen"]-.5,
          "surprise_minus_positive":bank["info_chosen"]-bank["positive_info"],
          "chosen_minus_decoy":tip("choice","own")-tip("decoy","own"),
          "forced_sentinel_minus_decoy":tip("sentinel","own")-tip("decoy","own"),
          "own_minus_swapped_write":tip("sentinel","own")-tip("sentinel","swapped"),
          "own_minus_omitted_write":tip("sentinel","own")-tip("sentinel","no_write")}
    values={f"{route}/{arm}/{mode}":est(tip(route,arm,mode)) for route in ("sentinel","choice","decoy")
            for arm in (ARMS if route=="sentinel" else ("own",)) for mode in ("carry","rest")}
    effect={k:float(v.mean()) for k,v in bank.items() if k.startswith("effect/")}
    archive=EVIDENCE/"raw.zip"
    buffer=BytesIO();np.savez_compressed(buffer,**full)
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:z.writestr("outcomes.npz",buffer.getvalue())
    io.write(EVIDENCE/"summary.json",dict(primary={k:est(v,.025/6) for k,v in axes.items()},actual=values,
           pre_shock_familiar_B=est(bank["pre_B"]),info_by_old_sentinel_valence={
             "previously_good":float(full["info_chosen"][:,:,0].mean()),"previously_bad":float(full["info_chosen"][:,:,1].mean())},
           view_effects=effect,archive_sha256=io.sha(archive),protocol_sha256=io.sha(EVIDENCE/"protocol.json"),
           source_sha256=io.sha(Path(__file__)),core_sha256=io.sha(Path(core.__file__))))


if __name__=="__main__":report()
