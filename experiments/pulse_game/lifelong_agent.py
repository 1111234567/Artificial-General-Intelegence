"""Run the frozen sensory graph and lifelong plasticity on CPU with NumPy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from .habitat_deploy import Forager


HERE=Path(__file__).resolve()
DEFAULT=HERE.with_name("reference-agent.npz")
if not DEFAULT.exists():
    DEFAULT=HERE.parents[2]/"data/pulse-game/morning-20261119/reference-agent.npz"


def save(path,front,route,motion,identity):
    arrays=dict(graph=front["matrix"],receptors=front["sensors"],odor_weights=front["ground"][0],odor_bias=front["ground"][1],
         transform=front["transform"],sensory_scalars=np.asarray(front["scalars"],np.float32),
         context_hidden=route["first"][0],context_hidden_bias=route["first"][1],
         context_read=route["second"][0],context_read_bias=route["second"][1],motion=np.asarray(motion,np.float32))
    meta=dict(format_version=1,identity=identity,capacity=int(front["capacity"]),age_cost=float(front.get("age_cost",0.)),
              route={k:(np.asarray(v).tolist() if isinstance(v,(list,tuple,np.ndarray)) else
                      v.item() if isinstance(v,np.generic) else v) for k,v in route.items() if k not in ("first","second")})
    np.savez_compressed(path,metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),**arrays)


def load(path=DEFAULT):
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    meta=json.loads(str(data["metadata"]))
    if meta["format_version"]!=1:raise ValueError("Unsupported phenotype format")
    f=dict(matrix=data["graph"],sensors=data["receptors"],ground=(data["odor_weights"],data["odor_bias"]),
           transform=data["transform"],scalars=data["sensory_scalars"].tolist(),capacity=meta["capacity"],age_cost=meta["age_cost"])
    r={**meta["route"],"first":(data["context_hidden"],data["context_hidden_bias"]),
       "second":(data["context_read"],data["context_read_bias"])}
    return f,r,data["motion"],meta


def new_lives(seeds,path=DEFAULT,*,keep_plastic=True,address="distributed"):
    f,r,m,_=load(path)
    if not keep_plastic:r={k:v for k,v in r.items() if k not in ("release","magnitude_floor")}
    # start=-1 disables the optional research snapshot. No experiment phase
    # identifier or hidden world label is needed by the standard controller.
    if address=="distributed":actor=Forager(f,r,seeds,start=-1,motor=m)
    elif address=="local":
        from .adaptive_local import LocalAdaptive
        actor=LocalAdaptive(f,r,seeds,motor=m,slots=6,radius=.09,inherit=True)
    elif address=="local-selective":
        from . import content_routing
        addon=Path(path).with_name("readout-options.json")
        settings=json.loads(addon.read_text(encoding="utf-8-sig"))["params"]
        actor=content_routing.Agent(f,r,seeds,motion=m,config=content_routing.Settings(**settings))
        actor.readout_sha256=hashlib.sha256(addon.read_bytes()).hexdigest()
    else:raise ValueError(address)
    actor.address_mode=address
    actor.phenotype_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    actor.keep_plastic=keep_plastic
    return actor


READER_ARRAYS=("prior","pred_i","pred_t","support_i","support_t","responsibility_sum","actual_updates")
STATE_ARRAYS=("h","g","cov","mem","signed","count","context","code","rlast","wlast","mv","motor",
              "noise","csum","ccount")


def save_life(path,actor):
    """Snapshot an active normal controller after an act call or a request boundary."""
    if actor.change!="true" or actor.start!=-1:raise ValueError("The portable format stores the deployed controller")
    destination=Path(path)
    arrays={name:getattr(actor,name) for name in STATE_ARRAYS if getattr(actor,name) is not None}
    if actor.address_mode in ("local","local-selective"):
        arrays.update({"places_"+k:getattr(actor.places,k) for k in ("prototypes","counts","current","overflows")})
    if actor.address_mode=="local-selective":
        arrays.update({"reader_"+k:getattr(actor,k) for k in READER_ARRAYS})
    meta=dict(schema=1,address=actor.address_mode,reader_sha256=getattr(actor,"readout_sha256",None),seeds=actor.seeds,request=actor.round,noise_tick=actor.it,n=actor.n,
              generator=[r.bit_generator.state for r in actor.rng],phenotype_sha256=actor.phenotype_sha256,
              keep_plastic=actor.keep_plastic)
    tmp=destination.with_suffix(".partial.npz")
    np.savez_compressed(tmp,meta=np.asarray(json.dumps(meta)),**arrays)
    os.replace(tmp,destination)


def restore_life(path,checkpoint=DEFAULT):
    """Continue with the next actual observation; the caller retains the environment state."""
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a["meta"]))
        if meta["schema"]!=1:raise ValueError("Unsupported lifetime snapshot")
        actor=new_lives(meta["seeds"],checkpoint,keep_plastic=meta["keep_plastic"],address=meta.get("address","distributed"))
        if actor.phenotype_sha256!=meta["phenotype_sha256"]:raise ValueError("Snapshot was written by another phenotype")
        actor.reset(meta["n"])
        for name in STATE_ARRAYS:
            state=getattr(actor,name)
            if state is None:continue
            v=a[name]
            if name!="noise" and state.shape!=v.shape:raise ValueError("Unexpected state shape: "+name)
            setattr(actor,name,v.copy())
        if actor.address_mode in ("local","local-selective"):
            for k in ("prototypes","counts","current","overflows"):setattr(actor.places,k,a["places_"+k].copy())
        if actor.address_mode=="local-selective":
            if meta["reader_sha256"]!=actor.readout_sha256:raise ValueError("Snapshot read/write rule changed")
            for k in READER_ARRAYS:setattr(actor,k,a["reader_"+k].copy())
        actor.round=meta["request"];actor.it=meta["noise_tick"]
        for rng,state in zip(actor.rng,meta["generator"],strict=True):rng.bit_generator.state=state
    return actor


def run_suite(name,roots,checkpoint,blocks,noise,keep_plastic,address="distributed"):
    from . import habitat_world,unified_world,longrun_world
    if name=="cross-game":
        from . import object_world,recall_world
        brain=new_lives(roots,checkpoint,keep_plastic=keep_plastic,address=address);brain.reset(len(roots))
        def forward(arena,hide):
            sim=habitat_world.Body(arena,silent_probes=hide);obs=sim.observe()
            while obs is not None:obs=sim.step(brain.act(obs))
            return sim.result()
        prime=habitat_world.make(roots,pairs=(12,12),visits=2,span=4,noise=.05,appearance="smooth",novel_seed=83649013)
        new,length=object_world.make(roots,problems=40,variable=True,noise=.04,nonce=83649014)
        new.tags[:]=prime.tags;new.appearance[:]=prime.tags[:,0,None,:]
        ret=recall_world.probe([v+9539 for v in roots],prime.tags,prime.relation,rounds=8)
        initial=habitat_world.metrics(prime,forward(prime,True))
        object_result=object_world.metrics(new,length,forward(new,False))
        final=forward(ret,True)
        return dict(scenario=name,address=address,worlds=len(roots),requests_per_life=brain.round+1,
               initial_both=float(initial["both"].mean()),second_choice_new_objects=float(np.mean(object_result["trial_2"])),
               return_to_old_rules=float(((final.contact_side==ret.truths)&(final.contact_side>=0))[ret.probe].mean()))
    if name=="transfer":
        n=len(roots)
        families=np.repeat(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]),n,axis=0)
        seeds=roots*4
        ep=unified_world.make(seeds,force=families,nonce=19176531,noise=noise)
        organism=new_lives(seeds,checkpoint,keep_plastic=keep_plastic,address=address)
        result=habitat_world.rollout(organism,ep.arena,silent_probes=True)
        scores=unified_world.measures(ep,result)
        report={}
        for k in (*ep.checks,"both","coverage"):
            v=scores[k].reshape(4,n)
            report[k]={"same":float(v[[0,3]].mean()),"conflicting":float(v[[1,2]].mean())}
        return dict(scenario=name,independent_worlds=n,lives=4*n,trials_per_life=ep.arena.menu.body.trials,
                    outcome=report)
    ep=longrun_world.make(roots,blocks=blocks,noise=noise,mode="reverse_return",nonce=8365331)
    organism=new_lives(roots,checkpoint,keep_plastic=keep_plastic,address=address)
    result=habitat_world.rollout(organism,ep.arena,silent_probes=True)
    regions=longrun_world.count_regions(ep,result);out={}
    for k,v in regions.items():
        count=sum(v["count"])
        out[k]=dict(opportunities=count,before=sum(v["pre"])/count if count else None,
                    after=sum(v["post"])/count if count else None)
    return dict(scenario=name,independent_worlds=len(roots),trials_per_life=ep.arena.menu.body.trials,
                ticks_per_life=16*ep.arena.menu.body.trials,outcome=out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent",type=Path,default=DEFAULT);parser.add_argument("--game",choices=["transfer","long","cross-game"],default="transfer")
    parser.add_argument("--worlds",type=int,default=24);parser.add_argument("--seed",type=int,default=20261119)
    parser.add_argument("--blocks",type=int,default=96);parser.add_argument("--noise",type=float,default=.05)
    parser.add_argument("--fixed-covariance",action="store_true")
    parser.add_argument("--address",choices=("distributed","local","local-selective"),default="distributed")
    args=parser.parse_args()
    if args.worlds<1 or not 0<=args.noise<=.5:parser.error("worlds >= 1 and noise in [0,.5] required")
    roots=np.random.default_rng(args.seed).integers(8_000_000,470_000_000,args.worlds).tolist()
    print(json.dumps(run_suite(args.game,roots,args.agent,args.blocks,args.noise,not args.fixed_covariance,args.address),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":main()



