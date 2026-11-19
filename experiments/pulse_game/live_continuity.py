"""Generate short stretches of a live environment successively; keep the entire animal state between them."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from . import habitat_world as bodies,longrun_world as env,lifelong_agent as deployed


ROOT=Path(__file__).resolve().parents[2]/"data/pulse-game"
OUT=ROOT/"native-long-life-v1"


def write(path,data):
    temp=path.with_suffix(".partial.json");temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8");temp.replace(path)


def private(seeds,total):
    tags=[];rules=[];switch=[]
    for s in seeds:
        r=np.random.default_rng(np.random.SeedSequence([s,76219]))
        angles=r.uniform(0.,.40)+np.array([0,r.uniform(.7,1.)])
        tags.append(np.column_stack((np.cos(angles),np.sin(angles)))[r.permutation(2)])
        rules.append(r.choice([-1,1],size=2))
        switch.append(np.array([total//3,2*total//3])+r.integers(-total//48,total//48+1,size=2))
    return np.asarray(tags),np.asarray(rules),np.asarray(switch)


def chunk(seeds,k,blocks,tags,rules,cuts):
    environmental=[int(np.random.SeedSequence([s,k,5931967]).generate_state(1)[0]) for s in seeds]
    return env.make(environmental,blocks=blocks,noise=.07,nonce=799217,private_rules=rules,public_habitats=tags,
                    absolute_switches=cuts,block_offset=k*blocks)


def play(actor,scene,record=False):
    sim=bodies.Body(scene.arena,silent_probes=True,record=record)
    obs=sim.observe()
    while obs is not None:obs=sim.step(actor.act(obs))
    return sim.result()


def study(spec):
    ensemble,worlds,chunks,bpc=spec[:4];address=spec[4] if len(spec)>4 else "distributed";n=worlds;total=bpc*chunks
    seeds=np.random.default_rng(57319871+ensemble).integers(9_000_000,465_000_000,n).tolist()
    dest=(OUT if address=="distributed" else ROOT/("native-long-life-local-v1" if address=="local" else "native-long-life-selective-v1"))/f"cohort-{ensemble}";dest.mkdir(parents=True,exist_ok=True)
    status=dest/"status.json"
    tags,rules,cuts=private(seeds,total)
    actor=deployed.new_lives(seeds,address=address)
    if status.exists():
        saved=json.loads(status.read_text())
        if (saved["settings"]!=list(spec) or saved["seeds"]!=seeds):raise RuntimeError("Interrupted study has different settings")
        actor=deployed.restore_life(dest/saved["snapshot"])
        begin=saved["done_chunks"];stats={reg:np.array(v,np.int64) for reg,v in saved["counters"].items()}
        rank=np.array(saved["B_visits"]);prev=np.array(saved["phase"]);chain=saved["action_chain"]
        det=saved["checkpoint_continuations"];cur_seconds=saved["compute_seconds"]
    else:
        actor.reset(n);begin=0;stats={};rank=np.zeros(n,int);prev=np.zeros(n,int);chain="0"*64;det=[];cur_seconds=0.
    start=time.perf_counter()
    for k in range(begin,chunks):
        # No later chunk is generated before the previous physical stretch finishes.
        w=chunk(seeds,k,bpc,tags,rules,cuts)
        rerun=(k in (1,chunks//2,chunks-1))
        copied=None
        if rerun:
            snapshot=dest/"continuation-test.npz";deployed.save_life(snapshot,actor);copied=deployed.restore_life(snapshot)
        ans=play(actor,w,rerun)
        if copied is not None:
            b=play(copied,w,True)
            fields=("contact_side","contact_tick","reward","pulses","observed_reward")
            if not all(np.array_equal(getattr(ans,v),getattr(b,v)) for v in fields):raise ValueError("Continuation changed a real action")
            for a in deployed.STATE_ARRAYS:
                if not np.array_equal(getattr(actor,a),getattr(copied,a)):raise ValueError("Continued state diverged: "+a)
            if address in ("local","local-selective"):
                for key in ("prototypes","counts","current","overflows"):
                    if not np.array_equal(getattr(actor.places,key),getattr(copied.places,key)):raise ValueError(key)
            if address=="local-selective":
                for key in deployed.READER_ARRAYS:
                    if not np.array_equal(getattr(actor,key),getattr(copied,key)):raise ValueError(key)
            det.append(dict(chunk=k,motor_bits=int(ans.pulses.size),worlds=n,all_state_equal=True))
            # Actually continue from the restored copy, not the old process object.
            actor=copied
        hit=(ans.contact_side>=0)&(ans.contact_side==w.arena.truths)
        def add(key,mask,pre,post):
            if key not in stats:stats[key]=np.zeros((n,3),np.int64)
            stats[key][:,0]+=mask;stats[key][:,1]+=mask*pre;stats[key][:,2]+=mask*post
        for j,(pre,post) in enumerate(w.query_at):
            block=k*bpc+j;phase=(block>=cuts[:,0]).astype(int)+(block>=cuts[:,1]).astype(int)
            rank=np.where(phase!=prev,0,rank);prev=phase
            B=w.block_context[:,j]==1;rank+=B
            if block>=8:
                add("all",np.ones(n,bool),hit[:,pre],hit[:,post])
                add("stable_A",~B,hit[:,pre],hit[:,post])
            if block>=total-50:add("final_50_visits",np.ones(n,bool),hit[:,pre],hit[:,post])
            for name,ph in (("change",1),("return",2)):
                for lo,hi in ((1,1),(2,2),(3,4),(5,8),(9,16),(17,100000)):
                    add(f"{name}_{lo}_{hi}",(phase==ph)&B&(rank>=lo)&(rank<=hi),hit[:,pre],hit[:,post])
        chain=hashlib.sha256(bytes.fromhex(chain)+ans.contact_side.tobytes()+ans.contact_tick.tobytes()+ans.reward.tobytes()).hexdigest()
        if not (np.isfinite(actor.g).all() and np.isfinite(actor.cov).all() and np.isfinite(actor.h).all()):
            raise FloatingPointError("Nonfinite continuous state")
        if (k+1)%10==0 or k+1==chunks:
            snap=f"life-after-{k+1:03d}.npz";deployed.save_life(dest/snap,actor)
            eig=np.linalg.eigvalsh(.5*(actor.cov+actor.cov.transpose(0,2,1)))
            write(status,dict(settings=list(spec),seeds=seeds,done_chunks=k+1,total_chunks=chunks,
                  requests_per_life=(k+1)*bpc*12,ticks_per_life=(k+1)*bpc*12*16,
                  actor_request_zero_based=actor.round,actor_tick_count=actor.it,
                  counters={m:v.tolist() for m,v in stats.items()},B_visits=rank.tolist(),phase=prev.tolist(),
                  action_chain=chain,checkpoint_continuations=det,compute_seconds=cur_seconds+time.perf_counter()-start,
                  snapshot=snap,min_covariance_eigen=float(eig.min()),max_covariance_eigen=float(eig.max()),
                  max_abs_rule_weight=float(np.abs(actor.g).max()),state_file_bytes=(dest/snap).stat().st_size))
            print(ensemble,k+1,"/",chunks,flush=True)
    return dict(ensemble=ensemble,worlds=n,requests=chunks*bpc*12)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worlds",type=int,default=12)
    p.add_argument("--chunks",type=int,default=200);p.add_argument("--blocks",type=int,default=40)
    p.add_argument("--cohorts",type=int,default=3)
    p.add_argument("--address",default="distributed",choices=("distributed","local","local-selective"))
    args=p.parse_args();directory=OUT if args.address=="distributed" else ROOT/("native-long-life-local-v1" if args.address=="local" else "native-long-life-selective-v1");directory.mkdir(exist_ok=True)
    jobs=[(i,args.worlds,args.chunks,args.blocks)+(() if args.address=="distributed" else (args.address,)) for i in range(args.cohorts)]
    with ProcessPoolExecutor(max_workers=args.cohorts) as pool:
        all_done=[f.result() for f in as_completed([pool.submit(study,job) for job in jobs])]
    write(directory/"complete.json",dict(results=all_done,checkpoint=str(deployed.DEFAULT),address=args.address))


if __name__=="__main__":main()


