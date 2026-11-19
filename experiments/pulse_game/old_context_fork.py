"""Matched-prefix tests of why an old rule is lost after prolonged conflicting experience."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from . import habitat_deploy as actor,habitat_world as physics,unified_confirm as sealed,unified_physical as transform
from . import unified_audit as arch,unified_world as task,research_audit as audit, world
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


class Fork(actor.Forager):
    def __init__(self,front,route,seeds,ep,motion,arm):
        super().__init__(front,route,seeds,start=ep.arena.start_test,motor=motion)
        self.ep,self.arm=ep,arm
        self.labels=ep.arena.menu.body.ctx
        self.original_g=None;self.old_code=None;self.held=None;self.details={}
    def address(self):
        actual=super().address()
        if self.round==self.ep.checks["return_A"] and self.arm in ("original_code","old_code_and_weights"):
            return self.old_code.copy()
        return actual
    def act(self,obs):
        if obs[:,world.TRIAL_START].all():
            j=self.round+1
            if j==self.ep.arena.start_test:
                self.original_g=self.g.copy()
                u=self.csum[:,0]/np.maximum(self.ccount[:,0,None],1)
                self.old_code=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-7)
            if j==self.ep.checks["return_A"]:
                new=self.g.copy();c=self.old_code;family=self.ep.arena.relation[:,0]
                self.details=dict(sign_before_B=(np.sum(self.original_g*c,1)*family>0).astype(float),
                     sign_after_B=(np.sum(new*c,1)*family>0).astype(float),
                     A_projection_change=((new-self.original_g)*c).sum(1))
                if self.arm in ("old_weights","old_code_and_weights"):self.g[:]=self.original_g
                if self.arm=="restore_only_A_component":self.g+=(((self.original_g-new)*c).sum(1)[:,None])*c
                if self.arm=="cross_life_sham":self.g[:]=self.original_g[np.arange(len(self.g))^1]
        return super().act(obs)


def one(name,model,subset):
    base=OUT/"unified-physical-evolution-v1"/model
    d=json.loads((base/"frozen.json").read_text());s=d["settings"]
    f,r,m=arch.resources(OUT/"unified-transfer-v1",s["model"])
    f,r,m=transform.materialize(f,r,m,np.asarray(d["best"]["genome"]),s["mode"])
    ep=sealed.make(subset,name);seeds=subset*4
    arms={};ref=None
    for tag in ("native","old_weights","original_code","old_code_and_weights","restore_only_A_component","cross_life_sham"):
        a=Fork(f,r,seeds,ep,m,tag);res=physics.rollout(a,ep.arena,silent_probes=True)
        at=ep.checks["return_A"]
        prefix=(res.contact_side[:,:at],res.contact_tick[:,:at],res.reward[:,:at])
        if ref is None:ref=prefix
        if not all(np.array_equal(x,y) for x,y in zip(prefix,ref,strict=True)):raise ValueError("Prefix differs")
        ms=task.measures(ep,res)
        arms[tag]={k:np.asarray(v).reshape(4,len(subset)).tolist() for k,v in
                   {**a.details,"A":ms["return_A"],"both":ms["both"]}.items()}
    return dict(model=model,case=name,seeds=subset,rows=arms)


def main():
    folder=audit.OUT/"A-return-forks";folder.mkdir(exist_ok=True)
    roots=np.random.default_rng(53091175).integers(7_000_000,475_000_000,128).tolist()
    names=sorted(p.parent.name for p in (OUT/"unified-physical-evolution-v1").glob("*--full-*/frozen.json"))
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs={pool.submit(one,c,n,roots):folder/(n+"-"+c+".json") for n in names for c in
            ("standard","long_competing_run","similar_habitats") if not (folder/(n+"-"+c+".json")).is_file()}
        for f in as_completed(fs):_write(fs[f],f.result());print(fs[f].name,flush=True)
    _write(folder/"complete.json",dict(models=names,worlds=len(roots)))


if __name__=="__main__":main()
