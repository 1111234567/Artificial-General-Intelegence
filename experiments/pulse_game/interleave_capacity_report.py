"""Recompute capacity and joint retention directly from paired actions and shared worlds."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .interleave_interference import DATA,EVIDENCE
from . import interleave_capacity as cap


def metrics(raw,branch,k):
    cut=raw["capacities"].tolist().index(k)
    b=cap.BRANCHES.index(branch)
    target=raw["right_value"][:,:k+1,-1].T
    actual=raw["side"][b,cut,:k+1]==target
    stats={"A":actual[0],"all":actual.all(0)}
    if k:
        stats.update(latest=actual[-1],both_latest=actual[0]&actual[-1],
                     B_mean=actual[1:].mean(0),both_mean=(actual[:1]&actual[1:]).mean(0))
    return {name:x.reshape(-1,4).mean(1) for name,x in stats.items()}


class Crossed:
    def __init__(self,n_roots,n_worlds,seed,draws=5000):
        rng=np.random.default_rng(seed)
        self.br=rng.multinomial(n_roots,np.ones(n_roots)/n_roots,draws)/n_roots
        self.bw=rng.multinomial(n_worlds,np.ones(n_worlds)/n_worlds,draws)/n_worlds

    def __call__(self,matrix):
        v=np.asarray(matrix,dtype=float)
        if v.shape!=(self.br.shape[1],self.bw.shape[1]):
            raise ValueError("Incorrect paired birth/world matrix")
        boot=(np.matmul(self.br,v)*self.bw).sum(1)
        return dict(mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                    by_birth=v.mean(1).tolist())


def stage(directory,roots,relative):
    values={k:{branch:{} for branch in cap.BRANCHES} for k in cap.CAPACITIES}
    trace_hashes={}
    for birth in roots:
        path=directory/relative.format(birth=birth)
        trace_hashes[path.relative_to(directory).as_posix()]=sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            for k,cells in values.items():
                for branch,keys in cells.items():
                    for metric,val in metrics(z,branch,k).items():
                        keys.setdefault(metric,[]).append(val)
    for cells in values.values():
        for keys in cells.values():
            for m,v in keys.items():keys[m]=np.stack(v)
    return values,trace_hashes


def summarize(stage_data,bs):
    return {k:{name:{metric:bs(v) for metric,v in d.items()} for name,d in by_mode.items()}
            for k,by_mode in stage_data.items()}


def difference(a,b,bs,arm_a="write",arm_b="write"):
    return {k:{m:bs(v-b[k][arm_b][m]) for m,v in groups[arm_a].items()}
            for k,groups in a.items()}


def chart(dataset,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    root=dataset["primary_matched"]
    fig,(left,right)=plt.subplots(1,2,figsize=(10.4,4.3),layout="constrained")
    xlabel="其间学习的其他对象数（测点等距）"
    x=np.arange(len(cap.CAPACITIES))
    for arm,mode,title,color in (
        ("prior","exposure_no_write","原模型，中间关闭写入","#8c919d"),
        ("prior","write","原单对象模型","#bc662f"),
        ("recent","write","训练时只问最近对象","#677caa"),
        ("uniform","write","训练时可问任一旧对象","#148374")):
        for ax,key,skip in ((left,"A",0),(right,"both_latest",1)):
            if ax is right and mode!="write":continue
            data=[root[arm][k][mode][key] for k in cap.CAPACITIES[skip:]]
            mu=np.array([v["mean"] for v in data])*100
            ci=np.array([v["ci95"] for v in data])*100
            ax.plot(x[skip:],mu,marker="o",label=title,color=color,lw=1.8,ms=4)
            if arm=="uniform":ax.fill_between(x[skip:],ci[:,0],ci[:,1],color=color,alpha=.14)
    for ax,base in ((left,50),(right,25)):
        ax.axhline(base,color="#b4b4b4",lw=.8)
        ax.set_xticks(x if ax is left else x[1:], cap.CAPACITIES if ax is left else cap.CAPACITIES[1:])
        ax.set_ylim(0,104);ax.grid(axis="y",alpha=.16);ax.set_xlabel(xlabel)
        ax.spines[["top","right"]].set_visible(False)
    left.set_ylabel("实际选择正确率（%）")
    left.set_title("回来时，最早的 A 还能答对吗？")
    right.set_title("同一状态能同时答对 A 和最近对象吗？")
    handles,labels=left.get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=2,frameon=False)
    fig.savefig(out,dpi=170)
    plt.close(fig)


def run(out):
    path=DATA/"retention-objective"
    roots=json.loads((path/"run.json").read_text())["births"]
    old_meta=json.loads((DATA/"frozen-capacity/summary.json").read_text())
    old_path=next(k for k in old_meta["raw_sha256"] if "local" in k and "odor" in k and str(roots[0]) in k)
    old_template=old_path.replace(str(roots[0]),"{birth}")
    result={"description":"Pointwise crossed bootstrap. First average the four paired law lanes within each "
                          "geometry and training birth, then resample births and the same held-out geometries "
                          "independently with replacement. Form differences before resampling. Query actions "
                          "come from the same pre-query living state.",
            "bootstrap_draws":5000,"raw_sha256":{}}
    for split in ("matched","replication","confirmation"):
        test=path if split!="confirmation" else DATA/"retention-objective-confirmation"
        group=roots if split!="confirmation" else list(range(811,817))
        evaluations=test/"evaluation"/split/"008000"
        if not (evaluations/f"capacity-uniform-{group[-1]}.npz").exists():continue
        tables={}
        for arm in ("uniform","recent"):
            tables[arm],hashes=stage(evaluations,group,f"capacity-{arm}-{{birth}}.npz")
            result["raw_sha256"][split+"/"+arm]=hashes
        base=tables["uniform"][0]["write"]["A"].shape
        bs=Crossed(*base,seed=722018)
        if split=="matched":
            tables["prior"],hashes=stage(DATA/"frozen-capacity",group,old_template)
            result["raw_sha256"][split+"/prior"]=hashes
        data={arm:summarize(v,bs) for arm,v in tables.items()}
        data["uniform_minus_recent"]=difference(tables["uniform"],tables["recent"],bs)
        data["uniform_write_minus_sham"]=difference(tables["uniform"],tables["uniform"],bs,"write","exposure_no_write")
        if "prior" in tables:
            data["uniform_minus_prior"]=difference(tables["uniform"],tables["prior"],bs)
            data["prior_write_minus_sham"]=difference(tables["prior"],tables["prior"],bs,"write","exposure_no_write")
        result["primary_"+split]=data
    out.mkdir(parents=True,exist_ok=True)
    chart(result,out/"capacity.png")
    (out/"capacity-crossed.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=EVIDENCE)
    run(p.parse_args().out)
