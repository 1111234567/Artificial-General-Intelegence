"""Verify stored motor choices and compare within-world payoffs in the two receipt-rehearsal cohorts."""

import json

import numpy as np

from .. import edge_site_replay as replay
from .. import noon_market_stats as stats


CONTRASTS=(("replay","random"),("replay","native"),("replay","uniform"),
           ("replay","b_only"),("replay","wrong_old"),("b_only","random"),("uniform","native"))


def verify_birth(birth):
    path=replay.DATA/f"birth-{birth}.npz"
    with np.load(path,allow_pickle=False) as z:
        if ((int(z["birth"]),int(z["worlds"]),int(z["seed"]))!=(birth,128,replay.SEED)
                or str(z["source"])!=replay.digest(replay.__file__)
                or str(z["plan"])!=replay.digest(replay.EVIDENCE/"replay-plan.md")):
            raise ValueError(f"Unexpected motor archive {path}")
        data={p:{} for p in replay.POLICIES}
        max_error=0.
        for angle in replay.ANGLES:
            target=z[f"{angle}/target"]
            laws=z[f"{angle}/laws"]
            reference=f"{angle}/random"
            for p in replay.POLICIES:
                prefix=f"{angle}/{p}"
                if p not in ("uniform","native"):
                    for sign in ("positive","negative"):
                        error=np.max(np.abs(z[f"{prefix}/{sign}"]-z[f"{reference}/{sign}"]))
                        max_error=max(float(error),max_error)
                    np.testing.assert_array_equal(z[f"{prefix}/edges"],z[f"{reference}/edges"])
                for delay in (4,32):
                    action=z[f"{prefix}/{delay}/action"]
                    if action.shape!=(512,2) or not np.isin(action,(-1,1)).all():
                        raise ValueError("Unpaid or missing motor choice")
                    a,b=(action==target).T
                    for label,subset in (("conflict",laws[:,0]!=laws[:,1]),("agree",laws[:,0]==laws[:,1])):
                        for name,observed in (("A",a),("B",b),("both",a&b)):
                            values=observed[subset].reshape(128,2).mean(axis=1)
                            data[p][f"{angle}/{label}/{delay}/{name}"]=values
        if max_error>3e-5:
            raise ValueError(f"Excess difference in same-sign deferred dose of {max_error}")
        max_groups={str(t):int(z[f"{t}/groups"].max()) for t in replay.ANGLES}
        for p in replay.POLICIES:
            for delay in (4,32):
                for name in ("A","B","both"):
                    data[p][f"primary/{delay}/{name}"]=np.mean(
                        [data[p][f"{angle}/conflict/{delay}/{name}"] for angle in (15,30)],axis=0)
        return data,dict(birth=birth,max_dose_error=max_error,max_groups=max_groups,
                         raw_sha256=replay.digest(path),parent_sha256=str(z["body"]),
                         physics=str(z["physics"]))


def main():
    result={}
    for cohort in ("original","independent"):
        births=replay.parents.births(cohort)
        collected=[verify_birth(b) for b in births]
        series={p:{k:np.stack([run[p][k] for run,_ in collected]) for k in collected[0][0][p]}
                for p in replay.POLICIES}
        result[cohort]=dict(births=births,worlds=128,archives=[meta for _,meta in collected],
                            scores=stats.summarize(series,128,births,replay.SEED+61,CONTRASTS))
    replay.storage._write_json(replay.EVIDENCE/"replay-results.json",result)
    descriptions={"native":"原生全写","uniform":"相同份额在各候选间均摊","random":"随机选地址",
                  "b_only":"只重演新 B","wrong_old":"重演时反转旧 A 回执","replay":"重演自己的 A、B",
                  "replay minus random":"重演减随机","replay minus native":"重演减全写",
                  "replay minus b_only":"重演减只用 B","replay minus wrong_old":"自己的回执减反转 A"}
    lines=["# 用一份亲历的旧回执和临时前向检验选择地址","",
           "身体照常学习 A 和 B。额外保存 A 当时的菜单（三个感觉分量）和由亲自选侧与实际味觉恢复的一个方向，用它来检验候选写入对旧行为的影响。B 的记录来自眼前接触。各候选写入之后在临时副本中呈现旧菜单，观察身体对原摆向和反向的运动概率。先在每组四条候选边中逐一更换地址，再合并并重新检查整体方案。正式回答只用最终选中的图，通过此前未见的测试顺序、摆向和动作随机数实际答两次。试验中的记忆回放和决策搜索是外加的；两对象都没有噪声，过去的味觉在回访时仍有效。","",
           "## 接近且相反的对象","",
           "下表是 15°、30° 冲突场景等权汇总，4 拍后回访。绝对值为正确率百分比，差值为百分点，区间同时重抽样身体与几何。每批六个出生、128 个新几何根，两种相反赋值在根内配对。",""]
    def fmt(entry):
        return f'{entry["mean"]*100:.2f} [{entry["ci95"][0]*100:.2f}, {entry["ci95"][1]*100:.2f}]'
    for cohort,title in (("original","原六个体"),("independent","独立六个体")):
        scores=result[cohort]["scores"]
        lines.extend([f"### {title}","","| 地址选择 | 先学的 A | 后学的 B | 同一生命两次都答对 |","|---|---:|---:|---:|"])
        for key,label in descriptions.items():
            cells=[fmt(scores[key]["primary/4/"+m]) for m in ("A","B","both")]
            lines.append("| "+" | ".join([label,*cells])+" |")
        lines.append("")
    lines.extend(["## 等待与规律关系","","重演自己 A、B 的真实联合正确率减同剂量随机的差，单位为百分点。","",
                  "| 角度、规律及等待 | 原六个体 | 独立六个体 |","|---|---:|---:|"])
    for angle in replay.ANGLES:
        for label,title in (("conflict","相反"),("agree","相同")):
            for delay in (4,32):
                cells=[fmt(result[c]["scores"]["replay minus random"][f"{angle}/{label}/{delay}/both"])
                       for c in ("original","independent")]
                lines.append("| "+" | ".join([f"{angle}°，{title}，{delay} 拍",*cells])+" |")
    meta=[m for c in result.values() for m in c["archives"]]
    max_dose=max(m["max_dose_error"] for m in meta)
    n=max(max(m["max_groups"].values()) for m in meta)
    lines.extend(["","所有选址比较逐生命核对了实际被减弱的地址数。按正、负符号分别核对从原生全写中扣回的权重量。"
                  f"和同组随机地址的最大差为 {max_dose:.2g}。单条生命有效候选最多 {n} 组；"
                  "每组四个临时候选，每个候选对 A 和 B 各试两个摆向，最后复核合并方案。"
                  "对照中的均摊可以改变更多地址，原生全写使用更多总剂量。","",
                  "[事前方案](replay-plan.md) · [逐条件与逐个体区间](replay-results.json) · "
                  "[从真实动作复算](../../../experiments/pulse_game/reporting/edge_site_replay_report.py)"])
    (replay.EVIDENCE/"replay.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
