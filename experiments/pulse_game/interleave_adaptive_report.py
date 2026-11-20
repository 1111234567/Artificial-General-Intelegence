"""Summarize the three independent reward-fitted gains on the same untouched deployment worlds."""

from hashlib import sha256
import json

import numpy as np

from . import interleave_adaptive_reference as study
from . import interleave_capacity_report as stats
from . import interleave_homeostasis as home, interleave_slow_reference as slow
from .interleave_interference import EVIDENCE


def report():
    seeds = study.SEEDS
    births = home.CONFIRM_BIRTHS
    ages = [slow.AGES.index(i) for i in (16, 64, 128)]
    hashes, coefficients = {}, {}
    for seed in seeds:
        path = study.fitted(seed)
        with np.load(path) as saved:
            if int(saved["step"]) != study.STEPS:
                raise ValueError("Incomplete training")
            coefficients[seed] = saved["theta"].tolist()
        hashes[str(path.relative_to(study.DEST).as_posix())] = sha256(path.read_bytes()).hexdigest()
    outcomes = {}
    for eco in home.ECOLOGIES:
        for reps in (1, 4):
            tables = {k: [] for k in ("adaptive", "constant", "fixed", "permuted", "mean")}
            for seed in seeds:
                groups = {k: [] for k in tables}
                for b in births:
                    file = study.DEST/str(seed)/"validation"/f"{b}-{eco}-{reps}.npz"
                    hashes[file.relative_to(study.DEST).as_posix()] = sha256(file.read_bytes()).hexdigest()
                    with np.load(file) as z:
                        for k in tables:
                            groups[k].append(z[k][ages].mean((0,2,3)))
                for k in tables:
                    tables[k].append(np.stack(groups[k]))
            tables = {k:np.stack(v) for k,v in tables.items()}
            for k in ("constant","fixed","mean","permuted"):
                tables["adaptive minus "+k] = tables["adaptive"] - tables[k]
            tables["constant minus fixed"] = tables["constant"] - tables["fixed"]
            count = tables["adaptive"].shape[2]
            paired = stats.Crossed(len(births),count,971733,5000)
            rng = np.random.default_rng(991713)
            fit_weights = rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),5000)/len(seeds)
            def measure(values):
                # Each fit is tested on all the same bodies and worlds.
                boot = np.zeros(5000)
                conditional = []
                for i, rows in enumerate(values):
                    each = ((paired.br @ rows) * paired.bw).sum(1)
                    boot += fit_weights[:,i] * each
                    conditional.append(dict(mean=float(rows.mean()), ci95=np.quantile(each,[.025,.975]).tolist()))
                return dict(mean=float(values.mean()),crossed_ci95=np.quantile(boot,[.025,.975]).tolist(),
                            fit_seeds=conditional,by_body=values.mean((0,2)).tolist())
            outcomes[f"{eco}/{reps}"] = {k:{m:measure(v[:,:,:,j]) for j,m in enumerate(study.READS)}
                                          for k,v in tables.items()}
    out=dict(fit_seeds=seeds,untuned_test_bodies=births,steps=study.STEPS,coefficients=coefficients,
             selected_ages=(16,64,128),raw_sha256=hashes,results=outcomes,
             definition="All real answers after one new X contact except old_before. Interval resamples optimizer "
                        "seeds, test births, and shared world geometries as separate crossed factors. The three "
                        "searches are few; their separate point estimates and conditional intervals are also given. "
                        "All eight private histories, ages, and paired intervention values stay together.")
    (EVIDENCE/"adaptive-reference.json").write_text(json.dumps(out,indent=1)+"\n", encoding="utf8")


if __name__=="__main__":
    report()
