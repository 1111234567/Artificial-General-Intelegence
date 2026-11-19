"""Compare the source-trained sensory graph to its cold initialization."""

import json

import numpy as np
import torch

from experiments.undirected.evolved_plasticity_world import CONTEXT
from .legacy_grounded import DEFAULT_GRAPH, GroundedAgent, Setting, base_config, play, public_calibration, read, sense, sha
from .legacy_grounded_outcomes import DEV_SEEDS, TRAIN_SEEDS, causal_predictions, collect, fit
from . import research_audit as audit


class NativeAffineDecoder(GroundedAgent):
    """Use the source decoder including its original bias."""

    def address(self, raw):
        self.sensory = self.graph.advance(self.sensory, raw)
        visible = raw[:, CONTEXT.start:CONTEXT.start + 2]
        updates = visible.any(axis=1)
        self.context[updates] = visible[updates]
        odor = self.sensory @ self.graph.odor + self.graph.bias
        key = (self.context[:,:,None] * odor[:,None,:]).reshape(self.n, 8)
        return key / np.sqrt(.01 + (key * key).sum(axis=1, keepdims=True))


def bootstrap_difference(left, right, key):
    if [x["seed"] for x in left] != [x["seed"] for x in right]:
        raise ValueError("Cannot pair unlike worlds")
    value = np.array([a[key] - b[key] for a,b in zip(left,right,strict=True)], np.float64)
    draws = np.random.default_rng(16183694).integers(len(value), size=(10000,len(value)))
    return {"difference":float(value.mean()),"ci95":np.quantile(value[draws].mean(1),[.025,.975]).tolist(),
            "worlds":len(value)}


def validate_prefix_calculation():
    torch.manual_seed(417990)
    key = torch.randn((2, 9, 5), dtype=torch.float64) * .2
    action = torch.randint(3,(2,9))
    receipt = torch.randint(2,(2,9),dtype=torch.float64)*2-1
    ridge = .15
    received = causal_predictions(key,action,receipt,ridge)
    sequential = torch.zeros_like(received)
    for row in range(len(key)):
        for step in range(key.shape[1]):
            used = (action[row, :step]==action[row,step])
            prior = key[row, :step][used]
            y = receipt[row, :step][used]
            system = torch.eye(key.shape[-1], dtype=key.dtype)*ridge + prior.T @ prior
            sequential[row,step] = key[row,step] @ torch.linalg.solve(system,prior.T@y)
    torch.testing.assert_close(received,sequential,rtol=0,atol=1e-12)
    changed = receipt.clone()
    changed[:,4:] = -receipt[:,4:]
    unused = causal_predictions(key,action,changed,ridge)
    torch.testing.assert_close(unused[:,:5],received[:,:5],rtol=0,atol=0)


def main():
    torch.set_num_threads(1)
    validate_prefix_calculation()
    root = audit.OUT
    path = DEFAULT_GRAPH
    record = torch.load(path,map_location="cpu",weights_only=True)
    initialized = sense(path, random_seed=record["seed"])
    trained = sense(path)
    own, dev = collect(initialized, TRAIN_SEEDS), collect(initialized, DEV_SEEDS)
    references = (collect(trained, TRAIN_SEEDS), collect(trained, DEV_SEEDS))
    for first, second in zip((own,dev),references,strict=True):
        if any(not np.array_equal(a,b) for a,b in zip(first[1:],second[1:],strict=True)):
            raise ValueError("The matched cold and trained graphs did not receive the same attempts and signs")
    mean = own[0].mean(axis=(0,1));std = own[0].std(axis=(0,1)).clip(min=.01)
    cfg = base_config()
    head, history = fit(own, dev, (mean,std), cfg.regularization, 550, shuffled=False)
    frozen = root/"legacy_grounded-cold-outcome-head.pt"
    torch.save({"state":head.state_dict(),
                "graph":{k:torch.as_tensor(v) for k,v in vars(initialized).items()},
                "same_frozen_source":str(path),"reference_graph_sha256":sha(path)},frozen)
    main = read(root/"legacy_grounded-outcomes.json")
    first = read(root/"legacy_grounded-confirmed.json")
    prior = Setting(**first["setting"])
    actual = Setting(**main["setting"])
    col = public_calibration(initialized)
    result={"matched_same_physical_attempts_and_receipts":True,"untrained_seed":record["seed"],
            "same_direction_mask":bool(np.array_equal(trained.dynamics!=0,initialized.dynamics!=0)),
            "public_fitting_max_change":{name:float(np.max(np.abs(getattr(trained,name)-value)))
                                        for name,value in vars(initialized).items()},
            "baseline_frozen":str(frozen), "baseline_frozen_sha256":sha(frozen), "history":history, "sets":{}}
    a = head.export()
    for key, blob in main["sets"].items():
        roots=tuple(blob["roots"])
        alternatives={
            "cold_actual":lambda seeds:GroundedAgent(cfg,initialized,None,actual,seeds,learned_key=a),
            "cold_public_factor":lambda seeds:GroundedAgent(cfg,initialized,col,prior,seeds),
            "source_full_affine":lambda seeds:NativeAffineDecoder(cfg,trained,None,prior,seeds),
        }
        comparison={}
        for name, make in alternatives.items():
            comparison[name]=[row for start in range(0,len(roots),8) for row in play(roots[start:start+8],make)]
        control_pairs=(("trained_actual_minus_cold_actual",blob["rows"]["actual"],comparison["cold_actual"]),
                       ("trained_factor_minus_cold_factor",first["sets"][key]["rows"]["trained_factor"],comparison["cold_public_factor"]))
        paired={name:{metric:bootstrap_difference(x,y,metric) for metric in ("strict","pair_strict_rate")}
                for name,x,y in control_pairs}
        result["sets"][key]={"roots":roots,"rows":comparison,"paired":paired}
        print(key,{name:tuple(round(float(np.mean([r[m] for r in rows])),6) for m in ("strict","pair_strict_rate"))
                   for name,rows in comparison.items()},paired,flush=True)
    (root/"legacy_grounded-graph-control.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
