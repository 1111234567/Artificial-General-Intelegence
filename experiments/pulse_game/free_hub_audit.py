"""Cut recurrent outputs after an intact prefix in frozen, degree-free networks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import free_graph_model as model, ring_logic_study as prior, ring_logic_world as env


EVIDENCE = prior.REPO / "docs/evidence/free-graph-2026-11-19"
PROTOCOL = EVIDENCE / "hubs-protocol.json"
DRAWS = 8


def matched_masks(arcs, seed, width=model.WIDTH):
    rng = np.random.default_rng([int(seed), 771109])
    adjacency = np.zeros((width, width), bool)  # target, source
    sources, targets = np.asarray(arcs).T
    adjacency[targets, sources] = True
    outdegree = adjacency.sum(axis=0)
    hub = int(rng.choice(np.flatnonzero(outdegree == outdegree.max())))
    recipients = np.flatnonzero(adjacency[:, hub])
    cuts = np.zeros((2 + DRAWS, width, width), bool)
    cuts[1, recipients, hub] = True
    for random_cut in cuts[2:]:
        for receiver in recipients:
            source = rng.choice(np.flatnonzero(adjacency[receiver]))
            random_cut[receiver, source] = True
    if (cuts & ~adjacency).any() or not np.all(cuts[1:].sum(2) == cuts[1].sum(1)):
        raise RuntimeError("Unmatched cuts")
    return hub, cuts


class QueryOnlyBank(model.Bank):
    def __init__(self, seed, cuts):
        super().__init__([seed]*len(cuts), ["fully_free"]*len(cuts))
        self.register_buffer("keep_messages", torch.from_numpy(~cuts), persistent=False)
        self.apply_cut = False

    def matrix(self, *, drop=False):
        recurrent = super().matrix(drop=drop)
        return recurrent*self.keep_messages if self.apply_cut else recurrent

    def forward(self, observations, state, *, retain=None, **kwargs):
        # ring_logic_study.score explicitly retains logits only on the scored fork.
        self.apply_cut = retain is not None
        return super().forward(observations, state, retain=retain, **kwargs)


def _group(job):
    from . import free_graph_study as study
    group, artifact, root_ids = job
    torch.set_num_threads(1)
    device = torch.device("cpu")
    source, rows = study.load_checkpoint(study.DEFAULT / artifact["path"], device,
                                         pick=study.selection_indices(artifact))
    episode = study.scoring_data(root_ids, "standard")
    outputs, graphs = {}, []
    for i, row in enumerate(rows):
        if row[0] != group or row[4] != "fully_free":
            continue
        rule = env.RULES[row[3]]
        hub, cuts = matched_masks(model.graph_for(row[2],row[4]), row[2])
        test = QueryOnlyBank(row[2], cuts)
        test.load_state_dict({k:v[i:i+1].repeat((len(cuts),)+(1,)*(v.ndim-1))
                              for k,v in source.state_dict().items()})
        # This also checks matching after optimization (the gain may differ by row).
        effective = test.matrix().detach().numpy()
        removed_power = np.sum((effective * cuts)**2, axis=(1,2))
        np.testing.assert_allclose(removed_power[1:], removed_power[1], rtol=2e-6, atol=1e-7)
        scored = prior.score(test, [row]*len(cuts), episode, batch_roots=8).astype(np.int8)
        actual_side, reward = scored[...,0], scored[...,2]
        target = env.truth()[row[3]].reshape(1,1,-1)
        if not np.array_equal((actual_side==target) & (actual_side>=0), reward>0):
            raise RuntimeError("Reward/physical side mismatch")
        outputs[rule] = scored
        graphs.append(dict(group=group, rule=rule, restart=row[1], seed=row[2], hub=hub,
                           removed=int(cuts[1].sum()), removed_power=float(removed_power[1]),
                           control_overlap=[int(np.sum(r & cuts[1])) for r in cuts[2:]],
                           cuts=[np.argwhere(r).tolist() for r in cuts[1:]]))
    if set(outputs) != set(env.RULES): raise RuntimeError(f"Missing selected group {group}")
    return group, outputs, graphs


def analyze(stacks, graph_rows, roots, metadata):
    rng = np.random.default_rng(4178941)
    ng = stacks[env.RULES[0]].shape[0]
    wb = rng.multinomial(ng, np.full(ng,1/ng), 20000)/ng
    wr = rng.multinomial(len(roots), np.full(len(roots),1/len(roots)),20000)/len(roots)
    summary = {}
    def estimate(data, alpha):
        sampled = ((wb @ data)*wr).sum(axis=1)
        return dict(mean=float(data.mean()), interval=np.quantile(sampled,[alpha,1-alpha]).tolist())
    for rule, x in stacks.items():
        correct = (x[...,2]>0).mean(axis=-1)
        intact, hub, scattered = correct[:,0], correct[:,1], correct[:,2:].mean(axis=1)
        summary[rule] = dict(intact=float(intact.mean()), hub_cut=float(hub.mean()), scattered_cut=float(scattered.mean()),
                             intact_minus_hub=estimate(intact-hub,.025),
                             intact_minus_scattered=estimate(intact-scattered,.025),
                             scattered_minus_hub=estimate(scattered-hub,.025/len(env.RULES)))
    archive = EVIDENCE/"hub-forks.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as output:
        for name,x in stacks.items():
            buffer=io.BytesIO()
            np.save(buffer,x,allow_pickle=False)
            output.writestr(name+".npy",buffer.getvalue())
    prior.write(EVIDENCE/"hub-results.json",dict(world_roots=roots, results=summary, graphs=graph_rows,
                axes=["birth group","intact, hub cut, eight controls","root","truth cell","side/tick/taste"],
                archive_sha256=prior.sha(archive), protocol_sha256=prior.sha(PROTOCOL),
                source_sha256=prior.sha(Path(__file__)), **metadata))


def main():
    from . import free_graph_study as study
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    config=prior.read(PROTOCOL)["worlds"]
    ids=list(range(config["start"],config["start"]+config["count"]))
    frozen=study.freeze_guard(study.DEFAULT)
    records=[a for a in frozen["files"] if a["design"]=="tied" and a["updates"]==1200]
    jobs=[]
    for a in records:
        groups=len(a["calibration_selected"])
        jobs.extend((g,a,ids) for g in range(a["first_group"],a["first_group"]+groups))
    if not jobs or len({j[0] for j in jobs})!=len(jobs): raise RuntimeError("Incomplete or overlapping shards")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        completed=sorted(pool.map(_group,jobs),key=lambda row:row[0])
    stacks={rule:np.stack([bank[rule] for _,bank,_ in completed]) for rule in env.RULES}
    graphs=[item for _,_,batch in completed for item in batch]
    analyze(stacks,graphs,ids,dict(frozen_sha256=prior.sha(study.DEFAULT/"frozen.json")))
    print(f"Confirmed {len(jobs)} selected birth groups across {len(ids)} new worlds.", flush=True)


if __name__=="__main__":
    main()
