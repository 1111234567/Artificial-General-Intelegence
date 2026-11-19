"""Paired world and independent-birth analysis for the sealed transfer study."""
import gzip
import io
import json
import platform
from pathlib import Path

import numpy as np

from .relation_study import BIRTHS, DATA, EVIDENCE, REG, digest, fit_path, read, test_roots, write


def by_world(path):
    """Average sibling lives within their world; never sample these as independent."""
    ids = np.asarray(test_roots())
    with np.load(path, allow_pickle=False) as data:
        seeds = data["_seed"]
        parity = data["_parity"]
        ix = np.array([np.flatnonzero(seeds == root) for root in ids])
        if ix.shape != (len(ids), 4) or len(seeds) != ix.size:
            raise ValueError(f"Missing crossed lives in {path}")
        if any(len(np.unique(parity[row], axis=0)) != 4 for row in ix):
            raise ValueError("Parities duplicated")
        return {k: data[k][ix].mean(axis=1) for k in data.files if not k.startswith("_") and data[k].ndim==1}


class Intervals:
    def __init__(self, births, worlds):
        rng = np.random.default_rng(396711)
        self.births = rng.multinomial(births, np.ones(births)/births, 4096) / births
        self.worlds = rng.multinomial(worlds, np.ones(worlds)/worlds, 4096) / worlds

    def estimate(self, x):
        values = np.asarray(x)
        boot = ((self.births @ values) * self.worlds).sum(axis=1)
        return {"mean":float(values.mean()), "ci95":np.quantile(boot, [0.025, 0.975]).tolist(),
                "birth_means":values.mean(axis=1).tolist()}


def plot(curves, summary_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
    doses = REG["doses"]
    for ax, metric, title in zip(axs, ("new","old"), ("New relation", "Return to old relation")):
        for arm, history, label, color in (
            ("adaptive","related","Trained rule / related","#125f7a"),
            ("adaptive","unrelated","Trained rule / unrelated","#ca6423"),
            ("local-selective","related","Hand-coded / related","#77549a"),
        ):
            points = [curves[arm][f"{history}/intact/{d}/{metric}"] for d in doses]
            ax.plot(doses, [p["mean"] for p in points], marker="o", color=color, label=label)
            ax.fill_between(doses, [p["ci95"][0] for p in points], [p["ci95"][1] for p in points],
                            alpha=.13, color=color, linewidth=0)
        ax.axhline(.5, color="#999999", linewidth=.8)
        ax.set(title=title, xlabel="B teaching contacts offered", xticks=doses, ylim=(0, 1.02))
    axs[0].set_ylabel("Correct physical choice per opportunity")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.09,1,1)); fig.savefig(summary_path,dpi=155); plt.close(fig)

def exposure_sensitivity():
    """Secondary, conditional estimate: keep both histories' complete matched opportunities."""
    deltas, fractions = [], []
    for seed in BIRTHS:
        with np.load(DATA / "test" / f"{seed}-adaptive.npz", allow_pickle=False) as a:
            ok = np.ones(len(a["_seed"]), dtype=bool)
            for history in ("related", "unrelated"):
                for d in (0, 2, 4):
                    prefix = f"{history}/intact/{d}/"
                    ok &= a[prefix+"history_receipts"] == 2*REG["history_pairs"]
                    ok &= a[prefix+"b_true_receipts"] == d
                    ok &= a[prefix+"matched_history_sides"].astype(bool)
            delta = {}
            for h in ("related", "unrelated"):
                delta[h] = (a[f"{h}/intact/2/new"]+a[f"{h}/intact/4/new"])/2-a[f"{h}/intact/0/new"]
            deltas.append(float((delta["related"]-delta["unrelated"])[ok].mean()) if ok.any() else None)
            fractions.append(float(ok.mean()))
    return dict(included_fraction_by_birth=fractions, early_gain_by_birth=deltas,
                mean_across_births=float(np.mean(deltas)) if None not in deltas else None)


def report():
    arms = ("adaptive","frozen","local-selective","local-only","initial","public_oracle","physical_oracle")
    curves, effects, rooted = {}, {}, {}
    for arm in arms:
        birth_ids = [0] if arm.endswith("oracle") else BIRTHS
        matrices = [by_world(DATA/"test"/f"{s}-{arm}.npz") for s in birth_ids]
        if any(set(v) != set(matrices[0]) for v in matrices): raise ValueError("Arm schemas differ by birth")
        bank = {key:np.stack([v[key] for v in matrices]) for key in matrices[0]}
        rooted[arm] = bank
        ci = Intervals(len(birth_ids), REG["test_worlds"])
        curves[arm] = {key:ci.estimate(v) for key,v in bank.items() if key.rsplit("/",1)[-1] in ("new","old","joint")}
        def get(history, dose, key="new", variant="intact"):
            return bank[f"{history}/{variant}/{dose}/{key}"]
        def advantage(dose):
            return get("related",dose) - get("unrelated",dose)
        early = (advantage(2)+advantage(4))/2
        row = dict(early_related_minus_unrelated=early, zero_dose_advantage=advantage(0),
                   early_gain_difference=early-advantage(0),
                   related_improvement=(get("related",2)+get("related",4))/2-get("related",0),
                   unrelated_improvement=(get("unrelated",2)+get("unrelated",4))/2-get("unrelated",0),
                   A_retention=get("related",16,"old")-get("related",0,"old"),
                   A0=get("related",0,"old"), A16=get("related",16,"old"), B16=get("related",16),
                   both16=get("related",16,"joint"))
        if arm == "adaptive":
            for variant in ("rollback", "shuffled"):
                for key in ("new","old","joint"):
                    row[f"intact_minus_{variant}_16_{key}"] = get("related",16,key)-get("related",16,key,variant)
        effects[arm] = {k:ci.estimate(v) for k,v in row.items()}
    ci = Intervals(len(BIRTHS), REG["test_worlds"])
    adaptive = rooted["adaptive"]
    comparisons = {}
    def early_gain(bank, history):
        prefix = history + "/intact/"
        return (bank[prefix+"2/new"]+bank[prefix+"4/new"])/2 - bank[prefix+"0/new"]

    for arm in ("frozen","local-selective","initial","local-only"):
        for d in REG["doses"]:
            for metric in ("new","old","joint"):
                k=f"related/intact/{d}/{metric}"
                comparisons[f"adaptive_minus_{arm}/{d}/{metric}"] = ci.estimate(adaptive[k]-rooted[arm][k])
        differences = {h:early_gain(adaptive,h)-early_gain(rooted[arm],h) for h in ("related","unrelated")}
        comparisons[f"adaptive_minus_{arm}/early_gain_difference"] = ci.estimate(differences["related"]-differences["unrelated"])
        for h, values in differences.items():
            comparisons[f"adaptive_minus_{arm}/{h}/early_gain"] = ci.estimate(values)
    counts = {arm:{k:float(v.mean()) for k,v in bank.items() if k.rsplit("/",1)[-1] not in ("new","old","joint")}
              for arm,bank in rooted.items()}
    extra = effects["adaptive"]
    gates = dict(faster_early_learning=extra["early_gain_difference"]["ci95"][0] > 0,
                 old_learned=extra["A0"]["ci95"][0] > 0.5, old_above_chance=extra["A16"]["ci95"][0] > 0.5,
                 old_retained_within_margin=extra["A_retention"]["ci95"][0] > -0.05)
    fits = {str(s):read(fit_path(s)) for s in BIRTHS}
    # Permutations in the sham condition couple roots inside each 32-root block.
    clustered = Intervals(len(BIRTHS), REG["test_worlds"]//32)
    shuffle = {key:clustered.estimate((adaptive[f"related/intact/16/{key}"]-adaptive[f"related/shuffled/16/{key}"])
                                      .reshape(len(BIRTHS),-1,32).mean(2)) for key in ("new","old","joint")}
    # With balanced B taste signs, zero-dose behavior must equal half of its physical coverage.
    for arm, bank in rooted.items():
        if arm == "physical_oracle": continue
        for h in ("related","unrelated"):
            pre=f"{h}/intact/0/"
            if not np.allclose(bank[pre+"new"], .5*bank[pre+"new_contact"]):
                raise ValueError(f"Zero-dose parities differ before B teaching: {arm} {h}")
    supplementary = dict(exact_receipt_and_side_match=exposure_sensitivity(), shuffle_block_intervals=shuffle,
                         related_absolute_early_gain_positive=extra["related_improvement"]["ci95"][0]>0,
                         unrelated_history_learned=curves["adaptive"]["unrelated/intact/0/old"]["ci95"][0]>.5)
    write(EVIDENCE/"relation-summary.json", dict(births=BIRTHS, roots=test_roots(), doses=REG["doses"],
          estimands=effects, curves=curves, paired_comparisons=comparisons, diagnostics=counts,
          gates=gates, success=all(gates.values()), fit_validation={k:v["validation_scores"] for k,v in fits.items()},
          supplementary=supplementary,seal_sha256=digest(EVIDENCE/"relation-seal.json")))
    # Per-root evidence travels with the repository. Full per-life rows stay in the run directory.
    portable = {a:{k:v.tolist() for k,v in bank.items() if k.rsplit("/",1)[-1] in ("new","old","joint")}
                for a,bank in rooted.items()}
    tables = {"relation-worlds.json.gz":dict(roots=test_roots(),births=BIRTHS,rows=portable),
              "relation-fits.json.gz":fits}
    for name, table in tables.items():
        with gzip.GzipFile(EVIDENCE/name, "wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as out:
                json.dump(table, out, separators=(",",":"))
    plot(curves, EVIDENCE/"relation-learning.png")
    dependencies = ("world.py","habitat_world.py","context_local.py","adaptive_local.py","habitat_deploy.py",
                    "content_routing.py","lifelong_agent.py","mixture_flash.py","mixture_world.py",
                    "relation_learner.py","relation_world.py","relation_study.py","relation_analysis.py")
    write(EVIDENCE/"relation-artifacts.json", dict(python=platform.python_version(),numpy=np.__version__,
          note="Post-run hashes from this workspace. Raw runs have four parity siblings per physical root.",
          sources={name:digest(Path(__file__).with_name(name)) for name in dependencies},
          raw_cases={p.name:digest(p) for p in sorted((DATA/"test").glob("*.npz"))},
          outputs={p.name:digest(p) for p in sorted(EVIDENCE.glob("relation-*")) if p.name!="relation-artifacts.json"}))
