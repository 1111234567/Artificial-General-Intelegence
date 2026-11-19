"""Apply the paired topology edit protocol to disjoint-ring births with bridges."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import module_graphs, plastic_topology_analysis as analysis, plastic_topology_study as study


DEFAULT = study.base.REPO / "data" / "pulse-game" / "modular-structural-logic-20261119"


def birth_mask(seed):
    src, dst = module_graphs.graph_for(seed).T
    mask = np.zeros((study.model.WIDTH, study.model.WIDTH), dtype=bool)
    mask[dst, src] = True
    return mask


def source_hashes():
    names = ("module_graphs.py", "plastic_topology_modules.py", "plastic_topology_analysis.py")
    return {**study.source_hashes(), **{n: study.base.sha(Path(__file__).with_name(n)) for n in names}}


def register(out, device):
    target = out / "registered.json"
    if target.exists():
        raise RuntimeError("Registration already exists")
    same = study.base.read(study.DEFAULT / "registered.json")
    seeds = same["seeds"]
    metadata = {str(seed): module_graphs.sample(seed).record() for draws in seeds for seed in draws}
    if not all(record["strongly_connected"] for record in metadata.values()):
        raise RuntimeError("An exact matched birth lacked mutual reachability")
    registration = {
        **same, "created_utc": study.time.strftime("%Y-%m-%dT%H:%M:%SZ", study.time.gmtime()),
        "related_protocol_sha256": study.base.sha(study.DEFAULT / "registered.json"),
        "birth_modules": metadata, "source_sha256": source_hashes(),
        "runtime": study.base.runtime(device),
        "topology": "Exactly the fixed modular 64-arc sampler. Every unit starts in one disjoint directed ring, with one incoming and one outgoing internal arc. 32 random inter-ring arcs give each ring one or more incoming and outgoing bridges. All drawn births were verified to be globally strongly connected without resampling. First 120 updates use the same fixed birth graph in all three arms. After update 120 and each following 120-update block, allow unrestricted nonself additions and any removals preserving one strong component, with no quota by vertex or initial ring, no protected ring or bridge arc, and no forced trade. Actual receiver degree normalizes one learnable gain per receiver. Each model has 449 continuous coefficients at each step.",
        "additional_observables": "Survival and exit of initial within-ring arcs and bridges, count of fully intact original rings, and growth of within-ring arcs outside the initial cycle. The original module labels are descriptive and are never used to propose, score, or reject candidate edits.",
    }
    study.base.write(target, registration)


def registered(out):
    registration = study.base.read(out / "registered.json")
    if registration["source_sha256"] != source_hashes():
        raise RuntimeError("Modular sources changed after registration")
    return registration


def train(out, device):
    registration = registered(out)
    if (out / "frozen.json").exists():
        raise RuntimeError("Checkpoints are already frozen")
    data = study.base.training_data("main_train", device)
    edit_batches = {step: study.edit_data(i, device) for i, step in enumerate(study.EDITS)}
    progress = out / "training-progress.json"
    artifacts = study.base.read(progress) if progress.exists() else []
    for first in range(len(artifacts) * study.SHARD, study.GROUPS, study.SHARD):
        rows = study.specs(registration["seeds"][first:first + study.SHARD], first)
        birth = np.stack([birth_mask(r[2]) for r in rows])
        masks = birth.copy()
        bank = study.model.Bank([r[2] for r in rows], masks).to(device)
        rounds, loss, elapsed = study.fit(bank, rows, masks, data, seed=1468173 ^ first,
                                         edit_batches=edit_batches)
        checkpoint = Path("checkpoints") / f"birth-{first:02d}.npz"
        study.save_npz(out / checkpoint, **{k: t.detach().cpu().numpy() for k, t in bank.named_parameters()},
                       birth=birth, final=masks)
        journal = Path("edits") / f"birth-{first:02d}.json"
        study.base.write(out / journal, {"rounds": rounds, "rows": rows,
                                        "birth": [study.topology(m) for m in birth],
                                        "final": [study.topology(m) for m in masks]})
        artifacts.append({"checkpoint": checkpoint.as_posix(), "checkpoint_sha256": study.base.sha(out / checkpoint),
                          "edits": journal.as_posix(), "edits_sha256": study.base.sha(out / journal),
                          "first_group": first, "rows": rows,
                          "continuous_parameters": sum(x[0].numel() for x in bank.parameters()),
                          "seconds_through_edit_install": elapsed, "last_mean_batch_loss": loss})
        study.base.write(progress, artifacts)
        print(f"fitted modular groups {first}–{first + study.SHARD - 1} in {elapsed:.1f}s", flush=True)
    study.base.write(out / "frozen.json", {"registered_sha256": study.base.sha(out / "registered.json"),
                                          "artifacts": artifacts})


def confirm(out, device):
    registered(out)
    artifacts = study.frozen(out)
    prefix, query, positions = study.env.sensory(study.FINAL_ROOTS, "standard")
    episode = study.env.body_tape(study.FINAL_ROOTS), prefix, query, positions, "standard"
    for artifact in artifacts:
        dest = out / "scores" / f"birth-{artifact['first_group']:02d}.npz"
        if dest.exists():
            continue
        rows = artifact["rows"]
        with np.load(out / artifact["checkpoint"], allow_pickle=False) as checkpoint:
            bank = study.model.Bank([row[2] for row in rows], checkpoint["final"]).to(device)
            bank.load_state_dict({k: torch.from_numpy(checkpoint[k]) for k in study.model.PARAMETERS})
        score = study.base.score(bank, rows, episode, batch_roots=4)
        study.save_npz(dest, standard=score)
        print(f"confirmed modular shard {artifact['first_group']}: {(score[..., 2] > 0).mean():.4f}", flush=True)


def ring_survival(out, artifacts):
    by_arc = {arm: [] for arm in study.ARMS}
    for artifact in artifacts:
        with np.load(out / artifact["checkpoint"], allow_pickle=False) as ckpt:
            for (_, _, seed, _, arm), final, birth in zip(artifact["rows"], ckpt["final"], ckpt["birth"], strict=True):
                record = module_graphs.sample(seed)
                src, dst = np.array(record.within).T
                surviving_ring_arcs = int(final[dst, src].sum())
                inter_src, inter_dst = np.array(record.bridges).T
                surviving_bridges = int(final[inter_dst, inter_src].sum())
                intact = sum(all(final[ring[(i + 1) % len(ring)], node]
                                 for i, node in enumerate(ring)) for ring in record.rings)
                membership = {node: i for i, ring in enumerate(record.rings) for node in ring}
                new_within = sum(membership[src] == membership[dst] for src, dst in study.arcs(final & ~birth))
                by_arc[arm].append((surviving_ring_arcs, surviving_bridges, intact, len(record.rings), new_within))
    keys = ("surviving_internal_arcs", "surviving_bridges", "intact_original_rings",
            "original_rings", "novel_within_ring_arcs")
    return {arm: {key: {**analysis.entries(col), "total": int(col.sum())}
                  for key, col in zip(keys, np.array(records).T, strict=True)}
            for arm, records in by_arc.items()}


def summarize(out, _device):
    registration = registered(out)
    artifacts = study.frozen(out)
    result = analysis.collect(out, registration, artifacts, birth_for=birth_mask)
    result["initial_ring_survival"] = ring_survival(out, artifacts)
    study.base.write(out / "summary.json", result)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "train", "confirm", "summarize"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    {"register": register, "train": train, "confirm": confirm, "summarize": summarize}[
        args.mode](args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
