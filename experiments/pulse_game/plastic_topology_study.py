"""Train, freeze, and score paired outcome-guided and random topology edits."""

from argparse import ArgumentParser
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import plastic_topology_model as model, ring_logic_world as env
from . import ring_logic_study as base
from .ring_logic_model import train_step


DEFAULT = base.REPO / "data" / "pulse-game" / "structural-logic-20261119"
ARMS = ("frozen", "guided", "random")
GROUPS = 8
RESTARTS = 2
SHARD = 4
LEARNING_RATE = .024
UPDATES = 1200
EDITS = tuple(range(120, UPDATES, 120))
EDIT_ROOT = 43267000
ROOTS_PER_EDIT = 4
FINAL_ROOTS = range(97381000, 97381128)
PROPOSALS = (0, 1, 1, 2, 2, -1, -1, -2, -2)
SOURCE_NAMES = ("plastic_topology_study.py", "plastic_topology_model.py",
                "free_graph_model.py", "ring_logic_study.py", "ring_logic_model.py",
                "ring_logic_world.py", "world.py")


def source_hashes():
    return {name: base.sha(Path(__file__).with_name(name)) for name in SOURCE_NAMES}


def mask_hash(mask):
    return hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()


def arcs(mask):
    return np.argwhere(mask)[:, ::-1].tolist()


def topology(mask):
    return {"arcs_source_target": arcs(mask), "sha256_target_source_uint8": mask_hash(mask),
            "edges": int(mask.sum()), "self_edges": int(mask.trace()),
            "in_degree": mask.sum(axis=1).tolist(), "out_degree": mask.sum(axis=0).tolist(),
            "strongly_connected": model.strong(mask)}


def register(out, device):
    target = out / "registered.json"
    if target.exists():
        raise RuntimeError("Registration already exists")
    seeds = np.random.default_rng(78322409).choice(2 ** 31, size=(16, 2), replace=False)[:GROUPS]
    edit_roots = list(range(EDIT_ROOT, EDIT_ROOT + len(EDITS) * ROOTS_PER_EDIT))
    split = {"gradient": list(base.roots("main_train")), "structure": edit_roots,
             "confirmation": list(FINAL_ROOTS)}
    if sum(map(len, split.values())) != len(set().union(*split.values())):
        raise RuntimeError("World splits overlap")
    base.write(target, {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary": "Physical first positive contact at the fixed final checkpoint over all 20 pairs of each novel episode root, pooling both restarts within paired birth groups. Guided minus frozen and guided minus random for each rule; birth and root bootstrap with six simultaneous Bonferroni intervals.",
        "rules": env.RULES, "arms": ARMS, "seeds": seeds.tolist(), "roots": split,
        "updates": UPDATES, "rate": LEARNING_RATE, "edit_after_update": EDITS,
        "candidates_per_round_signed_arc_count": PROPOSALS,
        "topology": "Uniform free strongly connected directed simple 64-arc birth sampled by free_graph_model, shared across all arms and tasks at each restart. Every nonself arc is eligible for growth. A retirement can remove any existing arcs that leave one strong component. Each recurrent arc uses its receiver's trainable gain divided by sqrt(actual in-degree). Redundant gated self retention is intrinsic. 449 continuous coefficients per circuit in every arm and round, independent across rows.",
        "training": "Use public signed tastes physically delivered by random-side exploration. Identical Adam minibatches, continuous coefficient births and optimizer budgets for all arms within each restart. A mask edit retains all coefficients and optimizer state. Each edit receives its own disjoint block of four structural-training roots with all 20 ordered pairs. BCE averaged across movement ticks, conditioning on explored contact side; exclude missing tastes. This is a receipt-prediction objective derived from physical outcomes.",
        "selection": "Guided chooses the strictly lowest probe loss if it beats the incumbent. Each signed size has two proposals when feasible. Ties follow proposal order with keep first. Random adopts a uniformly selected independently sampled feasible proposal of the same signed size at the same time; stratification excludes signed sizes unavailable to random. Random proposals incur the same scoring workload and their losses are rejected as selector input. Frozen retains the birth mask. Growth/retirement sets have no vertex degree quotas or forced compensating edits.",
        "evaluation": "No calibration or restart selection. Confirmation is opened only after writing and hashing the final checkpoints and edit journals. Timeout counts as incorrect.",
        "source_sha256": source_hashes(), "runtime": base.runtime(device),
    })


def registered(out):
    registration = base.read(out / "registered.json")
    if registration["source_sha256"] != source_hashes():
        raise RuntimeError("Source changed after registration")
    return registration


def specs(seeds, first):
    return [(group, restart, seed, rule, arm)
            for group, draws in enumerate(seeds, first)
            for restart, seed in enumerate(draws)
            for rule in range(len(env.RULES))
            for arm in ARMS]


def propose(mask, rng):
    inactive = np.argwhere(~mask & ~np.eye(len(mask), dtype=bool))
    active = np.argwhere(mask)
    candidates = []
    for amount in PROPOSALS:
        proposal = None
        selectable = inactive if amount > 0 else active
        if amount == 0:
            proposal = mask
        elif len(selectable) >= abs(amount):
            for _ in range(64 if amount < 0 else 1):
                loc = selectable[rng.choice(len(selectable), abs(amount), replace=False)]
                changed = mask.copy()
                changed[loc[:, 0], loc[:, 1]] = amount > 0
                if amount > 0 or model.strong(changed):
                    proposal = changed
                    break
        candidates.append(proposal)
    return candidates


def choice(guided, control_candidates, rng):
    """Select counterfactual guided receipts, then sample the paired random loci."""
    eligible = {PROPOSALS[i] for i, mask in enumerate(control_candidates) if mask is not None}
    pick = min((i for i, n in enumerate(PROPOSALS) if n in eligible and np.isfinite(guided[i])),
               key=lambda i: (guided[i], i))
    peers = [i for i, n in enumerate(PROPOSALS) if n == PROPOSALS[pick] and control_candidates[i] is not None]
    return pick, int(rng.choice(peers))


def edit_data(round_index, device):
    start = EDIT_ROOT + round_index * ROOTS_PER_EDIT
    roots = range(start, start + ROOTS_PER_EDIT)
    prefix, query, _ = env.sensory(roots, "standard")
    side, receipt = env.exploration(env.body_tape(roots), start ^ 0x94C557)
    cells = len(env.PAIRS)
    sensory = env.TIMING["standard"][0]
    query = query[:, :, :sensory + base.world.TICKS].reshape(-1, sensory + base.world.TICKS,
                                                           env.SENSES)
    return [torch.from_numpy(array).to(device) for array in (
        prefix.repeat(cells, axis=0)[None], query[None], side.reshape(-1),
        receipt.reshape(len(env.RULES), -1),
    )]


@torch.no_grad()
def counterfactual(bank, row_indices, masks, rules, data):
    prefix, query, side, receipt = data
    sensory = env.TIMING["standard"][0]
    output = []
    for start in range(0, len(masks), 48):
        block = slice(start, start + 48)
        edits = np.stack(masks[block])
        ids = torch.as_tensor(row_indices[block], device=bank.drive.device)
        rule = torch.as_tensor(rules[block], device=bank.drive.device)
        params = tuple(x[ids] for x in bank.parameters())
        normalized = model.normalize(torch.as_tensor(edits, device=bank.drive.device))
        state = bank.drive.new_zeros((len(ids), query.shape[1], bank.width))
        _, state = model.simulate(prefix, state, normalized, params)
        prediction, _ = model.simulate(query, state, normalized, params, retain=sensory)
        labels = receipt[rule]
        valid = labels != 0
        targets = (labels > 0).to(prediction.dtype)[..., None].expand_as(prediction)
        logits = prediction * (2 * side.clamp_min(0) - 1)[None, :, None]
        error = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
        # Keep each chunk on the execution device until the entire batch has finished.
        output.append((error * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1))
    values = torch.cat(output).cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite probe loss")
    return values


def edits(bank, rows, masks, step, data):
    start_time = time.perf_counter()
    variants = {}
    plan = []
    candidates, rules, row_indices = [], [], []
    for index, (group, restart, seed, rule, arm) in enumerate(rows):
        if arm == "frozen":
            continue
        rng = np.random.default_rng([seed, rule, step, ARMS.index(arm), 574812])
        variants[index] = propose(masks[index], rng)
        for j, mask in enumerate(variants[index]):
            if mask is not None:
                plan.append((index, j))
                candidates.append(mask)
                rules.append(rule)
                row_indices.append(index)
    prepared_time = time.perf_counter()
    values = counterfactual(bank, row_indices, candidates, rules, data)
    scored_time = time.perf_counter()
    loss = np.full((len(rows), len(PROPOSALS)), np.inf)
    for idx, value in zip(plan, values, strict=True):
        loss[idx] = value

    journal = []
    for index in range(0, len(rows), len(ARMS)):
        frozen, guided, random = range(index, index + len(ARMS))
        group, restart, seed, rule, _ = rows[frozen]
        rng = np.random.default_rng([seed, rule, step, 2334187])
        guided_pick, random_pick = choice(loss[guided], variants[random], rng)
        pair = {"group": group, "restart": restart, "seed": seed, "rule": env.RULES[rule],
                "signed_edit": PROPOSALS[guided_pick]}
        for current, picked in ((guided, guided_pick), (random, random_pick)):
            incumbent = masks[current]
            selected = variants[current][picked]
            pair[rows[current][4]] = {
                "chosen_candidate": picked, "prior_hash": mask_hash(incumbent),
                "result_hash": mask_hash(selected), "prior_count": int(incumbent.sum()),
                "result_count": int(selected.sum()), "strong": model.strong(selected),
                "candidates": [None if mask is None else {
                    "loss": float(loss[current, j]), "sha256": mask_hash(mask),
                    "growth": arcs(mask & ~incumbent), "retired": arcs(incumbent & ~mask),
                } for j, mask in enumerate(variants[current])],
            }
            masks[current] = selected
        if masks[guided].sum() != masks[random].sum():
            raise RuntimeError("Random control did not match the guided edge count")
        journal.append(pair)
    bank.edit(masks)
    if bank.drive.device.type == "cuda":
        torch.cuda.synchronize(bank.drive.device)
    return {"update": step, "proposal_seconds": prepared_time - start_time,
            "score_seconds": scored_time - prepared_time,
            "selection_and_install_seconds": time.perf_counter() - scored_time,
            "evaluated_candidates": len(plan), "paired": journal}


def fit(bank, rows, masks, data, *, seed, edit_batches, updates=UPDATES):
    device = bank.drive.device
    gpu = device.type == "cuda"
    targets = torch.tensor([row[3] for row in rows], device=device)
    groups = torch.arange(len(rows) // (len(env.RULES) * len(ARMS)), device=device)
    opt = torch.optim.Adam(bank.parameters(), lr=LEARNING_RATE, fused=gpu, capturable=gpu)
    torch.manual_seed(seed)

    def step():
        return train_step(bank, *data, targets, groups, env.TIMING["standard"][0], 1, opt)

    start = time.perf_counter()
    warmup = 0
    if gpu:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(base.WARMUP):
                loss = step()
        torch.cuda.current_stream().wait_stream(stream)
        warmup = base.WARMUP
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()

    rounds = []
    for count in range(warmup + 1, updates + 1):
        if gpu:
            graph.replay()
        else:
            loss = step()
        if count in edit_batches or count == updates:
            completed = int(opt.state[bank.drive]["step"].item())
            if completed != count or not torch.isfinite(loss):
                raise RuntimeError(f"Expected {count} finite Adam steps, got {completed}")
            if count in edit_batches:
                record = edits(bank, rows, masks, count, edit_batches[count])
                rounds.append(record)
                accepted = sum(p["signed_edit"] != 0 for p in record["paired"])
                print(f"updates={count} mean_loss={float(loss) / len(rows):.4f} "
                      f"edits={accepted}/{len(record['paired'])} elapsed={time.perf_counter() - start:.1f}s",
                      flush=True)
    return rounds, float(loss) / len(rows), time.perf_counter() - start


def save_npz(path, **data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".writing.npz")
    np.savez_compressed(temp, **data)
    temp.replace(path)


def train(out, device):
    registration = registered(out)
    if (out / "frozen.json").exists():
        raise RuntimeError("Checkpoints are already frozen")
    data = base.training_data("main_train", device)
    edit_batches = {step: edit_data(i, device) for i, step in enumerate(EDITS)}
    progress_path = out / "training-progress.json"
    artifacts = base.read(progress_path) if progress_path.exists() else []
    for first in range(len(artifacts) * SHARD, GROUPS, SHARD):
        rows = specs(registration["seeds"][first:first + SHARD], first)
        birth = np.stack([model.birth_mask(row[2]) for row in rows])
        masks = birth.copy()
        bank = model.Bank([row[2] for row in rows], masks).to(device)
        parameters = sum(x[0].numel() for x in bank.parameters())
        if parameters != 449:
            raise RuntimeError(f"Unexpected number of continuous coefficients: {parameters}")
        rounds, loss, elapsed = fit(bank, rows, masks, data, seed=1468173 ^ first,
                                    edit_batches=edit_batches)
        dest = out / "checkpoints" / f"birth-{first:02d}.npz"
        save_npz(dest, **{k: t.detach().cpu().numpy() for k, t in bank.named_parameters()},
                 birth=birth, final=masks)
        edit_path = out / "edits" / f"birth-{first:02d}.json"
        base.write(edit_path, {"rounds": rounds, "rows": rows,
                               "birth": [topology(m) for m in birth],
                               "final": [topology(m) for m in masks]})
        artifacts.append({"checkpoint": dest.relative_to(out).as_posix(),
                          "checkpoint_sha256": base.sha(dest),
                          "edits": edit_path.relative_to(out).as_posix(),
                          "edits_sha256": base.sha(edit_path), "first_group": first,
                          "rows": rows, "continuous_parameters": parameters,
                          "seconds_through_edit_install": elapsed, "last_mean_batch_loss": loss})
        base.write(progress_path, artifacts)
        print(f"fitted groups {first}…{first + len(rows) // 18 - 1} in {elapsed:.1f}s", flush=True)
    base.write(out / "frozen.json", {"registered_sha256": base.sha(out / "registered.json"),
                                    "artifacts": artifacts})


def frozen(out):
    seal = base.read(out / "frozen.json")
    if base.sha(out / "registered.json") != seal["registered_sha256"]:
        raise RuntimeError("Registration changed after freezing")
    for artifact in seal["artifacts"]:
        for key in ("checkpoint", "edits"):
            if base.sha(out / artifact[key]) != artifact[f"{key}_sha256"]:
                raise RuntimeError(f"Frozen {key} changed")
    return seal["artifacts"]


def confirm(out, device):
    registered(out)
    artifacts = frozen(out)
    profile = "standard"
    prefix, query, positions = env.sensory(FINAL_ROOTS, profile)
    episode = env.body_tape(FINAL_ROOTS), prefix, query, positions, profile
    for artifact in artifacts:
        first = artifact["first_group"]
        dest = out / "scores" / f"birth-{first:02d}.npz"
        if dest.exists():
            continue
        rows = artifact["rows"]
        with np.load(out / artifact["checkpoint"], allow_pickle=False) as checkpoint:
            bank = model.Bank([row[2] for row in rows], checkpoint["final"]).to(device)
            bank.load_state_dict({key: torch.from_numpy(checkpoint[key]) for key in model.PARAMETERS})
        scores = base.score(bank, rows, episode, batch_roots=4)
        save_npz(dest, standard=scores)
        print(f"confirmed shard {first}: mean first positive contact {(scores[..., 2] > 0).mean():.4f}",
              flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "train", "confirm"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    {"register": register, "train": train, "confirm": confirm}[args.mode](args.out,
                                                                          torch.device(args.device))


if __name__ == "__main__":
    main()
