"""Read-only cache diagnostics on an independent, explicitly seeded stream.

Oracle paths are reconstructed offline from already observed facts. Query
targets, oracle depth, and phase labels are removed before every model call.
Oracle-depth candidate accuracy is a diagnostic, never deployment performance.
No optimizer, parameter update, or training-mode execution is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.streaming.cache import CacheAgent
from experiments.streaming.stream import cycle, observed


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_hash(model, *, reader_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if reader_only and not (name == "log_scale" or name.startswith(("key.", "query.", "halt."))):
            continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def source_hashes():
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ("cache.py", "stream.py", "diagnose.py")}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def reconstruct_path(world, context, key, max_steps):
    """Read only the causal fact ledger; no task target/depth is an input."""
    path, visited = [], set()
    for _ in range(max_steps):
        identity = (context, key)
        if identity in visited:
            raise ValueError("Offline fact ledger contains an unterminated cycle")
        visited.add(identity)
        if identity not in world:
            raise ValueError("Offline fact ledger lacks an observed required record")
        value, terminal = world[identity]
        path.append((context, key, value, terminal))
        if terminal:
            return path, value
        key = value
    raise ValueError("Offline path exceeded its diagnostic safety bound")


def cache_records(memory, contexts, vocab):
    """Decode only valid hard-cache payloads, retaining physical slot indexes."""
    records = defaultdict(list)
    for slot, record in enumerate(memory.detach().cpu()):
        if record[-2] <= 0.5:
            continue
        context = int(record[:contexts].argmax())
        key = int(record[contexts:contexts + vocab].argmax())
        value = int(record[contexts + vocab:contexts + 2 * vocab].argmax())
        terminal = bool(record[contexts + 2 * vocab] > 0.5)
        records[(context, key)].append((value, terminal, slot))
    return records


def coverage_and_first_slots(path, records):
    found = []
    for context, key, value, terminal in path:
        exact = [slot for stored_value, stored_terminal, slot in records.get((context, key), ())
                 if value == stored_value and terminal == stored_terminal]
        found.append(exact)
    identity_slots = [slot for _, _, slot in records.get(path[0][:2], ())]
    return all(bool(slots) for slots in found), sum(bool(slots) for slots in found) / len(found), found[0], identity_slots


def average(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    covered = [row for row in rows if row["full_chain_coverage"]]
    present = [row for row in rows if row["first_hop_record_present"]]
    missing = [row for row in rows if not row["full_chain_coverage"]]
    return {
        "queries": len(rows),
        "hard_accuracy": average(rows, "hard_correct"),
        "soft_mixture_accuracy": average(rows, "soft_correct"),
        "hard_soft_disagreement": average(rows, "hard_soft_disagree"),
        "oracle_depth_candidate_accuracy": average(rows, "oracle_depth_correct"),
        "full_chain_coverage": average(rows, "full_chain_coverage"),
        "mean_required_record_fraction_present": average(rows, "record_coverage_fraction"),
        "covered_queries": len(covered),
        "hard_accuracy_if_full_chain_covered": average(covered, "hard_correct"),
        "oracle_depth_accuracy_if_full_chain_covered": average(covered, "oracle_depth_correct"),
        "hard_accuracy_if_chain_incomplete": average(missing, "hard_correct"),
        "first_hop_record_presence": average(rows, "first_hop_record_present"),
        "first_hop_correct_record_attention_mass": average(rows, "first_hop_correct_mass"),
        "first_hop_identity_attention_mass": average(rows, "first_hop_identity_mass"),
        "first_hop_attention_mass_if_correct_record_present": average(present, "first_hop_correct_mass"),
        "hard_reads": average(rows, "hard_reads"),
        "soft_expected_reads": average(rows, "soft_expected_reads"),
        "oracle_depth": average(rows, "depth"),
        "hard_stop_exact": average(rows, "stop_exact"),
        "hard_stop_early": average(rows, "stop_early"),
        "hard_stop_late": average(rows, "stop_late"),
        "oracle_depth_beyond_reader_budget": sum(row["oracle_depth_correct"] is None for row in rows),
    }


def grouped(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


@torch.inference_mode()
def diagnose_policy(model, config, policy, *, seed, batch, cycles, noise, queries, device):
    model.eval()
    before = parameter_hash(model)
    data_rng = torch.Generator(device=device).manual_seed(seed)
    policy_rng = torch.Generator(device=device).manual_seed(seed + 4_000_000)
    state = model.initial(batch, device)
    world = [dict() for _ in range(batch)]
    rows = []
    observed_digest = hashlib.sha256()
    start = time.perf_counter()
    writes, skipped = 0, 0.0
    for cycle_index in range(cycles):
        events = cycle(
            batch, data_rng, contexts=model.contexts, vocab=model.vocab,
            nodes=config.get("nodes", 4), noise=noise, queries=queries, device=device,
        )
        for event in events:
            # Even ignored metadata is literally absent from model-facing events.
            visible = replace(event, target=None, depth=None, phase="")
            observed_digest.update(event.kind.encode())
            observed_digest.update(observed(visible, model.contexts, model.vocab).cpu().numpy().tobytes())
            if event.kind == "fact":
                for row, context, key, value, terminal in zip(
                        world, event.context.tolist(), event.key.tolist(),
                        event.value.tolist(), event.terminal.tolist(), strict=True):
                    row[(context, key)] = (value, bool(terminal > 0.5))
                state, drop = model.write(state, visible, policy, policy_rng)
                writes += drop.numel()
                skipped += drop.sum().item()
                continue
            state = model.observe(state, visible)
            if event.kind != "query":
                continue
            paths, answers = [], []
            for index, row_world in enumerate(world):
                path, answer = reconstruct_path(
                    row_world, int(event.context[index]), int(event.key[index]),
                    max_steps=model.vocab * model.contexts + 1,
                )
                paths.append(path)
                answers.append(answer)
                # Task labels are checked only after the independent offline trace.
                if answer != int(event.target[index]) or len(path) != int(event.depth[index]):
                    raise RuntimeError("Offline oracle disagrees with generated diagnostic labels")
            memory_before, hidden_before = state.memory.clone(), state.hidden.clone()
            hard = model.read(state, visible)
            # A fixed budget yields all candidates, including those beyond the
            # deployable threshold halt. Oracle depth is never passed to read().
            full = model.read(state, visible, force_reads=model.max_reads)
            if not torch.equal(memory_before, state.memory) or not torch.equal(hidden_before, state.hidden):
                raise RuntimeError("Read-only inspection mutated cache or hidden state")
            if not torch.equal(hard["attention"][:, 0], full["attention"][:, 0]):
                raise RuntimeError("Forced inspection changed the first forward read")
            soft_answer = full["prob"].argmax(-1)
            for index, path in enumerate(paths):
                records = cache_records(state.memory[index], model.contexts, model.vocab)
                coverage, fraction, first_slots, identity_slots = coverage_and_first_slots(path, records)
                depth = len(path)
                oracle_correct = None
                if depth <= full["candidates"].shape[1]:
                    oracle_correct = int(full["candidates"][index, depth - 1].argmax()) == answers[index]
                reads = int(hard["reads"][index])
                attention = full["attention"][index, 0]
                rows.append({
                    "cycle": cycle_index, "row": index, "phase": event.phase,
                    "depth": depth, "hard_correct": int(hard["answer"][index]) == answers[index],
                    "soft_correct": int(soft_answer[index]) == answers[index],
                    "hard_soft_disagree": int(hard["answer"][index]) != int(soft_answer[index]),
                    "oracle_depth_correct": oracle_correct, "full_chain_coverage": coverage,
                    "record_coverage_fraction": fraction, "first_hop_record_present": bool(first_slots),
                    "first_hop_correct_mass": attention[first_slots].sum().item() if first_slots else 0.0,
                    "first_hop_identity_mass": attention[identity_slots].sum().item() if identity_slots else 0.0,
                    "hard_reads": reads, "soft_expected_reads": float(full["expected_reads"][index]),
                    "stop_exact": reads == depth, "stop_early": reads < depth, "stop_late": reads > depth,
                })
    phase_depth = defaultdict(list)
    for row in rows:
        phase_depth[f'{row["phase"]}/depth{row["depth"]}'].append(row)
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("Diagnostic execution changed parameter values")
    return {
        "policy": policy, "diagnostic_seed": seed, "batch": batch, "cycles": cycles,
        "noise": noise, "queries_per_phase": queries, "observed_stream_sha256": observed_digest.hexdigest(),
        "parameters_unchanged": True, "parameter_sha256_before": before,
        "parameter_sha256_after": after, "reader_parameter_sha256": parameter_hash(model, reader_only=True),
        "skip_rate": skipped / writes if writes else None, "summary": summarize(rows),
        "by_phase": grouped(rows, "phase"), "by_depth": grouped(rows, "depth"),
        "by_phase_depth": {name: summarize(group) for name, group in sorted(phase_depth.items())},
        "seconds": time.perf_counter() - start, "per_query": rows,
    }


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    saved_sources = checkpoint.get("source", {})
    current = source_hashes()
    source_match = {name: saved_sources.get(name) == current[name]
                    for name in ("cache.py", "stream.py")}
    if not all(source_match.values()):
        raise ValueError(f"Checkpoint and current forward/data source disagree: {source_match}")
    model = CacheAgent(**{name: config[name] for name in (
        "contexts", "vocab", "slots", "hidden", "max_reads", "width", "read_dim"
    ) if name in config}).to(device)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    model.eval()
    return model, config, checkpoint.get("policy", "learned"), source_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="data/streaming/diagnostic-v1")
    parser.add_argument("--seed", type=int, default=92_610_026)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--noise", type=int, default=8)
    parser.add_argument("--queries", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--writer-controls", default="mru",
                        help="Additional policies for a learned-writer checkpoint")
    args = parser.parse_args()
    if min(args.batch, args.cycles, args.queries) < 1 or args.noise < 0:
        parser.error("positive batch/cycles/queries and nonnegative noise required")
    checkpoints = [resolve(path).resolve(strict=True) for path in args.checkpoints]
    labels = [f"{path.parent.name}-{path.stem}" for path in checkpoints]
    if len(labels) != len(set(labels)):
        parser.error("Checkpoint labels must be unique")
    directory = resolve(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_before = source_hashes()
    manifest = {
        "configuration": vars(args), "sources": source_before,
        "checkpoints": {label: {"path": str(path), "sha256": file_hash(path)}
                        for label, path in zip(labels, checkpoints, strict=True)},
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "platform": platform.platform(), "device": args.device, "threads": 1},
        "protocol": [
            "Independent diagnostic seed; no formal heldout metrics used for selection.",
            "No optimizer, parameter updates, gradients, or training-mode model calls.",
            "Causal world oracle reconstructed from observed fact events only.",
            "Target, depth and phase removed from every model-facing event.",
            "Oracle-depth candidate accuracy is an offline diagnostic, never a deployment score.",
            "Full-chain coverage requires correct context/key/value/terminal payload for every required record.",
            "Soft mixture computed at the same fixed max_reads budget; hard accuracy uses ordinary halting.",
            "Read inspections checked not to mutate memory or hidden state; all parameter hashes checked.",
        ],
    }
    write_json(directory / "manifest.json", manifest)
    summaries = {}
    for label, path in zip(labels, checkpoints, strict=True):
        model, config, policy, source_match = load_model(path, args.device)
        policies = [policy]
        if policy == "learned":
            policies += [item for item in args.writer_controls.split(",") if item and item != policy]
        runs = {}
        for chosen in policies:
            result = diagnose_policy(
                model, config, chosen, seed=args.seed, batch=args.batch, cycles=args.cycles,
                noise=args.noise, queries=args.queries, device=args.device,
            )
            result.update(checkpoint=str(path), checkpoint_sha256=file_hash(path),
                          checkpoint_configuration=config, forward_source_match=source_match)
            write_json(directory / f"{label}-{chosen}.json", result)
            runs[chosen] = {key: value for key, value in result.items() if key not in {
                "per_query", "checkpoint_configuration", "by_phase_depth"
            }}
            print(json.dumps({"checkpoint": label, "policy": chosen,
                              **result["summary"]}), flush=True)
        summaries[label] = runs
    if source_hashes() != source_before:
        raise RuntimeError("Diagnostic source changed during execution")
    write_json(directory / "summary.json", summaries)
    write_json(directory / "artifact_hashes.json", {
        path.name: file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()
    })
    print(json.dumps({"event": "diagnostic_complete", "output": str(directory)}), flush=True)


if __name__ == "__main__":
    main()
