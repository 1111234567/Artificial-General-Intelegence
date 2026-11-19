"""Select and validate delay perturbations without choosing from held-out effects."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import attractor_search as engine
from .attractor_motif import Encoder, GAP, RULE, WINDOWS, choices, curriculum, probes, readout_on_prefix, responses, vector
from .spike_analysis import estimates


ROOT = engine.OUT / "research-2026-11-19"
FIT_RANGE = 8_901_000_001
HOLD_RANGE = 8_902_000_001
WINDOW = "early"


def experiment(v, birth, seeds):
    tape = curriculum(seeds)
    enc = Encoder(v, birth)
    start, wdiff, rate = enc.run(tape)
    w, guesses = readout_on_prefix(v, wdiff, tape)
    q = probes(start, seeds)
    baseline = responses(enc, q, collect=True)
    full = responses(enc, q, mask=np.ones_like(enc.weight), window=WINDOWS["all"])
    acc = lambda out: (choices(w, out, q) == q.goals).mean(axis=1)
    core = {"score": estimates(acc(baseline)), "full_gap_without_recurrence": estimates(acc(full)),
            "recurrence_effect": estimates(acc(baseline) - acc(full)),
            "prefix_learning": estimates((guesses[:, 32:] == tape.goal[:, 32:]).mean(axis=1)), "prefix_spikes": rate}
    return enc, q, w, baseline, core


def feature_signal(state, span):
    syn = state["synaptic"][slice(*span)]
    by_cue = syn.reshape(len(syn), -1, 8, engine.N)
    means = by_cue.mean(axis=1)
    between = np.square(means - means.mean(axis=1, keepdims=True)).mean(axis=(0, 1))
    total = np.var(syn, axis=1).mean(axis=0)
    ratio = between / np.maximum(total, 1e-9)
    return between, ratio


def cycles(enc):
    """Enumerate distinct simple directed cycles of two through four nodes."""
    good = []
    graph = [np.flatnonzero(enc.matrix[:, j]).tolist() for j in range(engine.N)]

    def scan(first, path):
        for after in graph[path[-1]]:
            if after == first and len(path) >= 2:
                good.append(tuple(path))
            elif len(path) < 4 and after > first and after not in path:
                scan(first, path + [after])

    for start in range(engine.N):
        scan(start, [start])
    return good


def cycle_mask(enc, subset):
    result = np.zeros_like(enc.weight, dtype=bool)
    for loop in subset:
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            result[b] |= enc.src[b] == a
    return result


def mask_properties(enc, mask, rate):
    values = enc.weight[mask]
    received = enc.to_matrix(mask * enc.weight) @ rate
    return np.array([len(values), (values > 0).sum(), (values < 0).sum(), np.linalg.norm(values),
                     np.sum(np.abs(values)), np.linalg.norm(received), (np.abs(values) * rate[enc.src[mask]]).sum(),
                     np.linalg.norm(enc.to_matrix(mask * enc.weight) * rate[None])], dtype=np.float64)


def props_dict(values):
    return dict(zip(("edges", "positive_edges", "negative_edges", "weight_l2", "absolute_weight_sum",
                     "mean_receiver_current_l2", "mean_absolute_current", "weighted_source_activity_l2"),
                    values.tolist(), strict=True))


def random_senders(enc, target, rng, activity, syn_signal):
    send = enc.matrix.sum(0)
    sign = send > 0
    count = len(target)
    pre = np.column_stack((activity, syn_signal, np.linalg.norm(enc.inp, axis=1),
                           np.linalg.norm(enc.proj, axis=0), (enc.matrix != 0).sum(axis=0),
                           np.linalg.norm(enc.matrix, axis=0), np.linalg.norm(enc.matrix, axis=1)))
    norm = np.maximum(pre.std(axis=0), 1e-6)
    priority = np.linalg.norm((pre[target][:, None] - pre[None]) / norm, axis=2)
    observed = set()
    options = np.arange(engine.N)
    for _ in range(1200):
        chosen = []
        for i, t in enumerate(target):
            eligible = (sign == sign[t]) & ~np.isin(options, target) & ~np.isin(options, chosen)
            ids = options[eligible]
            if not ids.size:
                break
            p = np.exp(-.42 * priority[i, ids])
            p /= p.sum()
            chosen.append(int(rng.choice(ids, p=p)))
        if len(chosen) == count:
            as_key = tuple(sorted(chosen))
            if as_key not in observed:
                observed.add(as_key)
                info = np.isin(enc.src, chosen)
                yield info, as_key


def random_receivers(enc, target, rng, activity):
    observed = set()
    rows, cols = np.nonzero(target)
    proposals = []
    for receiver, edge in zip(rows, cols, strict=True):
        w = enc.weight[receiver]
        ids = np.flatnonzero((np.sign(w) == np.sign(w[edge])) & ~target[receiver])
        if not len(ids):
            return
        a = activity[enc.src[receiver]]
        err = np.abs(w[ids] - w[edge]) / max(.05, abs(w[edge])) + np.abs(a[ids] - a[edge]) / max(.015, a[edge])
        p = np.exp(-.9 * err)
        proposals.append((ids, p / p.sum()))
    for _ in range(700):
        selected = tuple(int(rng.choice(ids, p=p)) for ids, p in proposals)
        if len(set(zip(rows, selected))) != len(rows) or selected in observed:
            continue
        observed.add(selected)
        candidate = np.zeros_like(target)
        candidate[rows, selected] = True
        yield candidate, selected


def pack(mask):
    return np.argwhere(mask).tolist()


def unpack(indices):
    mask = np.zeros((engine.N, engine.D), dtype=bool)
    if indices:
        mask[tuple(np.asarray(indices).T)] = True
    return mask


def match_controls(enc, q, baseline, target, sampler, span):
    """Freeze structural and activity matches without using action outcomes."""
    rate = baseline["state"]["synaptic"][slice(*span)].mean(axis=(0, 1))
    target_props = mask_properties(enc, target, rate)

    def error(values):
        normalized = (values - target_props) / np.maximum(np.abs(target_props), .012)
        return float(np.linalg.norm(normalized))

    generated = [(error(v), v, mask, index) for mask, index in sampler
                 for v in (mask_properties(enc, mask, rate),)]
    if not generated:
        raise ValueError("No feasible same-sign intervention")
    graded = sorted(generated, key=lambda v: v[0])[:32]
    target_run = responses(enc, q, mask=target, window=span)
    target_rate = target_run["pulses"][:, span[0]:GAP + 1].mean() / engine.N
    target_after = target_run["pulses"][:, span[1]:GAP + 1].mean() / engine.N
    candidates = []
    for _, props, mask, ids in graded:
        out = responses(enc, q, mask=mask, window=span)
        rate = out["pulses"][:, span[0]:GAP + 1].mean() / engine.N
        after = out["pulses"][:, span[1]:GAP + 1].mean() / engine.N
        penalty = 2.5 * abs(rate - target_rate) / .004 + 1.2 * abs(after - target_after) / .003
        candidates.append((error(props) + penalty, props, mask, ids, rate, after))
    choices_ = sorted(candidates, key=lambda v: v[0])[:8]
    return {"selected": pack(target), "props": props_dict(target_props), "spike_rate": float(target_rate),
            "recovery_spike_rate": float(target_after), "all_candidate_count": len(generated),
            "controls": [{"mask": pack(mask), "index": list(index), "spike_rate": float(rate),
                           "recovery_spike_rate": float(after), "props": props_dict(props)}
                          for _, props, mask, index, rate, after in choices_]}


def identify(enc, q, base, birth):
    span = WINDOWS[WINDOW]
    amount, reliability = feature_signal(base["state"], span)
    spike = base["state"]["spike"][slice(*span)].mean(axis=(0, 1))
    src_rate = base["state"]["synaptic"][slice(*span)].mean(axis=(0, 1))
    influence = np.sqrt(amount) * reliability**.25 * np.linalg.norm(enc.matrix, axis=0)**.4
    top = np.argsort(-influence, kind="stable")[:6]
    rng = np.random.default_rng(np.random.SeedSequence((birth, 481266)))
    senders = match_controls(enc, q, base, np.isin(enc.src, top), random_senders(enc, top, rng, spike, src_rate), span)
    loops = cycles(enc)

    def rank(loop):
        ids = np.asarray(loop)
        mass = np.abs(enc.matrix[np.roll(ids, -1), ids])
        return float(np.mean(np.sqrt(amount[ids]) * mass) * (max(1e-9, float(np.prod(reliability[ids])))**(.25 / len(ids))))

    ordered = sorted(loops, key=rank, reverse=True)
    picked = []
    covered = set()
    for loop in ordered:
        adjacent = list(zip(loop, loop[1:] + loop[:1], strict=True))
        replaceable = all(np.count_nonzero(np.sign(enc.weight[b]) == np.sign(enc.matrix[b, a])) > 1 for a, b in adjacent)
        if replaceable and not covered.intersection(loop):
            covered.update(loop)
            picked.append(loop)
        if len(picked) == 3:
            break
    cyc = cycle_mask(enc, picked)
    circuit = match_controls(enc, q, base, cyc, random_receivers(enc, cyc, rng, src_rate), span)
    circuit["chosen_cycles"] = [{"vertices": list(loop), "parents_signed_product": float(np.prod(
        enc.matrix[np.roll(loop, -1), loop])), "search_score": rank(loop)} for loop in picked]
    senders["selected_units"] = top.tolist()
    return {"sender_cut": senders, "cycle_cut": circuit, "node_signal": influence.tolist(), "full_cycles_considered": len(loops)}


def loop_activity(spikes, cycle_specs):
    """Count spike sequences compatible with a directed return, without causal attribution."""
    events = spikes.astype(bool)
    out = []
    for spec in cycle_specs:
        nodes = tuple(spec["vertices"])
        complete = np.zeros(events.shape[1], dtype=bool)
        any_step = np.zeros(events.shape[1], dtype=bool)
        for phase in range(len(nodes)):
            path = nodes[phase:] + nodes[:phase] + nodes[phase:phase + 1]
            arrived = events[:, :, path[0]]
            for node in path[1:]:
                recent = np.zeros_like(arrived)
                for delay in (1, 2, 3):
                    recent[delay:] |= arrived[:-delay]
                arrived = events[:, :, node] & recent
                if node == path[1]:
                    any_step |= arrived.any(axis=0)
            complete |= arrived.any(axis=0)
        out.append({"vertices": list(nodes), "compatible_directed_step": float(any_step.mean()),
                    "compatible_complete_return": float(complete.mean())})
    return out


def decoder(z):
    examples = z.reshape(-1, z.shape[-1]).astype(np.float64)
    tags = np.eye(8)[np.tile(np.arange(8), len(z))]
    mu = examples.mean(0)
    std = np.maximum(examples.std(0), 1e-6)
    design = (examples - mu) / std
    gram = design.T @ design + .02 * len(examples) * np.eye(examples.shape[-1])
    w = np.linalg.solve(gram, design.T @ (tags - tags.mean(0)))
    return {"mean": mu.tolist(), "sd": std.tolist(), "coef": w.tolist()}


def public_classes(embedding, params):
    x = (embedding - np.asarray(params["mean"])) / np.asarray(params["sd"])
    return (x @ np.asarray(params["coef"])).argmax(axis=-1)


def ev(enc, q, w, base, arrangement, params):
    true_action = choices(w, base, q)
    native = true_action == q.goals
    labels = np.arange(8)[None]
    native_decoded = public_classes(base["address"], params)
    base_beh = native.mean(1)
    base_public = (native_decoded == labels).mean(1)
    full_syn = base["state"]["synaptic"]
    span = WINDOWS[WINDOW]
    rate_sla = slice(span[0], GAP + 1)
    reference_current = base["mean_input"][slice(*span)]

    def run(mask, operation, when, swap=1):
        result = responses(enc, q, mask=mask, window=when, operation=operation, baseline=full_syn, swap=swap)
        picked = choices(w, result, q)
        decoded = public_classes(result["address"], params)
        diff = result["mean_input"][slice(*span)] - reference_current
        return {"performance": (picked == q.goals).mean(1), "decoded": (decoded == labels).mean(1),
                "rate": result["pulses"][:, rate_sla].mean() / engine.N,
                "after_spikes": result["pulses"][:, span[1]:GAP + 1].mean() / engine.N,
                "current_rms": float(np.sqrt(np.square(diff).mean())), "picked": picked, "pub": decoded}

    def reports(out):
        return {"correct": estimates(out["performance"]), "minus_baseline": estimates(out["performance"] - base_beh),
                "public_decoded": estimates(out["decoded"]), "public_minus_baseline": estimates(out["decoded"] - base_public),
                "spike_fraction": float(out["rate"]), "after_spikes": float(out["after_spikes"]),
                "current_rms_vs_baseline": out["current_rms"]}

    def contrast(lesson, pool):
        for key in ("performance", "decoded"):
            yield key, estimates(lesson[key] - np.mean([v[key] for v in pool], axis=0))

    output = {"baseline": {"behaviour": estimates(base_beh), "public_decoded": estimates(base_public),
                           "rate": float(base["pulses"][:, rate_sla].mean() / engine.N)}}
    for name in ("sender_cut", "cycle_cut"):
        group = arrangement[name]
        mask = unpack(group["selected"])
        masks = [unpack(v["mask"]) for v in group["controls"]]
        options = {}
        for operation in ("zero", "mean"):
            y = run(mask, operation, span)
            comparisons = [run(ctrl, operation, span) for ctrl in masks]
            options[operation] = {"candidate": reports(y), "controls": [reports(c) for c in comparisons],
                                  "candidate_minus_average_controls": dict(contrast(y, comparisons))}
        options["timing"] = {stage: reports(run(mask, "zero", WINDOWS[stage])) for stage in ("middle", "late", "post")}

        def donor_effect(a, b):
            original0, original1 = q.goals[:, np.arange(8) ^ 1], q.goals[:, np.arange(8) ^ 2]
            within = ((a["picked"] == original0).astype(float) - (a["picked"] == original1)
                      + (b["picked"] == original1) - (b["picked"] == original0)) / 2
            pub0, pub1 = labels ^ 1, labels ^ 2
            pub = ((a["pub"] == pub0).astype(float) - (a["pub"] == pub1)
                   + (b["pub"] == pub1) - (b["pub"] == pub0)) / 2
            return {"behaviour": within.mean(1), "public": pub.mean(1)}

        y1, y2 = [run(mask, "replay", span, which) for which in (1, 2)]
        change = donor_effect(y1, y2)
        controls = [donor_effect(*[run(c, "replay", span, which) for which in (1, 2)]) for c in masks]
        options["cue_swap"] = {"candidate": {k: estimates(v) for k, v in change.items()},
                               "minus_average_controls": {k: estimates(v - np.mean([r[k] for r in controls], axis=0))
                                                          for k, v in change.items()},
                               "native_score_swap1": reports(y1), "native_score_swap2": reports(y2)}
        if name == "cycle_cut":
            options["observed_compatible_sequences_in_first9"] = loop_activity(
                base["state"]["spike"][:9], group["chosen_cycles"])
        output[name] = options
    return output


def history(enc, q, seed):
    """Return/recovery assay beyond the trained readout horizon."""
    n = len(q.labels) // 8
    jitter = np.random.default_rng(seed).normal(0, .05, size=(n, engine.N)).astype(np.float32)
    direct = responses(enc, q, time=84, collect=True, distractor=False)
    shifted = responses(enc, q, time=84, collect=True, distractor=False, kick=(18, np.repeat(jitter, 8, axis=0)))
    reference = direct["state"]
    report = {}
    for at in (8, 18, 28, 40, 56, 80):
        row = {}
        for part in ("synaptic", "membrane", "trace", "features"):
            a = reference[part][at].reshape(n, 8, -1)
            b = shifted["state"][part][at].reshape(n, 8, -1)
            # Candidate-specific tag separation under exactly matched prefixes.
            code_diff = np.sqrt(np.square(a - a[:, np.arange(8) ^ 1]).mean(axis=(1, 2)))
            kick_diff = np.sqrt(np.square(a - b).mean(axis=(1, 2)))
            row[part] = {"two_cues_rms": estimates(code_diff), "same_cue_kick_rms": estimates(kick_diff)}
        row["spikes_over_next_four_ticks"] = float(direct["pulses"][:, at:at + 4].mean())
        report[str(at)] = row
    tails = [5, 9, 14, 19, 28, 40, 56, 80]
    return {"measurements": report, "spikes_per_tick": direct["pulses"].mean(0).tolist(),
            "no_more_spikes_after_tick": {t: float((direct["pulses"][:, t:].sum(1) == 0).mean()) for t in tails},
            "kick_no_more_spikes_after_tick": {t: float((shifted["pulses"][:, t:].sum(1) == 0).mean()) for t in tails}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("discover", "confirm"))
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--frozen", type=Path)
    args = ap.parse_args()
    v = vector()
    if args.stage == "discover":
        candidates = (43, 101, 137, 151, 152, 153, 154, 155, 156, 157, 158)
        worlds = tuple(range(FIT_RANGE, FIT_RANGE + 24))
        reviewed = []
        chosen = []
        anatomy = {}
        for seed in candidates:
            enc, q, w, ref, core = experiment(v, seed, worlds)
            eligible = core["score"]["mean"] >= .50 and core["recurrence_effect"]["mean"] >= .12
            reviewed.append({"birth": seed, "eligible_new_birth": eligible, "statistics": core})
            print(seed, core, flush=True)
            if seed == 43 or (seed != 43 and eligible and len(chosen) < 2):
                chosen.append(seed)
                masks = identify(enc, q, ref, seed)
                params = decoder(ref["address"])
                anatomy[str(seed)] = {"geometry": masks, "decoder": params,
                                      "baseline": core, "topological_observations": history(enc, q, seed),
                                      "exploratory_control_scores": ev(enc, q, w, ref, masks, params)}
                print("dissected", seed, flush=True)
        data = {"rule": RULE, "rationale": "437918 showed a positive long-gap recurrence contrast in all four archived births; "
                "representative 43 was less extreme than 11. One independent eligible birth is chosen by the displayed rule on discovery worlds only.",
                "train_worlds": worlds, "fixed_gap": GAP, "matched_window": WINDOWS[WINDOW], "births": chosen,
                "screen": reviewed, "anatomy": anatomy,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    else:
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        if frozen["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Motif selection or scoring code differs from source used in discovery")
        worlds = tuple(range(HOLD_RANGE, HOLD_RANGE + 64))
        out = {}
        for seed in frozen["births"]:
            frozen_seed = frozen["anatomy"][str(seed)]
            enc, q, w, ref, core = experiment(v, seed, worlds)
            ret = ev(enc, q, w, ref, frozen_seed["geometry"], frozen_seed["decoder"])
            out[str(seed)] = {"initial_baseline": core, "outcomes": ret, "dynamics": history(enc, q, seed)}
            print("held birth", seed, "core", core["score"],
                  {k: ret[k]["zero"]["candidate_minus_average_controls"]["performance"] for k in ("sender_cut", "cycle_cut")}, flush=True)
        data = {"held_worlds": worlds, "births": frozen["births"], "condition": "fixed 28 ticks, public intruder at tick 14; paired final recall",
                "results": out}
    args.artifact.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
