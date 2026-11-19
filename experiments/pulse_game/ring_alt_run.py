"""Discovery and out-of-map evidence for receptor-gated one-bit foraging."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .ring_alt_policy import ForagingSpec, ReceptorForager
from .ring_world import DERANGED, NEGATIVE, NEW, PERMS, POSITIVE, SIZE, TASTED, make_tape, rollout
from .spike_analysis import estimates


def preference_code(maps):
    return maps.reshape(-1, 8) @ (4 ** np.arange(8))


def world_panel(start, count, trials, *, panel):
    """World-seed filtering is evaluator-only and never passed to policy observations."""
    maps = np.array([np.stack((a, a[d])) for a in PERMS for d in DERANGED])
    code = preference_code(maps)
    code = np.random.default_rng(584421).permutation(code)
    part = code[:len(code) // 2] if panel == "development" else code[len(code) // 2:]
    # Keep the earliest generated worlds in the prescribed half, without looking at outcomes.
    search = make_tape(range(start, start + 5 * count), 1)
    included = np.flatnonzero(np.isin(preference_code(search.preferences), part))[:count]
    if len(included) != count:
        raise RuntimeError("not enough qualifying world seeds in allocation")
    return make_tape([search.seeds[i] for i in included], trials)


def new_policy(tape, spec, seed):
    action = [np.random.SeedSequence((s, 0x414C5452)) for s in tape.seeds]
    return ReceptorForager(spec, birth_seed=seed, action_seeds=action)


class ContactMatched:
    """At each bite, use the sign of the same ordinal bite by an unrelated donor."""

    def __init__(self, policy, signs):
        self.policy = policy
        self.signs = signs

    def reset(self, n):
        self.policy.reset(n)
        self.trial = -1
        self.ordinal = np.zeros(n, np.int8)
        self.aligned = np.zeros(n, np.int32)
        self.exposed = np.zeros(n, np.int32)

    def act(self, obs):
        if obs[0, NEW]:
            self.trial += 1
            self.ordinal.fill(0)
        got = np.flatnonzero(obs[:, TASTED].any(axis=1))
        if not got.size:
            return self.policy.act(obs)
        frame = obs.copy()
        sign = self.signs[got, self.trial, self.ordinal[got]]
        self.ordinal[got] += 1
        self.exposed[got] += 1
        self.aligned[got] += sign != 0
        frame[got, POSITIVE] = sign > 0
        frame[got, NEGATIVE] = sign < 0
        return self.policy.act(frame)


def contact_yoke(donor):
    n, trials, _ = donor.waveform.shape
    if n % 2:
        raise ValueError("contact yoke requires an even number of bodies")
    ordinal = np.cumsum(donor.waveform != 0, axis=2) - 1
    by_bite = np.zeros((n, trials, SIZE), dtype=np.int8)
    source = np.nonzero(donor.waveform)
    r, t, _ = source
    by_bite[r ^ 1, t, ordinal[source]] = donor.waveform[source]
    return by_bite


def pair_accuracy(tape, good, *, cutoff):
    counts = np.zeros((tape.n, 2), dtype=np.int32)
    for row in range(tape.n):
        for cue in range(4):
            visits = [np.flatnonzero((tape.cue[row, cutoff:] == cue) & (tape.ctx[row, cutoff:] == ctx)) + cutoff
                      for ctx in (0, 1)]
            pairs = min(len(visits[0]), len(visits[1]))
            counts[row, 0] += np.count_nonzero(good[row, visits[0][:pairs]] & good[row, visits[1][:pairs]])
            counts[row, 1] += pairs
    return np.divide(counts[:, 0], counts[:, 1], out=np.zeros(tape.n), where=counts[:, 1] > 0)


def summarize_result(tape, result, *, late_size=64):
    t = tape.trials
    cut = max(0, t - late_size)
    late = slice(cut, t)
    positive = result.first_correct
    success = result.solved
    b = result.bites
    scores = {
        "all_first": positive.mean(axis=1),
        "early_first": positive[:, :min(32, t)].mean(axis=1),
        "late_first": positive[:, late].mean(axis=1),
        "late_solved": success[:, late].mean(axis=1),
        "late_bites": b[:, late].mean(axis=1),
        "late_contact": (b[:, late] > 0).mean(axis=1),
        "late_first_tick": np.where(result.first_tick[:, late] >= 0, result.first_tick[:, late], 40).mean(axis=1),
        "late_context_pairs_both_correct": pair_accuracy(tape, positive, cutoff=cut),
        "late_cw_pulses": result.motion[:, late, 1].mean(1),
    }
    return {key: value.tolist() for key, value in scores.items()}


def execute(name, tape, spec, seed, *, retry=False, feedback="true", hidden_after=None, yoke=None, cue_ticks=40):
    policy = new_policy(tape, spec, seed)
    iface = ContactMatched(policy, yoke) if yoke is not None else policy
    result = rollout(iface, tape, retry=retry, feedback=feedback, hidden_after=hidden_after, cue_ticks=cue_ticks)
    scores = summarize_result(tape, result)
    info = {"name": name, "birth_seed": seed, "spec": asdict(spec), "scores": scores, "retry": retry,
            "feedback": "contact-matched-yoked" if yoke is not None else feedback,
            "hidden_after": hidden_after, "cue_ticks": cue_ticks, "first": result.first.tolist(),
            "first_correct": result.first_correct.tolist(), "success": result.solved.tolist(),
            "bites": result.bites.tolist(), "network": policy.stats()}
    if yoke is not None:
        info["yoke_nonzero_bites"] = iface.aligned.tolist()
        info["yoke_exposed_bites"] = iface.exposed.tolist()
    print(f"{seed:2d} {name:27s} {np.mean(scores['early_first']):.3f} → {np.mean(scores['late_first']):.3f}; "
          f"solved {np.mean(scores['late_solved']):.3f}, bites {np.mean(scores['late_bites']):.2f}", flush=True)
    return info, result


def discover(tape, seeds):
    base = ForagingSpec(outcomes="negative", scoring="linear")
    programs = (
        ("negative-filter", {}),
        ("negative-no-filter", {"cov_rate": 0.}),
        ("negative-filter-strong", {"cov_rate": .16}),
        ("negative-sparse-35", {"eligibility_sparsity": .35}),
        ("negative-sparse-65", {"eligibility_sparsity": .65}),
        ("negative-sparse-80", {"eligibility_sparsity": .80}),
        ("negative-sparse-no-filter", {"eligibility_sparsity": .65, "cov_rate": 0.}),
        ("negative-spiking", {"scoring": "spiking"}),
        ("negative-smooth", {"trace_decay": .86}),
        ("negative-fast-trace", {"trace_decay": .54}),
        ("negative-delay-2", {"feedback_delay": 2}),
        ("signed", {"outcomes": "signed"}),
        ("positive", {"outcomes": "positive"}),
        ("frozen", {"outcomes": "frozen"}),
        ("engineered-coincidence", {"representation": "coincidence"}),
    )
    return [execute(name, tape, replace(base, **changes), seed)[0]
            for seed in seeds for name, changes in programs]


def confirm(tape, seeds, spec, *, retry=False):
    results = []
    for seed in seeds:
        live, donor = execute("negative-live", tape, spec, seed, retry=retry)
        results.append(live)
        yoke = contact_yoke(donor)
        arms = (
            ("negative-yoked", spec, "none", None, yoke),
            ("negative-inverted", spec, "invert", None, None),
            ("negative-no-filter", replace(spec, cov_rate=0), "true", None, None),
            ("negative-no-feedback", spec, "none", None, None),
            ("frozen", replace(spec, outcomes="frozen"), "true", None, None),
            ("negative-drop-context", replace(spec, drop_context=True), "true", None, None),
            ("negative-blind", replace(spec, scoring="blind"), "true", None, None),
            ("positive-only", replace(spec, outcomes="positive"), "true", None, None),
            ("signed", replace(spec, outcomes="signed"), "true", None, None),
            ("contact-only", replace(spec, outcomes="contact"), "true", None, None),
            ("negative-counterclockwise", replace(spec, gait="counterclockwise"), "true", None, None),
            ("negative-stop-before-late", spec, "true", tape.trials - 64, None),
            ("negative-spike-readout", replace(spec, scoring="spiking"), "true", None, None),
            ("negative-no-reservoir-loops", replace(spec, recurrent_gain=0), "true", None, None),
            ("negative-sparse65-same-filter", replace(spec, eligibility_sparsity=.65), "true", None, None),
            ("negative-sparse65-no-filter", replace(spec, eligibility_sparsity=.65, cov_rate=0.), "true", None, None),
        )
        if retry:
            arms = tuple(arm for arm in arms if arm[0] in {
                "negative-yoked", "negative-no-feedback", "negative-blind", "signed", "positive-only"})
        for name, arm, feedback, cutoff, donor_sign in arms:
            results.append(execute(name, tape, arm, seed, retry=retry, feedback=feedback,
                                   hidden_after=cutoff, yoke=donor_sign)[0])
    return results


def evidence(results):
    names = list(dict.fromkeys(row["name"] for row in results))
    keys = ("early_first", "late_first", "late_solved", "late_bites", "late_context_pairs_both_correct", "late_contact")
    samples = {name: {key: np.mean([row["scores"][key] for row in results if row["name"] == name], axis=0)
                      for key in keys} for name in names}
    summary = {name: {key: estimates(value.reshape(-1, 2).mean(1) if name == "negative-yoked" else value)
                      for key, value in group.items()} for name, group in samples.items()}
    effects = {}
    if "negative-live" in samples:
        main = samples["negative-live"]
        for name, group in samples.items():
            if name == "negative-live":
                continue
            differences = {key: main[key] - group[key] for key in keys}
            if name == "negative-yoked":
                differences = {key: value.reshape(-1, 2).mean(1) for key, value in differences.items()}
            effects[name] = {key: estimates(value) for key, value in differences.items()}
    return {"arms": summary, "candidate_minus_comparator": effects}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discover", "confirm", "retry"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--trials", type=int, default=192)
    parser.add_argument("--births", type=int, nargs="+", required=True)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    panel = "development" if args.stage == "discover" else "evaluation"
    tape = world_panel(args.start, args.n, args.trials, panel=panel)
    if args.stage == "discover":
        rows = discover(tape, args.births)
    else:
        freeze = json.loads(args.frozen.read_text(encoding="utf-8"))
        rows = confirm(tape, args.births, ForagingSpec(**freeze["spec"]), retry=args.stage == "retry")
    dataset = {"stage": args.stage, "panel": panel, "seed_range_start": args.start, "trials": tape.trials,
               "world_seeds": tape.seeds, "map_codes": preference_code(tape.preferences).tolist(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "rows": rows, "summary": evidence(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
