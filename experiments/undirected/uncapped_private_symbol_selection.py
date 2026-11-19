"""Version the private-symbol search's selection score at generation boundaries."""

import argparse
import json
from pathlib import Path

from experiments.seed_evolution.large_runner import _lock, _utc, _write
from .evolved_plasticity_world import N_SYMBOLS


AMENDMENT = "selection-amendment.json"
TIE_RULE = "strict-then-recovery-v1"
WEIGHTED_RULE = "strict-weighted-recovery-v2"
POST_ANSWER_RULE = "strict-recovery-post-answer-v3"
RECOVERY_DIVISOR = 10


def requests_per_candidate(config):
    return config["actors"] * config["visits"] * N_SYMBOLS


def read_phases(out, requests):
    path = out / AMENDMENT
    if not path.is_file():
        return []
    amendment = json.loads(path.read_text(encoding="utf-8"))
    phases = amendment.get("phases", [amendment])
    if not isinstance(phases, list) or any(not isinstance(phase, dict) for phase in phases):
        raise ValueError(f"Invalid private-symbol selection amendment in {path}")
    previous = 0
    for phase in phases:
        first = phase.get("first_generation")
        rule = phase.get("selection_rule")
        if (rule not in (TIE_RULE, WEIGHTED_RULE, POST_ANSWER_RULE)
                or phase.get("requests_per_candidate") != requests
                or type(first) is not int or first <= previous
                or (rule != TIE_RULE and phase.get("recovered_points_per_request") != 1 / RECOVERY_DIVISOR)
                or (rule == POST_ANSWER_RULE and phase.get("post_answer_penalty_per_action") != 1 / RECOVERY_DIVISOR)):
            raise ValueError(f"Invalid private-symbol selection amendment in {path}")
        previous = first
    return phases


def rule_at(phases, generation):
    return next((phase["selection_rule"] for phase in reversed(phases)
                 if phase["first_generation"] <= generation), None)


def selected_score(row, requests, rule):
    strict, completion = row["strict"], row["completion"]
    if not 0. <= strict <= completion <= 1.:
        raise ValueError("Expected strict and ordinary completion rates in order")
    recovered = completion - strict
    if rule in (WEIGHTED_RULE, POST_ANSWER_RULE):
        tenths = RECOVERY_DIVISOR * strict + recovered
        if rule == POST_ANSWER_RULE:
            actions = row.get("post_answer_actions")
            if type(actions) is not int or actions < 0:
                raise ValueError("Expected a nonnegative integer post-answer action count")
            tenths -= actions / requests
        # Accumulate in tenths before dividing, preserving exact ties on the
        # fixed training tape's binary fractions.
        return tenths / RECOVERY_DIVISOR, recovered
    if rule == TIE_RULE:
        return strict + recovered / (requests + 1), recovered
    raise ValueError(f"Unknown private-symbol selection rule {rule!r}")


def score_fields(row, requests, rule):
    score, recovered = selected_score(row, requests, rule)
    return {**row, "selection_rule": rule, "selection_score": score,
            "recovered_completion_rate": recovered}


def validate_rule(rows, expected, location):
    if any(row.get("selection_rule") != expected for row in rows):
        raise ValueError(f"Selection rule differs from the scheduled phase in {location}")


def summarize(original, rows):
    result = original(rows)
    rule = rows[0].get("selection_rule")
    if rule is not None:
        result.update(selection_rule=rule,
                      recovered_completion_rate=result["completion"] - result["strict"])
    if rule == POST_ANSWER_RULE:
        winner = next(row for row in rows if row["id"] == result["best_candidate"])
        result["post_answer_actions"] = winner["post_answer_actions"]
    return result


def schedule_post_answer_penalty(out):
    """Schedule the first untouched generation while holding the experiment lock."""
    with _lock(out / ".lock"):
        previous = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        if previous.get("task") != "private_symbol":
            raise ValueError("This selection change requires a private-symbol experiment")
        requests = requests_per_candidate(previous)
        path = out / AMENDMENT
        phases = read_phases(out, requests)
        if phases and phases[-1]["selection_rule"] == POST_ANSWER_RULE:
            return phases[-1]
        started = max((int(folder.name) for folder in out.glob("pair-*/*/generations/*")
                       if folder.name.isdecimal() and (
                           (folder / "plan.json").is_file() or (folder / "summary.json").is_file()
                           or any((folder / "batches").glob("*.json.gz"))
                       )), default=0)
        first = max(started, phases[-1]["first_generation"] if phases else 0) + 1
        if first > previous["generations"]:
            raise ValueError("There is no untouched generation in this experiment")
        amendment = {
            "selection_rule": POST_ANSWER_RULE,
            "first_generation": first,
            "requests_per_candidate": requests,
            "recovered_points_per_request": 1 / RECOVERY_DIVISOR,
            "post_answer_penalty_per_action": 1 / RECOVERY_DIVISOR,
            "score": f"strict + (completion - strict - post_answer_actions / requests_per_candidate) / {RECOVERY_DIVISOR}",
            "recorded_utc": _utc(),
        }
        _write(path, {"phases": [*phases, amendment]})
        return amendment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(schedule_post_answer_penalty(args.run_dir.resolve()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
