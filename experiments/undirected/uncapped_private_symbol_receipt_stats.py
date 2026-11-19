"""World-seed paired contrasts for the private-symbol receipt intervention."""

from collections import Counter, defaultdict
from random import Random
from statistics import fmean, quantiles


QUESTIONS = ("same", "other_context", "other_raw")
DELAYS = (0, 1, 8)
PATH_ARMS = {"full": ("ack", "rej"), "no_write": ("ack_no_write", "rej_no_write")}
ARMS = tuple(arm for pair in PATH_ARMS.values() for arm in pair)
RATE_METRICS = ("repeat", "timely", "correct")
EFFECT_METRICS = (*RATE_METRICS, "truthful_correct")
BOOTSTRAP_DRAWS = 4096


def _matched_prefixes(rows):
    prefixes = defaultdict(dict)
    for row in rows:
        prefix = (row["seed"], row["champion"], row["actor"])
        cell = (row["delay"], row["key"])
        if cell in prefixes[prefix]:
            raise ValueError(f"duplicate question cell: {prefix}, {cell}")
        for field in ("previous", "target"):
            if type(row[field]) is not int or row[field] not in range(4):
                raise ValueError(f"invalid {field} symbol: {prefix}, {cell}")
        answers = row["answers"]
        if answers.keys() != set(ARMS) or any(type(answer) is not int or answer not in range(-1, 4)
                                              for answer in answers.values()):
            raise ValueError(f"invalid answer arms or symbols: {prefix}, {cell}")
        prefixes[prefix][cell] = row

    expected = {(delay, question) for delay in DELAYS for question in QUESTIONS}
    for prefix, cells in prefixes.items():
        if cells.keys() != expected:
            raise ValueError(f"incomplete or unexpected question cells: {prefix}")
        if len({row["previous"] for row in cells.values()}) != 1:
            raise ValueError(f"different teaching answer across paired cells: {prefix}")
        for question in QUESTIONS:
            if len({cells[delay, question]["target"] for delay in DELAYS}) != 1:
                raise ValueError(f"different target across delays: {prefix}, {question}")
    return prefixes


def _seed_scores(prefixes, delay):
    counts = {question: Counter() for question in QUESTIONS}
    for cells in prefixes:
        teaching = cells[delay, "same"]
        teaching_correct = teaching["previous"] == teaching["target"]
        for question in QUESTIONS:
            row = cells[delay, question]
            tally = counts[question]
            answers = row["answers"]
            for arm, answer in answers.items():
                tally[arm, "repeat"] += answer == row["previous"]
                tally[arm, "timely"] += answer != -1
                tally[arm, "correct"] += answer == row["target"]
            for path, (ack, rej) in PATH_ARMS.items():
                both_timely = answers[ack] != -1 and answers[rej] != -1
                changed = answers[ack] != answers[rej]
                tally[path, "changed"] += changed
                tally[path, "both_timely"] += both_timely
                tally[path, "switched"] += both_timely and changed
                truthful, false = (ack, rej) if teaching_correct else (rej, ack)
                tally[path, "truthful_correct"] += ((answers[truthful] == row["target"])
                                                      - (answers[false] == row["target"]))

    n = len(prefixes)
    scores = {}
    for question, tally in counts.items():
        rates = {arm: {metric: tally[arm, metric] / n for metric in RATE_METRICS} for arm in ARMS}
        effects, pairs = {}, {}
        for path, (ack, rej) in PATH_ARMS.items():
            effects[path] = {metric: (tally[ack, metric] - tally[rej, metric]) / n for metric in RATE_METRICS}
            effects[path]["truthful_correct"] = tally[path, "truthful_correct"] / n
            both_timely = tally[path, "both_timely"]
            pairs[path] = {"changed": tally[path, "changed"] / n,
                           "both_timely": both_timely / n,
                           "switch_given_both_timely": tally[path, "switched"] / both_timely if both_timely else None}
        effects["full_minus_no_write"] = {metric: effects["full"][metric] - effects["no_write"][metric]
                                         for metric in EFFECT_METRICS}
        scores[question] = {"rates": rates, "effects": effects, "pair_diagnostics": pairs}
    return scores


def _estimate(values, draws):
    mean = fmean(values)
    if draws is None:
        return mean
    if not draws:
        return {"mean": mean, "ci95": None}
    if min(values) == max(values):
        return {"mean": mean, "ci95": [mean, mean]}
    means = [fmean(values[index] for index in sample) for sample in draws]
    cutoffs = quantiles(means, n=40, method="inclusive")
    return {"mean": mean, "ci95": [cutoffs[0], cutoffs[-1]]}


def _report(scores, draws):
    paths = (*PATH_ARMS, "full_minus_no_write")
    by_question = {}
    for question in QUESTIONS:
        rates = {arm: {metric: fmean(seed[question]["rates"][arm][metric] for seed in scores)
                       for metric in RATE_METRICS} for arm in ARMS}
        effects = {path: {metric: _estimate([seed[question]["effects"][path][metric] for seed in scores], draws)
                          for metric in EFFECT_METRICS} for path in paths}
        pairs = {}
        for path in PATH_ARMS:
            seed_pairs = [seed[question]["pair_diagnostics"][path] for seed in scores]
            switched = [pair["switch_given_both_timely"] for pair in seed_pairs
                        if pair["switch_given_both_timely"] is not None]
            pairs[path] = {
                "changed": fmean(pair["changed"] for pair in seed_pairs),
                "both_timely": fmean(pair["both_timely"] for pair in seed_pairs),
                "switch_given_both_timely": fmean(switched) if switched else None,
                "world_seeds_with_both_timely": len(switched),
            }
        by_question[question] = {"rates": rates, "effects": effects, "pair_diagnostics": pairs}

    specificity = {}
    for control in QUESTIONS[1:]:
        specificity[control] = {
            path: {metric: _estimate([seed["same"]["effects"][path][metric] - seed[control]["effects"][path][metric]
                                      for seed in scores], draws) for metric in EFFECT_METRICS}
            for path in paths
        }
    return {"by_question": by_question, "specificity": specificity}


def summarize(rows):
    """Average paired answer effects across world seeds with equal seed weights."""
    prefixes = _matched_prefixes(rows)
    by_seed = defaultdict(list)
    by_champion = defaultdict(lambda: defaultdict(list))
    for (seed, champion, _actor), cells in prefixes.items():
        by_seed[seed].append(cells)
        by_champion[champion][seed].append(cells)

    seeds = sorted(by_seed)
    rng = Random(0)
    draws = ([rng.choices(range(len(seeds)), k=len(seeds)) for _ in range(BOOTSTRAP_DRAWS)]
             if len(seeds) > 1 else [])
    report = {
        "world_seeds": seeds,
        "eligible_prefixes": len(prefixes),
        "eligible_per_seed": {str(seed): len(by_seed[seed]) for seed in seeds},
        "method": {
            "unit": "World seed. Eligible champion/actor prefixes share every question, delay and receipt arm.",
            "weights": "Equal weight per eligible prefix within each seed and per seed in the summary.",
            "interval": "95% percentile bootstrap resamples matched world seeds together. A single seed yields null.",
            "bootstrap_draws": len(draws),
            "effects": "ACK minus REJ. Full minus no write subtracts the no-write contrast from the full contrast.",
            "specificity": "Same-question receipt effect minus each visible control's receipt effect.",
            "pair_diagnostics": ("Changed includes switches between timeout and answering. Conditional switch means average "
                                 "only world seeds with at least one pair answering under both receipts."),
            "descriptive_by_champion": "Equal-weight seed means without intervals, conditional on each champion.",
        },
        "metrics": {
            "repeat": "First timely answer repeats the teaching symbol. A timeout scores zero.",
            "timely": "First answer within the legal answer window, regardless of correctness.",
            "correct": "First timely answer equals this question's target. A timeout scores zero.",
            "truthful_correct": ("Correctness under truthful minus false teaching feedback. The original teaching target "
                                 "determines which feedback is truthful for every question."),
        },
        "delays": {},
    }
    if not seeds:
        return report

    for delay in DELAYS:
        scores = [_seed_scores(by_seed[seed], delay) for seed in seeds]
        delay_report = _report(scores, draws)
        delay_report["descriptive_by_champion"] = {}
        for champion, samples in by_champion.items():
            champion_scores = [_seed_scores(samples[seed], delay) for seed in sorted(samples)]
            delay_report["descriptive_by_champion"][champion] = {
                "eligible_per_seed": {str(seed): len(samples[seed]) for seed in sorted(samples)},
                **_report(champion_scores, None),
            }
        report["delays"][str(delay)] = delay_report
    return report
