"""Closed-loop Scout success indexed by the learner's previously earned receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_scout_model import ScoutRawAgent, load_scout_model
from .meta_scout_train import make_batch
from .scout_world import rollout_scout


def exposure_counts(tape, result):
    shape = tape.cue.shape
    visits = np.zeros((tape.n, 4), dtype=np.int32)
    by_context = np.zeros((tape.n, 4, 2), dtype=np.int32)
    prior = np.empty(shape, dtype=np.int32)
    prior_context = prior.copy()
    prior_other_context = prior.copy()
    obs = result.obs
    receipt = (obs[..., 13:15].sum(axis=-1) > 0) & (obs[..., world.CONTACT_ID].sum(axis=-1) > 0)
    has_receipt = receipt.any(axis=2)
    observed = result.observed_context.max(axis=2)
    rows = np.arange(tape.n)
    for trial in range(tape.trials):
        cue = tape.cue[:, trial]
        ctx = tape.ctx[:, trial]
        seen = observed[:, trial]
        prior[:, trial] = visits[rows, cue]
        prior_context[:, trial] = by_context[rows, cue, ctx]
        prior_other_context[:, trial] = by_context[rows, cue, 1 - ctx]
        signed = has_receipt[:, trial]
        visits[rows[signed], cue[signed]] += 1
        tagged = signed & (seen >= 0)
        by_context[rows[tagged], cue[tagged], seen[tagged]] += 1
    sensitive = result.sensitive_trial
    mandatory = sensitive & (tape.scout_side != (tape.left_id != tape.good_id))
    direct_path = ~sensitive & (tape.scout_side != (tape.left_id != tape.good_id))
    masks = {"cue_first_contact": prior == 0, "one_cue_receipt": prior == 1,
             "known_same_context_only": (prior_context > 0) & (prior_other_context == 0),
             "only_other_context_seen": (prior_context == 0) & (prior_other_context > 0),
             "one_receipt_per_context": (prior_context == 1) & (prior_other_context == 1),
             "both_contexts_observed": (prior_context > 0) & (prior_other_context > 0)}

    def group(mask, outcome):
        return {"n": int(mask.sum()), "correct": int((mask & outcome).sum())}

    return {name: {"overall": group(mask, result.correct),
                   "sensitive": group(mask & sensitive, result.correct),
                   "insensitive": group(mask & ~sensitive, result.correct),
                   "must_reverse": group(mask & mandatory, result.correct & result.reversed),
                   "direct_possible": group(mask & direct_path, result.correct & ~result.probed)}
            for name, mask in masks.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_340_800_000)
    parser.add_argument("--lives", type=int, default=512)
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    tape = make_batch(args.seed, args.lives, args.trials)
    result = rollout_scout(ScoutRawAgent(load_scout_model(args.checkpoint)), tape, record=True)
    data = {"checkpoint": str(args.checkpoint), "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "seeds": list(tape.seeds), "exposure_groups": exposure_counts(tape, result)}
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["exposure_groups"]))
