"""Replicate the frozen quest objective comparison on independent births and worlds."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path

import numpy as np

from . import interleave_eval as physical, interleave_quest_eval as evaluation, interleave_quest_train as quest


RUN = physical.RAW / "terminal-quest-replication"
EVIDENCE = physical.EVIDENCE / "terminal-quest-replication"
PARENT = physical.RAW / "joint-continuation-replication/full-direct"
BIRTHS = (811, 812, 813, 814, 815, 816)
DOMAINS = tuple(range(0x7EA230, 0x7EA235))
CONFIG = replace(quest.Config(), seed=971723)
PROFILES = ("familiar", "independent_odors", "new_layout")
HISTORIES = ("aba", "cba", "flipped")
WORLD_COUNT = 96


def _physical_roots(domain):
    start = 2_300_000_000
    generator = np.random.default_rng([domain, start])
    return tuple(start + int(root) for root in generator.choice(100_000_000, WORLD_COUNT, replace=False))


def registration():
    domains = (quest.TRAIN_DOMAIN, quest.DEV_DOMAIN, quest.VALIDATION_DOMAIN,
               quest.PHYSICAL_DOMAIN, quest.PHYSICAL_DONOR_DOMAIN)
    independent = {d for split in ("main", "replication") for d in physical.world_domains(split)}
    if set(DOMAINS) & {*domains, *independent}:
        raise ValueError("Replication and original episode domains must be disjoint")
    roots = _physical_roots(DOMAINS[3])
    if set(roots) & set(_physical_roots(domains[3])):
        raise ValueError("Quest physical environment roots must be disjoint")
    plan = dict(parent=str(PARENT / "local_active/step-008000.pt"), births=BIRTHS, objectives=quest.ARMS,
                config=asdict(CONFIG), domains=dict(zip(("train", "development", "native_confirmation", "physical", "donor"),
                                                       DOMAINS, strict=True)),
                worlds=WORLD_COUNT, physical_roots=roots, physical_profiles=PROFILES, physical_histories=HISTORIES,
                protocol="Same architecture, matched optimizer budget, signed support and silent terminal contact. "
                         "Native confirmation uses observable novel support order. Primary physical rate is "
                         "A+B+mark on conflict ABA from one snapshot, with paired mean, constant-left, uniform and "
                         "public-observation references. CBA and flipped-A histories test use of earlier taste.",
                source_sha256={Path(name).name: physical._file_digest(Path(name))
                               for name in (__file__, quest.__file__, evaluation.__file__)})
    path = EVIDENCE / "registration.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan)):
        raise ValueError("The replication launcher differs from its sealed registration")
    physical._atomic_json(path, plan)


def _configure_separate_process():
    # Reuse the original trainers and scorers. The default episode domain is
    # bound at function definition, so set it explicitly on every omitted draw.
    (quest.TRAIN_DOMAIN, quest.DEV_DOMAIN, quest.VALIDATION_DOMAIN,
     quest.PHYSICAL_DOMAIN, quest.PHYSICAL_DONOR_DOMAIN) = DOMAINS
    quest.make_tape = partial(quest.make_tape, domain=DOMAINS[0])
    evaluation.PROFILES = PROFILES
    evaluation.HISTORIES = HISTORIES


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "fit", "evaluate"))
    args = parser.parse_args(argv)
    registration()
    if args.phase == "register":
        return
    _configure_separate_process()
    if args.phase == "fit":
        quest.run(RUN, PARENT, BIRTHS, CONFIG, workers=1)
    else:
        evaluation.report(RUN, RUN / "heldout", EVIDENCE, worlds=WORLD_COUNT)


if __name__ == "__main__":
    main()
