"""Test visibility using the exact public receipt history collected by each graph."""

from argparse import ArgumentParser
from dataclasses import replace
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import graph_interactions as model, graph_interactions_study as study, mixture_world as food, world
from .unified_live_graph_study import Body, digest, read, write


def probe(menu, ridge, ties):
    tail = slice(-study.TAIL, None)
    fields = ("ctx", "cue", "left_id", "good_id", "distances", "wind")
    tape = replace(menu.body, **{name: getattr(menu.body, name)[:, tail] for name in fields})
    fresh = replace(menu, body=tape, chemicals=menu.chemicals[:, tail],
                    probability=menu.probability[:, tail], realized=menu.realized[:, tail])
    body = Body(fresh, None, study.TAIL)
    observed = body.observe()
    for trial in range(study.TAIL):
        right = ridge.choose(observed, ties[:, trial], explore=False)
        pulses = np.column_stack((~right, right))
        for _ in range(world.TICKS):
            observed = body.step(pulses)
    result = body.result()
    assert not np.any(result.observed_reward)
    reached = result.contact_side
    hit = reached >= 0
    expected = np.take_along_axis(fresh.probability, reached.clip(min=0)[..., None], axis=2)[..., 0] * hit
    return dict(accuracy=float((reached == fresh.probability.argmax(axis=2)).mean()),
                contact=float(hit.mean()), expected_taste=float(expected.mean()))


def report(out):
    study.require(out)
    summary = read(out / "summary.json")
    scores = {}
    rows = np.arange(len(study.TEST_BIRTHS) * len(study.TEST_WORLDS))
    for mode in study.MODES:
        with ZipFile(out / f"{mode}-raw.zip") as source:
            contents = source.read(f"{mode}.npz")
        if hashlib.sha256(contents).hexdigest() != summary["outcomes"][mode]["npz_sha256"]:
            raise RuntimeError("Unexpected confirmation arrays")
        with np.load(BytesIO(contents), allow_pickle=False) as raw:
            menu, _, noise, _ = model.make_menu(raw["worlds"], mode, study.TRIALS)
            food_match = menu.chemicals[:len(study.TEST_WORLDS)]
            np.testing.assert_array_equal(food_match, raw["raw_odors"])
            public = np.zeros((len(rows), world.OBS_SIZE), dtype=np.float32)
            receipt = np.zeros_like(public)
            scores[mode] = {}
            for arm in ("tanh", "square"):
                contact, shown = raw[f"{arm}/contact"], raw[f"{arm}/shown"]
                learners = {name: model.PublicRidge(len(rows), quadratic=name == "polynomial")
                            for name in ("linear", "polynomial")}
                learned = 0
                for trial in range(study.TRIALS - study.TAIL):
                    reached = contact[:, trial]
                    tasted = shown[:, trial]
                    heard = (reached >= 0) & (tasted != 0)
                    learned += int(heard.sum())
                    public[:, food.LEFT.start:food.RIGHT.stop] = menu.chemicals[:, trial].reshape(len(rows), -1)
                    receipt.fill(0)
                    receipt[rows[heard], world.CONTACT_ID.start + reached[heard]] = 1.
                    receipt[:, world.REWARD_POS] = tasted == 1
                    receipt[:, world.REWARD_NEG] = tasted == -1
                    for learner in learners.values():
                        learner.choose(public, noise[:, trial], explore=False)
                        learner.learn(receipt)
                scores[mode][arm] = dict(
                    training_receipts=learned,
                    methods={name: probe(menu, learner, noise[:, -study.TAIL:]) for name, learner in learners.items()},
                )
    write(out / "same-receipts-replay.json", dict(
        stage="descriptive post-confirmation control",
        source_sha256=digest(Path(__file__)), selection_source=dict(
            graph=digest(Path(model.__file__)), study=digest(Path(study.__file__)),
            summary=digest(out / "summary.json")),
        procedure="For each graph, train ridge controls exclusively on public raw mixtures and the one tasted side/sign actually collected by that graph. Measure these controls on the same novel silent tail with their own real physical pulses. No private taste used in fitting.",
        results=scores,
    ), exclusive=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    report(parser.parse_args().out)


if __name__ == "__main__":
    main()
