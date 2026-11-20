"""Run continuous lives with six real review choices and account for every earned taste."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import local_trace_graph as native
from . import noon_readback as preview
from . import noon_readback_fit as readouts
from . import noon_review_world as world


DATA = world.DATA / "repeated-readout"
PLAN = world.EVIDENCE / "noon-review-repeated-plan.md"
SOURCE = world.DATA / "online-readout" / "643793" / "live" / "step-4000.pt"
TRAIN_DOMAIN = 0x61A12C
EVAL_DOMAIN = 0x61A12D
SEEDS = {"main": {"original": 76202531, "independent": 76202532},
         "replicate": {"original": 76202541, "independent": 76202542},
         "trained": {"original": 76202561, "independent": 76202562}}
WARM, ROUNDS = 4, 6
POLICIES = ("learned", "random", "native", "weak")
FIT_ARMS = ("later", "total")
EVENTS = ("review", "seen", "new")


@dataclass(frozen=True)
class Tape:
    odor: torch.Tensor
    law: torch.Tensor
    ordinary_orientation: torch.Tensor
    ordinary_draw: torch.Tensor
    candidates: torch.Tensor
    menu_orientation: torch.Tensor
    scan_first: torch.Tensor
    choose_draw: torch.Tensor
    review_orientation: torch.Tensor
    review_draw: torch.Tensor
    seen_item: torch.Tensor

    @property
    def lanes(self):
        return len(self.odor)


@dataclass(frozen=True)
class Menu:
    odor: torch.Tensor
    law: torch.Tensor
    candidates: torch.Tensor
    menu_orientation: torch.Tensor
    review_orientation: torch.Tensor
    review_draw: torch.Tensor

    @property
    def lanes(self):
        return len(self.odor)


def make_tape(seed, worlds, *, births=(), step=None):
    if worlds < 1 or bool(births) != (step is not None):
        raise ValueError("Requirements are positive worlds and both or neither of births/step")
    roots = ([TRAIN_DOMAIN, seed, birth, step] for birth in births) if births else ([EVAL_DOMAIN, seed],)

    def generate(entropy):
        geometry, law_rng, events, menus = (np.random.default_rng(seq) for seq in np.random.SeedSequence(entropy).spawn(4))
        objects = WARM + ROUNDS
        norm = geometry.normal(size=(worlds, objects, 3))
        odors = (1.35 * norm / np.linalg.norm(norm, axis=2, keepdims=True)).astype(np.float32)
        law = (2 * law_rng.integers(0, 2, (worlds, objects)) - 1).astype(np.float32)
        twin_laws = np.stack((law, -law), axis=1).reshape(worlds * 2, objects)
        available = WARM + np.arange(ROUNDS)
        order = menus.random((worlds, ROUNDS, objects))
        allowed = np.arange(objects)[None, None, :] < available[None, :, None]
        choices = np.argsort(np.where(allowed, order, np.inf), axis=2)[:, :, :2]
        seen = (menus.random((worlds, ROUNDS + 1)) * (WARM + np.arange(ROUNDS + 1))[None]).astype(np.int64)

        def signs(rng, shape):
            return (2 * rng.integers(0, 2, shape) - 1).astype(np.float32)

        n_ordinary = WARM + 2 * ROUNDS + 1
        width = native.EDIT_DRAWS + 1
        public = (signs(events, (worlds, n_ordinary)), events.random((worlds, n_ordinary, width), dtype=np.float32),
                  choices, signs(menus, (worlds, ROUNDS, 2)), menus.integers(0, 2, (worlds, ROUNDS)),
                  menus.random((worlds, ROUNDS), dtype=np.float32), signs(events, (worlds, ROUNDS)),
                  events.random((worlds, ROUNDS, width), dtype=np.float32), seen)
        return np.repeat(odors, 2, axis=0), twin_laws, *(np.repeat(a, 2, axis=0) for a in public)

    cells = zip(*(generate(entry) for entry in roots), strict=True)
    return Tape(*(torch.from_numpy(np.concatenate(values, axis=0)) for values in cells))


def controller(path=SOURCE):
    model = readouts.Readout("logged")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["weights"])
    if sum(value.numel() for value in model.parameters()) != 17:
        raise ValueError("Expected the 17 learned shared coefficients")
    return model


def ordinary(view, state, tape, item, event):
    rows = torch.arange(tape.lanes)
    sign = tape.ordinary_orientation[:, event]
    left, right = world.sensory_for(tape, item, sign)
    truth = tape.law[rows, item] * sign
    return world.collect(view, state, left, right, truth, tape.ordinary_draw[:, event])


def _decision(action, signals, draw, policy, model):
    if policy == "native":
        return action.side, None
    fair = torch.where(draw < .5, 1., -1.)
    if policy == "random":
        return fair, None
    if policy == "weak":
        left, right = ((signals[..., 0] - signals[..., 1]).abs() * .5).unbind(dim=1)
        return torch.where(left == right, fair, torch.where(right < left, 1., -1.)), None
    if policy != "learned" or model is None:
        raise ValueError(f"Unsupported review decision {policy}")
    chance = model(signals.double(), action.motor_logit.double())
    side = torch.where(draw < chance.detach(), 1., -1.)
    return side, chance


def run_life(graph, rule, tape, policy, model=None, *, audit_graph=False):
    lanes_per_birth = tape.lanes // len(graph.bias)
    view = rule.expand(lanes_per_birth)
    initial = native.start(graph, lanes_per_birth)
    state = initial
    warm = []
    for item in range(WARM):
        event = ordinary(view, state, tape, item, item)
        state = event.state
        warm.append(event.earned)
    receipts, selections, objects, motor, probabilities, previews = [], [], [], [], [], []
    for turn in range(ROUNDS):
        menu = Menu(tape.odor, tape.law, tape.candidates[:, turn], tape.menu_orientation[:, turn],
                    tape.review_orientation[:, turn], tape.review_draw[:, turn])
        scanned, signals = preview.scan(view, state, menu, tape.scan_first[:, turn])
        draw = tape.choose_draw[:, turn]
        emitted = world.select(view, scanned, *world.candidate_menu(menu), draw)
        chosen, chance = _decision(emitted, signals, draw, policy, model)
        state, review = world.review(view, emitted.state, menu, chosen)
        seen = ordinary(view, state, tape, tape.seen_item[:, turn], WARM + 2 * turn)
        new = ordinary(view, seen.state, tape, WARM + turn, WARM + 2 * turn + 1)
        state = new.state
        receipts.append(torch.stack((review.earned, seen.earned, new.earned), dim=1))
        motor.append(torch.stack((review.motor.side, seen.motor.side, new.motor.side), dim=1))
        selections.append(chosen)
        objects.append(world.selected_item(menu, chosen))
        if chance is not None:
            probabilities.append(chance)
        if audit_graph:
            previews.append(signals)
    final = ordinary(view, state, tape, tape.seen_item[:, -1], WARM + 2 * ROUNDS)
    output = dict(warm=torch.stack(warm, dim=1), earned=torch.stack(receipts, dim=1), final=final.earned,
                  final_side=final.motor.side, motor=torch.stack(motor, dim=1), selected_side=torch.stack(selections, dim=1),
                  selected_object=torch.stack(objects, dim=1))
    if probabilities:
        output["policy_probability_right"] = torch.stack(probabilities, dim=1)
    if audit_graph:
        end = final.state.graph
        output["preview_motor"] = torch.stack(previews, dim=1)
        output["final_graph_change"] = torch.stack(((end.weights - initial.graph.weights).abs().flatten(1).sum(1),
                                                     (end.bias - initial.graph.bias).abs().sum(1),
                                                     (end.present != initial.graph.present).flatten(1).sum(1)), dim=1)
    return output


def resources(row):
    warm = row["warm"].sum(1)
    review, seen, novelty = row["earned"].sum(1).T
    final = row["final"]
    subsequent = seen + novelty + final
    return dict(warm=warm, review=review, seen=seen, new=novelty, final_seen=final, later=subsequent,
                after_warm=review + subsequent, total=warm + review + subsequent)


def nearby_effects(data, arm, reference):
    differences = data[arm]["earned"] != data[reference]["earned"]
    sides = data[arm]["selected_side"] != data[reference]["selected_side"]
    choices = data[arm]["selected_object"] != data[reference]["selected_object"]
    next_sensory = data[arm]["preview_motor"] != data[reference]["preview_motor"]
    return dict(choice_disagreement_by_round=choices.mean(0).tolist(),
                earned_taste_disagreement_by_round=differences.any(2).mean(0).tolist(),
                later_preview_disagreement_from_initial_choice=float(
                    next_sensory[sides[:, 0], 1:].any((1, 2, 3)).mean()) if sides[:, 0].any() else None)


def evaluator(cohort, split, worlds, stage, batch):
    torch.set_num_threads(1)
    roots = world.COHORTS[cohort]
    seed = SEEDS[split][cohort]
    paths = {"learned": SOURCE}
    if stage:
        paths.update({arm: DATA / "fit" / arm / f"step-{stage}.pt" for arm in FIT_ARMS})
    models = {arm: controller(path).requires_grad_(False) for arm, path in paths.items()}
    policies = (*POLICIES, *(arm for arm in FIT_ARMS if stage))
    parent_sha256 = {birth: prior.digest(world.checkpoints.checkpoint_path(world.parent_path(cohort), "odor",
                                                                           "uniform", birth, world.PARENT_STAGE))
                      for birth in roots}
    settings = prior.BirthIntervals(list(roots), 20_000, seed)
    files, records, histories = {}, {}, {}
    tape = make_tape(seed, worlds)
    root = DATA / "evaluation" / split / f"{stage:04d}" / cohort
    tape_path = root / f"tape-n{worlds}.npz"
    archive._write_raw(tape_path, **{name: value.numpy() for name, value in vars(tape).items()})
    files[tape_path.relative_to(DATA).as_posix()] = prior.digest(tape_path)
    for birth in roots:
        graph, rule = world.load_parent(cohort, birth)
        whole = {}
        for policy in policies:
            fragments = []
            for lo in range(0, tape.lanes, 2 * batch):
                cut = slice(lo, lo + 2 * batch)
                subset = Tape(**{name: value[cut] for name, value in vars(tape).items()})
                with torch.no_grad():
                    raw = run_life(graph, rule, subset, "learned" if policy in models else policy,
                                   models.get(policy), audit_graph=True)
                fragments.append({name: value.numpy() for name, value in raw.items()})
            whole[policy] = {name: np.concatenate([part[name] for part in fragments]) for name in fragments[0]}
            raw = whole[policy]
            if not all((raw[key] == -1).sum() + (raw[key] == 1).sum() == raw[key].size for key in ("warm", "earned", "final")):
                raise AssertionError("Lost a physically earned taste from the life ledger")
            if policy != "learned" and not np.array_equal(raw["warm"], whole["learned"]["warm"]):
                raise AssertionError("Policies diverged before the first available decision")
            file = root / f"{policy}-{birth}-n{worlds}.npz"
            archive._write_raw(file, **raw)
            files[file.relative_to(DATA).as_posix()] = prior.digest(file)
            records.setdefault(policy, []).append({name: float(value.mean()) for name, value in resources(raw).items()})
        histories[birth] = {policy: nearby_effects(whole, policy, "learned") for policy in policies if policy != "learned"}
        print(split, cohort, birth, {name: round(rows[-1]["total"], 4) for name, rows in records.items()}, flush=True)
    columns = {name: {key: np.array([row[key] for row in rows]) for key in rows[0]} for name, rows in records.items()}
    estimates = {policy: {key: settings.report(value) for key, value in results.items()} for policy, results in columns.items()}
    contrasts = [("learned", policy) for policy in POLICIES if policy != "learned"]
    if stage:
        contrasts.extend((arm, ref) for arm in FIT_ARMS for ref in ("learned", "random"))
        contrasts.append(("total", "later"))
    difference = {f"{left}_minus_{right}": {key: settings.report(value - columns[right][key])
                                          for key, value in columns[left].items()} for left, right in contrasts}
    report = dict(split=split, seed=seed, cohort=cohort, worlds=worlds, batch_worlds=batch,
                  roots=roots, stage=stage, rounds=ROUNDS,
                  real_contacts_per_life=WARM + ROUNDS * 3 + 1, phases=EVENTS, rates=estimates, differences=difference,
                  trajectories_diverge=histories, raw_sha256=files, parent_sha256=parent_sha256,
                  model_sha256={name: prior.digest(file) for name, file in paths.items()},
                  plan_sha256=prior.digest(PLAN), source_sha256=prior.digest(Path(__file__)))
    output = world.EVIDENCE / f"noon-review-repeated-{split}-{stage:04d}-{cohort}-n{worlds}.json"
    archive._write_json(output, report)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=SEEDS, required=True)
    p.add_argument("--stage", choices=(0, 800, 1600), type=int, default=0)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--cohorts", choices=world.COHORTS, nargs="+", default=list(world.COHORTS))
    a = p.parse_args()
    if not a.batch > 0 or (a.split == "trained") != bool(a.stage):
        raise ValueError("Select a positive batch and specify stage only for the trained split")
    with ProcessPoolExecutor(max_workers=len(a.cohorts)) as pool:
        jobs = [pool.submit(evaluator, name, a.split, a.worlds, a.stage, a.batch) for name in a.cohorts]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
