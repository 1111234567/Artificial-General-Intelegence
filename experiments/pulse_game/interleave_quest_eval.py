"""Independent native serial and physical same-snapshot tests of terminal quest endpoints."""

from argparse import ArgumentParser
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_eval as physical, interleave_quest_train as quest
from . import interleave_train as matched, self_param_writer as writer


PROFILES = ("familiar", "independent_odors", "new_layout", "novel_marks")
HISTORIES = ("aba", "cba", "flipped", "prime_zero", "prime_donor")
STAGES = (0, 500, 1000, 2000, 4000, 8000)
NATIVE_LABELS = ("A", "B", "mark", "both", "both_modalities")


def _frozen(run, arm, step, births):
    parts = [matched.load_checkpoint(run, arm, birth, step) for birth in births]
    graph = writer.Graph(*(torch.cat([getattr(g, field.name) for g, _ in parts]) for field in fields(writer.Graph)))
    rule = quest.local.Rule(births)
    states = [one.state_dict() for _, one in parts]
    rule.load_state_dict({key: torch.cat([state[key] for state in states]) for key in states[0]})
    return graph, rule.eval().requires_grad_(False)


@torch.no_grad()
def native_reference(tape, mode):
    """Use only the visible menu and tastes on selected contacts."""
    if mode not in ("uniform", "constant_left", "public_lookup"):
        raise ValueError(f"Unknown native reference {mode!r}")
    left, right, rewards, draws, ticks = [value.flatten(0, 2).numpy() for value in (
        tape.left, tape.right, tape.resources, tape.draws, tape.contact_tick)]
    shape = (*tape.shape, quest.CONTACTS)
    batch = len(left)
    choices = (draws[..., 0] < .5).astype(np.int8) if mode != "constant_left" else np.zeros(left.shape[:2], np.int8)
    resource = np.empty(left.shape[:2], np.float32)
    elapsed = np.empty(left.shape[:2], np.int8)
    pairs, items = [{} for _ in range(batch)], [{} for _ in range(batch)]

    def key(side):
        return np.where(side == 0, 0., side).tobytes()

    for slot in range(left.shape[1]):
        if mode == "public_lookup":
            for row, (l_view, r_view) in enumerate(zip(left[:, slot], right[:, slot], strict=True)):
                l, r = key(l_view), key(r_view)
                pair = tuple(sorted((l, r)))
                good = pairs[row].get(pair)
                if good is not None:
                    choices[row, slot] = int(good == r)
                elif items[row].get(l, 0) != items[row].get(r, 0):
                    choices[row, slot] = int(items[row].get(l, 0) < items[row].get(r, 0))
                if slot < quest.SUPPORT:
                    choice = choices[row, slot]
                    taste = int(rewards[row, slot, choice])
                    if l == r:
                        items[row][l] = taste
                    else:
                        pairs[row][pair] = (l, r)[choice if taste > 0 else 1 - choice]
        rows = np.arange(batch)
        resource[:, slot] = rewards[rows, slot, choices[:, slot]]
        elapsed[:, slot] = ticks[rows, slot, choices[:, slot]]
    taste = resource.copy()
    taste[:, quest.SUPPORT:] = 0.
    return quest.Experience(torch.from_numpy(resource.reshape(shape)), torch.from_numpy(taste.reshape(shape)),
                            torch.from_numpy((2 * choices - 1).reshape(shape)), torch.from_numpy(elapsed.reshape(shape)),
                            torch.zeros(shape), torch.zeros(shape))


def native_payload(experience, tape):
    if experience.taste[..., quest.SUPPORT:].count_nonzero() or (experience.tick < 0).any():
        raise AssertionError("Native quest scoring must observe full motion and no terminal taste")
    a, b, mark = quest.achieved(experience, tape)
    data = {field.name: getattr(experience, field.name).numpy() for field in fields(quest.Experience)}
    data.update({f"success/{k}": v.numpy() for k, v in zip(NATIVE_LABELS, (a, b, mark, a & b, a & b & mark), strict=True)})
    return {**data, "order": tape.order.numpy(), "law": tape.law.numpy()}


def _native_stats(payload, births):
    data = payload["law"]
    conflict = data[..., 0] != data[..., 1]
    result = {}
    for label in NATIVE_LABELS:
        values = payload[f"success/{label}"]
        stats = {}
        for stratum, mask in (("all", np.ones_like(conflict)), ("conflict", conflict), ("agree", ~conflict)):
            per_birth = {}
            for i, birth in enumerate(births):
                laws = data[i].reshape(-1, 3)
                used = mask[i].ravel()
                observed = values[i].ravel()
                by_law = {"".join("+" if sign > 0 else "-" for sign in law): float(observed[(laws == law).all(1)].mean())
                          for law in np.unique(laws[used], axis=0)}
                per_birth[str(birth)] = dict(mean=float(observed[used].mean()), worst=min(by_law.values()), by_law=by_law)
            stats[stratum] = dict(mean=float(np.mean([entry["mean"] for entry in per_birth.values()])),
                                  worst_within_birth=min(entry["worst"] for entry in per_birth.values()), by_birth=per_birth)
        result[label] = stats
    return result


def native_effect(left, right, label, *, conflict=None):
    if not np.array_equal(left["law"], right["law"]) or not np.array_equal(left["order"], right["order"]):
        raise ValueError("Terminal objectives must share the exact novel worlds and layouts")
    laws = left["law"][0, 0]
    keep = np.ones(len(laws), bool) if conflict is None else (laws[:, 0] != laws[:, 1]) == conflict
    differences = (left[f"success/{label}"].astype(float) - right[f"success/{label}"])[..., keep].mean(axis=2)
    n, w = differences.shape
    rng = np.random.default_rng([quest.VALIDATION_DOMAIN, 317])
    samples = 2500
    which_birth = rng.integers(n, size=(samples, n))
    which_world = rng.integers(w, size=(samples, w))
    estimate = differences[which_birth[:, :, None], which_world[:, None, :]].mean((1, 2))
    return dict(mean=float(differences.mean()), pointwise_95=np.quantile(estimate, [.025, .975]).tolist())


@torch.no_grad()
def evaluate_native(run, raw, config, births, endpoint, worlds):
    setup = {
        "development": (quest.DEV_DOMAIN, False, min(worlds, 48)),
        "confirmation": (quest.VALIDATION_DOMAIN, True, worlds),
    }
    reports = {}
    for split, (domain, novel, count) in setup.items():
        tape = quest.make_tape(births, replace(config, worlds=count), 0, domain=domain, novel=novel, shared=True)
        stages = tuple(step for step in STAGES if step < endpoint) + (endpoint,) if split == "development" else (0, endpoint)
        stats, results = {}, {}
        for mode in ("uniform", "constant_left", "public_lookup"):
            payload = native_payload(native_reference(tape, mode), tape)
            if mode == "public_lookup" and not payload["success/both_modalities"].all():
                raise AssertionError("Native observation-based solver could not complete the public quest")
            physical._atomic_npz(raw / "native" / split / f"{mode}.npz", payload)
            stats[mode] = _native_stats(payload, births)
        for step in stages:
            stage = stats.setdefault(str(step), {})
            for arm in quest.ARMS:
                graph, rule = _frozen(run, arm, step, births)
                payload = native_payload(quest.play(graph, rule, tape, config.max_wait), tape)
                physical._atomic_npz(raw / "native" / split / f"{arm}-{step:06d}.npz", payload)
                stage[arm] = _native_stats(payload, births)
                if step in (0, endpoint):
                    results[(step, arm)] = payload
        for field in fields(quest.Experience):
            if not np.array_equal(results[(0, quest.ARMS[0])][field.name], results[(0, quest.ARMS[1])][field.name]):
                raise AssertionError("Terminal objective arms had different initialization trajectories")
        left, right = (results[(endpoint, arm)] for arm in quest.ARMS)
        stats["conjunction_minus_mean"] = {label: {stratum: native_effect(left, right, label, conflict=flag)
                                                   for stratum, flag in (("all", None), ("conflict", True), ("agree", False))}
                                           for label in NATIVE_LABELS}
        reports[split] = stats
        print("native evaluation", split, flush=True)
    return reports


def _physical_effect(raw, profiles, histories, births):
    results = {}
    for profile in profiles:
        for history in histories:
            rows = []
            for model in ("conjunction", "mean"):
                entries = []
                for birth in births:
                    with np.load(raw / profile / history / f"{model}-{birth}.npz", allow_pickle=False) as archive:
                        keep = ("roots", "maps", "mark_law", *(f"success/{key}" for key in NATIVE_LABELS))
                        entries.append({"birth": birth, **{name: archive[name] for name in keep}})
                rows.append(entries)
            l, r = rows
            masks = l[0]["maps"]
            conflict = np.any(masks[:, physical.A] != masks[:, physical.B], axis=1)
            results.setdefault(profile, {})[history] = {
                label: {stratum: physical._comparison(l, r, f"success/{label}", 2500, keep)
                        for stratum, keep in (("all", None), ("conflict", conflict), ("agree", ~conflict))}
                for label in NATIVE_LABELS}
    return results


@torch.no_grad()
def evaluate_physical(run, raw, births, endpoint, worlds):
    roots = tuple(2_300_000_000 + int(n) for n in np.random.default_rng([quest.PHYSICAL_DOMAIN, 2_300_000_000]).choice(
        100_000_000, worlds, replace=False))
    specs, graphs = [], {}
    for mode in ("uniform", "constant_left", "public_lookup"):
        specs.append(dict(name=mode, birth=0))
    for name, arm, step in (("parent", quest.ARMS[0], 0), *( (arm, arm, endpoint) for arm in quest.ARMS)):
        for birth in births:
            model = name, birth
            graphs[model] = matched.load_checkpoint(run, arm, birth, step)
            specs.append(dict(name=name, birth=birth))
    plan = dict(split="quest_confirmation", rounds_per_context_pair=4, roots=roots, profiles=PROFILES,
                cases=HISTORIES, specs=specs, domain=quest.PHYSICAL_DOMAIN, donor_domain=quest.PHYSICAL_DONOR_DOMAIN)
    physical._atomic_json(raw / "protocol.json", plan)
    for history in HISTORIES:
        for profile in PROFILES:
            arena = physical.make_arena(roots, profile, history, domain=quest.PHYSICAL_DOMAIN,
                                        donor_domain=quest.PHYSICAL_DONOR_DOMAIN)
            for spec in specs:
                name, birth = spec["name"], spec["birth"]
                key = (name, birth)
                target = raw / profile / history / f"{name}-{birth}.npz"
                if target.exists():
                    continue
                world_for_actor = replace(arena, motor_draw=np.ones_like(arena.motor_draw)) if name == "constant_left" else arena
                if key in graphs:
                    actor = physical._GraphPolicy(*graphs[key], world_for_actor)
                else:
                    actor = physical._PublicPolicy(world_for_actor.motor_draw, learn=name == "public_lookup")
                measurement, _ = physical.rollout(world_for_actor, actor)
                data = physical.payload(world_for_actor, measurement)
                if name == "public_lookup" and history == "aba" and not data["success/both_modalities"].all():
                    raise AssertionError("A public-observation physical solver must solve every ABA quest")
                physical._atomic_npz(target, data)
            print("quest physical", profile, history, flush=True)
    result = physical.summarize(raw, plan)
    result["conjunction_minus_mean"] = _physical_effect(raw, PROFILES, HISTORIES, births)
    return result


def report(run, raw, evidence, *, worlds=96):
    torch.set_num_threads(1)
    fit = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = quest.Config(**fit["config"])
    births = tuple(fit["births"])
    protocol = dict(births=births, endpoint=config.updates, worlds=worlds, domains=fit["domains"],
                    fit_sha256=physical._file_digest(run / "run.json"),
                    checkpoint_sha256={arm: physical._file_digest(matched.checkpoint_path(run, arm, config.updates))
                                       for arm in quest.ARMS},
                    source_sha256={Path(name).name: physical._file_digest(Path(name))
                                   for name in (__file__, quest.__file__, physical.__file__)},
                    measures="Native terminal actions keep state through three serial signed-resource selections, "
                    "while taste remains absent. Novel layouts start with two marks, an observable order reserved "
                    "for confirmation, and increase "
                    "physical endpoint variation. Physical confirmation uses different roots, causal matched "
                    "histories and same-state A/B/mark forks. Both modalities means all three actual contacts are correct.",
                    analysis="CIs are pointwise crossed bootstrap, with full private law factorial retained per "
                    "world. Development has no authority to select the fixed final checkpoint.")
    physical._atomic_json(evidence / "protocol.json", protocol)
    native = evaluate_native(run, raw, config, births, config.updates, worlds)
    physical._atomic_json(evidence / "native-summary.json", native)
    body = evaluate_physical(run, raw / "physical", births, config.updates, worlds)
    physical._atomic_json(raw / "physical-summary.json", body)
    concise = physical.compact_report(body)
    concise["conjunction_minus_mean"] = body["conjunction_minus_mean"]
    physical._atomic_json(evidence / "physical-summary.json", concise)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=quest.HOME, type=Path)
    parser.add_argument("--raw", default=quest.HOME / "heldout", type=Path)
    parser.add_argument("--evidence", default=quest.EVIDENCE, type=Path)
    parser.add_argument("--worlds", default=96, type=int)
    args = parser.parse_args(argv)
    report(args.run, args.raw, args.evidence, worlds=args.worlds)


if __name__ == "__main__":
    main()
