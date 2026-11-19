"""Localize and transplant a receipt-responsive circuit in the frozen IID-1024 archive."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _utc, _write
from . import uncapped_directed_function_geometry_audit as geometry
from . import uncapped_private_symbol_iid_run as run
from .evolved_plasticity import ACTIONS, INPUTS, WEIGHT_LIMIT
from .evolved_plasticity_world import CONTEXT, FIRST_FINISH, N_ACTIONS, RAW, READY, _receipt_table
from .uncapped_directed_reward import _DirectedRewardPolicy, _stack_directed_genomes
from .uncapped_private_symbol_receipt_probe import batch_tape, clear_receipts, restore, snapshot, teaching, test_keys
from .uncapped_private_symbol_receipt_world import native_inputs
from .uncapped_reward_precision import _ACK, _REJ


BRANCH = run.OUT / "pair-01" / "downhill"
OUTPUT = run.OUT / "audits" / "iid-1024-module-transfer-2026-11-19.json"
SCREEN = tuple(range(97_422_310, 97_422_314))
LOCALIZE = tuple(range(97_422_410, 97_422_414))
HOLDOUT = tuple(range(97_422_510, 97_422_518))
PORTS, NODES, SRC, DST = geometry.PORTS, geometry.NODES, geometry.SRC, geometry.DST
SOURCE = 32
THRESHOLD = .05


class ProbePolicy(_DirectedRewardPolicy):
    def __init__(self, birth, actors, cuts):
        self.cut = cuts
        self.log = None
        self.hidden = None
        self.matrix_dirty = True
        super().__init__(birth, actors, torch.device("cpu"), "full")

    def _update_matrix(self):
        if not self.matrix_dirty:
            return
        super()._update_matrix()
        if self.cut is not None:
            self.matrix.flatten(-2).masked_fill_(self.cut, 0.)
        self.matrix_dirty = False

    def _change_weights(self, coactivity, event):
        if event.any():
            super()._change_weights(coactivity, event)
            self.matrix_dirty = True
        else:
            # As in the sealed traffic-free path, eligibility still decays.
            self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)

    def _change_support(self, coactivity, difference, receipt):
        super()._change_support(coactivity, difference, receipt)
        self.matrix_dirty = True

    def act(self, obs):
        actions = super().act(obs)
        if self.log is not None:
            self.log.append((actions[..., 0], obs[..., READY].bool()))
        if self.hidden is not None:
            self.hidden.append(self.activity[..., PORTS:].float())
        return actions


def _cut_tensor(cases, cuts):
    if not cuts:
        return None
    mask = torch.zeros((len(cases), 1, NODES * NODES), dtype=torch.bool)
    for i, name in enumerate(cases):
        loci = cuts.get(name)
        if loci is not None:
            mask[i, 0, DST[loci] * NODES + SRC[loci]] = True
    return mask


def _queries(tape, source):
    raw, context = tape.raw[source], tape.context[source]
    return {"same": (raw, context), "other_context": (raw, 1 - context),
            "other_raw": ((raw + 1) % run.N_SYMBOLS, context)}


@torch.no_grad()
def measure(cases, seeds, *, source=SOURCE, cuts=None, trace=False):
    tape = batch_tape(seeds, torch.device("cpu"))
    _, birth = _stack_directed_genomes(list(cases.values()))
    policy = ProbePolicy(birth, tape.raw.shape[1], _cut_tensor(cases, cuts))
    before = []
    policy.log = []
    obs, actions = native_inputs(policy, tape, _receipt_table("cpu").half(), source,
                                 lambda: before.append(snapshot(policy)))
    native, ready = (torch.stack(items) for items in zip(*policy.log, strict=True))
    policy.log = None
    legal = native.ge(FIRST_FINISH) & native.lt(N_ACTIONS)
    by_request = (legal & ready).reshape(-1, obs.size(1), len(cases), tape.raw.size(1))
    covered = by_request.any(1)[:source]
    early = (legal.reshape_as(by_request)[:source] & tape.stage[:source, :, None].eq(1)).any(1)
    native_off = (legal & ~ready).float().mean(0)
    first_covered, first_tick = (actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS) & obs[..., READY].bool()).max(1)
    pulse = first_tick[0] + 1
    eligible = first_covered[0] & pulse.lt(obs.size(1))
    previous = (actions[0].gather(0, first_tick[0][None])[0] - FIRST_FINISH) % run.N_SYMBOLS
    keys = _queries(tape, source)
    targets = {name: tape.maps[torch.arange(tape.raw.size(1)), context, raw] for name, (raw, context) in keys.items()}
    truth = previous.eq(targets["same"])
    receipt = pulse.clamp_max(obs.size(1) - 1)[None]
    ack, rej = (obs[0, ..., status].gather(0, receipt)[0].bool() for status in (_ACK, _REJ))
    if ((~(ack ^ rej) | (ack != truth)) & eligible).any():
        raise RuntimeError("native ACK/REJ does not match its eliciting answer")
    delays = (0, 1, 8) if "graft" in cases else (0, 1)
    matched = None
    if "graft" in cases:
        recipient_family = [i for i, name in enumerate(cases) if name != "donor"]
        eliciting_symbol = previous[recipient_family]
        arrival = pulse[recipient_family]
        matched = (eligible[recipient_family] & eliciting_symbol.eq(eliciting_symbol[0]) & arrival.eq(arrival[0])).all(0)
    clear_receipts(obs[1, 0], torch.ones_like(eligible))
    updates = torch.arange(obs.size(1))[None, :, None, None].eq(first_tick[:, None]) & first_covered[:, None]
    behavior, hidden = {}, {}
    for mode in ("ack", "rej"):
        restore(policy, before[0])
        policy.hidden = [] if trace else None
        teaching(policy, obs[0], actions[0], pulse, eligible, mode, False)
        if trace:
            hidden[mode] = torch.stack(policy.hidden)
        policy.hidden = None
        for delay in range(max(delays) + 1):
            if delay in delays:
                behavior[mode, delay] = test_keys(policy, tape.public[source + 9], keys)
            if delay == max(delays):
                break
            for i, frame in enumerate(obs[delay + 1]):
                if i:
                    policy.pending.copy_(updates[delay + 1, i - 1])
                elif delay:
                    policy.pending.copy_(updates[delay, -1])
                else:
                    policy.pending.zero_()
                policy.act(frame)

    rows = {int(seed): {} for seed in seeds}
    for world, seed in enumerate(seeds):
        sl = slice(world * run.ACTORS, (world + 1) * run.ACTORS)
        for variant, name in enumerate(cases):
            use = eligible[variant, sl]
            count = int(use.sum())
            row = {"events": count, "ready_coverage": float(covered[:, variant, sl].float().mean()),
                   "off_ready_finish_per_tick": float(native_off[variant, sl].mean()),
                   "early_finish_per_request": float(early[:, variant, sl].float().mean())}
            common = matched[sl] if matched is not None and name != "donor" else None
            row["matched_events"] = None if common is None else int(common.sum())
            for delay in delays:
                for key, target in targets.items():
                    av, rv = (behavior[mode, delay][key][variant, sl] for mode in ("ack", "rej"))
                    prior, target = previous[variant, sl], target[sl]
                    factual = torch.where(truth[variant, sl], av, rv)
                    inverse = torch.where(truth[variant, sl], rv, av)
                    for field, values in {
                        "switch": av.ne(rv), "repeat": av.eq(prior).float() - rv.eq(prior).float(),
                        "benefit": factual.eq(target).float() - inverse.eq(target).float(),
                        "correct": factual.eq(target), "ready": (av.ge(0).float() + rv.ge(0).float()) / 2.,
                    }.items():
                        row[f"d{delay}_{key}_{field}"] = float(values[use].float().mean()) if count else None
                        to_compare = row["matched_events"]
                        row[f"matched_d{delay}_{key}_{field}"] = float(values[common].float().mean()) if to_compare else None
                for field in ("repeat", "benefit"):
                    a, b = (row[f"d{delay}_{key}_{field}"] for key in ("same", "other_context"))
                    row[f"d{delay}_specific_{field}"] = a - b if count else None
                    a, b = (row[f"matched_d{delay}_{key}_{field}"] for key in ("same", "other_context"))
                    row[f"matched_d{delay}_specific_{field}"] = a - b if row["matched_events"] else None
            rows[seed][name] = row
    activity = {}
    if trace:
        diff = hidden["ack"] - hidden["rej"]
        pulse_or_after = torch.arange(obs.size(1))[:, None, None].ge(pulse) & eligible
        rms = ((diff.square() * pulse_or_after[..., None]).sum((0, 2))
               / pulse_or_after.sum((0, 2))[:, None].clamp_min(1)).sqrt()
        activity = {name: values for name, values in zip(cases, rms, strict=True)}
    return rows, activity


def means(rows):
    results = {}
    for variant in next(iter(rows.values())):
        columns = [row[variant] for row in rows.values()]
        results[variant] = {key: float(np.mean(available)) if available else None
                            for key in columns[0] if (available := [row[key] for row in columns if row[key] is not None])}
    return results


def choose_pair(screen):
    fields = ("d0_same_benefit", "d0_specific_benefit", "d0_same_repeat", "d0_same_switch")
    pairs = []
    for island in range(4):
        names = (f"champion_{island}", f"runner_up_{island}")
        for donor, recipient in (names, names[::-1]):
            a, b = screen[donor], screen[recipient]
            if min(a["events"], b["events"]) < 2 or any(field not in a or field not in b for field in fields):
                continue
            beneficial = a[fields[0]] >= THRESHOLD and a[fields[1]] >= THRESHOLD and a[fields[0]] > b[fields[0]]
            repeat_gap = abs(a[fields[2]]) - abs(b[fields[2]])
            pairs.append(((beneficial, a[fields[0]] - b[fields[0]] if beneficial else repeat_gap,
                           a[fields[3]]), donor, recipient))
    if not pairs:
        return None
    strength, donor, recipient = max(pairs)
    return {"donor": donor, "recipient": recipient, "screen_beneficial": strength[0], "criterion": strength[1],
            "donor_repeat_sign": 1 if screen[donor]["d0_same_repeat"] >= 0 else -1}


def module_arcs(donor, rms):
    readout = (DST >= INPUTS) & (DST < INPUTS + ACTIONS)
    weight = donor["birth_w"].masked_fill(~donor["birth_mask"], 0.)
    action_strength = torch.stack([weight[readout & SRC.eq(node)].norm() for node in range(PORTS, NODES)])
    scores = rms * action_strength
    hidden = scores.topk(2).indices + PORTS
    incoming_hub = torch.isin(DST, hidden)
    outgoing_hub = torch.isin(SRC, hidden)
    receipt = SRC.eq(_ACK) | SRC.eq(_REJ)
    key_ports = SRC.ge(RAW.start) & SRC.lt(RAW.stop) | SRC.ge(CONTEXT.start) & SRC.lt(CONTEXT.stop)
    action = SRC.ge(INPUTS) & SRC.lt(INPUTS + ACTIONS)
    core = incoming_hub & (outgoing_hub | receipt) | outgoing_hub & readout
    near = incoming_hub & (key_ports | action) & donor["birth_mask"]
    candidates = near.nonzero().flatten()
    extra = candidates[weight[candidates].abs().topk(min(6, len(candidates))).indices]
    arcs = torch.cat((core.nonzero().flatten(), extra)).unique()
    return hidden, arcs, scores


def graft(recipient, donor, loci):
    new = {**recipient}
    for field in ("birth_w", "birth_mask"):
        new[field] = recipient[field].clone()
        new[field][loci] = donor[field][loci]
    return new


def edit_types(a, b):
    aw = a["birth_w"].half().float().masked_fill(~a["birth_mask"], 0.)
    bw = b["birth_w"].half().float().masked_fill(~b["birth_mask"], 0.)
    delta = bw - aw
    changes = a["birth_mask"] ^ b["birth_mask"] | delta.ne(0.)
    kind = a["birth_mask"].int() + 2 * b["birth_mask"].int()
    return changes, kind, delta


def _roles(nodes):
    return torch.bucketize(nodes, torch.tensor([INPUTS, INPUTS + ACTIONS, PORTS]), right=True)


def matched_scatter(recipient, child, hubs, seed):
    """Scatter the observed FP16 displacements while preserving edit and endpoint types."""
    changes, kind, delta = edit_types(recipient, child)
    edited = changes.nonzero().flatten()
    sr, dr = _roles(SRC), _roles(DST)
    weight = recipient["birth_w"].half().float()
    exists = recipient["birth_mask"]
    free = ~torch.isin(SRC, hubs) & ~torch.isin(DST, hubs) & ~changes
    generator = geometry.generator(seed)
    scatter = graft(recipient, recipient, edited)
    for index in sorted(edited.tolist(), key=lambda i: (int(kind[i]) != 1, -abs(float(delta[i])))):
        edit = int(kind[index])  # 1 removes, 2 adds, 3 changes an active weight.
        possible = free & sr.eq(sr[index]) & dr.eq(dr[index]) & exists.eq(edit != 2)
        if edit == 3:
            moved = (weight + delta[index]).half().float()
            possible &= moved.abs().le(WEIGHT_LIMIT) & moved.ne(0.) & moved.ne(weight)
        options = possible.nonzero().flatten()
        if not len(options):
            return None
        if edit == 1:
            # Match the observed amount removed using the nearest few weights.
            near = (weight[options] + delta[index]).abs().topk(min(3, len(options)), largest=False).indices
            options = options[near]
        locus = options[torch.randint(len(options), (), generator=generator)]
        free[locus] = False
        scatter["birth_mask"][locus] = edit != 1
        scatter["birth_w"][locus] = 0. if edit == 1 else weight[locus] + delta[index]
    return scatter


def spec(recipient, child):
    changed, kind, delta = edit_types(recipient, child)
    return {"edits": int(changed.sum()), "kinds": dict(Counter(kind[changed].tolist())),
            "weight_l2": float(delta.norm()),
            "arcs": [{"source": int(SRC[i]), "target": int(DST[i]), "type": int(kind[i]), "delta": float(delta[i])}
                     for i in changed.nonzero().flatten().tolist()]}


def contrasts(rows, base, other):
    rng = np.random.default_rng(1450)
    result = {}
    for name in next(iter(rows.values()))[base]:
        points = [row[other][name] - row[base][name] for row in rows.values()
                  if row[base][name] is not None and row[other][name] is not None]
        if points:
            data = np.asarray(points)
            draw = rng.choice(data, size=(4096, len(points))).mean(1)
            result[name] = {"delta": float(data.mean()), "paired_worlds": len(data),
                            "ci95": np.quantile(draw, [.025, .975]).tolist() if len(data) > 1 else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reuse-discovery", type=Path, help="Reuse the same archive-bound screen and localization.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("provide a new output file")
    torch.set_num_threads(1)
    parents, parent_ids, hashes = geometry.load_parents(BRANCH, 1024, run.previous_run.rank)
    cached = None
    if args.reuse_discovery:
        cached = json.loads(args.reuse_discovery.read_text(encoding="utf-8"))
        if (cached["archive_sha256"] != hashes or cached["parents"] != parent_ids
                or tuple(cached["screen_seeds"]) != SCREEN or tuple(cached["localization"]["seeds"]) != LOCALIZE):
            parser.error("The discovery report must use the current archived parents and discovery seeds")
        screen_rows, screen, selection = (cached[key] for key in ("screen_rows", "screen", "selection"))
        if selection != choose_pair(screen):
            parser.error("The cached selection does not reproduce the declared rule")
    else:
        screen_rows, _ = measure(parents, SCREEN)
        screen = means(screen_rows)
        selection = choose_pair(screen)
    print(f"Screened all eight candidates. Selected {selection}", flush=True)
    report = {"schema": 1, "generation": 1024, "archive_sha256": hashes, "parents": parent_ids,
              "screen_seeds": SCREEN, "screen": screen, "screen_rows": screen_rows, "selection": selection,
              "device": "CPU, one Torch thread", "source_request_one_based": SOURCE + 1,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "precision": "Frozen IID FP16 eager policy, with eligibility kept active on traffic-free ticks.",
              "selection_rule": "Prefer a donor with discovery truthful gain and context specificity both at least .05. "
                                "Otherwise choose the same-island contrast in absolute ACK–REJ repeat response. "
                                "The fallback diagnoses a response without assuming it helps the task.",
              "measurement": "Native prefix and first READY FINISH. Fork only the sign of the next receipt and "
                             "retire the teaching request in both arms. Query unreinforced after zero, one, or eight "
                             "yoked distractors in holdout."}
    if args.reuse_discovery:
        report["reused_discovery_sha256"] = hashlib.sha256(args.reuse_discovery.read_bytes()).hexdigest()
    _write(args.output, {**report, "status": "screen complete"})
    if selection is not None:
        main_transfer(report, parents, selection, localization=None if cached is None else cached["localization"])
    report["completed_utc"] = _utc()
    _write(args.output, report)
    print(args.output, flush=True)


def main_transfer(report, parents, selection, localization=None):
    donor, recipient = (parents[selection[label]] for label in ("donor", "recipient"))
    if localization is None:
        local_rows, trace = measure({"donor": donor, "recipient": recipient}, LOCALIZE, trace=True)
        donor_hubs, donor_arcs, scores = module_arcs(donor, trace["donor"])
        # Pair the selected hidden roles by response magnitude and structural port signature.
        order, alignment = geometry.align(recipient, donor, trace["recipient"][:, None], trace["donor"][:, None])
        hubs = order.argsort()[donor_hubs]
        locus_order = geometry.relabel({**donor, "birth_w": torch.arange(len(SRC))}, order)["birth_w"]
        graft_arcs = torch.isin(locus_order, donor_arcs).nonzero().flatten()
        cut_rows, _ = measure({"donor": donor, "donor_cut": donor}, LOCALIZE, cuts={"donor_cut": donor_arcs})
        localization = {"seeds": LOCALIZE, "rows": local_rows, "donor_cut_rows": cut_rows,
                        "donor_cut_minus_donor": contrasts(cut_rows, "donor", "donor_cut"),
                        "donor_hidden_rms_response": trace["donor"].tolist(), "donor_module_scores": scores.tolist(),
                        "donor_hidden": donor_hubs.tolist(), "recipient_hidden": hubs.tolist(),
                        "alignment": {**alignment, "new_to_old": order.tolist()}}
    else:
        order = torch.tensor(localization["alignment"]["new_to_old"])
        hubs = torch.tensor(localization["recipient_hidden"])
        donor_hubs = torch.tensor(localization["donor_hidden"])
        loci = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(SRC, DST, strict=True))}
        graft_arcs = torch.tensor([loci[arc["source"], arc["target"]] for arc in localization["group_graft"]["arcs"]],
                                  dtype=torch.long)
    child = graft(recipient, geometry.relabel(donor, order), graft_arcs)
    target = spec(recipient, child)
    if "group_graft" in localization and json.loads(json.dumps(target)) != localization["group_graft"]:
        raise ValueError("The archived graft does not reproduce its FP16 edits")
    report["localization"] = {**localization, "group_graft": target}
    controls, specs = {}, {}
    for seed in range(977_430, 977_494):
        control = matched_scatter(recipient, child, hubs, seed)
        if control is None:
            continue
        matched = spec(recipient, control)
        if (matched["edits"] == target["edits"] and matched["kinds"] == target["kinds"]
                and abs(matched["weight_l2"] - target["weight_l2"]) <= .01 * target["weight_l2"]):
            label = f"scatter_{len(controls)}"
            controls[label], specs[label] = control, {**matched, "seed": seed}
            if len(controls) == 3:
                break
    print(f"Located donor hidden nodes {donor_hubs.tolist()}; graft edits {target['edits']}; matched controls {len(controls)}", flush=True)
    if target["edits"] == 0 or len(controls) < 3:
        report["holdout"] = {"available_controls": specs, "status": "insufficient matched interventions"}
        return
    cases = {"donor": donor, "recipient": recipient, "graft": child, **controls}
    holdout = {}
    for start in range(0, len(HOLDOUT), 2):
        rows, _ = measure(cases, HOLDOUT[start:start + 2])
        holdout.update(rows)
        print(f"Holdout {len(holdout)}/{len(HOLDOUT)} worlds", flush=True)
    for row in holdout.values():
        row["scatter_mean"] = {key: float(np.mean(available)) if available else None
                               for key in row["graft"]
                               for available in [[row[name][key] for name in controls if row[name][key] is not None]]}
    report["holdout"] = {"seeds": HOLDOUT, "paired_rows": holdout, "means": means(holdout), "controls": specs,
                         "graft_minus_recipient": contrasts(holdout, "recipient", "graft"),
                         "graft_minus_scatter": contrasts(holdout, "scatter_mean", "graft"),
                         "notation": "Paired percentile bootstrap over fixed held-out world seeds. "
                                     "No adjustment for multiple endpoints. Conditional response contrasts may "
                                     "be based on different eligible lives across interventions. Matched rows require "
                                     "every recipient arm to give the same teaching symbol at the same tick."}


if __name__ == "__main__":
    main()
