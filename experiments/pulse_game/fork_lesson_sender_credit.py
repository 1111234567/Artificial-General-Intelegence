"""Extend prediction credit through the same sender's full experience."""

from argparse import ArgumentParser
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from . import fork_lesson_sender as sender


FIT = sender.baseline
WORLD = sender.world
DATA = FIT.DATA.parent / "sender-credit"
EVIDENCE = sender.EVIDENCE
PLAN = EVIDENCE / "sender-credit-plan.md"
DOMAIN = 6238511
BATCH = 64
EXPLORATION = .25
LEARNING_RATE = .006
GRADIENT_LIMIT = 2.


def path(kind, birth):
    return DATA / "fit" / kind / f"birth-{birth}.pt"


def signature(kind, birth, parent, prior):
    return dict(kind=kind, birth=birth, parent=parent, prior=prior, step=sender.STEPS,
                world=FIT.digest(WORLD.__file__), source=FIT.digest(__file__), sender=FIT.digest(sender.__file__),
                plan=FIT.digest(PLAN))


def prior_run(kind, birth, parent):
    location = sender.path(kind, birth, "true")
    content = torch.load(location, weights_only=True, map_location="cpu")
    expected = dict(kind=kind, birth=birth, parent=parent, arm="true", step=sender.STEPS,
                    source=FIT.digest(sender.__file__), plan=FIT.digest(sender.PLAN))
    if any(content[key] != value for key, value in expected.items()):
        raise ValueError(f"Original matched sender fit has an incompatible identity: {location}")
    return content, FIT.digest(location)


def learning_event(model, scene, *, all_history):
    prefix_control = nullcontext() if all_history else torch.no_grad()
    with prefix_control:
        seen = WORLD.cue(model, WORLD.support(model, scene), scene)
    final, activities, _ = WORLD.choice_window(model, seen)
    predictions = FIT.forecast_scores(activities)
    with torch.no_grad():
        action = WORLD.choose(model, final, scene, exploration=EXPLORATION)
        _, outcome = WORLD.commit(model, final, scene, action)
        probability = (1 - EXPLORATION) * action.logit.sigmoid() + EXPLORATION / 2
    chosen = predictions.gather(1, action.side[:, None]).squeeze(1)
    loss = F.mse_loss(chosen, outcome.payoff)
    return loss, (scene.draws, probability, action.side, outcome.payoff, outcome.payoff, chosen.detach())


def verify_unchanged(model, start):
    masks = model.mask[20:, :20]
    original = start["return_weight"]
    current = model.return_weight.detach()
    if not torch.equal(original[~masks], current[~masks]):
        raise AssertionError("A disconnected sender weight was updated")
    for parameter, value in model.state_dict().items():
        if parameter != "return_weight" and not torch.equal(start[parameter], value):
            raise AssertionError(f"A frozen parameter or buffer changed: {parameter}")
    if int(masks.sum()) != 80:
        raise AssertionError("The sender no longer has the intended 80 outgoing axons")
    delta = current - original
    return dict(changed=int(delta.ne(0).sum()), possible=int(masks.sum()), max_change=float(delta.abs().max()))


def audit_initial(model, scene, previous):
    before = {}
    for label, scope in (("window", False), ("full", True)):
        loss, columns = learning_event(model, scene, all_history=scope)
        gradient, = torch.autograd.grad(loss, (model.return_weight,))
        norm = torch.linalg.vector_norm(gradient)
        ledger = torch.stack((*columns, norm.expand_as(columns[0])), 1)
        before[label] = ledger, gradient
    window, window_gradient = before["window"]
    full, full_gradient = before["full"]
    torch.testing.assert_close(window, previous, atol=0, rtol=0)
    torch.testing.assert_close(window[:, :-1], full[:, :-1], atol=0, rtol=0)
    difference = (full_gradient - window_gradient).norm()
    if float(difference) <= 1e-9:
        raise AssertionError("Extending credit through the earlier inputs had no derivative effect")
    return dict(window_gradient_norm=float(window[0, -1]), full_gradient_norm=float(full[0, -1]),
                gradient_difference_norm=float(difference))


def audit_ledger(kind, birth, ledger, reference):
    if ledger.shape != (sender.STEPS, BATCH, 7) or not torch.isfinite(ledger).all():
        raise AssertionError(f"Incomplete training ledger: {kind}/{birth}")
    draw, probability, side, earned, target, chosen, norm = ledger.unbind(2)
    if not torch.equal(draw, reference[:, :, 0]):
        raise AssertionError("The matched sender fits did not share their action draws")
    if not torch.equal(draw < probability, side.bool()) or not torch.equal(earned, target):
        raise AssertionError("A selected action or prediction target did not match the actual route")
    for index, action in enumerate(side.long()):
        scene = WORLD.scenes([DOMAIN, birth, index + 1], BATCH)
        target_actual = WORLD.oracle(scene).gather(1, action[:, None]).squeeze(1)
        if not torch.equal(scene.draws, draw[index]) or not torch.equal(earned[index], target_actual):
            raise AssertionError("Regenerating the private world disagreed with the real route ledger")
    return dict(physical_routes=int(side.numel()), selected_mse=float((chosen - earned).square().mean()),
                mean_delayed_return=float(earned.mean()), max_gradient_norm=float(norm.max()))


def train(kind, birth, *, smoke=False):
    model, parent = sender.load_parent(kind, birth)
    original, prior = prior_run(kind, birth, parent)
    model.return_weight.requires_grad_(True)
    if tuple(name for name, p in model.named_parameters() if p.requires_grad) != ("return_weight",):
        raise AssertionError("The alternate treatment exposed parameters outside the sender matrix")
    start = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam([model.return_weight], lr=LEARNING_RATE)
    initial = audit_initial(model, WORLD.scenes([DOMAIN, birth, 1], BATCH), original["ledger"][0])
    events = []
    count = 8 if smoke else sender.STEPS
    for step in range(1, count + 1):
        scene = WORLD.scenes([DOMAIN, birth, step], BATCH)
        loss, columns = learning_event(model, scene, all_history=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([model.return_weight], GRADIENT_LIMIT, error_if_nonfinite=True)
        optimizer.step()
        events.append(torch.stack((*columns, norm.expand_as(columns[0])), 1))
        if step % 256 == 0 or step == count:
            print(kind, birth, step, "mse", round(float(loss.detach()), 4), flush=True)
    changed = verify_unchanged(model, start)
    if smoke:
        print("smoke", initial, changed, flush=True)
        return
    ledger = torch.stack(events)
    audit = audit_ledger(kind, birth, ledger, original["ledger"])
    pack = dict(signature=signature(kind, birth, parent, prior), body=model.state_dict(), ledger=ledger,
                initial_gradient=initial, audit=audit, **changed)
    FIT.storage._atomic(path(kind, birth), lambda tmp: torch.save(pack, tmp))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sender.KINDS, required=True)
    parser.add_argument("--birth", type=int, choices=FIT.BIRTHS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.kind, args.birth, smoke=args.smoke)


if __name__ == "__main__":
    main()
