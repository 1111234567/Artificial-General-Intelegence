"""CPU semantic checks for the enabled version-free private-map task."""

from dataclasses import replace

import torch

from . import uncapped_private_symbol_world as world
from .evolved_plasticity_world import FIRST_FINISH, N_SYMBOLS, VERSION, _receipt_table
from .uncapped_ready_signal_run import schedule as versioned_schedule


class _ScriptedPolicy:
    def __init__(self, actions, *, record_observations=False):
        self.population = actions.shape[2]
        self.actions = iter(actions.flatten(0, 1).unbind())
        self.observations = [] if record_observations else None

    def act(self, observation):
        if self.observations is not None:
            self.observations.append(observation.clone())
        return next(self.actions)

    def summary(self):
        return {}


def _actions(tape, *, version=0, first_wrong=False, early=False, spare=False):
    requests, ticks, actors = tape.stage.shape
    actions = torch.zeros((requests, ticks, 1, actors, 2), dtype=torch.long)
    first_ready = tape.stage.eq(2).long().argmax(dim=1)
    for request in range(requests):
        for actor in range(actors):
            tick = first_ready[request, actor].item()
            target = tape.target[request, actor].item()
            actions[request, tick, 0, actor, 0] = (
                FIRST_FINISH + (target - FIRST_FINISH + 1) % N_SYMBOLS
                if first_wrong else target + version * N_SYMBOLS
            )
            if first_wrong:
                actions[request, tick + 1, 0, actor, 0] = target + version * N_SYMBOLS
            if early:
                actions[request, tick - 1, 0, actor, 0] = target + version * N_SYMBOLS
            if spare:
                actions[request, tick, 0, actor, 1] = 10
    return actions


def test_schedule_preserves_private_maps_and_events_without_version():
    args = dict(seed=712, actors=8, device="cpu", visits=8)
    original = versioned_schedule(**args, timing=world.TIMING)
    tape = world.schedule(**args)
    assert tape.public.shape[1] == 18
    expected_public = original.public.clone()
    expected_public[..., VERSION].zero_()
    assert torch.equal(tape.public, expected_public)
    for field in ("stage", "context", "raw", "maps"):
        assert torch.equal(getattr(tape, field), getattr(original, field)), field
    assert not (tape.maps[:, 0] == tape.maps[:, 1]).any()
    actor = torch.arange(args["actors"])
    assert torch.equal(tape.target, FIRST_FINISH + tape.maps[actor, tape.context, tape.raw])
    padded = world.schedule(**args, pad_to=30)
    assert torch.equal(padded.public[:, :18], tape.public)
    assert not padded.public[:, 18:].any()
    assert padded.stage[:, 18:].eq(3).all()


def test_four_training_world_blocks_match_the_previous_seed_partition():
    args = dict(seed=19, actors=16, device="cpu", visits=8)
    combined = world.training_schedule(**args)
    for block in range(world.WORLD_BLOCKS):
        independent = world.schedule(seed=args["seed"] * world.WORLD_BLOCKS + block,
                                     actors=4, device="cpu", visits=8)
        actors = slice(block * 4, (block + 1) * 4)
        assert torch.equal(combined.public[:, :, actors], independent.public)
        for field in ("target", "context", "raw"):
            assert torch.equal(getattr(combined, field)[:, actors], getattr(independent, field))
        assert torch.equal(combined.maps[actors], independent.maps)


def test_both_finish_halves_are_accepted_and_receipts_keep_emitted_identity():
    tape = world.schedule(seed=26, actors=2, device="cpu", visits=8)
    table = _receipt_table("cpu")
    results = []
    for version in (0, 1):
        actions = _actions(tape, version=version)
        policy = _ScriptedPolicy(actions, record_observations=True)
        metrics, observed = world.observed_rollout(policy, tape, table)
        assert torch.equal(observed, actions[..., 0])
        for key in ("completion", "strict", "first_finish_correct", "symbol_late_a",
                    "symbol_late_b", "symbol_late_mean", "symbol_first_mean",
                    "symbol_first_return_mean"):
            assert metrics[key].item() == 1, key
        assert metrics["wrong_answer_rate"].item() == metrics["off_ready_rate"].item() == 0
        first_tick = tape.stage[0, :, 0].eq(2).long().argmax().item()
        emitted = actions[0, first_tick, 0, 0, 0].item()
        incoming = policy.observations[first_tick + 1][0, 0, 11:24]
        assert torch.equal(incoming, table[emitted + 11])
        results.append(metrics)
    for key in results[0]:
        assert torch.equal(results[0][key], results[1][key]), key


def test_early_spare_and_first_wrong_semantics_remain_separate():
    tape = world.schedule(seed=20, actors=2, device="cpu", visits=8)
    table = _receipt_table("cpu")
    wrong = world.task_rollout(_ScriptedPolicy(_actions(tape, version=1, first_wrong=True)), tape, table)
    assert wrong["completion"].item() == 1
    assert wrong["strict"].item() == wrong["first_finish_correct"].item() == 0
    assert wrong["symbol_late_mean"].item() == 0
    early = world.task_rollout(_ScriptedPolicy(_actions(tape, version=1, early=True)), tape, table)
    assert early["completion"].item() == early["first_finish_correct"].item() == 1
    assert early["strict"].item() == 0
    assert early["premature_request_rate"].item() == 1
    assert early["off_ready_rate"].item() > 0
    spare = world.task_rollout(_ScriptedPolicy(_actions(tape, version=1, spare=True)), tape, table)
    assert spare["completion"].item() == spare["strict"].item() == 1
    assert spare["timing_clean"].item() == 0
    torch.testing.assert_close(spare["spare_rate"], torch.tensor([1 / 18]))
    assert spare["off_ready_rate"].item() == 0


def test_late_metric_uses_exactly_visits_five_through_eight():
    tape = world.schedule(seed=91, actors=2, device="cpu", visits=12)
    actions = _actions(tape, version=1)
    # Every first answer is right in visits 5-8 and wrong everywhere else.
    wrong_actions = _actions(tape, first_wrong=True)
    actions[:16] = wrong_actions[:16]
    actions[32:] = wrong_actions[32:]
    metrics = world.task_rollout(_ScriptedPolicy(actions), tape, _receipt_table("cpu"))
    assert metrics["symbol_late_mean"].item() == 1
    assert metrics["symbol_first_mean"].item() == metrics["symbol_first_return_mean"].item() == 0
    assert metrics["first_finish_correct"].item() < 1
    # Four-visit audits have no late sample and must remain finite.
    short = replace(tape, **{field: getattr(tape, field)[:16]
                            for field in ("public", "stage", "target", "context", "raw")}, visits=4)
    metrics = world.task_rollout(_ScriptedPolicy(_actions(short)), short, _receipt_table("cpu"))
    assert metrics["symbol_late_mean"].item() == 0
    assert all(value.isfinite().all() for value in metrics.values())


def test_canonicalization_retains_invalid_reserved_and_wait_codes():
    actions = torch.tensor([-9, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22])
    assert world.normalize_actions(actions).tolist() == [-9, -1, 0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 11, 22]
    source = world._kernel_source()
    assert "const bool accepted = legal &&" in source
    assert "receipt_idx[item * 2] = (valid0 ? act0 : 22)" in source
