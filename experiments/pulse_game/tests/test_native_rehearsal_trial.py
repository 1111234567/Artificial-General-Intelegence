"""Check the cloned parameter path and the independence of the real adoption path."""

from dataclasses import fields, replace

import torch

from experiments.pulse_game import native_rehearsal_trial as trial
from experiments.pulse_game import native_rehearsal_trial_eval as evaluation


def test_candidate_overlay_changes_shadow_before_main_activity_and_only_adopts_original():
    torch.set_num_threads(1)
    with torch.no_grad():
        enabled = trial.TrialGraph(731, "trial")
        disabled = trial.TrialGraph(731, "no_trial")
        for a, b in zip(enabled.state_dict().values(), disabled.state_dict().values(), strict=True):
            assert torch.equal(a, b)
        n = trial.REGION
        start = enabled.start(4)
        occupied = enabled.mask[:n, :n]
        draft = occupied.to(start.h.dtype)[None].expand(len(start.h), -1, -1) * 0.01
        start = replace(start, h=torch.full_like(start.h, 0.4), candidate=draft)
        quiet = start.h.new_zeros((len(start.h), 10))
        trialed, _ = enabled.tick(start, quiet)
        cut, _ = enabled.tick(start, quiet, read_override=False)
        other, _ = enabled.tick(start, quiet, candidate_override=-draft)
        stop, _ = enabled.tick(start, quiet, adoption_off=True)
        control, _ = disabled.tick(start, quiet)
        assert torch.equal(trialed.h[:, :n], cut.h[:, :n])
        assert not torch.equal(trialed.h[:, n:], cut.h[:, n:])
        for f in fields(trial.State):
            assert torch.equal(getattr(control, f.name), getattr(cut, f.name))
        for state in (trialed, cut, other):
            actual = evaluation.writable(state) - evaluation.writable(start)
            assert (actual[draft != 0] > 0).all()
            assert (actual <= draft).all()
        assert torch.equal(evaluation.writable(start), evaluation.writable(stop))
        assert torch.equal(trialed.h, stop.h)


def test_future_tape_is_hidden_from_support_and_quiet():
    torch.set_num_threads(1)
    model = trial.TrialGraph(731, "trial")
    episode = trial.live.scenes(82871, 3, evaluation=True)
    draws = episode.draws.clone()
    orientation = episode.orientation.clone()
    draws[:, 3:] = 1 - draws[:, 3:]
    orientation[:, 3:] *= -1
    poisoned = replace(episode, draws=draws, orientation=orientation, first_query=1 - episode.first_query)
    with torch.no_grad():
        states = [evaluation.run_quiet(model, trial.live.support(model, scene)[1])[0]
                  for scene in (episode, poisoned)]
    for f in fields(trial.State):
        assert torch.equal(getattr(states[0], f.name), getattr(states[1], f.name))
