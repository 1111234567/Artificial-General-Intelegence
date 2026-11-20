from dataclasses import fields

import torch

from experiments.pulse_game import interleave_curriculum as curriculum, interleave_train as train


def test_scaffold_rearranges_odor_retrieval_but_keeps_laws_marks_clock_and_draws():
    births = (731, 732)
    config = train.Config(pairs=4)
    tape = train.make_tape(births, config, 1)
    modes = torch.tensor([[0, 1, 2, 0], [1, 2, 1, 0]])
    changed = curriculum.arrange(tape, modes)
    odor = tape.context != train.MARK_CONTEXT
    assert torch.equal(odor, changed.context != train.MARK_CONTEXT)
    for name in ("draws", "waits", "law"):
        assert torch.equal(getattr(tape, name), getattr(changed, name))
    for name in curriculum.EVENT_FIELDS:
        original, after = (getattr(item, name) for item in (tape, changed))
        assert torch.equal(original[~odor], after[~odor])
        assert torch.equal(original[modes == 2], after[modes == 2])
    directions = changed.context[odor].reshape(*tape.batch_shape, 7)[:, :, 0]
    probe = changed.probe[odor].reshape(*tape.batch_shape, 7)[:, :, 0]
    direct = modes == 0
    assert torch.equal(directions[direct][:, :6:2], directions[direct][:, 1:6:2])
    assert (probe[direct][:, :6:2] == -1).all()
    assert torch.equal(directions[modes == 1][:, 0], directions[modes == 1][:, 2])
    assert torch.allclose(tape.objective_weight.sum(-1), changed.objective_weight.sum(-1))


def test_fully_phased_in_curriculum_uses_the_same_exact_warm_reference_tape():
    config = train.Config(updates=8000, pairs=4)
    step = 3500
    birth = (733,)
    tape, fraction = curriculum.world(birth, config, step)
    reference = train.make_tape(birth, config, step)
    assert fraction == (0., 0., 1.)
    for field in fields(train.Tape):
        assert torch.equal(getattr(tape, field.name), getattr(reference, field.name))
