import torch

from experiments.pulse_game import revisit_choice_value as study


def test_hidden_mark_laws_require_informative_demonstrations():
    scene = study.scenes([411, 7], 4, balanced=True)
    graph = study.fw.spawn(731, 8)
    for mode in ("blank", "yoked"):
        with torch.no_grad():
            action, earned, _ = study.interact(graph, scene, len(scene.book), treatment=mode)
        # Adjacent blocks of eight have opposite private laws and identical visible tapes.
        side = action.side.reshape(-1, 2, 8)
        reward = earned.reshape(-1, 2, 8)
        assert torch.equal(side[:, 0], side[:, 1])
        assert torch.equal(reward[:, 0], -reward[:, 1])


def test_simple_public_association_predicts_which_feeder_gains_a_unit():
    scene = study.scenes(438, 6, balanced=True)
    first_receipt = scene.first_mark * scene.book
    repeat_first_mark = scene.right_mark == scene.first_mark
    side = torch.where(repeat_first_mark, first_receipt, -first_receipt)
    assert torch.all(side * scene.right_mark * scene.book == 1)
