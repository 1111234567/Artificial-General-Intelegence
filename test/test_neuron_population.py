from dataclasses import replace
import sys

import numpy as np
import pytest
import torch

from experiments.pulse_game import attractor_search, dendrite_search
from experiments.pulse_game.dendrite_audit import reference
from src.neurons.lif_population import (
    BASE_GENE_RANGES, DENDRITE_GENE_RANGES, Schedule, evaluate_population,
)


@pytest.fixture
def schedule():
    def port(values):
        return np.array(values, dtype=np.uint8)

    return Schedule(
        cue=port([[0, 1, 3], [2, 0, 1]]), ctx=port([[0, 1, 0], [1, 1, 0]]),
        issue=port([[0, 1, 0], [1, 0, 1]]), query=port([[7, 8, 6], [7, 7, 6]]),
        length=port([[11, 11, 11], [11, 11, 11]]),
        raw_at=port([[255, 3, 255], [3, 255, 3]]),
        distractor=port([[0, 1, 7], [4, 5, 2]]), at=port([[4, 255, 5], [255, 2, 4]]),
        goal=port([[0, 2, 1], [1, 3, 2]]),
        tie=np.random.default_rng(329).random((2, 3, 4), dtype=np.float32), seeds=(819, 329),
    )


def _genes(mode):
    ranges = BASE_GENE_RANGES if mode == "base" else DENDRITE_GENE_RANGES
    return np.random.default_rng(309063).uniform(0.15, 0.70, (2, len(ranges)))


@pytest.mark.parametrize("mode", ["base", "basal", "add", "gain", "both"])
def test_cpu_population_matches_original_model(schedule, mode):
    vectors = _genes(mode)
    result = evaluate_population(vectors, 77, schedule, mode=mode, features=True)
    for i, vector in enumerate(vectors):
        if mode == "base":
            z, rate = attractor_search.cpu_features(vector, 77, schedule)
        else:
            z, rate = reference(vector, 77, schedule, mode)
        answers = attractor_search.cpu_answers(vector[:len(BASE_GENE_RANGES)], z, schedule)
        np.testing.assert_array_equal(result["z"][i], z)
        np.testing.assert_array_equal(result["answers"][i], answers)
        assert result["spike_fraction"][i] == rate
        assert result["fitness"][i] == (answers == schedule.goal).mean()


def test_no_feedback_and_noncontiguous_tie(schedule):
    tie = np.repeat(schedule.tie.astype(np.float64), 2, axis=-1)[..., ::2]
    result = evaluate_population(_genes("base"), 77, replace(schedule, tie=tie),
                                 cut=True, feedback=False)
    assert "z" not in result
    np.testing.assert_array_equal(result["answers"], np.broadcast_to(tie.argmax(-1), (2, 2, 3)))


@pytest.mark.parametrize(("field", "change"), [
    ("cue", lambda s: np.full(s.cue.shape, 256)),
    ("ctx", lambda s: s.ctx.astype(np.float32)),
    ("query", lambda s: s.length),
    ("raw_at", lambda s: s.length),
    ("distractor", lambda s: np.full(s.cue.shape, 8)),
    ("goal", lambda s: np.full(s.cue.shape, 4)),
    ("tie", lambda s: s.tie[..., :3]),
    ("seeds", lambda s: s.seeds[:1]),
])
def test_invalid_ports_rejected_before_cuda(schedule, field, change):
    with pytest.raises(ValueError, match=rf"schedule\.{field}"):
        evaluate_population(_genes("base"), 77, replace(schedule, **{field: change(schedule)}),
                            device="cuda:0")


def test_invalid_gene_packets(schedule):
    with pytest.raises(ValueError, match="vectors must"):
        evaluate_population(_genes("base"), 77, schedule, mode="both")
    vectors = _genes("base")
    vectors[0, 0] = np.nan
    with pytest.raises(ValueError, match="gene values"):
        evaluate_population(vectors, 77, schedule, device="cuda:0")


def test_dendrite_cuda_rejects_other_cuda_devices(schedule):
    with pytest.raises(ValueError, match="requires cuda:0"):
        evaluate_population(_genes("both"), 77, schedule, mode="both", device="cuda:1")


def test_cuda_rejects_overflowing_feature_offsets(schedule):
    shape = (36, 64)
    fields = {name: np.full(shape, value[0, 0], dtype=value.dtype)
              for name, value in vars(schedule).items() if isinstance(value, np.ndarray) and value.ndim == 2}
    many_lives = Schedule(**fields, tie=np.zeros((*shape, 4), dtype=np.float32), seeds=tuple(range(shape[0])))
    vectors = np.broadcast_to(_genes("base")[0], (60000, len(BASE_GENE_RANGES)))
    with pytest.raises(ValueError, match="index range"):
        evaluate_population(vectors, 77, many_lives, device="cuda:0", batch_size=len(vectors))


@pytest.mark.skipif(sys.platform != "win32" or not torch.cuda.is_available(),
                    reason="The original NVRTC backend requires Windows and CUDA")
@pytest.mark.parametrize("mode", ["base", "both"])
def test_cuda_population_matches_original_and_cpu_reference(schedule, mode):
    vectors = _genes(mode)
    result = evaluate_population(vectors, 77, schedule, mode=mode, device="cuda:0", features=True,
                                 batch_size=1)
    with torch.cuda.device(0):
        if mode == "base":
            expected = attractor_search.gpu_batch(vectors, 77, schedule, features=True)
        else:
            answers, rate, z = dendrite_search.run(vectors, 77, schedule, mode=mode, features=True)
            expected = {"answers": answers, "spike_fraction": rate, "z": z}
    for key, value in expected.items():
        np.testing.assert_array_equal(result[key], value)

    cpu = evaluate_population(vectors, 77, schedule, mode=mode, features=True)
    np.testing.assert_array_equal(result["answers"], cpu["answers"])
    np.testing.assert_array_equal(result["spike_fraction"], cpu["spike_fraction"])
    np.testing.assert_allclose(result["z"], cpu["z"], rtol=0, atol=1e-5)
