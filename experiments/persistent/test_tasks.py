"""Semantic checks using independent Python dictionary walks.

Run from the repository root with:
    python -m pytest experiments/persistent/test_tasks.py -q
"""

from dataclasses import replace

import pytest
import torch

from experiments.persistent.tasks import Batch, exact_oracle, make_batch


TENSOR_FIELDS = ("keys", "values", "query", "hops", "target", "path_targets")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _dictionary_paths(batch: Batch) -> list[list[int]]:
    """Reference from record pairs; no tensor lookup or stored gold labels."""
    results = []
    for row in range(batch.keys.shape[0]):
        records = {
            int(key): int(value)
            for key, value in zip(batch.keys[row].tolist(), batch.values[row].tolist())
        }
        position = int(batch.query[row])
        row_path = []
        for _ in range(batch.path_targets.shape[1]):
            position = records[position]
            row_path.append(position)
        results.append(row_path)
    return results


def _assert_semantics(batch: Batch, batch_size: int, n_records: int, max_hops: int) -> None:
    assert batch.keys.shape == batch.values.shape == (batch_size, n_records)
    assert batch.query.shape == batch.hops.shape == batch.target.shape == (batch_size,)
    assert batch.path_targets.shape == (batch_size, max_hops)
    assert all(getattr(batch, name).dtype == torch.long for name in TENSOR_FIELDS)
    for keys, values, query in zip(batch.keys.tolist(), batch.values.tolist(), batch.query.tolist()):
        assert len(set(keys)) == n_records
        assert sorted(keys) == sorted(values)
        assert query in keys
        assert all(0 <= key < batch.vocab for key in keys)
    assert bool(((batch.hops >= 1) & (batch.hops <= max_hops)).all())

    reference_paths = _dictionary_paths(batch)
    assert batch.path_targets.tolist() == reference_paths
    reference_targets = [
        path[hops - 1] for path, hops in zip(reference_paths, batch.hops.tolist())
    ]
    assert batch.target.tolist() == reference_targets
    assert exact_oracle(batch).tolist() == reference_targets


@pytest.mark.parametrize("seed", [0, 17, 8491])
@pytest.mark.parametrize("n_records,max_hops", [(3, 1), (7, 4), (19, 11), (64, 9)])
def test_random_batches_and_every_hop_match_dictionary_walks(seed: int, n_records: int, max_hops: int):
    generator = _generator(seed)
    for batch_size in (1, 13, 37):
        batch = make_batch(batch_size, n_records, 8, max_hops, generator=generator)
        _assert_semantics(batch, batch_size, n_records, max_hops)


@pytest.mark.parametrize("fixed_hops", [1, 3, 7])
def test_fixed_hops_keeps_full_analysis_path(fixed_hops: int):
    batch = make_batch(101, 13, 5, 7, generator=_generator(5), fixed_hops=fixed_hops)
    assert batch.hops.tolist() == [fixed_hops] * 101
    _assert_semantics(batch, 101, 13, 7)


def test_known_cycles_and_self_loop_have_exact_endpoints():
    # 2 -> 8 -> 5 -> 2, with a separate 11 -> 11 self-loop.
    batch = Batch(
        keys=torch.tensor([[5, 11, 2, 8], [8, 2, 11, 5]]),
        values=torch.tensor([[2, 11, 8, 5], [5, 8, 11, 2]]),
        query=torch.tensor([2, 11]),
        hops=torch.tensor([5, 4]),
        target=torch.tensor([5, 11]),
        path_targets=torch.tensor([[8, 5, 2, 8, 5, 2], [11, 11, 11, 11, 11, 11]]),
        gap=0,
        vocab=64,
    )
    _assert_semantics(batch, 2, 4, 6)


def test_record_shuffle_preserves_mapping_query_and_all_gold():
    batch = make_batch(71, 17, 13, 9, generator=_generator(126))
    shuffled = batch.shuffle_records(_generator(95))
    assert not torch.equal(batch.keys, shuffled.keys)
    for original_keys, original_values, new_keys, new_values in zip(
        batch.keys.tolist(), batch.values.tolist(), shuffled.keys.tolist(), shuffled.values.tolist()
    ):
        assert dict(zip(original_keys, original_values)) == dict(zip(new_keys, new_values))
    for field in ("query", "hops", "target", "path_targets"):
        assert torch.equal(getattr(batch, field), getattr(shuffled, field))
    assert shuffled.gap == batch.gap
    assert shuffled.vocab == batch.vocab
    _assert_semantics(shuffled, 71, 17, 9)


def test_noop_gap_changes_neither_random_draws_nor_answers():
    short = make_batch(43, 15, 0, 8, generator=_generator(614))
    long = make_batch(43, 15, 10_000, 8, generator=_generator(614))
    for field in TENSOR_FIELDS:
        assert torch.equal(getattr(short, field), getattr(long, field))
    changed = short.with_gap(213)
    assert changed.gap == 213
    assert torch.equal(exact_oracle(short), exact_oracle(changed))
    assert torch.equal(exact_oracle(short), exact_oracle(long))


def test_seed_reproducibility_and_explicit_rng_independence():
    global_state = torch.random.get_rng_state().clone()
    first_rng = _generator(975)
    first = make_batch(29, 11, 3, 6, generator=first_rng)
    assert torch.equal(global_state, torch.random.get_rng_state())

    # Arbitrary default-RNG draws cannot change batches drawn from local RNGs.
    try:
        torch.manual_seed(87622)
        torch.rand(113)
        second = make_batch(29, 11, 3, 6, generator=_generator(975))
    finally:
        torch.random.set_rng_state(global_state)
    for field in TENSOR_FIELDS:
        assert torch.equal(getattr(first, field), getattr(second, field))

    next_batch = make_batch(29, 11, 3, 6, generator=first_rng)
    assert not torch.equal(first.keys, next_batch.keys)
    other_seed = make_batch(29, 11, 3, 6, generator=_generator(976))
    assert not torch.equal(first.keys, other_seed.keys)


def test_default_rng_can_also_be_reseeded():
    global_state = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(527)
        first = make_batch(19, 8, 0, 4)
        torch.manual_seed(527)
        second = make_batch(19, 8, 0, 4)
    finally:
        torch.random.set_rng_state(global_state)
    for field in TENSOR_FIELDS:
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_oracle_never_reads_gold_labels():
    batch = make_batch(41, 21, 9, 12, generator=_generator(824))
    corrupted = replace(
        batch,
        target=torch.full_like(batch.target, -99),
        path_targets=torch.full_like(batch.path_targets, -99),
    )
    assert torch.equal(exact_oracle(batch), exact_oracle(corrupted))


def test_concatenated_prefix_and_suffix_match_each_step_of_full_walk():
    batch = make_batch(61, 23, 4, 13, generator=_generator(33))
    paths = _dictionary_paths(batch)
    # Starting a fresh walk after a prefix must reproduce every suffix step.
    # Each expected answer comes from the independently constructed full walk.
    for prefix in (1, 3, 7):
        for suffix in range(1, 14 - prefix):
            restarted = replace(
                batch,
                query=torch.tensor([path[prefix - 1] for path in paths]),
                hops=torch.full_like(batch.hops, suffix),
            )
            expected = [path[prefix + suffix - 1] for path in paths]
            assert exact_oracle(restarted).tolist() == expected
            assert batch.path_targets[:, prefix + suffix - 1].tolist() == expected


def test_sampling_distribution_has_no_record_position_or_label_shortcut():
    # Broad, deterministic bounds test for severe sampler mistakes.  These
    # checks do not assert independence of graph endpoints and query for
    # multi-hop permutation graphs: cycle lengths inherently relate them.
    batch_size, n_records, max_hops = 8192, 12, 6
    batch = make_batch(batch_size, n_records, 0, max_hops, generator=_generator(773))
    for name in ("query", "target"):
        counts = torch.bincount(getattr(batch, name), minlength=64)
        assert bool(((counts >= 75) & (counts <= 185)).all())
    hop_counts = torch.bincount(batch.hops, minlength=max_hops + 1)[1:]
    assert bool(((hop_counts >= 1150) & (hop_counts <= 1580)).all())
    key_counts = torch.bincount(batch.keys.flatten(), minlength=64)
    assert bool(((key_counts >= 1330) & (key_counts <= 1750)).all())

    query_columns = (batch.keys == batch.query[:, None]).nonzero()[:, 1]
    target_columns = (batch.keys == batch.target[:, None]).nonzero()[:, 1]
    for columns in (query_columns, target_columns):
        counts = torch.bincount(columns, minlength=n_records)
        assert bool(((counts >= 540) & (counts <= 825)).all())
    # One-hop chance of a self-loop is 1/N, not zero and not always identity.
    one_hop = batch.hops == 1
    self_loop_rate = (batch.target[one_hop] == batch.query[one_hop]).float().mean().item()
    assert 0.05 < self_loop_rate < 0.12

    # No batch-wide shared memory table or fixed key set is being recycled.
    distinct_key_sets = {tuple(sorted(row)) for row in batch.keys.tolist()}
    assert len(distinct_key_sets) > 0.99 * batch_size
    _assert_semantics(batch, batch_size, n_records, max_hops)


def test_clone_and_device_transfer_preserve_metadata_and_own_storage():
    batch = make_batch(7, 9, 12, 5, generator=_generator(87))
    copied = batch.clone().to("cpu")
    assert copied.gap == 12 and copied.vocab == 64
    for field in TENSOR_FIELDS:
        original, cloned = getattr(batch, field), getattr(copied, field)
        assert torch.equal(original, cloned)
        assert original.data_ptr() != cloned.data_ptr()
        assert cloned.device.type == "cpu"
    copied.keys[0, 0] = -1
    assert bool((batch.keys >= 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_generator_can_sample_directly_for_cuda_output():
    cpu = make_batch(17, 14, 7, 6, generator=_generator(42))
    gpu = make_batch(17, 14, 7, 6, device="cuda", generator=_generator(42))
    for field in TENSOR_FIELDS:
        assert getattr(gpu, field).device.type == "cuda"
        assert torch.equal(getattr(cpu, field), getattr(gpu, field).cpu())
    assert torch.equal(exact_oracle(gpu), gpu.target)
    shuffled = gpu.shuffle_records(_generator(18))
    assert torch.equal(exact_oracle(shuffled), gpu.target)


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"n_records": 1},
        {"n_records": 65},
        {"n_records": 3, "vocab": 2},
        {"gap": -1},
        {"max_hops": 0},
        {"vocab": 1},
        {"fixed_hops": 0},
        {"fixed_hops": 5},
        {"batch_size": True},
        {"gap": 1.5},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    arguments = dict(batch_size=8, n_records=7, gap=0, max_hops=4)
    arguments.update(changes)
    with pytest.raises(ValueError):
        make_batch(**arguments, generator=_generator(5))
