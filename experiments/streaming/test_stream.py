"""CPU checks for causal streams, raw cache decisions, and deployed halting."""
from dataclasses import fields, replace
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.streaming.cache import CacheAgent, State
from experiments.streaming.stream import Event, cycle, observed, oracle
from experiments.streaming import run_stream


@pytest.fixture(autouse=True)
def cpu_rng():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(41)
        yield


def fact(key, value, *, context=0, terminal=1.0):
    return Event("fact", torch.tensor([context]), torch.tensor([key]),
                 torch.tensor([value]), torch.tensor([terminal]))


def query(key=0, *, context=0, target=7, depth=2):
    return Event("query", torch.tensor([context]), torch.tensor([key]),
                 target=torch.tensor([target]), depth=torch.tensor([depth]), phase="probe")


def aligned_reader():
    """A known reader isolates cache/halting semantics from optimization noise."""
    model = CacheAgent(contexts=3, vocab=8, slots=4, hidden=8, max_reads=6)
    with torch.no_grad():
        model.key.weight.zero_()
        model.query.weight.zero_()
        width = model.contexts + model.vocab
        model.key.weight[:width] = torch.eye(width)
        model.query.weight[:width] = torch.eye(width)
        model.log_scale.fill_(math.log(100))
        model.halt.weight.copy_(torch.tensor([[12.0, 0.0]]))
        model.halt.bias.fill_(-6)
    state = model.initial(1, "cpu")
    for event in (fact(0, 1, terminal=0), fact(1, 7),
                  fact(2, 3, terminal=0), fact(3, 6)):
        state, _ = model.write(state, event, "fifo")
    return model, state


def follow(table, context, key):
    for depth in range(1, len(table) + 2):
        value, terminal = table[(context, key)]
        if terminal:
            return value, depth
        key = value
    raise AssertionError("observed facts did not terminate")


@pytest.mark.parametrize("nodes,return_to", [(4, "a"), (4, "b"), (6, "random")])
def test_all_queries_and_every_node_match_causal_dictionary(nodes, return_to):
    batch = 12
    events = cycle(batch, torch.Generator().manual_seed(123), nodes=nodes,
                   noise=3, queries=3, return_to=return_to)
    tables = [dict() for _ in range(batch)]
    phase_terminals = {}
    for event in events:
        if event.kind == "fact":
            for row in range(batch):
                tables[row][(int(event.context[row]), int(event.key[row]))] = (
                    int(event.value[row]), bool(event.terminal[row]))
        elif event.kind == "query":
            terminals = []
            for row, table in enumerate(tables):
                context, key = int(event.context[row]), int(event.key[row])
                expected, depth = follow(table, context, key)
                assert expected == int(event.target[row])
                assert depth == int(event.depth[row])
                records = sorted((k, v, t) for (c, k), (v, t) in table.items() if c == context)
                assert len(records) == nodes
                terminal_values = {v for _, v, t in records if t}
                assert len(terminal_values) == 2
                terminals.append(terminal_values)
                ks, vs, ts = (torch.tensor([[record[i] for record in records]]) for i in range(3))
                for node, _, _ in records:
                    answer, steps = oracle(ks, vs, ts, torch.tensor([node]))
                    assert (int(answer[0]), int(steps[0])) == follow(table, context, node)
            phase_terminals.setdefault(event.phase, terminals)
    for row in range(batch):
        a, b, revision = (phase_terminals[p][row] for p in ("a_acquire", "b_acquire", "revision"))
        assert a.isdisjoint(b) and a.isdisjoint(revision) and b.isdisjoint(revision)


def test_return_probe_has_no_reexposure_of_returned_context():
    events = cycle(24, torch.Generator().manual_seed(9), noise=4, queries=3)
    cues = [i for i, event in enumerate(events) if event.kind == "cue"]
    assert len(cues) == 3
    returned_context = events[cues[2]].context
    revision_start = next(i for i, e in enumerate(events) if e.kind == "query" and e.phase == "revision")
    return_queries = [i for i, e in enumerate(events) if e.kind == "query" and e.phase == "return"]
    assert return_queries and return_queries[-1] < revision_start
    for event in events[cues[2] + 1:return_queries[-1]]:
        if event.kind == "fact":
            assert not (event.context == returned_context).any()


def test_context_and_key_roles_are_randomized_and_noise_uses_same_keys():
    batch = 96
    events = cycle(batch, torch.Generator().manual_seed(13), noise=1, queries=1)
    cues = [e.context for e in events if e.kind == "cue"]
    assert len(cues[0].unique()) == len(cues[1].unique()) == 8
    assert (cues[0] != cues[1]).all()
    first_b = next(i for i, e in enumerate(events) if e.kind == "cue" and torch.equal(e.context, cues[1]))
    key_sets = [set() for _ in range(batch)]
    arrival_terminals = []
    for event in events[:first_b]:
        if event.kind == "fact":
            active = event.context == cues[0]
            if active.all():
                arrival_terminals.append(event.terminal)
            for row in range(batch):
                if active[row]:
                    key_sets[row].add(int(event.key[row]))
    assert set.union(*key_sets) == set(range(16))
    assert all(len(keys) == 4 for keys in key_sets)
    for event in events:
        if event.kind == "fact":
            assert all(int(event.key[row]) in key_sets[row] for row in range(batch))
    assert arrival_terminals[0].bool().any()
    assert (~arrival_terminals[-1].bool()).any()


def test_explicit_generator_does_not_consume_global_rng():
    before = torch.get_rng_state().clone()
    cycle(4, torch.Generator().manual_seed(14), noise=1)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("kwargs", [{"nodes": 1}, {"vocab": 5}, {"contexts": 2}])
def test_invalid_forest_dimensions_fail(kwargs):
    with pytest.raises(ValueError):
        cycle(2, torch.Generator().manual_seed(1), **kwargs)


def test_labels_and_phase_do_not_change_observation_or_reading():
    model, state = aligned_reader()
    model.eval()
    original = query()
    poisoned = replace(original, target=torch.tensor([2]), depth=torch.tensor([999]), phase="secret")
    assert torch.equal(observed(original, 3, 8), observed(poisoned, 3, 8))
    first, second = model.observe(state, original), model.observe(state, poisoned)
    assert torch.equal(first.hidden, second.hidden)
    a, b = model.read(first, original), model.read(second, poisoned)
    for key in ("answer", "reads", "candidates", "halt"):
        assert torch.equal(a[key], b[key])


@pytest.mark.parametrize("policy", ["learned", "fifo", "random", "mru", "no_write"])
def test_hard_cache_contains_only_real_records_and_unique_identities(policy):
    model = CacheAgent(slots=8)
    model.train()
    state = model.initial(4, "cpu")
    history = [[] for _ in range(4)]
    events = cycle(4, torch.Generator().manual_seed(23), noise=2, queries=1)
    policy_rng = torch.Generator().manual_seed(24)
    for event in events:
        if event.kind != "fact":
            state = model.observe(state, event)
            continue
        raw = observed(event, model.contexts, model.vocab)[:, :model.payload]
        for row in range(4):
            history[row].append(raw[row].clone())
        state, _ = model.write(state, event, policy, policy_rng)
        for row in range(4):
            slots = state.memory[row]
            assert ((slots[:, -2] == 0) | (slots[:, -2] == 1)).all()
            occupied = slots[slots[:, -2] == 1]
            identities = occupied[:, :model.contexts + model.vocab]
            assert identities.unique(dim=0).shape[0] == occupied.shape[0]
            for record in occupied:
                assert any(torch.equal(record[:model.payload], old) for old in history[row])
                assert record[-1].item().is_integer()


def test_bad_eviction_receives_nonzero_finite_writer_gradient():
    model = CacheAgent(slots=1, max_reads=2)
    with torch.no_grad():
        for parameter in model.retain.parameters():
            parameter.zero_()
        # Prefer evicting the older record, which is the wrong decision here.
        model.retain[0].weight[0, model.payload] = -1
        model.retain[-1].weight[0, 0] = 1
    state = model.initial(1, "cpu")
    state, _ = model.write(state, fact(0, 3), "fifo")
    state, _ = model.write(state, fact(1, 4), "learned")
    assert state.memory[0, 0, model.contexts + model.vocab + 4] == 1
    _, loss, _ = run_stream.rollout(model, state, [query(target=3, depth=1)], "learned",
                                     generator=torch.Generator().manual_seed(1), cost=0)
    loss.backward()
    gradients = [p.grad for p in model.retain.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0


def test_persistent_hidden_does_not_observe_fact_values_or_terminal_flags():
    model = CacheAgent()
    a, b = model.initial(1, "cpu"), model.initial(1, "cpu")
    for step in range(12):
        first = fact(step % 8, step % 16, context=step % 3, terminal=float(step % 2))
        second = replace(first, value=(first.value + 7) % 16, terminal=1 - first.terminal)
        a, b = model.observe(a, first), model.observe(b, second)
        assert torch.equal(a.hidden, b.hidden)
        assert torch.equal(a.demand_time, b.demand_time)


def test_demand_timestamps_are_causal_and_identical_for_all_policies():
    model = CacheAgent()
    states = {policy: model.initial(1, "cpu") for policy in ("learned", "mru", "fifo", "random")}
    expected = torch.full((1, model.contexts), -1.)
    events = [Event("cue", torch.tensor([2])), fact(0, 3, context=4),
              query(context=2), fact(1, 5, context=5), Event("cue", torch.tensor([5]))]
    rng = torch.Generator().manual_seed(12)
    for clock, event in enumerate(events, 1):
        if event.kind in {"cue", "query"}:
            expected[0, event.context.item()] = clock
        for policy, state in states.items():
            states[policy] = (model.write(state, event, policy, rng)[0]
                              if event.kind == "fact" else model.observe(state, event))
            assert torch.equal(states[policy].demand_time, expected)
        assert all(torch.equal(state.hidden, states["learned"].hidden) for state in states.values())
    assert expected[0, 4] == -1  # An observed fact is not an observed demand.


def test_query_does_not_reuse_hidden_answers_or_modify_persistent_state():
    model, state = aligned_reader()
    model.eval()
    hidden, memory = state.hidden.clone(), state.memory.clone()
    output = model.read(state, query())
    changed_hidden = replace(state, hidden=torch.randn_like(state.hidden) * 1000)
    other = model.read(changed_hidden, query())
    assert torch.equal(output["hard_prob"], other["hard_prob"])
    assert torch.equal(state.hidden, hidden) and torch.equal(state.memory, memory)
    erased = replace(state, memory=torch.zeros_like(state.memory))
    blank = model.read(erased, query())
    assert torch.allclose(blank["hard_prob"], torch.full((1, 8), 1 / 8))


def test_hard_stop_executes_fewer_reader_steps_and_force_overrides_it():
    model, state = aligned_reader()
    model.eval()
    ordinary = model.read(state, query())
    assert ordinary["answer"].item() == 7 and ordinary["reads"].item() == 2
    assert ordinary["attention"].shape[1] == 2
    early = model.read(state, query(), force_reads=1)
    assert early["reads"].item() == early["attention"].shape[1] == 1
    assert early["answer"].item() != 7
    forced = model.read(state, query(), force_reads=6)
    assert forced["reads"].item() == forced["attention"].shape[1] == 6


@pytest.mark.parametrize("halt_loss", ["expected", "mixture"])
def test_training_halt_loss_formula_is_explicit(halt_loss):
    model, state = aligned_reader()
    model.max_reads = 2
    model.train()
    with torch.no_grad():
        model.halt.weight.zero_()
        model.halt.bias.zero_()
    event = query()
    out = model.read(model.observe(state, event), event)
    selected = out["candidates"][0, :, event.target.item()]
    expected = (out["masses"][0] * -selected.clamp_min(1e-12).log()).sum() + .02 * out["expected_reads"][0]
    mixture_nll = -out["prob"][0, event.target.item()].clamp_min(1e-12).log() + .02 * out["expected_reads"][0]
    _, actual, _ = run_stream.rollout(model, state, [event], "fifo",
                                      generator=torch.Generator().manual_seed(1), cost=.02,
                                      halt_loss=halt_loss)
    assert torch.allclose(actual, expected if halt_loss == "expected" else mixture_nll)
    assert expected > mixture_nll + 1
    assert torch.allclose(out["masses"].sum(-1), torch.ones(1))
    actual.backward()
    assert torch.isfinite(model.halt.bias.grad).all()
    assert model.halt.bias.grad.abs().sum() > 0


@pytest.mark.parametrize("halt_loss", ["expected", "mixture"])
def test_evaluation_loss_uses_hard_stopped_answer(monkeypatch, halt_loss):
    model, state = aligned_reader()
    model.eval()
    with torch.no_grad():
        model.halt.weight.zero_()
        model.halt.bias.fill_(2)
    out = model.read(state, query())
    expected = -out["hard_prob"][0, 7].clamp_min(1e-12).log() + .02 * out["reads"][0]
    original_read = model.read
    reader_steps = []

    def poison_soft_output(*args, **kwargs):
        result = original_read(*args, **kwargs)
        # A perfect soft/future answer must not repair the deployed early error.
        result["prob"] = torch.nn.functional.one_hot(torch.tensor([7]), model.vocab).float()
        return result

    monkeypatch.setattr(model, "read", poison_soft_output)
    handle = model.query.register_forward_hook(lambda *args: reader_steps.append(1))
    _, actual, _ = run_stream.rollout(model, state, [query()], "fifo",
                                      generator=torch.Generator().manual_seed(1), cost=.02,
                                      halt_loss=halt_loss)
    handle.remove()
    assert torch.allclose(actual, expected)
    assert len(reader_steps) == 1
    assert out["answer"].item() != 7


def test_detach_preserves_forward_state_and_cuts_old_gradient_graph():
    model, state = aligned_reader()
    state = model.observe(state, Event("cue", torch.tensor([0])))
    detached = state.detach()
    for field in fields(state):
        original, carried = getattr(state, field.name), getattr(detached, field.name)
        if isinstance(original, torch.Tensor):
            assert torch.equal(original, carried)
    assert detached.clock == state.clock
    assert not detached.memory.requires_grad and not detached.hidden.requires_grad
    model.eval()
    assert torch.equal(model.read(state, query())["hard_prob"], model.read(detached, query())["hard_prob"])
    old_hidden = model.initial(1, "cpu").hidden.requires_grad_()
    first = replace(model.initial(1, "cpu"), hidden=old_hidden)
    next_state = model.observe(first, Event("cue", torch.tensor([0])))
    following = model.observe(next_state.detach(), Event("cue", torch.tensor([1])))
    following.hidden.sum().backward()
    assert old_hidden.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.event_state.parameters())


def test_rollout_carries_memory_across_calls_without_reset():
    model = CacheAgent(max_reads=2)
    model.eval()
    rng = torch.Generator().manual_seed(32)
    first, second = (cycle(2, rng, noise=0, queries=1) for _ in range(2))
    with torch.no_grad():
        state, _, _ = run_stream.rollout(model, model.initial(2, "cpu"), first, "fifo", generator=rng)
        carried, _, _ = run_stream.rollout(model, state.detach(), second, "fifo", generator=rng)
        combined, _, _ = run_stream.rollout(model, model.initial(2, "cpu"), first + second, "fifo", generator=rng)
    assert carried.clock == len(first) + len(second)
    for name in ("memory", "hidden", "recent"):
        assert torch.equal(getattr(carried, name), getattr(combined, name))


def test_random_writer_does_not_change_future_worlds_or_eval_weights(monkeypatch):
    captured = []
    original_cycle = run_stream.cycle

    def record_cycle(*args, **kwargs):
        events = original_cycle(*args, **kwargs)
        captured.append([observed(e, 8, 16).clone() for e in events])
        return events

    monkeypatch.setattr(run_stream, "cycle", record_cycle)
    model = CacheAgent(max_reads=2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    args = SimpleNamespace(device="cpu", eval_batch=2, queries=1, cost=.02,
                           contexts=8, vocab=16, nodes=4)
    run_stream.evaluate(model, args, "learned", 101, cycles=3, noise=0)
    worlds_a = captured[:]
    captured.clear()
    run_stream.evaluate(model, args, "random", 101, cycles=3, noise=0)
    assert len(worlds_a) == len(captured) == 3
    for a, b in zip(worlds_a, captured):
        assert len(a) == len(b)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
