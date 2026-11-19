"""Causal, randomized streams. Labels are separate from observable events.

Two randomly named contexts share keys but disagree about their chains. Other
contexts generate interference. A third phase returns to either previous
context before any of its facts are refreshed. The same context then changes.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class Event:
    kind: str
    context: torch.Tensor
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    terminal: torch.Tensor | None = None
    target: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    phase: str = ""


def oracle(keys, values, terminal, query):
    """Independent CPU dictionary interpreter; never supplied to a model."""
    answer, depths = [], []
    for ks, vs, ts, q in zip(keys.tolist(), values.tolist(), terminal.tolist(), query.tolist()):
        table = dict(zip(ks, zip(vs, ts)))
        for depth in range(1, len(ks) + 2):
            q, stop = table[q]
            if stop:
                answer.append(q)
                depths.append(depth)
                break
        else:
            raise ValueError("unterminated graph")
    return torch.tensor(answer, device=query.device), torch.tensor(depths, device=query.device)


def cycle(batch, generator, *, contexts=8, vocab=16, nodes=4,
          noise=8, queries=2, return_to="random", device="cpu"):
    """Generate one task cycle, not a state-reset boundary.

    A/B identities and relevant keys change each cycle. All contexts use the
    same randomly chosen keys, so key identity cannot indicate relevance.
    """
    if contexts < 3 or nodes > vocab or nodes < 2 or vocab < 6:
        raise ValueError("invalid task dimensions")
    rand = lambda *shape: torch.rand(shape, generator=generator, device=device)
    randint = lambda high, shape: torch.randint(high, shape, generator=generator, device=device)
    names = rand(batch, contexts).argsort(-1)
    keys = rand(batch, vocab).argsort(-1)[:, :nodes]
    rows = torch.arange(batch, device=device)

    endpoints = rand(batch, vocab).argsort(-1)[:, :6]

    def graph(finals):
        order = rand(batch, nodes).argsort(-1)
        path = keys.gather(1, order)
        split = 1 + randint(nodes - 1, (batch,))
        position = torch.arange(nodes, device=device)[None].expand(batch, -1)
        first = position < split[:, None]
        labels = torch.where(first, finals[:, :1], finals[:, 1:])
        depth = torch.where(first, split[:, None] - position, nodes - position)
        terminal = (depth == 1).float()
        values = torch.cat((path[:, 1:], finals[:, 1:]), 1)
        values = torch.where(terminal.bool(), labels, values)
        # Two different terminal answers prevent simply finding any terminal.
        return path, values, terminal, labels, depth

    ga, gb, gc = graph(endpoints[:, :2]), graph(endpoints[:, 2:4]), graph(endpoints[:, 4:6])
    if return_to == "a":
        choose_b = torch.zeros(batch, dtype=torch.bool, device=device)
    elif return_to == "b":
        choose_b = torch.ones(batch, dtype=torch.bool, device=device)
    else:
        choose_b = randint(2, (batch,)).bool()
    returned = tuple(torch.where(choose_b[:, None], b, a) if a.ndim == 2
                     else torch.where(choose_b, b, a) for a, b in zip(ga, gb))
    cr = torch.where(choose_b, names[:, 1], names[:, 0])
    events = []

    def distract():
        ctx = names[rows, 2 + randint(contexts - 2, (batch,))]
        key = keys[rows, randint(nodes, (batch,))]
        events.append(Event("fact", ctx, key, randint(vocab, (batch,)),
                            randint(2, (batch,)).float()))

    def facts(ctx, g):
        order = rand(batch, nodes).argsort(-1)
        for i in range(nodes):
            at = order[:, i]
            events.append(Event("fact", ctx, g[0][rows, at], g[1][rows, at], g[2][rows, at]))
            distract()

    def ask(ctx, g, phase):
        for _ in range(queries):
            start = randint(nodes, (batch,))
            events.append(Event("query", ctx, g[0][rows, start], target=g[3][rows, start].clone(),
                                depth=g[4][rows, start].clone(), phase=phase))
            distract()

    for ctx, g, phase in ((names[:, 0], ga, "a_acquire"), (names[:, 1], gb, "b_acquire")):
        events.append(Event("cue", ctx))
        facts(ctx, g)
        for _ in range(noise):
            distract()
        ask(ctx, g, phase)
    events.append(Event("cue", cr))
    for _ in range(noise):
        distract()
    ask(cr, returned, "return")  # no re-presentation before these probes
    facts(cr, gc)
    for _ in range(noise):
        distract()
    ask(cr, gc, "revision")
    return events


def observed(event, contexts, vocab):
    """All and only observable event fields, with no phase/depth/answer."""
    ctx = F.one_hot(event.context, contexts).float()
    empty = ctx.new_zeros((ctx.shape[0], vocab))
    key = empty if event.key is None else F.one_hot(event.key, vocab).float()
    value = empty if event.value is None else F.one_hot(event.value, vocab).float()
    terminal = ctx.new_zeros((ctx.shape[0], 1)) if event.terminal is None else event.terminal[:, None]
    kind = ctx.new_zeros((ctx.shape[0], 3))
    kind[:, {"cue": 0, "fact": 1, "query": 2}[event.kind]] = 1
    return torch.cat((ctx, key, value, terminal, kind), -1)
