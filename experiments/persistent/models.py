"""Small controlled models for memory retention and training through a read loop.

Raw facts are retained losslessly. Allocation/write policy is deliberately fixed;
only addressing, recurrent computation, output and (optionally) judgment learn.
This is a mechanism experiment, not a language agent or a lifelong learner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Output:
    logits: torch.Tensor             # [batch, computation steps, vocabulary]
    judgment: torch.Tensor           # logits for candidate correctness
    attention: torch.Tensor | None


class MemoryLoop(nn.Module):
    """One shared content-addressed reader, reused across all internal steps.

    'detach' changes backward connectivity only, never the numerical forward.
    No intermediate path labels or correctness labels enter forward().
    """

    def __init__(self, vocab=64, dim=64, mode="joint", judge_feedback=True):
        super().__init__()
        if mode not in {"single", "detach", "joint", "judge"}:
            raise ValueError(mode)
        self.vocab, self.dim, self.mode = vocab, dim, mode
        self.judge_feedback = judge_feedback
        self.embedding = nn.Embedding(vocab, dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.Tanh(),
                                    nn.Linear(dim, dim))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.judge = nn.Sequential(nn.Linear(2 * dim + 4, dim), nn.Tanh(),
                                   nn.Linear(dim, 1))

    def forward(self, keys, values, query, hops, *, steps=None,
                feedback_override=None):
        steps = int(hops.max().item()) if steps is None else steps
        emb = F.normalize(self.embedding.weight, dim=-1)
        memory_keys = F.normalize(self.key(emb[keys]), dim=-1)
        memory_values = emb[values]
        original = emb[query]
        state = original
        feedback = state.new_zeros((state.shape[0], 1))
        scale = self.log_scale.exp().clamp(max=100)
        logits, scores, attentions = [], [], []
        for step in range(steps):
            if step and self.mode == "detach":
                state = state.detach()
                feedback = feedback.detach()
            address = F.normalize(self.query(state), dim=-1)
            weights = (memory_keys @ address.unsqueeze(-1)).squeeze(-1)
            weights = (scale * weights).softmax(-1)
            read = (weights.unsqueeze(1) @ memory_values).squeeze(1)
            delta = self.update(torch.cat((state, read, feedback), dim=-1))
            state = F.normalize(read + delta, dim=-1)
            prediction = 20.0 * state @ emb.T
            confidence = prediction.softmax(-1).amax(-1, keepdim=True)
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            step_info = torch.stack((torch.full_like(hops, step + 1), hops), -1)
            features = torch.cat((state, original, confidence,
                                  entropy / math.log(max(2, keys.shape[1])),
                                  step_info.to(state.dtype) / 8.0), -1)
            score = self.judge(features).squeeze(-1)
            if self.mode == "judge" and self.judge_feedback:
                feedback = score.sigmoid().unsqueeze(-1)
                if feedback_override == "zero":
                    feedback = torch.zeros_like(feedback)
                elif feedback_override == "shuffle":
                    feedback = feedback.roll(1, 0)
            logits.append(prediction)
            scores.append(score)
            attentions.append(weights)
        return Output(torch.stack(logits, 1), torch.stack(scores, 1),
                      torch.stack(attentions, 1))


class CompressedGRU(nn.Module):
    """All previously observed facts pass through one fixed-size GRU state.

    The query is only revealed after facts and NOOP delay. Query/answer iterations
    use a shared decoder, with no subsequent access to the fact records.
    """

    def __init__(self, vocab=64, dim=64, hidden=128):
        super().__init__()
        self.vocab, self.dim, self.hidden = vocab, dim, hidden
        self.embedding = nn.Embedding(vocab, dim)
        self.encoder = nn.GRU(2 * dim + 1, hidden, batch_first=True)
        self.start = nn.Linear(hidden + dim, hidden)
        self.compute = nn.GRUCell(dim + 2, hidden)
        self.answer = nn.Linear(hidden, vocab)
        self.judge = nn.Linear(hidden, 1)

    def encode_chunk(self, records, state=None):
        """Carry the returned state into the next chunk; no implicit reset."""
        return self.encoder(records, state)

    def forward(self, keys, values, query, hops, *, gap=0, steps=None, **_):
        batch, count = keys.shape
        facts = torch.cat((self.embedding(keys), self.embedding(values),
                           self.embedding.weight.new_ones((batch, count, 1))), -1)
        if gap:
            facts = torch.cat((facts, facts.new_zeros((batch, gap, facts.shape[-1]))), 1)
        _, state = self.encode_chunk(facts)
        original = self.embedding(query)
        state = torch.tanh(self.start(torch.cat((state[-1], original), -1)))
        steps = int(hops.max().item()) if steps is None else steps
        logits, scores = [], []
        for step in range(steps):
            info = torch.stack((torch.full_like(hops, step + 1), hops), -1)
            state = self.compute(torch.cat((original, info.to(state.dtype) / 8), -1), state)
            logits.append(self.answer(state))
            scores.append(self.judge(state).squeeze(-1))
        return Output(torch.stack(logits, 1), torch.stack(scores, 1), None)


def selected_logits(output, hops):
    return output.logits[torch.arange(hops.numel(), device=hops.device), hops - 1]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
