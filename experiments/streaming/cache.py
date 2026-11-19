"""A finite raw-record cache with learned retention, addressing and halting.

Training uses hard forward eviction and a straight-through soft derivative;
inference uses only hard eviction and a threshold stopping policy. Empty-slot
allocation and identifying an existing (context,key) are cache interface rules.
They do not choose relevance, teach a path or give a stopping-time label.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .stream import observed


@dataclass
class State:
    memory: torch.Tensor
    hidden: torch.Tensor
    clock: int = 0
    recent: torch.Tensor | None = None
    demand_time: torch.Tensor | None = None

    def detach(self):
        return State(self.memory.detach(), self.hidden.detach(), self.clock,
                     None if self.recent is None else self.recent.clone(),
                     None if self.demand_time is None else self.demand_time.clone())


class CacheAgent(nn.Module):
    def __init__(self, contexts=8, vocab=16, slots=8, hidden=32, max_reads=6,
                 width=64, read_dim=32):
        super().__init__()
        self.contexts, self.vocab, self.slots = contexts, vocab, slots
        self.hidden_size, self.max_reads = hidden, max_reads
        self.payload = contexts + 2 * vocab + 1
        self.event_state = nn.GRUCell(self.payload + 3, hidden)
        # A shared scorer sees the candidate record, its age and demand history.
        self.retain = nn.Sequential(nn.Linear(self.payload + hidden + 3, width), nn.Tanh(), nn.Linear(width, 1))
        self.key = nn.Linear(contexts + vocab, read_dim, bias=False)
        self.query = nn.Linear(contexts + vocab, read_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(2.3))
        self.halt = nn.Linear(2, 1)

    def initial(self, batch, device):
        return State(torch.zeros(batch, self.slots, self.payload + 2, device=device),
                     torch.zeros(batch, self.hidden_size, device=device),
                     recent=torch.full((batch, 2), -1, device=device, dtype=torch.long),
                     demand_time=torch.full((batch, self.contexts), -1., device=device))

    def observe(self, state, event):
        features = observed(event, self.contexts, self.vocab)
        # Cross-event hidden state tracks demands, never fact values or answers.
        features[:, self.contexts + self.vocab:self.payload] = 0
        h = self.event_state(features, state.hidden)
        recent = state.recent
        demand = state.demand_time
        if event.kind in {"cue", "query"}:
            recent = torch.stack((event.context,
                                  torch.where(recent[:, 0] == event.context,
                                              recent[:, 1], recent[:, 0])), -1)
            demand = demand.scatter(1, event.context[:, None], float(state.clock + 1))
        return State(state.memory, h, state.clock + 1, recent, demand)

    def write(self, state, event, policy="learned", generator=None):
        state = self.observe(state, event)
        raw = observed(event, self.contexts, self.vocab)[:, :self.payload]
        record = torch.cat((raw, raw.new_ones((raw.shape[0], 1)),
                            raw.new_full((raw.shape[0], 1), state.clock)), -1)
        memory = state.memory
        valid = memory[:, :, -2] > .5
        # Existing-key location is shared by all baselines, not a teacher signal.
        identity = self.contexts + self.vocab
        matches = ((memory[:, :, :identity] - raw[:, None, :identity]).abs().sum(-1) < .1) & valid
        existing = matches.any(-1)
        empty = ~valid
        has_empty = empty.any(-1) & ~existing
        batch = raw.shape[0]
        if policy == "learned":
            candidates = torch.cat((memory, record[:, None]), 1)
            age = torch.log1p((state.clock - candidates[:, :, -1]).clamp_min(0)) / 5
            ctx = candidates[:, :, :self.contexts].argmax(-1)
            last_requested = state.demand_time.gather(1, ctx)
            seen = (last_requested >= 0).float()
            demand_age = torch.log1p((state.clock - last_requested).clamp_min(0)) / 5
            features = torch.cat((candidates[:, :, :self.payload],
                                  torch.stack((age, demand_age, seen), -1),
                                  state.hidden[:, None].expand(-1, self.slots + 1, -1)), -1)
            score = self.retain(features).squeeze(-1)
            allowed = torch.cat((torch.where(existing[:, None], matches, valid),
                                 torch.ones(batch, 1, dtype=torch.bool, device=raw.device)), -1)
            logits = (-score).masked_fill(~allowed, -1e9)
            prob = logits.softmax(-1)
            choice = prob.argmax(-1)
            hard = F.one_hot(choice, self.slots + 1).to(raw.dtype)
            weights = hard + (prob - prob.detach()) if self.training else hard
        else:
            if policy in {"fifo", "mru"}:
                choice = memory[:, :, -1].masked_fill(~valid, float("inf")).argmin(-1)
                if policy == "mru":
                    stored_ctx = memory[:, :, :self.contexts].argmax(-1)
                    relevant = (stored_ctx[:, :, None] == state.recent[:, None]).any(-1)
                    irrelevant = valid & ~relevant
                    remove = memory[:, :, -1].masked_fill(~irrelevant, float("inf")).argmin(-1)
                    choice = torch.where(irrelevant.any(-1), remove, choice)
            elif policy == "random":
                choice = torch.randint(self.slots, (batch,), generator=generator, device=raw.device)
            elif policy == "no_write":
                choice = torch.full((batch,), self.slots, device=raw.device, dtype=torch.long)
            else:
                raise ValueError(policy)
            choice = torch.where(existing, matches.long().argmax(-1), choice)
            if policy == "mru":
                relevant_incoming = (event.context[:, None] == state.recent).any(-1)
                choice = torch.where(relevant_incoming, choice, self.slots)
            weights = F.one_hot(choice, self.slots + 1).to(raw.dtype)
        # Fill free physical slots; relevance learning starts when budget fills.
        free_choice = F.one_hot(empty.long().argmax(-1), self.slots + 1).to(raw.dtype)
        if policy != "no_write":
            fill = has_empty
            if policy == "mru":
                fill = fill & (event.context[:, None] == state.recent).any(-1)
            weights = torch.where(fill[:, None], free_choice, weights)
        replace = weights[:, :self.slots, None]
        new_memory = memory * (1 - replace) + record[:, None] * replace
        return State(new_memory, state.hidden, state.clock, state.recent, state.demand_time), weights[:, -1].detach()

    def read(self, state, event, force_reads=None):
        """No target/depth/phase consumed. Read memory without mutating it."""
        c, v = self.contexts, self.vocab
        context = F.one_hot(event.context, c).float()
        node = F.one_hot(event.key, v).float()
        memory = state.memory
        keys = F.normalize(self.key(memory[:, :, :c + v]), dim=-1)
        values = memory[:, :, c + v:c + 2 * v]
        terminal = memory[:, :, self.payload - 1]
        valid = memory[:, :, -2] > .5
        survival = node.new_ones(node.shape[0])
        prediction = node.new_zeros(node.shape)
        expected = node.new_zeros(node.shape[0])
        hard_answer = node.new_zeros(node.shape)
        alive = torch.ones(node.shape[0], device=node.device, dtype=torch.bool)
        hard_reads = torch.zeros_like(alive, dtype=torch.long)
        attention_rows, halts, candidates, masses = [], [], [], []
        for step in range(self.max_reads):
            address = F.normalize(self.query(torch.cat((context, node), -1)), dim=-1)
            logits = self.log_scale.exp().clamp(max=100) * (keys @ address[:, :, None]).squeeze(-1)
            weights = logits.masked_fill(~valid, -1e9).softmax(-1)
            node = (weights[:, None] @ values).squeeze(1)
            # An empty cache returns uniform uncertainty, not a zero-loss vector.
            node = (node + 1e-8) / (node.sum(-1, keepdim=True) + v * 1e-8)
            end = (weights * terminal).sum(-1)
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1) / max(1.0, __import__('math').log(self.slots))
            halt = self.halt(torch.stack((end, entropy), -1)).squeeze(-1).sigmoid()
            if step == self.max_reads - 1:
                halt = torch.ones_like(halt)
            expected = expected + survival
            masses.append(survival * halt)
            prediction = prediction + (survival * halt)[:, None] * node
            survival = survival * (1 - halt)
            stop = halt >= .5 if force_reads is None else torch.full_like(alive, step + 1 >= force_reads)
            stop = stop | (step == self.max_reads - 1)
            finishing = alive & stop
            hard_answer = torch.where(finishing[:, None], node, hard_answer)
            hard_reads = hard_reads + alive.long()
            alive = alive & ~stop
            attention_rows.append(weights)
            halts.append(halt)
            candidates.append(node)
            if not self.training and not alive.any():
                break
        return dict(prob=prediction, hard_prob=hard_answer, answer=hard_answer.argmax(-1), reads=hard_reads,
                    expected_reads=expected, attention=torch.stack(attention_rows, 1),
                    halt=torch.stack(halts, 1), candidates=torch.stack(candidates, 1),
                    masses=torch.stack(masses, 1))
