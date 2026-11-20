"""Two regions of one recurrent graph, sharing current physical parameters."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from . import native_rehearsal_graph as live


REGION = 20
LIMIT = 1.15
KINDS = ("trial", "no_trial")


@dataclass
class State(live.State):
    candidate: torch.Tensor


class TrialGraph(live.NativeGraph):
    """Only the ordinary 10 sensory ports and the ordinary main motor ports touch the environment."""

    def __init__(self, birth, kind):
        if kind not in KINDS:
            raise ValueError(f"Unknown trial condition {kind!r}")
        super().__init__(birth, "copy")
        # The second region borrows the first region's recurrence instead of owning a second base matrix.
        self.return_weight = nn.Parameter(self.weight[REGION:, :REGION].detach().clone())
        self.weight = nn.Parameter(self.weight[:REGION].detach().clone())
        self.adopt_bias = nn.Parameter(torch.full((REGION,), 2.0))
        self.adopt_receiver = nn.Parameter(torch.full((REGION,), 0.1))
        self.adopt_sender = nn.Parameter(torch.full((REGION,), -0.1))
        self.read_candidate = kind == "trial"

    def start(self, batch):
        s = super().start(batch)
        return State(s.h, s.fast, s.trace, s.fast[:, :REGION, :REGION].clone())

    def tick(self, s, x, *, feedback_override=None, disable_feedback=False, candidate_override=None,
             read_override=None, birth_only=False, adoption_off=False):
        n = REGION
        primary, shadow = s.h.split(n, dim=1)
        main_fast = s.fast[:, :n, :n]
        main_weights = self.weight * self.mask[:n]
        intra = main_weights[:, :n]
        primary_current = F.linear(s.h, main_weights) + torch.bmm(s.fast[:, :n], s.h[:, :, None]).squeeze(2)
        back = F.linear(shadow, main_weights[:, n:]) + torch.bmm(s.fast[:, :n, n:], shadow[:, :, None]).squeeze(2)

        if feedback_override is not None or disable_feedback:
            replacement = torch.zeros_like(back) if feedback_override is None else feedback_override
            primary_current = primary_current - back + replacement

        projected = s.candidate if candidate_override is None else candidate_override
        enabled = self.read_candidate if read_override is None else read_override
        estimated_fast = (main_fast * (not birth_only) + projected * enabled).clamp(-LIMIT, LIMIT)
        bridge_weights = self.return_weight * self.mask[n:, :n]
        shadow_current = (F.linear(primary, bridge_weights) + F.linear(shadow, intra)
                          + torch.bmm(s.fast[:, n:, :n], primary[:, :, None]).squeeze(2)
                          + torch.bmm(estimated_fast, shadow[:, :, None]).squeeze(2))
        current = torch.cat((primary_current, shadow_current), dim=1)
        h = s.h + self.leak_logit.sigmoid() * (torch.tanh(current + self.bias + F.pad(x, (0, self.width - 10))) - s.h)
        trace = s.trace + self.keep_logit.neg().sigmoid() * (s.h - s.trace)

        # Evaluate the already available draft on the shadow before choosing how much should enter the body.
        shadow_now = h[:, n:]
        receiver = self.adopt_bias + self.adopt_receiver * shadow_now
        sender = self.adopt_sender * shadow_now
        adoption = (receiver[:, :, None] + sender[:, None, :]).sigmoid()
        accepted = s.candidate * adoption * (not adoption_off)
        mod = F.linear(h, self.gate * self.mask).tanh()
        post = (h + self.post_bias).tanh() * mod * self.edit_rate.sigmoid()
        new = post[:, :, None] * trace[:, None, :] * self.mask
        # The main recurrence gets the gated, pending draft. Bridges keep their ordinary live writes.
        proposed = torch.cat((torch.cat((accepted, new[:, :n, n:]), dim=2),
                              F.pad(new[:, n:, :n], (0, n))), dim=1)
        fast = (s.fast + proposed).clamp(-LIMIT, LIMIT)
        return State(h, fast, trace, new[:, :n, :n]), back
