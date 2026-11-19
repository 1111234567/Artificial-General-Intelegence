"""Measure first legal answers while reusing the sealed continuous rollout."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock

import torch

from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, N_SYMBOLS, READY, _int, _receipt_table
from .plasticity_continuous_world import VISITS, _FixedGuess, rollout_continuous
from .uncapped_bf16_backend import _BF16Policy, _device_tape
from .uncapped_cuda_backend import (
    CUDAGraphUnavailable, _Capture, _SHAPES_PER_DEVICE, _rollout_with_preloaded_receipts, _stack_genomes,
)
from .uncapped_plastic_graph import UncappedGraphBatch


_CACHE = {}
_CACHE_LOCK = Lock()


class _RecordPrimaryActions:
    def __init__(self, policy, tape, population):
        self.policy = policy
        requests, ticks, actors = tape.stage.shape
        self.actions = torch.empty((requests, ticks, population, actors), dtype=torch.long,
                                   device=tape.public.device)
        self.frames = iter(self.actions.view(requests * ticks, population, actors).unbind())

    def act(self, obs):
        action = self.policy.act(obs)
        next(self.frames).copy_(action[..., 0])
        return action


def _answer_metrics(actions, tape):
    finishes = actions.ge(FIRST_FINISH) & actions.lt(N_ACTIONS)
    public_ready = tape.public[..., 0, READY].bool().unsqueeze(2)
    legal = finishes & public_ready & tape.stage.eq(2).unsqueeze(2)
    # The first legal action precedes retirement, which requires a legal correct
    # action. Later actions cannot change its classification.
    attempted, first_tick = legal.max(dim=1)
    first_action = actions.gather(1, first_tick.unsqueeze(1)).squeeze(1)
    first_correct = first_action.eq(tape.target.unsqueeze(1))
    premature = (finishes & ~public_ready & tape.stage.eq(1).unsqueeze(2)).any(1)
    return {
        "wrong_answer_rate": (attempted & ~first_correct).float().mean((0, 2)),
        "first_finish_correct": (attempted & first_correct).float().mean((0, 2)),
        "premature_request_rate": premature.float().mean((0, 2)),
    }


def _observed_rollout(policy, tape, population, rollout):
    observer = _RecordPrimaryActions(policy, tape, population)
    metrics = rollout(observer, tape, population)
    return {**metrics, **_answer_metrics(observer.actions, tape)}


@torch.no_grad()
def eager_rollout(policy, tape, population):
    """Run a policy with the sealed world and report request-level answer rates."""
    return _observed_rollout(policy, tape, population, rollout_continuous)


def _bf16_rollout(policy, tape, table):
    rollout = _rollout_with_preloaded_receipts(table)
    metrics = _observed_rollout(policy, tape, policy.population, rollout)
    return {**policy.summary(), **metrics}


class _AnswerCapture(_Capture):
    def __init__(self, birth, tape, actors, device):
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.birth = {key: tensor.to(device) for key, tensor in birth.items()}
            self.tape = _device_tape(tape, device)
            self.receipt_table = _receipt_table(device).bfloat16()
            self.policy = _BF16Policy(self.birth, actors, device)
            _bf16_rollout(self.policy, self.tape, self.receipt_table)

        self.graph = torch.cuda.CUDAGraph()
        # Tracing rebinds both attributes. Replay still needs the reset buffers.
        self.entry_w = self.policy.w
        self.entry_activity = self.policy.activity
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.policy.reset(self.birth)
                self.results = _bf16_rollout(self.policy, self.tape, self.receipt_table)
        except RuntimeError as error:
            raise CUDAGraphUnavailable(f"answer feedback CUDA Graph capture failed: {error}") from error


class AnswerEvaluation:
    """Evaluate all candidates on the same supplied schedule and observation tape."""

    def __init__(self, schedule):
        self.schedule = schedule

    @torch.no_grad()
    def __call__(self, genomes, *, seed, actors, device, visits=VISITS, mode="full", capture=None):
        actors = _int("actors", actors, 1)
        device = torch.device(device)
        if mode != "full":
            raise ValueError("uncapped graph requires full plasticity")
        if capture is None:
            capture = device.type == "cuda" and len(genomes) >= 32

        if device.type != "cuda":
            if capture:
                raise CUDAGraphUnavailable("answer feedback CUDA Graph replay requires a CUDA device")
            tape = self.schedule(seed=seed, actors=actors, device=device, visits=visits)
            policy = UncappedGraphBatch(genomes, actors=actors, device=device, mode=mode, seed=seed)
            metrics = eager_rollout(policy, tape, len(genomes))
            return {**policy.summary(), **metrics}

        if not torch.cuda.is_available() or torch.is_autocast_enabled("cuda"):
            raise CUDAGraphUnavailable("answer feedback requires CUDA with autocast disabled")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise CUDAGraphUnavailable(f"{device} lacks native CUDA BF16 support")
            nodes, birth = _stack_genomes(genomes)
            tape = self.schedule(seed=seed, actors=actors, device="cpu", visits=visits)
            tape = replace(tape, public=tape.public.bfloat16())
            if not capture:
                policy = _BF16Policy({key: tensor.to(device) for key, tensor in birth.items()}, actors, device)
                return _bf16_rollout(policy, _device_tape(tape, device), _receipt_table(device).bfloat16())
            return _replay(birth, tape, len(genomes), nodes, actors, device)

    @torch.no_grad()
    def evaluate_baseline(self, *, seed, actors, device, visits=VISITS, fixed_symbol=0):
        """Use the public READY and version cues to guess one fixed symbol."""
        fixed_symbol = _int("fixed_symbol", fixed_symbol, 0)
        if fixed_symbol >= N_SYMBOLS:
            raise ValueError(f"fixed_symbol must be below {N_SYMBOLS}")
        tape = self.schedule(seed=seed, actors=actors, device=device, visits=visits)
        return eager_rollout(_FixedGuess(fixed_symbol), tape, 1)


def _replay(birth, tape, population, nodes, actors, device):
    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    key = (population, nodes, tape.public.shape, reduction)
    with _CACHE_LOCK:
        caller_stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            raise CUDAGraphUnavailable("answer feedback CUDA Graph replay cannot nest in an active capture")
        cache = _CACHE.setdefault(device.index, OrderedDict())
        captured = cache.get(key)
        if captured is None:
            if len(cache) >= _SHAPES_PER_DEVICE:
                _, retired = cache.popitem(last=False)
                retired.stream.synchronize()
                del retired
            captured = _AnswerCapture(birth, tape, actors, device)
            cache[key] = captured
        cache.move_to_end(key)
        return captured.evaluate(birth, tape, caller_stream)
