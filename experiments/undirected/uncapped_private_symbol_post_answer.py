"""Count primary actions between a correct answer and the next NEW signal."""

from pathlib import Path
import sys

import torch

from .evolved_plasticity_world import FIRST_FINISH, N_ACTIONS, N_SYMBOLS, WAIT


_THREADS = 256
_KERNELS = {}


def _torch_counts(actions, tape):
    stage = tape.stage.unsqueeze(2)
    target_symbol = (tape.target - FIRST_FINISH).remainder(N_SYMBOLS)[:, None, None, :]
    accepted = (
        stage.eq(2)
        & actions.ge(FIRST_FINISH)
        & actions.lt(N_ACTIONS)
        & (actions - FIRST_FINISH).remainder(N_SYMBOLS).eq(target_symbol)
    )
    completed, first_tick = accepted.max(dim=1, keepdim=True)
    ticks = torch.arange(actions.size(1), device=actions.device)[None, :, None, None]
    after_answer = completed & (ticks > first_tick)

    # The following block can begin with silent ticks before that actor sees NEW.
    previous_completed = torch.cat((torch.zeros_like(completed[:1]), completed[:-1]))
    before_next_new = previous_completed & stage.eq(0)
    return ((after_answer | before_next_new) & actions.ne(WAIT)).sum(dim=(0, 1, 3))


def _compile_kernel(device):
    from ._cuda_nvrtc import compile_kernels

    name = "private_symbol_post_answer"
    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    options = (f"-DPOST_ANSWER_THREADS={_THREADS}", f"-DPOST_ANSWER_WAIT={WAIT}",
               f"-DPOST_ANSWER_FIRST_FINISH={FIRST_FINISH}", f"-DPOST_ANSWER_ACTIONS={N_ACTIONS}",
               f"-DPOST_ANSWER_SYMBOLS={N_SYMBOLS}")
    return compile_kernels(source, name, device=device, options=options)[name]


def post_answer_actions(actions, tape):
    """Return integer counts per candidate from raw [request, tick, population, actor] actions."""
    requests, ticks, population, actors = actions.shape
    device = actions.device
    if actions.numel() == 0:
        return torch.zeros(population, dtype=torch.long, device=device)
    if (device.type == "cuda" and sys.platform == "win32"
            and actions.dtype == torch.long and actions.is_contiguous()
            and tape.stage.device == device and tape.stage.dtype == torch.uint8
            and tape.stage.shape == (requests, ticks, actors) and tape.stage.is_contiguous()
            and tape.target.device == device and tape.target.dtype == torch.long
            and tape.target.shape == (requests, actors) and tape.target.is_contiguous()):
        with torch.cuda.device(device):
            kernel = _KERNELS.get(device)
            if kernel is None and not torch.cuda.is_current_stream_capturing():
                kernel = _KERNELS[device] = _compile_kernel(device)
            if kernel is not None:
                counts = torch.empty(population, dtype=torch.long, device=device)
                kernel((population,), (_THREADS,), actions, tape.stage, tape.target, counts,
                       requests, ticks, population, actors)
                return counts
    return _torch_counts(actions, tape)
