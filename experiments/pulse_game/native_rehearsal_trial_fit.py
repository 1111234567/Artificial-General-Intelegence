"""Fit and seal the equally sized trial and candidate-masked graphs."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import native_rehearsal_fit as original
from . import native_rehearsal_trial as trial
from . import interleave_revisit_train as storage
from . import edge_site_train_fit as ancestors


DATA = original.DATA / "trial-training"
PLAN = original.EVIDENCE / "trial-plan.md"
PARENT_PATHS = ancestors.parent_paths()


def signature(kind, birth):
    source = (__file__, trial.__file__, original.__file__, trial.live.__file__, PLAN, PARENT_PATHS[str(birth)])
    return dict(kind=kind, birth=birth, steps=original.STEPS, batch=original.BATCH,
                sha256={Path(path).name: original.digest(path) for path in source})


def destination(kind, birth):
    return DATA / f"{kind}-{birth}.pt"


def fit(kind, birth, *, smoke=False):
    torch.set_num_threads(1)
    torch.manual_seed(birth)
    model = trial.TrialGraph(birth, kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    target = destination(kind, birth)
    identity = signature(kind, birth)
    start = 0
    ledger = []
    if target.is_file() and not smoke:
        saved = torch.load(target, map_location="cpu", weights_only=True)
        if saved["signature"] != identity:
            raise ValueError(f"Trial inputs differ from saved checkpoint {target}")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        ledger = saved["ledger"].tolist()

    started = perf_counter()
    total = 6 if smoke else original.STEPS
    for step in range(start + 1, total + 1):
        ledger.append(original.update(model, optimizer, kind, birth, step, original.BATCH))
        if not smoke and (step % 128 == 0 or step == total):
            target.parent.mkdir(parents=True, exist_ok=True)
            saved = dict(signature=identity, step=step, state=model.state_dict(),
                         optimizer=optimizer.state_dict(), ledger=torch.tensor(ledger))
            storage._atomic(target, lambda path: torch.save(saved, path))
            print(kind, birth, step, np.mean(ledger[-32:], 0)[:4].round(3).tolist(),
                  f"{perf_counter() - started:.2f}s", flush=True)
    if smoke:
        print(kind, birth, "smoke", np.asarray(ledger)[-1, :4].tolist(),
              f"{perf_counter() - started:.2f}s", flush=True)


def load_final(kind, birth):
    target = destination(kind, birth)
    saved = torch.load(target, map_location="cpu", weights_only=True)
    if saved["signature"] != signature(kind, birth) or saved["step"] != original.STEPS:
        raise ValueError(f"Trial checkpoint is unfinished or based on different inputs: {target}")
    model = trial.TrialGraph(birth, kind)
    model.load_state_dict(saved["state"])
    return model.eval(), saved["ledger"], original.digest(target)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=trial.KINDS)
    parser.add_argument("birth", type=int, choices=original.BIRTHS, nargs="?")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for birth in original.BIRTHS if args.birth is None else (args.birth,):
        fit(args.kind, birth, smoke=args.smoke)


if __name__ == "__main__":
    main()
