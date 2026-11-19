"""Compare complete FP64 updates from the same two canonical learned checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import time

import psutil
import torch

from experiments.coverage_exit.fast_window import install_window
from experiments.undirected.run import NativeTrainer, file_sha256


ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 1e-10, 1e-9


def compare_tensors(left, right):
    if set(left) != set(right):
        raise ValueError("different tensor inventories")
    largest, scaled = 0., 0.
    for name, a in left.items():
        b = right[name]
        if a.dtype != b.dtype or a.shape != b.shape:
            raise ValueError(f"different tensor layout for {name}")
        difference = (a - b).abs()
        largest = max(largest, float(difference.max()))
        ratio = float((difference / (ATOL + RTOL*a.abs())).max())
        if not torch.isfinite(difference).all() or not ratio <= 1:
            raise ValueError(f"FP64 equivalence failed for {name}: {ratio=}")
        scaled = max(scaled, ratio)
    return dict(max_absolute_error=largest, max_tolerance_ratio=scaled)


def capture_update(trainer):
    """Collect the derivatives actually used, excluding this update from timing."""
    captured = {}
    original_gradients = trainer.credit.gradients
    original_aux = trainer._auxiliary_gradients
    def live(*args, **kwargs):
        gradients, metrics = original_gradients(*args, **kwargs)
        captured["live"] = {k: v.detach().clone() for k, v in gradients.items()}
        return gradients, metrics
    def auxiliary(*args, **kwargs):
        gradients, metrics = original_aux(*args, **kwargs)
        captured["auxiliary"] = {k: v.detach().clone() for k, v in gradients.items()}
        return gradients, metrics
    trainer.credit.gradients, trainer._auxiliary_gradients = live, auxiliary
    try:
        point = trainer.train_update()
    finally:
        del trainer.credit.gradients, trainer._auxiliary_gradients
    captured["clipped"] = {k:(.5*v + .5*captured["auxiliary"][k])*point["update_metrics"]["clip_scale"]
                          for k, v in captured["live"].items()}
    return point, captured


def state_checks(a, b, pa, pb):
    # The models share input data, actions, independently sampled replay, and update budget.
    for key in ("tick", "world_tick", "hidden_resets", "action_hash", "data_hash", "event_hash", "live_world_hash"):
        if pa[key] != pb[key]:
            raise ValueError(f"paired conditions differ at {key}")
    if (a.last_auxiliary["info"]["start_tick"] != b.last_auxiliary["info"]["start_tick"] or a.work != b.work):
        raise ValueError("different sampled experience or logical update budget")
    def moments(trainer):
        return {f"{name}/{key}": value.detach() for name, param in trainer.model.named_parameters()
                for key, value in trainer.optimizer.state[param].items() if torch.is_tensor(value)}
    return dict(tick=a.ticks,
        parameters=compare_tensors(dict(a.model.named_parameters()), dict(b.model.named_parameters())),
        hidden=compare_tensors({"h": a.h}, {"h": b.h}),
        adam=compare_tensors(moments(a), moments(b)),
        combined_loss_abs=abs(pa["combined_loss"] - pb["combined_loss"]))


def benchmark(interface, repeats):
    branch = f"{interface}__undirected__recorded512__fixed__warmup"
    folder = ROOT / "data/undirected/main-v1/training/seed_000" / branch
    cp = json.loads((folder/"terminal.json").read_bytes())["checkpoints"]["131072"]
    if file_sha256(cp["path"]) != cp["sha256"]:
        raise ValueError("canonical checkpoint bytes differ")
    ref = NativeTrainer.from_checkpoint(cp["path"], device="cpu")
    opt = install_window(NativeTrainer.from_checkpoint(cp["path"], device="cpu"))
    pa, ga = capture_update(ref)
    pb, gb = capture_update(opt)
    proof = {k:compare_tensors(ga[k],gb[k]) for k in ga}
    states = [state_checks(ref,opt,pa,pb)]
    timings = {"reference":[], "optimized":[]}
    for index in range(repeats):
        points = {}
        order = [("reference",ref),("optimized",opt)]
        for name, trainer in (order if index % 2 == 0 else reversed(order)):
            started = time.perf_counter()
            points[name] = trainer.train_update()
            timings[name].append(time.perf_counter() - started)
        states.append(state_checks(ref,opt,points["reference"],points["optimized"]))
    medians = {k:median(v) for k,v in timings.items()}
    return dict(interface=interface, checkpoint=cp, warmups=1, repeats=repeats, order="alternating AB/BA",
        batch=ref.config.batch_size, tick_per_update=ref.config.chunk_ticks, width=ref.config.width,
        credit=ref.config.credit, dtype=ref.config.dtype, first_update_gradient_equivalence=proof,
        all_paired_updates=states, matched_action_data_world_and_logical_work=True,
        seconds=timings, median_seconds=medians, median_speedup=medians["reference"]/medians["optimized"],
        scope="CPU complete in-memory training updates with native live world, credit, auxiliary, Adam, and trace bookkeeping. External study scheduler, file IO and evaluation are excluded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats",type=int,default=6)
    parser.add_argument("--cpu",type=int)
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists():
        parser.error("need at least two measured repeats and a new output")
    proc = psutil.Process()
    if args.cpu is not None:
        proc.cpu_affinity([args.cpu])
    torch.set_num_threads(1)
    result = dict(platform=platform.platform(), torch=torch.__version__, python=platform.python_version(),
        executable=os.sys.executable, cpu_threads=torch.get_num_threads(), affinity=proc.cpu_affinity(),
        atol=ATOL, rtol=RTOL, host_cpu_percent_before=psutil.cpu_percent(interval=.5),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(),Path(__file__).with_name("fast_window.py").resolve())})
    result["cases"] = [benchmark(interface,args.repeats) for interface in ("ports","core")]
    result["host_cpu_percent_after"] = psutil.cpu_percent(interval=.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(args.output)
    for case in result["cases"]:
        print(case["interface"], case["median_seconds"],case["median_speedup"],flush=True)


if __name__ == "__main__":
    main()
