"""Mechanical long-path calibration; not a learned task result."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch.func import functional_call
from experiments.undirected.model import UndirectedModel
from experiments.undirected.credit import CreditConfig, TemporalCreditEngine, snapshot_parameters, weighted_loss


def main():
    torch.set_num_threads(1)
    model = UndirectedModel(input_size=2, n_jobs=1, n_actions=2, width=4,
                            mode="undirected", readout="linear").double()
    with torch.no_grad():
        model.edge_shared.zero_()
        model.edge_shared[0] = 2 ** .5
        model.encoder.weight.zero_()
        model.encoder.weight[:2, 0] = .03
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1
        model.head.weight[1, 0] = -1
        model.head.bias.zero_()
    names = ("tbptt128", "recorded512", "recorded1024", "rtrl", "rtrl_reset")
    engines = {name: TemporalCreditEngine(model, CreditConfig(method=name)) for name in names}
    records, gradients, metrics = [], {}, {}
    h = model.initial_state(1)
    for block in range(9):
        observations = torch.zeros(128, 1, 2, dtype=torch.float64)
        if block == 0:
            observations[0, 0, 0] = 1
        elapsed = tuple(0. if block == 0 and i == 0 else .25 for i in range(128))
        targets = torch.ones(128, 1, 1, dtype=torch.long)
        snapshot = snapshot_parameters(model)
        start = h.clone()
        with torch.no_grad():
            for observation, dt in zip(observations, elapsed):
                _, h = model(observation, h, dt)
        record = dict(observations=observations, targets=targets, elapsed=elapsed,
            start_h=start, end_h=h.clone(), start_tick=block * 128, end_tick=(block + 1) * 128)
        records.append((record, snapshot))
        for name, engine in engines.items():
            gradients[name], metrics[name] = engine.gradients(record, snapshot)
        if block < 8:
            with torch.no_grad():
                model.edge_shared[0].add_(1e-4 if block % 2 == 0 else -7e-5)
                model.head.bias[1].add_(1e-4)
    current = dict(model.named_parameters())

    def replay(offset=None):
        value = torch.zeros_like(records[0][0]["start_h"])
        logits = []
        for i, (record, snapshot) in enumerate(records):
            parameters = {name: old.to(current[name]) +
                (current[name] - current[name].detach() if offset is None else offset[name])
                for name, old in snapshot.items()}
            for observation, dt in zip(record["observations"], record["elapsed"]):
                prediction, value = functional_call(model, parameters, (observation, value, dt))
                if i == len(records) - 1:
                    logits.append(prediction)
        return weighted_loss(torch.stack(logits), records[-1][0]["targets"])[0], value

    loss, end = replay()
    reference = dict(zip(current, torch.autograd.grad(loss, tuple(current.values()))))
    max_error = max(float((gradients["rtrl"][name] - expected).abs().max()) for name, expected in reference.items())
    direction = {name: torch.zeros_like(value) for name, value in current.items()}
    direction["encoder.weight"][0, 0] = 1
    epsilon = 1e-6
    with torch.no_grad():
        plus, _ = replay({name: value * epsilon for name, value in direction.items()})
        minus, _ = replay({name: -value * epsilon for name, value in direction.items()})
    finite_difference = float((plus - minus) / (2 * epsilon))
    analytic = float(gradients["rtrl"]["encoder.weight"][0, 0])
    assert max_error < 1e-10
    assert torch.equal(end.detach(), h)
    assert abs(analytic - finite_difference) < 1e-7
    for name in ("tbptt128", "recorded512", "recorded1024", "rtrl_reset"):
        assert torch.count_nonzero(gradients[name]["encoder.weight"]) == 0
    assert gradients["rtrl"]["encoder.weight"].norm() > .01
    files = [Path(__file__), ROOT / "experiments/undirected/model.py", ROOT / "experiments/undirected/credit.py"]
    result = dict(format="undirected-credit-path-check-v1", total_ticks=1152, cue_tick=0,
        supervised_suffix=[1024, 1152], parameter_changes=8, update_ticks=128,
        graph="One shared undirected pair, manually placed near a slow mode; no training/behavior score.",
        derivative="Shared additive perturbation across the recorded changing-parameter path; no derivative through parameter changes or inputs.",
        rtrl_full_bptt_max_abs_error=max_error, primal_max_abs_error=float((end.detach() - h).abs().max()),
        directional_gradient=analytic, directional_finite_difference=finite_difference,
        encoder_gradient_norm={name:float(value["encoder.weight"].norm()) for name,value in gradients.items()},
        credit_path_ticks={name:metric["credit_path_ticks"] for name,metric in metrics.items()},
        sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    output = Path(__file__).with_name("credit-path-check.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
