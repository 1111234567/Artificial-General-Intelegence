"""Rebuild all control supports and independently replay frozen physical scores on CPU."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import archival_logic_model as model, archival_logic_study as study
from . import ring_logic_study as shared, ring_logic_world as env, world


def _advance(observations, state, weights, matrix, retain):
    values = []
    for tick, sensory in enumerate(observations.unbind(dim=1)):
        projected = sensory @ weights["drive"] + weights["bias"]
        candidate, gate = projected.chunk(2, dim=-1)
        candidate = (candidate + state @ matrix.T).tanh()
        gate = gate.sigmoid()
        state = gate * state + (1 - gate) * candidate
        if tick >= retain:
            values.append(state @ weights["readout"] + weights["offset"])
    return values, state


@torch.no_grad()
def verify(out):
    registration, frozen = study.frozen(out)
    scored = shared.read(out / "scored.json")
    if shared.sha(out / "scores.npz") != scored["scores_sha256"]:
        raise RuntimeError("Expected frozen physical scores were modified")
    graphs = shared.read(out / "graphs.json")
    original = study.load_support()
    np.testing.assert_array_equal(original, study.decode_graph(graphs["archive"]))
    for row in graphs["controls"]:
        duplicate, trace = study.switched(original, row["switch_seed"])
        np.testing.assert_array_equal(duplicate, study.decode_graph(row))
        if trace != row["trace"]:
            raise RuntimeError("The control membership trace did not replay exactly")
    choices = (0, study.STARTS - 1)
    replays = []
    with np.load(out / "scores.npz", allow_pickle=False) as scores:
        for profile in ("standard", "long"):
            roots = registration["world_roots"][profile][:32]
            tape = env.body_tape(roots)
            prefix, query, _ = env.sensory(roots, profile)
            sensory_ticks = env.TIMING[profile][0]
            p = torch.from_numpy(prefix)
            q = torch.from_numpy(query[:, :, :sensory_ticks + world.TICKS].reshape(-1, sensory_ticks + world.TICKS, env.SENSES))
            for artifact in frozen["checkpoints"]:
                step = artifact["updates"]
                weights = torch.load(out / artifact["path"], weights_only=True, map_location="cpu")
                reference = scores[f"{step}-{profile}"]
                for start in choices:
                    for rule_index, rule in enumerate(env.RULES):
                        for arm_index, arm in enumerate(model.ARMS):
                            support = original if arm == "archive" else study.decode_graph(graphs["controls"][start])
                            lane = (start * len(env.RULES) + rule_index) * len(model.ARMS) + arm_index
                            row = {key: param[lane] for key, param in weights.items()}
                            matrix = torch.zeros(study.WIDTH, study.WIDTH)
                            matrix[torch.from_numpy(support)] = row["edges"]
                            _, state = _advance(p, torch.zeros(len(p), study.WIDTH), row, matrix, p.shape[1])
                            logits, _ = _advance(q, state.repeat_interleave(len(env.PAIRS), dim=0), row,
                                                 matrix, sensory_ticks)
                            actual = env.contacts(tape, torch.stack(logits).ge(0).numpy(), rule_index)
                            expected = reference[start, rule_index, arm_index, :len(roots)].reshape(-1, 3)
                            replays.append({"steps": step, "profile": profile, "start": start,
                                            "rule": rule, "arm": arm, "physical_requests": len(actual),
                                            "different_side": int(np.count_nonzero(actual[:, 0] != expected[:, 0])),
                                            "different_tick": int(np.count_nonzero(actual[:, 1] != expected[:, 1])),
                                            "different_reward": int(np.count_nonzero(actual[:, 2] != expected[:, 2]))})
    shared.write(out / "verification.json", {"switch_traces_exact": len(graphs["controls"]),
                                            "implementation": "CPU separate recurrence, gated update and physical motor, no TF32",
                                            "rows": replays})
    print("replayed", len(replays), "rows", "side/tick/reward differences",
          [sum(row[key] for row in replays) for key in
           ("different_side", "different_tick", "different_reward")])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.DEFAULT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    verify(args.out)


if __name__ == "__main__":
    main()
