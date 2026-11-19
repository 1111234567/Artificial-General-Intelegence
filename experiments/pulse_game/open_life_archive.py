"""Package and replay compact physical pulses and tastes from the sealed experiment."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from . import open_life_study as study, ring_logic_study as io, world


PhysicalWorld = world.PulseWorld
REPLAY = "physical-actions.zip"


class RecordingWorld(PhysicalWorld):
    def __init__(self, tape, **kwargs):
        super().__init__(tape, **kwargs)
        self.commands = np.zeros((tape.n, tape.trials, world.TICKS), dtype=np.uint8)

    def step(self, pulses):
        self.commands[:, self.trial_index, self.tick_index] = pulses[:, 0] + 2 * pulses[:, 1]
        return super().step(pulses)


def packed_world(tape, lanes):
    n = len(study.TEST_ROOTS)
    fields = ("ctx", "cue", "left_id", "good_map", "distances", "wind")
    values = {key: getattr(tape, key)[:n] for key in fields}
    reference = {key: getattr(tape, key) for key in fields}
    for key, value in values.items():
        np.testing.assert_array_equal(value[lanes], reference[key])
    return values | {"seeds": np.asarray(study.TEST_ROOTS), "good_id": tape.good_id}


def serialize(archive, name, values):
    buffer = BytesIO()
    np.savez_compressed(buffer, **values)
    archive.writestr(ZipInfo(f"{name}.npz"), buffer.getvalue())


def per_root(raw, arm, case):
    physical, action = raw[f"{arm}/physical"], raw[f"{arm}/edits"]
    values = study.views(physical, case, raw["changed"])
    counts = {
        "timeouts": (physical[..., 0] < 0).sum(axis=(0, 1, 3)),
        "births": (action[..., 0] >= 0).sum(axis=(0, 1, 3)),
        "retirements": (action[..., 1] >= 0).sum(axis=(0, 1, 3)),
    }
    return {key: v.mean(axis=(0, 1)).tolist() for key, v in values.items()} | {key: v.tolist() for key, v in counts.items()}


def package(out, evidence):
    config = study.sealed_recipe(out)
    heads = study.original.heads(study.TEST_GROUPS)
    manifest = io.read(out / "confirmation.json")
    evidence.mkdir(parents=True, exist_ok=True)
    temporary = evidence / f"{REPLAY}.partial"
    participant = {}
    with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
        for case in study.CONDITIONS:
            native = out / f"{case}.npz"
            if manifest["files"][case] != io.sha(native):
                raise RuntimeError("Native physical archive hash changed")
            data, _ = study.tapes(study.TEST_ROOTS, heads[2], case, "confirmation")
            tape, _, lanes, _, _, _ = data
            values = packed_world(tape, lanes)
            with np.load(native, allow_pickle=False) as raw:
                for name in ("pairs", "flip_trial", "changed", "placebo_sign"):
                    values[name] = raw[name]
                participant[case] = {}
                for arm in study.ARMS:
                    bodies = []

                    def construct(*args, **kwargs):
                        body = RecordingWorld(*args, **kwargs)
                        bodies.append(body)
                        return body

                    with patch.object(world, "PulseWorld", construct):
                        result = study.run(heads, study.TEST_ROOTS, data, arm, config)
                    for key in ("physical", "edits", "event_records", "weight_step"):
                        np.testing.assert_array_equal(result[key], raw[f"{arm}/{key}"])
                    body = bodies[0]
                    values[f"{arm}/commands"] = body.commands
                    values[f"{arm}/physical"] = result["physical"]
                    values[f"{arm}/observed_taste"] = body.result().observed_reward
                    participant[case][arm] = per_root(raw, arm, case)
                    print("archived", case, arm, flush=True)
            serialize(archive, case, values)
    verify(temporary)
    destination = evidence / REPLAY
    temporary.replace(destination)
    io.write(evidence / "per-world.json", dict(
        root_order=study.TEST_ROOTS, groups=study.TEST_GROUPS, scope="Equal weight over all eight groups and three rules",
        cases=participant))
    io.write(evidence / "archive.json", dict(
        pulse_archive_sha256=io.sha(destination), per_world_sha256=io.sha(evidence / "per-world.json"),
        confirmation_sha256=io.sha(out / "confirmation.json"),
        archive_source_sha256=io.sha(Path(__file__)),
        commands_encoding="bit 0 left, bit 1 right. Arrays cover every physical motor slot, including receipt slot.",
        physical_fields=("contact_side", "contact_tick", "total_actual_signed_taste"),
        taste="observed_taste gives the true physical one-shot pulse by tick. For placebo, signed inputs to learning use placebo_sign only at its own actual contact.",
        replay="Every arm reuses the registered recipe and runs its own body. The archived command stream reconstructed all physical outputs and emitted taste pulses exactly. Native physical, edit, edge-generation and write arrays also matched during recording."))


def verify(path):
    with ZipFile(path) as archive:
        for case in study.CONDITIONS:
            with np.load(BytesIO(archive.read(f"{case}.npz")), allow_pickle=False) as data:
                roots = data["seeds"]
                lanes = np.arange(len(data["good_id"])) % len(roots)
                fields = ("ctx", "cue", "left_id", "good_map", "distances", "wind")
                tape = world.Tape(tuple(roots[lanes]), good_id=data["good_id"], **{key: data[key][lanes] for key in fields})
                for arm in study.ARMS:
                    body = PhysicalWorld(tape, feedback="true")
                    for commands in data[f"{arm}/commands"].transpose(1, 2, 0).reshape(-1, tape.n):
                        body.step(np.column_stack(((commands & 1) != 0, (commands & 2) != 0)))
                    outcome = body.result()
                    expected = data[f"{arm}/physical"].reshape(tape.n, tape.trials, 3)
                    actual = np.stack((outcome.contact_side, outcome.contact_tick, outcome.observed_reward.sum(axis=-1)), axis=-1)
                    np.testing.assert_array_equal(expected, actual)
                    np.testing.assert_array_equal(data[f"{arm}/observed_taste"], outcome.observed_reward)
            print("replayed", case, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.evidence / REPLAY)
    else:
        package(args.out, args.evidence)


if __name__ == "__main__":
    main()
