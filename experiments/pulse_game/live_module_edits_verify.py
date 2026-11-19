"""Audit all life histories against preregistered modular starts and final masks."""

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import live_module_edits as lab, live_module_edits_study as study
from . import module_graphs, ring_logic_study as io, world


def verify(out):
    frozen = io.read(out / "summary.json")
    heads = io.read(out / "confirmed-heads.json")
    children = []
    for row in heads["rows"]:
        internal = np.zeros((lab.WIDTH, lab.WIDTH), dtype=bool)
        source, target = np.asarray(module_graphs.sample(row[2], 64).within).T
        internal[target, source] = True
        children.append(internal)
    rows = {}
    for scene in study.CONDITIONS:
        path = out / f"{scene}.npz"
        if io.sha(path) != frozen["scores_sha256"][scene]:
            raise RuntimeError("Scored trajectories were modified")
        with np.load(path, allow_pickle=False) as data:
            initial = data["initial_mask"]
            for arm in study.ARMS:
                key = f"{scene}/{arm}"
                events = data[f"{arm}/edits"].reshape(len(initial), len(study.TEST_ROOTS), study.TRIALS, 2)
                final = data[f"{arm}/final_mask"].reshape(*events.shape[:2], lab.WIDTH, lab.WIDTH)
                retained_counts = np.zeros(2, dtype=np.int64)
                additions = retirements = 0
                for head, (original, inner, change_set) in enumerate(zip(initial, children, events, strict=True)):
                    for root, decisions in enumerate(change_set):
                        current = original.copy()
                        for grow, remove in decisions[(decisions >= 0).any(axis=-1)]:
                            if grow >= 0:
                                if current.flat[grow] or grow // lab.WIDTH == grow % lab.WIDTH:
                                    raise RuntimeError("A birth was duplicate or a self-connection")
                                current.flat[grow] = True
                                additions += 1
                            if remove >= 0:
                                if not current.flat[remove] or grow == remove:
                                    raise RuntimeError("An arc retirement was impossible")
                                current.flat[remove] = False
                                if not lab.strongly_connected(lab.WIDTH, zip(*np.nonzero(current.T), strict=True)):
                                    raise RuntimeError("Retirement broke strong connectivity")
                                retirements += 1
                        np.testing.assert_array_equal(final[head, root], current)
                        retained_counts += ((current & inner).sum(), (current & (original & ~inner)).sum())
                lives = np.prod(events.shape[:2])
                rows[key] = dict(lives=int(lives), births=additions, retirements=retirements,
                                 average_retained_internal_and_bridge_arcs=(retained_counts / lives).tolist())
    io.write(out / "verified-edits.json", dict(rows=rows, all_histories_reconstructed=True,
                                              all_mutations_satisfy_scc_guard=True))


def verify_trajectory(out):
    """Record every public observation and motor pulse in a separate placebo replay."""
    recorded = []

    class RecordBody(world.PulseWorld):
        def __init__(self, tape, **kwargs):
            super().__init__(tape, record=True, **kwargs)

        def result(self):
            body = super().result()
            recorded.append(body)
            return body

    info = io.read(out / "recipe.json")
    schedule = lab.Schedule(**info["schedule"])
    start = study.heads(study.TEST_GROUPS)
    with patch.object(world, "PulseWorld", RecordBody):
        for condition in study.CONDITIONS:
            paths = study.tapes(study.TEST_ROOTS, start[2], reversal=(condition == "reversal"))
            replay = study.run(start, study.TEST_ROOTS, paths, schedule, "independent_sign")
            with np.load(out / f"{condition}.npz", allow_pickle=False) as archive:
                for key, array in replay.items():
                    np.testing.assert_array_equal(array, archive[f"independent_sign/{key}"])
    stable, reversal = recorded
    public_columns = [i for i in range(world.OBS_SIZE) if i not in (world.REWARD_POS, world.REWARD_NEG)]
    np.testing.assert_array_equal(stable.obs[..., public_columns], reversal.obs[..., public_columns])
    np.testing.assert_array_equal(stable.pulses, reversal.pulses)
    io.write(out / "verified-trajectories.json", dict(
        arm="independent_sign", public_observations_equal_except_real_reward_channels=True,
        motor_pulse_trains_equal=True, physical_and_edit_replays_exact=True,
        lives=int(stable.pulses.shape[0]), physical_requests_per_life=study.TRIALS,
        motor_slots=int(np.prod(stable.pulses.shape[:-1])),
        different_real_reward_slots=int(np.count_nonzero(stable.observed_reward != reversal.observed_reward))))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--public-trajectory", action="store_true")
    args = parser.parse_args()
    verify(args.out)
    if args.public_trajectory:
        verify_trajectory(args.out)


if __name__ == "__main__":
    main()
