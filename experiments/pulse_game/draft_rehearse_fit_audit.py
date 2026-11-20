"""Check saved fit ledgers, real contacts, revisit rewards, and write budgets."""

from hashlib import sha256
import json

import numpy as np

from . import draft_rehearse_fit as fit


def audit():
    plan = fit.register()
    summary = json.loads((fit.EVIDENCE / "fit-summary.json").read_text(encoding="utf8"))
    if summary["protocol_sha256"] != plan or fit.digest(fit.ROOT / summary["heads"]) != summary["heads_sha256"]:
        raise ValueError("The published training artifacts do not match their hashes")
    signed_max = 0.0
    total = 0
    timings = []
    collected = {arm: [] for arm in fit.ARMS}
    for record in summary["replicas"]:
        replica, seed = record["replica"], record["seed"]
        location = fit.folder(replica)
        for path, digest in record["files"].items():
            if fit.digest(fit.ROOT / path) != digest:
                raise ValueError(f"Fit file {path} has a different hash")
        with np.load(location / "ledger.npz") as ledger, np.load(location / f"step-{fit.STEPS:04d}.npz") as final:
            metrics = ledger["metrics"]
            expected_stamps = ledger["event_sha256"]
            outcomes_by_birth = ledger["outcomes_by_birth"]
            timings.append(dict(zip(ledger["timings_columns"].tolist(),
                                    ledger["step_seconds"].sum(axis=1).tolist(), strict=True)))
            for arm in fit.ARMS:
                collected[arm].append(final[arm])
        with np.load(location / "step-0000.npz") as start:
            np.testing.assert_array_equal(np.stack([start[arm] for arm in fit.ARMS]), 0)
        assert np.isfinite(metrics).all()
        np.testing.assert_array_equal(metrics[:, :, 4], np.broadcast_to(metrics[0, :, 4], metrics[:, :, 4].shape))
        np.testing.assert_array_equal(metrics[:, :, 5], np.broadcast_to(metrics[0, :, 5], metrics[:, :, 5].shape))
        signed_max = max(signed_max, float(metrics[:, :, 6].max()))
        for end in range(fit.SEGMENT, fit.STEPS + 1, fit.SEGMENT):
            with np.load(location / f"routes-through-{end:04d}.npz") as file:
                actions, addresses, steps = file["actions"], file["addresses"], file["step"]
                assert str(file["protocol"]) == plan
                np.testing.assert_array_equal(file["columns"], fit.EVENTS)
                np.testing.assert_array_equal(file["arms"], fit.ARMS)
                np.testing.assert_array_equal(steps, np.arange(end - fit.SEGMENT + 1, end + 1))
            assert actions.shape == (len(fit.ARMS), fit.SEGMENT, len(fit.BIRTHS) * fit.WORLDS, len(fit.EVENTS))
            assert addresses.shape == (*actions.shape[:-1], fit.MAX_GROUPS)
            np.testing.assert_array_equal(actions[..., :6], np.broadcast_to(actions[0, ..., :6], actions[..., :6].shape))
            if end == fit.SEGMENT:
                np.testing.assert_array_equal(addresses[:, 0], np.broadcast_to(addresses[0, 0], addresses[:, 0].shape))
                np.testing.assert_array_equal(actions[:, 0, :, 10:],
                                              np.broadcast_to(actions[0, 0, :, 10:], actions[:, 0, :, 10:].shape))
            assert np.isin(actions[..., :10], (-1, 1)).all()
            assert np.isin(actions[..., 11:15], (-1, 1)).all()
            assert np.isin(actions[..., (10, 15)], (0, 1)).all()
            for index in (10, 15):
                balance = actions[..., index].reshape(len(fit.ARMS), fit.SEGMENT, len(fit.BIRTHS), fit.WORLDS)
                np.testing.assert_array_equal(balance.sum(-1), fit.WORLDS // 2)
                np.testing.assert_array_equal(balance, np.broadcast_to(balance[0], balance.shape))
            assert (addresses >= -1).all() and (addresses < fit.sites.net.NODES ** 2).all()
            actual_a, actual_b, actual_c = (actions[..., i] * actions[..., i + 1] for i in (2, 4, 0))
            retained = np.stack([actual_c[i] if arm == "wrong" else actual_a[i] for i, arm in enumerate(fit.ARMS)])
            np.testing.assert_array_equal(actions[..., 6] * actions[..., 7], retained)
            np.testing.assert_array_equal(actions[..., 8] * actions[..., 9], actual_b)
            writes = (addresses != -1).sum(axis=-1)
            np.testing.assert_array_equal(writes, np.broadcast_to(writes[0], writes.shape))
            np.testing.assert_allclose(writes.mean(-1), metrics[:, steps - 1, 4], rtol=2e-7)
            for time, step in enumerate(steps.tolist()):
                world, first, _ = fit.training_world(seed, step, fit.WORLDS)
                laws = np.stack((world.law_a.numpy(), world.law_b.numpy()), axis=-1)
                np.testing.assert_array_equal(actions[:, time, :, 10],
                                              np.broadcast_to(first.numpy(), actions[:, time, :, 10].shape))
                np.testing.assert_array_equal(actions[:, time, :, 13:15],
                                              actions[:, time, :, 11:13] * world.orders.numpy()[None, :, 3:] * laws)
                support = world.orders.numpy()[:, :2] * laws
                np.testing.assert_array_equal(actual_a[:, time], np.broadcast_to(support[:, 0], actual_a[:, time].shape))
                np.testing.assert_array_equal(actual_b[:, time], np.broadcast_to(support[:, 1], actual_b[:, time].shape))
                for index in range(len(fit.ARMS)):
                    payload = actions[index, time].tobytes() + addresses[index, time].tobytes()
                    assert sha256(payload).hexdigest() == expected_stamps[index, step - 1]
            correct_a, correct_b = np.moveaxis(actions[..., 13:15] > 0, -1, 0)
            joint = correct_a & correct_b
            observed = np.stack((correct_a, correct_b, joint, correct_a.astype(int) + correct_b + joint), axis=-1)
            np.testing.assert_allclose(observed.mean(axis=-2), metrics[:, steps - 1, :4], atol=2e-7)
            rooted = observed.reshape(len(fit.ARMS), fit.SEGMENT, len(fit.BIRTHS), fit.WORLDS, 4)
            np.testing.assert_allclose(rooted.mean(axis=-2), outcomes_by_birth[:, steps - 1], atol=2e-7)
            total += int(np.prod(actions.shape[:-1]))
    with np.load(fit.ROOT / summary["heads"]) as heads:
        for arm in fit.ARMS:
            np.testing.assert_array_equal(heads[arm], collected[arm])
            assert np.count_nonzero(heads[arm]) > 0
    assert total == len(fit.ARMS) * len(fit.SEEDS) * len(fit.BIRTHS) * fit.WORLDS * fit.STEPS
    fit.storage._write_json(fit.EVIDENCE / "fit-audit.json", {
        "protocol_sha256": plan, "fit_heads_sha256": summary["heads_sha256"],
        "audited_routes_across_arms_and_replicas": total,
        "per_run_numeric_dose_max_signed_error": signed_max,
        "compute_seconds_measured_before_archive_io": timings,
        "verified": ["fit checkpoint and action ledger hashes", "shared true contact logs across arms",
                     "zero-head arms produce identical first-update address and real revisit decisions",
                     "virtual taste sign transforms from the appropriate true receipt",
                     "revisit order and outcome emitted by the selected route",
                     "matching per-route counts, per-update dose and observed outcomes against aggregate curves"],
    })
    print(f"Verified {total:,} sampled routes, max signed dose error {signed_max:.3g}", flush=True)


if __name__ == "__main__":
    audit()
