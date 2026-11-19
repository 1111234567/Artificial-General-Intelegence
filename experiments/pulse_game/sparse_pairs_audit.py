"""Independently replay Q lifecycles and measure isolated whole-process memory."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from .sparse_pairs_model import BIRTH, EVIDENCE, PER_LANE, SUPPORT, WIDTH, coordinates
from .sparse_pairs_stock import receiver_stocks
from .sparse_pairs_study import ARMS, BIRTH_TEST, DATA, MODES, OUT, TAIL, TRIALS, WORLD_TEST, check, ratios, run
from .unified_live_graph import SIZE
from .unified_live_graph_study import digest, read, write


def verify_pairs(data, arm, config):
    get = lambda key: data[f"{arm}/{key}"]
    events, final = get("pair_events"), get("final_active")
    shown = get("shown") != 0
    receipt = shown.cumsum(axis=1)
    np.testing.assert_array_equal(receipt[:, -1], get("final_receipts"))
    all_lanes, receivers, firsts, seconds = coordinates(events["key"])
    exit_counts, surviving = 0, 0
    for lane in range(len(shown)):
        mask = data["initial_mask"][lane].copy()
        live = {}
        local = np.flatnonzero(all_lanes == lane)
        events_by_trial = np.searchsorted(events["trial"][local], np.arange(TRIALS + 1))
        final_life = final[(final["key"] // PER_LANE) == lane]
        later_writes = 0
        for trial, (grow, retire) in enumerate(get("edits")[lane]):
            if grow >= 0:
                mask.flat[grow] = True
            if retire >= 0:
                mask.flat[retire] = False
            for row in local[events_by_trial[trial]:events_by_trial[trial + 1]]:
                event = events[row]
                key, kind = int(event["key"]), int(event["type"])
                target, first, second = receivers[row], firsts[row], seconds[row]
                supported = mask[target, first] and mask[target, second]
                assert shown[lane, trial]
                if kind == BIRTH:
                    assert key not in live and supported
                    assert grow not in (SIZE * target + first, SIZE * target + second)
                    assert event["born"] == trial and event["born_receipt"] == receipt[lane, trial]
                    assert event["gradient"] != 0 and event["writes"] == 0
                    assert event["weight"] == np.clip(np.float32(config["pair_rate"]) * event["gradient"], -1.25, 1.25)
                    if arm != "random":
                        assert abs(event["gradient"]) >= config["policy"]["threshold"]
                    live[key] = trial, int(event["born_receipt"]), first, second, target
                else:
                    born, age, left, right, receiver = live.pop(key)
                    assert (born, age, left, right, receiver) == (event["born"], event["born_receipt"], first, second, target)
                    assert 0 <= event["writes"] <= receipt[lane, trial] - age
                    assert np.isnan(event["gradient"])
                    if kind == SUPPORT:
                        assert not supported and retire in (SIZE * target + first, SIZE * target + second)
                    elif kind == EVIDENCE:
                        assert supported
                        if arm != "random":
                            assert receipt[lane, trial] - age >= config["policy"]["grace"]
                            assert abs(event["weight"]) < config["policy"]["floor"]
                    else:
                        raise AssertionError("Unrecognized retirement reason")
                    later_writes += int(event["writes"])
                    exit_counts += 1
            assert len(live) == get("active")[lane, trial + 1]
            if retire >= 0:
                r, s = divmod(int(retire), SIZE)
                assert all(target != r or s not in (first, second) for _, _, first, second, target in live.values())
        np.testing.assert_array_equal(mask, get("final_mask")[lane])
        assert set(live) == set(final_life["key"])
        for row in final_life:
            born, age, *_ = live[row["key"]]
            assert (row["born"], row["born_receipt"]) == (born, age)
            assert 0 <= row["writes"] <= receipt[lane, -1] - age
        later_writes += int(final_life["writes"].sum())
        assert later_writes == get("writes")[lane, :, 3].sum()
        surviving += len(final_life)
    return dict(births=int((events["type"] == BIRTH).sum()), exits=exit_counts, final_installed=surviving,
                later_writes=int(get("writes")[..., 3].sum()))


def replay(out, data_root):
    check(out)
    summary_path = out / "summary.json"
    summary = read(summary_path)
    reports, prechange = {}, {}
    for mode in MODES:
        path = data_root / f"{mode}.npz"
        assert digest(path) == summary["scenarios"][mode]["raw_sha256"]
        with np.load(path, allow_pickle=False) as raw:
            data = {key: raw[key] for key in raw.files}
        before = np.arange(TRIALS)[None, :] < data["change"][:, None]
        report = {}
        for arm in ARMS:
            get = lambda key: data[f"{arm}/{key}"]
            assert not get("shown")[:, -TAIL:].any()
            assert np.array_equal(get("correct"), get("contact") == data["good_side"])
            assert not get("writes")[get("shown") == 0].any()
            for key in ("contact", "contact_tick", "taste", "shown", "correct", "edits", "writes"):
                values = get(key)[before]
                if mode == "stable":
                    prechange[arm, key] = values
                else:
                    np.testing.assert_array_equal(prechange[arm, key], values)
            report[arm] = dict(physical_trials=int(get("contact").size))
            if arm in ("guided", "random", "sham"):
                report[arm].update(verify_pairs(data, arm, summary["setup"]))
        measured_ratio = ratios(data["guided/active"], data["dense/active"])["peak_all_lives"]
        assert measured_ratio == summary["scenarios"][mode]["installed_vs_dense"]["peak_all_lives"]
        reports[mode] = report
    write(out / "checked.json", dict(summary_sha256=digest(summary_path), scenarios=reports), exclusive=True)


def stock_replay(out, data_root):
    summary_path = out / "stock-match-summary.json"
    summary = read(summary_path)
    reports, prechange = {}, {}
    for mode in MODES:
        path = data_root / f"stock-{mode}.npz"
        outcome = summary["results"][mode]
        assert digest(path) == outcome["raw_sha256"]
        with np.load(path, allow_pickle=False) as raw:
            data = {name: raw[name] for name in raw.files}
        planned, stock = receiver_stocks(data)
        np.testing.assert_array_equal(planned, data["planned"])
        np.testing.assert_array_equal(stock, data["target"])
        events = data["random/events"]
        lane, receiver = np.divmod(events["key"] // WIDTH, SIZE)
        delta = np.zeros_like(stock)
        np.add.at(delta, (lane, events["trial"], receiver), np.where(events["type"] == BIRTH, 1, -1))
        np.testing.assert_array_equal(delta.cumsum(axis=1), data["random/active"])
        final = data["random/final_active"]
        slots = np.bincount(final["key"] // WIDTH, minlength=len(stock) * SIZE).reshape(-1, SIZE)
        np.testing.assert_array_equal(slots, data["random/active"][:, -1])
        assert (events["writes"][events["type"] != BIRTH].sum() + final["writes"].sum() == data["random/writes"][..., 2].sum())
        assert not (data["random/active"] > data["random/capacity"]).any()
        assert (data["random/active"] == stock).mean() == outcome["receiver_slots_exact"]
        shown = data["random/shown"] != 0
        assert not shown[:, -TAIL:].any() and not data["random/writes"][~shown].any()
        before = np.arange(TRIALS)[None, :] < data["change"][:, None]
        for key in ("contact", "contact_tick", "taste", "shown", "correct", "edits", "writes", "active"):
            values = data[f"random/{key}"][before]
            if mode == "stable":
                prechange[key] = values
            else:
                np.testing.assert_array_equal(prechange[key], values)
        for arm, prefix in (("guided", ""), ("random", "random/")):
            correct = data[prefix + "contact"] == data["good_side"]
            np.testing.assert_array_equal(correct, data[prefix + "correct"])
            np.testing.assert_allclose(correct[:, -TAIL:].mean(), outcome["accuracy"][arm], rtol=0, atol=1e-15)
        reports[mode] = dict(receiver_cells=int(stock.size), receiver_cells_unmatched=int((data["random/active"] != stock).sum()),
                             quiet_cells_unmatched=int((data["random/active"][:, -TAIL:] != stock[:, -TAIL:]).sum()))
    write(out / "stock-match-checked.json", dict(summary_sha256=digest(summary_path), scenarios=reports), exclusive=True)


def pack(out, data_root):
    summaries = read(out / "summary.json")["scenarios"], read(out / "stock-match-summary.json")["results"]
    records = {}
    for archive, prefix, details in zip(("physical-records.zip", "stock-records.zip"), ("", "stock-"), summaries, strict=True):
        path = out / archive
        paths = [data_root / f"{prefix}{mode}.npz" for mode in MODES]
        hashes = {raw.name: digest(raw) for raw in paths}
        assert all(hashes[raw.name] == details[mode]["raw_sha256"] for raw, mode in zip(paths, MODES, strict=True))
        with ZipFile(path, "x", compression=ZIP_STORED) as handle:
            for raw in paths:
                handle.write(raw, raw.name)
        with ZipFile(path) as handle:
            assert handle.testzip() is None
            assert set(handle.namelist()) == set(hashes)
        records[archive] = dict(sha256=digest(path), members_sha256=hashes)
    write(out / "archive-digests.json", records, exclusive=True)


def working_set():
    if sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(peak_working_set_bytes=peak * (1 if sys.platform == "darwin" else 1024))

    import ctypes as ct
    from ctypes import wintypes as win

    class Counters(ct.Structure):
        _fields_ = [("cb", win.DWORD), ("page_faults", win.DWORD)] + [
            (name, ct.c_size_t) for name in ("peak", "current", "quota_peak_paged", "quota_paged", "quota_peak_nonpaged",
                                             "quota_nonpaged", "pagefile", "pagefile_peak", "private")]

    current_process = ct.windll.kernel32.GetCurrentProcess
    current_process.restype = win.HANDLE
    get_info = ct.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = win.HANDLE, ct.POINTER(Counters), win.DWORD
    get_info.restype = win.BOOL
    counters = Counters()
    counters.cb = ct.sizeof(Counters)
    if not get_info(current_process(), ct.byref(counters), counters.cb):
        raise ct.WinError()
    return dict(peak_working_set_bytes=counters.peak, current_working_set_bytes=counters.current,
                private_commit_bytes_at_read=counters.private)


def _memory_run(job):
    mode, arm, config = job
    before = working_set()
    raw = run(BIRTH_TEST, WORLD_TEST, mode, arm, config, capture_events=False, audit=False)
    after = working_set()
    return mode, arm, dict(before=before, after=after, peak_pair_array_bytes=int(raw["pair_storage_bytes"].max()),
                           final_pair_array_bytes=int(raw["pair_storage_bytes"][-1]), lanes=len(raw["birth"]))


def memory(out, workers):
    check(out)
    config = read(out / "development.json")["selected"]["setup"]
    modes = {mode: {} for mode in MODES}
    jobs = [(mode, arm, config) for mode in MODES for arm in ("dense", "guided")]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        for mode, arm, stats in pool.map(_memory_run, jobs):
            modes[mode][arm] = stats
    write(out / "process-memory.json", dict(platform=sys.platform, unit="bytes", scenarios=modes,
          protocol="Fresh process per arm/scenario, common 96-lane evaluation. Full 1,280 physical requests with output arrays, without pair event collection or archive compression. The OS-reported working set includes Python, NumPy, all body and graph state, reporting arrays, and scratch allocation; peak cannot be attributed to Q alone. The pair array byte count separately covers stored Q/cache/sampling arrays."), exclusive=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "stock-replay", "memory", "pack"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "memory":
        memory(args.out, args.workers)
    else:
        stages = {"replay": replay, "stock-replay": stock_replay, "pack": pack}
        stages[args.stage](args.out, args.data)


if __name__ == "__main__":
    main()
