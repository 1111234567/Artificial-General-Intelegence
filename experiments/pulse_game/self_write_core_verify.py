"""Reconcile lifetime action receipts with the logged worlds and endpoint graphs."""
import argparse
import hashlib
import json

import numpy as np

from . import self_write_core as core
from . import self_write_core_lifetime as life


def verify_variant(folder, summary_path, *, sparse_views=None):
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    training_view, runtime_view = sparse_views or ("own", "own")
    if sparse_views is not None:
        summary = summary[f"{training_view}_to_{runtime_view}"]
    scale = summary.get("edit_scale", 1.0)
    for filename, expected in summary["files_sha256"].items():
        with open(filename, "rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise AssertionError(f"Receipt digest mismatch: {filename}")
    runs = 0
    events = 0
    for root in summary["roots"]:
        prefix = "bootstrap-own" if sparse_views is None else f"sparse-bootstrap-{training_view}"
        birth = life.load_graph(life.EVIDENCE / f"{prefix}-{root}.npz")
        world = life.worlds(int(root))
        contact_count, worlds = world["category"].shape
        for arm in core.ARMS:
            for regime in life.REGIMES:
                prefix = folder / root / f"{regime}-{arm}"
                with np.load(f"{prefix}-receipts.npz") as raw:
                    logged = {k: raw[k] for k in raw.files}
                graph = core.repeated_graph(birth, worlds)
                state = core.rest(graph)
                for index in range(contact_count):
                    left, right = world["left"][index], world["right"][index]
                    evaluator = core.advance(graph, state, core.sense(left, right), core.MOTOR_TICKS)
                    side = np.where(core.motor(evaluator) > 0, 1, -1)
                    tasted = side * world["reported"][regime][index]
                    np.testing.assert_array_equal(side, logged["side"][index])
                    np.testing.assert_array_equal(tasted, logged["actual_taste"][index])
                    np.testing.assert_array_equal(side == world["hidden"][regime][index],
                                                  logged["correct"][index])
                    pre, state = core.contact_states(graph, evaluator, left, right, side, tasted)
                    if arm == "shuffle":
                        _, writer_state = core.contact_states(
                            graph, evaluator, left, right, side,
                            tasted[world["shuffles"][index]])
                    else:
                        writer_state = state
                    edit = core.proposal(graph, pre, writer_state, world["keys"][index],
                                         view=runtime_view, sparse=sparse_views is not None)
                    edit.edge_delta *= scale
                    edit.bias_delta *= scale
                    for field, archive in (
                        ("edge_delta", "proposed_edge_delta"), ("bias_delta", "proposed_bias_delta"),
                        ("birth", "birth_address"), ("retire", "retire_address"),
                        ("birth_value", "proposed_birth_value"),
                        ("birth_probability", "birth_probability"), ("retire_probability", "retire_probability"),
                    ):
                        np.testing.assert_array_equal(getattr(edit, field), logged[archive][index])
                    updated, receipt = core.apply(graph, edit, arm=arm)
                    for field in ("birth_status", "retire_status", "edge_count"):
                        np.testing.assert_array_equal(getattr(receipt, field), logged[field][index])
                    np.testing.assert_array_equal(receipt.edge_delta,
                                                  logged["executed_tuning_edge_delta"][index])
                    np.testing.assert_array_equal(updated.w - graph.w, logged["effective_edge_delta"][index])
                    np.testing.assert_array_equal(updated.b - graph.b, logged["effective_bias_delta"][index])
                    np.testing.assert_array_equal(receipt.retired_value, logged["retired_value"][index])
                    graph = updated
                with np.load(f"{prefix}-snapshots.npz") as final:
                    for field in ("w", "b", "mask"):
                        np.testing.assert_array_equal(getattr(graph, field), final[field])
                    np.testing.assert_array_equal(state, final["h"])
                assert all(map(core.connected, graph.mask))
                if arm in ("primitive", "none"):
                    for field in ("w", "b", "mask"):
                        np.testing.assert_array_equal(getattr(graph, field)[:, core.WRITER],
                                                      getattr(birth, field)[:, core.WRITER].repeat(worlds, 0))
                runs += 1
                events += contact_count * worlds
    label = summary_path.name
    if sparse_views is not None:
        label += f" ({training_view}-to-{runtime_view})"
    print(f"{label}: checked {runs} condition/root runs and {events} physical edit rounds",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--reduced", action="store_true")
    variants.add_argument("--sparse-view", choices=("own", "blind", "own-to-blind"))
    args = parser.parse_args()
    if args.sparse_view is not None:
        views = {"own": ("own", "own"), "blind": ("blind", "blind"),
                 "own-to-blind": ("own", "blind")}[args.sparse_view]
        verify_variant(life.DATA / f"sparse-{views[0]}-{views[1]}",
                       life.EVIDENCE / "sparse-lifetime-summary.json", sparse_views=views)
    elif args.reduced:
        verify_variant(life.DATA / "reduced-amplitude",
                       life.EVIDENCE / "lifetime-reduced-amplitude-summary.json")
    else:
        verify_variant(life.DATA, life.EVIDENCE / "lifetime-summary.json")


if __name__ == "__main__":
    main()
