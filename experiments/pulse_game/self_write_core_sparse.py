"""Fresh-root lifetime replication following hard sparse one-contact bootstrap."""
import argparse
import hashlib
import json

from . import self_write_core_lifetime as life


ROOTS = tuple(range(201, 209))
VARIANTS = (("own", "own"), ("blind", "blind"), ("own", "blind"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, nargs="+", default=ROOTS)
    args = parser.parse_args()
    all_results = {}
    for train_view, runtime_view in VARIANTS:
        folder = life.DATA / f"sparse-{train_view}-{runtime_view}"
        rows = {}
        name = f"{train_view}_to_{runtime_view}"
        for root in args.roots:
            checkpoint = life.EVIDENCE / f"sparse-bootstrap-{train_view}-{root}.npz"
            rows[root] = life.study(root, checkpoint, life.WORLDS, life.CONTACTS, save=True,
                                    data_path=folder, view=runtime_view, sparse=True)
            simple = [rows[root][f"{regime}-active"]["terminal"] for regime in life.REGIMES]
            print(f"{name} {root} " + " ".join(f"{score:.4f}" for score in simple), flush=True)
        manifest = {}
        for path in sorted(folder.glob("*/*.npz")):
            with path.open("rb") as reader:
                manifest[path.as_posix()] = hashlib.file_digest(reader, "sha256").hexdigest()
        all_results[name] = {"roots": rows, **life.aggregate(rows), "files_sha256": manifest}
    (life.EVIDENCE / "sparse-lifetime-summary.json").write_text(
        json.dumps(all_results, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
