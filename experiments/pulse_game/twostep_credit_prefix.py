"""Verify matched observed-reward training prefixes before changing the objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    a, b = read(args.left), read(args.right)
    if not 1 <= args.rounds <= min(len(a), len(b)):
        parser.error("both logs must cover the entire requested prefix")
    fields = set(a[0]).intersection(b[0]) - {"seconds"}
    unequal = [i for i, (left, right) in enumerate(zip(a[:args.rounds], b[:args.rounds])) if any(left.get(key) != right.get(key) for key in fields)]
    print(json.dumps({
        "left": args.left.name, "right": args.right.name, "prefix_rounds": args.rounds, "recorded_fields": sorted(fields),
        "differing_rounds": unequal,
        "identical_sampled_positive_rate": all(left.get("sampled_success") == right.get("sampled_success") for left, right in zip(a[:args.rounds], b[:args.rounds])),
    }, indent=2))


if __name__ == "__main__":
    main()
