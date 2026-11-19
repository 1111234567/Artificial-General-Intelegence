"""Apply the independent state-machine scorer to fresh normalization cases."""
from pathlib import Path
import json

from experiments.action_split import eval_audit as audit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/action-split/balanced-eval-audit-v1"


def main():
    audit.EVAL = ROOT / "data/action-split/balanced-eval-v1"
    audit.OUT = OUT
    audit.main()
    bindings = dict(passed=True,
        scope="Path-only reuse of the unchanged independently written tick-wise request-state-machine scorer on all30 fresh five-policy normalization-control evaluations.",
        wrapper_sha256=audit.sha(__file__), scorer_sha256=audit.sha(audit.__file__),
        evaluation_directory=str(audit.EVAL), metrics_sha256=audit.sha(OUT / "metrics.json"))
    with (OUT / "metrics_bindings.json").open("x", encoding="utf-8") as destination:
        json.dump(bindings, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
