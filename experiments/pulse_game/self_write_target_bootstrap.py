"""Bootstrap target-value writing on the elementary one-contact task."""
import argparse
import json

import numpy as np

from . import self_write_core_bootstrap as bootstrap
from . import self_write_target as target


BOOTSTRAP_BATCHES = 500


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, nargs="+", default=target.ROOTS)
    parser.add_argument("--view", choices=("own", "blind"), required=True)
    args = parser.parse_args()
    for root in args.roots:
        # Pair assay uses a fresh stream keyed by root; this diagnostic uses its own seed.
        graph, scores = bootstrap.train(root, BOOTSTRAP_BATCHES, view=args.view,
                                        eval_seed=92110421, field_fn=target.write_fields,
                                        step_fn=target.step)
        np.savez_compressed(target.EVIDENCE / f"bootstrap-{args.view}-{root}.npz", w=graph.w,
                            b=graph.b, mask=graph.mask,
                            metrics=json.dumps(scores), root=root, view=args.view)


if __name__ == "__main__":
    main()
