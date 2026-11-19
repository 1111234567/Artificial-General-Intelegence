"""Apply the same frozen probes to the independently confirmed joint-selected rate."""

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

from . import joint_live_integrated as integrated, joint_live_study as study
from . import live_edge_probe as probe, live_edge_probe_receipts as receipts, ring_logic_study as io


OUT = probe.OUT / 'integrated'
PROBE_ROOTS = tuple(range(69044000, 69044016))


def register(out):
    probe.register(out)
    path = out / 'registered.json'
    protocol = io.read(path)
    protocol['study'] = 'Rate selected by the joint development arm. The fitted heads are shared with the first rate-selection study, and both life and probe roots are distinct.'
    protocol['first_protocol_sha256'] = io.sha(probe.OUT / 'registered.json')
    protocol['replication_runner_sha256'] = io.sha(Path(__file__))
    io.write(path, protocol)
    replay = io.read(probe.OUT / 'receipt-replay-protocol.json')
    replay['replication'] = 'Same predetermined root offsets in fresh integrated life roots.'
    io.write(out / 'receipt-replay-protocol.json', replay)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'confirm', 'receipts'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--joint', type=Path, default=integrated.OUT)
    args = parser.parse_args()
    if set(PROBE_ROOTS) & (set(probe.PROBE_ROOTS) | set(probe.LIFE_ROOTS) | set(integrated.TEST_ROOTS)):
        raise RuntimeError('Integrated probe roots must be untouched and distinct from the first study')
    with (patch.multiple(probe, PROBE_ROOTS=PROBE_ROOTS, LIFE_ROOTS=integrated.TEST_ROOTS),
          patch.object(study, 'TEST_ROOTS', integrated.TEST_ROOTS)):
        if args.phase == 'register':
            register(args.out)
        else:
            if io.read(args.out / 'registered.json')['replication_runner_sha256'] != io.sha(Path(__file__)):
                raise RuntimeError('Registered replication runner changed')
            if args.phase == 'confirm':
                probe.confirm(args.out, args.joint)
            else:
                receipts.replay(args.out, args.joint)


if __name__ == '__main__':
    main()
