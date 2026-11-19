"""Replay real receipts and independently measure coefficient writes by edge generation."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import joint_live_model as joint, joint_live_study as study
from . import live_edge_probe as probe, live_edge_probe_history as history
from . import live_module_edits_study as starts, ring_logic_study as io


ROOT_OFFSETS = (0, 5, 10, 15)
FIELDS = ('lane', 'receiver', 'sender', 'cohort', 'born', 'end', 'active_at_end', 'birth_raw', 'end_raw',
          'signed_receipts', 'changed_receipts', 'first_change', 'path_raw', 'opposite_sign_steps')


class ReceiptAudit(joint.Population):
    def __init__(self, parameters, masks, schedule, random_seeds, **kwargs):
        super().__init__(parameters, masks, schedule, random_seeds, **kwargs)
        self.audit_initial = masks.copy()
        self.audit_rows = []
        self.audit_birth = np.where(masks, -1, -2)
        self.audit_cohort = np.where(masks, 0, -1).astype(np.int8)
        self.audit_ever = masks.copy()
        self.audit_reference = self.raw.copy()
        self.audit_first = np.full(masks.shape, -1, dtype=np.int16)
        self.audit_signed = np.zeros(masks.shape, dtype=np.int16)
        self.audit_changed = np.zeros_like(self.audit_signed)
        self.audit_path = np.zeros_like(self.raw)
        self.audit_opposite = np.zeros_like(self.audit_signed)

    def finish_request(self, contacted, shown, **kwargs):
        self.audit_receipt = (contacted >= 0) & (shown != 0)
        self.audit_signed += self.mask & self.audit_receipt[:, None, None]
        return super().finish_request(contacted, shown, **kwargs)

    def _update_weights(self, credit, **kwargs):
        before = self.raw.copy()
        super()._update_weights(credit, **kwargs)
        step = np.abs(self.raw - before)
        changed = step > 0
        justified = self.mask & self.audit_receipt[:, None, None]
        if np.any(changed & ~justified):
            raise RuntimeError('A raw coefficient changed without a signed receipt on an active edge')
        self.audit_changed += changed
        self.audit_first[changed & (self.audit_first < 0)] = self.trial
        self.audit_path += step
        self.audit_opposite += (np.sign(before) * np.sign(self.raw)) == -1

    def _record(self, lane, receiver, sender, end_raw, *, active):
        arc = lane, receiver, sender
        n = len(lane)
        self.audit_rows.append(np.column_stack((
            lane, receiver, sender, self.audit_cohort[arc], self.audit_birth[arc], np.full(n, self.trial),
            np.full(n, int(active)), self.audit_reference[arc], end_raw, self.audit_signed[arc],
            self.audit_changed[arc], self.audit_first[arc], self.audit_path[arc], self.audit_opposite[arc])))

    def _edit(self, growth, retirement, **kwargs):
        before = self.raw.copy()
        decisions, ledger = super()._edit(growth, retirement, **kwargs)
        lanes = np.flatnonzero(decisions[:, 1] >= 0)
        target, source = np.divmod(decisions[lanes, 1], joint.WIDTH)
        arc = lanes, target, source
        self._record(*arc, before[arc], active=False)
        self.audit_birth[arc] = -2
        self.audit_cohort[arc] = -1

        lanes = np.flatnonzero(decisions[:, 0] >= 0)
        target, source = np.divmod(decisions[lanes, 0], joint.WIDTH)
        arc = lanes, target, source
        if np.any(self.audit_birth[arc] != -2):
            raise RuntimeError('Reused a generation identifier for a new birth')
        self.audit_birth[arc] = self.trial
        self.audit_reference[arc] = self.raw[arc]
        self.audit_cohort[arc] = np.where(self.audit_initial[arc], 3, np.where(self.audit_ever[arc], 2, 1))
        self.audit_ever[arc] = True
        self.audit_first[arc] = -1
        for values in (self.audit_signed, self.audit_changed, self.audit_path, self.audit_opposite):
            values[arc] = 0
        return decisions, ledger

    def get_rows(self):
        arc = np.nonzero(self.mask)
        if not np.array_equal(self.audit_birth != -2, self.mask):
            raise RuntimeError('Audit support differs from final graph')
        self._record(*arc, self.raw[arc], active=True)
        return np.concatenate(self.audit_rows).astype(np.float32)


def cross_check(rows, data):
    edits = probe.flatten_lives(data['edits'])
    terminal = probe.flatten_lives(data['final_newborn_records'])
    retired = probe.flatten_lives(data['event_records'])
    columns = {name: i for i, name in enumerate(FIELDS)}
    fields = [joint.BIRTH_FIELDS.index(name) for name in (
        'trial', 'initial_raw', 'end_raw', 'later_receipts', 'changed_receipts', 'first_change', 'path_raw')]
    newborn = rows[rows[:, columns['born']] >= 0]
    observed = newborn[:, [columns[name] for name in (
        'born', 'birth_raw', 'end_raw', 'signed_receipts', 'changed_receipts', 'first_change', 'path_raw')]]
    mask = newborn[:, columns['active_at_end']].astype(bool)
    lane, target, source = newborn[:, :3].astype(int).T
    end = newborn[:, columns['end']].astype(int)
    opposite = newborn[:, columns['opposite_sign_steps']] <= newborn[:, columns['changed_receipts']]
    if not opposite.all():
        raise RuntimeError('Found sign flips without a direct coefficient change')
    expected = np.empty_like(observed)
    expected[mask] = terminal[lane[mask], target[mask], source[mask]][:, fields]
    dead = ~mask
    if not np.array_equal(edits[lane[dead], end[dead], 1], target[dead] * joint.WIDTH + source[dead]):
        raise RuntimeError('Retirement archive coordinates differ from replay')
    expected[dead] = retired[lane[dead], end[dead], 1][:, fields]
    np.testing.assert_array_equal(observed, expected)


def summarize(rows):
    columns = {name: i for i, name in enumerate(FIELDS)}
    result = {}
    for name, cohort in {'all_newborn': rows[:, 3] >= 1, **{
        name: rows[:, 3] == i for i, name in enumerate(history.COHORTS)
    }}.items():
        result[name] = {}
        for fate, alive in (('all', np.ones(len(rows), dtype=bool)), ('retired', rows[:, 6] == 0),
                            ('survived', rows[:, 6] == 1)):
            values = rows[cohort & alive]
            if len(values) == 0:
                result[name][fate] = {'edges': 0}
                continue
            birth, end, initial_raw, end_raw, receipts, moves, path, flips = values[:, [columns[name] for name in (
                'born', 'end', 'birth_raw', 'end_raw', 'signed_receipts', 'changed_receipts', 'path_raw',
                'opposite_sign_steps')]].T
            result[name][fate] = history.summarize(initial_raw, end_raw, end - birth, receipts) | dict(
                ever_changed=int((moves > 0).sum()), equal_at_end_despite_change=int(((moves > 0) & (initial_raw == end_raw)).sum()),
                ever_reversed_nonzero_sign=int((flips > 0).sum()), signed_updates_quantiles=np.quantile(moves, history.QUANTILES).tolist(),
                path_raw_quantiles=np.quantile(path, history.QUANTILES).tolist())
    return result


def replay(out, joint_out):
    provenance = probe.seal_provenance(joint_out)
    head = starts.heads(probe.GROUPS)
    seal = io.read(joint_out / 'recipe.json')
    roots = tuple(probe.LIFE_ROOTS[index] for index in ROOT_OFFSETS)
    report = {}
    for condition in probe.CONDITIONS:
        instances = []

        def instrument(*args, **kwargs):
            brain = ReceiptAudit(*args, **kwargs)
            instances.append(brain)
            return brain

        data = study.tapes(roots, head[2], condition)
        trajectory = study.run(head, roots, data, study.SCHEDULE, 'joint', seal['rate'], population=instrument)
        with np.load(joint_out / f'{condition}.npz', allow_pickle=False) as archived:
            for name, values in trajectory.items():
                np.testing.assert_array_equal(values, archived[f'joint/{name}'][:, :, ROOT_OFFSETS])
        rows = instances[0].get_rows()
        cross_check(rows, trajectory)
        dest = out / f'receipt-replay-{condition}.npz'
        np.savez_compressed(dest, edges=rows)
        report[condition] = dict(movement=summarize(rows), sha256=io.sha(dest))
        print('receipt replay', condition, len(rows), flush=True)
    io.write(out / 'receipt-replay.json', dict(
        protocol_sha256=io.sha(out / 'receipt-replay-protocol.json'), source_sha256=io.sha(Path(__file__)),
        artifact_hashes=provenance, field_names=FIELDS, starts=probe.GROUPS, roots=roots,
        trials_exactly_replayed=True, newborn_raw_updates_equal_to_parent_ledger=True, results=report))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=probe.OUT)
    parser.add_argument('--joint', type=Path, default=probe.JOINT)
    args = parser.parse_args()
    replay(args.out, args.joint)


if __name__ == '__main__':
    main()
