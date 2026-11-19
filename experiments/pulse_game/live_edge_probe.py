"""Freeze the jointly learned graph and test the numerical values of its newest edges."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import live_edge_probe_history as history
from . import live_module_edits as base_model, live_module_edits_study as starts
from . import ring_logic_study as io, ring_logic_world as sensory, world


OUT = io.REPO / 'docs/evidence/free-graph-2026-11-19/joint-edges'
JOINT = io.REPO / 'data/pulse-game/joint-live-20261119'
GROUPS = (4, 5, 6, 7, 12, 13, 14, 15)
LIFE_ROOTS = tuple(range(69023000, 69023016))
PROBE_ROOTS = tuple(range(69043000, 69043016))
CONDITIONS = ('stable', 'reversal')
ARMS = ('learned', 'reset_newborn', 'reset_initial', 'reset_all', 'reset_first_novel')
SOURCES = ('live_edge_probe.py', 'live_edge_probe_history.py', 'ring_logic_world.py', 'world.py',
           'live_module_edits.py', 'live_module_edits_study.py')
DRAW_COUNT = 20000
BATCH = 24


def code_hashes():
    return {name: io.sha(Path(__file__).with_name(name)) for name in SOURCES}


def register(out):
    path = out / 'registered.json'
    if path.exists():
        raise RuntimeError('Registration already exists')
    io.write(path, dict(
        registered_code_sha256=code_hashes(), cohort_role='Confirmatory audit on sealed joint lives and fresh queries',
        starts=GROUPS, life_roots=LIFE_ROOTS, probe_roots=PROBE_ROOTS,
        conditions=CONDITIONS, arms=ARMS,
        rule='Use every head/root/rule combination in the main jointly trained feedback arm, with no post hoc selection, for stable and reversed labels.',
        identity='Endpoints are receiver/sender matrix indices. Each growth event starts a fresh generation. Preserve untreated initial generations, novel addresses in their first generation, novel addresses regrown later, and regrown initial addresses as separate cohorts. Surviving within-life generations form the newborn intervention.',
        replacement='Use the true stored, unnormalized coefficient of each surviving generation at birth. Learned, reset newborn, reset initial, reset both, and reset only novel addresses in their first generation differ only in the selected raw scalars; graph, degree, hidden state, drive, bias, readout and offset agree before each fork.',
        evaluation='Fork each saved endpoint into 20 independent physical requests with all ordered odor pairs. Each fork starts from the identical saved state. One new probe root per paired life root, common within all rule, condition, and intervention comparisons; no visible signs, learning or edits. Score actual first contact against the terminal rule, timeouts as incorrect.',
        primary='Learned minus reset newborn in both stable and reversal, with simultaneous 95% Bonferroni percentile intervals. Positive difference means resetting newborn coefficients lowers physical performance with all other endpoint data fixed.',
        secondary='Learned minus reset initial, learned minus reset both, and learned minus reset first novel, plus the two-by-two reset interaction, in each condition. Simultaneous 95% Bonferroni percentile intervals in this separate eight-contrast family.',
        inference=f'{DRAW_COUNT} paired crossed bootstrap draws of independent head groups and root blocks (each maps one life root to one fresh probe root), with rules and odor pairs fixed.'
    ))


def verify_registration(out):
    if io.read(out / 'registered.json')['registered_code_sha256'] != code_hashes():
        raise RuntimeError('Registered audit sources changed')


def flatten_lives(array):
    return array.reshape((-1, *array.shape[3:]))


def score_probes(population, arrays, scorer, raw):
    """Start matched, feedback-silent forks from each post-life endpoint."""
    observer = sensory.TIMING['standard'][0]
    cells = len(sensory.PAIRS)
    count = len(raw)
    shape = (count, cells)
    physical = np.empty((*shape, 3), dtype=np.int8)
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        ix = np.arange(start, stop)
        def expand(value):
            return np.repeat(value[ix], cells, axis=0)

        raw_expand = expand(raw)
        masks = expand(arrays['final_mask'])
        params = {name: expand(arrays[f'final_{name}']) for name in ('drive', 'bias', 'readout', 'offset')}
        params['gain'] = expand(arrays['starting_gain'])
        seeds = [np.random.SeedSequence([69043023, int(scorer.roots[i]), int(scorer.rule[i]), cell])
                 for i in ix for cell in range(cells)]
        brain = population(params, masks, base_model.Schedule(1, 1, .5), seeds, learning_rate=0.)
        brain.state = expand(arrays['final_state'])
        brain.raw[...] = raw_expand
        brain.refresh_matrices()
        tape_rows = (scorer.root_index[ix, None] * cells + np.arange(cells)[None]).ravel()
        key = scorer.direction[ix].reshape(-1, 1)
        sense = scorer.tape
        left = sense.left_id[tape_rows]
        tape = replace(sense, seeds=tuple(sense.seeds[i] for i in tape_rows), ctx=sense.ctx[tape_rows],
                       cue=sense.cue[tape_rows], left_id=left, good_id=np.where(key, 1 - left, left),
                       good_map=sense.good_map[tape_rows], distances=sense.distances[tape_rows],
                       wind=sense.wind[tape_rows])
        body = world.PulseWorld(tape, feedback='none')
        for tick, stimulus in enumerate(scorer.stimuli[tape_rows].transpose(1, 0, 2)):
            go_right = brain.act(stimulus)
            if observer <= tick < observer + world.TICKS:
                body.step(np.column_stack((~go_right, go_right)))
        result = body.result()
        if np.any(result.observed_reward) or not np.array_equal(brain.raw, raw_expand):
            raise RuntimeError('A silent, frozen fork unexpectedly got feedback or changed a coefficient')
        if not np.array_equal(brain.mask, masks):
            raise RuntimeError('A silent, frozen fork unexpectedly edited support')
        physical[start:stop] = np.stack((result.contact_side, result.contact_tick, result.reward),
                                        axis=-1).reshape(stop - start, cells, 3)
    return physical


class Scorer:
    def __init__(self, rows, reversal):
        self.root_index = np.tile(np.arange(len(PROBE_ROOTS)), len(rows))
        _, observations, _ = sensory.sensory(PROBE_ROOTS, 'standard')
        self.stimuli = observations.reshape(-1, *observations.shape[2:])
        self.rule = np.repeat([row[3] for row in rows], len(PROBE_ROOTS))
        self.direction = sensory.truth()[self.rule] ^ reversal
        self.tape = sensory.body_tape(PROBE_ROOTS)
        self.roots = np.asarray(PROBE_ROOTS)[self.root_index]


def interventions(final_raw, reference, cohort):
    masks = {'learned': np.zeros_like(cohort, dtype=bool), 'reset_newborn': cohort >= 1,
             'reset_initial': cohort == 0, 'reset_all': cohort >= 0, 'reset_first_novel': cohort == 1}
    return {arm: np.where(masks[arm], reference, final_raw) for arm in ARMS}


def inspect_arrays(data, parameters, masks, rows):
    shape = (len(GROUPS), len(sensory.RULES), len(LIFE_ROOTS))
    if len(rows) != shape[0] * shape[1] or len(masks) != len(rows):
        raise RuntimeError('Wrong start cohort size')
    arrays = {name: data[f'joint/{name}'] for name in (
        'final_mask', 'final_raw', 'final_state', 'final_drive', 'final_bias', 'final_readout',
        'final_offset', 'final_newborn_trial', 'final_newborn_raw', 'edits', 'physical')}
    if any(array.shape[:3] != shape for array in arrays.values()):
        raise RuntimeError('Unexpected life shape')
    support = arrays['final_mask']
    initial = np.broadcast_to(masks.reshape(*shape[:2], 1, *masks.shape[-2:]), support.shape)
    births, generations, cohort = history.edge_identity(initial, arrays['edits'], support)
    saved_births = arrays['final_newborn_trial']
    if not np.array_equal(saved_births[births >= 0], births[births >= 0]) or np.any(saved_births[births == -1] >= 0):
        raise RuntimeError('Stored newborn tags disagree with independent generation replay')
    gain = parameters['gain'].reshape(*shape[:2], masks.shape[-1])
    arrays['starting_gain'] = np.broadcast_to(gain[..., None, :], (*shape, masks.shape[-1]))
    if not np.array_equal(data['initial_mask'].reshape(masks.shape), masks):
        raise RuntimeError('Initial support provenance does not match selected modular heads')
    saved_initial_raw = data['initial_raw'].reshape(masks.shape)
    expected_initial_raw = np.where(masks, parameters['gain'][..., None], np.float32(0))
    if not np.array_equal(saved_initial_raw, expected_initial_raw):
        raise RuntimeError('Initial raw values differ from the selected modular heads')
    initial_raw = saved_initial_raw.reshape(*shape[:2], 1, *masks.shape[-2:])
    reference = history.combine_coefficients(initial_raw, arrays['final_newborn_raw'], arrays['final_raw'], births)
    trial_count = arrays['edits'].shape[-2]
    age = np.where(births != history.ABSENT, trial_count - births - 1, -1)
    receipts = history.age_in_receipts(births, arrays['physical'][..., 2])
    records = dict(birth_trial=births, generation=generations, cohort=cohort, age_requests=age,
                   post_birth_signed_receipts=receipts, reference_raw=reference,
                   final_mask=support, final_raw=arrays['final_raw'])
    return arrays, records


def report(records, scores, rng):
    movement = {}
    for label, subset in {'all_newborn': records['cohort'] >= 1, **{
        name: records['cohort'] == i for i, name in enumerate(history.COHORTS)
    }}.items():
        reference, final, age, received = [records[name][subset] for name in (
            'reference_raw', 'final_raw', 'age_requests', 'post_birth_signed_receipts')]
        movement[label] = history.summarize(reference, final, age, received)
    accuracy = {arm: (scores[arm][..., 2] > 0).mean(axis=-1) for arm in ARMS}
    changed = {arm: float((scores[arm][..., 0] != scores['learned'][..., 0]).mean()) for arm in ARMS}
    point = {arm: {'accuracy': float(acc.mean()), 'per_rule': acc.mean(axis=(0, 2)).tolist(),
                   'timeout': float((scores[arm][..., 0] < 0).mean()), 'different_contact_side': changed[arm]}
             for arm, acc in accuracy.items()}
    differences = {arm: accuracy['learned'] - accuracy[arm] for arm in ARMS if arm != 'learned'}
    differences['factorial_interaction'] = (accuracy['learned'] - accuracy['reset_newborn']
                                           - accuracy['reset_initial'] + accuracy['reset_all'])
    groups = len(GROUPS)
    roots = len(LIFE_ROOTS)
    bg = rng.multinomial(groups, np.full(groups, 1 / groups), size=DRAW_COUNT) / groups
    br = rng.multinomial(roots, np.full(roots, 1 / roots), size=DRAW_COUNT) / roots
    contrasts = {}
    for arm, gain in differences.items():
        gain = gain.mean(axis=1)
        draws = ((bg @ gain) * br).sum(axis=-1)
        tail = .05 / (2 * (2 if arm == 'reset_newborn' else 8))
        contrasts[arm] = dict(difference=float(gain.mean()), simultaneous_95pct=np.quantile(draws, [tail, 1 - tail]).tolist(),
                              difference_by_group=gain.mean(axis=-1).tolist())
    return dict(movement=movement, physical=point, contrasts=contrasts)


def confirm(out, joint):
    verify_registration(out)
    from . import joint_live_model as model, joint_live_study as study

    if tuple(study.TEST_GROUPS) != GROUPS or tuple(study.TEST_ROOTS) != LIFE_ROOTS:
        raise RuntimeError('Joint confirmation split differs from audit registration')
    if set(PROBE_ROOTS) & (set(LIFE_ROOTS) | set(study.DEV_ROOTS)):
        raise RuntimeError('Probe roots overlap the life simulation roots')
    parameters, masks, rows, manifests = starts.heads(GROUPS)
    scores_by_condition = {}
    provenance = seal_provenance(joint)
    for condition in CONDITIONS:
        source = joint / f'{condition}.npz'
        with np.load(source, allow_pickle=False) as data:
            arrays, records = inspect_arrays(data, parameters, masks, rows)
        brain_arrays = {name: flatten_lives(array) for name, array in arrays.items()}
        scorer = Scorer(rows, condition == 'reversal')
        branch = interventions(records['final_raw'], records['reference_raw'], records['cohort'])
        output = {}
        for arm, raw in branch.items():
            physical = score_probes(model.Population, brain_arrays, scorer, flatten_lives(raw))
            output[arm] = physical.reshape(len(GROUPS), len(sensory.RULES), len(LIFE_ROOTS), len(sensory.PAIRS), 3)
        dest = out / f'{condition}.npz'
        np.savez_compressed(dest, **records, **{f'probes/{arm}': values for arm, values in output.items()})
        scores_by_condition[condition] = report(records, output, np.random.default_rng(69043004))
        print(condition, {arm: values['accuracy'] for arm, values in scores_by_condition[condition]['physical'].items()}, flush=True)
    io.write(out / 'summary.json', dict(
        protocol_sha256=io.sha(out / 'registered.json'), joint_provenance=provenance,
        checkpoint_heads=rows, checkpoint_manifests=manifests, groups=GROUPS, life_roots=LIFE_ROOTS,
        probe_roots=PROBE_ROOTS, quantiles=history.QUANTILES, change_thresholds=history.CHANGE_FLOORS,
        cohort_codes=history.COHORTS, physical_columns=['side', 'tick', 'external_reward'],
        results=scores_by_condition, raw_sha256={name: io.sha(out / f'{name}.npz') for name in CONDITIONS}))


def seal_provenance(joint):
    seal = io.read(joint / 'summary.json')
    names = ('registered.json', 'recipe.json', 'summary.json', *[f'{name}.npz' for name in CONDITIONS])
    hashes = {name: io.sha(joint / name) for name in names}
    for condition in CONDITIONS:
        if seal['scores_sha256'][condition] != hashes[f'{condition}.npz']:
            raise RuntimeError('Joint endpoint archive differs from its confirmation seal')
    return hashes


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'confirm'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--joint', type=Path, default=JOINT)
    args = parser.parse_args()
    if args.phase == 'register':
        register(args.out)
    else:
        confirm(args.out, args.joint)


if __name__ == '__main__':
    main()
