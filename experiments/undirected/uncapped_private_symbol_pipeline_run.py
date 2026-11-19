"""Run the span backend with generation sealing overlapped across arms."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import linecache

from . import uncapped_private_symbol_run as run
from . import uncapped_private_symbol_span_run as span


class _PendingCommit:
    """Keep at most one commit pending and surface its failure on the caller thread."""

    def __init__(self, executor):
        self.executor = executor
        self.pending = None

    def drain(self, job=None):
        if self.pending is None or (job is not None and self.pending[0] is not job):
            return
        future = self.pending[1]
        try:
            future.result()
        finally:
            if future.done():
                self.pending = None

    def submit(self, job, seal, *args):
        self.drain()
        self.pending = job, self.executor.submit(seal, job, *args)


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Generation pipeline expected one sealed runner fragment: {old!r}")
    return source.replace(old, new, 1)


def _record_batch(job, path, batch):
    run.runner.record_batch(path, batch)
    job.completed += len(batch)


def _runner_with_generation_tail(tail, base_main, journal_tail=None, score_batches=None):
    # The search protocol hashes run.py. Derive only the scheduling changes
    # from its source, and use its own journal, selection and sealing code.
    generation = inspect.getsource(run.run_generation)
    generation = _replace_once(generation, '    cfg = job.cfg\n',
                               '    if mirror is not None:\n'
                               '        _generation_tail.drain()\n'
                               '    cfg = job.cfg\n')
    journal_parameter = 'delta_evaluator, publish, *, mirror=None):\n'
    generation = _replace_once(generation, journal_parameter,
                               'delta_evaluator, publish, *, mirror=None, journal=None):\n')
    if score_batches is not None:
        begin = generation.index('    for path, group, genomes in execution:\n')
        finish = generation.index('        rows.extend(batch)\n', begin)
        generation = (generation[:begin]
                      + '    for path, batch, commit in _score_batches(execution, job, generation,\n'
                        '            mirror, training, seed, ACTORS, VISITS):\n'
                        '        if commit:\n'
                        '            runner.record_batch(path, batch)\n'
                        '            job.completed += len(batch)\n'
                      + generation[finish:])
    record = '            runner.record_batch(path, batch)\n            job.completed += len(batch)\n'
    generation = _replace_once(generation, record,
                               '            if journal is None:\n'
                               '                _record_batch(job, path, batch)\n'
                               '            else:\n'
                               '                journal.submit(job, _record_batch, path, batch)\n')
    report = '        if now >= next_progress or len(rows) == population:\n'
    generation = _replace_once(generation, report, report +
                               '            if journal is not None:\n'
                               '                journal.drain()\n')
    checkpoint = '    checkpoint = breeding.elites(cfg, rows, previous, warm_genomes=job.warm_bank)\n'
    generation = _replace_once(generation, checkpoint,
                               '    _generation_tail.submit(job, _seal_generation, generation, folder,\n'
                               '                            cfg, previous, rows, population, seed)\n'
                               '\n\n'
                               'def _seal_generation(job, generation, folder, cfg, previous, rows, population, seed):\n'
                               + checkpoint)

    main_source = inspect.getsource(base_main)
    parse = '    args = parser.parse_args(argv)\n'
    main_source = _replace_once(main_source, parse,
                                '    parser.add_argument("--async-journals", action="store_true",\n'
                                '                        help="Overlap one ordered batch journal with evaluation.")\n'
                                + parse)
    audit_path = '            audit_path = job.out / "audits" / f"{generation:04d}.json"\n'
    main_source = _replace_once(main_source, audit_path,
                                '            _generation_tail.drain()\n' + audit_path)
    arm_loop = '                    for index, job in enumerate(jobs):\n'
    main_source = _replace_once(main_source, arm_loop,
                                arm_loop + '                        _generation_tail.drain(job)\n')
    run_arm = '                                       mirror=mirror)\n'
    main_source = _replace_once(main_source, run_arm,
                                '                                       mirror=mirror,\n'
                                '                                       journal=_journal_tail if args.async_journals else None)\n')
    complete = '            publish(jobs[-1], args.generations, args.population, phase="complete", state="complete")\n'
    main_source = _replace_once(main_source, complete,
                                '            _generation_tail.drain()\n' + complete)
    failure = '        except (Exception, KeyboardInterrupt) as error:\n'
    main_source = _replace_once(main_source, failure, failure +
                                '            for commit in (_journal_tail, _generation_tail):\n'
                                '                if commit is not None:\n'
                                '                    try:\n'
                                '                        commit.drain()\n'
                                '                    except Exception as commit_error:\n'
                                '                        error.add_note(f"Pending artifact commit failed: {commit_error}")\n')

    # fast.main installs the span evaluator and parent replay before this runs.
    namespace = {**vars(run), "_generation_tail": tail, "_journal_tail": journal_tail,
                 "_record_batch": _record_batch, "_score_batches": score_batches}
    filename = f"{run.__file__}:generation-pipeline"
    source = generation + "\n\n" + main_source
    linecache.cache[filename] = len(source), None, source.splitlines(keepends=True), filename
    exec(compile(source, filename, "exec"), namespace)
    return namespace["main"]


def main():
    base_main = run.main

    def pipeline_main():
        with (ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation-seal") as seals,
              ThreadPoolExecutor(max_workers=1, thread_name_prefix="candidate-journal") as journals):
            return _runner_with_generation_tail(_PendingCommit(seals), base_main, _PendingCommit(journals))()

    run.main = pipeline_main
    try:
        span.main()
    finally:
        run.main = base_main


if __name__ == "__main__":
    main()
