"""Reuse identical private-symbol founder scores on matched arm tapes."""

from collections import deque
from functools import partial

from . import uncapped_private_symbol_pipeline_run as pipeline
from . import uncapped_private_symbol_span as span
from . import uncapped_private_symbol_run as run
from . import uncapped_private_symbol_selection as selection
from .uncapped_private_symbol_post_answer import post_answer_actions


def _founder(entry):
    return (entry["kind"] == "founder" and entry.get("parent") is None and entry.get("donor") is None
            and entry.get("inheritance_trial") is None)


def _scored_rollout(observe, policy, tape, table):
    metrics, actions = observe(policy, tape, table)
    metrics["post_answer_actions"] = post_answer_actions(actions, tape)
    return metrics


_training_rollout = partial(_scored_rollout, span.observed_rollout)
_holdout_rollout = partial(_scored_rollout, run.world.observed_rollout)


class _SharedFounderScores:
    def __init__(self):
        self.key = None
        self.rows = {}
        self.generation = None
        self.enabled = True
        self.checked_arms = set()
        self.requests = None
        self.phases = []
        self.rule = None

    def __call__(self, execution, job, generation, mirror, training, seed, actors, visits):
        cfg = job.cfg
        width, device = cfg["eval_batch_size"], cfg["device"]
        if self.requests is None:
            self.requests = actors * visits * selection.N_SYMBOLS
            self.phases = selection.read_phases(job.out.parent.parent, self.requests)
        rule = selection.rule_at(self.phases, generation)
        if rule == selection.POST_ANSWER_RULE and training.rollout is not _training_rollout:
            training.clear_cache()
            training.rollout = _training_rollout
        key = job.pair, generation, seed, actors, visits, width, device, cfg["nodes"], rule
        if job.out not in self.checked_arms:
            history = job.history
            if history:
                latest = history[-1]
                selection.validate_rule([latest], selection.rule_at(self.phases, latest["generation"]), job.out)
            self.checked_arms.add(job.out)
        if self.generation != generation:
            self.generation = generation
            self.enabled = not (job.out.parent.parent / ".disable-founder-reuse").is_file()
            if rule != self.rule:
                print(f"selection_rule={rule} active for generation {generation}", flush=True)
                self.rule = rule
        recording = self.enabled and job.arm == "downhill" and generation > 1 and mirror is None
        if recording:
            self.key, self.rows = key, {}
        cached = self.rows if job.arm == "nondecreasing" and self.key == key and mirror is None else {}
        if not recording and not cached:
            self.key, self.rows = None, {}

        def evaluate(genomes):
            metrics = training(genomes, seed=seed, actors=actors, visits=visits,
                               device=device, capture=device == "cuda")
            return run.runner.base.unpack_scores(metrics, len(genomes))

        def scored(entry, row):
            fields = run.score_fields(row, cfg["edge_pairs"])
            if rule is not None:
                fields = selection.score_fields(fields, self.requests, rule)
            return {**entry, **fields}

        def read(path, group):
            rows = run.runner.read_batch(path, group)
            selection.validate_rule(rows, rule, path)
            return rows

        if not cached:
            # Retain the original batch boundaries when nothing is reused.
            for path, group, genomes in execution:
                commit = not path.is_file()
                if not commit:
                    batch = read(path, group)
                elif mirror is not None:
                    batch = read(mirror / "batches" / path.name, group)
                else:
                    scores = [row for start in range(0, len(genomes), width)
                              for row in evaluate(genomes[start:start + width])]
                    batch = [scored(entry, row) for entry, row in zip(group, scores, strict=True)]
                if recording:
                    for entry, row in zip(group, batch, strict=True):
                        if _founder(entry):
                            self.rows[entry["id"]] = entry, row
                yield path, batch, commit
            return

        # Keep at most two ordinary journals in flight when filling the fixed
        # CUDA width with the unevaluated candidates. Commit in original order.
        pending, journals = [], deque()

        def flush():
            work = pending[:width]
            del pending[:width]
            genomes = [genome for _, _, _, genome in work]
            scores = evaluate(genomes + [genomes[-1]] * (width - len(genomes)))
            for (batch, index, entry, _), row in zip(work, scores[:len(work)], strict=True):
                batch[index] = scored(entry, row)

        def ready_journals():
            while journals and all(row is not None for row in journals[0][1]):
                yield journals.popleft()

        try:
            for path, group, genomes in execution:
                if path.is_file():
                    journals.append((path, read(path, group), False))
                else:
                    batch = [None] * len(group)
                    journals.append((path, batch, True))
                    for index, (entry, genome) in enumerate(zip(group, genomes, strict=True)):
                        shared = cached.get(entry["id"])
                        if shared is not None and shared[0] == entry:
                            batch[index] = dict(shared[1])
                        else:
                            pending.append((batch, index, entry, genome))
                    while len(pending) >= width:
                        flush()
                        yield from ready_journals()
                if len(journals) > 2 and pending:
                    flush()  # Bound the backlog if following journals were already committed.
                yield from ready_journals()
            if pending:
                flush()
            yield from journals
        finally:
            self.key, self.rows = None, {}


def main():
    previous = pipeline._runner_with_generation_tail
    previous_summary = run.summarize
    previous_audit = run.audit
    scoring = _SharedFounderScores()

    def with_shared_founders(tail, base_main, journal_tail=None):
        return previous(tail, base_main, journal_tail, score_batches=scoring)

    def audit(out, generation, config, checkpoint, factory, **options):
        phases = selection.read_phases(out.parent.parent, selection.requests_per_candidate(config))
        if selection.rule_at(phases, generation) == selection.POST_ANSWER_RULE:
            original_factory = factory

            def factory(schedule, **settings):
                settings.setdefault("rollout", _holdout_rollout)
                return original_factory(schedule, **settings)

        return previous_audit(out, generation, config, checkpoint, factory, **options)

    pipeline._runner_with_generation_tail = with_shared_founders
    run.summarize = partial(selection.summarize, previous_summary)
    run.audit = audit
    try:
        pipeline.main()
    finally:
        run.audit = previous_audit
        run.summarize = previous_summary
        pipeline._runner_with_generation_tail = previous


if __name__ == "__main__":
    main()
