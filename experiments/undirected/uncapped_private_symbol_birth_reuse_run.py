"""Reuse evaluated CPU birth genomes when sealing a private-symbol generation."""

from contextvars import ContextVar

from . import uncapped_private_symbol_pipeline_run as pipeline
from . import uncapped_private_symbol_run as run


class _BirthReuse:
    def __init__(self, build, batches):
        self.build = build
        self.original_batches = batches
        self.current = None
        self.sealing = ContextVar("private_symbol_sealing_births", default=None)

    def batches(self, folder, entries, config, previous, pool, depth):
        if self.current is not None:
            raise RuntimeError("The previous generation's birth genomes were not collected")
        births = {}
        self.current = folder, births
        for path, group, genomes in self.original_batches(folder, entries, config, previous, pool, depth):
            # The evaluator stacks new tensors. These original CPU objects
            # retain the exact values and dtypes used at birth.
            if genomes is not None:
                births.update((entry["id"], genome) for entry, genome in zip(group, genomes, strict=True))
            yield path, group, genomes

    def take(self, folder):
        if self.current is None:
            return {}  # A mirrored founder generation has only saved scores.
        open_folder, births = self.current
        if folder != open_folder:
            raise RuntimeError(f"Birth genome cache belongs to {open_folder}, expected {folder}")
        self.current = None
        return births

    def candidate(self, entry, previous, nodes, warm_genomes=None):
        cache = self.sealing.get()
        if cache is not None and cache[0] is previous and cache[1] == nodes:
            genome = cache[2].get(entry["id"])
            if genome is not None:
                return genome
        # Resuming a generation can read journal rows without building their
        # genomes. Regenerate those rows and mirrored founders as before.
        return self.build(entry, previous, nodes, warm_genomes=warm_genomes)

    def release(self):
        if self.current is not None:
            self.current[1].clear()
            self.current = None


class _ReusingGenerationTail:
    def __init__(self, tail, reuse):
        self.tail = tail
        self.reuse = reuse

    def drain(self, job=None):
        self.tail.drain(job)

    def submit(self, job, seal, generation, folder, cfg, previous, *args):
        births = self.reuse.take(folder)
        try:
            self.tail.submit(job, self._seal, seal, births, generation, folder, cfg, previous, *args)
        except BaseException:
            births.clear()
            raise

    def _seal(self, job, seal, births, generation, folder, cfg, previous, *args):
        token = self.reuse.sealing.set((previous, cfg["nodes"], births))
        try:
            seal(job, generation, folder, cfg, previous, *args)
        finally:
            self.reuse.sealing.reset(token)
            births.clear()


def main():
    original_build = run.breeding.candidate
    original_batches = run.runner.batches
    original_pipeline = pipeline._runner_with_generation_tail
    reuse = _BirthReuse(original_build, original_batches)

    def with_birth_reuse(tail, base_main, journal_tail=None):
        return original_pipeline(_ReusingGenerationTail(tail, reuse), base_main, journal_tail)

    run.breeding.candidate = reuse.candidate
    run.runner.batches = reuse.batches
    pipeline._runner_with_generation_tail = with_birth_reuse
    try:
        pipeline.main()
    finally:
        pipeline._runner_with_generation_tail = original_pipeline
        run.runner.batches = original_batches
        run.breeding.candidate = original_build
        reuse.release()


if __name__ == "__main__":
    main()
