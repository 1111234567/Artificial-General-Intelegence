"""Serialization adapter for tensor metadata added by the new trainer.

Frozen evaluation code/scoring/world/protocol are unchanged. The first attempted
new B0 record could not serialize review_selection.selection_counts (Tensor).
Only this metadata is converted to JSON types; no model input/output changes.
"""
from pathlib import Path
import torch

from experiments.review_selection import eval_review as e


def json_metadata(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metadata(v) for v in value]
    return value


def install():
    path = e.OUT/'serialization-adapter.json'
    value = dict(original_protocol_hash=e.read(e.PLAN)['protocol_hash'], source_file=str(Path(__file__).resolve()), source_sha256=e.sha(__file__),
        reason='First new B0 record failed before JSON write because trainer review_selection.selection_counts is a Tensor. Convert only review_selection metadata recursively to JSON values; original frozen evaluator/scorer/public streams/checkpoint data remain unchanged.',
        evaluation_source_sha256=e.sha(e.__file__))
    if path.exists():
        assert e.read(path) == value
    else:
        assert not any(p.name.startswith(('uniform_', 'conflict_', 'error_')) for p in e.OUT.glob('seed_*/*.json'))
        e.write(path, value)
    original = e.bind_checkpoint
    def bind(row, path, cp, seed, arm, snapshot, scenario, plan):
        adapted = dict(cp)
        adapted['review_selection'] = json_metadata(cp.get('review_selection'))
        return original(row, path, adapted, seed, arm, snapshot, scenario, plan)
    e.bind_checkpoint = bind


if __name__ == '__main__':
    install()
    e.main()
