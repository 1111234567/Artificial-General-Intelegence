"""Commit a training stage only after all of its inference exports have been published."""

from hashlib import sha256
import json

from . import interleave_revisit_train as storage
from . import local_trace_train as model_io
from .interleave_interference import EVIDENCE


def locked_plan(path, proposed, module, code_field):
    proposed=json.loads(json.dumps(proposed))
    if path.exists():
        saved=json.loads(path.read_text(encoding="utf8"))
        if saved!=proposed:
            # The original optimizer may resume with the audited save-order fix.
            # Any change to training inputs or any other source revision still fails.
            migration=EVIDENCE/"checkpoint-publication.json"
            known=json.loads(migration.read_text(encoding="utf8"))["transitions"] if migration.exists() else {}
            same={**proposed,code_field:saved.get(code_field)}==saved
            pair=[saved.get(code_field),proposed.get(code_field)]
            if not same or known.get(module)!=pair:
                raise ValueError(f"Training plan or its computational implementation changed: {path}")
    else:
        storage._write_json(path,proposed)
    return sha256(path.read_bytes()).hexdigest()


def needs_export(path, task, arm, births, step, stages):
    return step in stages and any(not model_io.checkpoint_path(path,task,arm,b,step).is_file() for b in births)
