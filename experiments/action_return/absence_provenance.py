"""Additive source-closure audit; no training, scoring, or protocol mutation."""
from datetime import datetime, timezone
from pathlib import Path
import json
from experiments.action_return.absence_study import OUT, ROOT, read_plan
from experiments.coverage_exit.retirement_study import read_plan as read_previous
from experiments.coverage_exit.retirement_magnitude_study import read_plan as read_magnitude
from experiments.continuity.run import state_digest
from experiments.undirected.run import _atomic_json, file_sha256


def audit():
    plan=read_plan()
    previous=[];closure={}
    for path,p in ((ROOT/'data/coverage-exit/retirement-v1/protocol.json',read_previous(ROOT/'data/coverage-exit/retirement-v1/protocol.json')),
                   (ROOT/'data/coverage-exit/magnitude-v1/protocol.json',read_magnitude())):
        for filename,sha in p['execution_files'].items():
            if file_sha256(filename)!=sha:raise ValueError('inherited execution file changed '+filename)
            if filename in closure and closure[filename]!=sha:raise ValueError('conflicting inherited source hash')
            closure[filename]=sha
        previous.append(dict(path=str(path),sha256=file_sha256(path),protocol_hash=p['protocol_hash']))
    required=[ROOT/'experiments/undirected/run.py',ROOT/'experiments/lifelong/evaluate.py',ROOT/'experiments/continuity/evaluate.py']
    if any(str(path) not in closure for path in required):raise ValueError('missing actual runtime closure')
    value=dict(format='absence-additive-source-closure-audit-v1',audit='passed',created_utc=datetime.now(timezone.utc).isoformat(),
               protocol_hash=plan['protocol_hash'],previous_protocols=previous,execution_files=closure,
               source_file=dict(path=str(Path(__file__).resolve()),sha256=file_sha256(__file__)),
               scope='Postlaunch read-only additive verification of unchanged original runtime/evaluator sources against prior prospective protocol hashes. No frozen new protocol, source, checkpoint, training or score was changed.')
    value['audit_hash']=state_digest(value);_atomic_json(OUT/'source_closure_audit.json',value);return value


if __name__=='__main__':print(json.dumps(audit(),indent=2))
