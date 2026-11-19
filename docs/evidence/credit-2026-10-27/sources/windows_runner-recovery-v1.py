"""Windows status-file retry adapter; frozen experiment algorithms stay unchanged.

Concurrent readers can briefly deny os.replace on Windows. Only coordinator
status writes are retried here. Failed/terminated attempts are archived before
the finite command manifest is resumed from its complete checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from experiments.credit import batch


def save_with_retry(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==99:
                raise
            time.sleep(.05)


def prepare_resume(status_path):
    if not status_path.exists():
        return
    data=json.loads(status_path.read_text(encoding="utf-8"))
    failed=[name for name,item in data["jobs"].items() if item.get("returncode") not in (None,0)]
    if not failed and "finished" not in data:
        return
    index=1
    while status_path.with_name(f"{status_path.stem}-attempt-{index:03d}.json").exists():
        index+=1
    archive=status_path.with_name(f"{status_path.stem}-attempt-{index:03d}.json")
    original=status_path.read_bytes()
    archive.write_bytes(original)
    entry=dict(archive=str(archive.resolve()),sha256=hashlib.sha256(original).hexdigest(),
        resumed_utc=datetime.now(timezone.utc).isoformat(),jobs=failed,
        reason="coordinator failed; each branch resumes its latest complete checkpoint")
    data.setdefault("resume_history",[]).append(entry)
    for name in failed:
        data["jobs"][name]={"returncode":None,"status":"queued_for_resume","previous_attempt":str(archive.resolve())}
    data.pop("finished",None)
    data["complete"]=False
    save_with_retry(status_path,data)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("batch","evaluation"),required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--status",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--training-status",type=Path)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    batch.save=save_with_retry
    prepare_resume(args.status)
    if args.kind=="batch":
        code=batch.execute(args.plan,args.status,args.workers)
    else:
        if args.training_status is None:
            parser.error("evaluation requires --training-status")
        from experiments.credit.evaluate_ready import run
        code=run(args.plan,args.status,args.training_status,args.workers)
    raise SystemExit(code)
