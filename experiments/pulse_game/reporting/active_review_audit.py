"""Verify the two final fits, exported bodies, physical receipts and paired reports."""

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .. import local_trace_train as checkpoint
from .. import noon_ticket as ticket


DATA = ticket.DATA
EVIDENCE = ticket.EVIDENCE
ROOT = ticket.ROOT
SOURCE = Path(__file__).resolve().parents[1]


def content_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}
        self.fits = []
        self.endpoints = []
        self.twin_decisions = 0
        self.commit_decisions = 0

    def check(self, path, expected=None):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in self.files:
            self.files[relative] = content_hash(path)
        if expected is not None and self.files[relative] != expected:
            raise ValueError(f"Digest mismatch at {relative}")
        return self.files[relative]

    def report(self, name):
        path = EVIDENCE / name
        report = json.loads(path.read_text(encoding="utf-8"))
        self.check(path)
        for relative, digest in report["hashes"].items():
            self.check(DATA / relative, digest)
        self.endpoints.append(name)
        return report


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify_fit(audit, cohort, arm, layout=None):
    if layout:
        location = DATA / "receipt-agreement" / layout / cohort
        folder = location / arm
        step, settings, plan = 12000, "coupling", "receipt-agreement-plan.json"
        parent = DATA / ("coverage" if layout=="bilateral" else "option-binding") / cohort
        parent_arm = "evidence" if layout=="bilateral" else "local"
    else:
        location = DATA / "commit" / cohort
        folder = location / arm
        step, settings, plan = 10000, "gate", "commit-plan.json"
        parent, parent_arm = ticket.source(cohort), "uniform"

    path = folder / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    audit.check(EVIDENCE/plan, manifest["plan"])
    for name, digest in manifest["sources"].items():
        audit.check(SOURCE/name, digest)
    if layout:
        audit.check(parent / parent_arm / "gain-8000.npz", manifest["gain"])
    ids = ticket.births(cohort)
    state = torch.load(folder/"resume.pt", map_location="cpu", weights_only=True)
    require(state["step"]==step and state["trace"].shape[:2]==(step,len(ids)), folder)
    require(state.get("meta", state.get("manifest"))==audit.check(path), "Resume manifest")
    require(state["trace"].isfinite().all().item(), "Nonfinite recorded fit")
    adam_steps = {int(s["step"]) for s in state["optimizer"]["state"].values()}
    require(adam_steps=={step}, f"Adam steps at {folder}: {adam_steps}")
    audit.check(folder/"resume.pt")
    coef = folder / f"{settings}-{step}.npz"
    audit.check(coef)
    with np.load(coef, allow_pickle=False) as head:
        np.testing.assert_array_equal(head["births"], ids)
        for name in ("gain","eta") if layout else ("gate",):
            np.testing.assert_array_equal(head[name],state[name].numpy())

    for lane,birth in enumerate(ids):
        audit.check(checkpoint.checkpoint_path(parent,"odor",parent_arm,birth,8000),manifest["parents"][str(birth)])
        body = checkpoint.checkpoint_path(location,"odor",arm,birth,step)
        audit.check(body)
        with np.load(body,allow_pickle=False) as z:
            require((int(z["birth"]),str(z["arm"]),int(z["step"]))==(birth,arm,step), body)
            for exported,saved in (("weights","w"),("bias","b"),("present","present")):
                np.testing.assert_array_equal(z[exported][0],state[saved][lane].numpy())
            for field,values in state["rule"].items():
                np.testing.assert_array_equal(z[f"rule/{field}"][0],values[lane].numpy())
    audit.fits.append(dict(cohort=cohort,arm=arm,layout=layout,steps=step,births=ids))


def physical_receipts(audit, report, paths):
    # Count every lane at each displayed test price (with the receipt-slip twins
    # separately represented) on the independent new-world bilateral verification.
    for path in paths:
        with np.load(DATA/path,allow_pickle=False) as z:
            for price in ("0.05","0.15","0.25","0.4","0.55"):
                def get(field):
                    return z[f"{price}/{field}"]
                choice = get("choice")
                twin = np.arange(len(choice)) ^ 11
                require(len(choice)==report["worlds"]*16, path)
                np.testing.assert_array_equal(get("actual_history"),get("actual_history")[twin])
                np.testing.assert_array_equal(get("trace"),get("trace")[twin])
                np.testing.assert_array_equal(choice,choice[twin])
                np.testing.assert_array_equal(get("target"),-get("target")[twin])
                audit.twin_decisions += len(choice)


def physical_commits(audit, report, paths):
    for path in paths:
        with np.load(DATA/path,allow_pickle=False) as z:
            for angle in (15,45,75):
                prefix = f"{angle}/4/"
                def get(field):
                    return z[prefix+field]
                gates = get("gate")
                require(len(gates)==report["worlds"]*8, path)
                row = np.arange(len(gates))
                actual = get("serial")[row,gates.astype(int)]
                paid = (actual==get("target")[:,3:5]).all(1)
                rates = (paid*get("weight")).reshape(-1,8).sum(1)
                np.testing.assert_allclose(rates,z[f"metric/{prefix}all/native/serial"])
                before = get("study")[:,-1,0]
                np.testing.assert_array_equal(before,before[row^1])
                audit.commit_decisions += len(row)


def main():
    torch.set_num_threads(1)
    audit = Evidence()
    for cohort in ("original","independent"):
        for layout,arm in product(("bilateral","local"),("agreement","scrambled","count")):
            verify_fit(audit,cohort,arm,layout)
        for arm in ("state","constant","confirmation","always"):
            verify_fit(audit,cohort,arm)
        for layout,kind in product(("bilateral","local"),("","transfer-")):
            audit.report(f"agreement-{layout}-{kind}{cohort}-12000.json")
        extra = audit.report(f"agreement-bilateral-confirmed-{cohort}-12000.json")
        physical_receipts(audit,extra,[p for p in extra["hashes"] if p.startswith("agreement-eval/")])
        for kind in ("","stress-"):
            audit.report(f"commit-{kind}{cohort}-10000.json")
        extra = audit.report(f"commit-confirmed-{cohort}-10000.json")
        physical_commits(audit,extra,[p for p in extra["hashes"] if p.startswith("commit-eval/")])

    # Use the pre-existing horizon seal independently of the two new fit families.
    path = EVIDENCE/"horizon-online-seal.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    audit.check(path)
    for group in ("training","held_out_and_sources","code_and_report","greedy_source_archives","summaries"):
        for filename, digest in seal[group].items():
            audit.check(ROOT/filename,digest)
    speed = json.loads((EVIDENCE/"ticket-speed.json").read_text(encoding="utf-8"))
    audit.check(DATA/"speed/frozen-ticket.npz",speed["raw_sha256"])
    audit.check(SOURCE/"noon_ticket_speed.py",speed["implementation_sha256"])
    output = dict(
        fits=audit.fits, body_exports=sum(len(f["births"]) for f in audit.fits),
        receipt_and_commit_reports=audit.endpoints,
        observationally_identical_pre_purchase_decisions=audit.twin_decisions,
        individually_reconstructed_serial_outcomes=audit.commit_decisions,
        distinct_files=len(audit.files), sha256=dict(sorted(audit.files.items())),
        scope="All final agreement and commit fits, endpoint/stress summaries, final exports and their raw data. "
              "Hidden-law twins and gate arithmetic checked separately on the fresh confirmation sets. "
              "Includes files named in the online horizon seal and the frozen speed timing archive.")
    target=EVIDENCE/"active-review-audit.json"
    target.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(output["distinct_files"],"files;",output["body_exports"],"exported bodies;",audit.twin_decisions,
          "pre-purchase choices;",audit.commit_decisions,"post-commit returns")


if __name__ == "__main__":
    main()
