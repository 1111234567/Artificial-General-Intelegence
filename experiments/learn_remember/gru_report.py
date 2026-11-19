"""Summarize retained training evidence; never substitutes fit for free tests."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.learn_remember.gru_train import OUT,sha,write_json


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    checkpoints=("random","short_mid","short","wide_mid","wide","withdrawal_mid","withdrawal","review_mid","review")
    output=dict(format="learn-remember-gru-training-report-v1",generated_time=time.time(),
        seeds=[0,1,2],selection="All three predetermined seeds included; smoke seed99 excluded from results",
        architecture="148246-parameter single-layer GRU192 receiving exactly public56; affine22 output, native11way argmax for each of2 slots; no auxiliary head, timer, manual gate or lookup memory",
        training_scope="Supervised imitation with complete teacher/mixed-by-lane/own continuous trajectories and repeated-current-block CE. Private causal oracle is a target only. No claim of unguided reinforcement learning.",
        hidden_protocol="One persistent numeric live hidden state per lane. It is carried through every observation and across optimizer/stage boundaries. Offline gradient passes start from that recorded collection anchor and detach only at128-tick boundaries. Repeated gradient passes do not advance/reset the live world or its live hidden state.",
        independence="Heldout frozen native-action tests and interventions are separate artifacts under data/learn-remember/evaluation-v1, not summarized as training metrics here.",
        reproduction_commands=[f".venv/Scripts/python.exe experiments/learn_remember/{script} --seed {seed}" for script in ("gru_train.py","gru_review.py","gru_audit.py") for seed in (0,1,2)],
        reproducibility="Seeds, source snapshots, configuration, RNG states, optimizer, complete live world and numerical hidden state are retained. CUDA training did not enable deterministic_algorithms; bit-identical retraining across driver/library/hardware changes is not asserted.",
        work_counter_semantics="replay_family3_during_withdrawal is a legacy main-branch counter inherited unchanged by review checkpoints. It is authoritative only for the original no-review withdrawal branch. For review, use review_updates, review_gradient_lane_ticks, review_family3_gradient_tick_slots and review_family3_nonwait_labels, plus the historical selection/position audit. Review explicitly does replay family3 history.",
        sources={str(p):sha(p) for p in [ROOT/"experiments/learn_remember/gru_train.py",ROOT/"experiments/learn_remember/gru_review.py",ROOT/"experiments/lifelong/world.py",ROOT/"experiments/continuity/world.py"]},
        per_seed={})
    for seed in (0,1,2):
        folder=OUT/f"seed_{seed:03d}"
        training=rows(folder/"updates.jsonl")
        review=rows(folder/"review_updates.jsonl")
        summary=dict(checkpoints={},training_rounds=training,review_rounds=review)
        audit_path=folder/"execution_audit.json"
        if audit_path.exists():
            summary["execution_audit"] = dict(path=str(audit_path),sha256=sha(audit_path))
            summary["withdrawal_residual_family3"]=json.loads(audit_path.read_text(encoding="utf-8"))["withdrawal_residual_family3"]
        for name in checkpoints:
            meta=folder/f"{name}.json"
            if meta.exists():
                record=json.loads(meta.read_text(encoding="utf-8"))
                if sha(record["path"])!=record["sha256"]:
                    raise AssertionError("checkpoint hash mismatch")
                summary["checkpoints"][name]=record
        summary["last_completely_own_training_round_by_stage"]={}
        for row in training:
            if row["teacher_lanes"]==0:
                m=row["execution_metrics"]
                summary["last_completely_own_training_round_by_stage"][row["stage"]]=dict(
                    round=row["round"],matured=m["matured_eligible"],ordinary=m["completed_matured"],strict=m["clean_completed_matured"],
                    note="Process metric on the actual continuing training world before this round's fit; inherited maturations may straddle a prior round. Not an independent frozen evaluation.")
        if (folder/"review_plan.json").exists():
            plan=json.loads((folder/"review_plan.json").read_text(encoding="utf-8"))
            stats=list(plan["historical_stats"].values())
            total=sum(s["all_tick_slots"] for s in stats)
            active=sum(s["active_tick_slots"] for s in stats)
            nonwait=sum(sum(s["nonwait_family_labels"]) for s in stats)
            old=sum(s["family_tick_slots"][3] for s in stats)
            old_active=sum(s["active_family_tick_slots"][3] for s in stats)
            old_labels=sum(s["nonwait_family_labels"][3] for s in stats)
            summary["review_history_family3"] = dict(all_tick_slots=total,family3_tick_slots=old,fraction=old/total,
                active_tick_slots=active,family3_active_tick_slots=old_active,active_fraction=old_active/active,
                nonwait_labels=nonwait,family3_nonwait_labels=old_labels,nonwait_label_fraction=old_labels/nonwait,
                total_gradient_budget_fraction=.125,family3_all_position_budget_fraction=.125*old/total)
        output["per_seed"][str(seed)]=summary
    write_json(OUT/"training_report.json",output)
    print(OUT/"training_report.json")


if __name__=="__main__":
    main()
