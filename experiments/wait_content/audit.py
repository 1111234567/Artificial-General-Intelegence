"""Independent arithmetic checks for the delayed-content training comparison."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.content_timing.audit import sha


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/wait-content"


def read(path):
    return Path(path).read_text(encoding="utf-8")


def check_pairs(seed):
    folder = BASE / f"data-v1/seed_{seed:03d}"
    pairs = np.load(folder / "anchors.npy", allow_pickle=False)
    with np.load(folder / "stream.npz", allow_pickle=False) as stream:
        teacher = stream["teacher_actions"]
        assert teacher.shape == (16384, 8, 2)
        assert pairs.ndim == 2 and pairs.shape[1] == 8 and pairs.dtype == np.int64
        block, lane, job, request, end, wait, target, move = pairs.T
        assert ((lane >= 0) & (lane < 4)).all() and ((job >= 0) & (job < 2)).all()
        assert ((wait < end) & (wait >= 0) & (end < len(teacher))).all()
        assert (wait//128 == block).all() and (end//128 == block).all()
        assert np.isin(move, [0, 1]).all()
        assert (teacher[end, lane, job] == target + 3).all()
        assert (teacher[wait, lane, job] == 0).all()
        req = stream["request_id"]
        assert (req[end, lane, job] == request).all() and (req[wait, lane, job] == request).all()
        phase = stream["phase"]
        assert (phase[end, lane, job] == 3).all() and (phase[wait, lane, job] == 2).all()
        dt = stream["duration"][wait,lane,job]
        long = dt >= 8
        delay = stream["now"][end] - stream["now"][wait]
        assert len(np.unique(np.column_stack((end,lane,job)), axis=0)) == len(pairs)
        assert len(np.unique(np.column_stack((wait,lane,job)), axis=0)) == len(pairs)
        counts = np.bincount(block, minlength=128)
        moves = np.bincount(block[move == 1], minlength=128)
        assert (moves*2 == counts).all()
        return dict(seed=seed, anchors=len(pairs), moved=int(moves.sum()),
                    moving_updates=int(np.sum(moves>0)), positions_in_action_loss=teacher[:,:4].size,
                    long_pairs=int(long.sum()), long_pairs_moved=int((long & (move==1)).sum()),
                    moved_wait_to_end_median_time=float(np.median(delay[move==1])),
                    target_histogram=np.bincount(target, minlength=8).tolist(),
                    per_update_pairs=counts.tolist(), per_update_moved=moves.tolist(),
                    bound={p.name:sha(p) for p in (folder/"anchors.npy", folder/"stream.npz")})


def rates(c):
    def div(num, den, unit):
        return unit*c[num]/c[den] if c[den] else None
    return dict(ordinary=div("primary_completed","primary_matured",100),
                strict=div("primary_clean_completed","primary_matured",100),
                begin=div("primary_feasible_began","primary_matured",100),
                idle=div("idle_actions","idle_opportunities",1000),
                idle_begin=div("idle_begin","idle_opportunities",1000),
                idle_finish=div("idle_finish","idle_opportunities",1000))


ARMS = ("untrained_anchor", "end_only", "wait_mixed")


def check_update(arm):
    """Rebuild the first moved update independently, using native per-tick execution."""
    import torch
    from experiments.undirected.model import PortGraphModel

    torch.set_num_threads(1)
    folder = BASE/f"train-v1/seed_000/{arm}"
    def load(name):
        return torch.load(folder/name, map_location="cpu", weights_only=False)
    before, after, batch = [load(n) for n in ("audit_before.pt","audit_after.pt","audit_batch.pt")]
    model = before["model"]
    base = PortGraphModel().double()
    prefix = "begin_head.base."
    base.load_state_dict({k[len(prefix):]:v for k,v in model.items() if k.startswith(prefix)})
    names = before["trainable_names"]
    edge = prefix+"edge_shared"
    parameters = {k:(base.edge_shared if k==edge else torch.nn.Parameter(model[k].clone())) for k in names}
    assert set(names) == {edge, "j0_coef", "j0_bias", "j1_coef", "j1_bias"}
    optimizer = torch.optim.Adam(list(parameters.values()), lr=.001)
    optimizer.load_state_dict(before["optimizer"])
    t0 = batch["block"] * 128
    data_folder = BASE/"data-v1/seed_000"
    a = np.load(data_folder/"anchors.npy")
    a = a[a[:,0] == batch["block"]]
    ts = np.where((arm=="wait_mixed") & (a[:,7]==1), a[:,5], a[:,4]) - t0
    positions = torch.from_numpy(np.column_stack((ts, a[:,1:3])))
    assert torch.equal(positions,batch["positions"])
    with np.load(data_folder/"stream.npz") as z:
        obs = torch.from_numpy(z["obs"][t0:t0+128,:4])
        act = torch.from_numpy(z["teacher_actions"][t0:t0+128,:4])
        elapsed = z["elapsed"][t0:t0+128]
    assert torch.equal(obs,batch["observations"]) and torch.equal(act,batch["actions"])
    h = before["hidden"].clone()
    hidden = []
    for o, dt in zip(obs,elapsed):
        _, h = base.step(o, h, float(dt))
        hidden.append(h.index_select(-1,model["begin_head.indices"]))
    h_error = float((h.detach()-after["hidden"]).abs().max())
    assert h_error < 1e-12
    x = torch.stack(hidden)
    begin_cols, finish_cols = [], []
    for j in range(2):
        bp = f"begin_head.j{j}_"
        fp = f"j{j}_"
        xb = (x-model[bp+"mean"])/model[bp+"std"]
        xf = (x-model[fp+"mean"])/model[fp+"std"]
        begin_cols.append(xb @ model[bp+"coeff"].T + model[bp+"bias"])
        finish_cols.append(xf @ parameters[fp+"coef"].T + parameters[fp+"bias"])
    begin = torch.stack(begin_cols,dim=-2)
    finish = torch.stack(finish_cols,dim=-2)
    def neglog(ref, choice):
        logits = torch.nn.functional.pad(ref, (1,0))
        return logits.logsumexp(-1)-logits.gather(-1,choice.unsqueeze(-1)).squeeze(-1)
    w = torch.ones_like(act,dtype=torch.float64)
    w[act==0] = .1
    bg = neglog(begin, torch.where((act==1)|(act==2),act,0))
    fg = neglog(finish, (act-2).clamp_min(0))
    keep = (act==0)|(act>=3)
    action_loss = (w*(bg+keep*fg)).sum()/w.sum()
    subset = finish[tuple(positions.T)]
    target = torch.from_numpy(a[:,6])
    extra = (subset.logsumexp(-1)-subset.gather(1,target[:,None]).squeeze(1)).mean()
    (action_loss+extra).backward()
    with torch.no_grad():
        base.edge_shared.grad[~before["mask"]] = 0.
    norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.)
    optimizer.step()
    errors = {k:float((p.detach()-after["model"][k]).abs().max()) for k,p in parameters.items()}
    assert max(errors.values()) < 2e-11, errors
    moment_error = 0.
    for idx, s in optimizer.state_dict()["state"].items():
        for k in ("exp_avg","exp_avg_sq","step"):
            moment_error = max(moment_error,float((s[k]-after["optimizer"]["state"][idx][k]).abs().max()))
    assert moment_error < 2e-11
    row = json.loads(read(folder/"updates.jsonl").splitlines()[batch["block"]])
    assert abs(row["loss"]-float((action_loss+extra).detach())) < 1e-10
    return dict(arm=arm, block=batch["block"], hidden_max_error=h_error, param_max_error=errors,
                adam_max_error=moment_error, loss=row["loss"], gradient_norm=float(norm))


def check_fork(item):
    trace = item["trace_artifact"]
    assert sha(trace["path"]) == trace["sha256"]
    with gzip.open(trace["path"], "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    assert len(rows) == 2048
    def field(name):
        return np.array([r[name] for r in rows])[:,4:8]
    active, started, enabled = (field(k) for k in ("active","started","enabled"))
    now = np.array([r["time"] for r in rows])[:,None,None]
    action, req, response = (field(k) for k in ("actions","request_id","result"))
    idle = ~active & enabled
    is_begin, is_finish = (action==1)|(action==2), action>=3
    assert np.array_equal(idle, field("idle"))
    # A not-yet-started slot has no ready time; the JSON trace encodes it as null.
    ready = field("ready_time").astype(np.float64)
    assert np.isfinite(ready[active & started]).all()
    processing = active & enabled & started & (now < ready-1e-6)
    legal = active & enabled & started & ~processing & (now <= field("deadline")+1e-6)
    exact = action == 3 + 4*field("version") + field("payload")
    assert np.array_equal(is_finish & (response==1), legal & exact)
    count = {
        "idle_opportunities":int(idle.sum()), "idle_actions":int((idle & (action!=0)).sum()),
        "idle_begin":int((idle & is_begin).sum()), "idle_finish":int((idle & is_finish).sum()),
        "processing_opportunities":int(processing.sum()), "processing_early_finish":int((processing & is_finish).sum()),
        "legal_finish_opportunities":int(legal.sum()), "legal_finish_attempts":int((legal & is_finish).sum()),
        "legal_finish_correct":int((legal & exact).sum()), "legal_finish_wrong":int((legal & is_finish & ~exact).sum()),
    }
    def keys(mask, ids):
        t, lane, job = np.where(mask)
        return set(zip(lane.tolist(),job.tolist(),ids[t,lane,job].tolist()))
    issued = keys(field("primary_command"),req)
    matured = field("matured_mask")
    ids = field("matured_request_id")
    final = keys(matured,ids) & issued
    feasible = is_begin & (response==1) & (now + field("duration") <= field("deadline")+1e-6)
    completed = final & keys(matured & field("matured_completed"),ids)
    clean = final & keys(matured & field("matured_clean_completed"),ids)
    count.update(primary_matured=len(final),primary_completed=len(completed),
        primary_clean_completed=len(clean),
        primary_feasible_began=len(final & keys(feasible,req)))
    scorer = item["shadow"]["heldout"]["counts"]
    for key,value in count.items():
        assert scorer.get(key,0) == value, (key,value,scorer.get(key))
    at_teacher_finish = field("teacher_actions") >= 3
    assert not (at_teacher_finish & ~legal).any()
    count.update(legal_wait=int((legal & (action==0)).sum()), legal_begin=int((legal & is_begin).sum()),
                 teacher_finish_positions=int(at_teacher_finish.sum()),
                 teacher_finish_wait=int((at_teacher_finish & (action==0)).sum()),
                 teacher_finish_begin=int((at_teacher_finish & is_begin).sum()),
                 teacher_finish_attempted=int((at_teacher_finish & is_finish).sum()))
    return count, final, (completed,clean)


def check_seed(seed):
    path = BASE/f"eval-v1/seed_{seed:03d}/result.json"
    source = json.loads(read(path))
    reference = {}
    success = {}
    pooled = defaultdict(Counter)
    for item in source["records"]:
        c, ids, events = check_fork(item)
        k = (item["condition"],item["family"])
        assert ids == reference.setdefault(k,ids)
        success[(*k,item["arm"])] = events
        pooled[item["arm"],item["condition"]].update(c)
    assert len(source["records"]) == 24 and len(reference)==8
    switches = []
    for (condition, family), both_sets in reference.items():
        for i,label in enumerate(("ordinary","strict")):
            original = success[condition,family,"end_only"][i]
            changed = success[condition,family,"wait_mixed"][i]
            switches.append(dict(condition=condition, family=family, measure=label, matured=len(both_sets),
                both=len(original & changed), only_end=len(original-changed), only_wait=len(changed-original)))
    return dict(seed=seed, evaluation_sha256=sha(path), paired_requests=switches,
                outcomes=[dict(seed=seed,arm=arm,condition=condition,counts=counts,rates=rates(counts))
                          for (arm,condition), counts in pooled.items()])


def check_logs(seed, budget):
    logs = []
    for arm in ARMS[1:]:
        path = BASE/f"train-v1/seed_{seed:03d}/{arm}/updates.jsonl"
        rows = [json.loads(line) for line in read(path).splitlines()]
        assert len(rows) == 128
        assert [r["auxiliary_count"] for r in rows] == budget["per_update_pairs"]
        assert sum(r["moved_positions"] for r in rows) == (budget["moved"] if arm=="wait_mixed" else 0)
        assert all(r["h_start_hash"]==previous["h_end_hash"] for previous,r in zip(rows, rows[1:]))
        logs.append(rows)
    for a,b in zip(*logs):
        for key in ("action_data_hash","pair_hash","auxiliary_targets","auxiliary_request_keys",
                    "action_positions","action_denominator","auxiliary_count"):
            assert a[key] == b[key]
        assert a["active_edges"] == b["active_edges"] and a["inactive_exact_zero"] and b["inactive_exact_zero"]


if __name__ == "__main__":
    budgets = [check_pairs(s) for s in range(3)]
    for seed,budget in enumerate(budgets):
        check_logs(seed,budget)
    updates = [check_update(a) for a in ARMS[1:]]
    # Fork records occupy memory only while their denominators are reconstructed.
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(check_seed,range(3)))
    destination = BASE/"independent-audit.json"
    value = dict(audit="passed", budgets=budgets, independent_update_replays=updates, seeds=outcomes,
                 original_forks=72, script_sha256=sha(__file__))
    destination.write_text(json.dumps(value,indent=2),encoding="utf-8")
    print(destination)
