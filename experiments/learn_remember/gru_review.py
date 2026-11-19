"""Fixed 1/8-budget historical review, forked from every seed's wide endpoint.

This never changes gru_train.py or its no-review branch. The positive control
uses precisely the last four complete wide streams, fixed before reading any
formal retention scores. Its new world follows its own action receipts; the
action-independent external opportunities and teacher lane schedule are paired.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from experiments.learn_remember.gru_train import (
    Config, GRUAgent, Trainer, OUT, FORMAT, load_checkpoint, sha, write_json,
)
from experiments.lifelong.world import ContextWorld, NEW, REPLACE


HISTORICAL_ROUNDS = (12, 13, 14, 15)


def historical_metadata(folder):
    """Read private fields only in a detached audit of recorded executions."""
    model, world, hidden, checkpoint = load_checkpoint(folder / "wide_mid.pt")
    stats, sources = {}, {}
    for index in range(8, 16):
        path = folder / f"trajectory_wide_{index:03d}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        counter = dict(all_tick_slots=0, active_tick_slots=0,
                       family_tick_slots=[0]*4, active_family_tick_slots=[0]*4,
                       nonwait_family_labels=[0]*4, command_families=[0]*4)
        for tick in range(len(data["obs"])):
            obs = world.observe()
            if not torch.equal(obs, data["obs"][tick]):
                raise AssertionError("historical public observation replay mismatch")
            if not torch.equal(world.oracle("midpoint"), data["labels"][tick]):
                raise AssertionError("historical label replay mismatch")
            families = world.current_family
            command = (world.event_kind == NEW) | (world.event_kind == REPLACE)
            counter["all_tick_slots"] += families.numel()
            counter["active_tick_slots"] += int(world.active.sum())
            for family in range(4):
                mask = families == family
                counter["family_tick_slots"][family] += int(mask.sum())
                counter["active_family_tick_slots"][family] += int((mask & world.active).sum())
                counter["nonwait_family_labels"][family] += int((mask & (data["labels"][tick] != 0)).sum())
                counter["command_families"][family] += int(((world.event_family == family) & command).sum())
            world.act(data["actions"][tick])
        if index in HISTORICAL_ROUNDS:
            stats[str(index)] = counter
            sources[str(index)] = dict(path=str(path), sha256=sha(path))
    return stats, sources


class ReviewTrainer(Trainer):
    def __init__(self, seed):
        folder = OUT / f"seed_{seed:03d}"
        checkpoint = torch.load(folder / "wide.pt", map_location="cpu", weights_only=False)
        super().__init__(Config(**checkpoint["config"]))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.world = ContextWorld.from_state_dict(checkpoint["world"], device="cpu")
        self.hidden = checkpoint["hidden"].clone()
        self.work = copy.deepcopy(checkpoint["work"])
        self.work.update(review_updates=0, review_gradient_lane_ticks=0,
                         review_family3_gradient_tick_slots=0,
                         review_family3_nonwait_labels=0)
        self.rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        self.metadata, self.sources = historical_metadata(folder)
        self.history = {index:torch.load(folder/f"trajectory_wide_{index:03d}.pt",
                           map_location="cpu", weights_only=False) for index in HISTORICAL_ROUNDS}
        if self.cfg.epochs % 8:
            raise ValueError("review requires epochs divisible by 8")
        write_json(self.folder / "review_plan.json", dict(
            format=FORMAT+"/review", source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
            main_source_sha256=sha(ROOT / "experiments/learn_remember/gru_train.py"),
            wide_checkpoint_sha256=sha(folder/"wide.pt"),
            fixed_before_formal_retention_results=True,
            historical_rounds=HISTORICAL_ROUNDS, historical_stats=self.metadata,
            historical_sources=self.sources,
            allocation="Each eighth epoch uses historical data; all other epochs new data. Same 32 epochs/round,128-tick updates,64 lanes,Adam state and learning rate as native no-review withdrawal.",
            control="Own closed-loop branch from same full wide endpoint; same action-independent external stream and teacher-lane RNG. Different actions/receipts and hidden trajectories are expected. Historical anchors are actual numeric hidden states from original collection; no private state enters policy.",
            history_selection="Round 12+(withdrawal_round modulo4), four entire sequences fixed without success selection."))

    def fit_review(self, live, historical, source_index):
        cfg = self.cfg
        sources = {}
        for name, data in (("live",live),("review",historical)):
            sources[name] = {key:data[key].to(cfg.device) for key in ("obs","labels","anchor")}
        weights = torch.ones(11, device=cfg.device)
        weights[0] = cfg.wait_weight
        total, counts = 0., {"live":0,"review":0}
        self.model.train()
        for epoch in range(cfg.epochs):
            name = "review" if (epoch+1)%8 == 0 else "live"
            data = sources[name]
            hidden = data["anchor"]
            for start in range(0,len(data["obs"]),cfg.block):
                obs = data["obs"][start:start+cfg.block]
                targets = data["labels"][start:start+cfg.block]
                logits, hidden = self.model(obs,hidden)
                loss = F.cross_entropy(logits.flatten(0,2), targets.flatten(), weight=weights)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
                self.optimizer.step()
                hidden = hidden.detach()
                total += float(loss.detach())
                counts[name] += 1
                self.work["gradient_updates"] += 1
                self.work["gradient_lane_ticks"] += obs.shape[0]*obs.shape[1]
                if name == "review":
                    self.work["review_updates"] += 1
                    self.work["review_gradient_lane_ticks"] += obs.shape[0]*obs.shape[1]
            if name == "review":
                stats = self.metadata[str(source_index)]
                self.work["review_family3_gradient_tick_slots"] += stats["family_tick_slots"][3]
                self.work["review_family3_nonwait_labels"] += stats["nonwait_family_labels"][3]
        return dict(mean_loss=total/sum(counts.values()), updates=counts,
                    historical_source_round=source_index, historical_sha256=self.sources[str(source_index)]["sha256"])

    def run_review(self):
        self.stage = "review"
        self.withdrawal_started = True
        self.world.set_family_policy((0,1,2))
        self.world.set_regime(duration_range=(1.,16.),gap_range=(.5,32.),name="gru-v1/withdrawal")
        for index in range(self.cfg.withdrawal_rounds):
            self.round = index
            beta = max(0.,1.-index/max(1,self.cfg.withdrawal_rounds//2))
            start = time.perf_counter()
            live = self.collect(beta)
            source_index = HISTORICAL_ROUNDS[index%len(HISTORICAL_ROUNDS)]
            result = self.fit_review(live,self.history[source_index],source_index)
            paired = torch.load(self.folder/f"trajectory_withdrawal_{index:03d}.pt",map_location="cpu",weights_only=False)
            if not torch.equal(paired["teacher_lanes"],live["teacher_lanes"]):
                raise AssertionError("teacher lane intervention schedule mismatch")
            if paired["exposure_after"] != live["exposure_after"]:
                raise AssertionError("external family opportunity mismatch")
            public_live=live["obs"].reshape(-1,self.cfg.batch,2,28)
            public_paired=paired["obs"].reshape(-1,self.cfg.batch,2,28)
            if not (torch.equal(public_live[...,:11],public_paired[...,:11]) and
                    torch.equal(public_live[...,24:28],public_paired[...,24:28])):
                raise AssertionError("public external event stream mismatch")
            row = dict(event="review_round",seed=self.cfg.seed,round=index,beta=beta,
                       teacher_lanes=int(live["teacher_lanes"].sum()),
                       external_family_exposures_match=True,public_external_events_match=True,teacher_lanes_match=True,
                       execution_metrics=live["metrics_delta"],fit=result,work=copy.deepcopy(self.work),
                       seconds=time.perf_counter()-start)
            with (self.folder/"review_updates.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            if index+1 == self.cfg.withdrawal_rounds//2:
                self.save("review_mid")
        self.save("review")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    ReviewTrainer(args.seed).run_review()


if __name__=="__main__":
    main()
