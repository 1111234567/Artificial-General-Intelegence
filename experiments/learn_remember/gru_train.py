"""A plain one-layer recurrent policy, trained on public ContextWorld streams.

The environment's oracle supplies supervised labels only. Training execution
is explicitly marked teacher/mixed/own; evaluation always executes the native
eleven-way argmax. The recurrent policy receives exactly the public 56 floats.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.lifelong.world import ContextWorld, NEW, REPLACE

OUT = ROOT / "data/learn-remember/gru-v1"
FORMAT = "learn-remember-plain-gru-v1"


class GRUAgent(nn.Module):
    def __init__(self, width=192, input_size=56, n_jobs=2, n_actions=11):
        super().__init__()
        self.width, self.input_size = int(width), int(input_size)
        self.n_jobs, self.n_actions = int(n_jobs), int(n_actions)
        self.core = nn.GRU(input_size, width, num_layers=1)
        self.head = nn.Linear(width, n_jobs * n_actions)

    def config(self):
        return dict(width=self.width, input_size=self.input_size,
                    n_jobs=self.n_jobs, n_actions=self.n_actions)

    def initial_state(self, batch_size, device=None):
        return self.head.weight.new_zeros(batch_size, self.width, device=device)

    def forward(self, obs, hidden):
        values, state = self.core(obs, hidden[None])
        return self.head(values).reshape(*obs.shape[:2], self.n_jobs, self.n_actions), state[0]

    def step(self, obs, state, dt=None, max_step=None):
        # dt is accepted solely for the existing evaluator's API. It is not
        # appended to inputs, and there is no external timing computation.
        values, state = self.forward(obs[None], state)
        return values[0], state

    def parameter_report(self):
        return dict(kind="plain_single_layer_gru", **self.config(),
                    total_parameters=sum(p.numel() for p in self.parameters()),
                    state_scalars=self.width, public_inputs_only=True)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["format"] != FORMAT:
        raise ValueError("unsupported checkpoint")
    model = GRUAgent(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    world = ContextWorld.from_state_dict(checkpoint["world"], device="cpu")
    return model, world, checkpoint["hidden"].to(device), checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class Config:
    seed: int = 0
    width: int = 192
    batch: int = 64
    ticks_per_round: int = 1024
    block: int = 128
    short_rounds: int = 6
    wide_rounds: int = 16
    withdrawal_rounds: int = 12
    epochs: int = 32
    lr: float = .001
    wait_weight: float = .25
    device: str = "cuda"


class Trainer:
    def __init__(self, config):
        self.cfg = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.rng = np.random.default_rng(50000 + config.seed)
        self.model = GRUAgent(config.width).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.world = ContextWorld(seed=20000 + config.seed, batch_size=config.batch,
                                  actor_ids=tuple(range(config.batch)), n_jobs=2,
                                  future_family=3, device="cpu", dt=.25)
        self.hidden = self.model.initial_state(config.batch, device="cpu")
        self.work = dict(environment_ticks=0, lane_ticks=0, gradient_updates=0,
                         gradient_lane_ticks=0, replay_family3_during_withdrawal=0,
                         withdrawal_residual_family3_commands=0,
                         withdrawal_residual_family3_labels=0,
                         withdrawal_residual_family3_active_tick_slots=0)
        self.folder = OUT / f"seed_{config.seed:03d}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stage, self.round = "random", 0
        self.withdrawal_started = False

    def save(self, name):
        checkpoint = dict(format=FORMAT, model_config=self.model.config(),
                          model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                          optimizer=self.optimizer.state_dict(), world=self.world.state_dict(),
                          hidden=self.hidden.detach().cpu(), config=asdict(self.cfg),
                          stage=self.stage, round=self.round, work=copy.deepcopy(self.work),
                          numpy_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                          source_sha256=sha(__file__), saved_time=time.time())
        path = self.folder / f"{name}.pt"
        torch.save(checkpoint, path)
        write_json(self.folder / f"{name}.json", dict(path=str(path), sha256=sha(path),
                   model=self.model.parameter_report(), stage=self.stage, round=self.round,
                   work=self.work, world=self.world.metrics(),
                   hidden_norm=float(self.hidden.norm())))
        print(json.dumps(dict(event="checkpoint", seed=self.cfg.seed, name=name,
                              path=str(path), work=self.work)), flush=True)
        return path

    def collect(self, teacher_probability):
        """One continuous actual world; only model obs and receipts enter h."""
        cfg = self.cfg
        policy = copy.deepcopy(self.model).cpu().eval()
        anchor = self.hidden.clone()
        obs_rows, target_rows, action_rows = [], [], []
        # Lane-level intervention creates complete successful demonstrations
        # alongside own action trajectories, rather than replacing arbitrary
        # error ticks and calling the resulting stream autonomous.
        teacher_lanes = torch.from_numpy(self.rng.random(cfg.batch) < teacher_probability)
        initial_counts = {k:int(v.sum()) for k,v in self.world.counts.items()}
        exposure_before = self.world.family_exposures()
        with torch.no_grad():
            for tick in range(cfg.ticks_per_round):
                obs = self.world.observe()
                labels = self.world.oracle("midpoint")
                logits, self.hidden = policy.step(obs, self.hidden)
                chosen = logits.argmax(-1)
                actions = torch.where(teacher_lanes[:, None], labels, chosen)
                obs_rows.append(obs.clone())
                target_rows.append(labels.clone())
                action_rows.append(actions.clone())
                if self.withdrawal_started:
                    old = self.world.current_family == 3
                    self.work["withdrawal_residual_family3_commands"] += int((
                        (self.world.event_family == 3) & ((self.world.event_kind == NEW) | (self.world.event_kind == REPLACE))).sum())
                    self.work["withdrawal_residual_family3_labels"] += int((old & (labels != 0)).sum())
                    self.work["withdrawal_residual_family3_active_tick_slots"] += int((old & self.world.active).sum())
                self.world.act(actions)
        self.work["environment_ticks"] += cfg.ticks_per_round
        self.work["lane_ticks"] += cfg.ticks_per_round * cfg.batch
        data = dict(obs=torch.stack(obs_rows), labels=torch.stack(target_rows),
                    actions=torch.stack(action_rows), anchor=anchor,
                    teacher_lanes=teacher_lanes, teacher_probability=teacher_probability,
                    stage=self.stage, round=self.round,
                    exposure_before=exposure_before, exposure_after=self.world.family_exposures(),
                    metrics_delta={k:int(v.sum())-initial_counts[k] for k,v in self.world.counts.items()})
        # Retained evidence distinguishes executed actions from labels; this
        # file is never admitted to future stages as a replay buffer.
        torch.save(data, self.folder / f"trajectory_{self.stage}_{self.round:03d}.pt")
        return data

    def fit(self, data):
        cfg = self.cfg
        observations = data["obs"].to(cfg.device)
        targets = data["labels"].to(cfg.device)
        weights = torch.ones(11, device=cfg.device)
        weights[0] = cfg.wait_weight
        self.model.train()
        loss_total = correct = denominator = 0
        for epoch in range(cfg.epochs):
            # Always replay this new contiguous block in temporal order,
            # starting from its actual collection anchor. No shuffled windows
            # or private state reconstructions are used.
            hidden = data["anchor"].to(cfg.device)
            for start in range(0, len(observations), cfg.block):
                obs = observations[start:start+cfg.block]
                labels = targets[start:start+cfg.block]
                logits, hidden = self.model(obs, hidden)
                loss = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), weight=weights)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                hidden = hidden.detach()
                loss_total += float(loss.detach())
                correct += int((logits.detach().argmax(-1) == labels).sum())
                denominator += labels.numel()
                self.work["gradient_updates"] += 1
                self.work["gradient_lane_ticks"] += obs.shape[0] * obs.shape[1]
        return dict(mean_loss=loss_total/(cfg.epochs * (len(observations)//cfg.block)),
                    training_all_action_accuracy=correct/denominator)

    def stage_run(self, stage, rounds, families, duration, gap):
        self.stage = stage
        self.world.set_family_policy(families)
        self.world.set_regime(duration_range=duration, gap_range=gap,
                              name=f"gru-v1/{stage}")
        self.withdrawal_started = stage == "withdrawal"
        for round_index in range(rounds):
            self.round = round_index
            # Every stage starts with plentiful successful expert execution;
            # its latter half is strictly native actions with expert labels.
            beta = max(0., 1. - round_index / max(1, rounds // 2))
            start = time.perf_counter()
            data = self.collect(beta)
            fit = self.fit(data)
            delta = data["metrics_delta"]
            row = dict(seed=self.cfg.seed, stage=stage, round=round_index,
                       teacher_probability=beta, teacher_lanes=int(data["teacher_lanes"].sum()),
                       actual_action_policy="teacher" if beta == 1 else "own" if beta == 0 else "mixed_by_lane",
                       label_policy="private causal oracle midpoint; never model input",
                       execution_metrics=delta, fit=fit, work=copy.deepcopy(self.work),
                       seconds=time.perf_counter()-start)
            with (self.folder / "updates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(dict(event="round", **row)), flush=True)
            if round_index + 1 == rounds // 2:
                self.save(stage + "_mid")
        self.save(stage)

    def run(self):
        # A zero-action causal boundary permits standard native fork evaluators
        # even for the untrained snapshot; no target is supplied to the model.
        obs = self.world.observe()
        with torch.no_grad():
            _, self.hidden = self.model.cpu().step(obs, self.hidden)
        self.world.act(torch.zeros((self.cfg.batch, 2), dtype=torch.long))
        self.model.to(self.cfg.device)
        self.save("random")
        self.stage_run("short", self.cfg.short_rounds, (3,), (1., 4.), (.5, 2.))
        self.stage_run("wide", self.cfg.wide_rounds, (0, 1, 2, 3), (1., 16.), (.5, 32.))
        self.stage_run("withdrawal", self.cfg.withdrawal_rounds, (0, 1, 2), (1., 16.), (.5, 32.))


def main():
    parser = argparse.ArgumentParser()
    for key, field in Config.__dataclass_fields__.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args()
    config = Config(**vars(args))
    if config.ticks_per_round % config.block:
        raise ValueError("ticks_per_round must be divisible by block")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"plan_seed_{config.seed:03d}.json", dict(format=FORMAT, config=asdict(config),
          source_sha256=sha(__file__), planned_seed_selection="All fixed seeds 0,1,2; no success selection",
          model="nn.GRU public56->192, one layer, Linear22; native11way argmax per slot",
          training="Continuous exact CPU world and hidden, GPU contiguous sequence supervised CE; current-round repeated epochs only; oracle label vs actual execution explicitly retained",
          withdrawal="future NEW families 0,1,2; no family3 historical replay; inherited live pending residual exposures counted separately"))
    Trainer(config).run()


if __name__ == "__main__":
    main()
