"""Train an address from the original simulator's actual random-attempt receipts."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.undirected import uncapped_private_symbol_world as private
from experiments.undirected.evolved_plasticity_world import NEW, READY, FIRST_FINISH, _receipt_table
from experiments.undirected.uncapped_reward_precision import _ACK, _REJ
from . import research_audit as audit
from .legacy_grounded import (
    DEFAULT_GRAPH, GroundedAgent, Setting, base_config, native_tape, paired_summary, play,
    public_calibration, read, sense, sha,
)


TRAIN_SEEDS = tuple(range(4_506_000_000, 4_506_000_064))
DEV_SEEDS = tuple(range(4_507_000_000, 4_507_000_016))
LATENT, HIDDEN = 8, 32
INIT = 57610043


class RandomAttemptCollector:
    """One uniform attempt at a random legal tick. Record only its real receipt."""

    def __init__(self, graph, actors, trials):
        self.graph = graph
        self.n = len(actors)
        self.rows = np.arange(self.n)
        rngs = [np.random.default_rng(np.random.SeedSequence([s, 0x647501])) for s in actors]
        self.choices = np.stack([r.integers(4, size=trials) for r in rngs], axis=1)
        self.delays = np.stack([r.integers(1, 5, size=trials) for r in rngs], axis=1)
        hidden = graph.dynamics.shape[0]
        self.state = np.zeros((self.n, hidden), np.float32)
        self.features = np.zeros((trials, self.n, hidden), np.float32)
        self.receipts = np.zeros((trials, self.n), np.int8)
        self.age = np.full(self.n, -1, np.int32)
        self.pending = self.age.copy()
        self.ready_count = np.zeros(self.n, np.int8)
        self.tried = np.zeros(self.n, bool)

    def act(self, observed):
        frame = observed[0].numpy()
        pos, neg = frame[:, _ACK] > 0, frame[:, _REJ] > 0
        if np.any(pos & neg):
            raise ValueError("The original world returned two signs for one attempt")
        arrived = (pos | neg) & (self.pending >= 0)
        idx = np.flatnonzero(arrived)
        self.receipts[self.pending[idx], idx] = pos[idx].astype(np.int8) - neg[idx].astype(np.int8)
        self.pending[idx] = -1
        start = frame[:, NEW] > 0
        self.age[start] += 1
        self.ready_count[start] = 0
        self.tried[start] = False
        self.state = self.graph.advance(self.state, frame)
        ready = frame[:, READY] > 0
        self.ready_count += ready
        chosen = ready & ~self.tried & (self.ready_count == self.delays[self.age, self.rows])
        idx = np.flatnonzero(chosen)
        self.tried[idx] = True
        self.pending[idx] = self.age[idx]
        self.features[self.age[idx], idx] = self.state[idx]
        action = np.zeros((1,self.n,2), np.int64)
        action[0,idx,0] = FIRST_FINISH + self.choices[self.age[idx], idx]
        return torch.from_numpy(action)


def collect(graph, seeds):
    events = []
    for offset in range(0, len(seeds), 8):
        group = seeds[offset:offset + 8]
        _, tape = native_tape(group)
        actors = [seed * 8 + n for seed in group for n in range(8)]
        collector = RandomAttemptCollector(graph, actors, len(tape.raw))
        private.rollout_with_table(_receipt_table("cpu").half())(collector, tape, 1)
        if (collector.receipts == 0).any() or (collector.pending >= 0).any():
            raise ValueError("An attempted action did not receive its signed original receipt")
        events.append((collector.features.transpose(1, 0, 2), collector.choices.T, collector.receipts.T))
    feature, action, sign = [np.concatenate(x) for x in zip(*events, strict=True)]
    return feature, action, sign


class Address(nn.Module):
    def __init__(self, mean, std, dimension=LATENT):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.first = nn.Linear(len(mean), HIDDEN)
        self.second = nn.Linear(HIDDEN, dimension)

    def forward(self, state):
        address = self.second(self.first((state - self.mean) / self.std).tanh())
        return address / torch.sqrt(.01 + address.square().sum(-1, keepdim=True))

    def export(self):
        array = lambda tensor: tensor.detach().cpu().numpy().copy()
        return NumpyAddress(array(self.mean), array(self.std), array(self.first.weight.T), array(self.first.bias),
                            array(self.second.weight.T), array(self.second.bias))


class NumpyAddress:
    def __init__(self, mean, std, first, first_bias, second, second_bias):
        self.mean, self.std = mean, std
        self.first, self.first_bias, self.second, self.second_bias = first, first_bias, second, second_bias
        self.dimension = len(second_bias)

    def __call__(self, state):
        address = np.tanh((state - self.mean) / self.std @ self.first + self.first_bias) @ self.second + self.second_bias
        return address / np.sqrt(.01 + (address * address).sum(1, keepdims=True))


def causal_predictions(keys, attempted, receipt, ridge):
    # Independent outcome synapses for each actually tried action. The Cholesky
    # term below is the batch equivalent of the prefixes of online RLS.
    same_action = attempted[:, :, None] == attempted[:, None, :]
    gram = (keys @ keys.transpose(1, 2)) * same_action
    reg = torch.eye(keys.shape[1], dtype=keys.dtype, device=keys.device) * ridge
    lower = torch.linalg.cholesky(gram + reg)
    impulse = torch.linalg.solve_triangular(lower, receipt[..., None], upper=False)
    return (lower.tril(diagonal=-1) @ impulse).squeeze(-1)


def data_tensors(data):
    return tuple(torch.as_tensor(x, dtype=t) for x, t in zip(data, (torch.float32, torch.long, torch.float32), strict=True))


def shuffled_within_action(data, seed):
    rng = np.random.default_rng(seed)
    x, a, y = data
    permuted = y.copy()
    for life in range(len(y)):
        for action in range(4):
            select = a[life] == action
            permuted[life, select] = rng.permutation(y[life, select])
    return x, a, permuted


def fit(train, dev, norm, ridge, steps, *, shuffled):
    torch.manual_seed(INIT)
    rng = np.random.default_rng(INIT + 1)
    network = Address(*norm)
    learner = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=.0003)
    if shuffled:
        train = shuffled_within_action(train, INIT + 2)
        dev = shuffled_within_action(dev, INIT + 3)
    x, a, y = data_tensors(train)
    xdev, adev, ydev = data_tensors(dev)
    history, state, score = [], None, float("inf")
    for step in range(steps + 1):
        if step > 0:
            batch = torch.as_tensor(rng.integers(len(x), size=24))
            logits = causal_predictions(network(x[batch]), a[batch], y[batch], ridge)
            loss = (logits - y[batch]).square().mean()
            learner.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 2.)
            learner.step()
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                valid = torch.cat([causal_predictions(network(xv), av, yv, ridge)
                                   for xv,av,yv in zip(xdev.split(32), adev.split(32), ydev.split(32), strict=True)])
                val = float((valid - ydev).square().mean())
            history.append({"step":step, "actual_outcome_brier_signed":val})
            if val < score:
                state = {k: value.detach().clone() for k,value in network.state_dict().items()}
                score = val
            print("shuffled" if shuffled else "actual", step, round(val, 6), flush=True)
    assert state is not None
    network.load_state_dict(state)
    return network, history


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=550)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = audit.OUT
    g = sense(args.graph)
    cfg = base_config()
    train, dev = collect(g, TRAIN_SEEDS), collect(g, DEV_SEEDS)
    norm = (train[0].mean(axis=(0,1)), train[0].std(axis=(0,1)).clip(min=.01))
    trained, history = fit(train, dev, norm, cfg.regularization, args.steps, shuffled=False)
    shuffled, shadow_hist = fit(train, dev, norm, cfg.regularization, args.steps, shuffled=True)
    torch.manual_seed(INIT)
    random = Address(*norm)
    for label, model in (("actual",trained),("shuffled",shuffled),("initial",random)):
        torch.save({"model":model.state_dict(),"source_graph_sha256":sha(args.graph),"training_seeds":TRAIN_SEEDS,
                    "development_seeds":DEV_SEEDS,"prior":"only actual attempted-sign outcome prefixes"},
                   root/f"legacy_grounded-outcomes-{label}.pt")
    primary = read(root/"legacy_grounded-confirmed.json")
    setting = replace(Setting(**primary["setting"]), learning_rate=1.0)
    cal = public_calibration(g)
    native = {"train_lives":len(train[0]),"train_actually_received_labels":train[2].size,
              "true_train_reward_positive_rate":float((train[2]>0).mean()),
              "dev_lives":len(dev[0]),"dev_actually_received_labels":dev[2].size,
              "train_seeds":TRAIN_SEEDS,"development_seeds":DEV_SEEDS}
    result={"graph_sha256":sha(args.graph),"source_sha256":sha(Path(__file__)),
            "setting":setting.__dict__,"recorded_legal_exploration":native,
            "learning_curves":{"actual":history,"pulled_apart_by_action":shadow_hist},"sets":{}}
    keys = {name: model.export() for name,model in (("actual",trained),("shuffled",shuffled),("initial",random))}
    variants = {"actual": (keys["actual"],"none"), "shuffled": (keys["shuffled"],"none"),
                "initial": (keys["initial"],"none"), "no_write": (keys["actual"],"no_write"),
                "wrong_binding": (keys["actual"],"wrong_binding"), "calibrated_factor_ratematched": (None,"none")}
    for label,prior in primary["sets"].items():
        roots=tuple(prior["roots"])
        rows={"trained_factor":prior["rows"]["trained_factor"],"table_retry":prior["rows"]["table_retry"]}
        for arm,(k,intervention) in variants.items():
            fn = lambda actors: GroundedAgent(cfg,g,cal,setting,actors,intervention=intervention,learned_key=k)
            rows[arm] = [r for offset in range(0,len(roots),8) for r in play(roots[offset:offset+8],fn)]
            print(label,arm,round(np.mean([r["strict"] for r in rows[arm]]),5),
                  round(np.mean([r["pair_strict_rate"] for r in rows[arm]]),5),flush=True)
        result["sets"][label]={"roots":roots,"rows":rows,"summary":paired_summary(rows)}
        (root/"legacy_grounded-outcomes.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
