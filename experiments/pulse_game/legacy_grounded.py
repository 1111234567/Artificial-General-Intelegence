"""Retain the original request world while replacing its sensory encoder."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.undirected import local_rule_evaluator as evaluator, local_rule_world as worlds
from experiments.undirected import uncapped_private_symbol_world as private
from experiments.undirected.evolved_plasticity_world import NEW, READY, CONTEXT, RAW, FIRST_FINISH, _receipt_table
from experiments.undirected.uncapped_answer_feedback import _RecordPrimaryActions
from experiments.undirected.uncapped_reward_precision import _ACK, _REJ
from . import ring_spikes, ring_world, symmetry_grounded
from .attractor_search import OUT
from .bridge_controls import PublicMemory
from .legacy_bridge import BridgeAgent, Options, batch as legacy_batch
from . import research_audit as audit


DEFAULT_GRAPH = OUT / "symmetry-grounded-v1" / "public-delta-s7243-before.pt"
RING = OUT / "ring-value-search-v1" / "frozen.json"
CALIBRATION_SEEDS = tuple(range(4_502_000_000, 4_502_000_012))
DEVELOPMENT_SEEDS = tuple(range(4_501_000_000, 4_501_000_008))
EXTERNAL_SEEDS = tuple(range(4_503_000_000, 4_503_000_032))
TORCH_THREADS = 1


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Sense:
    dynamics: np.ndarray
    afferents: np.ndarray
    odor: np.ndarray
    bias: np.ndarray

    def advance(self, state, frame):
        cue = frame[:, RAW]
        context = frame[:, CONTEXT.start:CONTEXT.start + 2]
        return state @ self.dynamics + np.column_stack((cue, context)) @ self.afferents


@dataclass(frozen=True)
class Calibration:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray

    def score(self, state):
        return (state - self.mean) / self.scale @ self.weight + self.bias


@dataclass(frozen=True)
class Setting:
    mode: str = "calibrated_soft"
    value_gain: float = 0.36
    ready_gain: float = 1.10
    motor_leak: float = 0.55
    learning_rate: float | None = None


def sense(path, *, random_seed=None):
    if random_seed is None:
        f = symmetry_grounded.load(path)
    else:
        record = torch.load(path, map_location="cpu", weights_only=True)
        torch.manual_seed(random_seed)
        f = symmetry_grounded.Grounded(symmetry_grounded.core.Config(**record["spec"])).export()
    return Sense(f["matrix"], f["sensors"], *f["ground"])


@lru_cache(maxsize=6)
def native_tape(seeds):
    tapes = [worlds.schedule(seed=s, actors=8, visits=16, device="cpu") for s in seeds]
    dims = dict(public=2, stage=2, raw=1, context=1, target=1, maps=0)
    packed = replace(tapes[0], **{k: torch.cat([getattr(t, k) for t in tapes], dim=d) for k, d in dims.items()})
    return tapes, packed


def public_calibration(graph, seeds=CALIBRATION_SEEDS):
    """Fit six public sensory lines on recurrent states, with no target or outcome input."""
    _, tape = native_tape(tuple(seeds))
    port = tape.public[..., 0, :].numpy()
    state = np.zeros((port.shape[2], graph.dynamics.shape[0]), np.float32)
    inputs, sensory = [], []
    for request in port:
        onset = np.column_stack((request[0][:, RAW], request[0][:, CONTEXT.start:CONTEXT.start + 2]))
        for frame in request:
            state = graph.advance(state, frame)
            ready = frame[:, READY] > 0
            inputs.append(state[ready])
            sensory.append(onset[ready])
    x = np.concatenate(inputs).astype(np.float64)
    y = np.concatenate(sensory).astype(np.float64)
    mu = x.mean(0)
    std = x.std(0).clip(min=.01)
    normalized = (x - mu) / std
    w = np.linalg.solve(normalized.T @ normalized + .001 * len(x) * np.eye(x.shape[1]),
                        normalized.T @ (y - y.mean(0)))
    return Calibration(mu.astype(np.float32), std.astype(np.float32), w.astype(np.float32), y.mean(0).astype(np.float32))


def prepare_key(scores, domain, exact):
    if exact:
        return np.eye(domain, dtype=np.float32)[scores.argmax(1)]
    logits = 6 * scores
    weights = np.exp(logits - logits.max(1, keepdims=True))
    return weights / weights.sum(1, keepdims=True)


class GroundedAgent(BridgeAgent):
    def __init__(self, cfg, graph, calibration, setting, seeds, *, intervention="none", learned_key=None):
        self.graph = graph
        self.calibration = calibration
        self.setting = setting
        self.intervention = intervention
        self.learned_key = learned_key
        latent = 8 if learned_key is None else learned_key.dimension
        rate = cfg.learning_rate if setting.learning_rate is None else setting.learning_rate
        if intervention == "no_write":
            rate = 0.0
        cfg = replace(cfg, units=graph.dynamics.shape[0], latent=latent, learning_rate=rate)
        opt = Options(actuator="lif", key="continuous", ready_gain=setting.ready_gain,
                      value_gain=setting.value_gain, motor_leak=setting.motor_leak)
        super().__init__(cfg, opt, seeds, 3)
        self.sensory = np.zeros((len(seeds), graph.dynamics.shape[0]), np.float32)
        self.context = np.zeros((len(seeds), 2), np.float32)
        self.eye_row = np.arange(self.n)

    def address(self, raw):
        self.sensory = self.graph.advance(self.sensory, raw)
        if self.learned_key is not None:
            return self.learned_key(self.sensory)
        if self.setting.mode == "source_factor":
            visible = raw[:, CONTEXT.start:CONTEXT.start + 2]
            update = visible.any(1)
            self.context[update] = visible[update]
            odor = self.sensory @ self.graph.odor
            context = self.context
        else:
            score = self.calibration.score(self.sensory)
            exact = self.setting.mode == "calibrated_categorical"
            odor = prepare_key(score[:, :4], 4, exact)
            context = prepare_key(score[:, 4:], 2, exact)
        v = (context[:, :, None] * odor[:, None, :]).reshape(self.n, 8)
        return v / np.sqrt(.01 + (v * v).sum(1, keepdims=True))

    def act(self, observation):
        raw = observation[0].numpy()
        learner = self.p
        ready = raw[:, READY] != 0
        receipt = np.zeros((self.n, ring_world.OBS), np.float32)
        receipt[:, ring_world.POSITIVE] = raw[:, _ACK]
        receipt[:, ring_world.NEGATIVE] = raw[:, _REJ]
        if self.intervention == "wrong_binding":
            amount = 4 if self.learned_key is None else max(1, learner.cfg.latent // 2)
            learner.last_key = np.roll(learner.last_key, amount, axis=1)
        learner._learn(receipt)
        key = self.address(raw)
        values = np.einsum("bfk,bk->bf", learner.w, key)
        random = self._tie()
        # Same READY-modulated, laterally inhibited LIF motor interface as the original bridge.
        self.v *= self.opt.motor_leak
        self.v += .04 + self.opt.ready_gain * ready[:, None] + self.opt.value_gain * np.tanh(values) + .035 * (random - .5)
        np.maximum(self.v, -1, out=self.v)
        choice = self.v.argmax(1)
        fired = self.v[self.eye_row, choice] >= 1
        self.v[fired] -= .68
        self.v[np.flatnonzero(fired), choice[fired]] = 0
        self.motor.fill(0)
        self.motor[np.flatnonzero(fired), choice[fired]] = 1
        learner.last_odor[:] = self.motor * ready[:, None]
        learner.last_key[:] = key
        learner.last_pred = values[self.eye_row, choice]
        self.tick += 1
        actions = np.zeros((1, self.n, 2), np.int64)
        actions[0, fired, 0] = FIRST_FINISH + choice[fired]
        return torch.from_numpy(actions)


class RandomActions:
    def __init__(self, seeds):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed for seed in seeds]))
        self.n = len(seeds)

    def act(self, observation):
        f = observation[0].numpy()
        result = np.zeros((1, self.n, 2), np.int64)
        valid = f[:, READY] != 0
        random = self.rng.integers(4, size=self.n)
        result[0, valid, 0] = FIRST_FINISH + random[valid]
        return torch.from_numpy(result)


def play(seeds, factory):
    tapes, group = native_tape(tuple(seeds))
    actors = [seed * 8 + i for seed in seeds for i in range(8)]
    agent = factory(actors)
    recorder = _RecordPrimaryActions(agent, group, 1)
    results = private.rollout_with_table(_receipt_table("cpu").half())(recorder, group, 1)
    rows = []
    for i, seed in enumerate(seeds):
        tape = tapes[i]
        actions = recorder.actions[..., 8*i:8*(i+1)]
        raw = evaluator.trace_statistics(actions, tape, worlds.partners(tape.raw, tape.context))
        row = {k: v.item() for k, v in raw.items()}
        row["selection_score"] = (
            row["strict"] + .1 * (row["completion"] - row["strict"])
            + (row["pairs_both_strict"] - .1 * row["post_answer_actions"]) / 512
        )
        row["seed"] = int(seed)
        if isinstance(agent, GroundedAgent):
            row["actual_writes"] = int(agent.p.writes[8*i:8*(i+1)].sum())
        rows.append(row)
    for k in ("strict", "completion"):
        np.testing.assert_allclose(np.mean([row[k] for row in rows]), results[k].item(), rtol=0, atol=1e-7)
    return rows


def report_interval(values):
    x = np.asarray(values, np.float64)
    samples = np.random.default_rng(16183694).integers(len(x), size=(3000, len(x)))
    return {"mean": float(x.mean()), "ci95": np.quantile(x[samples].mean(1), [.025, .975]).tolist(), "worlds": len(x)}


def paired_summary(arms):
    keys = ("strict", "completion", "first_finish_correct", "pair_strict_rate", "premature_request_rate", "ready_coverage")
    roots = tuple(row["seed"] for row in next(iter(arms.values())))
    different = any(tuple(row["seed"] for row in a) != roots for a in arms.values())
    if different:
        raise ValueError("Worlds do not pair")
    results = {key: {m: report_interval([r[m] for r in rows]) for m in keys} for key, rows in arms.items()}
    if "trained_factor" in arms:
        selected = "trained_factor"
        results["differences_trained_factor"] = {
            key: {m: report_interval([a[m] - b[m] for a, b in zip(arms[selected], rows, strict=True)])
                  for m in ("strict", "pair_strict_rate")} for key, rows in arms.items() if key != selected}
    return results


def select(cfg, graph, calibration):
    data = []
    for mode in ("source_factor", "calibrated_soft", "calibrated_categorical"):
        for gain in (.22, .36, .59):
            setting = Setting(mode=mode, value_gain=gain)
            r = play(DEVELOPMENT_SEEDS, lambda actors: GroundedAgent(cfg, graph, calibration, setting, actors))
            chosen = np.mean([x["selection_score"] for x in r])
            data.append({"setting": asdict(setting), "selection": float(chosen), "rows": r})
            print("development", mode, gain, round(float(chosen), 5),
                  round(np.mean([x["strict"] for x in r]), 5), flush=True)
    return max(data, key=lambda x: x["selection"]), data


def base_config():
    return ring_spikes.Config(**read(RING)["best"]["config"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--step", choices=("select", "confirm"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(TORCH_THREADS)
    cfg = base_config()
    root = audit.OUT
    root.mkdir(parents=True, exist_ok=True)
    graph = sense(args.graph)
    cal = public_calibration(graph)
    chosen = root / "legacy_grounded-selected.json"
    if args.step == "select":
        best, trials = select(cfg, graph, cal)
        payload = {"source_graph": str(args.graph), "source_graph_sha256": sha(args.graph),
                   "source_sha256": sha(Path(__file__)), "calibration_seeds": CALIBRATION_SEEDS,
                   "development_seeds": DEVELOPMENT_SEEDS, "best": best, "trials": trials}
        chosen.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    freeze = read(chosen)
    if sha(args.graph) != freeze["source_graph_sha256"]:
        raise ValueError("The frozen parent changed")
    setting = Setting(**freeze["best"]["setting"])
    random = sense(args.graph, random_seed=830019)
    rc = public_calibration(random)
    old = read(OUT.parents[0] / "undirected" / "2026-11-19-local-rule-v1" / "audits" / "0128.json")["seeds"]
    native_report = read(root / "legacy-bridge-confirmed.json")["arms"]["continuous_lif"]["true"]
    legacy_options = Options(**read(root / "legacy-bridge-selected.json")["selected"]["continuous_lif"]["options"])
    arms = {
        "trained_factor": lambda actors: GroundedAgent(cfg, graph, cal, setting, actors),
        "trained_original_decoder": lambda actors: GroundedAgent(cfg, graph, cal, replace(setting, mode="source_factor"), actors),
        "random_graph_factor": lambda actors: GroundedAgent(cfg, random, rc, setting, actors),
        "no_write": lambda actors: GroundedAgent(cfg, graph, cal, setting, actors, intervention="no_write"),
        "wrong_context_binding": lambda actors: GroundedAgent(cfg, graph, cal, setting, actors, intervention="wrong_binding"),
        "cut_sensory_recurrence": lambda actors: GroundedAgent(
            cfg, replace(graph, dynamics=np.zeros_like(graph.dynamics)), cal, setting, actors),
        "random_actions": RandomActions,
        "table_once": lambda actors: PublicMemory(len(actors), repeat=False),
        "table_retry": lambda actors: PublicMemory(len(actors), repeat=True),
    }
    report = {"source_checkpoint": str(args.graph), "source_sha256": sha(args.graph), "graph_nodes": graph.dynamics.shape[0],
              "directed_nonzero_edges": int(np.count_nonzero(graph.dynamics)), "setting": asdict(setting),
              "controls": {"factor": "outer product of the separately estimated public cue and context",
                           "calibration": "public onset cue/context recovered at READY from continuous sensory activity",
                           "wrong_binding": "swap context halves of the last key exclusively on the write path",
                           "random": "untrained graph topology, fitted on the same public calibration signals"},
              "sets": {}}
    for split, source in (("original32", tuple(old)), ("new32", EXTERNAL_SEEDS)):
        seeds = source[:args.limit]
        rows = {}
        for name, agent in arms.items():
            rows[name] = [r for start in range(0,len(seeds),8) for r in play(seeds[start:start+8],agent)]
            print(split, name, float(np.mean([r["strict"] for r in rows[name]])),
                  float(np.mean([r["pair_strict_rate"] for r in rows[name]])), flush=True)
        means = {}
        if split == "original32":
            baseline = [r for r in native_report if r["seed"] in seeds]
        else:
            baseline = [r for graph_seed in (3,19,37,53) for start in range(0,len(seeds),8)
                        for r in legacy_batch(cfg,legacy_options,graph_seed,seeds[start:start+8])]
        for seed in seeds:
            matching = [r for r in baseline if r["seed"] == seed]
            assert len(matching) == 4
            means[seed] = {key: float(np.mean([r[key] for r in matching]))
                           for key in rows["trained_factor"][0] if key != "actual_writes"}
        rows["previous_bridge"] = [means[seed] for seed in seeds]
        report["sets"][split] = {"roots": seeds, "rows": rows, "summary": paired_summary(rows)}
        (root / "legacy_grounded-confirmed.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
