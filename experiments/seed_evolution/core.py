"""Evolve the initialization of a frozen-trace policy and train its memory heads."""
from __future__ import annotations

from functools import cache
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from experiments.learned_binding import audit, common, data, interaction, pretrain
from experiments.learned_binding.frozen_trace import FrozenTracePolicy, _check_trainability
from experiments.learned_binding.model import ENCODER_WIDTH


ROOT = Path(__file__).resolve().parents[2]
DATA_FREEZE = ROOT / "data/learned-binding/v1/data-freeze.json"
ENCODER_INIT_SD = 1 / math.sqrt(3 * ENCODER_WIDTH)
HEAD_INIT_SD = .02
ENCODER_WEIGHTS = ("encoder.weight_ih_l0", "encoder.weight_hh_l0")
ENCODER_BIASES = ("encoder.bias_ih_l0", "encoder.bias_hh_l0")
HEAD_WEIGHTS = tuple(f"{name}.weight" for name in ("query", "key", "value", "gate"))


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _source_seed(value):
    seed = _integer("source_seed", value)
    if seed not in common.SETTINGS["seeds"]:
        raise ValueError("source_seed must select old executor 0, 1 or 2")
    return seed


def _new_policy(source_seed, rng_seed):
    # Construct the historical executor first. Only the new policy's CPU
    # initializer should consume draws from the genome seed.
    with torch.random.fork_rng(devices=[]):
        executor = common.executor(_source_seed(source_seed))
        torch.random.default_generator.manual_seed(_integer("rng_seed", rng_seed))
        return FrozenTracePolicy(executor=executor,
                                 original_mappings=common.mapping_table(source_seed, "related"))


def _copy_tensor(value):
    return value.detach().cpu().clone()


def new_genome(source_seed: int, rng_seed: int) -> dict[str, torch.Tensor]:
    """Draw fresh policy coordinates while keeping the old executor and mapping fixed."""
    model = _new_policy(source_seed, rng_seed)
    return {name: _copy_tensor(value) for name, value in model.state_dict().items()
            if not name.startswith("executor.")}


def genome_hash(genome):
    return audit.parameter_hash(dict(sorted(genome.items())))


def _check_tensors(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("incompatible policy state keys")
    for name, value in actual.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype):
            raise ValueError(f"incompatible policy tensor {name}")


def model_from_genome(source_seed, genome) -> FrozenTracePolicy:
    """Load policy initialization into a fresh CPU model with the fixed executor."""
    model = _new_policy(source_seed, 0)
    state = model.state_dict()
    _check_tensors(genome, {key: value for key, value in state.items()
                            if not key.startswith("executor.")})
    if not torch.equal(genome["original_mappings"].cpu(), state["original_mappings"]):
        raise ValueError("genome original_mappings does not match source executor")
    model.load_state_dict({**state, **genome}, strict=True)
    _check_trainability(model)
    return model


def mutate_genome(parent, mutation_seed, sigma=.15) -> dict[str, torch.Tensor]:
    """Add independently seeded Gaussian noise to the original random coordinates."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    child = {name: _copy_tensor(value) for name, value in parent.items()}
    generator = torch.Generator(device="cpu").manual_seed(_integer("mutation_seed", mutation_seed))
    for names, sd in ((ENCODER_WEIGHTS + ENCODER_BIASES, ENCODER_INIT_SD),
                      (HEAD_WEIGHTS, HEAD_INIT_SD)):
        for name in names:
            value = child[name]
            noise = torch.randn(value.shape, generator=generator, dtype=value.dtype)
            if name in ENCODER_BIASES:
                noise[ENCODER_WIDTH:2 * ENCODER_WIDTH] = 0
            value.add_(noise, alpha=sigma * sd)
    return child


@cache
def _training_streams(source_seed):
    """Verify the v1 files once per process and retain only CE input tensors."""
    manifest = json.loads(DATA_FREEZE.read_text(encoding="utf-8"))
    rows = sorted((row for row in manifest["streams"] if row["seed"] == source_seed),
                  key=lambda row: row["stream"])
    if [row["stream"] for row in rows] != list(range(common.SETTINGS["training_streams"])):
        raise ValueError(f"incomplete or duplicate v1 training streams for seed {source_seed}")
    streams = []
    ticks, batch = common.SETTINGS["train_ticks"], common.SETTINGS["train_batch"]
    for row in rows:
        path = ROOT / Path(row["path"].replace("\\", "/"))
        if common.sha(path) != row["sha256"]:
            raise ValueError(f"frozen v1 training stream hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obs, actions = payload["obs"], payload["actions"]
        if (obs.shape != (ticks, batch, 56) or obs.dtype != torch.float32
                or actions.shape != (ticks, batch, 2) or actions.dtype != torch.long):
            raise ValueError(f"incompatible v1 training tensors: {path}")
        streams.append((obs, actions))
    return tuple(streams)


def _stream_index(step):
    dense = common.SETTINGS["dense_updates"]
    return step % 8 if step < dense else (step - dense) % 16


def _training_snapshot(model, optimizer, source_seed, step):
    optimizer_state = optimizer.state_dict()
    optimizer_state["state"] = {
        index: {name: _copy_tensor(value) if isinstance(value, torch.Tensor) else value
                for name, value in state.items()}
        for index, state in optimizer_state["state"].items()
    }
    return dict(model={name: _copy_tensor(value) for name, value in model.state_dict().items()},
                model_config=model.config(), optimizer=optimizer_state, updates=step,
                source_seed=source_seed)


def inner_train(source_seed, genome, steps=64, checkpoints=(0, 64), device="cpu"):
    """Fit the original 2,258 trainable scalars on the frozen v1 stream schedule."""
    source_seed = _source_seed(source_seed)
    steps = _integer("steps", steps)
    checkpoints = {_integer("checkpoint", step) for step in checkpoints}
    if steps > common.SETTINGS["ce_updates"] or any(step > steps for step in checkpoints):
        raise ValueError("checkpoints must lie within the requested v1 training budget")
    torch.set_num_threads(1)
    model = model_from_genome(source_seed, genome).to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=common.SETTINGS["ce_lr"])
    initial_executor = common.parameter_hash(model.executor)
    initial_encoder = common.parameter_hash(model.encoder)
    initial_genome = genome_hash(genome)
    snapshots = {}
    if 0 in checkpoints:
        snapshots[0] = _training_snapshot(model, optimizer, source_seed, 0)
    streams = _training_streams(source_seed) if steps else ()
    indices = [_stream_index(step) for step in range(steps)]
    log = []
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for update, index in enumerate(indices, start=1):
            obs, actions = (value.to(device) for value in streams[index])
            logits, _ = model(obs, model.initial_state(obs.shape[1]))
            loss = pretrain.loss_for(logits, actions)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite CE at update {update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True))
            optimizer.step()
            if update in checkpoints:
                snapshots[update] = _training_snapshot(model, optimizer, source_seed, update)
            if update == 1 or update == steps or update % 64 == 0 or update in checkpoints:
                finish = actions >= 3
                finish_accuracy = (float((logits.argmax(-1)[finish] == actions[finish]).float().mean())
                                   if finish.any() else None)
                log.append(dict(update=update, stream=index, loss=float(loss.detach()),
                                grad_norm=grad_norm, finish_accuracy=finish_accuracy))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    if (initial_executor != common.parameter_hash(model.executor)
            or initial_encoder != common.parameter_hash(model.encoder)
            or initial_genome != genome_hash(genome)
            or any(parameter.grad is not None for name, parameter in model.named_parameters()
                   if name.startswith(("encoder.", "executor.")))):
        raise AssertionError("inner training changed an initialization or a frozen component")
    metadata = dict(source_seed=source_seed, steps=steps, device=str(torch.device(device)),
                    genome_hash=initial_genome, executor_hash=initial_executor,
                    encoder_hash=initial_encoder, stream_indices=indices,
                    initial_state=dict(h=0, z=0, memory=0), reset_every_stream=True,
                    optimizer="Adam", fresh_optimizer=True, learning_rate=common.SETTINGS["ce_lr"],
                    wait_weight=common.SETTINGS["wait_weight"], grad_clip_norm=5.,
                    trainable_parameters=sum(parameter.numel() for parameter in trainable),
                    encoder_unchanged=True, executor_unchanged=True, tf32=False)
    return dict(checkpoints=snapshots, log=log, metadata=metadata)


def _model_for_score(source_seed, model_state):
    model = _new_policy(source_seed, 0)
    reference = model.state_dict()
    _check_tensors(model_state, reference)
    for name, value in reference.items():
        if name.startswith("executor.") or name == "original_mappings":
            if not torch.equal(value, model_state[name].cpu()):
                raise ValueError(f"scoring state changed fixed source tensor {name}")
    model.load_state_dict(model_state, strict=True)
    return model.eval()


def _sampling_seed(source_seed, stream_id, random_seed):
    domain = np.random.SeedSequence([0x53454544, source_seed, random_seed, stream_id])
    return int(domain.generate_state(1, dtype=np.uint64)[0])


def _new_subject(model, source_seed, stream_id, actors, random_seed):
    subject = interaction.Subject.__new__(interaction.Subject)
    subject.seed, subject.kind, subject.stream_id = source_seed, "learned", stream_id
    spec = common.SEEDS[source_seed]
    subject.contexts = (spec["history_cue"], spec["B_cue"])
    subject.world = data.make_world(
        source_seed, batch=actors, duration_range=tuple(common.SETTINGS["support_duration"]),
        gap_range=tuple(common.SETTINGS["support_gap"]), stream_id=stream_id,
        random_seed=random_seed, split="eval")
    subject.model, subject.executor = model, model.executor
    subject.original = model.original_mappings.detach().clone()
    subject.state = model.initial_state(actors)
    subject.binder = None
    subject.rng = torch.Generator(device="cpu").manual_seed(_sampling_seed(source_seed, stream_id, random_seed))
    subject.pending_obs = None
    subject.work = dict(ticks=0, model_steps=0, sampled_action_rows=0, teacher_action_rows=0,
                        public_success_finishes=0, manual_public_examples=0)
    return subject


def _conflict_fitness(groups):
    rates = []
    for name in ("A_conflict", "B_conflict"):
        row = groups[name]
        if row["n"] <= 0:
            raise ValueError(f"zero matured eligible requests for {name}")
        rates.append(row["strict"] / row["n"])
    return sum(rates) / len(rates)


def _boundary(snapshot):
    world = snapshot["world"]
    return dict(tick=world["tick"], after_act=world["acted"], event_hash=world["event_hash"],
                parameter_hash=snapshot["parameter_hash"], rng_hash=common.tensor_hash(snapshot["rng"]),
                state_hashes={name: common.tensor_hash(value) for name, value in snapshot["state"].items()},
                work=snapshot["work"])


def _support_summary(block, teacher):
    start, final = block["start"], block["final"]
    start_counts, final_counts = start["world"]["counts"], final["world"]["counts"]
    return dict(start=_boundary(start), final=_boundary(final), ticks=len(block["actions"]),
                policy="teacher_midpoint" if teacher else "sample_all_11_actions",
                exogenous_hash=audit._exogenous_hash(block["obs"].numpy()),
                actions_hash=common.tensor_hash(block["actions"]),
                accepted_finishes=int((final_counts["completed_all"] - start_counts["completed_all"]).sum()),
                rejected=int((final_counts["rejected"] - start_counts["rejected"]).sum()),
                gate_mean=float(block["gate"].mean()), effective_gate_mean=float(block["effective_gate"].mean()))


def _probe_summary(probe):
    report = probe["report"]
    summary = {name: report[name] for name in ("idle_opportunities", "idle_actions", "command_translations",
               "correct_translations", "memory_norm", "gate_statistics", "effective_gate_sum", "first_groups")}
    summary.update(start=_boundary(probe["start"]), final=_boundary(probe["final"]), policy="argmax",
                   exogenous_hash=audit._exogenous_hash(probe["arrays"]["obs"]),
                   actions_hash=common.tensor_hash(torch.from_numpy(probe["arrays"]["actions"])),
                   right_censored=len(report["right_censored"]), inherited=len(report["left_boundary_matured"]))
    for name, numerator, denominator in (("idle_action_rate", "idle_actions", "idle_opportunities"),
                                         ("command_translation_accuracy", "correct_translations", "command_translations")):
        summary[name] = report[numerator] / report[denominator] if report[denominator] else None
    return summary


def _score_stream(model, source_seed, stream_id, actors, random_seed, support_ticks, probe_ticks,
                  read_enabled, write_enabled, teacher_support, b_ticks):
    subject = _new_subject(model, source_seed, stream_id, actors, random_seed)
    if not teacher_support:
        subject.world.oracle = audit._forbid_oracle
    support = subject.support(support_ticks, teacher=teacher_support, sample=not teacher_support,
                              read_enabled=read_enabled, write_enabled=write_enabled)
    subject.world.oracle = audit._forbid_oracle
    archive = dict(support=support)
    b_summary = None
    if b_ticks:
        subject.world.set_family_policy((subject.contexts[1],))
        archive["b_support"] = subject.support(b_ticks, teacher=False, sample=True,
                                               read_enabled=read_enabled, write_enabled=write_enabled)
        b_summary = _support_summary(archive["b_support"], False)
        subject.world.set_family_policy(subject.contexts)
    probe = interaction.probe(subject, probe_ticks, read_enabled=read_enabled, write_enabled=write_enabled)
    archive["probe"] = probe
    if support["start"]["parameter_hash"] != probe["final"]["parameter_hash"]:
        raise AssertionError("evaluation changed controller parameters")
    groups = probe["report"]["groups"]
    support_summary, probe_summary = _support_summary(support, teacher_support), _probe_summary(probe)
    start_world = support["start"]["world"]
    stream = dict(stream_id=stream_id, fitness=_conflict_fitness(groups), groups=groups,
                  contexts=list(subject.contexts), actor_ids=list(subject.world.actor_ids),
                  private_mapping_hash=common.tensor_hash(start_world["random_mapping"]["mappings"]),
                  sampling_seed=_sampling_seed(source_seed, stream_id, random_seed),
                  world_start_hash=start_world["event_hash"], world_support_end_hash=support["final"]["world"]["event_hash"],
                  world_end_hash=probe["final"]["world"]["event_hash"],
                  support_exogenous_hash=support_summary["exogenous_hash"],
                  probe_exogenous_hash=probe_summary["exogenous_hash"],
                  b_support_exogenous_hash=None if b_summary is None else b_summary["exogenous_hash"],
                  support=support_summary, b_support=b_summary, probe=probe_summary)
    return stream, archive


def score_model(source_seed, model_state, stream_ids=(1100, 1101), support_ticks=2048,
                probe_ticks=2048, actors=16, random_seed=78111, read_enabled=True,
                write_enabled=True, *, archive=False, teacher_support=False, b_ticks=0):
    """Score strict matured conflict requests after actual continuous experience."""
    source_seed = _source_seed(source_seed)
    stream_ids = tuple(_integer("stream_id", index) for index in stream_ids)
    if not stream_ids or len(set(stream_ids)) != len(stream_ids):
        raise ValueError("stream_ids must be nonempty and distinct")
    support_ticks = _integer("support_ticks", support_ticks, 1)
    probe_ticks = _integer("probe_ticks", probe_ticks, 1)
    actors = _integer("actors", actors, 1)
    random_seed = _integer("random_seed", random_seed)
    b_ticks = _integer("b_ticks", b_ticks)
    if any(type(flag) is not bool for flag in (read_enabled, write_enabled, archive, teacher_support)):
        raise TypeError("read_enabled, write_enabled, archive and teacher_support must be booleans")
    torch.set_num_threads(1)
    model = _model_for_score(source_seed, model_state)
    parameter_hash = common.parameter_hash(model)
    streams, archives = [], {}
    for stream_id in stream_ids:
        stream, trace = _score_stream(model, source_seed, stream_id, actors, random_seed, support_ticks,
                                      probe_ticks, read_enabled, write_enabled, teacher_support, b_ticks)
        streams.append(stream)
        if archive:
            archives[stream_id] = trace
        del trace
    result = dict(source_seed=source_seed, parameter_hash=parameter_hash,
                  fitness=sum(stream["fitness"] for stream in streams) / len(streams), streams=streams,
                  support_ticks=support_ticks, probe_ticks=probe_ticks, actors=actors, random_seed=random_seed,
                  read_enabled=read_enabled, write_enabled=write_enabled, teacher_support=teacher_support, b_ticks=b_ticks)
    if archive:
        result["archives"] = archives
    return result
