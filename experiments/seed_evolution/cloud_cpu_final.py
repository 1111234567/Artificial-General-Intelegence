"""Continue frozen search checkpoints through the final CE budget on CPU."""
from __future__ import annotations

import torch

from experiments.learned_binding import common, pretrain
from experiments.seed_evolution import audit, core, hybrid_backend, large_runner as runner
from experiments.seed_evolution.lazy_training_streams import LazyTrainingStreams


def _continue_training(seed, genome, origin, steps, provenance):
    start = origin["updates"]
    torch.set_num_threads(1)
    model = core.model_from_genome(seed, genome).train()
    trainable = audit.trainable_parameters(model)
    executor = {name: value.detach().clone() for name, value in model.executor.state_dict().items()}
    encoder_hash = common.parameter_hash(model.encoder)
    genome_hash = core.genome_hash(genome)
    initial = core._training_snapshot(model, audit.make_adam(trainable), seed, 0)
    audit.same_tree(initial["model_config"], origin["model_config"], "search model configuration")
    core._check_tensors(origin["model"], initial["model"])
    audit.audit_frozen_checkpoint(origin, seed, start, genome, executor)

    model.load_state_dict(origin["model"])
    optimizer = audit.make_adam(trainable, origin["optimizer"], start)
    snapshots = {0: initial, start: origin}
    save_steps = {start + 1, steps - 1, steps}
    indices = [core._stream_index(step) for step in range(start, steps)]
    log = []
    with LazyTrainingStreams().install(), hybrid_backend._fp32():
        streams = core._training_streams(seed)
        for update, index in enumerate(indices, start=start + 1):
            obs, actions = streams[index]
            logits, _ = model(obs, model.initial_state(obs.shape[1]))
            loss = pretrain.loss_for(logits, actions)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite CE at update {update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable.values(), 5., error_if_nonfinite=True))
            optimizer.step()
            if update in save_steps:
                snapshots[update] = core._training_snapshot(model, optimizer, seed, update)
            if update % 64 == 0 or update in save_steps:
                finish = actions >= 3
                finish_accuracy = (float((logits.argmax(-1)[finish] == actions[finish]).float().mean())
                                   if finish.any() else None)
                log.append(dict(update=update, stream=index, loss=float(loss.detach()),
                                grad_norm=grad_norm, finish_accuracy=finish_accuracy))

    audit.audit_frozen_checkpoint(snapshots[steps], seed, steps, genome, executor)
    if (core.genome_hash(genome) != genome_hash
            or any(parameter.grad is not None for parameter in model.parameters()
                   if not parameter.requires_grad)):
        raise ValueError("CPU continuation changed the initialization or reached a frozen parameter")
    metadata = dict(source_seed=seed, steps=steps, device="cpu", genome_hash=genome_hash,
                    executor_hash=common.parameter_hash(model.executor), encoder_hash=encoder_hash,
                    stream_indices=indices, initial_state=dict(h=0, z=0, memory=0),
                    reset_every_stream=True, optimizer="Adam", fresh_optimizer=False,
                    learning_rate=common.SETTINGS["ce_lr"], wait_weight=common.SETTINGS["wait_weight"],
                    grad_clip_norm=5., trainable_parameters=sum(p.numel() for p in trainable.values()),
                    encoder_unchanged=True, executor_unchanged=True, tf32=False, amp=False,
                    dtype="float32", torch_version=str(torch.__version__), **provenance)
    return dict(checkpoints=snapshots, log=log, metadata=metadata)


def train_final(winner):
    """Freeze a finalist rooted in the committed model and Adam state at step 64."""
    seed, cid = winner["rep"], winner["id"]
    start, steps = runner.CONFIG["inner_steps"], runner.CONFIG["final_steps"]
    candidate = runner._candidate_dir(seed, cid)
    path = runner._final_dir(winner)
    with runner._lock(path / ".final.lock"):
        record = runner.base.read(candidate / "candidate.json")
        if (record["rep"], record["id"]) != (seed, cid):
            raise ValueError(f"winner differs from its committed candidate: {candidate}")
        artifacts = record["artifacts"]
        for name in ("initial.pt", "inner64.pt", "training.json"):
            runner._verify_hash(candidate / name, artifacts[name])
        source = runner.base.read(candidate / "training.json")
        metadata64 = source["metadata"]
        if (source["checkpoint_sha256"] != artifacts["inner64.pt"]
                or (metadata64["source_seed"], metadata64["steps"], metadata64["genome_hash"])
                != (seed, start, record["genome_hash"])):
            raise ValueError(f"search training differs from its committed candidate: {candidate}")
        provenance = dict(origin64_sha256=artifacts["inner64.pt"],
                          origin64_training_sha256=artifacts["training.json"],
                          origin64_device=metadata64["device"],
                          origin64_execution=metadata64.get("execution"),
                          continuation_start_update=start + 1, continuation_updates=steps - start)

        checkpoint_path, metadata_path = path / "training.pt", path / "training.json"
        metadata = runner._recover_pair(checkpoint_path, metadata_path, "sha256")
        if metadata is None:
            genome = runner._load(candidate / "initial.pt")
            origin = runner._load(candidate / "inner64.pt")
            if (origin["updates"] != start or core.genome_hash(genome) != record["genome_hash"]
                    or runner.base.model_hash(origin["model"]) != record["trained_hash"]):
                raise ValueError(f"winner state differs from its committed candidate: {candidate}")
            trained = _continue_training(seed, genome, origin, steps, provenance)
            metadata = runner._commit_pair(checkpoint_path, metadata_path, trained,
                                           dict(winner=winner, metadata=trained["metadata"],
                                                log=trained["log"]), "sha256")
        common.assert_tree_equal(metadata["winner"], winner, "trained winner")
        stored = metadata["metadata"]
        common.assert_tree_equal({name: stored.get(name) for name in provenance},
                                 provenance, "final continuation provenance")
        if stored["device"] != "cpu" or stored["fresh_optimizer"] is not False:
            raise ValueError(f"unexpected final training mode: {path}")
        return dict(winner=winner, path=checkpoint_path.relative_to(runner.ROOT).as_posix(),
                    sha256=metadata["sha256"])
