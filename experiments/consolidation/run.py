"""Causal experience replay versus compute-matched current-example rehearsal.

New external observations, optimizer steps, teacher, and initialization are paired.
Evaluation reads only model parameters, and never updates memory or the model.
The observed revision is a context-local condition, not inferred change detection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.consolidation.world import World
from experiments.consolidation.memory import Reservoir
from experiments.consolidation.model import Predictor

CONDITIONS = {'A0': (0, 0), 'B0': (1, 0), 'C0': (2, 0), 'A1': (0, 1)}
METHODS = ('online', 'repeat', 'replay')


def seed_for(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f'consolidation-v1/{seed}/{name}'.encode()).digest()[:8], 'little') % (2**63 - 1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(tensors) -> str:
    h = hashlib.sha256()
    for name, t in sorted(tensors.items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def schedule(kind: str, rounds: int = 6, revision_rounds: int = 3) -> list[str]:
    if kind == 'blocked':
        return [k for k in ('A0', 'B0', 'C0') for _ in range(rounds)] + [k for k in ('A1', 'B0', 'C0') for _ in range(revision_rounds)]
    if kind == 'recurrent':
        return ['A0', 'B0', 'C0'] * rounds + ['A1', 'B0', 'C0'] * revision_rounds
    raise ValueError(kind)


@dataclass(frozen=True)
class Config:
    input_dim: int = 32
    classes: int = 4
    width: int = 256
    n_train: int = 16384
    n_val: int = 2048
    n_test: int = 8192
    batch: int = 256
    capacity: int = 8192
    steps_per_block: int = 100
    rounds: int = 6
    revision_rounds: int = 3
    lr: float = .001
    device: str = 'cuda'
    eval_split: str = 'val'
    threads: int = 2


@torch.no_grad()
def evaluate(model: Predictor, world: World, split: str, chunk: int = 2048) -> dict:
    model.eval()
    result = {}
    for domain in (split, 'transfer_' + split):
        raw = world.splits[domain]
        agreement = world.diagnostics(raw)['ab_agree']
        values = {}
        for key, (ctx, version) in CONDITIONS.items():
            correct = []
            losses = []
            for start in range(0, len(raw), chunk):
                x = raw[start:start + chunk]
                y = world.targets(x, ctx, version)
                logits = model(world.features(x, ctx, version))
                correct.append(logits.argmax(1).eq(y))
                losses.append(F.cross_entropy(logits, y, reduction='none'))
            good = torch.cat(correct)
            losses = torch.cat(losses)
            item = {'accuracy': good.float().mean().item(), 'nll': losses.mean().item(), 'n': len(raw)}
            for group, mask in (('ab_agree', agreement), ('ab_disagree', ~agreement)):
                item[group] = {'n': int(mask.sum()), 'accuracy': good[mask].float().mean().item() if mask.any() else None}
            values[key] = item
        result[domain] = values
    model.train()
    return result


def update(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def run_one(config: Config, world: World, seed: int, method: str, order: str, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    if config.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    model = Predictor(config.input_dim + 5, config.classes, config.width, seed_for(seed, 'model')).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    memory = Reservoir(config.capacity, config.input_dim + 5, config.device, seed_for(seed, 'memory'))
    sample_rng = {key: torch.Generator().manual_seed(seed_for(seed, 'stream-' + key)) for key in CONDITIONS}
    external_hash = {key: hashlib.sha256() for key in CONDITIONS}
    initial_hash = tensor_digest(model.state_dict())
    trajectory = []
    initial = evaluate(model, world, config.eval_split)
    start_time = time.perf_counter()
    updates = 0
    new_examples = 0
    replayed_examples = 0
    visits = {key: 0 for key in CONDITIONS}
    last_seen = {key: None for key in CONDITIONS}
    all_schedule = schedule(order, config.rounds, config.revision_rounds)
    before = initial
    for block, key in enumerate(all_schedule):
        ctx, version = CONDITIONS[key]
        visits[key] += 1
        before_age = {k: None if v is None else block * config.steps_per_block - v for k, v in last_seen.items()}
        prequential_good = 0
        first_good = None
        for step in range(config.steps_per_block):
            indices = torch.randint(config.n_train, (config.batch,), generator=sample_rng[key])
            external_hash[key].update(indices.numpy().tobytes())
            x, y = world.batch('train', indices.to(config.device), ctx, version)
            # Read pre-feedback prediction before either parameter update.
            with torch.no_grad():
                good = int(model(x).argmax(1).eq(y).sum())
            prequential_good += good
            if first_good is None:
                first_good = good / config.batch
            # Historical rehearsal excludes this external minibatch.
            old = memory.sample(config.batch) if method == 'replay' and len(memory) else None
            if old is not None and len(old[0]) != config.batch:
                raise RuntimeError('replay must preserve batch size')
            update(model, optimizer, x, y)
            updates += 1
            if method in ('repeat', 'replay'):
                rx, ry = old if old is not None else (x, y)
                update(model, optimizer, rx, ry)
                updates += 1
                if old is not None:
                    replayed_examples += len(rx)
            memory.add(x, y)
            new_examples += len(x)
        last_seen[key] = (block + 1) * config.steps_per_block
        after_age = {k: None if v is None else (block + 1) * config.steps_per_block - v for k, v in last_seen.items()}
        after = evaluate(model, world, config.eval_split)
        item = {'block': block, 'condition': key, 'visit': visits[key],
                'new_examples': new_examples, 'optimizer_steps': updates,
                'prequential_accuracy': prequential_good / (config.steps_per_block * config.batch),
                'first_batch_before_feedback': first_good,
                'before_age_external_steps': before_age, 'after_age_external_steps': after_age,
                'before': before, 'after': after}
        trajectory.append(item)
        before = after
        with (folder / 'trajectory.json').open('w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
        print(json.dumps({'seed': seed, 'method': method, 'order': order, 'block': block,
                          'condition': key, 'prequential': item['prequential_accuracy'],
                          'current': {k: round(v['accuracy'], 4) for k, v in after[config.eval_split].items()}}), flush=True)
    if config.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    memory_state = memory.state_dict()
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'memory': memory_state, 'config': asdict(config), 'seed': seed,
                  'method': method, 'order': order}
    torch.save(checkpoint, folder / 'final.pt')
    result = {'seed': seed, 'method': method, 'order': order, 'config': asdict(config),
              'parameters': sum(p.numel() for p in model.parameters()), 'initial_hash': initial_hash,
              'external_stream_hashes': {k: h.hexdigest() for k, h in external_hash.items()},
              'initial': initial, 'final': after, 'trajectory': trajectory,
              'new_examples': new_examples, 'optimizer_steps': updates,
              'training_examples': updates * config.batch, 'replayed_examples': replayed_examples,
              'buffer_size': len(memory), 'buffer_seen': memory.seen,
              'buffer_bytes': memory.bytes, 'buffer_replacements': memory.replacements,
              'buffer_hash': tensor_digest({k: memory_state[k] for k in ('features', 'targets', 'priorities')}),
              'elapsed_seconds': elapsed,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if config.device.startswith('cuda') else 0,
              'checkpoint_sha256': digest(folder / 'final.pt')}
    (folder / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--orders', default='blocked,recurrent')
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(field.default), default=field.default)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    if config.eval_split not in ('val', 'test'):
        raise ValueError('eval_split must be val or test')
    if min(config.steps_per_block, config.batch, config.rounds, config.revision_rounds, config.capacity) < 1:
        raise ValueError('positive sizes required')
    methods, orders = args.methods.split(','), args.orders.split(',')
    if any(m not in METHODS for m in methods) or any(o not in ('blocked', 'recurrent') for o in orders):
        raise ValueError('unknown method or schedule')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('refusing to overwrite nonempty output')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('world.py'), Path(__file__).with_name('memory.py'), Path(__file__).with_name('model.py')]
    source_dir = args.output / 'source'
    source_dir.mkdir()
    for path in sources:
        (source_dir / path.name).write_bytes(path.read_bytes())
    manifest = {'config': asdict(config), 'seeds': args.seeds, 'methods': methods, 'orders': orders,
                'python': platform.python_version(), 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name() if config.device.startswith('cuda') else None,
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                'selection': 'fixed final checkpoint; pilot validation only; no test selection',
                'limits': ['Supervised environment labels, not self-evaluation or reward-only learning.',
                           'Explicit context and local rule-version cues; no autonomous change detection.',
                           'Uniform random-priority replay; no learned importance or learned replay scheduling.',
                           'Evaluation reads weights only; fixed-size raw buffer used exclusively for training.',
                           'The positive-positive quadrant is a narrow distribution-shift probe.',
                           'Online uses half the optimizer steps; repeat versus replay is the compute-matched comparison.',
                           'All methods maintain the same buffer; only replay reads it for learning.']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = []
    world_hashes = {}
    for seed in map(int, args.seeds.split(',')):
        world = World(seed, input_dim=config.input_dim, classes=config.classes, n_train=config.n_train,
                      n_val=config.n_val, n_test=config.n_test, device=config.device)
        world_hashes[str(seed)] = tensor_digest({'teacher_w': world.teacher_w, 'teacher_delta': world.teacher_delta, **world.splits})
        for order in orders:
            for method in methods:
                folder = args.output / f'{order}-{method}-seed{seed}'
                run_one(config, world, seed, method, order, folder)
                if any(digest(p) != manifest['source_sha256'][str(p.relative_to(ROOT))] for p in sources):
                    raise RuntimeError('source changed during this run; output must not be treated as fixed-source evidence')
                paths.append(str(folder.relative_to(args.output) / 'results.json'))
                (args.output / 'index.json').write_text(json.dumps({'completed': paths}, indent=2), encoding='utf-8')
    (args.output / 'verification.json').write_text(json.dumps({'world_hashes': world_hashes, 'source_unchanged': True}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
