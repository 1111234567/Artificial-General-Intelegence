"""Meta-learn fast synapse addresses for previously unseen, continuously varying foods.

The optimization sees only public odor/context and the signed taste of whichever
side an exploring physical body reached. A prefix fits fast synapses, later
tasted signs score their predictions. No preferred side or latent utility enters
the loss. In independent games a pair of spiking motors acts on these values.
"""
from dataclasses import asdict, dataclass, replace
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import mixture_brain, mixture_world as game, world
from .attractor_search import OUT


@dataclass(frozen=True)
class AddressConfig:
    form: str = "compartment"
    latent: int = 24
    hidden: int = 64


class Addresses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width, latent = cfg.hidden, cfg.latent
        if cfg.form == "dense":
            self.input = nn.Linear(6, width)
        elif cfg.form in ("compartment", "chemical_product"):
            self.input = nn.Linear(4, width)
            self.second = nn.Linear(4, width)
            self.context = nn.Linear(2, width)
        else:
            raise ValueError(cfg.form)
        self.output = nn.Linear(width, latent)
        self.log_ridge = nn.Parameter(torch.tensor(-.6))

    def forward(self, features):
        if self.cfg.form == "dense":
            h = self.input(features).tanh()
        else:
            # Two independently weighted inputs per cell. No list of chemical
            # interactions, context-specific key slots or true value parameters.
            odor, ctx = features[..., :4], features[..., 4:]
            a, b = self.input(odor), self.second(odor)
            q = (a + a * b).tanh()
            h = q * self.context(ctx) if self.cfg.form == "compartment" else (q + self.context(ctx)).tanh()
        k = self.output(h)
        return k / torch.sqrt(.25 + k.square().sum(-1, keepdim=True))

    @property
    def ridge(self):
        return .04 + self.log_ridge.exp()

    def export(self):
        def weights(layer):
            return layer.weight.detach().cpu().numpy().T.copy(), layer.bias.detach().cpu().numpy().copy()
        return {"input": weights(self.input), "output": weights(self.output),
                **({} if self.cfg.form == "dense" else {"second": weights(self.second), "context": weights(self.context)}),
                "ridge": float(self.ridge.detach().cpu()), "form": self.cfg.form,
                "latent": self.cfg.latent}


class Explorer:
    def __init__(self, seeds, trials):
        self.sides = np.stack([np.random.default_rng(np.random.SeedSequence([s, 822197])).integers(2, size=trials)
                               for s in seeds])

    def reset(self, n):
        self.trial = -1

    def act(self, obs):
        if obs[:, world.TRIAL_START].all():
            self.trial += 1
        return np.eye(2, dtype=np.float32)[self.sides[:, self.trial]] * obs[:, world.ACTIVE, None]


def exposures(seed, n, trials, folder):
    path = folder / f"public-exposures-s{seed}-n{n}-l{trials}.npz"
    if path.exists():
        with np.load(path) as old:
            return old["features"], old["signs"]
    if n % 3:
        raise ValueError("Balance the three environment families")
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for family in ("linear", "mixed", "interaction"):
        seeds = rng.integers(10_000_000, 470_000_000, n // 3).tolist()
        for i in range(0, len(seeds), 128):
            ss = seeds[i:i+128]
            menu = game.make(ss, trials, rule=family)
            agent = Explorer(ss, trials)
            # Only the observed outcome and public chemical at actual contact pass upstream.
            result = game.rollout(agent, menu)
            assert (result.contact_side >= 0).all()
            signed = result.observed_reward.sum(2).astype(np.float32)
            assert np.array_equal(signed, result.reward)
            chosen = np.take_along_axis(menu.chemicals, result.contact_side[:, :, None, None], 2)[:, :, 0]
            ctx = np.eye(2, dtype=np.float32)[menu.body.ctx]
            xs.append(np.concatenate([2 * chosen - 1, ctx], 2))
            ys.append(signed)
    x, y = np.concatenate(xs), np.concatenate(ys)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, features=x, signs=y)
    os.replace(tmp, path)
    return x, y


def future_loss(model, x, y, history):
    key = model(x)
    prefix = key[:, :history]
    gram = prefix.transpose(1, 2) @ prefix
    rhs = prefix.transpose(1, 2) @ y[:, :history, None]
    eye = torch.eye(model.cfg.latent, device=key.device)
    coeff = torch.linalg.solve(gram + model.ridge * eye, rhs)
    pred = (key[:, history:] @ coeff).squeeze(-1)
    return (y[:, history:] - pred).square().mean()


def train(folder, data, val, form, seed, updates, device, *, latent=24):
    torch.manual_seed(seed)
    cfg = AddressConfig(form=form, latent=latent)
    model = Addresses(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.0003)
    rng = np.random.default_rng(seed)
    x, y = [torch.as_tensor(v, device=device) for v in data]
    vx, vy = [torch.as_tensor(v, device=device) for v in val]
    best = float("inf")
    start = time.perf_counter()
    for it in range(1, updates+1):
        ix = torch.as_tensor(rng.integers(len(x), size=128), device=device)
        life = x[ix]
        sign = y[ix]
        # Each pass picks an actual prefix. A random contiguous segment also
        # changes which experienced bouts can serve as first encounters.
        offset = int(rng.integers(0,9))
        depth = int(rng.choice([8,16,32,48,64]))
        loss = future_loss(model, life[:, offset:], sign[:, offset:], depth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        opt.step()
        if it == 1 or it % 100 == 0 or it == updates:
            with torch.no_grad():
                scores = [float(future_loss(model, vx[j:j+128], vy[j:j+128], d))
                          for j in range(0, len(vx), 128) for d in (16,32,64)]
            value = float(np.mean(scores))
            path = folder / f"{form}-s{seed}.pt"
            if value < best:
                best = value
                tmp = path.with_suffix(".partial.pt")
                torch.save(dict(config=asdict(cfg), state=model.state_dict(), step=it, validation=value,
                                seed=seed, ridge=float(model.ridge.detach()), used_labels="actually sampled signed taste",
                                train_lives=len(x)), tmp)
                os.replace(tmp, path)
            print(form, seed, it, "Brier(signed)", round(value, 5), "best", round(best, 5),
                  "s", round(time.perf_counter()-start, 1), flush=True)
    return best


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = Addresses(AddressConfig(**data["config"]))
    model.load_state_dict(data["state"])
    return model.export()


class LearnedForager(mixture_brain.MixtureLearner):
    """The two continuous spiking motor cells choose a food without location labels."""
    def __init__(self, params, net, seeds, *, learn=True, freeze=None, mask_context=False):
        self.params = params
        rule = mixture_brain.Rule(representation="spikes", latent=params["latent"], ridge=params["ridge"],
                                  rate=float(learn))
        super().__init__(rule, net, seeds, freeze=freeze, mask_context=mask_context)

    def _features(self, x, ctx):
        p = self.params
        def apply(v, name):
            w, b = p[name]
            return v @ w + b
        if p["form"] == "dense":
            public = np.concatenate([x, np.broadcast_to(ctx[:,None,:], (*x.shape[:2], 2))], 2)
            h = np.tanh(apply(public, "input"))
        else:
            a, b = apply(x, "input"), apply(x, "second")
            q = np.tanh(a + a*b)
            context = apply(ctx, "context")[:,None,:]
            h = q*context if p["form"] == "compartment" else np.tanh(q+context)
        key = apply(h, "output")
        return key / np.sqrt(.25 + (key*key).sum(2, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="mixture-meta-v1")
    ap.add_argument("--forms", nargs="+", default=["dense", "compartment", "chemical_product"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3361, 5383])
    ap.add_argument("--updates", type=int, default=900)
    ap.add_argument("--lives", type=int, default=4608)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    folder = OUT / args.folder
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    data = exposures(6531907, args.lives, 96, folder)
    val = exposures(865143, 768, 96, folder)
    for seed in args.seeds:
        for form in args.forms:
            train(folder, data, val, form, seed, args.updates, args.device)


if __name__ == "__main__":
    main()
