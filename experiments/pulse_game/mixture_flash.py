"""A physical taste task with independently timed single-tick odor and habitat pulses."""
from dataclasses import dataclass
import os

import numpy as np

from . import mixture_world as game, world
from .mixture_metric import Explorer


def timing(seeds, trials):
    odor = np.empty((len(seeds), trials), np.int8)
    ctx = np.empty_like(odor)
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 195319]))
        order = rng.random(trials) < .5
        early = rng.integers(0, 2, size=trials)
        late = rng.integers(2, 4, size=trials)
        odor[j] = np.where(order, early, late)
        ctx[j] = np.where(order, late, early)
    return odor, ctx


class FlashBody(game.Body):
    def __init__(self, menu, **kwargs):
        self.odor_tick, self.context_tick = timing(menu.body.seeds, menu.body.trials)
        super().__init__(menu, **kwargs)

    def observe(self):
        obs = super().observe()
        j, t = self.trial_index, self.tick_index
        obs[self.odor_tick[:, j] != t, game.LEFT.start:game.RIGHT.stop] = .5
        obs[self.context_tick[:, j] != t, game.CTX] = 0.
        return obs


def rollout(agent, menu, **kwargs):
    sim = FlashBody(menu, **kwargs)
    agent.reset(menu.body.n)
    obs = sim.observe()
    while obs is not None:
        obs = sim.step(agent.act(obs))
    return sim.result()


def public_examples(folder, seed, lives, trials):
    """Roll real bodies; extract pulse contents, eventual actual contact time and signed taste."""
    path = folder/f"flashes-public-{seed}-{lives}-{trials}.npz"
    if path.exists():
        with np.load(path) as raw:
            return {k:raw[k] for k in raw.files}
    if lives % 3:
        raise ValueError("Use balanced private taste families")
    rng = np.random.default_rng(seed)
    bank = {k:[] for k in ("odor","ctx","ot","ct","side","touch","reward")}
    for family in ("linear","interaction","mixed"):
        seeds = rng.integers(10_000_000,470_000_000,lives//3).tolist()
        for start in range(0,len(seeds),96):
            ss = seeds[start:start+96]
            menu = game.make(ss,trials,rule=family)
            explorer = Explorer(ss,trials)
            res = rollout(explorer,menu,record=True)
            if not (res.contact_side>=0).all():
                raise ValueError("An exploration bout had no contact")
            visual = res.obs[:,:,:,game.LEFT.start:game.RIGHT.stop].reshape(len(ss),trials,world.TICKS,2,4)*2.-1.
            ctx = res.obs[:,:,:,game.CTX]
            odor_on = np.any(visual!=0,axis=(3,4))
            ctx_on = ctx.any(3)
            if not np.all((odor_on.sum(2)==1)&(ctx_on.sum(2)==1)):
                raise AssertionError("Pulse protocol changed")
            signs=res.observed_reward.sum(2)
            if not np.array_equal(signs,res.reward):raise AssertionError("Public sign mismatch")
            row=dict(odor=visual.sum(2),ctx=ctx.sum(2),ot=odor_on.argmax(2).astype(np.int8),
                     ct=ctx_on.argmax(2).astype(np.int8),side=res.contact_side,touch=res.contact_tick,reward=signs)
            for k,v in row.items():bank[k].append(v)
    all_data={k:np.concatenate(items) for k,items in bank.items()}
    tmp=path.with_suffix(".partial.npz")
    np.savez_compressed(tmp,**all_data)
    os.replace(tmp,path)
    return all_data
