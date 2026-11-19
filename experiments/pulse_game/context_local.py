"""A small context-addressed rule memory with the frozen sensor and actuator."""

from __future__ import annotations

import numpy as np

from . import habitat_deploy, mixture_world as food, world


class Addresses:
    """Allocate cosine-nearest addresses only from physically presented pulses."""

    def __init__(self, lives: int, slots: int, radius: float) -> None:
        if slots < 1 or not 0 <= radius <= np.pi:
            raise ValueError("invalid angular address capacity or radius")
        self.minimum_similarity = np.cos(radius)
        self.prototypes = np.zeros((lives, slots, 2), dtype=np.float32)
        self.counts = np.zeros((lives, slots), dtype=np.int32)
        self.current = np.full(lives, -1, dtype=np.int32)
        self.overflows = np.zeros(lives, dtype=np.int32)
        self.basis = np.eye(slots, dtype=np.float32)

    def register(self, rows: np.ndarray, pulse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pulse = pulse / np.maximum(np.linalg.norm(pulse, axis=1, keepdims=True), 1e-8)
        proto = self.prototypes[rows]
        exists = self.counts[rows] > 0
        similarity = np.where(exists, (proto * pulse[:, None]).sum(axis=2), -np.inf)
        winner = similarity.argmax(axis=1)
        new = similarity.max(axis=1) < self.minimum_similarity
        free = (~exists).any(axis=1)
        allocated = new & free
        winner[allocated] = (~exists[allocated]).argmax(axis=1)
        if self.basis.shape[0] > 1:
            self.overflows[rows[new & ~free]] += 1
        before = self.counts[rows, winner]
        updated = self.prototypes[rows, winner] * before[:, None] + pulse
        self.prototypes[rows, winner] = updated / np.maximum(np.linalg.norm(updated, axis=1, keepdims=True), 1e-8)
        self.counts[rows, winner] = before + 1
        self.current[rows] = winner
        return rows[allocated], winner[allocated]

    def code(self) -> np.ndarray:
        return self.basis[np.maximum(self.current, 0)] * (self.current[:, None] >= 0)


class LocalForager(habitat_deploy.Forager):
    """Replace only the rule address and scope of retention.

    The original odor front, signed food-event bank, actual contact update
    rule, actuator, and body remain in use. New addresses can inherit the
    first discovered address as a default. No habitat labels enter the actor.
    """

    def __init__(
        self, perception, route, seeds, *, motor, slots: int = 3, radius: float = 0.09,
        inherit: bool = True, local_retention: bool = True, record_rules: bool = False,
    ) -> None:
        chosen = {**route, "slots": slots}
        # This reference uses exactly the fitted rule's ordinary update.
        # Extra cross-address covariance manipulations would change its meaning.
        if "release" in chosen or "magnitude_floor" in chosen:
            raise ValueError("use a frozen unmodified RLS or delta rule")
        super().__init__(perception, chosen, seeds, start=-1, motor=motor)
        self.slots = slots
        self.radius = radius
        self.inherit = inherit
        self.local_retention = local_retention
        self.record_rules = record_rules

    def reset(self, n):
        super().reset(n)
        self.addresses = Addresses(n, self.slots, self.radius)
        self._present = np.empty(0, dtype=np.intp)
        self.history = []
        self.default_records = []

    def address(self):
        if self._present.size:
            born, slot = self.addresses.register(self._present, self.context[self._present])
            if self.inherit:
                self.g[born, slot] = self.g[born, 0]
            if self.record_rules:
                self.history.append((self.round, self.addresses.current.copy(), self.g.copy()))
        return self.addresses.code()

    def act(self, obs):
        self._present = np.flatnonzero(obs[:, food.CTX].any(axis=1))
        if self.record_rules and obs[:, world.TRIAL_START].all():
            self.default_records.append(self.g[:, 0].copy())
        receive = obs[:, world.REWARD_POS] != obs[:, world.REWARD_NEG]
        received = np.flatnonzero(receive & obs[:, world.CONTACT_ID].any(axis=1))
        if received.size and self.local_retention:
            prior = self.g[received].copy()
            untouched = (self.addresses.counts[received] > 0) & (self.code[received] == 0)
        pulse = super().act(obs)
        if received.size and self.local_retention:
            self.g[received] = np.where(untouched, prior, self.g[received])
        return pulse


class BoundNativeForager(habitat_deploy.Forager):
    """Keep the original learned code and bank its complete plastic state."""

    def __init__(self, perception, route, seeds, *, motor, slots: int = 3, radius: float = 0.09):
        super().__init__(perception, route, seeds, start=-1, motor=motor)
        self.slots = slots
        self.radius = radius

    def reset(self, n):
        super().reset(n)
        self.addresses = Addresses(n, self.slots, self.radius)
        self.rule_bank = np.repeat(self.g[:, None], self.slots, axis=1)
        self.covariance_bank = None if self.cov is None else np.repeat(self.cov[:, None], self.slots, axis=1)
        self._present = np.empty(0, dtype=np.intp)

    def address(self):
        rows = self._present
        if rows.size:
            previous = self.addresses.current[rows]
            known = previous >= 0
            previous_rows = rows[known]
            old = previous[known]
            self.rule_bank[previous_rows, old] = self.g[previous_rows]
            if self.covariance_bank is not None:
                self.covariance_bank[previous_rows, old] = self.cov[previous_rows]
            born, slot = self.addresses.register(rows, self.context[rows])
            self.rule_bank[born, slot] = self.rule_bank[born, 0]
            current = self.addresses.current[rows]
            self.g[rows] = self.rule_bank[rows, current]
            if self.covariance_bank is not None:
                self.cov[rows] = self.covariance_bank[rows, current]
        return super().address()

    def act(self, obs):
        self._present = np.flatnonzero(obs[:, food.CTX].any(axis=1))
        return super().act(obs)
