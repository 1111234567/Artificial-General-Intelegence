"""Room values and an acquired directed transition memory for two-step control."""

from __future__ import annotations

import numpy as np

from experiments.pulse_game.twostep_world import (
    AVAILABLE, ENTRANCE_RECEIPT, ENTRANCE_STAGE, ROOM_ID, ROOM_STAGE, STAGE_START, VISIT, TwoStepTape,
)
from experiments.pulse_game.world import ACTIVE, CONTACT_ID, LEFT_ID, REWARD_NEG, REWARD_POS, TRIAL_START


_ONE_HOT = np.eye(2, dtype=np.float32)


class RouteMemory:
    """Local fast writes keyed by public room and actual endpoint receipts."""

    def __init__(self, worlds: int, *, write_routes: bool = True, write_visit_values: bool = True, mean_visits: bool = False) -> None:
        self.rows = np.arange(worlds)
        self.transition_counts = np.zeros((worlds, 2, 2), dtype=np.float32)
        self.value = np.zeros((worlds, 2, 2), dtype=np.float32)
        self.fruit_counts = np.zeros((worlds, 2, 2), dtype=np.int32)
        self.visit_totals = np.zeros((worlds, 2, 2), dtype=np.float32)
        self.visit_counts = np.zeros((worlds, 2, 2), dtype=np.int32)
        self.cached = np.zeros((worlds, 2), dtype=np.float32)
        self.cached_counts = np.zeros((worlds, 2), dtype=np.int32)
        self.entry = np.full(worlds, -1, dtype=np.int8)
        self.room = np.full(worlds, -1, dtype=np.int8)
        self.write_routes = write_routes
        self.write_visit_values = write_visit_values
        self.mean_visits = mean_visits

    def observe(self, obs: np.ndarray) -> None:
        start = obs[:, TRIAL_START] > 0
        self.entry[start] = -1
        self.room[start] = -1

        seen_entry = np.flatnonzero(obs[:, ENTRANCE_RECEIPT].sum(axis=1) == 1)
        self.entry[seen_entry] = obs[seen_entry, ENTRANCE_RECEIPT].argmax(axis=1)

        seen_room = np.flatnonzero((obs[:, ROOM_ID].sum(axis=1) == 1) & (obs[:, STAGE_START] > 0))
        self.room[seen_room] = obs[seen_room, ROOM_ID].argmax(axis=1)
        if self.write_routes:
            linked = seen_room[self.entry[seen_room] >= 0]
            self.transition_counts[linked, self.entry[linked], self.room[linked]] += 1

        signed = obs[:, REWARD_POS] - obs[:, REWARD_NEG]
        terminal = (obs[:, CONTACT_ID].sum(axis=1) == 1) & (signed != 0)
        address = terminal & (self.room >= 0)
        if not self.write_visit_values:
            address &= obs[:, VISIT] == 0
        bound = np.flatnonzero(address)
        fruit = obs[bound, CONTACT_ID].argmax(axis=1)
        key = bound, self.room[bound], fruit
        self.value[key] = 0.2 * self.value[key] + 0.8 * signed[bound]
        self.fruit_counts[key] += 1
        if self.mean_visits:
            in_visit = obs[bound, VISIT] > 0
            sampled = bound[in_visit]
            visit_key = sampled, self.room[sampled], fruit[in_visit]
            self.visit_totals[visit_key] += signed[sampled]
            self.visit_counts[visit_key] += 1
            self.value[visit_key] = self.visit_totals[visit_key] / self.visit_counts[visit_key]

        with_entry = np.flatnonzero(terminal & (self.entry >= 0) & (obs[:, VISIT] == 0))
        key = with_entry, self.entry[with_entry]
        self.cached_counts[key] += 1
        current = self.cached[key]
        self.cached[key] = current + (signed[with_entry] - current) / self.cached_counts[key]

    def probabilities(self) -> np.ndarray:
        """Pool evidence for the two possible symmetric entrance permutations."""
        counts = self.transition_counts
        aligned = counts[:, 0, 0] + counts[:, 1, 1]
        total = counts.sum(axis=(1, 2))
        p = (aligned + 0.5) / (total + 1)
        return np.stack((p, 1 - p, 1 - p, p), axis=1).reshape(-1, 2, 2)

    def neural_features(self, obs: np.ndarray) -> np.ndarray:
        self.observe(obs)
        graph = self.probabilities().reshape(-1, 4) - 0.5
        values = self.value.reshape(-1, 4)
        count = (self.fruit_counts.clip(0, 3) / 3).astype(np.float32).reshape(-1, 4)
        visits = (self.transition_counts.sum(axis=2).clip(0, 6) / 6).astype(np.float32)
        return np.column_stack((obs, graph, values, count, visits))


class DirectedRoutePolicy:
    """Reference solver using fast associations and a continuous planning loop.

    Fruit populations drive room populations. The transition weights pass
    room activation to entrances and weakly circulate entrance activation
    back to rooms. Lateral inhibition separates competing entrances. The
    supplied directional motor primitive drives the currently selected
    object. The cached control changes only the post-visit entrance readout.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        cached_probe: bool = False,
        write_routes: bool = True,
        write_visit_values: bool = True,
        mean_visits: bool = False,
        global_probe_fruit: bool = False,
    ) -> None:
        self.seed = seed
        self.cached_probe = cached_probe
        self.write_routes = write_routes
        self.write_visit_values = write_visit_values
        self.mean_visits = mean_visits
        self.global_probe_fruit = global_probe_fruit

    def reset(self, n: int) -> None:
        self.memory = RouteMemory(
            n, write_routes=self.write_routes, write_visit_values=self.write_visit_values, mean_visits=self.mean_visits,
        )
        self._rng = np.random.default_rng(self.seed)
        self._entered_visit = False
        self._tie = np.zeros(n, dtype=np.int8)
        self._target = np.zeros(n, dtype=np.int8)
        self._fruit = np.zeros((n, 2, 2), dtype=np.float32)
        self._room = np.zeros((n, 2), dtype=np.float32)
        self._route = np.zeros((n, 2), dtype=np.float32)
        self._motor = np.zeros((n, 2), dtype=np.float32)

    def _tick_graph(self) -> None:
        p = self.memory.probabilities()
        fruit_current = self.memory.value + 0.12 * self._room[:, :, None]
        self._fruit = 0.36 * self._fruit + 0.64 * np.tanh(fruit_current)
        feedback = (p * self._route[:, :, None]).sum(axis=1)
        room_current = self._fruit.max(axis=2) + 0.12 * feedback
        self._room = 0.46 * self._room + 0.54 * np.tanh(room_current)
        feedforward = (p * self._room[:, None, :]).sum(axis=2)
        self._route = 0.44 * self._route + 0.56 * np.tanh(feedforward - 0.18 * self._route[:, ::-1])

    def _pick(self, scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.where(np.abs(scores[:, 0] - scores[:, 1]) < 1e-6, self._tie[rows], scores[:, 1] > scores[:, 0])

    def _set_entry(self, rows: np.ndarray) -> None:
        if self._entered_visit and self.cached_probe:
            self._target[rows] = self._pick(self.memory.cached[rows], rows)
            return
        score = self._route[rows]
        if not self._entered_visit:
            exposures = self.memory.transition_counts[rows].sum(axis=2)
            minimum = exposures.min(axis=1)
            sample = self._pick(-exposures, rows)
            self._target[rows] = np.where(minimum < 6, sample, self._pick(score, rows))
        else:
            self._target[rows] = self._pick(score, rows)

    def _set_fruit(self, rows: np.ndarray) -> None:
        if self._entered_visit and self.global_probe_fruit:
            self._target[rows] = self._pick(self._fruit[rows].max(axis=1), rows)
            return
        known = rows[self.memory.room[rows] >= 0]
        room = self.memory.room[known]
        scores = self._fruit[known, room]
        counts = self.memory.fruit_counts[known, room]
        self._target[rows] = self._tie[rows]
        choice = self._pick(scores, known)
        if not self._entered_visit:
            earlier = self._pick(-counts, known)
            choice = np.where(counts.min(axis=1) < 2, earlier, choice)
        self._target[known] = choice

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.memory.observe(obs)
        self._tick_graph()
        if obs[0, TRIAL_START]:
            self._tie[:] = self._rng.integers(2, size=len(self._tie))
            self._entered_visit |= bool(obs[0, VISIT])
        fresh = obs[:, STAGE_START] > 0
        entries = np.flatnonzero(fresh & (obs[:, ENTRANCE_STAGE] > 0))
        room = np.flatnonzero(fresh & (obs[:, ROOM_STAGE] > 0))
        if entries.size:
            self._set_entry(entries)
        if room.size:
            self._set_fruit(room)
        direction = (obs[:, LEFT_ID].argmax(axis=1) != self._target).astype(np.int8)
        availability = obs[:, AVAILABLE]
        forced = availability.sum(axis=1) == 1
        direction = np.where(forced, availability.argmax(axis=1), direction)
        self._motor = 0.4 * self._motor + 1.05 * _ONE_HOT[direction] * obs[:, ACTIVE, None]
        output = self._motor >= 1
        self._motor -= output
        return output


class OraclePolicy:
    """Evaluator-only control with access to the hidden transition and values."""

    def __init__(self, tape: TwoStepTape) -> None:
        self.tape = tape

    def reset(self, n: int) -> None:
        if n != self.tape.n:
            raise ValueError("oracle needs the corresponding tape")
        self._episode = -1
        self._rows = np.arange(n)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs[0, TRIAL_START]:
            self._episode += 1
        entrance = self.tape.initial_entrance if self._episode < self.tape.acquisition else self.tape.final_entrance
        room = obs[:, ROOM_ID].argmax(axis=1)
        fruit = self.tape.preferred_fruit(self._rows, self._episode, room)
        target = np.where(obs[:, ENTRANCE_STAGE] > 0, entrance, fruit)
        left = obs[:, LEFT_ID].argmax(axis=1)
        direction = (left != target).astype(np.int8)
        available = obs[:, AVAILABLE]
        direction = np.where(available.sum(axis=1) == 1, available.argmax(axis=1), direction)
        return (_ONE_HOT[direction] > 0) & (obs[:, ACTIVE, None] > 0)


class CachedTerminalPolicy(DirectedRoutePolicy):
    """Train an entrance value from actual terminal returns along its own route."""

    def __init__(self, seed: int = 0, *, mean_visits: bool = False) -> None:
        super().__init__(seed, cached_probe=True, write_routes=False, mean_visits=mean_visits)

    def _set_entry(self, rows: np.ndarray) -> None:
        choice = self._pick(self.memory.cached[rows], rows)
        if not self._entered_visit:
            visited = self.memory.cached_counts[rows]
            choice = np.where(visited.min(axis=1) < 6, self._pick(-visited, rows), choice)
        self._target[rows] = choice
