"""Exact equilibrium response witnesses for matched 32-cell circuits."""

import json
from dataclasses import dataclass
from fractions import Fraction


WIDTH = 32
PREDECESSOR = tuple((node - 1) % WIDTH for node in range(WIDTH))
ANTIPODE = tuple((node + WIDTH // 2) % WIDTH for node in range(WIDTH))


@dataclass(frozen=True)
class Witness:
    arm: str
    keep: Fraction
    scaled_gain: tuple[Fraction, ...]
    sources: tuple[tuple[int, ...], ...]
    drive: tuple[int, ...]
    order: int

    def moments(self):
        """Choose the same scalar input direction and readout."""
        state = tuple(map(Fraction, self.drive))
        samples = []
        for _ in range(2 * self.order):
            samples.append(sum(weight * cell for weight, cell in zip(self.drive, state, strict=True)))
            state = tuple(self.keep * cell + gain * sum(state[source] for source in incoming)
                          for cell, gain, incoming in zip(state, self.scaled_gain, self.sources,
                                                          strict=True))
        return samples


def witnesses():
    nodes = range(WIDTH)
    keep = Fraction(1, 2)
    paired = {0: Fraction(1, 4), 16: Fraction(1, 4),
              2: Fraction(3, 8), 18: Fraction(3, 8)}
    branched = tuple(zip(PREDECESSOR, ANTIPODE, strict=True))
    single_input = (1,) + (0,) * (WIDTH - 1)
    return {
        "self_negative": Witness("self", keep,
                                 tuple(-Fraction(node + 1, 2 * WIDTH) - keep for node in nodes),
                                 tuple((node,) for node in nodes), (1,) * WIDTH, WIDTH),
        "directed_rotating": Witness("directed", keep, (Fraction(1, 4),) * WIDTH,
                                      tuple((node,) for node in PREDECESSOR), single_input, WIDTH),
        "multiloop_negative": Witness("multiloop", Fraction(1, 8),
                                      tuple(paired.get(node, Fraction(0)) for node in nodes),
                                      branched, tuple(int(node in (0, 2)) for node in nodes), 4),
        "multiloop_rotating": Witness("multiloop", keep, (Fraction(1, 8),) * WIDTH,
                                      branched, single_input, WIDTH),
    }


def _polynomial(samples, order):
    """Recover the scalar response polynomial from an invertible Hankel matrix."""
    rows = [samples[start:start + order] + [samples[start + order]] for start in range(order)]
    for col in range(order):
        pivot = next((row for row in range(col, order) if rows[row][col]), None)
        if pivot is None:
            raise ArithmeticError(f"The Hankel matrix is singular at column {col}")
        rows[pivot], rows[col] = rows[col], rows[pivot]
        for row in range(col + 1, order):
            ratio = rows[row][col] / rows[col][col]
            if ratio:
                for entry in range(col, order + 1):
                    rows[row][entry] -= ratio * rows[col][entry]
    result = [Fraction(0)] * order
    for col in reversed(range(order)):
        remainder = sum(rows[col][entry] * result[entry] for entry in range(col + 1, order))
        result[col] = (rows[col][-1] - remainder) / rows[col][col]
    return [-value for value in result] + [Fraction(1)]


def certificates():
    result = {}
    for name, witness in witnesses().items():
        samples = witness.moments()
        polynomial = _polynomial(samples, witness.order)
        result[name] = {
            "width": WIDTH,
            "hankel_rank": witness.order,
            "first_six_impulse_derivatives": list(map(str, samples[:6])),
            "minimal_polynomial_constant_to_leading": list(map(str, polynomial)),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(certificates(), indent=2))
