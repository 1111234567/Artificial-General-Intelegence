"""Fixed teachers and disjoint data for the consolidation experiment.

The model observes only Gaussian inputs, context A/B/C, and an explicit
context-local rule version.  Version is a supplied change signal; this
environment does not test learned change detection.  Only A changes in the
experimental stream.  Calling B or C with version 1 leaves their rules intact.

Targets and A/B agreement are environment-side feedback/evaluation.  They
must not be passed to a model as extra input features.  No data or teacher is
sampled in response to a model's behavior or to feedback requests.
"""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor
from torch.nn import functional as F


CONTEXTS = ("A", "B", "C")
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _generator(seed: int, stream: int) -> torch.Generator:
    # Separate streams keep teachers and every split independent of split sizes
    # and of the process-wide RNG used by models and training minibatches.
    stream_seed = (seed + 1_000_003 * stream) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(stream_seed)


def _sample_region(n: int, width: int, transfer: bool, rng: torch.Generator) -> Tensor:
    """Rejection-sample a standard normal conditioned on the requested region."""
    chunks = []
    remaining = n
    while remaining:
        candidates = torch.randn(4096, width, generator=rng, dtype=torch.float32, device="cpu")
        joint_positive = (candidates[:, 0] > 0) & (candidates[:, 1] > 0)
        accepted = candidates[joint_positive if transfer else ~joint_positive]
        accepted = accepted[:remaining]
        chunks.append(accepted)
        remaining -= len(accepted)
    return torch.cat(chunks) if chunks else torch.empty((0, width), dtype=torch.float32, device="cpu")


class World:
    """An immutable-in-use task definition with five independent raw-x splits.

    ``teacher_w`` and ``teacher_delta`` have shape [classes, input_dim], with
    independently sampled unit L2-norm rows.  Let ``a = argmax(W @ x)``.
    A/v0 gives a; B/either version gives argmax((W + 0.5 D) @ x);
    C/either version gives (a + 1) % classes; A/v1 gives (a + 2) % classes.
    Thus C always conflicts with A/v0, and A/v1 is a real revision that is also
    distinct from C.  A/B agreement is determined by x, never supplied as an
    observation.

    Train, val, and test exclude x[0] > 0 and x[1] > 0 jointly.  Transfer val
    and test contain only this held-out conjunction.  Individual positive
    factors still occur in the training distribution.  ``transfer_val`` has
    n_val rows and ``transfer_test`` has n_test rows.

    All random draws use private CPU generators before transfer to ``device``.
    Split construction cannot consume model RNG or alter the fixed teachers.
    """

    def __init__(
        self,
        seed: int,
        input_dim: int = 32,
        classes: int = 4,
        n_train: int = 16_384,
        n_val: int = 4_096,
        n_test: int = 8_192,
        device: torch.device | str = "cpu",
    ) -> None:
        self.seed = _integer("seed", seed, 0)
        self.input_dim = _integer("input_dim", input_dim, 2)
        self.classes = _integer("classes", classes, 3)
        n_train = _integer("n_train", n_train, 0)
        n_val = _integer("n_val", n_val, 0)
        n_test = _integer("n_test", n_test, 0)
        self.device = torch.device(device)
        self.feature_dim = self.input_dim + len(CONTEXTS) + 2

        self.teacher_w = F.normalize(
            torch.randn(
                self.classes, self.input_dim,
                generator=_generator(self.seed, 0), dtype=torch.float32, device="cpu",
            ),
            dim=-1,
        ).to(self.device)
        self.teacher_delta = F.normalize(
            torch.randn(
                self.classes, self.input_dim,
                generator=_generator(self.seed, 1), dtype=torch.float32, device="cpu",
            ),
            dim=-1,
        ).to(self.device)

        specifications = (
            ("train", n_train, False),
            ("val", n_val, False),
            ("test", n_test, False),
            ("transfer_val", n_val, True),
            ("transfer_test", n_test, True),
        )
        self.splits = {
            name: _sample_region(n, self.input_dim, transfer, _generator(self.seed, 10 + i)).to(
                self.device
            )
            for i, (name, n, transfer) in enumerate(specifications)
        }

    def _x(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have last dimension {self.input_dim}, got {tuple(x.shape)}")
        return x

    def _observation(
        self, value: str | int | Tensor, shape: torch.Size, name: str, count: int
    ) -> Tensor:
        if isinstance(value, str):
            if name != "context" or value not in CONTEXTS:
                raise ValueError(f"unknown {name}: {value!r}")
            value = CONTEXTS.index(value)
        value = torch.as_tensor(value, device=self.device)
        if value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} must contain integer indices")
        if not bool(((value >= 0) & (value < count)).all()):
            raise ValueError(f"{name} indices must be in [0, {count})")
        try:
            return torch.broadcast_to(value.long(), shape)
        except RuntimeError as error:
            raise ValueError(f"{name} shape must broadcast to input batch shape {tuple(shape)}") from error

    def features(self, x: Tensor, context: str | int | Tensor, version: int | Tensor) -> Tensor:
        """Return [..., input_dim + 5]: raw x, 3-way context, 2-way version.

        Context indices are A=0, B=1, C=2.  Context and version may be scalars
        or integer tensors broadcastable to x.shape[:-1].  No target, teacher
        score, agreement mask, or inferred conflict identity enters this path.
        """
        x = self._x(x)
        contexts = self._observation(context, x.shape[:-1], "context", 3)
        versions = self._observation(version, x.shape[:-1], "version", 2)
        return torch.cat(
            (x, F.one_hot(contexts, 3).to(x.dtype), F.one_hot(versions, 2).to(x.dtype)),
            dim=-1,
        )

    @torch.no_grad()
    def targets(self, x: Tensor, context: str | int | Tensor, version: int | Tensor) -> Tensor:
        """Environment-only feedback/evaluation labels, with shape x.shape[:-1]."""
        x = self._x(x)
        contexts = self._observation(context, x.shape[:-1], "context", 3)
        versions = self._observation(version, x.shape[:-1], "version", 2)
        a = (x @ self.teacher_w.T).argmax(dim=-1)
        b = (x @ (self.teacher_w + 0.5 * self.teacher_delta).T).argmax(dim=-1)
        revised_a = (a + 2 * versions) % self.classes
        c = (a + 1) % self.classes
        return torch.where(contexts == 1, b, torch.where(contexts == 2, c, revised_a))

    @torch.no_grad()
    def diagnostics(self, x: Tensor) -> dict[str, Tensor]:
        """Evaluator-only actual A/v0 versus B/v0 agreement for each input."""
        return {"ab_agree": self.targets(x, "A", 0) == self.targets(x, "B", 0)}

    def batch(
        self,
        split: str,
        indices: Tensor,
        context: str | int | Tensor,
        version: int | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Select arbitrary tensor indices and return (observations, feedback).

        Index order and repeated indices are preserved.  Scalar and multidim
        integer indices, as well as a split-length boolean mask, follow normal
        PyTorch indexing.  Forward the first item to the model; the second is
        available only when the experiment supplies feedback or evaluates.
        """
        x = self.splits[split][torch.as_tensor(indices, device=self.device)]
        return self.features(x, context, version), self.targets(x, context, version)
