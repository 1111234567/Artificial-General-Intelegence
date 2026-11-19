"""Read-only recurrent-weight interventions for an existing dense controller.

Weights use the frozen model's convention W[target, source]. The DAG cuts
remove only distinct-node feedback: diagonal self recurrence is always kept.
They are large distribution shifts, so loss relative to the original network
alone cannot identify a task contribution of loops. The random comparator
matches retained incoming edge counts, and a second pair additionally matches
per-target off-diagonal L2 norms without amplifying any retained weight.

Selection uses only the original nonzero support and a private CPU generator.
There is no training, label, observation, hidden-state or world access here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral
from typing import Any, TypeVar

import torch
from torch import nn


FORMAT_VERSION = "loops-weight-interventions-v1"
INTERVENTIONS = (
    "none", "diagonal_only", "dag_cut", "random_cut",
    "dag_norm_matched", "random_norm_matched",
    "dag_quarter", "random_quarter",
)
_Model = TypeVar("_Model", bound=nn.Module)


def _validate_weight(weight: torch.Tensor) -> None:
    if (not isinstance(weight, torch.Tensor) or weight.layout != torch.strided
            or weight.ndim != 2 or weight.shape[0] < 1
            or weight.shape[0] != weight.shape[1]
            or not weight.is_floating_point()):
        raise ValueError("recurrent weight must be a nonempty square real floating tensor")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("recurrent weight must be finite")


def _tensor_sha256(value: torch.Tensor) -> str:
    saved = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{saved.dtype}/{tuple(saved.shape)}".encode())
    digest.update(saved.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _row_norms(weight: torch.Tensor) -> torch.Tensor:
    offdiagonal = weight.detach().cpu().double().clone()
    offdiagonal.fill_diagonal_(0)
    return torch.linalg.vector_norm(offdiagonal, dim=1)


def _masked(weight: torch.Tensor, offdiagonal_mask: torch.Tensor) -> torch.Tensor:
    keep = offdiagonal_mask.to(device=weight.device).clone()
    keep.fill_diagonal_(True)
    return weight.masked_fill(~keep, 0)


def _rescaled(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    result = weight * scale.to(device=weight.device, dtype=weight.dtype)[:, None]
    # Keep self recurrence exact, including signed zeros, irrespective of scale.
    result.diagonal().copy_(weight.diagonal())
    return result


@torch.no_grad()
def build_interventions(weight: torch.Tensor, seed: int) -> tuple[
        dict[str, torch.Tensor], dict[str, Any]]:
    """Return detached independent weights and JSON-safe construction metadata.

    One private generator draws the node order, then samples a uniform subset
    of existing off-diagonal edges independently for each target row. Its
    subset size equals that row's DAG retained count. No candidate is rejected
    based on cyclicity, weights, task results or later observations. A random
    comparator can itself be acyclic, especially for tiny or sparse matrices.

    The normalized pair shares row target min(norm(DAG row), norm(random row)).
    Empty rows keep scale 1 and target norm 0. The two raw cuts are also exposed
    because norm matching changes retained edge magnitudes. Outgoing degrees,
    spectrum, signed input balance and trajectory distribution are unmatched.
    The quarter pair interpolates 25% toward those matched endpoints, retaining
    original nonzero support; it is not acyclic or row-norm matched in general.
    """
    _validate_weight(weight)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    original = weight.detach().clone()
    width = original.shape[0]
    support = original.cpu().ne(0)
    support.fill_diagonal_(False)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(width, generator=generator)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(width)
    dag_mask = support & (rank[:, None] > rank[None, :])
    counts = dag_mask.sum(dim=1)
    random_mask = torch.zeros_like(dag_mask)
    for target in range(width):
        retained = int(counts[target])
        if retained:
            candidates = support[target].nonzero(as_tuple=True)[0]
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:retained]]
            random_mask[target, chosen] = True

    dag, random = _masked(original, dag_mask), _masked(original, random_mask)
    dag_norms, random_norms = _row_norms(dag), _row_norms(random)
    common_norms = torch.minimum(dag_norms, random_norms)
    dag_scale, random_scale = torch.ones(width, dtype=torch.float64), torch.ones(width, dtype=torch.float64)
    for norms, scale in ((dag_norms, dag_scale), (random_norms, random_scale)):
        nonempty = norms > 0
        scale[nonempty] = common_norms[nonempty] / norms[nonempty]
    matched_dag, matched_random = _rescaled(dag, dag_scale), _rescaled(random, random_scale)
    quarter_dag, quarter_random = .75 * original + .25 * matched_dag, .75 * original + .25 * matched_random
    quarter_dag.diagonal().copy_(original.diagonal())
    quarter_random.diagonal().copy_(original.diagonal())
    weights = dict(zip(INTERVENTIONS, (
        original.clone(), _masked(original, torch.zeros_like(support)), dag, random,
        matched_dag, matched_random, quarter_dag, quarter_random,
    )))
    masks = dict(zip(INTERVENTIONS, (support, torch.zeros_like(support),
                                   dag_mask, random_mask, dag_mask, random_mask, support, support)))
    scales = {name: torch.ones(width, dtype=torch.float64) for name in INTERVENTIONS}
    scales.update(dag_norm_matched=dag_scale, random_norm_matched=random_scale)
    records = {}
    for name, result in weights.items():
        partial = name.endswith("quarter")
        mask = masks[name]
        actual_support = result.detach().cpu().ne(0)
        actual_support.fill_diagonal_(False)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite intervention weight")
        if not torch.equal(actual_support, mask):
            raise FloatingPointError("intervention rescaling lost a retained nonzero edge")
        if not torch.equal(result.diagonal(), original.diagonal()):
            raise AssertionError("intervention changed diagonal recurrence")
        records[name] = {
            "weight_sha256": _tensor_sha256(result),
            "offdiagonal_mask_sha256": _tensor_sha256(mask),
            "retained_offdiagonal_edges": int(mask.sum()),
            "retained_offdiagonal_per_target": mask.sum(1).tolist(),
            "retained_offdiagonal_per_source": mask.sum(0).tolist(),
            "offdiagonal_row_l2": _row_norms(result).tolist(),
            "row_scales": None if partial else scales[name].tolist(),
            "diagonal_preserved": True,
            "distinct_node_acyclic_by_construction": name in (
                "diagonal_only", "dag_cut", "dag_norm_matched"),
            "rescaling": ("convex_interpolation_with_matched_endpoint" if partial else
                          "shared_minimum_raw_row_offdiagonal_l2" if name.endswith("norm_matched") else "none"),
            "interpolation_alpha": .25 if partial else None,
            "interpolation_endpoint": ("dag_norm_matched" if name == "dag_quarter" else
                                       "random_norm_matched" if name == "random_quarter" else None),
            "interpolated_row_l2_matching_claimed": False,
        }
    metadata = {
        "format": FORMAT_VERSION, "seed": int(seed), "width": width,
        "source_weight_sha256": _tensor_sha256(original),
        "matrix_convention": "W[target, source]",
        "node_order": order.tolist(),
        "dag_retention_rule": "rank(source) < rank(target); diagonal always retained",
        "rng": "private CPU torch.Generator; node randperm then target-row subset randperms",
        "selection_inputs": ["source nonzero offdiagonal support", "private seed"],
        "common_target_offdiagonal_row_l2": common_norms.tolist(),
        "matched_properties": [
            "DAG/random pairs retain identical incoming nonzero edge counts for every target",
            "normalized pair shares per-target offdiagonal L2; retained magnitudes never increase",
            "all variants preserve original diagonal self recurrence exactly",
        ],
        "unmatched_properties": [
            "outgoing degree per source", "spectrum and spectral radius",
            "signed input balance and edge geometry", "nonlinear state and policy distributions",
        ],
        "limitations": [
            "DAG cuts remove distinct-node cycles but retain diagonal self recurrence",
            "cutting roughly half of a dense matrix is a large out-of-distribution intervention",
            "performance loss relative to none alone cannot be attributed specifically to loops",
            "matched random cuts can also be acyclic; no rejection sampling enforces cycles",
            "row norm matching rescales retained weights and does not match dynamics or spectrum",
            "quarter variants partially attenuate toward matched endpoints; they are not acyclic or row-norm matched in general",
        ],
        "interventions": records,
    }
    return weights, metadata


@torch.no_grad()
def clone_with_recurrent(model: _Model, weight: torch.Tensor) -> _Model:
    """Make an isolated frozen eval clone; never touch live state or history.

    Pass this clone and the unchanged full hidden value to the existing frozen
    ``diagnose_fork`` observer, which clones the world and hidden state itself.
    There is no initialization, burn-in, retraining or hidden-state correction.
    """
    _validate_weight(weight)
    recurrent = getattr(model, "recurrent", None)
    if (not isinstance(recurrent, torch.Tensor) or recurrent.shape != weight.shape
            or recurrent.dtype != weight.dtype):
        raise ValueError("model must have recurrent weight of the same shape and dtype")
    result = deepcopy(model)
    result.recurrent.copy_(weight.to(device=result.recurrent.device))
    result.eval().requires_grad_(False)
    for parameter in result.parameters():
        parameter.grad = None
    return result
