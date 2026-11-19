"""Existing directed and undirected tanh graph learning implementations.

Use LocalRulePolicy.act for custom public observation streams. The local-rule
evaluators own the private-symbol experiment's world and scoring. The fixed
port and learned interface models use temporal derivatives from recorded
blocks through TemporalCreditEngine, with optional pair structure updates.
"""

from experiments.undirected.credit import CreditConfig, TemporalCreditEngine, snapshot_parameters
from experiments.undirected.fast_local_life import Evaluation as ResidentLocalRuleEvaluator, FastLocalLife
from experiments.undirected.fast_training_evaluator import FastEvaluation as FastLocalTrainingEvaluator
from experiments.undirected.local_rule_evaluator import Evaluation as LocalRuleEvaluator
from experiments.undirected.local_rule_evaluator import stack_birth as stack_local_birth
from experiments.undirected.local_rule_policy import (
    PLASTIC_FIELDS, LocalRulePolicy, default_plastic, make_policy as _make_local_policy,
)
from experiments.undirected.model import PortGraphModel, UndirectedModel
from experiments.undirected.plasticity import PairConfig, PairOperator

__all__ = [
    "PLASTIC_FIELDS", "LocalRulePolicy", "default_plastic", "stack_local_birth", "make_local_policy",
    "LocalRuleEvaluator", "ResidentLocalRuleEvaluator", "FastLocalTrainingEvaluator", "FastLocalLife",
    "PortGraphModel", "UndirectedModel", "CreditConfig", "TemporalCreditEngine", "snapshot_parameters",
    "PairConfig", "PairOperator",
]


def make_local_policy(birth, actors, device, mode, *, backend="auto",
                      arc_kernel="standard", arc_tiles=1):
    """Construct the native tick policy and optionally install a CUDA arc kernel.

    ``backend`` selects the existing Torch or native CUDA tick path. ``tiled``
    and ``compact`` specialize the latter for 96 nodes and Tensor Cores. They
    keep the original implementation on other devices and graph sizes. Set
    ``arc_tiles`` to 1, 2, or 4 for ``tiled``. Use the resulting policy before
    capturing any CUDA graph that calls its ``act`` method.
    """
    if arc_kernel not in ("standard", "tiled", "compact"):
        raise ValueError("expected arc_kernel 'standard', 'tiled' or 'compact'")
    policy = _make_local_policy(birth, actors, device, mode, backend=backend)
    if arc_kernel == "tiled":
        from experiments.undirected.fast_local_arcs import install

        return install(policy, tiles=arc_tiles)
    if arc_kernel == "compact":
        from experiments.undirected.fast_local_arcs_compact import install

        return install(policy)
    return policy
