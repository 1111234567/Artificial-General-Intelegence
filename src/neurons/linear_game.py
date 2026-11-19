"""Whole-game accelerators used with the linear sensory graph.

These are the original population scorers. Their public ports, physics and
fitness statistics belong to the corresponding game. For training on arbitrary
inputs use LinearGraph in linear.py.
"""

from experiments.pulse_game.symmetry_policy import Cohort as PairedPolicyCohort
from experiments.pulse_game.unified_physical import Cohort as UnifiedPolicyCohort
from experiments.pulse_game.longrun_plastic import Cohort as LongLifeCohort

__all__ = ["PairedPolicyCohort", "UnifiedPolicyCohort", "LongLifeCohort"]
