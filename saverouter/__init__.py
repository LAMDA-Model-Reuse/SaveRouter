"""SAVERouter: fixed-K sparse supervision for economical LLM routing."""

from .data import SparseSupervision
from .features import TextFeatureEncoder
from .grouping import GroupAssigner
from .metrics import supervision_amortized_metrics
from .router import FixedKSparseRouter
from .supervision import collect_fixed_k_supervision, simulate_fixed_k_supervision

__version__ = "0.1.0"

__all__ = [
    "FixedKSparseRouter",
    "GroupAssigner",
    "SparseSupervision",
    "TextFeatureEncoder",
    "collect_fixed_k_supervision",
    "simulate_fixed_k_supervision",
    "supervision_amortized_metrics",
]
