"""Semantic Region APIs."""

from .builder import (
    LOWERING_CONTRACT_STATUS_COMPOSITE,
    LOWERING_CONTRACT_STATUS_DIRECT,
    LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE,
    LOWERING_CONTRACT_STATUS_SPECIALIZED,
    LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE,
    build_graph_optimization_input,
    build_semantic_region,
)
from .schema import GraphOptimizationInput, SemanticRegionRecord

__all__ = [
    "GraphOptimizationInput",
    "LOWERING_CONTRACT_STATUS_COMPOSITE",
    "LOWERING_CONTRACT_STATUS_DIRECT",
    "LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE",
    "LOWERING_CONTRACT_STATUS_SPECIALIZED",
    "LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE",
    "SemanticRegionRecord",
    "build_graph_optimization_input",
    "build_semantic_region",
]
