# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Capability registry lookup."""

from __future__ import annotations

from ..atomic import AtomicDAGRecord
from ..semantic import SemanticRegionRecord
from ..source import SourceRecord
from .schema import CapabilityRecord, RegistryResolution


LOOKUP_ORDER = (
    "semantic_region",
    "lowering_contract",
    "atomic_requirements",
    "shape_dtype_layout_constraints",
    "source_origin_policy",
    "runtime_admission",
)


class CapabilityRegistry:
    """Resolve support by the active lookup order."""

    def __init__(self, capabilities: tuple[CapabilityRecord, ...] | list[CapabilityRecord]):
        self._capabilities = tuple(capabilities)

    def resolve(
        self,
        *,
        source: SourceRecord,
        atomic: AtomicDAGRecord,
        semantic_region: SemanticRegionRecord,
    ) -> RegistryResolution:
        trace: list[str] = []
        trace.append(LOOKUP_ORDER[0])
        candidates = [
            capability
            for capability in self._capabilities
            if capability.semantic_region_key == semantic_region.semantic_region_key
        ]
        if not candidates:
            return _failure(trace, "no capability for Semantic Region")

        trace.append(LOOKUP_ORDER[1])
        candidates = [
            capability
            for capability in candidates
            if not capability.lowering_contract_ids
            or set(capability.lowering_contract_ids).issubset(
                set(
                    semantic_region.lowering_contract_ids
                    or (
                        (semantic_region.matched_contract_id,)
                        if semantic_region.matched_contract_id is not None
                        else ()
                    )
                )
            )
        ]
        if not candidates:
            return _failure(trace, "lowering contract not available for Semantic Region")

        trace.append(LOOKUP_ORDER[2])
        atomic_families = set(atomic.atomic_family_histogram)
        candidates = [
            capability
            for capability in candidates
            if set(capability.required_atomic_families).issubset(atomic_families)
        ]
        if not candidates:
            return _failure(trace, "atomic requirements not satisfied")

        trace.append(LOOKUP_ORDER[3])
        candidates = [
            capability
            for capability in candidates
            if _constraints_match(capability.shape_constraints, semantic_region.shape_constraints)
            and _constraints_match(capability.dtype_constraints, semantic_region.dtype_constraints)
            and _constraints_match(capability.layout_constraints, semantic_region.layout_constraints)
        ]
        if not candidates:
            return _failure(trace, "shape/dtype/layout constraints not satisfied")

        trace.append(LOOKUP_ORDER[4])
        candidates = [
            capability for capability in candidates if source.source_origin in capability.allowed_source_origins
        ]
        if not candidates:
            return _failure(trace, "source origin policy rejected input")

        trace.append(LOOKUP_ORDER[5])
        for capability in candidates:
            if capability.runtime_available:
                return RegistryResolution(
                    supported=True,
                    capability_id=capability.capability_id,
                    failure_reason=None,
                    query_trace=tuple(trace),
                )
        return _failure(trace, "runtime admission unavailable")


def _constraints_match(required: object, observed: object) -> bool:
    if not required:
        return True
    return required == observed


def _failure(trace: list[str], reason: str) -> RegistryResolution:
    return RegistryResolution(
        supported=False,
        capability_id=None,
        failure_reason=reason,
        query_trace=tuple(trace),
    )
