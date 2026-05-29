# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Runtime admission helpers for the final route."""

from __future__ import annotations

from dataclasses import dataclass

from ..atomic import AtomicDAGRecord
from ..contracts import ContractValidationResult
from ..manifest import OperatorManifestRecord
from ..registry import RegistryResolution
from ..semantic import SemanticRegionRecord
from ..source import SourceRecord


@dataclass(frozen=True)
class RuntimeAdmissionResult:
    admitted: bool
    status: str
    reason: str | None


def admit_runtime(
    *,
    source: SourceRecord,
    atomic: AtomicDAGRecord,
    validation: ContractValidationResult,
    semantic_region: SemanticRegionRecord | None,
    manifest_record: OperatorManifestRecord,
    registry_resolution: RegistryResolution,
) -> RuntimeAdmissionResult:
    """Admit only the SourceRecord-to-Semantic-Region route."""

    if atomic.source_id != source.source_id:
        return RuntimeAdmissionResult(False, "rejected", "source and Atomic DAG mismatch")
    if not validation.passed:
        return RuntimeAdmissionResult(False, "gap", validation.failure_reason)
    if semantic_region is None:
        return RuntimeAdmissionResult(False, "gap", "missing Semantic Region")
    if manifest_record.semantic_region_hash != semantic_region.semantic_region_hash:
        return RuntimeAdmissionResult(False, "rejected", "manifest and Semantic Region mismatch")
    if not registry_resolution.supported:
        return RuntimeAdmissionResult(False, "gap", registry_resolution.failure_reason)
    return RuntimeAdmissionResult(True, "admitted", None)

