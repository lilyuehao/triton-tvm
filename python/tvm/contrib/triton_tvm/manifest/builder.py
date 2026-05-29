# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Manifest construction helpers."""

from __future__ import annotations

from typing import Any, Mapping

from ..atomic import AtomicDAGRecord
from ..semantic import SemanticRegionRecord
from ..source import SourceRecord
from .schema import ManifestGapRecord, OperatorManifestRecord


def build_operator_manifest_record(
    *,
    model_id: str,
    operator_id: str,
    execution_order: int,
    source: SourceRecord,
    atomic: AtomicDAGRecord | None,
    semantic_region: SemanticRegionRecord | None,
    shape_signature: Mapping[str, Any] | None = None,
    dtype_signature: Mapping[str, Any] | None = None,
    layout_signature: Mapping[str, Any] | None = None,
    runtime_admission_status: str = "not_checked",
    gap_reason: str | None = None,
    optional_model_anchor: str | None = None,
) -> OperatorManifestRecord:
    """Build an operator manifest entry without duplicating proof facts."""

    if semantic_region is not None and atomic is None:
        raise ValueError("Semantic Region proof requires an Atomic DAG")
    if semantic_region is not None and semantic_region.input_source_id != source.source_id:
        raise ValueError("Semantic Region source does not match manifest source")
    if semantic_region is not None and semantic_region.input_atomic_dag_hash != atomic.atomic_dag_hash:
        raise ValueError("Semantic Region Atomic DAG does not match manifest Atomic DAG")
    return OperatorManifestRecord(
        model_id=model_id,
        operator_id=operator_id,
        execution_order=execution_order,
        source_id=source.source_id,
        atomic_dag_hash=atomic.atomic_dag_hash if atomic is not None else None,
        semantic_region_hash=(
            semantic_region.semantic_region_hash if semantic_region is not None else None
        ),
        semantic_region_key=(
            semantic_region.semantic_region_key if semantic_region is not None else None
        ),
        shape_signature=dict(shape_signature or {}),
        dtype_signature=dict(dtype_signature or {}),
        layout_signature=dict(layout_signature or {}),
        runtime_admission_status=runtime_admission_status,
        gap_reason=gap_reason,
        optional_model_anchor=optional_model_anchor,
    )


def build_manifest_gap(
    *, model_id: str, operator_id: str, reason: str, source_id: str | None = None
) -> ManifestGapRecord:
    return ManifestGapRecord(
        model_id=model_id,
        operator_id=operator_id,
        source_id=source_id,
        reason=reason,
    )


def recover_manifest_proof_facts(
    record: OperatorManifestRecord,
    *,
    sources: Mapping[str, SourceRecord],
    atomics: Mapping[str, AtomicDAGRecord],
    semantics: Mapping[str, SemanticRegionRecord],
) -> dict[str, Any]:
    """Recover denormalized proof facts through joins."""

    source = sources[record.source_id]
    atomic = atomics[record.atomic_dag_hash] if record.atomic_dag_hash else None
    semantic = semantics[record.semantic_region_hash] if record.semantic_region_hash else None
    return {
        "source_id": source.source_id,
        "source_origin": source.source_origin.value,
        "artifact_kind": source.artifact_kind.value,
        "source_hash": source.source_hash,
        "ttir_hash": atomic.ttir_hash if atomic is not None else source.ttir_hash,
        "validator_id": semantic.validator_id if semantic is not None else None,
        "matched_contract_id": semantic.matched_contract_id if semantic is not None else None,
    }

