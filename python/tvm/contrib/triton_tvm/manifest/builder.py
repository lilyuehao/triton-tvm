# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Manifest construction helpers."""

from __future__ import annotations

from typing import Any, Mapping

from ..atomic import AtomicDAGRecord
from ..contracts import ContractValidationResult
from ..semantic import SemanticRegionRecord
from ..source import SourceRecord
from ..source.identity import stable_hash
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
    validation: ContractValidationResult | None = None,
    declared_semantic_region_key: str | None = None,
    source_route_record_ids: tuple[str, ...] | None = None,
    source_ids: tuple[str, ...] | None = None,
    atomic_dag_hashes: tuple[str, ...] | None = None,
    atomic_dag_bundle_hash: str | None = None,
    lowering_contract_ids: tuple[str, ...] | None = None,
    lowering_contract_status: str | None = None,
) -> OperatorManifestRecord:
    """Build an operator manifest entry without duplicating proof facts."""

    if semantic_region is not None and atomic is None:
        raise ValueError("Semantic Region proof requires an Atomic DAG")
    if semantic_region is not None and semantic_region.input_source_id != source.source_id:
        raise ValueError("Semantic Region source does not match manifest source")
    if semantic_region is not None and semantic_region.input_atomic_dag_hash != atomic.atomic_dag_hash:
        raise ValueError("Semantic Region Atomic DAG does not match manifest Atomic DAG")
    route_record_ids = tuple(source_route_record_ids or (operator_id,))
    manifest_source_ids = tuple(source_ids or (source.source_id,))
    manifest_atomic_hashes = tuple(
        atomic_dag_hashes
        or ((atomic.atomic_dag_hash,) if atomic is not None else ())
    )
    bundle_hash = atomic_dag_bundle_hash
    if bundle_hash is None and manifest_atomic_hashes:
        bundle_hash = stable_hash({"atomic_dag_hashes": manifest_atomic_hashes})
    contract_ids = tuple(
        lowering_contract_ids
        or (
            semantic_region.lowering_contract_ids
            if semantic_region is not None and semantic_region.lowering_contract_ids
            else (
                (semantic_region.matched_contract_id,)
                if semantic_region is not None
                else (
                    (validation.matched_contract_id,)
                    if validation is not None and validation.matched_contract_id is not None
                    else ()
                )
            )
        )
    )
    validation_status = validation.validation_status if validation is not None else None
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
        declared_semantic_region_key=(
            declared_semantic_region_key
            or (
                semantic_region.declared_semantic_region_key
                if semantic_region is not None
                else None
            )
        ),
        source_route_record_ids=route_record_ids,
        source_ids=manifest_source_ids,
        atomic_dag_hashes=manifest_atomic_hashes,
        atomic_dag_bundle_hash=bundle_hash,
        lowering_contract_ids=contract_ids,
        lowering_contract_status=(
            lowering_contract_status
            or (semantic_region.lowering_contract_status if semantic_region is not None else None)
        ),
        matched_contract_id=(
            semantic_region.matched_contract_id
            if semantic_region is not None
            else (validation.matched_contract_id if validation is not None else None)
        ),
        contract_validation_status=validation_status,
        semantic_proof_status=(
            semantic_region.semantic_proof_status if semantic_region is not None else None
        ),
        proof_source=semantic_region.proof_source if semantic_region is not None else None,
        support_claim_source=(
            semantic_region.support_claim_source if semantic_region is not None else None
        ),
        wrapper_provider_claim=(
            semantic_region.wrapper_provider_claim if semantic_region is not None else False
        ),
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
        "semantic_region_key": semantic.semantic_region_key if semantic is not None else None,
        "declared_semantic_region_key": (
            semantic.declared_semantic_region_key if semantic is not None else None
        ),
        "lowering_contract_ids": (
            semantic.lowering_contract_ids if semantic is not None else record.lowering_contract_ids
        ),
        "lowering_contract_status": (
            semantic.lowering_contract_status
            if semantic is not None
            else record.lowering_contract_status
        ),
    }
