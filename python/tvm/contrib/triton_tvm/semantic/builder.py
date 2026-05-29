# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Semantic Region construction."""

from __future__ import annotations

from ..atomic import AtomicDAGRecord
from ..contracts import ContractValidationResult
from ..source import SourceRecord
from ..source.identity import stable_hash
from .schema import GraphOptimizationInput, SemanticRegionRecord


def build_semantic_region(
    source: SourceRecord,
    atomic: AtomicDAGRecord,
    validation: ContractValidationResult,
) -> SemanticRegionRecord:
    """Create a Semantic Region only from a passing contract validation result."""

    if not validation.passed:
        raise ValueError("failed contract validation cannot create a Semantic Region")
    if validation.input_source_id != source.source_id:
        raise ValueError("validation source does not match SourceRecord")
    if validation.input_atomic_dag_hash != atomic.atomic_dag_hash:
        raise ValueError("validation Atomic DAG does not match AtomicDAGRecord")
    if validation.matched_contract_id is None:
        raise ValueError("passing validation requires a matched contract")
    key = validation.matched_contract_id
    payload = {
        "semantic_region_key": key,
        "input_atomic_dag_hash": atomic.atomic_dag_hash,
        "input_source_id": source.source_id,
        "validator_id": validation.validator_id,
        "validator_version": validation.validator_version,
        "required_atomic_families": validation.required_atomic_families,
        "shape_constraints": validation.shape_constraints,
        "dtype_constraints": validation.dtype_constraints,
        "layout_constraints": validation.layout_constraints,
    }
    return SemanticRegionRecord(
        semantic_region_hash=stable_hash(payload),
        semantic_region_key=key,
        input_atomic_dag_hash=atomic.atomic_dag_hash,
        input_source_id=source.source_id,
        validator_id=validation.validator_id,
        validator_version=validation.validator_version,
        matched_contract_id=validation.matched_contract_id,
        required_atomic_families=validation.required_atomic_families,
        shape_constraints=dict(validation.shape_constraints),
        dtype_constraints=dict(validation.dtype_constraints),
        layout_constraints=dict(validation.layout_constraints),
        optimization_eligibility={"graph_optimization_input": True},
    )


def build_graph_optimization_input(
    semantic_region: SemanticRegionRecord, atomic: AtomicDAGRecord
) -> GraphOptimizationInput:
    """Bundle semantic and atomic identities for future optimization."""

    if semantic_region.input_atomic_dag_hash != atomic.atomic_dag_hash:
        raise ValueError("Semantic Region does not match Atomic DAG")
    return GraphOptimizationInput(
        semantic_region=semantic_region,
        atomic_dag_hash=atomic.atomic_dag_hash,
        source_id=atomic.source_id,
    )

