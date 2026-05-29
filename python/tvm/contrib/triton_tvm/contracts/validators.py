# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Contract validators for Atomic DAG promotion."""

from __future__ import annotations

from typing import Any, Mapping

from ..atomic import (
    ATOMIC_FAMILY_DOT,
    ATOMIC_FAMILY_MEMORY,
    ATOMIC_FAMILY_UNSUPPORTED,
    AtomicDAGRecord,
)
from ..source import SourceRecord
from .schema import ContractValidationResult


VALIDATOR_ID = "triton_tvm_contract_validator"
VALIDATOR_VERSION = "1"

_CONTRACT_REQUIRED_FAMILIES = {
    "pointwise_flat": (ATOMIC_FAMILY_MEMORY,),
    "matmul": (ATOMIC_FAMILY_DOT,),
    "matmul_bias_epilogue": (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY),
    "conv2d_nchw_static": (ATOMIC_FAMILY_MEMORY,),
    "attention_decomposed": (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY),
}


def validate_atomic_contract(
    source: SourceRecord,
    atomic: AtomicDAGRecord,
    *,
    contract_id: str,
    shape_constraints: Mapping[str, Any] | None = None,
    dtype_constraints: Mapping[str, Any] | None = None,
    layout_constraints: Mapping[str, Any] | None = None,
) -> ContractValidationResult:
    """Validate whether an Atomic DAG satisfies a known semantic contract."""

    required = _CONTRACT_REQUIRED_FAMILIES.get(contract_id)
    if required is None:
        return _failed(source, atomic, contract_id, (), "unknown contract")
    if atomic.unsupported_atomic_ops:
        return _failed(
            source,
            atomic,
            contract_id,
            required,
            f"{len(atomic.unsupported_atomic_ops)} unsupported atomic ops",
        )
    families = set(atomic.atomic_family_histogram)
    missing = tuple(family for family in required if family not in families)
    if missing:
        return _failed(
            source,
            atomic,
            contract_id,
            required,
            "missing required atomic families: " + ", ".join(missing),
        )
    return ContractValidationResult(
        validator_id=VALIDATOR_ID,
        validator_version=VALIDATOR_VERSION,
        input_source_id=source.source_id,
        input_atomic_dag_hash=atomic.atomic_dag_hash,
        validation_status="passed",
        matched_contract_id=contract_id,
        failure_reason=None,
        required_atomic_families=required,
        shape_constraints=dict(shape_constraints or {}),
        dtype_constraints=dict(dtype_constraints or {}),
        layout_constraints=dict(layout_constraints or {}),
    )


def _failed(
    source: SourceRecord,
    atomic: AtomicDAGRecord,
    contract_id: str,
    required: tuple[str, ...],
    reason: str,
) -> ContractValidationResult:
    return ContractValidationResult(
        validator_id=VALIDATOR_ID,
        validator_version=VALIDATOR_VERSION,
        input_source_id=source.source_id,
        input_atomic_dag_hash=atomic.atomic_dag_hash,
        validation_status="failed",
        matched_contract_id=contract_id if contract_id in _CONTRACT_REQUIRED_FAMILIES else None,
        failure_reason=reason,
        required_atomic_families=required,
        shape_constraints={},
        dtype_constraints={},
        layout_constraints={},
    )


def supported_contract_ids() -> tuple[str, ...]:
    """Return active contract identifiers."""

    return tuple(_CONTRACT_REQUIRED_FAMILIES)

