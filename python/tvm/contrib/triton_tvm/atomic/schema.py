# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Atomic DAG schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..source import SourceArtifactKind, SourceOrigin


ATOMIC_DAG_SCHEMA_VERSION = "atomic_dag_v1"

ATOMIC_FAMILY_LAUNCH_INDEX = "launch_index"
ATOMIC_FAMILY_SHAPE_VIEW = "shape_view"
ATOMIC_FAMILY_POINTER_INDEX = "pointer_index"
ATOMIC_FAMILY_MEMORY = "memory"
ATOMIC_FAMILY_ARITH = "arith"
ATOMIC_FAMILY_MATH = "math"
ATOMIC_FAMILY_COMPARE_SELECT = "compare_select"
ATOMIC_FAMILY_CAST_DTYPE = "cast_dtype"
ATOMIC_FAMILY_REDUCTION = "reduction"
ATOMIC_FAMILY_DOT = "dot"
ATOMIC_FAMILY_UNSUPPORTED = "unsupported_atomic_op"

SUPPORTED_ATOMIC_FAMILIES = (
    ATOMIC_FAMILY_LAUNCH_INDEX,
    ATOMIC_FAMILY_SHAPE_VIEW,
    ATOMIC_FAMILY_POINTER_INDEX,
    ATOMIC_FAMILY_MEMORY,
    ATOMIC_FAMILY_ARITH,
    ATOMIC_FAMILY_MATH,
    ATOMIC_FAMILY_COMPARE_SELECT,
    ATOMIC_FAMILY_CAST_DTYPE,
    ATOMIC_FAMILY_REDUCTION,
    ATOMIC_FAMILY_DOT,
)


@dataclass(frozen=True)
class AtomicOpRecord:
    """TTIR-derived atomic operation."""

    atomic_id: str
    atomic_family: str
    ttir_op_name: str
    operands: tuple[str, ...]
    results: tuple[str, ...]
    dtype: str | None
    shape: tuple[int | str, ...] | None
    attrs: Mapping[str, Any]
    source_location: str | None
    producer_ids: tuple[str, ...]
    consumer_ids: tuple[str, ...]
    local_scope_id: str | None
    unsupported_reason: str | None


@dataclass(frozen=True)
class AtomicDAGRecord:
    """Stable low-level proof structure and future graph optimization input."""

    atomic_dag_hash: str
    source_id: str
    source_origin: SourceOrigin
    artifact_kind: SourceArtifactKind
    function_name: str | None
    target: str
    source_hash: str | None
    ttir_hash: str | None
    atomic_ops: tuple[AtomicOpRecord, ...]
    unsupported_atomic_ops: tuple[AtomicOpRecord, ...]
    atomic_family_histogram: Mapping[str, int]
    producer_consumer_consistent: bool
    schema_version: str = ATOMIC_DAG_SCHEMA_VERSION

