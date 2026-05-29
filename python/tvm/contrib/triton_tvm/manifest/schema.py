# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Model/operator manifest records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OperatorManifestRecord:
    """Manifest record keyed by model_id plus operator_id."""

    model_id: str
    operator_id: str
    execution_order: int
    source_id: str
    atomic_dag_hash: str | None
    semantic_region_hash: str | None
    semantic_region_key: str | None
    shape_signature: Mapping[str, Any]
    dtype_signature: Mapping[str, Any]
    layout_signature: Mapping[str, Any]
    runtime_admission_status: str
    gap_reason: str | None
    optional_model_anchor: str | None = None

    @property
    def identity(self) -> tuple[str, str]:
        return (self.model_id, self.operator_id)


@dataclass(frozen=True)
class ManifestGapRecord:
    """Unsupported operator gap."""

    model_id: str
    operator_id: str
    source_id: str | None
    reason: str

