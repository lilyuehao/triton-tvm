# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Semantic Region records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SEMANTIC_REGION_SCHEMA_VERSION = "semantic_region_v1"


@dataclass(frozen=True)
class SemanticRegionRecord:
    """Validator-driven sidecar over an Atomic DAG."""

    semantic_region_hash: str
    semantic_region_key: str
    input_atomic_dag_hash: str
    input_source_id: str
    validator_id: str
    validator_version: str
    matched_contract_id: str
    required_atomic_families: tuple[str, ...]
    shape_constraints: Mapping[str, Any]
    dtype_constraints: Mapping[str, Any]
    layout_constraints: Mapping[str, Any]
    optimization_eligibility: Mapping[str, Any]
    schema_version: str = SEMANTIC_REGION_SCHEMA_VERSION


@dataclass(frozen=True)
class GraphOptimizationInput:
    """Typed input for future graph optimization work."""

    semantic_region: SemanticRegionRecord
    atomic_dag_hash: str
    source_id: str

