# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Contract validation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTRACT_SCHEMA_VERSION = "contract_validation_v1"


@dataclass(frozen=True)
class ContractValidationResult:
    """Result of validating an Atomic DAG against a semantic contract."""

    validator_id: str
    validator_version: str
    input_source_id: str
    input_atomic_dag_hash: str
    validation_status: str
    matched_contract_id: str | None
    failure_reason: str | None
    required_atomic_families: tuple[str, ...]
    shape_constraints: Mapping[str, Any]
    dtype_constraints: Mapping[str, Any]
    layout_constraints: Mapping[str, Any]
    schema_version: str = CONTRACT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.validation_status == "passed"

