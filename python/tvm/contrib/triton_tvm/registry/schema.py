# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Capability registry schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..source import SourceOrigin


@dataclass(frozen=True)
class CapabilityRecord:
    """One registry capability entry."""

    capability_id: str
    semantic_region_key: str
    required_atomic_families: tuple[str, ...]
    shape_constraints: Mapping[str, Any]
    dtype_constraints: Mapping[str, Any]
    layout_constraints: Mapping[str, Any]
    allowed_source_origins: tuple[SourceOrigin, ...]
    runtime_available: bool


@dataclass(frozen=True)
class RegistryResolution:
    """Capability lookup result."""

    supported: bool
    capability_id: str | None
    failure_reason: str | None
    query_trace: tuple[str, ...]

