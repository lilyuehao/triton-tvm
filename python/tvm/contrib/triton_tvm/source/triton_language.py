# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Native Triton language source adapters."""

from __future__ import annotations

from .normalize import normalize_source_record, validate_native_triton_audit
from .schema import SourceArtifactKind, SourceOrigin, SourceRecord


def native_triton_source(
    *,
    source: str,
    function_name: str | None,
    target: str,
    producer_name: str | None = "triton",
    producer_version: str | None = None,
) -> SourceRecord:
    """Create a SourceRecord for auditable Triton language source."""

    record = normalize_source_record(
        content=source,
        ttir=None,
        source_origin=SourceOrigin.NATIVE_TRITON_LANGUAGE,
        artifact_kind=SourceArtifactKind.TRITON_PYTHON_SOURCE,
        function_name=function_name,
        target=target,
        producer_name=producer_name,
        producer_version=producer_version,
    )
    validate_native_triton_audit(record)
    return record


def native_triton_ttir(
    *,
    ttir: str,
    function_name: str | None,
    target: str,
    parent_source_id: str | None,
    parent_ttir_hash: str | None = None,
) -> SourceRecord:
    """Create a SourceRecord for TTIR with native Triton provenance."""

    record = normalize_source_record(
        content=None,
        ttir=ttir,
        source_origin=SourceOrigin.NATIVE_TRITON_LANGUAGE,
        artifact_kind=SourceArtifactKind.TTIR_MODULE,
        function_name=function_name,
        target=target,
        parent_source_id=parent_source_id,
        parent_ttir_hash=parent_ttir_hash,
    )
    validate_native_triton_audit(record)
    return record

