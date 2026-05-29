# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Input normalization helpers for source records."""

from __future__ import annotations

from typing import Any, Mapping

from .identity import build_source_id, source_hash_from_text, ttir_hash_from_text
from .schema import SourceArtifactKind, SourceOrigin, SourceRecord


def normalize_source_record(
    *,
    content: str | None,
    ttir: str | None,
    source_origin: SourceOrigin | str,
    artifact_kind: SourceArtifactKind | str,
    function_name: str | None,
    target: str,
    producer_name: str | None = None,
    producer_version: str | None = None,
    parent_source_id: str | None = None,
    parent_ttir_hash: str | None = None,
    debug_origin: Mapping[str, Any] | None = None,
) -> SourceRecord:
    """Create a SourceRecord with stable hashes from source and TTIR material."""

    origin = SourceOrigin(source_origin)
    artifact = SourceArtifactKind(artifact_kind)
    src_hash = source_hash_from_text(content) if content is not None else None
    module_hash = ttir_hash_from_text(ttir) if ttir is not None else None
    identity_hash = src_hash or module_hash
    if identity_hash is None:
        raise ValueError("source normalization requires source text or TTIR text")
    source_id = build_source_id(
        origin=origin.value,
        artifact=artifact.value,
        content_hash=identity_hash,
        function_name=function_name,
    )
    return SourceRecord(
        source_id=source_id,
        source_origin=origin,
        artifact_kind=artifact,
        source_hash=src_hash,
        ttir_hash=module_hash,
        function_name=function_name,
        target=target,
        producer_name=producer_name,
        producer_version=producer_version,
        parent_source_id=parent_source_id,
        parent_ttir_hash=parent_ttir_hash,
        debug_origin=debug_origin,
    )


def validate_native_triton_audit(record: SourceRecord) -> None:
    """Reject production native Triton records that lack auditable source provenance."""

    if record.source_origin != SourceOrigin.NATIVE_TRITON_LANGUAGE:
        return
    if record.artifact_kind == SourceArtifactKind.TRITON_PYTHON_SOURCE and record.source_hash:
        return
    if record.parent_source_id or record.parent_ttir_hash:
        return
    raise ValueError("native Triton language production records require auditable provenance")

