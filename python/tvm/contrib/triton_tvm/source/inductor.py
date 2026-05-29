# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""TorchInductor source adapters."""

from __future__ import annotations

from .normalize import normalize_source_record
from .schema import SourceArtifactKind, SourceOrigin, SourceRecord


def exported_inductor_source(
    *,
    source: str,
    function_name: str | None,
    target: str,
    producer_version: str | None = None,
) -> SourceRecord:
    """Create a SourceRecord for an exported TorchInductor kernel."""

    return normalize_source_record(
        content=source,
        ttir=None,
        source_origin=SourceOrigin.TORCH_INDUCTOR,
        artifact_kind=SourceArtifactKind.EXPORTED_INDUCTOR_KERNEL,
        function_name=function_name,
        target=target,
        producer_name="torch_inductor",
        producer_version=producer_version,
    )


def captured_inductor_jit_kernel(
    *,
    ttir: str,
    function_name: str | None,
    target: str,
    producer_version: str | None = None,
) -> SourceRecord:
    """Create a SourceRecord for captured TorchInductor JIT material."""

    return normalize_source_record(
        content=None,
        ttir=ttir,
        source_origin=SourceOrigin.TORCH_INDUCTOR,
        artifact_kind=SourceArtifactKind.CAPTURED_JIT_KERNEL,
        function_name=function_name,
        target=target,
        producer_name="torch_inductor",
        producer_version=producer_version,
    )

