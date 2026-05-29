# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Source identity records for Triton-TVM inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class SourceOrigin(str, Enum):
    """Where active input material comes from."""

    TORCH_INDUCTOR = "torch_inductor"
    NATIVE_TRITON_LANGUAGE = "native_triton_language"
    TEST_FIXTURE = "test_fixture"


class SourceArtifactKind(str, Enum):
    """The concrete input artifact being normalized."""

    TRITON_PYTHON_SOURCE = "triton_python_source"
    TTIR_MODULE = "ttir_module"
    CAPTURED_JIT_KERNEL = "captured_jit_kernel"
    EXPORTED_INDUCTOR_KERNEL = "exported_inductor_kernel"
    NORMALIZED_TTIR_GRAPH = "normalized_ttir_graph"


@dataclass(frozen=True)
class SourceRecord:
    """Stable source identity.

    This layer does not express operator support, schedules, runtime admission,
    model families, or capability credit.
    """

    source_id: str
    source_origin: SourceOrigin
    artifact_kind: SourceArtifactKind
    source_hash: str | None
    ttir_hash: str | None
    function_name: str | None
    target: str
    producer_name: str | None = None
    producer_version: str | None = None
    parent_source_id: str | None = None
    parent_ttir_hash: str | None = None
    debug_origin: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("SourceRecord.source_id is required")
        if not self.target:
            raise ValueError("SourceRecord.target is required")
        if not isinstance(self.source_origin, SourceOrigin):
            object.__setattr__(self, "source_origin", SourceOrigin(self.source_origin))
        if not isinstance(self.artifact_kind, SourceArtifactKind):
            object.__setattr__(self, "artifact_kind", SourceArtifactKind(self.artifact_kind))
        if self.artifact_kind == SourceArtifactKind.NORMALIZED_TTIR_GRAPH:
            if not self.producer_name or not self.producer_version:
                raise ValueError("normalized TTIR graph records require producer metadata")
            if not (self.parent_source_id or self.parent_ttir_hash):
                raise ValueError("normalized TTIR graph records require parent provenance")
            if not (self.source_hash or self.ttir_hash):
                raise ValueError("normalized TTIR graph records require a stable content hash")


def production_origins() -> tuple[SourceOrigin, SourceOrigin]:
    """Return active production origins."""

    return (SourceOrigin.TORCH_INDUCTOR, SourceOrigin.NATIVE_TRITON_LANGUAGE)


def is_production_origin(origin: SourceOrigin | str) -> bool:
    """Whether an origin may be counted in production capability reports."""

    return SourceOrigin(origin) in production_origins()

