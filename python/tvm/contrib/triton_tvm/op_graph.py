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
"""Normalized TTIR graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TTIRType:
    """Normalized view of a TTIR type used by the Python prototype."""

    raw: str
    dtype: str
    shape: tuple[int, ...] = ()
    is_pointer: bool = False

    @property
    def is_tensor(self) -> bool:
        return bool(self.shape)


@dataclass(frozen=True)
class TTIRParam:
    """A `tt.func` parameter."""

    name: str
    type: TTIRType


@dataclass
class TTIROp:
    """A normalized operation from textual TTIR."""

    name: str
    results: list[str]
    operands: list[str]
    result_types: list[TTIRType]
    attrs: dict[str, Any] = field(default_factory=dict)
    regions: list[list["TTIROp"]] = field(default_factory=list)
    raw: str = ""


@dataclass
class NormalizedTTIROpGraph:
    """A single-function TTIR module in a stable Python representation."""

    function_name: str
    params: list[TTIRParam]
    ops: list[TTIROp]
    attrs: dict[str, Any] = field(default_factory=dict)
    raw_ttir: str = ""

    def op_by_result(self) -> dict[str, TTIROp]:
        """Return the defining op for every SSA result."""
        return {result: op for op in self.ops for result in op.results}

    def param_by_name(self) -> dict[str, TTIRParam]:
        """Return function parameters keyed by their source names."""
        return {param.name: param for param in self.params}
