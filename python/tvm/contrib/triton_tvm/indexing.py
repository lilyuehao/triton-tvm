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
"""Pointwise TTIR index-expression classification.

This module is intentionally internal to the Triton TVM prototype.  It keeps
the M7 pointwise index taxonomy shared between translation and capability
reports, while leaving the public API surface unchanged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .errors import UnsupportedTTIROpError
from .op_graph import NormalizedTTIROpGraph, TTIROp


@dataclass(frozen=True)
class TTIRIndexInfo:
    """A supported index expression rendered for TVMScript and reports."""

    kind: str
    expr: str
    extent_expr: str
    summary_expr: str
    factor: int | None = None


class TTIRIndexClassifier:
    """Classify the Grid1D pointwise index subset used by M7."""

    def __init__(
        self,
        graph: NormalizedTTIROpGraph,
        *,
        block_size: int | None = None,
        extent_expr: str = "N",
        lane_index_ssa: str | None = None,
    ):
        self.graph = graph
        self.defs = graph.op_by_result()
        self.block_size = block_size or self._find_block_size()
        self.extent_expr = extent_expr
        self.lane_index_ssa = lane_index_ssa or self._find_lane_index_ssa()

    def classify(self, value: str) -> TTIRIndexInfo:
        """Return a supported index classification or raise a stable blocker."""
        info = self._classify(value)
        return info

    def addptr_summary(self) -> dict[str, Any]:
        """Return additive report fields for every ``tt.addptr`` offset."""
        entries = []
        kind_counts: Counter[str] = Counter()
        unsupported_count = 0
        for op in [op for op in self.graph.ops if op.name == "tt.addptr"]:
            offset = op.operands[1] if len(op.operands) > 1 else ""
            entry: dict[str, Any] = {
                "results": list(op.results),
                "operands": list(op.operands),
                "offset": offset,
                "supported": False,
                "kind": "unsupported",
                "expr": "",
                "reason": "",
            }
            try:
                info = self.classify(offset)
                entry.update(
                    {
                        "supported": True,
                        "kind": info.kind,
                        "expr": info.summary_expr,
                        "extent_expr": info.extent_expr,
                    }
                )
                kind_counts[info.kind] += 1
            except UnsupportedTTIROpError as err:
                entry["reason"] = str(err)
                unsupported_count += 1
                kind_counts["unsupported"] += 1
            entries.append(entry)

        return {
            "classifier_version": 1,
            "lane_index": self.lane_index_ssa or "",
            "classified_addptrs": entries,
            "index_kinds": dict(sorted(kind_counts.items())),
            "unsupported_index_count": unsupported_count,
        }

    def _classify(self, value: str) -> TTIRIndexInfo:
        if self._is_lane_index_value(value):
            return TTIRIndexInfo(
                kind="flat",
                expr="i",
                extent_expr=self.extent_expr,
                summary_expr="i",
            )

        constant = self._constant_number_value(value)
        if isinstance(constant, int):
            return TTIRIndexInfo(
                kind="constant",
                expr=f"T.int64({constant})",
                extent_expr=f"T.int64({max(constant + 1, 1)})",
                summary_expr=str(constant),
                factor=constant,
            )

        op = self.defs.get(value)
        if op is not None and op.name in (
            "arith.extsi",
            "arith.extui",
            "arith.truncf",
        ) and len(op.operands) == 1:
            source = self._classify(op.operands[0])
            dtype = op.result_types[0].dtype if op.result_types else "int64"
            if dtype == "int64" and source.expr == "i":
                return source
            return TTIRIndexInfo(
                kind=source.kind,
                expr=f'T.Cast("{dtype}", {source.expr})',
                extent_expr=source.extent_expr,
                summary_expr=source.summary_expr,
                factor=source.factor,
            )

        if op is not None and op.name in ("arith.sitofp", "arith.fptosi") and len(op.operands) == 1:
            source = self._classify(op.operands[0])
            dtype = op.result_types[0].dtype if op.result_types else "int64"
            return TTIRIndexInfo(
                kind=f"{op.name.removeprefix('arith.')}_{source.kind}",
                expr=f'T.Cast("{dtype}", {source.expr})',
                extent_expr=source.extent_expr,
                summary_expr=f"{op.name.removeprefix('arith.')}({source.summary_expr})",
                factor=source.factor,
            )

        if op is None or len(op.operands) != 2:
            raise UnsupportedTTIROpError("unsupported composed indexing pattern for pointwise_flat")

        lhs, rhs = op.operands
        if op.name == "arith.addi":
            lhs_info = self._classify(lhs)
            rhs_info = self._classify(rhs)
            return self._combine_add(lhs_info, rhs_info)

        if op.name in ("arith.muli", "arith.mulf"):
            lhs_const = self._constant_number_value(lhs)
            rhs_const = self._constant_number_value(rhs)
            if lhs_const is not None:
                return self._scale_index(self._classify(rhs), lhs_const, op.name)
            if rhs_const is not None:
                return self._scale_index(self._classify(lhs), rhs_const, op.name)
            raise UnsupportedTTIROpError("unsupported composed indexing pattern for pointwise_flat")

        if op.name in ("arith.divsi", "arith.remsi"):
            constant = self._positive_int_constant(rhs)
            lhs_info = self._classify(lhs)
            if op.name == "arith.divsi":
                return TTIRIndexInfo(
                    kind="div",
                    expr=f"({lhs_info.expr} // T.int64({constant}))",
                    extent_expr=f"T.ceildiv({lhs_info.extent_expr}, T.int64({constant}))",
                    summary_expr=f"({lhs_info.summary_expr} // {constant})",
                    factor=constant,
                )
            return TTIRIndexInfo(
                kind="rem",
                expr=f"({lhs_info.expr} % T.int64({constant}))",
                extent_expr=f"T.int64({constant})",
                summary_expr=f"({lhs_info.summary_expr} % {constant})",
                factor=constant,
            )

        if op.name == "arith.divf":
            constant = self._positive_number_constant(rhs)
            lhs_info = self._classify(lhs)
            dtype = op.result_types[0].dtype if op.result_types else "float32"
            return TTIRIndexInfo(
                kind=f"divf_{lhs_info.kind}",
                expr=f"({lhs_info.expr} / {_tir_const(constant, dtype)})",
                extent_expr=lhs_info.extent_expr,
                summary_expr=f"({lhs_info.summary_expr} / {constant:g})",
                factor=lhs_info.factor,
            )

        raise UnsupportedTTIROpError("unsupported composed indexing pattern for pointwise_flat")

    def _combine_add(self, lhs: TTIRIndexInfo, rhs: TTIRIndexInfo) -> TTIRIndexInfo:
        if lhs.kind == "constant":
            return self._offset_index(rhs, lhs.factor or 0)
        if rhs.kind == "constant":
            return self._offset_index(lhs, rhs.factor or 0)
        kind = "affine_combo"
        if {lhs.kind, rhs.kind} == {"div", "rem"}:
            kind = "div_rem_combo"
        elif lhs.kind.startswith("mul") or rhs.kind.startswith("mul"):
            kind = "affine_add"
        return TTIRIndexInfo(
            kind=kind,
            expr=f"({lhs.expr} + {rhs.expr})",
            extent_expr=f"({lhs.extent_expr} + {rhs.extent_expr})",
            summary_expr=f"({lhs.summary_expr} + {rhs.summary_expr})",
        )

    def _offset_index(self, source: TTIRIndexInfo, offset: int) -> TTIRIndexInfo:
        if offset == 0:
            return source
        op = "+" if offset >= 0 else "-"
        abs_offset = abs(offset)
        return TTIRIndexInfo(
            kind="offset" if source.kind == "flat" else f"{source.kind}_offset",
            expr=f"({source.expr} {op} T.int64({abs_offset}))",
            extent_expr=(
                f"({source.extent_expr} + T.int64({offset}))"
                if offset > 0
                else source.extent_expr
            ),
            summary_expr=f"({source.summary_expr} {op} {abs_offset})",
            factor=source.factor,
        )

    def _scale_index(
        self, source: TTIRIndexInfo, factor: int | float, op_name: str
    ) -> TTIRIndexInfo:
        if factor == 0:
            return TTIRIndexInfo(
                kind="constant",
                expr="T.int64(0)",
                extent_expr="T.int64(1)",
                summary_expr="0",
                factor=0,
            )
        if factor < 0:
            raise UnsupportedTTIROpError("negative stride indexing is not supported")
        if isinstance(factor, int):
            return TTIRIndexInfo(
                kind="mul" if source.kind == "flat" else f"{source.kind}_mul",
                expr=f"({source.expr} * T.int64({factor}))",
                extent_expr=f"({source.extent_expr} * T.int64({factor}))",
                summary_expr=f"({source.summary_expr} * {factor})",
                factor=factor,
            )
        dtype = "float32"
        return TTIRIndexInfo(
            kind=f"mulf_{source.kind}",
            expr=f"({source.expr} * {_tir_const(factor, dtype)})",
            extent_expr=source.extent_expr,
            summary_expr=f"({source.summary_expr} * {factor:g})",
            factor=source.factor,
        )

    def _find_block_size(self) -> int | None:
        for op in self.graph.ops:
            if op.name == "tt.make_range":
                start = int(op.attrs.get("start", 0))
                end = int(op.attrs.get("end", 0))
                if end > start:
                    return end - start
        return None

    def _find_lane_index_ssa(self) -> str | None:
        for op in self.graph.ops:
            if op.name != "arith.addi" or not op.results or len(op.operands) != 2:
                continue
            lhs, rhs = op.operands
            if (self._is_pid_block_splat(lhs) and self._is_make_range(rhs)) or (
                self._is_pid_block_splat(rhs) and self._is_make_range(lhs)
            ):
                return op.results[0]
        return None

    def _is_make_range(self, value: str) -> bool:
        op = self.defs.get(value)
        return op is not None and op.name == "tt.make_range"

    def _is_pid_block_splat(self, value: str) -> bool:
        op = self.defs.get(value)
        if op is None or op.name != "tt.splat" or not op.operands:
            return False
        source = self.defs.get(op.operands[0])
        if source is None or source.name != "arith.muli" or len(source.operands) != 2:
            return False
        return any(self._is_program_id(operand) for operand in source.operands) and any(
            self._is_block_size_constant(operand) for operand in source.operands
        )

    def _is_program_id(self, value: str) -> bool:
        op = self.defs.get(value)
        return op is not None and op.name == "tt.get_program_id" and op.attrs.get("axis") == "x"

    def _is_block_size_constant(self, value: str) -> bool:
        constant = self._constant_number_value(value)
        return isinstance(constant, int) and self.block_size is not None and constant == self.block_size

    def _is_lane_index_value(self, value: str) -> bool:
        if value == self.lane_index_ssa:
            return True
        op = self.defs.get(value)
        return (
            op is not None
            and op.name in ("arith.extsi", "arith.extui")
            and len(op.operands) == 1
            and op.operands[0] == self.lane_index_ssa
        )

    def _positive_int_constant(self, value: str) -> int:
        constant = self._constant_number_value(value)
        if not isinstance(constant, int) or constant <= 0:
            raise UnsupportedTTIROpError("pointwise indexed constants must be positive")
        return constant

    def _positive_number_constant(self, value: str) -> int | float:
        constant = self._constant_number_value(value)
        if constant is None or constant <= 0:
            raise UnsupportedTTIROpError("pointwise indexed constants must be positive")
        return constant

    def _constant_number_value(self, value: str) -> int | float | None:
        op = self.defs.get(value)
        if op is None:
            return None
        if op.name == "tt.splat" and len(op.operands) == 1:
            return self._constant_number_value(op.operands[0])
        if op.name != "arith.constant":
            return None
        constant = op.attrs.get("value")
        if isinstance(constant, bool):
            return None
        if isinstance(constant, int):
            return constant
        if isinstance(constant, float):
            if constant.is_integer():
                return int(constant)
            return constant
        return None


def summarize_ttir_indexing(graph: NormalizedTTIROpGraph) -> dict[str, Any]:
    """Return additive M7 classifier fields for capability reports."""
    return TTIRIndexClassifier(graph).addptr_summary()


def _tir_const(value: int | float, dtype: str) -> str:
    if dtype.startswith("float") or dtype == "bfloat16":
        return f"T.{dtype}({float(value)!r})"
    if dtype.startswith("int") or dtype.startswith("uint"):
        return f"T.{dtype}({int(value)})"
    return repr(value)
