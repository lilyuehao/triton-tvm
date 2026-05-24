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
"""Translate normalized TTIR into TIRX."""

from __future__ import annotations

import hashlib
import json
import keyword
import re
import struct
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import tvm

from .contracts import (
    get_triton_tvm_contract,
    normalize_triton_tvm_contract,
    validate_triton_tvm_contract,
)
from .errors import UnsupportedContractError, UnsupportedTargetPolicyError, UnsupportedTTIROpError
from .indexing import TTIRIndexClassifier, TTIRIndexInfo
from .op_graph import NormalizedTTIROpGraph, TTIROp, TTIRType
from .ttir import TTIRInput, normalize_ttir_input


_M25_POINTWISE_OPS = frozenset(
    {
        "arith.addf",
        "arith.addi",
        "arith.cmpi",
        "arith.constant",
        "arith.extsi",
        "arith.extui",
        "arith.mulf",
        "arith.muli",
        "arith.subf",
        "arith.subi",
        "tt.addptr",
        "tt.get_program_id",
        "tt.load",
        "tt.make_range",
        "tt.return",
        "tt.splat",
        "tt.store",
    }
)

_M35_INDEXED_EXTRA_OPS = frozenset(
    {
        "arith.andi",
        "arith.cmpf",
        "arith.divf",
        "arith.divsi",
        "arith.extf",
        "arith.fptosi",
        "arith.fptoui",
        "arith.ori",
        "arith.remsi",
        "arith.select",
        "arith.sitofp",
        "arith.truncf",
        "math.absf",
        "math.cos",
        "math.exp",
        "math.log",
        "math.sin",
        "tt.bitcast",
        "tt.extern_elementwise",
        "tt.precise_sqrt",
    }
)

_SUPPORTED_OPS_BY_CONTRACT = {
    "pointwise_minimal": _M25_POINTWISE_OPS,
    "pointwise_flat": _M25_POINTWISE_OPS | _M35_INDEXED_EXTRA_OPS,
    "reduction_minimal": frozenset(
        {
            "arith.addf",
            "arith.addi",
            "arith.cmpi",
            "arith.constant",
            "arith.divf",
            "arith.extf",
            "arith.extsi",
            "arith.extui",
            "arith.mulf",
            "arith.muli",
            "arith.select",
            "arith.sitofp",
            "arith.subf",
            "arith.truncf",
            "math.rsqrt",
            "tt.addptr",
            "tt.get_program_id",
            "tt.load",
            "tt.make_range",
            "tt.reduce",
            "tt.reduce.return",
            "tt.return",
            "tt.splat",
            "tt.store",
        }
    ),
}
_SUPPORTED_OPS_BY_CONTRACT["norm_single_row"] = _SUPPORTED_OPS_BY_CONTRACT[
    "reduction_minimal"
]
_M8_RANK2_OPS = _SUPPORTED_OPS_BY_CONTRACT["reduction_minimal"] | frozenset(
    {
        "arith.andi",
        "arith.cmpf",
        "arith.divsi",
        "arith.ori",
        "arith.remsi",
        "math.exp",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.extern_elementwise",
    }
)
_SUPPORTED_OPS_BY_CONTRACT["row_reduction"] = _M8_RANK2_OPS
_SUPPORTED_OPS_BY_CONTRACT["norm_row"] = _M8_RANK2_OPS
_SUPPORTED_OPS_BY_CONTRACT["softmax_row"] = _M8_RANK2_OPS
_SUPPORTED_OPS_BY_CONTRACT["masked_softmax_row"] = _M8_RANK2_OPS

_BINARY_OP_SYMBOLS = {
    "arith.addf": "+",
    "arith.addi": "+",
    "arith.andi": "and",
    "arith.divf": "/",
    "arith.divsi": "//",
    "arith.mulf": "*",
    "arith.muli": "*",
    "arith.ori": "or",
    "arith.remsi": "%",
    "arith.subf": "-",
    "arith.subi": "-",
}

_TRANSLATOR_VERSION = "triton_tvm_python_pre_m5_contracts_v1"
_CUDA_TARGET_POLICY_VERSION = "cuda_thread_binding_v1"
_CACHE_POLICY = "disabled"
_M8_ROW_CONTRACTS = frozenset(
    {"row_reduction", "norm_row", "softmax_row", "masked_softmax_row"}
)


@dataclass(frozen=True)
class _ExtentInfo:
    kind: str
    expr: str
    param_name: str = ""
    value: int | None = None


_NOT_APPLICABLE_REDUCTION_EXTENT = _ExtentInfo(kind="not_applicable", expr="")


@dataclass(frozen=True)
class _PointwiseTargetPolicy:
    target: str
    target_kind: str
    launch_policy_id: str
    version: str
    target_attrs: str
    supported_contracts: frozenset[str]
    block_var: str
    lane_var: str
    block_thread_tag: str
    lane_thread_tag: str

    def program_id_expr(self, axis: str) -> str:
        if axis != "x":
            raise UnsupportedTTIROpError(
                f"tt.get_program_id axis {axis!r} is not supported by "
                f"{self.launch_policy_id}"
            )
        return self.block_var

    def lane_expr(self, start: int) -> str:
        lane = f'T.Cast("int64", {self.lane_var})'
        if start:
            return f"(T.int64({start}) + {lane})"
        return lane


@dataclass(frozen=True)
class TritonTVMMeta:
    """Metadata carried through build/runtime for a translated Triton kernel."""

    kernel_name: str
    signature: dict[str, str]
    constexprs: dict[str, Any]
    grid: Any
    target: str
    target_kind: str
    contract: str
    canonical_contract: str
    requested_contract: str
    emit: str
    translator_version: str
    contract_version: str
    target_policy_version: str
    target_attrs: str
    triton_version: str
    tvm_version: str
    ttir_hash: str
    source_hash: str
    extent_param: str
    extent_kind: str
    extent_value: int | None
    reduction_extent_param: str
    reduction_extent_kind: str
    reduction_extent_value: int | None
    buffer_extents: dict[str, str]
    block_size: int
    indexing_kind: str
    execution_kind: str
    accumulator_dtype_policy: str
    epsilon_policy: str
    mask_policy: str
    axis_policy: str
    layout_policy: str
    launch_policy_id: str
    abi: list[dict[str, str]]
    cache_policy: str
    disk_cache_enabled: bool
    fallback_reason: str
    cache_key: str


def translate_ttir(
    ttir_or_graph: TTIRInput,
    *,
    grid,
    target: str = "cuda",
    emit: str = "tirx",
    contract: str = "pointwise_minimal",
) -> tuple[tvm.IRModule, TritonTVMMeta]:
    """Translate normalized TTIR into a contract-driven TIRX IRModule.

    ``str`` and ``TTIRArtifact`` inputs are accepted for the public convenience
    API, but textual parsing is delegated to ``ttir.normalize_ttir_input``.
    Unsupported TTIR, target policies, contracts, and stream/cache semantics are
    reported explicitly; this path never silently falls back to native Triton.
    """
    if emit != "tirx":
        raise ValueError(f"Only emit='tirx' is supported by this prototype, got {emit!r}")
    canonical_contract = normalize_triton_tvm_contract(contract)
    contract_spec = get_triton_tvm_contract(canonical_contract)
    target_policy = _pointwise_target_policy(target)
    _validate_policy_contract_support(target_policy, canonical_contract)

    graph, artifact = normalize_ttir_input(ttir_or_graph)
    _validate_supported_subset(graph, canonical_contract)

    if canonical_contract in _M8_ROW_CONTRACTS:
        builder = _TIRXRank2RowBuilder(graph, target_policy, canonical_contract)
    elif canonical_contract in ("reduction_minimal", "norm_single_row"):
        builder = _TIRXReductionBuilder(graph, target_policy, grid)
    else:
        builder = _TIRXTemplateBuilder(graph, target_policy, canonical_contract)
    source = builder.build_source()
    from tvm.script import ir as I  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    irmod = tvm.script.from_source(source, {"I": I, "T": T})
    validate_triton_tvm_contract(irmod, canonical_contract)

    ttir_text = graph.raw_ttir
    signature = artifact.signature if artifact is not None else _signature_from_graph(graph)
    constexprs = artifact.constexprs if artifact is not None else {}
    triton_version = artifact.triton_version if artifact is not None else _current_triton_version()
    source_hash = artifact.source_hash if artifact is not None else ""
    ttir_hash = hashlib.sha256(ttir_text.encode("utf-8")).hexdigest()
    abi = _abi_from_graph(graph)
    cache_key = _cache_key(
        kernel_name=graph.function_name,
        signature=signature,
        constexprs=constexprs,
        target=target_policy.target,
        target_kind=target_policy.target_kind,
        target_policy=target_policy.launch_policy_id,
        canonical_contract=canonical_contract,
        emit=emit,
        translator_version=_TRANSLATOR_VERSION,
        contract_version=contract_spec.version,
        target_policy_version=target_policy.version,
        tvm_version=tvm.__version__,
        triton_version=triton_version,
        ttir_hash=ttir_hash,
        source_hash=source_hash,
        abi=abi,
        extent_param=builder.extent_info.param_name,
        extent_kind=builder.extent_info.kind,
        extent_value=builder.extent_info.value,
        reduction_extent_param=builder.reduction_extent_info.param_name,
        reduction_extent_kind=builder.reduction_extent_info.kind,
        reduction_extent_value=builder.reduction_extent_info.value,
        buffer_extents=builder.buffer_extents,
        block_size=builder.block_size,
        indexing_kind=contract_spec.indexing_kind,
        execution_kind=contract_spec.execution_kind,
        accumulator_dtype_policy=contract_spec.accumulator_dtype_policy,
        epsilon_policy=contract_spec.epsilon_policy,
        mask_policy=contract_spec.mask_policy,
        axis_policy=contract_spec.axis_policy,
        layout_policy=contract_spec.layout_policy,
        launch_policy_id=target_policy.launch_policy_id,
        target_attrs=target_policy.target_attrs,
        cache_policy=_CACHE_POLICY,
    )
    meta = TritonTVMMeta(
        kernel_name=graph.function_name,
        signature=signature,
        constexprs=constexprs,
        grid=grid,
        target=target_policy.target,
        target_kind=target_policy.target_kind,
        contract=canonical_contract,
        canonical_contract=canonical_contract,
        requested_contract=canonical_contract,
        emit=emit,
        translator_version=_TRANSLATOR_VERSION,
        contract_version=contract_spec.version,
        target_policy_version=target_policy.version,
        target_attrs=target_policy.target_attrs,
        triton_version=triton_version,
        tvm_version=tvm.__version__,
        ttir_hash=ttir_hash,
        source_hash=source_hash,
        extent_param=builder.extent_info.param_name,
        extent_kind=builder.extent_info.kind,
        extent_value=builder.extent_info.value,
        reduction_extent_param=builder.reduction_extent_info.param_name,
        reduction_extent_kind=builder.reduction_extent_info.kind,
        reduction_extent_value=builder.reduction_extent_info.value,
        buffer_extents=builder.buffer_extents,
        block_size=builder.block_size,
        indexing_kind=contract_spec.indexing_kind,
        execution_kind=contract_spec.execution_kind,
        accumulator_dtype_policy=contract_spec.accumulator_dtype_policy,
        epsilon_policy=contract_spec.epsilon_policy,
        mask_policy=contract_spec.mask_policy,
        axis_policy=contract_spec.axis_policy,
        layout_policy=contract_spec.layout_policy,
        launch_policy_id=target_policy.launch_policy_id,
        abi=abi,
        cache_policy=_CACHE_POLICY,
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=cache_key,
    )
    return irmod, meta


class _TIRXTemplateBuilder:
    def __init__(
        self,
        graph: NormalizedTTIROpGraph,
        target_policy: _PointwiseTargetPolicy,
        contract: str,
    ):
        self.graph = graph
        self.target_policy = target_policy
        self.contract = contract
        self.defs = graph.op_by_result()
        self.params = graph.param_by_name()
        self.names = {param.name: _sanitize_identifier(param.name) for param in graph.params}
        self.ptr_params = [param for param in graph.params if param.type.is_pointer]
        self.scalar_params = [param for param in graph.params if not param.type.is_pointer]
        self.block_size = self._find_block_size()
        self.stores = self._find_stores()
        self.lane_index_ssa = self._find_lane_index_ssa()
        self._validate_flat_mask_policy()
        self.extent_info = self._find_extent_info()
        self.reduction_extent_info = _NOT_APPLICABLE_REDUCTION_EXTENT
        self.index_classifier = TTIRIndexClassifier(
            self.graph,
            block_size=self.block_size,
            extent_expr=self.extent_info.expr,
            lane_index_ssa=self.lane_index_ssa,
        )
        self.buffer_extents = self._infer_buffer_extents()
        self.safe_no_other_loads = self._prove_safe_masked_loads_without_other()
        self.aliases: dict[str, str] = {}
        if self.lane_index_ssa:
            self.aliases[self.lane_index_ssa] = "i"

    def build_source(self) -> str:
        extent = self.extent_info.expr

        mask_values = self._collect_mask_values()
        mask_defs = [(mask, self.expr(mask)) for mask in mask_values]
        mask_alias_by_expr: dict[str, str] = {}
        for i, (mask, _) in enumerate(mask_defs):
            mask_expr = mask_defs[i][1]
            if mask_expr in mask_alias_by_expr:
                self.aliases[mask] = mask_alias_by_expr[mask_expr]
                continue
            alias = "mask" if not mask_alias_by_expr else f"mask_{len(mask_alias_by_expr)}"
            mask_alias_by_expr[mask_expr] = alias
            self.aliases[mask] = alias

        lines: list[str] = [
            "# from tvm.script import ir as I",
            "# from tvm.script import tirx as T",
            "",
            "@I.ir_module",
            "class Module:",
            "    @T.prim_func",
            f"    def {_sanitize_identifier(self.graph.function_name)}({self._param_signature()}):",
            "        T.func_attr({"
            f'"global_symbol": "{self.graph.function_name}", '
            '"tirx.noalias": True, '
            f'"target": T.target({self.target_policy.target_attrs!r})'
            "})",
        ]
        for param in self.ptr_params:
            name = self.names[param.name]
            buffer_extent = self.buffer_extents.get(param.name, extent)
            lines.append(
                "        "
                f'{name} = T.match_buffer({name}_handle, ({buffer_extent},), '
                f'"{param.type.dtype}")'
            )

        lines.extend(
            [
                "        "
                f"for {self.target_policy.block_var} in T.thread_binding("
                f"0, T.ceildiv({extent}, T.int64({self.block_size})), "
                f'thread="{self.target_policy.block_thread_tag}"):',
                "            "
                f"for {self.target_policy.lane_var} in T.thread_binding(0, {self.block_size}, "
                f'thread="{self.target_policy.lane_thread_tag}"):',
                "                "
                f"i = {self.target_policy.block_var} * T.int64({self.block_size}) + "
                f'T.Cast("int64", {self.target_policy.lane_var})',
            ]
        )
        emitted_mask_aliases: set[str] = set()
        for mask, mask_expr in mask_defs:
            alias = self.aliases[mask]
            if alias in emitted_mask_aliases:
                continue
            emitted_mask_aliases.add(alias)
            lines.append(f"                {alias} = {mask_expr}")
        for store in self.stores:
            store_ptr, store_value = store.operands[0], store.operands[1]
            out_buffer, out_index = self.pointer_ref(store_ptr, allow_store_bitcast=True)
            value_expr = self.store_value_expr(store)
            store_mask = store.operands[2] if len(store.operands) >= 3 else None
            mask_expr = (
                self.aliases.get(store_mask, self.expr(store_mask))
                if store_mask is not None
                else mask_alias_by_expr.get(f"(i < {extent})", f"(i < {extent})")
            )
            lines.append(f"                if {mask_expr}:")
            lines.append(f"                    {out_buffer}[{out_index}] = {value_expr}")
        return "\n".join(lines) + "\n"

    def expr(self, value: str) -> str:
        if value in self.aliases:
            return self.aliases[value]
        if value in self.params and not self.params[value].type.is_pointer:
            return self.names[value]
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR value %{value}")

        op = self.defs[value]
        if op.name == "arith.constant":
            ty = op.result_types[0] if op.result_types else TTIRType("i64", "int64")
            dtype = ty.dtype
            if dtype.startswith("int") or dtype.startswith("uint"):
                dtype = "int64"
            return _tir_const(op.attrs.get("value", 0), dtype)
        if op.name == "tt.get_program_id":
            axis = op.attrs.get("axis")
            return self.target_policy.program_id_expr(axis)
        if op.name == "tt.make_range":
            start = int(op.attrs.get("start", 0))
            return self.target_policy.lane_expr(start)
        if op.name == "tt.splat":
            return self.expr(op.operands[0])
        if op.name in (
            "arith.extf",
            "arith.extsi",
            "arith.extui",
            "arith.fptosi",
            "arith.fptoui",
            "arith.sitofp",
            "arith.truncf",
        ):
            source = self.expr(op.operands[0])
            dtype = op.result_types[0].dtype if op.result_types else "int64"
            if source == "i":
                return "i"
            return f'T.Cast("{dtype}", {source})'
        if op.name in _BINARY_OP_SYMBOLS:
            self._validate_binary_op(op)
            lhs = self.expr(op.operands[0])
            rhs = self.expr(op.operands[1])
            symbol = _BINARY_OP_SYMBOLS[op.name]
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.cmpi":
            lhs = self.expr(op.operands[0])
            rhs = self.expr(op.operands[1])
            if self._is_lane_index_value(op.operands[0]):
                bound = self._extent_bound(op.operands[1])
                if bound is not None:
                    rhs = bound.expr
            elif self._is_lane_index_value(op.operands[1]):
                bound = self._extent_bound(op.operands[0])
                if bound is not None:
                    lhs = bound.expr
            symbol = _cmp_symbol(op.attrs.get("predicate", ""))
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.cmpf":
            lhs = self.expr(op.operands[0])
            rhs = self.expr(op.operands[1])
            symbol = _cmpf_symbol(op.attrs.get("predicate", ""))
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.select":
            cond = self.expr(op.operands[0])
            true_value = self.expr(op.operands[1])
            false_value = self.expr(op.operands[2])
            return f"T.Select({cond}, {true_value}, {false_value})"
        if op.name in ("math.absf", "math.cos", "math.exp", "math.log", "math.sin"):
            source = self.expr(op.operands[0])
            tir_op = {
                "math.absf": "abs",
                "math.cos": "cos",
                "math.exp": "exp",
                "math.log": "log",
                "math.sin": "sin",
            }[op.name]
            return f"T.{tir_op}({source})"
        if op.name == "tt.precise_sqrt":
            source = self.expr(op.operands[0])
            return f"T.sqrt({source})"
        if op.name == "tt.extern_elementwise":
            source = self.expr(op.operands[0])
            symbol = _unquote_attr(op.attrs.get("symbol", ""))
            mapping = {
                "__nv_erff": "erf",
                "__nv_expf": "exp",
                "__nv_tanhf": "tanh",
                "__nv_rsqrtf": "rsqrt",
            }
            if symbol not in mapping:
                raise UnsupportedTTIROpError(
                    f"tt.extern_elementwise symbol {symbol!r} is not supported yet"
                )
            return f"T.{mapping[symbol]}({source})"
        if op.name == "tt.load":
            if len(op.operands) == 1:
                buffer_name, index = self.pointer_ref(op.operands[0])
                return f"{buffer_name}[{index}]"
            if len(op.operands) < 3:
                result = op.results[0] if op.results else value
                if result not in self.safe_no_other_loads:
                    raise UnsupportedTTIROpError(
                        "unsafe masked tt.load without other is not supported"
                    )
                buffer_name, index = self.pointer_ref(op.operands[0])
                return f"{buffer_name}[{index}]"
            buffer_name, index = self.pointer_ref(op.operands[0])
            mask = self.expr(op.operands[1])
            other = self.expr(op.operands[2])
            return f"T.if_then_else({mask}, {buffer_name}[{index}], {other})"
        if op.name == "tt.bitcast":
            raise UnsupportedTTIROpError("tt.bitcast is only supported for observed bool stores")

        raise UnsupportedTTIROpError(f"{op.name} is not supported yet")

    def store_value_expr(self, store: TTIROp) -> str:
        value = store.operands[1]
        if self._is_observed_bool_bitcast_store(store):
            value_op = self.defs[value]
            return self.expr(value_op.operands[0])
        return self.expr(value)

    def pointer_ref(self, value: str, *, allow_store_bitcast: bool = False) -> tuple[str, str]:
        if value in self.params and self.params[value].type.is_pointer:
            return self.names[value], "0"
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR pointer %{value}")
        op = self.defs[value]
        if op.name == "tt.splat":
            return self.pointer_ref(op.operands[0], allow_store_bitcast=allow_store_bitcast)
        if op.name == "tt.bitcast":
            if not allow_store_bitcast or not self._is_observed_bool_pointer_bitcast(op):
                raise UnsupportedTTIROpError(
                    "unsupported tt.bitcast pointer pattern in pointwise store"
                )
            return self.pointer_ref(op.operands[0], allow_store_bitcast=allow_store_bitcast)
        if op.name == "tt.addptr":
            base, base_index = self.pointer_ref(
                op.operands[0], allow_store_bitcast=allow_store_bitcast
            )
            if self.lane_index_ssa is None:
                raise UnsupportedTTIROpError(
                    "non-contiguous pointer pattern is not supported yet"
                )
            if self._allows_indexed_pointwise_capability():
                offset = self._index_info(op.operands[1]).expr
            else:
                if op.operands[1] != self.lane_index_ssa:
                    raise UnsupportedTTIROpError(
                        "non-contiguous pointer pattern is not supported yet"
                    )
                offset = self.expr(op.operands[1])
            if base_index == "0":
                return base, offset
            return base, f"({base_index} + {offset})"
        raise UnsupportedTTIROpError(f"{op.name} cannot be used as a pointwise pointer")

    def _param_signature(self) -> str:
        items: list[str] = []
        for param in self.graph.params:
            name = self.names[param.name]
            if param.type.is_pointer:
                items.append(f"{name}_handle: T.handle")
            else:
                items.append(f"{name}: T.{param.type.dtype}")
        return ", ".join(items)

    def _find_block_size(self) -> int:
        for op in self.graph.ops:
            if op.name == "tt.make_range":
                start = int(op.attrs.get("start", 0))
                end = int(op.attrs.get("end", 0))
                if end <= start:
                    raise UnsupportedTTIROpError("tt.make_range must have positive extent")
                if start != 0:
                    raise UnsupportedTTIROpError(
                        "tt.make_range with non-zero start is not supported yet"
                    )
                return end - start
        raise UnsupportedTTIROpError("pointwise contracts require tt.make_range to determine BLOCK")

    def _find_stores(self) -> list[TTIROp]:
        stores = [op for op in self.graph.ops if op.name == "tt.store"]
        if len(stores) > 1:
            if self.contract == "pointwise_minimal":
                raise UnsupportedTTIROpError(
                    "multiple tt.store requires contract='pointwise_flat'"
                )
        if not stores:
            raise UnsupportedTTIROpError("pointwise contracts require at least one tt.store")
        for store in stores:
            if len(store.operands) < 2:
                raise UnsupportedTTIROpError("tt.store requires pointer and value operands")
        return stores

    def _find_extent_info(self) -> _ExtentInfo:
        candidates = self._extent_candidates_from_masks(self._collect_store_mask_values())
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise UnsupportedTTIROpError(
                "ambiguous extent candidates from store mask pattern: "
                + ", ".join(candidate.expr for candidate in candidates)
            )

        candidates = self._extent_candidates_from_masks(self._collect_mask_values())
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise UnsupportedTTIROpError(
                "ambiguous extent candidates from mask/index pattern: "
                + ", ".join(candidate.expr for candidate in candidates)
            )

        fallback = self._fallback_extent_info()
        if fallback is not None:
            return fallback
        raise UnsupportedTTIROpError(
            "pointwise contracts require a unique extent from "
            "the mask/index pattern"
        )

    def _extent_candidates_from_masks(self, masks: list[str]) -> list[_ExtentInfo]:
        candidates: list[_ExtentInfo] = []
        for mask in masks:
            op = self.defs.get(mask)
            if op is None or op.name != "arith.cmpi" or len(op.operands) != 2:
                continue
            lhs, rhs = op.operands
            predicate = op.attrs.get("predicate", "")
            candidate = None
            if predicate in ("slt", "ult", "sle", "ule") and self._is_lane_index_value(lhs):
                candidate = self._extent_bound(rhs)
            elif predicate in ("sgt", "ugt", "sge", "uge") and self._is_lane_index_value(rhs):
                candidate = self._extent_bound(lhs)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _fallback_extent_info(self) -> _ExtentInfo | None:
        scalar_int_params = [
            param
            for param in self.scalar_params
            if param.type.dtype.startswith(("int", "uint"))
        ]
        numel_params = [
            param
            for param in scalar_int_params
            if "numel" in param.name or param.name in ("n", "N")
        ]
        candidates = numel_params or scalar_int_params
        if len(candidates) != 1:
            return None
        param = candidates[0]
        name = self.names[param.name]
        expr = name if param.type.dtype == "int64" else f'T.Cast("int64", {name})'
        return _ExtentInfo(kind="runtime_param", expr=expr, param_name=param.name)

    def _extent_bound(self, value: str) -> _ExtentInfo | None:
        param = self._scalar_splat_param(value)
        if param is not None:
            dtype = self.params[param].type.dtype
            name = self.names[param]
            expr = name if dtype == "int64" else f'T.Cast("int64", {name})'
            return _ExtentInfo(kind="runtime_param", expr=expr, param_name=param)
        constant = self._constant_int_value(value)
        if constant is not None:
            if constant <= 0:
                raise UnsupportedTTIROpError("pointwise extent constant must be positive")
            return _ExtentInfo(
                kind="constant",
                expr=f"T.int64({constant})",
                value=constant,
            )
        return None

    def _infer_buffer_extents(self) -> dict[str, str]:
        extents = {param.name: self.extent_info.expr for param in self.ptr_params}
        inferred: dict[str, str] = {}
        for op in self.graph.ops:
            if op.name != "tt.addptr" or len(op.operands) != 2:
                continue
            base = self._base_pointer_param(op.operands[0])
            if base is None:
                continue
            index_info = self._index_info(op.operands[1])
            previous = inferred.get(base)
            if previous is not None and previous != index_info.extent_expr:
                merged = f"({previous} + {index_info.extent_expr})"
                inferred[base] = merged
                extents[base] = merged
                continue
            inferred[base] = index_info.extent_expr
            extents[base] = index_info.extent_expr
        return extents

    def _index_info(self, value: str) -> TTIRIndexInfo:
        if not self._allows_indexed_pointwise_capability():
            if self._is_lane_index_value(value):
                return TTIRIndexInfo(
                    kind="flat",
                    expr="i",
                    extent_expr=self.extent_info.expr,
                    summary_expr="i",
                )
            raise UnsupportedTTIROpError("non-contiguous pointer pattern is not supported yet")
        return self.index_classifier.classify(value)

    def _prove_safe_masked_loads_without_other(self) -> set[str]:
        loads = [
            op
            for op in self.graph.ops
            if op.name == "tt.load" and len(op.operands) == 2 and op.results
        ]
        if not loads:
            return set()
        if not self._allows_indexed_pointwise_capability():
            return set()

        store_masks = set(self._collect_store_mask_values())
        if not store_masks:
            raise UnsupportedTTIROpError(
                "masked tt.load without other escapes the guarded store value"
            )
        for load in loads:
            if len(load.operands) < 2 or load.operands[1] not in store_masks:
                raise UnsupportedTTIROpError(
                    "masked tt.load without other must use a store-dominating mask"
                )

        uses: dict[str, list[tuple[TTIROp, int]]] = {}
        for op in self.graph.ops:
            for i, operand in enumerate(op.operands):
                uses.setdefault(operand, []).append((op, i))

        safe: set[str] = set()
        for load in loads:
            root = load.results[0]
            load_mask = load.operands[1]
            seen: set[str] = set()
            stack = [root]
            while stack:
                value = stack.pop()
                if value in seen:
                    continue
                seen.add(value)
                for use_op, operand_index in uses.get(value, []):
                    if use_op.name == "tt.store":
                        if (
                            operand_index == 1
                            and len(use_op.operands) >= 3
                            and use_op.operands[2] == load_mask
                        ):
                            continue
                        raise UnsupportedTTIROpError(
                            "masked tt.load without other escapes the guarded store value"
                        )
                    if use_op.name in ("tt.addptr", "tt.load"):
                        raise UnsupportedTTIROpError(
                            "masked tt.load without other cannot feed pointer or load operands"
                        )
                    if use_op.name == "tt.return":
                        raise UnsupportedTTIROpError(
                            "masked tt.load without other cannot escape through tt.return"
                        )
                    if (
                        use_op.name not in _SUPPORTED_OPS_BY_CONTRACT["pointwise_flat"]
                        or not use_op.results
                    ):
                        raise UnsupportedTTIROpError(
                            "masked tt.load without other has an unsupported use"
                        )
                    stack.extend(use_op.results)
            safe.add(root)
        return safe

    def _validate_flat_mask_policy(self) -> None:
        masked_no_other_loads = [
            op for op in self.graph.ops if op.name == "tt.load" and len(op.operands) == 2
        ]
        if not masked_no_other_loads:
            return
        store_masks = set(self._collect_store_mask_values())
        if not store_masks:
            return
        for load in masked_no_other_loads:
            if load.operands[1] not in store_masks:
                raise UnsupportedTTIROpError(
                    "masked tt.load without other requires load/store masks to use "
                    "the same flat extent predicate"
                )

    def _find_lane_index_ssa(self) -> str | None:
        for op in self.graph.ops:
            if op.name != "arith.addi" or not op.results:
                continue
            if len(op.operands) != 2:
                continue
            lhs, rhs = op.operands
            if (self._is_pid_block_splat(lhs) and self._is_make_range(rhs)) or (
                self._is_pid_block_splat(rhs) and self._is_make_range(lhs)
            ):
                return op.results[0]
        return None

    def _allows_indexed_pointwise_capability(self) -> bool:
        return self.contract == "pointwise_flat"

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
        op = self.defs.get(value)
        return (
            op is not None
            and op.name == "arith.constant"
            and int(op.attrs.get("value", -1)) == self.block_size
        )

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

    def _scalar_splat_param(self, value: str) -> str | None:
        if value in self.params and not self.params[value].type.is_pointer:
            return value
        op = self.defs.get(value)
        if op is None or op.name != "tt.splat" or len(op.operands) != 1:
            return None
        source = op.operands[0]
        if source in self.params and not self.params[source].type.is_pointer:
            return source
        return None

    def _constant_int_value(self, value: str) -> int | None:
        op = self.defs.get(value)
        if op is None:
            return None
        if op.name == "tt.splat" and len(op.operands) == 1:
            return self._constant_int_value(op.operands[0])
        if op.name != "arith.constant":
            return None
        constant = op.attrs.get("value")
        if isinstance(constant, bool):
            return None
        if isinstance(constant, int):
            return constant
        if isinstance(constant, float) and constant.is_integer():
            return int(constant)
        return None

    def _positive_constant(self, value: str) -> int | None:
        constant = self._constant_int_value(value)
        if constant is None:
            return None
        if constant <= 0:
            raise UnsupportedTTIROpError("pointwise indexed constants must be positive")
        return constant

    def _base_pointer_param(self, value: str) -> str | None:
        if value in self.params and self.params[value].type.is_pointer:
            return value
        op = self.defs.get(value)
        if op is None or not op.operands:
            return None
        if op.name in ("tt.addptr", "tt.bitcast", "tt.splat"):
            return self._base_pointer_param(op.operands[0])
        return None

    def _is_observed_bool_pointer_bitcast(self, op: TTIROp) -> bool:
        if op.name != "tt.bitcast" or not op.result_types or not op.operands:
            return False
        source_op = self.defs.get(op.operands[0])
        if source_op is None or not source_op.result_types:
            return False
        source_type = source_op.result_types[0]
        result_type = op.result_types[0]
        return (
            source_type.is_pointer
            and result_type.is_pointer
            and source_type.dtype == "bool"
            and result_type.dtype in ("int8", "uint8")
        )

    def _is_observed_bool_bitcast_store(self, store: TTIROp) -> bool:
        if len(store.operands) < 2:
            return False
        ptr_op = self.defs.get(store.operands[0])
        value_op = self.defs.get(store.operands[1])
        if ptr_op is None or value_op is None:
            return False
        if not self._is_observed_bool_pointer_bitcast(ptr_op):
            return False
        if value_op.name != "arith.extui" or not value_op.operands:
            return False
        source_op = self.defs.get(value_op.operands[0])
        return (
            source_op is not None
            and bool(source_op.result_types)
            and source_op.result_types[0].dtype == "bool"
        )

    def _validate_binary_op(self, op: TTIROp) -> None:
        if op.name not in ("arith.andi", "arith.ori"):
            return
        dtype = op.result_types[0].dtype if op.result_types else ""
        if dtype != "bool":
            raise UnsupportedTTIROpError(f"{op.name} is only supported for bool tensors")

    def _collect_mask_values(self) -> list[str]:
        masks: list[str] = []
        for op in self.graph.ops:
            mask = None
            if op.name == "tt.load" and len(op.operands) >= 2:
                mask = op.operands[1]
            elif op.name == "tt.store" and len(op.operands) >= 3:
                mask = op.operands[2]
            if mask is not None and mask not in masks:
                masks.append(mask)
        return masks

    def _collect_store_mask_values(self) -> list[str]:
        masks: list[str] = []
        for op in self.graph.ops:
            if op.name == "tt.store" and len(op.operands) >= 3:
                mask = op.operands[2]
                if mask not in masks:
                    masks.append(mask)
        return masks


class _TIRXReductionBuilder:
    def __init__(
        self,
        graph: NormalizedTTIROpGraph,
        target_policy: _PointwiseTargetPolicy,
        grid,
    ):
        self.graph = graph
        self.target_policy = target_policy
        self.row_count = _reduction_grid_extent(grid)
        self.defs = graph.op_by_result()
        self.params = graph.param_by_name()
        self.names = {param.name: _sanitize_identifier(param.name) for param in graph.params}
        self.ptr_params = [param for param in graph.params if param.type.is_pointer]
        self.scalar_params = [param for param in graph.params if not param.type.is_pointer]
        self.block_size = self._find_block_size()
        self.lane_ssa = self._find_lane_ssa()
        self.stores = self._find_stores()
        self.reductions = self._find_reductions()
        self._validate_reductions()
        self._validate_memory_masks()
        self.extent_info = self._find_extent_info()
        self.reduction_extent_info = self.extent_info
        self.buffer_extents = self._infer_buffer_extents()
        self.reduction_aliases: dict[str, str] = {}

    def build_source(self) -> str:
        lines: list[str] = [
            "# from tvm.script import ir as I",
            "# from tvm.script import tirx as T",
            "",
            "@I.ir_module",
            "class Module:",
            "    @T.prim_func",
            f"    def {_sanitize_identifier(self.graph.function_name)}({self._param_signature()}):",
            "        T.func_attr({"
            f'"global_symbol": "{self.graph.function_name}", '
            '"tirx.noalias": True, '
            f'"target": T.target({self.target_policy.target_attrs!r})'
            "})",
        ]
        for param in self.ptr_params:
            name = self.names[param.name]
            buffer_extent = self.buffer_extents.get(param.name, self.extent_info.expr)
            lines.append(
                "        "
                f'{name} = T.match_buffer({name}_handle, ({buffer_extent},), '
                f'"{param.type.dtype}")'
            )
        for i, reduce_op in enumerate(self.reductions):
            dtype = self._value_dtype(reduce_op.results[0])
            lines.append(
                f'        red_{i} = T.alloc_buffer((1,), "{dtype}", scope="local")'
            )

        lines.extend(
            [
                "        "
                f"for {self.target_policy.block_var} in T.thread_binding("
                f"0, T.int64({self.row_count}), "
                f'thread="{self.target_policy.block_thread_tag}"):',
                "            "
                f"for {self.target_policy.lane_var} in T.thread_binding(0, {self.block_size}, "
                f'thread="{self.target_policy.lane_thread_tag}"):',
                "                "
                f"row = T.Cast(\"int64\", {self.target_policy.block_var})",
                "                "
                f"i = T.Cast(\"int64\", {self.target_policy.lane_var})",
                "                if i == T.int64(0):",
            ]
        )
        for i, reduce_op in enumerate(self.reductions):
            acc = f"red_{i}"
            dtype = self._value_dtype(reduce_op.results[0])
            lines.append(f"                    {acc}[0] = {_tir_const(0, dtype)}")
            lines.append(f"                    for rk in T.serial(0, {self.block_size}):")
            value_expr = self.expr(reduce_op.operands[0], lane_var="rk")
            lines.append(f"                        {acc}[0] = {acc}[0] + {value_expr}")
            self.reduction_aliases[reduce_op.results[0]] = f"{acc}[0]"

        for store in self.stores:
            store_ptr = store.operands[0]
            store_mask = store.operands[2] if len(store.operands) >= 3 else None
            if store_mask is not None:
                lines.append(f"                    for sk in T.serial(0, {self.block_size}):")
                out_buffer, out_index = self.pointer_ref(store_ptr, lane_var="sk")
                value_expr = self.expr(store.operands[1], lane_var="sk")
                mask_expr = self.expr(store_mask, lane_var="sk")
                lines.append(f"                        if {mask_expr}:")
                lines.append(f"                            {out_buffer}[{out_index}] = {value_expr}")
            else:
                out_buffer, out_index = self.pointer_ref(store_ptr, lane_var="i")
                value_expr = self.expr(store.operands[1], lane_var="i")
                lines.append(f"                    {out_buffer}[{out_index}] = {value_expr}")
        return "\n".join(lines) + "\n"

    def expr(self, value: str, *, lane_var: str) -> str:
        if value in self.reduction_aliases:
            return self.reduction_aliases[value]
        if value in self.params and not self.params[value].type.is_pointer:
            return self.names[value]
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR value %{value}")

        op = self.defs[value]
        if op.name == "arith.constant":
            ty = op.result_types[0] if op.result_types else TTIRType("i64", "int64")
            dtype = ty.dtype
            if dtype.startswith("int") or dtype.startswith("uint"):
                dtype = "int64"
            return _tir_const(op.attrs.get("value", 0), dtype)
        if op.name == "tt.get_program_id":
            if op.attrs.get("axis") != "x":
                raise UnsupportedTTIROpError("reduction_minimal only supports program_id axis x")
            return "row"
        if op.name == "tt.make_range":
            start = int(op.attrs.get("start", 0))
            if start:
                return f"(T.int64({start}) + {self._lane_expr(lane_var)})"
            return self._lane_expr(lane_var)
        if op.name == "tt.splat":
            return self.expr(op.operands[0], lane_var=lane_var)
        if op.name in ("arith.extf", "arith.extsi", "arith.extui", "arith.sitofp", "arith.truncf"):
            source = self.expr(op.operands[0], lane_var=lane_var)
            dtype = op.result_types[0].dtype if op.result_types else "int64"
            if source in ("row", "i") and dtype == "int64":
                return source
            if source.startswith('T.Cast("int64"') and dtype == "int64":
                return source
            return f'T.Cast("{dtype}", {source})'
        if op.name in _BINARY_OP_SYMBOLS:
            lhs = self.expr(op.operands[0], lane_var=lane_var)
            rhs = self.expr(op.operands[1], lane_var=lane_var)
            symbol = _BINARY_OP_SYMBOLS[op.name]
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.cmpi":
            lhs = self.expr(op.operands[0], lane_var=lane_var)
            rhs = self.expr(op.operands[1], lane_var=lane_var)
            symbol = _cmp_symbol(op.attrs.get("predicate", ""))
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.select":
            cond = self.expr(op.operands[0], lane_var=lane_var)
            true_value = self.expr(op.operands[1], lane_var=lane_var)
            false_value = self.expr(op.operands[2], lane_var=lane_var)
            return f"T.Select({cond}, {true_value}, {false_value})"
        if op.name == "math.rsqrt":
            source = self.expr(op.operands[0], lane_var=lane_var)
            return f"T.rsqrt({source})"
        if op.name == "tt.load":
            if len(op.operands) == 1:
                buffer_name, index = self.pointer_ref(op.operands[0], lane_var=lane_var)
                return f"{buffer_name}[{index}]"
            if len(op.operands) < 3:
                raise UnsupportedTTIROpError(
                    "masked tt.load without explicit zero other is not supported by "
                    "reduction_minimal"
                )
            if not self._is_zero_value(op.operands[2]):
                raise UnsupportedTTIROpError(
                    "reduction_minimal only supports masked tt.load with zero other"
                )
            buffer_name, index = self.pointer_ref(op.operands[0], lane_var=lane_var)
            mask = self.expr(op.operands[1], lane_var=lane_var)
            other = self.expr(op.operands[2], lane_var=lane_var)
            return f"T.if_then_else({mask}, {buffer_name}[{index}], {other})"
        if op.name == "tt.reduce":
            if value not in self.reduction_aliases:
                raise UnsupportedTTIROpError(
                    "tt.reduce result is used before its accumulator is emitted"
                )
            return self.reduction_aliases[value]

        raise UnsupportedTTIROpError(f"{op.name} is not supported by reduction_minimal")

    def pointer_ref(self, value: str, *, lane_var: str) -> tuple[str, str]:
        if value in self.params and self.params[value].type.is_pointer:
            return self.names[value], "T.int64(0)"
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR pointer %{value}")
        op = self.defs[value]
        if op.name == "tt.splat":
            return self.pointer_ref(op.operands[0], lane_var=lane_var)
        if op.name == "tt.addptr":
            base, base_index = self.pointer_ref(op.operands[0], lane_var=lane_var)
            offset = self.pointer_index(op.operands[1], lane_var=lane_var)
            if base_index == "T.int64(0)":
                return base, offset
            return base, f"({base_index} + {offset})"
        raise UnsupportedTTIROpError(f"{op.name} cannot be used as a reduction pointer")

    def pointer_index(self, value: str, *, lane_var: str) -> str:
        if self._is_lane_value(value):
            return self._lane_expr(lane_var)
        if self._is_row_value(value):
            return "row"
        op = self.defs.get(value)
        if op is not None and op.name in ("arith.extsi", "arith.extui") and len(op.operands) == 1:
            return self.pointer_index(op.operands[0], lane_var=lane_var)
        if op is not None and op.name == "arith.addi" and len(op.operands) == 2:
            lhs = self.pointer_index(op.operands[0], lane_var=lane_var)
            rhs = self.pointer_index(op.operands[1], lane_var=lane_var)
            if lhs == "T.int64(0)":
                return rhs
            if rhs == "T.int64(0)":
                return lhs
            return f"({lhs} + {rhs})"
        if op is not None and op.name == "arith.muli" and len(op.operands) == 2:
            lhs, rhs = op.operands
            lhs_extent = self._row_extent_multiplier(lhs)
            rhs_extent = self._row_extent_multiplier(rhs)
            if lhs_extent is not None and self._is_row_value(rhs):
                return f"(row * {lhs_extent})"
            if rhs_extent is not None and self._is_row_value(lhs):
                return f"(row * {rhs_extent})"
        constant = self._constant_int_value(value)
        if constant is not None:
            return f"T.int64({constant})"
        raise UnsupportedTTIROpError("unsupported non-row-major reduction pointer pattern")

    def _param_signature(self) -> str:
        items: list[str] = []
        for param in self.graph.params:
            name = self.names[param.name]
            if param.type.is_pointer:
                items.append(f"{name}_handle: T.handle")
            else:
                items.append(f"{name}: T.{param.type.dtype}")
        return ", ".join(items)

    def _find_block_size(self) -> int:
        for op in self.graph.ops:
            if op.name == "tt.make_range":
                start = int(op.attrs.get("start", 0))
                end = int(op.attrs.get("end", 0))
                if end <= start:
                    raise UnsupportedTTIROpError("tt.make_range must have positive extent")
                if start != 0:
                    raise UnsupportedTTIROpError(
                        "tt.make_range with non-zero start is not supported by reduction_minimal"
                    )
                return end - start
        raise UnsupportedTTIROpError(
            "reduction_minimal requires tt.make_range to determine BLOCK"
        )

    def _find_lane_ssa(self) -> str:
        for op in self.graph.ops:
            if op.name == "tt.make_range" and op.results:
                return op.results[0]
        raise UnsupportedTTIROpError("reduction_minimal requires a lane range")

    def _find_stores(self) -> list[TTIROp]:
        stores = [op for op in self.graph.ops if op.name == "tt.store"]
        if not stores:
            raise UnsupportedTTIROpError("reduction_minimal requires at least one tt.store")
        for store in stores:
            if len(store.operands) < 2:
                raise UnsupportedTTIROpError("tt.store requires pointer and value operands")
        return stores

    def _find_reductions(self) -> list[TTIROp]:
        reductions = [op for op in self.graph.ops if op.name == "tt.reduce"]
        if not reductions:
            raise UnsupportedTTIROpError("reduction_minimal requires at least one tt.reduce")
        return reductions

    def _validate_reductions(self) -> None:
        for op in self.reductions:
            if op.attrs.get("axis") != 0:
                raise UnsupportedTTIROpError("reduction_minimal only supports tt.reduce axis=0")
            if len(op.operands) != 1 or len(op.results) != 1:
                raise UnsupportedTTIROpError(
                    "reduction_minimal only supports single-input tt.reduce"
                )
            if len(op.regions) != 1:
                raise UnsupportedTTIROpError(
                    "reduction_minimal requires tt.reduce to carry one combiner region"
                )
            region = op.regions[0]
            combine_ops = [region_op for region_op in region if region_op.name != "tt.reduce.return"]
            returns = [region_op for region_op in region if region_op.name == "tt.reduce.return"]
            if len(combine_ops) != 1 or len(returns) != 1:
                raise UnsupportedTTIROpError(
                    "reduction_minimal only supports a single sum combiner"
                )
            combine = combine_ops[0]
            if combine.name not in ("arith.addf", "arith.addi"):
                raise UnsupportedTTIROpError(
                    "reduction_minimal only supports sum reduction combiners"
                )
            if not combine.results or returns[0].operands != combine.results[:1]:
                raise UnsupportedTTIROpError(
                    "tt.reduce.return must return the sum combiner result"
                )

    def _validate_memory_masks(self) -> None:
        masks: list[str] = []
        for op in self.graph.ops:
            if op.name == "tt.load":
                if len(op.operands) == 1:
                    continue
                if len(op.operands) < 3:
                    raise UnsupportedTTIROpError(
                        "masked tt.load without explicit zero other is not supported by "
                        "reduction_minimal"
                    )
                if not self._is_zero_value(op.operands[2]):
                    raise UnsupportedTTIROpError(
                        "reduction_minimal only supports masked tt.load with zero other"
                    )
                mask = op.operands[1]
            elif op.name == "tt.store" and len(op.operands) >= 3:
                mask = op.operands[2]
            else:
                continue
            if mask not in masks:
                masks.append(mask)
        if len(masks) > 1:
            raise UnsupportedTTIROpError(
                "reduction_minimal requires all load/store masks to share one extent"
            )

    def _find_extent_info(self) -> _ExtentInfo:
        candidates: list[_ExtentInfo] = []
        for mask in self._collect_mask_values():
            op = self.defs.get(mask)
            if op is None or op.name != "arith.cmpi" or len(op.operands) != 2:
                continue
            lhs, rhs = op.operands
            predicate = op.attrs.get("predicate", "")
            candidate = None
            if predicate in ("slt", "ult", "sle", "ule") and self._is_lane_value(lhs):
                candidate = self._extent_bound(rhs)
            elif predicate in ("sgt", "ugt", "sge", "uge") and self._is_lane_value(rhs):
                candidate = self._extent_bound(lhs)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)

        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise UnsupportedTTIROpError(
                "reduction_minimal requires a unique extent from its mask"
            )
        raise UnsupportedTTIROpError(
            "ambiguous reduction extent candidates: "
            + ", ".join(candidate.expr for candidate in candidates)
        )

    def _extent_bound(self, value: str) -> _ExtentInfo | None:
        param = self._scalar_splat_param(value)
        if param is not None:
            dtype = self.params[param].type.dtype
            name = self.names[param]
            expr = name if dtype == "int64" else f'T.Cast("int64", {name})'
            return _ExtentInfo(kind="runtime_param", expr=expr, param_name=param)
        constant = self._constant_int_value(value)
        if constant is not None:
            if constant <= 0:
                raise UnsupportedTTIROpError("reduction extent constant must be positive")
            return _ExtentInfo(kind="constant", expr=f"T.int64({constant})", value=constant)
        return None

    def _infer_buffer_extents(self) -> dict[str, str]:
        extents = {param.name: self.extent_info.expr for param in self.ptr_params}
        for op in [op for op in self.graph.ops if op.name in ("tt.load", "tt.store")]:
            if not op.operands:
                continue
            base = self._base_pointer_param(op.operands[0])
            if base is None:
                continue
            _, index = self.pointer_ref(op.operands[0], lane_var="i")
            inferred = self._extent_for_index(index)
            previous = extents.get(base)
            if previous is None or self._extent_priority(inferred) >= self._extent_priority(previous):
                extents[base] = inferred
        return extents

    def _extent_for_index(self, index: str) -> str:
        row_extent = f"T.int64({self.row_count})"
        if "row" in index and "i" in index:
            return f"({row_extent} * {self.extent_info.expr})"
        if "row" in index:
            return row_extent
        return self.extent_info.expr

    def _extent_priority(self, extent: str) -> int:
        if "*" in extent:
            return 3
        if extent == f"T.int64({self.row_count})":
            return 2
        return 1

    def _collect_mask_values(self) -> list[str]:
        masks: list[str] = []
        for op in self.graph.ops:
            mask = None
            if op.name == "tt.load" and len(op.operands) >= 2:
                mask = op.operands[1]
            elif op.name == "tt.store" and len(op.operands) >= 3:
                mask = op.operands[2]
            if mask is not None and mask not in masks:
                masks.append(mask)
        return masks

    def _lane_expr(self, lane_var: str) -> str:
        if lane_var == "i":
            return "i"
        return f'T.Cast("int64", {lane_var})'

    def _is_lane_value(self, value: str) -> bool:
        if value == self.lane_ssa:
            return True
        op = self.defs.get(value)
        return (
            op is not None
            and op.name in ("arith.extsi", "arith.extui")
            and len(op.operands) == 1
            and op.operands[0] == self.lane_ssa
        )

    def _is_row_value(self, value: str) -> bool:
        op = self.defs.get(value)
        if op is None:
            return False
        if op.name == "tt.get_program_id" and op.attrs.get("axis") == "x":
            return True
        return (
            op.name in ("arith.extsi", "arith.extui")
            and len(op.operands) == 1
            and self._is_row_value(op.operands[0])
        )

    def _row_extent_multiplier(self, value: str) -> str | None:
        if self.extent_info.param_name and value == self.extent_info.param_name:
            return self.extent_info.expr
        constant = self._constant_int_value(value)
        if constant is not None:
            if constant <= 0:
                raise UnsupportedTTIROpError("row-major reduction multiplier must be positive")
            return f"T.int64({constant})"
        return None

    def _scalar_splat_param(self, value: str) -> str | None:
        if value in self.params and not self.params[value].type.is_pointer:
            return value
        op = self.defs.get(value)
        if op is None or op.name != "tt.splat" or len(op.operands) != 1:
            return None
        source = op.operands[0]
        if source in self.params and not self.params[source].type.is_pointer:
            return source
        return None

    def _constant_int_value(self, value: str) -> int | None:
        op = self.defs.get(value)
        if op is None:
            return None
        if op.name == "tt.splat" and len(op.operands) == 1:
            return self._constant_int_value(op.operands[0])
        if op.name != "arith.constant":
            return None
        constant = op.attrs.get("value")
        if isinstance(constant, bool):
            return None
        if isinstance(constant, int):
            return constant
        if isinstance(constant, float) and constant.is_integer():
            return int(constant)
        return None

    def _is_zero_value(self, value: str) -> bool:
        op = self.defs.get(value)
        if op is None:
            return False
        if op.name == "tt.splat" and len(op.operands) == 1:
            return self._is_zero_value(op.operands[0])
        if op.name != "arith.constant":
            return False
        constant = op.attrs.get("value")
        return constant in (0, 0.0, False)

    def _base_pointer_param(self, value: str) -> str | None:
        if value in self.params and self.params[value].type.is_pointer:
            return value
        op = self.defs.get(value)
        if op is None or not op.operands:
            return None
        if op.name in ("tt.addptr", "tt.splat"):
            return self._base_pointer_param(op.operands[0])
        return None

    def _value_dtype(self, value: str) -> str:
        if value in self.params:
            return self.params[value].type.dtype
        op = self.defs.get(value)
        if op is not None and op.result_types:
            return op.result_types[0].dtype
        raise UnsupportedTTIROpError(f"Cannot infer dtype for %{value}")


class _TIRXRank2RowBuilder:
    def __init__(
        self,
        graph: NormalizedTTIROpGraph,
        target_policy: _PointwiseTargetPolicy,
        contract: str,
    ):
        self.graph = graph
        self.target_policy = target_policy
        self.contract = contract
        self.defs = graph.op_by_result()
        self.params = graph.param_by_name()
        self.names = {param.name: _sanitize_identifier(param.name) for param in graph.params}
        self.ptr_params = [param for param in graph.params if param.type.is_pointer]
        self.scalar_params = [param for param in graph.params if not param.type.is_pointer]
        self.x_range, self.r_range = self._find_axis_ranges()
        self.block_size = self._range_extent(self.x_range)
        self.r_block_size = self._range_extent(self.r_range)
        self.reductions = self._find_reductions()
        self._validate_reductions()
        self.stores = self._find_stores()
        self.extent_info = self._find_x_extent_info()
        self.r_extent_info = self._find_r_extent_info()
        self.reduction_extent_info = self.r_extent_info
        self.buffer_extents = self._infer_buffer_extents()
        self.reduction_aliases: dict[str, str] = {}

    def build_source(self) -> str:
        extent = self.extent_info.expr
        red_index = 'T.Cast("int64", tx)'
        lines: list[str] = [
            "# from tvm.script import ir as I",
            "# from tvm.script import tirx as T",
            "",
            "@I.ir_module",
            "class Module:",
            "    @T.prim_func",
            f"    def {_sanitize_identifier(self.graph.function_name)}({self._param_signature()}):",
            "        T.func_attr({"
            f'"global_symbol": "{self.graph.function_name}", '
            '"tirx.noalias": True, '
            f'"target": T.target({self.target_policy.target_attrs!r})'
            "})",
        ]
        for param in self.ptr_params:
            name = self.names[param.name]
            buffer_extent = self.buffer_extents.get(param.name, self._default_buffer_extent())
            lines.append(
                "        "
                f'{name} = T.match_buffer({name}_handle, ({buffer_extent},), '
                f'"{param.type.dtype}")'
            )
        for i, reduce_op in enumerate(self.reductions):
            dtype = self._value_dtype(reduce_op.results[0])
            lines.append(
                f'        red_{i} = T.alloc_buffer(({self.block_size},), "{dtype}", scope="local")'
            )

        lines.extend(
            [
                "        "
                f"for {self.target_policy.block_var} in T.thread_binding("
                f"0, T.ceildiv({extent}, T.int64({self.block_size})), "
                f'thread="{self.target_policy.block_thread_tag}"):',
                "            "
                f"for {self.target_policy.lane_var} in T.thread_binding(0, {self.block_size}, "
                f'thread="{self.target_policy.lane_thread_tag}"):',
                "                txi = T.Cast(\"int64\", "
                f"{self.target_policy.lane_var})",
                "                row_i = "
                f"T.Cast(\"int64\", {self.target_policy.block_var}) * "
                f"T.int64({self.block_size}) + txi",
                f"                if row_i < {extent}:",
            ]
        )

        for i, reduce_op in enumerate(self.reductions):
            acc = f"red_{i}"
            kind = self._reduction_kind(reduce_op)
            dtype = self._value_dtype(reduce_op.results[0])
            identity = self._reduction_identity(kind, dtype)
            lines.append(f"                    {acc}[{red_index}] = {identity}")
            lines.append(f"                    for rk in T.serial(0, {self.r_block_size}):")
            lines.append('                        r = T.Cast("int64", rk)')
            value_expr = self.expr(reduce_op.operands[0], r_var="r")
            update = self._reduction_update(kind, f"{acc}[{red_index}]", value_expr)
            lines.append(f"                        {acc}[{red_index}] = {update}")
            self.reduction_aliases[reduce_op.results[0]] = f"{acc}[{red_index}]"

        for store in self.stores:
            store_r_extent = self._store_r_loop_extent(store)
            store_ptr = store.operands[0]
            store_value = store.operands[1]
            store_mask = store.operands[2] if len(store.operands) >= 3 else None
            if store_r_extent > 1:
                lines.append(f"                    for sk in T.serial(0, {store_r_extent}):")
                lines.append('                        r = T.Cast("int64", sk)')
                out_buffer, out_index = self.pointer_ref(store_ptr, r_var="r")
                value_expr = self.expr(store_value, r_var="r")
                mask_expr = self.expr(store_mask, r_var="r") if store_mask is not None else "True"
                lines.append(f"                        if {mask_expr}:")
                lines.append(f"                            {out_buffer}[{out_index}] = {value_expr}")
            else:
                lines.append("                    r = T.int64(0)")
                out_buffer, out_index = self.pointer_ref(store_ptr, r_var="r")
                value_expr = self.expr(store_value, r_var="r")
                mask_expr = self.expr(store_mask, r_var="r") if store_mask is not None else "True"
                lines.append(f"                    if {mask_expr}:")
                lines.append(f"                        {out_buffer}[{out_index}] = {value_expr}")
        return "\n".join(lines) + "\n"

    def expr(self, value: str, *, r_var: str) -> str:
        if value in self.reduction_aliases:
            return self.reduction_aliases[value]
        if value in self.params and not self.params[value].type.is_pointer:
            dtype = self.params[value].type.dtype
            name = self.names[value]
            if dtype.startswith(("int", "uint")) and dtype != "int64":
                return f'T.Cast("int64", {name})'
            return name
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR value %{value}")

        op = self.defs[value]
        if op.name == "arith.constant":
            ty = op.result_types[0] if op.result_types else TTIRType("i64", "int64")
            dtype = ty.dtype
            if dtype.startswith("int") or dtype.startswith("uint"):
                dtype = "int64"
            return _tir_const(op.attrs.get("value", 0), dtype)
        if op.name == "tt.get_program_id":
            if op.attrs.get("axis") != "x":
                raise UnsupportedTTIROpError(f"{self.contract} only supports program_id axis x")
            return f'T.Cast("int64", {self.target_policy.block_var})'
        if op.name == "tt.make_range":
            if value == self.r_range and self.x_range != self.r_range:
                return r_var
            return 'T.Cast("int64", tx)'
        if op.name == "tt.expand_dims":
            axis = op.attrs.get("axis")
            source = op.operands[0]
            if self._is_make_range_value(source):
                if axis == 0:
                    return r_var
                if axis == 1:
                    return 'T.Cast("int64", tx)'
            return self.expr(source, r_var=r_var)
        if op.name in ("tt.broadcast", "tt.splat"):
            return self.expr(op.operands[0], r_var=r_var)
        if op.name in (
            "arith.extf",
            "arith.extsi",
            "arith.extui",
            "arith.fptosi",
            "arith.fptoui",
            "arith.sitofp",
            "arith.truncf",
        ):
            source = self.expr(op.operands[0], r_var=r_var)
            dtype = op.result_types[0].dtype if op.result_types else "int64"
            return f'T.Cast("{dtype}", {source})'
        if op.name in _BINARY_OP_SYMBOLS:
            lhs = self.expr(op.operands[0], r_var=r_var)
            rhs = self.expr(op.operands[1], r_var=r_var)
            symbol = _BINARY_OP_SYMBOLS[op.name]
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.cmpi":
            lhs = self.expr(op.operands[0], r_var=r_var)
            rhs = self.expr(op.operands[1], r_var=r_var)
            symbol = _cmp_symbol(op.attrs.get("predicate", ""))
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.cmpf":
            lhs = self.expr(op.operands[0], r_var=r_var)
            rhs = self.expr(op.operands[1], r_var=r_var)
            symbol = _cmpf_symbol(op.attrs.get("predicate", ""))
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.select":
            cond = self.expr(op.operands[0], r_var=r_var)
            true_value = self.expr(op.operands[1], r_var=r_var)
            false_value = self.expr(op.operands[2], r_var=r_var)
            return f"T.Select({cond}, {true_value}, {false_value})"
        if op.name in ("math.exp", "math.rsqrt"):
            source = self.expr(op.operands[0], r_var=r_var)
            tir_op = {"math.exp": "exp", "math.rsqrt": "rsqrt"}[op.name]
            return f"T.{tir_op}({source})"
        if op.name == "tt.extern_elementwise":
            source = self.expr(op.operands[0], r_var=r_var)
            symbol = _unquote_attr(op.attrs.get("symbol", ""))
            mapping = {
                "__nv_expf": "exp",
                "__nv_rsqrtf": "rsqrt",
                "__nv_erff": "erf",
            }
            if symbol not in mapping:
                raise UnsupportedTTIROpError(
                    f"tt.extern_elementwise symbol {symbol!r} is not supported by {self.contract}"
                )
            return f"T.{mapping[symbol]}({source})"
        if op.name == "tt.load":
            buffer_name, index = self.pointer_ref(op.operands[0], r_var=r_var)
            if len(op.operands) < 3:
                return f"{buffer_name}[{index}]"
            mask = self.expr(op.operands[1], r_var=r_var)
            other = self.expr(op.operands[2], r_var=r_var)
            return f"T.if_then_else({mask}, {buffer_name}[{index}], {other})"
        if op.name == "tt.reduce":
            if value not in self.reduction_aliases:
                raise UnsupportedTTIROpError(
                    "tt.reduce result is used before its accumulator is emitted"
                )
            return self.reduction_aliases[value]

        raise UnsupportedTTIROpError(f"{op.name} is not supported by {self.contract}")

    def pointer_ref(self, value: str, *, r_var: str) -> tuple[str, str]:
        if value in self.params and self.params[value].type.is_pointer:
            return self.names[value], "T.int64(0)"
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR pointer %{value}")
        op = self.defs[value]
        if op.name in ("tt.splat", "tt.broadcast", "tt.expand_dims"):
            return self.pointer_ref(op.operands[0], r_var=r_var)
        if op.name == "tt.addptr":
            base, base_index = self.pointer_ref(op.operands[0], r_var=r_var)
            offset = self.expr(op.operands[1], r_var=r_var)
            if base_index == "T.int64(0)":
                return base, offset
            return base, f"({base_index} + {offset})"
        raise UnsupportedTTIROpError(f"{op.name} cannot be used as a rank-2 pointer")

    def _param_signature(self) -> str:
        items: list[str] = []
        for param in self.graph.params:
            name = self.names[param.name]
            if param.type.is_pointer:
                items.append(f"{name}_handle: T.handle")
            else:
                items.append(f"{name}: T.{param.type.dtype}")
        return ", ".join(items)

    def _find_axis_ranges(self) -> tuple[str, str]:
        ranges = [op for op in self.graph.ops if op.name == "tt.make_range" and op.results]
        if not ranges:
            raise UnsupportedTTIROpError(f"{self.contract} requires tt.make_range")
        x_candidates: list[str] = []
        r_candidates: list[str] = []
        for range_op in ranges:
            value = range_op.results[0]
            for user in self.graph.ops:
                if user.name != "tt.expand_dims" or not user.operands or user.operands[0] != value:
                    continue
                if user.attrs.get("axis") == 1:
                    x_candidates.append(value)
                elif user.attrs.get("axis") == 0:
                    r_candidates.append(value)
            if value.startswith("x"):
                x_candidates.append(value)
            if value.startswith(("r", "r0")):
                r_candidates.append(value)
        x_range = x_candidates[0] if x_candidates else ""
        r_range = r_candidates[0] if r_candidates else (ranges[-1].results[0])
        return x_range, r_range

    def _range_extent(self, value: str) -> int:
        if not value:
            return 1
        op = self.defs.get(value)
        if op is None or op.name != "tt.make_range":
            raise UnsupportedTTIROpError(f"Cannot resolve rank-2 range %{value}")
        start = int(op.attrs.get("start", 0))
        end = int(op.attrs.get("end", 0))
        if end <= start:
            raise UnsupportedTTIROpError("tt.make_range must have positive extent")
        if start != 0:
            raise UnsupportedTTIROpError(f"{self.contract} requires zero-based ranges")
        return end - start

    def _find_reductions(self) -> list[TTIROp]:
        reductions = [op for op in self.graph.ops if op.name == "tt.reduce"]
        if not reductions:
            raise UnsupportedTTIROpError(f"{self.contract} requires at least one tt.reduce")
        return reductions

    def _find_stores(self) -> list[TTIROp]:
        stores = [op for op in self.graph.ops if op.name == "tt.store"]
        if not stores:
            raise UnsupportedTTIROpError(f"{self.contract} requires at least one tt.store")
        for store in stores:
            if len(store.operands) < 2:
                raise UnsupportedTTIROpError("tt.store requires pointer and value operands")
        return stores

    def _validate_reductions(self) -> None:
        for op in self.reductions:
            if op.attrs.get("axis") != 1:
                raise UnsupportedTTIROpError(f"{self.contract} only supports tt.reduce axis=1")
            if len(op.operands) != 1 or len(op.results) != 1:
                raise UnsupportedTTIROpError(f"{self.contract} only supports single-input tt.reduce")
            if len(op.regions) != 1:
                raise UnsupportedTTIROpError(f"{self.contract} requires one tt.reduce region")
            self._reduction_kind(op)

    def _reduction_kind(self, op: TTIROp) -> str:
        region = op.regions[0]
        combine_ops = [region_op for region_op in region if region_op.name != "tt.reduce.return"]
        returns = [region_op for region_op in region if region_op.name == "tt.reduce.return"]
        if len(returns) != 1:
            raise UnsupportedTTIROpError(f"{self.contract} requires one tt.reduce.return")
        if len(combine_ops) == 1 and combine_ops[0].name in ("arith.addf", "arith.addi"):
            combine = combine_ops[0]
            if combine.results and returns[0].operands == combine.results[:1]:
                return "sum"
        if any(region_op.name == "arith.select" for region_op in combine_ops) and any(
            region_op.name == "arith.cmpf" for region_op in combine_ops
        ):
            return "max"
        raise UnsupportedTTIROpError(f"{self.contract} only supports sum and max reductions")

    def _reduction_identity(self, kind: str, dtype: str) -> str:
        if kind == "sum":
            return _tir_const(0, dtype)
        if kind == "max":
            return f'T.min_value("{dtype}")'
        raise UnsupportedTTIROpError(f"unsupported reduction kind {kind!r}")

    def _reduction_update(self, kind: str, acc: str, value: str) -> str:
        if kind == "sum":
            return f"({acc} + {value})"
        if kind == "max":
            return f"T.max({acc}, {value})"
        raise UnsupportedTTIROpError(f"unsupported reduction kind {kind!r}")

    def _find_x_extent_info(self) -> _ExtentInfo:
        for op in self.graph.ops:
            if op.name != "arith.cmpi" or len(op.operands) != 2:
                continue
            shape = self._value_shape(op.results[0]) if op.results else ()
            if not self._is_x_mask_shape(shape):
                continue
            lhs, rhs = op.operands
            predicate = op.attrs.get("predicate", "")
            if predicate in ("slt", "ult", "sle", "ule"):
                candidate = self._extent_bound(rhs)
            elif predicate in ("sgt", "ugt", "sge", "uge"):
                candidate = self._extent_bound(lhs)
            else:
                candidate = None
            if candidate is not None:
                return candidate
        return _ExtentInfo(kind="constant", expr=f"T.int64({self.block_size})", value=self.block_size)

    def _find_r_extent_info(self) -> _ExtentInfo:
        for op in self.graph.ops:
            if op.name != "arith.cmpi" or len(op.operands) != 2:
                continue
            shape = self._value_shape(op.results[0]) if op.results else ()
            if not self._is_r_mask_shape(shape):
                continue
            lhs, rhs = op.operands
            predicate = op.attrs.get("predicate", "")
            if predicate in ("slt", "ult", "sle", "ule"):
                candidate = self._extent_bound(rhs)
            elif predicate in ("sgt", "ugt", "sge", "uge"):
                candidate = self._extent_bound(lhs)
            else:
                candidate = None
            if candidate is not None:
                return candidate
        return _ExtentInfo(
            kind="constant",
            expr=f"T.int64({self.r_block_size})",
            value=self.r_block_size,
        )

    def _extent_bound(self, value: str) -> _ExtentInfo | None:
        param = self._scalar_splat_param(value)
        if param is not None:
            dtype = self.params[param].type.dtype
            name = self.names[param]
            expr = name if dtype == "int64" else f'T.Cast("int64", {name})'
            return _ExtentInfo(kind="runtime_param", expr=expr, param_name=param)
        constant = self._constant_int_value(value)
        if constant is not None:
            if constant <= 0:
                raise UnsupportedTTIROpError("rank-2 row extent constant must be positive")
            return _ExtentInfo(kind="constant", expr=f"T.int64({constant})", value=constant)
        return None

    def _infer_buffer_extents(self) -> dict[str, str]:
        default = self._default_buffer_extent()
        extents = {param.name: default for param in self.ptr_params}
        for store in self.stores:
            base = self._base_pointer_param(store.operands[0])
            if base is None:
                continue
            shape = self._value_shape(store.operands[0])
            if len(shape) >= 2 and shape[1] == 1:
                extents[base] = self.extent_info.expr
        return extents

    def _default_buffer_extent(self) -> str:
        return f"({self.extent_info.expr} * {self._physical_r_extent_expr()})"

    def _physical_r_extent_expr(self) -> str:
        if self.r_extent_info.kind == "constant" and self.r_extent_info.value is not None:
            return f"T.int64({max(self.r_extent_info.value, self.r_block_size)})"
        return self.r_extent_info.expr

    def _store_r_loop_extent(self, store: TTIROp) -> int:
        shape = self._value_shape(store.operands[0])
        if len(shape) >= 2:
            return max(1, int(shape[1]))
        return 1

    def _value_shape(self, value: str) -> tuple[int, ...]:
        if value in self.params:
            return self.params[value].type.shape
        op = self.defs.get(value)
        if op is not None and op.result_types:
            return op.result_types[0].shape
        return ()

    def _value_dtype(self, value: str) -> str:
        if value in self.params:
            return self.params[value].type.dtype
        op = self.defs.get(value)
        if op is not None and op.result_types:
            return op.result_types[0].dtype
        raise UnsupportedTTIROpError(f"Cannot infer dtype for %{value}")

    def _is_x_mask_shape(self, shape: tuple[int, ...]) -> bool:
        if not shape:
            return True
        if len(shape) == 1:
            return shape[0] == self.block_size
        return len(shape) == 2 and shape[0] == self.block_size and shape[1] == 1

    def _is_r_mask_shape(self, shape: tuple[int, ...]) -> bool:
        return len(shape) == 2 and shape[0] == 1 and shape[1] == self.r_block_size

    def _scalar_splat_param(self, value: str) -> str | None:
        if value in self.params and not self.params[value].type.is_pointer:
            return value
        op = self.defs.get(value)
        if op is None or op.name != "tt.splat" or len(op.operands) != 1:
            return None
        source = op.operands[0]
        if source in self.params and not self.params[source].type.is_pointer:
            return source
        return None

    def _constant_int_value(self, value: str) -> int | None:
        op = self.defs.get(value)
        if op is None:
            return None
        if op.name in ("tt.splat", "tt.broadcast", "tt.expand_dims") and len(op.operands) == 1:
            return self._constant_int_value(op.operands[0])
        if op.name != "arith.constant":
            return None
        constant = op.attrs.get("value")
        if isinstance(constant, bool):
            return None
        if isinstance(constant, int):
            return constant
        if isinstance(constant, float) and constant.is_integer():
            return int(constant)
        return None

    def _is_make_range_value(self, value: str) -> bool:
        op = self.defs.get(value)
        return op is not None and op.name == "tt.make_range"

    def _base_pointer_param(self, value: str) -> str | None:
        if value in self.params and self.params[value].type.is_pointer:
            return value
        op = self.defs.get(value)
        if op is None or not op.operands:
            return None
        if op.name in ("tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims"):
            return self._base_pointer_param(op.operands[0])
        return None


def _validate_supported_subset(graph: NormalizedTTIROpGraph, contract: str) -> None:
    supported_ops = _SUPPORTED_OPS_BY_CONTRACT.get(contract)
    if supported_ops is None:
        raise UnsupportedContractError(
            f"Contract {contract!r} is not implemented yet; "
            "only pointwise, reduction_minimal, and norm_single_row contracts are "
            "supported by this translator."
        )
    for op in _iter_ops_with_regions(graph.ops):
        if op.name not in supported_ops:
            raise UnsupportedTTIROpError(f"{op.name} is not supported by contract {contract!r}")
        if op.name == "tt.load" and len(op.operands) < 3:
            if contract in ("reduction_minimal", "norm_single_row") and len(op.operands) == 1:
                continue
            if contract in _M8_ROW_CONTRACTS and len(op.operands) in (1, 2):
                continue
            if contract != "pointwise_flat":
                raise UnsupportedTTIROpError(
                    f"masked tt.load without other is not supported by contract {contract!r}"
                )


def _iter_ops_with_regions(ops: list[TTIROp]):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _iter_ops_with_regions(region)


def _reduction_grid_extent(grid) -> int:
    if callable(grid):
        raise ValueError("reduction_minimal requires a static 1D grid")
    if isinstance(grid, int):
        value = int(grid)
    elif isinstance(grid, tuple | list) and len(grid) == 1:
        value = int(grid[0])
    else:
        raise ValueError("reduction_minimal requires a static 1D grid")
    if value <= 0:
        raise ValueError("reduction_minimal grid extent must be positive")
    return value


def _signature_from_graph(graph: NormalizedTTIROpGraph) -> dict[str, str]:
    return {param.name: _signature_type(param.type) for param in graph.params}


def _signature_type(ty: TTIRType) -> str:
    dtype = _triton_dtype(ty.dtype)
    if ty.is_pointer:
        return "*" + dtype
    return dtype


def _abi_from_graph(graph: NormalizedTTIROpGraph) -> list[dict[str, str]]:
    abi = []
    for param in graph.params:
        abi.append(
            {
                "name": param.name,
                "kind": "pointer" if param.type.is_pointer else "scalar",
                "dtype": param.type.dtype,
            }
        )
    return abi


def _cache_key(**kwargs) -> str:
    payload = json.dumps(_stable_jsonable(kwargs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _stable_jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable_jsonable(item) for item in value]
    raise TypeError(f"Unstable value in Triton TVM cache key payload: {type(value)!r}")


def _current_triton_version() -> str:
    try:
        import triton  # pylint: disable=import-outside-toplevel

        return triton.__version__
    except ImportError:
        return ""


def _normalize_target(target) -> tuple[tvm.target.Target, str]:
    if isinstance(target, tvm.target.Target):
        target_obj = target
    else:
        try:
            target_obj = tvm.target.Target(target)
        except ValueError:
            target_obj = _parse_legacy_cuda_target_string(target)
    return target_obj, str(target_obj)


def _target_kind(target) -> str:
    target_obj, _ = _normalize_target(target)
    return target_obj.kind.name


def _pointwise_target_policy(target) -> _PointwiseTargetPolicy:
    target_obj, target_attrs = _normalize_target(target)
    target_kind = target_obj.kind.name
    if target_kind == "cuda":
        return _PointwiseTargetPolicy(
            target=target_attrs,
            target_kind=target_kind,
            launch_policy_id="cuda_block_thread",
            version=_CUDA_TARGET_POLICY_VERSION,
            target_attrs=target_attrs,
            supported_contracts=frozenset(
                {
                    "pointwise_minimal",
                    "pointwise_flat",
                    "reduction_minimal",
                    "norm_single_row",
                    "row_reduction",
                    "norm_row",
                    "softmax_row",
                    "masked_softmax_row",
                }
            ),
            block_var="bx",
            lane_var="tx",
            block_thread_tag="blockIdx.x",
            lane_thread_tag="threadIdx.x",
        )
    raise UnsupportedTargetPolicyError(
        "No backend policy registered for target kind "
        f"{target_kind!r} and Triton TVM contracts."
    )


def _validate_policy_contract_support(
    target_policy: _PointwiseTargetPolicy, contract: str
) -> None:
    if contract not in target_policy.supported_contracts:
        raise UnsupportedContractError(
            f"Backend policy {target_policy.launch_policy_id!r} does not support "
            f"contract {contract!r}."
        )


def _parse_legacy_cuda_target_string(target) -> tvm.target.Target:
    if not isinstance(target, str):
        raise ValueError(f"Unsupported target type for pointwise policy: {type(target)!r}")
    match = re.fullmatch(r"cuda\s+-arch=([\w.]+)", target.strip())
    if not match:
        raise ValueError(f"Cannot parse target string {target!r}")
    return tvm.target.Target({"kind": "cuda", "arch": match.group(1)})


def _sanitize_identifier(name: str) -> str:
    name = re.sub(r"\W", "_", name)
    if not name or name[0].isdigit() or keyword.iskeyword(name):
        name = "_" + name
    return name


def _tir_const(value: Any, dtype: str) -> str:
    if dtype == "bool":
        return "True" if bool(value) else "False"
    if dtype.startswith("float") or dtype == "bfloat16":
        if isinstance(value, int) and dtype == "float32":
            value = struct.unpack("!f", int(value).to_bytes(4, "big", signed=False))[0]
        if value == float("-inf"):
            return f'T.min_value("{dtype}")'
        if value == float("inf"):
            return f'T.max_value("{dtype}")'
        return f"T.{dtype}({float(value)!r})"
    if dtype.startswith("int") or dtype.startswith("uint"):
        return f"T.{dtype}({int(value)})"
    return repr(value)


def _cmp_symbol(predicate: str) -> str:
    mapping = {
        "eq": "==",
        "ne": "!=",
        "slt": "<",
        "ult": "<",
        "sle": "<=",
        "ule": "<=",
        "sgt": ">",
        "ugt": ">",
        "sge": ">=",
        "uge": ">=",
    }
    if predicate not in mapping:
        raise UnsupportedTTIROpError(f"arith.cmpi predicate {predicate!r} is not supported yet")
    return mapping[predicate]


def _cmpf_symbol(predicate: str) -> str:
    mapping = {
        "oeq": "==",
        "olt": "<",
        "ole": "<=",
        "ogt": ">",
        "oge": ">=",
        "une": "!=",
    }
    if predicate not in mapping:
        raise UnsupportedTTIROpError(f"arith.cmpf predicate {predicate!r} is not supported yet")
    return mapping[predicate]


def _unquote_attr(value: Any) -> str:
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


def _triton_dtype(dtype: str) -> str:
    mapping = {
        "float16": "fp16",
        "float32": "fp32",
        "float64": "fp64",
        "bfloat16": "bf16",
    }
    return mapping.get(dtype, dtype)
