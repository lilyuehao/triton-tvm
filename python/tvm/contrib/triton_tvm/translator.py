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
from dataclasses import dataclass
from typing import Any

import tvm

from .contracts import (
    get_triton_tvm_contract,
    normalize_triton_tvm_contract,
    validate_triton_tvm_contract,
)
from .errors import UnsupportedContractError, UnsupportedTargetPolicyError, UnsupportedTTIROpError
from .frontend import TTIRArtifact
from .op_graph import NormalizedTTIROpGraph, TTIROp, TTIRType
from .ttir import TTIRReader


_SUPPORTED_OPS = {
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

_BINARY_OP_SYMBOLS = {
    "arith.addf": "+",
    "arith.addi": "+",
    "arith.mulf": "*",
    "arith.muli": "*",
    "arith.subf": "-",
    "arith.subi": "-",
}

_TRANSLATOR_VERSION = "triton_tvm_python_m25_policy_v1"
_CUDA_TARGET_POLICY_VERSION = "cuda_thread_binding_v1"


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
    block_size: int
    indexing_kind: str
    launch_policy_id: str
    abi: list[dict[str, str]]
    cache_key: str


def translate_ttir(
    ttir_or_graph: str | TTIRArtifact | NormalizedTTIROpGraph,
    *,
    grid,
    target: str = "cuda",
    emit: str = "tirx",
    contract: str = "pointwise_minimal",
) -> tuple[tvm.IRModule, TritonTVMMeta]:
    """Translate TTIR into a contract-driven pointwise TIRX IRModule."""
    if emit != "tirx":
        raise ValueError(f"Only emit='tirx' is supported by this prototype, got {emit!r}")
    requested_contract = contract
    canonical_contract = normalize_triton_tvm_contract(contract)
    contract_spec = get_triton_tvm_contract(canonical_contract)
    target_policy = _pointwise_target_policy(target)
    _validate_policy_contract_support(target_policy, canonical_contract)

    graph, artifact = _as_graph(ttir_or_graph)
    _validate_supported_subset(graph, canonical_contract)

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
        grid=grid,
        target=target_policy.target,
        target_kind=target_policy.target_kind,
        canonical_contract=canonical_contract,
        emit=emit,
        translator_version=_TRANSLATOR_VERSION,
        contract_version=contract_spec.version,
        target_policy_version=target_policy.version,
        triton_version=triton_version,
        ttir_hash=ttir_hash,
        source_hash=source_hash,
        extent_param=builder.extent_param.name,
        block_size=builder.block_size,
        indexing_kind=contract_spec.indexing_kind,
        launch_policy_id=target_policy.launch_policy_id,
        target_attrs=target_policy.target_attrs,
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
        requested_contract=requested_contract,
        emit=emit,
        translator_version=_TRANSLATOR_VERSION,
        contract_version=contract_spec.version,
        target_policy_version=target_policy.version,
        target_attrs=target_policy.target_attrs,
        triton_version=triton_version,
        tvm_version=tvm.__version__,
        ttir_hash=ttir_hash,
        source_hash=source_hash,
        extent_param=builder.extent_param.name,
        block_size=builder.block_size,
        indexing_kind=contract_spec.indexing_kind,
        launch_policy_id=target_policy.launch_policy_id,
        abi=abi,
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
        self.extent_param = self._find_extent_param()
        self.aliases: dict[str, str] = {}
        if self.lane_index_ssa:
            self.aliases[self.lane_index_ssa] = "i"

    def build_source(self) -> str:
        extent = self.names[self.extent_param.name]

        mask_values = self._collect_mask_values()
        mask_defs = [(mask, self.expr(mask)) for mask in mask_values]
        for i, (mask, _) in enumerate(mask_defs):
            self.aliases[mask] = "mask" if i == 0 else f"mask_{i}"

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
            lines.append(
                "        "
                f'{name} = T.match_buffer({name}_handle, ({extent},), "{param.type.dtype}")'
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
        for mask, mask_expr in mask_defs:
            lines.append(f"                {self.aliases[mask]} = {mask_expr}")
        for store in self.stores:
            store_ptr, store_value = store.operands[0], store.operands[1]
            out_buffer, out_index = self.pointer_ref(store_ptr)
            value_expr = self.expr(store_value)
            store_mask = store.operands[2] if len(store.operands) >= 3 else None
            if store_mask is not None:
                mask_expr = self.aliases.get(store_mask, self.expr(store_mask))
                lines.append(f"                if {mask_expr}:")
                lines.append(f"                    {out_buffer}[{out_index}] = {value_expr}")
            else:
                lines.append(f"                {out_buffer}[{out_index}] = {value_expr}")
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
            return _tir_const(op.attrs.get("value", 0), ty.dtype)
        if op.name == "tt.get_program_id":
            axis = op.attrs.get("axis")
            return self.target_policy.program_id_expr(axis)
        if op.name == "tt.make_range":
            start = int(op.attrs.get("start", 0))
            return self.target_policy.lane_expr(start)
        if op.name == "tt.splat":
            return self.expr(op.operands[0])
        if op.name in ("arith.extsi", "arith.extui"):
            source = self.expr(op.operands[0])
            dtype = op.result_types[0].dtype if op.result_types else "int64"
            if source == "i":
                return "i"
            return f'T.Cast("{dtype}", {source})'
        if op.name in _BINARY_OP_SYMBOLS:
            lhs = self.expr(op.operands[0])
            rhs = self.expr(op.operands[1])
            symbol = _BINARY_OP_SYMBOLS[op.name]
            return f"({lhs} {symbol} {rhs})"
        if op.name == "arith.cmpi":
            lhs = self.expr(op.operands[0])
            rhs = self.expr(op.operands[1])
            symbol = _cmp_symbol(op.attrs.get("predicate", ""))
            return f"({lhs} {symbol} {rhs})"
        if op.name == "tt.load":
            if len(op.operands) < 3:
                raise UnsupportedTTIROpError("masked tt.load without other is not supported yet")
            buffer_name, index = self.pointer_ref(op.operands[0])
            mask = self.expr(op.operands[1])
            other = self.expr(op.operands[2])
            return f"T.if_then_else({mask}, {buffer_name}[{index}], {other})"

        raise UnsupportedTTIROpError(f"{op.name} is not supported yet")

    def pointer_ref(self, value: str) -> tuple[str, str]:
        if value in self.params and self.params[value].type.is_pointer:
            return self.names[value], "0"
        if value not in self.defs:
            raise UnsupportedTTIROpError(f"Cannot resolve TTIR pointer %{value}")
        op = self.defs[value]
        if op.name == "tt.splat":
            return self.pointer_ref(op.operands[0])
        if op.name == "tt.addptr":
            base, base_index = self.pointer_ref(op.operands[0])
            if self.lane_index_ssa is None or op.operands[1] != self.lane_index_ssa:
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

    def _find_extent_param(self):
        if not self.scalar_params:
            raise UnsupportedTTIROpError(
                "pointwise contracts require a runtime scalar extent parameter"
            )

        candidates: list[str] = []
        for mask in self._collect_mask_values():
            op = self.defs.get(mask)
            if op is None or op.name != "arith.cmpi" or len(op.operands) != 2:
                continue
            lhs, rhs = op.operands
            predicate = op.attrs.get("predicate", "")
            candidate = None
            if predicate in ("slt", "ult", "sle", "ule") and self._is_lane_index_value(lhs):
                candidate = self._scalar_splat_param(rhs)
            elif predicate in ("sgt", "ugt", "sge", "uge") and self._is_lane_index_value(rhs):
                candidate = self._scalar_splat_param(lhs)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)

        if len(candidates) == 1:
            return self.params[candidates[0]]
        if not candidates:
            raise UnsupportedTTIROpError(
                "pointwise contracts require a unique runtime extent scalar from "
                "the mask/index pattern"
            )
        raise UnsupportedTTIROpError(
            "ambiguous runtime extent scalar candidates from mask/index pattern: "
            + ", ".join(f"%{name}" for name in candidates)
        )

    def _validate_flat_mask_policy(self) -> None:
        masks = self._collect_mask_values()
        if len(masks) > 1:
            raise UnsupportedTTIROpError(
                "pointwise contracts require load/store masks to use the same "
                "flat extent predicate"
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


def _as_graph(
    ttir_or_graph: str | TTIRArtifact | NormalizedTTIROpGraph,
) -> tuple[NormalizedTTIROpGraph, TTIRArtifact | None]:
    if isinstance(ttir_or_graph, NormalizedTTIROpGraph):
        return ttir_or_graph, None
    if isinstance(ttir_or_graph, TTIRArtifact):
        return TTIRReader().read(ttir_or_graph.ttir), ttir_or_graph
    if isinstance(ttir_or_graph, str):
        return TTIRReader().read(ttir_or_graph), None
    raise TypeError(f"Unsupported TTIR input type: {type(ttir_or_graph)!r}")


def _validate_supported_subset(graph: NormalizedTTIROpGraph, contract: str) -> None:
    for op in graph.ops:
        if op.name not in _SUPPORTED_OPS:
            raise UnsupportedTTIROpError(f"{op.name} is not supported by contract {contract!r}")
        if op.name == "tt.load" and len(op.operands) < 3:
            raise UnsupportedTTIROpError(
                f"masked tt.load without other is not supported by contract {contract!r}"
            )


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
    payload = json.dumps(kwargs, sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
            supported_contracts=frozenset({"pointwise_minimal", "pointwise_flat"}),
            block_var="bx",
            lane_var="tx",
            block_thread_tag="blockIdx.x",
            lane_thread_tag="threadIdx.x",
        )
    raise UnsupportedTargetPolicyError(
        "No backend policy registered for target kind "
        f"{target_kind!r} and pointwise contracts."
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


def _triton_dtype(dtype: str) -> str:
    mapping = {
        "float16": "fp16",
        "float32": "fp32",
        "float64": "fp64",
        "bfloat16": "bf16",
    }
    return mapping.get(dtype, dtype)
