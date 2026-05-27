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
"""Adapter passes for the Triton-to-TVM prototype."""

from __future__ import annotations

import ast
import keyword
import re

import tvm

from .contracts import validate_triton_tvm_contract
from .matmul import (
    M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
)


def ValidateTritonKernelTIR(contract: str = "pointwise_minimal"):
    """Return the validation pass for a Triton TVM contract.

    Pre-M5 keeps validation separate from normalization.  The translator emits
    canonical TIRX directly today; a pass named ``NormalizeTritonKernelTIR``
    should only return once it performs real IR rewrites.
    """

    @tvm.ir.transform.module_pass(opt_level=0, name="triton_tvm.ValidateTritonKernelTIR")
    def _pass(mod, _ctx):  # pylint: disable=unused-argument
        validate_triton_tvm_contract(mod, contract)
        return mod

    return _pass


def M129OptimizeNativeWrapperMatmul():
    """Return the M12.9 backend pass for fixed-shape ViT wrapper matmuls.

    The M12.2 native provider intentionally used a correctness-first CUDA
    schedule with one output element per block and a single active thread.  This
    opt-in pass keeps the same provider contract and rewrites only matching
    M12 fixed-shape wrapper matmul/addmm PrimFuncs into a row-threaded schedule:
    one CUDA block per output row, one thread per output column, and serial-K
    accumulation per output element.
    """

    @tvm.ir.transform.module_pass(opt_level=0, name="triton_tvm.M129OptimizeNativeWrapperMatmul")
    def _pass(mod, _ctx):  # pylint: disable=unused-argument
        updated = tvm.IRModule(dict(mod.functions), attrs=mod.attrs)
        changed = False
        for gvar, func in mod.functions.items():
            if not _is_m129_native_wrapper_matmul_candidate(func):
                continue
            source = _build_m129_row_threaded_wrapper_matmul_source(gvar.name_hint, func)
            replacement = next(iter(tvm.script.from_source(source).functions.values()))
            updated.update_func(gvar, replacement)
            changed = True
        if changed:
            updated = updated.with_attr("triton_tvm.m12_9_backend_pass", "row_threaded_matmul")
        return updated

    return _pass


def _is_m129_native_wrapper_matmul_candidate(func) -> bool:
    attrs = getattr(func, "attrs", None) or {}
    return (
        str(attrs.get("triton_tvm.contract", "")) == "matmul_minimal"
        and str(attrs.get("triton_tvm.implementation_kind", "")) == "native_tir_schedule"
        and str(attrs.get("triton_tvm.schedule_id", "")) == NATIVE_TIR_MATMUL_SCHEDULE_ID
        and str(attrs.get("triton_tvm.matmul_source_kind", ""))
        in {"wrapper_extern_gemm", "wrapper_extern_addmm_bias"}
    )


def _build_m129_row_threaded_wrapper_matmul_source(name_hint: str, func) -> str:
    attrs = func.attrs or {}
    source_kind = _attr_str(attrs, "triton_tvm.matmul_source_kind")
    is_addmm = source_kind == "wrapper_extern_addmm_bias"
    m = _attr_int(attrs, "triton_tvm.matmul_m")
    n = _attr_int(attrs, "triton_tvm.matmul_n")
    k = _attr_int(attrs, "triton_tvm.matmul_k")
    a_stride = _attr_tuple(attrs, "triton_tvm.a_stride")
    b_storage_shape = _attr_tuple(attrs, "triton_tvm.b_storage_shape")
    b_storage_stride = _attr_tuple(attrs, "triton_tvm.b_storage_stride")
    c_stride = _attr_tuple(attrs, "triton_tvm.c_stride")
    transposed_b = _attr_bool(attrs, "triton_tvm.transposed_b")
    func_name = _sanitize_identifier(
        _attr_str(attrs, "global_symbol", default=name_hint) or name_hint
    )
    b_read = "b[vn, vk]" if transposed_b else "b[vk, vn]"
    reads = f"T.reads(a[vm, vk], {b_read})"
    init_value = "T.float32(0)"
    signature = "a_handle: T.handle, b_handle: T.handle, c_handle: T.handle"
    bias_buffer = ""
    if is_addmm:
        bias_shape = _attr_tuple(attrs, "triton_tvm.bias_shape")
        bias_stride = _attr_tuple(attrs, "triton_tvm.bias_stride")
        signature = (
            "bias_handle: T.handle, a_handle: T.handle, b_handle: T.handle, "
            "c_handle: T.handle"
        )
        bias_buffer = (
            f'        bias = T.match_buffer(bias_handle, ({_tuple_items(bias_shape)}), '
            f'"float32", strides=({_tuple_items(bias_stride)}))'
        )
        if len(bias_shape) == 1:
            reads = f"T.reads(a[vm, vk], {b_read}, bias[vn])"
            init_value = "bias[vn]"
        else:
            reads = f"T.reads(a[vm, vk], {b_read}, bias[vm, vn])"
            init_value = "bias[vm, vn]"

    lines = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}({signature}):",
        "        T.func_attr({"
        + ", ".join(_m129_func_attr_lines(attrs, source_kind, m, n, k))
        + "})",
    ]
    if bias_buffer:
        lines.append(bias_buffer)
    lines.extend(
        [
            f'        a = T.match_buffer(a_handle, ({m}, {k}), "float32", '
            f"strides=({_tuple_items(a_stride)}))",
            f'        b = T.match_buffer(b_handle, ({b_storage_shape[0]}, {b_storage_shape[1]}), '
            f'"float32", strides=({_tuple_items(b_storage_stride)}))',
            f'        c = T.match_buffer(c_handle, ({m}, {n}), "float32", '
            f"strides=({_tuple_items(c_stride)}))",
            f'        for blockIdx_x in T.thread_binding(0, {m}, thread="blockIdx.x"):',
            f'            for threadIdx_x in T.thread_binding(0, {n}, thread="threadIdx.x"):',
            "                mi = blockIdx_x",
            "                ni = threadIdx_x",
            f"                for kk in T.serial(0, {k}):",
            '                    with T.sblock("matmul"):',
            f"                        vm = T.axis.spatial({m}, mi)",
            f"                        vn = T.axis.spatial({n}, ni)",
            f"                        vk = T.axis.reduce({k}, kk)",
            f"                        {reads}",
            "                        T.writes(c[vm, vn])",
            "                        T.sblock_attr({"
            + ", ".join(_m129_block_attr_lines(attrs, source_kind, m, n, k))
            + "})",
            "                        with T.init():",
            f"                            c[vm, vn] = {init_value}",
            "                        c[vm, vn] = c[vm, vn] + "
            f'T.Cast("float32", a[vm, vk]) * T.Cast("float32", {b_read})',
        ]
    )
    return "\n".join(lines) + "\n"


def _m129_func_attr_lines(attrs, source_kind: str, m: int, n: int, k: int) -> list[str]:
    string_keys = [
        "global_symbol",
        "triton_tvm.contract",
        "triton_tvm.matmul_source_kind",
        "triton_tvm.a_dtype",
        "triton_tvm.b_dtype",
        "triton_tvm.output_dtype",
        "triton_tvm.accumulator_dtype",
        "triton_tvm.input_precision",
        "triton_tvm.tf32_policy",
        "triton_tvm.bounds_policy",
        "triton_tvm.mask_kind",
        "triton_tvm.epilogue_kind",
        "triton_tvm.alias_policy",
        "triton_tvm.implementation_kind",
        "triton_tvm.extern_symbol",
        "triton_tvm.extern_packed_func",
        "triton_tvm.extern_runtime_kind",
        "triton_tvm.extern_runtime_replacement",
        "triton_tvm.extern_runtime_replacement_reason",
        "triton_tvm.extern_gemm_runtime_status",
        "triton_tvm.extern_gemm_provider_kind",
        "triton_tvm.extern_gemm_runtime_claim",
        "triton_tvm.a_layout",
        "triton_tvm.b_layout",
        "triton_tvm.c_layout",
        "triton_tvm.a_stride",
        "triton_tvm.b_stride",
        "triton_tvm.c_stride",
        "triton_tvm.b_storage_shape",
        "triton_tvm.b_storage_stride",
    ]
    lines = ['"tirx.noalias": True', '"target": T.target("cuda")']
    for key in string_keys:
        lines.append(f'"{key}": {_quote(_attr_str(attrs, key))}')
    lines.extend(
        [
            f'"triton_tvm.schedule_id": {_quote(M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID)}',
            '"triton_tvm.extern_runtime_replacement_available": True',
            f'"triton_tvm.extern_gemm_provider_abi_version": '
            f'{_attr_int(attrs, "triton_tvm.extern_gemm_provider_abi_version")}',
            '"triton_tvm.extern_gemm_performance_claim": False',
            '"triton_tvm.extern_gemm_uses_host_staging": False',
            f'"triton_tvm.matmul_m": {m}',
            f'"triton_tvm.matmul_n": {n}',
            f'"triton_tvm.matmul_k": {k}',
            f'"triton_tvm.transposed_b": {_bool_literal(_attr_bool(attrs, "triton_tvm.transposed_b"))}',
            '"triton_tvm.m12_9_backend_pass": "row_threaded_wrapper_matmul"',
        ]
    )
    if source_kind == "wrapper_extern_addmm_bias":
        lines.extend(
            [
                f'"triton_tvm.bias_param": {_quote(_attr_str(attrs, "triton_tvm.bias_param"))}',
                f'"triton_tvm.bias_shape": {_quote(_attr_str(attrs, "triton_tvm.bias_shape"))}',
                f'"triton_tvm.bias_stride": {_quote(_attr_str(attrs, "triton_tvm.bias_stride"))}',
                f'"triton_tvm.bias_rank": {_attr_int(attrs, "triton_tvm.bias_rank")}',
                f'"triton_tvm.alpha": {_quote(_attr_str(attrs, "triton_tvm.alpha"))}',
                f'"triton_tvm.beta": {_quote(_attr_str(attrs, "triton_tvm.beta"))}',
            ]
        )
    return lines


def _m129_block_attr_lines(attrs, source_kind: str, m: int, n: int, k: int) -> list[str]:
    lines = [
        '"triton_tvm.contract": "matmul_minimal"',
        f'"triton_tvm.matmul_source_kind": {_quote(source_kind)}',
        f'"triton_tvm.accumulator_dtype": {_quote(_attr_str(attrs, "triton_tvm.accumulator_dtype"))}',
        f'"triton_tvm.implementation_kind": {_quote(_attr_str(attrs, "triton_tvm.implementation_kind"))}',
        f'"triton_tvm.schedule_id": {_quote(M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID)}',
        f'"triton_tvm.matmul_m": {m}',
        f'"triton_tvm.matmul_n": {n}',
        f'"triton_tvm.matmul_k": {k}',
        f'"triton_tvm.a_layout": {_quote(_attr_str(attrs, "triton_tvm.a_layout"))}',
        f'"triton_tvm.b_layout": {_quote(_attr_str(attrs, "triton_tvm.b_layout"))}',
        f'"triton_tvm.c_layout": {_quote(_attr_str(attrs, "triton_tvm.c_layout"))}',
        f'"triton_tvm.input_precision": {_quote(_attr_str(attrs, "triton_tvm.input_precision"))}',
        f'"triton_tvm.bounds_policy": {_quote(_attr_str(attrs, "triton_tvm.bounds_policy"))}',
        f'"triton_tvm.mask_kind": {_quote(_attr_str(attrs, "triton_tvm.mask_kind"))}',
        f'"triton_tvm.epilogue_kind": {_quote(_attr_str(attrs, "triton_tvm.epilogue_kind"))}',
        '"triton_tvm.m12_9_backend_pass": "row_threaded_wrapper_matmul"',
    ]
    if source_kind == "wrapper_extern_addmm_bias":
        lines.extend(
            [
                f'"triton_tvm.bias_shape": {_quote(_attr_str(attrs, "triton_tvm.bias_shape"))}',
                f'"triton_tvm.bias_stride": {_quote(_attr_str(attrs, "triton_tvm.bias_stride"))}',
                f'"triton_tvm.bias_rank": {_attr_int(attrs, "triton_tvm.bias_rank")}',
            ]
        )
    return lines


def _attr_str(attrs, name: str, *, default: str = "") -> str:
    value = attrs.get(name, default)
    return "" if value is None else str(value)


def _attr_int(attrs, name: str) -> int:
    return int(attrs[name])


def _attr_bool(attrs, name: str) -> bool:
    value = attrs[name]
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"1", "true", "t"}:
        return True
    if text in {"0", "false", "f"}:
        return False
    return bool(value)


def _attr_tuple(attrs, name: str) -> tuple[int, ...]:
    value = attrs[name]
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, int):
        return (int(parsed),)
    return tuple(int(item) for item in parsed)


def _tuple_items(values: tuple[int, ...]) -> str:
    if len(values) == 1:
        return f"{values[0]},"
    return ", ".join(str(value) for value in values)


def _quote(value: str) -> str:
    return repr(str(value))


def _bool_literal(value: bool) -> str:
    return "True" if value else "False"


def _sanitize_identifier(value: str) -> str:
    ident = re.sub(r"\W", "_", value)
    if not ident or ident[0].isdigit() or keyword.iskeyword(ident):
        ident = f"kernel_{ident}"
    return ident
