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
"""Contract checks for translated Triton-flavored TIRX.

Contract boundary:

- ``pointwise_minimal`` is the canonical single-store pointwise contract.
- ``pointwise_flat`` is the canonical pointwise contract.  It allows multiple
  masked stores and multiple outputs, with M3.5 indexed forms tracked as
  capability metadata instead of a separate contract name.
- ``reduction_minimal`` is the M4 row-wise reduction contract.  It keeps the
  CUDA block/thread launch shell, but uses a single-lane local accumulator for a
  correctness-first reduction subset.
- ``norm_single_row`` is the M4 single-row LN/RMS contract over the same
  correctness-first reduction lowering.
- ``cuda_minimal`` and ``cuda_pointwise_flat`` are deprecated compatibility
  aliases.  They are accepted only at the public boundary and immediately
  canonicalized.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import tvm

from .errors import TritonTVMContractError, UnsupportedContractError


_CONTRACT_ALIASES = {
    "pointwise_minimal": "pointwise_minimal",
    "pointwise_flat": "pointwise_flat",
    "reduction_minimal": "reduction_minimal",
    "norm_single_row": "norm_single_row",
    "row_reduction": "row_reduction",
    "norm_row": "norm_row",
    "softmax_row": "softmax_row",
    "masked_softmax_row": "masked_softmax_row",
    "cuda_minimal": "pointwise_minimal",
    "cuda_pointwise_flat": "pointwise_flat",
}

_LEGACY_CONTRACT_ALIASES = {
    "cuda_minimal": "pointwise_minimal",
    "cuda_pointwise_flat": "pointwise_flat",
}


@dataclass(frozen=True)
class TritonTVMContract:
    """Target-neutral contract metadata for translated pointwise kernels."""

    name: str
    indexing_kind: str
    memory_model: str
    requires_extent_param: bool
    supports_multiple_outputs: bool
    version: str
    execution_kind: str = "thread_parallel_pointwise"
    accumulator_dtype_policy: str = "not_applicable"
    epsilon_policy: str = "not_applicable"
    mask_policy: str = "flat_extent_guard"
    axis_policy: str = "flat_1d"
    layout_policy: str = "flat_contiguous"


_CONTRACTS = {
    "pointwise_minimal": TritonTVMContract(
        name="pointwise_minimal",
        indexing_kind="flat_contiguous",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="pointwise_v1",
    ),
    "pointwise_flat": TritonTVMContract(
        name="pointwise_flat",
        indexing_kind="pointwise",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=True,
        version="pointwise_pre_m5_v1",
    ),
    "reduction_minimal": TritonTVMContract(
        name="reduction_minimal",
        indexing_kind="block_reduction",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="reduction_minimal_v1",
        execution_kind="serial_m4_single_lane",
        accumulator_dtype_policy="preserve_ttir_reduction_dtype",
        epsilon_policy="not_applicable",
        mask_policy="masked_reduction_loads_require_zero_other",
        axis_policy="axis_0_only",
        layout_policy="row_major_only",
    ),
    "norm_single_row": TritonTVMContract(
        name="norm_single_row",
        indexing_kind="single_row_norm",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="norm_single_row_v1",
        execution_kind="serial_m4_single_lane",
        accumulator_dtype_policy="preserve_ttir_reduction_dtype",
        epsilon_policy="runtime_and_constexpr_eps_supported",
        mask_policy="masked_reduction_loads_require_zero_other",
        axis_policy="axis_0_only",
        layout_policy="single_row_row_major",
    ),
    "row_reduction": TritonTVMContract(
        name="row_reduction",
        indexing_kind="rank2_row_reduction",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="row_reduction_m8_v1",
        execution_kind="serial_m8_rank2_lane",
        accumulator_dtype_policy="preserve_ttir_reduction_dtype",
        epsilon_policy="not_applicable",
        mask_policy="rank2_masks_explicit_or_dominated",
        axis_policy="axis_1_only",
        layout_policy="rank2_row_major",
    ),
    "norm_row": TritonTVMContract(
        name="norm_row",
        indexing_kind="rank2_row_norm",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="norm_row_m8_v1",
        execution_kind="serial_m8_rank2_lane",
        accumulator_dtype_policy="preserve_ttir_reduction_dtype",
        epsilon_policy="runtime_and_constexpr_eps_supported",
        mask_policy="rank2_masks_explicit_or_dominated",
        axis_policy="axis_1_only",
        layout_policy="rank2_row_major",
    ),
    "softmax_row": TritonTVMContract(
        name="softmax_row",
        indexing_kind="rank2_row_softmax",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="softmax_row_m8_v1",
        execution_kind="serial_m8_rank2_lane",
        accumulator_dtype_policy="preserve_ttir_reduction_dtype",
        epsilon_policy="not_applicable",
        mask_policy="rank2_masks_explicit_or_dominated",
        axis_policy="axis_1_only",
        layout_policy="rank2_row_major",
    ),
    "masked_softmax_row": TritonTVMContract(
        name="masked_softmax_row",
        indexing_kind="rank2_masked_row_softmax",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="masked_softmax_row_m8_v1",
        execution_kind="serial_m8_rank2_lane",
        accumulator_dtype_policy="preserve_ttir_reduction_dtype",
        epsilon_policy="not_applicable",
        mask_policy="rank2_masks_explicit_or_dominated",
        axis_policy="axis_1_only",
        layout_policy="rank2_row_major_or_causal",
    ),
}


def normalize_triton_tvm_contract(contract: str) -> str:
    """Return the canonical contract name for a public contract or alias."""
    try:
        canonical = _CONTRACT_ALIASES[contract]
    except KeyError as err:
        raise UnsupportedContractError(
            f"Unsupported Triton TVM contract: {contract}"
        ) from err
    if contract in _LEGACY_CONTRACT_ALIASES:
        warnings.warn(
            f"Triton TVM contract {contract!r} is deprecated; use "
            f"{canonical!r} instead.",
            FutureWarning,
            stacklevel=2,
        )
    return canonical


def get_triton_tvm_contract(contract: str) -> TritonTVMContract:
    """Return target-neutral metadata for a public contract or alias."""
    canonical = normalize_triton_tvm_contract(contract)
    return _CONTRACTS[canonical]


def validate_pointwise_minimal_contract(irmod: tvm.IRModule) -> None:
    """Validate the single-store pointwise contract for the active target policy."""
    _validate_pointwise_common(irmod, "pointwise_minimal")
    for gvar, func in irmod.functions.items():
        _validate_pointwise_minimal_body(gvar.name_hint, func)


def validate_pointwise_flat_contract(irmod: tvm.IRModule) -> None:
    """Validate the multi-store pointwise contract and its capability variants."""
    _validate_pointwise_common(irmod, "pointwise_flat")
    for gvar, func in irmod.functions.items():
        _validate_pointwise_flat_body(gvar.name_hint, func)
        _validate_pointwise_capability_body(gvar.name_hint, func)


def validate_reduction_minimal_contract(irmod: tvm.IRModule) -> None:
    """Validate the M4 row-wise reduction contract."""
    _validate_pointwise_common(irmod, "reduction_minimal")
    for gvar, func in irmod.functions.items():
        _validate_reduction_minimal_body(gvar.name_hint, func)


def validate_norm_single_row_contract(irmod: tvm.IRModule) -> None:
    """Validate the M4 single-row LN/RMS contract."""
    _validate_pointwise_common(irmod, "norm_single_row")
    for gvar, func in irmod.functions.items():
        _validate_reduction_minimal_body(gvar.name_hint, func)


def validate_row_reduction_contract(irmod: tvm.IRModule) -> None:
    """Validate the M8 rank-2 row reduction contract."""
    _validate_pointwise_common(irmod, "row_reduction")
    for gvar, func in irmod.functions.items():
        _validate_serial_row_contract_body(gvar.name_hint, func, "row_reduction")


def validate_norm_row_contract(irmod: tvm.IRModule) -> None:
    """Validate the M8 rank-2 row norm contract."""
    _validate_pointwise_common(irmod, "norm_row")
    for gvar, func in irmod.functions.items():
        _validate_serial_row_contract_body(gvar.name_hint, func, "norm_row")


def validate_softmax_row_contract(irmod: tvm.IRModule) -> None:
    """Validate the M8 rank-2 row softmax contract."""
    _validate_pointwise_common(irmod, "softmax_row")
    for gvar, func in irmod.functions.items():
        _validate_serial_row_contract_body(gvar.name_hint, func, "softmax_row")


def validate_masked_softmax_row_contract(irmod: tvm.IRModule) -> None:
    """Validate the M8 rank-2 masked row softmax contract."""
    _validate_pointwise_common(irmod, "masked_softmax_row")
    for gvar, func in irmod.functions.items():
        _validate_serial_row_contract_body(gvar.name_hint, func, "masked_softmax_row")


def validate_cuda_minimal_contract(irmod: tvm.IRModule) -> None:
    """Validate the legacy CUDA single-store alias."""
    _warn_legacy_contract("cuda_minimal", "pointwise_minimal")
    validate_pointwise_minimal_contract(irmod)


def validate_cuda_pointwise_flat_contract(irmod: tvm.IRModule) -> None:
    """Validate the legacy CUDA flat-contiguous alias."""
    _warn_legacy_contract("cuda_pointwise_flat", "pointwise_flat")
    validate_pointwise_flat_contract(irmod)


def validate_triton_tvm_contract(irmod: tvm.IRModule, contract: str) -> None:
    """Validate an IRModule against a named Triton TVM contract."""
    contract = normalize_triton_tvm_contract(contract)
    if contract == "pointwise_minimal":
        validate_pointwise_minimal_contract(irmod)
    elif contract == "pointwise_flat":
        validate_pointwise_flat_contract(irmod)
    elif contract == "reduction_minimal":
        validate_reduction_minimal_contract(irmod)
    elif contract == "norm_single_row":
        validate_norm_single_row_contract(irmod)
    elif contract == "row_reduction":
        validate_row_reduction_contract(irmod)
    elif contract == "norm_row":
        validate_norm_row_contract(irmod)
    elif contract == "softmax_row":
        validate_softmax_row_contract(irmod)
    elif contract == "masked_softmax_row":
        validate_masked_softmax_row_contract(irmod)
    else:
        raise UnsupportedContractError(
            f"Contract {contract!r} does not have a validator in this prototype"
        )


def _warn_legacy_contract(alias: str, canonical: str) -> None:
    warnings.warn(
        f"Triton TVM contract {alias!r} is deprecated; use {canonical!r} instead.",
        FutureWarning,
        stacklevel=2,
    )


def _validate_pointwise_common(irmod: tvm.IRModule, contract: str) -> None:
    if irmod.attrs is not None and irmod.attrs.get("external_mods", None) is not None:
        raise TritonTVMContractError(f"{contract} contract must not use external_mods")

    if not irmod.functions:
        raise TritonTVMContractError(f"{contract} contract requires at least one PrimFunc")

    for gvar, func in irmod.functions.items():
        if not isinstance(func, tvm.tirx.PrimFunc):
            raise TritonTVMContractError(f"{gvar.name_hint} is not a tirx.PrimFunc")
        attrs = func.attrs
        if attrs is None:
            raise TritonTVMContractError(f"{gvar.name_hint} is missing function attrs")
        target = attrs.get("target", None)
        if target is None or tvm.target.Target(target).kind.name != "cuda":
            raise TritonTVMContractError(
                f"{gvar.name_hint} must use the supported Triton TVM target policy 'cuda'"
            )
        if not bool(attrs.get("tirx.noalias", False)):
            raise TritonTVMContractError(f"{gvar.name_hint} must set tirx.noalias")
        if attrs.get("global_symbol", None) is None:
            raise TritonTVMContractError(f"{gvar.name_hint} must set global_symbol")

        _validate_cuda_launch_body(gvar.name_hint, func)


def _validate_cuda_launch_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    body = func.body
    if isinstance(body, tirx.SeqStmt):
        body = next(
            (stmt for stmt in body.seq if _is_thread_for(stmt, "blockIdx.x")),
            body,
        )
    if not _is_thread_for(body, "blockIdx.x"):
        raise TritonTVMContractError(f"{name} must bind the outer loop to blockIdx.x")

    inner = body.body
    if not _is_thread_for(inner, "threadIdx.x"):
        raise TritonTVMContractError(f"{name} must bind the inner loop to threadIdx.x")


def _validate_pointwise_minimal_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    body = func.body
    stores = [stmt for stmt in _walk_stmt(body) if isinstance(stmt, tirx.BufferStore)]
    if len(stores) != 1:
        raise TritonTVMContractError(
            f"{name} pointwise_minimal contract requires exactly one BufferStore, "
            f"got {len(stores)}"
        )

    if not _store_guard_states(body)[0][1]:
        raise TritonTVMContractError(f"{name} must guard the store with an IfThenElse mask")


def _validate_pointwise_flat_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    guarded_stores = _store_guard_states(func.body)
    if not guarded_stores:
        raise TritonTVMContractError(
            f"{name} pointwise_flat contract requires at least one BufferStore"
        )

    unguarded = sum(1 for _, guarded in guarded_stores if not guarded)
    if unguarded:
        raise TritonTVMContractError(
            f"{name} must guard every BufferStore with an IfThenElse mask"
        )


def _validate_pointwise_capability_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    guarded_stores = _store_guard_records(func.body)
    if not guarded_stores:
        raise TritonTVMContractError(
            f"{name} pointwise_flat contract requires at least one BufferStore"
        )

    unguarded = sum(1 for _, guard in guarded_stores if guard is None)
    if unguarded:
        raise TritonTVMContractError(
            f"{name} must guard every BufferStore with an IfThenElse mask"
        )

    first_guard = guarded_stores[0][1]
    assert first_guard is not None
    first_guard_key = _expr_key(first_guard)
    if not _is_supported_pointwise_mask_guard(first_guard):
        raise TritonTVMContractError(
            f"{name} pointwise_flat stores must use an i < extent mask guard"
        )

    allowed_loads: set[str] = set()
    for store, guard in guarded_stores:
        assert guard is not None
        if _expr_key(guard) != first_guard_key:
            raise TritonTVMContractError(
                f"{name} pointwise_flat stores must use the same mask guard"
            )
        for index in store.indices:
            _validate_supported_pointwise_index(name, index)
            if _buffer_load_keys(index):
                raise TritonTVMContractError(
                    f"{name} pointwise_flat store indices must not contain BufferLoad"
                )
        allowed_loads.update(_buffer_load_keys(store.value))

    all_loads = _buffer_load_keys(func.body)
    disallowed_loads = sorted(all_loads - allowed_loads)
    if disallowed_loads:
        raise TritonTVMContractError(
            f"{name} pointwise_flat direct BufferLoad must appear only in "
            f"guarded store values: {', '.join(disallowed_loads[:3])}"
        )

    for load in _buffer_load_nodes(func.body):
        if not isinstance(load, tirx.BufferLoad):
            continue
        for index in load.indices:
            _validate_supported_pointwise_index(name, index)


def _validate_reduction_minimal_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    _validate_serial_row_contract_body(name, func, "reduction_minimal")


def _validate_serial_row_contract_body(
    name: str, func: tvm.tirx.PrimFunc, contract: str
) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    stores = [stmt for stmt in _walk_stmt(func.body) if isinstance(stmt, tirx.BufferStore)]
    if not stores:
        raise TritonTVMContractError(
            f"{name} {contract} contract requires at least one BufferStore"
        )

    serial_loops = [
        stmt
        for stmt in _walk_stmt(func.body)
        if isinstance(stmt, tirx.For) and int(stmt.kind) == int(tirx.ForKind.SERIAL)
    ]
    if not serial_loops:
        raise TritonTVMContractError(
            f"{name} {contract} contract requires a serial reduction loop"
        )


def _is_thread_for(stmt, thread_tag: str) -> bool:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    return (
        isinstance(stmt, tirx.For)
        and int(stmt.kind) == int(tirx.ForKind.THREAD_BINDING)
        and stmt.thread_binding is not None
        and stmt.thread_binding.thread_tag == thread_tag
    )


def _walk_stmt(stmt):
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    yield stmt
    if isinstance(stmt, tirx.For):
        yield from _walk_stmt(stmt.body)
    elif isinstance(stmt, tirx.SeqStmt):
        for child in stmt.seq:
            yield from _walk_stmt(child)
    elif isinstance(stmt, tirx.IfThenElse):
        yield from _walk_stmt(stmt.then_case)
        if stmt.else_case is not None:
            yield from _walk_stmt(stmt.else_case)
    elif isinstance(stmt, tirx.While):
        yield from _walk_stmt(stmt.body)
    elif hasattr(tirx, "AttrStmt") and isinstance(stmt, tirx.AttrStmt):
        yield from _walk_stmt(stmt.body)


def _store_guard_states(stmt, guarded: bool = False):
    records = _store_guard_records(stmt, True if guarded else None)
    return [(store, guard is not None) for store, guard in records]


def _store_guard_records(stmt, guard=None):
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    if isinstance(stmt, tirx.BufferStore):
        return [(stmt, guard)]
    if isinstance(stmt, tirx.For):
        return _store_guard_records(stmt.body, guard)
    if isinstance(stmt, tirx.SeqStmt):
        stores = []
        for child in stmt.seq:
            stores.extend(_store_guard_records(child, guard))
        return stores
    if isinstance(stmt, tirx.IfThenElse):
        stores = _store_guard_records(stmt.then_case, stmt.condition)
        if stmt.else_case is not None:
            stores.extend(_store_guard_records(stmt.else_case, guard))
        return stores
    if isinstance(stmt, tirx.While):
        return _store_guard_records(stmt.body, guard)
    if hasattr(tirx, "AttrStmt") and isinstance(stmt, tirx.AttrStmt):
        return _store_guard_records(stmt.body, guard)
    return []


def _expr_key(expr) -> str:
    return str(expr)


def _buffer_load_nodes(node) -> list:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    loads = []

    def _visit(visited):
        if isinstance(visited, tirx.BufferLoad):
            loads.append(visited)

    tirx.stmt_functor.post_order_visit(node, _visit)
    return loads


def _buffer_load_keys(node) -> set[str]:
    return {_expr_key(load) for load in _buffer_load_nodes(node)}


def _validate_supported_pointwise_index(name: str, expr) -> None:
    if _is_supported_pointwise_index(expr):
        return
    raise TritonTVMContractError(
        f"{name} pointwise_flat only supports classified M7 pointwise index expressions"
    )


def _is_supported_pointwise_index(expr) -> bool:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    if _is_lane_index_expr(expr):
        return True
    if isinstance(expr, tirx.IntImm):
        return int(expr.value) >= 0
    if isinstance(expr, tirx.Cast):
        return _is_supported_pointwise_index(expr.value)
    if isinstance(expr, (tirx.Add, tirx.Sub)):
        return _is_supported_pointwise_index(expr.a) and _is_supported_pointwise_index(expr.b)
    if isinstance(expr, (tirx.FloorMod, tirx.FloorDiv)):
        return _is_supported_pointwise_index(expr.a) and _positive_int_imm(expr.b) is not None
    if isinstance(expr, tirx.Mul):
        if _positive_int_imm(expr.a) is not None:
            return _is_supported_pointwise_index(expr.b)
        if _positive_int_imm(expr.b) is not None:
            return _is_supported_pointwise_index(expr.a)
        if _positive_float_imm(expr.a) is not None:
            return _is_supported_pointwise_index(expr.b)
        if _positive_float_imm(expr.b) is not None:
            return _is_supported_pointwise_index(expr.a)
    return False


def _is_supported_pointwise_mask_guard(expr) -> bool:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    if isinstance(expr, tirx.Var) and str(expr).startswith("mask"):
        return True
    return isinstance(expr, (tirx.LT, tirx.LE)) and _is_lane_index_expr(expr.a)


def _is_lane_index_expr(expr) -> bool:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    return isinstance(expr, tirx.Var) and str(expr) == "i"


def _positive_int_imm(expr) -> int | None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    if not isinstance(expr, tirx.IntImm):
        return None
    value = int(expr.value)
    if value <= 0:
        return None
    return value


def _positive_float_imm(expr) -> float | None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    if not isinstance(expr, tirx.FloatImm):
        return None
    value = float(expr.value)
    if value <= 0.0:
        return None
    return value
