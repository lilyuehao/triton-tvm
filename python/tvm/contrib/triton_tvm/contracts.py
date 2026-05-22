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
- ``pointwise_flat`` is the canonical M2.5 flat-contiguous pointwise contract.  It
  allows multiple masked stores and multiple outputs.
- ``pointwise_indexed`` is the M3.5 indexed pointwise contract.  It keeps the
  same launch and guarded-store shape as ``pointwise_flat``, but permits a small
  set of affine/broadcast pointer indices from Inductor pointwise kernels.
- ``reduction_minimal`` is the M4 row-wise reduction contract.  It keeps the
  CUDA block/thread launch shell, but uses a single-lane local accumulator for a
  correctness-first reduction subset.
- ``cuda_minimal``, ``cuda_pointwise_flat``, and ``cuda_pointwise_indexed`` are
  compatibility aliases kept while the implementation grows target policies
  beyond CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import tvm

from .errors import TritonTVMContractError, UnsupportedContractError


_CONTRACT_ALIASES = {
    "pointwise_minimal": "pointwise_minimal",
    "pointwise_flat": "pointwise_flat",
    "pointwise_indexed": "pointwise_indexed",
    "reduction_minimal": "reduction_minimal",
    "cuda_minimal": "pointwise_minimal",
    "cuda_pointwise_flat": "pointwise_flat",
    "cuda_pointwise_indexed": "pointwise_indexed",
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
        indexing_kind="flat_contiguous",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=True,
        version="pointwise_v1",
    ),
    "pointwise_indexed": TritonTVMContract(
        name="pointwise_indexed",
        indexing_kind="indexed_pointwise",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=True,
        version="pointwise_indexed_v2",
    ),
    "reduction_minimal": TritonTVMContract(
        name="reduction_minimal",
        indexing_kind="block_reduction",
        memory_model="flat_buffer",
        requires_extent_param=True,
        supports_multiple_outputs=False,
        version="reduction_minimal_v1",
    ),
}


def normalize_triton_tvm_contract(contract: str) -> str:
    """Return the canonical contract name for a public contract or alias."""
    try:
        return _CONTRACT_ALIASES[contract]
    except KeyError as err:
        raise UnsupportedContractError(
            f"Unsupported Triton TVM contract: {contract}"
        ) from err


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
    """Validate the flat-contiguous multi-store pointwise contract."""
    _validate_pointwise_common(irmod, "pointwise_flat")
    for gvar, func in irmod.functions.items():
        _validate_pointwise_flat_body(gvar.name_hint, func)


def validate_pointwise_indexed_contract(irmod: tvm.IRModule) -> None:
    """Validate the indexed pointwise contract."""
    _validate_pointwise_common(irmod, "pointwise_indexed")
    for gvar, func in irmod.functions.items():
        _validate_pointwise_indexed_body(gvar.name_hint, func)


def validate_reduction_minimal_contract(irmod: tvm.IRModule) -> None:
    """Validate the M4 row-wise reduction contract."""
    _validate_pointwise_common(irmod, "reduction_minimal")
    for gvar, func in irmod.functions.items():
        _validate_reduction_minimal_body(gvar.name_hint, func)


def validate_cuda_minimal_contract(irmod: tvm.IRModule) -> None:
    """Validate the legacy CUDA single-store alias."""
    validate_pointwise_minimal_contract(irmod)


def validate_cuda_pointwise_flat_contract(irmod: tvm.IRModule) -> None:
    """Validate the legacy CUDA flat-contiguous alias."""
    validate_pointwise_flat_contract(irmod)


def validate_cuda_pointwise_indexed_contract(irmod: tvm.IRModule) -> None:
    """Validate the legacy CUDA indexed pointwise alias."""
    validate_pointwise_indexed_contract(irmod)


def validate_triton_tvm_contract(irmod: tvm.IRModule, contract: str) -> None:
    """Validate an IRModule against a named Triton TVM contract."""
    contract = normalize_triton_tvm_contract(contract)
    if contract == "pointwise_minimal":
        validate_pointwise_minimal_contract(irmod)
    elif contract == "pointwise_flat":
        validate_pointwise_flat_contract(irmod)
    elif contract == "pointwise_indexed":
        validate_pointwise_indexed_contract(irmod)
    elif contract == "reduction_minimal":
        validate_reduction_minimal_contract(irmod)
    else:
        raise UnsupportedContractError(
            f"Contract {contract!r} does not have a validator in this prototype"
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


def _validate_pointwise_indexed_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    guarded_stores = _store_guard_records(func.body)
    if not guarded_stores:
        raise TritonTVMContractError(
            f"{name} pointwise_indexed contract requires at least one BufferStore"
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
            f"{name} pointwise_indexed stores must use an i < extent mask guard"
        )

    allowed_loads: set[str] = set()
    for store, guard in guarded_stores:
        assert guard is not None
        if _expr_key(guard) != first_guard_key:
            raise TritonTVMContractError(
                f"{name} pointwise_indexed stores must use the same mask guard"
            )
        for index in store.indices:
            _validate_supported_pointwise_index(name, index)
            if _buffer_load_keys(index):
                raise TritonTVMContractError(
                    f"{name} pointwise_indexed store indices must not contain BufferLoad"
                )
        allowed_loads.update(_buffer_load_keys(store.value))

    all_loads = _buffer_load_keys(func.body)
    disallowed_loads = sorted(all_loads - allowed_loads)
    if disallowed_loads:
        raise TritonTVMContractError(
            f"{name} pointwise_indexed direct BufferLoad must appear only in "
            f"guarded store values: {', '.join(disallowed_loads[:3])}"
        )

    for load in _buffer_load_nodes(func.body):
        if not isinstance(load, tirx.BufferLoad):
            continue
        for index in load.indices:
            _validate_supported_pointwise_index(name, index)


def _validate_reduction_minimal_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    stores = [stmt for stmt in _walk_stmt(func.body) if isinstance(stmt, tirx.BufferStore)]
    if not stores:
        raise TritonTVMContractError(
            f"{name} reduction_minimal contract requires at least one BufferStore"
        )

    serial_loops = [
        stmt
        for stmt in _walk_stmt(func.body)
        if isinstance(stmt, tirx.For) and int(stmt.kind) == int(tirx.ForKind.SERIAL)
    ]
    if not serial_loops:
        raise TritonTVMContractError(
            f"{name} reduction_minimal contract requires a serial reduction loop"
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
        f"{name} pointwise_indexed only supports buffer indices i, i % C, i // C, and i * C"
    )


def _is_supported_pointwise_index(expr) -> bool:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    if _is_lane_index_expr(expr):
        return True
    if isinstance(expr, (tirx.FloorMod, tirx.FloorDiv)):
        return _is_lane_index_expr(expr.a) and _positive_int_imm(expr.b) is not None
    if isinstance(expr, tirx.Mul):
        return (
            (_is_lane_index_expr(expr.a) and _positive_int_imm(expr.b) is not None)
            or (_is_lane_index_expr(expr.b) and _positive_int_imm(expr.a) is not None)
        )
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
