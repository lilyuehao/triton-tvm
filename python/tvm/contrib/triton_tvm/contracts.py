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
- ``matmul_minimal`` is the M9 semantic matmul contract.  It validates the
  unresolved semantic ``matmul`` block, the M9.3 correctness-first native
  schedule selected after target policy, and the M9.4 explicit wrapper extern
  GEMM packed-call artifact.
- ``cuda_minimal`` and ``cuda_pointwise_flat`` are deprecated compatibility
  aliases.  They are accepted only at the public boundary and immediately
  canonicalized.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import tvm

from .errors import TritonTVMContractError, UnsupportedContractError
from .matmul import (
    EXTERN_GEMM_PACKED_FUNC,
    EXTERN_GEMM_PROVIDER_ABI_VERSION,
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    EXTERN_GEMM_RUNTIME_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
    EXTERN_GEMM_RUNTIME_REPLACEMENT,
    EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
    EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    EXTERN_GEMM_SYMBOL,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_TILE_M,
    TILED_TIR_MATMUL_TILE_N,
)


_CONTRACT_ALIASES = {
    "pointwise_minimal": "pointwise_minimal",
    "pointwise_flat": "pointwise_flat",
    "reduction_minimal": "reduction_minimal",
    "norm_single_row": "norm_single_row",
    "row_reduction": "row_reduction",
    "norm_row": "norm_row",
    "softmax_row": "softmax_row",
    "masked_softmax_row": "masked_softmax_row",
    "matmul_minimal": "matmul_minimal",
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
    "matmul_minimal": TritonTVMContract(
        name="matmul_minimal",
        indexing_kind="rank2_matmul",
        memory_model="rank2_row_major_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version="matmul_minimal_m9_v1",
        execution_kind="semantic_then_native_m9_matmul_sblock",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="spatial_mn_reduce_k",
        layout_policy="rank2_row_major",
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


def validate_matmul_minimal_contract(irmod: tvm.IRModule) -> None:
    """Validate the M9 semantic/native/extern-artifact matmul contract."""
    _validate_semantic_common(irmod, "matmul_minimal")
    for gvar, func in irmod.functions.items():
        _validate_matmul_minimal_body(gvar.name_hint, func)


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
    elif contract == "matmul_minimal":
        validate_matmul_minimal_contract(irmod)
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
    _validate_common_func_attrs(irmod, contract)
    for gvar, func in irmod.functions.items():
        _validate_cuda_launch_body(gvar.name_hint, func)


def _validate_semantic_common(irmod: tvm.IRModule, contract: str) -> None:
    _validate_common_func_attrs(irmod, contract)


def _validate_common_func_attrs(irmod: tvm.IRModule, contract: str) -> None:
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


def _validate_matmul_minimal_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    attrs = func.attrs or {}
    if str(attrs.get("triton_tvm.implementation_kind", "")) == "extern_gemm":
        _validate_matmul_extern_gemm_artifact(name, func)
        return
    func_dims = _required_matmul_dims(name, attrs, "function attrs")

    try:
        sch = tvm.s_tir.Schedule(func)
        block_rv = sch.get_sblock("matmul")
        loops = sch.get_loops(block_rv)
        block = sch.get(block_rv)
    except Exception as err:  # pylint: disable=broad-except
        raise TritonTVMContractError(
            f"{name} matmul_minimal contract requires a matchable matmul block"
        ) from err

    if len(loops) != 3:
        raise TritonTVMContractError(
            f"{name} matmul_minimal contract requires exactly 3 matmul loops, "
            f"got {len(loops)}"
        )
    if len(block.iter_vars) != 3:
        raise TritonTVMContractError(
            f"{name} matmul_minimal contract requires 3 block axes"
        )
    iter_types = [int(iter_var.iter_type) for iter_var in block.iter_vars]
    if iter_types != [0, 0, 2]:
        raise TritonTVMContractError(
            f"{name} matmul_minimal axes must be spatial, spatial, reduction"
        )
    if block.init is None:
        raise TritonTVMContractError(f"{name} matmul_minimal requires T.init")
    if len(block.reads) != 2 or len(block.writes) != 1:
        raise TritonTVMContractError(
            f"{name} matmul_minimal requires two reads and one write"
        )

    attrs = block.annotations
    if attrs is None or attrs.get("triton_tvm.contract", None) != "matmul_minimal":
        raise TritonTVMContractError(
            f"{name} matmul block must preserve triton_tvm.contract=matmul_minimal"
        )
    if attrs.get("triton_tvm.matmul_source_kind", None) != "tt_dot":
        raise TritonTVMContractError(
            f"{name} matmul block must preserve source kind tt_dot"
        )
    block_dims = _required_matmul_dims(name, attrs, "block attrs")
    if block_dims != func_dims:
        raise TritonTVMContractError(
            f"{name} matmul block dims must match function attrs"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.accumulator_dtype",
        "float32",
        "matmul block",
    )
    _require_attr_value(name, attrs, "triton_tvm.bounds_policy", "exact", "matmul block")
    _require_attr_value(name, attrs, "triton_tvm.mask_kind", "none", "matmul block")
    _require_attr_value(name, attrs, "triton_tvm.epilogue_kind", "none", "matmul block")
    _validate_matmul_block_axes(name, block, block_dims)
    _validate_matmul_buffer_regions(name, block, block_dims)
    _validate_matmul_stores(name, block)
    implementation_kind = str(attrs.get("triton_tvm.implementation_kind", ""))
    if implementation_kind == "unresolved":
        _validate_matmul_loop_extents(name, sch, loops, block_dims, implementation_kind)
        return
    if implementation_kind == "native_tir_schedule":
        schedule_id = str(attrs.get("triton_tvm.schedule_id", ""))
        if schedule_id not in (NATIVE_TIR_MATMUL_SCHEDULE_ID, TILED_TIR_MATMUL_SCHEDULE_ID):
            raise TritonTVMContractError(
                f"{name} native matmul schedule has unknown triton_tvm.schedule_id"
            )
        func_schedule_id = str(func.attrs.get("triton_tvm.schedule_id", ""))
        if func_schedule_id != schedule_id:
            raise TritonTVMContractError(
                f"{name} native matmul function attrs must match block schedule_id"
            )
        if not _is_thread_for(sch.get(loops[0]), "blockIdx.x"):
            raise TritonTVMContractError(
                f"{name} native matmul schedule must bind the outer loop to blockIdx.x"
            )
        if not _is_thread_for(sch.get(loops[1]), "threadIdx.x"):
            raise TritonTVMContractError(
                f"{name} native matmul schedule must bind the inner loop to threadIdx.x"
            )
        _validate_matmul_schedule_attrs(name, attrs, block_dims, schedule_id)
        _validate_matmul_loop_extents(
            name,
            sch,
            loops,
            block_dims,
            implementation_kind,
            schedule_id=schedule_id,
        )
        return
    raise TritonTVMContractError(
        f"{name} matmul block has unsupported implementation_kind={implementation_kind!r}"
    )


def _validate_matmul_extern_gemm_artifact(name: str, func: tvm.tirx.PrimFunc) -> None:
    attrs = func.attrs or {}
    if attrs.get("triton_tvm.contract", None) != "matmul_minimal":
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact must preserve triton_tvm.contract=matmul_minimal"
        )
    if attrs.get("triton_tvm.matmul_source_kind", None) != "wrapper_extern_gemm":
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact must preserve source kind wrapper_extern_gemm"
        )
    if str(attrs.get("triton_tvm.extern_symbol", "")) != EXTERN_GEMM_SYMBOL:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact must preserve extern_symbol={EXTERN_GEMM_SYMBOL}"
        )
    dims = _required_matmul_dims(name, attrs, "extern GEMM artifact attrs")
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_packed_func",
        EXTERN_GEMM_PACKED_FUNC,
        "extern GEMM artifact",
    )
    _validate_matmul_extern_runtime_metadata(name, attrs)
    for attr_name, expected in (
        ("triton_tvm.accumulator_dtype", "float32"),
        ("triton_tvm.output_dtype", "float32"),
        ("triton_tvm.bounds_policy", "exact"),
        ("triton_tvm.mask_kind", "none"),
        ("triton_tvm.epilogue_kind", "none"),
        ("triton_tvm.a_layout", "row_major"),
        ("triton_tvm.c_layout", "row_major"),
    ):
        _require_attr_value(name, attrs, attr_name, expected, "extern GEMM artifact")
    b_layout = str(attrs.get("triton_tvm.b_layout", ""))
    if b_layout not in ("row_major", "transposed_weight_view"):
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact has unsupported b_layout={b_layout!r}"
        )
    _validate_matmul_extern_buffers(name, func, dims, b_layout, attrs)

    script = _prim_func_script(func)
    packed_call = f'T.call_packed("{EXTERN_GEMM_PACKED_FUNC}"'
    if script.count(packed_call) != 1:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact must contain exactly one "
            f"{EXTERN_GEMM_PACKED_FUNC} packed call"
        )
    if "call_extern" in script:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact must not use T.call_extern"
        )
    if 'thread="blockIdx.x"' in script or 'thread="threadIdx.x"' in script:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact must not contain a native CUDA schedule"
        )


def _required_matmul_dims(name: str, attrs, label: str) -> tuple[int, int, int]:
    dims = []
    for dim_name in ("triton_tvm.matmul_m", "triton_tvm.matmul_n", "triton_tvm.matmul_k"):
        value = _int_attr(attrs, dim_name)
        if value is None:
            raise TritonTVMContractError(f"{name} {label} missing {dim_name}")
        if value <= 0:
            raise TritonTVMContractError(f"{name} {label} {dim_name} must be positive")
        dims.append(value)
    return tuple(dims)  # type: ignore[return-value]


def _int_attr(attrs, attr_name: str) -> int | None:
    value = attrs.get(attr_name, None)
    if value is None:
        return None
    value = _int_imm_value(value)
    return value


def _bool_attr(attrs, attr_name: str) -> bool | None:
    value = attrs.get(attr_name, None)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    int_value = _int_imm_value(value)
    if int_value is not None:
        return bool(int_value)
    text = str(value).lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def _require_attr_value(
    name: str,
    attrs,
    attr_name: str,
    expected: str,
    label: str,
) -> None:
    actual = attrs.get(attr_name, None)
    if actual is None:
        raise TritonTVMContractError(f"{name} {label} missing {attr_name}")
    if str(actual) != expected:
        raise TritonTVMContractError(
            f"{name} {label} requires {attr_name}={expected}"
        )


def _validate_matmul_block_axes(name: str, block, dims: tuple[int, int, int]) -> None:
    for axis_name, iter_var, expected in zip(("m", "n", "k"), block.iter_vars, dims):
        extent = None
        if getattr(iter_var, "dom", None) is not None:
            extent = _int_imm_value(iter_var.dom.extent)
        if extent != expected:
            raise TritonTVMContractError(
                f"{name} matmul {axis_name} axis extent must be {expected}"
            )


def _validate_matmul_loop_extents(
    name: str,
    sch,
    loops,
    dims: tuple[int, int, int],
    implementation_kind: str,
    *,
    schedule_id: str = "",
) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    m, n, k = dims
    if implementation_kind == "unresolved":
        expected_extents = (m, n, k)
        expected_kinds = (tirx.ForKind.SERIAL, tirx.ForKind.SERIAL, tirx.ForKind.SERIAL)
    elif schedule_id == TILED_TIR_MATMUL_SCHEDULE_ID:
        expected_extents = (
            (m // TILED_TIR_MATMUL_TILE_M) * (n // TILED_TIR_MATMUL_TILE_N),
            TILED_TIR_MATMUL_TILE_M * TILED_TIR_MATMUL_TILE_N,
            k,
        )
        expected_kinds = (
            tirx.ForKind.THREAD_BINDING,
            tirx.ForKind.THREAD_BINDING,
            tirx.ForKind.SERIAL,
        )
    else:
        expected_extents = (m * n, 1, k)
        expected_kinds = (
            tirx.ForKind.THREAD_BINDING,
            tirx.ForKind.THREAD_BINDING,
            tirx.ForKind.SERIAL,
        )
    for idx, loop_rv in enumerate(loops):
        loop = sch.get(loop_rv)
        extent = _int_imm_value(loop.extent)
        if extent != expected_extents[idx]:
            raise TritonTVMContractError(
                f"{name} matmul loop {idx} extent must be {expected_extents[idx]}"
            )
        if int(loop.kind) != int(expected_kinds[idx]):
            raise TritonTVMContractError(
                f"{name} matmul loop {idx} kind does not match {implementation_kind}"
            )


def _validate_matmul_buffer_regions(name: str, block, dims: tuple[int, int, int]) -> None:
    m, n, k = dims
    expected = (
        (block.reads[0], (m, k), "A read"),
        (block.reads[1], (k, n), "B read"),
        (block.writes[0], (m, n), "C write"),
    )
    for region, shape, label in expected:
        _validate_buffer_region(name, region, shape, label)


def _validate_buffer_region(
    name: str,
    buffer_region,
    expected_shape: tuple[int, int],
    label: str,
) -> None:
    buffer_shape = tuple(_int_imm_value(dim) for dim in buffer_region.buffer.shape)
    if buffer_shape != expected_shape:
        raise TritonTVMContractError(
            f"{name} matmul {label} buffer shape must be {expected_shape}"
        )
    if len(buffer_region.region) != 2:
        raise TritonTVMContractError(f"{name} matmul {label} must be rank-2")
    region_extents = tuple(_int_imm_value(rng.extent) for rng in buffer_region.region)
    if region_extents != (1, 1):
        raise TritonTVMContractError(
            f"{name} matmul {label} region must access one element per axis"
        )


def _validate_matmul_stores(name: str, block) -> None:
    from tvm import tirx  # pylint: disable=import-outside-toplevel

    init_stores = [
        stmt for stmt in _walk_stmt(block.init) if isinstance(stmt, tirx.BufferStore)
    ]
    update_stores = [
        stmt for stmt in _walk_stmt(block.body) if isinstance(stmt, tirx.BufferStore)
    ]
    if len(init_stores) != 1 or len(update_stores) != 1:
        raise TritonTVMContractError(
            f"{name} matmul block requires one init store and one update store"
        )
    write_buffer = str(block.writes[0].buffer.name)
    for store in init_stores + update_stores:
        if str(store.buffer.name) != write_buffer:
            raise TritonTVMContractError(
                f"{name} matmul init/update stores must write the declared output buffer"
            )
        if len(store.indices) != 2:
            raise TritonTVMContractError(
                f"{name} matmul init/update stores must be rank-2"
            )


def _validate_matmul_extern_buffers(
    name: str,
    func: tvm.tirx.PrimFunc,
    dims: tuple[int, int, int],
    b_layout: str,
    attrs,
) -> None:
    m, n, k = dims
    buffers = list(func.buffer_map.values())
    if len(buffers) != 3:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact requires exactly three buffers"
        )
    transposed_b = _bool_attr(attrs, "triton_tvm.transposed_b")
    expected_transposed = b_layout == "transposed_weight_view"
    if transposed_b is not expected_transposed:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact transposed_b must match b_layout"
        )
    expected_b_shape = (n, k) if expected_transposed else (k, n)
    expected_b_stride = (k, 1) if expected_transposed else (n, 1)
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.b_stride",
        "1, " + str(k) if expected_transposed else f"{n}, 1",
        "extern GEMM logical metadata",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.b_storage_shape",
        f"{expected_b_shape[0]}, {expected_b_shape[1]}",
        "extern GEMM storage metadata",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.b_storage_stride",
        f"{expected_b_stride[0]}, {expected_b_stride[1]}",
        "extern GEMM storage metadata",
    )
    expected_shapes = ((m, k), expected_b_shape, (m, n))
    expected_strides = ((k, 1), expected_b_stride, (n, 1))
    for buffer, shape, strides in zip(buffers, expected_shapes, expected_strides):
        buffer_shape = tuple(_int_imm_value(dim) for dim in buffer.shape)
        if buffer_shape != shape:
            raise TritonTVMContractError(
                f"{name} extern GEMM artifact buffer shape must be {shape}"
            )
        buffer_strides = tuple(_int_imm_value(stride) for stride in buffer.strides)
        if buffer_strides != strides:
            raise TritonTVMContractError(
                f"{name} extern GEMM artifact buffer strides must be {strides}"
            )


def _validate_matmul_schedule_attrs(
    name: str,
    attrs,
    dims: tuple[int, int, int],
    schedule_id: str,
) -> None:
    m, n, _ = dims
    if schedule_id == NATIVE_TIR_MATMUL_SCHEDULE_ID:
        return
    if schedule_id != TILED_TIR_MATMUL_SCHEDULE_ID:
        raise TritonTVMContractError(
            f"{name} native matmul schedule has unknown triton_tvm.schedule_id"
        )
    tile_m = _int_attr(attrs, "triton_tvm.tile_m")
    tile_n = _int_attr(attrs, "triton_tvm.tile_n")
    if tile_m != TILED_TIR_MATMUL_TILE_M or tile_n != TILED_TIR_MATMUL_TILE_N:
        raise TritonTVMContractError(
            f"{name} tiled matmul attrs must preserve 8x8 tile shape"
        )
    if m % TILED_TIR_MATMUL_TILE_M or n % TILED_TIR_MATMUL_TILE_N:
        raise TritonTVMContractError(
            f"{name} tiled matmul schedule requires M and N multiples of 8"
        )


def _validate_matmul_extern_runtime_metadata(name: str, attrs) -> None:
    status = str(attrs.get("triton_tvm.extern_gemm_runtime_status", ""))
    provider_kind = str(attrs.get("triton_tvm.extern_gemm_provider_kind", ""))
    provider_abi_version = _int_attr(attrs, "triton_tvm.extern_gemm_provider_abi_version")
    performance_claim = _bool_attr(attrs, "triton_tvm.extern_gemm_performance_claim")
    uses_host_staging = _bool_attr(attrs, "triton_tvm.extern_gemm_uses_host_staging")
    replacement_available = _bool_attr(
        attrs,
        "triton_tvm.extern_runtime_replacement_available",
    )

    if status == EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY:
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_kind",
            EXTERN_GEMM_RUNTIME_KIND,
            "extern GEMM artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement",
            EXTERN_GEMM_RUNTIME_REPLACEMENT,
            "extern GEMM artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement_reason",
            EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
            "extern GEMM artifact",
        )
        if replacement_available is not False:
            raise TritonTVMContractError(
                f"{name} extern GEMM artifact must keep runtime replacement unavailable"
            )
        if provider_kind != EXTERN_GEMM_PROVIDER_NONE or provider_abi_version != 0:
            raise TritonTVMContractError(
                f"{name} artifact-only extern GEMM must use provider none"
            )
        if performance_claim is not False or uses_host_staging is not False:
            raise TritonTVMContractError(
                f"{name} artifact-only extern GEMM must not claim provider execution"
            )
        return

    if status != EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED:
        raise TritonTVMContractError(
            f"{name} extern GEMM artifact has unsupported runtime status"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_kind",
        EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
        "extern GEMM runtime provider",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_replacement",
        EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
        "extern GEMM runtime provider",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_replacement_reason",
        EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
        "extern GEMM runtime provider",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_gemm_runtime_claim",
        EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
        "extern GEMM runtime provider",
    )
    if replacement_available is not True:
        raise TritonTVMContractError(
            f"{name} runtime-resolved extern GEMM must mark replacement available"
        )
    if provider_kind != EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED:
        raise TritonTVMContractError(
            f"{name} runtime-resolved extern GEMM must name python_torch_host_staged"
        )
    if provider_abi_version != EXTERN_GEMM_PROVIDER_ABI_VERSION:
        raise TritonTVMContractError(
            f"{name} runtime-resolved extern GEMM must preserve provider ABI v1"
        )
    if performance_claim is not False or uses_host_staging is not True:
        raise TritonTVMContractError(
            f"{name} runtime-resolved extern GEMM must be correctness-only host staged"
        )


def _int_imm_value(value) -> int | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prim_func_script(func: tvm.tirx.PrimFunc) -> str:
    try:
        return func.script()
    except Exception:  # pylint: disable=broad-except
        return str(func)


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
