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
- M11 ``conv2d_*_nchw_static_v1`` contracts validate explicit wrapper
  convolution packed calls, including the M11.4 opt-in correctness provider for
  the first ViT patch-embedding convolution and the M11.6 static conv2d
  correctness envelope.
- M11.6 ``pointwise_grid2d_static_v1`` validates native static Grid2D
  pointwise/fusion scaffold artifacts.
- ``cuda_minimal`` and ``cuda_pointwise_flat`` are deprecated compatibility
  aliases.  They are accepted only at the public boundary and immediately
  canonicalized.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import tvm

from .attention import (
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_EXTERN_PACKED_FUNC,
    ATTENTION_EXTERN_SYMBOL,
    ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA,
    ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ATTENTION_NATIVE_DECOMPOSED_PROVIDER_REASON,
    ATTENTION_PROVIDER_ABI_VERSION,
    ATTENTION_PROVIDER_NONE,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY,
    ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
    ATTENTION_RUNTIME_KIND_PROVIDER,
    ATTENTION_RUNTIME_PROVIDER_REASON,
    ATTENTION_RUNTIME_REPLACEMENT,
    ATTENTION_RUNTIME_REPLACEMENT_REASON,
    ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY,
    ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
)
from .errors import TritonTVMContractError, UnsupportedContractError
from .matmul import (
    EXTERN_ADDMM_BIAS_PACKED_FUNC,
    EXTERN_ADDMM_BIAS_NATIVE_TVM_RUNTIME_PROVIDER_REASON,
    EXTERN_ADDMM_BIAS_RUNTIME_PROVIDER_REASON,
    EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON,
    EXTERN_ADDMM_BIAS_SYMBOL,
    EXTERN_GEMM_PACKED_FUNC,
    EXTERN_GEMM_NATIVE_TVM_RUNTIME_PROVIDER_REASON,
    EXTERN_GEMM_PROVIDER_ABI_VERSION,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
    EXTERN_GEMM_RUNTIME_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
    EXTERN_GEMM_RUNTIME_REPLACEMENT,
    EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
    EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    EXTERN_GEMM_SYMBOL,
    MATMUL_PERF_ENVELOPE_ID,
    M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
    SIMT_TIR_MATMUL_SCHEDULE_ID,
    SIMT_TIR_MATMUL_TILE_M,
    SIMT_TIR_MATMUL_TILE_N,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TENSORCORE_TIR_MATMUL_TILE_K,
    TENSORCORE_TIR_MATMUL_TILE_M,
    TENSORCORE_TIR_MATMUL_TILE_N,
    TT_DOT_NATIVE_SOURCE_KINDS,
    TILED_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_TILE_M,
    TILED_TIR_MATMUL_TILE_N,
)
from .vision import (
    VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
    VISION_CONTRACT_VERSION,
    VISION_CONV2D_CONTRACTS,
    VISION_EXTERN_PACKED_FUNC,
    VISION_EXTERN_SYMBOL,
    VISION_GRID2D_CONTRACT_VERSION,
    M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
    M11_GRID2D_PROVIDER_NATIVE_TVM,
    M11_GRID2D_RUNTIME_CLAIM,
    M11_GRID2D_RUNTIME_READY,
    VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
    VISION_RUNTIME_KIND_ARTIFACT_ONLY,
    VISION_RUNTIME_KIND_PROVIDER,
    VISION_PROVIDER_ABI_VERSION,
    VISION_PROVIDER_DEVICE_TORCH_CUDA,
    VISION_PROVIDER_NONE,
    VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    VISION_RUNTIME_CLAIM_PERFORMANCE_ELIGIBLE,
    VISION_RUNTIME_DEVICE_PROVIDER_REASON,
    VISION_RUNTIME_PROVIDER_REASON,
    VISION_RUNTIME_REPLACEMENT,
    VISION_RUNTIME_REPLACEMENT_REASON,
    VISION_RUNTIME_STATUS_ARTIFACT_ONLY,
    VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    is_m11_6_static_conv2d_runtime_scope,
    is_m11_7_device_conv2d_runtime_scope,
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
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL: ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_VIT_FULL: ATTENTION_CONTRACT_VIT_FULL,
    VISION_CONTRACT_CONV2D_NCHW_STATIC: VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC: VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
    VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC: (
        VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC
    ),
    VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC: VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC: VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
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
    ATTENTION_CONTRACT_VIT_FULL: TritonTVMContract(
        name=ATTENTION_CONTRACT_VIT_FULL,
        indexing_kind="rank4_sdpa",
        memory_model="rank4_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version="attention_vit_full_m10_v1",
        execution_kind="extern_attention_sdpa_artifact",
        accumulator_dtype_policy="torch_sdpa_provider_default",
        epsilon_policy="not_applicable",
        mask_policy="none_or_padding",
        axis_policy="batch_head_sequence_head_dim",
        layout_policy="rank4_static",
    ),
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL: TritonTVMContract(
        name=ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
        indexing_kind="rank4_sdpa",
        memory_model="rank4_buffers_with_additive_mask",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version="attention_llama_causal_prefill_m10_v1",
        execution_kind="native_decomposed_attention_qk_masked_softmax_av",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="causal_additive_mask",
        axis_policy="batch_head_sequence_head_dim",
        layout_policy="rank4_static_prefill",
    ),
    VISION_CONTRACT_CONV2D_NCHW_STATIC: TritonTVMContract(
        name=VISION_CONTRACT_CONV2D_NCHW_STATIC,
        indexing_kind="rank4_conv2d",
        memory_model="rank4_nchw_oihw_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version=VISION_CONTRACT_VERSION,
        execution_kind="extern_conv2d_artifact_only",
        accumulator_dtype_policy="provider_default_not_runtime_resolved",
        epsilon_policy="not_applicable",
        mask_policy="not_applicable",
        axis_policy="batch_output_channel_spatial",
        layout_policy="logical_nchw_oihw_static",
    ),
    VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC: TritonTVMContract(
        name=VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
        indexing_kind="rank4_conv2d_1x1",
        memory_model="rank4_nchw_oihw_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version=VISION_CONTRACT_VERSION,
        execution_kind="extern_conv2d_artifact_only",
        accumulator_dtype_policy="provider_default_not_runtime_resolved",
        epsilon_policy="not_applicable",
        mask_policy="not_applicable",
        axis_policy="batch_output_channel_spatial",
        layout_policy="logical_nchw_oihw_static",
    ),
    VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC: TritonTVMContract(
        name=VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC,
        indexing_kind="rank4_depthwise_conv2d",
        memory_model="rank4_nchw_oihw_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version=VISION_CONTRACT_VERSION,
        execution_kind="extern_conv2d_artifact_only",
        accumulator_dtype_policy="provider_default_not_runtime_resolved",
        epsilon_policy="not_applicable",
        mask_policy="not_applicable",
        axis_policy="batch_output_channel_spatial",
        layout_policy="logical_nchw_oihw_static",
    ),
    VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC: TritonTVMContract(
        name=VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC,
        indexing_kind="rank4_grouped_conv2d",
        memory_model="rank4_nchw_oihw_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version=VISION_CONTRACT_VERSION,
        execution_kind="extern_conv2d_artifact_only",
        accumulator_dtype_policy="provider_default_not_runtime_resolved",
        epsilon_policy="not_applicable",
        mask_policy="not_applicable",
        axis_policy="batch_output_channel_spatial",
        layout_policy="logical_nchw_oihw_static",
    ),
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC: TritonTVMContract(
        name=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        indexing_kind="grid2d_affine_pointwise",
        memory_model="rank2_f32_buffers",
        requires_extent_param=False,
        supports_multiple_outputs=False,
        version=VISION_GRID2D_CONTRACT_VERSION,
        execution_kind="native_tvm_grid2d",
        accumulator_dtype_policy="fp32_elementwise",
        epsilon_policy="not_applicable",
        mask_policy="guarded_x_axis_store",
        axis_policy="program_id_x_y",
        layout_policy="rank2_row_major_static",
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


def validate_attention_vit_full_contract(irmod: tvm.IRModule) -> None:
    """Validate the M10 wrapper SDPA packed-call artifact contract."""
    _validate_semantic_common(irmod, ATTENTION_CONTRACT_VIT_FULL)
    for gvar, func in irmod.functions.items():
        _validate_attention_vit_full_body(gvar.name_hint, func)


def validate_attention_llama_causal_prefill_contract(irmod: tvm.IRModule) -> None:
    """Validate the M10.5 native decomposed Llama prefill contract."""
    _validate_semantic_common(irmod, ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL)
    for gvar, func in irmod.functions.items():
        _validate_attention_llama_causal_prefill_body(gvar.name_hint, func)


def validate_vision_conv2d_contract(irmod: tvm.IRModule) -> None:
    """Validate the M11 artifact-only wrapper conv2d contracts."""
    for gvar, func in irmod.functions.items():
        contract = str((func.attrs or {}).get("triton_tvm.contract", ""))
        if contract not in VISION_CONV2D_CONTRACTS:
            raise TritonTVMContractError(
                f"{gvar.name_hint} vision conv2d artifact has unsupported contract"
            )
    for gvar, func in irmod.functions.items():
        _validate_vision_conv2d_artifact_body(gvar.name_hint, func)


def validate_pointwise_grid2d_static_contract(irmod: tvm.IRModule) -> None:
    """Validate the M11.6 native static Grid2D pointwise/fusion contract."""
    _validate_common_func_attrs(irmod, VISION_CONTRACT_POINTWISE_GRID2D_STATIC)
    for gvar, func in irmod.functions.items():
        _validate_pointwise_grid2d_static_body(gvar.name_hint, func)


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
    elif contract == ATTENTION_CONTRACT_VIT_FULL:
        validate_attention_vit_full_contract(irmod)
    elif contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        validate_attention_llama_causal_prefill_contract(irmod)
    elif contract in VISION_CONV2D_CONTRACTS:
        validate_vision_conv2d_contract(irmod)
    elif contract == VISION_CONTRACT_POINTWISE_GRID2D_STATIC:
        validate_pointwise_grid2d_static_contract(irmod)
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
    if str(attrs.get("triton_tvm.implementation_kind", "")) == "extern_addmm_bias":
        _validate_matmul_extern_addmm_bias_artifact(name, func)
        return
    if str(attrs.get("triton_tvm.schedule_id", "")) == TENSORCORE_TIR_MATMUL_SCHEDULE_ID:
        _validate_matmul_tensorcore_artifact(name, func)
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
    if len(block.writes) != 1:
        raise TritonTVMContractError(f"{name} matmul_minimal requires one write")

    attrs = block.annotations
    if attrs is None or attrs.get("triton_tvm.contract", None) != "matmul_minimal":
        raise TritonTVMContractError(
            f"{name} matmul block must preserve triton_tvm.contract=matmul_minimal"
        )
    source_kind = str(attrs.get("triton_tvm.matmul_source_kind", ""))
    if source_kind not in (
        set(TT_DOT_NATIVE_SOURCE_KINDS)
        | {"wrapper_extern_gemm", "wrapper_extern_addmm_bias"}
    ):
        raise TritonTVMContractError(
            f"{name} matmul block has unsupported source kind {source_kind!r}"
        )
    func_source_kind = str(func.attrs.get("triton_tvm.matmul_source_kind", ""))
    if func_source_kind and func_source_kind != source_kind:
        raise TritonTVMContractError(
            f"{name} matmul block source kind must match function attrs"
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
    expected_epilogue = "bias_add" if source_kind == "wrapper_extern_addmm_bias" else "none"
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.epilogue_kind",
        expected_epilogue,
        "matmul block",
    )
    expected_reads = 3 if source_kind == "wrapper_extern_addmm_bias" else 2
    if len(block.reads) != expected_reads:
        raise TritonTVMContractError(
            f"{name} matmul_minimal requires {expected_reads} reads for {source_kind}"
        )
    _validate_matmul_block_axes(name, block, block_dims)
    _validate_matmul_buffer_regions(name, block, block_dims, attrs)
    _validate_matmul_stores(name, block)
    implementation_kind = str(attrs.get("triton_tvm.implementation_kind", ""))
    if implementation_kind == "unresolved":
        _validate_matmul_loop_extents(name, sch, loops, block_dims, implementation_kind)
        return
    if implementation_kind == "native_tir_schedule":
        schedule_id = str(attrs.get("triton_tvm.schedule_id", ""))
        if schedule_id not in (
            NATIVE_TIR_MATMUL_SCHEDULE_ID,
            M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            SIMT_TIR_MATMUL_SCHEDULE_ID,
            TILED_TIR_MATMUL_SCHEDULE_ID,
        ):
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
        if source_kind in {"wrapper_extern_gemm", "wrapper_extern_addmm_bias"}:
            _validate_native_wrapper_matmul_runtime_metadata(name, func.attrs or {}, source_kind)
        return
    raise TritonTVMContractError(
        f"{name} matmul block has unsupported implementation_kind={implementation_kind!r}"
    )


def _validate_attention_vit_full_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    attrs = func.attrs or {}
    if attrs.get("triton_tvm.contract", None) != ATTENTION_CONTRACT_VIT_FULL:
        raise TritonTVMContractError(
            f"{name} attention artifact must preserve triton_tvm.contract="
            f"{ATTENTION_CONTRACT_VIT_FULL}"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_contract",
        ATTENTION_CONTRACT_VIT_FULL,
        "attention artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_phase",
        "full_attention",
        "attention artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_mask_kind",
        "none_or_padding",
        "attention artifact",
    )
    if _bool_attr(attrs, "triton_tvm.attention_causal") is not False:
        raise TritonTVMContractError(f"{name} attention_vit_full_v1 must be non-causal")
    implementation_kind = str(attrs.get("triton_tvm.implementation_kind", ""))
    if implementation_kind not in (
        ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA,
        ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ):
        raise TritonTVMContractError(
            f"{name} attention artifact has unsupported implementation_kind"
        )
    q_shape = _required_attention_shape(name, attrs, "triton_tvm.q_shape")
    if len(q_shape) != 4:
        raise TritonTVMContractError(f"{name} attention_vit_full_v1 requires rank-4 Q")
    for attr_name in (
        "triton_tvm.k_shape",
        "triton_tvm.v_shape",
        "triton_tvm.output_shape",
    ):
        if _required_attention_shape(name, attrs, attr_name) != q_shape:
            raise TritonTVMContractError(
                f"{name} attention_vit_full_v1 requires matching Q/K/V/output shapes"
            )
    for dtype_attr in (
        "triton_tvm.q_dtype",
        "triton_tvm.k_dtype",
        "triton_tvm.v_dtype",
        "triton_tvm.output_dtype",
    ):
        _require_attr_value(name, attrs, dtype_attr, "float32", "attention artifact")
    if implementation_kind == ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA:
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_symbol",
            ATTENTION_EXTERN_SYMBOL,
            "attention artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_packed_func",
            ATTENTION_EXTERN_PACKED_FUNC,
            "attention artifact",
        )
    else:
        script = tvm.IRModule({"main": func}).script()
        if "call_packed" in script:
            raise TritonTVMContractError(
                f"{name} native decomposed attention must not call packed externs"
            )
        for block_name in (
            "attention_qk_matmul",
            "attention_row_softmax",
            "attention_av_matmul",
        ):
            if block_name not in script:
                raise TritonTVMContractError(
                    f"{name} native decomposed attention missing {block_name}"
                )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.qk_matmul_boundary",
            "matmul_minimal",
            "native attention",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.softmax_boundary",
            "softmax_row",
            "native attention",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.av_matmul_boundary",
            "matmul_minimal",
            "native attention",
        )
    _validate_attention_runtime_metadata(name, attrs)


def _validate_attention_llama_causal_prefill_body(
    name: str,
    func: tvm.tirx.PrimFunc,
) -> None:
    attrs = func.attrs or {}
    if attrs.get("triton_tvm.contract", None) != ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        raise TritonTVMContractError(
            f"{name} Llama prefill attention must preserve triton_tvm.contract="
            f"{ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL}"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_contract",
        ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
        "Llama prefill attention",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_phase",
        "causal_prefill",
        "Llama prefill attention",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_mask_kind",
        "causal",
        "Llama prefill attention",
    )
    if _bool_attr(attrs, "triton_tvm.attention_causal") is not True:
        raise TritonTVMContractError(
            f"{name} attention_llama_causal_prefill_v1 must be causal"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.implementation_kind",
        ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
        "Llama prefill attention",
    )
    q_shape = _required_attention_shape(name, attrs, "triton_tvm.q_shape")
    if len(q_shape) != 4:
        raise TritonTVMContractError(
            f"{name} attention_llama_causal_prefill_v1 requires rank-4 Q"
        )
    for attr_name in (
        "triton_tvm.k_shape",
        "triton_tvm.v_shape",
        "triton_tvm.output_shape",
    ):
        if _required_attention_shape(name, attrs, attr_name) != q_shape:
            raise TritonTVMContractError(
                f"{name} Llama prefill requires matching Q/K/V/output shapes"
            )
    expected_mask_shape = (q_shape[0], q_shape[1], q_shape[2], q_shape[2])
    if _required_attention_shape(name, attrs, "triton_tvm.mask_shape") != expected_mask_shape:
        raise TritonTVMContractError(
            f"{name} Llama prefill requires mask shape B,H,S,S"
        )
    mask_stride = _int_tuple_attr(attrs, "triton_tvm.mask_stride")
    if len(mask_stride) != 4 or any(stride < 0 for stride in mask_stride):
        raise TritonTVMContractError(
            f"{name} Llama prefill requires rank-4 non-negative mask stride"
        )
    for dtype_attr in (
        "triton_tvm.q_dtype",
        "triton_tvm.k_dtype",
        "triton_tvm.v_dtype",
        "triton_tvm.output_dtype",
        "triton_tvm.mask_dtype",
    ):
        _require_attr_value(name, attrs, dtype_attr, "float32", "Llama prefill attention")
    for forbidden_attr in (
        "triton_tvm.kv_cache_layout",
        "triton_tvm.kv_cache_update",
        "triton_tvm.kv_cache_read",
    ):
        if attrs.get(forbidden_attr, None) is not None:
            raise TritonTVMContractError(
                f"{name} M10.5 Llama prefill must not expose {forbidden_attr}"
            )
    script = tvm.IRModule({"main": func}).script()
    if "call_packed" in script:
        raise TritonTVMContractError(
            f"{name} native decomposed Llama prefill must not call packed externs"
        )
    for block_name in (
        "attention_qk_matmul",
        "attention_mask_add",
        "attention_row_softmax",
        "attention_av_matmul",
    ):
        if block_name not in script:
            raise TritonTVMContractError(
                f"{name} native decomposed Llama prefill missing {block_name}"
            )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.qk_matmul_boundary",
        "matmul_minimal",
        "native Llama prefill",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.softmax_boundary",
        "masked_softmax_row",
        "native Llama prefill",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.av_matmul_boundary",
        "matmul_minimal",
        "native Llama prefill",
    )
    _validate_attention_runtime_metadata(name, attrs)


def _validate_vision_conv2d_artifact_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    attrs = func.attrs or {}
    contract = str(attrs.get("triton_tvm.contract", ""))
    if contract not in VISION_CONV2D_CONTRACTS:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact has unsupported contract={contract!r}"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.vision_contract",
        contract,
        "vision conv2d artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.vision_contract_version",
        VISION_CONTRACT_VERSION,
        "vision conv2d artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.vision_source_kind",
        "wrapper_extern_convolution",
        "vision conv2d artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.implementation_kind",
        VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
        "vision conv2d artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_symbol",
        VISION_EXTERN_SYMBOL,
        "vision conv2d artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_packed_func",
        VISION_EXTERN_PACKED_FUNC,
        "vision conv2d artifact",
    )
    _validate_vision_conv2d_runtime_metadata(name, attrs)

    input_shape = _required_vision_shape(name, attrs, "triton_tvm.input_shape")
    weight_shape = _required_vision_shape(name, attrs, "triton_tvm.weight_shape")
    output_shape = _required_vision_shape(name, attrs, "triton_tvm.output_shape")
    input_stride = _required_vision_shape(name, attrs, "triton_tvm.input_stride")
    weight_stride = _required_vision_shape(name, attrs, "triton_tvm.weight_stride")
    output_stride = _required_vision_shape(name, attrs, "triton_tvm.output_stride")
    stride = _required_vision_pair(name, attrs, "triton_tvm.stride")
    padding = _required_vision_pair(name, attrs, "triton_tvm.padding", allow_zero=True)
    dilation = _required_vision_pair(name, attrs, "triton_tvm.dilation")
    output_padding = _required_vision_pair(
        name,
        attrs,
        "triton_tvm.output_padding",
        allow_zero=True,
    )
    groups = _int_attr(attrs, "triton_tvm.groups")
    if groups is None or groups <= 0:
        raise TritonTVMContractError(f"{name} vision conv2d groups must be positive")
    if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
        raise TritonTVMContractError(f"{name} vision conv2d requires rank-4 buffers")
    if len(input_stride) != 4 or len(weight_stride) != 4 or len(output_stride) != 4:
        raise TritonTVMContractError(f"{name} vision conv2d requires rank-4 strides")
    for dtype_attr in (
        "triton_tvm.input_dtype",
        "triton_tvm.weight_dtype",
        "triton_tvm.output_dtype",
    ):
        _require_attr_value(name, attrs, dtype_attr, "float32", "vision conv2d artifact")
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.bias_policy",
        "none",
        "vision conv2d artifact",
    )
    if _bool_attr(attrs, "triton_tvm.transposed") is not False:
        raise TritonTVMContractError(f"{name} vision conv2d artifact must not be transposed")
    if output_padding != (0, 0):
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact must keep output_padding zero"
        )
    if weight_shape[1] * groups != input_shape[1]:
        raise TritonTVMContractError(
            f"{name} vision conv2d grouped input channels mismatch"
        )
    expected_output = _vision_conv2d_output_shape(
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
    )
    if output_shape != expected_output:
        raise TritonTVMContractError(
            f"{name} vision conv2d output shape must be {expected_output}"
        )
    _validate_vision_conv2d_accounting_metadata(
        name,
        attrs,
        input_shape,
        weight_shape,
        output_shape,
    )
    if contract == VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC and weight_shape[2:] != (1, 1):
        raise TritonTVMContractError(f"{name} vision 1x1 conv requires 1x1 weight")
    if contract == VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC:
        if groups != input_shape[1] or weight_shape[0] != input_shape[1] or weight_shape[1] != 1:
            raise TritonTVMContractError(f"{name} depthwise conv contract mismatch")
    if contract == VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC:
        if groups <= 1 or (
            groups == input_shape[1] and weight_shape[0] == input_shape[1] and weight_shape[1] == 1
        ):
            raise TritonTVMContractError(f"{name} grouped conv contract mismatch")
    if contract == VISION_CONTRACT_CONV2D_NCHW_STATIC and groups != 1:
        raise TritonTVMContractError(f"{name} regular conv2d contract requires groups=1")
    if (
        str(attrs.get("triton_tvm.vision_runtime_status", ""))
        == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED
        and not _vision_conv2d_runtime_scope_m11_6_or_m11_7(
            contract,
            input_shape,
            weight_shape,
            output_shape,
            stride,
            padding,
            dilation,
            groups,
            attrs,
        )
    ):
        raise TritonTVMContractError(
            f"{name} M11.6/M11.7 runtime-resolved conv2d is limited to static "
            "NCHW/OIHW groups=1 stride/padding envelope"
        )

    script = _prim_func_script(func)
    packed_call = f'T.call_packed("{VISION_EXTERN_PACKED_FUNC}"'
    if script.count(packed_call) != 1:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact must contain exactly one "
            f"{VISION_EXTERN_PACKED_FUNC} packed call"
        )
    if "call_extern" in script:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact must not use T.call_extern"
        )
    if 'thread="blockIdx.x"' in script or 'thread="threadIdx.x"' in script:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact must not contain a native CUDA schedule"
        )


def _validate_pointwise_grid2d_static_body(name: str, func: tvm.tirx.PrimFunc) -> None:
    attrs = func.attrs or {}
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.contract",
        VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        "Grid2D pointwise artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.grid2d_contract",
        VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        "Grid2D pointwise artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.grid2d_contract_version",
        VISION_GRID2D_CONTRACT_VERSION,
        "Grid2D pointwise artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.implementation_kind",
        M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        "Grid2D pointwise artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.grid2d_runtime_status",
        M11_GRID2D_RUNTIME_READY,
        "Grid2D pointwise artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.grid2d_provider_kind",
        M11_GRID2D_PROVIDER_NATIVE_TVM,
        "Grid2D pointwise artifact",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.grid2d_runtime_claim",
        M11_GRID2D_RUNTIME_CLAIM,
        "Grid2D pointwise artifact",
    )
    if _bool_attr(attrs, "triton_tvm.grid2d_performance_claim") is not True:
        raise TritonTVMContractError(f"{name} Grid2D native artifact must claim performance")
    if _bool_attr(attrs, "triton_tvm.grid2d_uses_host_staging") is not False:
        raise TritonTVMContractError(f"{name} Grid2D native artifact must not use host staging")
    x_extent = _int_attr(attrs, "triton_tvm.grid2d_x_extent")
    y_extent = _int_attr(attrs, "triton_tvm.grid2d_y_extent")
    block_size = _int_attr(attrs, "triton_tvm.grid2d_block_size")
    if x_extent is None or y_extent is None or block_size is None:
        raise TritonTVMContractError(f"{name} Grid2D native artifact missing static shape attrs")
    if x_extent <= 0 or y_extent <= 0 or block_size <= 0:
        raise TritonTVMContractError(f"{name} Grid2D native artifact extents must be positive")
    if _int_attr(attrs, "triton_tvm.grid2d_runtime_launch_count") != 1:
        raise TritonTVMContractError(f"{name} Grid2D native artifact must record one launch")
    if _int_attr(attrs, "triton_tvm.grid2d_artifact_call_count") != 1:
        raise TritonTVMContractError(f"{name} Grid2D native artifact must record one artifact")
    expected_io = int(x_extent) * int(y_extent) * 4 * 3
    if _int_attr(attrs, "triton_tvm.grid2d_total_io_bytes") != expected_io:
        raise TritonTVMContractError(
            f"{name} Grid2D native artifact total IO bytes must be {expected_io}"
        )
    if _int_attr(attrs, "triton_tvm.grid2d_host_staging_bytes") != 0:
        raise TritonTVMContractError(f"{name} Grid2D native artifact host staging must be zero")

    script = _prim_func_script(func)
    for token in ('thread="blockIdx.x"', 'thread="blockIdx.y"', 'thread="threadIdx.x"'):
        if token not in script:
            raise TritonTVMContractError(f"{name} Grid2D native artifact missing {token}")
    if "call_packed" in script or "call_extern" in script:
        raise TritonTVMContractError(f"{name} Grid2D native artifact must not use extern calls")

    guarded_stores = _store_guard_states(func.body)
    if not guarded_stores:
        raise TritonTVMContractError(
            f"{name} Grid2D native artifact requires at least one guarded store"
        )
    if any(not guarded for _, guarded in guarded_stores):
        raise TritonTVMContractError(
            f"{name} Grid2D native artifact must guard every store"
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


def _validate_matmul_extern_addmm_bias_artifact(
    name: str,
    func: tvm.tirx.PrimFunc,
) -> None:
    attrs = func.attrs or {}
    if attrs.get("triton_tvm.contract", None) != "matmul_minimal":
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact must preserve triton_tvm.contract=matmul_minimal"
        )
    if attrs.get("triton_tvm.matmul_source_kind", None) != "wrapper_extern_addmm_bias":
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact must preserve source kind wrapper_extern_addmm_bias"
        )
    if str(attrs.get("triton_tvm.extern_symbol", "")) != EXTERN_ADDMM_BIAS_SYMBOL:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact must preserve extern_symbol={EXTERN_ADDMM_BIAS_SYMBOL}"
        )
    dims = _required_matmul_dims(name, attrs, "extern addmm bias artifact attrs")
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_packed_func",
        EXTERN_ADDMM_BIAS_PACKED_FUNC,
        "extern addmm bias artifact",
    )
    _validate_matmul_extern_runtime_metadata(
        name,
        attrs,
        artifact_reason=EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON,
        provider_reason=EXTERN_ADDMM_BIAS_RUNTIME_PROVIDER_REASON,
        label="extern addmm bias artifact",
    )
    for attr_name, expected in (
        ("triton_tvm.accumulator_dtype", "float32"),
        ("triton_tvm.output_dtype", "float32"),
        ("triton_tvm.bounds_policy", "exact"),
        ("triton_tvm.mask_kind", "none"),
        ("triton_tvm.epilogue_kind", "bias_add"),
        ("triton_tvm.a_layout", "row_major"),
        ("triton_tvm.c_layout", "row_major"),
        ("triton_tvm.alpha", "1"),
        ("triton_tvm.beta", "1"),
    ):
        _require_attr_value(name, attrs, attr_name, expected, "extern addmm bias artifact")
    b_layout = str(attrs.get("triton_tvm.b_layout", ""))
    if b_layout not in ("row_major", "transposed_weight_view"):
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact has unsupported b_layout={b_layout!r}"
        )
    _validate_matmul_extern_addmm_bias_buffers(name, func, dims, b_layout, attrs)

    script = _prim_func_script(func)
    packed_call = f'T.call_packed("{EXTERN_ADDMM_BIAS_PACKED_FUNC}"'
    if script.count(packed_call) != 1:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact must contain exactly one "
            f"{EXTERN_ADDMM_BIAS_PACKED_FUNC} packed call"
        )
    if "call_extern" in script:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact must not use T.call_extern"
        )
    if 'thread="blockIdx.x"' in script or 'thread="threadIdx.x"' in script:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact must not contain a native CUDA schedule"
        )


def _validate_matmul_tensorcore_artifact(name: str, func: tvm.tirx.PrimFunc) -> None:
    attrs = func.attrs or {}
    if attrs.get("triton_tvm.contract", None) != "matmul_minimal":
        raise TritonTVMContractError(
            f"{name} TensorCore matmul must preserve triton_tvm.contract=matmul_minimal"
        )
    if str(attrs.get("triton_tvm.matmul_source_kind", "")) not in TT_DOT_NATIVE_SOURCE_KINDS:
        raise TritonTVMContractError(
            f"{name} TensorCore matmul must preserve a tt.dot source kind"
        )
    if str(attrs.get("triton_tvm.implementation_kind", "")) != "native_tir_schedule":
        raise TritonTVMContractError(
            f"{name} TensorCore matmul must be a native_tir_schedule"
        )
    if str(attrs.get("triton_tvm.schedule_id", "")) != TENSORCORE_TIR_MATMUL_SCHEDULE_ID:
        raise TritonTVMContractError(
            f"{name} TensorCore matmul has unknown triton_tvm.schedule_id"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.matmul_perf_envelope",
        MATMUL_PERF_ENVELOPE_ID,
        "TensorCore matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.accumulator_dtype",
        "float32",
        "TensorCore matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.output_dtype",
        "float32",
        "TensorCore matmul",
    )
    _require_attr_value(name, attrs, "triton_tvm.a_dtype", "float16", "TensorCore matmul")
    _require_attr_value(name, attrs, "triton_tvm.b_dtype", "float16", "TensorCore matmul")
    _require_attr_value(name, attrs, "triton_tvm.bounds_policy", "exact", "TensorCore matmul")
    _require_attr_value(name, attrs, "triton_tvm.mask_kind", "none", "TensorCore matmul")
    _require_attr_value(name, attrs, "triton_tvm.epilogue_kind", "none", "TensorCore matmul")
    dims = _required_matmul_dims(name, attrs, "TensorCore matmul attrs")
    m, n, k = dims
    if (
        m % TENSORCORE_TIR_MATMUL_TILE_M
        or n % TENSORCORE_TIR_MATMUL_TILE_N
        or k % 16
    ):
        raise TritonTVMContractError(
            f"{name} TensorCore matmul requires M/N/K multiples of 16"
        )
    if _int_attr(attrs, "triton_tvm.tile_m") != TENSORCORE_TIR_MATMUL_TILE_M:
        raise TritonTVMContractError(f"{name} TensorCore matmul must preserve tile_m=16")
    if _int_attr(attrs, "triton_tvm.tile_n") != TENSORCORE_TIR_MATMUL_TILE_N:
        raise TritonTVMContractError(f"{name} TensorCore matmul must preserve tile_n=16")
    if _int_attr(attrs, "triton_tvm.tile_k") != TENSORCORE_TIR_MATMUL_TILE_K:
        raise TritonTVMContractError(f"{name} TensorCore matmul must preserve tile_k=4")
    _validate_matmul_tensorcore_buffers(name, func, dims)

    script = _prim_func_script(func)
    if "T.ptx_mma" not in script and "tirx.ptx_mma" not in script:
        raise TritonTVMContractError(f"{name} TensorCore matmul must use T.ptx_mma")
    if "m8n8k4" not in script:
        raise TritonTVMContractError(f"{name} TensorCore matmul must use m8n8k4")
    if "blockIdx.x" not in script or "threadIdx.x" not in script:
        raise TritonTVMContractError(
            f"{name} TensorCore matmul must launch blockIdx.x and threadIdx.x"
        )
    if "call_packed" in script or "call_extern" in script:
        raise TritonTVMContractError(
            f"{name} TensorCore matmul must not use extern calls"
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


def _required_attention_shape(name: str, attrs, attr_name: str) -> tuple[int, ...]:
    shape = _int_tuple_attr(attrs, attr_name)
    if not shape:
        raise TritonTVMContractError(f"{name} attention artifact missing {attr_name}")
    if any(extent <= 0 for extent in shape):
        raise TritonTVMContractError(f"{name} attention artifact {attr_name} must be positive")
    return shape


def _required_vision_shape(name: str, attrs, attr_name: str) -> tuple[int, ...]:
    shape = _int_tuple_attr(attrs, attr_name)
    if not shape:
        raise TritonTVMContractError(f"{name} vision conv2d artifact missing {attr_name}")
    if any(extent <= 0 for extent in shape):
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact {attr_name} must be positive"
        )
    return shape


def _required_vision_pair(
    name: str,
    attrs,
    attr_name: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, int]:
    values = _int_tuple_attr(attrs, attr_name)
    if len(values) != 2:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact {attr_name} must be a pair"
        )
    if allow_zero:
        invalid = any(value < 0 for value in values)
    else:
        invalid = any(value <= 0 for value in values)
    if invalid:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact {attr_name} has invalid values"
        )
    return (values[0], values[1])


def _vision_conv2d_output_shape(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    batch, _, input_h, input_w = input_shape
    output_channels, _, kernel_h, kernel_w = weight_shape
    output_h = (input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    output_w = (input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    if output_h <= 0 or output_w <= 0:
        raise TritonTVMContractError(f"vision conv2d output shape must be positive")
    return (batch, output_channels, output_h, output_w)


def _vision_conv2d_runtime_scope_m11_6_or_m11_7(
    contract: str,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    attrs,
) -> bool:
    from .vision import VisionConv2DSemantics  # pylint: disable=import-outside-toplevel

    semantics = VisionConv2DSemantics(
        source_kind="wrapper_extern_convolution",
        source_name=VISION_EXTERN_SYMBOL,
        kernel_name="contract_validator",
        vision_contract=contract,
        vision_op_family="convolution",
        input_param="input",
        weight_param="weight",
        output_param="output",
        input_shape=tuple(input_shape),  # type: ignore[arg-type]
        weight_shape=tuple(weight_shape),  # type: ignore[arg-type]
        output_shape=tuple(output_shape),  # type: ignore[arg-type]
        input_stride=_int_tuple_attr(attrs, "triton_tvm.input_stride"),
        weight_stride=_int_tuple_attr(attrs, "triton_tvm.weight_stride"),
        output_stride=_int_tuple_attr(attrs, "triton_tvm.output_stride"),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_policy=str(attrs.get("triton_tvm.bias_policy", "")),
        transposed=bool(_bool_attr(attrs, "triton_tvm.transposed")),
        output_padding=_int_tuple_attr(attrs, "triton_tvm.output_padding"),
        input_dtype=str(attrs.get("triton_tvm.input_dtype", "")),
        weight_dtype=str(attrs.get("triton_tvm.weight_dtype", "")),
        output_dtype=str(attrs.get("triton_tvm.output_dtype", "")),
    )
    provider_kind = str(
        attrs.get("triton_tvm.vision_provider_kind", VISION_PROVIDER_NONE)
        or VISION_PROVIDER_NONE
    )
    if provider_kind == VISION_PROVIDER_DEVICE_TORCH_CUDA:
        return is_m11_7_device_conv2d_runtime_scope(semantics)
    return is_m11_6_static_conv2d_runtime_scope(semantics)


def _validate_vision_conv2d_accounting_metadata(
    name: str,
    attrs,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    input_bytes = _shape_numel(input_shape) * 4
    weight_bytes = _shape_numel(weight_shape) * 4
    output_bytes = _shape_numel(output_shape) * 4
    total_io_bytes = input_bytes + weight_bytes + output_bytes
    uses_host_staging = bool(_bool_attr(attrs, "triton_tvm.vision_uses_host_staging"))
    host_staging_bytes = total_io_bytes if uses_host_staging else 0
    expected = {
        "triton_tvm.vision_input_bytes": input_bytes,
        "triton_tvm.vision_weight_bytes": weight_bytes,
        "triton_tvm.vision_output_bytes": output_bytes,
        "triton_tvm.vision_total_io_bytes": total_io_bytes,
        "triton_tvm.vision_host_staging_bytes": host_staging_bytes,
        "triton_tvm.vision_total_accounted_bytes": total_io_bytes + host_staging_bytes,
    }
    for attr_name, expected_value in expected.items():
        value = _int_attr(attrs, attr_name)
        if value != expected_value:
            raise TritonTVMContractError(
                f"{name} vision conv2d artifact {attr_name} must be {expected_value}"
            )


def _shape_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def _validate_vision_conv2d_runtime_metadata(name: str, attrs) -> None:
    status = str(attrs.get("triton_tvm.vision_runtime_status", ""))
    provider_kind = str(
        attrs.get("triton_tvm.vision_provider_kind", VISION_PROVIDER_NONE)
        or VISION_PROVIDER_NONE
    )
    provider_abi_version = _int_attr(attrs, "triton_tvm.vision_provider_abi_version") or 0
    uses_host_staging = _bool_attr(attrs, "triton_tvm.vision_uses_host_staging")
    performance_claim = _bool_attr(attrs, "triton_tvm.vision_performance_claim")
    launch_count = _int_attr(attrs, "triton_tvm.vision_runtime_launch_count")
    artifact_calls = _int_attr(attrs, "triton_tvm.vision_artifact_call_count")
    if artifact_calls != 1:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact must record one artifact call"
        )
    if status == VISION_RUNTIME_STATUS_ARTIFACT_ONLY:
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_kind",
            VISION_RUNTIME_KIND_ARTIFACT_ONLY,
            "vision conv2d artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement",
            VISION_RUNTIME_REPLACEMENT,
            "vision conv2d artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement_reason",
            VISION_RUNTIME_REPLACEMENT_REASON,
            "vision conv2d artifact",
        )
        if _bool_attr(attrs, "triton_tvm.extern_runtime_replacement_available") is not False:
            raise TritonTVMContractError(
                f"{name} vision conv2d artifact must keep runtime replacement unavailable"
            )
        if provider_kind != VISION_PROVIDER_NONE or provider_abi_version != 0:
            raise TritonTVMContractError(
                f"{name} artifact-only vision conv2d must not claim a provider"
            )
        if uses_host_staging is not False:
            raise TritonTVMContractError(
                f"{name} artifact-only vision conv2d must not use host staging"
            )
        if performance_claim is not False:
            raise TritonTVMContractError(
                f"{name} artifact-only vision conv2d must not claim performance"
            )
        if launch_count != 0:
            raise TritonTVMContractError(
                f"{name} vision conv2d artifact must record zero runtime launches"
            )
    elif status == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED:
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_kind",
            VISION_RUNTIME_KIND_PROVIDER,
            "vision conv2d runtime provider",
        )
        if _bool_attr(attrs, "triton_tvm.extern_runtime_replacement_available") is not True:
            raise TritonTVMContractError(
                f"{name} vision conv2d runtime provider must be replacement-available"
            )
        if provider_kind not in {
            VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
            VISION_PROVIDER_DEVICE_TORCH_CUDA,
        }:
            raise TritonTVMContractError(
                f"{name} vision conv2d runtime provider kind mismatch"
            )
        if provider_abi_version != VISION_PROVIDER_ABI_VERSION:
            raise TritonTVMContractError(
                f"{name} vision conv2d runtime provider ABI mismatch"
            )
        if provider_kind == VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED:
            _require_attr_value(
                name,
                attrs,
                "triton_tvm.extern_runtime_replacement",
                VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                "vision conv2d runtime provider",
            )
            _require_attr_value(
                name,
                attrs,
                "triton_tvm.extern_runtime_replacement_reason",
                VISION_RUNTIME_PROVIDER_REASON,
                "vision conv2d runtime provider",
            )
            _require_attr_value(
                name,
                attrs,
                "triton_tvm.vision_runtime_claim",
                VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                "vision conv2d runtime provider",
            )
            if performance_claim is not False:
                raise TritonTVMContractError(
                    f"{name} host-staged vision conv2d provider must not claim performance"
                )
            if uses_host_staging is not True:
                raise TritonTVMContractError(
                    f"{name} host-staged vision conv2d provider must record host staging"
                )
        else:
            _require_attr_value(
                name,
                attrs,
                "triton_tvm.extern_runtime_replacement",
                VISION_PROVIDER_DEVICE_TORCH_CUDA,
                "vision conv2d runtime provider",
            )
            _require_attr_value(
                name,
                attrs,
                "triton_tvm.extern_runtime_replacement_reason",
                VISION_RUNTIME_DEVICE_PROVIDER_REASON,
                "vision conv2d runtime provider",
            )
            _require_attr_value(
                name,
                attrs,
                "triton_tvm.vision_runtime_claim",
                VISION_RUNTIME_CLAIM_PERFORMANCE_ELIGIBLE,
                "vision conv2d runtime provider",
            )
            if performance_claim is not True:
                raise TritonTVMContractError(
                    f"{name} device vision conv2d provider must claim performance"
                )
            if uses_host_staging is not False:
                raise TritonTVMContractError(
                    f"{name} device vision conv2d provider must not record host staging"
                )
        if launch_count != 1:
            raise TritonTVMContractError(
                f"{name} vision conv2d runtime provider must record one runtime launch"
            )
    else:
        raise TritonTVMContractError(
            f"{name} vision conv2d artifact has unsupported runtime status={status!r}"
        )
    for attr_name in (
        "triton_tvm.vision_input_bytes",
        "triton_tvm.vision_weight_bytes",
        "triton_tvm.vision_output_bytes",
        "triton_tvm.vision_total_io_bytes",
        "triton_tvm.vision_host_staging_bytes",
        "triton_tvm.vision_total_accounted_bytes",
    ):
        value = _int_attr(attrs, attr_name)
        if value is None or value < 0:
            raise TritonTVMContractError(
                f"{name} vision conv2d artifact {attr_name} must be non-negative"
            )


def _validate_attention_runtime_metadata(name: str, attrs) -> None:
    status = str(attrs.get("triton_tvm.attention_runtime_status", ""))
    provider_kind = str(attrs.get("triton_tvm.attention_provider_kind", ""))
    provider_abi_version = _int_attr(attrs, "triton_tvm.attention_provider_abi_version")
    performance_claim = _bool_attr(attrs, "triton_tvm.attention_performance_claim")
    uses_host_staging = _bool_attr(attrs, "triton_tvm.attention_uses_host_staging")
    replacement_available = _bool_attr(
        attrs,
        "triton_tvm.extern_runtime_replacement_available",
    )
    _validate_attention_accounting_metadata(name, attrs)

    if status == ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY:
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_kind",
            ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY,
            "attention artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement",
            ATTENTION_RUNTIME_REPLACEMENT,
            "attention artifact",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement_reason",
            ATTENTION_RUNTIME_REPLACEMENT_REASON,
            "attention artifact",
        )
        if replacement_available is not False:
            raise TritonTVMContractError(
                f"{name} attention artifact must keep runtime replacement unavailable"
            )
        if provider_kind != ATTENTION_PROVIDER_NONE or provider_abi_version != 0:
            raise TritonTVMContractError(
                f"{name} artifact-only attention must use provider none"
            )
        if performance_claim is not False or uses_host_staging is not False:
            raise TritonTVMContractError(
                f"{name} artifact-only attention must not claim provider execution"
            )
        return

    if status != ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED:
        raise TritonTVMContractError(
            f"{name} attention artifact has unsupported runtime status"
        )
    implementation_kind = str(attrs.get("triton_tvm.implementation_kind", ""))
    if implementation_kind == ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED:
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_kind",
            ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
            "native attention provider",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement",
            ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
            "native attention provider",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement_reason",
            ATTENTION_NATIVE_DECOMPOSED_PROVIDER_REASON,
            "native attention provider",
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.attention_runtime_claim",
            ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
            "native attention provider",
        )
        if replacement_available is not True:
            raise TritonTVMContractError(
                f"{name} native attention must mark replacement available"
            )
        if provider_kind != ATTENTION_PROVIDER_NATIVE_DECOMPOSED:
            raise TritonTVMContractError(
                f"{name} native attention must name native_decomposed provider"
            )
        if provider_abi_version != ATTENTION_PROVIDER_ABI_VERSION:
            raise TritonTVMContractError(
                f"{name} native attention must preserve provider ABI v1"
            )
        if performance_claim is not False or uses_host_staging is not False:
            raise TritonTVMContractError(
                f"{name} native attention must be correctness-only without host staging"
            )
        return

    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_kind",
        ATTENTION_RUNTIME_KIND_PROVIDER,
        "attention runtime provider",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_replacement",
        ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
        "attention runtime provider",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_replacement_reason",
        ATTENTION_RUNTIME_PROVIDER_REASON,
        "attention runtime provider",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.attention_runtime_claim",
        ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
        "attention runtime provider",
    )
    if replacement_available is not True:
        raise TritonTVMContractError(
            f"{name} runtime-resolved attention must mark replacement available"
        )
    if provider_kind != ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED:
        raise TritonTVMContractError(
            f"{name} runtime-resolved attention must name python_torch_host_staged"
        )
    if provider_abi_version != ATTENTION_PROVIDER_ABI_VERSION:
        raise TritonTVMContractError(
            f"{name} runtime-resolved attention must preserve provider ABI v1"
        )
    if performance_claim is not False or uses_host_staging is not True:
        raise TritonTVMContractError(
            f"{name} runtime-resolved attention must be correctness-only host staged"
        )


def _validate_attention_accounting_metadata(name: str, attrs) -> None:
    int_attrs = {
        attr_name: _int_attr(attrs, f"triton_tvm.{attr_name}")
        for attr_name in (
            "attention_runtime_launch_count",
            "attention_artifact_call_count",
            "attention_qkv_bytes",
            "attention_mask_bytes",
            "attention_output_bytes",
            "attention_total_io_bytes",
            "attention_intermediate_buffer_bytes",
            "attention_host_staging_bytes",
            "attention_total_accounted_bytes",
        )
    }
    missing = [attr_name for attr_name, value in int_attrs.items() if value is None]
    if missing:
        raise TritonTVMContractError(
            f"{name} attention artifact missing hardening accounting attrs: "
            + ", ".join(missing)
        )
    negative = [attr_name for attr_name, value in int_attrs.items() if int(value or 0) < 0]
    if negative:
        raise TritonTVMContractError(
            f"{name} attention hardening accounting attrs must be non-negative: "
            + ", ".join(negative)
        )

    qkv = int_attrs["attention_qkv_bytes"] or 0
    mask = int_attrs["attention_mask_bytes"] or 0
    output = int_attrs["attention_output_bytes"] or 0
    total_io = int_attrs["attention_total_io_bytes"] or 0
    intermediate = int_attrs["attention_intermediate_buffer_bytes"] or 0
    host_staging = int_attrs["attention_host_staging_bytes"] or 0
    total_accounted = int_attrs["attention_total_accounted_bytes"] or 0
    launch_count = int_attrs["attention_runtime_launch_count"] or 0
    artifact_calls = int_attrs["attention_artifact_call_count"] or 0
    status = str(attrs.get("triton_tvm.attention_runtime_status", ""))
    implementation_kind = str(attrs.get("triton_tvm.implementation_kind", ""))

    if total_io != qkv + mask + output:
        raise TritonTVMContractError(
            f"{name} attention total IO bytes must equal QKV + mask + output bytes"
        )
    if total_accounted != total_io + intermediate:
        raise TritonTVMContractError(
            f"{name} attention total accounted bytes must include intermediate bytes"
        )
    if artifact_calls != 1:
        raise TritonTVMContractError(
            f"{name} attention artifact must record exactly one artifact call"
        )
    if status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED and launch_count != 1:
        raise TritonTVMContractError(
            f"{name} runtime-resolved attention must record one runtime launch"
        )
    if status == ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY and launch_count != 0:
        raise TritonTVMContractError(
            f"{name} artifact-only attention must record zero runtime launches"
        )
    if implementation_kind == ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED:
        if intermediate <= 0:
            raise TritonTVMContractError(
                f"{name} native attention must account intermediate buffers"
            )
        if host_staging != 0:
            raise TritonTVMContractError(
                f"{name} native attention must not account host staging"
            )
    elif host_staging not in (0, total_io):
        raise TritonTVMContractError(
            f"{name} host-staged attention must account host staging as IO bytes"
        )


def _int_attr(attrs, attr_name: str) -> int | None:
    value = attrs.get(attr_name, None)
    if value is None:
        return None
    value = _int_imm_value(value)
    return value


def _int_tuple_attr(attrs, attr_name: str) -> tuple[int, ...]:
    value = attrs.get(attr_name, None)
    if value is None:
        return ()
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


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
    elif schedule_id == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
        expected_extents = (m, n, k)
        expected_kinds = (
            tirx.ForKind.THREAD_BINDING,
            tirx.ForKind.THREAD_BINDING,
            tirx.ForKind.SERIAL,
        )
    elif schedule_id == SIMT_TIR_MATMUL_SCHEDULE_ID:
        expected_extents = (
            (m // SIMT_TIR_MATMUL_TILE_M) * (n // SIMT_TIR_MATMUL_TILE_N),
            SIMT_TIR_MATMUL_TILE_M * SIMT_TIR_MATMUL_TILE_N,
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


def _validate_matmul_buffer_regions(name: str, block, dims: tuple[int, int, int], attrs) -> None:
    m, n, k = dims
    source_kind = str(attrs.get("triton_tvm.matmul_source_kind", ""))
    b_layout = str(attrs.get("triton_tvm.b_layout", ""))
    b_shape = (n, k) if b_layout == "transposed_weight_view" else (k, n)
    expected = (
        (block.reads[0], (m, k), "A read"),
        (block.reads[1], b_shape, "B read"),
        (block.writes[0], (m, n), "C write"),
    )
    for region, shape, label in expected:
        _validate_buffer_region(name, region, shape, label)
    if source_kind == "wrapper_extern_addmm_bias":
        bias_shape = _int_tuple_attr(attrs, "triton_tvm.bias_shape")
        if bias_shape not in ((n,), (m, n)):
            raise TritonTVMContractError(
                f"{name} native wrapper addmm bias shape must be N or MxN"
            )
        _validate_buffer_region_any_rank(name, block.reads[2], bias_shape, "bias read")


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


def _validate_buffer_region_any_rank(
    name: str,
    buffer_region,
    expected_shape: tuple[int, ...],
    label: str,
) -> None:
    buffer_shape = tuple(_int_imm_value(dim) for dim in buffer_region.buffer.shape)
    if buffer_shape != expected_shape:
        raise TritonTVMContractError(
            f"{name} matmul {label} buffer shape must be {expected_shape}"
        )
    if len(buffer_region.region) != len(expected_shape):
        raise TritonTVMContractError(
            f"{name} matmul {label} rank must be {len(expected_shape)}"
        )
    region_extents = tuple(_int_imm_value(rng.extent) for rng in buffer_region.region)
    if region_extents != tuple(1 for _ in expected_shape):
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


def _validate_matmul_extern_addmm_bias_buffers(
    name: str,
    func: tvm.tirx.PrimFunc,
    dims: tuple[int, int, int],
    b_layout: str,
    attrs,
) -> None:
    m, n, k = dims
    buffers = list(func.buffer_map.values())
    if len(buffers) != 4:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact requires exactly four buffers"
        )
    transposed_b = _bool_attr(attrs, "triton_tvm.transposed_b")
    expected_transposed = b_layout == "transposed_weight_view"
    if transposed_b is not expected_transposed:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact transposed_b must match b_layout"
        )
    expected_b_shape = (n, k) if expected_transposed else (k, n)
    expected_b_stride = (k, 1) if expected_transposed else (n, 1)
    bias_shape = _int_tuple_attr(attrs, "triton_tvm.bias_shape")
    bias_stride = _int_tuple_attr(attrs, "triton_tvm.bias_stride")
    bias_rank = _int_attr(attrs, "triton_tvm.bias_rank")
    if bias_shape == (n,):
        expected_bias_stride = (1,)
        expected_bias_rank = 1
    elif bias_shape == (m, n):
        expected_bias_stride = (n, 1)
        expected_bias_rank = 2
    else:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact supports only rank-1 N or rank-2 MxN bias"
        )
    if bias_stride != expected_bias_stride or bias_rank != expected_bias_rank:
        raise TritonTVMContractError(
            f"{name} extern addmm bias artifact bias metadata does not match shape"
        )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.b_stride",
        "1, " + str(k) if expected_transposed else f"{n}, 1",
        "extern addmm bias logical metadata",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.b_storage_shape",
        f"{expected_b_shape[0]}, {expected_b_shape[1]}",
        "extern addmm bias storage metadata",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.b_storage_stride",
        f"{expected_b_stride[0]}, {expected_b_stride[1]}",
        "extern addmm bias storage metadata",
    )
    expected_shapes = (bias_shape, (m, k), expected_b_shape, (m, n))
    expected_strides = (expected_bias_stride, (k, 1), expected_b_stride, (n, 1))
    for buffer, shape, strides in zip(buffers, expected_shapes, expected_strides):
        buffer_shape = tuple(_int_imm_value(dim) for dim in buffer.shape)
        if buffer_shape != shape:
            raise TritonTVMContractError(
                f"{name} extern addmm bias artifact buffer shape must be {shape}"
            )
        buffer_strides = tuple(_int_imm_value(stride) for stride in buffer.strides)
        if buffer_strides != strides:
            raise TritonTVMContractError(
                f"{name} extern addmm bias artifact buffer strides must be {strides}"
            )


def _validate_matmul_tensorcore_buffers(
    name: str,
    func: tvm.tirx.PrimFunc,
    dims: tuple[int, int, int],
) -> None:
    m, n, k = dims
    buffers = list(func.buffer_map.values())
    if len(buffers) != 3:
        raise TritonTVMContractError(
            f"{name} TensorCore matmul requires exactly three buffers"
        )
    expected = ((m, k, "float16"), (k, n, "float16"), (m, n, "float32"))
    for buffer, (dim0, dim1, dtype) in zip(buffers, expected):
        buffer_shape = tuple(_int_imm_value(dim) for dim in buffer.shape)
        if buffer_shape != (dim0, dim1):
            raise TritonTVMContractError(
                f"{name} TensorCore matmul buffer shape must be {(dim0, dim1)}"
            )
        if str(buffer.dtype) != dtype:
            raise TritonTVMContractError(
                f"{name} TensorCore matmul buffer dtype must be {dtype}"
            )


def _validate_matmul_schedule_attrs(
    name: str,
    attrs,
    dims: tuple[int, int, int],
    schedule_id: str,
) -> None:
    m, n, _ = dims
    if schedule_id in (NATIVE_TIR_MATMUL_SCHEDULE_ID, M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID):
        return
    if schedule_id == SIMT_TIR_MATMUL_SCHEDULE_ID:
        tile_m = _int_attr(attrs, "triton_tvm.tile_m")
        tile_n = _int_attr(attrs, "triton_tvm.tile_n")
        if tile_m != SIMT_TIR_MATMUL_TILE_M or tile_n != SIMT_TIR_MATMUL_TILE_N:
            raise TritonTVMContractError(
                f"{name} SIMT matmul attrs must preserve 16x16 tile shape"
            )
        if m % SIMT_TIR_MATMUL_TILE_M or n % SIMT_TIR_MATMUL_TILE_N:
            raise TritonTVMContractError(
                f"{name} SIMT matmul schedule requires M and N multiples of 16"
            )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.matmul_perf_envelope",
            MATMUL_PERF_ENVELOPE_ID,
            "SIMT matmul",
        )
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


def _validate_matmul_extern_runtime_metadata(
    name: str,
    attrs,
    *,
    artifact_reason: str = EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
    provider_reason: str = EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
    label: str = "extern GEMM artifact",
) -> None:
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
            label,
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement",
            EXTERN_GEMM_RUNTIME_REPLACEMENT,
            label,
        )
        _require_attr_value(
            name,
            attrs,
            "triton_tvm.extern_runtime_replacement_reason",
            artifact_reason,
            label,
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
        provider_reason,
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


def _validate_native_wrapper_matmul_runtime_metadata(
    name: str,
    attrs,
    source_kind: str,
) -> None:
    provider_reason = (
        EXTERN_ADDMM_BIAS_NATIVE_TVM_RUNTIME_PROVIDER_REASON
        if source_kind == "wrapper_extern_addmm_bias"
        else EXTERN_GEMM_NATIVE_TVM_RUNTIME_PROVIDER_REASON
    )
    extern_symbol = (
        EXTERN_ADDMM_BIAS_SYMBOL if source_kind == "wrapper_extern_addmm_bias" else EXTERN_GEMM_SYMBOL
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_symbol",
        extern_symbol,
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_packed_func",
        "",
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_kind",
        EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_replacement",
        EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_runtime_replacement_reason",
        provider_reason,
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_gemm_runtime_status",
        EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_gemm_provider_kind",
        EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        "native wrapper matmul",
    )
    _require_attr_value(
        name,
        attrs,
        "triton_tvm.extern_gemm_runtime_claim",
        EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
        "native wrapper matmul",
    )
    if _bool_attr(attrs, "triton_tvm.extern_runtime_replacement_available") is not True:
        raise TritonTVMContractError(
            f"{name} native wrapper matmul must mark replacement available"
        )
    if _int_attr(attrs, "triton_tvm.extern_gemm_provider_abi_version") != (
        EXTERN_GEMM_PROVIDER_ABI_VERSION
    ):
        raise TritonTVMContractError(f"{name} native wrapper matmul provider ABI mismatch")
    if _bool_attr(attrs, "triton_tvm.extern_gemm_performance_claim") is not False:
        raise TritonTVMContractError(
            f"{name} native wrapper matmul must not claim performance in M12.2"
        )
    if _bool_attr(attrs, "triton_tvm.extern_gemm_uses_host_staging") is not False:
        raise TritonTVMContractError(f"{name} native wrapper matmul must not host stage")


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
