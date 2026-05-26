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
"""M11 vision/convolution report and artifact boundary helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Any

from .errors import UnsupportedTTIROpError


VISION_CONTRACT_CONV2D_NCHW_STATIC = "conv2d_nchw_static_v1"
VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC = "conv2d_1x1_nchw_static_v1"
VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC = "depthwise_conv2d_nchw_static_v1"
VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC = "grouped_conv2d_nchw_static_v1"
VISION_CONTRACT_POINTWISE_GRID2D_STATIC = "pointwise_grid2d_static_v1"
VISION_CONV2D_CONTRACTS = (
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
    VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC,
)
VISION_CONTRACT_VERSION = "m11_v1"
VISION_GRID2D_CONTRACT_VERSION = "m11_6_grid2d_static_v1"

VISION_SOURCE_KIND_WRAPPER_CONV2D = "wrapper_extern_convolution"
VISION_OP_FAMILY_CONVOLUTION = "convolution"
VISION_SEMANTICS_STATUS_ACCEPTED = "vision_semantics_accepted"
VISION_SEMANTICS_STATUS_UNSUPPORTED = "vision_semantics_unsupported"

VISION_EXTERN_SYMBOL = "extern_kernels.convolution"
VISION_EXTERN_PACKED_FUNC = "tvm.contrib.triton_tvm.extern_conv2d"
VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D = "extern_conv2d"
VISION_RUNTIME_KIND_ARTIFACT_ONLY = "artifact_only"
VISION_RUNTIME_REPLACEMENT = "not_available"
VISION_RUNTIME_REPLACEMENT_REASON = "extern_conv2d_runtime_replacement_gate_closed"
VISION_RUNTIME_STATUS_ARTIFACT_ONLY = "artifact_only"
VISION_RUNTIME_STATUS_RUNTIME_RESOLVED = "runtime_resolved"
VISION_RUNTIME_STATUS_UNSUPPORTED = "unsupported"
VISION_RUNTIME_KIND_PROVIDER = "runtime_provider"
VISION_PROVIDER_NONE = "none"
VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED = "python_torch_host_staged"
VISION_PROVIDER_ABI_VERSION = 1
VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY = "correctness_only"
VISION_RUNTIME_PROVIDER_REASON = "extern_conv2d_python_torch_host_staged_provider_enabled"
VISION_RUNTIME_PROVIDER_UNKNOWN_REASON = "vision_conv2d_runtime_provider_unknown_m11_4"
VISION_M11_4_RUNTIME_PROVIDER_SCOPE_REASON = "vision_conv2d_provider_scope_m11_4_vit_patch_only"
VISION_RUNTIME_PROVIDER_SCOPE_REASON = "vision_conv2d_provider_scope_m11_6_static_conv2d_only"
VISION_M11_4_INTERFACE_STATUS = "m11_4_runtime_resolved_vit_patch_conv_v1"
VISION_M11_5_HARDENING_STATUS = "m11_5_vision_runtime_hardened_v1"
VISION_M11_5_RUNTIME_SCOPE_STATUS = "m11_5_exact_vit_patch_runtime_scope_v1"
VISION_M11_5_CORPUS_DIFF_BASELINE_ID = "m11_5_vit_patch_provider_corpus_baseline_v1"
VISION_M11_6_INTERFACE_STATUS = "m11_6_vision_runtime_grid2d_readiness_v1"
VISION_M11_6_RUNTIME_SCOPE_STATUS = "m11_6_static_conv2d_runtime_scope_v1"
VISION_M11_6_CORPUS_DIFF_BASELINE_ID = (
    "m11_6_vision_runtime_grid2d_readiness_baseline_v1"
)
VISION_M11_4_VIT_PATCH_INPUT_SHAPE = (1, 3, 32, 32)
VISION_M11_4_VIT_PATCH_WEIGHT_SHAPE = (64, 3, 16, 16)
VISION_M11_4_VIT_PATCH_OUTPUT_SHAPE = (1, 64, 2, 2)
VISION_M11_4_VIT_PATCH_STRIDE = (16, 16)
VISION_M11_4_VIT_PATCH_PADDING = (0, 0)
VISION_M11_4_VIT_PATCH_DILATION = (1, 1)

M11_GRID_TAXONOMY_VERSION = "m11_3_grid_taxonomy_v1"
M11_GRID2D_READINESS_VERSION = "m11_6_grid2d_readiness_v1"
M11_GRID_STATUS_CLASSIFIED = "m11_grid_classified"
M11_GRID_UNSUPPORTED_REASON = "m11_3_taxonomy_only_no_grid_runtime"
M11_GRID2D_ARTIFACT_READY = "m11_6_grid2d_artifact_ready"
M11_GRID2D_ARTIFACT_REJECTED = "m11_6_grid2d_artifact_rejected"
M11_GRID2D_RUNTIME_READY = "runtime_resolved"
M11_GRID2D_RUNTIME_UNSUPPORTED = "unsupported"
M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM = "native_tvm_grid2d"
M11_GRID2D_PROVIDER_NATIVE_TVM = "native_tvm_grid2d"
M11_GRID2D_RUNTIME_CLAIM = "native_tvm_grid2d_correctness_and_perf_scaffold"
M11_GRID2D_UNSUPPORTED_CONCAT_REASON = (
    "grid2d_concat_split_multi_output_layout_deferred_m11_6"
)
M11_GRID2D_UNSUPPORTED_DYNAMIC_REASON = "grid2d_requires_static_grid2d_shape_hints_m11_6"
M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE = "grid_conv_adjacent_pointwise"
M11_GRID_FAMILY_BN_SILU_FUSION = "grid_bn_silu_fusion"
M11_GRID_FAMILY_CONCAT_SPLIT = "grid_concat_split"
M11_GRID_FAMILY_POOL_OR_SOFTMAX = "grid_pool_or_softmax"
M11_GRID_FAMILY_YOLO_DECODE_POSTPROCESS = "grid_yolo_decode_postprocess"
M11_GRID_FAMILY_OTHER_MULTIDIM_POINTWISE = "grid_other_multidim_pointwise"
M11_GRID_FAMILIES = (
    M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
    M11_GRID_FAMILY_BN_SILU_FUSION,
    M11_GRID_FAMILY_CONCAT_SPLIT,
    M11_GRID_FAMILY_POOL_OR_SOFTMAX,
    M11_GRID_FAMILY_YOLO_DECODE_POSTPROCESS,
    M11_GRID_FAMILY_OTHER_MULTIDIM_POINTWISE,
)
M11_GRID_REPORT_FIELDS = (
    "m11_grid_status",
    "m11_grid_family",
    "m11_grid_launch_kind",
    "m11_grid_program_axes",
    "m11_grid_size_hints",
    "m11_grid_operator_tags",
    "unsupported_m11_grid_reason",
    "m11_grid_contract",
    "m11_grid_contract_version",
    "m11_grid_artifact_status",
    "m11_grid_runtime_status",
    "m11_grid_implementation_kind",
    "m11_grid_provider_kind",
    "m11_grid_runtime_claim",
    "m11_grid_performance_claim",
    "m11_grid_runtime_launch_count",
    "m11_grid_artifact_call_count",
    "m11_grid_host_staging_bytes",
    "unsupported_m11_grid_runtime_reason",
)

VISION_REPORT_FIELDS = (
    "vision_source_kind",
    "vision_contract",
    "vision_contract_version",
    "vision_op_family",
    "vision_semantics_status",
    "vision_layout",
    "vision_input_shape",
    "vision_weight_shape",
    "vision_output_shape",
    "vision_input_stride",
    "vision_weight_stride",
    "vision_output_stride",
    "vision_stride",
    "vision_padding",
    "vision_dilation",
    "vision_groups",
    "vision_bias_policy",
    "vision_transposed",
    "vision_output_padding",
    "vision_input_dtype",
    "vision_weight_dtype",
    "vision_output_dtype",
    "vision_implementation_kind",
    "vision_extern_symbol",
    "vision_extern_packed_func",
    "vision_runtime_kind",
    "vision_runtime_replacement",
    "vision_runtime_replacement_available",
    "vision_runtime_replacement_reason",
    "vision_runtime_status",
    "vision_provider_kind",
    "vision_provider_abi_version",
    "vision_runtime_claim",
    "vision_performance_claim",
    "vision_uses_host_staging",
    "vision_runtime_launch_count",
    "vision_artifact_call_count",
    "vision_input_bytes",
    "vision_weight_bytes",
    "vision_output_bytes",
    "vision_total_io_bytes",
    "vision_host_staging_bytes",
    "vision_total_accounted_bytes",
    "unsupported_vision_reason",
    "unsupported_vision_runtime_reason",
)


@dataclass(frozen=True)
class VisionConv2DSemantics:
    """Structured M11 wrapper convolution semantics."""

    source_kind: str
    source_name: str
    kernel_name: str
    vision_contract: str
    vision_op_family: str
    input_param: str
    weight_param: str
    output_param: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    input_stride: tuple[int, int, int, int]
    weight_stride: tuple[int, int, int, int]
    output_stride: tuple[int, int, int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    bias_policy: str
    transposed: bool
    output_padding: tuple[int, int]
    input_dtype: str = "float32"
    weight_dtype: str = "float32"
    output_dtype: str = "float32"

    @property
    def vision_layout(self) -> str:
        input_layout = _nchw_layout(self.input_shape, self.input_stride)
        output_layout = _nchw_layout(self.output_shape, self.output_stride)
        weight_layout = _oihw_layout(self.weight_shape, self.weight_stride)
        if (
            input_layout == "nchw_channels_last"
            and output_layout == "nchw_channels_last"
            and weight_layout == "oihw_channels_last"
        ):
            return "nchw_channels_last"
        if (
            input_layout == "nchw_contiguous"
            and output_layout == "nchw_contiguous"
            and weight_layout == "oihw_contiguous"
        ):
            return "nchw_contiguous"
        return "nchw_strided"

    def as_report_fields(self) -> dict[str, Any]:
        """Return stable JSON-able M11 report fields."""
        return {
            "vision_source_kind": self.source_kind,
            "vision_contract": self.vision_contract,
            "vision_contract_version": VISION_CONTRACT_VERSION,
            "vision_op_family": self.vision_op_family,
            "vision_semantics_status": VISION_SEMANTICS_STATUS_ACCEPTED,
            "vision_layout": self.vision_layout,
            "vision_input_shape": _tuple_text(self.input_shape),
            "vision_weight_shape": _tuple_text(self.weight_shape),
            "vision_output_shape": _tuple_text(self.output_shape),
            "vision_input_stride": _tuple_text(self.input_stride),
            "vision_weight_stride": _tuple_text(self.weight_stride),
            "vision_output_stride": _tuple_text(self.output_stride),
            "vision_stride": _tuple_text(self.stride),
            "vision_padding": _tuple_text(self.padding),
            "vision_dilation": _tuple_text(self.dilation),
            "vision_groups": self.groups,
            "vision_bias_policy": self.bias_policy,
            "vision_transposed": self.transposed,
            "vision_output_padding": _tuple_text(self.output_padding),
            "vision_input_dtype": self.input_dtype,
            "vision_weight_dtype": self.weight_dtype,
            "vision_output_dtype": self.output_dtype,
            "unsupported_vision_reason": "",
        }


@dataclass(frozen=True)
class TargetVisionDecision:
    """M11 target decision for wrapper convolution records."""

    vision_contract_ok: bool
    implementation_kind: str
    extern_symbol: str = ""
    extern_packed_func: str = ""
    vision_runtime_kind: str = ""
    vision_runtime_replacement: str = ""
    vision_runtime_replacement_available: bool = False
    vision_runtime_replacement_reason: str = ""
    vision_runtime_status: str = ""
    vision_provider_kind: str = ""
    vision_provider_abi_version: int = 0
    vision_runtime_claim: str = ""
    vision_performance_claim: bool = False
    vision_uses_host_staging: bool = False
    vision_runtime_launch_count: int = 0
    vision_artifact_call_count: int = 0
    vision_input_bytes: int = 0
    vision_weight_bytes: int = 0
    vision_output_bytes: int = 0
    vision_total_io_bytes: int = 0
    vision_host_staging_bytes: int = 0
    vision_total_accounted_bytes: int = 0
    unsupported_vision_reason: str = ""
    unsupported_vision_runtime_reason: str = ""

    def with_accounting(self, semantics: VisionConv2DSemantics) -> "TargetVisionDecision":
        """Return the same decision with deterministic M11.4 byte accounting."""
        return replace(self, **_vision_runtime_accounting_fields(semantics, self))

    def as_report_fields(self) -> dict[str, Any]:
        """Return stable JSON-able decision fields."""
        return {
            "vision_implementation_kind": self.implementation_kind,
            "vision_extern_symbol": self.extern_symbol,
            "vision_extern_packed_func": self.extern_packed_func,
            "vision_runtime_kind": self.vision_runtime_kind,
            "vision_runtime_replacement": self.vision_runtime_replacement,
            "vision_runtime_replacement_available": self.vision_runtime_replacement_available,
            "vision_runtime_replacement_reason": self.vision_runtime_replacement_reason,
            "vision_runtime_status": self.vision_runtime_status,
            "vision_provider_kind": self.vision_provider_kind,
            "vision_provider_abi_version": self.vision_provider_abi_version,
            "vision_runtime_claim": self.vision_runtime_claim,
            "vision_performance_claim": self.vision_performance_claim,
            "vision_uses_host_staging": self.vision_uses_host_staging,
            "vision_runtime_launch_count": self.vision_runtime_launch_count,
            "vision_artifact_call_count": self.vision_artifact_call_count,
            "vision_input_bytes": self.vision_input_bytes,
            "vision_weight_bytes": self.vision_weight_bytes,
            "vision_output_bytes": self.vision_output_bytes,
            "vision_total_io_bytes": self.vision_total_io_bytes,
            "vision_host_staging_bytes": self.vision_host_staging_bytes,
            "vision_total_accounted_bytes": self.vision_total_accounted_bytes,
            "unsupported_vision_reason": self.unsupported_vision_reason,
            "unsupported_vision_runtime_reason": self.unsupported_vision_runtime_reason,
        }


@dataclass(frozen=True)
class TargetVisionPolicy:
    """M11.2/M11.4 policy for wrapper conv2d records."""

    target_kind: str = "cuda"
    vision_runtime_provider: str = VISION_PROVIDER_NONE

    def decide(
        self,
        semantics: VisionConv2DSemantics,
        *,
        vision_contract_ok: bool,
    ) -> TargetVisionDecision:
        """Materialize accepted wrapper conv2d as an explicit artifact only."""
        if not vision_contract_ok:
            return TargetVisionDecision(
                vision_contract_ok=False,
                implementation_kind="unsupported",
                vision_runtime_status=VISION_RUNTIME_STATUS_UNSUPPORTED,
                unsupported_vision_reason="vision_contract_failed",
            )
        if self.vision_runtime_provider not in {
            VISION_PROVIDER_NONE,
            VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
        }:
            return TargetVisionDecision(
                vision_contract_ok=True,
                implementation_kind="unsupported",
                vision_runtime_status=VISION_RUNTIME_STATUS_UNSUPPORTED,
                vision_provider_kind=self.vision_runtime_provider,
                vision_provider_abi_version=0,
                unsupported_vision_runtime_reason=VISION_RUNTIME_PROVIDER_UNKNOWN_REASON,
            ).with_accounting(semantics)
        if self.vision_runtime_provider == VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED:
            if is_m11_6_static_conv2d_runtime_scope(semantics):
                return TargetVisionDecision(
                    vision_contract_ok=True,
                    implementation_kind=VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
                    extern_symbol=VISION_EXTERN_SYMBOL,
                    extern_packed_func=VISION_EXTERN_PACKED_FUNC,
                    vision_runtime_kind=VISION_RUNTIME_KIND_PROVIDER,
                    vision_runtime_replacement=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                    vision_runtime_replacement_available=True,
                    vision_runtime_replacement_reason=VISION_RUNTIME_PROVIDER_REASON,
                    vision_runtime_status=VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
                    vision_provider_kind=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                    vision_provider_abi_version=VISION_PROVIDER_ABI_VERSION,
                    vision_runtime_claim=VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                    vision_performance_claim=False,
                    vision_uses_host_staging=True,
                    vision_artifact_call_count=1,
                ).with_accounting(semantics)
            return TargetVisionDecision(
                vision_contract_ok=True,
                implementation_kind=VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
                extern_symbol=VISION_EXTERN_SYMBOL,
                extern_packed_func=VISION_EXTERN_PACKED_FUNC,
                vision_runtime_kind=VISION_RUNTIME_KIND_ARTIFACT_ONLY,
                vision_runtime_replacement=VISION_RUNTIME_REPLACEMENT,
                vision_runtime_replacement_available=False,
                vision_runtime_replacement_reason=VISION_RUNTIME_REPLACEMENT_REASON,
                vision_runtime_status=VISION_RUNTIME_STATUS_ARTIFACT_ONLY,
                vision_provider_kind=VISION_PROVIDER_NONE,
                vision_provider_abi_version=0,
                vision_runtime_claim="",
                vision_performance_claim=False,
                vision_uses_host_staging=False,
                vision_artifact_call_count=1,
                unsupported_vision_runtime_reason=VISION_RUNTIME_PROVIDER_SCOPE_REASON,
            ).with_accounting(semantics)
        return TargetVisionDecision(
            vision_contract_ok=True,
            implementation_kind=VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
            extern_symbol=VISION_EXTERN_SYMBOL,
            extern_packed_func=VISION_EXTERN_PACKED_FUNC,
            vision_runtime_kind=VISION_RUNTIME_KIND_ARTIFACT_ONLY,
            vision_runtime_replacement=VISION_RUNTIME_REPLACEMENT,
            vision_runtime_replacement_available=False,
            vision_runtime_replacement_reason=VISION_RUNTIME_REPLACEMENT_REASON,
            vision_runtime_status=VISION_RUNTIME_STATUS_ARTIFACT_ONLY,
            vision_provider_kind=VISION_PROVIDER_NONE,
            vision_provider_abi_version=0,
            vision_runtime_claim="",
            vision_performance_claim=False,
            vision_uses_host_staging=False,
            vision_runtime_launch_count=0,
            vision_artifact_call_count=1,
        ).with_accounting(semantics)


@dataclass(frozen=True)
class VisionGrid2DPointwiseSemantics:
    """M11.6 native Grid2D pointwise/fusion scaffold semantics."""

    case_name: str
    model: str
    grid_family: str
    x_extent: int
    y_extent: int
    op_kind: str
    block_size: int = 128
    dtype: str = "float32"

    @property
    def shape_hints(self) -> str:
        return f"x={self.x_extent}, y={self.y_extent}"

    @property
    def element_count(self) -> int:
        return int(self.x_extent) * int(self.y_extent)

    @property
    def total_io_bytes(self) -> int:
        return self.element_count * _dtype_nbytes(self.dtype) * 3


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str = "float32"


def classify_m11_captured_grid_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return M11.3 captured-grid taxonomy fields for a kernel record."""
    defaults = _empty_m11_grid_report_fields()
    status = record.get("translate_status", {})
    if status.get("ok", False) or str(record.get("blocker_class", "")) != "grid":
        return defaults

    kernel_name = str(record.get("kernel_name", ""))
    text = " ".join(
        [
            kernel_name,
            str(record.get("model_family", "")),
            str(record.get("model_case", "")),
        ]
    ).lower()
    tags = _m11_grid_operator_tags(text)
    fields = {
        "m11_grid_status": M11_GRID_STATUS_CLASSIFIED,
        "m11_grid_family": _m11_grid_family_from_tags(tags),
        "m11_grid_launch_kind": str(record.get("grid_type", "")),
        "m11_grid_program_axes": _m11_grid_program_axes(record),
        "m11_grid_size_hints": _m11_grid_size_hints(record),
        "m11_grid_operator_tags": _m11_grid_tags_text(tags),
        "unsupported_m11_grid_reason": M11_GRID_UNSUPPORTED_REASON,
    }
    fields.update(_m11_grid2d_readiness_fields(fields, record))
    return fields


def _empty_m11_grid_report_fields() -> dict[str, Any]:
    return {field: "" for field in M11_GRID_REPORT_FIELDS}


def _m11_grid2d_readiness_fields(
    taxonomy_fields: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    family = str(taxonomy_fields.get("m11_grid_family", ""))
    launch_kind = str(taxonomy_fields.get("m11_grid_launch_kind", ""))
    size_hints = _m11_grid_size_hint_values(record)
    if launch_kind != "Grid2D" or size_hints[0] <= 0 or size_hints[1] <= 0:
        return {
            "m11_grid_contract": "",
            "m11_grid_contract_version": "",
            "m11_grid_artifact_status": M11_GRID2D_ARTIFACT_REJECTED,
            "m11_grid_runtime_status": M11_GRID2D_RUNTIME_UNSUPPORTED,
            "m11_grid_implementation_kind": "",
            "m11_grid_provider_kind": "",
            "m11_grid_runtime_claim": "",
            "m11_grid_performance_claim": False,
            "m11_grid_runtime_launch_count": 0,
            "m11_grid_artifact_call_count": 0,
            "m11_grid_host_staging_bytes": 0,
            "unsupported_m11_grid_runtime_reason": M11_GRID2D_UNSUPPORTED_DYNAMIC_REASON,
        }
    if family == M11_GRID_FAMILY_CONCAT_SPLIT:
        return {
            "m11_grid_contract": VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
            "m11_grid_contract_version": VISION_GRID2D_CONTRACT_VERSION,
            "m11_grid_artifact_status": M11_GRID2D_ARTIFACT_REJECTED,
            "m11_grid_runtime_status": M11_GRID2D_RUNTIME_UNSUPPORTED,
            "m11_grid_implementation_kind": "",
            "m11_grid_provider_kind": "",
            "m11_grid_runtime_claim": "",
            "m11_grid_performance_claim": False,
            "m11_grid_runtime_launch_count": 0,
            "m11_grid_artifact_call_count": 0,
            "m11_grid_host_staging_bytes": 0,
            "unsupported_m11_grid_runtime_reason": M11_GRID2D_UNSUPPORTED_CONCAT_REASON,
        }
    if family in {
        M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
        M11_GRID_FAMILY_BN_SILU_FUSION,
    }:
        return {
            "m11_grid_contract": VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
            "m11_grid_contract_version": VISION_GRID2D_CONTRACT_VERSION,
            "m11_grid_artifact_status": M11_GRID2D_ARTIFACT_READY,
            "m11_grid_runtime_status": M11_GRID2D_RUNTIME_READY,
            "m11_grid_implementation_kind": M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
            "m11_grid_provider_kind": M11_GRID2D_PROVIDER_NATIVE_TVM,
            "m11_grid_runtime_claim": M11_GRID2D_RUNTIME_CLAIM,
            "m11_grid_performance_claim": True,
            "m11_grid_runtime_launch_count": 1,
            "m11_grid_artifact_call_count": 1,
            "m11_grid_host_staging_bytes": 0,
            "unsupported_m11_grid_runtime_reason": "",
        }
    return {
        "m11_grid_contract": VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        "m11_grid_contract_version": VISION_GRID2D_CONTRACT_VERSION,
        "m11_grid_artifact_status": M11_GRID2D_ARTIFACT_REJECTED,
        "m11_grid_runtime_status": M11_GRID2D_RUNTIME_UNSUPPORTED,
        "m11_grid_implementation_kind": "",
        "m11_grid_provider_kind": "",
        "m11_grid_runtime_claim": "",
        "m11_grid_performance_claim": False,
        "m11_grid_runtime_launch_count": 0,
        "m11_grid_artifact_call_count": 0,
        "m11_grid_host_staging_bytes": 0,
        "unsupported_m11_grid_runtime_reason": "grid2d_family_deferred_m11_6",
    }


def _m11_grid_operator_tags(text: str) -> tuple[str, ...]:
    tags: list[str] = []
    if "pool" in text:
        tags.append("pool")
    if "softmax" in text:
        tags.append("softmax")
    if any(
        token in text
        for token in ("decode", "postprocess", "nms", "sigmoid", "stack", "fill_select")
    ) or "unsafe_index" in text:
        tags.append("yolo_decode_postprocess")
    if "native_batch_norm" in text or "batch_norm" in text or "_bn_" in text:
        tags.append("batch_norm")
    if "silu" in text:
        tags.append("silu")
    if "cat" in text:
        tags.append("cat")
    if "split" in text:
        tags.append("split")
    if "view" in text:
        tags.append("view")
    if "convolution" in text or "conv" in text:
        tags.append("convolution")
    if not tags:
        tags.append("multidim_pointwise")
    return tuple(dict.fromkeys(tags))


def _m11_grid_family_from_tags(tags: tuple[str, ...]) -> str:
    tag_set = set(tags)
    if tag_set.intersection({"pool", "softmax"}):
        return M11_GRID_FAMILY_POOL_OR_SOFTMAX
    if "yolo_decode_postprocess" in tag_set:
        return M11_GRID_FAMILY_YOLO_DECODE_POSTPROCESS
    if "batch_norm" in tag_set:
        return M11_GRID_FAMILY_BN_SILU_FUSION
    if tag_set.intersection({"cat", "split", "view"}):
        return M11_GRID_FAMILY_CONCAT_SPLIT
    if tag_set.intersection({"convolution", "silu"}):
        return M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE
    return M11_GRID_FAMILY_OTHER_MULTIDIM_POINTWISE


def _m11_grid_program_axes(record: dict[str, Any]) -> str:
    indexing_summary = record.get("indexing_summary", {}) or {}
    axes = indexing_summary.get("program_id_axes", [])
    if not axes:
        axes = list((record.get("size_hints", {}) or {}).keys())
    return ", ".join(str(axis) for axis in axes)


def _m11_grid_size_hints(record: dict[str, Any]) -> str:
    size_hints = record.get("size_hints", {}) or {}
    if not size_hints:
        return ""
    ordered_keys = [key for key in ("x", "y", "z") if key in size_hints]
    ordered_keys.extend(sorted(str(key) for key in size_hints if key not in {"x", "y", "z"}))
    return ", ".join(f"{key}={size_hints[key]}" for key in ordered_keys)


def _m11_grid_size_hint_values(record: dict[str, Any]) -> tuple[int, int]:
    size_hints = record.get("size_hints", {}) or {}
    try:
        return (int(size_hints.get("x", 0) or 0), int(size_hints.get("y", 0) or 0))
    except (TypeError, ValueError):
        return (0, 0)


def _m11_grid_tags_text(tags: tuple[str, ...]) -> str:
    return ", ".join(tags)


def wrapper_conv2d_vision_report_fields(
    *,
    op_name: str,
    op_family: str,
    source: str = "",
    full_source: str = "",
    line_no: int = 0,
    case_name: str = "",
    model_case: str = "",
    vision_runtime_provider: str = VISION_PROVIDER_NONE,
) -> dict[str, Any]:
    """Return M11 report fields for one wrapper ``extern_kernels.convolution`` call."""
    if op_family != "deferred_convolution":
        return _default_report_fields()

    fields = _default_report_fields()
    fields.update(
        {
            "vision_source_kind": VISION_SOURCE_KIND_WRAPPER_CONV2D,
            "vision_contract_version": VISION_CONTRACT_VERSION,
            "vision_op_family": VISION_OP_FAMILY_CONVOLUTION,
            "vision_runtime_status": VISION_RUNTIME_STATUS_UNSUPPORTED,
        }
    )
    try:
        semantics = extract_vision_conv2d_semantics_from_wrapper_extern(
            {
                "op_name": op_name,
                "source": source,
                "full_source": full_source,
                "line_no": line_no,
                "case_name": case_name or model_case,
                "kernel_name": case_name or model_case or "wrapper_extern_conv2d",
            }
        )
        decision = TargetVisionPolicy(
            vision_runtime_provider=vision_runtime_provider,
        ).decide(semantics, vision_contract_ok=True)
    except UnsupportedTTIROpError as err:
        fields.update(
            {
                "vision_semantics_status": VISION_SEMANTICS_STATUS_UNSUPPORTED,
                "unsupported_vision_reason": str(err),
            }
        )
        return fields

    fields.update(semantics.as_report_fields())
    fields.update(decision.as_report_fields())
    return fields


def extract_vision_conv2d_semantics_from_wrapper_extern(
    source_or_call: Any,
    *,
    case_name: str = "",
    kernel_name: str = "",
) -> VisionConv2DSemantics:
    """Extract M11 conv2d semantics from an Inductor wrapper call."""
    source, full_source, op_name, line_no, call_case, call_kernel = _wrapper_call_fields(
        source_or_call
    )
    case_name = case_name or call_case
    kernel_name = kernel_name or call_kernel or case_name or "wrapper_extern_conv2d"
    if op_name and op_name != VISION_EXTERN_SYMBOL:
        raise UnsupportedTTIROpError(f"unsupported vision wrapper op {op_name!r}")
    if not full_source:
        full_source = source
    if not full_source:
        raise UnsupportedTTIROpError("vision_conv2d_requires_wrapper_source")

    context = _extract_conv2d_context(full_source, line_no=line_no)
    call = context["call"]
    input_name = context["input_name"]
    weight_name = context["weight_name"]
    output_name = context["output_name"]
    input_meta = context["input_meta"]
    weight_meta = context["weight_meta"]
    output_meta = context["output_meta"]

    if input_meta is None:
        raise UnsupportedTTIROpError("vision_conv2d_missing_input_metadata")
    if weight_meta is None:
        raise UnsupportedTTIROpError("vision_conv2d_missing_weight_metadata")
    if output_meta is None:
        raise UnsupportedTTIROpError("vision_conv2d_missing_output_metadata")

    stride = _parse_int_tuple_keyword(call, "stride", required=True)
    padding = _parse_int_tuple_keyword(call, "padding", required=True)
    dilation = _parse_int_tuple_keyword(call, "dilation", required=True)
    output_padding = _parse_int_tuple_keyword(call, "output_padding", required=True)
    groups = _parse_int_keyword(call, "groups", required=True)
    transposed = _parse_bool_keyword(call, "transposed", required=True)
    bias_policy = _parse_bias_policy(call)

    if len(input_meta.shape) != 4:
        raise UnsupportedTTIROpError("vision_conv2d_requires_rank4_input")
    if len(weight_meta.shape) != 4:
        raise UnsupportedTTIROpError("vision_conv2d_requires_rank4_weight")
    if len(output_meta.shape) != 4:
        raise UnsupportedTTIROpError("vision_conv2d_requires_rank4_output")
    if input_meta.dtype != "float32" or weight_meta.dtype != "float32":
        raise UnsupportedTTIROpError("vision_conv2d_requires_fp32_input_weight")
    output_dtype = output_meta.dtype or input_meta.dtype
    if output_dtype != "float32":
        raise UnsupportedTTIROpError("vision_conv2d_requires_fp32_output")
    if transposed:
        raise UnsupportedTTIROpError("vision_conv2d_transposed_not_supported_m11_2")
    if output_padding != (0, 0):
        raise UnsupportedTTIROpError("vision_conv2d_output_padding_not_supported_m11_2")
    if bias_policy != "none":
        raise UnsupportedTTIROpError("vision_conv2d_bias_not_supported_m11_2")
    if groups <= 0:
        raise UnsupportedTTIROpError("vision_conv2d_groups_must_be_positive")
    if len(stride) != 2 or any(value <= 0 for value in stride):
        raise UnsupportedTTIROpError("vision_conv2d_stride_must_be_positive_pair")
    if len(padding) != 2 or any(value < 0 for value in padding):
        raise UnsupportedTTIROpError("vision_conv2d_padding_must_be_non_negative_pair")
    if len(dilation) != 2 or any(value <= 0 for value in dilation):
        raise UnsupportedTTIROpError("vision_conv2d_dilation_must_be_positive_pair")

    input_shape = _rank4(input_meta.shape, "input")
    weight_shape = _rank4(weight_meta.shape, "weight")
    output_shape = _rank4(output_meta.shape, "output")
    input_stride = _rank4(input_meta.stride, "input_stride")
    weight_stride = _rank4(weight_meta.stride, "weight_stride")
    output_stride = _rank4(output_meta.stride, "output_stride")

    if weight_shape[1] * groups != input_shape[1]:
        raise UnsupportedTTIROpError("vision_conv2d_group_input_channel_mismatch")
    expected_output = _conv2d_output_shape(
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
    )
    if output_shape != expected_output:
        raise UnsupportedTTIROpError("vision_conv2d_output_shape_mismatch")

    contract = _conv2d_contract(input_shape, weight_shape, groups)
    return VisionConv2DSemantics(
        source_kind=VISION_SOURCE_KIND_WRAPPER_CONV2D,
        source_name=VISION_EXTERN_SYMBOL,
        kernel_name=kernel_name,
        vision_contract=contract,
        vision_op_family=VISION_OP_FAMILY_CONVOLUTION,
        input_param=input_name,
        weight_param=weight_name,
        output_param=output_name,
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        input_stride=input_stride,
        weight_stride=weight_stride,
        output_stride=output_stride,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_policy=bias_policy,
        transposed=transposed,
        output_padding=output_padding,
        input_dtype=input_meta.dtype,
        weight_dtype=weight_meta.dtype,
        output_dtype=output_dtype,
    )


def build_extern_conv2d_tirx_source(
    semantics: VisionConv2DSemantics,
    decision: TargetVisionDecision,
    *,
    target_attrs: str = "cuda",
) -> str:
    """Build the explicit M11.2 packed-call artifact for wrapper conv2d."""
    if semantics.vision_contract not in VISION_CONV2D_CONTRACTS:
        raise UnsupportedTTIROpError("extern conv2d artifact requires an M11 conv2d contract")
    if (
        decision.implementation_kind != VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D
        or decision.extern_symbol != VISION_EXTERN_SYMBOL
    ):
        raise UnsupportedTTIROpError("extern conv2d artifact requires extern_conv2d decision")

    func_name = _sanitize_identifier(semantics.kernel_name or "wrapper_extern_conv2d")
    input_name = _sanitize_identifier(semantics.input_param or "inp")
    weight_name = _sanitize_identifier(semantics.weight_param or "weight")
    output_name = _sanitize_identifier(semantics.output_param or "out")
    input_buffer_stride = semantics.input_stride
    weight_buffer_stride = semantics.weight_stride
    output_buffer_stride = semantics.output_stride
    if decision.vision_runtime_status == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED:
        input_buffer_stride = _compact_rank4_stride(semantics.input_shape)
        weight_buffer_stride = _compact_rank4_stride(semantics.weight_shape)
        output_buffer_stride = _compact_rank4_stride(semantics.output_shape)
    lines: list[str] = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}({input_name}_handle: T.handle, "
        f"{weight_name}_handle: T.handle, {output_name}_handle: T.handle):",
        "        T.func_attr({"
        f'"global_symbol": "{semantics.kernel_name}", '
        '"tirx.noalias": True, '
        f'"target": T.target({target_attrs!r}), '
        f'"triton_tvm.contract": "{semantics.vision_contract}", '
        f'"triton_tvm.vision_source_kind": "{semantics.source_kind}", '
        f'"triton_tvm.vision_contract": "{semantics.vision_contract}", '
        f'"triton_tvm.vision_contract_version": "{VISION_CONTRACT_VERSION}", '
        f'"triton_tvm.vision_op_family": "{semantics.vision_op_family}", '
        f'"triton_tvm.vision_layout": "{semantics.vision_layout}", '
        f'"triton_tvm.input_shape": "{_tuple_text(semantics.input_shape)}", '
        f'"triton_tvm.weight_shape": "{_tuple_text(semantics.weight_shape)}", '
        f'"triton_tvm.output_shape": "{_tuple_text(semantics.output_shape)}", '
        f'"triton_tvm.input_stride": "{_tuple_text(semantics.input_stride)}", '
        f'"triton_tvm.weight_stride": "{_tuple_text(semantics.weight_stride)}", '
        f'"triton_tvm.output_stride": "{_tuple_text(semantics.output_stride)}", '
        f'"triton_tvm.stride": "{_tuple_text(semantics.stride)}", '
        f'"triton_tvm.padding": "{_tuple_text(semantics.padding)}", '
        f'"triton_tvm.dilation": "{_tuple_text(semantics.dilation)}", '
        f'"triton_tvm.groups": {semantics.groups}, '
        f'"triton_tvm.bias_policy": "{semantics.bias_policy}", '
        f'"triton_tvm.transposed": {_bool_literal(semantics.transposed)}, '
        f'"triton_tvm.output_padding": "{_tuple_text(semantics.output_padding)}", '
        f'"triton_tvm.input_dtype": "{semantics.input_dtype}", '
        f'"triton_tvm.weight_dtype": "{semantics.weight_dtype}", '
        f'"triton_tvm.output_dtype": "{semantics.output_dtype}", '
        f'"triton_tvm.implementation_kind": "{decision.implementation_kind}", '
        f'"triton_tvm.extern_symbol": "{decision.extern_symbol}", '
        f'"triton_tvm.extern_packed_func": "{decision.extern_packed_func}", '
        f'"triton_tvm.extern_runtime_kind": "{decision.vision_runtime_kind}", '
        f'"triton_tvm.extern_runtime_replacement": "{decision.vision_runtime_replacement}", '
        f'"triton_tvm.extern_runtime_replacement_available": '
        f'{_bool_literal(decision.vision_runtime_replacement_available)}, '
        f'"triton_tvm.extern_runtime_replacement_reason": '
        f'"{decision.vision_runtime_replacement_reason}", '
        f'"triton_tvm.vision_runtime_status": "{decision.vision_runtime_status}", '
        f'"triton_tvm.vision_provider_kind": "{decision.vision_provider_kind}", '
        f'"triton_tvm.vision_provider_abi_version": '
        f'{decision.vision_provider_abi_version}, '
        f'"triton_tvm.vision_runtime_claim": "{decision.vision_runtime_claim}", '
        f'"triton_tvm.vision_performance_claim": '
        f'{_bool_literal(decision.vision_performance_claim)}, '
        f'"triton_tvm.vision_uses_host_staging": '
        f'{_bool_literal(decision.vision_uses_host_staging)}, '
        f'"triton_tvm.vision_runtime_launch_count": {decision.vision_runtime_launch_count}, '
        f'"triton_tvm.vision_artifact_call_count": {decision.vision_artifact_call_count}, '
        f'"triton_tvm.vision_input_bytes": {decision.vision_input_bytes}, '
        f'"triton_tvm.vision_weight_bytes": {decision.vision_weight_bytes}, '
        f'"triton_tvm.vision_output_bytes": {decision.vision_output_bytes}, '
        f'"triton_tvm.vision_total_io_bytes": {decision.vision_total_io_bytes}, '
        f'"triton_tvm.vision_host_staging_bytes": {decision.vision_host_staging_bytes}, '
        f'"triton_tvm.vision_total_accounted_bytes": '
        f'{decision.vision_total_accounted_bytes}, '
        f'"triton_tvm.unsupported_vision_runtime_reason": '
        f'"{decision.unsupported_vision_runtime_reason}"'
        "})",
        "        "
        f"{input_name} = T.match_buffer({input_name}_handle, "
        f"({_tuple_text(semantics.input_shape)}), "
        f'"{semantics.input_dtype}", strides=({_tuple_text(input_buffer_stride)}))',
        "        "
        f"{weight_name} = T.match_buffer({weight_name}_handle, "
        f"({_tuple_text(semantics.weight_shape)}), "
        f'"{semantics.weight_dtype}", strides=({_tuple_text(weight_buffer_stride)}))',
        "        "
        f"{output_name} = T.match_buffer({output_name}_handle, "
        f"({_tuple_text(semantics.output_shape)}), "
        f'"{semantics.output_dtype}", strides=({_tuple_text(output_buffer_stride)}))',
        "        "
        f'T.evaluate(T.call_packed("{VISION_EXTERN_PACKED_FUNC}", '
        f"{input_name}, {weight_name}, {output_name}, "
        f"{semantics.stride[0]}, {semantics.stride[1]}, "
        f"{semantics.padding[0]}, {semantics.padding[1]}, "
        f"{semantics.dilation[0]}, {semantics.dilation[1]}, {semantics.groups}))",
    ]
    return "\n".join(lines) + "\n"


def build_native_grid2d_pointwise_tirx_source(
    semantics: VisionGrid2DPointwiseSemantics,
    *,
    target_attrs: str = "cuda",
) -> str:
    """Build a native TVM Grid2D pointwise/fusion scaffold artifact."""
    if semantics.x_extent <= 0 or semantics.y_extent <= 0:
        raise UnsupportedTTIROpError("grid2d pointwise requires positive static extents")
    if semantics.block_size <= 0:
        raise UnsupportedTTIROpError("grid2d pointwise requires a positive block size")
    if semantics.dtype != "float32":
        raise UnsupportedTTIROpError("grid2d pointwise M11.6 supports float32 only")
    if semantics.op_kind not in _GRID2D_POINTWISE_OPS:
        raise UnsupportedTTIROpError(f"unsupported grid2d pointwise op {semantics.op_kind!r}")

    func_name = _sanitize_identifier(semantics.case_name)
    x_extent = int(semantics.x_extent)
    y_extent = int(semantics.y_extent)
    block_size = int(semantics.block_size)
    op_expr = _grid2d_pointwise_expr(semantics.op_kind)
    lines: list[str] = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}(x_handle: T.handle, y_handle: T.handle, out_handle: T.handle):",
        "        T.func_attr({"
        f'"global_symbol": "{func_name}", '
        '"tirx.noalias": True, '
        f'"target": T.target({target_attrs!r}), '
        f'"triton_tvm.contract": "{VISION_CONTRACT_POINTWISE_GRID2D_STATIC}", '
        f'"triton_tvm.grid2d_contract": "{VISION_CONTRACT_POINTWISE_GRID2D_STATIC}", '
        f'"triton_tvm.grid2d_contract_version": "{VISION_GRID2D_CONTRACT_VERSION}", '
        f'"triton_tvm.grid2d_family": "{semantics.grid_family}", '
        f'"triton_tvm.grid2d_op_kind": "{semantics.op_kind}", '
        f'"triton_tvm.grid2d_shape_hints": "{semantics.shape_hints}", '
        f'"triton_tvm.grid2d_x_extent": {x_extent}, '
        f'"triton_tvm.grid2d_y_extent": {y_extent}, '
        f'"triton_tvm.grid2d_block_size": {block_size}, '
        f'"triton_tvm.implementation_kind": "{M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM}", '
        f'"triton_tvm.grid2d_runtime_status": "{M11_GRID2D_RUNTIME_READY}", '
        f'"triton_tvm.grid2d_provider_kind": "{M11_GRID2D_PROVIDER_NATIVE_TVM}", '
        f'"triton_tvm.grid2d_runtime_claim": "{M11_GRID2D_RUNTIME_CLAIM}", '
        '"triton_tvm.grid2d_performance_claim": True, '
        '"triton_tvm.grid2d_uses_host_staging": False, '
        '"triton_tvm.grid2d_runtime_launch_count": 1, '
        '"triton_tvm.grid2d_artifact_call_count": 1, '
        f'"triton_tvm.grid2d_total_io_bytes": {semantics.total_io_bytes}, '
        '"triton_tvm.grid2d_host_staging_bytes": 0'
        "})",
        f'        x = T.match_buffer(x_handle, ({y_extent}, {x_extent}), "{semantics.dtype}")',
        f'        y = T.match_buffer(y_handle, ({y_extent}, {x_extent}), "{semantics.dtype}")',
        f'        out = T.match_buffer(out_handle, ({y_extent}, {x_extent}), "{semantics.dtype}")',
        f'        for by in T.thread_binding(0, {y_extent}, thread="blockIdx.y"):',
        "            "
        f'for bx in T.thread_binding(0, T.ceildiv(T.int64({x_extent}), '
        f'T.int64({block_size})), thread="blockIdx.x"):',
        f'                for tx in T.thread_binding(0, {block_size}, thread="threadIdx.x"):',
        f'                    ix = T.Cast("int64", bx) * T.int64({block_size}) + '
        'T.Cast("int64", tx)',
        '                    iy = T.Cast("int64", by)',
        f"                    if ix < T.int64({x_extent}):",
        f"                        out[iy, ix] = {op_expr}",
    ]
    return "\n".join(lines) + "\n"


_GRID2D_POINTWISE_OPS = {
    "add",
    "mul",
    "sub",
    "select_relu",
    "silu",
    "bn_affine_silu",
}


def _grid2d_pointwise_expr(op_kind: str) -> str:
    x_value = "x[iy, ix]"
    y_value = "y[iy, ix]"
    if op_kind == "add":
        return f"{x_value} + {y_value}"
    if op_kind == "mul":
        return f"{x_value} * {y_value}"
    if op_kind == "sub":
        return f"{x_value} - {y_value}"
    if op_kind == "select_relu":
        return f"T.Select({x_value} > T.float32(0.0), {x_value}, {y_value})"
    if op_kind == "silu":
        return f"{x_value} / (T.float32(1.0) + T.exp(T.float32(0.0) - {x_value}))"
    if op_kind == "bn_affine_silu":
        affine = f"({x_value} * {y_value})"
        return f"{affine} / (T.float32(1.0) + T.exp(T.float32(0.0) - {affine}))"
    raise UnsupportedTTIROpError(f"unsupported grid2d pointwise op {op_kind!r}")


def _extract_conv2d_context(full_source: str, *, line_no: int) -> dict[str, Any]:
    tree = ast.parse(full_source)
    env: dict[str, _TensorMetadata] = {}
    target_call: ast.Call | None = None
    input_name = ""
    weight_name = ""
    output_name = ""
    input_meta: _TensorMetadata | None = None
    weight_meta: _TensorMetadata | None = None
    output_meta: _TensorMetadata | None = None

    for stmt in _iter_statements(tree.body):
        conv_assignment = _conv_assignment(stmt)
        if target_call is None and conv_assignment is not None:
            call, assigned_output = conv_assignment
            call_line = int(getattr(call, "lineno", 0) or 0)
            if line_no > 0 and call_line != line_no:
                _update_tensor_env(stmt, env)
                continue
            output_name = assigned_output
            if not output_name:
                raise UnsupportedTTIROpError("vision_conv2d_output_assignment_required")
            if len(call.args) < 2:
                raise UnsupportedTTIROpError("vision_conv2d_requires_input_and_weight")
            input_name = _name_from_expr(call.args[0])
            weight_name = _name_from_expr(call.args[1])
            if not input_name or not weight_name:
                raise UnsupportedTTIROpError("vision_conv2d_requires_named_input_weight")
            target_call = call
            input_meta = env.get(input_name)
            weight_meta = env.get(weight_name)
            continue

        if target_call is None:
            _update_tensor_env(stmt, env)
            continue

        output_meta = _assert_size_stride_metadata(stmt, output_name, env.get(output_name))
        if output_meta is not None:
            break
        alias = _alias_target(stmt, output_name)
        if alias:
            output_name = alias

    if target_call is None:
        raise UnsupportedTTIROpError("vision_conv2d_call_not_found")
    return {
        "call": target_call,
        "input_name": input_name,
        "weight_name": weight_name,
        "output_name": output_name,
        "input_meta": input_meta,
        "weight_meta": weight_meta,
        "output_meta": output_meta,
    }


def _iter_statements(statements: list[ast.stmt]) -> Any:
    for stmt in statements:
        yield stmt
        nested: list[list[ast.stmt]] = []
        for attr_name in ("body", "orelse", "finalbody"):
            value = getattr(stmt, attr_name, None)
            if isinstance(value, list):
                nested.append(value)
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                nested.append(handler.body)
        for body in nested:
            yield from _iter_statements(body)


def _conv_assignment(stmt: ast.stmt) -> tuple[ast.Call, str] | None:
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        if _attribute_chain(stmt.value.func) != VISION_EXTERN_SYMBOL:
            return None
        return stmt.value, _single_assignment_target(stmt)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        if _attribute_chain(stmt.value.func) == VISION_EXTERN_SYMBOL:
            return stmt.value, ""
    if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
        if _attribute_chain(stmt.value.func) == VISION_EXTERN_SYMBOL:
            return stmt.value, ""
    return None


def _update_tensor_env(stmt: ast.stmt, env: dict[str, _TensorMetadata]) -> None:
    if isinstance(stmt, ast.Expr):
        call = stmt.value
        if isinstance(call, ast.Call):
            metadata = _assert_size_stride_metadata(stmt, "", None)
            if metadata is not None:
                target = _name_from_expr(call.args[0]) if call.args else ""
                if target:
                    env[target] = metadata
        return

    if not isinstance(stmt, ast.Assign):
        return
    target = _single_assignment_target(stmt)
    if not target:
        return
    value = stmt.value
    if isinstance(value, ast.Name) and value.id in env:
        env[target] = env[value.id]
        return
    if isinstance(value, ast.Call):
        func_name = _attribute_chain(value.func)
        if func_name in {
            "empty_strided_cuda",
            "empty_strided",
            "rand_strided",
        }:
            metadata = _metadata_from_shape_stride_call(value, existing=None)
            if metadata is not None:
                env[target] = metadata
            return
        if func_name == "reinterpret_tensor":
            metadata = _metadata_from_reinterpret_tensor(value, env)
            if metadata is not None:
                env[target] = metadata
            return
        if func_name == "copy_misaligned" and value.args:
            source = _name_from_expr(value.args[0])
            if source in env:
                env[target] = env[source]


def _assert_size_stride_metadata(
    stmt: ast.stmt,
    target_name: str,
    existing: _TensorMetadata | None,
) -> _TensorMetadata | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if _attribute_chain(call.func) != "assert_size_stride":
        return None
    if len(call.args) < 3:
        return None
    observed_target = _name_from_expr(call.args[0])
    if target_name and observed_target != target_name:
        return None
    shape = _literal_int_tuple(call.args[1])
    stride = _literal_int_tuple(call.args[2])
    if shape is None or stride is None:
        return None
    dtype = existing.dtype if existing is not None else "float32"
    return _TensorMetadata(shape=shape, stride=stride, dtype=dtype)


def _metadata_from_shape_stride_call(
    call: ast.Call,
    *,
    existing: _TensorMetadata | None,
) -> _TensorMetadata | None:
    if len(call.args) < 2:
        return None
    shape = _literal_int_tuple(call.args[0])
    stride = _literal_int_tuple(call.args[1])
    if shape is None or stride is None:
        return None
    dtype = existing.dtype if existing is not None else "float32"
    if len(call.args) >= 3:
        dtype = _dtype_from_expr(call.args[2]) or dtype
    for keyword in call.keywords:
        if keyword.arg == "dtype":
            dtype = _dtype_from_expr(keyword.value) or dtype
    return _TensorMetadata(shape=shape, stride=stride, dtype=dtype)


def _metadata_from_reinterpret_tensor(
    call: ast.Call,
    env: dict[str, _TensorMetadata],
) -> _TensorMetadata | None:
    if len(call.args) < 3:
        return None
    source = _name_from_expr(call.args[0])
    shape = _literal_int_tuple(call.args[1])
    stride = _literal_int_tuple(call.args[2])
    if shape is None or stride is None:
        return None
    dtype = env[source].dtype if source in env else "float32"
    return _TensorMetadata(shape=shape, stride=stride, dtype=dtype)


def _alias_target(stmt: ast.stmt, source_name: str) -> str:
    if not isinstance(stmt, ast.Assign):
        return ""
    if isinstance(stmt.value, ast.Name) and stmt.value.id == source_name:
        return _single_assignment_target(stmt)
    return ""


def _single_assignment_target(stmt: ast.Assign) -> str:
    if len(stmt.targets) != 1:
        return ""
    return _name_from_expr(stmt.targets[0])


def _wrapper_call_fields(source_or_call: Any) -> tuple[str, str, str, int, str, str]:
    if isinstance(source_or_call, dict):
        return (
            str(source_or_call.get("source", "")),
            str(source_or_call.get("full_source", "")),
            str(source_or_call.get("op_name", "")),
            int(source_or_call.get("line_no", 0) or 0),
            str(source_or_call.get("case_name", "")),
            str(source_or_call.get("kernel_name", "")),
        )
    return str(source_or_call), str(source_or_call), "", 0, "", ""


def _parse_int_tuple_keyword(
    call: ast.Call,
    name: str,
    *,
    required: bool,
) -> tuple[int, int]:
    for keyword in call.keywords:
        if keyword.arg == name:
            value = _literal_int_tuple(keyword.value)
            if value is None or len(value) != 2:
                raise UnsupportedTTIROpError(f"vision_conv2d_{name}_must_be_pair")
            return (int(value[0]), int(value[1]))
    if required:
        raise UnsupportedTTIROpError(f"vision_conv2d_missing_{name}")
    return (0, 0)


def _parse_int_keyword(call: ast.Call, name: str, *, required: bool) -> int:
    for keyword in call.keywords:
        if keyword.arg == name:
            value = _literal_int(keyword.value)
            if value is None:
                raise UnsupportedTTIROpError(f"vision_conv2d_{name}_must_be_int")
            return value
    if required:
        raise UnsupportedTTIROpError(f"vision_conv2d_missing_{name}")
    return 0


def _parse_bool_keyword(call: ast.Call, name: str, *, required: bool) -> bool:
    for keyword in call.keywords:
        if keyword.arg == name:
            value = _literal_bool(keyword.value)
            if value is None:
                raise UnsupportedTTIROpError(f"vision_conv2d_{name}_must_be_bool")
            return value
    if required:
        raise UnsupportedTTIROpError(f"vision_conv2d_missing_{name}")
    return False


def _parse_bias_policy(call: ast.Call) -> str:
    for keyword in call.keywords:
        if keyword.arg == "bias":
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                return "none"
            return "bias"
    if len(call.args) >= 3:
        if isinstance(call.args[2], ast.Constant) and call.args[2].value is None:
            return "none"
        return "bias"
    return "none"


def _conv2d_output_shape(
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    batch, _, input_h, input_w = input_shape
    output_channels, _, kernel_h, kernel_w = weight_shape
    output_h = (input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    output_w = (input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    if output_h <= 0 or output_w <= 0:
        raise UnsupportedTTIROpError("vision_conv2d_output_shape_must_be_positive")
    return (batch, output_channels, output_h, output_w)


def _conv2d_contract(
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    groups: int,
) -> str:
    input_channels = input_shape[1]
    output_channels, channels_per_group, kernel_h, kernel_w = weight_shape
    if groups == input_channels and output_channels == input_channels and channels_per_group == 1:
        return VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC
    if groups > 1:
        return VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC
    if kernel_h == 1 and kernel_w == 1:
        return VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC
    return VISION_CONTRACT_CONV2D_NCHW_STATIC


def _is_m11_4_vit_patch_conv2d(semantics: VisionConv2DSemantics) -> bool:
    return (
        semantics.vision_contract == VISION_CONTRACT_CONV2D_NCHW_STATIC
        and semantics.input_shape == VISION_M11_4_VIT_PATCH_INPUT_SHAPE
        and semantics.weight_shape == VISION_M11_4_VIT_PATCH_WEIGHT_SHAPE
        and semantics.output_shape == VISION_M11_4_VIT_PATCH_OUTPUT_SHAPE
        and semantics.stride == VISION_M11_4_VIT_PATCH_STRIDE
        and semantics.padding == VISION_M11_4_VIT_PATCH_PADDING
        and semantics.dilation == VISION_M11_4_VIT_PATCH_DILATION
        and semantics.groups == 1
        and semantics.bias_policy == "none"
        and not semantics.transposed
        and semantics.output_padding == (0, 0)
        and semantics.input_dtype == "float32"
        and semantics.weight_dtype == "float32"
        and semantics.output_dtype == "float32"
    )


def is_m11_6_static_conv2d_runtime_scope(semantics: VisionConv2DSemantics) -> bool:
    """Return whether M11.6 admits a conv2d record to the correctness provider."""
    if (
        semantics.input_shape[0] != 1
        or semantics.groups != 1
        or semantics.bias_policy != "none"
        or semantics.transposed
        or semantics.output_padding != (0, 0)
        or semantics.dilation != (1, 1)
        or semantics.input_dtype != "float32"
        or semantics.weight_dtype != "float32"
        or semantics.output_dtype != "float32"
    ):
        return False
    if semantics.vision_contract == VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC:
        return (
            semantics.weight_shape[2:] == (1, 1)
            and semantics.stride == (1, 1)
            and semantics.padding == (0, 0)
        )
    if semantics.vision_contract != VISION_CONTRACT_CONV2D_NCHW_STATIC:
        return False
    if semantics.stride[0] != semantics.stride[1] or semantics.padding[0] != semantics.padding[1]:
        return False
    return semantics.stride[0] in {1, 2, 16} and semantics.padding[0] in {0, 1}


def _vision_runtime_accounting_fields(
    semantics: VisionConv2DSemantics,
    decision: TargetVisionDecision,
) -> dict[str, int]:
    dtype_bytes = _dtype_nbytes(semantics.input_dtype)
    weight_dtype_bytes = _dtype_nbytes(semantics.weight_dtype)
    output_dtype_bytes = _dtype_nbytes(semantics.output_dtype)
    input_bytes = _numel(semantics.input_shape) * dtype_bytes
    weight_bytes = _numel(semantics.weight_shape) * weight_dtype_bytes
    output_bytes = _numel(semantics.output_shape) * output_dtype_bytes
    total_io_bytes = input_bytes + weight_bytes + output_bytes
    host_staging_bytes = total_io_bytes if decision.vision_uses_host_staging else 0
    runtime_launch_count = (
        1
        if decision.vision_runtime_status == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED
        else 0
    )
    return {
        "vision_runtime_launch_count": runtime_launch_count,
        "vision_input_bytes": input_bytes,
        "vision_weight_bytes": weight_bytes,
        "vision_output_bytes": output_bytes,
        "vision_total_io_bytes": total_io_bytes,
        "vision_host_staging_bytes": host_staging_bytes,
        "vision_total_accounted_bytes": total_io_bytes + host_staging_bytes,
    }


def _dtype_nbytes(dtype: str) -> int:
    if dtype in {"float16", "bfloat16"}:
        return 2
    if dtype == "float64":
        return 8
    return 4


def _compact_rank4_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (channels * height * width, height * width, width, 1)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _rank4(values: tuple[int, ...], label: str) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise UnsupportedTTIROpError(f"vision_conv2d_{label}_must_be_rank4")
    if any(value <= 0 for value in values):
        raise UnsupportedTTIROpError(f"vision_conv2d_{label}_must_be_positive")
    return (values[0], values[1], values[2], values[3])


def _nchw_layout(shape: tuple[int, ...], stride: tuple[int, ...]) -> str:
    if len(shape) != 4 or len(stride) != 4:
        return "nchw_unknown"
    _, channels, height, width = shape
    contiguous = (channels * height * width, height * width, width, 1)
    channels_last = (channels * height * width, 1, width * channels, channels)
    if stride == contiguous:
        return "nchw_contiguous"
    if stride == channels_last:
        return "nchw_channels_last"
    return "nchw_strided"


def _oihw_layout(shape: tuple[int, ...], stride: tuple[int, ...]) -> str:
    if len(shape) != 4 or len(stride) != 4:
        return "oihw_unknown"
    _, input_channels, height, width = shape
    contiguous = (input_channels * height * width, height * width, width, 1)
    channels_last = (input_channels * height * width, 1, width * input_channels, input_channels)
    if stride == contiguous:
        return "oihw_contiguous"
    if stride == channels_last:
        return "oihw_channels_last"
    return "oihw_strided"


def _default_report_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field_name in VISION_REPORT_FIELDS:
        if field_name in {
            "vision_groups",
            "vision_provider_abi_version",
            "vision_runtime_launch_count",
            "vision_artifact_call_count",
            "vision_input_bytes",
            "vision_weight_bytes",
            "vision_output_bytes",
            "vision_total_io_bytes",
            "vision_host_staging_bytes",
            "vision_total_accounted_bytes",
        }:
            fields[field_name] = 0
        elif field_name in {
            "vision_transposed",
            "vision_runtime_replacement_available",
            "vision_performance_claim",
            "vision_uses_host_staging",
        }:
            fields[field_name] = False
        else:
            fields[field_name] = ""
    return fields


class _VisionProviderRegistration:
    """Restorable packed-func registration for the M11.4 conv2d provider."""

    def __init__(self, name: str, previous: Any):
        self._name = name
        self._previous = previous
        self._closed = False

    def close(self) -> None:
        """Restore the previous packed function or remove the conv2d provider."""
        if self._closed:
            return
        import tvm  # pylint: disable=import-outside-toplevel
        import tvm_ffi  # pylint: disable=import-outside-toplevel

        if self._previous is None:
            tvm_ffi.remove_global_func(self._name)
        else:
            tvm.register_global_func(self._name, self._previous, override=True)
        self._closed = True

    def __enter__(self) -> "_VisionProviderRegistration":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # pylint: disable=unused-argument
        self.close()


def register_python_torch_extern_conv2d() -> _VisionProviderRegistration:
    """Register the opt-in correctness-only host-staged extern conv2d provider."""
    import tvm  # pylint: disable=import-outside-toplevel

    previous = tvm.get_global_func(VISION_EXTERN_PACKED_FUNC, allow_missing=True)

    def _extern_conv2d(
        input_tensor,
        weight_tensor,
        output_tensor,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
    ):
        import torch  # pylint: disable=import-outside-toplevel
        import torch.nn.functional as torch_functional  # pylint: disable=import-outside-toplevel

        if not torch.cuda.is_available():
            raise RuntimeError("python_torch_host_staged extern_conv2d requires Torch CUDA")
        input_shape = _tensor_shape(input_tensor)
        weight_shape = _tensor_shape(weight_tensor)
        output_shape = _tensor_shape(output_tensor)
        _validate_python_torch_conv_tensor("input", input_tensor, "float32", input_shape)
        _validate_python_torch_conv_tensor("weight", weight_tensor, "float32", weight_shape)
        _validate_python_torch_conv_tensor("output", output_tensor, "float32", output_shape)

        result = torch_functional.conv2d(
            torch.from_numpy(input_tensor.numpy()).to(device="cuda"),
            torch.from_numpy(weight_tensor.numpy()).to(device="cuda"),
            bias=None,
            stride=(int(stride_h), int(stride_w)),
            padding=(int(padding_h), int(padding_w)),
            dilation=(int(dilation_h), int(dilation_w)),
            groups=int(groups),
        )
        if tuple(int(dim) for dim in result.shape) != output_shape:
            raise ValueError(
                f"extern_conv2d output shape must be {output_shape}, got "
                f"{tuple(int(dim) for dim in result.shape)}"
            )
        output_tensor.copyfrom(result.detach().cpu().numpy())

    tvm.register_global_func(VISION_EXTERN_PACKED_FUNC, _extern_conv2d, override=True)
    return _VisionProviderRegistration(VISION_EXTERN_PACKED_FUNC, previous)


def _validate_python_torch_conv_tensor(
    role: str,
    tensor: Any,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    if not hasattr(tensor, "dtype") or not hasattr(tensor, "shape"):
        raise TypeError(f"extern_conv2d {role} must be a TVM tensor")
    if str(tensor.dtype) != dtype:
        raise TypeError(
            f"python_torch_host_staged extern_conv2d supports {dtype} {role}, "
            f"got {tensor.dtype}"
        )
    actual_shape = _tensor_shape(tensor)
    if actual_shape != shape:
        raise ValueError(f"extern_conv2d {role} shape must be {shape}, got {actual_shape}")


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _attribute_chain(node: ast.AST) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _name_from_expr(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _literal_int_tuple(node: ast.AST) -> tuple[int, ...] | None:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    values: list[int] = []
    for item in node.elts:
        value = _literal_int(item)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_int(node.operand)
        return -value if value is not None else None
    return None


def _literal_bool(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    return None


def _dtype_from_expr(node: ast.AST) -> str:
    text = _attribute_chain(node)
    if text.endswith("float32"):
        return "float32"
    if text.endswith("float16"):
        return "float16"
    if text.endswith("bfloat16"):
        return "bfloat16"
    if text.endswith("float64"):
        return "float64"
    return ""


def _tuple_text(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values)


def _bool_literal(value: bool) -> str:
    return "True" if value else "False"


def _sanitize_identifier(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized
