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
"""Attention ABI, artifact, and correctness-provider helpers.

M10 starts with wrapper-level SDPA records.  The default path materializes an
explicit packed-call artifact for the observed ViT full-attention shape, while
opt-in providers cover correctness-only host-staged SDPA and native decomposed
ViT attention.
"""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass, replace
from typing import Any

from .errors import UnsupportedTTIROpError


ATTENTION_ABI_VERSION = 1
ATTENTION_CONTRACT_VIT_FULL = "attention_vit_full_v1"
ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL = "attention_llama_causal_prefill_v1"
ATTENTION_CONTRACT_LLAMA_DECODE = "attention_llama_decode_v1"
ATTENTION_CONTRACT_UNCLASSIFIED = "attention_unclassified"
ATTENTION_RUNTIME_STATUS_DEFERRED = "deferred_attention_runtime"
ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY = "artifact_only"
ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED = "runtime_resolved"
ATTENTION_RUNTIME_STATUS_RUNTIME_FAILED = "runtime_failed"
ATTENTION_SOURCE_KIND_WRAPPER_SDPA = "wrapper_aten_scaled_dot_product_attention"
ATTENTION_ABI_STATUS_CLASSIFIED = "abi_classified_runtime_deferred"
ATTENTION_ABI_STATUS_UNCLASSIFIED = "unsupported_attention_abi"
ATTENTION_SEMANTICS_STATUS_ACCEPTED = "attention_semantics_accepted"
ATTENTION_SEMANTICS_STATUS_UNSUPPORTED = "attention_semantics_unsupported"
ATTENTION_EXTERN_SYMBOL = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
ATTENTION_EXTERN_PACKED_FUNC = "tvm.contrib.triton_tvm.extern_attention_sdpa"
ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA = "extern_attention_sdpa"
ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED = "native_decomposed_attention"
ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY = "artifact_only"
ATTENTION_RUNTIME_KIND_PROVIDER = "runtime_provider"
ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED = "native_decomposed"
ATTENTION_RUNTIME_REPLACEMENT = "not_available"
ATTENTION_RUNTIME_REPLACEMENT_REASON = "attention_sdpa_runtime_replacement_gate_closed"
ATTENTION_RUNTIME_PROVIDER_REASON = "attention_sdpa_python_torch_host_staged_provider_enabled"
ATTENTION_NATIVE_DECOMPOSED_PROVIDER_REASON = "attention_sdpa_native_decomposed_provider_enabled"
ATTENTION_PROVIDER_NONE = "none"
ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED = "python_torch_host_staged"
ATTENTION_PROVIDER_NATIVE_DECOMPOSED = "native_decomposed"
ATTENTION_RUNTIME_PROVIDERS = (
    ATTENTION_PROVIDER_NONE,
    ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
)
ATTENTION_PROVIDER_ABI_VERSION = 1
ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY = "correctness_only"

ATTENTION_TARGET_CONTRACTS = (
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_LLAMA_DECODE,
)

ATTENTION_ABI_REPORT_FIELDS = (
    "attention_source_kind",
    "attention_contract",
    "attention_abi_status",
    "attention_abi_version",
    "attention_phase",
    "attention_causal",
    "attention_mask_kind",
    "attention_rope_policy",
    "attention_kv_cache_policy",
    "attention_sequence_policy",
    "attention_runtime_status",
    "unsupported_attention_reason",
)

ATTENTION_RUNTIME_REPORT_FIELDS = (
    "attention_semantics_status",
    "attention_q_shape",
    "attention_k_shape",
    "attention_v_shape",
    "attention_output_shape",
    "attention_q_stride",
    "attention_k_stride",
    "attention_v_stride",
    "attention_output_stride",
    "attention_mask_param",
    "attention_mask_shape",
    "attention_mask_stride",
    "attention_mask_dtype",
    "attention_q_dtype",
    "attention_k_dtype",
    "attention_v_dtype",
    "attention_output_dtype",
    "attention_scale",
    "attention_implementation_kind",
    "attention_extern_symbol",
    "attention_extern_packed_func",
    "attention_runtime_kind",
    "attention_runtime_replacement",
    "attention_runtime_replacement_available",
    "attention_runtime_replacement_reason",
    "attention_provider_kind",
    "attention_provider_abi_version",
    "attention_runtime_claim",
    "attention_performance_claim",
    "attention_uses_host_staging",
    "attention_runtime_launch_count",
    "attention_artifact_call_count",
    "attention_qkv_bytes",
    "attention_mask_bytes",
    "attention_output_bytes",
    "attention_total_io_bytes",
    "attention_intermediate_buffer_bytes",
    "attention_host_staging_bytes",
    "attention_total_accounted_bytes",
    "unsupported_attention_runtime_reason",
)

ATTENTION_REPORT_FIELDS = ATTENTION_ABI_REPORT_FIELDS + ATTENTION_RUNTIME_REPORT_FIELDS


@dataclass(frozen=True)
class AttentionABIClassification:
    """Structured Pre-M10 classification for one wrapper-level attention call."""

    attention_source_kind: str = ""
    attention_contract: str = ""
    attention_abi_status: str = ""
    attention_abi_version: int = 0
    attention_phase: str = ""
    attention_causal: bool = False
    attention_mask_kind: str = ""
    attention_rope_policy: str = ""
    attention_kv_cache_policy: str = ""
    attention_sequence_policy: str = ""
    attention_runtime_status: str = ""
    unsupported_attention_reason: str = ""

    def as_report_fields(self) -> dict[str, Any]:
        """Return additive report fields for model-corpus extern records."""
        return {
            field_name: getattr(self, field_name)
            for field_name in ATTENTION_ABI_REPORT_FIELDS
        }


@dataclass(frozen=True)
class AttentionSemantics:
    """M10 wrapper SDPA semantic payload for observed attention calls."""

    source_kind: str
    source_name: str
    kernel_name: str
    attention_contract: str
    q_param: str
    k_param: str
    v_param: str
    out_param: str
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    q_stride: tuple[int, ...]
    k_stride: tuple[int, ...]
    v_stride: tuple[int, ...]
    output_stride: tuple[int, ...]
    mask_param: str = ""
    mask_shape: tuple[int, ...] = ()
    mask_stride: tuple[int, ...] = ()
    mask_dtype: str = ""
    q_dtype: str = "float32"
    k_dtype: str = "float32"
    v_dtype: str = "float32"
    output_dtype: str = "float32"
    mask_kind: str = "none_or_padding"
    causal: bool = False
    scale: float = 1.0
    phase: str = "full_attention"
    target_kind: str = "cuda"
    implementation_kind: str = "unresolved"
    unsupported_attention_runtime_reason: str = ""

    @property
    def batch(self) -> int:
        return self.q_shape[0]

    @property
    def heads(self) -> int:
        return self.q_shape[1]

    @property
    def sequence(self) -> int:
        return self.q_shape[2]

    @property
    def head_dim(self) -> int:
        return self.q_shape[3]

    def as_report_fields(self) -> dict[str, Any]:
        """Return report/cache fields derived directly from semantics."""
        return {
            "attention_semantics_status": ATTENTION_SEMANTICS_STATUS_ACCEPTED,
            "attention_q_shape": _tuple_text(self.q_shape),
            "attention_k_shape": _tuple_text(self.k_shape),
            "attention_v_shape": _tuple_text(self.v_shape),
            "attention_output_shape": _tuple_text(self.output_shape),
            "attention_q_stride": _tuple_text(self.q_stride),
            "attention_k_stride": _tuple_text(self.k_stride),
            "attention_v_stride": _tuple_text(self.v_stride),
            "attention_output_stride": _tuple_text(self.output_stride),
            "attention_mask_param": self.mask_param,
            "attention_mask_shape": _tuple_text(self.mask_shape),
            "attention_mask_stride": _tuple_text(self.mask_stride),
            "attention_mask_dtype": self.mask_dtype,
            "attention_q_dtype": self.q_dtype,
            "attention_k_dtype": self.k_dtype,
            "attention_v_dtype": self.v_dtype,
            "attention_output_dtype": self.output_dtype,
            "attention_scale": self.scale,
        }


@dataclass(frozen=True)
class TargetAttentionDecision:
    """Target/provider decision for accepted wrapper SDPA semantics."""

    attention_contract_ok: bool
    implementation_kind: str
    extern_symbol: str = ""
    extern_packed_func: str = ""
    extern_runtime_kind: str = ""
    extern_runtime_replacement: str = ""
    extern_runtime_replacement_available: bool = False
    extern_runtime_replacement_reason: str = ""
    attention_runtime_status: str = ""
    attention_provider_kind: str = ""
    attention_provider_abi_version: int = 0
    attention_runtime_claim: str = ""
    attention_performance_claim: bool = False
    attention_uses_host_staging: bool = False
    attention_runtime_launch_count: int = 0
    attention_artifact_call_count: int = 0
    attention_qkv_bytes: int = 0
    attention_mask_bytes: int = 0
    attention_output_bytes: int = 0
    attention_total_io_bytes: int = 0
    attention_intermediate_buffer_bytes: int = 0
    attention_host_staging_bytes: int = 0
    attention_total_accounted_bytes: int = 0
    unsupported_attention_runtime_reason: str = ""

    def with_accounting(self, semantics: AttentionSemantics) -> "TargetAttentionDecision":
        """Return the same decision with deterministic M10 hardening accounting."""
        return replace(self, **_attention_runtime_accounting_fields(semantics, self))

    def as_report_fields(self) -> dict[str, Any]:
        """Return additive report fields for model-corpus extern records."""
        return {
            "attention_implementation_kind": self.implementation_kind,
            "attention_extern_symbol": self.extern_symbol,
            "attention_extern_packed_func": self.extern_packed_func,
            "attention_runtime_kind": self.extern_runtime_kind,
            "attention_runtime_replacement": self.extern_runtime_replacement,
            "attention_runtime_replacement_available": (
                self.extern_runtime_replacement_available
            ),
            "attention_runtime_replacement_reason": (
                self.extern_runtime_replacement_reason
            ),
            "attention_runtime_status": self.attention_runtime_status,
            "attention_provider_kind": self.attention_provider_kind,
            "attention_provider_abi_version": self.attention_provider_abi_version,
            "attention_runtime_claim": self.attention_runtime_claim,
            "attention_performance_claim": self.attention_performance_claim,
            "attention_uses_host_staging": self.attention_uses_host_staging,
            "attention_runtime_launch_count": self.attention_runtime_launch_count,
            "attention_artifact_call_count": self.attention_artifact_call_count,
            "attention_qkv_bytes": self.attention_qkv_bytes,
            "attention_mask_bytes": self.attention_mask_bytes,
            "attention_output_bytes": self.attention_output_bytes,
            "attention_total_io_bytes": self.attention_total_io_bytes,
            "attention_intermediate_buffer_bytes": (
                self.attention_intermediate_buffer_bytes
            ),
            "attention_host_staging_bytes": self.attention_host_staging_bytes,
            "attention_total_accounted_bytes": self.attention_total_accounted_bytes,
            "unsupported_attention_runtime_reason": (
                self.unsupported_attention_runtime_reason
            ),
        }


@dataclass(frozen=True)
class TargetAttentionPolicy:
    """M10 target policy for wrapper SDPA artifact/runtime selection."""

    target_kind: str = "cuda"
    attention_runtime_provider: str = ATTENTION_PROVIDER_NONE

    def decide(
        self,
        semantics: AttentionSemantics,
        *,
        attention_contract_ok: bool,
    ) -> TargetAttentionDecision:
        """Choose the artifact/runtime state for an accepted SDPA payload."""
        if self.attention_runtime_provider not in ATTENTION_RUNTIME_PROVIDERS:
            return TargetAttentionDecision(
                attention_contract_ok=attention_contract_ok,
                implementation_kind="unsupported",
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
                attention_provider_kind=self.attention_runtime_provider,
                unsupported_attention_runtime_reason="attention_runtime_provider_unknown_m10_hardening",
            ).with_accounting(semantics)
        if not attention_contract_ok:
            return TargetAttentionDecision(
                attention_contract_ok=False,
                implementation_kind="unsupported",
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
                unsupported_attention_runtime_reason="attention_contract_failed",
            )
        unsupported_reason = self._unsupported_reason(semantics)
        if unsupported_reason:
            return TargetAttentionDecision(
                attention_contract_ok=True,
                implementation_kind="unsupported",
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
                attention_provider_kind=ATTENTION_PROVIDER_NONE,
                attention_provider_abi_version=0,
                attention_performance_claim=False,
                attention_uses_host_staging=False,
                unsupported_attention_runtime_reason=unsupported_reason,
            ).with_accounting(semantics)
        if semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_DECODE:
            return TargetAttentionDecision(
                attention_contract_ok=True,
                implementation_kind="unsupported",
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
                unsupported_attention_runtime_reason=(
                    "attention_llama_decode_synthetic_only_m10_6"
                ),
            ).with_accounting(semantics)

        if (
            semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
            and self.attention_runtime_provider != ATTENTION_PROVIDER_NATIVE_DECOMPOSED
        ):
            return TargetAttentionDecision(
                attention_contract_ok=True,
                implementation_kind="unsupported",
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
                attention_provider_kind=ATTENTION_PROVIDER_NONE,
                attention_provider_abi_version=0,
                attention_performance_claim=False,
                attention_uses_host_staging=False,
                unsupported_attention_runtime_reason=(
                    "attention_llama_prefill_native_decomposed_provider_required_m10_5"
                ),
            ).with_accounting(semantics)

        if (
            self.attention_runtime_provider == ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED
            and semantics.attention_contract == ATTENTION_CONTRACT_VIT_FULL
        ):
            return TargetAttentionDecision(
                attention_contract_ok=True,
                implementation_kind=ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA,
                extern_symbol=ATTENTION_EXTERN_SYMBOL,
                extern_packed_func=ATTENTION_EXTERN_PACKED_FUNC,
                extern_runtime_kind=ATTENTION_RUNTIME_KIND_PROVIDER,
                extern_runtime_replacement=ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                extern_runtime_replacement_available=True,
                extern_runtime_replacement_reason=ATTENTION_RUNTIME_PROVIDER_REASON,
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
                attention_provider_kind=ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                attention_provider_abi_version=ATTENTION_PROVIDER_ABI_VERSION,
                attention_runtime_claim=ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                attention_performance_claim=False,
                attention_uses_host_staging=True,
            ).with_accounting(semantics)
        if self.attention_runtime_provider == ATTENTION_PROVIDER_NATIVE_DECOMPOSED:
            return TargetAttentionDecision(
                attention_contract_ok=True,
                implementation_kind=ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
                extern_runtime_kind=ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
                extern_runtime_replacement=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
                extern_runtime_replacement_available=True,
                extern_runtime_replacement_reason=ATTENTION_NATIVE_DECOMPOSED_PROVIDER_REASON,
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
                attention_provider_kind=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
                attention_provider_abi_version=ATTENTION_PROVIDER_ABI_VERSION,
                attention_runtime_claim=ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                attention_performance_claim=False,
                attention_uses_host_staging=False,
            ).with_accounting(semantics)
        if semantics.attention_contract == ATTENTION_CONTRACT_VIT_FULL:
            return TargetAttentionDecision(
                attention_contract_ok=True,
                implementation_kind=ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA,
                extern_symbol=ATTENTION_EXTERN_SYMBOL,
                extern_packed_func=ATTENTION_EXTERN_PACKED_FUNC,
                extern_runtime_kind=ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY,
                extern_runtime_replacement=ATTENTION_RUNTIME_REPLACEMENT,
                extern_runtime_replacement_available=False,
                extern_runtime_replacement_reason=ATTENTION_RUNTIME_REPLACEMENT_REASON,
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY,
                attention_provider_kind=ATTENTION_PROVIDER_NONE,
                attention_provider_abi_version=0,
                attention_runtime_claim="",
                attention_performance_claim=False,
                attention_uses_host_staging=False,
            ).with_accounting(semantics)
        return TargetAttentionDecision(
            attention_contract_ok=True,
            implementation_kind="unsupported",
            attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
            attention_provider_kind=ATTENTION_PROVIDER_NONE,
            attention_provider_abi_version=0,
            attention_performance_claim=False,
            attention_uses_host_staging=False,
            unsupported_attention_runtime_reason="attention_contract_runtime_not_enabled_m10_6",
        ).with_accounting(semantics)

    def _unsupported_reason(self, semantics: AttentionSemantics) -> str:
        if self.target_kind != "cuda" or semantics.target_kind != "cuda":
            return "attention_target_kind_not_supported"
        if semantics.q_dtype != "float32" or semantics.output_dtype != "float32":
            return "attention_dtype_not_supported_m10_5"
        if semantics.k_dtype != "float32" or semantics.v_dtype != "float32":
            return "attention_dtype_not_supported_m10_5"
        if semantics.attention_contract == ATTENTION_CONTRACT_VIT_FULL:
            if semantics.causal:
                return "attention_causal_not_supported_m10_3"
            if semantics.mask_kind != "none_or_padding":
                return "attention_mask_not_supported_m10_3"
            return ""
        if semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
            if not semantics.causal:
                return "attention_llama_prefill_requires_causal_contract_m10_5"
            if semantics.mask_kind != "causal":
                return "attention_llama_prefill_requires_causal_mask_m10_5"
            expected_mask_shape = (
                semantics.batch,
                semantics.heads,
                semantics.sequence,
                semantics.sequence,
            )
            if semantics.mask_shape != expected_mask_shape:
                return "attention_llama_prefill_mask_shape_not_supported_m10_5"
            if semantics.mask_dtype != "float32":
                return "attention_llama_prefill_mask_dtype_not_supported_m10_5"
            return ""
        if semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_DECODE:
            return "attention_llama_decode_synthetic_only_m10_6"
        return "attention_contract_runtime_not_enabled_m10_6"


def _attention_runtime_accounting_fields(
    semantics: AttentionSemantics,
    decision: TargetAttentionDecision,
) -> dict[str, int]:
    """Compute deterministic launch and byte accounting for M10 hardening.

    These counters are report metadata only.  They do not promote the corpus
    provider records into performance claims.
    """
    q_bytes = _numel(semantics.q_shape) * _dtype_nbytes(semantics.q_dtype)
    k_bytes = _numel(semantics.k_shape) * _dtype_nbytes(semantics.k_dtype)
    v_bytes = _numel(semantics.v_shape) * _dtype_nbytes(semantics.v_dtype)
    output_bytes = _numel(semantics.output_shape) * _dtype_nbytes(semantics.output_dtype)
    mask_bytes = (
        _numel(semantics.mask_shape) * _dtype_nbytes(semantics.mask_dtype)
        if semantics.mask_shape and semantics.mask_dtype
        else 0
    )
    qkv_bytes = q_bytes + k_bytes + v_bytes
    total_io_bytes = qkv_bytes + mask_bytes + output_bytes
    intermediate_bytes = 0
    if decision.implementation_kind == ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED:
        output_elems = _numel(semantics.output_shape)
        fp32_bytes = _dtype_nbytes("float32")
        intermediate_bytes = (
            2 * output_elems * semantics.sequence * fp32_bytes
            + 2 * output_elems * fp32_bytes
        )
    runtime_launch_count = (
        1
        if decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
        else 0
    )
    artifact_call_count = 1 if decision.implementation_kind in {
        ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA,
        ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    } else 0
    host_staging_bytes = total_io_bytes if decision.attention_uses_host_staging else 0
    return {
        "attention_runtime_launch_count": runtime_launch_count,
        "attention_artifact_call_count": artifact_call_count,
        "attention_qkv_bytes": qkv_bytes,
        "attention_mask_bytes": mask_bytes,
        "attention_output_bytes": output_bytes,
        "attention_total_io_bytes": total_io_bytes,
        "attention_intermediate_buffer_bytes": intermediate_bytes,
        "attention_host_staging_bytes": host_staging_bytes,
        "attention_total_accounted_bytes": total_io_bytes + intermediate_bytes,
    }


def classify_wrapper_sdpa_attention(
    *,
    op_name: str,
    op_family: str,
    source: str = "",
    model_family: str = "",
    model_case: str = "",
    case_name: str = "",
) -> AttentionABIClassification:
    """Classify an Inductor wrapper SDPA call for the Pre-M10 ABI gate."""
    if op_family != "deferred_attention":
        return AttentionABIClassification()
    if op_name != ATTENTION_EXTERN_SYMBOL:
        return _unclassified("unsupported_attention_op")

    text = " ".join([source, model_family, model_case, case_name]).lower()
    family = model_family.lower()
    if not family:
        if "vit" in text:
            family = "vit"
        elif "llama" in text:
            family = "llama"

    if family == "vit":
        return AttentionABIClassification(
            attention_source_kind=ATTENTION_SOURCE_KIND_WRAPPER_SDPA,
            attention_contract=ATTENTION_CONTRACT_VIT_FULL,
            attention_abi_status=ATTENTION_ABI_STATUS_CLASSIFIED,
            attention_abi_version=ATTENTION_ABI_VERSION,
            attention_phase="full_attention",
            attention_causal=False,
            attention_mask_kind="none_or_padding",
            attention_rope_policy="not_applicable",
            attention_kv_cache_policy="not_applicable",
            attention_sequence_policy="static_full_sequence",
            attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
        )

    if family == "llama":
        if _looks_like_decode(text):
            return AttentionABIClassification(
                attention_source_kind=ATTENTION_SOURCE_KIND_WRAPPER_SDPA,
                attention_contract=ATTENTION_CONTRACT_LLAMA_DECODE,
                attention_abi_status=ATTENTION_ABI_STATUS_CLASSIFIED,
                attention_abi_version=ATTENTION_ABI_VERSION,
                attention_phase="decode",
                attention_causal=True,
                attention_mask_kind="causal_single_token",
                attention_rope_policy="rope_pointwise_deferred",
                attention_kv_cache_policy="kv_cache_layout_deferred",
                attention_sequence_policy="single_token_decode",
                attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
            )
        return AttentionABIClassification(
            attention_source_kind=ATTENTION_SOURCE_KIND_WRAPPER_SDPA,
            attention_contract=ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
            attention_abi_status=ATTENTION_ABI_STATUS_CLASSIFIED,
            attention_abi_version=ATTENTION_ABI_VERSION,
            attention_phase="causal_prefill",
            attention_causal=True,
            attention_mask_kind="causal",
            attention_rope_policy="rope_pointwise_deferred",
            attention_kv_cache_policy="prefill_no_cache_update",
            attention_sequence_policy="static_or_runtime_prefill_sequence",
            attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
        )

    return _unclassified("unsupported_attention_model_family")


def wrapper_sdpa_attention_report_fields(
    *,
    op_name: str,
    op_family: str,
    source: str = "",
    model_family: str = "",
    model_case: str = "",
    case_name: str = "",
    attention_runtime_provider: str = ATTENTION_PROVIDER_NONE,
) -> dict[str, Any]:
    """Return ABI plus M10 runtime fields for one wrapper SDPA call."""
    if op_family != "deferred_attention":
        return _default_report_fields()

    classification = classify_wrapper_sdpa_attention(
        op_name=op_name,
        op_family=op_family,
        source=source,
        model_family=model_family,
        model_case=model_case,
        case_name=case_name,
    )
    fields = {
        **_default_report_fields(),
        **classification.as_report_fields(),
    }
    if classification.attention_contract not in (
        ATTENTION_CONTRACT_VIT_FULL,
        ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ):
        reason = "attention_contract_runtime_not_enabled_m10_6"
        if classification.attention_contract == ATTENTION_CONTRACT_LLAMA_DECODE:
            reason = "attention_llama_decode_synthetic_only_m10_6"
        fields.update(
            {
                "attention_semantics_status": ATTENTION_SEMANTICS_STATUS_UNSUPPORTED,
                "unsupported_attention_runtime_reason": reason,
            }
        )
        return fields

    try:
        semantics = extract_attention_semantics_from_wrapper_sdpa(
            {
                "op_name": op_name,
                "source": source,
                "case_name": case_name or model_case,
                "kernel_name": case_name or model_case or "wrapper_attention_sdpa",
            },
            model_family=model_family,
            model_case=model_case,
        )
        decision = TargetAttentionPolicy(
            attention_runtime_provider=attention_runtime_provider
        ).decide(
            semantics,
            attention_contract_ok=True,
        )
    except UnsupportedTTIROpError as err:
        fields.update(
            {
                "attention_semantics_status": ATTENTION_SEMANTICS_STATUS_UNSUPPORTED,
                "unsupported_attention_runtime_reason": str(err),
            }
        )
        return fields

    fields.update(semantics.as_report_fields())
    fields.update(decision.as_report_fields())
    return fields


def extract_attention_semantics_from_wrapper_sdpa(
    source_or_call: Any,
    *,
    case_name: str = "",
    kernel_name: str = "",
    model_family: str = "",
    model_case: str = "",
) -> AttentionSemantics:
    """Extract M10 wrapper SDPA semantics from supported source forms."""
    source, op_name, call_case, call_kernel = _wrapper_call_source_fields(source_or_call)
    case_name = case_name or call_case
    kernel_name = kernel_name or call_kernel or case_name or "wrapper_attention_sdpa"
    if op_name and op_name != ATTENTION_EXTERN_SYMBOL:
        raise UnsupportedTTIROpError(f"Unsupported wrapper attention op {op_name!r}")
    if not source:
        raise UnsupportedTTIROpError("wrapper_attention_sdpa requires a source line")

    classification = classify_wrapper_sdpa_attention(
        op_name=op_name or ATTENTION_EXTERN_SYMBOL,
        op_family="deferred_attention",
        source=source,
        model_family=model_family,
        model_case=model_case,
        case_name=case_name,
    )
    call = _first_call(source, ATTENTION_EXTERN_SYMBOL)
    if call is None:
        raise UnsupportedTTIROpError("wrapper_attention_sdpa requires ATen SDPA")
    if len(call.args) < 5:
        raise UnsupportedTTIROpError("wrapper_attention_sdpa requires q, k, v, mask, causal")

    if classification.attention_contract == ATTENTION_CONTRACT_VIT_FULL:
        return _extract_vit_full_attention_semantics(call, kernel_name)
    if classification.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        return _extract_llama_prefill_attention_semantics(call, kernel_name)
    if classification.attention_contract == ATTENTION_CONTRACT_LLAMA_DECODE:
        raise UnsupportedTTIROpError("attention_llama_decode_v1 is synthetic-only in M10.6")
    raise UnsupportedTTIROpError("unsupported wrapper attention contract")


def _extract_vit_full_attention_semantics(
    call: ast.Call,
    kernel_name: str,
) -> AttentionSemantics:
    """Extract observed ViT full-attention wrapper SDPA semantics."""

    q_tensor = _parse_reinterpret_tensor(call.args[0], "q")
    k_tensor = _parse_reinterpret_tensor(call.args[1], "k")
    v_tensor = _parse_reinterpret_tensor(call.args[2], "v")
    mask_kind = _parse_mask_kind(call.args[3])
    causal = _parse_bool_arg(call.args[4], "causal")
    scale = _parse_optional_float_keyword(call, "scale")
    if scale is None:
        raise UnsupportedTTIROpError("wrapper_attention_sdpa requires explicit numeric scale")

    q_shape = q_tensor["shape"]
    k_shape = k_tensor["shape"]
    v_shape = v_tensor["shape"]
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise UnsupportedTTIROpError("attention_vit_full_v1 requires rank-4 Q/K/V")
    if q_shape != k_shape or q_shape != v_shape:
        raise UnsupportedTTIROpError("attention_vit_full_v1 requires matching Q/K/V shapes")
    if q_shape[-1] <= 0 or q_shape[2] <= 0:
        raise UnsupportedTTIROpError("attention_vit_full_v1 requires positive sequence/head dims")
    if mask_kind != "none_or_padding":
        raise UnsupportedTTIROpError("attention_vit_full_v1 only supports no mask in M10.3")
    if causal:
        raise UnsupportedTTIROpError("attention_vit_full_v1 requires causal=False")

    output_stride = _contiguous_stride(q_shape)
    return AttentionSemantics(
        source_kind=ATTENTION_SOURCE_KIND_WRAPPER_SDPA,
        source_name=ATTENTION_EXTERN_SYMBOL,
        kernel_name=kernel_name,
        attention_contract=ATTENTION_CONTRACT_VIT_FULL,
        q_param=str(q_tensor["base"]),
        k_param=str(k_tensor["base"]),
        v_param=str(v_tensor["base"]),
        out_param="out",
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        output_shape=q_shape,
        q_stride=q_tensor["stride"],
        k_stride=k_tensor["stride"],
        v_stride=v_tensor["stride"],
        output_stride=output_stride,
        mask_kind=mask_kind,
        causal=causal,
        scale=float(scale),
    )


def _extract_llama_prefill_attention_semantics(
    call: ast.Call,
    kernel_name: str,
) -> AttentionSemantics:
    """Extract observed Llama causal-prefill wrapper SDPA semantics."""
    v_tensor = _parse_reinterpret_tensor(call.args[2], "v")
    mask_tensor = _parse_reinterpret_tensor(call.args[3], "mask")
    sdpa_is_causal = _parse_bool_arg(call.args[4], "causal")
    scale = _parse_optional_float_keyword(call, "scale")
    if scale is None:
        raise UnsupportedTTIROpError("attention_llama_causal_prefill_v1 requires scale")
    if sdpa_is_causal:
        raise UnsupportedTTIROpError(
            "attention_llama_causal_prefill_v1 requires additive mask with is_causal=False"
        )

    v_shape = v_tensor["shape"]
    v_stride = v_tensor["stride"]
    if len(v_shape) != 4:
        raise UnsupportedTTIROpError("attention_llama_causal_prefill_v1 requires rank-4 V")
    if v_shape[-1] <= 0 or v_shape[2] <= 0:
        raise UnsupportedTTIROpError(
            "attention_llama_causal_prefill_v1 requires positive sequence/head dims"
        )

    q_tensor = _parse_optional_reinterpret_tensor(call.args[0], "q")
    k_tensor = _parse_optional_reinterpret_tensor(call.args[1], "k")
    q_shape = q_tensor["shape"] if q_tensor else v_shape
    k_shape = k_tensor["shape"] if k_tensor else v_shape
    q_stride = q_tensor["stride"] if q_tensor else v_stride
    k_stride = k_tensor["stride"] if k_tensor else v_stride
    if q_shape != v_shape or k_shape != v_shape:
        raise UnsupportedTTIROpError(
            "attention_llama_causal_prefill_v1 requires matching Q/K/V shapes"
        )
    expected_mask_shape = (v_shape[0], v_shape[1], v_shape[2], v_shape[2])
    if mask_tensor["shape"] != expected_mask_shape:
        raise UnsupportedTTIROpError(
            "attention_llama_causal_prefill_v1 requires [B,H,S,S] causal mask"
        )

    return AttentionSemantics(
        source_kind=ATTENTION_SOURCE_KIND_WRAPPER_SDPA,
        source_name=ATTENTION_EXTERN_SYMBOL,
        kernel_name=kernel_name,
        attention_contract=ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
        q_param=str(q_tensor["base"]) if q_tensor else _node_text(call.args[0]),
        k_param=str(k_tensor["base"]) if k_tensor else _node_text(call.args[1]),
        v_param=str(v_tensor["base"]),
        out_param="out",
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        output_shape=v_shape,
        q_stride=q_stride,
        k_stride=k_stride,
        v_stride=v_stride,
        output_stride=q_stride,
        mask_param=str(mask_tensor["base"]),
        mask_shape=mask_tensor["shape"],
        mask_stride=mask_tensor["stride"],
        mask_dtype="float32",
        mask_kind="causal",
        causal=True,
        scale=float(scale),
        phase="causal_prefill",
    )


def build_attention_sdpa_tirx_source(
    semantics: AttentionSemantics,
    decision: TargetAttentionDecision,
    *,
    target_attrs: str = "cuda",
) -> str:
    """Build the explicit M10.2 packed-call artifact for wrapper SDPA."""
    if semantics.attention_contract != ATTENTION_CONTRACT_VIT_FULL:
        raise UnsupportedTTIROpError("attention artifact requires attention_vit_full_v1")
    if (
        decision.implementation_kind != ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA
        or decision.extern_symbol != ATTENTION_EXTERN_SYMBOL
    ):
        raise UnsupportedTTIROpError("attention artifact requires extern SDPA policy decision")

    func_name = _sanitize_identifier(semantics.kernel_name or "wrapper_attention_sdpa")
    q_name = _sanitize_identifier(semantics.q_param or "q")
    k_name = _sanitize_identifier(semantics.k_param or "k")
    v_name = _sanitize_identifier(semantics.v_param or "v")
    out_name = _sanitize_identifier(semantics.out_param or "out")
    shape = _tuple_text(semantics.q_shape)
    causal_expr = "T.bool(True)" if semantics.causal else "T.bool(False)"
    scale_expr = f"T.float32({semantics.scale:.17g})"
    lines: list[str] = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}({q_name}_handle: T.handle, {k_name}_handle: T.handle, "
        f"{v_name}_handle: T.handle, {out_name}_handle: T.handle):",
        "        T.func_attr({"
        f'"global_symbol": "{semantics.kernel_name}", '
        '"tirx.noalias": True, '
        f'"target": T.target({target_attrs!r}), '
        f'"triton_tvm.contract": "{ATTENTION_CONTRACT_VIT_FULL}", '
        f'"triton_tvm.attention_source_kind": "{semantics.source_kind}", '
        f'"triton_tvm.attention_contract": "{semantics.attention_contract}", '
        f'"triton_tvm.attention_phase": "{semantics.phase}", '
        f'"triton_tvm.attention_mask_kind": "{semantics.mask_kind}", '
        f'"triton_tvm.attention_causal": {_bool_literal(semantics.causal)}, '
        f'"triton_tvm.attention_scale": "{semantics.scale:.17g}", '
        f'"triton_tvm.q_dtype": "{semantics.q_dtype}", '
        f'"triton_tvm.k_dtype": "{semantics.k_dtype}", '
        f'"triton_tvm.v_dtype": "{semantics.v_dtype}", '
        f'"triton_tvm.output_dtype": "{semantics.output_dtype}", '
        f'"triton_tvm.q_shape": "{_tuple_text(semantics.q_shape)}", '
        f'"triton_tvm.k_shape": "{_tuple_text(semantics.k_shape)}", '
        f'"triton_tvm.v_shape": "{_tuple_text(semantics.v_shape)}", '
        f'"triton_tvm.output_shape": "{_tuple_text(semantics.output_shape)}", '
        f'"triton_tvm.q_stride": "{_tuple_text(semantics.q_stride)}", '
        f'"triton_tvm.k_stride": "{_tuple_text(semantics.k_stride)}", '
        f'"triton_tvm.v_stride": "{_tuple_text(semantics.v_stride)}", '
        f'"triton_tvm.output_stride": "{_tuple_text(semantics.output_stride)}", '
        f'"triton_tvm.implementation_kind": "{decision.implementation_kind}", '
        f'"triton_tvm.extern_symbol": "{decision.extern_symbol}", '
        f'"triton_tvm.extern_packed_func": "{decision.extern_packed_func}", '
        f'"triton_tvm.extern_runtime_kind": "{decision.extern_runtime_kind}", '
        f'"triton_tvm.extern_runtime_replacement": "{decision.extern_runtime_replacement}", '
        f'"triton_tvm.extern_runtime_replacement_available": '
        f'{_bool_literal(decision.extern_runtime_replacement_available)}, '
        f'"triton_tvm.extern_runtime_replacement_reason": '
        f'"{decision.extern_runtime_replacement_reason}", '
        f'"triton_tvm.attention_runtime_status": "{decision.attention_runtime_status}", '
        f'"triton_tvm.attention_provider_kind": "{decision.attention_provider_kind}", '
        f'"triton_tvm.attention_provider_abi_version": '
        f'{decision.attention_provider_abi_version}, '
        f'"triton_tvm.attention_runtime_claim": "{decision.attention_runtime_claim}", '
        f'"triton_tvm.attention_performance_claim": '
        f'{_bool_literal(decision.attention_performance_claim)}, '
        f'"triton_tvm.attention_uses_host_staging": '
        f'{_bool_literal(decision.attention_uses_host_staging)}, '
        f'"triton_tvm.attention_runtime_launch_count": '
        f'{decision.attention_runtime_launch_count}, '
        f'"triton_tvm.attention_artifact_call_count": '
        f'{decision.attention_artifact_call_count}, '
        f'"triton_tvm.attention_qkv_bytes": {decision.attention_qkv_bytes}, '
        f'"triton_tvm.attention_mask_bytes": {decision.attention_mask_bytes}, '
        f'"triton_tvm.attention_output_bytes": {decision.attention_output_bytes}, '
        f'"triton_tvm.attention_total_io_bytes": {decision.attention_total_io_bytes}, '
        f'"triton_tvm.attention_intermediate_buffer_bytes": '
        f'{decision.attention_intermediate_buffer_bytes}, '
        f'"triton_tvm.attention_host_staging_bytes": '
        f'{decision.attention_host_staging_bytes}, '
        f'"triton_tvm.attention_total_accounted_bytes": '
        f'{decision.attention_total_accounted_bytes}'
        "})",
        "        "
        f'{q_name} = T.match_buffer({q_name}_handle, ({shape}), "{semantics.q_dtype}")',
        "        "
        f'{k_name} = T.match_buffer({k_name}_handle, ({shape}), "{semantics.k_dtype}")',
        "        "
        f'{v_name} = T.match_buffer({v_name}_handle, ({shape}), "{semantics.v_dtype}")',
        "        "
        f'{out_name} = T.match_buffer({out_name}_handle, ({shape}), "{semantics.output_dtype}")',
        "        "
        f'T.evaluate(T.call_packed("{ATTENTION_EXTERN_PACKED_FUNC}", '
        f"{q_name}, {k_name}, {v_name}, {out_name}, "
        f"{semantics.batch}, {semantics.heads}, {semantics.sequence}, "
        f"{semantics.head_dim}, {scale_expr}, {causal_expr}))",
    ]
    return "\n".join(lines) + "\n"


def build_native_decomposed_attention_tirx_source(
    semantics: AttentionSemantics,
    decision: TargetAttentionDecision,
    *,
    target_attrs: str = "cuda",
) -> str:
    """Build the M10 native decomposed attention artifact.

    The schedule is intentionally correctness-first: one CUDA block computes one
    output element by recomputing the row-local QK scores, row softmax, and AV
    accumulation in local buffers.
    """
    if semantics.attention_contract not in (
        ATTENTION_CONTRACT_VIT_FULL,
        ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ):
        raise UnsupportedTTIROpError("native attention requires an M10 native contract")
    if decision.implementation_kind != ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED:
        raise UnsupportedTTIROpError(
            "native attention requires native_decomposed policy decision"
        )
    if (
        semantics.attention_contract == ATTENTION_CONTRACT_VIT_FULL
        and (semantics.causal or semantics.mask_kind != "none_or_padding")
    ):
        raise UnsupportedTTIROpError("native attention requires unmasked non-causal ViT")
    if semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        expected_mask_shape = (
            semantics.batch,
            semantics.heads,
            semantics.sequence,
            semantics.sequence,
        )
        if not semantics.causal or semantics.mask_kind != "causal":
            raise UnsupportedTTIROpError("native Llama prefill requires causal mask semantics")
        if semantics.mask_shape != expected_mask_shape or not semantics.mask_param:
            raise UnsupportedTTIROpError("native Llama prefill requires [B,H,S,S] mask")
    if (
        semantics.q_dtype != "float32"
        or semantics.k_dtype != "float32"
        or semantics.v_dtype != "float32"
        or semantics.output_dtype != "float32"
        or (
            semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
            and semantics.mask_dtype != "float32"
        )
    ):
        raise UnsupportedTTIROpError("native attention M10 supports fp32 only")

    func_name = _sanitize_identifier(semantics.kernel_name or "wrapper_attention_sdpa")
    q_name = _sanitize_identifier(semantics.q_param or "q")
    k_name = _sanitize_identifier(semantics.k_param or "k")
    v_name = _sanitize_identifier(semantics.v_param or "v")
    mask_name = _sanitize_identifier(semantics.mask_param or "mask")
    out_name = _sanitize_identifier(semantics.out_param or "out")
    batch, heads, sequence, head_dim = semantics.q_shape
    output_elems = batch * heads * sequence * head_dim
    q_shape = _shape_literal(semantics.q_shape)
    k_shape = _shape_literal(semantics.k_shape)
    v_shape = _shape_literal(semantics.v_shape)
    output_shape = _shape_literal(semantics.output_shape)
    mask_shape = _shape_literal(semantics.mask_shape) if semantics.mask_shape else ""
    q_stride = _shape_literal(semantics.q_stride)
    k_stride = _shape_literal(semantics.k_stride)
    v_stride = _shape_literal(semantics.v_stride)
    output_stride = _shape_literal(semantics.output_stride)
    mask_stride = _shape_literal(semantics.mask_stride) if semantics.mask_stride else ""
    scale_expr = f"T.float32({semantics.scale:.17g})"
    has_mask = semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
    softmax_boundary = "masked_softmax_row" if has_mask else "softmax_row"
    mask_attr_lines = []
    mask_buffer_lines = []
    if has_mask:
        mask_attr_lines = [
            f'"triton_tvm.mask_param": "{semantics.mask_param}", ',
            f'"triton_tvm.mask_shape": "{_tuple_text(semantics.mask_shape)}", ',
            f'"triton_tvm.mask_stride": "{_tuple_text(semantics.mask_stride)}", ',
            f'"triton_tvm.mask_dtype": "{semantics.mask_dtype}", ',
        ]
        mask_buffer_lines = [
            "        "
            f'{mask_name} = T.match_buffer({mask_name}_handle, {mask_shape}, '
            f'"{semantics.mask_dtype}", strides={mask_stride})',
        ]
    mask_attr_text = "".join(mask_attr_lines)
    mask_arg = f", {mask_name}_handle: T.handle" if has_mask else ""
    mask_add_lines = []
    if has_mask:
        mask_add_lines = [
            f"                for j in T.serial(0, {sequence}):",
            '                    with T.sblock("attention_mask_add"):',
            f"                        bo = T.axis.spatial({output_elems}, block)",
            f"                        vb = T.axis.spatial({batch}, b)",
            f"                        vh = T.axis.spatial({heads}, h)",
            f"                        vi = T.axis.spatial({sequence}, i)",
            f"                        vj = T.axis.spatial({sequence}, j)",
            f"                        T.reads(scores[bo, vj], {mask_name}[vb, vh, vi, vj])",
            "                        T.writes(scores[bo, vj])",
            f"                        scores[bo, vj] = scores[bo, vj] + "
            f"{mask_name}[vb, vh, vi, vj]",
        ]
    lines: list[str] = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}({q_name}_handle: T.handle, {k_name}_handle: T.handle, "
        f"{v_name}_handle: T.handle{mask_arg}, {out_name}_handle: T.handle):",
        "        T.func_attr({"
        f'"global_symbol": "{semantics.kernel_name}", '
        '"tirx.noalias": True, '
        f'"target": T.target({target_attrs!r}), '
        f'"triton_tvm.contract": "{semantics.attention_contract}", '
        f'"triton_tvm.attention_source_kind": "{semantics.source_kind}", '
        f'"triton_tvm.attention_contract": "{semantics.attention_contract}", '
        f'"triton_tvm.attention_phase": "{semantics.phase}", '
        f'"triton_tvm.attention_mask_kind": "{semantics.mask_kind}", '
        f'"triton_tvm.attention_causal": {_bool_literal(semantics.causal)}, '
        f'"triton_tvm.attention_scale": "{semantics.scale:.17g}", '
        f'"triton_tvm.q_dtype": "{semantics.q_dtype}", '
        f'"triton_tvm.k_dtype": "{semantics.k_dtype}", '
        f'"triton_tvm.v_dtype": "{semantics.v_dtype}", '
        f'"triton_tvm.output_dtype": "{semantics.output_dtype}", '
        f'"triton_tvm.q_shape": "{_tuple_text(semantics.q_shape)}", '
        f'"triton_tvm.k_shape": "{_tuple_text(semantics.k_shape)}", '
        f'"triton_tvm.v_shape": "{_tuple_text(semantics.v_shape)}", '
        f'"triton_tvm.output_shape": "{_tuple_text(semantics.output_shape)}", '
        f'"triton_tvm.q_stride": "{_tuple_text(semantics.q_stride)}", '
        f'"triton_tvm.k_stride": "{_tuple_text(semantics.k_stride)}", '
        f'"triton_tvm.v_stride": "{_tuple_text(semantics.v_stride)}", '
        f'"triton_tvm.output_stride": "{_tuple_text(semantics.output_stride)}", '
        f"{mask_attr_text}"
        f'"triton_tvm.implementation_kind": "{decision.implementation_kind}", '
        f'"triton_tvm.attention_decomposition": "qk_softmax_av", '
        f'"triton_tvm.qk_matmul_boundary": "matmul_minimal", '
        f'"triton_tvm.softmax_boundary": "{softmax_boundary}", '
        f'"triton_tvm.av_matmul_boundary": "matmul_minimal", '
        f'"triton_tvm.extern_symbol": "{decision.extern_symbol}", '
        f'"triton_tvm.extern_packed_func": "{decision.extern_packed_func}", '
        f'"triton_tvm.extern_runtime_kind": "{decision.extern_runtime_kind}", '
        f'"triton_tvm.extern_runtime_replacement": "{decision.extern_runtime_replacement}", '
        f'"triton_tvm.extern_runtime_replacement_available": '
        f'{_bool_literal(decision.extern_runtime_replacement_available)}, '
        f'"triton_tvm.extern_runtime_replacement_reason": '
        f'"{decision.extern_runtime_replacement_reason}", '
        f'"triton_tvm.attention_runtime_status": "{decision.attention_runtime_status}", '
        f'"triton_tvm.attention_provider_kind": "{decision.attention_provider_kind}", '
        f'"triton_tvm.attention_provider_abi_version": '
        f'{decision.attention_provider_abi_version}, '
        f'"triton_tvm.attention_runtime_claim": "{decision.attention_runtime_claim}", '
        f'"triton_tvm.attention_performance_claim": '
        f'{_bool_literal(decision.attention_performance_claim)}, '
        f'"triton_tvm.attention_uses_host_staging": '
        f'{_bool_literal(decision.attention_uses_host_staging)}, '
        f'"triton_tvm.attention_runtime_launch_count": '
        f'{decision.attention_runtime_launch_count}, '
        f'"triton_tvm.attention_artifact_call_count": '
        f'{decision.attention_artifact_call_count}, '
        f'"triton_tvm.attention_qkv_bytes": {decision.attention_qkv_bytes}, '
        f'"triton_tvm.attention_mask_bytes": {decision.attention_mask_bytes}, '
        f'"triton_tvm.attention_output_bytes": {decision.attention_output_bytes}, '
        f'"triton_tvm.attention_total_io_bytes": {decision.attention_total_io_bytes}, '
        f'"triton_tvm.attention_intermediate_buffer_bytes": '
        f'{decision.attention_intermediate_buffer_bytes}, '
        f'"triton_tvm.attention_host_staging_bytes": '
        f'{decision.attention_host_staging_bytes}, '
        f'"triton_tvm.attention_total_accounted_bytes": '
        f'{decision.attention_total_accounted_bytes}'
        "})",
        "        "
        f'{q_name} = T.match_buffer({q_name}_handle, {q_shape}, '
        f'"{semantics.q_dtype}", strides={q_stride})',
        "        "
        f'{k_name} = T.match_buffer({k_name}_handle, {k_shape}, '
        f'"{semantics.k_dtype}", strides={k_stride})',
        "        "
        f'{v_name} = T.match_buffer({v_name}_handle, {v_shape}, '
        f'"{semantics.v_dtype}", strides={v_stride})',
        *mask_buffer_lines,
        "        "
        f'{out_name} = T.match_buffer({out_name}_handle, {output_shape}, '
        f'"{semantics.output_dtype}", strides={output_stride})',
        f'        scores = T.alloc_buffer(({output_elems}, {sequence}), "float32", scope="local")',
        f'        probs = T.alloc_buffer(({output_elems}, {sequence}), "float32", scope="local")',
        f'        maxv = T.alloc_buffer(({output_elems},), "float32", scope="local")',
        f'        den = T.alloc_buffer(({output_elems},), "float32", scope="local")',
        "        "
        f'for block in T.thread_binding(0, {output_elems}, thread="blockIdx.x"):',
        '            for tx in T.thread_binding(0, 1, thread="threadIdx.x"):',
        f"                b = block // {heads * sequence * head_dim}",
        f"                rem0 = block % {heads * sequence * head_dim}",
        f"                h = rem0 // {sequence * head_dim}",
        f"                rem1 = rem0 % {sequence * head_dim}",
        f"                i = rem1 // {head_dim}",
        f"                d = rem1 % {head_dim}",
        f"                for j in T.serial(0, {sequence}):",
        f"                    for kd in T.serial(0, {head_dim}):",
        '                        with T.sblock("attention_qk_matmul"):',
        f"                            bo = T.axis.spatial({output_elems}, block)",
        f"                            vb = T.axis.spatial({batch}, b)",
        f"                            vh = T.axis.spatial({heads}, h)",
        f"                            vi = T.axis.spatial({sequence}, i)",
        f"                            vj = T.axis.spatial({sequence}, j)",
        f"                            vk = T.axis.reduce({head_dim}, kd)",
        f"                            T.reads({q_name}[vb, vh, vi, vk], "
        f"{k_name}[vb, vh, vj, vk])",
        "                            T.writes(scores[bo, vj])",
        "                            with T.init():",
        "                                scores[bo, vj] = T.float32(0)",
        f"                            scores[bo, vj] = scores[bo, vj] + {q_name}[vb, vh, vi, vk] "
        f"* {k_name}[vb, vh, vj, vk] * {scale_expr}",
        *mask_add_lines,
        "                maxv[block] = T.float32(-3.4028234663852886e38)",
        f"                for j in T.serial(0, {sequence}):",
        "                    maxv[block] = T.max(maxv[block], scores[block, j])",
        "                den[block] = T.float32(0)",
        f"                for j in T.serial(0, {sequence}):",
        "                    probs[block, j] = T.exp(scores[block, j] - maxv[block])",
        "                    den[block] = den[block] + probs[block, j]",
        f"                for j in T.serial(0, {sequence}):",
        '                    with T.sblock("attention_row_softmax"):',
        f"                        bo = T.axis.spatial({output_elems}, block)",
        f"                        vj = T.axis.spatial({sequence}, j)",
        "                        T.reads(probs[bo, vj], den[bo])",
        "                        T.writes(probs[bo, vj])",
        "                        probs[bo, vj] = probs[bo, vj] / den[bo]",
        f"                for j in T.serial(0, {sequence}):",
        '                    with T.sblock("attention_av_matmul"):',
        f"                        bo = T.axis.spatial({output_elems}, block)",
        f"                        vb = T.axis.spatial({batch}, b)",
        f"                        vh = T.axis.spatial({heads}, h)",
        f"                        vi = T.axis.spatial({sequence}, i)",
        f"                        vd = T.axis.spatial({head_dim}, d)",
        f"                        vj = T.axis.reduce({sequence}, j)",
        f"                        T.reads(probs[bo, vj], {v_name}[vb, vh, vj, vd])",
        f"                        T.writes({out_name}[vb, vh, vi, vd])",
        "                        with T.init():",
        f"                            {out_name}[vb, vh, vi, vd] = T.float32(0)",
        f"                        {out_name}[vb, vh, vi, vd] = "
        f"{out_name}[vb, vh, vi, vd] + probs[bo, vj] * {v_name}[vb, vh, vj, vd]",
    ]
    return "\n".join(lines) + "\n"


class _AttentionProviderRegistration:
    """Restorable packed-func registration for the Python/Torch SDPA proof."""

    def __init__(self, name: str, previous: Any):
        self._name = name
        self._previous = previous
        self._closed = False

    def close(self) -> None:
        """Restore the previous packed function or remove the proof provider."""
        if self._closed:
            return
        import tvm  # pylint: disable=import-outside-toplevel
        import tvm_ffi  # pylint: disable=import-outside-toplevel

        if self._previous is None:
            tvm_ffi.remove_global_func(self._name)
        else:
            tvm.register_global_func(self._name, self._previous, override=True)
        self._closed = True

    def __enter__(self) -> "_AttentionProviderRegistration":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # pylint: disable=unused-argument
        self.close()


def register_python_torch_attention_sdpa() -> _AttentionProviderRegistration:
    """Register the opt-in correctness-only host-staged SDPA provider."""
    import tvm  # pylint: disable=import-outside-toplevel

    previous = tvm.get_global_func(ATTENTION_EXTERN_PACKED_FUNC, allow_missing=True)

    def _extern_attention_sdpa(q, k, v, out, batch, heads, sequence, head_dim, scale, causal):
        import torch  # pylint: disable=import-outside-toplevel
        import torch.nn.functional as torch_functional  # pylint: disable=import-outside-toplevel

        shape = (int(batch), int(heads), int(sequence), int(head_dim))
        _validate_python_torch_attention_tensor("Q", q, "float32", shape)
        _validate_python_torch_attention_tensor("K", k, "float32", shape)
        _validate_python_torch_attention_tensor("V", v, "float32", shape)
        _validate_python_torch_attention_tensor("out", out, "float32", shape)

        result = torch_functional.scaled_dot_product_attention(
            torch.from_numpy(q.numpy()),
            torch.from_numpy(k.numpy()),
            torch.from_numpy(v.numpy()),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=bool(causal),
            scale=float(scale),
        )
        out.copyfrom(result.numpy())

    tvm.register_global_func(ATTENTION_EXTERN_PACKED_FUNC, _extern_attention_sdpa, override=True)
    return _AttentionProviderRegistration(ATTENTION_EXTERN_PACKED_FUNC, previous)


def _unclassified(reason: str) -> AttentionABIClassification:
    return AttentionABIClassification(
        attention_source_kind=ATTENTION_SOURCE_KIND_WRAPPER_SDPA,
        attention_contract=ATTENTION_CONTRACT_UNCLASSIFIED,
        attention_abi_status=ATTENTION_ABI_STATUS_UNCLASSIFIED,
        attention_abi_version=ATTENTION_ABI_VERSION,
        attention_phase="unknown",
        attention_mask_kind="unknown",
        attention_rope_policy="unknown",
        attention_kv_cache_policy="unknown",
        attention_sequence_policy="unknown",
        attention_runtime_status=ATTENTION_RUNTIME_STATUS_DEFERRED,
        unsupported_attention_reason=reason,
    )


def _looks_like_decode(text: str) -> bool:
    return any(
        token in text
        for token in (
            "decode",
            "cache",
            "kv_cache",
            "past_key_value",
            "past_key_values",
            "single-token",
            "single_token",
            "single token",
        )
    )


def _default_report_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field_name in ATTENTION_REPORT_FIELDS:
        if field_name in {"attention_abi_version", "attention_provider_abi_version"}:
            fields[field_name] = 0
        elif field_name in {
            "attention_runtime_launch_count",
            "attention_artifact_call_count",
            "attention_qkv_bytes",
            "attention_mask_bytes",
            "attention_output_bytes",
            "attention_total_io_bytes",
            "attention_intermediate_buffer_bytes",
            "attention_host_staging_bytes",
            "attention_total_accounted_bytes",
        }:
            fields[field_name] = 0
        elif field_name in {
            "attention_causal",
            "attention_runtime_replacement_available",
            "attention_performance_claim",
            "attention_uses_host_staging",
        }:
            fields[field_name] = False
        elif field_name == "attention_scale":
            fields[field_name] = None
        else:
            fields[field_name] = ""
    return fields


def _wrapper_call_source_fields(source_or_call: Any) -> tuple[str, str, str, str]:
    if isinstance(source_or_call, str):
        return source_or_call, "", "", ""
    if isinstance(source_or_call, dict):
        return (
            str(source_or_call.get("source", "")),
            str(source_or_call.get("op_name", "")),
            str(source_or_call.get("case_name", "")),
            str(source_or_call.get("kernel_name", "")),
        )
    return (
        str(getattr(source_or_call, "source", "")),
        str(getattr(source_or_call, "op_name", "")),
        str(getattr(source_or_call, "case_name", "")),
        str(getattr(source_or_call, "kernel_name", "")),
    )


def _first_call(source: str, op_name: str) -> ast.Call | None:
    try:
        tree = ast.parse(source.strip())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _attribute_chain_ast(node.func) == op_name:
            return node
    return None


def _parse_reinterpret_tensor(node: ast.AST, role: str) -> dict[str, Any]:
    if not isinstance(node, ast.Call) or _attribute_chain_ast(node.func) != "reinterpret_tensor":
        raise UnsupportedTTIROpError(
            f"wrapper_attention_sdpa {role} must be reinterpret_tensor(...)"
        )
    if len(node.args) < 4:
        raise UnsupportedTTIROpError(
            f"wrapper_attention_sdpa {role} reinterpret_tensor must include base, shape, "
            "stride, offset"
        )
    shape = _literal_int_tuple(node.args[1], "shape")
    stride = _literal_int_tuple(node.args[2], "stride")
    offset = _literal_int(node.args[3], "offset")
    if len(shape) != 4 or len(stride) != 4:
        raise UnsupportedTTIROpError(
            f"wrapper_attention_sdpa {role} reinterpret_tensor must be rank-4"
        )
    if offset != 0:
        raise UnsupportedTTIROpError(
            f"wrapper_attention_sdpa {role} reinterpret_tensor offset must be 0"
        )
    return {
        "base": _node_text(node.args[0]),
        "shape": shape,
        "stride": stride,
    }


def _parse_optional_reinterpret_tensor(node: ast.AST, role: str) -> dict[str, Any] | None:
    if isinstance(node, ast.Call) and _attribute_chain_ast(node.func) == "reinterpret_tensor":
        return _parse_reinterpret_tensor(node, role)
    return None


def _parse_mask_kind(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and node.value is None:
        return "none_or_padding"
    return "unsupported_mask"


def _parse_bool_arg(node: ast.AST, label: str) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    raise UnsupportedTTIROpError(f"wrapper_attention_sdpa {label} must be a bool literal")


def _parse_optional_float_keyword(call: ast.Call, name: str) -> float | None:
    for keyword_node in call.keywords:
        if keyword_node.arg == name:
            return _literal_float(keyword_node.value, name)
    return None


def _literal_int_tuple(node: ast.AST, label: str) -> tuple[int, ...]:
    if not isinstance(node, ast.Tuple):
        raise UnsupportedTTIROpError(f"wrapper_attention_sdpa {label} must be a tuple")
    values = []
    for elt in node.elts:
        values.append(_literal_int(elt, label))
    return tuple(values)


def _literal_int(node: ast.AST, label: str) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    raise UnsupportedTTIROpError(f"wrapper_attention_sdpa {label} must be an int literal")


def _literal_float(node: ast.AST, label: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    raise UnsupportedTTIROpError(f"wrapper_attention_sdpa {label} must be numeric")


def _attribute_chain_ast(node: ast.AST) -> str:
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


def _node_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= int(extent)
    return tuple(reversed(stride))


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _dtype_nbytes(dtype: str) -> int:
    if dtype in {"float64", "int64"}:
        return 8
    if dtype in {"float32", "int32"}:
        return 4
    if dtype in {"float16", "bfloat16", "int16"}:
        return 2
    if dtype in {"bool", "int8", "uint8"}:
        return 1
    return 4


def _tuple_text(values: tuple[int, ...]) -> str:
    return ", ".join(str(int(value)) for value in values)


def _shape_literal(values: tuple[int, ...]) -> str:
    text = _tuple_text(values)
    if len(values) == 1:
        text += ","
    return f"({text})"


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"\W", "_", name)
    if not sanitized or sanitized[0].isdigit() or keyword.iskeyword(sanitized):
        sanitized = f"_{sanitized}"
    return sanitized


def _bool_literal(value: bool) -> str:
    return "True" if value else "False"


def _validate_python_torch_attention_tensor(
    role: str,
    tensor: Any,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    if not hasattr(tensor, "dtype") or not hasattr(tensor, "shape"):
        raise TypeError(f"attention_sdpa {role} must be a TVM tensor")
    if str(tensor.dtype) != dtype:
        raise TypeError(
            f"python_torch_host_staged attention_sdpa supports {dtype} {role}, "
            f"got {tensor.dtype}"
        )
    actual_shape = tuple(int(dim) for dim in tensor.shape)
    if actual_shape != shape:
        raise ValueError(f"attention_sdpa {role} shape must be {shape}, got {actual_shape}")
