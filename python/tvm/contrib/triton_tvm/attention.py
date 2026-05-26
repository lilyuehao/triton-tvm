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
"""Pre-M10 attention ABI classification helpers.

This module deliberately stops at ABI/report classification.  It does not lower
or execute attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ATTENTION_ABI_VERSION = 1
ATTENTION_CONTRACT_VIT_FULL = "attention_vit_full_v1"
ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL = "attention_llama_causal_prefill_v1"
ATTENTION_CONTRACT_LLAMA_DECODE = "attention_llama_decode_v1"
ATTENTION_CONTRACT_UNCLASSIFIED = "attention_unclassified"
ATTENTION_RUNTIME_STATUS_DEFERRED = "deferred_attention_runtime"
ATTENTION_SOURCE_KIND_WRAPPER_SDPA = "wrapper_aten_scaled_dot_product_attention"
ATTENTION_ABI_STATUS_CLASSIFIED = "abi_classified_runtime_deferred"
ATTENTION_ABI_STATUS_UNCLASSIFIED = "unsupported_attention_abi"

ATTENTION_TARGET_CONTRACTS = (
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_LLAMA_DECODE,
)

ATTENTION_REPORT_FIELDS = (
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
            for field_name in ATTENTION_REPORT_FIELDS
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
    if op_name != "torch.ops.aten._scaled_dot_product_efficient_attention.default":
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
