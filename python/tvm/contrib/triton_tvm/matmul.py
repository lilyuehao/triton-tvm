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
"""Matmul semantic extraction and runtime proof helpers for Triton TVM."""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import UnsupportedTTIROpError
from .op_graph import NormalizedTTIROpGraph, TTIROp, TTIRType


NATIVE_TIR_MATMUL_SCHEDULE_ID = "cuda_block_per_output_serial_k_v1"
TILED_TIR_MATMUL_SCHEDULE_ID = "cuda_block_tile_8x8_serial_k_v1"
TILED_TIR_MATMUL_TILE_M = 8
TILED_TIR_MATMUL_TILE_N = 8
SIMT_TIR_MATMUL_SCHEDULE_ID = "cuda_block_tile_16x16_simt_v1"
SIMT_TIR_MATMUL_TILE_M = 16
SIMT_TIR_MATMUL_TILE_N = 16
TENSORCORE_TIR_MATMUL_SCHEDULE_ID = "cuda_ptx_mma_m8n8k4_warp_tile_16x16_fp16fp32_v1"
TENSORCORE_TIR_MATMUL_TILE_M = 16
TENSORCORE_TIR_MATMUL_TILE_N = 16
TENSORCORE_TIR_MATMUL_TILE_K = 4
MATMUL_PERF_ENVELOPE_ID = "matmul_perf_core_v1"
MATMUL_SCHEDULE_REGISTRY_VERSION = "m9p_schedule_registry_v1"
MATMUL_TUNE_KEY_VERSION = "m9p_tune_key_v1"
MATMUL_PERF_STATUS_NOT_MEASURED = "not_measured"
MATMUL_PERF_STATUS_UNAVAILABLE = "unavailable"
TORCH_CUDA_CUBLAS_BASELINE_ID = "torch_cuda_cublas_baseline_v1"
TRITON_NATIVE_MATMUL_BASELINE_ID = "native_triton_matmul_baseline_v1"
EXTERN_GEMM_SYMBOL = "extern_kernels.mm"
EXTERN_GEMM_PACKED_FUNC = "tvm.contrib.triton_tvm.extern_gemm"
EXTERN_ADDMM_BIAS_SYMBOL = "extern_kernels.addmm"
EXTERN_ADDMM_BIAS_PACKED_FUNC = "tvm.contrib.triton_tvm.extern_addmm_bias"
EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON = "extern_addmm_bias_runtime_replacement_gate_closed"
EXTERN_ADDMM_BIAS_RUNTIME_PROVIDER_REASON = (
    "extern_addmm_bias_python_torch_host_staged_provider_enabled"
)
EXTERN_GEMM_RUNTIME_KIND = "artifact_only"
EXTERN_GEMM_RUNTIME_REPLACEMENT = "not_available"
EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON = "extern_gemm_runtime_replacement_gate_closed"
EXTERN_GEMM_RUNTIME_PROVIDER_KIND = "runtime_provider"
EXTERN_GEMM_PROVIDER_NONE = "none"
EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED = "python_torch_host_staged"
EXTERN_GEMM_PROVIDER_ABI_VERSION = 1
EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY = "artifact_only"
EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED = "runtime_resolved"
EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_FAILED = "runtime_failed"
EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY = "correctness_only"
EXTERN_GEMM_RUNTIME_PROVIDER_REASON = "extern_gemm_python_torch_host_staged_provider_enabled"


@dataclass(frozen=True)
class MatmulScheduleCandidate:
    """Frozen M9.P schedule candidate registry record."""

    schedule_id: str
    package_id: str
    implementation_kind: str
    applicability_predicate: str
    tunable_params: dict[str, Any] = field(default_factory=dict)
    builder_hook: str = ""
    correctness_status: str = "not_measured"
    latency_result: dict[str, Any] = field(default_factory=dict)
    selected_reason: str = ""
    rejected_reason: str = ""
    performance_claim: bool = False
    performance_claim_metadata: dict[str, Any] = field(default_factory=dict)

    def cache_payload(self) -> dict[str, Any]:
        """Return stable JSON-able schedule metadata for reports/cache keys."""
        return asdict(self)


_MATMUL_SCHEDULE_REGISTRY: tuple[MatmulScheduleCandidate, ...] = (
    MatmulScheduleCandidate(
        schedule_id=TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
        package_id="M9.PA",
        implementation_kind="native_tir_schedule",
        applicability_predicate="matmul_perf_core_v1_tt_dot_fp16_16x16_exact",
        tunable_params={
            "tile_m": TENSORCORE_TIR_MATMUL_TILE_M,
            "tile_n": TENSORCORE_TIR_MATMUL_TILE_N,
            "tile_k": TENSORCORE_TIR_MATMUL_TILE_K,
            "warps_per_tile": 1,
            "ptx_mma_shape": "m8n8k4",
        },
        builder_hook="_build_native_tensorcore_schedule_source",
        performance_claim=True,
        performance_claim_metadata={
            "claim_scope": "fp16 TensorCore PTX-MMA candidate under matmul_perf_core_v1",
            "baseline_required": True,
        },
    ),
    MatmulScheduleCandidate(
        schedule_id=SIMT_TIR_MATMUL_SCHEDULE_ID,
        package_id="M9.PC",
        implementation_kind="native_tir_schedule",
        applicability_predicate="matmul_perf_core_v1_tt_dot_fp16_or_bf16_16x16_exact_no_tensorcore",
        tunable_params={
            "tile_m": SIMT_TIR_MATMUL_TILE_M,
            "tile_n": SIMT_TIR_MATMUL_TILE_N,
            "threads_per_tile": SIMT_TIR_MATMUL_TILE_M * SIMT_TIR_MATMUL_TILE_N,
            "serial_k": True,
        },
        builder_hook="_build_native_simt_schedule_source",
        performance_claim=False,
        performance_claim_metadata={
            "claim_scope": "non-TensorCore SIMT fallback under matmul_perf_core_v1",
            "baseline_required": True,
        },
    ),
    MatmulScheduleCandidate(
        schedule_id=TILED_TIR_MATMUL_SCHEDULE_ID,
        package_id="M9.7",
        implementation_kind="native_tir_schedule",
        applicability_predicate="tt_dot_fp16_or_bf16_mn_multiple_of_8",
        tunable_params={
            "tile_m": TILED_TIR_MATMUL_TILE_M,
            "tile_n": TILED_TIR_MATMUL_TILE_N,
            "serial_k": True,
        },
        builder_hook="_build_native_tiled_schedule_source",
        performance_claim=False,
        performance_claim_metadata={"claim_scope": "correctness_first_serial_k"},
    ),
    MatmulScheduleCandidate(
        schedule_id=NATIVE_TIR_MATMUL_SCHEDULE_ID,
        package_id="M9.3",
        implementation_kind="native_tir_schedule",
        applicability_predicate="accepted_tt_dot_semantics",
        tunable_params={"serial_k": True, "one_output_element_per_block": True},
        builder_hook="_build_native_per_output_schedule_source",
        performance_claim=False,
        performance_claim_metadata={"claim_scope": "correctness_first_serial_k"},
    ),
)


def matmul_schedule_candidate_records() -> tuple[dict[str, Any], ...]:
    """Return the frozen M9.P schedule candidate registry as report records."""
    return tuple(candidate.cache_payload() for candidate in _MATMUL_SCHEDULE_REGISTRY)


def matmul_schedule_candidate_ids() -> tuple[str, ...]:
    """Return schedule ids in registry priority order."""
    return tuple(candidate.schedule_id for candidate in _MATMUL_SCHEDULE_REGISTRY)


def matmul_schedule_registry_version() -> str:
    """Return the frozen schedule registry version."""
    return MATMUL_SCHEDULE_REGISTRY_VERSION


@dataclass(frozen=True)
class MatmulSemantics:
    """Structured M9 matmul semantics before target implementation selection."""

    source_kind: str
    source_name: str
    kernel_name: str
    m: int
    n: int
    k: int
    batch_dims: tuple[int, ...]
    a_param: str
    b_param: str
    c_param: str
    a_dtype: str
    b_dtype: str
    accumulator_dtype: str
    output_dtype: str
    a_layout: str
    b_layout: str
    c_layout: str
    a_stride: tuple[int, int]
    b_stride: tuple[int, int]
    c_stride: tuple[int, int]
    transpose_a: bool
    transpose_b: bool
    block_m: int
    block_n: int
    block_k: int
    input_precision: str
    tf32_policy: str
    bounds_policy: str
    mask_kind: str
    epilogue_kind: str
    alias_policy: str
    noalias: bool
    target_kind: str
    implementation_kind: str
    fallback_reason: str
    bias_param: str = ""
    bias_shape: tuple[int, ...] = ()
    bias_stride: tuple[int, ...] = ()
    alpha: float = 1.0
    beta: float = 1.0

    def cache_payload(self) -> dict[str, Any]:
        """Return stable JSON-able matmul metadata for cache keys and reports."""
        return asdict(self)


@dataclass(frozen=True)
class TargetMatmulDecision:
    """Target implementation decision for a validated matmul semantic block."""

    matmul_contract_ok: bool
    implementation_kind: str
    schedule_id: str = ""
    extern_symbol: str = ""
    extern_packed_func: str = ""
    extern_runtime_kind: str = ""
    extern_runtime_replacement: str = ""
    extern_runtime_replacement_available: bool = False
    extern_runtime_replacement_reason: str = ""
    extern_gemm_runtime_status: str = ""
    extern_gemm_provider_kind: str = ""
    extern_gemm_provider_abi_version: int = 0
    extern_gemm_runtime_claim: str = ""
    extern_gemm_performance_claim: bool = False
    extern_gemm_uses_host_staging: bool = False
    unsupported_matmul_reason: str = ""
    matmul_perf_envelope: str = ""
    candidate_schedule_ids: tuple[str, ...] = ()
    selected_schedule_id: str = ""
    schedule_reject_reasons: dict[str, str] = field(default_factory=dict)
    perf_guard_status: str = ""
    tune_key: dict[str, Any] = field(default_factory=dict)

    def cache_payload(self) -> dict[str, Any]:
        """Return stable JSON-able policy metadata for cache keys and reports."""
        return asdict(self)


@dataclass(frozen=True)
class TargetMatmulPolicy:
    """Minimal M9 target policy for matmul implementation selection.

    TTIR ``tt.dot`` uses the correctness-first native CUDA TIR schedule: one
    CUDA block computes one output element and serializes K. Wrapper-level
    ``extern_kernels.mm`` is represented as an explicit packed-call artifact
    because this environment does not provide a registered cublas/cublaslt TVM
    packed runtime.
    """

    target_kind: str = "cuda"
    extern_gemm_runtime_provider: str = EXTERN_GEMM_PROVIDER_NONE

    def decide(
        self,
        semantics: MatmulSemantics,
        *,
        matmul_contract_ok: bool,
    ) -> TargetMatmulDecision:
        """Choose the implementation kind for a validated matmul semantic payload."""
        if not matmul_contract_ok:
            return TargetMatmulDecision(
                matmul_contract_ok=False,
                implementation_kind="unsupported",
                unsupported_matmul_reason="matmul_contract_failed",
            )

        unsupported_reason = self._unsupported_reason(semantics)
        if unsupported_reason:
            return TargetMatmulDecision(
                matmul_contract_ok=True,
                implementation_kind="unsupported",
                unsupported_matmul_reason=unsupported_reason,
            )

        if semantics.source_kind == "wrapper_extern_gemm":
            if self.extern_gemm_runtime_provider == EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED:
                return TargetMatmulDecision(
                    matmul_contract_ok=True,
                    implementation_kind="extern_gemm",
                    extern_symbol=EXTERN_GEMM_SYMBOL,
                    extern_packed_func=EXTERN_GEMM_PACKED_FUNC,
                    extern_runtime_kind=EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
                    extern_runtime_replacement=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                    extern_runtime_replacement_available=True,
                    extern_runtime_replacement_reason=EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
                    extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
                    extern_gemm_provider_kind=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                    extern_gemm_provider_abi_version=EXTERN_GEMM_PROVIDER_ABI_VERSION,
                    extern_gemm_runtime_claim=EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                    extern_gemm_performance_claim=False,
                    extern_gemm_uses_host_staging=True,
                    matmul_perf_envelope=MATMUL_PERF_ENVELOPE_ID,
                    perf_guard_status=MATMUL_PERF_STATUS_UNAVAILABLE,
                )
            return TargetMatmulDecision(
                matmul_contract_ok=True,
                implementation_kind="extern_gemm",
                extern_symbol=EXTERN_GEMM_SYMBOL,
                extern_packed_func=EXTERN_GEMM_PACKED_FUNC,
                extern_runtime_kind=EXTERN_GEMM_RUNTIME_KIND,
                extern_runtime_replacement=EXTERN_GEMM_RUNTIME_REPLACEMENT,
                extern_runtime_replacement_available=False,
                extern_runtime_replacement_reason=EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
                extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
                extern_gemm_provider_kind=EXTERN_GEMM_PROVIDER_NONE,
                extern_gemm_provider_abi_version=0,
                extern_gemm_runtime_claim="",
                extern_gemm_performance_claim=False,
                extern_gemm_uses_host_staging=False,
                matmul_perf_envelope=MATMUL_PERF_ENVELOPE_ID,
                perf_guard_status=MATMUL_PERF_STATUS_UNAVAILABLE,
            )

        if semantics.source_kind == "wrapper_extern_addmm_bias":
            if self.extern_gemm_runtime_provider == EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED:
                return TargetMatmulDecision(
                    matmul_contract_ok=True,
                    implementation_kind="extern_addmm_bias",
                    extern_symbol=EXTERN_ADDMM_BIAS_SYMBOL,
                    extern_packed_func=EXTERN_ADDMM_BIAS_PACKED_FUNC,
                    extern_runtime_kind=EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
                    extern_runtime_replacement=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                    extern_runtime_replacement_available=True,
                    extern_runtime_replacement_reason=EXTERN_ADDMM_BIAS_RUNTIME_PROVIDER_REASON,
                    extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
                    extern_gemm_provider_kind=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                    extern_gemm_provider_abi_version=EXTERN_GEMM_PROVIDER_ABI_VERSION,
                    extern_gemm_runtime_claim=EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                    extern_gemm_performance_claim=False,
                    extern_gemm_uses_host_staging=True,
                    matmul_perf_envelope=MATMUL_PERF_ENVELOPE_ID,
                    perf_guard_status=MATMUL_PERF_STATUS_UNAVAILABLE,
                )
            return TargetMatmulDecision(
                matmul_contract_ok=True,
                implementation_kind="extern_addmm_bias",
                extern_symbol=EXTERN_ADDMM_BIAS_SYMBOL,
                extern_packed_func=EXTERN_ADDMM_BIAS_PACKED_FUNC,
                extern_runtime_kind=EXTERN_GEMM_RUNTIME_KIND,
                extern_runtime_replacement=EXTERN_GEMM_RUNTIME_REPLACEMENT,
                extern_runtime_replacement_available=False,
                extern_runtime_replacement_reason=EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON,
                extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
                extern_gemm_provider_kind=EXTERN_GEMM_PROVIDER_NONE,
                extern_gemm_provider_abi_version=0,
                extern_gemm_runtime_claim="",
                extern_gemm_performance_claim=False,
                extern_gemm_uses_host_staging=False,
                matmul_perf_envelope=MATMUL_PERF_ENVELOPE_ID,
                perf_guard_status=MATMUL_PERF_STATUS_UNAVAILABLE,
            )

        schedule_id = _native_matmul_schedule_id(semantics)
        return TargetMatmulDecision(
            matmul_contract_ok=True,
            implementation_kind="native_tir_schedule",
            schedule_id=schedule_id,
            **_native_matmul_decision_metadata(semantics, schedule_id),
        )

    def _unsupported_reason(self, semantics: MatmulSemantics) -> str:
        if self.target_kind != "cuda" or semantics.target_kind != "cuda":
            return "matmul_target_kind_not_supported"
        if semantics.batch_dims:
            return "batched_matmul_not_supported"
        if semantics.transpose_a or semantics.transpose_b:
            return "transposed_matmul_not_supported"
        if semantics.bounds_policy != "exact":
            return "matmul_bounds_policy_not_supported"
        if semantics.mask_kind != "none":
            return "masked_matmul_not_supported"
        if (
            semantics.source_kind != "wrapper_extern_addmm_bias"
            and semantics.epilogue_kind != "none"
        ):
            return "matmul_epilogue_not_supported"
        if (
            semantics.accumulator_dtype != "float32"
            or semantics.output_dtype != "float32"
        ):
            return "matmul_accumulator_dtype_not_supported"
        if semantics.source_kind == "wrapper_extern_addmm_bias":
            if semantics.epilogue_kind != "bias_add":
                return "extern_addmm_bias_requires_bias_add_epilogue"
            if (
                semantics.a_layout != "row_major"
                or semantics.b_layout not in ("row_major", "transposed_weight_view")
                or semantics.c_layout != "row_major"
            ):
                return "non_row_major_matmul_not_supported"
            if semantics.a_dtype != "float32" or semantics.b_dtype != "float32":
                return "matmul_input_dtype_not_supported"
            if semantics.bias_shape not in ((semantics.n,), (semantics.m, semantics.n)):
                return "extern_addmm_bias_shape_not_supported"
            if semantics.alpha != 1.0 or semantics.beta != 1.0:
                return "extern_addmm_bias_requires_alpha_beta_one"
            return ""
        if semantics.source_kind == "wrapper_extern_gemm":
            if (
                semantics.a_layout != "row_major"
                or semantics.b_layout not in ("row_major", "transposed_weight_view")
                or semantics.c_layout != "row_major"
            ):
                return "non_row_major_matmul_not_supported"
            if semantics.a_dtype != "float32" or semantics.b_dtype != "float32":
                return "matmul_input_dtype_not_supported"
            return ""
        if semantics.source_kind != "tt_dot":
            return "matmul_source_kind_not_supported"
        if (
            semantics.a_layout != "row_major"
            or semantics.b_layout != "row_major"
            or semantics.c_layout != "row_major"
        ):
            return "non_row_major_matmul_not_supported"
        if (
            semantics.a_dtype not in ("float16", "bfloat16")
            or semantics.b_dtype not in ("float16", "bfloat16")
        ):
            return "matmul_input_dtype_not_supported"
        return ""


def extract_matmul_semantics_from_ttir(graph: NormalizedTTIROpGraph) -> MatmulSemantics:
    """Extract the first supported M9 ``tt.dot`` matmul semantic shape.

    M9.1 intentionally accepts only exact, unmasked, rank-2 row-major GEMM:
    ``A[M, K] x B[K, N] -> C[M, N]`` with fp32 accumulation/output.
    """

    defs = graph.op_by_result()
    dot_ops = [op for op in graph.ops if op.name == "tt.dot"]
    if len(dot_ops) != 1:
        raise UnsupportedTTIROpError(
            f"matmul_minimal requires exactly one tt.dot, got {len(dot_ops)}"
        )

    dot = dot_ops[0]
    if len(dot.operands) != 2 or len(dot.results) != 1:
        raise UnsupportedTTIROpError("matmul_minimal requires binary tt.dot with one result")
    if len(dot.result_types) != 1:
        raise UnsupportedTTIROpError("matmul_minimal tt.dot must have exactly one result type")

    a_type = _value_type(defs, dot.operands[0])
    b_type = _value_type(defs, dot.operands[1])
    c_type = dot.result_types[0]
    if len(a_type.shape) != 2 or len(b_type.shape) != 2 or len(c_type.shape) != 2:
        raise UnsupportedTTIROpError("matmul_minimal only supports rank-2 tt.dot tensors")

    m, k = a_type.shape
    b_k, n = b_type.shape
    c_m, c_n = c_type.shape
    if k != b_k:
        raise UnsupportedTTIROpError(
            f"matmul_minimal K mismatch: A K={k}, B K={b_k}"
        )
    if (m, n) != (c_m, c_n):
        raise UnsupportedTTIROpError(
            "matmul_minimal output shape must match A[M,K] x B[K,N]"
        )
    if c_type.dtype != "float32":
        raise UnsupportedTTIROpError("matmul_minimal requires fp32 output/accumulator")

    a_load = _load_op(defs, dot.operands[0], "A")
    b_load = _load_op(defs, dot.operands[1], "B")
    if len(a_load.operands) != 1 or len(b_load.operands) != 1:
        raise UnsupportedTTIROpError("matmul_minimal only supports unmasked tt.load")

    stores = [
        op
        for op in graph.ops
        if op.name == "tt.store" and len(op.operands) >= 2 and op.operands[1] == dot.results[0]
    ]
    if len(stores) != 1:
        raise UnsupportedTTIROpError(
            f"matmul_minimal requires one tt.store of the tt.dot result, got {len(stores)}"
        )
    store = stores[0]
    if len(store.operands) != 2:
        raise UnsupportedTTIROpError("matmul_minimal only supports unmasked tt.store")

    a_param = _base_pointer_param(graph, a_load.operands[0])
    b_param = _base_pointer_param(graph, b_load.operands[0])
    c_param = _base_pointer_param(graph, store.operands[0])
    if not a_param or not b_param or not c_param:
        raise UnsupportedTTIROpError("matmul_minimal requires pointer-backed A/B/C buffers")
    if len({a_param, b_param, c_param}) != 3:
        raise UnsupportedTTIROpError("matmul_minimal requires distinct A/B/C buffers")

    input_precision = _unquote_attr(dot.attrs.get("inputPrecision", "tf32"))
    return MatmulSemantics(
        source_kind="tt_dot",
        source_name=dot.results[0],
        kernel_name=graph.function_name,
        m=m,
        n=n,
        k=k,
        batch_dims=(),
        a_param=a_param,
        b_param=b_param,
        c_param=c_param,
        a_dtype=a_type.dtype,
        b_dtype=b_type.dtype,
        accumulator_dtype=c_type.dtype,
        output_dtype=c_type.dtype,
        a_layout="row_major",
        b_layout="row_major",
        c_layout="row_major",
        a_stride=(k, 1),
        b_stride=(n, 1),
        c_stride=(n, 1),
        transpose_a=False,
        transpose_b=False,
        block_m=m,
        block_n=n,
        block_k=k,
        input_precision=input_precision,
        tf32_policy=input_precision,
        bounds_policy="exact",
        mask_kind="none",
        epilogue_kind="none",
        alias_policy="distinct_buffers",
        noalias=True,
        target_kind="cuda",
        implementation_kind="unresolved",
        fallback_reason="",
    )


def extract_matmul_semantics_from_wrapper_extern(
    source_or_call: Any,
    *,
    case_name: str = "",
    kernel_name: str = "",
) -> MatmulSemantics:
    """Extract M9.4 wrapper ``extern_kernels.mm`` semantics from a source record.

    The accepted wrapper shape is the static rank-2 Inductor form:
    ``extern_kernels.mm(reinterpret_tensor(A, (M,K), (K,1), 0),
    reinterpret_tensor(B, (K,N), stride, 0), out=C)``.  ``stride`` may be
    row-major ``(N, 1)`` or the real-corpus transposed-weight view ``(1, K)``.
    """

    source, op_name, call_case, call_kernel = _wrapper_call_source_fields(source_or_call)
    case_name = case_name or call_case
    if op_name == EXTERN_ADDMM_BIAS_SYMBOL:
        return extract_matmul_semantics_from_wrapper_extern_addmm(
            source_or_call,
            case_name=case_name,
            kernel_name=kernel_name or call_kernel or case_name or "wrapper_extern_addmm_bias",
        )
    if op_name and op_name != EXTERN_GEMM_SYMBOL:
        raise UnsupportedTTIROpError(f"Unsupported wrapper matmul op {op_name!r}")
    if not source:
        raise UnsupportedTTIROpError("wrapper_extern_gemm requires a source line")
    if _first_call(source, EXTERN_ADDMM_BIAS_SYMBOL) is not None:
        return extract_matmul_semantics_from_wrapper_extern_addmm(
            source_or_call,
            case_name=case_name,
            kernel_name=kernel_name or call_kernel or case_name or "wrapper_extern_addmm_bias",
        )
    kernel_name = kernel_name or call_kernel or case_name or "wrapper_extern_gemm"

    call = _first_call(source, EXTERN_GEMM_SYMBOL)
    if call is None:
        raise UnsupportedTTIROpError("wrapper_extern_gemm requires extern_kernels.mm")
    if len(call.args) < 2:
        raise UnsupportedTTIROpError("wrapper_extern_gemm requires lhs and rhs arguments")

    lhs = _parse_reinterpret_tensor(call.args[0], "lhs")
    rhs = _parse_reinterpret_tensor(call.args[1], "rhs")
    out_param = _parse_out_param(call)

    m, k = lhs["shape"]
    b_k, n = rhs["shape"]
    if k != b_k:
        raise UnsupportedTTIROpError(
            f"wrapper_extern_gemm K mismatch: A K={k}, B K={b_k}"
        )
    if lhs["stride"] != (k, 1):
        raise UnsupportedTTIROpError(
            "wrapper_extern_gemm lhs must be row-major reinterpret_tensor stride (K, 1)"
        )
    if rhs["stride"] == (n, 1):
        b_layout = "row_major"
    elif rhs["stride"] == (1, k):
        b_layout = "transposed_weight_view"
    else:
        raise UnsupportedTTIROpError(
            "wrapper_extern_gemm rhs must be row-major stride (N, 1) "
            "or transposed-weight-view stride (1, K)"
        )

    return MatmulSemantics(
        source_kind="wrapper_extern_gemm",
        source_name=EXTERN_GEMM_SYMBOL,
        kernel_name=kernel_name,
        m=m,
        n=n,
        k=k,
        batch_dims=(),
        a_param=str(lhs["base"]),
        b_param=str(rhs["base"]),
        c_param=out_param,
        a_dtype="float32",
        b_dtype="float32",
        accumulator_dtype="float32",
        output_dtype="float32",
        a_layout="row_major",
        b_layout=b_layout,
        c_layout="row_major",
        a_stride=lhs["stride"],
        b_stride=rhs["stride"],
        c_stride=(n, 1),
        transpose_a=False,
        transpose_b=False,
        block_m=m,
        block_n=n,
        block_k=k,
        input_precision="extern_fp32",
        tf32_policy="extern_runtime_default",
        bounds_policy="exact",
        mask_kind="none",
        epilogue_kind="none",
        alias_policy="wrapper_extern_buffers",
        noalias=True,
        target_kind="cuda",
        implementation_kind="unresolved",
        fallback_reason="",
    )


def extract_matmul_semantics_from_wrapper_extern_addmm(
    source_or_call: Any,
    *,
    case_name: str = "",
    kernel_name: str = "",
) -> MatmulSemantics:
    """Extract the M9.PE minimal wrapper ``extern_kernels.addmm`` bias slice."""

    source, op_name, call_case, call_kernel = _wrapper_call_source_fields(source_or_call)
    case_name = case_name or call_case
    kernel_name = kernel_name or call_kernel or case_name or "wrapper_extern_addmm_bias"
    if op_name and op_name != EXTERN_ADDMM_BIAS_SYMBOL:
        raise UnsupportedTTIROpError(f"Unsupported wrapper addmm op {op_name!r}")
    if not source:
        raise UnsupportedTTIROpError("wrapper_extern_addmm_bias requires a source line")

    call = _first_call(source, EXTERN_ADDMM_BIAS_SYMBOL)
    if call is None:
        raise UnsupportedTTIROpError("wrapper_extern_addmm_bias requires extern_kernels.addmm")
    if len(call.args) < 3:
        raise UnsupportedTTIROpError("wrapper_extern_addmm_bias requires bias, lhs, and rhs")

    lhs = _parse_reinterpret_tensor(call.args[1], "lhs")
    rhs = _parse_reinterpret_tensor(call.args[2], "rhs")
    out_param = _parse_out_param(call)
    alpha = _parse_optional_float_keyword(call, "alpha", 1.0)
    beta = _parse_optional_float_keyword(call, "beta", 1.0)

    m, k = lhs["shape"]
    b_k, n = rhs["shape"]
    if k != b_k:
        raise UnsupportedTTIROpError(
            f"wrapper_extern_addmm_bias K mismatch: A K={k}, B K={b_k}"
        )
    if lhs["stride"] != (k, 1):
        raise UnsupportedTTIROpError(
            "wrapper_extern_addmm_bias lhs must be row-major reinterpret_tensor stride (K, 1)"
        )
    if rhs["stride"] == (n, 1):
        b_layout = "row_major"
    elif rhs["stride"] == (1, k):
        b_layout = "transposed_weight_view"
    else:
        raise UnsupportedTTIROpError(
            "wrapper_extern_addmm_bias rhs must be row-major stride (N, 1) "
            "or transposed-weight-view stride (1, K)"
        )
    if alpha != 1.0 or beta != 1.0:
        raise UnsupportedTTIROpError("wrapper_extern_addmm_bias requires alpha=1 and beta=1")

    bias = _parse_addmm_bias(call.args[0], m=m, n=n)
    return MatmulSemantics(
        source_kind="wrapper_extern_addmm_bias",
        source_name=EXTERN_ADDMM_BIAS_SYMBOL,
        kernel_name=kernel_name,
        m=m,
        n=n,
        k=k,
        batch_dims=(),
        a_param=str(lhs["base"]),
        b_param=str(rhs["base"]),
        c_param=out_param,
        a_dtype="float32",
        b_dtype="float32",
        accumulator_dtype="float32",
        output_dtype="float32",
        a_layout="row_major",
        b_layout=b_layout,
        c_layout="row_major",
        a_stride=lhs["stride"],
        b_stride=rhs["stride"],
        c_stride=(n, 1),
        transpose_a=False,
        transpose_b=False,
        block_m=m,
        block_n=n,
        block_k=k,
        input_precision="extern_fp32",
        tf32_policy="extern_runtime_default",
        bounds_policy="exact",
        mask_kind="none",
        epilogue_kind="bias_add",
        alias_policy="wrapper_extern_buffers",
        noalias=True,
        target_kind="cuda",
        implementation_kind="unresolved",
        fallback_reason="",
        bias_param=str(bias["base"]),
        bias_shape=bias["shape"],
        bias_stride=bias["stride"],
        alpha=alpha,
        beta=beta,
    )


def build_extern_gemm_tirx_source(
    semantics: MatmulSemantics,
    decision: TargetMatmulDecision,
    *,
    target_attrs: str = "cuda",
) -> str:
    """Build the explicit M9.4 packed-call artifact for wrapper extern GEMM."""

    if semantics.source_kind != "wrapper_extern_gemm":
        raise UnsupportedTTIROpError("extern GEMM artifact requires wrapper_extern_gemm semantics")
    if decision.implementation_kind != "extern_gemm" or decision.extern_symbol != EXTERN_GEMM_SYMBOL:
        raise UnsupportedTTIROpError("extern GEMM artifact requires extern_gemm policy decision")

    func_name = _sanitize_identifier(semantics.kernel_name or "wrapper_extern_gemm")
    a_name = _sanitize_identifier(semantics.a_param or "a")
    b_name = _sanitize_identifier(semantics.b_param or "b")
    c_name = _sanitize_identifier(semantics.c_param or "out")
    transposed_b = semantics.b_layout == "transposed_weight_view"
    b_storage_shape = (semantics.n, semantics.k) if transposed_b else (semantics.k, semantics.n)
    b_storage_stride = (semantics.k, 1) if transposed_b else semantics.b_stride
    b_transposed_expr = "T.bool(True)" if transposed_b else "T.bool(False)"
    lines: list[str] = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}({a_name}_handle: T.handle, {b_name}_handle: T.handle, "
        f"{c_name}_handle: T.handle):",
        "        T.func_attr({"
        f'"global_symbol": "{semantics.kernel_name}", '
        '"tirx.noalias": True, '
        f'"target": T.target({target_attrs!r}), '
        '"triton_tvm.contract": "matmul_minimal", '
        f'"triton_tvm.matmul_source_kind": "{semantics.source_kind}", '
        f'"triton_tvm.a_dtype": "{semantics.a_dtype}", '
        f'"triton_tvm.b_dtype": "{semantics.b_dtype}", '
        f'"triton_tvm.output_dtype": "{semantics.output_dtype}", '
        f'"triton_tvm.accumulator_dtype": "{semantics.accumulator_dtype}", '
        f'"triton_tvm.input_precision": "{semantics.input_precision}", '
        f'"triton_tvm.tf32_policy": "{semantics.tf32_policy}", '
        f'"triton_tvm.bounds_policy": "{semantics.bounds_policy}", '
        f'"triton_tvm.mask_kind": "{semantics.mask_kind}", '
        f'"triton_tvm.epilogue_kind": "{semantics.epilogue_kind}", '
        f'"triton_tvm.alias_policy": "{semantics.alias_policy}", '
        f'"triton_tvm.implementation_kind": "{decision.implementation_kind}", '
        f'"triton_tvm.extern_symbol": "{decision.extern_symbol}", '
        f'"triton_tvm.extern_packed_func": "{decision.extern_packed_func}", '
        f'"triton_tvm.extern_runtime_kind": "{decision.extern_runtime_kind}", '
        f'"triton_tvm.extern_runtime_replacement": "{decision.extern_runtime_replacement}", '
        f'"triton_tvm.extern_runtime_replacement_available": '
        f'{_bool_literal(decision.extern_runtime_replacement_available)}, '
        f'"triton_tvm.extern_runtime_replacement_reason": '
        f'"{decision.extern_runtime_replacement_reason}", '
        f'"triton_tvm.extern_gemm_runtime_status": "{decision.extern_gemm_runtime_status}", '
        f'"triton_tvm.extern_gemm_provider_kind": "{decision.extern_gemm_provider_kind}", '
        f'"triton_tvm.extern_gemm_provider_abi_version": '
        f'{decision.extern_gemm_provider_abi_version}, '
        f'"triton_tvm.extern_gemm_runtime_claim": "{decision.extern_gemm_runtime_claim}", '
        f'"triton_tvm.extern_gemm_performance_claim": '
        f'{_bool_literal(decision.extern_gemm_performance_claim)}, '
        f'"triton_tvm.extern_gemm_uses_host_staging": '
        f'{_bool_literal(decision.extern_gemm_uses_host_staging)}, '
        f'"triton_tvm.matmul_m": {semantics.m}, '
        f'"triton_tvm.matmul_n": {semantics.n}, '
        f'"triton_tvm.matmul_k": {semantics.k}, '
        f'"triton_tvm.a_layout": "{semantics.a_layout}", '
        f'"triton_tvm.b_layout": "{semantics.b_layout}", '
        f'"triton_tvm.c_layout": "{semantics.c_layout}", '
        f'"triton_tvm.a_stride": "{_stride_text(semantics.a_stride)}", '
        f'"triton_tvm.b_stride": "{_stride_text(semantics.b_stride)}", '
        f'"triton_tvm.c_stride": "{_stride_text(semantics.c_stride)}", '
        f'"triton_tvm.b_storage_shape": "{_stride_text(b_storage_shape)}", '
        f'"triton_tvm.b_storage_stride": "{_stride_text(b_storage_stride)}", '
        f'"triton_tvm.transposed_b": {_bool_literal(transposed_b)}'
        "})",
        "        "
        f'{a_name} = T.match_buffer({a_name}_handle, ({semantics.m}, {semantics.k}), '
        f'"{semantics.a_dtype}", strides=({_stride_text(semantics.a_stride)}))',
        "        "
        f'{b_name} = T.match_buffer({b_name}_handle, '
        f'({b_storage_shape[0]}, {b_storage_shape[1]}), '
        f'"{semantics.b_dtype}", strides=({_stride_text(b_storage_stride)}))',
        "        "
        f'{c_name} = T.match_buffer({c_name}_handle, ({semantics.m}, {semantics.n}), '
        f'"{semantics.output_dtype}", strides=({_stride_text(semantics.c_stride)}))',
        "        "
        f'T.evaluate(T.call_packed("{EXTERN_GEMM_PACKED_FUNC}", '
        f"{a_name}, {b_name}, {c_name}, {semantics.m}, {semantics.n}, "
        f"{semantics.k}, {b_transposed_expr}))",
    ]
    return "\n".join(lines) + "\n"


def build_extern_addmm_bias_tirx_source(
    semantics: MatmulSemantics,
    decision: TargetMatmulDecision,
    *,
    target_attrs: str = "cuda",
) -> str:
    """Build the explicit M9.PE packed-call artifact for wrapper addmm+bias."""

    if semantics.source_kind != "wrapper_extern_addmm_bias":
        raise UnsupportedTTIROpError(
            "extern addmm bias artifact requires wrapper_extern_addmm_bias semantics"
        )
    if (
        decision.implementation_kind != "extern_addmm_bias"
        or decision.extern_symbol != EXTERN_ADDMM_BIAS_SYMBOL
    ):
        raise UnsupportedTTIROpError(
            "extern addmm bias artifact requires extern_addmm_bias policy decision"
        )
    if semantics.bias_shape not in ((semantics.n,), (semantics.m, semantics.n)):
        raise UnsupportedTTIROpError("extern addmm bias artifact requires rank-1 N or rank-2 MxN bias")

    func_name = _sanitize_identifier(semantics.kernel_name or "wrapper_extern_addmm_bias")
    bias_name = _sanitize_identifier(semantics.bias_param or "bias")
    a_name = _sanitize_identifier(semantics.a_param or "a")
    b_name = _sanitize_identifier(semantics.b_param or "b")
    c_name = _sanitize_identifier(semantics.c_param or "out")
    transposed_b = semantics.b_layout == "transposed_weight_view"
    b_storage_shape = (semantics.n, semantics.k) if transposed_b else (semantics.k, semantics.n)
    b_storage_stride = (semantics.k, 1) if transposed_b else semantics.b_stride
    b_transposed_expr = "T.bool(True)" if transposed_b else "T.bool(False)"
    bias_rank = len(semantics.bias_shape)
    bias_shape_text = _tuple_text(semantics.bias_shape)
    bias_stride_text = _tuple_text(semantics.bias_stride)
    lines: list[str] = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        f"    def {func_name}({bias_name}_handle: T.handle, {a_name}_handle: T.handle, "
        f"{b_name}_handle: T.handle, {c_name}_handle: T.handle):",
        "        T.func_attr({"
        f'"global_symbol": "{semantics.kernel_name}", '
        '"tirx.noalias": True, '
        f'"target": T.target({target_attrs!r}), '
        '"triton_tvm.contract": "matmul_minimal", '
        f'"triton_tvm.matmul_source_kind": "{semantics.source_kind}", '
        f'"triton_tvm.a_dtype": "{semantics.a_dtype}", '
        f'"triton_tvm.b_dtype": "{semantics.b_dtype}", '
        f'"triton_tvm.output_dtype": "{semantics.output_dtype}", '
        f'"triton_tvm.accumulator_dtype": "{semantics.accumulator_dtype}", '
        f'"triton_tvm.input_precision": "{semantics.input_precision}", '
        f'"triton_tvm.tf32_policy": "{semantics.tf32_policy}", '
        f'"triton_tvm.bounds_policy": "{semantics.bounds_policy}", '
        f'"triton_tvm.mask_kind": "{semantics.mask_kind}", '
        f'"triton_tvm.epilogue_kind": "{semantics.epilogue_kind}", '
        f'"triton_tvm.alias_policy": "{semantics.alias_policy}", '
        f'"triton_tvm.implementation_kind": "{decision.implementation_kind}", '
        f'"triton_tvm.extern_symbol": "{decision.extern_symbol}", '
        f'"triton_tvm.extern_packed_func": "{decision.extern_packed_func}", '
        f'"triton_tvm.extern_runtime_kind": "{decision.extern_runtime_kind}", '
        f'"triton_tvm.extern_runtime_replacement": "{decision.extern_runtime_replacement}", '
        f'"triton_tvm.extern_runtime_replacement_available": '
        f'{_bool_literal(decision.extern_runtime_replacement_available)}, '
        f'"triton_tvm.extern_runtime_replacement_reason": '
        f'"{decision.extern_runtime_replacement_reason}", '
        f'"triton_tvm.extern_gemm_runtime_status": "{decision.extern_gemm_runtime_status}", '
        f'"triton_tvm.extern_gemm_provider_kind": "{decision.extern_gemm_provider_kind}", '
        f'"triton_tvm.extern_gemm_provider_abi_version": '
        f'{decision.extern_gemm_provider_abi_version}, '
        f'"triton_tvm.extern_gemm_runtime_claim": "{decision.extern_gemm_runtime_claim}", '
        f'"triton_tvm.extern_gemm_performance_claim": '
        f'{_bool_literal(decision.extern_gemm_performance_claim)}, '
        f'"triton_tvm.extern_gemm_uses_host_staging": '
        f'{_bool_literal(decision.extern_gemm_uses_host_staging)}, '
        f'"triton_tvm.matmul_m": {semantics.m}, '
        f'"triton_tvm.matmul_n": {semantics.n}, '
        f'"triton_tvm.matmul_k": {semantics.k}, '
        f'"triton_tvm.a_layout": "{semantics.a_layout}", '
        f'"triton_tvm.b_layout": "{semantics.b_layout}", '
        f'"triton_tvm.c_layout": "{semantics.c_layout}", '
        f'"triton_tvm.a_stride": "{_stride_text(semantics.a_stride)}", '
        f'"triton_tvm.b_stride": "{_stride_text(semantics.b_stride)}", '
        f'"triton_tvm.c_stride": "{_stride_text(semantics.c_stride)}", '
        f'"triton_tvm.b_storage_shape": "{_tuple_text(b_storage_shape)}", '
        f'"triton_tvm.b_storage_stride": "{_stride_text(b_storage_stride)}", '
        f'"triton_tvm.transposed_b": {_bool_literal(transposed_b)}, '
        f'"triton_tvm.bias_param": "{semantics.bias_param}", '
        f'"triton_tvm.bias_shape": "{bias_shape_text}", '
        f'"triton_tvm.bias_stride": "{bias_stride_text}", '
        f'"triton_tvm.bias_rank": {bias_rank}, '
        f'"triton_tvm.alpha": "{semantics.alpha:g}", '
        f'"triton_tvm.beta": "{semantics.beta:g}"'
        "})",
        "        "
        f'{bias_name} = T.match_buffer({bias_name}_handle, '
        f'({_shape_literal(semantics.bias_shape)}), "float32", '
        f"strides=({_shape_literal(semantics.bias_stride)}))",
        "        "
        f'{a_name} = T.match_buffer({a_name}_handle, ({semantics.m}, {semantics.k}), '
        f'"{semantics.a_dtype}", strides=({_stride_text(semantics.a_stride)}))',
        "        "
        f'{b_name} = T.match_buffer({b_name}_handle, '
        f'({b_storage_shape[0]}, {b_storage_shape[1]}), '
        f'"{semantics.b_dtype}", strides=({_stride_text(b_storage_stride)}))',
        "        "
        f'{c_name} = T.match_buffer({c_name}_handle, ({semantics.m}, {semantics.n}), '
        f'"{semantics.output_dtype}", strides=({_stride_text(semantics.c_stride)}))',
        "        "
        f'T.evaluate(T.call_packed("{EXTERN_ADDMM_BIAS_PACKED_FUNC}", '
        f"{bias_name}, {a_name}, {b_name}, {c_name}, {semantics.m}, {semantics.n}, "
        f"{semantics.k}, {b_transposed_expr}, T.int64({bias_rank})))",
    ]
    return "\n".join(lines) + "\n"


class _ExternGemmProviderRegistration:
    """Restorable packed-func registration for the Python/Torch runtime proof."""

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

    def __enter__(self) -> "_ExternGemmProviderRegistration":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # pylint: disable=unused-argument
        self.close()


def register_python_torch_extern_gemm() -> _ExternGemmProviderRegistration:
    """Register the opt-in correctness-only host-staged extern GEMM provider.

    This provider is an M9.6 runtime proof.  It stages inputs through host
    NumPy arrays, computes with PyTorch on host tensors, and copies the result
    back to the TVM output tensor.  It intentionally makes no performance claim.
    """

    import tvm  # pylint: disable=import-outside-toplevel

    previous = tvm.get_global_func(EXTERN_GEMM_PACKED_FUNC, allow_missing=True)

    def _extern_gemm(a, b_storage, c, m, n, k, transposed_b):
        import torch  # pylint: disable=import-outside-toplevel

        _validate_python_torch_provider_tensor("A", a, "float32", (int(m), int(k)))
        expected_b_shape = (int(n), int(k)) if bool(transposed_b) else (int(k), int(n))
        _validate_python_torch_provider_tensor("B", b_storage, "float32", expected_b_shape)
        _validate_python_torch_provider_tensor("C", c, "float32", (int(m), int(n)))

        a_host = torch.from_numpy(a.numpy())
        b_host = torch.from_numpy(b_storage.numpy())
        logical_b = b_host.transpose(0, 1) if bool(transposed_b) else b_host
        result = torch.matmul(a_host, logical_b)
        c.copyfrom(result.numpy())

    tvm.register_global_func(EXTERN_GEMM_PACKED_FUNC, _extern_gemm, override=True)
    return _ExternGemmProviderRegistration(EXTERN_GEMM_PACKED_FUNC, previous)


def register_python_torch_extern_addmm_bias() -> _ExternGemmProviderRegistration:
    """Register the opt-in correctness-only host-staged addmm+bias provider."""

    import tvm  # pylint: disable=import-outside-toplevel

    previous = tvm.get_global_func(EXTERN_ADDMM_BIAS_PACKED_FUNC, allow_missing=True)

    def _extern_addmm_bias(bias, a, b_storage, c, m, n, k, transposed_b, bias_rank):
        import torch  # pylint: disable=import-outside-toplevel

        m = int(m)
        n = int(n)
        k = int(k)
        bias_rank = int(bias_rank)
        _validate_python_torch_provider_tensor("A", a, "float32", (m, k))
        expected_b_shape = (n, k) if bool(transposed_b) else (k, n)
        _validate_python_torch_provider_tensor("B", b_storage, "float32", expected_b_shape)
        _validate_python_torch_provider_tensor("C", c, "float32", (m, n))
        expected_bias_shape = (n,) if bias_rank == 1 else (m, n)
        _validate_python_torch_provider_tensor("bias", bias, "float32", expected_bias_shape)

        bias_host = torch.from_numpy(bias.numpy())
        a_host = torch.from_numpy(a.numpy())
        b_host = torch.from_numpy(b_storage.numpy())
        logical_b = b_host.transpose(0, 1) if bool(transposed_b) else b_host
        result = torch.matmul(a_host, logical_b) + bias_host
        c.copyfrom(result.numpy())

    tvm.register_global_func(EXTERN_ADDMM_BIAS_PACKED_FUNC, _extern_addmm_bias, override=True)
    return _ExternGemmProviderRegistration(EXTERN_ADDMM_BIAS_PACKED_FUNC, previous)


def _validate_python_torch_provider_tensor(
    role: str,
    tensor: Any,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    if not hasattr(tensor, "dtype") or not hasattr(tensor, "shape"):
        raise TypeError(f"extern_gemm {role} must be a TVM tensor")
    if str(tensor.dtype) != dtype:
        raise TypeError(
            f"python_torch_host_staged extern_gemm supports {dtype} {role}, "
            f"got {tensor.dtype}"
        )
    actual_shape = tuple(int(dim) for dim in tensor.shape)
    if actual_shape != shape:
        raise ValueError(f"extern_gemm {role} shape must be {shape}, got {actual_shape}")


def _native_matmul_schedule_id(semantics: MatmulSemantics) -> str:
    if _tensorcore_matmul_reject_reason(semantics) == "":
        return TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    if _simt_matmul_reject_reason(semantics) == "":
        return SIMT_TIR_MATMUL_SCHEDULE_ID
    if (
        semantics.m % TILED_TIR_MATMUL_TILE_M == 0
        and semantics.n % TILED_TIR_MATMUL_TILE_N == 0
        and semantics.source_kind == "tt_dot"
        and semantics.a_layout == "row_major"
        and semantics.b_layout == "row_major"
        and semantics.c_layout == "row_major"
        and semantics.a_dtype in ("float16", "bfloat16")
        and semantics.b_dtype in ("float16", "bfloat16")
    ):
        return TILED_TIR_MATMUL_SCHEDULE_ID
    return NATIVE_TIR_MATMUL_SCHEDULE_ID


def _native_matmul_decision_metadata(
    semantics: MatmulSemantics,
    selected_schedule_id: str,
) -> dict[str, Any]:
    reject_reasons: dict[str, str] = {}
    tensorcore_reject_reason = _tensorcore_matmul_reject_reason(semantics)
    if selected_schedule_id != TENSORCORE_TIR_MATMUL_SCHEDULE_ID:
        reject_reasons[TENSORCORE_TIR_MATMUL_SCHEDULE_ID] = tensorcore_reject_reason or (
            "tensorcore_candidate_not_selected"
        )
    if selected_schedule_id != SIMT_TIR_MATMUL_SCHEDULE_ID:
        simt_reject_reason = _simt_matmul_reject_reason(semantics)
        reject_reasons[SIMT_TIR_MATMUL_SCHEDULE_ID] = simt_reject_reason or (
            "lower_priority_candidate_not_selected"
        )
    if selected_schedule_id != TILED_TIR_MATMUL_SCHEDULE_ID:
        tiled_reject_reason = _tiled_matmul_reject_reason(semantics)
        reject_reasons[TILED_TIR_MATMUL_SCHEDULE_ID] = tiled_reject_reason or (
            "lower_priority_candidate_not_selected"
        )
    if selected_schedule_id != NATIVE_TIR_MATMUL_SCHEDULE_ID:
        reject_reasons[NATIVE_TIR_MATMUL_SCHEDULE_ID] = "fallback_candidate_not_selected"

    return {
        "matmul_perf_envelope": MATMUL_PERF_ENVELOPE_ID,
        "candidate_schedule_ids": matmul_schedule_candidate_ids(),
        "selected_schedule_id": selected_schedule_id,
        "schedule_reject_reasons": reject_reasons,
        "perf_guard_status": MATMUL_PERF_STATUS_NOT_MEASURED,
        "tune_key": build_matmul_tune_key_payload(
            semantics,
            schedule_id=selected_schedule_id,
            tune_params=_schedule_tune_params(selected_schedule_id),
        ),
    }


def build_matmul_tune_key_payload(
    semantics: MatmulSemantics,
    *,
    schedule_id: str,
    tune_params: dict[str, Any] | None = None,
    target_arch: str = "",
    device_name: str = "",
    framework_versions: dict[str, str] | None = None,
    cuda_target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen M9.P cache/tune key payload."""
    return {
        "version": MATMUL_TUNE_KEY_VERSION,
        "registry_version": MATMUL_SCHEDULE_REGISTRY_VERSION,
        "matmul_perf_envelope": MATMUL_PERF_ENVELOPE_ID,
        "m": semantics.m,
        "n": semantics.n,
        "k": semantics.k,
        "batch_dims": list(semantics.batch_dims),
        "a_dtype": semantics.a_dtype,
        "b_dtype": semantics.b_dtype,
        "accumulator_dtype": semantics.accumulator_dtype,
        "output_dtype": semantics.output_dtype,
        "a_layout": semantics.a_layout,
        "b_layout": semantics.b_layout,
        "c_layout": semantics.c_layout,
        "a_stride": list(semantics.a_stride),
        "b_stride": list(semantics.b_stride),
        "c_stride": list(semantics.c_stride),
        "transpose_a": semantics.transpose_a,
        "transpose_b": semantics.transpose_b,
        "epilogue_kind": semantics.epilogue_kind,
        "target_arch": target_arch,
        "device_name": device_name,
        "schedule_id": schedule_id,
        "tune_params": dict(tune_params or {}),
        "framework_versions": dict(framework_versions or {}),
        "cuda_target_metadata": dict(cuda_target_metadata or {}),
    }


def _schedule_tune_params(schedule_id: str) -> dict[str, Any]:
    for candidate in _MATMUL_SCHEDULE_REGISTRY:
        if candidate.schedule_id == schedule_id:
            return dict(candidate.tunable_params)
    return {}


def _tensorcore_matmul_reject_reason(semantics: MatmulSemantics) -> str:
    envelope_reason = _matmul_perf_core_reject_reason(semantics)
    if envelope_reason:
        return envelope_reason
    if semantics.a_dtype != "float16" or semantics.b_dtype != "float16":
        return "tensorcore_candidate_requires_fp16_inputs"
    return ""


def _simt_matmul_reject_reason(semantics: MatmulSemantics) -> str:
    envelope_reason = _matmul_perf_core_reject_reason(semantics)
    if envelope_reason:
        return envelope_reason
    if _tensorcore_matmul_reject_reason(semantics) == "":
        return "simt_candidate_not_selected_tensorcore_available"
    if semantics.m % SIMT_TIR_MATMUL_TILE_M or semantics.n % SIMT_TIR_MATMUL_TILE_N:
        return "simt_candidate_requires_mn_multiples_of_16"
    return ""


def _tiled_matmul_reject_reason(semantics: MatmulSemantics) -> str:
    if semantics.source_kind != "tt_dot":
        return "tiled_candidate_requires_tt_dot_source"
    if semantics.m % TILED_TIR_MATMUL_TILE_M or semantics.n % TILED_TIR_MATMUL_TILE_N:
        return "tiled_candidate_requires_mn_multiples_of_8"
    if semantics.a_layout != "row_major" or semantics.b_layout != "row_major":
        return "tiled_candidate_requires_row_major_ab"
    if semantics.c_layout != "row_major":
        return "tiled_candidate_requires_row_major_c"
    if semantics.a_dtype not in ("float16", "bfloat16") or semantics.b_dtype not in (
        "float16",
        "bfloat16",
    ):
        return "tiled_candidate_requires_fp16_or_bf16_inputs"
    return ""


def _matmul_perf_core_reject_reason(semantics: MatmulSemantics) -> str:
    if semantics.source_kind != "tt_dot":
        return "matmul_perf_core_requires_tt_dot_native_source"
    if semantics.batch_dims:
        return "matmul_perf_core_rejects_batch_dims"
    if semantics.transpose_a:
        return "matmul_perf_core_rejects_transpose_a"
    if semantics.transpose_b:
        return "matmul_perf_core_rejects_transpose_b"
    if semantics.bounds_policy != "exact":
        return "matmul_perf_core_requires_exact_bounds"
    if semantics.mask_kind != "none":
        return "matmul_perf_core_rejects_masks"
    if semantics.epilogue_kind != "none":
        return "matmul_perf_core_rejects_epilogue"
    if semantics.accumulator_dtype != "float32" or semantics.output_dtype != "float32":
        return "matmul_perf_core_requires_fp32_accumulator_output"
    if semantics.a_dtype not in ("float16", "bfloat16") or semantics.b_dtype not in (
        "float16",
        "bfloat16",
    ):
        return "matmul_perf_core_requires_fp16_or_bf16_inputs"
    if semantics.a_layout != "row_major" or semantics.c_layout != "row_major":
        return "matmul_perf_core_requires_row_major_a_c"
    if semantics.b_layout not in ("row_major", "transposed_weight_view"):
        return "matmul_perf_core_requires_row_major_or_transposed_weight_b"
    if semantics.m % 16 or semantics.n % 16 or semantics.k % 16:
        return "matmul_perf_core_requires_mnk_multiples_of_16"
    return ""


def _value_type(defs: dict[str, TTIROp], value: str) -> TTIRType:
    op = defs.get(value)
    if op is None or not op.result_types:
        raise UnsupportedTTIROpError(f"Cannot infer matmul_minimal type for %{value}")
    return op.result_types[0]


def _load_op(defs: dict[str, TTIROp], value: str, role: str) -> TTIROp:
    op = defs.get(value)
    if op is None or op.name != "tt.load":
        raise UnsupportedTTIROpError(f"matmul_minimal {role} operand must come from tt.load")
    return op


def _base_pointer_param(graph: NormalizedTTIROpGraph, value: str) -> str | None:
    params = graph.param_by_name()
    defs = graph.op_by_result()
    if value in params and params[value].type.is_pointer:
        return value
    op = defs.get(value)
    if op is None or not op.operands:
        return None
    if op.name in ("tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims"):
        return _base_pointer_param(graph, op.operands[0])
    return None


def _unquote_attr(value: Any) -> str:
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


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
    source = source.strip()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.parse(f"_tmp = {source}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _attribute_chain_ast(node.func) == op_name:
            return node
    return None


def _parse_reinterpret_tensor(node: ast.AST, role: str) -> dict[str, Any]:
    if not isinstance(node, ast.Call) or _attribute_chain_ast(node.func) != "reinterpret_tensor":
        raise UnsupportedTTIROpError(
            f"wrapper_extern_gemm {role} must be reinterpret_tensor(...)"
        )
    if len(node.args) < 4:
        raise UnsupportedTTIROpError(
            f"wrapper_extern_gemm {role} reinterpret_tensor must include base, shape, stride, offset"
        )
    shape = _int_tuple(node.args[1], f"{role} shape")
    stride = _int_tuple(node.args[2], f"{role} stride")
    if len(shape) != 2 or len(stride) != 2:
        raise UnsupportedTTIROpError(
            f"wrapper_extern_gemm {role} reinterpret_tensor must be rank-2"
        )
    offset = _literal_int(node.args[3], f"{role} offset")
    if offset != 0:
        raise UnsupportedTTIROpError(
            f"wrapper_extern_gemm {role} reinterpret_tensor offset must be 0"
        )
    return {
        "base": _expr_name(node.args[0]),
        "shape": shape,
        "stride": stride,
    }


def _parse_addmm_bias(node: ast.AST, *, m: int, n: int) -> dict[str, Any]:
    if isinstance(node, ast.Call) and _attribute_chain_ast(node.func) == "reinterpret_tensor":
        if len(node.args) < 4:
            raise UnsupportedTTIROpError(
                "wrapper_extern_addmm_bias bias reinterpret_tensor must include base, "
                "shape, stride, offset"
            )
        shape = _int_tuple(node.args[1], "bias shape")
        stride = _int_tuple(node.args[2], "bias stride")
        offset = _literal_int(node.args[3], "bias offset")
        if offset != 0:
            raise UnsupportedTTIROpError(
                "wrapper_extern_addmm_bias bias reinterpret_tensor offset must be 0"
            )
        if shape == (n,):
            expected_stride = (1,)
        elif shape == (m, n):
            expected_stride = (n, 1)
        else:
            raise UnsupportedTTIROpError(
                "wrapper_extern_addmm_bias bias must be rank-1 N or rank-2 MxN"
            )
        if stride != expected_stride:
            raise UnsupportedTTIROpError(
                "wrapper_extern_addmm_bias bias stride must match supported bias shape"
            )
        return {
            "base": _expr_name(node.args[0]),
            "shape": shape,
            "stride": stride,
        }
    return {
        "base": _expr_name(node),
        "shape": (n,),
        "stride": (1,),
    }


def _parse_optional_float_keyword(call: ast.Call, name: str, default: float) -> float:
    for keyword_node in call.keywords:
        if keyword_node.arg == name:
            return _literal_float(keyword_node.value, name)
    return default


def _parse_out_param(call: ast.Call) -> str:
    for keyword_node in call.keywords:
        if keyword_node.arg == "out":
            if isinstance(keyword_node.value, ast.Constant) and keyword_node.value.value is None:
                raise UnsupportedTTIROpError("wrapper_extern_gemm requires out= buffer")
            return _expr_name(keyword_node.value)
    raise UnsupportedTTIROpError("wrapper_extern_gemm requires out= buffer")


def _int_tuple(node: ast.AST, label: str) -> tuple[int, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise UnsupportedTTIROpError(f"wrapper_extern_gemm {label} must be a static tuple")
    return tuple(_literal_int(elt, label) for elt in node.elts)


def _literal_int(node: ast.AST, label: str) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_int(node.operand, label)
    raise UnsupportedTTIROpError(f"wrapper_extern_gemm {label} must be static integers")


def _literal_float(node: ast.AST, label: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_float(node.operand, label)
    raise UnsupportedTTIROpError(f"wrapper_extern_addmm_bias {label} must be a static number")


def _expr_name(node: ast.AST) -> str:
    name = _attribute_chain_ast(node)
    if name:
        return name
    raise UnsupportedTTIROpError("wrapper_extern_gemm buffers must be named values")


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


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"\W", "_", name)
    if not sanitized or sanitized[0].isdigit() or keyword.iskeyword(sanitized):
        sanitized = f"_{sanitized}"
    return sanitized


def _stride_text(stride: tuple[int, int]) -> str:
    return _tuple_text(stride)


def _tuple_text(values: tuple[int, ...]) -> str:
    return ", ".join(str(int(value)) for value in values)


def _shape_literal(values: tuple[int, ...]) -> str:
    text = _tuple_text(values)
    if len(values) == 1:
        return text + ","
    return text


def _bool_literal(value: bool) -> str:
    return "True" if bool(value) else "False"
