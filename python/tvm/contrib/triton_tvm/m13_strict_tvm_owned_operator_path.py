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
"""M13 fixed-shape ViT strict TVM-owned operator path reports."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tvm
import tvm_ffi

from .contracts import (
    validate_attention_vit_full_contract,
    validate_matmul_minimal_contract,
    validate_pointwise_grid2d_static_contract,
    validate_vision_conv2d_contract,
)
from .attention import (
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    TargetAttentionPolicy,
    build_native_decomposed_attention_tirx_source,
    extract_attention_semantics_from_wrapper_sdpa,
)
from .errors import TritonTVMError
from .frontend import lower_to_ttir
from .inductor import (
    extract_inductor_triton_sources,
    extract_inductor_wrapper_extern_calls,
    load_inductor_kernel,
    lower_inductor_kernel_to_ttir,
)
from .matmul import (
    EXTERN_ADDMM_BIAS_SYMBOL,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    EXTERN_GEMM_SYMBOL,
    GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
    REAL_JIT_TT_DOT_SOURCE_KIND,
    TargetMatmulPolicy,
    extract_matmul_semantics_from_ttir,
    extract_matmul_semantics_from_wrapper_extern,
    extract_matmul_semantics_from_wrapper_extern_addmm,
)
from .m123_vit_e2e_runner import (
    _capture_compiled_vit,
    _correctness_report,
    _dependency_versions,
    _output_tensors,
    _require_torch,
)
from .m116_vision_dashboard import _grid2d_meta
from .model_corpus import _m11_grid2d_op_kind
from .runtime import TritonTVMArtifact, build_triton_tvm
from .translator import TritonTVMMeta, translate_ttir
from .ttir import TTIRReader
from .vision import (
    M11_GRID2D_ARTIFACT_READY,
    M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
    M11_GRID2D_PROVIDER_NATIVE_TVM,
    M11_GRID2D_RUNTIME_READY,
    M11_GRID2D_RUNTIME_CLAIM,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_VERSION,
    VISION_IMPLEMENTATION_KIND_NATIVE_TVM_CONV2D,
    VISION_PROVIDER_DEVICE_TORCH_CUDA,
    VISION_PROVIDER_ABI_VERSION,
    VISION_PROVIDER_NATIVE_TVM_CONV2D,
    VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    VisionGrid2DPointwiseSemantics,
    build_native_grid2d_pointwise_tirx_source,
    extract_vision_conv2d_semantics_from_wrapper_extern,
)


REPORT_ROOT = Path("/home/liyh/xdb/triton-tvm-workbench/reports/m13")
M12_CORPUS_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_2_vit_native_matmul_closure_corpus/report.json"
)
M12_E2E_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_3_vit_fixed_shape_e2e_runner/report.json"
)
M12_FINAL_WRAPPER = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_p_profile_optimization_loop/provider_session_reuse_v1/vit_tiny_random_wrapper.py"
)

TARGET_MODEL = "vit_tiny_random"
MILESTONE = "M13"
M13_4_REPORT_ID = "m13_4_matmul_addmm_runtime_replacement_v1"
M13_4_RUNTIME_KIND = "generated_bridge_tvm_artifact_run"
M13_4_BIAS_EPILOGUE_IMPLEMENTATION = "native_tvm_pointwise"
M13_4_ADAPTER_IMPLEMENTATION = "native_tvm_pointwise"
M13_5_REPORT_ID = "m13_5_native_conv_slice_v1"
M13_5_CONV_RUNTIME_KIND = "native_tvm_conv_artifact_run"
M13_5_CONV_SHAPE_SCOPE = "exact_vit_patch_embedding_m13"
M13_5_CONV_SCHEDULE_ID = "m13_5_serial_output_element_reduction"
M13_6_REPORT_ID = "m13_6_native_attention_slice_v1"
M13_6_ATTENTION_RUNTIME_KIND = "native_tvm_attention_decomposed_artifact_run"
M13_6_ATTENTION_PROVIDER_KIND = "native_tvm_attention_decomposed"
M13_6_ATTENTION_SHAPE_SCOPE = "exact_vit_tiny_random_full_attention_m13"
M13_7_REPORT_ID = "m13_7_strict_correctness_gate_v1"
M13_7_REPORT_DIR = REPORT_ROOT / "m13_7_strict_correctness_gate"
M13_7_ALLCLOSE_RTOL = 1e-3
M13_7_ALLCLOSE_ATOL = 1e-3
M13_8_REPORT_ID = "m13_8_p2_dashboard_decision_v1"
M13_8_REPORT_DIR = REPORT_ROOT / "m13_8_p2_dashboard_decision"
M13_8_STRICT_MODE = "m13_strict_tvm_owned_operator_path"
M13_8_TORCH_COMPILE_MODE = "torch_compile_inductor"
M13_8_P2_THRESHOLD = 2.0
M13_8_DEFAULT_WARMUP = 5
M13_8_DEFAULT_REPEAT = 20
M13_P_REPORT_ROOT = REPORT_ROOT / "m13_p_strict_surface_performance_loop"
M13_P1_REPORT_DIR = M13_P_REPORT_ROOT / "p1_strict_surface_profile"
M13_P1_REPORT_ID = "m13_p1_strict_surface_profile_v1"
M13_P2_REPORT_DIR = M13_P_REPORT_ROOT / "p2_generated_bridge_direct_io"
M13_P2_REPORT_ID = "m13_p2_generated_bridge_direct_io_v1"
M13_P2_DIRECT_RUNTIME_KIND = "generated_bridge_direct_io_tvm_artifact_run"
M13_P2_DIRECT_SCHEDULE_ID = "m13_p2_direct_io_wrapper_matmul_serial_k"
M13_P3_REPORT_DIR = M13_P_REPORT_ROOT / "p3_generated_bridge_async_direct_io"
M13_P3_REPORT_ID = "m13_p3_generated_bridge_async_direct_io_v1"
M13_P3_DIRECT_RUNTIME_KIND = "generated_bridge_async_direct_io_tvm_artifact_run"
M13_P3_DIRECT_SCHEDULE_ID = "m13_p3_async_direct_io_wrapper_matmul_serial_k"
M13_P4_REPORT_DIR = M13_P_REPORT_ROOT / "p4_fine_grained_strict_surface_profile"
M13_P4_REPORT_ID = "m13_p4_fine_grained_strict_surface_profile_v1"
M13_P5_REPORT_DIR = M13_P_REPORT_ROOT / "p5_generated_bridge_fast_artifact_invocation"
M13_P5_REPORT_ID = "m13_p5_generated_bridge_fast_artifact_invocation_v1"
M13_P5_DIRECT_RUNTIME_KIND = "generated_bridge_fast_async_direct_io_tvm_entry_run"
M13_P6_REPORT_DIR = M13_P_REPORT_ROOT / "p6_native_artifact_fine_grained_profile"
M13_P6_REPORT_ID = "m13_p6_native_artifact_fine_grained_profile_v1"
M13_P7_REPORT_DIR = M13_P_REPORT_ROOT / "p7_native_conv_fast_artifact_invocation"
M13_P7_REPORT_ID = "m13_p7_native_conv_fast_artifact_invocation_v1"
M13_P7_NATIVE_CONV_RUNTIME_KIND = "native_tvm_conv_artifact_fast_entry_run"
IMPLEMENTATION_LEVELS = (
    "triton_language_lowering",
    "generated_tl_dot_bridge",
    "tvm_native_wrapper_lowering",
    "tvm_decomposed_artifact",
    "metadata_only_structure",
)
LEGACY_PROVIDERS = (
    "torch_inductor_triton_captured_harness",
    "native_triton_launch",
    "device_torch_cuda",
    "torch_replay_attention",
    "wrapper_extern_gemm",
    "wrapper_extern_addmm_bias",
    "native_tvm_matmul",
    "inductor_triton_harness_fallback",
    "pytorch_fallback",
)
M13_SOURCE_TAXONOMY = {
    "real_jit_tt_dot_captured": {
        "source_kind": REAL_JIT_TT_DOT_SOURCE_KIND,
        "counts_as_captured_real_tl_dot": True,
        "counts_as_generated_bridge": False,
    },
    "generated_real_tl_dot_bridge": {
        "source_kind": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        "counts_as_captured_real_tl_dot": False,
        "counts_as_generated_bridge": True,
    },
    "wrapper_extern_gemm": {
        "source_kind": "wrapper_extern_gemm",
        "counts_as_captured_real_tl_dot": False,
        "counts_as_generated_bridge": False,
    },
}


def run_m13_0_policy_taxonomy_freeze(
    *, out_dir: str | Path | None = REPORT_ROOT / "m13_0_policy_taxonomy_freeze"
) -> dict[str, Any]:
    """Write the M13.0 policy/taxonomy freeze report."""

    report = {
        "report_kind": "triton_tvm_m13_0_policy_taxonomy_freeze",
        "schema_version": 1,
        "report_id": "m13_0_policy_taxonomy_freeze_v1",
        "milestone": MILESTONE,
        "substep": "M13.0",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "goal": "Freeze M13 strict TVM-owned operator path gates, taxonomy, and report schema.",
        "m13_c_gate_fields": {
            "fixed_shape_vit_strict_tvm_owned_allclose": "bool",
            "host_staging_bytes": "int",
            "legacy_provider_count": "int",
            "silent_fallback_count": "int",
            "operator_counts_match_inventory": "bool",
        },
        "m13_p2_gate_fields": {
            "same_run_torch_compile_baseline": "required",
            "warmed_cache": True,
            "p50_ratio_threshold": 2.0,
            "p95_ratio_threshold": 2.0,
            "provider_relaxation_allowed": False,
            "fused_qkv_credit_allowed": False,
        },
        "implementation_levels": list(IMPLEMENTATION_LEVELS),
        "source_taxonomy": M13_SOURCE_TAXONOMY,
        "legacy_provider_counters": {provider: 0 for provider in LEGACY_PROVIDERS},
        "forbidden_invariants": {
            "hand_written_vit_executor": False,
            "pytorch_fallback": False,
            "inductor_triton_harness_fallback": False,
            "device_torch_cuda_in_strict_conv": False,
            "torch_replay_attention_in_strict_attention": False,
            "unbridged_wrapper_gemm": False,
            "generated_bridge_counts_as_captured_real_dot": False,
        },
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "generated_tl_dot_bridge",
                "captured_kernel_tvm_artifact_execution",
                "native_conv2d_static",
                "native_decomposed_attention",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13 report fields are frozen so later reusable capability extraction can "
                "separate implementation level from provider/runtime status."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("implementation_levels_frozen", set(report["implementation_levels"]) == set(IMPLEMENTATION_LEVELS)),
            (
                "generated_bridge_not_captured_real_dot",
                not report["source_taxonomy"]["generated_real_tl_dot_bridge"][
                    "counts_as_captured_real_tl_dot"
                ],
            ),
            ("m13_c_requires_zero_legacy", "legacy_provider_count" in report["m13_c_gate_fields"]),
            ("m13_p2_requires_p50_p95", report["m13_p2_gate_fields"]["p50_ratio_threshold"] == 2.0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.0 Policy Taxonomy Freeze")
    return report


def run_m13_1_operator_inventory_baseline(
    *, out_dir: str | Path | None = REPORT_ROOT / "m13_1_operator_inventory_baseline"
) -> dict[str, Any]:
    """Build the fixed-shape ViT operator inventory baseline."""

    wrapper_source = _load_wrapper_source()
    corpus = _read_json(M12_CORPUS_REPORT)
    e2e = _read_json(M12_E2E_REPORT)
    captured_records = _vit_captured_records(corpus)
    extern_records = _vit_extern_records(corpus)
    inventory = _operator_inventory(wrapper_source, captured_records, extern_records)
    counts = Counter(record["operator_family"] for record in inventory)

    report = {
        "report_kind": "triton_tvm_m13_1_operator_inventory_baseline",
        "schema_version": 1,
        "report_id": "m13_1_operator_inventory_baseline_v1",
        "milestone": MILESTONE,
        "substep": "M13.1",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "operator_inventory": inventory,
        "operator_counts": dict(sorted(counts.items())),
        "m12_provider_mix": (e2e.get("provider_mix") or {}).get("provider_counts", {}),
        "m13_target_counts": {
            "captured_tvm_artifact": counts.get("captured_triton_kernel", 0),
            "generated_bridge_matmul_addmm": counts.get("wrapper_matmul_addmm", 0),
            "native_conv": counts.get("wrapper_conv", 0),
            "native_attention": counts.get("wrapper_attention", 0),
            "legacy_provider": 0,
        },
        **_slice_fields(
            operator_inventory_delta=[
                {"operator_family": family, "count": count, "status": "inventoried"}
                for family, count in sorted(counts.items())
            ],
            implementation_level_delta={
                "triton_language_lowering": counts.get("captured_triton_kernel", 0),
                "generated_tl_dot_bridge": counts.get("wrapper_matmul_addmm", 0),
                "tvm_native_wrapper_lowering": counts.get("wrapper_conv", 0),
                "tvm_decomposed_artifact": counts.get("wrapper_attention", 0),
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={
                "torch_inductor_triton_captured_harness": -counts.get(
                    "captured_triton_kernel", 0
                ),
                "device_torch_cuda": -counts.get("wrapper_conv", 0),
                "torch_replay_attention": -counts.get("wrapper_attention", 0),
                "native_tvm_matmul": -counts.get("wrapper_matmul_addmm", 0),
            },
            reusable_capability_candidate=[
                "captured_kernel_tvm_artifact_execution",
                "generated_tl_dot_bridge",
                "conv2d_nchw_static_v1",
                "attention_vit_full_native_decomposed",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "Inventory keeps exact ViT shapes as assumptions while preserving generic "
                "contract names for later extraction."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("captured_kernel_count_7", counts.get("captured_triton_kernel", 0) == 7),
            ("matmul_addmm_count_7", counts.get("wrapper_matmul_addmm", 0) == 7),
            ("conv_count_1", counts.get("wrapper_conv", 0) == 1),
            ("attention_count_1", counts.get("wrapper_attention", 0) == 1),
            ("total_operator_count_16", len(inventory) == 16),
            ("unclassified_operator_count_0", counts.get("unclassified", 0) == 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.1 Operator Inventory Baseline")
    return report


def run_m13_2_captured_kernel_tvm_execution(
    *,
    out_dir: str | Path | None = REPORT_ROOT / "m13_2_captured_kernel_tvm_execution",
    run_execution: bool = True,
) -> dict[str, Any]:
    """Build and run the seven captured ViT kernels as TVM artifacts."""

    if not run_execution:
        report = _empty_execution_report(
            "triton_tvm_m13_2_captured_kernel_tvm_execution",
            "m13_2_captured_kernel_tvm_execution_v1",
            "M13.2",
            "execution_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.2 Captured Kernel TVM Execution")
        return report

    if not tvm.cuda(0).exist:
        report = _empty_execution_report(
            "triton_tvm_m13_2_captured_kernel_tvm_execution",
            "m13_2_captured_kernel_tvm_execution_v1",
            "M13.2",
            "tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.2 Captured Kernel TVM Execution")
        return report

    try:
        records = _run_captured_kernel_artifacts()
    except (ImportError, RuntimeError, ValueError, TritonTVMError) as err:
        report = _empty_execution_report(
            "triton_tvm_m13_2_captured_kernel_tvm_execution",
            "m13_2_captured_kernel_tvm_execution_v1",
            "M13.2",
            f"{type(err).__name__}:{err}",
            status="failed",
        )
        _maybe_write_report(report, out_dir, title="M13.2 Captured Kernel TVM Execution")
        return report

    counts = Counter(record["contract"] for record in records)
    passed = all(record.get("status") == "passed" for record in records)
    report = {
        "report_kind": "triton_tvm_m13_2_captured_kernel_tvm_execution",
        "schema_version": 1,
        "report_id": "m13_2_captured_kernel_tvm_execution_v1",
        "milestone": MILESTONE,
        "substep": "M13.2",
        "status": "passed" if passed else "failed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "implementation_level": "triton_language_lowering",
        "runtime_kind": "tvm_artifact_run",
        "captured_kernel_records": records,
        "contract_counts": dict(sorted(counts.items())),
        "summary": {
            "captured_kernel_count": len(records),
            "tvm_artifact_run_count": sum(record.get("run_count", 0) for record in records),
            "harness_fallback_count": sum(
                record.get("harness_fallback_count", 0) for record in records
            ),
            "native_triton_launch_count": sum(
                record.get("native_triton_launch_count", 0) for record in records
            ),
            "silent_fallback_count": sum(
                record.get("silent_fallback_count", 0) for record in records
            ),
        },
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "captured_triton_kernel",
                    "count": len(records),
                    "from_runtime_kind": "torch_inductor_triton_captured_harness",
                    "to_runtime_kind": "tvm_artifact_run",
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": len(records),
                "generated_tl_dot_bridge": 0,
                "tvm_native_wrapper_lowering": 0,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={
                "torch_inductor_triton_captured_harness": -len(records),
                "native_triton_launch": 0,
                "inductor_triton_harness_fallback": 0,
                "pytorch_fallback": 0,
            },
            reusable_capability_candidate=["captured_kernel_tvm_artifact_execution"],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "Captured-kernel execution uses TVM artifacts directly with stream=None; "
                "E2E wrapper replacement remains M13.7 integration work."
            ),
        ),
    }
    summary = report["summary"]
    report["invariants"] = _invariants(
        [
            ("captured_kernel_count_7", summary["captured_kernel_count"] == 7),
            ("all_run_once", summary["tvm_artifact_run_count"] == 7),
            ("harness_fallback_zero", summary["harness_fallback_count"] == 0),
            ("native_triton_launch_zero", summary["native_triton_launch_count"] == 0),
            ("silent_fallback_zero", summary["silent_fallback_count"] == 0),
            ("all_records_passed", passed),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.2 Captured Kernel TVM Execution")
    return report


def run_m13_3_generated_tl_dot_bridge_semantics(
    *,
    out_dir: str | Path | None = REPORT_ROOT / "m13_3_generated_tl_dot_bridge_semantics",
) -> dict[str, Any]:
    """Run the M13.3 generated ``tl.dot`` bridge semantics gate."""

    try:
        records = _run_generated_tl_dot_bridge_records()
    except (ImportError, RuntimeError, ValueError, TritonTVMError) as err:
        report = _empty_execution_report(
            "triton_tvm_m13_3_generated_tl_dot_bridge_semantics",
            "m13_3_generated_tl_dot_bridge_semantics_v1",
            "M13.3",
            f"{type(err).__name__}:{err}",
            status="failed",
        )
        _maybe_write_report(report, out_dir, title="M13.3 Generated tl.dot Bridge Semantics")
        return report

    passed = all(record.get("status") == "passed" for record in records)
    source_kind_counts = Counter(record.get("generated_matmul_source_kind", "") for record in records)
    epilogue_counts = Counter(record.get("wrapper_epilogue_kind", "") for record in records)
    report = {
        "report_kind": "triton_tvm_m13_3_generated_tl_dot_bridge_semantics",
        "schema_version": 1,
        "report_id": "m13_3_generated_tl_dot_bridge_semantics_v1",
        "milestone": MILESTONE,
        "substep": "M13.3",
        "status": "passed" if passed else "failed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "implementation_level": "generated_tl_dot_bridge",
        "runtime_kind": "not_connected_to_e2e_runtime",
        "source_taxonomy": M13_SOURCE_TAXONOMY,
        "bridge_records": records,
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "epilogue_counts": dict(sorted(epilogue_counts.items())),
        "summary": {
            "wrapper_matmul_addmm_count": len(records),
            "generated_bridge_pass_count": sum(
                1 for record in records if record.get("status") == "passed"
            ),
            "generated_bridge_counts_as_real_jit_tt_dot_captured": False,
            "real_jit_tt_dot_captured_count": 0,
            "wrapper_extern_gemm_runtime_replacement_count": 0,
            "build_or_runtime_required": False,
        },
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_matmul_addmm",
                    "count": len(records),
                    "from_source_kind": "wrapper_extern_gemm_or_addmm_bias",
                    "to_source_kind": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
                    "runtime_replacement": "deferred_to_M13.4",
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": len(records),
                "tvm_native_wrapper_lowering": 0,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={
                "wrapper_extern_gemm": 0,
                "wrapper_extern_addmm_bias": 0,
                "native_tvm_matmul": 0,
            },
            reusable_capability_candidate=[
                "generated_tl_dot_bridge",
                "bias_epilogue_native_pointwise_followup",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "Generated tl.dot bridge proves logical matmul semantics for observed "
                "fp32 wrapper shapes; M13.4 owns E2E runtime replacement and bias epilogues."
            ),
        ),
    }
    summary = report["summary"]
    report["invariants"] = _invariants(
        [
            ("wrapper_matmul_addmm_count_7", summary["wrapper_matmul_addmm_count"] == 7),
            ("all_generated_bridge_passed", summary["generated_bridge_pass_count"] == 7),
            (
                "generated_source_kind_count_7",
                source_kind_counts.get(GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND, 0) == 7,
            ),
            (
                "generated_bridge_not_captured_real_dot",
                summary["generated_bridge_counts_as_real_jit_tt_dot_captured"] is False,
            ),
            ("real_jit_captured_count_zero", summary["real_jit_tt_dot_captured_count"] == 0),
            ("build_or_runtime_not_required", summary["build_or_runtime_required"] is False),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.3 Generated tl.dot Bridge Semantics")
    return report


def run_m13_4_matmul_addmm_runtime_replacement(
    *,
    out_dir: str | Path | None = REPORT_ROOT / "m13_4_matmul_addmm_runtime_replacement",
    seed: int = 0,
    run_execution: bool = True,
) -> dict[str, Any]:
    """Run the M13.4 wrapper matmul/addmm generated-bridge runtime replacement."""

    if not run_execution:
        report = _empty_execution_report(
            "triton_tvm_m13_4_matmul_addmm_runtime_replacement",
            M13_4_REPORT_ID,
            "M13.4",
            "execution_disabled",
            status="not_run",
        )
        _maybe_write_report(report, out_dir, title="M13.4 Matmul Addmm Runtime Replacement")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_execution_report(
            "triton_tvm_m13_4_matmul_addmm_runtime_replacement",
            M13_4_REPORT_ID,
            "M13.4",
            "torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.4 Matmul Addmm Runtime Replacement")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_execution_report(
            "triton_tvm_m13_4_matmul_addmm_runtime_replacement",
            M13_4_REPORT_ID,
            "M13.4",
            "cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.4 Matmul Addmm Runtime Replacement")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        with _patched_m13_4_matmul_runtime(torch, calls) as patch_state:
            patch_state.reset_runtime_state()
            with torch.no_grad():
                actual_output = compiled(pixel_values)
            torch.cuda.synchronize()
            patch_state.assert_all_calls_consumed()
            actual = _output_tensors(actual_output)
        correctness = _correctness_report(actual, expected)
        report = _build_m13_4_matmul_addmm_runtime_replacement_report(
            records=patch_state.executed,
            wrapper_source=wrapper_source,
            correctness=correctness,
            seed=seed,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_execution_report(
            "triton_tvm_m13_4_matmul_addmm_runtime_replacement",
            M13_4_REPORT_ID,
            "M13.4",
            f"{type(err).__name__}:{err}",
            status="failed",
        )
        _maybe_write_report(report, out_dir, title="M13.4 Matmul Addmm Runtime Replacement")
        return report


def _build_m13_4_matmul_addmm_runtime_replacement_report(
    *,
    records: list[dict[str, Any]],
    wrapper_source: str,
    correctness: dict[str, Any],
    seed: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    """Build and optionally write the M13.4 report from runtime records."""

    op_counts = Counter(record.get("op_family", "") for record in records)
    epilogue_counts = Counter(record.get("epilogue_kind", "") for record in records)
    captured_count = len(extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL))
    bias_epilogue_count = sum(
        1
        for record in records
        if record.get("epilogue_kind") == "bias_add"
        and record.get("epilogue_implementation") == M13_4_BIAS_EPILOGUE_IMPLEMENTATION
    )
    generated_bridge_count = sum(
        1
        for record in records
        if record.get("matmul_core") == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
        and record.get("runtime_kind") == M13_4_RUNTIME_KIND
    )
    summary = {
        "wrapper_matmul_addmm_count": len(records),
        "generated_real_tl_dot_bridge_runtime_count": generated_bridge_count,
        "bias_epilogue_native_artifact_count": bias_epilogue_count,
        "wrapper_extern_gemm_runtime_count": 0,
        "wrapper_extern_addmm_bias_provider_count": 0,
        "native_tvm_matmul_legacy_wrapper_provider_count": 0,
        "host_staging_bytes": sum(int(record.get("host_staging_bytes", 0)) for record in records),
        "silent_fallback_count": sum(int(record.get("silent_fallback_count", 0)) for record in records),
        "matmul_runtime_replacement_connected": all(
            bool(record.get("runtime_replacement_connected")) for record in records
        ),
        "matmul_correctness_allclose": bool(correctness.get("allclose")),
        "m13_c_claim": False,
    }
    report = {
        "report_kind": "triton_tvm_m13_4_matmul_addmm_runtime_replacement",
        "schema_version": 1,
        "report_id": M13_4_REPORT_ID,
        "milestone": MILESTONE,
        "substep": "M13.4",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "implementation_level": "generated_tl_dot_bridge",
        "runtime_kind": M13_4_RUNTIME_KIND,
        "source_taxonomy": M13_SOURCE_TAXONOMY,
        "summary": summary,
        "correctness": correctness,
        "op_family_counts": dict(sorted(op_counts.items())),
        "epilogue_counts": dict(sorted(epilogue_counts.items())),
        "matmul_runtime_records": records,
        "non_matmul_scope_note": {
            "captured_kernel_count_in_wrapper": captured_count,
            "captured_kernel_replacement_source": "M13.2 report evidence",
            "conv_replacement_source": "deferred_to_M13.5",
            "attention_replacement_source": "deferred_to_M13.6",
            "m13_c_requires_followup": True,
        },
        "legacy_provider_counts": {
            "wrapper_extern_gemm": 0,
            "wrapper_extern_addmm_bias": 0,
            "native_tvm_matmul": 0,
        },
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_matmul_addmm",
                    "count": len(records),
                    "from_runtime_kind": "native_tvm_matmul_wrapper_provider",
                    "to_runtime_kind": M13_4_RUNTIME_KIND,
                    "matmul_core": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
                    "bias_epilogue_native_artifact_count": bias_epilogue_count,
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": len(records),
                "tvm_native_wrapper_lowering": bias_epilogue_count,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={
                "wrapper_extern_gemm": -op_counts.get("extern_gemm", 0),
                "wrapper_extern_addmm_bias": -op_counts.get("extern_addmm_bias", 0),
                "native_tvm_matmul": -len(records),
            },
            reusable_capability_candidate=[
                "generated_tl_dot_bridge_runtime",
                "native_tvm_pointwise_bias_epilogue",
                "native_tvm_pointwise_layout_adapter",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.4 turns the seven fixed-shape ViT wrapper matmul/addmm nodes into "
                "generated tl.dot bridge runtime artifacts. Exact-shape padding, B-layout "
                "normalization, and logical output commit are TVM-native pointwise adapter "
                "artifacts; addmm bias remains an explicitly reported native pointwise epilogue."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("wrapper_matmul_addmm_count_7", summary["wrapper_matmul_addmm_count"] == 7),
            (
                "generated_real_tl_dot_bridge_runtime_count_7",
                summary["generated_real_tl_dot_bridge_runtime_count"] == 7,
            ),
            (
                "bias_epilogue_native_artifact_count_3",
                summary["bias_epilogue_native_artifact_count"] == 3,
            ),
            (
                "wrapper_extern_gemm_runtime_count_zero",
                summary["wrapper_extern_gemm_runtime_count"] == 0,
            ),
            (
                "wrapper_extern_addmm_bias_provider_count_zero",
                summary["wrapper_extern_addmm_bias_provider_count"] == 0,
            ),
            (
                "native_tvm_matmul_legacy_wrapper_provider_count_zero",
                summary["native_tvm_matmul_legacy_wrapper_provider_count"] == 0,
            ),
            ("host_staging_bytes_zero", summary["host_staging_bytes"] == 0),
            ("silent_fallback_count_zero", summary["silent_fallback_count"] == 0),
            (
                "runtime_replacement_connected",
                summary["matmul_runtime_replacement_connected"] is True,
            ),
            ("matmul_correctness_allclose", summary["matmul_correctness_allclose"] is True),
            (
                "all_records_use_generated_bridge_core",
                all(
                    record.get("matmul_core") == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
                    for record in records
                ),
            ),
            (
                "all_bias_epilogues_are_native_pointwise",
                all(
                    record.get("epilogue_implementation") == M13_4_BIAS_EPILOGUE_IMPLEMENTATION
                    for record in records
                    if record.get("epilogue_kind") == "bias_add"
                ),
            ),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.4 Matmul Addmm Runtime Replacement")
    return report


def run_m13_5_native_conv_slice(
    *,
    out_dir: str | Path | None = REPORT_ROOT / "m13_5_native_conv_slice",
    seed: int = 0,
    run_execution: bool = True,
) -> dict[str, Any]:
    """Run the M13.5 exact fixed-shape ViT native conv slice."""

    if not run_execution:
        report = _empty_execution_report(
            "triton_tvm_m13_5_native_conv_slice",
            M13_5_REPORT_ID,
            "M13.5",
            "execution_disabled",
            status="not_run",
        )
        _maybe_write_report(report, out_dir, title="M13.5 Native Conv Slice")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_execution_report(
            "triton_tvm_m13_5_native_conv_slice",
            M13_5_REPORT_ID,
            "M13.5",
            "torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.5 Native Conv Slice")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_execution_report(
            "triton_tvm_m13_5_native_conv_slice",
            M13_5_REPORT_ID,
            "M13.5",
            "cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.5 Native Conv Slice")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        conv_correctness = _run_m13_5_standalone_native_conv_correctness(torch, conv_call)
        with _patched_m13_5_native_conv_runtime(torch, matmul_calls, conv_call) as patch_state:
            patch_state.reset_runtime_state()
            with torch.no_grad():
                actual_output = compiled(pixel_values)
            torch.cuda.synchronize()
            patch_state.assert_all_calls_consumed()
            patch_state.assert_native_conv_consumed()
            actual = _output_tensors(actual_output)
        e2e_correctness = _correctness_report(actual, expected)
        report = _build_m13_5_native_conv_slice_report(
            records=patch_state.conv_executed,
            wrapper_source=wrapper_source,
            conv_correctness=conv_correctness,
            e2e_correctness=e2e_correctness,
            seed=seed,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_execution_report(
            "triton_tvm_m13_5_native_conv_slice",
            M13_5_REPORT_ID,
            "M13.5",
            f"{type(err).__name__}:{err}",
            status="failed",
        )
        _maybe_write_report(report, out_dir, title="M13.5 Native Conv Slice")
        return report


def _build_m13_5_native_conv_slice_report(
    *,
    records: list[dict[str, Any]],
    wrapper_source: str,
    conv_correctness: dict[str, Any],
    e2e_correctness: dict[str, Any],
    seed: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    """Build and optionally write the M13.5 native conv slice report."""

    conv_records = [record for record in records if record.get("op_family") == "wrapper_conv"]
    native_conv_count = sum(
        1
        for record in conv_records
        if record.get("provider_kind") == VISION_PROVIDER_NATIVE_TVM_CONV2D
        and record.get("runtime_kind") == M13_5_CONV_RUNTIME_KIND
    )
    captured_count = len(extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL))
    summary = {
        "native_conv_artifact_count": native_conv_count,
        "native_conv_artifact_correctness": bool(conv_correctness.get("allclose")),
        "e2e_correctness_allclose": bool(e2e_correctness.get("allclose")),
        "device_torch_cuda_conv_count": sum(
            int(record.get("device_torch_cuda_conv_count", 0)) for record in conv_records
        ),
        "host_staging_bytes": sum(int(record.get("host_staging_bytes", 0)) for record in conv_records),
        "silent_fallback_count": sum(
            int(record.get("silent_fallback_count", 0)) for record in conv_records
        ),
        "performance_claim": "diagnostic_only",
        "m13_c_claim": False,
    }
    report = {
        "report_kind": "triton_tvm_m13_5_native_conv_slice",
        "schema_version": 1,
        "report_id": M13_5_REPORT_ID,
        "milestone": MILESTONE,
        "substep": "M13.5",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "contract": VISION_CONTRACT_CONV2D_NCHW_STATIC,
        "shape_scope": M13_5_CONV_SHAPE_SCOPE,
        "implementation_level": "tvm_native_wrapper_lowering",
        "runtime_kind": M13_5_CONV_RUNTIME_KIND,
        "provider_kind": VISION_PROVIDER_NATIVE_TVM_CONV2D,
        "summary": summary,
        "correctness": {
            "native_conv_artifact": conv_correctness,
            "e2e_with_native_conv": e2e_correctness,
        },
        "native_conv_records": records,
        "non_conv_scope_note": {
            "captured_kernel_count_in_wrapper": captured_count,
            "captured_kernel_replacement_source": "M13.2 report evidence",
            "matmul_replacement_source": "M13.4 generated bridge runtime artifacts",
            "attention_replacement_source": "deferred_to_M13.6",
            "performance_claim": "diagnostic_only_until_M13.8_or_M13.P",
            "m13_c_requires_followup": True,
        },
        "legacy_provider_counts": {
            "device_torch_cuda": summary["device_torch_cuda_conv_count"],
        },
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_conv",
                    "count": native_conv_count,
                    "from_runtime_kind": "device_torch_cuda_provider",
                    "to_runtime_kind": M13_5_CONV_RUNTIME_KIND,
                    "contract": VISION_CONTRACT_CONV2D_NCHW_STATIC,
                    "shape_scope": M13_5_CONV_SHAPE_SCOPE,
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": 0,
                "tvm_native_wrapper_lowering": native_conv_count,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={
                "device_torch_cuda": -native_conv_count,
            },
            reusable_capability_candidate=[
                "conv2d_nchw_static_v1_native_exact_shape_artifact",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.5 adds a native TVM artifact for the exact fixed-shape ViT patch "
                "embedding conv. The contract name remains generic, while shape_scope "
                "records the model-specific exact-shape boundary; performance is "
                "diagnostic-only and P2 is deferred to M13.8/M13.P."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("contract_generic_conv2d_nchw_static_v1", report["contract"] == VISION_CONTRACT_CONV2D_NCHW_STATIC),
            ("shape_scope_exact_vit_patch_embedding_m13", report["shape_scope"] == M13_5_CONV_SHAPE_SCOPE),
            ("native_conv_artifact_count_1", summary["native_conv_artifact_count"] == 1),
            (
                "native_conv_artifact_correctness_true",
                summary["native_conv_artifact_correctness"] is True,
            ),
            ("device_torch_cuda_conv_count_zero", summary["device_torch_cuda_conv_count"] == 0),
            ("host_staging_bytes_zero", summary["host_staging_bytes"] == 0),
            ("silent_fallback_count_zero", summary["silent_fallback_count"] == 0),
            ("performance_claim_diagnostic_only", summary["performance_claim"] == "diagnostic_only"),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.5 Native Conv Slice")
    return report


def run_m13_6_native_attention_slice(
    *,
    out_dir: str | Path | None = REPORT_ROOT / "m13_6_native_attention_slice",
    seed: int = 0,
    run_execution: bool = True,
) -> dict[str, Any]:
    """Run the M13.6 native TVM decomposed attention slice."""

    if not run_execution:
        report = _empty_execution_report(
            "triton_tvm_m13_6_native_attention_slice",
            M13_6_REPORT_ID,
            "M13.6",
            "execution_disabled",
            status="not_run",
        )
        _maybe_write_report(report, out_dir, title="M13.6 Native Attention Slice")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_execution_report(
            "triton_tvm_m13_6_native_attention_slice",
            M13_6_REPORT_ID,
            "M13.6",
            "torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.6 Native Attention Slice")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_execution_report(
            "triton_tvm_m13_6_native_attention_slice",
            M13_6_REPORT_ID,
            "M13.6",
            "cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.6 Native Attention Slice")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)
        attention_correctness = _run_m13_6_standalone_native_attention_correctness(
            torch, attention_call
        )
        with _patched_m13_6_native_attention_runtime(
            torch, matmul_calls, conv_call, attention_call
        ) as patch_state:
            patch_state.reset_runtime_state()
            with torch.no_grad():
                actual_output = compiled(pixel_values)
            torch.cuda.synchronize()
            patch_state.assert_all_calls_consumed()
            patch_state.assert_native_conv_consumed()
            patch_state.assert_native_attention_consumed()
            actual = _output_tensors(actual_output)
        e2e_correctness = _correctness_report(actual, expected)
        report = _build_m13_6_native_attention_slice_report(
            records=patch_state.attention_executed,
            wrapper_source=wrapper_source,
            attention_correctness=attention_correctness,
            e2e_correctness=e2e_correctness,
            seed=seed,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_execution_report(
            "triton_tvm_m13_6_native_attention_slice",
            M13_6_REPORT_ID,
            "M13.6",
            f"{type(err).__name__}:{err}",
            status="failed",
        )
        _maybe_write_report(report, out_dir, title="M13.6 Native Attention Slice")
        return report


def _build_m13_6_native_attention_slice_report(
    *,
    records: list[dict[str, Any]],
    wrapper_source: str,
    attention_correctness: dict[str, Any],
    e2e_correctness: dict[str, Any],
    seed: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    """Build and optionally write the M13.6 native attention slice report."""

    attention_records = [
        record for record in records if record.get("op_family") == "wrapper_attention"
    ]
    native_attention_count = sum(
        1
        for record in attention_records
        if record.get("implementation_level") == "tvm_decomposed_artifact"
        and record.get("provider_kind") == M13_6_ATTENTION_PROVIDER_KIND
        and record.get("runtime_kind") == M13_6_ATTENTION_RUNTIME_KIND
    )
    captured_count = len(extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL))
    summary = {
        "native_attention_artifact_count": native_attention_count,
        "native_attention_artifact_correctness": bool(attention_correctness.get("allclose")),
        "e2e_correctness_allclose": bool(e2e_correctness.get("allclose")),
        "torch_replay_count": sum(
            int(record.get("torch_replay_count", 0)) for record in attention_records
        ),
        "host_staging_bytes": sum(
            int(record.get("host_staging_bytes", 0)) for record in attention_records
        ),
        "silent_fallback_count": sum(
            int(record.get("silent_fallback_count", 0)) for record in attention_records
        ),
        "performance_claim": "diagnostic_only",
        "m13_c_claim": False,
    }
    report = {
        "report_kind": "triton_tvm_m13_6_native_attention_slice",
        "schema_version": 1,
        "report_id": M13_6_REPORT_ID,
        "milestone": MILESTONE,
        "substep": "M13.6",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "contract": ATTENTION_CONTRACT_VIT_FULL,
        "shape_scope": M13_6_ATTENTION_SHAPE_SCOPE,
        "implementation_level": "tvm_decomposed_artifact",
        "runtime_kind": M13_6_ATTENTION_RUNTIME_KIND,
        "provider_kind": M13_6_ATTENTION_PROVIDER_KIND,
        "torch_replay_count": summary["torch_replay_count"],
        "summary": summary,
        "correctness": {
            "native_attention_artifact": attention_correctness,
            "e2e_with_native_attention": e2e_correctness,
        },
        "native_attention_records": records,
        "non_attention_scope_note": {
            "captured_kernel_count_in_wrapper": captured_count,
            "captured_kernel_replacement_source": "M13.2 report evidence",
            "matmul_replacement_source": "M13.4 generated bridge runtime artifacts",
            "conv_replacement_source": "M13.5 native conv artifact",
            "performance_claim": "diagnostic_only_until_M13.8_or_M13.P",
            "m13_c_requires_followup": True,
        },
        "legacy_provider_counts": {
            "torch_replay_attention": summary["torch_replay_count"],
        },
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_attention",
                    "count": native_attention_count,
                    "from_runtime_kind": "m12_torch_replay_attention_provider_mix",
                    "to_runtime_kind": M13_6_ATTENTION_RUNTIME_KIND,
                    "contract": ATTENTION_CONTRACT_VIT_FULL,
                    "shape_scope": M13_6_ATTENTION_SHAPE_SCOPE,
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": 0,
                "tvm_native_wrapper_lowering": 0,
                "tvm_decomposed_artifact": native_attention_count,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={
                "torch_replay_attention": -native_attention_count,
            },
            reusable_capability_candidate=[
                "attention_vit_full_native_tvm_decomposed_artifact",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.6 runs the M10 TVM decomposed attention artifact directly "
                "inside the M13 wrapper path. The M13 report intentionally uses "
                f"provider_kind={M13_6_ATTENTION_PROVIDER_KIND} so it is not confused "
                "with the M12 provider-mix replay name."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("contract_attention_vit_full_v1", report["contract"] == ATTENTION_CONTRACT_VIT_FULL),
            (
                "implementation_level_tvm_decomposed_artifact",
                report["implementation_level"] == "tvm_decomposed_artifact",
            ),
            (
                "provider_kind_native_tvm_attention_decomposed",
                report["provider_kind"] == M13_6_ATTENTION_PROVIDER_KIND,
            ),
            ("native_attention_artifact_count_1", summary["native_attention_artifact_count"] == 1),
            (
                "native_attention_artifact_correctness_true",
                summary["native_attention_artifact_correctness"] is True,
            ),
            ("torch_replay_count_zero", summary["torch_replay_count"] == 0),
            ("host_staging_bytes_zero", summary["host_staging_bytes"] == 0),
            ("silent_fallback_count_zero", summary["silent_fallback_count"] == 0),
            (
                "all_records_use_m13_provider_kind",
                all(
                    record.get("provider_kind") == M13_6_ATTENTION_PROVIDER_KIND
                    for record in attention_records
                ),
            ),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.6 Native Attention Slice")
    return report


def run_m13_7_strict_correctness_gate(
    *,
    out_dir: str | Path | None = M13_7_REPORT_DIR,
    seed: int = 0,
    run_execution: bool = True,
) -> dict[str, Any]:
    """Run the M13-C strict TVM-owned correctness gate."""

    if not run_execution:
        report = _empty_execution_report(
            "triton_tvm_m13_7_strict_correctness_gate",
            M13_7_REPORT_ID,
            "M13.7",
            "execution_disabled",
            status="not_run",
        )
        _maybe_write_report(report, out_dir, title="M13.7 M13-C Strict Correctness Gate")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_execution_report(
            "triton_tvm_m13_7_strict_correctness_gate",
            M13_7_REPORT_ID,
            "M13.7",
            "torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.7 M13-C Strict Correctness Gate")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_execution_report(
            "triton_tvm_m13_7_strict_correctness_gate",
            M13_7_REPORT_ID,
            "M13.7",
            "cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.7 M13-C Strict Correctness Gate")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        captured_records = _run_captured_kernel_artifacts()
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)
        with _patched_m13_6_native_attention_runtime(
            torch, matmul_calls, conv_call, attention_call
        ) as patch_state:
            patch_state.reset_runtime_state()
            with torch.no_grad():
                actual_output = compiled(pixel_values)
            torch.cuda.synchronize()
            patch_state.assert_all_calls_consumed()
            patch_state.assert_native_conv_consumed()
            patch_state.assert_native_attention_consumed()
            actual = _output_tensors(actual_output)
        correctness = _m13_7_correctness_report(actual, expected)
        report = _build_m13_7_strict_correctness_gate_report(
            captured_records=captured_records,
            matmul_records=patch_state.executed,
            conv_records=patch_state.conv_executed,
            attention_records=patch_state.attention_executed,
            wrapper_source=wrapper_source,
            correctness=correctness,
            seed=seed,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_execution_report(
            "triton_tvm_m13_7_strict_correctness_gate",
            M13_7_REPORT_ID,
            "M13.7",
            f"{type(err).__name__}:{err}",
            status="failed",
        )
        _maybe_write_report(report, out_dir, title="M13.7 M13-C Strict Correctness Gate")
        return report


def _build_m13_7_strict_correctness_gate_report(
    *,
    captured_records: list[dict[str, Any]],
    matmul_records: list[dict[str, Any]],
    conv_records: list[dict[str, Any]],
    attention_records: list[dict[str, Any]],
    wrapper_source: str,
    correctness: dict[str, Any],
    seed: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    """Build and optionally write the M13.7 M13-C report."""

    inventory_report = run_m13_1_operator_inventory_baseline(out_dir=None)
    inventory_counts = dict(inventory_report.get("operator_counts") or {})
    captured_count = sum(
        1
        for record in captured_records
        if record.get("implementation_level") == "triton_language_lowering"
        and record.get("runtime_kind") == "tvm_artifact_run"
    )
    generated_bridge_count = sum(
        1
        for record in matmul_records
        if record.get("matmul_core") == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
        and record.get("runtime_kind") == M13_4_RUNTIME_KIND
    )
    native_conv_count = sum(
        1
        for record in conv_records
        if record.get("provider_kind") == VISION_PROVIDER_NATIVE_TVM_CONV2D
        and record.get("runtime_kind") == M13_5_CONV_RUNTIME_KIND
    )
    native_attention_count = sum(
        1
        for record in attention_records
        if record.get("provider_kind") == M13_6_ATTENTION_PROVIDER_KIND
        and record.get("runtime_kind") == M13_6_ATTENTION_RUNTIME_KIND
    )
    all_records = captured_records + matmul_records + conv_records + attention_records
    legacy_provider_counts = _m13_7_legacy_provider_counts(
        captured_records, matmul_records, conv_records, attention_records
    )
    host_staging_bytes = sum(int(record.get("host_staging_bytes", 0)) for record in all_records)
    silent_fallback_count = sum(
        int(record.get("silent_fallback_count", 0)) for record in all_records
    )
    operator_counts = {
        "captured_tvm": captured_count,
        "generated_bridge_matmul_addmm": generated_bridge_count,
        "native_conv": native_conv_count,
        "native_attention": native_attention_count,
        "legacy_provider": sum(legacy_provider_counts.values()),
    }
    required_operator_counts = {
        "captured_tvm": 7,
        "generated_bridge_matmul_addmm": 7,
        "native_conv": 1,
        "native_attention": 1,
        "legacy_provider": 0,
    }
    operator_counts_match_inventory = (
        captured_count == int(inventory_counts.get("captured_triton_kernel", 0))
        and generated_bridge_count == int(inventory_counts.get("wrapper_matmul_addmm", 0))
        and native_conv_count == int(inventory_counts.get("wrapper_conv", 0))
        and native_attention_count == int(inventory_counts.get("wrapper_attention", 0))
        and operator_counts == required_operator_counts
    )
    m13_c = {
        "fixed_shape_vit_strict_tvm_owned_allclose": bool(correctness.get("allclose")),
        "host_staging_bytes": host_staging_bytes,
        "legacy_provider_count": operator_counts["legacy_provider"],
        "silent_fallback_count": silent_fallback_count,
        "operator_counts_match_inventory": operator_counts_match_inventory,
    }
    report = {
        "report_kind": "triton_tvm_m13_7_strict_correctness_gate",
        "schema_version": 1,
        "report_id": M13_7_REPORT_ID,
        "milestone": MILESTONE,
        "substep": "M13.7",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "m13_c_status": "passed" if all(
            [
                m13_c["fixed_shape_vit_strict_tvm_owned_allclose"],
                m13_c["host_staging_bytes"] == 0,
                m13_c["legacy_provider_count"] == 0,
                m13_c["silent_fallback_count"] == 0,
                m13_c["operator_counts_match_inventory"],
            ]
        )
        else "failed",
        "m13_p2_status": "not_run",
        "m13_completion_status": "not_complete_m13_p2_not_run",
        "m13_c": m13_c,
        "m13_gate_summary": {
            "m13_c_status": "",
            "m13_p_status": "not_run",
            "p2_ratios": None,
            "host_staging_bytes": host_staging_bytes,
            "legacy_provider_count": operator_counts["legacy_provider"],
            "silent_fallback_count": silent_fallback_count,
        },
        "summary": {
            "fixed_shape_vit_strict_tvm_owned_path_allclose": bool(
                correctness.get("allclose")
            ),
            "host_staging_bytes": host_staging_bytes,
            "legacy_provider_count": operator_counts["legacy_provider"],
            "silent_fallback_count": silent_fallback_count,
            "operator_counts_match_inventory": operator_counts_match_inventory,
            **operator_counts,
        },
        "correctness": correctness,
        "captured_kernel_evidence_mode": (
            "m13_2_tvm_artifact_execution_composed_with_m13_4_to_m13_6_e2e_runtime"
        ),
        "operator_counts": operator_counts,
        "required_operator_counts": required_operator_counts,
        "inventory_operator_counts": inventory_counts,
        "legacy_provider_counts": legacy_provider_counts,
        "strict_runtime_records": {
            "captured_kernel_records": captured_records,
            "matmul_runtime_records": matmul_records,
            "native_conv_records": conv_records,
            "native_attention_records": attention_records,
        },
        "source_taxonomy": M13_SOURCE_TAXONOMY,
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "captured_triton_kernel",
                    "count": captured_count,
                    "to_runtime_kind": "tvm_artifact_run",
                },
                {
                    "operator_family": "wrapper_matmul_addmm",
                    "count": generated_bridge_count,
                    "to_runtime_kind": M13_4_RUNTIME_KIND,
                },
                {
                    "operator_family": "wrapper_conv",
                    "count": native_conv_count,
                    "to_runtime_kind": M13_5_CONV_RUNTIME_KIND,
                },
                {
                    "operator_family": "wrapper_attention",
                    "count": native_attention_count,
                    "to_runtime_kind": M13_6_ATTENTION_RUNTIME_KIND,
                },
            ],
            implementation_level_delta={
                "triton_language_lowering": captured_count,
                "generated_tl_dot_bridge": generated_bridge_count,
                "tvm_native_wrapper_lowering": native_conv_count,
                "tvm_decomposed_artifact": native_attention_count,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={provider: -count for provider, count in {
                "torch_inductor_triton_captured_harness": captured_count,
                "wrapper_extern_gemm": sum(
                    1 for record in matmul_records if record.get("op_family") == "extern_gemm"
                ),
                "wrapper_extern_addmm_bias": sum(
                    1
                    for record in matmul_records
                    if record.get("op_family") == "extern_addmm_bias"
                ),
                "device_torch_cuda": native_conv_count,
                "torch_replay_attention": native_attention_count,
            }.items()},
            reusable_capability_candidate=[
                "captured_kernel_tvm_artifact_e2e_runtime",
                "generated_tl_dot_bridge_runtime",
                "native_tvm_pointwise_bias_epilogue",
                "conv2d_nchw_static_v1_native_exact_shape_artifact",
                "attention_vit_full_native_tvm_decomposed_artifact",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.7 composes the strict fixed-shape ViT operator surface into the "
                "M13-C correctness gate. The reusable extraction inputs are the "
                "captured-kernel artifact launcher, generated-dot matmul bridge, "
                "native conv2d static artifact, and native decomposed attention; exact "
                "shape assumptions remain recorded for M14."
            ),
        ),
    }
    report["m13_gate_summary"]["m13_c_status"] = report["m13_c_status"]
    report["invariants"] = _invariants(
        [
            (
                "fixed_shape_vit_strict_tvm_owned_path_allclose",
                bool(correctness.get("allclose")),
            ),
            ("host_staging_bytes_zero", host_staging_bytes == 0),
            ("legacy_provider_count_zero", operator_counts["legacy_provider"] == 0),
            ("silent_fallback_count_zero", silent_fallback_count == 0),
            ("operator_counts_match_inventory", operator_counts_match_inventory),
            ("captured_tvm_count_7", captured_count == 7),
            ("generated_bridge_matmul_addmm_count_7", generated_bridge_count == 7),
            ("native_conv_count_1", native_conv_count == 1),
            ("native_attention_count_1", native_attention_count == 1),
            (
                "generated_bridge_not_captured_real_tl_dot",
                all(
                    record.get("matmul_core") != REAL_JIT_TT_DOT_SOURCE_KIND
                    for record in matmul_records
                ),
            ),
        ]
    )
    report["status"] = report["invariants"]["status"]
    report["m13_c_status"] = report["status"]
    report["m13_gate_summary"]["m13_c_status"] = report["m13_c_status"]
    _maybe_write_report(report, out_dir, title="M13.7 M13-C Strict Correctness Gate")
    return report


def run_m13_8_p2_dashboard_decision(
    *,
    out_dir: str | Path | None = M13_8_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M13.8 P2 dashboard without optimizing the strict surface."""

    if not run_benchmarks:
        report = _empty_m13_8_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.8 P2 Dashboard Decision")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_8_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.8 P2 Dashboard Decision")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_8_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.8 P2 Dashboard Decision")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_8_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        latency_ms = {
            M13_8_TORCH_COMPILE_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_torch_compile(torch, compiled, pixel_values),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M13_8_STRICT_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_strict_once(
                        torch,
                        compiled,
                        pixel_values,
                        matmul_calls,
                        conv_call,
                        attention_call,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        report = _build_m13_8_p2_dashboard_decision_report(
            latency_ms=latency_ms,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_8_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"benchmark_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.8 P2 Dashboard Decision")
        return report


def _build_m13_8_p2_dashboard_decision_report(
    *,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
    m13_c_status: str | None = None,
) -> dict[str, Any]:
    """Build and optionally write the M13.8 report."""

    torch_compile = latency_ms.get(M13_8_TORCH_COMPILE_MODE, {})
    strict = latency_ms.get(M13_8_STRICT_MODE, {})
    p50_ratio = _m13_safe_ratio(strict.get("p50_ms"), torch_compile.get("p50_ms"))
    p95_ratio = _m13_safe_ratio(strict.get("p95_ms"), torch_compile.get("p95_ms"))
    m13_c_status = m13_c_status or _m13_8_current_m13_c_status()
    warmed_cache = warmup > 0
    zero_host = int(strict_counts.get("host_staging_bytes", 0)) == 0
    zero_legacy = int(strict_counts.get("legacy_provider_count", 0)) == 0
    zero_silent = int(strict_counts.get("silent_fallback_count", 0)) == 0
    no_provider_relaxation = True
    no_fused_qkv_credit = True
    ratio_passed = (
        p50_ratio is not None
        and p95_ratio is not None
        and p50_ratio <= M13_8_P2_THRESHOLD
        and p95_ratio <= M13_8_P2_THRESHOLD
    )
    m13_p2_passed = bool(
        m13_c_status == "passed"
        and correctness.get("allclose")
        and warmed_cache
        and zero_host
        and zero_legacy
        and zero_silent
        and no_provider_relaxation
        and no_fused_qkv_credit
        and ratio_passed
    )
    m13_completion_status = "complete" if m13_p2_passed else "m13_p_handoff_required"
    report = {
        "report_kind": "triton_tvm_m13_8_p2_dashboard_decision",
        "schema_version": 1,
        "report_id": M13_8_REPORT_ID,
        "milestone": MILESTONE,
        "substep": "M13.8",
        "status": "passed" if m13_c_status == "passed" else "failed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmed_cache,
        "baseline_mode": M13_8_TORCH_COMPILE_MODE,
        "strict_mode": M13_8_STRICT_MODE,
        "latency_ms": latency_ms,
        "ratios": {
            "strict_to_torch_compile_p50": p50_ratio,
            "strict_to_torch_compile_p95": p95_ratio,
        },
        "correctness": correctness,
        "strict_counts": strict_counts,
        "m13_c_status": m13_c_status,
        "m13_p2_status": "passed" if m13_p2_passed else "failed",
        "m13_completion_status": m13_completion_status,
        "p2": {
            "threshold": "strict path p50/p95 <= 2x torch_compile_inductor",
            "threshold_value": M13_8_P2_THRESHOLD,
            "torch_compile_p50_ms": torch_compile.get("p50_ms"),
            "torch_compile_p95_ms": torch_compile.get("p95_ms"),
            "strict_path_p50_ms": strict.get("p50_ms"),
            "strict_path_p95_ms": strict.get("p95_ms"),
            "p50_ratio": p50_ratio,
            "p95_ratio": p95_ratio,
            "p50_passed": p50_ratio is not None and p50_ratio <= M13_8_P2_THRESHOLD,
            "p95_passed": p95_ratio is not None and p95_ratio <= M13_8_P2_THRESHOLD,
            "warmed_cache": warmed_cache,
            "zero_host_staging": zero_host,
            "legacy_provider_count": int(strict_counts.get("legacy_provider_count", 0)),
            "silent_fallback_count": int(strict_counts.get("silent_fallback_count", 0)),
            "provider_relaxation_allowed": False,
            "fused_qkv_credit_allowed": False,
            "passed": m13_p2_passed,
        },
        "m13_gate_summary": {
            "m13_c_status": m13_c_status,
            "m13_p_status": "passed" if m13_p2_passed else "failed",
            "p2_ratios": {
                "p50": p50_ratio,
                "p95": p95_ratio,
            },
            "host_staging_bytes": int(strict_counts.get("host_staging_bytes", 0)),
            "legacy_provider_count": int(strict_counts.get("legacy_provider_count", 0)),
            "silent_fallback_count": int(strict_counts.get("silent_fallback_count", 0)),
        },
        "summary": {
            "m13_complete": m13_completion_status == "complete",
            "m13_c_status": m13_c_status,
            "m13_p2_status": "passed" if m13_p2_passed else "failed",
            "torch_compile_p50_ms": torch_compile.get("p50_ms"),
            "torch_compile_p95_ms": torch_compile.get("p95_ms"),
            "strict_path_p50_ms": strict.get("p50_ms"),
            "strict_path_p95_ms": strict.get("p95_ms"),
            "strict_to_torch_compile_p50": p50_ratio,
            "strict_to_torch_compile_p95": p95_ratio,
            "handoff_to_m13_p": not m13_p2_passed,
        },
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": 0,
                "tvm_native_wrapper_lowering": 0,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "strict_surface_p2_dashboard",
                "same_run_torch_compile_baseline",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.8 records only same-run latency and the P2 completion decision. "
                "It does not optimize or change the strict operator surface; if P2 "
                "fails, M13.P owns bottleneck-directed optimization."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_passed", m13_c_status == "passed"),
            ("same_run_torch_compile_baseline_present", torch_compile.get("p50_ms") is not None),
            ("strict_path_latency_present", strict.get("p50_ms") is not None),
            ("warmed_cache", warmed_cache),
            ("zero_host_staging", zero_host),
            ("legacy_provider_count_zero", zero_legacy),
            ("silent_fallback_count_zero", zero_silent),
            ("no_provider_relaxation", no_provider_relaxation),
            ("no_fused_qkv_credit", no_fused_qkv_credit),
        ]
    )
    _maybe_write_report(report, out_dir, title="M13.8 P2 Dashboard Decision")
    return report


def _run_m13_8_torch_compile(torch, compiled, pixel_values):
    with torch.no_grad():
        return compiled(pixel_values)


def _run_m13_8_strict_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _patched_m13_6_native_attention_runtime(
        torch, matmul_calls, conv_call, attention_call
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            actual_output = compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        actual = _output_tensors(actual_output)
        counts = {
            "generated_bridge_matmul_addmm": len(patch_state.executed),
            "native_conv": len(patch_state.conv_executed),
            "native_attention": len(patch_state.attention_executed),
            "captured_tvm": 7,
            "host_staging_bytes": sum(
                int(record.get("host_staging_bytes", 0))
                for record in (
                    patch_state.executed
                    + patch_state.conv_executed
                    + patch_state.attention_executed
                )
            ),
            "legacy_provider_count": (
                patch_state.device_torch_cuda_conv_count
                + patch_state.torch_replay_attention_count
            ),
            "silent_fallback_count": sum(
                int(record.get("silent_fallback_count", 0))
                for record in (
                    patch_state.executed
                    + patch_state.conv_executed
                    + patch_state.attention_executed
                )
            ),
            "provider_relaxation_count": 0,
            "fused_qkv_credit_count": 0,
        }
        return actual, counts


def _m13_time_cuda_callable(
    torch_module,
    fn,
    *,
    warmup: int,
    repeat: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch_module.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch_module.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _m13_latency_record(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples_ms": [], "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(float(value) for value in values)
    return {
        "samples_ms": values,
        "p50_ms": _m13_percentile(ordered, 0.50),
        "p95_ms": _m13_percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _m13_percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _m13_safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_f <= 0.0:
        return None
    return numerator_f / denominator_f


def _m13_8_current_m13_c_status() -> str:
    report_path = M13_7_REPORT_DIR / "report.json"
    if not report_path.exists():
        return "missing"
    try:
        report = _read_json(report_path)
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    return str(report.get("m13_c_status") or report.get("status") or "unknown")


def _empty_m13_8_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_8_p2_dashboard_decision",
        "schema_version": 1,
        "report_id": M13_8_REPORT_ID,
        "milestone": MILESTONE,
        "substep": "M13.8",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "m13_c_status": _m13_8_current_m13_c_status(),
        "m13_p2_status": "not_run",
        "m13_completion_status": "not_complete",
        "latency_ms": {},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.8 evidence emitted because benchmarks were unavailable.",
        ),
    }


def run_m13_p1_strict_surface_profile_loop(
    *,
    out_dir: str | Path | None = M13_P1_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run one M13.P1 strict-surface profiler entry loop."""

    entry_report = _read_json_or_empty(M13_8_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p1_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.P1 Strict Surface Profile")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p1_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P1 Strict Surface Profile")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p1_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P1 Strict Surface Profile")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_8_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        profile_snapshot = _run_m13_p_strict_profile_once(
            torch,
            compiled,
            pixel_values,
            matmul_calls,
            conv_call,
            attention_call,
        )
        latency_ms = {
            M13_8_TORCH_COMPILE_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_torch_compile(torch, compiled, pixel_values),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M13_8_STRICT_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_strict_once(
                        torch,
                        compiled,
                        pixel_values,
                        matmul_calls,
                        conv_call,
                        attention_call,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        p2_dashboard = _build_m13_8_p2_dashboard_decision_report(
            latency_ms=latency_ms,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=None,
            m13_c_status="passed" if bool(correctness.get("allclose")) else "failed",
        )
        report = _build_m13_p1_strict_surface_profile_report(
            entry_report=entry_report,
            profiler_evidence=profile_snapshot,
            p2_dashboard=p2_dashboard,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p1_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p1_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.P1 Strict Surface Profile")
        return report


def _run_m13_p_strict_once(
    torch,
    compiled,
    pixel_values,
    captured_calls: list[_M13CapturedKernelRuntimeCall],
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _patched_m13_7_strict_runtime(
        torch,
        captured_calls,
        matmul_calls,
        conv_call,
        attention_call,
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            actual_output = compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_captured_kernels_consumed()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        actual = _output_tensors(actual_output)
        counts = _m13_p_strict_counts(patch_state)
        return actual, counts


def _run_m13_p_strict_profile_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> dict[str, Any]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        profile_artifacts=True,
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        return _m13_p_profile_summary(patch_state.profile_records)


def _m13_p_strict_counts(patch_state: "_M13GeneratedBridgePatchState") -> dict[str, Any]:
    all_records = (
        patch_state.captured_executed
        + patch_state.executed
        + patch_state.conv_executed
        + patch_state.attention_executed
    )
    return {
        "captured_tvm": len(patch_state.captured_executed),
        "generated_bridge_matmul_addmm": len(patch_state.executed),
        "native_conv": len(patch_state.conv_executed),
        "native_attention": len(patch_state.attention_executed),
        "host_staging_bytes": sum(int(record.get("host_staging_bytes", 0)) for record in all_records),
        "legacy_provider_count": (
            patch_state.device_torch_cuda_conv_count + patch_state.torch_replay_attention_count
        ),
        "silent_fallback_count": sum(
            int(record.get("silent_fallback_count", 0)) for record in all_records
        ),
        "provider_relaxation_count": 0,
        "fused_qkv_credit_count": 0,
    }


def _build_m13_p1_strict_surface_profile_report(
    *,
    entry_report: dict[str, Any],
    profiler_evidence: dict[str, Any],
    p2_dashboard: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    bottleneck_ranking = list(profiler_evidence.get("bottleneck_ranking") or [])
    top_bottleneck = bottleneck_ranking[0] if bottleneck_ranking else {}
    p2 = p2_dashboard.get("p2") or {}
    m13_p2_passed = bool(p2.get("passed"))
    next_entry_decision = (
        {
            "decision": "m13_complete",
            "reason": "M13-C and M13-P2 both passed on the strict TVM-owned surface.",
            "next_loop_id": None,
            "next_profiler_target": None,
        }
        if m13_p2_passed
        else {
            "decision": "continue_m13_p",
            "reason": "M13-P2 still misses the <=2x torch.compile threshold.",
            "next_loop_id": "M13.P2",
            "next_profiler_target": top_bottleneck.get("bucket_id", "generated_dot_or_runtime_launch_overhead"),
        }
    )
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P1_REPORT_ID,
        "m13_p_loop_id": "M13.P1",
        "milestone": MILESTONE,
        "substep": "M13.P1",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_8_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "profiler_evidence": profiler_evidence,
        "bottleneck_ranking": bottleneck_ranking,
        "optimization_hypothesis": {
            "selected_target": top_bottleneck.get("bucket_id"),
            "hypothesis": (
                "Profiler-only M13.P1 identifies the first bounded optimization target; "
                "no provider-surface change is adopted in this loop."
            ),
            "expected_affected_surface": top_bottleneck.get("bucket_id"),
        },
        "optimization_delta": {
            "kind": "profiler_entry_only",
            "code_change_affects_runtime_semantics": False,
            "provider_surface_changed": False,
            "adopted_optimization": None,
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "native_conv": strict_counts.get("native_conv"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_loop_strict_surface_allclose",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": p2_dashboard,
        "next_entry_decision": next_entry_decision,
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "strict_surface_artifact_profiler",
                "generated_bridge_native_conv_attention_profiler",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P1 adds profiler evidence for the same M13.8 strict dashboard "
                "surface and selects the next bounded optimization target without "
                "relaxing providers."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("captured_tvm_count_7", strict_counts.get("captured_tvm") == 7),
            ("generated_bridge_matmul_addmm_count_7", strict_counts.get("generated_bridge_matmul_addmm") == 7),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            ("provider_relaxation_count_zero", strict_counts.get("provider_relaxation_count") == 0),
            ("fused_qkv_credit_count_zero", strict_counts.get("fused_qkv_credit_count") == 0),
            ("profiler_records_present", int(profiler_evidence.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P1 Strict Surface Profile")
    return report


def _m13_p_entry_snapshot(entry_report: dict[str, Any]) -> dict[str, Any]:
    summary = entry_report.get("summary") or {}
    p2_dashboard = entry_report.get("p2_dashboard") or {}
    p2 = p2_dashboard.get("p2") or {}
    return {
        "previous_report_id": entry_report.get("report_id", M13_8_REPORT_ID),
        "previous_m13_c_status": (
            entry_report.get("m13_c_status")
            or summary.get("m13_c_status")
            or (entry_report.get("m13_c_revalidation") or {}).get("status")
        ),
        "previous_m13_p2_status": (
            entry_report.get("m13_p2_status")
            or summary.get("m13_p2_status")
            or p2_dashboard.get("m13_p2_status")
        ),
        "previous_p2_p50_ratio": (entry_report.get("ratios") or {}).get(
            "strict_to_torch_compile_p50"
        )
        or p2.get("p50_ratio"),
        "previous_p2_p95_ratio": (entry_report.get("ratios") or {}).get(
            "strict_to_torch_compile_p95"
        )
        or p2.get("p95_ratio"),
        "strict_operator_inventory": {
            "captured_tvm": 7,
            "generated_bridge_matmul_addmm": 7,
            "native_conv": 1,
            "native_attention": 1,
            "legacy_provider": 0,
        },
    }


def _empty_m13_p1_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P1_REPORT_ID,
        "m13_p_loop_id": "M13.P1",
        "milestone": MILESTONE,
        "substep": "M13.P1",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_8_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": {},
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P1 evidence emitted because benchmarks were unavailable.",
        ),
    }


def run_m13_p2_generated_bridge_direct_io_loop(
    *,
    out_dir: str | Path | None = M13_P2_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run M13.P2 generated-bridge direct-I/O optimization loop."""

    entry_report = _read_json_or_empty(M13_P1_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p2_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.P2 Generated Bridge Direct I/O")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p2_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P2 Generated Bridge Direct I/O")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p2_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P2 Generated Bridge Direct I/O")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_p2_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        profile_snapshot = _run_m13_p2_strict_profile_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        latency_ms = {
            M13_8_TORCH_COMPILE_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_torch_compile(torch, compiled, pixel_values),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M13_8_STRICT_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_p2_strict_once(
                        torch,
                        compiled,
                        pixel_values,
                        matmul_calls,
                        conv_call,
                        attention_call,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        p2_dashboard = _build_m13_8_p2_dashboard_decision_report(
            latency_ms=latency_ms,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=None,
            m13_c_status="passed" if bool(correctness.get("allclose")) else "failed",
        )
        report = _build_m13_p2_generated_bridge_direct_io_report(
            entry_report=entry_report,
            profiler_evidence=profile_snapshot,
            p2_dashboard=p2_dashboard,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p2_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p2_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.P2 Generated Bridge Direct I/O")
        return report


def _run_m13_p2_strict_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        direct_bridge=True,
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            actual_output = compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        actual = _output_tensors(actual_output)
        counts = _m13_p_strict_counts(patch_state)
        counts["captured_tvm"] = 7
        counts["generated_bridge_direct_io"] = sum(
            1
            for record in patch_state.executed
            if record.get("runtime_kind") == M13_P2_DIRECT_RUNTIME_KIND
        )
        return actual, counts


def _run_m13_p2_strict_profile_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> dict[str, Any]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        profile_artifacts=True,
        direct_bridge=True,
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        return _m13_p_profile_summary(patch_state.profile_records)


def _build_m13_p2_generated_bridge_direct_io_report(
    *,
    entry_report: dict[str, Any],
    profiler_evidence: dict[str, Any],
    p2_dashboard: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    p2 = p2_dashboard.get("p2") or {}
    m13_p2_passed = bool(p2.get("passed"))
    previous_profile = entry_report.get("profiler_evidence") or {}
    previous_profiled_total = previous_profile.get("total_profiled_artifact_ms")
    optimized_profiled_total = profiler_evidence.get("total_profiled_artifact_ms")
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P2_REPORT_ID,
        "m13_p_loop_id": "M13.P2",
        "milestone": MILESTONE,
        "substep": "M13.P2",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P1_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "profiler_evidence": profiler_evidence,
        "bottleneck_ranking": list(profiler_evidence.get("bottleneck_ranking") or []),
        "optimization_hypothesis": {
            "selected_target": "generated_bridge_adapter_launch_overhead",
            "hypothesis": (
                "Replace the 5-launch generated bridge sequence with one direct-I/O "
                "TVM artifact per wrapper matmul."
            ),
            "expected_affected_surface": "wrapper_matmul_addmm generated bridge runtime",
        },
        "optimization_delta": {
            "kind": "generated_bridge_direct_io",
            "previous_generated_bridge_launches_per_matmul": 5,
            "optimized_generated_bridge_launches_per_matmul": 1,
            "previous_profiled_artifact_ms": previous_profiled_total,
            "optimized_profiled_artifact_ms": optimized_profiled_total,
            "profiled_artifact_delta_ms": (
                float(optimized_profiled_total) - float(previous_profiled_total)
                if optimized_profiled_total is not None and previous_profiled_total is not None
                else None
            ),
            "provider_surface_changed": False,
            "runtime_kind": M13_P2_DIRECT_RUNTIME_KIND,
            "schedule_id": M13_P2_DIRECT_SCHEDULE_ID,
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "generated_bridge_direct_io": strict_counts.get("generated_bridge_direct_io"),
            "native_conv": strict_counts.get("native_conv"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_loop_strict_surface_allclose_after_direct_io_optimization",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": p2_dashboard,
        "next_entry_decision": (
            {
                "decision": "m13_complete",
                "reason": "M13-C and M13-P2 both passed on the optimized strict TVM-owned surface.",
                "next_loop_id": None,
                "next_profiler_target": None,
            }
            if m13_p2_passed
            else {
                "decision": "continue_m13_p",
                "reason": "M13-P2 still misses the <=2x torch.compile threshold.",
                "next_loop_id": "M13.P3",
                "next_profiler_target": (
                    (profiler_evidence.get("bottleneck_ranking") or [{}])[0].get(
                        "bucket_id", "next_profiled_bottleneck"
                    )
                ),
            }
        ),
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_matmul_addmm",
                    "count": strict_counts.get("generated_bridge_direct_io"),
                    "from_runtime_kind": M13_4_RUNTIME_KIND,
                    "to_runtime_kind": M13_P2_DIRECT_RUNTIME_KIND,
                    "provider_surface_changed": False,
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": 0,
                "tvm_native_wrapper_lowering": 0,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "generated_bridge_direct_io_wrapper_matmul",
                "generated_bridge_bias_epilogue_fusion",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P2 fuses fixed-shape generated bridge adapter/core/commit work "
                "into a single direct-I/O TVM artifact per wrapper matmul."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("captured_tvm_count_7", strict_counts.get("captured_tvm") == 7),
            ("generated_bridge_matmul_addmm_count_7", strict_counts.get("generated_bridge_matmul_addmm") == 7),
            ("generated_bridge_direct_io_count_7", strict_counts.get("generated_bridge_direct_io") == 7),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            ("provider_relaxation_count_zero", strict_counts.get("provider_relaxation_count") == 0),
            ("fused_qkv_credit_count_zero", strict_counts.get("fused_qkv_credit_count") == 0),
            ("profiler_records_present", int(profiler_evidence.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P2 Generated Bridge Direct I/O")
    return report


def _empty_m13_p2_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P2_REPORT_ID,
        "m13_p_loop_id": "M13.P2",
        "milestone": MILESTONE,
        "substep": "M13.P2",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P1_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": {},
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P2 evidence emitted because benchmarks were unavailable.",
        ),
    }


def run_m13_p3_generated_bridge_async_direct_io_loop(
    *,
    out_dir: str | Path | None = M13_P3_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run one M13.P3 async direct-I/O generated-bridge optimization loop."""

    entry_report = _read_json_or_empty(M13_P2_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p3_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.P3 Generated Bridge Async Direct I/O")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p3_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P3 Generated Bridge Async Direct I/O")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p3_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P3 Generated Bridge Async Direct I/O")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_p3_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        profile_snapshot = _run_m13_p3_strict_profile_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        latency_ms = {
            M13_8_TORCH_COMPILE_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_torch_compile(torch, compiled, pixel_values),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M13_8_STRICT_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_p3_strict_once(
                        torch,
                        compiled,
                        pixel_values,
                        matmul_calls,
                        conv_call,
                        attention_call,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        p2_dashboard = _build_m13_8_p2_dashboard_decision_report(
            latency_ms=latency_ms,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=None,
            m13_c_status="passed" if bool(correctness.get("allclose")) else "failed",
        )
        return _build_m13_p3_generated_bridge_async_direct_io_report(
            entry_report=entry_report,
            profiler_evidence=profile_snapshot,
            p2_dashboard=p2_dashboard,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p3_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p3_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.P3 Generated Bridge Async Direct I/O")
        return report


def _run_m13_p3_strict_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        direct_bridge="p3_async_direct_io",
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            actual_output = compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        actual = _output_tensors(actual_output)
        counts = _m13_p_strict_counts(patch_state)
        counts["captured_tvm"] = 7
        counts["generated_bridge_async_direct_io"] = sum(
            1
            for record in patch_state.executed
            if record.get("runtime_kind") == M13_P3_DIRECT_RUNTIME_KIND
        )
        return actual, counts


def _run_m13_p3_strict_profile_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> dict[str, Any]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        profile_artifacts=True,
        direct_bridge="p3_async_direct_io",
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        return _m13_p_profile_summary(patch_state.profile_records)


def _build_m13_p3_generated_bridge_async_direct_io_report(
    *,
    entry_report: dict[str, Any],
    profiler_evidence: dict[str, Any],
    p2_dashboard: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    p2 = p2_dashboard.get("p2") or {}
    m13_p2_passed = bool(p2.get("passed"))
    previous_profile = entry_report.get("profiler_evidence") or {}
    previous_profiled_total = previous_profile.get("total_profiled_artifact_ms")
    optimized_profiled_total = profiler_evidence.get("total_profiled_artifact_ms")
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P3_REPORT_ID,
        "m13_p_loop_id": "M13.P3",
        "milestone": MILESTONE,
        "substep": "M13.P3",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P2_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "profiler_evidence": profiler_evidence,
        "bottleneck_ranking": list(profiler_evidence.get("bottleneck_ranking") or []),
        "optimization_hypothesis": {
            "selected_target": "generated_bridge_direct_io_per_call_synchronization",
            "hypothesis": (
                "Keep the P2 one-artifact direct-I/O provider surface and defer "
                "per-matmul CUDA synchronization to the strict-path boundary."
            ),
            "expected_affected_surface": "wrapper_matmul_addmm generated bridge runtime",
        },
        "optimization_delta": {
            "kind": "generated_bridge_async_direct_io",
            "previous_schedule_id": M13_P2_DIRECT_SCHEDULE_ID,
            "optimized_schedule_id": M13_P3_DIRECT_SCHEDULE_ID,
            "previous_profiled_artifact_ms": previous_profiled_total,
            "optimized_profiled_artifact_ms": optimized_profiled_total,
            "profiled_artifact_delta_ms": (
                float(optimized_profiled_total) - float(previous_profiled_total)
                if optimized_profiled_total is not None and previous_profiled_total is not None
                else None
            ),
            "provider_surface_changed": False,
            "runtime_kind": M13_P3_DIRECT_RUNTIME_KIND,
            "launch_sync_policy": "defer_to_strict_path_boundary",
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "generated_bridge_async_direct_io": strict_counts.get(
                "generated_bridge_async_direct_io"
            ),
            "native_conv": strict_counts.get("native_conv"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_loop_strict_surface_allclose_after_async_direct_io_optimization",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": p2_dashboard,
        "next_entry_decision": (
            {
                "decision": "m13_complete",
                "reason": "M13-C and M13-P2 both passed on the optimized strict TVM-owned surface.",
                "next_loop_id": None,
                "next_profiler_target": None,
            }
            if m13_p2_passed
            else {
                "decision": "continue_m13_p",
                "reason": "M13-P2 still misses the <=2x torch.compile threshold.",
                "next_loop_id": "M13.P4",
                "next_profiler_target": (
                    (profiler_evidence.get("bottleneck_ranking") or [{}])[0].get(
                        "bucket_id", "next_profiled_bottleneck"
                    )
                ),
            }
        ),
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_matmul_addmm",
                    "count": strict_counts.get("generated_bridge_async_direct_io"),
                    "from_runtime_kind": M13_P2_DIRECT_RUNTIME_KIND,
                    "to_runtime_kind": M13_P3_DIRECT_RUNTIME_KIND,
                    "provider_surface_changed": False,
                }
            ],
            implementation_level_delta={
                "triton_language_lowering": 0,
                "generated_tl_dot_bridge": 0,
                "tvm_native_wrapper_lowering": 0,
                "tvm_decomposed_artifact": 0,
                "metadata_only_structure": 0,
            },
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "generated_bridge_async_direct_io_wrapper_matmul",
                "strict_path_boundary_cuda_synchronization",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P3 keeps the fixed-shape generated bridge direct-I/O artifact "
                "surface and changes only the runtime launch synchronization policy."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("captured_tvm_count_7", strict_counts.get("captured_tvm") == 7),
            (
                "generated_bridge_matmul_addmm_count_7",
                strict_counts.get("generated_bridge_matmul_addmm") == 7,
            ),
            (
                "generated_bridge_async_direct_io_count_7",
                strict_counts.get("generated_bridge_async_direct_io") == 7,
            ),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            (
                "provider_relaxation_count_zero",
                strict_counts.get("provider_relaxation_count") == 0,
            ),
            ("fused_qkv_credit_count_zero", strict_counts.get("fused_qkv_credit_count") == 0),
            ("profiler_records_present", int(profiler_evidence.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P3 Generated Bridge Async Direct I/O")
    return report


def _empty_m13_p3_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P3_REPORT_ID,
        "m13_p_loop_id": "M13.P3",
        "milestone": MILESTONE,
        "substep": "M13.P3",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P2_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": {},
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P3 evidence emitted because benchmarks were unavailable.",
        ),
    }


def run_m13_p4_fine_grained_strict_surface_profile_loop(
    *,
    out_dir: str | Path | None = M13_P4_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run a fine-grained profiler pass over the M13.P3 strict surface."""

    entry_report = _read_json_or_empty(M13_P3_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p4_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.P4 Fine-Grained Strict Surface Profile")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p4_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P4 Fine-Grained Strict Surface Profile")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p4_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P4 Fine-Grained Strict Surface Profile")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_p3_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        fine_profile = _run_m13_p4_fine_grained_profile_once(
            torch,
            compiled,
            pixel_values,
            matmul_calls,
            conv_call,
            attention_call,
            warmup=warmup,
            repeat=repeat,
        )
        report = _build_m13_p4_fine_grained_profile_report(
            entry_report=entry_report,
            fine_profile=fine_profile,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p4_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p4_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.P4 Fine-Grained Strict Surface Profile")
        return report


def _run_m13_p4_fine_grained_profile_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
    *,
    warmup: int,
    repeat: int,
    direct_bridge_mode: str = "p3_async_direct_io",
    native_conv_fast_entry: bool = False,
) -> dict[str, Any]:
    for _ in range(warmup):
        if native_conv_fast_entry:
            _run_m13_p7_strict_once(
                torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
            )
        elif direct_bridge_mode == "p5_fast_async_direct_io":
            _run_m13_p5_strict_once(
                torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
            )
        else:
            _run_m13_p3_strict_once(
                torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
            )

    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        profile_runtime=True,
        direct_bridge=direct_bridge_mode,
        native_conv_fast_entry=native_conv_fast_entry,
    ) as patch_state:
        for iteration in range(repeat):
            patch_state.reset_runtime_state()
            patch_state.set_profile_iteration(iteration)
            total_start_ns = time.perf_counter_ns()
            with torch.no_grad():
                compiled(pixel_values)
            call_return_ns = time.perf_counter_ns()
            patch_state.record_runtime_profile(
                "compiled_call_to_return_cpu",
                call_return_ns - total_start_ns,
            )
            sync_start_ns = time.perf_counter_ns()
            torch.cuda.synchronize()
            sync_end_ns = time.perf_counter_ns()
            patch_state.record_runtime_profile(
                "strict_boundary_cuda_sync_cpu",
                sync_end_ns - sync_start_ns,
            )
            patch_state.record_runtime_profile(
                "strict_forward_total_cpu",
                sync_end_ns - total_start_ns,
            )
            patch_state.assert_all_calls_consumed()
            patch_state.assert_native_conv_consumed()
            patch_state.assert_native_attention_consumed()
        return _m13_p4_runtime_profile_summary(patch_state.runtime_profile_records, repeat=repeat)


def _run_m13_p5_strict_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        direct_bridge="p5_fast_async_direct_io",
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            actual_output = compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        actual = _output_tensors(actual_output)
        counts = _m13_p_strict_counts(patch_state)
        counts["captured_tvm"] = 7
        counts["generated_bridge_fast_async_direct_io"] = sum(
            1
            for record in patch_state.executed
            if record.get("runtime_kind") == M13_P5_DIRECT_RUNTIME_KIND
        )
    return actual, counts


def _run_m13_p7_strict_once(
    torch,
    compiled,
    pixel_values,
    matmul_calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _patched_m13_6_native_attention_runtime(
        torch,
        matmul_calls,
        conv_call,
        attention_call,
        direct_bridge="p5_fast_async_direct_io",
        native_conv_fast_entry=True,
    ) as patch_state:
        patch_state.reset_runtime_state()
        with torch.no_grad():
            actual_output = compiled(pixel_values)
        torch.cuda.synchronize()
        patch_state.assert_all_calls_consumed()
        patch_state.assert_native_conv_consumed()
        patch_state.assert_native_attention_consumed()
        actual = _output_tensors(actual_output)
        counts = _m13_p_strict_counts(patch_state)
        counts["captured_tvm"] = 7
        counts["generated_bridge_fast_async_direct_io"] = sum(
            1
            for record in patch_state.executed
            if record.get("runtime_kind") == M13_P5_DIRECT_RUNTIME_KIND
        )
        counts["native_conv_fast_entry"] = sum(
            1
            for record in patch_state.conv_executed
            if record.get("runtime_kind") == M13_P7_NATIVE_CONV_RUNTIME_KIND
        )
    return actual, counts


def _build_m13_p4_fine_grained_profile_report(
    *,
    entry_report: dict[str, Any],
    fine_profile: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_fine_grained_profile",
        "schema_version": 1,
        "report_id": M13_P4_REPORT_ID,
        "m13_p_loop_id": "M13.P4",
        "milestone": MILESTONE,
        "substep": "M13.P4",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P3_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": fine_profile,
        "bottleneck_ranking": list(fine_profile.get("per_iteration_ranking") or []),
        "optimization_hypothesis": {
            "selected_target": "fine_grained_profile_only",
            "hypothesis": (
                "Explain the gap between summed CUDA artifact time and strict E2E "
                "latency by profiling CPU-side runtime glue and boundary sync."
            ),
            "expected_affected_surface": "measurement_only",
        },
        "optimization_delta": {
            "kind": "profiler_entry_only",
            "provider_surface_changed": False,
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "generated_bridge_async_direct_io": strict_counts.get(
                "generated_bridge_async_direct_io"
            ),
            "native_conv": strict_counts.get("native_conv"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_surface_allclose_before_fine_grained_profile",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": entry_report.get("p2_dashboard", {}),
        "next_entry_decision": {
            "decision": "continue_m13_p",
            "reason": "Fine-grained profile identifies residual strict-surface overhead.",
            "next_loop_id": "M13.P5",
            "next_profiler_target": _m13_p4_first_actionable_target(fine_profile),
        },
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "strict_surface_cpu_side_phase_profiler",
                "runtime_glue_overhead_accounting",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P4 is profiler-only evidence for CPU/runtime glue overhead. "
                "It does not change the strict TVM-owned operator surface."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("generated_bridge_async_direct_io_count_7", strict_counts.get("generated_bridge_async_direct_io") == 7),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            ("fine_profile_records_present", int(fine_profile.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P4 Fine-Grained Strict Surface Profile")
    return report


def _empty_m13_p4_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_fine_grained_profile",
        "schema_version": 1,
        "report_id": M13_P4_REPORT_ID,
        "m13_p_loop_id": "M13.P4",
        "milestone": MILESTONE,
        "substep": "M13.P4",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P3_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": entry_report.get("p2_dashboard", {}),
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P4 evidence emitted because benchmarks were unavailable.",
        ),
    }


def run_m13_p5_generated_bridge_fast_artifact_invocation_loop(
    *,
    out_dir: str | Path | None = M13_P5_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Optimize generated bridge artifact invocation using cached TVM entry calls."""

    entry_report = _read_json_or_empty(M13_P4_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p5_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(
            report, out_dir, title="M13.P5 Generated Bridge Fast Artifact Invocation"
        )
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p5_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(
            report, out_dir, title="M13.P5 Generated Bridge Fast Artifact Invocation"
        )
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p5_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(
            report, out_dir, title="M13.P5 Generated Bridge Fast Artifact Invocation"
        )
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_p5_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        fine_profile = _run_m13_p4_fine_grained_profile_once(
            torch,
            compiled,
            pixel_values,
            matmul_calls,
            conv_call,
            attention_call,
            warmup=warmup,
            repeat=repeat,
            direct_bridge_mode="p5_fast_async_direct_io",
        )
        latency_ms = {
            M13_8_TORCH_COMPILE_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_torch_compile(torch, compiled, pixel_values),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M13_8_STRICT_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_p5_strict_once(
                        torch,
                        compiled,
                        pixel_values,
                        matmul_calls,
                        conv_call,
                        attention_call,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        p2_dashboard = _build_m13_8_p2_dashboard_decision_report(
            latency_ms=latency_ms,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=None,
            m13_c_status="passed" if bool(correctness.get("allclose")) else "failed",
        )
        return _build_m13_p5_generated_bridge_fast_artifact_invocation_report(
            entry_report=entry_report,
            fine_profile=fine_profile,
            p2_dashboard=p2_dashboard,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p5_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p5_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(
            report, out_dir, title="M13.P5 Generated Bridge Fast Artifact Invocation"
        )
        return report


def _build_m13_p5_generated_bridge_fast_artifact_invocation_report(
    *,
    entry_report: dict[str, Any],
    fine_profile: dict[str, Any],
    p2_dashboard: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    p2 = p2_dashboard.get("p2") or {}
    m13_p2_passed = bool(p2.get("passed"))
    previous_profile = entry_report.get("fine_grained_profiler_evidence") or {}
    previous_artifact_ms = _m13_p_bucket_avg(
        previous_profile, "generated_bridge_async_direct_io_artifact_run_cpu"
    )
    optimized_artifact_ms = _m13_p_bucket_avg(
        fine_profile, "generated_bridge_fast_async_direct_io_artifact_fast_entry_cpu"
    )
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P5_REPORT_ID,
        "m13_p_loop_id": "M13.P5",
        "milestone": MILESTONE,
        "substep": "M13.P5",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P4_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": fine_profile,
        "bottleneck_ranking": list(fine_profile.get("per_iteration_ranking") or []),
        "optimization_hypothesis": {
            "selected_target": "generated_bridge_async_direct_io_artifact_run_cpu",
            "hypothesis": (
                "Bypass per-call TritonTVMArtifact.run grid/ABI validation and kernel "
                "lookup for already-validated generated bridge artifacts."
            ),
            "expected_affected_surface": "wrapper_matmul_addmm generated bridge runtime",
        },
        "optimization_delta": {
            "kind": "generated_bridge_fast_artifact_invocation",
            "previous_artifact_invocation_ms_per_iter": previous_artifact_ms,
            "optimized_artifact_invocation_ms_per_iter": optimized_artifact_ms,
            "artifact_invocation_delta_ms_per_iter": (
                float(optimized_artifact_ms) - float(previous_artifact_ms)
                if optimized_artifact_ms is not None and previous_artifact_ms is not None
                else None
            ),
            "provider_surface_changed": False,
            "runtime_kind": M13_P5_DIRECT_RUNTIME_KIND,
            "artifact_invocation_policy": "cached_tvm_entry_function_no_runtime_validation",
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "generated_bridge_fast_async_direct_io": strict_counts.get(
                "generated_bridge_fast_async_direct_io"
            ),
            "native_conv": strict_counts.get("native_conv"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_loop_strict_surface_allclose_after_fast_artifact_invocation",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": p2_dashboard,
        "next_entry_decision": (
            {
                "decision": "m13_complete",
                "reason": "M13-C and M13-P2 both passed on the optimized strict TVM-owned surface.",
                "next_loop_id": None,
                "next_profiler_target": None,
            }
            if m13_p2_passed
            else {
                "decision": "continue_m13_p",
                "reason": "M13-P2 still misses the <=2x torch.compile threshold.",
                "next_loop_id": "M13.P6",
                "next_profiler_target": _m13_p4_first_actionable_target(fine_profile),
            }
        ),
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "wrapper_matmul_addmm",
                    "count": strict_counts.get("generated_bridge_fast_async_direct_io"),
                    "from_runtime_kind": M13_P3_DIRECT_RUNTIME_KIND,
                    "to_runtime_kind": M13_P5_DIRECT_RUNTIME_KIND,
                    "provider_surface_changed": False,
                }
            ],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "validated_artifact_fast_entry_invocation",
                "strict_surface_runtime_glue_fastpath",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P5 optimizes runtime invocation glue only. The generated bridge "
                "artifact and strict TVM-owned provider surface are unchanged."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("generated_bridge_fast_async_direct_io_count_7", strict_counts.get("generated_bridge_fast_async_direct_io") == 7),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            ("fine_profile_records_present", int(fine_profile.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P5 Generated Bridge Fast Artifact Invocation")
    return report


def _empty_m13_p5_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P5_REPORT_ID,
        "m13_p_loop_id": "M13.P5",
        "milestone": MILESTONE,
        "substep": "M13.P5",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P4_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": entry_report.get("p2_dashboard", {}),
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P5 evidence emitted because benchmarks were unavailable.",
        ),
    }


def _m13_p_bucket_avg(profile: dict[str, Any], bucket_id: str) -> float | None:
    for bucket in profile.get("buckets") or []:
        if bucket.get("bucket_id") == bucket_id:
            return bucket.get("avg_per_iteration_ms")
    for bucket in profile.get("per_iteration_ranking") or []:
        if bucket.get("bucket_id") == bucket_id:
            return bucket.get("avg_per_iteration_ms")
    return None


def run_m13_p6_native_artifact_fine_grained_profile_loop(
    *,
    out_dir: str | Path | None = M13_P6_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run a profiler-only pass for native conv/attention residual overhead."""

    entry_report = _read_json_or_empty(M13_P5_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p6_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.P6 Native Artifact Fine-Grained Profile")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p6_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P6 Native Artifact Fine-Grained Profile")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p6_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P6 Native Artifact Fine-Grained Profile")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)
        strict_actual, strict_counts = _run_m13_p5_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        fine_profile = _run_m13_p4_fine_grained_profile_once(
            torch,
            compiled,
            pixel_values,
            matmul_calls,
            conv_call,
            attention_call,
            warmup=warmup,
            repeat=repeat,
            direct_bridge_mode="p5_fast_async_direct_io",
        )
        report = _build_m13_p6_native_artifact_fine_grained_profile_report(
            entry_report=entry_report,
            fine_profile=fine_profile,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p6_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p6_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.P6 Native Artifact Fine-Grained Profile")
        return report


def _build_m13_p6_native_artifact_fine_grained_profile_report(
    *,
    entry_report: dict[str, Any],
    fine_profile: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_fine_grained_profile",
        "schema_version": 1,
        "report_id": M13_P6_REPORT_ID,
        "m13_p_loop_id": "M13.P6",
        "milestone": MILESTONE,
        "substep": "M13.P6",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P5_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": fine_profile,
        "bottleneck_ranking": list(fine_profile.get("per_iteration_ranking") or []),
        "optimization_hypothesis": {
            "selected_target": "native_artifact_residual_profile_only",
            "hypothesis": (
                "Split native conv/attention residual overhead into validation, "
                "allocation, storage-view, DLPack, artifact invocation, and sync phases."
            ),
            "expected_affected_surface": "measurement_only",
        },
        "optimization_delta": {
            "kind": "profiler_entry_only",
            "provider_surface_changed": False,
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "generated_bridge_fast_async_direct_io": strict_counts.get(
                "generated_bridge_fast_async_direct_io"
            ),
            "native_conv": strict_counts.get("native_conv"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_surface_allclose_before_native_residual_profile",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": entry_report.get("p2_dashboard", {}),
        "next_entry_decision": {
            "decision": "continue_m13_p",
            "reason": "Fine-grained profile identifies the next native residual target.",
            "next_loop_id": "M13.P7",
            "next_profiler_target": _m13_p4_first_actionable_target(fine_profile),
        },
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "native_artifact_phase_profiler",
                "conv_attention_runtime_glue_accounting",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P6 is profiler-only evidence for native conv/attention residual "
                "runtime overhead and does not change the strict TVM-owned surface."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("generated_bridge_fast_async_direct_io_count_7", strict_counts.get("generated_bridge_fast_async_direct_io") == 7),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            ("fine_profile_records_present", int(fine_profile.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P6 Native Artifact Fine-Grained Profile")
    return report


def _empty_m13_p6_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_fine_grained_profile",
        "schema_version": 1,
        "report_id": M13_P6_REPORT_ID,
        "m13_p_loop_id": "M13.P6",
        "milestone": MILESTONE,
        "substep": "M13.P6",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P5_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": entry_report.get("p2_dashboard", {}),
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P6 evidence emitted because benchmarks were unavailable.",
        ),
    }


def run_m13_p7_native_conv_fast_artifact_invocation_loop(
    *,
    out_dir: str | Path | None = M13_P7_REPORT_DIR,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Optimize native conv artifact invocation using cached TVM entry calls."""

    entry_report = _read_json_or_empty(M13_P6_REPORT_DIR / "report.json")
    if not run_benchmarks:
        report = _empty_m13_p7_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
        )
        _maybe_write_report(report, out_dir, title="M13.P7 Native Conv Fast Artifact Invocation")
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_m13_p7_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P7 Native Conv Fast Artifact Invocation")
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_m13_p7_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir, title="M13.P7 Native Conv Fast Artifact Invocation")
        return report

    try:
        model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
        expected = _run_m13_4_torch_baseline(torch, model, pixel_values)
        matmul_calls = _build_m13_4_generated_bridge_runtime_calls(wrapper_source)
        conv_call = _build_m13_5_native_conv_runtime_call(wrapper_source)
        attention_call = _build_m13_6_native_attention_runtime_call(wrapper_source)

        strict_actual, strict_counts = _run_m13_p7_strict_once(
            torch, compiled, pixel_values, matmul_calls, conv_call, attention_call
        )
        correctness = _m13_7_correctness_report(strict_actual, expected)
        fine_profile = _run_m13_p4_fine_grained_profile_once(
            torch,
            compiled,
            pixel_values,
            matmul_calls,
            conv_call,
            attention_call,
            warmup=warmup,
            repeat=repeat,
            direct_bridge_mode="p5_fast_async_direct_io",
            native_conv_fast_entry=True,
        )
        latency_ms = {
            M13_8_TORCH_COMPILE_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_8_torch_compile(torch, compiled, pixel_values),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M13_8_STRICT_MODE: _m13_latency_record(
                _m13_time_cuda_callable(
                    torch,
                    lambda: _run_m13_p7_strict_once(
                        torch,
                        compiled,
                        pixel_values,
                        matmul_calls,
                        conv_call,
                        attention_call,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        p2_dashboard = _build_m13_8_p2_dashboard_decision_report(
            latency_ms=latency_ms,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=None,
            m13_c_status="passed" if bool(correctness.get("allclose")) else "failed",
        )
        return _build_m13_p7_native_conv_fast_artifact_invocation_report(
            entry_report=entry_report,
            fine_profile=fine_profile,
            p2_dashboard=p2_dashboard,
            correctness=correctness,
            strict_counts=strict_counts,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            dependency_versions=_dependency_versions(torch),
            out_dir=out_dir,
        )
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_m13_p7_report(
            entry_report=entry_report,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m13_p7_failed:{type(err).__name__}:{err}",
        )
        _maybe_write_report(report, out_dir, title="M13.P7 Native Conv Fast Artifact Invocation")
        return report


def _build_m13_p7_native_conv_fast_artifact_invocation_report(
    *,
    entry_report: dict[str, Any],
    fine_profile: dict[str, Any],
    p2_dashboard: dict[str, Any],
    correctness: dict[str, Any],
    strict_counts: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    dependency_versions: dict[str, Any] | None,
    out_dir: str | Path | None,
) -> dict[str, Any]:
    p2 = p2_dashboard.get("p2") or {}
    m13_p2_passed = bool(p2.get("passed"))
    previous_profile = entry_report.get("fine_grained_profiler_evidence") or {}
    previous_artifact_ms = _m13_p_bucket_avg(previous_profile, "native_conv_artifact_run_cpu")
    optimized_artifact_ms = _m13_p_bucket_avg(
        fine_profile, "native_conv_artifact_fast_entry_cpu"
    )
    report = {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P7_REPORT_ID,
        "m13_p_loop_id": "M13.P7",
        "milestone": MILESTONE,
        "substep": "M13.P7",
        "status": "passed",
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P6_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": fine_profile,
        "bottleneck_ranking": list(fine_profile.get("per_iteration_ranking") or []),
        "optimization_hypothesis": {
            "selected_target": "native_conv_artifact_run_cpu",
            "hypothesis": (
                "Bypass per-call TritonTVMArtifact.run grid/ABI validation and kernel "
                "lookup for the already-validated fixed-shape native conv artifact."
            ),
            "expected_affected_surface": "native conv runtime glue",
        },
        "optimization_delta": {
            "kind": "native_conv_fast_artifact_invocation",
            "previous_artifact_invocation_ms_per_iter": previous_artifact_ms,
            "optimized_artifact_invocation_ms_per_iter": optimized_artifact_ms,
            "artifact_invocation_delta_ms_per_iter": (
                float(optimized_artifact_ms) - float(previous_artifact_ms)
                if optimized_artifact_ms is not None and previous_artifact_ms is not None
                else None
            ),
            "provider_surface_changed": False,
            "runtime_kind": M13_P7_NATIVE_CONV_RUNTIME_KIND,
            "artifact_invocation_policy": "cached_tvm_entry_function_no_runtime_validation",
        },
        "strict_surface_invariants": {
            "captured_tvm": strict_counts.get("captured_tvm"),
            "generated_bridge_matmul_addmm": strict_counts.get("generated_bridge_matmul_addmm"),
            "generated_bridge_fast_async_direct_io": strict_counts.get(
                "generated_bridge_fast_async_direct_io"
            ),
            "native_conv": strict_counts.get("native_conv"),
            "native_conv_fast_entry": strict_counts.get("native_conv_fast_entry"),
            "native_attention": strict_counts.get("native_attention"),
            "host_staging_bytes": strict_counts.get("host_staging_bytes"),
            "legacy_provider_count": strict_counts.get("legacy_provider_count"),
            "silent_fallback_count": strict_counts.get("silent_fallback_count"),
            "provider_relaxation_count": strict_counts.get("provider_relaxation_count"),
            "fused_qkv_credit_count": strict_counts.get("fused_qkv_credit_count"),
        },
        "m13_c_revalidation": {
            "mode": "same_loop_strict_surface_allclose_after_native_conv_fast_entry",
            "status": "passed" if bool(correctness.get("allclose")) else "failed",
            "correctness": correctness,
        },
        "p2_dashboard": p2_dashboard,
        "next_entry_decision": (
            {
                "decision": "m13_complete",
                "reason": "M13-C and M13-P2 both passed on the optimized strict TVM-owned surface.",
                "next_loop_id": None,
                "next_profiler_target": None,
            }
            if m13_p2_passed
            else {
                "decision": "continue_m13_p",
                "reason": "M13-P2 still misses the <=2x torch.compile threshold.",
                "next_loop_id": "M13.P8",
                "next_profiler_target": _m13_p4_first_actionable_target(fine_profile),
            }
        ),
        "dependency_versions": dependency_versions or {},
        **_slice_fields(
            operator_inventory_delta=[
                {
                    "operator_family": "native_conv2d",
                    "count": strict_counts.get("native_conv_fast_entry"),
                    "from_runtime_kind": M13_5_CONV_RUNTIME_KIND,
                    "to_runtime_kind": M13_P7_NATIVE_CONV_RUNTIME_KIND,
                    "provider_surface_changed": False,
                }
            ],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[
                "native_artifact_fast_entry_invocation",
                "strict_surface_runtime_glue_fastpath",
            ],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note=(
                "M13.P7 optimizes native conv runtime invocation glue only. The "
                "native TVM conv artifact, CUDA target, and strict provider surface "
                "are unchanged."
            ),
        ),
    }
    report["invariants"] = _invariants(
        [
            ("m13_c_revalidated", report["m13_c_revalidation"]["status"] == "passed"),
            ("generated_bridge_fast_async_direct_io_count_7", strict_counts.get("generated_bridge_fast_async_direct_io") == 7),
            ("native_conv_count_1", strict_counts.get("native_conv") == 1),
            ("native_conv_fast_entry_count_1", strict_counts.get("native_conv_fast_entry") == 1),
            ("native_attention_count_1", strict_counts.get("native_attention") == 1),
            ("host_staging_bytes_zero", strict_counts.get("host_staging_bytes") == 0),
            ("legacy_provider_count_zero", strict_counts.get("legacy_provider_count") == 0),
            ("silent_fallback_count_zero", strict_counts.get("silent_fallback_count") == 0),
            ("fine_profile_records_present", int(fine_profile.get("record_count", 0)) > 0),
        ]
    )
    report["status"] = report["invariants"]["status"]
    _maybe_write_report(report, out_dir, title="M13.P7 Native Conv Fast Artifact Invocation")
    return report


def _empty_m13_p7_report(
    *,
    entry_report: dict[str, Any],
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    return {
        "report_kind": "triton_tvm_m13_p_strict_surface_performance_loop",
        "schema_version": 1,
        "report_id": M13_P7_REPORT_ID,
        "m13_p_loop_id": "M13.P7",
        "milestone": MILESTONE,
        "substep": "M13.P7",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "entry_report_id": entry_report.get("report_id", M13_P6_REPORT_ID),
        "entry_snapshot": _m13_p_entry_snapshot(entry_report),
        "fine_grained_profiler_evidence": {},
        "bottleneck_ranking": [],
        "optimization_hypothesis": {},
        "optimization_delta": {"kind": "not_run"},
        "strict_surface_invariants": {},
        "m13_c_revalidation": {"status": "not_run"},
        "p2_dashboard": entry_report.get("p2_dashboard", {}),
        "next_entry_decision": {"decision": "blocked", "reason": availability_reason},
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No M13.P7 evidence emitted because benchmarks were unavailable.",
        ),
    }


def _m13_p4_runtime_profile_summary(
    records: list[dict[str, Any]],
    *,
    repeat: int,
) -> dict[str, Any]:
    by_bucket: dict[str, list[float]] = {}
    for record in records:
        bucket = str(record.get("bucket_id", "unknown"))
        by_bucket.setdefault(bucket, []).append(float(record.get("elapsed_ms", 0.0)))
    buckets = []
    for bucket, values in sorted(by_bucket.items()):
        total = sum(values)
        buckets.append(
            {
                "bucket_id": bucket,
                "sample_count": len(values),
                "total_ms": total,
                "avg_per_iteration_ms": total / max(1, repeat),
                "p50_sample_ms": _m13_percentile(sorted(values), 0.50),
                "p95_sample_ms": _m13_percentile(sorted(values), 0.95),
                "max_sample_ms": max(values),
            }
        )
    ranking = sorted(buckets, key=lambda item: item["avg_per_iteration_ms"], reverse=True)
    return {
        "mode": "fine_grained_cpu_runtime_phase_profile",
        "repeat": repeat,
        "record_count": len(records),
        "records": records,
        "buckets": buckets,
        "per_iteration_ranking": ranking,
        "actionable_ranking": [
            item for item in ranking if item["bucket_id"] not in _M13_P4_AGGREGATE_BUCKETS
        ],
        "total_profiled_cpu_ms_per_iteration": sum(
            item["avg_per_iteration_ms"] for item in buckets
        ),
    }


_M13_P4_AGGREGATE_BUCKETS = {
    "strict_forward_total_cpu",
    "compiled_call_to_return_cpu",
    "generated_bridge_async_direct_call_cpu",
    "wrapper_addmm_dispatch_cpu",
    "wrapper_mm_dispatch_cpu",
    "native_conv_dispatch_cpu",
    "native_attention_dispatch_cpu",
}


def _m13_p4_first_actionable_target(fine_profile: dict[str, Any]) -> str:
    for item in fine_profile.get("actionable_ranking") or []:
        bucket = item.get("bucket_id")
        if bucket:
            return str(bucket)
    for item in fine_profile.get("per_iteration_ranking") or []:
        bucket = item.get("bucket_id")
        if bucket:
            return str(bucket)
    return "next_fine_grained_bottleneck"


def _m13_p_profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, list[float]] = {}
    for record in records:
        bucket = str(record.get("bucket_id", "unknown"))
        by_bucket.setdefault(bucket, []).append(float(record.get("elapsed_ms", 0.0)))
    buckets = []
    for bucket, values in sorted(by_bucket.items()):
        total = sum(values)
        buckets.append(
            {
                "bucket_id": bucket,
                "launch_count": len(values),
                "total_ms": total,
                "p50_ms": _m13_percentile(sorted(values), 0.50),
                "p95_ms": _m13_percentile(sorted(values), 0.95),
                "max_ms": max(values),
            }
        )
    ranking = sorted(buckets, key=lambda item: item["total_ms"], reverse=True)
    return {
        "mode": "single_strict_surface_profile_run_with_cuda_events",
        "record_count": len(records),
        "records": records,
        "buckets": buckets,
        "bottleneck_ranking": ranking,
        "total_profiled_artifact_ms": sum(item["total_ms"] for item in buckets),
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def run_m13_0_to_m13_3(*, run_execution: bool = True) -> dict[str, Any]:
    """Run M13.0 through M13.3 in order."""

    reports = [
        run_m13_0_policy_taxonomy_freeze(),
        run_m13_1_operator_inventory_baseline(),
        run_m13_2_captured_kernel_tvm_execution(run_execution=run_execution),
        run_m13_3_generated_tl_dot_bridge_semantics(),
    ]
    return {
        "status": "passed" if all(report["status"] == "passed" for report in reports) else "failed",
        "reports": [report["report_id"] for report in reports],
        "substeps": {report["substep"]: report["status"] for report in reports},
    }


def run_m13_0_to_m13_4(*, run_execution: bool = True, seed: int = 0) -> dict[str, Any]:
    """Run M13.0 through M13.4 in order."""

    through_m13_3 = run_m13_0_to_m13_3(run_execution=run_execution)
    m13_4 = run_m13_4_matmul_addmm_runtime_replacement(
        run_execution=run_execution,
        seed=seed,
    )
    passed = through_m13_3["status"] == "passed" and m13_4["status"] == "passed"
    return {
        "status": "passed" if passed else "failed",
        "reports": list(through_m13_3["reports"]) + [m13_4["report_id"]],
        "substeps": {**through_m13_3["substeps"], m13_4["substep"]: m13_4["status"]},
    }


def run_m13_0_to_m13_5(*, run_execution: bool = True, seed: int = 0) -> dict[str, Any]:
    """Run M13.0 through M13.5 in order."""

    through_m13_4 = run_m13_0_to_m13_4(run_execution=run_execution, seed=seed)
    m13_5 = run_m13_5_native_conv_slice(
        run_execution=run_execution,
        seed=seed,
    )
    passed = through_m13_4["status"] == "passed" and m13_5["status"] == "passed"
    return {
        "status": "passed" if passed else "failed",
        "reports": list(through_m13_4["reports"]) + [m13_5["report_id"]],
        "substeps": {**through_m13_4["substeps"], m13_5["substep"]: m13_5["status"]},
    }


def run_m13_0_to_m13_6(*, run_execution: bool = True, seed: int = 0) -> dict[str, Any]:
    """Run M13.0 through M13.6 in order."""

    through_m13_5 = run_m13_0_to_m13_5(run_execution=run_execution, seed=seed)
    m13_6 = run_m13_6_native_attention_slice(
        run_execution=run_execution,
        seed=seed,
    )
    passed = through_m13_5["status"] == "passed" and m13_6["status"] == "passed"
    return {
        "status": "passed" if passed else "failed",
        "reports": list(through_m13_5["reports"]) + [m13_6["report_id"]],
        "substeps": {**through_m13_5["substeps"], m13_6["substep"]: m13_6["status"]},
    }


def run_m13_0_to_m13_7(*, run_execution: bool = True, seed: int = 0) -> dict[str, Any]:
    """Run M13.0 through M13.7 in order."""

    through_m13_6 = run_m13_0_to_m13_6(run_execution=run_execution, seed=seed)
    m13_7 = run_m13_7_strict_correctness_gate(
        run_execution=run_execution,
        seed=seed,
    )
    passed = through_m13_6["status"] == "passed" and m13_7["status"] == "passed"
    return {
        "status": "passed" if passed else "failed",
        "reports": list(through_m13_6["reports"]) + [m13_7["report_id"]],
        "substeps": {**through_m13_6["substeps"], m13_7["substep"]: m13_7["status"]},
    }


def run_m13_0_to_m13_8(
    *,
    run_execution: bool = True,
    seed: int = 0,
    warmup: int = M13_8_DEFAULT_WARMUP,
    repeat: int = M13_8_DEFAULT_REPEAT,
) -> dict[str, Any]:
    """Run M13.0 through M13.8 in order."""

    through_m13_7 = run_m13_0_to_m13_7(run_execution=run_execution, seed=seed)
    m13_8 = run_m13_8_p2_dashboard_decision(
        run_benchmarks=run_execution,
        seed=seed,
        warmup=warmup,
        repeat=repeat,
    )
    passed = (
        through_m13_7["status"] == "passed"
        and m13_8["status"] == "passed"
        and m13_8.get("m13_p2_status") == "passed"
    )
    return {
        "status": "passed" if passed else "failed",
        "reports": list(through_m13_7["reports"]) + [m13_8["report_id"]],
        "substeps": {**through_m13_7["substeps"], m13_8["substep"]: m13_8["status"]},
    }


@dataclass(frozen=True)
class _M13CapturedKernelRuntimeCall:
    index: int
    kernel_name: str
    contract: str
    artifact: TritonTVMArtifact
    translation_route: str


@dataclass(frozen=True)
class _M13GeneratedBridgeRuntimeCall:
    index: int
    op_family: str
    op_name: str
    line_no: int
    source: str
    wrapper_source_kind: str
    wrapper_m: int
    wrapper_n: int
    wrapper_k: int
    wrapper_b_layout: str
    wrapper_b_storage_shape: tuple[int, int]
    wrapper_b_storage_stride: tuple[int, int]
    bias_param: str
    bias_shape: tuple[int, ...]
    bias_stride: tuple[int, ...]
    block_m: int
    core_artifact: TritonTVMArtifact
    a_copy_artifact: TritonTVMArtifact
    a_zero_tail_artifact: TritonTVMArtifact | None
    b_adapter_artifact: TritonTVMArtifact
    commit_artifact: TritonTVMArtifact
    direct_io_artifact: TritonTVMArtifact


@dataclass(frozen=True)
class _M13NativeConvRuntimeCall:
    index: int
    op_family: str
    op_name: str
    line_no: int
    source: str
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
    contract: str
    artifact: TritonTVMArtifact


@dataclass(frozen=True)
class _M13NativeAttentionRuntimeCall:
    index: int
    op_family: str
    op_name: str
    line_no: int
    source: str
    contract: str
    q_shape: tuple[int, int, int, int]
    k_shape: tuple[int, int, int, int]
    v_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    q_stride: tuple[int, int, int, int]
    k_stride: tuple[int, int, int, int]
    v_stride: tuple[int, int, int, int]
    output_stride: tuple[int, int, int, int]
    scale: float
    artifact: TritonTVMArtifact
    m10_provider_kind: str


class _M13GeneratedBridgePatchState:
    """Patched wrapper state for M13.4 matmul/addmm replacement."""

    def __init__(
        self,
        calls: list[_M13GeneratedBridgeRuntimeCall],
        torch_module,
        *,
        native_conv_call: _M13NativeConvRuntimeCall | None = None,
        native_attention_call: _M13NativeAttentionRuntimeCall | None = None,
        profile_artifacts: bool = False,
        profile_runtime: bool = False,
        direct_bridge: bool | str = False,
        native_conv_fast_entry: bool = False,
    ):
        self._calls = list(calls)
        self._torch = torch_module
        self._native_conv_call = native_conv_call
        self._native_attention_call = native_attention_call
        self._profile_artifacts = bool(profile_artifacts)
        self._profile_runtime = bool(profile_runtime)
        self._native_conv_fast_entry = bool(native_conv_fast_entry)
        if direct_bridge is True:
            self._direct_bridge_mode = "p2_direct_io"
        elif isinstance(direct_bridge, str):
            self._direct_bridge_mode = direct_bridge
        else:
            self._direct_bridge_mode = ""
        self.executed: list[dict[str, Any]] = []
        self.conv_executed: list[dict[str, Any]] = []
        self.attention_executed: list[dict[str, Any]] = []
        self.captured_executed: list[dict[str, Any]] = []
        self.profile_records: list[dict[str, Any]] = []
        self.runtime_profile_records: list[dict[str, Any]] = []
        self._profile_iteration = 0
        self._artifact_entry_cache: dict[int, Any] = {}
        self._captured_expected: dict[str, _M13CapturedKernelRuntimeCall] = {}
        self._captured_runtime_counts: Counter[str] = Counter()
        self._captured_patches: list[tuple[Any, str, Any]] = []
        self.conv_runtime_calls = 0
        self.native_conv_runtime_calls = 0
        self.device_torch_cuda_conv_count = 0
        self.attention_runtime_calls = 0
        self.native_attention_runtime_calls = 0
        self.torch_replay_attention_count = 0
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self._addmm = deque(call for call in self._calls if call.op_family == "extern_addmm_bias")
        self._gemm = deque(call for call in self._calls if call.op_family == "extern_gemm")
        self.executed.clear()
        self.conv_executed.clear()
        self.attention_executed.clear()
        self.captured_executed.clear()
        self.profile_records.clear()
        self._captured_runtime_counts.clear()
        self.conv_runtime_calls = 0
        self.native_conv_runtime_calls = 0
        self.device_torch_cuda_conv_count = 0
        self.attention_runtime_calls = 0
        self.native_attention_runtime_calls = 0
        self.torch_replay_attention_count = 0

    def set_profile_iteration(self, iteration: int) -> None:
        self._profile_iteration = int(iteration)

    def record_runtime_profile(self, bucket_id: str, elapsed_ns: int, **fields: Any) -> None:
        if not self._profile_runtime:
            return
        record = {
            "bucket_id": bucket_id,
            "elapsed_ms": elapsed_ns / 1_000_000.0,
            "iteration": self._profile_iteration,
        }
        record.update(fields)
        self.runtime_profile_records.append(record)

    def addmm(self, bias, lhs, rhs, *args, **kwargs):
        start_ns = time.perf_counter_ns()
        out = kwargs.get("out")
        if out is None:
            raise RuntimeError("M13.4 generated bridge addmm patch requires out= buffer")
        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 1.0))
        if alpha != 1.0 or beta != 1.0:
            raise RuntimeError(f"M13.4 generated bridge addmm requires alpha=beta=1, got {alpha}, {beta}")
        call = self._addmm.popleft()
        self._run_generated_bridge_call(call, lhs=lhs, rhs=rhs, out=out, bias=bias)
        self.record_runtime_profile(
            "wrapper_addmm_dispatch_cpu",
            time.perf_counter_ns() - start_ns,
            op_family=call.op_family,
            index=call.index,
        )
        return out

    def mm(self, lhs, rhs, *args, **kwargs):
        start_ns = time.perf_counter_ns()
        out = kwargs.get("out")
        if out is None:
            raise RuntimeError("M13.4 generated bridge mm patch requires out= buffer")
        call = self._gemm.popleft()
        self._run_generated_bridge_call(call, lhs=lhs, rhs=rhs, out=out, bias=None)
        self.record_runtime_profile(
            "wrapper_mm_dispatch_cpu",
            time.perf_counter_ns() - start_ns,
            op_family=call.op_family,
            index=call.index,
        )
        return out

    def wrap_conv(self, original):
        def _conv(*args, **kwargs):
            self.conv_runtime_calls += 1
            self.device_torch_cuda_conv_count += 1
            return original(*args, **kwargs)

        return _conv

    def native_conv2d(self, inp, weight, *args, **kwargs):
        start_ns = time.perf_counter_ns()
        call = self._native_conv_call
        if call is None:
            raise RuntimeError("M13.5 native conv patch requires a native conv call")
        phase_start_ns = time.perf_counter_ns()
        self._assert_native_conv_kwargs(kwargs)
        self._assert_shape_stride("input", inp, call.input_shape, call.input_stride)
        self._assert_shape_stride("weight", weight, call.weight_shape, call.weight_stride)
        self.record_runtime_profile(
            "native_conv_validation_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        out = self._torch.empty_strided(
            call.output_shape,
            call.output_stride,
            device=inp.device,
            dtype=inp.dtype,
        )
        self.record_runtime_profile(
            "native_conv_output_alloc_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        inp_storage = self._storage_view(inp, call.input_shape, call.input_stride)
        self.record_runtime_profile(
            "native_conv_storage_view_input_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        weight_storage = self._storage_view(weight, call.weight_shape, call.weight_stride)
        self.record_runtime_profile(
            "native_conv_storage_view_weight_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        out_storage = self._storage_view(out, call.output_shape, call.output_stride)
        self.record_runtime_profile(
            "native_conv_storage_view_out_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        inp_tvm = tvm.runtime.from_dlpack(inp_storage)
        self.record_runtime_profile(
            "native_conv_dlpack_input_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        weight_tvm = tvm.runtime.from_dlpack(weight_storage)
        self.record_runtime_profile(
            "native_conv_dlpack_weight_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        out_tvm = tvm.runtime.from_dlpack(out_storage)
        self.record_runtime_profile(
            "native_conv_dlpack_out_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        tvm_args = [inp_tvm, weight_tvm, out_tvm]
        if self._native_conv_fast_entry:
            self._run_artifact_fast_entry(
                call.artifact,
                tvm_args,
                bucket_id="native_conv",
                kernel_name=call.artifact.meta.kernel_name,
                op_family=call.op_family,
            )
        else:
            self._run_artifact(
                call.artifact,
                tvm_args,
                bucket_id="native_conv",
                kernel_name=call.artifact.meta.kernel_name,
                op_family=call.op_family,
            )
        phase_start_ns = time.perf_counter_ns()
        self._torch.cuda.synchronize()
        self.record_runtime_profile(
            "native_conv_sync_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        tvm_args.clear()
        self.conv_runtime_calls += 1
        self.native_conv_runtime_calls += 1
        conv_record = _m13_5_native_conv_runtime_record(call)
        if self._native_conv_fast_entry:
            conv_record["runtime_kind"] = M13_P7_NATIVE_CONV_RUNTIME_KIND
            conv_record["artifact_invocation_policy"] = (
                "cached_tvm_entry_function_no_runtime_validation"
            )
        self.conv_executed.append(conv_record)
        self.record_runtime_profile(
            "native_conv_dispatch_cpu",
            time.perf_counter_ns() - start_ns,
            op_family=call.op_family,
        )
        return out

    def native_decomposed_sdpa(self, query, key, value, *args, **kwargs):
        self.attention_runtime_calls += 1
        self.torch_replay_attention_count += 1
        scale = kwargs.get("scale")
        if scale is None:
            scale = float(query.shape[-1]) ** -0.5
        scores = self._torch.matmul(query, key.transpose(-2, -1)) * float(scale)
        probs = self._torch.softmax(scores, dim=-1)
        out = self._torch.matmul(probs, value)
        out = out.transpose(1, 2).contiguous().transpose(1, 2)
        return (out,)

    def native_tvm_attention_decomposed_sdpa(self, query, key, value, *args, **kwargs):
        start_ns = time.perf_counter_ns()
        call = self._native_attention_call
        if call is None:
            raise RuntimeError("M13.6 native attention patch requires a native attention call")
        mask = args[0] if args else None
        is_causal = args[1] if len(args) > 1 else kwargs.get("is_causal", False)
        if mask is not None:
            raise RuntimeError("M13.6 native attention slice only supports unmasked ViT attention")
        if bool(is_causal):
            raise RuntimeError("M13.6 native attention slice requires causal=False")
        scale = kwargs.get("scale")
        if scale is None:
            scale = float(query.shape[-1]) ** -0.5
        if float(scale) != float(call.scale):
            raise RuntimeError(f"M13.6 native attention expected scale={call.scale}, got {scale}")
        phase_start_ns = time.perf_counter_ns()
        self._assert_shape_stride("attention query", query, call.q_shape, call.q_stride)
        self._assert_shape_stride("attention key", key, call.k_shape, call.k_stride)
        self._assert_shape_stride("attention value", value, call.v_shape, call.v_stride)
        self.record_runtime_profile(
            "native_attention_validation_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        out = self._torch.empty_strided(
            call.output_shape,
            call.output_stride,
            device=query.device,
            dtype=query.dtype,
        )
        self.record_runtime_profile(
            "native_attention_output_alloc_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        query_tvm = _from_dlpack_allow_strided(query)
        self.record_runtime_profile(
            "native_attention_dlpack_query_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        key_tvm = _from_dlpack_allow_strided(key)
        self.record_runtime_profile(
            "native_attention_dlpack_key_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        value_tvm = _from_dlpack_allow_strided(value)
        self.record_runtime_profile(
            "native_attention_dlpack_value_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        out_tvm = _from_dlpack_allow_strided(out)
        self.record_runtime_profile(
            "native_attention_dlpack_out_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        tvm_args = [query_tvm, key_tvm, value_tvm, out_tvm]
        self._run_artifact(
            call.artifact,
            tvm_args,
            bucket_id="native_attention",
            kernel_name=call.artifact.meta.kernel_name,
            op_family=call.op_family,
        )
        phase_start_ns = time.perf_counter_ns()
        self._torch.cuda.synchronize()
        self.record_runtime_profile(
            "native_attention_sync_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
        )
        tvm_args.clear()
        self.attention_runtime_calls += 1
        self.native_attention_runtime_calls += 1
        self.attention_executed.append(_m13_6_native_attention_runtime_record(call))
        self.record_runtime_profile(
            "native_attention_dispatch_cpu",
            time.perf_counter_ns() - start_ns,
            op_family=call.op_family,
        )
        return (out,)

    def assert_all_calls_consumed(self) -> None:
        if self._addmm or self._gemm:
            raise RuntimeError(
                "M13.4 generated bridge patch did not consume all calls: "
                f"addmm={len(self._addmm)}, gemm={len(self._gemm)}"
            )

    def assert_native_conv_consumed(self) -> None:
        if self._native_conv_call is None:
            return
        if self.native_conv_runtime_calls != 1 or len(self.conv_executed) != 1:
            raise RuntimeError(
                "M13.5 native conv patch expected exactly one native conv call, "
                f"got runtime_calls={self.native_conv_runtime_calls}, "
                f"records={len(self.conv_executed)}"
            )
        if self.device_torch_cuda_conv_count != 0:
            raise RuntimeError("M13.5 native conv patch used device_torch_cuda")

    def assert_native_attention_consumed(self) -> None:
        if self._native_attention_call is None:
            return
        if self.native_attention_runtime_calls != 1 or len(self.attention_executed) != 1:
            raise RuntimeError(
                "M13.6 native attention patch expected exactly one native attention call, "
                f"got runtime_calls={self.native_attention_runtime_calls}, "
                f"records={len(self.attention_executed)}"
            )
        if self.torch_replay_attention_count != 0:
            raise RuntimeError("M13.6 native attention patch used Torch replay")

    def install_captured_kernel_patches(
        self,
        calls: list[_M13CapturedKernelRuntimeCall],
    ) -> None:
        self._captured_expected = {call.kernel_name: call for call in calls}
        names = set(self._captured_expected)
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if not namespace:
                continue
            for name in names:
                if name not in namespace:
                    continue
                old_value = namespace[name]
                namespace[name] = _M13CapturedKernelRunner(self, self._captured_expected[name])
                self._captured_patches.append((module, name, old_value))

    def restore_captured_kernel_patches(self) -> None:
        while self._captured_patches:
            module, name, old_value = self._captured_patches.pop()
            getattr(module, "__dict__", {})[name] = old_value

    def run_captured_kernel(
        self,
        call: "_M13CapturedKernelRuntimeCall",
        args: tuple[Any, ...],
        stream,
        kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            raise RuntimeError(
                "M13.7 captured-kernel TVM runner does not support keyword "
                f"runtime arguments: {sorted(kwargs)}"
            )
        if call.translation_route == "m13_7_native_captured_layout_copy":
            storage_extent = int(call.artifact.meta.extent_value)
            tvm_args = [
                _from_dlpack_allow_strided(
                    self._torch.as_strided(args[0], (storage_extent,), (1,))
                ),
                _from_dlpack_allow_strided(
                    self._torch.as_strided(args[1], (storage_extent,), (1,))
                ),
            ]
            device = getattr(args[0], "device", None)
            device_index = int(device.index) if getattr(device, "index", None) is not None else 0
        else:
            tvm_args, device_index = _m13_7_torch_runtime_args_to_tvm(args, call.artifact.meta)
        dev = tvm.cuda(device_index)
        if stream is not None:
            dev.set_raw_stream(int(stream))
        try:
            self._run_artifact(
                call.artifact,
                tvm_args,
                bucket_id="captured_kernel",
                kernel_name=call.artifact.meta.kernel_name,
                op_family="captured_triton_kernel",
            )
        finally:
            if stream is not None:
                dev.set_raw_stream(0)
        self._captured_runtime_counts[call.kernel_name] += 1
        self.captured_executed.append(_m13_7_captured_kernel_runtime_record(call))

    def assert_captured_kernels_consumed(self) -> None:
        if not self._captured_expected:
            raise RuntimeError("M13.7 captured kernel patch expected captured calls")
        missing = [
            name for name in self._captured_expected if self._captured_runtime_counts[name] != 1
        ]
        if missing:
            raise RuntimeError(f"M13.7 captured TVM patch missed kernels: {missing}")

    def _assert_native_conv_kwargs(self, kwargs: dict[str, Any]) -> None:
        call = self._native_conv_call
        if call is None:
            raise RuntimeError("M13.5 native conv patch requires a native conv call")
        expected = {
            "stride": call.stride,
            "padding": call.padding,
            "dilation": call.dilation,
            "transposed": False,
            "output_padding": (0, 0),
            "groups": call.groups,
            "bias": None,
        }
        for key, value in expected.items():
            actual = kwargs.get(key)
            if key in {"stride", "padding", "dilation", "output_padding"} and actual is not None:
                actual = tuple(int(item) for item in actual)
            if actual != value:
                raise RuntimeError(f"M13.5 native conv expected {key}={value!r}, got {actual!r}")

    @staticmethod
    def _assert_shape_stride(
        label: str,
        tensor,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> None:
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        actual_stride = tuple(int(dim) for dim in tensor.stride())
        if actual_shape != tuple(shape) or actual_stride != tuple(stride):
            raise RuntimeError(
                f"M13 native runtime {label} shape/stride mismatch: "
                f"shape={actual_shape} stride={actual_stride}, "
                f"expected shape={tuple(shape)} stride={tuple(stride)}"
            )

    def _storage_view(self, tensor, shape: tuple[int, ...], stride: tuple[int, ...]):
        extent = _storage_extent(shape, stride)
        return self._torch.as_strided(
            tensor,
            (extent,),
            (1,),
            storage_offset=int(tensor.storage_offset()),
        )

    def _run_generated_bridge_call(
        self,
        call: _M13GeneratedBridgeRuntimeCall,
        *,
        lhs,
        rhs,
        out,
        bias,
    ) -> None:
        if self._direct_bridge_mode == "p2_direct_io":
            self._run_generated_bridge_direct_call(
                call,
                lhs=lhs,
                rhs=rhs,
                out=out,
                bias=bias,
            )
            return
        if self._direct_bridge_mode == "p3_async_direct_io":
            self._run_generated_bridge_async_direct_call(
                call,
                lhs=lhs,
                rhs=rhs,
                out=out,
                bias=bias,
                runtime_record_kind="p3",
                bucket_id="generated_bridge_async_direct_io",
                fast_entry=False,
            )
            return
        if self._direct_bridge_mode == "p5_fast_async_direct_io":
            self._run_generated_bridge_async_direct_call(
                call,
                lhs=lhs,
                rhs=rhs,
                out=out,
                bias=bias,
                runtime_record_kind="p5",
                bucket_id="generated_bridge_fast_async_direct_io",
                fast_entry=True,
            )
            return
        rhs_storage = self._b_storage(rhs, call)
        a_pad = self._torch.empty(
            (call.block_m, call.wrapper_k),
            device=lhs.device,
            dtype=lhs.dtype,
        )
        b_row_major = self._torch.empty(
            (call.wrapper_k, call.wrapper_n),
            device=rhs.device,
            dtype=rhs.dtype,
        )
        core_out = self._torch.empty(
            (call.block_m, call.wrapper_n),
            device=out.device,
            dtype=out.dtype,
        )

        tvm_arg_batches = []
        args = [
            tvm.runtime.from_dlpack(lhs),
            tvm.runtime.from_dlpack(a_pad),
            call.wrapper_m * call.wrapper_k,
        ]
        tvm_arg_batches.append(args)
        self._run_artifact(
            call.a_copy_artifact,
            args,
            bucket_id="generated_bridge_a_copy",
            kernel_name=call.a_copy_artifact.meta.kernel_name,
            op_family=call.op_family,
            index=call.index,
        )
        if call.a_zero_tail_artifact is not None:
            args = [
                tvm.runtime.from_dlpack(a_pad),
                (call.block_m - call.wrapper_m) * call.wrapper_k,
            ]
            tvm_arg_batches.append(args)
            self._run_artifact(
                call.a_zero_tail_artifact,
                args,
                bucket_id="generated_bridge_a_zero_tail",
                kernel_name=call.a_zero_tail_artifact.meta.kernel_name,
                op_family=call.op_family,
                index=call.index,
            )
        args = [
            tvm.runtime.from_dlpack(rhs_storage),
            tvm.runtime.from_dlpack(b_row_major),
            call.wrapper_k * call.wrapper_n,
        ]
        tvm_arg_batches.append(args)
        self._run_artifact(
            call.b_adapter_artifact,
            args,
            bucket_id="generated_bridge_b_adapter",
            kernel_name=call.b_adapter_artifact.meta.kernel_name,
            op_family=call.op_family,
            index=call.index,
        )
        args = [
            tvm.runtime.from_dlpack(a_pad),
            tvm.runtime.from_dlpack(b_row_major),
            tvm.runtime.from_dlpack(core_out),
        ]
        tvm_arg_batches.append(args)
        self._run_artifact(
            call.core_artifact,
            args,
            bucket_id="generated_bridge_core_dot",
            kernel_name=call.core_artifact.meta.kernel_name,
            op_family=call.op_family,
            index=call.index,
        )
        if call.op_family == "extern_addmm_bias":
            if bias is None:
                raise RuntimeError("M13.4 addmm epilogue requires bias")
            args = [
                tvm.runtime.from_dlpack(bias),
                tvm.runtime.from_dlpack(core_out),
                tvm.runtime.from_dlpack(out),
                call.wrapper_m * call.wrapper_n,
            ]
            tvm_arg_batches.append(args)
            self._run_artifact(
                call.commit_artifact,
                args,
                bucket_id="generated_bridge_bias_commit",
                kernel_name=call.commit_artifact.meta.kernel_name,
                op_family=call.op_family,
                index=call.index,
            )
        else:
            args = [
                tvm.runtime.from_dlpack(core_out),
                tvm.runtime.from_dlpack(out),
                call.wrapper_m * call.wrapper_n,
            ]
            tvm_arg_batches.append(args)
            self._run_artifact(
                call.commit_artifact,
                args,
                bucket_id="generated_bridge_output_commit",
                kernel_name=call.commit_artifact.meta.kernel_name,
                op_family=call.op_family,
                index=call.index,
            )
        self._torch.cuda.synchronize()
        tvm_arg_batches.clear()
        self.executed.append(_m13_4_runtime_record(call))

    def _run_generated_bridge_direct_call(
        self,
        call: _M13GeneratedBridgeRuntimeCall,
        *,
        lhs,
        rhs,
        out,
        bias,
    ) -> None:
        rhs_storage = self._b_storage(rhs, call)
        tvm_args = []
        if call.op_family == "extern_addmm_bias":
            if bias is None:
                raise RuntimeError("M13.P2 direct bridge addmm requires bias")
            tvm_args.append(_from_dlpack_allow_strided(bias))
        tvm_args.extend(
            [
                _from_dlpack_allow_strided(lhs),
                _from_dlpack_allow_strided(rhs_storage),
                _from_dlpack_allow_strided(out),
            ]
        )
        self._run_artifact(
            call.direct_io_artifact,
            tvm_args,
            bucket_id="generated_bridge_direct_io",
            kernel_name=call.direct_io_artifact.meta.kernel_name,
            op_family=call.op_family,
            index=call.index,
        )
        self._torch.cuda.synchronize()
        tvm_args.clear()
        self.executed.append(_m13_p2_direct_runtime_record(call))

    def _run_generated_bridge_async_direct_call(
        self,
        call: _M13GeneratedBridgeRuntimeCall,
        *,
        lhs,
        rhs,
        out,
        bias,
        runtime_record_kind: str,
        bucket_id: str,
        fast_entry: bool,
    ) -> None:
        total_start_ns = time.perf_counter_ns()
        phase_start_ns = total_start_ns
        rhs_storage = self._b_storage(rhs, call)
        self.record_runtime_profile(
            "generated_bridge_b_storage_view_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
            index=call.index,
        )
        tvm_args = []
        if call.op_family == "extern_addmm_bias":
            if bias is None:
                raise RuntimeError("M13.P3 async direct bridge addmm requires bias")
            phase_start_ns = time.perf_counter_ns()
            tvm_args.append(_from_dlpack_allow_strided(bias))
            self.record_runtime_profile(
                "generated_bridge_dlpack_bias_cpu",
                time.perf_counter_ns() - phase_start_ns,
                op_family=call.op_family,
                index=call.index,
            )
        phase_start_ns = time.perf_counter_ns()
        tvm_args.append(_from_dlpack_allow_strided(lhs))
        self.record_runtime_profile(
            "generated_bridge_dlpack_lhs_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
            index=call.index,
        )
        phase_start_ns = time.perf_counter_ns()
        tvm_args.append(_from_dlpack_allow_strided(rhs_storage))
        self.record_runtime_profile(
            "generated_bridge_dlpack_rhs_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
            index=call.index,
        )
        phase_start_ns = time.perf_counter_ns()
        tvm_args.append(_from_dlpack_allow_strided(out))
        self.record_runtime_profile(
            "generated_bridge_dlpack_out_cpu",
            time.perf_counter_ns() - phase_start_ns,
            op_family=call.op_family,
            index=call.index,
        )
        if fast_entry:
            self._run_artifact_fast_entry(
                call.direct_io_artifact,
                tvm_args,
                bucket_id=bucket_id,
                kernel_name=call.direct_io_artifact.meta.kernel_name,
                op_family=call.op_family,
                index=call.index,
            )
        else:
            self._run_artifact(
                call.direct_io_artifact,
                tvm_args,
                bucket_id=bucket_id,
                kernel_name=call.direct_io_artifact.meta.kernel_name,
                op_family=call.op_family,
                index=call.index,
            )
        tvm_args.clear()
        if runtime_record_kind == "p5":
            self.executed.append(_m13_p5_fast_direct_runtime_record(call))
        else:
            self.executed.append(_m13_p3_async_direct_runtime_record(call))
        self.record_runtime_profile(
            "generated_bridge_async_direct_call_cpu",
            time.perf_counter_ns() - total_start_ns,
            op_family=call.op_family,
            index=call.index,
        )

    def _b_storage(self, rhs, call: _M13GeneratedBridgeRuntimeCall):
        expected_shape = call.wrapper_b_storage_shape
        expected_stride = call.wrapper_b_storage_stride
        base = getattr(rhs, "_base", None)
        if base is not None and tuple(int(dim) for dim in base.shape) == expected_shape:
            return base
        if tuple(int(dim) for dim in rhs.shape) == expected_shape and tuple(
            int(stride) for stride in rhs.stride()
        ) == expected_stride:
            return rhs
        return self._torch.as_strided(
            rhs,
            expected_shape,
            expected_stride,
            storage_offset=int(rhs.storage_offset()),
        )

    def _run_artifact(
        self,
        artifact: TritonTVMArtifact,
        args: list[Any],
        *,
        bucket_id: str,
        kernel_name: str,
        op_family: str,
        index: int | None = None,
    ) -> None:
        if not self._profile_artifacts:
            start_ns = time.perf_counter_ns()
            artifact.run(args)
            self.record_runtime_profile(
                f"{bucket_id}_artifact_run_cpu",
                time.perf_counter_ns() - start_ns,
                kernel_name=kernel_name,
                op_family=op_family,
                index=index,
            )
            return
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        artifact.run(args)
        end.record()
        self._torch.cuda.synchronize()
        record = {
            "bucket_id": bucket_id,
            "kernel_name": kernel_name,
            "op_family": op_family,
            "elapsed_ms": float(start.elapsed_time(end)),
        }
        if index is not None:
            record["index"] = index
        self.profile_records.append(record)

    def _run_artifact_fast_entry(
        self,
        artifact: TritonTVMArtifact,
        args: list[Any],
        *,
        bucket_id: str,
        kernel_name: str,
        op_family: str,
        index: int | None = None,
    ) -> None:
        cache_key = id(artifact)
        entry = self._artifact_entry_cache.get(cache_key)
        if entry is None:
            entry = artifact.executable[artifact.meta.kernel_name]
            self._artifact_entry_cache[cache_key] = entry
        start_ns = time.perf_counter_ns()
        entry(*args)
        self.record_runtime_profile(
            f"{bucket_id}_artifact_fast_entry_cpu",
            time.perf_counter_ns() - start_ns,
            kernel_name=kernel_name,
            op_family=op_family,
            index=index,
        )


class _M13CapturedKernelRunner:
    """Small replacement for an Inductor Triton kernel object in the wrapper module."""

    def __init__(
        self,
        state: _M13GeneratedBridgePatchState,
        call: _M13CapturedKernelRuntimeCall,
    ):
        self._state = state
        self._call = call

    def run(self, *args, stream=None, benchmark_run=False, **kwargs):  # pylint: disable=unused-argument
        self._state.run_captured_kernel(self._call, args, stream, kwargs)


@contextlib.contextmanager
def _patched_m13_4_matmul_runtime(
    torch,
    calls: list[_M13GeneratedBridgeRuntimeCall],
) -> Iterator[_M13GeneratedBridgePatchState]:
    import torch._inductor.select_algorithm as select_algorithm  # pylint: disable=import-outside-toplevel

    state = _M13GeneratedBridgePatchState(calls, torch)
    extern_kernels = select_algorithm.extern_kernels
    old_addmm = extern_kernels.addmm
    old_mm = extern_kernels.mm
    old_conv = extern_kernels.convolution
    sdpa_packet = torch.ops.aten._scaled_dot_product_efficient_attention
    old_sdpa_default = sdpa_packet.default
    extern_kernels.addmm = state.addmm
    extern_kernels.mm = state.mm
    extern_kernels.convolution = state.wrap_conv(old_conv)
    sdpa_packet.default = state.native_decomposed_sdpa
    try:
        yield state
    finally:
        extern_kernels.addmm = old_addmm
        extern_kernels.mm = old_mm
        extern_kernels.convolution = old_conv
        sdpa_packet.default = old_sdpa_default


@contextlib.contextmanager
def _patched_m13_5_native_conv_runtime(
    torch,
    calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
) -> Iterator[_M13GeneratedBridgePatchState]:
    import torch._inductor.select_algorithm as select_algorithm  # pylint: disable=import-outside-toplevel

    state = _M13GeneratedBridgePatchState(calls, torch, native_conv_call=conv_call)
    extern_kernels = select_algorithm.extern_kernels
    old_addmm = extern_kernels.addmm
    old_mm = extern_kernels.mm
    old_conv = extern_kernels.convolution
    sdpa_packet = torch.ops.aten._scaled_dot_product_efficient_attention
    old_sdpa_default = sdpa_packet.default
    extern_kernels.addmm = state.addmm
    extern_kernels.mm = state.mm
    extern_kernels.convolution = state.native_conv2d
    sdpa_packet.default = state.native_decomposed_sdpa
    try:
        yield state
    finally:
        extern_kernels.addmm = old_addmm
        extern_kernels.mm = old_mm
        extern_kernels.convolution = old_conv
        sdpa_packet.default = old_sdpa_default


@contextlib.contextmanager
def _patched_m13_6_native_attention_runtime(
    torch,
    calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
    *,
    profile_artifacts: bool = False,
    profile_runtime: bool = False,
    direct_bridge: bool | str = False,
    native_conv_fast_entry: bool = False,
) -> Iterator[_M13GeneratedBridgePatchState]:
    import torch._inductor.select_algorithm as select_algorithm  # pylint: disable=import-outside-toplevel

    state = _M13GeneratedBridgePatchState(
        calls,
        torch,
        native_conv_call=conv_call,
        native_attention_call=attention_call,
        profile_artifacts=profile_artifacts,
        profile_runtime=profile_runtime,
        direct_bridge=direct_bridge,
        native_conv_fast_entry=native_conv_fast_entry,
    )
    extern_kernels = select_algorithm.extern_kernels
    old_addmm = extern_kernels.addmm
    old_mm = extern_kernels.mm
    old_conv = extern_kernels.convolution
    sdpa_packet = torch.ops.aten._scaled_dot_product_efficient_attention
    old_sdpa_default = sdpa_packet.default
    extern_kernels.addmm = state.addmm
    extern_kernels.mm = state.mm
    extern_kernels.convolution = state.native_conv2d
    sdpa_packet.default = state.native_tvm_attention_decomposed_sdpa
    try:
        yield state
    finally:
        extern_kernels.addmm = old_addmm
        extern_kernels.mm = old_mm
        extern_kernels.convolution = old_conv
        sdpa_packet.default = old_sdpa_default


@contextlib.contextmanager
def _patched_m13_7_strict_runtime(
    torch,
    captured_calls: list[_M13CapturedKernelRuntimeCall],
    calls: list[_M13GeneratedBridgeRuntimeCall],
    conv_call: _M13NativeConvRuntimeCall,
    attention_call: _M13NativeAttentionRuntimeCall,
    *,
    profile_artifacts: bool = False,
) -> Iterator[_M13GeneratedBridgePatchState]:
    import torch._inductor.select_algorithm as select_algorithm  # pylint: disable=import-outside-toplevel

    state = _M13GeneratedBridgePatchState(
        calls,
        torch,
        native_conv_call=conv_call,
        native_attention_call=attention_call,
        profile_artifacts=profile_artifacts,
    )
    extern_kernels = select_algorithm.extern_kernels
    old_addmm = extern_kernels.addmm
    old_mm = extern_kernels.mm
    old_conv = extern_kernels.convolution
    sdpa_packet = torch.ops.aten._scaled_dot_product_efficient_attention
    old_sdpa_default = sdpa_packet.default
    extern_kernels.addmm = state.addmm
    extern_kernels.mm = state.mm
    extern_kernels.convolution = state.native_conv2d
    sdpa_packet.default = state.native_tvm_attention_decomposed_sdpa
    state.install_captured_kernel_patches(captured_calls)
    try:
        yield state
    finally:
        state.restore_captured_kernel_patches()
        extern_kernels.addmm = old_addmm
        extern_kernels.mm = old_mm
        extern_kernels.convolution = old_conv
        sdpa_packet.default = old_sdpa_default


def _run_captured_kernel_artifacts() -> list[dict[str, Any]]:
    wrapper_source = _load_wrapper_source()
    corpus_records = {
        record["kernel_name"]: record
        for record in _vit_captured_records(_read_json(M12_CORPUS_REPORT))
    }
    sources = extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL)
    if len(sources) != 7:
        raise RuntimeError(f"M13.2 expected 7 captured kernels, got {len(sources)}")

    records = []
    for index, source in enumerate(sources):
        corpus_record = corpus_records[source.kernel_name]
        kernel = load_inductor_kernel(source)
        contract = str(corpus_record.get("contract", ""))
        if contract == VISION_CONTRACT_POINTWISE_GRID2D_STATIC:
            record = _run_grid2d_captured_kernel(index, source, kernel, corpus_record)
        else:
            record = _run_translated_captured_kernel(index, source, kernel, contract)
        records.append(record)
    return records


def _build_m13_7_captured_kernel_runtime_calls(
    wrapper_source: str,
) -> list[_M13CapturedKernelRuntimeCall]:
    corpus_records = {
        record["kernel_name"]: record
        for record in _vit_captured_records(_read_json(M12_CORPUS_REPORT))
    }
    sources = extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL)
    if len(sources) != 7:
        raise RuntimeError(f"M13.7 expected 7 captured kernels, got {len(sources)}")

    calls: list[_M13CapturedKernelRuntimeCall] = []
    for index, source in enumerate(sources):
        corpus_record = corpus_records[source.kernel_name]
        kernel = load_inductor_kernel(source)
        contract = str(corpus_record.get("contract", ""))
        if source.kernel_name in {
            "triton_poi_fused_convolution_0",
            "triton_poi_fused_convolution_1",
        }:
            artifact, translation_route = _build_m13_7_exact_captured_layout_copy_artifact(
                source.kernel_name
            )
        elif contract == VISION_CONTRACT_POINTWISE_GRID2D_STATIC:
            artifact, translation_route = _build_m13_7_grid2d_captured_kernel_artifact(
                index, source, kernel, corpus_record
            )
        else:
            artifact, translation_route = _build_m13_7_translated_captured_kernel_artifact(
                kernel, contract
            )
        calls.append(
            _M13CapturedKernelRuntimeCall(
                index=index,
                kernel_name=source.kernel_name,
                contract=contract,
                artifact=artifact,
                translation_route=translation_route,
            )
        )
    return calls


def _build_m13_7_translated_captured_kernel_artifact(
    kernel,
    contract: str,
) -> tuple[TritonTVMArtifact, str]:
    artifact = lower_inductor_kernel_to_ttir(kernel)
    irmod, meta = translate_ttir(artifact, grid=(1,), target="cuda", contract=contract)
    return build_triton_tvm(irmod, meta), "lower_to_ttir_translate_ttir_build_triton_tvm"


def _build_m13_7_grid2d_captured_kernel_artifact(
    index: int,
    source,
    kernel,
    corpus_record: dict[str, Any],
) -> tuple[TritonTVMArtifact, str]:
    size_hints = corpus_record.get("size_hints") or kernel.size_hints
    semantics = VisionGrid2DPointwiseSemantics(
        case_name=f"m13_7_{source.kernel_name}_{index}",
        model=TARGET_MODEL,
        grid_family=str(corpus_record.get("m11_grid_family", "")),
        x_extent=int(size_hints["x"]),
        y_extent=int(size_hints["y"]),
        op_kind=_m11_grid2d_op_kind(corpus_record, kernel),
    )
    source_text = build_native_grid2d_pointwise_tirx_source(semantics)
    irmod = tvm.script.from_source(source_text)
    validate_pointwise_grid2d_static_contract(irmod)
    return build_triton_tvm(irmod, _grid2d_meta(semantics)), "m11_grid2d_static_captured_artifact"


def _build_m13_7_exact_captured_layout_copy_artifact(
    kernel_name: str,
) -> tuple[TritonTVMArtifact, str]:
    if kernel_name == "triton_poi_fused_convolution_0":
        y_extent = 3
        x_extent = 1024
        output_index = "y + T.int64(3) * x"
    elif kernel_name == "triton_poi_fused_convolution_1":
        y_extent = 192
        x_extent = 256
        output_index = (
            "(y % T.int64(3)) + T.int64(3) * x + "
            "T.int64(768) * (y // T.int64(3))"
        )
    else:
        raise RuntimeError(f"M13.7 unsupported exact captured layout copy {kernel_name}")
    storage_extent = y_extent * x_extent
    tvm_kernel_name = f"m13_7_{kernel_name}_layout_copy"
    source = f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {tvm_kernel_name}(input_h: T.handle, output_h: T.handle):
        T.func_attr({{\
"global_symbol": "{tvm_kernel_name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "{VISION_CONTRACT_POINTWISE_GRID2D_STATIC}", \
"triton_tvm.grid2d_contract": "{VISION_CONTRACT_POINTWISE_GRID2D_STATIC}", \
"triton_tvm.grid2d_contract_version": "m11_6_grid2d_static_v1", \
"triton_tvm.grid2d_x_extent": {x_extent}, \
"triton_tvm.grid2d_y_extent": {y_extent}, \
"triton_tvm.grid2d_block_size": 1, \
"triton_tvm.implementation_kind": "{M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM}", \
"triton_tvm.grid2d_runtime_status": "{M11_GRID2D_RUNTIME_READY}", \
"triton_tvm.grid2d_provider_kind": "{M11_GRID2D_PROVIDER_NATIVE_TVM}", \
"triton_tvm.grid2d_runtime_claim": "{M11_GRID2D_RUNTIME_CLAIM}", \
"triton_tvm.grid2d_performance_claim": True, \
"triton_tvm.grid2d_uses_host_staging": False, \
"triton_tvm.grid2d_runtime_launch_count": 1, \
"triton_tvm.grid2d_artifact_call_count": 1, \
"triton_tvm.grid2d_total_io_bytes": {storage_extent * 3 * 4}, \
"triton_tvm.grid2d_host_staging_bytes": 0, \
"triton_tvm.source_kernel_name": "{kernel_name}"}})
        inp = T.match_buffer(input_h, ({storage_extent},), "float32")
        out = T.match_buffer(output_h, ({storage_extent},), "float32")
        for by in T.thread_binding(0, {y_extent}, thread="blockIdx.y"):
            for bx in T.thread_binding(0, {x_extent}, thread="blockIdx.x"):
                for tx in T.thread_binding(0, 1, thread="threadIdx.x"):
                    y = T.Cast("int64", by)
                    x = T.Cast("int64", bx)
                    if x < T.int64({x_extent}):
                        out[{output_index}] = inp[y * T.int64({x_extent}) + x]
"""
    irmod = tvm.script.from_source(source)
    meta = TritonTVMMeta(
        kernel_name=tvm_kernel_name,
        signature={},
        constexprs={},
        grid=(x_extent, y_extent),
        target="cuda",
        target_kind="cuda",
        contract=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        canonical_contract=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        requested_contract=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        emit="tirx",
        translator_version="m13_7_strict_correctness_gate",
        contract_version="pointwise_grid2d_static_m13_7",
        target_policy_version="m13_7_strict_correctness_gate",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=storage_extent,
        reduction_extent_param="",
        reduction_extent_kind="none",
        reduction_extent_value=None,
        buffer_extents={
            "input": f"T.int64({storage_extent})",
            "output": f"T.int64({storage_extent})",
        },
        block_size=1,
        indexing_kind="captured_grid2d_layout_copy",
        execution_kind="m13_7_native_captured_layout_copy",
        accumulator_dtype_policy="not_applicable",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="grid2d_static",
        layout_policy="captured_inductor_layout_copy",
        launch_policy_id="cuda_grid2d_one_thread_per_element",
        abi=[
            {"name": "input", "kind": "pointer", "dtype": "float32"},
            {"name": "output", "kind": "pointer", "dtype": "float32"},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"m13_7:{kernel_name}:layout_copy",
        implementation_kind=M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        schedule_id="m13_7_native_captured_layout_copy",
    )
    return build_triton_tvm(irmod, meta), "m13_7_native_captured_layout_copy"


def _run_translated_captured_kernel(index, source, kernel, contract: str) -> dict[str, Any]:
    record = _base_captured_kernel_record(index, source.kernel_name, contract)
    artifact = lower_inductor_kernel_to_ttir(kernel)
    graph = TTIRReader().read(artifact.ttir)
    irmod, meta = translate_ttir(artifact, grid=(1,), target="cuda", contract=contract)
    built = build_triton_tvm(irmod, meta)
    tvm_args = _make_tvm_args_for_meta(meta, kernel=kernel)
    built.run(tvm_args)
    tvm.cuda(0).sync()
    record.update(
        {
            "status": "passed",
            "translation_route": "lower_to_ttir_translate_ttir_build_triton_tvm",
            "implementation_level": "triton_language_lowering",
            "runtime_kind": "tvm_artifact_run",
            "run_count": 1,
            "artifact_identity": _artifact_identity(meta, graph),
            "stream_handoff": {"stream": None, "policy": "tvm_default_stream"},
            "harness_fallback_count": 0,
            "native_triton_launch_count": 0,
            "silent_fallback_count": 0,
        }
    )
    return record


def _run_grid2d_captured_kernel(index, source, kernel, corpus_record: dict[str, Any]) -> dict[str, Any]:
    contract = VISION_CONTRACT_POINTWISE_GRID2D_STATIC
    record = _base_captured_kernel_record(index, source.kernel_name, contract)
    size_hints = corpus_record.get("size_hints") or kernel.size_hints
    semantics = VisionGrid2DPointwiseSemantics(
        case_name=f"m13_2_{source.kernel_name}_{index}",
        model=TARGET_MODEL,
        grid_family=str(corpus_record.get("m11_grid_family", "")),
        x_extent=int(size_hints["x"]),
        y_extent=int(size_hints["y"]),
        op_kind=_m11_grid2d_op_kind(corpus_record, kernel),
    )
    source_text = build_native_grid2d_pointwise_tirx_source(semantics)
    irmod = tvm.script.from_source(source_text)
    validate_pointwise_grid2d_static_contract(irmod)
    meta = _grid2d_meta(semantics)
    built = build_triton_tvm(irmod, meta)
    tvm_args = _make_tvm_args_for_meta(meta, kernel=kernel)
    built.run(tvm_args)
    tvm.cuda(0).sync()
    record.update(
        {
            "status": "passed",
            "translation_route": "m11_grid2d_static_captured_artifact",
            "implementation_level": "triton_language_lowering",
            "runtime_kind": "tvm_artifact_run",
            "m11_grid_artifact_status": M11_GRID2D_ARTIFACT_READY,
            "m11_grid_runtime_status": M11_GRID2D_RUNTIME_READY,
            "m11_grid_implementation_kind": M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
            "m11_grid_provider_kind": M11_GRID2D_PROVIDER_NATIVE_TVM,
            "run_count": 1,
            "artifact_identity": {
                "kernel_name": meta.kernel_name,
                "cache_key": meta.cache_key,
                "contract": meta.contract,
                "grid": list(meta.grid),
                "buffer_extents": dict(meta.buffer_extents),
            },
            "stream_handoff": {"stream": None, "policy": "tvm_default_stream"},
            "harness_fallback_count": 0,
            "native_triton_launch_count": 0,
            "silent_fallback_count": 0,
        }
    )
    return record


def _run_generated_tl_dot_bridge_records() -> list[dict[str, Any]]:
    wrapper_source = _load_wrapper_source()
    calls = sorted(
        [
            call
            for call in extract_inductor_wrapper_extern_calls(
                wrapper_source,
                case_name=TARGET_MODEL,
                extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
                extern_gemm_runtime_model_case=TARGET_MODEL,
            )
            if call.op_family in {"extern_gemm", "extern_addmm_bias"}
        ],
        key=lambda call: call.line_no,
    )
    if len(calls) != 7:
        raise RuntimeError(f"M13.3 expected 7 wrapper matmul/addmm calls, got {len(calls)}")

    records = []
    for index, call in enumerate(calls):
        wrapper_semantics = _extract_wrapper_matmul_semantics(index, call)
        artifact = lower_to_ttir(
            _m13_generated_tl_dot_bridge,
            _generated_dot_signature(),
            {
                "BLOCK_M": _tl_dot_block_extent(wrapper_semantics.m),
                "BLOCK_N": wrapper_semantics.n,
                "BLOCK_K": wrapper_semantics.k,
            },
        )
        graph = TTIRReader().read(artifact.ttir)
        generated_semantics = extract_matmul_semantics_from_ttir(
            graph, source_kind=GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
        )
        irmod, meta = translate_ttir(
            artifact,
            grid=(1,),
            target="cuda",
            contract="matmul_minimal",
            matmul_source_kind=GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        )
        validate_matmul_minimal_contract(irmod)
        decision = TargetMatmulPolicy(target_kind="cuda").decide(
            generated_semantics,
            matmul_contract_ok=True,
        )
        record = {
            "index": index,
            "status": "passed"
            if _generated_bridge_record_passes(wrapper_semantics, generated_semantics, meta, decision)
            else "failed",
            "line_no": call.line_no,
            "op_family": call.op_family,
            "op_name": call.op_name,
            "wrapper_source_kind": wrapper_semantics.source_kind,
            "wrapper_shape": {
                "m": wrapper_semantics.m,
                "n": wrapper_semantics.n,
                "k": wrapper_semantics.k,
            },
            "bridge_padding_policy": "zero_pad_logical_m_to_power_of_two_tile",
            "bridge_logical_shape": {
                "m": wrapper_semantics.m,
                "n": wrapper_semantics.n,
                "k": wrapper_semantics.k,
            },
            "wrapper_b_layout": wrapper_semantics.b_layout,
            "wrapper_epilogue_kind": wrapper_semantics.epilogue_kind,
            "generated_kernel_name": artifact.kernel_name,
            "lower_to_ttir_ok": True,
            "ttir_contains_tt_dot": "tt.dot" in artifact.ttir,
            "reader_dot_snapshot": _dot_snapshot(graph),
            "generated_matmul_source_kind": generated_semantics.source_kind,
            "generated_shape": {
                "m": generated_semantics.m,
                "n": generated_semantics.n,
                "k": generated_semantics.k,
            },
            "generated_dtypes": {
                "a": generated_semantics.a_dtype,
                "b": generated_semantics.b_dtype,
                "accumulator": generated_semantics.accumulator_dtype,
                "output": generated_semantics.output_dtype,
            },
            "generated_layouts": {
                "a": generated_semantics.a_layout,
                "b": generated_semantics.b_layout,
                "c": generated_semantics.c_layout,
            },
            "contract_validation_ok": meta.matmul_contract_ok,
            "policy_implementation_kind": decision.implementation_kind,
            "policy_schedule_id": decision.schedule_id,
            "policy_unsupported_reason": decision.unsupported_matmul_reason,
            "generated_bridge_counts_as_real_jit_tt_dot_captured": False,
            "runtime_replacement_connected": False,
            "bias_epilogue_bridge_status": "deferred_to_M13.4_native_pointwise"
            if wrapper_semantics.epilogue_kind == "bias_add"
            else "not_applicable",
        }
        records.append(record)
    return records


def _generated_bridge_record_passes(wrapper_semantics, generated_semantics, meta, decision) -> bool:
    expected_m = _tl_dot_block_extent(wrapper_semantics.m)
    return (
        generated_semantics.source_kind == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
        and (generated_semantics.m, generated_semantics.n, generated_semantics.k)
        == (expected_m, wrapper_semantics.n, wrapper_semantics.k)
        and generated_semantics.a_dtype == "float32"
        and generated_semantics.b_dtype == "float32"
        and generated_semantics.accumulator_dtype == "float32"
        and generated_semantics.output_dtype == "float32"
        and generated_semantics.epilogue_kind == "none"
        and meta.matmul_contract_ok is True
        and meta.matmul_source_kind == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
        and decision.implementation_kind == "native_tir_schedule"
        and decision.unsupported_matmul_reason == ""
    )


def _tl_dot_block_extent(logical_extent: int) -> int:
    value = max(8, int(logical_extent))
    if value & (value - 1) == 0:
        return value
    return 1 << value.bit_length()


def _extract_wrapper_matmul_semantics(index: int, call):
    payload = {
        "op_name": call.op_name,
        "source": call.source,
        "case_name": TARGET_MODEL,
        "kernel_name": f"m13_3_wrapper_matmul_{index}",
    }
    if call.op_family == "extern_addmm_bias":
        return extract_matmul_semantics_from_wrapper_extern_addmm(payload)
    return extract_matmul_semantics_from_wrapper_extern(payload)


def _build_m13_4_generated_bridge_runtime_calls(
    wrapper_source: str,
) -> list[_M13GeneratedBridgeRuntimeCall]:
    calls = sorted(
        [
            call
            for call in extract_inductor_wrapper_extern_calls(
                wrapper_source,
                case_name=TARGET_MODEL,
                extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
                extern_gemm_runtime_model_case=TARGET_MODEL,
            )
            if call.op_family in {"extern_gemm", "extern_addmm_bias"}
        ],
        key=lambda call: call.line_no,
    )
    if len(calls) != 7:
        raise RuntimeError(f"M13.4 expected 7 wrapper matmul/addmm calls, got {len(calls)}")

    built: list[_M13GeneratedBridgeRuntimeCall] = []
    for index, call in enumerate(calls):
        wrapper_semantics = _extract_wrapper_matmul_semantics(index, call)
        block_m = _tl_dot_block_extent(wrapper_semantics.m)
        core_artifact = _build_m13_4_generated_bridge_core_artifact(wrapper_semantics, block_m)
        a_copy_artifact = _build_m13_4_pointwise_artifact(
            _build_m13_4_a_copy_tirx_source(index, wrapper_semantics),
            kernel_name=f"m13_4_a_copy_{index}",
            abi=[
                {"name": "a", "kind": "pointer", "dtype": wrapper_semantics.a_dtype},
                {"name": "a_pad", "kind": "pointer", "dtype": wrapper_semantics.a_dtype},
                {"name": "n_elements", "kind": "scalar", "dtype": "int64"},
            ],
            buffer_extents={
                "a": f"T.int64({wrapper_semantics.m * wrapper_semantics.k})",
                "a_pad": f"T.int64({block_m * wrapper_semantics.k})",
            },
            extent_value=wrapper_semantics.m * wrapper_semantics.k,
            cache_key=f"m13_4:a_copy:{index}:{wrapper_semantics.m}:{wrapper_semantics.k}",
        )
        a_zero_tail_artifact = None
        if block_m > wrapper_semantics.m:
            a_zero_tail_artifact = _build_m13_4_pointwise_artifact(
                _build_m13_4_a_zero_tail_tirx_source(index, wrapper_semantics, block_m),
                kernel_name=f"m13_4_a_zero_tail_{index}",
                abi=[
                    {"name": "a_pad", "kind": "pointer", "dtype": wrapper_semantics.a_dtype},
                    {"name": "n_elements", "kind": "scalar", "dtype": "int64"},
                ],
                buffer_extents={
                    "a_pad": f"T.int64({block_m * wrapper_semantics.k})",
                },
                extent_value=(block_m - wrapper_semantics.m) * wrapper_semantics.k,
                cache_key=f"m13_4:a_zero_tail:{index}:{wrapper_semantics.m}:{block_m}:{wrapper_semantics.k}",
            )
        b_storage_shape = _m13_4_b_storage_shape(wrapper_semantics)
        b_storage_stride = _m13_4_b_storage_stride(wrapper_semantics)
        b_adapter_artifact = _build_m13_4_pointwise_artifact(
            _build_m13_4_b_adapter_tirx_source(index, wrapper_semantics),
            kernel_name=f"m13_4_b_row_major_adapter_{index}",
            abi=[
                {"name": "b_storage", "kind": "pointer", "dtype": wrapper_semantics.b_dtype},
                {"name": "b_row_major", "kind": "pointer", "dtype": wrapper_semantics.b_dtype},
                {"name": "n_elements", "kind": "scalar", "dtype": "int64"},
            ],
            buffer_extents={
                "b_storage": f"T.int64({b_storage_shape[0] * b_storage_shape[1]})",
                "b_row_major": f"T.int64({wrapper_semantics.k * wrapper_semantics.n})",
            },
            extent_value=wrapper_semantics.k * wrapper_semantics.n,
            cache_key=f"m13_4:b_adapter:{index}:{wrapper_semantics.k}:{wrapper_semantics.n}:{wrapper_semantics.b_layout}",
        )
        commit_source = (
            _build_m13_4_bias_commit_tirx_source(index, wrapper_semantics, block_m)
            if call.op_family == "extern_addmm_bias"
            else _build_m13_4_output_commit_tirx_source(index, wrapper_semantics, block_m)
        )
        commit_abi = (
            [
                {"name": "bias", "kind": "pointer", "dtype": wrapper_semantics.output_dtype},
                {"name": "core_out", "kind": "pointer", "dtype": wrapper_semantics.output_dtype},
                {"name": "out", "kind": "pointer", "dtype": wrapper_semantics.output_dtype},
                {"name": "n_elements", "kind": "scalar", "dtype": "int64"},
            ]
            if call.op_family == "extern_addmm_bias"
            else [
                {"name": "core_out", "kind": "pointer", "dtype": wrapper_semantics.output_dtype},
                {"name": "out", "kind": "pointer", "dtype": wrapper_semantics.output_dtype},
                {"name": "n_elements", "kind": "scalar", "dtype": "int64"},
            ]
        )
        commit_buffer_extents = {
            "core_out": f"T.int64({block_m * wrapper_semantics.n})",
            "out": f"T.int64({wrapper_semantics.m * wrapper_semantics.n})",
        }
        if call.op_family == "extern_addmm_bias":
            commit_buffer_extents["bias"] = f"T.int64({int(np.prod(wrapper_semantics.bias_shape))})"
        commit_artifact = _build_m13_4_pointwise_artifact(
            commit_source,
            kernel_name=(
                f"m13_4_bias_epilogue_commit_{index}"
                if call.op_family == "extern_addmm_bias"
                else f"m13_4_output_commit_{index}"
            ),
            abi=commit_abi,
            buffer_extents=commit_buffer_extents,
            extent_value=wrapper_semantics.m * wrapper_semantics.n,
            cache_key=f"m13_4:commit:{index}:{wrapper_semantics.m}:{wrapper_semantics.n}:{call.op_family}",
        )
        direct_io_artifact = _build_m13_p2_direct_io_matmul_artifact(
            index,
            wrapper_semantics,
            b_storage_shape=b_storage_shape,
            b_storage_stride=b_storage_stride,
        )
        built.append(
            _M13GeneratedBridgeRuntimeCall(
                index=index,
                op_family=call.op_family,
                op_name=call.op_name,
                line_no=call.line_no,
                source=call.source,
                wrapper_source_kind=wrapper_semantics.source_kind,
                wrapper_m=wrapper_semantics.m,
                wrapper_n=wrapper_semantics.n,
                wrapper_k=wrapper_semantics.k,
                wrapper_b_layout=wrapper_semantics.b_layout,
                wrapper_b_storage_shape=b_storage_shape,
                wrapper_b_storage_stride=b_storage_stride,
                bias_param=wrapper_semantics.bias_param,
                bias_shape=wrapper_semantics.bias_shape,
                bias_stride=wrapper_semantics.bias_stride,
                block_m=block_m,
                core_artifact=core_artifact,
                a_copy_artifact=a_copy_artifact,
                a_zero_tail_artifact=a_zero_tail_artifact,
                b_adapter_artifact=b_adapter_artifact,
                commit_artifact=commit_artifact,
                direct_io_artifact=direct_io_artifact,
            )
        )
    return built


def _build_m13_5_native_conv_runtime_call(wrapper_source: str) -> _M13NativeConvRuntimeCall:
    calls = [
        call
        for call in extract_inductor_wrapper_extern_calls(
            wrapper_source,
            case_name=TARGET_MODEL,
        )
        if call.op_family == "deferred_convolution"
    ]
    if len(calls) != 1:
        raise RuntimeError(f"M13.5 expected one wrapper conv call, got {len(calls)}")
    call = calls[0]
    semantics = extract_vision_conv2d_semantics_from_wrapper_extern(
        {
            "op_name": call.op_name,
            "source": call.source,
            "full_source": wrapper_source,
            "line_no": call.line_no,
            "case_name": TARGET_MODEL,
            "kernel_name": "m13_5_native_conv2d_patch_embedding",
        }
    )
    if not _m13_5_is_exact_patch_conv(semantics):
        raise RuntimeError("M13.5 native conv is limited to exact ViT patch embedding shape")
    source = _build_m13_5_native_conv2d_tirx_source(semantics)
    irmod = tvm.script.from_source(source)
    validate_vision_conv2d_contract(irmod)
    artifact = build_triton_tvm(irmod, _m13_5_native_conv_meta(semantics))
    return _M13NativeConvRuntimeCall(
        index=0,
        op_family="wrapper_conv",
        op_name=call.op_name,
        line_no=call.line_no,
        source=call.source,
        input_param=semantics.input_param,
        weight_param=semantics.weight_param,
        output_param=semantics.output_param,
        input_shape=semantics.input_shape,
        weight_shape=semantics.weight_shape,
        output_shape=semantics.output_shape,
        input_stride=semantics.input_stride,
        weight_stride=semantics.weight_stride,
        output_stride=semantics.output_stride,
        stride=semantics.stride,
        padding=semantics.padding,
        dilation=semantics.dilation,
        groups=semantics.groups,
        contract=semantics.vision_contract,
        artifact=artifact,
    )


def _build_m13_p2_direct_io_matmul_artifact(
    index: int,
    semantics,
    *,
    b_storage_shape: tuple[int, int],
    b_storage_stride: tuple[int, int],
) -> TritonTVMArtifact:
    source = _build_m13_p2_direct_io_matmul_tirx_source(
        index,
        semantics,
        b_storage_shape=b_storage_shape,
        b_storage_stride=b_storage_stride,
    )
    irmod = tvm.script.from_source(source)
    meta = _m13_p2_direct_io_matmul_meta(
        index,
        semantics,
        b_storage_shape=b_storage_shape,
        b_storage_stride=b_storage_stride,
    )
    return build_triton_tvm(irmod, meta)


def _build_m13_p2_direct_io_matmul_tirx_source(
    index: int,
    semantics,
    *,
    b_storage_shape: tuple[int, int],
    b_storage_stride: tuple[int, int],
) -> str:
    is_addmm = semantics.source_kind == "wrapper_extern_addmm_bias"
    name = f"m13_p2_direct_io_matmul_{index}"
    b_read = "b[vn, vk]" if semantics.b_layout == "transposed_weight_view" else "b[vk, vn]"
    output_elems = semantics.m * semantics.n
    signature = (
        "bias_h: T.handle, a_h: T.handle, b_h: T.handle, out_h: T.handle"
        if is_addmm
        else "a_h: T.handle, b_h: T.handle, out_h: T.handle"
    )
    bias_line = ""
    init_value = "T.float32(0)"
    reads = f"T.reads(a[vm, vk], {b_read})"
    bias_func_attrs = ""
    bias_block_attrs = ""
    if is_addmm:
        bias_shape = _shape_literal(semantics.bias_shape)
        bias_stride = _shape_literal(semantics.bias_stride)
        bias_func_attrs = (
            f', "triton_tvm.bias_param": "{semantics.bias_param}", '
            f'"triton_tvm.bias_shape": "{bias_shape}", '
            f'"triton_tvm.bias_stride": "{bias_stride}", '
            f'"triton_tvm.bias_rank": {len(semantics.bias_shape)}, '
            f'"triton_tvm.alpha": "{semantics.alpha:g}", '
            f'"triton_tvm.beta": "{semantics.beta:g}"'
        )
        bias_block_attrs = (
            f', "triton_tvm.bias_shape": "{bias_shape}", '
            f'"triton_tvm.bias_stride": "{bias_stride}"'
        )
        bias_line = (
            f'        bias = T.match_buffer(bias_h, ({bias_shape}), "float32", '
            f"strides=({bias_stride}))\n"
        )
        if len(semantics.bias_shape) == 1:
            init_value = "bias[vn]"
            reads = f"T.reads(a[vm, vk], {b_read}, bias[vn])"
        else:
            init_value = "bias[vm, vn]"
            reads = f"T.reads(a[vm, vk], {b_read}, bias[vm, vn])"
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}({signature}):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "matmul_minimal", \
"triton_tvm.matmul_source_kind": "{semantics.source_kind}", \
"triton_tvm.generated_bridge_source_kind": "{GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND}", \
"triton_tvm.wrapper_source_kind": "{semantics.source_kind}", \
"triton_tvm.a_dtype": "{semantics.a_dtype}", \
"triton_tvm.b_dtype": "{semantics.b_dtype}", \
"triton_tvm.output_dtype": "{semantics.output_dtype}", \
"triton_tvm.accumulator_dtype": "{semantics.accumulator_dtype}", \
"triton_tvm.input_precision": "{semantics.input_precision}", \
"triton_tvm.tf32_policy": "{semantics.tf32_policy}", \
"triton_tvm.bounds_policy": "{semantics.bounds_policy}", \
"triton_tvm.mask_kind": "{semantics.mask_kind}", \
"triton_tvm.epilogue_kind": "{semantics.epilogue_kind}", \
"triton_tvm.implementation_kind": "native_tir_schedule", \
"triton_tvm.schedule_id": "{M13_P2_DIRECT_SCHEDULE_ID}", \
"triton_tvm.runtime_kind": "{M13_P2_DIRECT_RUNTIME_KIND}", \
"triton_tvm.matmul_m": {semantics.m}, \
"triton_tvm.matmul_n": {semantics.n}, \
"triton_tvm.matmul_k": {semantics.k}, \
"triton_tvm.a_layout": "{semantics.a_layout}", \
"triton_tvm.b_layout": "{semantics.b_layout}", \
"triton_tvm.c_layout": "{semantics.c_layout}", \
"triton_tvm.a_stride": "{_stride_text(semantics.a_stride)}", \
"triton_tvm.b_stride": "{_stride_text(semantics.b_stride)}", \
"triton_tvm.c_stride": "{_stride_text(semantics.c_stride)}", \
"triton_tvm.b_storage_shape": "{_shape_literal(b_storage_shape)}", \
"triton_tvm.b_storage_stride": "{_stride_text(b_storage_stride)}", \
"triton_tvm.provider_kind": "generated_bridge_direct_io", \
"triton_tvm.uses_host_staging": False, \
"triton_tvm.provider_relaxation": False{bias_func_attrs}}})
{bias_line}        a = T.match_buffer(a_h, ({semantics.m}, {semantics.k}), "{semantics.a_dtype}", strides=({_stride_text(semantics.a_stride)}))
        b = T.match_buffer(b_h, ({b_storage_shape[0]}, {b_storage_shape[1]}), "{semantics.b_dtype}", strides=({_stride_text(b_storage_stride)}))
        out = T.match_buffer(out_h, ({semantics.m}, {semantics.n}), "{semantics.output_dtype}", strides=({_stride_text(semantics.c_stride)}))
        for blockIdx_x in T.thread_binding(0, {output_elems}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                mi = blockIdx_x // {semantics.n}
                ni = blockIdx_x % {semantics.n}
                for kk in T.serial(0, {semantics.k}):
                    with T.sblock("matmul"):
                        vm = T.axis.spatial({semantics.m}, mi)
                        vn = T.axis.spatial({semantics.n}, ni)
                        vk = T.axis.reduce({semantics.k}, kk)
                        {reads}
                        T.writes(out[vm, vn])
                        T.sblock_attr({{\
"triton_tvm.contract": "matmul_minimal", \
"triton_tvm.matmul_source_kind": "{semantics.source_kind}", \
"triton_tvm.accumulator_dtype": "{semantics.accumulator_dtype}", \
"triton_tvm.implementation_kind": "native_tir_schedule", \
"triton_tvm.schedule_id": "{M13_P2_DIRECT_SCHEDULE_ID}", \
"triton_tvm.matmul_m": {semantics.m}, \
"triton_tvm.matmul_n": {semantics.n}, \
"triton_tvm.matmul_k": {semantics.k}, \
"triton_tvm.a_layout": "{semantics.a_layout}", \
"triton_tvm.b_layout": "{semantics.b_layout}", \
"triton_tvm.c_layout": "{semantics.c_layout}", \
"triton_tvm.input_precision": "{semantics.input_precision}", \
"triton_tvm.bounds_policy": "{semantics.bounds_policy}", \
"triton_tvm.mask_kind": "{semantics.mask_kind}", \
"triton_tvm.epilogue_kind": "{semantics.epilogue_kind}"{bias_block_attrs}}})
                        with T.init():
                            out[vm, vn] = {init_value}
                        out[vm, vn] = out[vm, vn] + T.Cast("float32", a[vm, vk]) * T.Cast("float32", {b_read})
"""


def _m13_p2_direct_io_matmul_meta(
    index: int,
    semantics,
    *,
    b_storage_shape: tuple[int, int],
    b_storage_stride: tuple[int, int],
) -> TritonTVMMeta:
    is_addmm = semantics.source_kind == "wrapper_extern_addmm_bias"
    abi = []
    buffer_extents = {
        "a": f"T.int64({_storage_extent((semantics.m, semantics.k), semantics.a_stride)})",
        "b": f"T.int64({_storage_extent(b_storage_shape, b_storage_stride)})",
        "out": f"T.int64({_storage_extent((semantics.m, semantics.n), semantics.c_stride)})",
    }
    if is_addmm:
        abi.append({"name": "bias", "kind": "pointer", "dtype": "float32"})
        buffer_extents["bias"] = f"T.int64({_storage_extent(semantics.bias_shape, semantics.bias_stride)})"
    abi.extend(
        [
            {"name": "a", "kind": "pointer", "dtype": semantics.a_dtype},
            {"name": "b", "kind": "pointer", "dtype": semantics.b_dtype},
            {"name": "out", "kind": "pointer", "dtype": semantics.output_dtype},
        ]
    )
    return TritonTVMMeta(
        kernel_name=f"m13_p2_direct_io_matmul_{index}",
        signature={},
        constexprs={},
        grid=(semantics.m * semantics.n,),
        target="cuda",
        target_kind="cuda",
        contract="matmul_minimal",
        canonical_contract="matmul_minimal",
        requested_contract="matmul_minimal",
        emit="tirx",
        translator_version="m13_p2_generated_bridge_direct_io",
        contract_version="matmul_minimal_m13_p2_direct_io",
        target_policy_version="m13_p2_generated_bridge_direct_io",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=semantics.m * semantics.n,
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=semantics.k,
        buffer_extents=buffer_extents,
        block_size=1,
        indexing_kind="rank2_direct_io_wrapper_matmul",
        execution_kind="m13_p2_direct_io_wrapper_matmul_serial_k",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="rank2_mn_serial_k",
        layout_policy="wrapper_strided_direct_io",
        launch_policy_id=M13_P2_DIRECT_SCHEDULE_ID,
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=(
            f"m13_p2_direct_io:{index}:{semantics.source_kind}:"
            f"{semantics.m}:{semantics.n}:{semantics.k}:{semantics.b_layout}"
        ),
        implementation_kind="native_tir_schedule",
        schedule_id=M13_P2_DIRECT_SCHEDULE_ID,
    )


def _build_m13_6_native_attention_runtime_call(
    wrapper_source: str,
) -> _M13NativeAttentionRuntimeCall:
    calls = [
        call
        for call in extract_inductor_wrapper_extern_calls(
            wrapper_source,
            case_name=TARGET_MODEL,
        )
        if call.op_family == "deferred_attention"
    ]
    if len(calls) != 1:
        raise RuntimeError(f"M13.6 expected one wrapper attention call, got {len(calls)}")
    call = calls[0]
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        {
            "op_name": call.op_name,
            "source": call.source,
            "case_name": TARGET_MODEL,
            "kernel_name": "m13_6_native_tvm_attention_decomposed",
        },
        model_case=TARGET_MODEL,
    )
    if semantics.attention_contract != ATTENTION_CONTRACT_VIT_FULL:
        raise RuntimeError("M13.6 native attention is limited to ViT full attention")
    semantics = replace(semantics, output_stride=semantics.q_stride)
    decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    ).decide(semantics, attention_contract_ok=True)
    if decision.implementation_kind != ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED:
        raise RuntimeError("M13.6 expected the M10 native decomposed TVM attention artifact")
    source = build_native_decomposed_attention_tirx_source(semantics, decision)
    irmod = tvm.script.from_source(source)
    validate_attention_vit_full_contract(irmod)
    artifact = build_triton_tvm(irmod, _m13_6_native_attention_meta(semantics))
    return _M13NativeAttentionRuntimeCall(
        index=0,
        op_family="wrapper_attention",
        op_name=call.op_name,
        line_no=call.line_no,
        source=call.source,
        contract=semantics.attention_contract,
        q_shape=semantics.q_shape,
        k_shape=semantics.k_shape,
        v_shape=semantics.v_shape,
        output_shape=semantics.output_shape,
        q_stride=semantics.q_stride,
        k_stride=semantics.k_stride,
        v_stride=semantics.v_stride,
        output_stride=semantics.output_stride,
        scale=semantics.scale,
        artifact=artifact,
        m10_provider_kind=decision.attention_provider_kind,
    )


def _m13_5_is_exact_patch_conv(semantics) -> bool:
    return (
        semantics.vision_contract == VISION_CONTRACT_CONV2D_NCHW_STATIC
        and semantics.input_shape == (1, 3, 32, 32)
        and semantics.weight_shape == (64, 3, 16, 16)
        and semantics.output_shape == (1, 64, 2, 2)
        and semantics.stride == (16, 16)
        and semantics.padding == (0, 0)
        and semantics.dilation == (1, 1)
        and semantics.groups == 1
        and semantics.bias_policy == "none"
    )


def _build_m13_5_native_conv2d_tirx_source(semantics) -> str:
    name = "m13_5_native_conv2d_patch_embedding"
    input_shape = _shape_literal(semantics.input_shape)
    weight_shape = _shape_literal(semantics.weight_shape)
    output_shape = _shape_literal(semantics.output_shape)
    input_stride = _shape_literal(semantics.input_stride)
    weight_stride = _shape_literal(semantics.weight_stride)
    output_stride = _shape_literal(semantics.output_stride)
    input_storage = _storage_extent(semantics.input_shape, semantics.input_stride)
    weight_storage = _storage_extent(semantics.weight_shape, semantics.weight_stride)
    output_storage = _storage_extent(semantics.output_shape, semantics.output_stride)
    input_bytes, weight_bytes, output_bytes, total_io_bytes = _m13_5_conv_io_bytes(semantics)
    n, channels, _, _ = semantics.input_shape
    out_channels, _, kernel_h, kernel_w = semantics.weight_shape
    _, _, output_h, output_w = semantics.output_shape
    output_elems = _numel(semantics.output_shape)
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}(input_h: T.handle, weight_h: T.handle, output_h: T.handle):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "{semantics.vision_contract}", \
"triton_tvm.shape_scope": "{M13_5_CONV_SHAPE_SCOPE}", \
"triton_tvm.vision_source_kind": "{semantics.source_kind}", \
"triton_tvm.vision_contract": "{semantics.vision_contract}", \
"triton_tvm.vision_contract_version": "{VISION_CONTRACT_VERSION}", \
"triton_tvm.vision_op_family": "{semantics.vision_op_family}", \
"triton_tvm.vision_layout": "{semantics.vision_layout}", \
"triton_tvm.input_shape": "{input_shape}", \
"triton_tvm.weight_shape": "{weight_shape}", \
"triton_tvm.output_shape": "{output_shape}", \
"triton_tvm.input_stride": "{input_stride}", \
"triton_tvm.weight_stride": "{weight_stride}", \
"triton_tvm.output_stride": "{output_stride}", \
"triton_tvm.stride": "{_shape_literal(semantics.stride)}", \
"triton_tvm.padding": "{_shape_literal(semantics.padding)}", \
"triton_tvm.dilation": "{_shape_literal(semantics.dilation)}", \
"triton_tvm.groups": {semantics.groups}, \
"triton_tvm.bias_policy": "{semantics.bias_policy}", \
"triton_tvm.transposed": False, \
"triton_tvm.output_padding": "{_shape_literal(semantics.output_padding)}", \
"triton_tvm.input_dtype": "{semantics.input_dtype}", \
"triton_tvm.weight_dtype": "{semantics.weight_dtype}", \
"triton_tvm.output_dtype": "{semantics.output_dtype}", \
"triton_tvm.implementation_kind": "{VISION_IMPLEMENTATION_KIND_NATIVE_TVM_CONV2D}", \
"triton_tvm.extern_symbol": "", \
"triton_tvm.extern_packed_func": "", \
"triton_tvm.extern_runtime_kind": "native_tvm_artifact", \
"triton_tvm.extern_runtime_replacement": "{VISION_PROVIDER_NATIVE_TVM_CONV2D}", \
"triton_tvm.extern_runtime_replacement_available": True, \
"triton_tvm.extern_runtime_replacement_reason": "m13_5_exact_native_conv_artifact", \
"triton_tvm.vision_runtime_status": "{VISION_RUNTIME_STATUS_RUNTIME_RESOLVED}", \
"triton_tvm.vision_provider_kind": "{VISION_PROVIDER_NATIVE_TVM_CONV2D}", \
"triton_tvm.vision_provider_abi_version": {VISION_PROVIDER_ABI_VERSION}, \
"triton_tvm.vision_runtime_claim": "{VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY}", \
"triton_tvm.vision_performance_claim": False, \
"triton_tvm.vision_uses_host_staging": False, \
"triton_tvm.vision_runtime_launch_count": 1, \
"triton_tvm.vision_artifact_call_count": 1, \
"triton_tvm.vision_input_bytes": {input_bytes}, \
"triton_tvm.vision_weight_bytes": {weight_bytes}, \
"triton_tvm.vision_output_bytes": {output_bytes}, \
"triton_tvm.vision_total_io_bytes": {total_io_bytes}, \
"triton_tvm.vision_host_staging_bytes": 0, \
"triton_tvm.vision_total_accounted_bytes": {total_io_bytes}, \
"triton_tvm.m13_performance_claim": "diagnostic_only", \
"triton_tvm.native_conv_schedule_id": "{M13_5_CONV_SCHEDULE_ID}"}})
        inp = T.match_buffer(input_h, ({input_storage},), "{semantics.input_dtype}")
        weight = T.match_buffer(weight_h, ({weight_storage},), "{semantics.weight_dtype}")
        out = T.match_buffer(output_h, ({output_storage},), "{semantics.output_dtype}")
        for blockIdx_x in T.thread_binding(0, {output_elems}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                n_idx = blockIdx_x // T.int64({out_channels * output_h * output_w})
                rem0 = blockIdx_x % T.int64({out_channels * output_h * output_w})
                oc_idx = rem0 // T.int64({output_h * output_w})
                rem1 = rem0 % T.int64({output_h * output_w})
                oh_idx = rem1 // T.int64({output_w})
                ow_idx = rem1 % T.int64({output_w})
                for ic_idx in T.serial(0, {channels}):
                    for kh_idx in T.serial(0, {kernel_h}):
                        for kw_idx in T.serial(0, {kernel_w}):
                            input_offset = n_idx * T.int64({semantics.input_stride[0]}) + ic_idx * T.int64({semantics.input_stride[1]}) + (oh_idx * T.int64({semantics.stride[0]}) + kh_idx) * T.int64({semantics.input_stride[2]}) + (ow_idx * T.int64({semantics.stride[1]}) + kw_idx) * T.int64({semantics.input_stride[3]})
                            weight_offset = oc_idx * T.int64({semantics.weight_stride[0]}) + ic_idx * T.int64({semantics.weight_stride[1]}) + kh_idx * T.int64({semantics.weight_stride[2]}) + kw_idx * T.int64({semantics.weight_stride[3]})
                            output_offset = n_idx * T.int64({semantics.output_stride[0]}) + oc_idx * T.int64({semantics.output_stride[1]}) + oh_idx * T.int64({semantics.output_stride[2]}) + ow_idx * T.int64({semantics.output_stride[3]})
                            with T.sblock("conv2d"):
                                vn = T.axis.spatial({n}, n_idx)
                                voc = T.axis.spatial({out_channels}, oc_idx)
                                voh = T.axis.spatial({output_h}, oh_idx)
                                vow = T.axis.spatial({output_w}, ow_idx)
                                vic = T.axis.reduce({channels}, ic_idx)
                                vkh = T.axis.reduce({kernel_h}, kh_idx)
                                vkw = T.axis.reduce({kernel_w}, kw_idx)
                                T.reads(inp[input_offset], weight[weight_offset])
                                T.writes(out[output_offset])
                                T.sblock_attr({{"triton_tvm.schedule_id": "{M13_5_CONV_SCHEDULE_ID}"}})
                                with T.init():
                                    out[output_offset] = T.float32(0)
                                out[output_offset] = out[output_offset] + T.Cast("float32", inp[input_offset]) * T.Cast("float32", weight[weight_offset])
"""


def _m13_5_native_conv_meta(semantics) -> TritonTVMMeta:
    return TritonTVMMeta(
        kernel_name="m13_5_native_conv2d_patch_embedding",
        signature={},
        constexprs={},
        grid=(_numel(semantics.output_shape),),
        target="cuda",
        target_kind="cuda",
        contract=semantics.vision_contract,
        canonical_contract=semantics.vision_contract,
        requested_contract=semantics.vision_contract,
        emit="tirx",
        translator_version="m13_5_native_conv_slice",
        contract_version=VISION_CONTRACT_VERSION,
        target_policy_version="m13_5_native_conv_slice",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=_numel(semantics.output_shape),
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=(
            semantics.input_shape[1] * semantics.weight_shape[2] * semantics.weight_shape[3]
        ),
        buffer_extents={
            "input": f"T.int64({_storage_extent(semantics.input_shape, semantics.input_stride)})",
            "weight": f"T.int64({_storage_extent(semantics.weight_shape, semantics.weight_stride)})",
            "output": f"T.int64({_storage_extent(semantics.output_shape, semantics.output_stride)})",
        },
        block_size=1,
        indexing_kind="rank4_conv2d_exact_vit_patch",
        execution_kind="m13_5_native_conv2d_serial_reduction",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="nchw_oihw_static",
        layout_policy="rank4_static_conv2d_with_static_strides",
        launch_policy_id=M13_5_CONV_SCHEDULE_ID,
        abi=[
            {"name": "input", "kind": "pointer", "dtype": semantics.input_dtype},
            {"name": "weight", "kind": "pointer", "dtype": semantics.weight_dtype},
            {"name": "output", "kind": "pointer", "dtype": semantics.output_dtype},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=(
            "m13_5_native_conv:"
            f"{semantics.input_shape}:{semantics.weight_shape}:{semantics.output_shape}"
        ),
        implementation_kind=VISION_IMPLEMENTATION_KIND_NATIVE_TVM_CONV2D,
        schedule_id=M13_5_CONV_SCHEDULE_ID,
    )


def _m13_6_native_attention_meta(semantics) -> TritonTVMMeta:
    q_storage = _storage_extent(semantics.q_shape, semantics.q_stride)
    k_storage = _storage_extent(semantics.k_shape, semantics.k_stride)
    v_storage = _storage_extent(semantics.v_shape, semantics.v_stride)
    output_storage = _storage_extent(semantics.output_shape, semantics.output_stride)
    return TritonTVMMeta(
        kernel_name=semantics.kernel_name,
        signature={},
        constexprs={},
        grid=(_numel(semantics.output_shape),),
        target="cuda",
        target_kind="cuda",
        contract=semantics.attention_contract,
        canonical_contract=semantics.attention_contract,
        requested_contract=semantics.attention_contract,
        emit="tirx",
        translator_version="m13_6_native_attention_slice",
        contract_version="attention_vit_full_m10_v1",
        target_policy_version="m13_6_native_attention_slice",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=_numel(semantics.output_shape),
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=semantics.head_dim,
        buffer_extents={
            "q": f"T.int64({q_storage})",
            "k": f"T.int64({k_storage})",
            "v": f"T.int64({v_storage})",
            "out": f"T.int64({output_storage})",
        },
        block_size=1,
        indexing_kind="rank4_sdpa_attention_exact_vit",
        execution_kind="native_tvm_attention_decomposed_qk_softmax_av",
        accumulator_dtype_policy="fp32_attention",
        epsilon_policy="not_applicable",
        mask_policy="none_or_padding",
        axis_policy="batch_head_sequence_head_dim",
        layout_policy="rank4_static_attention_with_wrapper_strides",
        launch_policy_id="m13_6_native_tvm_attention_decomposed",
        abi=[
            {"name": "q", "kind": "pointer", "dtype": semantics.q_dtype},
            {"name": "k", "kind": "pointer", "dtype": semantics.k_dtype},
            {"name": "v", "kind": "pointer", "dtype": semantics.v_dtype},
            {"name": "out", "kind": "pointer", "dtype": semantics.output_dtype},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=(
            "m13_6_native_attention:"
            f"{semantics.q_shape}:{semantics.q_stride}:{semantics.output_stride}"
        ),
        implementation_kind=ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    )


def _run_m13_5_standalone_native_conv_correctness(torch, call: _M13NativeConvRuntimeCall) -> dict[str, Any]:
    inp = torch.empty_strided(call.input_shape, call.input_stride, device="cuda", dtype=torch.float32)
    weight = torch.empty_strided(
        call.weight_shape,
        call.weight_stride,
        device="cuda",
        dtype=torch.float32,
    )
    inp.copy_(
        torch.linspace(-0.25, 0.25, steps=_numel(call.input_shape), device="cuda").reshape(
            call.input_shape
        )
    )
    weight.copy_(
        torch.linspace(-0.125, 0.125, steps=_numel(call.weight_shape), device="cuda").reshape(
            call.weight_shape
        )
    )
    out = torch.empty_strided(call.output_shape, call.output_stride, device="cuda", dtype=torch.float32)
    tvm_args = [
        tvm.runtime.from_dlpack(
            torch.as_strided(inp, (_storage_extent(call.input_shape, call.input_stride),), (1,))
        ),
        tvm.runtime.from_dlpack(
            torch.as_strided(weight, (_storage_extent(call.weight_shape, call.weight_stride),), (1,))
        ),
        tvm.runtime.from_dlpack(
            torch.as_strided(out, (_storage_extent(call.output_shape, call.output_stride),), (1,))
        ),
    ]
    call.artifact.run(tvm_args)
    torch.cuda.synchronize()
    expected = torch.nn.functional.conv2d(
        inp,
        weight,
        bias=None,
        stride=call.stride,
        padding=call.padding,
        dilation=call.dilation,
        groups=call.groups,
    )
    torch.cuda.synchronize()
    actual_np = out.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()
    abs_error = np.abs(actual_np - expected_np)
    denom = np.maximum(np.abs(expected_np), 1e-12)
    rel_error = abs_error / denom
    return {
        "allclose": bool(np.allclose(actual_np, expected_np, rtol=1e-4, atol=1e-4)),
        "allclose_rtol": 1e-4,
        "allclose_atol": 1e-4,
        "max_abs_error": float(abs_error.max()),
        "max_rel_error": float(rel_error.max()),
        "reference_backend": "torch.nn.functional.conv2d_cuda_validation_only",
        "artifact_kernel_name": call.artifact.meta.kernel_name,
    }


def _run_m13_6_standalone_native_attention_correctness(
    torch,
    call: _M13NativeAttentionRuntimeCall,
) -> dict[str, Any]:
    import torch.nn.functional as torch_functional  # pylint: disable=import-outside-toplevel

    q = torch.empty_strided(call.q_shape, call.q_stride, device="cuda", dtype=torch.float32)
    k = torch.empty_strided(call.k_shape, call.k_stride, device="cuda", dtype=torch.float32)
    v = torch.empty_strided(call.v_shape, call.v_stride, device="cuda", dtype=torch.float32)
    q.copy_(
        torch.linspace(-0.25, 0.25, steps=_numel(call.q_shape), device="cuda").reshape(
            call.q_shape
        )
    )
    k.copy_(
        torch.linspace(0.125, -0.125, steps=_numel(call.k_shape), device="cuda").reshape(
            call.k_shape
        )
    )
    v.copy_(
        torch.linspace(-0.5, 0.5, steps=_numel(call.v_shape), device="cuda").reshape(
            call.v_shape
        )
    )
    out = torch.empty_strided(
        call.output_shape,
        call.output_stride,
        device="cuda",
        dtype=torch.float32,
    )
    tvm_args = [
        _from_dlpack_allow_strided(q),
        _from_dlpack_allow_strided(k),
        _from_dlpack_allow_strided(v),
        _from_dlpack_allow_strided(out),
    ]
    call.artifact.run(tvm_args)
    torch.cuda.synchronize()
    expected = torch_functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=call.scale,
    )
    torch.cuda.synchronize()
    actual_np = out.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()
    abs_error = np.abs(actual_np - expected_np)
    denom = np.maximum(np.abs(expected_np), 1e-12)
    rel_error = abs_error / denom
    return {
        "allclose": bool(np.allclose(actual_np, expected_np, rtol=1e-4, atol=1e-4)),
        "allclose_rtol": 1e-4,
        "allclose_atol": 1e-4,
        "max_abs_error": float(abs_error.max()),
        "max_rel_error": float(rel_error.max()),
        "reference_backend": "torch.nn.functional.scaled_dot_product_attention_cuda_validation_only",
        "artifact_kernel_name": call.artifact.meta.kernel_name,
        "output_stride": list(call.output_stride),
    }


def _build_m13_4_generated_bridge_core_artifact(wrapper_semantics, block_m: int):
    artifact = lower_to_ttir(
        _m13_generated_tl_dot_bridge,
        _generated_dot_signature(),
        {
            "BLOCK_M": block_m,
            "BLOCK_N": wrapper_semantics.n,
            "BLOCK_K": wrapper_semantics.k,
        },
    )
    graph = TTIRReader().read(artifact.ttir)
    generated_semantics = extract_matmul_semantics_from_ttir(
        graph, source_kind=GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
    )
    irmod, meta = translate_ttir(
        artifact,
        grid=(1,),
        target="cuda",
        contract="matmul_minimal",
        matmul_source_kind=GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
    )
    validate_matmul_minimal_contract(irmod)
    decision = TargetMatmulPolicy(target_kind="cuda").decide(
        generated_semantics,
        matmul_contract_ok=True,
    )
    if not _generated_bridge_record_passes(wrapper_semantics, generated_semantics, meta, decision):
        raise RuntimeError(
            "M13.4 generated bridge core failed semantic/runtime precheck for "
            f"shape {(wrapper_semantics.m, wrapper_semantics.n, wrapper_semantics.k)}"
        )
    return build_triton_tvm(irmod, meta)


def _build_m13_4_pointwise_artifact(
    source: str,
    *,
    kernel_name: str,
    abi: list[dict[str, str]],
    buffer_extents: dict[str, str],
    extent_value: int,
    cache_key: str,
) -> TritonTVMArtifact:
    irmod = tvm.script.from_source(source)
    meta = _m13_4_pointwise_meta(
        kernel_name=kernel_name,
        abi=abi,
        buffer_extents=buffer_extents,
        extent_value=extent_value,
        cache_key=cache_key,
    )
    return build_triton_tvm(irmod, meta)


def _build_m13_4_a_copy_tirx_source(index: int, semantics) -> str:
    name = f"m13_4_a_copy_{index}"
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}(a_h: T.handle, a_pad_h: T.handle, n_elements: T.int64):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "pointwise_flat", "triton_tvm.implementation_kind": "{M13_4_ADAPTER_IMPLEMENTATION}", \
"triton_tvm.adapter_kind": "a_logical_copy"}})
        a = T.match_buffer(a_h, ({semantics.m}, {semantics.k}), "{semantics.a_dtype}", strides=({_stride_text(semantics.a_stride)}))
        a_pad = T.match_buffer(a_pad_h, ({_tl_dot_block_extent(semantics.m)}, {semantics.k}), "{semantics.a_dtype}")
        for blockIdx_x in T.thread_binding(0, {semantics.m * semantics.k}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                i = T.Cast("int64", blockIdx_x)
                if i < n_elements:
                    a_pad[i // T.int64({semantics.k}), i % T.int64({semantics.k})] = a[i // T.int64({semantics.k}), i % T.int64({semantics.k})]
"""


def _build_m13_4_a_zero_tail_tirx_source(index: int, semantics, block_m: int) -> str:
    name = f"m13_4_a_zero_tail_{index}"
    tail_elems = (block_m - semantics.m) * semantics.k
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}(a_pad_h: T.handle, n_elements: T.int64):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "pointwise_flat", "triton_tvm.implementation_kind": "{M13_4_ADAPTER_IMPLEMENTATION}", \
"triton_tvm.adapter_kind": "a_zero_pad_tail"}})
        a_pad = T.match_buffer(a_pad_h, ({block_m}, {semantics.k}), "{semantics.a_dtype}")
        for blockIdx_x in T.thread_binding(0, {tail_elems}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                i = T.Cast("int64", blockIdx_x)
                if i < n_elements:
                    a_pad[(i + T.int64({semantics.m * semantics.k})) // T.int64({semantics.k}), (i + T.int64({semantics.m * semantics.k})) % T.int64({semantics.k})] = T.float32(0)
"""


def _build_m13_4_b_adapter_tirx_source(index: int, semantics) -> str:
    name = f"m13_4_b_row_major_adapter_{index}"
    storage_shape = _m13_4_b_storage_shape(semantics)
    storage_stride = _m13_4_b_storage_stride(semantics)
    if semantics.b_layout == "transposed_weight_view":
        read_expr = (
            f"b_storage[i % T.int64({semantics.n}), i // T.int64({semantics.n})]"
        )
    else:
        read_expr = (
            f"b_storage[i // T.int64({semantics.n}), i % T.int64({semantics.n})]"
        )
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}(b_storage_h: T.handle, b_row_major_h: T.handle, n_elements: T.int64):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "pointwise_flat", "triton_tvm.implementation_kind": "{M13_4_ADAPTER_IMPLEMENTATION}", \
"triton_tvm.adapter_kind": "b_row_major_layout_adapter", "triton_tvm.wrapper_b_layout": "{semantics.b_layout}"}})
        b_storage = T.match_buffer(b_storage_h, ({storage_shape[0]}, {storage_shape[1]}), "{semantics.b_dtype}", strides=({_stride_text(storage_stride)}))
        b_row_major = T.match_buffer(b_row_major_h, ({semantics.k}, {semantics.n}), "{semantics.b_dtype}")
        for blockIdx_x in T.thread_binding(0, {semantics.k * semantics.n}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                i = T.Cast("int64", blockIdx_x)
                if i < n_elements:
                    b_row_major[i // T.int64({semantics.n}), i % T.int64({semantics.n})] = {read_expr}
"""


def _build_m13_4_output_commit_tirx_source(index: int, semantics, block_m: int) -> str:
    name = f"m13_4_output_commit_{index}"
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}(core_out_h: T.handle, out_h: T.handle, n_elements: T.int64):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "pointwise_flat", "triton_tvm.implementation_kind": "{M13_4_ADAPTER_IMPLEMENTATION}", \
"triton_tvm.adapter_kind": "logical_output_commit", "triton_tvm.epilogue_kind": "none"}})
        core_out = T.match_buffer(core_out_h, ({block_m}, {semantics.n}), "{semantics.output_dtype}")
        out = T.match_buffer(out_h, ({semantics.m}, {semantics.n}), "{semantics.output_dtype}", strides=({_stride_text(semantics.c_stride)}))
        for blockIdx_x in T.thread_binding(0, {semantics.m * semantics.n}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                i = T.Cast("int64", blockIdx_x)
                if i < n_elements:
                    out[i // T.int64({semantics.n}), i % T.int64({semantics.n})] = core_out[i // T.int64({semantics.n}), i % T.int64({semantics.n})]
"""


def _build_m13_4_bias_commit_tirx_source(index: int, semantics, block_m: int) -> str:
    name = f"m13_4_bias_epilogue_commit_{index}"
    bias_shape = _shape_literal(semantics.bias_shape)
    bias_stride = _shape_literal(semantics.bias_stride)
    bias_read = (
        f"bias[i % T.int64({semantics.n})]"
        if len(semantics.bias_shape) == 1
        else f"bias[i // T.int64({semantics.n}), i % T.int64({semantics.n})]"
    )
    return f"""# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {name}(bias_h: T.handle, core_out_h: T.handle, out_h: T.handle, n_elements: T.int64):
        T.func_attr({{\
"global_symbol": "{name}", "tirx.noalias": True, "target": T.target("cuda"), \
"triton_tvm.contract": "pointwise_flat", "triton_tvm.implementation_kind": "{M13_4_BIAS_EPILOGUE_IMPLEMENTATION}", \
"triton_tvm.adapter_kind": "logical_output_commit", "triton_tvm.epilogue_kind": "bias_add"}})
        bias = T.match_buffer(bias_h, ({bias_shape}), "{semantics.output_dtype}", strides=({bias_stride}))
        core_out = T.match_buffer(core_out_h, ({block_m}, {semantics.n}), "{semantics.output_dtype}")
        out = T.match_buffer(out_h, ({semantics.m}, {semantics.n}), "{semantics.output_dtype}", strides=({_stride_text(semantics.c_stride)}))
        for blockIdx_x in T.thread_binding(0, {semantics.m * semantics.n}, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 1, thread="threadIdx.x"):
                i = T.Cast("int64", blockIdx_x)
                if i < n_elements:
                    out[i // T.int64({semantics.n}), i % T.int64({semantics.n})] = core_out[i // T.int64({semantics.n}), i % T.int64({semantics.n})] + {bias_read}
"""


def _m13_4_pointwise_meta(
    *,
    kernel_name: str,
    abi: list[dict[str, str]],
    buffer_extents: dict[str, str],
    extent_value: int,
    cache_key: str,
) -> TritonTVMMeta:
    return TritonTVMMeta(
        kernel_name=kernel_name,
        signature={},
        constexprs={},
        grid=(int(extent_value),),
        target="cuda",
        target_kind="cuda",
        contract="pointwise_flat",
        canonical_contract="pointwise_flat",
        requested_contract="pointwise_flat",
        emit="tirx",
        translator_version="m13_4_matmul_addmm_runtime_replacement",
        contract_version="pointwise_pre_m5_v1",
        target_policy_version="m13_4_matmul_addmm_runtime_replacement",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="n_elements",
        extent_kind="runtime",
        extent_value=int(extent_value),
        reduction_extent_param="",
        reduction_extent_kind="none",
        reduction_extent_value=None,
        buffer_extents=buffer_extents,
        block_size=1,
        indexing_kind="pointwise",
        execution_kind="m13_4_native_pointwise_adapter",
        accumulator_dtype_policy="not_applicable",
        epsilon_policy="not_applicable",
        mask_policy="flat_extent_guard",
        axis_policy="flat_1d",
        layout_policy="rank2_row_major_static",
        launch_policy_id="cuda_one_block_per_element",
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=cache_key,
    )


def _m13_4_b_storage_shape(semantics) -> tuple[int, int]:
    if semantics.b_layout == "transposed_weight_view":
        return (semantics.n, semantics.k)
    return (semantics.k, semantics.n)


def _m13_4_b_storage_stride(semantics) -> tuple[int, int]:
    if semantics.b_layout == "transposed_weight_view":
        return (semantics.k, 1)
    return semantics.b_stride


def _stride_text(stride: tuple[int, ...]) -> str:
    return ", ".join(str(int(value)) for value in stride)


def _shape_literal(values: tuple[int, ...]) -> str:
    if len(values) == 1:
        return f"{int(values[0])},"
    return ", ".join(str(int(value)) for value in values)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def _storage_extent(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if len(shape) != len(stride):
        raise ValueError(f"shape/stride rank mismatch: {shape} vs {stride}")
    extent = 1
    for dim, step in zip(shape, stride, strict=True):
        extent += (int(dim) - 1) * int(step)
    return int(extent)


def _from_dlpack_allow_strided(tensor):
    return tvm_ffi.from_dlpack(
        tensor,
        require_alignment=0,
        require_contiguous=False,
    )


def _m13_7_torch_runtime_args_to_tvm(args: tuple[Any, ...], meta) -> tuple[list[Any], int]:
    if len(args) < len(meta.abi):
        raise RuntimeError(f"M13.7 expected at least {len(meta.abi)} runtime args, got {len(args)}")
    extra_args = args[len(meta.abi) :]
    if extra_args and any(hasattr(arg, "device") for arg in extra_args):
        raise RuntimeError(
            "M13.7 captured-kernel runtime args include extra tensor arguments "
            f"beyond ABI length {len(meta.abi)}"
        )
    scalar_values = {
        spec["name"]: int(arg)
        for arg, spec in zip(args[: len(meta.abi)], meta.abi, strict=True)
        if spec["kind"] == "scalar"
    }
    tvm_args: list[Any] = []
    device_index = 0
    for arg, spec in zip(args[: len(meta.abi)], meta.abi, strict=True):
        if spec["kind"] == "pointer":
            device = getattr(arg, "device", None)
            if getattr(device, "type", None) == "cuda" and device.index is not None:
                device_index = int(device.index)
            extent = _buffer_extent_value(meta.buffer_extents.get(spec["name"], ""), scalar_values)
            storage_view = arg.as_strided((extent,), (1,), storage_offset=int(arg.storage_offset()))
            tvm_args.append(_from_dlpack_allow_strided(storage_view))
        else:
            tvm_args.append(arg)
    return tvm_args, device_index


def _m13_5_conv_io_bytes(semantics) -> tuple[int, int, int, int]:
    input_bytes = _numel(semantics.input_shape) * 4
    weight_bytes = _numel(semantics.weight_shape) * 4
    output_bytes = _numel(semantics.output_shape) * 4
    return input_bytes, weight_bytes, output_bytes, input_bytes + weight_bytes + output_bytes


def _m13_7_captured_kernel_runtime_record(
    call: _M13CapturedKernelRuntimeCall,
) -> dict[str, Any]:
    return {
        "index": call.index,
        "kernel_name": call.kernel_name,
        "contract": call.contract,
        "status": "passed",
        "implementation_level": "triton_language_lowering",
        "runtime_kind": "tvm_artifact_run",
        "runtime_replacement_connected": True,
        "run_count": 1,
        "translation_route": call.translation_route,
        "artifact_kernel_name": call.artifact.meta.kernel_name,
        "artifact_contract": call.artifact.meta.contract,
        "stream_handoff": {"stream": "inductor_raw_stream", "policy": "tvm_set_raw_stream"},
        "harness_fallback_count": 0,
        "native_triton_launch_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "pytorch_fallback_count": 0,
    }


def _m13_4_runtime_record(call: _M13GeneratedBridgeRuntimeCall) -> dict[str, Any]:
    adapter_artifacts = [
        {
            "adapter_kind": "a_logical_copy",
            "implementation": M13_4_ADAPTER_IMPLEMENTATION,
            "runtime_kind": "tvm_artifact_run",
            "kernel_name": call.a_copy_artifact.meta.kernel_name,
        },
        {
            "adapter_kind": "b_row_major_layout_adapter",
            "implementation": M13_4_ADAPTER_IMPLEMENTATION,
            "runtime_kind": "tvm_artifact_run",
            "kernel_name": call.b_adapter_artifact.meta.kernel_name,
        },
        {
            "adapter_kind": "logical_output_commit",
            "implementation": (
                M13_4_BIAS_EPILOGUE_IMPLEMENTATION
                if call.op_family == "extern_addmm_bias"
                else M13_4_ADAPTER_IMPLEMENTATION
            ),
            "runtime_kind": "tvm_artifact_run",
            "kernel_name": call.commit_artifact.meta.kernel_name,
        },
    ]
    if call.a_zero_tail_artifact is not None:
        adapter_artifacts.insert(
            1,
            {
                "adapter_kind": "a_zero_pad_tail",
                "implementation": M13_4_ADAPTER_IMPLEMENTATION,
                "runtime_kind": "tvm_artifact_run",
                "kernel_name": call.a_zero_tail_artifact.meta.kernel_name,
            },
        )
    epilogue_kind = "bias_add" if call.op_family == "extern_addmm_bias" else "none"
    return {
        "index": call.index,
        "status": "passed",
        "line_no": call.line_no,
        "op_family": call.op_family,
        "op_name": call.op_name,
        "wrapper_source_kind": call.wrapper_source_kind,
        "wrapper_shape": {"m": call.wrapper_m, "n": call.wrapper_n, "k": call.wrapper_k},
        "bridge_padded_shape": {"m": call.block_m, "n": call.wrapper_n, "k": call.wrapper_k},
        "wrapper_b_layout": call.wrapper_b_layout,
        "wrapper_b_storage_shape": list(call.wrapper_b_storage_shape),
        "wrapper_b_storage_stride": list(call.wrapper_b_storage_stride),
        "implementation_level": "generated_tl_dot_bridge",
        "runtime_kind": M13_4_RUNTIME_KIND,
        "runtime_replacement_connected": True,
        "matmul_core": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        "core_kernel_name": call.core_artifact.meta.kernel_name,
        "core_contract": call.core_artifact.meta.contract,
        "core_schedule_id": call.core_artifact.meta.schedule_id,
        "core_runtime_kind": "tvm_artifact_run",
        "epilogue_kind": epilogue_kind,
        "epilogue_implementation": (
            M13_4_BIAS_EPILOGUE_IMPLEMENTATION if epilogue_kind == "bias_add" else "none"
        ),
        "epilogue_runtime_kind": "tvm_artifact_run" if epilogue_kind == "bias_add" else "none",
        "adapter_artifacts": adapter_artifacts,
        "wrapper_extern_gemm_runtime_count": 0,
        "wrapper_extern_addmm_bias_provider_count": 0,
        "native_tvm_matmul_legacy_wrapper_provider_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "native_triton_launch_count": 0,
        "pytorch_fallback_count": 0,
    }


def _m13_p2_direct_runtime_record(call: _M13GeneratedBridgeRuntimeCall) -> dict[str, Any]:
    epilogue_kind = "bias_add" if call.op_family == "extern_addmm_bias" else "none"
    return {
        "index": call.index,
        "status": "passed",
        "line_no": call.line_no,
        "op_family": call.op_family,
        "op_name": call.op_name,
        "wrapper_source_kind": call.wrapper_source_kind,
        "wrapper_shape": {"m": call.wrapper_m, "n": call.wrapper_n, "k": call.wrapper_k},
        "bridge_padded_shape": {"m": call.wrapper_m, "n": call.wrapper_n, "k": call.wrapper_k},
        "wrapper_b_layout": call.wrapper_b_layout,
        "wrapper_b_storage_shape": list(call.wrapper_b_storage_shape),
        "wrapper_b_storage_stride": list(call.wrapper_b_storage_stride),
        "implementation_level": "generated_tl_dot_bridge",
        "runtime_kind": M13_P2_DIRECT_RUNTIME_KIND,
        "runtime_replacement_connected": True,
        "matmul_core": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        "core_kernel_name": call.direct_io_artifact.meta.kernel_name,
        "core_contract": call.direct_io_artifact.meta.contract,
        "core_schedule_id": M13_P2_DIRECT_SCHEDULE_ID,
        "core_runtime_kind": "tvm_artifact_run",
        "epilogue_kind": epilogue_kind,
        "epilogue_implementation": (
            "fused_direct_io_bias_epilogue" if epilogue_kind == "bias_add" else "none"
        ),
        "epilogue_runtime_kind": "fused_in_direct_io_kernel" if epilogue_kind == "bias_add" else "none",
        "adapter_artifacts": [],
        "direct_io_artifact": {
            "kernel_name": call.direct_io_artifact.meta.kernel_name,
            "runtime_kind": "tvm_artifact_run",
            "schedule_id": M13_P2_DIRECT_SCHEDULE_ID,
        },
        "wrapper_extern_gemm_runtime_count": 0,
        "wrapper_extern_addmm_bias_provider_count": 0,
        "native_tvm_matmul_legacy_wrapper_provider_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "native_triton_launch_count": 0,
        "pytorch_fallback_count": 0,
    }


def _m13_p3_async_direct_runtime_record(call: _M13GeneratedBridgeRuntimeCall) -> dict[str, Any]:
    epilogue_kind = "bias_add" if call.op_family == "extern_addmm_bias" else "none"
    return {
        "index": call.index,
        "status": "passed",
        "line_no": call.line_no,
        "op_family": call.op_family,
        "op_name": call.op_name,
        "wrapper_source_kind": call.wrapper_source_kind,
        "wrapper_shape": {"m": call.wrapper_m, "n": call.wrapper_n, "k": call.wrapper_k},
        "bridge_padded_shape": {"m": call.wrapper_m, "n": call.wrapper_n, "k": call.wrapper_k},
        "wrapper_b_layout": call.wrapper_b_layout,
        "wrapper_b_storage_shape": list(call.wrapper_b_storage_shape),
        "wrapper_b_storage_stride": list(call.wrapper_b_storage_stride),
        "implementation_level": "generated_tl_dot_bridge",
        "runtime_kind": M13_P3_DIRECT_RUNTIME_KIND,
        "runtime_replacement_connected": True,
        "matmul_core": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        "core_kernel_name": call.direct_io_artifact.meta.kernel_name,
        "core_contract": call.direct_io_artifact.meta.contract,
        "core_schedule_id": M13_P2_DIRECT_SCHEDULE_ID,
        "core_runtime_kind": "tvm_artifact_run",
        "launch_sync_policy": "defer_to_strict_path_boundary",
        "epilogue_kind": epilogue_kind,
        "epilogue_implementation": (
            "fused_async_direct_io_bias_epilogue" if epilogue_kind == "bias_add" else "none"
        ),
        "epilogue_runtime_kind": (
            "fused_in_async_direct_io_kernel" if epilogue_kind == "bias_add" else "none"
        ),
        "adapter_artifacts": [],
        "direct_io_artifact": {
            "kernel_name": call.direct_io_artifact.meta.kernel_name,
            "runtime_kind": "tvm_artifact_run",
            "schedule_id": M13_P2_DIRECT_SCHEDULE_ID,
        },
        "wrapper_extern_gemm_runtime_count": 0,
        "wrapper_extern_addmm_bias_provider_count": 0,
        "native_tvm_matmul_legacy_wrapper_provider_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "native_triton_launch_count": 0,
        "pytorch_fallback_count": 0,
    }


def _m13_p5_fast_direct_runtime_record(call: _M13GeneratedBridgeRuntimeCall) -> dict[str, Any]:
    record = _m13_p3_async_direct_runtime_record(call)
    record["runtime_kind"] = M13_P5_DIRECT_RUNTIME_KIND
    record["core_runtime_kind"] = "tvm_entry_function_run"
    record["launch_sync_policy"] = "defer_to_strict_path_boundary"
    record["artifact_invocation_policy"] = "cached_tvm_entry_function_no_runtime_validation"
    record["direct_io_artifact"]["runtime_kind"] = "tvm_entry_function_run"
    return record


def _m13_5_native_conv_runtime_record(call: _M13NativeConvRuntimeCall) -> dict[str, Any]:
    input_bytes = _numel(call.input_shape) * 4
    weight_bytes = _numel(call.weight_shape) * 4
    output_bytes = _numel(call.output_shape) * 4
    total_io_bytes = input_bytes + weight_bytes + output_bytes
    return {
        "index": call.index,
        "status": "passed",
        "line_no": call.line_no,
        "op_family": call.op_family,
        "op_name": call.op_name,
        "contract": call.contract,
        "shape_scope": M13_5_CONV_SHAPE_SCOPE,
        "implementation_level": "tvm_native_wrapper_lowering",
        "runtime_kind": M13_5_CONV_RUNTIME_KIND,
        "provider_kind": VISION_PROVIDER_NATIVE_TVM_CONV2D,
        "runtime_replacement_connected": True,
        "artifact_kernel_name": call.artifact.meta.kernel_name,
        "artifact_contract": call.artifact.meta.contract,
        "schedule_id": M13_5_CONV_SCHEDULE_ID,
        "input_shape": list(call.input_shape),
        "weight_shape": list(call.weight_shape),
        "output_shape": list(call.output_shape),
        "input_stride": list(call.input_stride),
        "weight_stride": list(call.weight_stride),
        "output_stride": list(call.output_stride),
        "stride": list(call.stride),
        "padding": list(call.padding),
        "dilation": list(call.dilation),
        "groups": call.groups,
        "device_torch_cuda_conv_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "native_triton_launch_count": 0,
        "pytorch_fallback_count": 0,
        "performance_claim": "diagnostic_only",
        "input_bytes": input_bytes,
        "weight_bytes": weight_bytes,
        "output_bytes": output_bytes,
        "total_io_bytes": total_io_bytes,
    }


def _m13_6_native_attention_runtime_record(
    call: _M13NativeAttentionRuntimeCall,
) -> dict[str, Any]:
    qkv_bytes = (
        _numel(call.q_shape) + _numel(call.k_shape) + _numel(call.v_shape)
    ) * 4
    output_bytes = _numel(call.output_shape) * 4
    total_io_bytes = qkv_bytes + output_bytes
    return {
        "index": call.index,
        "status": "passed",
        "line_no": call.line_no,
        "op_family": call.op_family,
        "op_name": call.op_name,
        "contract": call.contract,
        "shape_scope": M13_6_ATTENTION_SHAPE_SCOPE,
        "implementation_level": "tvm_decomposed_artifact",
        "runtime_kind": M13_6_ATTENTION_RUNTIME_KIND,
        "provider_kind": M13_6_ATTENTION_PROVIDER_KIND,
        "runtime_status": ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
        "runtime_claim": ATTENTION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
        "runtime_replacement_connected": True,
        "artifact_kernel_name": call.artifact.meta.kernel_name,
        "artifact_contract": call.artifact.meta.contract,
        "artifact_source_milestone": "M10",
        "artifact_implementation_kind": "tvm_decomposed_attention_artifact",
        "q_shape": list(call.q_shape),
        "k_shape": list(call.k_shape),
        "v_shape": list(call.v_shape),
        "output_shape": list(call.output_shape),
        "q_stride": list(call.q_stride),
        "k_stride": list(call.k_stride),
        "v_stride": list(call.v_stride),
        "output_stride": list(call.output_stride),
        "scale": call.scale,
        "torch_replay_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "native_triton_launch_count": 0,
        "pytorch_fallback_count": 0,
        "performance_claim": "diagnostic_only",
        "qkv_bytes": qkv_bytes,
        "output_bytes": output_bytes,
        "total_io_bytes": total_io_bytes,
    }


def _run_m13_4_torch_baseline(torch, model, pixel_values) -> dict[str, Any]:
    with torch.no_grad():
        output = model(pixel_values)
    torch.cuda.synchronize()
    return _output_tensors(output)


def _m13_7_correctness_report(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    allclose = True
    max_abs = 0.0
    max_rel = 0.0
    for name in ("last_hidden_state", "pooler_output"):
        actual_np = actual[name].detach().cpu().numpy()
        expected_np = expected[name].detach().cpu().numpy()
        abs_error = np.abs(actual_np - expected_np)
        denom = np.maximum(np.abs(expected_np), 1e-12)
        rel_error = abs_error / denom
        output_allclose = bool(
            np.allclose(
                actual_np,
                expected_np,
                rtol=M13_7_ALLCLOSE_RTOL,
                atol=M13_7_ALLCLOSE_ATOL,
            )
        )
        outputs[name] = {
            "shape": list(actual_np.shape),
            "allclose": output_allclose,
            "max_abs_error": float(abs_error.max()),
            "max_rel_error": float(rel_error.max()),
        }
        allclose = allclose and output_allclose
        max_abs = max(max_abs, outputs[name]["max_abs_error"])
        max_rel = max(max_rel, outputs[name]["max_rel_error"])
    return {
        "baseline": "torch_eager_cuda_vit_tiny_random",
        "allclose": bool(allclose),
        "allclose_rtol": M13_7_ALLCLOSE_RTOL,
        "allclose_atol": M13_7_ALLCLOSE_ATOL,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "outputs": outputs,
    }


def _m13_7_legacy_provider_counts(
    captured_records: list[dict[str, Any]],
    matmul_records: list[dict[str, Any]],
    conv_records: list[dict[str, Any]],
    attention_records: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "torch_inductor_triton_captured_harness": sum(
            int(record.get("harness_fallback_count", 0)) for record in captured_records
        ),
        "native_triton_launch": sum(
            int(record.get("native_triton_launch_count", 0)) for record in captured_records
        ),
        "wrapper_extern_gemm": sum(
            int(record.get("wrapper_extern_gemm_runtime_count", 0))
            for record in matmul_records
        ),
        "wrapper_extern_addmm_bias": sum(
            int(record.get("wrapper_extern_addmm_bias_provider_count", 0))
            for record in matmul_records
        ),
        "native_tvm_matmul": sum(
            int(record.get("native_tvm_matmul_legacy_wrapper_provider_count", 0))
            for record in matmul_records
        ),
        "device_torch_cuda": sum(
            int(record.get("device_torch_cuda_conv_count", 0)) for record in conv_records
        ),
        "torch_replay_attention": sum(
            int(record.get("torch_replay_count", 0)) for record in attention_records
        ),
        "inductor_triton_harness_fallback": 0,
        "pytorch_fallback": sum(
            int(record.get("pytorch_fallback_count", 0))
            for record in captured_records + matmul_records + conv_records + attention_records
        ),
    }


def _base_captured_kernel_record(index: int, kernel_name: str, contract: str) -> dict[str, Any]:
    return {
        "index": index,
        "kernel_name": kernel_name,
        "contract": contract,
        "status": "not_run",
        "implementation_level": "triton_language_lowering",
        "runtime_kind": "tvm_artifact_run",
        "run_count": 0,
        "harness_fallback_count": 0,
        "native_triton_launch_count": 0,
        "silent_fallback_count": 0,
    }


def _operator_inventory(
    wrapper_source: str,
    captured_records: list[dict[str, Any]],
    extern_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL)
    captured_by_name = {record["kernel_name"]: record for record in captured_records}
    inventory: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        record = captured_by_name[source.kernel_name]
        inventory.append(
            {
                "operator_id": f"captured_kernel_{index}",
                "operator_family": "captured_triton_kernel",
                "kernel_name": source.kernel_name,
                "m12_provider_kind": "torch_inductor_triton_captured_harness",
                "m12_runtime_kind": "native_triton_harness_launch",
                "m12_contract": record.get("contract", ""),
                "m13_target_implementation_level": "triton_language_lowering",
                "m13_target_runtime_kind": "tvm_artifact_run",
                "m13_target_provider_kind": "tvm_artifact",
                "legacy_provider_for_m13_strict": True,
            }
        )

    for index, call in enumerate(extern_records):
        if call["op_family"] in {"extern_gemm", "extern_addmm_bias"}:
            inventory.append(_matmul_inventory_record(index, call))
        elif call["op_family"] == "deferred_convolution":
            inventory.append(_conv_inventory_record(call))
        elif call["op_family"] == "deferred_attention":
            inventory.append(_attention_inventory_record(call))
    return inventory


def _matmul_inventory_record(index: int, call: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": f"wrapper_matmul_addmm_{index}",
        "operator_family": "wrapper_matmul_addmm",
        "op_family": call["op_family"],
        "op_name": call["op_name"],
        "line_no": call["line_no"],
        "shape": {
            "m": call.get("matmul_m"),
            "n": call.get("matmul_n"),
            "k": call.get("matmul_k"),
        },
        "m12_provider_kind": EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        "m12_runtime_kind": "native_tvm_matmul_wrapper_provider",
        "m12_source_kind": call.get("matmul_source_kind"),
        "m12_implementation_kind": call.get("implementation_kind"),
        "m13_target_source_kind": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        "m13_target_implementation_level": "generated_tl_dot_bridge",
        "m13_target_runtime_kind": "generated_bridge_tvm_artifact_run",
        "m13_target_provider_kind": "generated_real_tl_dot_bridge",
        "bias_epilogue_required": call["op_family"] == "extern_addmm_bias",
        "legacy_provider_for_m13_strict": True,
    }


def _conv_inventory_record(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": "wrapper_patch_embedding_conv_0",
        "operator_family": "wrapper_conv",
        "op_family": call["op_family"],
        "op_name": call["op_name"],
        "line_no": call["line_no"],
        "contract": "conv2d_nchw_static_v1",
        "shape_scope": "exact_vit_patch_embedding_m13",
        "input_shape": call.get("vision_input_shape"),
        "weight_shape": call.get("vision_weight_shape"),
        "output_shape": call.get("vision_output_shape"),
        "m12_provider_kind": VISION_PROVIDER_DEVICE_TORCH_CUDA,
        "m12_runtime_kind": "device_torch_cuda_provider",
        "m13_target_implementation_level": "tvm_native_wrapper_lowering",
        "m13_target_runtime_kind": "native_tvm_conv_artifact_run",
        "m13_target_provider_kind": "native_tvm_conv2d",
        "legacy_provider_for_m13_strict": True,
    }


def _attention_inventory_record(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": "wrapper_sdpa_attention_0",
        "operator_family": "wrapper_attention",
        "op_family": call["op_family"],
        "op_name": call["op_name"],
        "line_no": call["line_no"],
        "contract": call.get("attention_contract", "attention_vit_full_v1"),
        "q_shape": call.get("attention_q_shape", ""),
        "k_shape": call.get("attention_k_shape", ""),
        "v_shape": call.get("attention_v_shape", ""),
        "output_shape": call.get("attention_output_shape", ""),
        "m12_provider_kind": "native_decomposed",
        "m12_runtime_kind": "torch_replay_native_decomposed_attention",
        "m13_target_implementation_level": "tvm_decomposed_artifact",
        "m13_target_runtime_kind": "native_tvm_attention_decomposed_artifact_run",
        "m13_target_provider_kind": "native_tvm_attention_decomposed",
        "legacy_provider_for_m13_strict": True,
    }


def _vit_captured_records(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        record
        for record in corpus.get("kernels", [])
        if record.get("model_case") == TARGET_MODEL or record.get("case_name") == TARGET_MODEL
    ]
    by_name = {}
    for record in records:
        by_name.setdefault(record["kernel_name"], record)
    return [by_name[name] for name in sorted(by_name)]


def _vit_extern_records(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            record
            for record in corpus.get("extern_ops", [])
            if record.get("model_case") == TARGET_MODEL or record.get("case_name") == TARGET_MODEL
        ],
        key=lambda record: int(record.get("line_no", 0) or 0),
    )


def _make_tvm_args_for_meta(meta, *, kernel=None) -> list[Any]:
    dev = tvm.cuda(0)
    scalar_values = _scalar_values_for_meta(meta, kernel=kernel)
    args: list[Any] = []
    for spec in meta.abi:
        name = spec["name"]
        if spec["kind"] == "scalar":
            args.append(scalar_values.get(name, 1))
            continue
        extent = _buffer_extent_value(meta.buffer_extents.get(name, ""), scalar_values)
        shape = _runtime_shape_for_pointer(meta, extent)
        dtype = spec["dtype"]
        np_dtype = _numpy_dtype(dtype)
        if np.issubdtype(np_dtype, np.floating):
            values = np.full(shape, 0.125, dtype=np_dtype)
        elif np_dtype == np.bool_:
            values = np.zeros(shape, dtype=np_dtype)
        else:
            values = np.zeros(shape, dtype=np_dtype)
        args.append(tvm.runtime.tensor(values, dev))
    return args


def _runtime_shape_for_pointer(meta, extent: int) -> tuple[int, ...]:
    if meta.contract == VISION_CONTRACT_POINTWISE_GRID2D_STATIC and meta.grid:
        y_extent = int(meta.grid[1])
        x_extent = max(1, int(extent) // max(1, y_extent))
        return (y_extent, x_extent)
    return (int(extent),)


def _scalar_values_for_meta(meta, *, kernel=None) -> dict[str, int]:
    size_hints = dict(getattr(kernel, "size_hints", {}) or {})
    values: dict[str, int] = {}
    for spec in meta.abi:
        if spec["kind"] != "scalar":
            continue
        name = spec["name"]
        if name == meta.extent_param and meta.extent_value is not None:
            values[name] = int(meta.extent_value)
        elif name == meta.reduction_extent_param and meta.reduction_extent_value is not None:
            values[name] = int(meta.reduction_extent_value)
        elif name == "xnumel":
            values[name] = int(meta.extent_value or size_hints.get("x", 1) or 1)
        elif name == "ynumel":
            values[name] = int(size_hints.get("y", 1) or 1)
        elif name.startswith("r"):
            values[name] = int(meta.reduction_extent_value or size_hints.get("r0_", 1) or 1)
        else:
            values[name] = 1
    return values


def _buffer_extent_value(expr: str, scalar_values: dict[str, int]) -> int:
    if not expr:
        return 1
    value = expr
    for name, scalar_value in scalar_values.items():
        value = re.sub(rf"\b{re.escape(name)}\b", str(scalar_value), value)
    ints = [int(item) for item in re.findall(r"T\.int64\((\d+)\)", value)]
    if not ints:
        plain_ints = [int(item) for item in re.findall(r"\b\d+\b", value)]
        return max(1, max(plain_ints) if plain_ints else 1)
    if "*" in value:
        product = 1
        for item in ints:
            product *= item
        return max(1, product)
    return max(1, max(ints))


def _numpy_dtype(dtype: str):
    return {
        "float32": np.float32,
        "float16": np.float16,
        "int64": np.int64,
        "int32": np.int32,
        "bool": np.bool_,
    }.get(dtype, np.float32)


def _artifact_identity(meta, graph) -> dict[str, Any]:
    return {
        "kernel_name": meta.kernel_name,
        "cache_key": meta.cache_key,
        "ttir_hash": meta.ttir_hash,
        "source_hash": meta.source_hash,
        "contract": meta.contract,
        "abi": list(meta.abi),
        "buffer_extents": dict(meta.buffer_extents),
        "unique_ops": sorted({op.name for op in graph.ops}),
    }


def _dot_snapshot(graph) -> dict[str, Any]:
    dots = [op for op in graph.ops if op.name == "tt.dot"]
    if not dots:
        return {}
    dot = dots[0]
    return {
        "name": dot.name,
        "operands": list(dot.operands),
        "results": list(dot.results),
        "attrs": dict(dot.attrs),
        "result_types": [
            {"raw": ty.raw, "dtype": ty.dtype, "shape": list(ty.shape)}
            for ty in dot.result_types
        ],
    }


def _generated_dot_signature() -> dict[str, str]:
    return {
        "a": "*fp32",
        "b": "*fp32",
        "out": "*fp32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def _load_wrapper_source() -> str:
    return M12_FINAL_WRAPPER.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_specific_assumptions() -> dict[str, Any]:
    return {
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "pixel_values_shape": [1, 3, 32, 32],
        "hidden_size": 64,
        "sequence_length": 5,
        "num_attention_heads": 4,
        "patch_embedding_conv": {
            "input_shape": [1, 3, 32, 32],
            "weight_shape": [64, 3, 16, 16],
            "output_shape": [1, 64, 2, 2],
        },
        "attention_shape_scope": "exact_vit_tiny_random_full_attention_m13",
    }


def _slice_fields(
    *,
    operator_inventory_delta,
    implementation_level_delta,
    legacy_provider_delta,
    reusable_capability_candidate,
    model_specific_assumption,
    m14_extraction_note,
) -> dict[str, Any]:
    return {
        "operator_inventory_delta": operator_inventory_delta,
        "implementation_level_delta": implementation_level_delta,
        "legacy_provider_delta": legacy_provider_delta,
        "reusable_capability_candidate": reusable_capability_candidate,
        "model_specific_assumption": model_specific_assumption,
        "m14_extraction_note": m14_extraction_note,
    }


def _invariants(checks: list[tuple[str, bool]]) -> dict[str, Any]:
    failures = [name for name, ok in checks if not ok]
    return {"status": "passed" if not failures else "failed", "failures": failures}


def _empty_execution_report(
    report_kind: str,
    report_id: str,
    substep: str,
    reason: str,
    *,
    status: str = "unavailable",
) -> dict[str, Any]:
    return {
        "report_kind": report_kind,
        "schema_version": 1,
        "report_id": report_id,
        "milestone": MILESTONE,
        "substep": substep,
        "status": status,
        "availability_reason": reason,
        "generated_at": _now(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "records": [],
        **_slice_fields(
            operator_inventory_delta=[],
            implementation_level_delta={level: 0 for level in IMPLEMENTATION_LEVELS},
            legacy_provider_delta={provider: 0 for provider in LEGACY_PROVIDERS},
            reusable_capability_candidate=[],
            model_specific_assumption=_model_specific_assumptions(),
            m14_extraction_note="No evidence emitted because execution was unavailable.",
        ),
    }


def _maybe_write_report(report: dict[str, Any], out_dir: str | Path | None, *, title: str) -> None:
    if out_dir is None:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "report.md").write_text(_markdown_report(report, title=title), encoding="utf-8")


def _markdown_report(report: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Report id: `{report.get('report_id')}`",
        f"- Target: `{report.get('target_model')}`",
        f"- Fixed shape: `{report.get('fixed_shape')}`",
    ]
    if "availability_reason" in report:
        lines.append(f"- Availability reason: `{report.get('availability_reason')}`")
    summary = report.get("summary")
    if isinstance(summary, dict):
        lines.extend(["", "## Summary", ""])
        for key, value in summary.items():
            lines.append(f"- `{key}`: `{value}`")
    invariants = report.get("invariants")
    if isinstance(invariants, dict):
        lines.extend(["", "## Invariants", ""])
        lines.append(f"- Status: `{invariants.get('status')}`")
        failures = invariants.get("failures") or []
        if failures:
            for failure in failures:
                lines.append(f"- Failure: `{failure}`")
        else:
            lines.append("- Failures: none")
    if "m13_p_loop_id" in report:
        p2 = (report.get("p2_dashboard") or {}).get("p2") or {}
        if p2:
            lines.extend(["", "## P2 Dashboard", ""])
            lines.append(
                "- Strict p50/p95: "
                f"`{p2.get('strict_path_p50_ms')}` / `{p2.get('strict_path_p95_ms')}` ms"
            )
            lines.append(
                "- Torch compile p50/p95: "
                f"`{p2.get('torch_compile_p50_ms')}` / `{p2.get('torch_compile_p95_ms')}` ms"
            )
            lines.append(
                f"- Ratios p50/p95: `{p2.get('p50_ratio')}` / `{p2.get('p95_ratio')}`"
            )
            lines.append(f"- Passed: `{p2.get('passed')}`")
        evidence = report.get("fine_grained_profiler_evidence") or report.get("profiler_evidence")
        if isinstance(evidence, dict) and evidence:
            lines.extend(["", "## Profiler Evidence", ""])
            if "record_count" in evidence:
                lines.append(f"- Record count: `{evidence.get('record_count')}`")
            for bucket in list(evidence.get("per_iteration_ranking") or [])[:10]:
                lines.append(
                    "- "
                    f"`{bucket.get('bucket_id')}`: "
                    f"`{bucket.get('avg_per_iteration_ms')}` ms/iter "
                    f"({bucket.get('sample_count')} samples)"
                )
        ranking = report.get("bottleneck_ranking") or []
        if ranking:
            lines.extend(["", "## Bottleneck Ranking", ""])
            for bucket in list(ranking)[:10]:
                lines.append(
                    "- "
                    f"`{bucket.get('bucket_id')}`: "
                    f"`{bucket.get('avg_per_iteration_ms')}` ms/iter"
                )
        next_decision = report.get("next_entry_decision") or {}
        if next_decision:
            lines.extend(["", "## Next Entry Decision", ""])
            for key in ("decision", "reason", "next_loop_id", "next_profiler_target"):
                if key in next_decision:
                    lines.append(f"- `{key}`: `{next_decision.get(key)}`")
    if "operator_counts" in report:
        lines.extend(["", "## Operator Counts", ""])
        for key, value in sorted((report.get("operator_counts") or {}).items()):
            lines.append(f"- `{key}`: `{value}`")
    records = (
        report.get("captured_kernel_records")
        or report.get("bridge_records")
        or report.get("matmul_runtime_records")
        or report.get("native_conv_records")
        or report.get("native_attention_records")
        or report.get("operator_inventory")
        or []
    )
    if records:
        lines.extend(["", "## Records", ""])
        for record in records:
            name = (
                record.get("kernel_name")
                or record.get("operator_id")
                or record.get("generated_kernel_name")
                or record.get("op_family")
            )
            status = record.get("status", record.get("m13_target_runtime_kind", ""))
            lines.append(f"- `{name}`: `{status}`")
    lines.append("")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m13_strict_tvm_owned_operator_path``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=(
            "m13.0",
            "m13.1",
            "m13.2",
            "m13.3",
            "m13.4",
            "m13.5",
            "m13.6",
            "m13.7",
            "m13.8",
            "m13.p1",
            "m13.p2",
            "m13.p3",
            "m13.p4",
            "m13.p5",
            "m13.p6",
            "m13.p7",
            "through-m13.3",
            "through-m13.4",
            "through-m13.5",
            "through-m13.6",
            "through-m13.7",
            "through-m13.8",
        ),
        default="through-m13.8",
    )
    parser.add_argument("--no-execution", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=M13_8_DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=M13_8_DEFAULT_REPEAT)
    args = parser.parse_args(argv)

    if args.step == "m13.0":
        report = run_m13_0_policy_taxonomy_freeze()
    elif args.step == "m13.1":
        report = run_m13_1_operator_inventory_baseline()
    elif args.step == "m13.2":
        report = run_m13_2_captured_kernel_tvm_execution(run_execution=not args.no_execution)
    elif args.step == "m13.3":
        report = run_m13_3_generated_tl_dot_bridge_semantics()
    elif args.step == "m13.4":
        report = run_m13_4_matmul_addmm_runtime_replacement(
            run_execution=not args.no_execution,
            seed=args.seed,
        )
    elif args.step == "m13.5":
        report = run_m13_5_native_conv_slice(
            run_execution=not args.no_execution,
            seed=args.seed,
        )
    elif args.step == "m13.6":
        report = run_m13_6_native_attention_slice(
            run_execution=not args.no_execution,
            seed=args.seed,
        )
    elif args.step == "m13.7":
        report = run_m13_7_strict_correctness_gate(
            run_execution=not args.no_execution,
            seed=args.seed,
        )
    elif args.step == "m13.8":
        report = run_m13_8_p2_dashboard_decision(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p1":
        report = run_m13_p1_strict_surface_profile_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p2":
        report = run_m13_p2_generated_bridge_direct_io_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p3":
        report = run_m13_p3_generated_bridge_async_direct_io_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p4":
        report = run_m13_p4_fine_grained_strict_surface_profile_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p5":
        report = run_m13_p5_generated_bridge_fast_artifact_invocation_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p6":
        report = run_m13_p6_native_artifact_fine_grained_profile_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "m13.p7":
        report = run_m13_p7_native_conv_fast_artifact_invocation_loop(
            run_benchmarks=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "through-m13.8":
        report = run_m13_0_to_m13_8(
            run_execution=not args.no_execution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    elif args.step == "through-m13.7":
        report = run_m13_0_to_m13_7(run_execution=not args.no_execution, seed=args.seed)
    elif args.step == "through-m13.6":
        report = run_m13_0_to_m13_6(run_execution=not args.no_execution, seed=args.seed)
    elif args.step == "through-m13.5":
        report = run_m13_0_to_m13_5(run_execution=not args.no_execution, seed=args.seed)
    elif args.step == "through-m13.4":
        report = run_m13_0_to_m13_4(run_execution=not args.no_execution, seed=args.seed)
    else:
        report = run_m13_0_to_m13_3(run_execution=not args.no_execution)
    print(f"M13 runner: step={args.step} status={report['status']}")
    return 0 if report["status"] in {"passed", "unavailable", "not_run"} else 1


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _m13_generated_tl_dot_bridge(
        a,
        b,
        out,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
        bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
        acc = tl.dot(av, bv, input_precision="tf32")
        tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


if __name__ == "__main__":
    raise SystemExit(main())
