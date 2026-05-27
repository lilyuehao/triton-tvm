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
"""M12.16 backend-general P2 gap analysis report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tvm

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import EXTERN_GEMM_PROVIDER_NATIVE_TVM, M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
from .m1215_vit_p2_reentry import P2_ELIGIBLE_MODE
from .m124_vit_e2e_dashboard import P2_THRESHOLD
from .m129_vit_provider_runtime_optimization import BACKEND_PASS_MODE, FUSED_QKV_MODE
from .m123_vit_e2e_runner import TARGET_MODEL
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_16_backend_general_p2_gap_analysis"
REPORT_ID = "m12_16_backend_general_p2_gap_analysis_v1"
MILESTONE = "M12.16"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_16_backend_general_p2_gap_analysis"
)
DEFAULT_M127_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_7_native_matmul_provider_cost/report.json"
)
DEFAULT_M129_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_9_provider_runtime_optimization/report.json"
)
DEFAULT_M1210_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_10_route_cleanup/report.json"
)
DEFAULT_M1213_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_13_runtime_overhead_general_measurement/report.json"
)
DEFAULT_M1214_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_14_real_tt_dot_schedule_handoff/report.json"
)
DEFAULT_M1215_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_15_fixed_shape_vit_p2_reentry/report.json"
)
NEXT_ACTION_AFTER_M1216 = "m12_p_profile_optimization_loop"
EXPECTED_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}
REQUIRED_ANALYSIS_BUCKETS = (
    "native_tvm_matmul_provider",
    "native_tvm_matmul_kernel",
    "native_tvm_matmul_runtime_glue",
    "remaining_e2e_residual",
    "torch_inductor_captured_harness",
    "native_decomposed_attention",
    "device_torch_cuda_conv",
    "reusable_runtime_overhead_microbench",
)


def run_backend_general_p2_gap_analysis(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    seed: int = 0,
    m127_report: str | Path = DEFAULT_M127_REPORT,
    m129_report: str | Path = DEFAULT_M129_REPORT,
    m1210_report: str | Path = DEFAULT_M1210_REPORT,
    m1213_report: str | Path = DEFAULT_M1213_REPORT,
    m1214_report: str | Path = DEFAULT_M1214_REPORT,
    m1215_report: str | Path = DEFAULT_M1215_REPORT,
) -> dict[str, Any]:
    """Build M12.16 from already generated source evidence."""

    try:
        source_m127 = _read_json_report(m127_report)
        source_m129 = _read_json_report(m129_report)
        source_m1210 = _read_json_report(m1210_report)
        source_m1213 = _read_json_report(m1213_report)
        source_m1214 = _read_json_report(m1214_report)
        source_m1215 = _read_json_report(m1215_report)
    except FileNotFoundError as err:
        report = _empty_report(
            status="unavailable",
            availability_reason=f"source_report_missing:{err.filename}",
            seed=seed,
            m127_report=m127_report,
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1213_report=m1213_report,
            m1214_report=m1214_report,
            m1215_report=m1215_report,
        )
        _maybe_write_report(report, out_dir)
        return report
    except json.JSONDecodeError as err:
        report = _empty_report(
            status="failed",
            availability_reason=f"source_report_invalid_json:{err}",
            seed=seed,
            m127_report=m127_report,
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1213_report=m1213_report,
            m1214_report=m1214_report,
            m1215_report=m1215_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    return build_backend_general_p2_gap_analysis_report(
        seed=seed,
        m127_report=source_m127,
        m127_report_path=m127_report,
        m129_report=source_m129,
        m129_report_path=m129_report,
        m1210_report=source_m1210,
        m1210_report_path=m1210_report,
        m1213_report=source_m1213,
        m1213_report_path=m1213_report,
        m1214_report=source_m1214,
        m1214_report_path=m1214_report,
        m1215_report=source_m1215,
        m1215_report_path=m1215_report,
        out_dir=out_dir,
    )


def build_backend_general_p2_gap_analysis_report(
    *,
    seed: int,
    m127_report: dict[str, Any],
    m127_report_path: str | Path,
    m129_report: dict[str, Any],
    m129_report_path: str | Path,
    m1210_report: dict[str, Any],
    m1210_report_path: str | Path,
    m1213_report: dict[str, Any],
    m1213_report_path: str | Path,
    m1214_report: dict[str, Any],
    m1214_report_path: str | Path,
    m1215_report: dict[str, Any],
    m1215_report_path: str | Path,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build the M12.16 gap-analysis report from source reports."""

    gap_budget = _p2_gap_budget(m1215_report)
    provider_mix = _p2_eligible_provider_mix(m1215_report)
    buckets = _analysis_buckets(
        m127_report=m127_report,
        m129_report=m129_report,
        m1213_report=m1213_report,
        m1215_report=m1215_report,
    )
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "source_reports": {
            "m12_7_report": str(m127_report_path),
            "m12_9_report": str(m129_report_path),
            "m12_10_report": str(m1210_report_path),
            "m12_13_report": str(m1213_report_path),
            "m12_14_report": str(m1214_report_path),
            "m12_15_report": str(m1215_report_path),
        },
        "source_report_status": _source_report_status(
            m127_report,
            m129_report,
            m1210_report,
            m1213_report,
            m1214_report,
            m1215_report,
        ),
        "analysis_scope": {
            "name": "backend_general_p2_gap_analysis",
            "kind": "source_backed_gap_attribution",
            "changes_runtime_or_schedule": False,
            "p2_eligible_mode": P2_ELIGIBLE_MODE,
            "p2_threshold": "<=2x torch_compile_inductor p50/p95",
            "requires_fixed_shape_vit_p2": True,
            "counts_m12_14_as_schedule_readiness": True,
            "counts_m12_14_as_e2e_performance_evidence": False,
            "counts_fused_qkv_for_p2": False,
        },
        "p2_gap_budget": gap_budget,
        "p2_eligible_provider_mix": provider_mix,
        "backend_pass": _backend_pass_context(m1215_report),
        "gap_buckets": buckets,
        "call_shape_analysis": _call_shape_analysis(m127_report, m1215_report),
        "attribution": _attribution(buckets, gap_budget),
        "candidate_actions": _candidate_actions(),
        "selected_next_slice": {
            "action": NEXT_ACTION_AFTER_M1216,
            "primary_bucket": "native_tvm_matmul_provider",
            "first_step": (
                "profile and reduce the backend-pass native TVM matmul provider cost "
                "under the exact M12.15 fixed-shape ViT path"
            ),
            "requires_new_p2_gate_before_m12_close": True,
            "model_specific": False,
            "counts_for_triton_tvm_backend_progress": True,
        },
        "forbidden_scope": _forbidden_scope(),
        "performance_ready_e2e": False,
        "performance_claim": False,
        "p2_passed": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
            "strict_full_tvm_native": False,
            "full_tvm_runnable_models": 0,
        },
        "next_default_action": {
            "action": NEXT_ACTION_AFTER_M1216,
            "reason": (
                "M12.16 attributes the remaining P2 gap to backend-general provider "
                "cost first; continue inside the repeatable M12.P profile/optimization loop."
            ),
        },
        "dependency_versions": {"tvm": str(tvm.__version__)},
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _source_report_status(
    m127_report: dict[str, Any],
    m129_report: dict[str, Any],
    m1210_report: dict[str, Any],
    m1213_report: dict[str, Any],
    m1214_report: dict[str, Any],
    m1215_report: dict[str, Any],
) -> dict[str, Any]:
    m1210_backend = ((m1210_report.get("route_claims") or {}).get(BACKEND_PASS_MODE)) or {}
    m1210_fused = ((m1210_report.get("route_claims") or {}).get(FUSED_QKV_MODE)) or {}
    m1215_policy = m1215_report.get("source_evidence_policy") or {}
    return {
        "m12_7_status": m127_report.get("status"),
        "m12_7_invariants": (m127_report.get("invariants") or {}).get("status"),
        "m12_7_optimization_validated": (m127_report.get("optimization") or {}).get("validated"),
        "m12_7_optimized_p2_estimate_passed": (m127_report.get("p2_gap_recovery") or {}).get(
            "optimized_p2_estimate_passed"
        ),
        "m12_9_status": m129_report.get("status"),
        "m12_9_invariants": (m129_report.get("invariants") or {}).get("status"),
        "m12_9_best_mode": (m129_report.get("best_mode") or {}).get("mode"),
        "m12_10_status": m1210_report.get("status"),
        "m12_10_invariants": (m1210_report.get("invariants") or {}).get("status"),
        "m12_10_backend_pass_counts_for_progress": m1210_backend.get(
            "counts_for_triton_tvm_backend_progress"
        ),
        "m12_10_backend_pass_model_specific": m1210_backend.get("model_specific_artifact"),
        "m12_10_fused_qkv_counts_for_backend_progress": m1210_fused.get(
            "counts_for_triton_tvm_backend_progress"
        ),
        "m12_10_fused_qkv_label": m1210_fused.get("route_diagnostic_label"),
        "m12_13_status": m1213_report.get("status"),
        "m12_13_invariants": (m1213_report.get("invariants") or {}).get("status"),
        "m12_13_runtime_overhead_measurement_complete": (
            m1213_report.get("summary") or {}
        ).get("runtime_overhead_measurement_complete"),
        "m12_14_status": m1214_report.get("status"),
        "m12_14_invariants": (m1214_report.get("invariants") or {}).get("status"),
        "m12_14_real_tt_dot_schedule_handoff_complete": m1214_report.get(
            "real_tt_dot_schedule_handoff_complete"
        ),
        "m12_14_p2_passed": m1214_report.get("p2_passed"),
        "m12_14_performance_ready_e2e": m1214_report.get("performance_ready_e2e"),
        "m12_15_status": m1215_report.get("status"),
        "m12_15_invariants": (m1215_report.get("invariants") or {}).get("status"),
        "m12_15_p2_passed": m1215_report.get("p2_passed"),
        "m12_15_performance_ready_e2e": m1215_report.get("performance_ready_e2e"),
        "m12_15_next_action": (m1215_report.get("next_default_action") or {}).get("action")
        or (m1215_report.get("summary") or {}).get("next_action"),
        "m12_15_p2_eligible_mode": m1215_report.get("p2_eligible_mode"),
        "m12_15_m12_14_counts_as_p2_evidence": m1215_policy.get(
            "m12_14_counts_as_p2_evidence"
        ),
        "m12_15_fused_qkv_excluded_from_p2": FUSED_QKV_MODE
        in (m1215_policy.get("p2_excluded_modes") or []),
    }


def _p2_gap_budget(m1215_report: dict[str, Any]) -> dict[str, Any]:
    latency = m1215_report.get("latency_ms") or {}
    compile_record = latency.get("torch_compile_inductor") or {}
    eligible_record = latency.get(P2_ELIGIBLE_MODE) or {}
    compile_p50 = _to_float_or_none(compile_record.get("p50_ms"))
    compile_p95 = _to_float_or_none(compile_record.get("p95_ms"))
    eligible_p50 = _to_float_or_none(eligible_record.get("p50_ms"))
    eligible_p95 = _to_float_or_none(eligible_record.get("p95_ms"))
    p50_ceiling = _mul_or_none(compile_p50, P2_THRESHOLD)
    p95_ceiling = _mul_or_none(compile_p95, P2_THRESHOLD)
    p50_excess = _positive_excess(eligible_p50, p50_ceiling)
    p95_excess = _positive_excess(eligible_p95, p95_ceiling)
    return {
        "threshold": "eligible Triton TVM p50/p95 <= 2x torch_compile_inductor",
        "eligible_mode": P2_ELIGIBLE_MODE,
        "torch_compile_p50_ms": compile_p50,
        "torch_compile_p95_ms": compile_p95,
        "eligible_p50_ms": eligible_p50,
        "eligible_p95_ms": eligible_p95,
        "p50_ceiling_ms": p50_ceiling,
        "p95_ceiling_ms": p95_ceiling,
        "p50_excess_ms": p50_excess,
        "p95_excess_ms": p95_excess,
        "ratio_p50": _safe_ratio(eligible_p50, compile_p50),
        "ratio_p95": _safe_ratio(eligible_p95, compile_p95),
        "p2_passed": bool((m1215_report.get("p2") or {}).get("passed")),
    }


def _p2_eligible_provider_mix(m1215_report: dict[str, Any]) -> dict[str, Any]:
    return (m1215_report.get("provider_mix") or {}).get(P2_ELIGIBLE_MODE) or {}


def _backend_pass_context(m1215_report: dict[str, Any]) -> dict[str, Any]:
    backend_pass = m1215_report.get("backend_pass") or {}
    return {
        "pass_name": backend_pass.get("pass_name"),
        "schedule_id": backend_pass.get("schedule_id"),
        "transformed_call_count": backend_pass.get("transformed_call_count"),
        "expected_call_count": backend_pass.get("expected_call_count"),
        "p2_eligible": backend_pass.get("p2_eligible"),
        "model_specific_artifact": backend_pass.get("model_specific_artifact"),
        "counts_for_triton_tvm_backend_progress": backend_pass.get(
            "counts_for_triton_tvm_backend_progress"
        ),
    }


def _analysis_buckets(
    *,
    m127_report: dict[str, Any],
    m129_report: dict[str, Any],
    m1213_report: dict[str, Any],
    m1215_report: dict[str, Any],
) -> list[dict[str, Any]]:
    profile = ((m129_report.get("provider_overhead_profile") or {}).get(BACKEND_PASS_MODE)) or {}
    aggregate = profile.get("aggregate_sum_ms") or {}
    m127_no_sync = (
        (m127_report.get("native_matmul_profile") or {})
        .get("policies", {})
        .get("no_per_call_sync", {})
        .get("aggregate_categories", {})
    )
    provider_total_single = _to_float_or_none(aggregate.get("provider_total_wall_ms"))
    kernel_single = _to_float_or_none(aggregate.get("kernel_event_ms"))
    runtime_glue_single = _delta(provider_total_single, kernel_single)
    eligible_p50 = _to_float_or_none(
        ((m1215_report.get("latency_ms") or {}).get(P2_ELIGIBLE_MODE) or {}).get("p50_ms")
    )
    residual_p50 = _positive_delta(eligible_p50, provider_total_single)
    measurements = m1213_report.get("measurements") or {}
    m1213_dominant = (m1213_report.get("attribution") or {}).get("dominant_bucket")
    provider_mix = _p2_eligible_provider_mix(m1215_report)
    counts = provider_mix.get("provider_counts") or {}
    return [
        {
            "bucket_id": "native_tvm_matmul_provider",
            "label": "P2-eligible native TVM matmul provider total",
            "priority": "primary",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "count": _int_or_zero(counts.get(EXTERN_GEMM_PROVIDER_NATIVE_TVM)),
            "evidence_kind": "m12_9_backend_pass_single_profile_plus_m12_7_historical_p50",
            "backend_pass_single_profile_ms": provider_total_single,
            "historical_no_sync_p50_ms": _record_p50(m127_no_sync, "provider_total_wall_ms"),
            "historical_no_sync_p95_ms": _record_p95(m127_no_sync, "provider_total_wall_ms"),
            "reason": (
                "Largest measured backend-general provider bucket; covers all seven "
                "P2-eligible wrapper matmul/addmm calls."
            ),
        },
        {
            "bucket_id": "native_tvm_matmul_kernel",
            "label": "Native TVM matmul kernel event time",
            "priority": "primary_subbucket",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "count": _int_or_zero(counts.get(EXTERN_GEMM_PROVIDER_NATIVE_TVM)),
            "evidence_kind": "m12_9_backend_pass_single_profile_plus_m12_7_historical_p50",
            "backend_pass_single_profile_ms": kernel_single,
            "historical_no_sync_p50_ms": _record_p50(m127_no_sync, "kernel_event_ms"),
            "historical_no_sync_p95_ms": _record_p95(m127_no_sync, "kernel_event_ms"),
            "reason": "Kernel time is the dominant measured subbucket inside matmul provider cost.",
        },
        {
            "bucket_id": "native_tvm_matmul_runtime_glue",
            "label": "Native TVM matmul provider glue outside kernel event",
            "priority": "primary_subbucket",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "count": _int_or_zero(counts.get(EXTERN_GEMM_PROVIDER_NATIVE_TVM)),
            "evidence_kind": "derived_from_provider_total_minus_kernel_event",
            "backend_pass_single_profile_ms": runtime_glue_single,
            "historical_no_sync_p50_ms": _delta(
                _record_p50(m127_no_sync, "provider_total_wall_ms"),
                _record_p50(m127_no_sync, "kernel_event_ms"),
            ),
            "reason": "Separates dispatch/DLPack/storage overhead from raw kernel time.",
        },
        {
            "bucket_id": "remaining_e2e_residual",
            "label": "Remaining E2E time outside measured backend-pass matmul provider",
            "priority": "instrumentation",
            "backend_general": True,
            "model_specific": False,
            "actionable": False,
            "count": (
                _int_or_zero(counts.get("torch_inductor_triton_captured_harness"))
                + _int_or_zero(counts.get(ATTENTION_PROVIDER_NATIVE_DECOMPOSED))
                + _int_or_zero(counts.get(VISION_PROVIDER_DEVICE_TORCH_CUDA))
            ),
            "evidence_kind": "derived_e2e_minus_backend_pass_native_matmul_provider",
            "estimated_p50_ms": residual_p50,
            "reason": (
                "Residual includes captured harness, attention, conv, and wrapper/runtime "
                "overhead; keep it explicit so M12.16 does not over-attribute the whole "
                "E2E miss to matmul."
            ),
        },
        {
            "bucket_id": "torch_inductor_captured_harness",
            "label": "Captured Inductor Triton harness kernels",
            "priority": "instrumentation",
            "backend_general": False,
            "model_specific": False,
            "actionable": False,
            "count": _int_or_zero(counts.get("torch_inductor_triton_captured_harness")),
            "evidence_kind": "provider_mix_count_only",
            "estimated_ms": None,
            "reason": (
                "Still in the explicit provider mix but not yet attributed by M12.16; "
                "instrument only after matmul provider cost is resolved or disproved."
            ),
        },
        {
            "bucket_id": "native_decomposed_attention",
            "label": "Native decomposed attention provider",
            "priority": "watch",
            "backend_general": True,
            "model_specific": False,
            "actionable": False,
            "count": _int_or_zero(counts.get(ATTENTION_PROVIDER_NATIVE_DECOMPOSED)),
            "evidence_kind": "provider_mix_count_only",
            "estimated_ms": None,
            "reason": "Correctness-only provider remains in path; quantify before optimizing.",
        },
        {
            "bucket_id": "device_torch_cuda_conv",
            "label": "Device Torch CUDA convolution provider",
            "priority": "watch",
            "backend_general": False,
            "model_specific": False,
            "actionable": False,
            "count": _int_or_zero(counts.get(VISION_PROVIDER_DEVICE_TORCH_CUDA)),
            "evidence_kind": "provider_mix_count_only",
            "estimated_ms": None,
            "reason": "Runtime-resolved device provider remains in path; quantify before optimizing.",
        },
        {
            "bucket_id": "reusable_runtime_overhead_microbench",
            "label": "Reusable runtime overhead microbench",
            "priority": "secondary",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "count": 1,
            "evidence_kind": "m12_13_standalone_microbench",
            "dominant_bucket": m1213_dominant,
            "dominant_p50_ms": _record_p50(measurements, str(m1213_dominant)),
            "artifact_run_p50_ms": _record_p50(measurements, "artifact_run_wall_ms"),
            "packed_func_lookup_p50_ms": _record_p50(measurements, "packed_func_lookup_ms"),
            "dlpack_conversion_p50_ms": _record_p50(measurements, "dlpack_conversion_ms"),
            "reason": "Reusable runtime overhead is real but smaller than the current matmul provider bucket.",
        },
    ]


def _call_shape_analysis(
    m127_report: dict[str, Any],
    m1215_report: dict[str, Any],
) -> list[dict[str, Any]]:
    backend_calls = (m1215_report.get("backend_pass") or {}).get("per_call") or []
    historical_calls = (
        (m127_report.get("native_matmul_profile") or {})
        .get("policies", {})
        .get("no_per_call_sync", {})
        .get("per_call", [])
    )
    historical_by_index = {int(call.get("index", -1)): call for call in historical_calls}
    rows = []
    for call in backend_calls:
        index = int(call.get("index", -1))
        historical = historical_by_index.get(index) or {}
        categories = historical.get("categories") or {}
        rows.append(
            {
                "index": index,
                "op_family": call.get("op_family"),
                "shape_mnk": call.get("shape_mnk"),
                "m12_15_schedule_id": call.get("schedule_id"),
                "historical_no_sync_schedule_id": historical.get("schedule_id"),
                "historical_provider_total_p50_ms": _record_p50(
                    categories, "provider_total_wall_ms"
                ),
                "historical_kernel_event_p50_ms": _record_p50(categories, "kernel_event_ms"),
                "historical_artifact_dispatch_p50_ms": _record_p50(
                    categories, "artifact_dispatch_wall_ms"
                ),
            }
        )
    return rows


def _attribution(buckets: list[dict[str, Any]], gap_budget: dict[str, Any]) -> dict[str, Any]:
    measured = []
    for bucket in buckets:
        value = _to_float_or_none(bucket.get("backend_pass_single_profile_ms"))
        if value is None:
            value = _to_float_or_none(bucket.get("historical_no_sync_p50_ms"))
        if value is None:
            value = _to_float_or_none(bucket.get("dominant_p50_ms"))
        if value is None:
            value = _to_float_or_none(bucket.get("estimated_p50_ms"))
        if value is not None:
            measured.append((value, bucket["bucket_id"]))
    dominant_value, dominant_bucket = max(measured) if measured else (None, None)
    p50_excess = _to_float_or_none(gap_budget.get("p50_excess_ms"))
    p95_excess = _to_float_or_none(gap_budget.get("p95_excess_ms"))
    return {
        "dominant_measured_bucket": dominant_bucket,
        "dominant_measured_ms": dominant_value,
        "remaining_p2_gap_p50_ms": p50_excess,
        "remaining_p2_gap_p95_ms": p95_excess,
        "primary_interpretation": (
            "M12.16 attributes the first backend-general recovery target to "
            "native_tvm_matmul provider cost; captured harness, attention, and conv "
            "need instrumentation but are not selected ahead of the measured provider bucket."
        ),
    }


def _candidate_actions() -> list[dict[str, Any]]:
    return [
        {
            "action": NEXT_ACTION_AFTER_M1216,
            "bucket": "native_tvm_matmul_provider",
            "priority": 1,
            "backend_general": True,
            "model_specific": False,
            "description": (
                "Enter the repeatable M12.P loop: profile the exact M12.15 backend-pass "
                "path per call, classify the bottleneck, optimize a backend-general path, "
                "then verify and rerun the unchanged P2 gate when evidence justifies it."
            ),
        },
        {
            "action": "instrument_remaining_provider_mix",
            "bucket": "torch_inductor_captured_harness",
            "priority": 2,
            "backend_general": True,
            "model_specific": False,
            "description": "Add timing attribution for captured harness, attention, and conv providers.",
        },
        {
            "action": "reuse_runtime_overhead_reduction_if_matmul_disproven",
            "bucket": "reusable_runtime_overhead_microbench",
            "priority": 3,
            "backend_general": True,
            "model_specific": False,
            "description": "Apply M12.13 runtime-overhead findings only if provider-cost recovery stalls.",
        },
    ]


def _forbidden_scope() -> dict[str, Any]:
    return {
        "no_p2_redefinition": True,
        "no_model_specific_vit_executor": True,
        "no_wrapper_line_or_buffer_replay": True,
        "no_fused_qkv_p2_credit": True,
        "no_m12_14_as_e2e_performance_evidence": True,
        "no_strict_full_native_claim": True,
        "no_m12_completion_claim": True,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("milestone") != MILESTONE:
        failures.append("m12_16_wrong_milestone")

    source = report.get("source_report_status") or {}
    expected_sources = (
        ("m12_7", "m12_16_m12_7"),
        ("m12_9", "m12_16_m12_9"),
        ("m12_10", "m12_16_m12_10"),
        ("m12_13", "m12_16_m12_13"),
        ("m12_14", "m12_16_m12_14"),
        ("m12_15", "m12_16_m12_15"),
    )
    for source_key, failure_prefix in expected_sources:
        if source.get(f"{source_key}_status") != "passed":
            failures.append(f"{failure_prefix}_not_passed")
        if source.get(f"{source_key}_invariants") != "passed":
            failures.append(f"{failure_prefix}_invariants_not_passed")

    if source.get("m12_7_optimization_validated") is not True:
        failures.append("m12_16_m12_7_optimization_not_validated")
    if source.get("m12_7_optimized_p2_estimate_passed") is not False:
        failures.append("m12_16_m12_7_not_p2_miss_context")
    if source.get("m12_10_backend_pass_counts_for_progress") is not True:
        failures.append("m12_16_backend_pass_not_backend_progress")
    if source.get("m12_10_backend_pass_model_specific") is not False:
        failures.append("m12_16_backend_pass_model_specific_in_route_cleanup")
    if source.get("m12_10_fused_qkv_counts_for_backend_progress") is not False:
        failures.append("m12_16_fused_qkv_counts_for_backend_progress")
    if source.get("m12_10_fused_qkv_label") != "model_specific_diagnostic_only":
        failures.append("m12_16_fused_qkv_not_diagnostic_only")
    if source.get("m12_13_runtime_overhead_measurement_complete") is not True:
        failures.append("m12_16_m12_13_runtime_measurement_incomplete")
    if source.get("m12_14_real_tt_dot_schedule_handoff_complete") is not True:
        failures.append("m12_16_m12_14_schedule_handoff_missing")
    if source.get("m12_14_p2_passed") is not False:
        failures.append("m12_16_m12_14_treated_as_p2")
    if source.get("m12_14_performance_ready_e2e") is not False:
        failures.append("m12_16_m12_14_treated_as_performance_ready")
    if source.get("m12_15_p2_passed") is not False:
        failures.append("m12_16_m12_15_not_p2_miss")
    if source.get("m12_15_performance_ready_e2e") is not False:
        failures.append("m12_16_m12_15_performance_ready_unexpected")
    if source.get("m12_15_next_action") != "m12_16_backend_general_p2_gap_analysis":
        failures.append("m12_16_m12_15_next_action_mismatch")
    if source.get("m12_15_p2_eligible_mode") != P2_ELIGIBLE_MODE:
        failures.append("m12_16_wrong_p2_eligible_mode")
    if source.get("m12_15_m12_14_counts_as_p2_evidence") is not False:
        failures.append("m12_16_m12_14_counts_as_p2_evidence")
    if source.get("m12_15_fused_qkv_excluded_from_p2") is not True:
        failures.append("m12_16_fused_qkv_not_excluded_from_p2")

    scope = report.get("analysis_scope") or {}
    if scope.get("changes_runtime_or_schedule") is not False:
        failures.append("m12_16_scope_changes_runtime_or_schedule")
    if scope.get("p2_eligible_mode") != P2_ELIGIBLE_MODE:
        failures.append("m12_16_scope_wrong_p2_mode")
    if scope.get("counts_m12_14_as_e2e_performance_evidence") is not False:
        failures.append("m12_16_scope_counts_m12_14_as_e2e")
    if scope.get("counts_fused_qkv_for_p2") is not False:
        failures.append("m12_16_scope_counts_fused_qkv")

    gap = report.get("p2_gap_budget") or {}
    if gap.get("eligible_mode") != P2_ELIGIBLE_MODE:
        failures.append("m12_16_gap_budget_wrong_eligible_mode")
    if gap.get("p2_passed") is not False:
        failures.append("m12_16_gap_budget_not_p2_miss")
    for key in ("p50_excess_ms", "p95_excess_ms", "eligible_p50_ms", "eligible_p95_ms"):
        if _to_float_or_none(gap.get(key)) is None:
            failures.append(f"m12_16_gap_budget_missing_{key}")
    if _to_float_or_none(gap.get("p50_excess_ms")) is not None and float(
        gap.get("p50_excess_ms")
    ) <= 0.0:
        failures.append("m12_16_gap_budget_p50_not_positive")
    if _to_float_or_none(gap.get("p95_excess_ms")) is not None and float(
        gap.get("p95_excess_ms")
    ) <= 0.0:
        failures.append("m12_16_gap_budget_p95_not_positive")
    compile_p50 = _to_float_or_none(gap.get("torch_compile_p50_ms"))
    compile_p95 = _to_float_or_none(gap.get("torch_compile_p95_ms"))
    eligible_p50 = _to_float_or_none(gap.get("eligible_p50_ms"))
    eligible_p95 = _to_float_or_none(gap.get("eligible_p95_ms"))
    p50_ceiling = _to_float_or_none(gap.get("p50_ceiling_ms"))
    p95_ceiling = _to_float_or_none(gap.get("p95_ceiling_ms"))
    if not _close_or_missing(p50_ceiling, _mul_or_none(compile_p50, P2_THRESHOLD)):
        failures.append("m12_16_gap_budget_p50_ceiling_mismatch")
    if not _close_or_missing(p95_ceiling, _mul_or_none(compile_p95, P2_THRESHOLD)):
        failures.append("m12_16_gap_budget_p95_ceiling_mismatch")
    if not _close_or_missing(gap.get("p50_excess_ms"), _positive_excess(eligible_p50, p50_ceiling)):
        failures.append("m12_16_gap_budget_p50_excess_mismatch")
    if not _close_or_missing(gap.get("p95_excess_ms"), _positive_excess(eligible_p95, p95_ceiling)):
        failures.append("m12_16_gap_budget_p95_excess_mismatch")
    if not _close_or_missing(gap.get("ratio_p50"), _safe_ratio(eligible_p50, compile_p50)):
        failures.append("m12_16_gap_budget_p50_ratio_mismatch")
    if not _close_or_missing(gap.get("ratio_p95"), _safe_ratio(eligible_p95, compile_p95)):
        failures.append("m12_16_gap_budget_p95_ratio_mismatch")

    provider_mix = report.get("p2_eligible_provider_mix") or {}
    if provider_mix.get("complete_provider_report") is not True:
        failures.append("m12_16_provider_mix_incomplete")
    if provider_mix.get("no_silent_fallback") is not True:
        failures.append("m12_16_provider_mix_silent_fallback")
    counts = {
        str(provider): int(count or 0)
        for provider, count in (provider_mix.get("provider_counts") or {}).items()
    }
    if counts != EXPECTED_PROVIDER_COUNTS:
        failures.append("m12_16_provider_counts_mismatch")

    backend_pass = report.get("backend_pass") or {}
    if backend_pass.get("schedule_id") != M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
        failures.append("m12_16_backend_pass_wrong_schedule")
    if backend_pass.get("transformed_call_count") != backend_pass.get("expected_call_count"):
        failures.append("m12_16_backend_pass_did_not_transform_all_calls")
    if backend_pass.get("p2_eligible") is not True:
        failures.append("m12_16_backend_pass_not_p2_eligible")
    if backend_pass.get("model_specific_artifact") is not False:
        failures.append("m12_16_backend_pass_model_specific")
    if backend_pass.get("counts_for_triton_tvm_backend_progress") is not True:
        failures.append("m12_16_backend_pass_not_backend_progress")

    gap_buckets = report.get("gap_buckets") or []
    bucket_by_id = {bucket.get("bucket_id"): bucket for bucket in gap_buckets}
    bucket_ids = set(bucket_by_id)
    for bucket in REQUIRED_ANALYSIS_BUCKETS:
        if bucket not in bucket_ids:
            failures.append(f"m12_16_missing_gap_bucket_{bucket}")
    primary = bucket_by_id.get("native_tvm_matmul_provider") or {}
    if primary.get("actionable") is not True or primary.get("model_specific") is not False:
        failures.append("m12_16_primary_bucket_not_backend_general_actionable")
    residual = bucket_by_id.get("remaining_e2e_residual") or {}
    if residual.get("actionable") is not False:
        failures.append("m12_16_residual_bucket_actionable")
    if residual.get("evidence_kind") != "derived_e2e_minus_backend_pass_native_matmul_provider":
        failures.append("m12_16_residual_bucket_evidence_mismatch")
    if _to_float_or_none(residual.get("estimated_p50_ms")) is None:
        failures.append("m12_16_residual_bucket_missing_estimate")
    count_only_expectations = {
        "torch_inductor_captured_harness": "provider_mix_count_only",
        "native_decomposed_attention": "provider_mix_count_only",
        "device_torch_cuda_conv": "provider_mix_count_only",
    }
    for bucket_id, evidence_kind in count_only_expectations.items():
        bucket = bucket_by_id.get(bucket_id) or {}
        if bucket.get("actionable") is not False:
            failures.append(f"m12_16_count_only_bucket_actionable_{bucket_id}")
        if bucket.get("evidence_kind") != evidence_kind:
            failures.append(f"m12_16_count_only_bucket_evidence_mismatch_{bucket_id}")
        if bucket.get("estimated_ms") is not None:
            failures.append(f"m12_16_count_only_bucket_has_estimate_{bucket_id}")
    primary_bucket = (report.get("selected_next_slice") or {}).get("primary_bucket")
    if primary_bucket != "native_tvm_matmul_provider":
        failures.append("m12_16_wrong_primary_bucket")
    selected = report.get("selected_next_slice") or {}
    if selected.get("model_specific") is not False:
        failures.append("m12_16_selected_slice_model_specific")
    if selected.get("counts_for_triton_tvm_backend_progress") is not True:
        failures.append("m12_16_selected_slice_not_backend_progress")

    forbidden = report.get("forbidden_scope") or {}
    for key in (
        "no_p2_redefinition",
        "no_model_specific_vit_executor",
        "no_wrapper_line_or_buffer_replay",
        "no_fused_qkv_p2_credit",
        "no_m12_14_as_e2e_performance_evidence",
        "no_strict_full_native_claim",
        "no_m12_completion_claim",
    ):
        if forbidden.get(key) is not True:
            failures.append(f"m12_16_forbidden_scope_missing_{key}")

    for key in (
        "performance_ready_e2e",
        "performance_claim",
        "p2_passed",
        "backend_general_complete",
        "strict_full_tvm_native",
    ):
        if report.get(key) is not False:
            failures.append(f"m12_16_forbidden_claim_{key}")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_16_full_tvm_runnable_claim_present")
    completion = report.get("completion_gate") or {}
    if completion.get("m12_complete") is not False:
        failures.append("m12_16_m12_complete_claim_present")
    if completion.get("requires_fixed_shape_vit_p2") is not True:
        failures.append("m12_16_p2_gate_not_preserved")
    if completion.get("strict_full_tvm_native") is not False:
        failures.append("m12_16_completion_strict_native_claim_present")
    if int(completion.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_16_completion_full_runnable_claim_present")
    if (report.get("next_default_action") or {}).get("action") != NEXT_ACTION_AFTER_M1216:
        failures.append("m12_16_next_action_mismatch")

    return {"status": "passed" if not failures else "failed", "invariant_failures": failures}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    gap = report.get("p2_gap_budget") or {}
    attribution = report.get("attribution") or {}
    return {
        "m12_16_complete": report.get("status") == "passed",
        "m12_complete": (report.get("completion_gate") or {}).get("m12_complete"),
        "p2_passed": False,
        "performance_ready_e2e": False,
        "p2_eligible_mode": P2_ELIGIBLE_MODE,
        "eligible_mode_p50_ms": gap.get("eligible_p50_ms"),
        "eligible_mode_p95_ms": gap.get("eligible_p95_ms"),
        "p50_excess_ms": gap.get("p50_excess_ms"),
        "p95_excess_ms": gap.get("p95_excess_ms"),
        "dominant_measured_bucket": attribution.get("dominant_measured_bucket"),
        "selected_next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(
    *,
    status: str,
    availability_reason: str,
    seed: int,
    m127_report: str | Path,
    m129_report: str | Path,
    m1210_report: str | Path,
    m1213_report: str | Path,
    m1214_report: str | Path,
    m1215_report: str | Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "source_reports": {
            "m12_7_report": str(m127_report),
            "m12_9_report": str(m129_report),
            "m12_10_report": str(m1210_report),
            "m12_13_report": str(m1213_report),
            "m12_14_report": str(m1214_report),
            "m12_15_report": str(m1215_report),
        },
        "source_report_status": {},
        "analysis_scope": {
            "name": "backend_general_p2_gap_analysis",
            "kind": "source_backed_gap_attribution",
            "changes_runtime_or_schedule": False,
            "p2_eligible_mode": P2_ELIGIBLE_MODE,
        },
        "p2_gap_budget": {},
        "p2_eligible_provider_mix": {},
        "backend_pass": {},
        "gap_buckets": [],
        "call_shape_analysis": [],
        "attribution": {"dominant_measured_bucket": None},
        "candidate_actions": [],
        "selected_next_slice": {},
        "forbidden_scope": _forbidden_scope(),
        "performance_ready_e2e": False,
        "performance_claim": False,
        "p2_passed": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
            "strict_full_tvm_native": False,
            "full_tvm_runnable_models": 0,
        },
        "next_default_action": {"action": "fix_m12_16_source_reports"},
        "dependency_versions": {"tvm": str(tvm.__version__)},
        "invariants": {"status": "not_run", "invariant_failures": []},
        "summary": {
            "m12_16_complete": False,
            "m12_complete": False,
            "p2_passed": False,
            "performance_ready_e2e": False,
            "selected_next_action": "fix_m12_16_source_reports",
        },
    }
    return report


def _read_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maybe_write_report(report: dict[str, Any], out_dir: str | Path | None) -> None:
    if out_dir is None:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "report.md").write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    gap = report.get("p2_gap_budget") or {}
    summary = report.get("summary") or {}
    lines = [
        "# M12.16 Backend-General P2 Gap Analysis",
        "",
        f"- Status: `{report.get('status')}`",
        f"- P2 eligible mode: `{(report.get('analysis_scope') or {}).get('p2_eligible_mode')}`",
        f"- P2 passed: `{report.get('p2_passed')}`",
        f"- Performance ready E2E: `{report.get('performance_ready_e2e')}`",
        f"- Dominant measured bucket: `{summary.get('dominant_measured_bucket')}`",
        f"- Next action: `{summary.get('selected_next_action')}`",
        "",
        "## P2 Gap",
        "",
        "| Metric | ms / ratio |",
        "| --- | ---: |",
        f"| Torch compile p50 | {_fmt(gap.get('torch_compile_p50_ms'))} |",
        f"| Torch compile p95 | {_fmt(gap.get('torch_compile_p95_ms'))} |",
        f"| Eligible p50 | {_fmt(gap.get('eligible_p50_ms'))} |",
        f"| Eligible p95 | {_fmt(gap.get('eligible_p95_ms'))} |",
        f"| P2 ceiling p50 | {_fmt(gap.get('p50_ceiling_ms'))} |",
        f"| P2 ceiling p95 | {_fmt(gap.get('p95_ceiling_ms'))} |",
        f"| Excess p50 | {_fmt(gap.get('p50_excess_ms'))} |",
        f"| Excess p95 | {_fmt(gap.get('p95_excess_ms'))} |",
        f"| Ratio p50 | {_fmt(gap.get('ratio_p50'))} |",
        f"| Ratio p95 | {_fmt(gap.get('ratio_p95'))} |",
        "",
        "## Gap Buckets",
        "",
        "| Bucket | Priority | Count | Evidence | Estimate ms | Actionable |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for bucket in report.get("gap_buckets") or []:
        estimate = (
            bucket.get("backend_pass_single_profile_ms")
            if bucket.get("backend_pass_single_profile_ms") is not None
            else bucket.get("historical_no_sync_p50_ms")
        )
        if estimate is None:
            estimate = bucket.get("dominant_p50_ms")
        if estimate is None:
            estimate = bucket.get("estimated_p50_ms")
        lines.append(
            f"| `{bucket.get('bucket_id')}` | `{bucket.get('priority')}` | "
            f"{bucket.get('count')} | `{bucket.get('evidence_kind')}` | "
            f"{_fmt(estimate)} | `{bucket.get('actionable')}` |"
        )
    lines.extend(
        [
            "",
            "## Selected Next Slice",
            "",
            f"- Action: `{(report.get('selected_next_slice') or {}).get('action')}`",
            f"- Primary bucket: `{(report.get('selected_next_slice') or {}).get('primary_bucket')}`",
            f"- First step: {(report.get('selected_next_slice') or {}).get('first_step')}",
            "",
            "## Forbidden Scope",
            "",
        ]
    )
    for key, value in (report.get("forbidden_scope") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Invariants", ""])
    invariants = report.get("invariants") or {}
    lines.append(f"- Status: `{invariants.get('status')}`")
    for failure in invariants.get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _record_p50(records: dict[str, Any], key: str) -> float | None:
    return _to_float_or_none((records.get(key) or {}).get("p50_ms"))


def _record_p95(records: dict[str, Any], key: str) -> float | None:
    return _to_float_or_none((records.get(key) or {}).get("p95_ms"))


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mul_or_none(value: float | None, factor: float) -> float | None:
    if value is None:
        return None
    return value * factor


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_f = _to_float_or_none(numerator)
    denominator_f = _to_float_or_none(denominator)
    if numerator_f is None or denominator_f is None or denominator_f <= 0.0:
        return None
    return numerator_f / denominator_f


def _positive_excess(value: float | None, ceiling: float | None) -> float | None:
    if value is None or ceiling is None:
        return None
    return max(0.0, value - ceiling)


def _positive_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return max(0.0, before - after)


def _close_or_missing(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    left_f = _to_float_or_none(left)
    right_f = _to_float_or_none(right)
    if left_f is None or right_f is None:
        return False
    return abs(left_f - right_f) <= atol


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return before - after


def _fmt(value: Any) -> str:
    converted = _to_float_or_none(value)
    if converted is None:
        return "n/a"
    return f"{converted:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m1216_backend_general_p2_gap_analysis``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m127-report", type=Path, default=DEFAULT_M127_REPORT)
    parser.add_argument("--m129-report", type=Path, default=DEFAULT_M129_REPORT)
    parser.add_argument("--m1210-report", type=Path, default=DEFAULT_M1210_REPORT)
    parser.add_argument("--m1213-report", type=Path, default=DEFAULT_M1213_REPORT)
    parser.add_argument("--m1214-report", type=Path, default=DEFAULT_M1214_REPORT)
    parser.add_argument("--m1215-report", type=Path, default=DEFAULT_M1215_REPORT)
    args = parser.parse_args(argv)

    report = run_backend_general_p2_gap_analysis(
        out_dir=args.out_dir,
        seed=args.seed,
        m127_report=args.m127_report,
        m129_report=args.m129_report,
        m1210_report=args.m1210_report,
        m1213_report=args.m1213_report,
        m1214_report=args.m1214_report,
        m1215_report=args.m1215_report,
    )
    summary = report.get("summary") or {}
    print(
        f"M12.16 backend-general P2 gap analysis: status={report['status']} "
        f"dominant_bucket={summary.get('dominant_measured_bucket')} "
        f"p50_excess_ms={summary.get('p50_excess_ms')} "
        f"p95_excess_ms={summary.get('p95_excess_ms')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
