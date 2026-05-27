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
"""M12.15 fixed-shape ViT P2 re-entry gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tvm

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import EXTERN_GEMM_PROVIDER_NATIVE_TVM, M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
from .m123_vit_e2e_runner import (
    NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
    TARGET_MODEL,
    _build_native_matmul_calls,
    _correctness_report,
    _dependency_versions,
    _provider_mix_report,
    _require_torch,
    execute_vit_fixed_shape_provider_path,
    prepare_vit_fixed_shape_e2e,
)
from .m124_vit_e2e_dashboard import DEFAULT_REPEAT, DEFAULT_WARMUP, P1_THRESHOLD, P2_THRESHOLD
from .m129_vit_provider_runtime_optimization import BACKEND_PASS_MODE, CURRENT_MODE, FUSED_QKV_MODE
from .passes import M129OptimizeNativeWrapperMatmul
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_15_vit_p2_reentry"
REPORT_ID = "m12_15_fixed_shape_vit_p2_reentry_v1"
MILESTONE = "M12.15"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_15_fixed_shape_vit_p2_reentry"
)
DEFAULT_M129_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_9_provider_runtime_optimization/report.json"
)
DEFAULT_M1210_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_10_route_cleanup/report.json"
)
DEFAULT_M1214_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_14_real_tt_dot_schedule_handoff/report.json"
)
BASELINES = ("torch_eager_cuda", "torch_compile_inductor")
MEASURED_PROVIDER_MODES = (CURRENT_MODE, BACKEND_PASS_MODE)
P2_ELIGIBLE_MODE = BACKEND_PASS_MODE
NEXT_ACTION_AFTER_P2_MISS = "m12_16_backend_general_p2_gap_analysis"
NEXT_ACTION_AFTER_P2_PASS = "close_m12_p2_passed"
EXPECTED_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}


def run_vit_p2_reentry(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
    m129_report: str | Path = DEFAULT_M129_REPORT,
    m1210_report: str | Path = DEFAULT_M1210_REPORT,
    m1214_report: str | Path = DEFAULT_M1214_REPORT,
) -> dict[str, Any]:
    """Run the M12.15 fixed-shape ViT P2 re-entry gate."""

    try:
        source_m129 = _read_json_report(m129_report)
        source_m1210 = _read_json_report(m1210_report)
        source_m1214 = _read_json_report(m1214_report)
    except FileNotFoundError as err:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason=f"source_report_missing:{err.filename}",
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1214_report=m1214_report,
        )
        _maybe_write_report(report, out_dir)
        return report
    except json.JSONDecodeError as err:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"source_report_invalid_json:{err}",
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1214_report=m1214_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    if not run_benchmarks:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1214_report=m1214_report,
            source_m129=source_m129,
            source_m1210=source_m1210,
            source_m1214=source_m1214,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1214_report=m1214_report,
            source_m129=source_m129,
            source_m1210=source_m1210,
            source_m1214=source_m1214,
        )
        _maybe_write_report(report, out_dir)
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1214_report=m1214_report,
            source_m129=source_m129,
            source_m1210=source_m1210,
            source_m1214=source_m1214,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        backend_pass = M129OptimizeNativeWrapperMatmul()
        optimized_calls = _build_native_matmul_calls(
            prepared.wrapper_source,
            native_matmul_passes=[backend_pass],
        )
        optimized_prepared = replace(prepared, matmul_calls=optimized_calls)

        current_actual, current_state = execute_vit_fixed_shape_provider_path(
            prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )
        backend_actual, backend_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )
        correctness = {
            CURRENT_MODE: _correctness_report(current_actual, prepared.expected),
            BACKEND_PASS_MODE: _correctness_report(backend_actual, prepared.expected),
        }
        provider_mix = {
            CURRENT_MODE: _provider_mix_report(
                wrapper_source=prepared.wrapper_source,
                matmul_calls=prepared.matmul_calls,
                patch_state=current_state,
            ),
            BACKEND_PASS_MODE: _provider_mix_report(
                wrapper_source=prepared.wrapper_source,
                matmul_calls=optimized_prepared.matmul_calls,
                patch_state=backend_state,
            ),
        }

        latency_ms = {
            "torch_eager_cuda": _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: _run_torch_eager(prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            "torch_compile_inductor": _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: _run_torch_compile(prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            CURRENT_MODE: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            BACKEND_PASS_MODE: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        optimized_prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        report = build_vit_p2_reentry_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            m129_report=source_m129,
            m129_report_path=m129_report,
            m1210_report=source_m1210,
            m1210_report_path=m1210_report,
            m1214_report=source_m1214,
            m1214_report_path=m1214_report,
            latency_ms=latency_ms,
            correctness=correctness,
            provider_mix=provider_mix,
            backend_pass_summary=_backend_pass_summary(optimized_prepared.matmul_calls),
            dependency_versions=_dependency_versions(prepared.torch),
            out_dir=out_dir,
        )
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "vit_tiny_random_wrapper.py").write_text(
                prepared.wrapper_source, encoding="utf-8"
            )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m12_15_failed:{type(err).__name__}:{err}",
            m129_report=m129_report,
            m1210_report=m1210_report,
            m1214_report=m1214_report,
            source_m129=source_m129,
            source_m1210=source_m1210,
            source_m1214=source_m1214,
        )
        _maybe_write_report(report, out_dir)
        return report


def build_vit_p2_reentry_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    m129_report: dict[str, Any],
    m129_report_path: str | Path,
    m1210_report: dict[str, Any],
    m1210_report_path: str | Path,
    m1214_report: dict[str, Any],
    m1214_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    backend_pass_summary: dict[str, Any],
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build the M12.15 P2 re-entry report from measured or synthetic data."""

    ratios = _ratio_report(latency_ms)
    host_staging_bytes = 0
    p0 = _p0_report(correctness, provider_mix)
    p1 = _p1_report(p0, ratios, host_staging_bytes)
    p2 = _p2_report(p0, ratios, host_staging_bytes, warmup)
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
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "source_reports": {
            "m12_9_report": str(m129_report_path),
            "m12_10_report": str(m1210_report_path),
            "m12_14_report": str(m1214_report_path),
        },
        "source_report_status": _source_report_status(m129_report, m1210_report, m1214_report),
        "source_evidence_policy": _source_evidence_policy(),
        "baselines": list(BASELINES),
        "measured_provider_modes": list(MEASURED_PROVIDER_MODES),
        "p2_eligible_mode": P2_ELIGIBLE_MODE,
        "latency_ms": latency_ms,
        "ratios": ratios,
        "correctness": correctness,
        "provider_mix": provider_mix,
        "backend_pass": backend_pass_summary,
        "diagnostic_exclusions": _diagnostic_exclusions(m129_report, m1210_report),
        "host_staging_bytes": host_staging_bytes,
        "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "m12_performance_tier": _performance_tier(p0, p1, p2),
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
        "next_default_action": {},
        "notes": [
            "M12.15 reruns the fixed-shape ViT P2 gate after M12.14.",
            "M12.14 is required backend-general schedule-readiness evidence, not E2E performance evidence.",
            "The M12.9 fused-QKV artifact remains diagnostic-only and is excluded from P2 eligibility.",
        ],
        "dependency_versions": dependency_versions or {},
    }
    report["invariants"] = _invariants(report)
    gate_ready = report["invariants"]["status"] == "passed"
    report["p2"]["passed"] = bool(gate_ready and report["p2"]["passed"])
    report["p2_passed"] = bool(report["p2"]["passed"])
    report["m12_performance_tier"] = _performance_tier(
        report["p0"],
        report["p1"],
        report["p2"],
    )
    report["performance_ready_e2e"] = bool(report["p2_passed"])
    report["performance_claim"] = bool(report["p2_passed"])
    report["completion_gate"] = _completion_gate(report)
    report["invariants"] = _invariants(report)
    report["next_default_action"] = _next_default_action(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _source_report_status(
    m129_report: dict[str, Any],
    m1210_report: dict[str, Any],
    m1214_report: dict[str, Any],
) -> dict[str, Any]:
    m1210_claims = m1210_report.get("route_claims") or {}
    fused_claim = m1210_claims.get(FUSED_QKV_MODE) or {}
    m1214_invariants = m1214_report.get("invariants") or {}
    return {
        "m12_9_status": m129_report.get("status"),
        "m12_9_invariants": (m129_report.get("invariants") or {}).get("status"),
        "m12_9_best_mode": (m129_report.get("best_mode") or {}).get("mode"),
        "m12_9_best_mode_p2_passed": (m129_report.get("p2_context") or {}).get(
            "best_mode_p2_passed"
        ),
        "m12_9_performance_ready_e2e": m129_report.get("performance_ready_e2e"),
        "m12_10_status": m1210_report.get("status"),
        "m12_10_invariants": (m1210_report.get("invariants") or {}).get("status"),
        "m12_10_fused_qkv_counts_for_backend_progress": fused_claim.get(
            "counts_for_triton_tvm_backend_progress"
        ),
        "m12_10_fused_qkv_label": fused_claim.get("route_diagnostic_label"),
        "m12_14_status": m1214_report.get("status"),
        "m12_14_invariants": m1214_invariants.get("status"),
        "m12_14_real_tt_dot_schedule_handoff_complete": m1214_report.get(
            "real_tt_dot_schedule_handoff_complete"
        ),
        "m12_14_p2_passed": m1214_report.get("p2_passed"),
        "m12_14_performance_ready_e2e": m1214_report.get("performance_ready_e2e"),
        "m12_14_next_action": (m1214_report.get("next_default_action") or {}).get("action"),
    }


def _source_evidence_policy() -> dict[str, Any]:
    return {
        "m12_14_required": True,
        "m12_14_counts_as_backend_general_schedule_readiness": True,
        "m12_14_counts_as_e2e_performance_evidence": False,
        "m12_14_counts_as_p2_evidence": False,
        "p2_eligible_mode": P2_ELIGIBLE_MODE,
        "p2_excluded_modes": [FUSED_QKV_MODE],
        "model_specific_artifacts_allowed_as_diagnostics_only": True,
    }


def _diagnostic_exclusions(
    m129_report: dict[str, Any],
    m1210_report: dict[str, Any],
) -> dict[str, Any]:
    m129_modes = m129_report.get("optimization_modes") or {}
    m1210_claims = m1210_report.get("route_claims") or {}
    fused_source = m1210_claims.get(FUSED_QKV_MODE) or m129_modes.get(FUSED_QKV_MODE) or {}
    return {
        FUSED_QKV_MODE: {
            "excluded_from_p2": True,
            "model_specific_artifact": bool(fused_source.get("model_specific_artifact", True)),
            "counts_for_triton_tvm_backend_progress": bool(
                fused_source.get("counts_for_triton_tvm_backend_progress", False)
            ),
            "route_diagnostic_label": fused_source.get(
                "route_diagnostic_label",
                "model_specific_diagnostic_only",
            ),
            "reason": (
                "M12.9 fixed-shape fused QKV is diagnostic-only per M12.10 and "
                "cannot be used as the M12.15 P2 path."
            ),
        }
    }


def _backend_pass_summary(matmul_calls) -> dict[str, Any]:
    per_call = []
    transformed = 0
    for call in matmul_calls:
        func = next(iter(call.artifact.irmod.functions.values()))
        schedule_id = str((func.attrs or {}).get("triton_tvm.schedule_id", ""))
        if schedule_id == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
            transformed += 1
        per_call.append(
            {
                "index": call.index,
                "op_family": call.op_family,
                "shape_mnk": [call.matmul_m, call.matmul_n, call.matmul_k],
                "schedule_id": schedule_id,
            }
        )
    return {
        "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
        "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
        "transformed_call_count": transformed,
        "expected_call_count": len(matmul_calls),
        "p2_eligible": True,
        "model_specific_artifact": False,
        "counts_for_triton_tvm_backend_progress": True,
        "per_call": per_call,
    }


def _run_torch_eager(prepared):
    with prepared.torch.no_grad():
        return prepared.model(prepared.pixel_values)


def _run_torch_compile(prepared):
    with prepared.torch.no_grad():
        return prepared.compiled(prepared.pixel_values)


def _time_cuda_callable(
    torch_module,
    fn: Callable[[], Any],
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


def _latency_record(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples_ms": [], "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(float(value) for value in values)
    return {
        "samples_ms": values,
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio_report(latency_ms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    ratios: dict[str, Any] = {}
    for mode in MEASURED_PROVIDER_MODES:
        record = latency_ms.get(mode, {})
        ratios[f"{mode}_to_torch_compile_p50"] = _safe_ratio(
            record.get("p50_ms"),
            compile_latency.get("p50_ms"),
        )
        ratios[f"{mode}_to_torch_compile_p95"] = _safe_ratio(
            record.get("p95_ms"),
            compile_latency.get("p95_ms"),
        )
    return ratios


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_f <= 0.0:
        return None
    return numerator_f / denominator_f


def _p0_report(
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligible_correctness = correctness.get(P2_ELIGIBLE_MODE, {})
    eligible_mix = provider_mix.get(P2_ELIGIBLE_MODE, {})
    report = {
        "eligible_mode": P2_ELIGIBLE_MODE,
        "correctness": bool(eligible_correctness.get("allclose")),
        "no_silent_fallback": bool(eligible_mix.get("no_silent_fallback")),
        "complete_provider_report": bool(eligible_mix.get("complete_provider_report")),
    }
    report["passed"] = bool(all(value for key, value in report.items() if key != "eligible_mode"))
    return report


def _p1_report(
    p0: dict[str, Any],
    ratios: dict[str, Any],
    host_staging_bytes: int,
) -> dict[str, Any]:
    p50 = ratios.get(f"{P2_ELIGIBLE_MODE}_to_torch_compile_p50")
    p95 = ratios.get(f"{P2_ELIGIBLE_MODE}_to_torch_compile_p95")
    return {
        "threshold": "<=3x torch_compile_inductor p50/p95",
        "eligible_mode": P2_ELIGIBLE_MODE,
        "inductor_baseline_ok": _ratio_pair_ok(p50, p95, P1_THRESHOLD),
        "zero_host_staging": host_staging_bytes == 0,
        "passed": bool(
            p0.get("passed")
            and host_staging_bytes == 0
            and _ratio_pair_ok(p50, p95, P1_THRESHOLD)
        ),
    }


def _p2_report(
    p0: dict[str, Any],
    ratios: dict[str, Any],
    host_staging_bytes: int,
    warmup: int,
) -> dict[str, Any]:
    p50 = ratios.get(f"{P2_ELIGIBLE_MODE}_to_torch_compile_p50")
    p95 = ratios.get(f"{P2_ELIGIBLE_MODE}_to_torch_compile_p95")
    return {
        "threshold": "<=2x torch_compile_inductor p50/p95",
        "eligible_mode": P2_ELIGIBLE_MODE,
        "torch_compile_ratio_p50": p50,
        "torch_compile_ratio_p95": p95,
        "inductor_baseline_ok": _ratio_pair_ok(p50, p95, P2_THRESHOLD),
        "warmed_cache": warmup > 0,
        "zero_host_staging": host_staging_bytes == 0,
        "complete_provider_mix": bool(p0.get("complete_provider_report")),
        "correctness": bool(p0.get("correctness")),
        "passed": bool(
            p0.get("passed")
            and host_staging_bytes == 0
            and warmup > 0
            and _ratio_pair_ok(p50, p95, P2_THRESHOLD)
        ),
    }


def _ratio_pair_ok(p50: Any, p95: Any, threshold: float) -> bool:
    try:
        return float(p50) <= threshold and float(p95) <= threshold
    except (TypeError, ValueError):
        return False


def _performance_tier(p0: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any]) -> str:
    if p2.get("passed"):
        return "P2"
    if p1.get("passed"):
        return "P1"
    if p0.get("passed"):
        return "P0"
    return "unavailable"


def _completion_gate(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "m12_complete": bool(report.get("p2_passed")),
        "requires_fixed_shape_vit_p2": True,
        "p2_passed": bool(report.get("p2_passed")),
        "performance_ready_e2e": bool(report.get("performance_ready_e2e")),
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
    }


def _next_default_action(report: dict[str, Any]) -> dict[str, Any]:
    if (report.get("invariants") or {}).get("status") != "passed":
        return {
            "action": "fix_m12_15_p2_reentry_invariants",
            "reason": "M12.15 source or gate invariants failed before an M12 close decision.",
        }
    if report.get("p2_passed"):
        return {
            "action": NEXT_ACTION_AFTER_P2_PASS,
            "reason": "Fixed-shape ViT passed the unchanged M12 P2 gate.",
        }
    return {
        "action": NEXT_ACTION_AFTER_P2_MISS,
        "reason": (
            "Fixed-shape ViT still misses P2 after the backend-general re-entry gate; "
            "continue with backend-general P2 gap analysis, not model-specific executor work."
        ),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("milestone") != MILESTONE:
        failures.append("m12_15_wrong_milestone")

    source_status = report.get("source_report_status") or {}
    if source_status.get("m12_9_status") != "passed":
        failures.append("m12_15_m12_9_not_passed")
    if source_status.get("m12_9_invariants") != "passed":
        failures.append("m12_15_m12_9_invariants_not_passed")
    if source_status.get("m12_10_status") != "passed":
        failures.append("m12_15_m12_10_not_passed")
    if source_status.get("m12_10_invariants") != "passed":
        failures.append("m12_15_m12_10_invariants_not_passed")
    if source_status.get("m12_14_status") != "passed":
        failures.append("m12_15_m12_14_not_passed")
    if source_status.get("m12_14_invariants") != "passed":
        failures.append("m12_15_m12_14_invariants_not_passed")
    if source_status.get("m12_14_real_tt_dot_schedule_handoff_complete") is not True:
        failures.append("m12_15_m12_14_schedule_handoff_missing")
    if source_status.get("m12_14_p2_passed") is not False:
        failures.append("m12_15_m12_14_treated_as_p2_evidence")
    if source_status.get("m12_14_performance_ready_e2e") is not False:
        failures.append("m12_15_m12_14_treated_as_performance_ready")

    policy = report.get("source_evidence_policy") or {}
    if policy.get("m12_14_required") is not True:
        failures.append("m12_15_m12_14_not_required")
    if policy.get("m12_14_counts_as_e2e_performance_evidence") is not False:
        failures.append("m12_15_m12_14_counts_as_e2e_performance")
    if policy.get("m12_14_counts_as_p2_evidence") is not False:
        failures.append("m12_15_m12_14_counts_as_p2_evidence")
    if policy.get("p2_eligible_mode") != P2_ELIGIBLE_MODE:
        failures.append("m12_15_wrong_p2_eligible_mode")
    if FUSED_QKV_MODE not in (policy.get("p2_excluded_modes") or []):
        failures.append("m12_15_fused_qkv_not_excluded_from_p2")
    if report.get("p2_eligible_mode") != P2_ELIGIBLE_MODE:
        failures.append("m12_15_report_wrong_p2_eligible_mode")
    if report.get("p2_eligible_mode") == FUSED_QKV_MODE:
        failures.append("m12_15_fused_qkv_selected_for_p2")
    if source_status.get("m12_10_fused_qkv_counts_for_backend_progress") is not False:
        failures.append("m12_15_fused_qkv_counts_for_backend_progress")
    if source_status.get("m12_10_fused_qkv_label") != "model_specific_diagnostic_only":
        failures.append("m12_15_fused_qkv_not_diagnostic_only")
    fused_exclusion = (report.get("diagnostic_exclusions") or {}).get(FUSED_QKV_MODE) or {}
    if fused_exclusion.get("excluded_from_p2") is not True:
        failures.append("m12_15_fused_qkv_exclusion_missing")
    if fused_exclusion.get("counts_for_triton_tvm_backend_progress") is not False:
        failures.append("m12_15_fused_qkv_exclusion_counts_for_progress")

    if report.get("warmed_cache") is not True:
        failures.append("m12_15_warmed_cache_required")
    if int(report.get("host_staging_bytes", 0) or 0) != 0:
        failures.append("m12_15_host_staging_nonzero")
    if report.get("backend_pass", {}).get("transformed_call_count") != report.get(
        "backend_pass", {}
    ).get("expected_call_count"):
        failures.append("m12_15_backend_pass_did_not_transform_all_calls")
    if report.get("backend_pass", {}).get("schedule_id") != M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
        failures.append("m12_15_backend_pass_wrong_schedule")

    latency_ms = report.get("latency_ms") or {}
    for name in list(BASELINES) + list(MEASURED_PROVIDER_MODES):
        record = latency_ms.get(name) or {}
        if _non_positive_or_missing(record.get("p50_ms")) or _non_positive_or_missing(
            record.get("p95_ms")
        ):
            failures.append(f"m12_15_missing_latency_{name}")

    correctness = report.get("correctness") or {}
    provider_mix = report.get("provider_mix") or {}
    for mode in MEASURED_PROVIDER_MODES:
        if correctness.get(mode, {}).get("allclose") is not True:
            failures.append(f"m12_15_correctness_failed_{mode}")
        if not _provider_mix_ok(provider_mix.get(mode, {})):
            failures.append(f"m12_15_provider_mix_incomplete_{mode}")

    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_15_strict_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_15_full_tvm_runnable_claim_present")
    if report.get("backend_general_complete") is not False:
        failures.append("m12_15_backend_general_complete_claim_present")
    if report.get("performance_ready_e2e") and not report.get("p2", {}).get("passed"):
        failures.append("m12_15_performance_ready_without_p2")
    if report.get("performance_claim") and not report.get("p2", {}).get("passed"):
        failures.append("m12_15_performance_claim_without_p2")
    completion = report.get("completion_gate") or {}
    if completion.get("requires_fixed_shape_vit_p2") is not True:
        failures.append("m12_15_p2_gate_not_preserved")
    if completion.get("m12_complete") and not report.get("p2", {}).get("passed"):
        failures.append("m12_15_m12_complete_without_p2")
    if completion.get("strict_full_tvm_native") is not False:
        failures.append("m12_15_completion_strict_native_claim_present")
    if int(completion.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_15_completion_full_runnable_claim_present")

    return {"status": "passed" if not failures else "failed", "invariant_failures": failures}


def _provider_mix_ok(provider_mix: dict[str, Any]) -> bool:
    if provider_mix.get("complete_provider_report") is not True:
        return False
    if provider_mix.get("no_silent_fallback") is not True:
        return False
    counts = {
        str(provider): int(count or 0)
        for provider, count in (provider_mix.get("provider_counts") or {}).items()
    }
    return counts == EXPECTED_PROVIDER_COUNTS


def _non_positive_or_missing(value: Any) -> bool:
    try:
        return float(value) <= 0.0
    except (TypeError, ValueError):
        return True


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    latency = report.get("latency_ms") or {}
    eligible = latency.get(P2_ELIGIBLE_MODE) or {}
    compile_latency = latency.get("torch_compile_inductor") or {}
    return {
        "m12_15_complete": report.get("status") == "passed",
        "m12_complete": (report.get("completion_gate") or {}).get("m12_complete"),
        "p2_passed": bool(report.get("p2_passed")),
        "performance_ready_e2e": bool(report.get("performance_ready_e2e")),
        "p2_eligible_mode": P2_ELIGIBLE_MODE,
        "eligible_mode_p50_ms": eligible.get("p50_ms"),
        "eligible_mode_p95_ms": eligible.get("p95_ms"),
        "torch_compile_p50_ms": compile_latency.get("p50_ms"),
        "torch_compile_p95_ms": compile_latency.get("p95_ms"),
        "torch_compile_ratio_p50": (report.get("p2") or {}).get("torch_compile_ratio_p50"),
        "torch_compile_ratio_p95": (report.get("p2") or {}).get("torch_compile_ratio_p95"),
        "next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    m129_report: str | Path,
    m1210_report: str | Path,
    m1214_report: str | Path,
    source_m129: dict[str, Any] | None = None,
    source_m1210: dict[str, Any] | None = None,
    source_m1214: dict[str, Any] | None = None,
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
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "source_reports": {
            "m12_9_report": str(m129_report),
            "m12_10_report": str(m1210_report),
            "m12_14_report": str(m1214_report),
        },
        "source_report_status": _source_report_status(
            source_m129 or {}, source_m1210 or {}, source_m1214 or {}
        ),
        "source_evidence_policy": _source_evidence_policy(),
        "baselines": list(BASELINES),
        "measured_provider_modes": list(MEASURED_PROVIDER_MODES),
        "p2_eligible_mode": P2_ELIGIBLE_MODE,
        "latency_ms": {},
        "ratios": {},
        "correctness": {},
        "provider_mix": {},
        "backend_pass": {
            "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
            "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            "transformed_call_count": 0,
            "expected_call_count": 0,
            "p2_eligible": True,
        },
        "diagnostic_exclusions": _diagnostic_exclusions(source_m129 or {}, source_m1210 or {}),
        "host_staging_bytes": 0,
        "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        "p0": {"passed": False},
        "p1": {"passed": False},
        "p2": {"passed": False, "eligible_mode": P2_ELIGIBLE_MODE},
        "m12_performance_tier": "unavailable",
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
        "next_default_action": {"action": "fix_m12_15_p2_reentry_invariants"},
        "dependency_versions": {},
        "invariants": {"status": "not_run", "invariant_failures": []},
        "summary": {
            "m12_15_complete": False,
            "m12_complete": False,
            "p2_passed": False,
            "performance_ready_e2e": False,
            "p2_eligible_mode": P2_ELIGIBLE_MODE,
            "next_action": "fix_m12_15_p2_reentry_invariants",
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
    latency = report.get("latency_ms") or {}
    summary = report.get("summary") or {}
    lines = [
        "# M12.15 Fixed-Shape ViT P2 Re-entry",
        "",
        f"- Status: `{report.get('status')}`",
        f"- P2 eligible mode: `{report.get('p2_eligible_mode')}`",
        f"- P2 passed: `{report.get('p2_passed')}`",
        f"- M12 complete: `{(report.get('completion_gate') or {}).get('m12_complete')}`",
        f"- Performance ready E2E: `{report.get('performance_ready_e2e')}`",
        f"- Next action: `{(report.get('next_default_action') or {}).get('action')}`",
        "",
        "## Latency",
        "",
        "| Path | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for name in list(BASELINES) + list(MEASURED_PROVIDER_MODES):
        record = latency.get(name) or {}
        lines.append(f"| `{name}` | {_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |")
    lines.extend(
        [
            "",
            "## P2",
            "",
            f"- Threshold: `{(report.get('p2') or {}).get('threshold')}`",
            f"- Ratio p50: `{summary.get('torch_compile_ratio_p50')}`",
            f"- Ratio p95: `{summary.get('torch_compile_ratio_p95')}`",
            f"- Warmed cache: `{report.get('warmed_cache')}`",
            f"- Host staging bytes: `{report.get('host_staging_bytes')}`",
            "",
            "## Source Evidence",
            "",
            f"- M12.14 required: `{(report.get('source_evidence_policy') or {}).get('m12_14_required')}`",
            "- M12.14 counts as E2E performance evidence: "
            f"`{(report.get('source_evidence_policy') or {}).get('m12_14_counts_as_e2e_performance_evidence')}`",
            f"- Fused-QKV excluded from P2: `{FUSED_QKV_MODE in ((report.get('source_evidence_policy') or {}).get('p2_excluded_modes') or [])}`",
            "",
            "## Invariants",
            "",
            f"- Status: `{(report.get('invariants') or {}).get('status')}`",
        ]
    )
    for failure in (report.get("invariants") or {}).get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m1215_vit_p2_reentry``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--m129-report", type=Path, default=DEFAULT_M129_REPORT)
    parser.add_argument("--m1210-report", type=Path, default=DEFAULT_M1210_REPORT)
    parser.add_argument("--m1214-report", type=Path, default=DEFAULT_M1214_REPORT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    report = run_vit_p2_reentry(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        run_benchmarks=not args.no_benchmarks,
        m129_report=args.m129_report,
        m1210_report=args.m1210_report,
        m1214_report=args.m1214_report,
    )
    summary = report.get("summary") or {}
    print(
        f"M12.15 fixed-shape ViT P2 re-entry: status={report['status']} "
        f"p2_passed={report.get('p2_passed')} "
        f"ratio_p50={summary.get('torch_compile_ratio_p50')} "
        f"ratio_p95={summary.get('torch_compile_ratio_p95')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
