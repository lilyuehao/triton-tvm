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
"""M12.P repeatable profiling/performance optimization loop."""

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
    NATIVE_MATMUL_DISPATCH_MODE_ARTIFACT_RUN,
    NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
    NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL,
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
from .m124_vit_e2e_dashboard import DEFAULT_REPEAT, DEFAULT_WARMUP, P2_THRESHOLD
from .m129_vit_provider_runtime_optimization import BACKEND_PASS_MODE
from .passes import M129OptimizeNativeWrapperMatmul
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_p_profile_optimization_loop"
REPORT_ID = "m12_p_iter_001_prebound_packed_call_v1"
MILESTONE = "M12.P"
ITERATION_ID = "provider_runtime_glue_prebound_packed_call_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_p_profile_optimization_loop/provider_runtime_glue_prebound_packed_call_v1"
)
MINIMAL_RECORD_REPORT_ID = "m12_p_iter_002_minimal_record_v1"
MINIMAL_RECORD_ITERATION_ID = "provider_runtime_glue_minimal_record_v1"
DEFAULT_MINIMAL_RECORD_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_p_profile_optimization_loop/provider_runtime_glue_minimal_record_v1"
)
DEFAULT_M1215_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_15_fixed_shape_vit_p2_reentry/report.json"
)
DEFAULT_M1216_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_16_backend_general_p2_gap_analysis/report.json"
)
DEFAULT_PREBOUND_ITERATION_REPORT = DEFAULT_OUT_DIR / "report.json"
M12P_PREBOUND_MODE = "triton_tvm_e2e_m12p_prebound_packed_call"
M12P_MINIMAL_RECORD_MODE = "triton_tvm_e2e_m12p_minimal_record"
BASELINES = ("torch_eager_cuda", "torch_compile_inductor")
MEASURED_PROVIDER_MODES = (BACKEND_PASS_MODE, M12P_PREBOUND_MODE)
MINIMAL_RECORD_MEASURED_PROVIDER_MODES = (M12P_PREBOUND_MODE, M12P_MINIMAL_RECORD_MODE)
EXPECTED_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}
NEXT_ACTION_CONTINUE = "m12_p_profile_optimization_loop"
NEXT_ACTION_CLOSE = "close_m12_p2_passed"

PREBOUND_OPTIMIZATION_SCOPE = {
    "target_bucket": "native_tvm_matmul_runtime_glue",
    "optimization": "prebound_packed_call",
    "description": (
        "Use the already-built TVM PackedFunc for M12 fixed-shape native matmul "
        "provider dispatch, avoiding per-call artifact.run validation and lookup."
    ),
    "backend_general": True,
    "model_specific": False,
    "counts_for_triton_tvm_backend_progress": True,
    "changes_runtime_or_schedule": True,
    "dispatch_mode": NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL,
    "sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
}
MINIMAL_RECORD_OPTIMIZATION_SCOPE = {
    "target_bucket": "native_tvm_matmul_runtime_glue",
    "optimization": "prebound_minimal_record",
    "description": (
        "Use the prebound TVM PackedFunc path with minimal non-profile diagnostics: "
        "keep provider records for counts, but skip per-call Python timing buckets during "
        "latency execution."
    ),
    "backend_general": True,
    "model_specific": False,
    "counts_for_triton_tvm_backend_progress": True,
    "changes_runtime_or_schedule": True,
    "dispatch_mode": NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
    "sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
}


def run_m12p_profile_optimization_loop(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
    m1215_report: str | Path = DEFAULT_M1215_REPORT,
    m1216_report: str | Path = DEFAULT_M1216_REPORT,
    previous_iteration_report: str | Path | None = None,
    report_id: str = REPORT_ID,
    iteration_id: str = ITERATION_ID,
    baseline_mode: str = BACKEND_PASS_MODE,
    baseline_dispatch_mode: str = NATIVE_MATMUL_DISPATCH_MODE_ARTIFACT_RUN,
    candidate_mode: str = M12P_PREBOUND_MODE,
    candidate_dispatch_mode: str = NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL,
    measured_provider_modes: tuple[str, ...] = MEASURED_PROVIDER_MODES,
    optimization_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one M12.P profile -> optimize -> verify -> decide iteration."""

    iteration_context = {
        "report_id": report_id,
        "iteration_id": iteration_id,
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "measured_provider_modes": measured_provider_modes,
        "optimization_scope": optimization_scope,
    }
    try:
        source_m1215 = _read_json_report(m1215_report)
        source_m1216 = _read_json_report(m1216_report)
        source_previous = (
            _read_json_report(previous_iteration_report)
            if previous_iteration_report is not None
            else None
        )
    except FileNotFoundError as err:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason=f"source_report_missing:{err.filename}",
            m1215_report=m1215_report,
            m1216_report=m1216_report,
            **iteration_context,
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
            m1215_report=m1215_report,
            m1216_report=m1216_report,
            **iteration_context,
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
            m1215_report=m1215_report,
            m1216_report=m1216_report,
            source_m1215=source_m1215,
            source_m1216=source_m1216,
            source_previous=source_previous,
            previous_iteration_report=previous_iteration_report,
            **iteration_context,
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
            m1215_report=m1215_report,
            m1216_report=m1216_report,
            source_m1215=source_m1215,
            source_m1216=source_m1216,
            source_previous=source_previous,
            previous_iteration_report=previous_iteration_report,
            **iteration_context,
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
            m1215_report=m1215_report,
            m1216_report=m1216_report,
            source_m1215=source_m1215,
            source_m1216=source_m1216,
            source_previous=source_previous,
            previous_iteration_report=previous_iteration_report,
            **iteration_context,
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

        baseline_actual, baseline_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=baseline_dispatch_mode,
        )
        candidate_actual, candidate_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=candidate_dispatch_mode,
        )
        correctness = {
            baseline_mode: _correctness_report(baseline_actual, prepared.expected),
            candidate_mode: _correctness_report(candidate_actual, prepared.expected),
        }
        provider_mix = {
            baseline_mode: _provider_mix_report(
                wrapper_source=prepared.wrapper_source,
                matmul_calls=optimized_prepared.matmul_calls,
                patch_state=baseline_state,
            ),
            candidate_mode: _provider_mix_report(
                wrapper_source=prepared.wrapper_source,
                matmul_calls=optimized_prepared.matmul_calls,
                patch_state=candidate_state,
            ),
        }
        _, baseline_profile_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            profile_native_matmul=True,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=baseline_dispatch_mode,
        )
        _, candidate_profile_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            profile_native_matmul=True,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=candidate_dispatch_mode,
        )

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
            baseline_mode: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        optimized_prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                        native_matmul_dispatch_mode=baseline_dispatch_mode,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            candidate_mode: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        optimized_prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                        native_matmul_dispatch_mode=candidate_dispatch_mode,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }

        report = build_m12p_profile_optimization_loop_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            m1215_report=source_m1215,
            m1215_report_path=m1215_report,
            m1216_report=source_m1216,
            m1216_report_path=m1216_report,
            previous_iteration_report=source_previous,
            previous_iteration_report_path=previous_iteration_report,
            report_id=report_id,
            iteration_id=iteration_id,
            baseline_mode=baseline_mode,
            candidate_mode=candidate_mode,
            measured_provider_modes=measured_provider_modes,
            latency_ms=latency_ms,
            correctness=correctness,
            provider_mix=provider_mix,
            backend_pass_summary=_backend_pass_summary(optimized_prepared.matmul_calls),
            profile={
                baseline_mode: _profile_summary(baseline_profile_state.executed),
                candidate_mode: _profile_summary(candidate_profile_state.executed),
            },
            dependency_versions=_dependency_versions(prepared.torch),
            optimization_scope=optimization_scope,
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
            availability_reason=f"m12_p_iteration_failed:{type(err).__name__}:{err}",
            m1215_report=m1215_report,
            m1216_report=m1216_report,
            source_m1215=source_m1215,
            source_m1216=source_m1216,
            source_previous=source_previous,
            previous_iteration_report=previous_iteration_report,
            **iteration_context,
        )
        _maybe_write_report(report, out_dir)
        return report


def run_m12p_minimal_record_optimization_loop(
    *,
    out_dir: str | Path | None = DEFAULT_MINIMAL_RECORD_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
    m1215_report: str | Path = DEFAULT_M1215_REPORT,
    m1216_report: str | Path = DEFAULT_M1216_REPORT,
    previous_iteration_report: str | Path | None = DEFAULT_PREBOUND_ITERATION_REPORT,
) -> dict[str, Any]:
    """Run the second M12.P runtime-glue iteration with minimal record dispatch."""

    return run_m12p_profile_optimization_loop(
        out_dir=out_dir,
        warmup=warmup,
        repeat=repeat,
        seed=seed,
        run_benchmarks=run_benchmarks,
        m1215_report=m1215_report,
        m1216_report=m1216_report,
        previous_iteration_report=previous_iteration_report,
        report_id=MINIMAL_RECORD_REPORT_ID,
        iteration_id=MINIMAL_RECORD_ITERATION_ID,
        baseline_mode=M12P_PREBOUND_MODE,
        baseline_dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL,
        candidate_mode=M12P_MINIMAL_RECORD_MODE,
        candidate_dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
        measured_provider_modes=MINIMAL_RECORD_MEASURED_PROVIDER_MODES,
        optimization_scope=MINIMAL_RECORD_OPTIMIZATION_SCOPE,
    )


def build_m12p_profile_optimization_loop_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    m1215_report: dict[str, Any],
    m1215_report_path: str | Path,
    m1216_report: dict[str, Any],
    m1216_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    backend_pass_summary: dict[str, Any],
    profile: dict[str, dict[str, Any]],
    previous_iteration_report: dict[str, Any] | None = None,
    previous_iteration_report_path: str | Path | None = None,
    report_id: str = REPORT_ID,
    iteration_id: str = ITERATION_ID,
    baseline_mode: str = BACKEND_PASS_MODE,
    candidate_mode: str = M12P_PREBOUND_MODE,
    measured_provider_modes: tuple[str, ...] = MEASURED_PROVIDER_MODES,
    dependency_versions: dict[str, Any] | None = None,
    changed_files: list[str] | None = None,
    optimization_scope: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build one M12.P iteration report from measured or synthetic inputs."""

    host_staging_bytes = 0
    ratios = _ratio_report(latency_ms, measured_provider_modes=measured_provider_modes)
    p2 = _p2_report(
        ratios,
        host_staging_bytes,
        warmup,
        candidate_mode=candidate_mode,
    )
    recovered = _recovered_latency(
        latency_ms,
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
    )
    profile_buckets = _profile_buckets(
        profile,
        latency_ms,
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
    )
    source_reports = {
        "m12_15_report": str(m1215_report_path),
        "m12_16_report": str(m1216_report_path),
    }
    if previous_iteration_report_path is not None:
        source_reports["previous_iteration_report"] = str(previous_iteration_report_path)
    scope = dict(optimization_scope or PREBOUND_OPTIMIZATION_SCOPE)
    scope["changed_files"] = changed_files or [
        "python/tvm/contrib/triton_tvm/m123_vit_e2e_runner.py",
        "python/tvm/contrib/triton_tvm/m12p_profile_optimization_loop.py",
        "tests/python/contrib/test_triton_tvm_model_corpus.py",
    ]
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": report_id,
        "milestone": MILESTONE,
        "iteration_id": iteration_id,
        "loop_phase": "profile_classify_optimize_verify_decide",
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "source_reports": source_reports,
        "source_report_status": _source_report_status(
            m1215_report,
            m1216_report,
            previous_iteration_report=previous_iteration_report,
        ),
        "entry_p2_snapshot": _entry_p2_snapshot(m1216_report),
        "previous_iteration_snapshot": _previous_iteration_snapshot(previous_iteration_report),
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "measured_provider_modes": list(measured_provider_modes),
        "p2_eligible_mode": candidate_mode,
        "latency_ms": latency_ms,
        "ratios": ratios,
        "recovered_ms": recovered,
        "correctness": correctness,
        "provider_mix": provider_mix,
        "backend_pass": backend_pass_summary,
        "profile": profile,
        "profile_buckets": profile_buckets,
        "selected_primary_bucket": "native_tvm_matmul_runtime_glue",
        "optimization_scope": scope,
        "host_staging_bytes": host_staging_bytes,
        "p2": p2,
        "p2_passed": False,
        "performance_ready_e2e": False,
        "performance_claim": False,
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
        "forbidden_scope": _forbidden_scope(),
        "dependency_versions": dependency_versions or {},
    }
    report["invariants"] = _invariants(report)
    gate_passed = bool(report["invariants"]["status"] == "passed" and p2["passed"])
    report["p2"]["passed"] = gate_passed
    report["p2_passed"] = gate_passed
    report["performance_ready_e2e"] = gate_passed
    report["performance_claim"] = gate_passed
    report["backend_general_complete"] = gate_passed
    report["completion_gate"] = {
        "m12_complete": gate_passed,
        "requires_fixed_shape_vit_p2": True,
        "p2_passed": gate_passed,
        "performance_ready_e2e": gate_passed,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
    }
    report["next_default_action"] = _next_default_action(report)
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


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


def _profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = (
        "shape_abi_validation_ms",
        "b_storage_resolution_ms",
        "dlpack_conversion_ms",
        "artifact_dispatch_wall_ms",
        "kernel_event_ms",
        "dispatch_minus_kernel_estimate_ms",
        "provider_total_wall_ms",
    )
    aggregate = {category: 0.0 for category in categories}
    samples = {category: 0 for category in categories}
    per_call = []
    for record in records:
        profile = record.get("profile") or {}
        row = {
            "index": record.get("index"),
            "op_family": record.get("op_family"),
            "shape_mnk": [
                record.get("matmul_m"),
                record.get("matmul_n"),
                record.get("matmul_k"),
            ],
            "schedule_id": record.get("schedule_id"),
            "dispatch_mode": profile.get("dispatch_mode"),
            "categories": {},
        }
        for category in categories:
            value = _to_float_or_none(profile.get(category))
            row["categories"][category] = value
            if value is not None:
                aggregate[category] += value
                samples[category] += 1
        per_call.append(row)
    return {
        "record_count": len(records),
        "aggregate_sum_ms": {
            category: round(total, 6) if samples[category] else None
            for category, total in aggregate.items()
        },
        "per_call": per_call,
    }


def _profile_buckets(
    profile: dict[str, dict[str, Any]],
    latency_ms: dict[str, dict[str, Any]],
    *,
    baseline_mode: str = BACKEND_PASS_MODE,
    candidate_mode: str = M12P_PREBOUND_MODE,
) -> list[dict[str, Any]]:
    baseline = profile.get(baseline_mode, {}).get("aggregate_sum_ms") or {}
    candidate = profile.get(candidate_mode, {}).get("aggregate_sum_ms") or {}
    candidate_provider = _to_float_or_none(candidate.get("provider_total_wall_ms"))
    candidate_kernel = _to_float_or_none(candidate.get("kernel_event_ms"))
    candidate_glue = _delta(candidate_provider, candidate_kernel)
    candidate_p50 = _to_float_or_none(latency_ms.get(candidate_mode, {}).get("p50_ms"))
    return [
        {
            "bucket_id": "native_tvm_matmul_provider",
            "priority": "primary",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "baseline_ms": _to_float_or_none(baseline.get("provider_total_wall_ms")),
            "candidate_ms": candidate_provider,
            "recovered_ms": _positive_delta(
                _to_float_or_none(baseline.get("provider_total_wall_ms")),
                candidate_provider,
            ),
        },
        {
            "bucket_id": "native_tvm_matmul_kernel",
            "priority": "primary_subbucket",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "baseline_ms": _to_float_or_none(baseline.get("kernel_event_ms")),
            "candidate_ms": candidate_kernel,
            "recovered_ms": _positive_delta(
                _to_float_or_none(baseline.get("kernel_event_ms")),
                candidate_kernel,
            ),
        },
        {
            "bucket_id": "native_tvm_matmul_runtime_glue",
            "priority": "selected",
            "backend_general": True,
            "model_specific": False,
            "actionable": True,
            "baseline_ms": _delta(
                _to_float_or_none(baseline.get("provider_total_wall_ms")),
                _to_float_or_none(baseline.get("kernel_event_ms")),
            ),
            "candidate_ms": candidate_glue,
            "recovered_ms": _positive_delta(
                _delta(
                    _to_float_or_none(baseline.get("provider_total_wall_ms")),
                    _to_float_or_none(baseline.get("kernel_event_ms")),
                ),
                candidate_glue,
            ),
        },
        {
            "bucket_id": "remaining_e2e_residual",
            "priority": "watch",
            "backend_general": True,
            "model_specific": False,
            "actionable": False,
            "estimated_p50_ms": _positive_delta(candidate_p50, candidate_provider),
            "evidence_kind": "derived_candidate_e2e_minus_profiled_matmul_provider",
        },
    ]


def _source_report_status(
    m1215_report: dict[str, Any],
    m1216_report: dict[str, Any],
    *,
    previous_iteration_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = {
        "m12_15_status": m1215_report.get("status"),
        "m12_15_invariants": (m1215_report.get("invariants") or {}).get("status"),
        "m12_15_p2_passed": m1215_report.get("p2_passed"),
        "m12_15_performance_ready_e2e": m1215_report.get("performance_ready_e2e"),
        "m12_15_p2_eligible_mode": m1215_report.get("p2_eligible_mode"),
        "m12_16_status": m1216_report.get("status"),
        "m12_16_invariants": (m1216_report.get("invariants") or {}).get("status"),
        "m12_16_p2_passed": m1216_report.get("p2_passed"),
        "m12_16_next_action": (m1216_report.get("next_default_action") or {}).get("action"),
        "m12_16_selected_primary_bucket": (m1216_report.get("selected_next_slice") or {}).get(
            "primary_bucket"
        ),
    }
    if previous_iteration_report is not None:
        status.update(
            {
                "previous_iteration_id": previous_iteration_report.get("iteration_id"),
                "previous_iteration_status": previous_iteration_report.get("status"),
                "previous_iteration_invariants": (
                    previous_iteration_report.get("invariants") or {}
                ).get("status"),
                "previous_iteration_p2_passed": previous_iteration_report.get("p2_passed"),
                "previous_iteration_candidate_mode": previous_iteration_report.get(
                    "candidate_mode"
                ),
            }
        )
    return status


def _entry_p2_snapshot(m1216_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "m12_16_backend_general_p2_gap_analysis",
        "p2_gap_budget": m1216_report.get("p2_gap_budget") or {},
        "dominant_measured_bucket": (m1216_report.get("summary") or {}).get(
            "dominant_measured_bucket"
        )
        or (m1216_report.get("attribution") or {}).get("dominant_measured_bucket"),
        "selected_next_slice": m1216_report.get("selected_next_slice") or {},
    }


def _previous_iteration_snapshot(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "iteration_id": report.get("iteration_id"),
        "status": report.get("status"),
        "invariants_status": (report.get("invariants") or {}).get("status"),
        "p2_passed": report.get("p2_passed"),
        "candidate_mode": report.get("candidate_mode"),
        "candidate_p50_ms": (report.get("latency_ms") or {})
        .get(report.get("candidate_mode"), {})
        .get("p50_ms"),
        "candidate_p95_ms": (report.get("latency_ms") or {})
        .get(report.get("candidate_mode"), {})
        .get("p95_ms"),
    }


def _ratio_report(
    latency_ms: dict[str, dict[str, Any]],
    *,
    measured_provider_modes: tuple[str, ...] = MEASURED_PROVIDER_MODES,
) -> dict[str, Any]:
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    ratios: dict[str, Any] = {}
    for mode in measured_provider_modes:
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


def _p2_report(
    ratios: dict[str, Any],
    host_staging_bytes: int,
    warmup: int,
    *,
    candidate_mode: str = M12P_PREBOUND_MODE,
) -> dict[str, Any]:
    p50 = ratios.get(f"{candidate_mode}_to_torch_compile_p50")
    p95 = ratios.get(f"{candidate_mode}_to_torch_compile_p95")
    return {
        "threshold": "<=2x torch_compile_inductor p50/p95",
        "eligible_mode": candidate_mode,
        "torch_compile_ratio_p50": p50,
        "torch_compile_ratio_p95": p95,
        "inductor_baseline_ok": _ratio_pair_ok(p50, p95, P2_THRESHOLD),
        "warmed_cache": warmup > 0,
        "zero_host_staging": host_staging_bytes == 0,
        "passed": bool(
            warmup > 0
            and host_staging_bytes == 0
            and _ratio_pair_ok(p50, p95, P2_THRESHOLD)
        ),
    }


def _recovered_latency(
    latency_ms: dict[str, dict[str, Any]],
    *,
    baseline_mode: str = BACKEND_PASS_MODE,
    candidate_mode: str = M12P_PREBOUND_MODE,
) -> dict[str, Any]:
    baseline = latency_ms.get(baseline_mode, {})
    candidate = latency_ms.get(candidate_mode, {})
    return {
        "p50_ms": _delta(
            _to_float_or_none(baseline.get("p50_ms")),
            _to_float_or_none(candidate.get("p50_ms")),
        ),
        "p95_ms": _delta(
            _to_float_or_none(baseline.get("p95_ms")),
            _to_float_or_none(candidate.get("p95_ms")),
        ),
    }


def _next_default_action(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("p2_passed"):
        return {
            "action": NEXT_ACTION_CLOSE,
            "reason": "M12.P iteration passed the unchanged fixed-shape ViT P2 gate.",
        }
    return {
        "action": NEXT_ACTION_CONTINUE,
        "next_loop_target": "native_tvm_matmul_kernel",
        "reason": (
            "Prebound packed-call dispatch closed one runtime-glue substep, but P2 still "
            "requires another M12.P iteration under the unchanged gate."
        ),
    }


def _forbidden_scope() -> dict[str, Any]:
    return {
        "no_m12_17_numbering": True,
        "no_p2_redefinition": True,
        "no_model_specific_vit_executor": True,
        "no_wrapper_line_or_buffer_replay": True,
        "no_fused_qkv_p2_credit": True,
        "no_m12_14_as_e2e_performance_evidence": True,
        "no_strict_full_native_claim": True,
        "no_native_tvm_conv_schedule_claim": True,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("milestone") != MILESTONE:
        failures.append("m12_p_wrong_milestone")
    iteration_id = report.get("iteration_id")
    if iteration_id not in {ITERATION_ID, MINIMAL_RECORD_ITERATION_ID}:
        failures.append("m12_p_wrong_iteration_id")
    candidate_mode = str(report.get("candidate_mode", ""))
    baseline_mode = str(report.get("baseline_mode", ""))
    measured_provider_modes = tuple(report.get("measured_provider_modes") or MEASURED_PROVIDER_MODES)

    source = report.get("source_report_status") or {}
    if source.get("m12_15_status") != "passed":
        failures.append("m12_p_m12_15_not_passed")
    if source.get("m12_15_invariants") != "passed":
        failures.append("m12_p_m12_15_invariants_not_passed")
    if source.get("m12_15_p2_passed") is not False:
        failures.append("m12_p_m12_15_not_p2_miss")
    if source.get("m12_16_status") != "passed":
        failures.append("m12_p_m12_16_not_passed")
    if source.get("m12_16_invariants") != "passed":
        failures.append("m12_p_m12_16_invariants_not_passed")
    if source.get("m12_16_p2_passed") is not False:
        failures.append("m12_p_m12_16_not_p2_miss")
    if source.get("m12_16_next_action") != NEXT_ACTION_CONTINUE:
        failures.append("m12_p_m12_16_next_action_mismatch")
    if source.get("m12_16_selected_primary_bucket") != "native_tvm_matmul_provider":
        failures.append("m12_p_wrong_entry_primary_bucket")
    if iteration_id == MINIMAL_RECORD_ITERATION_ID:
        if baseline_mode != M12P_PREBOUND_MODE:
            failures.append("m12_p_previous_candidate_not_baseline")
        if candidate_mode != M12P_MINIMAL_RECORD_MODE:
            failures.append("m12_p_wrong_minimal_record_candidate")
        if source.get("previous_iteration_id") != ITERATION_ID:
            failures.append("m12_p_previous_iteration_id_missing")
        if source.get("previous_iteration_status") != "passed":
            failures.append("m12_p_previous_iteration_not_passed")
        if source.get("previous_iteration_invariants") != "passed":
            failures.append("m12_p_previous_iteration_invariants_not_passed")
        if source.get("previous_iteration_p2_passed") is not True:
            failures.append("m12_p_previous_iteration_not_p2_passed")
        if source.get("previous_iteration_candidate_mode") != M12P_PREBOUND_MODE:
            failures.append("m12_p_previous_iteration_candidate_mismatch")
    elif iteration_id == ITERATION_ID:
        if baseline_mode != BACKEND_PASS_MODE:
            failures.append("m12_p_wrong_first_iteration_baseline")
        if candidate_mode != M12P_PREBOUND_MODE:
            failures.append("m12_p_wrong_first_iteration_candidate")

    if report.get("warmed_cache") is not True:
        failures.append("m12_p_warmed_cache_required")
    if int(report.get("host_staging_bytes", 0) or 0) != 0:
        failures.append("m12_p_host_staging_nonzero")

    scope = report.get("optimization_scope") or {}
    if scope.get("target_bucket") != "native_tvm_matmul_runtime_glue":
        failures.append("m12_p_wrong_optimization_bucket")
    if scope.get("backend_general") is not True:
        failures.append("m12_p_optimization_not_backend_general")
    if scope.get("model_specific") is not False:
        failures.append("m12_p_optimization_model_specific")
    if scope.get("counts_for_triton_tvm_backend_progress") is not True:
        failures.append("m12_p_optimization_not_backend_progress")
    expected_dispatch = (
        NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD
        if iteration_id == MINIMAL_RECORD_ITERATION_ID
        else NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL
    )
    if scope.get("dispatch_mode") != expected_dispatch:
        failures.append("m12_p_wrong_dispatch_mode")
    if scope.get("sync_policy") != NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC:
        failures.append("m12_p_wrong_sync_policy")

    backend_pass = report.get("backend_pass") or {}
    if backend_pass.get("schedule_id") != M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
        failures.append("m12_p_backend_pass_wrong_schedule")
    if backend_pass.get("transformed_call_count") != backend_pass.get("expected_call_count"):
        failures.append("m12_p_backend_pass_did_not_transform_all_calls")
    if backend_pass.get("p2_eligible") is not True:
        failures.append("m12_p_backend_pass_not_p2_eligible")
    if backend_pass.get("model_specific_artifact") is not False:
        failures.append("m12_p_backend_pass_model_specific")

    latency = report.get("latency_ms") or {}
    for mode in list(BASELINES) + list(measured_provider_modes):
        record = latency.get(mode) or {}
        if _non_positive_or_missing(record.get("p50_ms")):
            failures.append(f"m12_p_missing_latency_p50_{mode}")
        if _non_positive_or_missing(record.get("p95_ms")):
            failures.append(f"m12_p_missing_latency_p95_{mode}")

    correctness = report.get("correctness") or {}
    provider_mix = report.get("provider_mix") or {}
    for mode in measured_provider_modes:
        if correctness.get(mode, {}).get("allclose") is not True:
            failures.append(f"m12_p_correctness_failed_{mode}")
        if not _provider_mix_ok(provider_mix.get(mode, {})):
            failures.append(f"m12_p_provider_mix_incomplete_{mode}")

    profile = report.get("profile") or {}
    for mode in measured_provider_modes:
        mode_profile = profile.get(mode) or {}
        if int(mode_profile.get("record_count", 0) or 0) != 7:
            failures.append(f"m12_p_profile_record_count_{mode}")
        aggregate = mode_profile.get("aggregate_sum_ms") or {}
        for key in ("provider_total_wall_ms", "kernel_event_ms", "artifact_dispatch_wall_ms"):
            if _to_float_or_none(aggregate.get(key)) is None:
                failures.append(f"m12_p_profile_missing_{mode}_{key}")

    if report.get("p2_eligible_mode") != candidate_mode:
        failures.append("m12_p_wrong_p2_eligible_mode")
    p2 = report.get("p2") or {}
    if p2.get("threshold") != "<=2x torch_compile_inductor p50/p95":
        failures.append("m12_p_p2_threshold_changed")
    if p2.get("passed") is not True:
        for key in ("performance_ready_e2e", "performance_claim", "backend_general_complete"):
            if report.get(key) is not False:
                failures.append(f"m12_p_forbidden_claim_without_p2_{key}")
        completion = report.get("completion_gate") or {}
        if completion.get("m12_complete") is not False:
            failures.append("m12_p_m12_complete_without_p2")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_p_strict_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_p_full_tvm_runnable_claim_present")

    forbidden = report.get("forbidden_scope") or {}
    for key in (
        "no_m12_17_numbering",
        "no_p2_redefinition",
        "no_model_specific_vit_executor",
        "no_wrapper_line_or_buffer_replay",
        "no_fused_qkv_p2_credit",
        "no_m12_14_as_e2e_performance_evidence",
        "no_strict_full_native_claim",
        "no_native_tvm_conv_schedule_claim",
    ):
        if forbidden.get(key) is not True:
            failures.append(f"m12_p_forbidden_scope_missing_{key}")

    return {"status": "failed" if failures else "passed", "invariant_failures": failures}


def _provider_mix_ok(mix: dict[str, Any]) -> bool:
    if mix.get("complete_provider_report") is not True:
        return False
    if mix.get("no_silent_fallback") is not True:
        return False
    counts = {str(key): int(value or 0) for key, value in (mix.get("provider_counts") or {}).items()}
    return counts == EXPECTED_PROVIDER_COUNTS


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    buckets = {bucket["bucket_id"]: bucket for bucket in report.get("profile_buckets", [])}
    candidate_mode = report.get("candidate_mode")
    return {
        "iteration_id": report.get("iteration_id"),
        "status": report.get("status"),
        "selected_primary_bucket": report.get("selected_primary_bucket"),
        "candidate_mode": candidate_mode,
        "candidate_p50_ms": (report.get("latency_ms") or {})
        .get(candidate_mode, {})
        .get("p50_ms"),
        "candidate_p95_ms": (report.get("latency_ms") or {})
        .get(candidate_mode, {})
        .get("p95_ms"),
        "candidate_ratio_p50": (report.get("ratios") or {}).get(
            f"{candidate_mode}_to_torch_compile_p50"
        ),
        "candidate_ratio_p95": (report.get("ratios") or {}).get(
            f"{candidate_mode}_to_torch_compile_p95"
        ),
        "recovered_p50_ms": (report.get("recovered_ms") or {}).get("p50_ms"),
        "recovered_p95_ms": (report.get("recovered_ms") or {}).get("p95_ms"),
        "provider_candidate_ms": buckets.get("native_tvm_matmul_provider", {}).get(
            "candidate_ms"
        ),
        "runtime_glue_candidate_ms": buckets.get("native_tvm_matmul_runtime_glue", {}).get(
            "candidate_ms"
        ),
        "p2_passed": report.get("p2_passed"),
        "next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    m1215_report: str | Path,
    m1216_report: str | Path,
    source_m1215: dict[str, Any] | None = None,
    source_m1216: dict[str, Any] | None = None,
    source_previous: dict[str, Any] | None = None,
    previous_iteration_report: str | Path | None = None,
    report_id: str = REPORT_ID,
    iteration_id: str = ITERATION_ID,
    baseline_mode: str = BACKEND_PASS_MODE,
    candidate_mode: str = M12P_PREBOUND_MODE,
    measured_provider_modes: tuple[str, ...] = MEASURED_PROVIDER_MODES,
    optimization_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_reports = {
        "m12_15_report": str(m1215_report),
        "m12_16_report": str(m1216_report),
    }
    if previous_iteration_report is not None:
        source_reports["previous_iteration_report"] = str(previous_iteration_report)
    scope = dict(optimization_scope or PREBOUND_OPTIMIZATION_SCOPE)
    return {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": report_id,
        "milestone": MILESTONE,
        "iteration_id": iteration_id,
        "loop_phase": "profile_classify_optimize_verify_decide",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "source_reports": source_reports,
        "source_report_status": _source_report_status(
            source_m1215 or {},
            source_m1216 or {},
            previous_iteration_report=source_previous,
        ),
        "entry_p2_snapshot": _entry_p2_snapshot(source_m1216 or {}),
        "previous_iteration_snapshot": _previous_iteration_snapshot(source_previous),
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "measured_provider_modes": list(measured_provider_modes),
        "p2_eligible_mode": candidate_mode,
        "latency_ms": {},
        "ratios": {},
        "recovered_ms": {"p50_ms": None, "p95_ms": None},
        "correctness": {},
        "provider_mix": {},
        "backend_pass": {
            "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
            "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            "transformed_call_count": 0,
            "expected_call_count": 0,
            "p2_eligible": True,
            "model_specific_artifact": False,
            "counts_for_triton_tvm_backend_progress": True,
            "per_call": [],
        },
        "profile": {},
        "profile_buckets": [],
        "selected_primary_bucket": "native_tvm_matmul_runtime_glue",
        "optimization_scope": scope,
        "host_staging_bytes": 0,
        "p2": {"passed": False},
        "p2_passed": False,
        "performance_ready_e2e": False,
        "performance_claim": False,
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
        "forbidden_scope": _forbidden_scope(),
        "invariants": {"status": "not_run", "invariant_failures": []},
        "next_default_action": {"action": NEXT_ACTION_CONTINUE},
        "summary": {"next_action": NEXT_ACTION_CONTINUE},
    }


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
        "# M12.P Profile Optimization Loop Iteration",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Iteration: `{report.get('iteration_id')}`",
        f"- Primary bucket: `{report.get('selected_primary_bucket')}`",
        f"- Candidate mode: `{report.get('candidate_mode')}`",
        f"- P2 passed: `{report.get('p2_passed')}`",
        f"- Performance-ready E2E: `{report.get('performance_ready_e2e')}`",
        f"- Next action: `{summary.get('next_action')}`",
        "",
        "## Latency",
        "",
        "| Path | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for name, record in latency.items():
        lines.append(f"| `{name}` | {_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |")
    lines.extend(
        [
            "",
            "## Profile Buckets",
            "",
            "| Bucket | Baseline ms | Candidate ms | Recovered ms |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for bucket in report.get("profile_buckets", []):
        lines.append(
            f"| `{bucket.get('bucket_id')}` | {_fmt(bucket.get('baseline_ms'))} | "
            f"{_fmt(bucket.get('candidate_ms'))} | {_fmt(bucket.get('recovered_ms'))} |"
        )
    lines.extend(
        [
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


def _read_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_f <= 0.0:
        return None
    return numerator_f / denominator_f


def _ratio_pair_ok(p50: Any, p95: Any, threshold: float) -> bool:
    try:
        return float(p50) <= threshold and float(p95) <= threshold
    except (TypeError, ValueError):
        return False


def _to_float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    left_f = _to_float_or_none(left)
    right_f = _to_float_or_none(right)
    if left_f is None or right_f is None:
        return None
    return round(left_f - right_f, 6)


def _positive_delta(left: Any, right: Any) -> float | None:
    delta = _delta(left, right)
    if delta is None:
        return None
    return round(max(0.0, delta), 6)


def _non_positive_or_missing(value: Any) -> bool:
    value_f = _to_float_or_none(value)
    return value_f is None or value_f <= 0.0


def _fmt(value: Any) -> str:
    value_f = _to_float_or_none(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m12p_profile_optimization_loop``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--iteration",
        choices=("prebound_packed_call", "minimal_record"),
        default="prebound_packed_call",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--m1215-report", type=Path, default=DEFAULT_M1215_REPORT)
    parser.add_argument("--m1216-report", type=Path, default=DEFAULT_M1216_REPORT)
    parser.add_argument(
        "--previous-iteration-report",
        type=Path,
        default=DEFAULT_PREBOUND_ITERATION_REPORT,
    )
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    if args.iteration == "minimal_record":
        out_dir = args.out_dir or DEFAULT_MINIMAL_RECORD_OUT_DIR
        report = run_m12p_minimal_record_optimization_loop(
            out_dir=out_dir,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            run_benchmarks=not args.no_benchmarks,
            m1215_report=args.m1215_report,
            m1216_report=args.m1216_report,
            previous_iteration_report=args.previous_iteration_report,
        )
    else:
        out_dir = args.out_dir or DEFAULT_OUT_DIR
        report = run_m12p_profile_optimization_loop(
            out_dir=out_dir,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            run_benchmarks=not args.no_benchmarks,
            m1215_report=args.m1215_report,
            m1216_report=args.m1216_report,
        )
    summary = report.get("summary") or {}
    print(
        f"M12.P iteration: status={report['status']} "
        f"p2={report.get('p2_passed')} "
        f"candidate_p50={summary.get('candidate_p50_ms')} out_dir={out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
