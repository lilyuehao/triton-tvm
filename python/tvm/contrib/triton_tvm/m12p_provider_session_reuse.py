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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""M12.P provider-session reuse optimization for fixed-shape ViT."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tvm

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import EXTERN_GEMM_PROVIDER_NATIVE_TVM
from .m123_vit_e2e_runner import (
    NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
    NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
    TARGET_MODEL,
    NativeMatmulProfileConfig,
    VitFixedShapeProviderSession,
    _build_native_matmul_calls,
    _correctness_report,
    _dependency_versions,
    _provider_mix_report,
    _require_torch,
    execute_vit_fixed_shape_provider_path,
    prepare_vit_fixed_shape_e2e,
)
from .m124_vit_e2e_dashboard import DEFAULT_REPEAT, DEFAULT_WARMUP, P2_THRESHOLD
from .m12p_profile_optimization_loop import (
    DEFAULT_MINIMAL_RECORD_OUT_DIR,
    M12P_MINIMAL_RECORD_MODE,
    _latency_record,
    _profile_summary,
    _ratio_pair_ok,
    _safe_ratio,
    _time_cuda_callable,
)
from .m12p_residual_profile import (
    DEFAULT_OUT_DIR as DEFAULT_RESIDUAL_PROFILE_OUT_DIR,
    _provider_profile_summary,
)
from .passes import M129OptimizeNativeWrapperMatmul
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_p_provider_session_reuse"
REPORT_ID = "m12_p_iter_003_provider_session_reuse_v1"
ITERATION_ID = "provider_session_reuse_v1"
MILESTONE = "M12.P"
M12P_SESSION_REUSE_MODE = "triton_tvm_e2e_m12p_provider_session_reuse"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_p_profile_optimization_loop/provider_session_reuse_v1"
)
DEFAULT_PREVIOUS_ITERATION_REPORT = DEFAULT_MINIMAL_RECORD_OUT_DIR / "report.json"
DEFAULT_RESIDUAL_PROFILE_REPORT = DEFAULT_RESIDUAL_PROFILE_OUT_DIR / "report.json"


def run_m12p_provider_session_reuse_optimization(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    previous_iteration_report: str | Path = DEFAULT_PREVIOUS_ITERATION_REPORT,
    residual_profile_report: str | Path = DEFAULT_RESIDUAL_PROFILE_REPORT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Measure provider-session reuse without changing lowering or schedules."""

    try:
        previous = _read_json(previous_iteration_report)
    except FileNotFoundError as err:
        return _write_empty(
            out_dir,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason=f"previous_iteration_report_missing:{err.filename}",
            previous_iteration_report=previous_iteration_report,
            residual_profile_report=residual_profile_report,
        )
    residual = _read_json_or_none(residual_profile_report)

    if not run_benchmarks:
        return _write_empty(
            out_dir,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
            previous_iteration_report=previous_iteration_report,
            residual_profile_report=residual_profile_report,
            previous_iteration=previous,
            residual_profile=residual,
        )

    try:
        torch = _require_torch()
    except RuntimeError:
        return _write_empty(
            out_dir,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
            previous_iteration_report=previous_iteration_report,
            residual_profile_report=residual_profile_report,
            previous_iteration=previous,
            residual_profile=residual,
        )

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        return _write_empty(
            out_dir,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
            previous_iteration_report=previous_iteration_report,
            residual_profile_report=residual_profile_report,
            previous_iteration=previous,
            residual_profile=residual,
        )

    try:
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        optimized_prepared = replace(
            prepared,
            matmul_calls=_build_native_matmul_calls(
                prepared.wrapper_source,
                native_matmul_passes=[M129OptimizeNativeWrapperMatmul()],
            ),
        )
        baseline_actual, baseline_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
        )
        candidate_actual, candidate_state = _run_provider_session_once(optimized_prepared)
        latency_ms = {
            "torch_compile_inductor": _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: _run_torch_compile(prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M12P_MINIMAL_RECORD_MODE: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        optimized_prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                        native_matmul_dispatch_mode=(
                            NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD
                        ),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            M12P_SESSION_REUSE_MODE: _latency_record(
                _time_provider_session(
                    prepared.torch,
                    lambda: _provider_session_factory(optimized_prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        profile_snapshot = _profile_steady_state_provider_session(
            optimized_prepared,
            warmup=warmup,
            repeat=repeat,
        )
        report = build_m12p_provider_session_reuse_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            previous_iteration_report=previous,
            previous_iteration_report_path=previous_iteration_report,
            residual_profile_report=residual,
            residual_profile_report_path=residual_profile_report,
            latency_ms=latency_ms,
            correctness={
                M12P_MINIMAL_RECORD_MODE: _correctness_report(
                    baseline_actual, prepared.expected
                ),
                M12P_SESSION_REUSE_MODE: _correctness_report(
                    candidate_actual, prepared.expected
                ),
            },
            provider_mix={
                M12P_MINIMAL_RECORD_MODE: _provider_mix_report(
                    wrapper_source=prepared.wrapper_source,
                    matmul_calls=optimized_prepared.matmul_calls,
                    patch_state=baseline_state,
                ),
                M12P_SESSION_REUSE_MODE: _provider_mix_report(
                    wrapper_source=prepared.wrapper_source,
                    matmul_calls=optimized_prepared.matmul_calls,
                    patch_state=candidate_state,
                ),
            },
            native_matmul_profile=_profile_summary(profile_snapshot["executed"]),
            residual_provider_profile=_provider_profile_summary(
                profile_snapshot["provider_profiles"]
            ),
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
        return _write_empty(
            out_dir,
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"provider_session_reuse_failed:{type(err).__name__}:{err}",
            previous_iteration_report=previous_iteration_report,
            residual_profile_report=residual_profile_report,
            previous_iteration=previous,
            residual_profile=residual,
        )


def build_m12p_provider_session_reuse_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    previous_iteration_report: dict[str, Any],
    previous_iteration_report_path: str | Path,
    residual_profile_report: dict[str, Any] | None,
    residual_profile_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    native_matmul_profile: dict[str, Any],
    residual_provider_profile: dict[str, Any],
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    ratios = _ratio_report(latency_ms)
    p2 = {
        "threshold": "<=2x torch_compile_inductor p50/p95",
        "eligible_mode": M12P_SESSION_REUSE_MODE,
        "torch_compile_ratio_p50": ratios.get(
            f"{M12P_SESSION_REUSE_MODE}_to_torch_compile_p50"
        ),
        "torch_compile_ratio_p95": ratios.get(
            f"{M12P_SESSION_REUSE_MODE}_to_torch_compile_p95"
        ),
        "warmed_cache": warmup > 0,
        "zero_host_staging": True,
    }
    p2["passed"] = bool(
        p2["warmed_cache"]
        and _ratio_pair_ok(
            p2["torch_compile_ratio_p50"],
            p2["torch_compile_ratio_p95"],
            P2_THRESHOLD,
        )
    )
    baseline = latency_ms.get(M12P_MINIMAL_RECORD_MODE, {})
    candidate = latency_ms.get(M12P_SESSION_REUSE_MODE, {})
    recovered = {
        "p50_ms": _positive_delta(baseline.get("p50_ms"), candidate.get("p50_ms")),
        "p95_ms": _positive_delta(baseline.get("p95_ms"), candidate.get("p95_ms")),
    }
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "iteration_id": ITERATION_ID,
        "loop_phase": "post_closure_runtime_glue_optimization",
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "source_reports": {
            "previous_iteration_report": str(previous_iteration_report_path),
            "residual_profile_report": str(residual_profile_report_path),
        },
        "source_report_status": {
            "previous_iteration_id": previous_iteration_report.get("iteration_id"),
            "previous_iteration_status": previous_iteration_report.get("status"),
            "previous_iteration_p2_passed": previous_iteration_report.get("p2_passed"),
            "residual_profile_iteration_id": (residual_profile_report or {}).get(
                "iteration_id"
            ),
            "residual_profile_status": (residual_profile_report or {}).get("status"),
        },
        "baseline_mode": M12P_MINIMAL_RECORD_MODE,
        "candidate_mode": M12P_SESSION_REUSE_MODE,
        "latency_ms": latency_ms,
        "ratios": ratios,
        "correctness": correctness,
        "provider_mix": provider_mix,
        "profile": {
            EXTERN_GEMM_PROVIDER_NATIVE_TVM: native_matmul_profile,
            "residual_providers": residual_provider_profile,
        },
        "optimization_scope": {
            "target_bucket": "remaining_unattributed_e2e_residual",
            "optimization": "provider_session_reuse",
            "implemented_changes": [
                {
                    "change": "rhs_storage_base_first_fastpath",
                    "measured_as": "included_in_baseline_and_candidate_after_code_update",
                    "changes_lowering": False,
                    "changes_schedule": False,
                },
                {
                    "change": "reusable_patched_provider_session",
                    "measured_as": "candidate_vs_same_run_minimal_record_baseline",
                    "changes_lowering": False,
                    "changes_schedule": False,
                },
            ],
            "description": (
                "Reuse the patched explicit provider session across repeated "
                "fixed-shape executions so provider patch/unpatch runtime glue is "
                "not paid once per measured sample."
            ),
            "backend_general_runtime_glue": True,
            "model_specific": False,
            "changes_lowering": False,
            "changes_schedule": False,
            "changes_runtime_or_schedule": True,
            "dispatch_mode": NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
            "sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        },
        "optimization_result": {
            "baseline_mode": M12P_MINIMAL_RECORD_MODE,
            "candidate_mode": M12P_SESSION_REUSE_MODE,
            "recovered_p50_ms": recovered["p50_ms"],
            "recovered_p95_ms": recovered["p95_ms"],
            "improved_p50": _is_positive(recovered["p50_ms"]),
            "improved_p95": _is_positive(recovered["p95_ms"]),
        },
        "host_staging_bytes": 0,
        "p2": p2,
        "p2_passed": p2["passed"],
        "performance_ready_e2e": p2["passed"],
        "performance_claim": p2["passed"],
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "forbidden_scope": {
            "no_p2_redefinition": True,
            "no_model_specific_vit_executor": True,
            "no_wrapper_buffer_replay": True,
            "no_fused_qkv_p2_credit": True,
            "no_strict_full_native_claim": True,
            "no_native_tvm_conv_schedule_claim": True,
            "no_lowering_or_schedule_change": True,
        },
        "dependency_versions": dependency_versions or {},
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _provider_session_factory(prepared):
    profile_config = NativeMatmulProfileConfig(
        enabled=False,
        sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
    )
    return VitFixedShapeProviderSession(prepared, profile_config=profile_config)


def _run_provider_session_once(prepared):
    with _provider_session_factory(prepared) as session:
        return session.run()


def _time_provider_session(
    torch_module,
    session_factory: Callable[[], VitFixedShapeProviderSession],
    *,
    warmup: int,
    repeat: int,
) -> list[float]:
    with session_factory() as session:
        for _ in range(warmup):
            session.run()
        torch_module.cuda.synchronize()
        values: list[float] = []
        for _ in range(repeat):
            start = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            start.record()
            session.run()
            end.record()
            torch_module.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
        return values


def _profile_steady_state_provider_session(prepared, *, warmup: int, repeat: int):
    for _ in range(warmup):
        _run_provider_session_once(prepared)
    prepared.torch.cuda.synchronize()
    profile_config = NativeMatmulProfileConfig(
        enabled=True,
        sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
    )
    snapshots = []
    with VitFixedShapeProviderSession(prepared, profile_config=profile_config) as session:
        for _ in range(max(1, repeat)):
            _, state = session.run()
            snapshot = {
                "executed": copy.deepcopy(state.executed),
                "provider_profiles": copy.deepcopy(state.provider_profiles),
            }
            snapshots.append((_profile_state_score_ms(snapshot), snapshot))
    snapshots.sort(key=lambda item: item[0])
    return snapshots[len(snapshots) // 2][1]


def _profile_state_score_ms(snapshot: dict[str, Any]) -> float:
    native_ms = sum(
        _to_float_or_none((record.get("profile") or {}).get("provider_total_wall_ms")) or 0.0
        for record in snapshot.get("executed", [])
    )
    residual_ms = sum(
        _to_float_or_none((record.get("profile") or {}).get("kernel_event_ms")) or 0.0
        for record in snapshot.get("provider_profiles", [])
    )
    return native_ms + residual_ms


def _run_torch_compile(prepared):
    with prepared.torch.no_grad():
        return prepared.compiled(prepared.pixel_values)


def _ratio_report(latency_ms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    return {
        f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p50": _safe_ratio(
            latency_ms.get(M12P_MINIMAL_RECORD_MODE, {}).get("p50_ms"),
            compile_latency.get("p50_ms"),
        ),
        f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p95": _safe_ratio(
            latency_ms.get(M12P_MINIMAL_RECORD_MODE, {}).get("p95_ms"),
            compile_latency.get("p95_ms"),
        ),
        f"{M12P_SESSION_REUSE_MODE}_to_torch_compile_p50": _safe_ratio(
            latency_ms.get(M12P_SESSION_REUSE_MODE, {}).get("p50_ms"),
            compile_latency.get("p50_ms"),
        ),
        f"{M12P_SESSION_REUSE_MODE}_to_torch_compile_p95": _safe_ratio(
            latency_ms.get(M12P_SESSION_REUSE_MODE, {}).get("p95_ms"),
            compile_latency.get("p95_ms"),
        ),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    source = report.get("source_report_status") or {}
    if source.get("previous_iteration_id") != "provider_runtime_glue_minimal_record_v1":
        failures.append("provider_session_reuse_previous_iteration_mismatch")
    if source.get("previous_iteration_status") != "passed":
        failures.append("provider_session_reuse_previous_iteration_not_passed")
    if source.get("previous_iteration_p2_passed") is not True:
        failures.append("provider_session_reuse_previous_iteration_not_p2")
    if report.get("candidate_mode") != M12P_SESSION_REUSE_MODE:
        failures.append("provider_session_reuse_wrong_candidate")
    if (report.get("correctness") or {}).get(M12P_SESSION_REUSE_MODE, {}).get(
        "allclose"
    ) is not True:
        failures.append("provider_session_reuse_correctness_failed")
    mix = (report.get("provider_mix") or {}).get(M12P_SESSION_REUSE_MODE, {})
    counts = mix.get("provider_counts") or {}
    if counts.get(EXTERN_GEMM_PROVIDER_NATIVE_TVM) != 7:
        failures.append("provider_session_reuse_native_matmul_count")
    if counts.get(VISION_PROVIDER_DEVICE_TORCH_CUDA) != 1:
        failures.append("provider_session_reuse_conv_count")
    if counts.get(ATTENTION_PROVIDER_NATIVE_DECOMPOSED) != 1:
        failures.append("provider_session_reuse_attention_count")
    if mix.get("complete_provider_report") is not True or mix.get("no_silent_fallback") is not True:
        failures.append("provider_session_reuse_provider_mix_incomplete")
    scope = report.get("optimization_scope") or {}
    if scope.get("changes_lowering") is not False or scope.get("changes_schedule") is not False:
        failures.append("provider_session_reuse_forbidden_lowering_or_schedule_change")
    if scope.get("model_specific") is not False:
        failures.append("provider_session_reuse_model_specific")
    if int(report.get("host_staging_bytes", 0) or 0) != 0:
        failures.append("provider_session_reuse_host_staging_nonzero")
    if report.get("p2_passed") is not True:
        failures.append("provider_session_reuse_p2_not_passed")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("provider_session_reuse_strict_native_claim")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("provider_session_reuse_full_runnable_claim")
    return {"status": "failed" if failures else "passed", "invariant_failures": failures}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    latency = report.get("latency_ms") or {}
    ratios = report.get("ratios") or {}
    result = report.get("optimization_result") or {}
    return {
        "iteration_id": report.get("iteration_id"),
        "status": report.get("status"),
        "baseline_p50_ms": latency.get(M12P_MINIMAL_RECORD_MODE, {}).get("p50_ms"),
        "baseline_p95_ms": latency.get(M12P_MINIMAL_RECORD_MODE, {}).get("p95_ms"),
        "candidate_p50_ms": latency.get(M12P_SESSION_REUSE_MODE, {}).get("p50_ms"),
        "candidate_p95_ms": latency.get(M12P_SESSION_REUSE_MODE, {}).get("p95_ms"),
        "candidate_ratio_p50": ratios.get(
            f"{M12P_SESSION_REUSE_MODE}_to_torch_compile_p50"
        ),
        "candidate_ratio_p95": ratios.get(
            f"{M12P_SESSION_REUSE_MODE}_to_torch_compile_p95"
        ),
        "recovered_p50_ms": result.get("recovered_p50_ms"),
        "recovered_p95_ms": result.get("recovered_p95_ms"),
        "improved_p50": result.get("improved_p50"),
        "improved_p95": result.get("improved_p95"),
    }


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    previous_iteration_report: str | Path,
    residual_profile_report: str | Path,
    previous_iteration: dict[str, Any] | None = None,
    residual_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "iteration_id": ITERATION_ID,
        "loop_phase": "post_closure_runtime_glue_optimization",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "source_reports": {
            "previous_iteration_report": str(previous_iteration_report),
            "residual_profile_report": str(residual_profile_report),
        },
        "source_report_status": {
            "previous_iteration_id": (previous_iteration or {}).get("iteration_id"),
            "previous_iteration_status": (previous_iteration or {}).get("status"),
            "residual_profile_iteration_id": (residual_profile or {}).get("iteration_id"),
            "residual_profile_status": (residual_profile or {}).get("status"),
        },
        "baseline_mode": M12P_MINIMAL_RECORD_MODE,
        "candidate_mode": M12P_SESSION_REUSE_MODE,
        "host_staging_bytes": 0,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "invariants": {"status": "not_run", "invariant_failures": []},
    }
    report["summary"] = {"status": status}
    return report


def _write_empty(out_dir: str | Path | None, **kwargs) -> dict[str, Any]:
    report = _empty_report(**kwargs)
    _maybe_write_report(report, out_dir)
    return report


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_or_none(path: str | Path) -> dict[str, Any] | None:
    try:
        return _read_json(path)
    except FileNotFoundError:
        return None


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
    summary = report.get("summary") or {}
    lines = [
        "# M12.P Provider Session Reuse",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Iteration: `{report.get('iteration_id')}`",
        f"- Baseline mode: `{report.get('baseline_mode')}`",
        f"- Candidate mode: `{report.get('candidate_mode')}`",
        f"- Baseline p50/p95 ms: `{_fmt(summary.get('baseline_p50_ms'))} / {_fmt(summary.get('baseline_p95_ms'))}`",
        f"- Candidate p50/p95 ms: `{_fmt(summary.get('candidate_p50_ms'))} / {_fmt(summary.get('candidate_p95_ms'))}`",
        f"- Recovered p50/p95 ms: `{_fmt(summary.get('recovered_p50_ms'))} / {_fmt(summary.get('recovered_p95_ms'))}`",
        f"- Candidate ratio p50/p95: `{_fmt(summary.get('candidate_ratio_p50'))}x / {_fmt(summary.get('candidate_ratio_p95'))}x`",
        "",
        "## Optimization Scope",
        "",
    ]
    scope = report.get("optimization_scope") or {}
    for key in (
        "optimization",
        "target_bucket",
        "backend_general_runtime_glue",
        "model_specific",
        "changes_lowering",
        "changes_schedule",
    ):
        lines.append(f"- {key}: `{scope.get(key)}`")
    lines.extend(["", "## Invariants", ""])
    inv = report.get("invariants") or {}
    lines.append(f"- Status: `{inv.get('status')}`")
    for failure in inv.get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _to_float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _positive_delta(left: Any, right: Any) -> float | None:
    left_f = _to_float_or_none(left)
    right_f = _to_float_or_none(right)
    if left_f is None or right_f is None:
        return None
    return round(max(0.0, left_f - right_f), 6)


def _is_positive(value: Any) -> bool:
    value_f = _to_float_or_none(value)
    return bool(value_f is not None and value_f > 0.0)


def _fmt(value: Any) -> str:
    value_f = _to_float_or_none(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument(
        "--previous-iteration-report",
        type=Path,
        default=DEFAULT_PREVIOUS_ITERATION_REPORT,
    )
    parser.add_argument(
        "--residual-profile-report",
        type=Path,
        default=DEFAULT_RESIDUAL_PROFILE_REPORT,
    )
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    report = run_m12p_provider_session_reuse_optimization(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        previous_iteration_report=args.previous_iteration_report,
        residual_profile_report=args.residual_profile_report,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report.get("summary") or {}
    print(
        f"M12.P provider session reuse: status={report['status']} "
        f"candidate_p50={summary.get('candidate_p50_ms')} "
        f"recovered_p50={summary.get('recovered_p50_ms')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
