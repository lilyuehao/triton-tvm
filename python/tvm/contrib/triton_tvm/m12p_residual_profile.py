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
"""M12.P post-closure residual profiling for fixed-shape ViT."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tvm

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import EXTERN_GEMM_PROVIDER_NATIVE_TVM
from .m123_vit_e2e_runner import (
    NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
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
from .m12p_profile_optimization_loop import (
    DEFAULT_MINIMAL_RECORD_OUT_DIR,
    M12P_MINIMAL_RECORD_MODE,
    _latency_record,
    _profile_summary,
    _ratio_pair_ok,
    _safe_ratio,
    _time_cuda_callable,
)
from .passes import M129OptimizeNativeWrapperMatmul
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_p_residual_profile"
REPORT_ID = "m12_p_residual_profile_001_v1"
ITERATION_ID = "remaining_e2e_residual_profile_v1"
MILESTONE = "M12.P"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_p_profile_optimization_loop/remaining_e2e_residual_profile_v1"
)
DEFAULT_PREVIOUS_ITERATION_REPORT = DEFAULT_MINIMAL_RECORD_OUT_DIR / "report.json"


def run_m12p_residual_profile(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    previous_iteration_report: str | Path = DEFAULT_PREVIOUS_ITERATION_REPORT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Profile the post-M12.P residual without changing lowering or schedules."""

    try:
        previous = _read_json(previous_iteration_report)
    except FileNotFoundError as err:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason=f"previous_iteration_report_missing:{err.filename}",
            previous_iteration_report=previous_iteration_report,
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
            previous_iteration_report=previous_iteration_report,
            previous_iteration=previous,
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
            previous_iteration_report=previous_iteration_report,
            previous_iteration=previous,
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
            previous_iteration_report=previous_iteration_report,
            previous_iteration=previous,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        optimized_prepared = replace(
            prepared,
            matmul_calls=_build_native_matmul_calls(
                prepared.wrapper_source,
                native_matmul_passes=[M129OptimizeNativeWrapperMatmul()],
            ),
        )
        actual, state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
        )
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
        }
        profile_state = _profile_steady_state_minimal_record(
            optimized_prepared,
            warmup=warmup,
            repeat=repeat,
        )
        report = build_m12p_residual_profile_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            previous_iteration_report=previous,
            previous_iteration_report_path=previous_iteration_report,
            latency_ms=latency_ms,
            correctness={M12P_MINIMAL_RECORD_MODE: _correctness_report(actual, prepared.expected)},
            provider_mix={
                M12P_MINIMAL_RECORD_MODE: _provider_mix_report(
                    wrapper_source=prepared.wrapper_source,
                    matmul_calls=optimized_prepared.matmul_calls,
                    patch_state=state,
                )
            },
            native_matmul_profile=_profile_summary(profile_state.executed),
            residual_provider_profile=_provider_profile_summary(profile_state.provider_profiles),
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
            availability_reason=f"residual_profile_failed:{type(err).__name__}:{err}",
            previous_iteration_report=previous_iteration_report,
            previous_iteration=previous,
        )
        _maybe_write_report(report, out_dir)
        return report


def _profile_steady_state_minimal_record(
    prepared: Any,
    *,
    warmup: int,
    repeat: int,
):
    for _ in range(warmup):
        execute_vit_fixed_shape_provider_path(
            prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
        )
    prepared.torch.cuda.synchronize()

    profiled_states = []
    for _ in range(max(1, repeat)):
        _, state = execute_vit_fixed_shape_provider_path(
            prepared,
            profile_native_matmul=True,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            native_matmul_dispatch_mode=NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
        )
        profiled_states.append((_profile_state_score_ms(state), state))
    profiled_states.sort(key=lambda item: item[0])
    return profiled_states[len(profiled_states) // 2][1]


def _profile_state_score_ms(state: Any) -> float:
    native_ms = sum(
        _to_float_or_none((record.get("profile") or {}).get("provider_total_wall_ms")) or 0.0
        for record in state.executed
    )
    residual_ms = sum(
        _to_float_or_none((record.get("profile") or {}).get("kernel_event_ms")) or 0.0
        for record in state.provider_profiles
    )
    return native_ms + residual_ms


def build_m12p_residual_profile_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    previous_iteration_report: dict[str, Any],
    previous_iteration_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    native_matmul_profile: dict[str, Any],
    residual_provider_profile: dict[str, Any],
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build the residual profile report from measured or synthetic inputs."""

    ratios = _ratio_report(latency_ms)
    p2 = {
        "threshold": "<=2x torch_compile_inductor p50/p95",
        "eligible_mode": M12P_MINIMAL_RECORD_MODE,
        "torch_compile_ratio_p50": ratios.get(
            f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p50"
        ),
        "torch_compile_ratio_p95": ratios.get(
            f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p95"
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
    profile_buckets = _profile_buckets(
        latency_ms=latency_ms,
        native_matmul_profile=native_matmul_profile,
        residual_provider_profile=residual_provider_profile,
    )
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "iteration_id": ITERATION_ID,
        "loop_phase": "profile_residual_only",
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "source_reports": {
            "previous_iteration_report": str(previous_iteration_report_path),
        },
        "previous_iteration_snapshot": {
            "iteration_id": previous_iteration_report.get("iteration_id"),
            "status": previous_iteration_report.get("status"),
            "invariants_status": (previous_iteration_report.get("invariants") or {}).get(
                "status"
            ),
            "p2_passed": previous_iteration_report.get("p2_passed"),
            "candidate_mode": previous_iteration_report.get("candidate_mode"),
        },
        "candidate_mode": M12P_MINIMAL_RECORD_MODE,
        "latency_ms": latency_ms,
        "ratios": ratios,
        "correctness": correctness,
        "provider_mix": provider_mix,
        "profile": {
            EXTERN_GEMM_PROVIDER_NATIVE_TVM: native_matmul_profile,
            "residual_providers": residual_provider_profile,
        },
        "profile_selection": {
            "method": "median_total_profile_cost_after_unprofiled_warmup",
            "representative_execution_count": 1,
            "warmup": warmup,
            "repeat": repeat,
        },
        "profile_buckets": profile_buckets,
        "selected_next_slice": _selected_next_slice(profile_buckets),
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
        },
        "dependency_versions": dependency_versions or {},
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _provider_profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, dict[str, Any]] = {}
    for record in records:
        provider = str(record.get("provider_kind", "unknown"))
        entry = by_provider.setdefault(
            provider,
            {
                "record_count": 0,
                "aggregate_sum_ms": {
                    "provider_total_wall_ms": 0.0,
                    "kernel_event_ms": 0.0,
                    "dispatch_minus_kernel_estimate_ms": 0.0,
                },
                "per_call": [],
            },
        )
        entry["record_count"] += 1
        profile = record.get("profile") or {}
        row = {
            "index": record.get("index"),
            "op_family": record.get("op_family"),
            "provider_kind": provider,
            "categories": {},
        }
        for key in (
            "provider_total_wall_ms",
            "kernel_event_ms",
            "dispatch_minus_kernel_estimate_ms",
        ):
            value = _to_float_or_none(profile.get(key))
            row["categories"][key] = value
            if value is not None:
                entry["aggregate_sum_ms"][key] += value
        entry["per_call"].append(row)
    for entry in by_provider.values():
        entry["aggregate_sum_ms"] = {
            key: round(value, 6) for key, value in entry["aggregate_sum_ms"].items()
        }
    return {"record_count": len(records), "by_provider": by_provider}


def _profile_buckets(
    *,
    latency_ms: dict[str, dict[str, Any]],
    native_matmul_profile: dict[str, Any],
    residual_provider_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    e2e_p50 = _to_float_or_none(latency_ms.get(M12P_MINIMAL_RECORD_MODE, {}).get("p50_ms"))
    native = (native_matmul_profile.get("aggregate_sum_ms") or {}).get(
        "provider_total_wall_ms"
    )
    providers = residual_provider_profile.get("by_provider") or {}
    conv = ((providers.get(VISION_PROVIDER_DEVICE_TORCH_CUDA) or {}).get("aggregate_sum_ms") or {}).get(
        "kernel_event_ms"
    )
    attention = (
        (providers.get(ATTENTION_PROVIDER_NATIVE_DECOMPOSED) or {}).get("aggregate_sum_ms")
        or {}
    ).get("kernel_event_ms")
    measured_total = sum(
        value
        for value in (
            _to_float_or_none(native),
            _to_float_or_none(conv),
            _to_float_or_none(attention),
        )
        if value is not None
    )
    return [
        {
            "bucket_id": "native_tvm_matmul_provider",
            "priority": "measured",
            "backend_general": True,
            "candidate_ms": _to_float_or_none(native),
            "evidence_kind": "profiled_provider_wall",
        },
        {
            "bucket_id": "device_torch_cuda_conv",
            "priority": "measured_residual_provider",
            "backend_general": False,
            "candidate_ms": _to_float_or_none(conv),
            "evidence_kind": "profiled_cuda_event",
        },
        {
            "bucket_id": "native_decomposed_attention",
            "priority": "measured_residual_provider",
            "backend_general": False,
            "candidate_ms": _to_float_or_none(attention),
            "evidence_kind": "profiled_cuda_event",
        },
        {
            "bucket_id": "remaining_unattributed_e2e_residual",
            "priority": "next_profile_target",
            "backend_general": True,
            "estimated_p50_ms": _positive_delta(e2e_p50, measured_total),
            "evidence_kind": "e2e_p50_minus_profiled_provider_components",
        },
    ]


def _selected_next_slice(profile_buckets: list[dict[str, Any]]) -> dict[str, Any]:
    residual = next(
        bucket
        for bucket in profile_buckets
        if bucket["bucket_id"] == "remaining_unattributed_e2e_residual"
    )
    return {
        "action": "continue_residual_profile",
        "primary_bucket": residual["bucket_id"],
        "reason": (
            "Conv and attention providers are now measured; the remaining residual should "
            "be attributed before another optimization."
        ),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    previous = report.get("previous_iteration_snapshot") or {}
    if previous.get("iteration_id") != "provider_runtime_glue_minimal_record_v1":
        failures.append("residual_profile_previous_iteration_mismatch")
    if previous.get("status") != "passed" or previous.get("invariants_status") != "passed":
        failures.append("residual_profile_previous_iteration_not_passed")
    if previous.get("p2_passed") is not True:
        failures.append("residual_profile_previous_iteration_not_p2")
    if report.get("candidate_mode") != M12P_MINIMAL_RECORD_MODE:
        failures.append("residual_profile_wrong_candidate")
    if (report.get("correctness") or {}).get(M12P_MINIMAL_RECORD_MODE, {}).get("allclose") is not True:
        failures.append("residual_profile_correctness_failed")
    mix = (report.get("provider_mix") or {}).get(M12P_MINIMAL_RECORD_MODE, {})
    counts = mix.get("provider_counts") or {}
    if counts.get(EXTERN_GEMM_PROVIDER_NATIVE_TVM) != 7:
        failures.append("residual_profile_native_matmul_count")
    if counts.get(VISION_PROVIDER_DEVICE_TORCH_CUDA) != 1:
        failures.append("residual_profile_conv_count")
    if counts.get(ATTENTION_PROVIDER_NATIVE_DECOMPOSED) != 1:
        failures.append("residual_profile_attention_count")
    if mix.get("complete_provider_report") is not True or mix.get("no_silent_fallback") is not True:
        failures.append("residual_profile_provider_mix_incomplete")
    native_profile = (report.get("profile") or {}).get(EXTERN_GEMM_PROVIDER_NATIVE_TVM) or {}
    if int(native_profile.get("record_count", 0) or 0) != 7:
        failures.append("residual_profile_native_profile_count")
    providers = ((report.get("profile") or {}).get("residual_providers") or {}).get(
        "by_provider"
    ) or {}
    if int((providers.get(VISION_PROVIDER_DEVICE_TORCH_CUDA) or {}).get("record_count", 0) or 0) != 1:
        failures.append("residual_profile_conv_profile_count")
    if int((providers.get(ATTENTION_PROVIDER_NATIVE_DECOMPOSED) or {}).get("record_count", 0) or 0) != 1:
        failures.append("residual_profile_attention_profile_count")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("residual_profile_strict_native_claim")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("residual_profile_full_runnable_claim")
    return {"status": "failed" if failures else "passed", "invariant_failures": failures}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    buckets = {bucket["bucket_id"]: bucket for bucket in report.get("profile_buckets", [])}
    ratios = report.get("ratios") or {}
    return {
        "iteration_id": report.get("iteration_id"),
        "status": report.get("status"),
        "candidate_p50_ms": (report.get("latency_ms") or {})
        .get(M12P_MINIMAL_RECORD_MODE, {})
        .get("p50_ms"),
        "candidate_ratio_p50": ratios.get(
            f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p50"
        ),
        "native_tvm_matmul_provider_ms": buckets.get("native_tvm_matmul_provider", {}).get(
            "candidate_ms"
        ),
        "device_torch_cuda_conv_ms": buckets.get("device_torch_cuda_conv", {}).get(
            "candidate_ms"
        ),
        "native_decomposed_attention_ms": buckets.get("native_decomposed_attention", {}).get(
            "candidate_ms"
        ),
        "remaining_unattributed_e2e_residual_ms": buckets.get(
            "remaining_unattributed_e2e_residual", {}
        ).get("estimated_p50_ms"),
        "selected_next_action": (report.get("selected_next_slice") or {}).get("action"),
    }


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    previous_iteration_report: str | Path,
    previous_iteration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "iteration_id": ITERATION_ID,
        "loop_phase": "profile_residual_only",
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "source_reports": {"previous_iteration_report": str(previous_iteration_report)},
        "previous_iteration_snapshot": {
            "iteration_id": (previous_iteration or {}).get("iteration_id"),
            "status": (previous_iteration or {}).get("status"),
        },
        "candidate_mode": M12P_MINIMAL_RECORD_MODE,
        "host_staging_bytes": 0,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "invariants": {"status": "not_run", "invariant_failures": []},
    }
    report["summary"] = {"status": status}
    return report


def _run_torch_compile(prepared):
    with prepared.torch.no_grad():
        return prepared.compiled(prepared.pixel_values)


def _ratio_report(latency_ms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    candidate = latency_ms.get(M12P_MINIMAL_RECORD_MODE, {})
    return {
        f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p50": _safe_ratio(
            candidate.get("p50_ms"), compile_latency.get("p50_ms")
        ),
        f"{M12P_MINIMAL_RECORD_MODE}_to_torch_compile_p95": _safe_ratio(
            candidate.get("p95_ms"), compile_latency.get("p95_ms")
        ),
    }


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


def _read_json(path: str | Path) -> dict[str, Any]:
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
    summary = report.get("summary") or {}
    lines = [
        "# M12.P Residual Profile",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Iteration: `{report.get('iteration_id')}`",
        f"- Candidate mode: `{report.get('candidate_mode')}`",
        f"- Candidate p50 ms: `{_fmt(summary.get('candidate_p50_ms'))}`",
        f"- Native matmul provider ms: `{_fmt(summary.get('native_tvm_matmul_provider_ms'))}`",
        f"- Conv ms: `{_fmt(summary.get('device_torch_cuda_conv_ms'))}`",
        f"- Attention ms: `{_fmt(summary.get('native_decomposed_attention_ms'))}`",
        f"- Remaining unattributed residual ms: `{_fmt(summary.get('remaining_unattributed_e2e_residual_ms'))}`",
        "",
        "## Profile Buckets",
        "",
        "| Bucket | ms | evidence |",
        "| --- | ---: | --- |",
    ]
    for bucket in report.get("profile_buckets", []):
        value = bucket.get("candidate_ms", bucket.get("estimated_p50_ms"))
        lines.append(
            f"| `{bucket.get('bucket_id')}` | {_fmt(value)} | `{bucket.get('evidence_kind')}` |"
        )
    lines.extend(["", "## Invariants", ""])
    inv = report.get("invariants") or {}
    lines.append(f"- Status: `{inv.get('status')}`")
    for failure in inv.get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    value_f = _to_float_or_none(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m12p_residual_profile``."""

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
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    report = run_m12p_residual_profile(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        previous_iteration_report=args.previous_iteration_report,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report.get("summary") or {}
    print(
        f"M12.P residual profile: status={report['status']} "
        f"candidate_p50={summary.get('candidate_p50_ms')} "
        f"remaining={summary.get('remaining_unattributed_e2e_residual_ms')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
