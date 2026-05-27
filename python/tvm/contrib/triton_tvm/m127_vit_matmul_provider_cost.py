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
"""M12.7 fixed-shape ViT native matmul provider-cost recovery report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tvm

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import EXTERN_GEMM_PROVIDER_NATIVE_TVM
from .m123_vit_e2e_runner import (
    NATIVE_MATMUL_SYNC_POLICY_LEGACY,
    NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
    TARGET_MODEL,
    _correctness_report,
    _dependency_versions,
    _provider_mix_report,
    _require_torch,
    execute_vit_fixed_shape_provider_path,
    prepare_vit_fixed_shape_e2e,
)
from .m124_vit_e2e_dashboard import DEFAULT_REPEAT, DEFAULT_WARMUP
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_7_native_matmul_provider_cost"
REPORT_ID = "m12_7_native_matmul_provider_cost_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_7_native_matmul_provider_cost"
)
DEFAULT_HARDENING_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_6_hardening/report.json"
)
DEFAULT_FREEZE_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_5_5_perf_optimization_freeze/report.json"
)
MAX_PROFILE_REPEAT = 5
P2_THRESHOLD = 2.0

EXPECTED_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}
PROFILE_CATEGORIES = (
    "shape_abi_validation_ms",
    "b_storage_resolution_ms",
    "artifact_lookup_ms",
    "dlpack_conversion_ms",
    "artifact_dispatch_wall_ms",
    "kernel_event_ms",
    "dispatch_minus_kernel_estimate_ms",
    "pre_sync_wall_ms",
    "post_sync_wall_ms",
    "synchronization_wall_ms",
    "provider_total_wall_ms",
)
DOMINANT_COST_CATEGORIES = (
    "shape_abi_validation_ms",
    "b_storage_resolution_ms",
    "artifact_lookup_ms",
    "dlpack_conversion_ms",
    "artifact_dispatch_wall_ms",
    "kernel_event_ms",
    "synchronization_wall_ms",
)


def run_vit_native_matmul_provider_cost(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
    hardening_report: str | Path = DEFAULT_HARDENING_REPORT,
    freeze_report: str | Path = DEFAULT_FREEZE_REPORT,
) -> dict[str, Any]:
    """Run M12.7 profiling/recovery measurements and optionally write a report."""

    if not run_benchmarks:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
            hardening_report=hardening_report,
            freeze_report=freeze_report,
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
            hardening_report=hardening_report,
            freeze_report=freeze_report,
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
            hardening_report=hardening_report,
            freeze_report=freeze_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        hardening = _read_json_report(hardening_report)
        freeze = _read_json_report(freeze_report)
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        profile_repeat = max(1, min(int(repeat or 1), MAX_PROFILE_REPEAT))
        legacy_profile = _profile_provider_path(
            prepared,
            sync_policy=NATIVE_MATMUL_SYNC_POLICY_LEGACY,
            profile_repeat=profile_repeat,
        )
        optimized_profile = _profile_provider_path(
            prepared,
            sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            profile_repeat=profile_repeat,
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
            "triton_tvm_e2e_legacy_sync": _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_LEGACY,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            "triton_tvm_e2e_no_per_call_sync": _latency_record(
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
        }
        report = build_vit_native_matmul_provider_cost_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            profile_repeat=profile_repeat,
            hardening_report=hardening,
            hardening_report_path=hardening_report,
            freeze_report=freeze,
            freeze_report_path=freeze_report,
            latency_ms=latency_ms,
            legacy_correctness=legacy_profile["correctness"],
            legacy_provider_mix=legacy_profile["provider_mix"],
            legacy_profile_runs=legacy_profile["profile_runs"],
            optimized_correctness=optimized_profile["correctness"],
            optimized_provider_mix=optimized_profile["provider_mix"],
            optimized_profile_runs=optimized_profile["profile_runs"],
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
            availability_reason=f"m12_7_failed:{type(err).__name__}:{err}",
            hardening_report=hardening_report,
            freeze_report=freeze_report,
        )
        _maybe_write_report(report, out_dir)
        return report


def build_vit_native_matmul_provider_cost_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    profile_repeat: int,
    hardening_report: dict[str, Any],
    hardening_report_path: str | Path,
    freeze_report: dict[str, Any],
    freeze_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    legacy_correctness: dict[str, Any],
    legacy_provider_mix: dict[str, Any],
    legacy_profile_runs: list[list[dict[str, Any]]] | list[dict[str, Any]],
    optimized_correctness: dict[str, Any],
    optimized_provider_mix: dict[str, Any],
    optimized_profile_runs: list[list[dict[str, Any]]] | list[dict[str, Any]],
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build a full M12.7 report from measured or synthetic profiling records."""

    legacy_summary = _summarize_native_matmul_profile_runs(legacy_profile_runs)
    optimized_summary = _summarize_native_matmul_profile_runs(optimized_profile_runs)
    dominant = _dominant_cost_category(legacy_summary)
    optimization = _optimization_report(latency_ms, legacy_summary, optimized_summary)
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "profile_repeat": profile_repeat,
        "warmed_cache": warmup > 0,
        "source_reports": {
            "hardening_report": str(hardening_report_path),
            "freeze_report": str(freeze_report_path),
        },
        "source_report_status": {
            "m12_6_hardening_status": hardening_report.get("status"),
            "m12_6_hardening_invariants": (hardening_report.get("invariants") or {}).get(
                "status"
            ),
            "m12_5_5_freeze_status": freeze_report.get("status"),
        },
        "latency_ms": latency_ms,
        "p2_gap_recovery": _p2_gap_recovery(latency_ms, freeze_report),
        "correctness": {
            NATIVE_MATMUL_SYNC_POLICY_LEGACY: legacy_correctness,
            NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC: optimized_correctness,
        },
        "provider_mix": {
            NATIVE_MATMUL_SYNC_POLICY_LEGACY: legacy_provider_mix,
            NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC: optimized_provider_mix,
        },
        "native_matmul_profile": {
            "profile_repeat": profile_repeat,
            "policies": {
                NATIVE_MATMUL_SYNC_POLICY_LEGACY: legacy_summary,
                NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC: optimized_summary,
            },
        },
        "dominant_cost_category": dominant,
        "optimization": optimization,
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": 0,
        "next_action": "M12.8 P2 Gate Re-run and Close/Continue Decision",
        "notes": [
            "M12.7 profiles and validates native_tvm_matmul provider cost only.",
            "The official P2 promotion remains deferred to the M12.8 dashboard rerun.",
        ],
        "dependency_versions": dependency_versions or {"tvm": str(tvm.__version__)},
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    if report["status"] == "passed":
        report["optimization"]["enabled"] = True
        report["optimization"]["validated"] = True
    else:
        report["optimization"]["enabled"] = False
        report["optimization"]["validated"] = False
    _maybe_write_report(report, out_dir)
    return report


def _profile_provider_path(prepared, *, sync_policy: str, profile_repeat: int) -> dict[str, Any]:
    profile_runs: list[list[dict[str, Any]]] = []
    actual = None
    patch_state = None
    for _ in range(profile_repeat):
        actual, patch_state = execute_vit_fixed_shape_provider_path(
            prepared,
            profile_native_matmul=True,
            native_matmul_sync_policy=sync_policy,
        )
        profile_runs.append(patch_state.executed)
    assert actual is not None and patch_state is not None
    correctness = _correctness_report(actual, prepared.expected)
    provider_mix = _provider_mix_report(
        wrapper_source=prepared.wrapper_source,
        matmul_calls=prepared.matmul_calls,
        patch_state=patch_state,
    )
    return {
        "correctness": correctness,
        "provider_mix": provider_mix,
        "profile_runs": profile_runs,
    }


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
    return {
        "samples_ms": values,
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def _summarize_native_matmul_profile_runs(
    profile_runs: list[list[dict[str, Any]]] | list[dict[str, Any]],
) -> dict[str, Any]:
    runs = _normalize_profile_runs(profile_runs)
    aggregate_samples = {
        category: [_sum_profile_category(run, category) for run in runs]
        for category in PROFILE_CATEGORIES
    }
    aggregate = {category: _stat_record(values) for category, values in aggregate_samples.items()}
    per_call = []
    call_indexes = sorted(
        {
            int(record.get("index", -1))
            for run in runs
            for record in run
            if record.get("index") is not None
        }
    )
    for index in call_indexes:
        records = [record for run in runs for record in run if int(record.get("index", -1)) == index]
        first = records[0] if records else {}
        categories = {
            category: _stat_record(_profile_values(records, category))
            for category in PROFILE_CATEGORIES
        }
        per_call.append(
            {
                "index": index,
                "op_family": first.get("op_family"),
                "line_no": first.get("line_no"),
                "shape": [
                    first.get("matmul_m"),
                    first.get("matmul_n"),
                    first.get("matmul_k"),
                ],
                "provider_kind": first.get("provider_kind"),
                "schedule_id": first.get("schedule_id"),
                "categories": categories,
            }
        )
    record_count_per_run = [len(run) for run in runs]
    summary = {
        "run_count": len(runs),
        "record_count_per_run": record_count_per_run,
        "complete_call_count": bool(record_count_per_run and all(count == 7 for count in record_count_per_run)),
        "aggregate_categories": aggregate,
        "per_call": per_call,
    }
    summary["dominant_cost_category"] = _dominant_cost_category(summary)
    return summary


def _normalize_profile_runs(
    profile_runs: list[list[dict[str, Any]]] | list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not profile_runs:
        return []
    first = profile_runs[0]
    if isinstance(first, dict):
        return [profile_runs]  # type: ignore[list-item]
    return profile_runs  # type: ignore[return-value]


def _sum_profile_category(records: list[dict[str, Any]], category: str) -> float:
    values = _profile_values(records, category)
    return sum(values)


def _profile_values(records: list[dict[str, Any]], category: str) -> list[float]:
    values = []
    for record in records:
        profile = record.get("profile", {})
        if not isinstance(profile, dict):
            continue
        value = profile.get(category)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _stat_record(values: list[float]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "sum_ms": sum(values) if values else 0.0,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def _dominant_cost_category(profile_summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = profile_summary.get("aggregate_categories", {})
    candidates = []
    for category in DOMINANT_COST_CATEGORIES:
        p50 = _to_float_or_none((aggregate.get(category) or {}).get("p50_ms"))
        if p50 is not None:
            candidates.append((category, p50))
    if not candidates:
        return {"category": None, "aggregate_p50_ms": None}
    category, value = max(candidates, key=lambda item: item[1])
    return {"category": category, "aggregate_p50_ms": value}


def _optimization_report(
    latency_ms: dict[str, dict[str, Any]],
    legacy_summary: dict[str, Any],
    optimized_summary: dict[str, Any],
) -> dict[str, Any]:
    legacy_total = _to_float_or_none(
        legacy_summary.get("aggregate_categories", {})
        .get("provider_total_wall_ms", {})
        .get("p50_ms")
    )
    optimized_total = _to_float_or_none(
        optimized_summary.get("aggregate_categories", {})
        .get("provider_total_wall_ms", {})
        .get("p50_ms")
    )
    legacy_e2e = latency_ms.get("triton_tvm_e2e_legacy_sync", {})
    optimized_e2e = latency_ms.get("triton_tvm_e2e_no_per_call_sync", {})
    legacy_p50 = _to_float_or_none(legacy_e2e.get("p50_ms"))
    optimized_p50 = _to_float_or_none(optimized_e2e.get("p50_ms"))
    legacy_p95 = _to_float_or_none(legacy_e2e.get("p95_ms"))
    optimized_p95 = _to_float_or_none(optimized_e2e.get("p95_ms"))
    return {
        "target": "native_tvm_matmul_provider_cost",
        "change": "remove_redundant_native_matmul_per_call_sync_for_fixed_shape_path",
        "sync_policy_before": NATIVE_MATMUL_SYNC_POLICY_LEGACY,
        "sync_policy_after": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        "enabled": False,
        "validated": False,
        "provider_total_p50_delta_ms": _delta(legacy_total, optimized_total),
        "e2e_p50_delta_ms": _delta(legacy_p50, optimized_p50),
        "e2e_p95_delta_ms": _delta(legacy_p95, optimized_p95),
    }


def _p2_gap_recovery(
    latency_ms: dict[str, dict[str, Any]],
    freeze_report: dict[str, Any],
) -> dict[str, Any]:
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    legacy_latency = latency_ms.get("triton_tvm_e2e_legacy_sync", {})
    optimized_latency = latency_ms.get("triton_tvm_e2e_no_per_call_sync", {})
    compile_p50 = _to_float_or_none(compile_latency.get("p50_ms"))
    compile_p95 = _to_float_or_none(compile_latency.get("p95_ms"))
    legacy_p50 = _to_float_or_none(legacy_latency.get("p50_ms"))
    legacy_p95 = _to_float_or_none(legacy_latency.get("p95_ms"))
    optimized_p50 = _to_float_or_none(optimized_latency.get("p50_ms"))
    optimized_p95 = _to_float_or_none(optimized_latency.get("p95_ms"))
    p50_ceiling = compile_p50 * P2_THRESHOLD if compile_p50 is not None else None
    p95_ceiling = compile_p95 * P2_THRESHOLD if compile_p95 is not None else None
    return {
        "threshold": "triton_tvm_e2e p50/p95 <= 2x torch_compile_inductor",
        "source_frozen_gap": freeze_report.get("p2_gap", {}),
        "torch_compile_p50_ms": compile_p50,
        "torch_compile_p95_ms": compile_p95,
        "p50_ceiling_ms": p50_ceiling,
        "p95_ceiling_ms": p95_ceiling,
        "legacy_sync_p50_excess_ms": _positive_excess(legacy_p50, p50_ceiling),
        "legacy_sync_p95_excess_ms": _positive_excess(legacy_p95, p95_ceiling),
        "optimized_p50_excess_ms": _positive_excess(optimized_p50, p50_ceiling),
        "optimized_p95_excess_ms": _positive_excess(optimized_p95, p95_ceiling),
        "recovered_p50_ms": _positive_delta(
            _positive_excess(legacy_p50, p50_ceiling),
            _positive_excess(optimized_p50, p50_ceiling),
        ),
        "recovered_p95_ms": _positive_delta(
            _positive_excess(legacy_p95, p95_ceiling),
            _positive_excess(optimized_p95, p95_ceiling),
        ),
        "optimized_p2_estimate_passed": bool(
            optimized_p50 is not None
            and optimized_p95 is not None
            and p50_ceiling is not None
            and p95_ceiling is not None
            and optimized_p50 <= p50_ceiling
            and optimized_p95 <= p95_ceiling
        ),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    source_status = report.get("source_report_status", {})
    if source_status.get("m12_6_hardening_status") != "passed":
        failures.append("m12_7_hardening_report_not_passed")
    if source_status.get("m12_6_hardening_invariants") != "passed":
        failures.append("m12_7_hardening_invariants_not_passed")
    if report.get("performance_ready_e2e") is not False:
        failures.append("m12_7_performance_ready_set_before_m12_8")
    if report.get("performance_claim") is not False:
        failures.append("m12_7_performance_claim_set_before_m12_8")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_7_strict_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_7_full_tvm_runnable_claim_present")
    if int(report.get("host_staging_bytes", 0) or 0) != 0:
        failures.append("m12_7_host_staging_nonzero")
    for policy in (NATIVE_MATMUL_SYNC_POLICY_LEGACY, NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC):
        correctness = report.get("correctness", {}).get(policy, {})
        if correctness.get("allclose") is not True:
            failures.append(f"m12_7_correctness_failed_{policy}")
        provider_mix = report.get("provider_mix", {}).get(policy, {})
        if not _provider_mix_ok(provider_mix):
            failures.append(f"m12_7_provider_mix_mismatch_{policy}")
        profile = report.get("native_matmul_profile", {}).get("policies", {}).get(policy, {})
        if profile.get("complete_call_count") is not True:
            failures.append(f"m12_7_profile_call_count_mismatch_{policy}")
    dominant = report.get("dominant_cost_category", {})
    if dominant.get("category") not in DOMINANT_COST_CATEGORIES:
        failures.append("m12_7_dominant_cost_category_missing")
    for name in (
        "torch_compile_inductor",
        "triton_tvm_e2e_legacy_sync",
        "triton_tvm_e2e_no_per_call_sync",
    ):
        record = report.get("latency_ms", {}).get(name, {})
        if _non_positive_or_missing(record.get("p50_ms")) or _non_positive_or_missing(
            record.get("p95_ms")
        ):
            failures.append(f"m12_7_missing_latency_{name}")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


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


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    hardening_report: str | Path,
    freeze_report: str | Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "profile_repeat": 0,
        "warmed_cache": warmup > 0,
        "availability_reason": availability_reason,
        "source_reports": {
            "hardening_report": str(hardening_report),
            "freeze_report": str(freeze_report),
        },
        "source_report_status": {},
        "latency_ms": {
            "torch_compile_inductor": _latency_record([]),
            "triton_tvm_e2e_legacy_sync": _latency_record([]),
            "triton_tvm_e2e_no_per_call_sync": _latency_record([]),
        },
        "p2_gap_recovery": {},
        "correctness": {},
        "provider_mix": {},
        "native_matmul_profile": {"profile_repeat": 0, "policies": {}},
        "dominant_cost_category": {"category": None, "aggregate_p50_ms": None},
        "optimization": {
            "target": "native_tvm_matmul_provider_cost",
            "change": "remove_redundant_native_matmul_per_call_sync_for_fixed_shape_path",
            "sync_policy_before": NATIVE_MATMUL_SYNC_POLICY_LEGACY,
            "sync_policy_after": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            "enabled": False,
            "validated": False,
        },
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": 0,
        "next_action": "M12.8 P2 Gate Re-run and Close/Continue Decision",
        "dependency_versions": {"tvm": str(tvm.__version__)},
    }
    report["invariants"] = {
        "status": "failed",
        "invariant_failures": [f"m12_7_{availability_reason}"],
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
    latency = report.get("latency_ms", {})
    gap = report.get("p2_gap_recovery", {})
    dominant = report.get("dominant_cost_category", {})
    optimization = report.get("optimization", {})
    lines = [
        "# M12.7 Native Matmul Provider Cost",
        "",
        f"- Status: {report.get('status')}",
        f"- Target model: `{report.get('target_model')}`",
        f"- Warmup/repeat/profile_repeat: "
        f"{report.get('warmup')} / {report.get('repeat')} / {report.get('profile_repeat')}",
        f"- Dominant legacy category: `{dominant.get('category')}` "
        f"({_fmt(dominant.get('aggregate_p50_ms'))} ms aggregate p50)",
        f"- Optimization validated: {optimization.get('validated')}",
        f"- Performance-ready E2E: {report.get('performance_ready_e2e')}",
        f"- Host staging bytes: {report.get('host_staging_bytes')}",
        f"- Next action: {report.get('next_action')}",
        "",
        "## E2E Timing",
        "",
        "| Path | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for name in (
        "torch_compile_inductor",
        "triton_tvm_e2e_legacy_sync",
        "triton_tvm_e2e_no_per_call_sync",
    ):
        record = latency.get(name, {})
        lines.append(
            f"| `{name}` | {_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |"
        )
    lines.extend(
        [
            "",
            "## P2 Gap Recovery Estimate",
            "",
            f"- Optimized P2 estimate passed: {gap.get('optimized_p2_estimate_passed')}",
            f"- Recovered p50/p95 ms: "
            f"{_fmt(gap.get('recovered_p50_ms'))}, {_fmt(gap.get('recovered_p95_ms'))}",
            "",
            "## Native Matmul Profile",
            "",
            "| Policy | Provider total p50 ms | Sync p50 ms | Kernel p50 ms |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    policies = report.get("native_matmul_profile", {}).get("policies", {})
    for policy in (NATIVE_MATMUL_SYNC_POLICY_LEGACY, NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC):
        aggregate = policies.get(policy, {}).get("aggregate_categories", {})
        lines.append(
            f"| `{policy}` | "
            f"{_fmt((aggregate.get('provider_total_wall_ms') or {}).get('p50_ms'))} | "
            f"{_fmt((aggregate.get('synchronization_wall_ms') or {}).get('p50_ms'))} | "
            f"{_fmt((aggregate.get('kernel_event_ms') or {}).get('p50_ms'))} |"
        )
    failures = ", ".join(report.get("invariants", {}).get("invariant_failures", [])) or "none"
    lines.extend(["", "## Invariants", "", f"- Failures: {failures}", ""])
    return "\n".join(lines)


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _non_positive_or_missing(value: Any) -> bool:
    converted = _to_float_or_none(value)
    return converted is None or converted <= 0.0


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return before - after


def _positive_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return max(0.0, before - after)


def _positive_excess(value: float | None, ceiling: float | None) -> float | None:
    if value is None or ceiling is None:
        return None
    return max(0.0, value - ceiling)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardening-report", type=Path, default=DEFAULT_HARDENING_REPORT)
    parser.add_argument("--freeze-report", type=Path, default=DEFAULT_FREEZE_REPORT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    report = run_vit_native_matmul_provider_cost(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        run_benchmarks=not args.no_benchmarks,
        hardening_report=args.hardening_report,
        freeze_report=args.freeze_report,
    )
    print(
        "M12.7 native matmul provider cost: "
        f"status={report['status']} "
        f"dominant={report.get('dominant_cost_category', {}).get('category')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
