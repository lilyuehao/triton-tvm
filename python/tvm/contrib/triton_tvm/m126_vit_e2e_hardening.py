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
"""M12.6 fixed-shape ViT E2E surface hardening report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import (
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
)
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_6_vit_e2e_surface_hardening"
HARDENING_ID = "m12_6_vit_e2e_surface_hardening_v1"
CORPUS_DIFF_BASELINE_ID = "m12_6_vit_fixed_shape_surface_baseline_v1"
DEFAULT_CORPUS_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_2_vit_native_matmul_closure_corpus/report.json"
)
DEFAULT_DASHBOARD_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_4_vit_e2e_dashboard/report.json"
)
DEFAULT_REVIEW_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_5_optimization_review/report.json"
)
DEFAULT_FREEZE_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_5_5_perf_optimization_freeze/report.json"
)
DEFAULT_OUT_DIR = Path("/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_6_hardening")

EXPECTED_BASELINES = ("torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e")
EXPECTED_DASHBOARD_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}
EXPECTED_PROVIDER_IDS = frozenset(
    {
        "torch_inductor_triton_captured_harness",
        "native_tvm_tirx",
        VISION_PROVIDER_DEVICE_TORCH_CUDA,
        ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
        EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    }
)
KNOWN_EXTERN_GEMM_RUNTIME_PROVIDER_IDS = (
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
)
REASON_FIELDS = (
    "fallback_reason",
    "unsupported_reason",
    "unsupported_matmul_reason",
    "unsupported_attention_runtime_reason",
    "unsupported_vision_runtime_reason",
    "unsupported_m11_grid_runtime_reason",
)


def run_vit_e2e_hardening(
    *,
    corpus_report: str | Path = DEFAULT_CORPUS_REPORT,
    dashboard_report: str | Path = DEFAULT_DASHBOARD_REPORT,
    review_report: str | Path = DEFAULT_REVIEW_REPORT,
    freeze_report: str | Path = DEFAULT_FREEZE_REPORT,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Validate the M12 E2E/report/provider surface and write a hardening report."""

    corpus_path = Path(corpus_report)
    dashboard_path = Path(dashboard_report)
    review_path = Path(review_report)
    freeze_path = Path(freeze_report)
    corpus = _read_json(corpus_path)
    dashboard = _read_json(dashboard_path)
    review = _read_json(review_path)
    freeze = _read_json(freeze_path)

    checks = {
        "provider_id_validation": _provider_id_validation(corpus, dashboard),
        "dashboard_schema_guards": _dashboard_schema_guards(dashboard),
        "freeze_schema_guards": _freeze_schema_guards(freeze),
        "report_cache_invariants": _report_cache_invariants(
            corpus,
            dashboard,
            review,
            freeze,
        ),
        "no_stale_fallback_reasons": _no_stale_fallback_reasons(corpus, dashboard),
        "no_hidden_host_staging": _no_hidden_host_staging(corpus, dashboard, freeze),
        "no_accidental_strict_native_claim": _no_accidental_strict_native_claim(
            corpus,
            dashboard,
            freeze,
        ),
        "performance_ready_gate": _performance_ready_gate(dashboard, review, freeze),
        "corpus_diff_guard": _corpus_diff_guard(corpus),
    }
    failures = sorted(
        {
            failure
            for check in checks.values()
            for failure in check.get("invariant_failures", [])
        }
    )
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "hardening_id": HARDENING_ID,
        "status": "passed" if not failures else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": "vit_tiny_random",
        "source_reports": {
            "corpus_report": str(corpus_path),
            "dashboard_report": str(dashboard_path),
            "review_report": str(review_path),
            "freeze_report": str(freeze_path),
        },
        "performance_optimization_introduced": False,
        "provider_id_policy": {
            "known_extern_gemm_runtime_provider_ids": list(
                KNOWN_EXTERN_GEMM_RUNTIME_PROVIDER_IDS
            ),
            "allowed_m12_e2e_provider_ids": sorted(EXPECTED_PROVIDER_IDS),
        },
        "checks": checks,
        "performance_ready_e2e": bool(dashboard.get("performance_ready_e2e")),
        "p1_passed": bool(dashboard.get("p1", {}).get("passed")),
        "p2_passed": bool(dashboard.get("p2", {}).get("passed")),
        "strict_full_tvm_native": bool(dashboard.get("strict_full_tvm_native")),
        "full_tvm_runnable_models": int(dashboard.get("full_tvm_runnable_models", 0) or 0),
        "host_staging_bytes": int(dashboard.get("host_staging_bytes", 0) or 0),
        "primary_next_target": "native_tvm_matmul_provider_cost",
        "next_action": "M12.7 P2 Recovery - Native Matmul Provider Cost",
    }
    report["invariants"] = {
        "status": "passed" if not failures else "failed",
        "invariant_failures": failures,
    }
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_path / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _provider_id_validation(corpus: dict[str, Any], dashboard: dict[str, Any]) -> dict[str, Any]:
    observed = sorted(_observed_provider_ids(corpus, dashboard))
    unknown = sorted(provider for provider in observed if provider not in EXPECTED_PROVIDER_IDS)
    failures = ["m12_6_provider_id_not_allowed"] if unknown else []
    provider_counts = dashboard.get("provider_mix", {}).get("provider_counts", {})
    normalized_counts = {
        str(provider): int(count or 0) for provider, count in provider_counts.items()
    }
    if normalized_counts != EXPECTED_DASHBOARD_PROVIDER_COUNTS:
        failures.append("m12_6_provider_counts_mismatch")
    return {
        "status": "passed" if not failures else "failed",
        "observed_provider_ids": observed,
        "expected_provider_counts": dict(sorted(EXPECTED_DASHBOARD_PROVIDER_COUNTS.items())),
        "observed_provider_counts": dict(sorted(normalized_counts.items())),
        "unknown_provider_ids": unknown,
        "invariant_failures": sorted(set(failures)),
    }


def _observed_provider_ids(corpus: dict[str, Any], dashboard: dict[str, Any]) -> set[str]:
    providers: set[str] = set()
    provider_counts = dashboard.get("provider_mix", {}).get("provider_counts", {})
    providers.update(str(provider) for provider in provider_counts if str(provider))
    for section in ("wrapper_convolution", "wrapper_attention", "wrapper_matmul"):
        provider = str(dashboard.get("provider_mix", {}).get(section, {}).get("provider_kind", ""))
        if provider:
            providers.add(provider)
    m12 = corpus.get("m12", {})
    closure = m12.get("m12_2_native_matmul_closure", {})
    providers.update(str(provider) for provider in (closure.get("provider_counts") or {}) if str(provider))
    provider = str(closure.get("provider_kind", ""))
    if provider:
        providers.add(provider)
    for record in closure.get("records", []) or []:
        provider = str(record.get("provider_kind", ""))
        if provider:
            providers.add(provider)
    for node in m12.get("vit_fixed_shape_execution_plan", {}).get("nodes", []) or []:
        provider = str(node.get("provider_kind", ""))
        if provider:
            providers.add(provider)
    return providers


def _dashboard_schema_guards(dashboard: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    _require_fields(
        dashboard,
        (
            "report_kind",
            "schema_version",
            "dashboard_id",
            "status",
            "target_model",
            "fixed_shape",
            "warmup",
            "repeat",
            "warmed_cache",
            "baselines",
            "latency_ms",
            "ratios",
            "correctness",
            "p0",
            "p1",
            "p2",
            "m12_performance_tier",
            "performance_ready_e2e",
            "performance_claim",
            "strict_full_tvm_native",
            "full_tvm_runnable_models",
            "host_staging_bytes",
            "launch_count_estimate",
            "provider_mix",
            "device_bytes",
            "summary",
            "invariants",
        ),
        failures,
        prefix="dashboard",
    )
    if dashboard.get("report_kind") != "triton_tvm_m12_4_vit_fixed_shape_e2e_dashboard":
        failures.append("m12_6_dashboard_report_kind_mismatch")
    if int(dashboard.get("schema_version", 0) or 0) != 1:
        failures.append("m12_6_dashboard_schema_version_mismatch")
    if dashboard.get("target_model") != "vit_tiny_random" or dashboard.get("fixed_shape") is not True:
        failures.append("m12_6_dashboard_target_or_shape_mismatch")
    if tuple(dashboard.get("baselines", []) or []) != EXPECTED_BASELINES:
        failures.append("m12_6_dashboard_baselines_mismatch")

    latency = dashboard.get("latency_ms", {})
    measured = dashboard.get("status") == "measured"
    for baseline in EXPECTED_BASELINES:
        record = latency.get(baseline)
        if not isinstance(record, dict):
            failures.append("m12_6_dashboard_latency_baseline_missing")
            continue
        _require_fields(record, ("p50_ms", "p95_ms", "samples_ms"), failures, prefix=baseline)
        if measured and (
            _non_positive_or_missing(record.get("p50_ms"))
            or _non_positive_or_missing(record.get("p95_ms"))
        ):
            failures.append("m12_6_dashboard_measured_latency_missing")

    _require_fields(
        dashboard.get("ratios", {}),
        (
            "triton_tvm_to_torch_eager_p50",
            "triton_tvm_to_torch_eager_p95",
            "triton_tvm_to_torch_compile_p50",
            "triton_tvm_to_torch_compile_p95",
        ),
        failures,
        prefix="ratios",
    )
    _require_fields(
        dashboard.get("p0", {}),
        ("correctness", "no_silent_fallback", "complete_provider_report", "passed"),
        failures,
        prefix="p0",
    )
    _require_fields(
        dashboard.get("p1", {}),
        ("threshold", "eager_baseline_ok", "inductor_baseline_ok", "zero_host_staging", "passed"),
        failures,
        prefix="p1",
    )
    _require_fields(
        dashboard.get("p2", {}),
        (
            "threshold",
            "inductor_baseline_ok",
            "warmed_cache",
            "zero_host_staging",
            "complete_provider_mix",
            "correctness",
            "passed",
        ),
        failures,
        prefix="p2",
    )
    if str(dashboard.get("p2", {}).get("threshold", "")) != "<=2x torch_compile_inductor p50/p95":
        failures.append("m12_6_dashboard_p2_threshold_changed")
    if (dashboard.get("invariants") or {}).get("status") != "passed":
        failures.append("m12_6_dashboard_embedded_invariants_failed")
    return _check_result(failures)


def _freeze_schema_guards(freeze: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    _require_fields(
        freeze,
        (
            "report_kind",
            "schema_version",
            "freeze_id",
            "status",
            "current_gate_state",
            "p2_gap",
            "primary_target",
            "frozen_targets",
            "acceptance_criteria",
            "out_of_scope",
            "p2_recovery_sequence",
            "invariants",
        ),
        failures,
        prefix="freeze",
    )
    if freeze.get("report_kind") != "triton_tvm_m12_5_5_vit_perf_optimization_freeze":
        failures.append("m12_6_freeze_report_kind_mismatch")
    if int(freeze.get("schema_version", 0) or 0) != 1:
        failures.append("m12_6_freeze_schema_version_mismatch")
    if freeze.get("primary_target") != "native_tvm_matmul_provider_cost":
        failures.append("m12_6_freeze_primary_target_changed")
    _require_fields(
        freeze.get("p2_gap", {}),
        (
            "threshold",
            "torch_compile_p50_ms",
            "torch_compile_p95_ms",
            "triton_tvm_p50_ms",
            "triton_tvm_p95_ms",
            "p50_ceiling_ms",
            "p95_ceiling_ms",
            "p50_excess_ms",
            "p95_excess_ms",
        ),
        failures,
        prefix="p2_gap",
    )
    if str(freeze.get("p2_gap", {}).get("threshold", "")) != (
        "triton_tvm_e2e p50/p95 <= 2x torch_compile_inductor"
    ):
        failures.append("m12_6_freeze_p2_threshold_changed")
    sequence = freeze.get("p2_recovery_sequence", []) or []
    m126 = next((item for item in sequence if item.get("slice") == "M12.6"), {})
    m127 = next((item for item in sequence if item.get("slice") == "M12.7"), {})
    if m126.get("performance_optimization") is not False:
        failures.append("m12_6_freeze_m126_not_marked_non_optimizing")
    if m127.get("primary_target") != "native_tvm_matmul_provider_cost":
        failures.append("m12_6_freeze_m127_primary_target_changed")
    if (freeze.get("invariants") or {}).get("status") != "passed":
        failures.append("m12_6_freeze_embedded_invariants_failed")
    return _check_result(failures)


def _report_cache_invariants(
    corpus: dict[str, Any],
    dashboard: dict[str, Any],
    review: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    m12 = corpus.get("m12", {})
    plan = m12.get("vit_fixed_shape_execution_plan", {})
    closure = m12.get("m12_2_native_matmul_closure", {})
    if m12.get("status") != "m12_2_native_vit_matmul_closed":
        failures.append("m12_6_corpus_m12_status_mismatch")
    if plan.get("captured_kernel_count") != 7 or plan.get("translated_captured_kernel_count") != 7:
        failures.append("m12_6_vit_captured_kernel_count_mismatch")
    if plan.get("wrapper_call_count") != 9 or plan.get("wrapper_matmul_call_count") != 7:
        failures.append("m12_6_vit_wrapper_count_mismatch")
    if plan.get("artifact_only_matmul_blockers") != 0:
        failures.append("m12_6_vit_artifact_only_matmul_blocker_present")
    if plan.get("native_runtime_resolved_matmul_count") != 7:
        failures.append("m12_6_vit_native_matmul_count_mismatch")
    if plan.get("no_hidden_fallback") is not True:
        failures.append("m12_6_vit_hidden_fallback_present")
    if closure.get("status") != "passed":
        failures.append("m12_6_native_matmul_closure_not_passed")
    if closure.get("runtime_resolved_total") != 7:
        failures.append("m12_6_native_matmul_runtime_count_mismatch")
    if closure.get("runtime_resolved_extern_gemm") != 4:
        failures.append("m12_6_native_matmul_gemm_count_mismatch")
    if closure.get("runtime_resolved_extern_addmm_bias") != 3:
        failures.append("m12_6_native_matmul_addmm_count_mismatch")
    if closure.get("host_staging_bytes") != 0:
        failures.append("m12_6_native_matmul_host_staging_nonzero")
    if closure.get("performance_claim") is not False or closure.get("performance_claim_count") != 0:
        failures.append("m12_6_native_matmul_performance_claim_present")
    if closure.get("provider_kind") != EXTERN_GEMM_PROVIDER_NATIVE_TVM:
        failures.append("m12_6_native_matmul_provider_mismatch")
    if closure.get("runtime_claim") != EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE:
        failures.append("m12_6_native_matmul_runtime_claim_mismatch")
    if closure.get("schedule_id") != NATIVE_TIR_MATMUL_SCHEDULE_ID:
        failures.append("m12_6_native_matmul_schedule_changed")
    if dashboard.get("status") != "measured":
        failures.append("m12_6_dashboard_not_measured")
    if dashboard.get("m12_performance_tier") != "P1":
        failures.append("m12_6_dashboard_tier_not_p1")
    if review.get("disposition") != "open_m12_7_p2_recovery":
        failures.append("m12_6_review_disposition_changed")
    if review.get("primary_target") != "native_tvm_matmul_provider_cost":
        failures.append("m12_6_review_primary_target_changed")
    if freeze.get("primary_target") != "native_tvm_matmul_provider_cost":
        failures.append("m12_6_freeze_primary_target_changed")
    return _check_result(failures)


def _no_stale_fallback_reasons(corpus: dict[str, Any], dashboard: dict[str, Any]) -> dict[str, Any]:
    stale = []
    m12 = corpus.get("m12", {})
    for index, record in enumerate(
        m12.get("m12_2_native_matmul_closure", {}).get("records", []) or []
    ):
        if record.get("runtime_status") == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED:
            if _has_reason_field(record):
                stale.append(f"closure.records[{index}]")
    for index, node in enumerate(m12.get("vit_fixed_shape_execution_plan", {}).get("nodes", []) or []):
        runtime_status = str(node.get("runtime_status", ""))
        translated = bool(node.get("translate_ok", False))
        if (runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED or translated) and (
            _has_reason_field(node)
        ):
            stale.append(f"plan.nodes[{index}]")
    for section_name in ("wrapper_convolution", "wrapper_attention", "wrapper_matmul"):
        section = dashboard.get("provider_mix", {}).get(section_name, {})
        if section.get("runtime_status") == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED:
            if _has_reason_field(section):
                stale.append(f"dashboard.provider_mix.{section_name}")
    failures = ["m12_6_runtime_resolved_stale_fallback_reason"] if stale else []
    return {
        "status": "passed" if not failures else "failed",
        "stale_reason_locations": stale,
        "invariant_failures": failures,
    }


def _no_hidden_host_staging(
    corpus: dict[str, Any],
    dashboard: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    nonzero_locations = []
    for location, value in _host_staging_values(
        {
            "corpus_m12": corpus.get("m12", {}),
            "dashboard": dashboard,
            "freeze_gate": freeze.get("current_gate_state", {}),
        }
    ):
        if int(value or 0) != 0:
            nonzero_locations.append(location)
    closure_records = corpus.get("m12", {}).get("m12_2_native_matmul_closure", {}).get("records", [])
    if any(bool(record.get("uses_host_staging", False)) for record in closure_records or []):
        failures.append("m12_6_native_matmul_uses_host_staging")
    if nonzero_locations:
        failures.append("m12_6_host_staging_nonzero")
    return {
        "status": "passed" if not failures else "failed",
        "nonzero_host_staging_locations": nonzero_locations,
        "invariant_failures": sorted(set(failures)),
    }


def _no_accidental_strict_native_claim(
    corpus: dict[str, Any],
    dashboard: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    policy = corpus.get("m12", {}).get("policy", {})
    gate = freeze.get("current_gate_state", {})
    for name, value in (
        ("corpus_policy", policy.get("strict_full_tvm_native")),
        ("dashboard", dashboard.get("strict_full_tvm_native")),
        ("freeze_gate", gate.get("strict_full_tvm_native")),
    ):
        if value is not False:
            failures.append(f"m12_6_strict_native_claim_present_{name}")
    for name, value in (
        ("corpus_policy", policy.get("full_tvm_runnable_models")),
        ("dashboard", dashboard.get("full_tvm_runnable_models")),
        ("freeze_gate", gate.get("full_tvm_runnable_models")),
    ):
        if int(value or 0) != 0:
            failures.append(f"m12_6_full_tvm_runnable_claim_present_{name}")
    return _check_result(failures)


def _performance_ready_gate(
    dashboard: dict[str, Any],
    review: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    p2_passed = bool(dashboard.get("p2", {}).get("passed"))
    if bool(dashboard.get("performance_ready_e2e")) and not p2_passed:
        failures.append("m12_6_dashboard_performance_ready_without_p2")
    if bool(review.get("performance_ready_e2e")) and not bool(review.get("p2_passed")):
        failures.append("m12_6_review_performance_ready_without_p2")
    gate = freeze.get("current_gate_state", {})
    if bool(gate.get("performance_ready_e2e")) and not bool(gate.get("p2_passed")):
        failures.append("m12_6_freeze_performance_ready_without_p2")
    if bool(dashboard.get("performance_claim")) != p2_passed:
        failures.append("m12_6_dashboard_performance_claim_not_tied_to_p2")
    if bool(freeze.get("current_gate_state", {}).get("p2_passed")):
        failures.append("m12_6_expected_pre_p2_freeze_state_changed")
    return _check_result(failures)


def _corpus_diff_guard(corpus: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    summary = corpus.get("summary", {})
    model_summary = corpus.get("model_summary", {})
    pre_m12 = corpus.get("pre_m12", {})
    residual = pre_m12.get("residual_debt_counts", {})
    status_buckets = summary.get("status_buckets", {})
    if int(summary.get("total_kernels", 0) or 0) != 89:
        failures.append("m12_6_corpus_total_kernel_count_changed")
    if int(summary.get("translated_kernels", 0) or 0) != 72:
        failures.append("m12_6_corpus_translated_kernel_count_changed")
    if status_buckets != {"contract_error": 17, "translated": 72}:
        failures.append("m12_6_corpus_status_buckets_changed")
    if int(model_summary.get("fallback_kernels", 0) or 0) != 17:
        failures.append("m12_6_corpus_fallback_kernel_count_changed")
    if int(model_summary.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_6_corpus_full_tvm_runnable_claim_present")
    expected_residual = {
        "captured_grid_concat_split": 17,
        "wrapper_extern_gemm_artifact_only": 7,
        "wrapper_extern_addmm_bias_artifact_only": 0,
        "runtime_resolved_attention": 2,
        "runtime_resolved_convolution": 65,
    }
    for key, expected in expected_residual.items():
        if int(residual.get(key, 0) or 0) != expected:
            failures.append("m12_6_pre_m12_residual_debt_changed")
            break
    return {
        "status": "passed" if not failures else "failed",
        "baseline_id": CORPUS_DIFF_BASELINE_ID,
        "captured_kernel_count": int(summary.get("total_kernels", 0) or 0),
        "translated_kernel_count": int(summary.get("translated_kernels", 0) or 0),
        "status_buckets": dict(sorted(status_buckets.items())),
        "invariant_failures": sorted(set(failures)),
    }


def _has_reason_field(record: dict[str, Any]) -> bool:
    return any(str(record.get(field, "") or "").strip() for field in REASON_FIELDS)


def _host_staging_values(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if key.endswith("host_staging_bytes") or key == "host_staging_bytes":
                yield location, value
            yield from _host_staging_values(value, location)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _host_staging_values(value, f"{prefix}[{index}]")


def _require_fields(
    payload: dict[str, Any],
    fields: Iterable[str],
    failures: list[str],
    *,
    prefix: str,
) -> None:
    for field in fields:
        if field not in payload:
            failures.append(f"m12_6_missing_{prefix}_{field}")


def _non_positive_or_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return float(value) <= 0.0
    except (TypeError, ValueError):
        return True


def _check_result(failures: list[str]) -> dict[str, Any]:
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M12.6 E2E Surface Hardening",
        "",
        f"- Status: {report['status']}",
        f"- Target model: `{report['target_model']}`",
        f"- P1/P2 passed: {report['p1_passed']} / {report['p2_passed']}",
        f"- Performance-ready E2E: {report['performance_ready_e2e']}",
        f"- Host staging bytes: {report['host_staging_bytes']}",
        f"- Strict full TVM native: {report['strict_full_tvm_native']}",
        f"- Full TVM runnable models: {report['full_tvm_runnable_models']}",
        f"- Next action: {report['next_action']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Failures |",
        "| --- | --- | --- |",
    ]
    for name, check in report["checks"].items():
        failures = ", ".join(check.get("invariant_failures", [])) or "none"
        lines.append(f"| `{name}` | {check.get('status')} | {failures} |")
    lines.extend(
        [
            "",
            "## Provider Ids",
            "",
            "- Allowed M12 E2E provider ids: "
            + ", ".join(f"`{item}`" for item in report["provider_id_policy"]["allowed_m12_e2e_provider_ids"]),
            "- Known extern GEMM runtime provider ids: "
            + ", ".join(
                f"`{item}`"
                for item in report["provider_id_policy"][
                    "known_extern_gemm_runtime_provider_ids"
                ]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-report", type=Path, default=DEFAULT_CORPUS_REPORT)
    parser.add_argument("--dashboard-report", type=Path, default=DEFAULT_DASHBOARD_REPORT)
    parser.add_argument("--review-report", type=Path, default=DEFAULT_REVIEW_REPORT)
    parser.add_argument("--freeze-report", type=Path, default=DEFAULT_FREEZE_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    report = run_vit_e2e_hardening(
        corpus_report=args.corpus_report,
        dashboard_report=args.dashboard_report,
        review_report=args.review_report,
        freeze_report=args.freeze_report,
        out_dir=args.out_dir,
    )
    print(
        "M12.6 E2E hardening: "
        f"status={report['status']} next={report['next_action']} out_dir={args.out_dir}"
    )
    return 0 if report["invariants"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
