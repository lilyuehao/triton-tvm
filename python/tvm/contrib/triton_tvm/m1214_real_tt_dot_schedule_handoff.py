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
"""M12.14 real ``tt.dot`` schedule-handoff report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import validate_matmul_minimal_contract
from .errors import TritonTVMError
from .frontend import lower_to_ttir
from .matmul import (
    M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
    REAL_JIT_TT_DOT_SOURCE_KIND,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_SCHEDULE_ID,
    matmul_schedule_candidate_records,
)
from .translator import translate_ttir


REPORT_KIND = "triton_tvm_m12_14_real_tt_dot_schedule_handoff"
REPORT_ID = "m12_14_real_tt_dot_schedule_handoff_v1"
MILESTONE = "M12.14"
NEXT_DEFAULT_ACTION = "m12_15_fixed_shape_vit_p2_reentry"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_14_real_tt_dot_schedule_handoff"
)
DEFAULT_M1211_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_11_backend_general_replan/report.json"
)
DEFAULT_M1212_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_12_real_tl_dot_bridge_harness/report.json"
)
DEFAULT_M1213_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_13_runtime_overhead_general_measurement/report.json"
)
TENSORCORE_CASE = "real_jit_tl_dot_fp16_16x16x16_tensorcore"
TILED_CASE = "real_jit_tl_dot_fp16_8x8x16_tiled"


def run_real_tt_dot_schedule_handoff(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    m1211_report: str | Path = DEFAULT_M1211_REPORT,
    m1212_report: str | Path = DEFAULT_M1212_REPORT,
    m1213_report: str | Path = DEFAULT_M1213_REPORT,
) -> dict[str, Any]:
    """Lower real Triton JIT ``tl.dot`` cases and write the M12.14 report."""

    try:
        source_m1211 = _read_json_report(m1211_report)
        source_m1212 = _read_json_report(m1212_report)
        source_m1213 = _read_json_report(m1213_report)
    except FileNotFoundError as err:
        report = _empty_report(
            status="unavailable",
            availability_reason=f"source_report_missing:{err.filename}",
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            m1213_report=m1213_report,
        )
        _maybe_write_report(report, out_dir)
        return report
    except json.JSONDecodeError as err:
        report = _empty_report(
            status="failed",
            availability_reason=f"source_report_invalid_json:{err}",
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            m1213_report=m1213_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        cases = _collect_schedule_handoff_cases()
    except (ImportError, RuntimeError, ValueError, TritonTVMError) as err:
        report = _empty_report(
            status="unavailable",
            availability_reason=f"m12_14_unavailable:{type(err).__name__}:{err}",
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            m1213_report=m1213_report,
            source_m1211=source_m1211,
            source_m1212=source_m1212,
            source_m1213=source_m1213,
        )
        _maybe_write_report(report, out_dir)
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_report(
            status="failed",
            availability_reason=f"m12_14_failed:{type(err).__name__}:{err}",
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            m1213_report=m1213_report,
            source_m1211=source_m1211,
            source_m1212=source_m1212,
            source_m1213=source_m1213,
        )
        _maybe_write_report(report, out_dir)
        return report

    return build_real_tt_dot_schedule_handoff_report(
        m1211_report=source_m1211,
        m1211_report_path=m1211_report,
        m1212_report=source_m1212,
        m1212_report_path=m1212_report,
        m1213_report=source_m1213,
        m1213_report_path=m1213_report,
        schedule_handoff_cases=cases,
        out_dir=out_dir,
    )


def build_real_tt_dot_schedule_handoff_report(
    *,
    m1211_report: dict[str, Any],
    m1211_report_path: str | Path,
    m1212_report: dict[str, Any],
    m1212_report_path: str | Path,
    m1213_report: dict[str, Any],
    m1213_report_path: str | Path,
    schedule_handoff_cases: list[dict[str, Any]],
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build a machine-readable M12.14 schedule-handoff report."""

    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_reports": {
            "m12_11_report": str(m1211_report_path),
            "m12_12_report": str(m1212_report_path),
            "m12_13_report": str(m1213_report_path),
        },
        "source_report_status": _source_report_status(
            m1211_report, m1212_report, m1213_report
        ),
        "handoff_surface": {
            "name": "real_tt_dot_schedule_handoff",
            "source_kind": REAL_JIT_TT_DOT_SOURCE_KIND,
            "contract": "matmul_minimal",
            "requires_real_triton_jit_lowering": True,
            "uses_vit_wrapper": False,
            "uses_wrapper_buffer_replay": False,
            "uses_fused_qkv_artifact": False,
            "mutates_schedule_registry": False,
        },
        "schedule_handoff_cases": list(schedule_handoff_cases),
        "reusable_tt_dot_schedule_candidates": _reusable_schedule_candidates(
            schedule_handoff_cases
        ),
        "wrapper_specific_schedule_boundary": _wrapper_specific_schedule_boundary(
            schedule_handoff_cases
        ),
        "route_boundaries": _route_boundaries(),
        "next_default_action": {
            "action": NEXT_DEFAULT_ACTION,
            "reason": (
                "M12.14 completes the backend-general real tt.dot schedule handoff; "
                "the next default returns to the fixed-shape ViT P2 gate."
            ),
        },
        "real_tt_dot_schedule_handoff_complete": True,
        "performance_claim": False,
        "performance_ready_e2e": False,
        "p2_passed": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
        },
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _collect_schedule_handoff_cases() -> list[dict[str, Any]]:
    kernels = _real_tt_dot_kernels()
    return [
        _run_case(
            TENSORCORE_CASE,
            kernels["dot"],
            block_m=16,
            block_n=16,
            block_k=16,
            expected_schedule_id=TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
        ),
        _run_case(
            TILED_CASE,
            kernels["dot"],
            block_m=8,
            block_n=8,
            block_k=16,
            expected_schedule_id=TILED_TIR_MATMUL_SCHEDULE_ID,
        ),
    ]


def _run_case(
    case_name: str,
    jit_fn,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    expected_schedule_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_name": case_name,
        "status": "failed",
        "lower_to_ttir_ok": False,
        "ttir_contains_tt_dot": False,
        "matmul_source_kind": "",
        "matmul_contract_ok": False,
        "implementation_kind": "",
        "schedule_id": "",
        "selected_schedule_id": "",
        "expected_schedule_id": expected_schedule_id,
        "candidate_schedule_ids": [],
        "schedule_reject_reasons": {},
        "matmul_perf_envelope": "",
        "perf_guard_status": "",
        "shape": {"m": block_m, "n": block_n, "k": block_k},
        "observed_reason": "",
        "is_reusable_tt_dot_schedule": False,
        "uses_wrapper_specific_schedule": False,
        "counts_as_handoff_readiness": False,
    }
    try:
        artifact = lower_to_ttir(
            jit_fn,
            _signature(),
            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
        )
        result["lower_to_ttir_ok"] = True
        result["kernel_name"] = artifact.kernel_name
        result["triton_version"] = artifact.triton_version
        result["ttir_contains_tt_dot"] = "tt.dot" in artifact.ttir
        irmod, meta = translate_ttir(artifact, grid=(1,), contract="matmul_minimal")
        validate_matmul_minimal_contract(irmod)
        result.update(
            {
                "matmul_source_kind": meta.matmul_source_kind,
                "matmul_contract_ok": bool(meta.matmul_contract_ok),
                "implementation_kind": meta.implementation_kind,
                "schedule_id": meta.schedule_id,
                "selected_schedule_id": meta.selected_schedule_id,
                "candidate_schedule_ids": list(meta.candidate_schedule_ids),
                "schedule_reject_reasons": dict(meta.schedule_reject_reasons or {}),
                "matmul_perf_envelope": meta.matmul_perf_envelope,
                "perf_guard_status": meta.perf_guard_status,
                "tune_key": dict(meta.tune_key or {}),
                "observed_reason": meta.unsupported_matmul_reason,
            }
        )
        result["uses_wrapper_specific_schedule"] = (
            result["schedule_id"] == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
        )
        result["is_reusable_tt_dot_schedule"] = (
            result["matmul_source_kind"] == REAL_JIT_TT_DOT_SOURCE_KIND
            and result["implementation_kind"] == "native_tir_schedule"
            and result["schedule_id"] != M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
        )
        result["counts_as_handoff_readiness"] = bool(
            result["is_reusable_tt_dot_schedule"]
            and result["schedule_id"] == expected_schedule_id
            and result["matmul_contract_ok"] is True
        )
        result["status"] = "passed" if result["counts_as_handoff_readiness"] else "failed"
    except Exception as err:  # pylint: disable=broad-except
        result["error_type"] = type(err).__name__
        result["error_message"] = str(err)
        result["observed_reason"] = str(err)
    return result


def _real_tt_dot_kernels() -> dict[str, Any]:
    import triton  # pylint: disable=import-outside-toplevel
    import triton.language as tl  # pylint: disable=import-outside-toplevel

    @triton.jit
    def dot(a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
        bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
        acc = tl.dot(av, bv, input_precision="tf32")
        tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)

    return {"dot": dot}


def _signature() -> dict[str, str]:
    return {
        "a": "*fp16",
        "b": "*fp16",
        "out": "*fp32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def _source_report_status(
    m1211_report: dict[str, Any],
    m1212_report: dict[str, Any],
    m1213_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "m12_11_status": m1211_report.get("status"),
        "m12_11_invariants": (m1211_report.get("invariants") or {}).get("status"),
        "m12_11_primary_route": m1211_report.get("primary_route"),
        "m12_11_auxiliary_order": m1211_report.get("auxiliary_order"),
        "m12_12_status": m1212_report.get("status"),
        "m12_12_invariants": (m1212_report.get("invariants") or {}).get("status"),
        "m12_12_accepted_source_kind": (
            m1212_report.get("accepted_case") or {}
        ).get("matmul_source_kind"),
        "m12_13_status": m1213_report.get("status"),
        "m12_13_invariants": (m1213_report.get("invariants") or {}).get("status"),
        "m12_13_runtime_overhead_measurement_complete": (
            m1213_report.get("summary") or {}
        ).get("runtime_overhead_measurement_complete"),
    }


def _reusable_schedule_candidates(
    schedule_handoff_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {case.get("schedule_id") for case in schedule_handoff_cases}
    records = []
    for candidate in matmul_schedule_candidate_records():
        schedule_id = candidate["schedule_id"]
        wrapper_specific = schedule_id == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
        records.append(
            {
                "schedule_id": schedule_id,
                "package_id": candidate.get("package_id", ""),
                "builder_hook": candidate.get("builder_hook", ""),
                "applicability_predicate": candidate.get("applicability_predicate", ""),
                "reusable_for_real_tt_dot": not wrapper_specific,
                "selected_for_real_tt_dot": schedule_id in selected and not wrapper_specific,
                "counts_as_handoff_readiness": schedule_id in selected and not wrapper_specific,
                "wrapper_specific": wrapper_specific,
            }
        )
    return records


def _wrapper_specific_schedule_boundary(
    schedule_handoff_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = {case.get("schedule_id") for case in schedule_handoff_cases}
    return {
        "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
        "selected_for_real_tt_dot": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID in selected,
        "counts_as_handoff_readiness": False,
        "reason": (
            "M12.9 wrapper schedule is fixed-shape ViT wrapper-specific evidence; "
            "M12.14 readiness must come from real lowering-derived tt.dot semantics."
        ),
    }


def _route_boundaries() -> dict[str, Any]:
    return {
        "no_vit_wrapper_replay": True,
        "no_wrapper_buffer_replay": True,
        "no_wrapper_line_replay": True,
        "no_fused_qkv_artifact": True,
        "no_wrapper_specific_schedule_readiness": True,
        "no_p2_or_performance_claim": True,
        "no_backend_general_complete_claim": True,
        "schedule_registry_not_mutated": True,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("milestone") != MILESTONE:
        failures.append("m12_14_wrong_milestone")

    source_status = report.get("source_report_status") or {}
    if source_status.get("m12_11_status") != "route_frozen":
        failures.append("m12_14_m12_11_not_route_frozen")
    if source_status.get("m12_11_invariants") != "passed":
        failures.append("m12_14_m12_11_invariants_not_passed")
    if source_status.get("m12_11_primary_route") != "real_tl_dot_bridge":
        failures.append("m12_14_m12_11_primary_route_changed")
    if source_status.get("m12_11_auxiliary_order") != [
        "runtime_overhead",
        "tt_dot_schedule_handoff",
    ]:
        failures.append("m12_14_m12_11_auxiliary_order_changed")
    if source_status.get("m12_12_status") != "passed":
        failures.append("m12_14_m12_12_not_passed")
    if source_status.get("m12_12_invariants") != "passed":
        failures.append("m12_14_m12_12_invariants_not_passed")
    if source_status.get("m12_12_accepted_source_kind") != REAL_JIT_TT_DOT_SOURCE_KIND:
        failures.append("m12_14_m12_12_not_real_jit_tt_dot")
    if source_status.get("m12_13_status") != "passed":
        failures.append("m12_14_m12_13_not_passed")
    if source_status.get("m12_13_invariants") != "passed":
        failures.append("m12_14_m12_13_invariants_not_passed")

    surface = report.get("handoff_surface") or {}
    if surface.get("source_kind") != REAL_JIT_TT_DOT_SOURCE_KIND:
        failures.append("m12_14_handoff_surface_not_real_jit_tt_dot")
    if surface.get("contract") != "matmul_minimal":
        failures.append("m12_14_handoff_surface_not_matmul_minimal")
    for key in (
        "requires_real_triton_jit_lowering",
        "mutates_schedule_registry",
    ):
        expected = key != "mutates_schedule_registry"
        if surface.get(key) is not expected:
            failures.append(f"m12_14_handoff_surface_bad_{key}")
    for key in ("uses_vit_wrapper", "uses_wrapper_buffer_replay", "uses_fused_qkv_artifact"):
        if surface.get(key) is not False:
            failures.append(f"m12_14_handoff_surface_uses_{key}")

    cases = {case.get("case_name"): case for case in report.get("schedule_handoff_cases") or []}
    tensorcore = cases.get(TENSORCORE_CASE)
    tiled = cases.get(TILED_CASE)
    if not _case_ok(tensorcore, TENSORCORE_TIR_MATMUL_SCHEDULE_ID):
        failures.append("m12_14_tensorcore_handoff_missing")
    if not _case_ok(tiled, TILED_TIR_MATMUL_SCHEDULE_ID):
        failures.append("m12_14_tiled_handoff_missing")

    wrapper = report.get("wrapper_specific_schedule_boundary") or {}
    if wrapper.get("schedule_id") != M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
        failures.append("m12_14_wrapper_boundary_wrong_schedule")
    if wrapper.get("selected_for_real_tt_dot") is not False:
        failures.append("m12_14_wrapper_schedule_selected_for_real_tt_dot")
    if wrapper.get("counts_as_handoff_readiness") is not False:
        failures.append("m12_14_wrapper_schedule_counts_as_handoff")

    candidates = report.get("reusable_tt_dot_schedule_candidates") or []
    wrapper_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("schedule_id") == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
    ]
    if not wrapper_candidates:
        failures.append("m12_14_wrapper_schedule_boundary_missing")
    for candidate in wrapper_candidates:
        if candidate.get("counts_as_handoff_readiness") is not False:
            failures.append("m12_14_wrapper_candidate_counts_as_handoff")
        if candidate.get("selected_for_real_tt_dot") is not False:
            failures.append("m12_14_wrapper_candidate_selected")

    boundaries = report.get("route_boundaries") or {}
    for key in (
        "no_vit_wrapper_replay",
        "no_wrapper_buffer_replay",
        "no_wrapper_line_replay",
        "no_fused_qkv_artifact",
        "no_wrapper_specific_schedule_readiness",
        "no_p2_or_performance_claim",
        "no_backend_general_complete_claim",
        "schedule_registry_not_mutated",
    ):
        if boundaries.get(key) is not True:
            failures.append(f"m12_14_boundary_violation_{key}")

    if (report.get("next_default_action") or {}).get("action") != NEXT_DEFAULT_ACTION:
        failures.append("m12_14_next_action_not_m12_15_p2_reentry")
    if report.get("real_tt_dot_schedule_handoff_complete") is not True:
        failures.append("m12_14_completion_flag_missing")
    for key in (
        "performance_claim",
        "performance_ready_e2e",
        "p2_passed",
        "backend_general_complete",
        "strict_full_tvm_native",
    ):
        if report.get(key) is not False:
            failures.append(f"m12_14_forbidden_claim_{key}")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_14_full_tvm_runnable_claim_present")
    completion = report.get("completion_gate") or {}
    if completion.get("m12_complete") is not False:
        failures.append("m12_14_m12_complete_claim_present")
    if completion.get("requires_fixed_shape_vit_p2") is not True:
        failures.append("m12_14_p2_gate_not_preserved")

    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _case_ok(case: dict[str, Any] | None, expected_schedule_id: str) -> bool:
    if not case:
        return False
    return (
        case.get("status") == "passed"
        and case.get("lower_to_ttir_ok") is True
        and case.get("ttir_contains_tt_dot") is True
        and case.get("matmul_source_kind") == REAL_JIT_TT_DOT_SOURCE_KIND
        and case.get("matmul_contract_ok") is True
        and case.get("implementation_kind") == "native_tir_schedule"
        and case.get("schedule_id") == expected_schedule_id
        and case.get("selected_schedule_id") == expected_schedule_id
        and case.get("uses_wrapper_specific_schedule") is False
        and case.get("counts_as_handoff_readiness") is True
        and case.get("observed_reason", "") == ""
    )


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    cases = {case.get("case_name"): case for case in report.get("schedule_handoff_cases") or []}
    return {
        "m12_14_complete": report.get("status") == "passed",
        "real_tt_dot_schedule_handoff_complete": bool(
            report.get("status") == "passed"
            and (report.get("invariants") or {}).get("status") == "passed"
        ),
        "tensorcore_schedule_selected": (cases.get(TENSORCORE_CASE) or {}).get("schedule_id")
        == TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
        "wrapper_specific_schedule_excluded": (
            report.get("wrapper_specific_schedule_boundary") or {}
        ).get("counts_as_handoff_readiness")
        is False,
        "performance_claim": False,
        "m12_complete": False,
        "next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(
    *,
    status: str,
    availability_reason: str,
    m1211_report: str | Path,
    m1212_report: str | Path,
    m1213_report: str | Path,
    source_m1211: dict[str, Any] | None = None,
    source_m1212: dict[str, Any] | None = None,
    source_m1213: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_reports": {
            "m12_11_report": str(m1211_report),
            "m12_12_report": str(m1212_report),
            "m12_13_report": str(m1213_report),
        },
        "source_report_status": _source_report_status(
            source_m1211 or {}, source_m1212 or {}, source_m1213 or {}
        ),
        "handoff_surface": {
            "name": "real_tt_dot_schedule_handoff",
            "source_kind": REAL_JIT_TT_DOT_SOURCE_KIND,
            "contract": "matmul_minimal",
            "requires_real_triton_jit_lowering": True,
            "uses_vit_wrapper": False,
            "uses_wrapper_buffer_replay": False,
            "uses_fused_qkv_artifact": False,
            "mutates_schedule_registry": False,
        },
        "schedule_handoff_cases": [],
        "reusable_tt_dot_schedule_candidates": _reusable_schedule_candidates([]),
        "wrapper_specific_schedule_boundary": _wrapper_specific_schedule_boundary([]),
        "route_boundaries": _route_boundaries(),
        "next_default_action": {"action": NEXT_DEFAULT_ACTION},
        "real_tt_dot_schedule_handoff_complete": False,
        "performance_claim": False,
        "performance_ready_e2e": False,
        "p2_passed": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
        },
        "invariants": {"status": "not_run", "invariant_failures": []},
        "summary": {
            "m12_14_complete": False,
            "real_tt_dot_schedule_handoff_complete": False,
            "tensorcore_schedule_selected": False,
            "wrapper_specific_schedule_excluded": True,
            "performance_claim": False,
            "m12_complete": False,
            "next_action": NEXT_DEFAULT_ACTION,
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
    lines = [
        "# M12.14 Real `tt.dot` Schedule Handoff",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Surface: `{(report.get('handoff_surface') or {}).get('name')}`",
        f"- Source kind: `{(report.get('handoff_surface') or {}).get('source_kind')}`",
        f"- TensorCore selected: `{(report.get('summary') or {}).get('tensorcore_schedule_selected')}`",
        f"- Performance claim: `{report.get('performance_claim')}`",
        f"- P2 passed: `{report.get('p2_passed')}`",
        f"- Next action: `{(report.get('next_default_action') or {}).get('action')}`",
        "",
        "## Handoff Cases",
        "",
        "| Case | Status | Source | Schedule | Counts |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report.get("schedule_handoff_cases") or []:
        lines.append(
            f"| `{case.get('case_name')}` | `{case.get('status')}` | "
            f"`{case.get('matmul_source_kind')}` | `{case.get('schedule_id')}` | "
            f"`{case.get('counts_as_handoff_readiness')}` |"
        )
    lines.extend(["", "## Wrapper Boundary", ""])
    wrapper = report.get("wrapper_specific_schedule_boundary") or {}
    for key in ("schedule_id", "selected_for_real_tt_dot", "counts_as_handoff_readiness"):
        lines.append(f"- `{key}`: `{wrapper.get(key)}`")
    lines.extend(["", "## Invariants", ""])
    invariants = report.get("invariants") or {}
    lines.append(f"- Status: `{invariants.get('status')}`")
    for failure in invariants.get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m1214_real_tt_dot_schedule_handoff``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--m1211-report", type=Path, default=DEFAULT_M1211_REPORT)
    parser.add_argument("--m1212-report", type=Path, default=DEFAULT_M1212_REPORT)
    parser.add_argument("--m1213-report", type=Path, default=DEFAULT_M1213_REPORT)
    args = parser.parse_args(argv)

    report = run_real_tt_dot_schedule_handoff(
        out_dir=args.out_dir,
        m1211_report=args.m1211_report,
        m1212_report=args.m1212_report,
        m1213_report=args.m1213_report,
    )
    print(
        f"M12.14 real tt.dot schedule handoff: status={report['status']} "
        f"tensorcore_selected={(report.get('summary') or {}).get('tensorcore_schedule_selected')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
