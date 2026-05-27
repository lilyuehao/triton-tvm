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
"""M12.11 backend-general route-freeze report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m123_vit_e2e_runner import TARGET_MODEL
from .m129_vit_provider_runtime_optimization import BACKEND_PASS_MODE, FUSED_QKV_MODE


REPORT_KIND = "triton_tvm_m12_11_backend_general_replan"
REPORT_ID = "m12_11_backend_general_replan_v1"
MILESTONE = "M12.11"
PRIMARY_ROUTE = "real_tl_dot_bridge"
AUXILIARY_ORDER = ["runtime_overhead", "tt_dot_schedule_handoff"]
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_11_backend_general_replan"
)
DEFAULT_M1210_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_10_route_cleanup/report.json"
)


def run_backend_general_replan(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    m1210_report: str | Path = DEFAULT_M1210_REPORT,
) -> dict[str, Any]:
    """Read the M12.10 report and write the M12.11 route-freeze report."""

    try:
        source = _read_json_report(m1210_report)
    except FileNotFoundError:
        report = _empty_report(
            status="unavailable",
            availability_reason="m12_10_report_missing",
            m1210_report=m1210_report,
        )
        _maybe_write_report(report, out_dir)
        return report
    except json.JSONDecodeError as err:
        report = _empty_report(
            status="failed",
            availability_reason=f"m12_10_report_invalid_json:{err}",
            m1210_report=m1210_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    return build_backend_general_replan_report(
        m1210_report=source,
        m1210_report_path=m1210_report,
        out_dir=out_dir,
    )


def run_m1211_backend_general_replan(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    m1210_report: str | Path = DEFAULT_M1210_REPORT,
) -> dict[str, Any]:
    """Compatibility wrapper with the milestone in the function name."""

    return run_backend_general_replan(out_dir=out_dir, m1210_report=m1210_report)


def build_backend_general_replan_report(
    *,
    m1210_report: dict[str, Any],
    m1210_report_path: str | Path,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build the M12.11 backend-general route-freeze report."""

    route_matrix = _route_matrix(m1210_report)
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": "route_frozen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "source_reports": _source_reports(m1210_report, m1210_report_path),
        "source_report_status": _source_report_status(m1210_report),
        "primary_route": PRIMARY_ROUTE,
        "auxiliary_order": list(AUXILIARY_ORDER),
        "route_freeze": {
            "intent": "route_freeze_not_performance_claim",
            "selected_primary_route": PRIMARY_ROUTE,
            "selected_auxiliary_order": list(AUXILIARY_ORDER),
            "success_metric": "machine_readable_report_and_docs",
            "implementation_deferred_to": ["M12.12", "M12.13", "M12.14"],
        },
        "non_goals": _non_goals(),
        "route_matrix": route_matrix,
        "next_milestones": _next_milestones(),
        "next_default_action": {
            "action": "m12_12_real_tl_dot_bridge_harness",
            "reason": (
                "M12.11 froze the route; the next implementation slice must "
                "bridge real Triton JIT tl.dot TTIR into matmul_minimal."
            ),
        },
        "rejected_routes": _rejected_routes(route_matrix),
        "completion_gate": _completion_gate(m1210_report),
        "performance_ready_e2e": False,
        "performance_claim": False,
        "p2_passed": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "notes": [
            "M12.11 freezes the replacement route after M12.10; it does not implement it.",
            "The first implementation slice must bridge real Triton JIT tl.dot TTIR into the existing tt.dot matmul semantics.",
            "Runtime overhead is the first auxiliary track; schedule handoff follows real tt.dot lowering evidence.",
        ],
    }
    report["invariants"] = _invariants(report)
    if report["invariants"]["status"] != "passed":
        report["status"] = "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def build_m1211_backend_general_replan_report(
    *,
    m1210_report: dict[str, Any],
    m1210_report_path: str | Path,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Compatibility wrapper with the milestone in the function name."""

    return build_backend_general_replan_report(
        m1210_report=m1210_report,
        m1210_report_path=m1210_report_path,
        out_dir=out_dir,
    )


def _source_reports(m1210_report: dict[str, Any], m1210_report_path: str | Path) -> dict[str, str]:
    source_reports = {"m12_10_report": str(m1210_report_path)}
    m1210_sources = m1210_report.get("source_reports") or {}
    if "m12_9_report" in m1210_sources:
        source_reports["m12_9_report"] = str(m1210_sources["m12_9_report"])
    return source_reports


def _source_report_status(m1210_report: dict[str, Any]) -> dict[str, Any]:
    invariants = m1210_report.get("invariants") or {}
    completion_gate = m1210_report.get("completion_gate") or {}
    return {
        "m12_10_status": m1210_report.get("status"),
        "m12_10_invariants": invariants.get("status"),
        "m12_10_next_action": (m1210_report.get("next_default_action") or {}).get("action"),
        "m12_10_m12_complete": completion_gate.get("m12_complete"),
        "m12_10_performance_ready_e2e": m1210_report.get("performance_ready_e2e"),
    }


def _route_matrix(m1210_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    claims = m1210_report.get("route_claims") or {}
    backend_claim = claims.get(BACKEND_PASS_MODE, {})
    fused_claim = claims.get(FUSED_QKV_MODE, {})
    return {
        PRIMARY_ROUTE: {
            "role": "primary",
            "decision": "selected",
            "next_milestone": "M12.12",
            "implementation_status": "planned",
            "m12_11_implements": False,
            "backend_general": True,
            "model_specific_artifact": False,
            "counts_for_triton_tvm_backend_progress_when_implemented": True,
            "success_criteria": [
                "Use a standalone real Triton JIT tl.dot kernel as input.",
                "Lower through lower_to_ttir into textual TTIR containing tt.dot.",
                "Classify the resulting static rank-2 fp16/bf16 dot through matmul_minimal.",
                "Report masked, layout, or block-pointer cases as explicit unsupported gaps.",
            ],
        },
        "runtime_overhead": {
            "role": "auxiliary_1",
            "decision": "selected_after_primary",
            "next_milestone": "M12.13",
            "implementation_status": "planned",
            "m12_11_implements": False,
            "backend_general": True,
            "model_specific_artifact": False,
            "targets": [
                "TVM packed-call overhead",
                "DLPack conversion overhead",
                "artifact dispatch overhead",
                "launch overhead without per-call synchronization",
            ],
        },
        "tt_dot_schedule_handoff": {
            "role": "auxiliary_2",
            "decision": "selected_after_runtime",
            "next_milestone": "M12.14",
            "implementation_status": "planned",
            "m12_11_implements": False,
            "backend_general": True,
            "model_specific_artifact": False,
            "depends_on": PRIMARY_ROUTE,
            "handoff_rule": "TensorCore/GPU schedules must consume real lowering-derived tt_dot semantics, not wrapper-specific artifacts.",
        },
        "m129_backend_pass": {
            "role": "historical_evidence",
            "decision": "keep_as_backend_general_progress_evidence",
            "next_milestone": None,
            "implementation_status": "complete_before_m12_11",
            "backend_general": bool(backend_claim.get("backend_general")),
            "model_specific_artifact": bool(backend_claim.get("model_specific_artifact")),
            "counts_for_triton_tvm_backend_progress": bool(
                backend_claim.get("counts_for_triton_tvm_backend_progress")
            ),
            "route_diagnostic_label": backend_claim.get(
                "route_diagnostic_label", "backend_general_progress"
            ),
            "readiness_disposition": "evidence_only_not_route_completion",
        },
        "m129_fused_qkv_artifact": {
            "role": "rejected_readiness",
            "decision": "diagnostic_only",
            "next_milestone": None,
            "implementation_status": "complete_before_m12_11",
            "backend_general": False,
            "model_specific_artifact": True,
            "counts_for_triton_tvm_backend_progress": False,
            "route_diagnostic_label": fused_claim.get(
                "route_diagnostic_label", "model_specific_diagnostic_only"
            ),
            "readiness_disposition": "rejected_for_backend_readiness",
            "reason": "Fixed-shape ViT QKV fused artifact; diagnostic evidence only.",
        },
    }


def _next_milestones() -> list[dict[str, Any]]:
    return [
        {
            "milestone": "M12.12",
            "name": "real_tl_dot_bridge_harness",
            "track": PRIMARY_ROUTE,
            "goal": "Bridge real Triton JIT tl.dot TTIR into the existing matmul_minimal tt.dot semantics.",
            "acceptance": [
                "fp16/bf16 static rank-2 dot is accepted from real Triton lowering.",
                "The accepted path is independent of Inductor wrapper names and ViT buffers.",
                "Masks, layout transforms, and block pointers are explicit unsupported gaps if present.",
            ],
        },
        {
            "milestone": "M12.13",
            "name": "runtime_overhead_general_measurement",
            "track": "runtime_overhead",
            "goal": "Measure reusable runtime overhead outside a model-specific ViT graph.",
            "acceptance": [
                "Packed-call, DLPack, artifact dispatch, and launch overhead are separately reported.",
                "The report does not depend on fixed-shape QKV or wrapper replay.",
            ],
        },
        {
            "milestone": "M12.14",
            "name": "real_tt_dot_schedule_handoff",
            "track": "tt_dot_schedule_handoff",
            "goal": "Feed real lowering-derived tt_dot semantics into reusable TVM GPU schedules.",
            "acceptance": [
                "TensorCore/GPU schedule eligibility is keyed by reusable tt_dot semantics.",
                "M12.9 wrapper-specific schedules are not treated as the readiness source.",
            ],
        },
    ]


def _non_goals() -> list[str]:
    return [
        "hand-written fixed-shape ViT executor",
        "wrapper-buffer replay",
        "wrapper line-number replay",
        "M12.9 fused-QKV counted as backend readiness",
        "P2 redefinition",
        "M12.11 performance improvement claim",
        "strict full-native TVM claim",
    ]


def _rejected_routes(route_matrix: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fused = route_matrix["m129_fused_qkv_artifact"]
    return [
        {
            "route": "hand_written_vit_executor",
            "reason": "Model-specific executor work bypasses Triton language lowering.",
            "counts_for_triton_tvm_backend_progress": False,
        },
        {
            "route": "wrapper_buffer_replay",
            "reason": "Wrapper buffer-name or line-number replay is not backend-general.",
            "counts_for_triton_tvm_backend_progress": False,
        },
        {
            "route": "m129_fused_qkv_artifact",
            "reason": fused["reason"],
            "route_diagnostic_label": fused["route_diagnostic_label"],
            "counts_for_triton_tvm_backend_progress": False,
        },
    ]


def _completion_gate(m1210_report: dict[str, Any]) -> dict[str, Any]:
    m1210_completion = m1210_report.get("completion_gate") or {}
    return {
        "m12_complete": False,
        "requires_fixed_shape_vit_p2": True,
        "p2_passed": False,
        "m12_10_m12_complete": bool(m1210_completion.get("m12_complete")),
        "performance_ready_e2e": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("milestone") != MILESTONE:
        failures.append("m12_11_wrong_milestone")
    if report.get("status") != "route_frozen":
        failures.append("m12_11_status_not_route_frozen")
    if report.get("primary_route") != PRIMARY_ROUTE:
        failures.append("m12_11_primary_route_not_real_tl_dot_bridge")
    if report.get("auxiliary_order") != AUXILIARY_ORDER:
        failures.append("m12_11_auxiliary_order_not_runtime_before_schedule")

    source_status = report.get("source_report_status") or {}
    if source_status.get("m12_10_status") != "passed":
        failures.append("m12_11_m12_10_report_not_passed")
    if source_status.get("m12_10_invariants") != "passed":
        failures.append("m12_11_m12_10_invariants_not_passed")
    if source_status.get("m12_10_next_action") != "m12_11_backend_general_replan":
        failures.append("m12_11_m12_10_next_action_not_replan")

    matrix = report.get("route_matrix") or {}
    if PRIMARY_ROUTE not in matrix:
        failures.append("m12_11_missing_real_tl_dot_route")
    fused = matrix.get("m129_fused_qkv_artifact") or {}
    if fused.get("route_diagnostic_label") != "model_specific_diagnostic_only":
        failures.append("m12_11_fused_qkv_not_diagnostic_only")
    if fused.get("readiness_disposition") != "rejected_for_backend_readiness":
        failures.append("m12_11_fused_qkv_not_rejected_for_readiness")
    if fused.get("counts_for_triton_tvm_backend_progress") is not False:
        failures.append("m12_11_fused_qkv_counts_for_backend_progress")
    if fused.get("backend_general") is not False:
        failures.append("m12_11_fused_qkv_marked_backend_general")
    if fused.get("model_specific_artifact") is not True:
        failures.append("m12_11_fused_qkv_not_model_specific")

    milestones = [item.get("milestone") for item in report.get("next_milestones", [])]
    if milestones != ["M12.12", "M12.13", "M12.14"]:
        failures.append("m12_11_next_milestones_not_ordered")
    if (report.get("next_default_action") or {}).get(
        "action"
    ) != "m12_12_real_tl_dot_bridge_harness":
        failures.append("m12_11_next_action_not_real_tl_dot_bridge_harness")
    non_goals = set(report.get("non_goals") or [])
    for required in _non_goals():
        if required not in non_goals:
            failures.append(f"m12_11_missing_non_goal_{required.replace(' ', '_')}")

    if report.get("performance_ready_e2e") is not False:
        failures.append("m12_11_performance_ready_claim_present")
    if report.get("performance_claim") is not False:
        failures.append("m12_11_performance_claim_present")
    if report.get("p2_passed") is not False:
        failures.append("m12_11_p2_pass_claim_present")
    if report.get("backend_general_complete") is not False:
        failures.append("m12_11_backend_general_completion_claim_present")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_11_strict_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_11_full_tvm_runnable_claim_present")

    completion = report.get("completion_gate") or {}
    if completion.get("m12_complete") is not False:
        failures.append("m12_11_m12_complete_claim_present")
    if completion.get("requires_fixed_shape_vit_p2") is not True:
        failures.append("m12_11_p2_gate_not_preserved")

    return {"status": "passed" if not failures else "failed", "invariant_failures": failures}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "m12_11_complete": report.get("status") == "route_frozen"
        and (report.get("invariants") or {}).get("status") == "passed",
        "m12_complete": False,
        "route_frozen": report.get("status") == "route_frozen",
        "primary_route": report.get("primary_route"),
        "auxiliary_order": report.get("auxiliary_order"),
        "next_milestones": [
            item.get("milestone") for item in report.get("next_milestones", [])
        ],
        "next_action": (report.get("next_default_action") or {}).get("action"),
        "blocked_actions": [
            "continue_blind_provider_runtime_optimization",
            "hand_written_vit_executor",
            "wrapper_buffer_replay",
        ],
    }


def _empty_report(
    *,
    status: str,
    availability_reason: str,
    m1210_report: str | Path,
) -> dict[str, Any]:
    return {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "source_reports": {"m12_10_report": str(m1210_report)},
        "primary_route": PRIMARY_ROUTE,
        "auxiliary_order": list(AUXILIARY_ORDER),
        "non_goals": _non_goals(),
        "route_matrix": {},
        "next_milestones": _next_milestones(),
        "next_default_action": {"action": "m12_12_real_tl_dot_bridge_harness"},
        "invariants": {"status": "not_run", "invariant_failures": []},
        "performance_ready_e2e": False,
        "performance_claim": False,
        "p2_passed": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
    }


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
        "# M12.11 Backend-General Replan",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Milestone: `{report.get('milestone')}`",
        f"- Primary route: `{report.get('primary_route')}`",
        f"- Auxiliary order: `{report.get('auxiliary_order')}`",
        f"- Next action: `{(report.get('next_default_action') or {}).get('action')}`",
        f"- Source M12.10 status: `{(report.get('source_report_status') or {}).get('m12_10_status')}`",
        f"- M12 complete: `{(report.get('completion_gate') or {}).get('m12_complete')}`",
        "",
        "## Route Matrix",
        "",
        "| Route | Role | Decision | Next milestone |",
        "| --- | --- | --- | --- |",
    ]
    for route, item in (report.get("route_matrix") or {}).items():
        lines.append(
            "| "
            f"`{route}` | "
            f"`{item.get('role')}` | "
            f"`{item.get('decision')}` | "
            f"`{item.get('next_milestone')}` |"
        )
    lines.extend(["", "## Next Milestones", ""])
    for item in report.get("next_milestones", []):
        lines.append(
            f"- `{item.get('milestone')}` `{item.get('name')}`: {item.get('goal')}"
        )
    lines.extend(["", "## Rejected Routes", ""])
    for item in report.get("rejected_routes", []):
        lines.append(f"- `{item.get('route')}`: {item.get('reason')}")
    lines.extend(["", "## Non-Goals", ""])
    for item in report.get("non_goals", []):
        lines.append(f"- {item}")
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


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m1211_backend_general_replan``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--m1210-report", type=Path, default=DEFAULT_M1210_REPORT)
    args = parser.parse_args(argv)

    report = run_backend_general_replan(out_dir=args.out_dir, m1210_report=args.m1210_report)
    print(
        f"M12.11 backend-general replan: status={report['status']} "
        f"primary={report.get('primary_route')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"route_frozen", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
