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
"""M12.10 route cleanup report for the fixed-shape ViT E2E track."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m123_vit_e2e_runner import TARGET_MODEL
from .m129_vit_provider_runtime_optimization import (
    BACKEND_PASS_MODE,
    CURRENT_MODE,
    FUSED_QKV_MODE,
)


REPORT_KIND = "triton_tvm_m12_10_route_cleanup"
REPORT_ID = "m12_10_route_cleanup_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_10_route_cleanup"
)
DEFAULT_M129_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_9_provider_runtime_optimization/report.json"
)
REQUIRED_CLAIM_FIELDS = (
    "backend_general",
    "model_specific_artifact",
    "counts_for_triton_tvm_backend_progress",
)


def run_vit_route_cleanup(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    m129_report: str | Path = DEFAULT_M129_REPORT,
) -> dict[str, Any]:
    """Read the M12.9 report and write the M12.10 route-cleanup report."""

    try:
        source = _read_json_report(m129_report)
    except FileNotFoundError:
        report = _empty_report(
            status="unavailable",
            availability_reason="m12_9_report_missing",
            m129_report=m129_report,
        )
        _maybe_write_report(report, out_dir)
        return report
    except json.JSONDecodeError as err:
        report = _empty_report(
            status="failed",
            availability_reason=f"m12_9_report_invalid_json:{err}",
            m129_report=m129_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    return build_vit_route_cleanup_report(
        m129_report=source,
        m129_report_path=m129_report,
        out_dir=out_dir,
    )


def build_vit_route_cleanup_report(
    *,
    m129_report: dict[str, Any],
    m129_report_path: str | Path,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build the M12.10 report from an M12.9 optimization report."""

    route_claims = _route_claims(m129_report)
    route_policy = _route_policy()
    next_default_action = {
        "action": "m12_11_backend_general_replan",
        "blocked_action": "continue_blind_provider_runtime_optimization",
        "reason": (
            "M12.9 improved the fixed-shape ViT dashboard but exposed drift "
            "toward model-specific executor work."
        ),
    }
    completion_gate = _completion_gate(m129_report)
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "source_reports": {"m12_9_report": str(m129_report_path)},
        "source_report_status": {
            "m12_9_status": m129_report.get("status"),
            "m12_9_invariants": (m129_report.get("invariants") or {}).get("status"),
            "m12_9_best_mode": (m129_report.get("best_mode") or {}).get("mode"),
            "m12_9_p2_passed": (m129_report.get("p2_context") or {}).get(
                "best_mode_p2_passed"
            ),
            "m12_9_performance_ready_e2e": m129_report.get("performance_ready_e2e"),
        },
        "required_claim_fields": list(REQUIRED_CLAIM_FIELDS),
        "route_claims": route_claims,
        "route_policy": route_policy,
        "completion_gate": completion_gate,
        "next_default_action": next_default_action,
        "accepted_future_work": [
            "Triton language lowering coverage",
            "TVM backend schedule or pass for a reusable IR class",
            "Reusable runtime overhead reduction",
            "Model-specific evidence labeled diagnostic only",
        ],
        "rejected_future_work": [
            "hand-written fixed-shape ViT executor",
            "wrapper buffer-name replay",
            "wrapper line-number replay",
            "one-off QKV or larger subgraph artifact counted as backend readiness",
            "P2-driven shortcut that bypasses Triton language lowering",
        ],
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "notes": [
            "M12.10 is route cleanup and report-surface hardening, not a performance slice.",
            "The fixed-shape ViT P2 gate remains the M12 completion gate.",
            "P2 pressure must not justify model-specific executor work.",
        ],
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _route_claims(m129_report: dict[str, Any]) -> dict[str, Any]:
    modes = m129_report.get("optimization_modes") or {}
    ordered_modes = [CURRENT_MODE, BACKEND_PASS_MODE, FUSED_QKV_MODE]
    return {
        mode: _canonical_route_claim(mode, modes.get(mode, {}))
        for mode in ordered_modes
        if mode in modes or mode == FUSED_QKV_MODE
    }


def _canonical_route_claim(mode: str, source: dict[str, Any]) -> dict[str, Any]:
    if mode == BACKEND_PASS_MODE:
        defaults = {
            "backend_general": True,
            "model_specific_artifact": False,
            "counts_for_triton_tvm_backend_progress": True,
            "route_diagnostic_label": "backend_general_progress",
            "reason": "TVM module pass over native wrapper matmul TIR.",
        }
    elif mode == FUSED_QKV_MODE:
        defaults = {
            "backend_general": False,
            "model_specific_artifact": True,
            "counts_for_triton_tvm_backend_progress": False,
            "route_diagnostic_label": "model_specific_diagnostic_only",
            "reason": "Fixed-shape ViT QKV fused artifact; diagnostic evidence only.",
        }
    else:
        defaults = {
            "backend_general": False,
            "model_specific_artifact": False,
            "counts_for_triton_tvm_backend_progress": False,
            "route_diagnostic_label": "baseline_only",
            "reason": "M12.9 comparison baseline, not a new optimization claim.",
        }

    claim = {
        "mode": mode,
        "source_kind": source.get("kind"),
        "p50_ms": source.get("p50_ms"),
        "p95_ms": source.get("p95_ms"),
    }
    for field in REQUIRED_CLAIM_FIELDS:
        claim[field] = bool(source[field]) if field in source else defaults[field]
    claim["route_diagnostic_label"] = source.get(
        "route_diagnostic_label",
        defaults["route_diagnostic_label"],
    )
    claim["requires_future_generalization"] = bool(
        source.get("requires_future_generalization", claim["model_specific_artifact"])
    )
    claim["reason"] = source.get("diagnostic_reason", defaults["reason"])
    return claim


def _route_policy() -> dict[str, bool]:
    return {
        "m12_10_is_cleanup_not_optimization": True,
        "no_hand_written_vit_executor": True,
        "no_wrapper_buffer_replay_as_backend_progress": True,
        "model_specific_artifacts_are_diagnostic_only": True,
        "inductor_wrappers_are_corpus_input": True,
        "p2_cannot_justify_model_specific_executor": True,
        "m12_11_reserved_for_backend_general_replan": True,
    }


def _completion_gate(m129_report: dict[str, Any]) -> dict[str, Any]:
    p2 = m129_report.get("p2_context") or {}
    return {
        "m12_complete": False,
        "requires_fixed_shape_vit_p2": True,
        "m12_9_best_mode": (m129_report.get("best_mode") or {}).get("mode"),
        "m12_9_best_mode_p2_passed": bool(p2.get("best_mode_p2_passed")),
        "performance_ready_e2e": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    source_status = report.get("source_report_status") or {}
    if source_status.get("m12_9_status") != "passed":
        failures.append("m12_10_m12_9_report_not_passed")
    if source_status.get("m12_9_invariants") != "passed":
        failures.append("m12_10_m12_9_invariants_not_passed")
    for mode, claim in (report.get("route_claims") or {}).items():
        for field in REQUIRED_CLAIM_FIELDS:
            if field not in claim:
                failures.append(f"m12_10_missing_claim_field_{mode}_{field}")
        if claim.get("model_specific_artifact") and claim.get(
            "counts_for_triton_tvm_backend_progress"
        ):
            failures.append(f"m12_10_model_specific_counts_as_backend_progress_{mode}")
    fused = (report.get("route_claims") or {}).get(FUSED_QKV_MODE, {})
    if fused.get("route_diagnostic_label") != "model_specific_diagnostic_only":
        failures.append("m12_10_fused_qkv_not_diagnostic_only")
    if fused.get("counts_for_triton_tvm_backend_progress") is not False:
        failures.append("m12_10_fused_qkv_counts_for_backend_progress")
    policy = report.get("route_policy") or {}
    for key, value in policy.items():
        if value is not True:
            failures.append(f"m12_10_route_policy_false_{key}")
    if (report.get("next_default_action") or {}).get("action") != "m12_11_backend_general_replan":
        failures.append("m12_10_next_action_not_m12_11")
    if report.get("performance_ready_e2e") is not False:
        failures.append("m12_10_performance_ready_claim_present")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_10_strict_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_10_full_tvm_runnable_claim_present")
    return {"status": "passed" if not failures else "failed", "invariant_failures": failures}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    claims = report.get("route_claims") or {}
    diagnostic_modes = [
        mode
        for mode, claim in claims.items()
        if claim.get("route_diagnostic_label") == "model_specific_diagnostic_only"
    ]
    backend_progress_modes = [
        mode for mode, claim in claims.items() if claim.get("counts_for_triton_tvm_backend_progress")
    ]
    return {
        "m12_10_complete": report.get("status") == "passed",
        "m12_complete": False,
        "diagnostic_only_modes": diagnostic_modes,
        "backend_progress_modes": backend_progress_modes,
        "next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(
    *,
    status: str,
    availability_reason: str,
    m129_report: str | Path,
) -> dict[str, Any]:
    return {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "source_reports": {"m12_9_report": str(m129_report)},
        "route_claims": {},
        "route_policy": _route_policy(),
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
        },
        "next_default_action": {"action": "m12_11_backend_general_replan"},
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "invariants": {"status": "not_run", "invariant_failures": []},
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
        "# M12.10 Route Cleanup",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Source M12.9 status: `{(report.get('source_report_status') or {}).get('m12_9_status')}`",
        f"- Next action: `{(report.get('next_default_action') or {}).get('action')}`",
        f"- M12 complete: `{(report.get('completion_gate') or {}).get('m12_complete')}`",
        "",
        "## Route Claims",
        "",
        "| Mode | backend_general | model_specific_artifact | counts_for_progress | label |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for mode, claim in (report.get("route_claims") or {}).items():
        lines.append(
            "| "
            f"`{mode}` | "
            f"`{claim.get('backend_general')}` | "
            f"`{claim.get('model_specific_artifact')}` | "
            f"`{claim.get('counts_for_triton_tvm_backend_progress')}` | "
            f"`{claim.get('route_diagnostic_label')}` |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
        ]
    )
    for key, value in (report.get("route_policy") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
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
    """CLI for ``python -m tvm.contrib.triton_tvm.m1210_vit_route_cleanup``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--m129-report", type=Path, default=DEFAULT_M129_REPORT)
    args = parser.parse_args(argv)

    report = run_vit_route_cleanup(out_dir=args.out_dir, m129_report=args.m129_report)
    print(
        f"M12.10 route cleanup: status={report['status']} "
        f"next={(report.get('next_default_action') or {}).get('action')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
