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
"""M12.5 fixed-shape ViT optimization opportunity review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_KIND = "triton_tvm_m12_5_vit_optimization_review"
REVIEW_ID = "m12_5_vit_optimization_review_v1"
DEFAULT_DASHBOARD_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_4_vit_e2e_dashboard/report.json"
)
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_5_optimization_review"
)


def run_optimization_review(
    *,
    dashboard_report: str | Path = DEFAULT_DASHBOARD_REPORT,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Review the M12.4 dashboard and write the M12.5 opportunity report."""

    dashboard_path = Path(dashboard_report)
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    p1_passed = bool(dashboard.get("p1", {}).get("passed"))
    p2_passed = bool(dashboard.get("p2", {}).get("passed"))
    measured = dashboard.get("status") == "measured"

    disposition = _disposition(measured=measured, p1_passed=p1_passed, p2_passed=p2_passed)
    primary_target = _primary_target(dashboard, disposition)
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "status": "blocked" if disposition == "blocked_on_m12_4" else "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dashboard_report": str(dashboard_path),
        "target_model": dashboard.get("target_model", "vit_tiny_random"),
        "dashboard_status": dashboard.get("status"),
        "m12_performance_tier": dashboard.get("m12_performance_tier", "unavailable"),
        "p1_passed": p1_passed,
        "p2_passed": p2_passed,
        "performance_ready_e2e": bool(dashboard.get("performance_ready_e2e")),
        "disposition": disposition,
        "focused_follow_up_recommended": disposition == "open_m12_7_p2_recovery",
        "p3_recommended": False,
        "primary_target": primary_target,
        "next_action": _next_action(disposition),
        "opportunities": _opportunities(dashboard, primary_target, disposition),
        "notes": _notes(disposition),
    }
    report["invariants"] = _invariants(report)
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_path / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _disposition(*, measured: bool, p1_passed: bool, p2_passed: bool) -> str:
    if p2_passed:
        return "close_m12_after_p2"
    if measured and p1_passed:
        return "open_m12_7_p2_recovery"
    return "blocked_on_m12_4"


def _primary_target(dashboard: dict[str, Any], disposition: str) -> str:
    if disposition == "blocked_on_m12_4":
        return "m12_4_dashboard_measurement"
    if disposition == "close_m12_after_p2":
        return "no_p3_by_default"
    provider_counts = dashboard.get("provider_mix", {}).get("provider_counts", {})
    if int(provider_counts.get("native_tvm_matmul", 0) or 0) >= 7:
        return "native_tvm_matmul_provider_cost"
    return "captured_kernel_harness_overhead"


def _next_action(disposition: str) -> str:
    if disposition == "close_m12_after_p2":
        return "M12.6 hardening"
    if disposition == "open_m12_7_p2_recovery":
        return (
            "M12.6 hardening, then M12.7 P2 recovery on native_tvm_matmul_provider_cost."
        )
    return "Regenerate M12.4 measured dashboard before continuing."


def _opportunities(
    dashboard: dict[str, Any],
    primary_target: str,
    disposition: str,
) -> list[dict[str, Any]]:
    provider_counts = dashboard.get("provider_mix", {}).get("provider_counts", {})
    launch_count = dashboard.get("launch_count_estimate")
    categories = [
        (
            "native_tvm_matmul_provider_cost",
            f"{provider_counts.get('native_tvm_matmul', 0)} wrapper matmul/addmm launches",
            "Review fixed-shape native matmul schedule cost before changing semantics.",
        ),
        (
            "launch_count",
            f"launch_count_estimate={launch_count}",
            "Look for launch fusion/reuse only after preserving explicit provider accounting.",
        ),
        (
            "captured_kernel_harness_overhead",
            f"{provider_counts.get('torch_inductor_triton_captured_harness', 0)} captured-kernel harness launches",
            "Measure separately before claiming strict native TVM replacement.",
        ),
        (
            "python_packed_call_dispatch",
            "dashboard uses Python-level provider patching and packed-call boundaries",
            "Keep as review item; CUDA event timing does not fully price CPU dispatch.",
        ),
        (
            "buffer_allocation_reuse",
            "E2E dashboard does not yet expose allocation lifetime breakdown",
            "Profile allocations only if the measured P2 gap justifies it.",
        ),
        (
            "layout_conversions",
            "ViT path includes wrapper views/transpose-sensitive matmul inputs",
            "Inspect only after identifying a measured layout-conversion cost.",
        ),
        (
            "attention_conv_provider_cost",
            "1 native_decomposed attention replay and 1 device_torch_cuda conv provider call",
            "Keep provider boundary explicit; no new attention or native conv claim in M12.",
        ),
    ]
    opportunities: list[dict[str, Any]] = []
    for category, evidence, action in categories:
        priority = _priority(category, primary_target, disposition)
        opportunities.append(
            {
                "category": category,
                "priority": priority,
                "evidence": evidence,
                "recommended_action": action,
            }
        )
    return opportunities


def _priority(category: str, primary_target: str, disposition: str) -> str:
    if disposition == "blocked_on_m12_4":
        return "blocked"
    if disposition == "close_m12_after_p2":
        return "residual"
    if category == primary_target:
        return "primary"
    if category in {"launch_count", "captured_kernel_harness_overhead", "python_packed_call_dispatch"}:
        return "secondary"
    return "watch"


def _notes(disposition: str) -> list[str]:
    if disposition == "close_m12_after_p2":
        return [
            "M12.5 does not open P3 by default after P2.",
            "Residual opportunities are documented for future milestones.",
        ]
    if disposition == "open_m12_7_p2_recovery":
        return [
            "M12.4 measured P1 but missed P2.",
            "The active follow-up is M12.7 P2 recovery after M12.6 hardening.",
            "The recovery work must preserve M12 provider boundaries and strict-full-native semantics.",
        ]
    return [
        "M12.5 is blocked until M12.4 has a measured dashboard with at least P1 evidence.",
    ]


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("performance_ready_e2e") and not report.get("p2_passed"):
        failures.append("m12_5_performance_ready_without_p2")
    if report.get("disposition") == "close_m12_after_p2" and report.get("p3_recommended"):
        failures.append("m12_5_p3_recommended_after_p2")
    if report.get("disposition") == "open_m12_7_p2_recovery" and not report.get("p1_passed"):
        failures.append("m12_5_p2_recovery_without_p1")
    if not report.get("opportunities"):
        failures.append("m12_5_missing_opportunity_categories")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M12.5 Optimization Opportunity Review",
        "",
        f"- Status: {report.get('status')}",
        f"- Source dashboard: `{report.get('source_dashboard_report')}`",
        f"- M12 performance tier: {report.get('m12_performance_tier')}",
        f"- P1/P2 passed: {report.get('p1_passed')} / {report.get('p2_passed')}",
        f"- Disposition: `{report.get('disposition')}`",
        f"- Focused follow-up recommended: {report.get('focused_follow_up_recommended')}",
        f"- Primary target: `{report.get('primary_target')}`",
        f"- Next action: {report.get('next_action')}",
        "",
        "| Category | Priority | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in report.get("opportunities", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['category']}`",
                    item["priority"],
                    item["evidence"],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-report", type=Path, default=DEFAULT_DASHBOARD_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    report = run_optimization_review(
        dashboard_report=args.dashboard_report,
        out_dir=args.out_dir,
    )
    print(
        "M12.5 optimization review: "
        f"status={report['status']} disposition={report['disposition']} "
        f"primary={report['primary_target']} out_dir={args.out_dir}"
    )
    return 0 if report["invariants"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
