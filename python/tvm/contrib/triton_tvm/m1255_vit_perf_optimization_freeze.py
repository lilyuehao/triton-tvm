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
"""M12.5.5 fixed-shape ViT performance optimization target freeze."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_KIND = "triton_tvm_m12_5_5_vit_perf_optimization_freeze"
FREEZE_ID = "m12_5_5_vit_perf_optimization_freeze_v1"
DEFAULT_DASHBOARD_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_4_vit_e2e_dashboard/report.json"
)
DEFAULT_REVIEW_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_5_optimization_review/report.json"
)
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_5_5_perf_optimization_freeze"
)


def run_perf_optimization_freeze(
    *,
    dashboard_report: str | Path = DEFAULT_DASHBOARD_REPORT,
    review_report: str | Path = DEFAULT_REVIEW_REPORT,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Freeze the scoped M12 post-review performance optimization targets."""

    dashboard_path = Path(dashboard_report)
    review_path = Path(review_report)
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    p2_gap = _p2_gap(dashboard)
    primary_target = str(review.get("primary_target", "native_tvm_matmul_provider_cost"))
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dashboard_report": str(dashboard_path),
        "source_review_report": str(review_path),
        "target_model": dashboard.get("target_model", "vit_tiny_random"),
        "current_gate_state": _current_gate_state(dashboard, review),
        "p2_gap": p2_gap,
        "primary_target": primary_target,
        "frozen_targets": _frozen_targets(dashboard, review, p2_gap, primary_target),
        "acceptance_criteria": _acceptance_criteria(dashboard),
        "out_of_scope": _out_of_scope(),
        "p2_recovery_sequence": [
            {
                "slice": "M12.6",
                "name": "E2E Surface Hardening",
                "performance_optimization": False,
            },
            {
                "slice": "M12.7",
                "name": "P2 Recovery - Native Matmul Provider Cost",
                "primary_target": "native_tvm_matmul_provider_cost",
            },
            {
                "slice": "M12.8",
                "name": "P2 Gate Re-run and Close/Continue Decision",
            },
        ],
        "next_action": (
            "Use this frozen target list as the M12.6 hardening input and as the "
            "decision record for the M12.7 P2 recovery slice."
        ),
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


def _current_gate_state(dashboard: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    return {
        "dashboard_status": dashboard.get("status"),
        "m12_performance_tier": dashboard.get("m12_performance_tier"),
        "p1_passed": bool(dashboard.get("p1", {}).get("passed")),
        "p2_passed": bool(dashboard.get("p2", {}).get("passed")),
        "performance_ready_e2e": bool(dashboard.get("performance_ready_e2e")),
        "host_staging_bytes": int(dashboard.get("host_staging_bytes", 0) or 0),
        "strict_full_tvm_native": bool(dashboard.get("strict_full_tvm_native")),
        "full_tvm_runnable_models": int(dashboard.get("full_tvm_runnable_models", 0) or 0),
        "review_disposition": review.get("disposition"),
        "focused_follow_up_recommended": bool(
            review.get("focused_follow_up_recommended", review.get("p3_recommended"))
        ),
    }


def _p2_gap(dashboard: dict[str, Any]) -> dict[str, Any]:
    latency = dashboard.get("latency_ms", {})
    ratios = dashboard.get("ratios", {})
    compile_latency = latency.get("torch_compile_inductor", {})
    triton_latency = latency.get("triton_tvm_e2e", {})
    compile_p50 = _to_float_or_none(compile_latency.get("p50_ms"))
    compile_p95 = _to_float_or_none(compile_latency.get("p95_ms"))
    triton_p50 = _to_float_or_none(triton_latency.get("p50_ms"))
    triton_p95 = _to_float_or_none(triton_latency.get("p95_ms"))
    p50_ceiling = compile_p50 * 2.0 if compile_p50 is not None else None
    p95_ceiling = compile_p95 * 2.0 if compile_p95 is not None else None
    p50_excess = _positive_excess(triton_p50, p50_ceiling)
    p95_excess = _positive_excess(triton_p95, p95_ceiling)
    return {
        "threshold": "triton_tvm_e2e p50/p95 <= 2x torch_compile_inductor",
        "torch_compile_p50_ms": compile_p50,
        "torch_compile_p95_ms": compile_p95,
        "triton_tvm_p50_ms": triton_p50,
        "triton_tvm_p95_ms": triton_p95,
        "p50_ceiling_ms": p50_ceiling,
        "p95_ceiling_ms": p95_ceiling,
        "p50_excess_ms": p50_excess,
        "p95_excess_ms": p95_excess,
        "p50_required_reduction_pct_of_current": _pct(p50_excess, triton_p50),
        "p95_required_reduction_pct_of_current": _pct(p95_excess, triton_p95),
        "ratio_p50": _to_float_or_none(ratios.get("triton_tvm_to_torch_compile_p50")),
        "ratio_p95": _to_float_or_none(ratios.get("triton_tvm_to_torch_compile_p95")),
    }


def _frozen_targets(
    dashboard: dict[str, Any],
    review: dict[str, Any],
    p2_gap: dict[str, Any],
    primary_target: str,
) -> list[dict[str, Any]]:
    provider_counts = dashboard.get("provider_mix", {}).get("provider_counts", {})
    largest_excess = max(
        float(p2_gap.get("p50_excess_ms") or 0.0),
        float(p2_gap.get("p95_excess_ms") or 0.0),
    )
    target_specs = [
        {
            "target_id": "native_tvm_matmul_provider_cost",
            "priority": "primary",
            "evidence": [
                f"{provider_counts.get('native_tvm_matmul', 0)} wrapper matmul/addmm launches",
                "M12.5/M12.5.5 selected this as the primary M12.7 P2 recovery target",
            ],
            "frozen_action": (
                "Measure and reduce the fixed-shape native_tvm_matmul wrapper cost before "
                "changing model scope or attention/conv semantics."
            ),
            "success_signal": (
                "Recover at least the current P2 gap, approximately "
                f"{largest_excess:.6f} ms E2E, while preserving all M12 correctness and "
                "provider invariants."
            ),
        },
        {
            "target_id": "launch_sequence_overhead",
            "priority": "secondary",
            "evidence": [f"launch_count_estimate={dashboard.get('launch_count_estimate')}"],
            "frozen_action": "Profile launch sequence cost after the primary matmul path is isolated.",
            "success_signal": "Any launch reduction must keep provider reporting complete and explicit.",
        },
        {
            "target_id": "captured_kernel_harness_overhead",
            "priority": "secondary_measurement",
            "evidence": [
                f"{provider_counts.get('torch_inductor_triton_captured_harness', 0)} captured-kernel harness launches"
            ],
            "frozen_action": (
                "Measure separately; do not treat this as strict native TVM replacement work "
                "inside M12.5.5."
            ),
            "success_signal": (
                "Only promote into M12.7 input if the measurement shows a larger gap than matmul."
            ),
        },
        {
            "target_id": "python_packed_call_dispatch",
            "priority": "diagnostic",
            "evidence": ["CUDA event timing does not fully price CPU dispatch overhead"],
            "frozen_action": "Keep a host-side timing/profiling note, not a P2 gate change.",
            "success_signal": "Do not change P2 thresholds based on host dispatch evidence alone.",
        },
        {
            "target_id": "buffer_allocation_and_layout",
            "priority": "watch",
            "evidence": [
                "dashboard lacks allocation lifetime breakdown",
                "ViT path includes transpose/view-sensitive wrapper matmul inputs",
            ],
            "frozen_action": "Inspect only if matmul and launch profiling do not explain the P2 miss.",
            "success_signal": "Any fix must keep zero host staging and fixed-shape scope.",
        },
        {
            "target_id": "attention_conv_provider_cost",
            "priority": "watch",
            "evidence": [
                f"{provider_counts.get('native_decomposed', 0)} native_decomposed attention replay",
                f"{provider_counts.get('device_torch_cuda', 0)} device_torch_cuda conv provider call",
            ],
            "frozen_action": (
                "Do not reopen attention semantics or native TVM conv scheduling without a later "
                "explicit milestone."
            ),
            "success_signal": "Provider boundaries remain explicit and zero-host-staged.",
        },
    ]
    review_priorities = {
        item.get("category"): item.get("priority") for item in review.get("opportunities", [])
    }
    for target in target_specs:
        target["selected_by_m12_5"] = target["target_id"] == primary_target
        target["m12_5_priority"] = review_priorities.get(target["target_id"], "")
    return target_specs


def _acceptance_criteria(dashboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "p2_threshold_unchanged": "p50 and p95 each <= 2x torch_compile_inductor",
        "correctness_allclose_required": bool(dashboard.get("correctness", {}).get("allclose")),
        "host_staging_bytes_required": 0,
        "complete_provider_mix_required": True,
        "no_silent_fallback_required": True,
        "strict_full_tvm_native_required": False,
        "full_tvm_runnable_models_required": 0,
        "scope": "fixed-shape vit_tiny_random only",
    }


def _out_of_scope() -> list[str]:
    return [
        "YOLO closure",
        "Llama GEMM closure",
        "dynamic shape support",
        "new attention semantics",
        "native TVM conv scheduling claim",
        "strict full-native TVM replacement claim",
        "general concat/split Grid2D closure",
    ]


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    gate = report.get("current_gate_state", {})
    targets = report.get("frozen_targets", [])
    selected = [target for target in targets if target.get("selected_by_m12_5")]
    if gate.get("performance_ready_e2e") and not gate.get("p2_passed"):
        failures.append("m12_5_5_performance_ready_without_p2")
    if gate.get("host_staging_bytes") != 0:
        failures.append("m12_5_5_host_staging_nonzero")
    if gate.get("strict_full_tvm_native"):
        failures.append("m12_5_5_strict_native_claim_present")
    if gate.get("full_tvm_runnable_models") != 0:
        failures.append("m12_5_5_full_tvm_runnable_claim_present")
    if report.get("primary_target") != "native_tvm_matmul_provider_cost":
        failures.append("m12_5_5_primary_target_not_frozen")
    if len(selected) != 1:
        failures.append("m12_5_5_selected_target_count_invalid")
    if not report.get("p2_gap", {}).get("p50_excess_ms"):
        failures.append("m12_5_5_missing_p50_gap")
    if not report.get("p2_gap", {}).get("p95_excess_ms"):
        failures.append("m12_5_5_missing_p95_gap")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    gap = report["p2_gap"]
    gate = report["current_gate_state"]
    lines = [
        "# M12.5.5 Performance Optimization Freeze",
        "",
        f"- Status: {report['status']}",
        f"- Target model: `{report['target_model']}`",
        f"- Current tier: {gate.get('m12_performance_tier')}",
        f"- P1/P2 passed: {gate.get('p1_passed')} / {gate.get('p2_passed')}",
        f"- Primary frozen target: `{report['primary_target']}`",
        f"- Host staging bytes: {gate.get('host_staging_bytes')}",
        f"- Performance-ready E2E: {gate.get('performance_ready_e2e')}",
        "",
        "## P2 Gap",
        "",
        "| Metric | Current | P2 ceiling | Excess |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| p50 ms | {_fmt(gap.get('triton_tvm_p50_ms'))} | "
            f"{_fmt(gap.get('p50_ceiling_ms'))} | {_fmt(gap.get('p50_excess_ms'))} |"
        ),
        (
            f"| p95 ms | {_fmt(gap.get('triton_tvm_p95_ms'))} | "
            f"{_fmt(gap.get('p95_ceiling_ms'))} | {_fmt(gap.get('p95_excess_ms'))} |"
        ),
        "",
        "## Frozen Targets",
        "",
        "| Target | Priority | Selected | Success signal |",
        "| --- | --- | --- | --- |",
    ]
    for target in report["frozen_targets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target['target_id']}`",
                    target["priority"],
                    str(target["selected_by_m12_5"]),
                    target["success_signal"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
        ]
    )
    for key, value in report["acceptance_criteria"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Out Of Scope", ""])
    for item in report["out_of_scope"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _positive_excess(value: float | None, ceiling: float | None) -> float | None:
    if value is None or ceiling is None:
        return None
    return max(0.0, value - ceiling)


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return 100.0 * numerator / denominator


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-report", type=Path, default=DEFAULT_DASHBOARD_REPORT)
    parser.add_argument("--review-report", type=Path, default=DEFAULT_REVIEW_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    report = run_perf_optimization_freeze(
        dashboard_report=args.dashboard_report,
        review_report=args.review_report,
        out_dir=args.out_dir,
    )
    print(
        "M12.5.5 optimization freeze: "
        f"status={report['status']} primary={report['primary_target']} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["invariants"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
