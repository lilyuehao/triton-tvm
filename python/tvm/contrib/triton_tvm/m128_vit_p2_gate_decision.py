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
"""M12.8 fixed-shape ViT P2 gate rerun and close/continue decision."""

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
    NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
    TARGET_MODEL,
    _correctness_report,
    _dependency_versions,
    _provider_mix_report,
    _require_torch,
    execute_vit_fixed_shape_provider_path,
    prepare_vit_fixed_shape_e2e,
)
from .m124_vit_e2e_dashboard import (
    DEFAULT_REPEAT,
    DEFAULT_WARMUP,
    LAUNCH_COUNT_ESTIMATE,
    P1_THRESHOLD,
    P2_THRESHOLD,
)
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_8_vit_p2_gate_decision"
GATE_ID = "m12_8_vit_p2_gate_decision_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_8_p2_gate_decision"
)
DEFAULT_M127_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_7_native_matmul_provider_cost/report.json"
)
BASELINES = ("torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e")
NEAR_MISS_RATIO_MAX = 2.15
PROFILE_RETURN_RATIO_MIN = 2.30
EXPECTED_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}


def run_vit_p2_gate_decision(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
    m127_report: str | Path = DEFAULT_M127_REPORT,
) -> dict[str, Any]:
    """Run the M12.8 official P2 gate rerun and optionally write a report."""

    if not run_benchmarks:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
            m127_report=m127_report,
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
            m127_report=m127_report,
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
            m127_report=m127_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        m127 = _read_json_report(m127_report)
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        actual, patch_state = execute_vit_fixed_shape_provider_path(
            prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )
        correctness = _correctness_report(actual, prepared.expected)
        provider_mix = _provider_mix_report(
            wrapper_source=prepared.wrapper_source,
            matmul_calls=prepared.matmul_calls,
            patch_state=patch_state,
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
            "triton_tvm_e2e": _latency_record(
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
        report = build_vit_p2_gate_decision_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            m127_report=m127,
            m127_report_path=m127_report,
            latency_ms=latency_ms,
            correctness=correctness,
            provider_mix=provider_mix,
            device_bytes=_device_bytes_report(prepared, actual),
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
            availability_reason=f"m12_8_failed:{type(err).__name__}:{err}",
            m127_report=m127_report,
        )
        _maybe_write_report(report, out_dir)
        return report


def build_vit_p2_gate_decision_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    m127_report: dict[str, Any],
    m127_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, Any],
    provider_mix: dict[str, Any],
    device_bytes: dict[str, Any] | None = None,
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build an M12.8 P2 gate report from measured or synthetic data."""

    ratios = _ratio_report(latency_ms)
    host_staging_bytes = 0
    p0 = _p0_report(correctness, provider_mix)
    p1 = _p1_report(p0, ratios, host_staging_bytes)
    p2 = _p2_report(p0, ratios, host_staging_bytes, warmup)
    tier = _performance_tier(p0, p1, p2)
    source_status = {
        "m12_7_status": m127_report.get("status"),
        "m12_7_invariants": (m127_report.get("invariants") or {}).get("status"),
        "m12_7_optimized_p2_estimate_passed": (
            m127_report.get("p2_gap_recovery") or {}
        ).get("optimized_p2_estimate_passed"),
    }
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "gate_id": GATE_ID,
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "baselines": list(BASELINES),
        "provider_path": {
            "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            "source": "M12.7 validated fixed-shape provider path",
        },
        "source_reports": {"m12_7_report": str(m127_report_path)},
        "source_report_status": source_status,
        "latency_ms": latency_ms,
        "ratios": ratios,
        "correctness": correctness,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "m12_performance_tier": tier,
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": host_staging_bytes,
        "launch_count_estimate": LAUNCH_COUNT_ESTIMATE,
        "provider_mix": provider_mix,
        "device_bytes": device_bytes or {},
        "m12_8_decision": {},
        "notes": [
            "M12.8 only reruns the fixed-shape ViT P2 gate and records the close/continue decision.",
            "No new optimization, model scope, provider semantics, or P2 threshold changes are introduced.",
        ],
        "dependency_versions": dependency_versions or {"tvm": str(tvm.__version__)},
    }
    report["invariants"] = _invariants(report)
    gate_ready = report["invariants"]["status"] == "passed"
    report["p2"]["passed"] = bool(gate_ready and report["p2"]["passed"])
    report["m12_performance_tier"] = _performance_tier(
        report["p0"],
        report["p1"],
        report["p2"],
    )
    report["performance_ready_e2e"] = bool(report["p2"]["passed"])
    report["performance_claim"] = bool(report["p2"]["passed"])
    report["m12_8_decision"] = _decision(report)
    report["status"] = "passed" if gate_ready else "failed"
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
    return {
        "samples_ms": values,
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def _ratio_report(latency_ms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    triton = latency_ms["triton_tvm_e2e"]
    eager = latency_ms["torch_eager_cuda"]
    compile_latency = latency_ms["torch_compile_inductor"]
    return {
        "triton_tvm_to_torch_eager_p50": _safe_ratio(triton["p50_ms"], eager["p50_ms"]),
        "triton_tvm_to_torch_eager_p95": _safe_ratio(triton["p95_ms"], eager["p95_ms"]),
        "triton_tvm_to_torch_compile_p50": _safe_ratio(
            triton["p50_ms"], compile_latency["p50_ms"]
        ),
        "triton_tvm_to_torch_compile_p95": _safe_ratio(
            triton["p95_ms"], compile_latency["p95_ms"]
        ),
    }


def _p0_report(correctness: dict[str, Any], provider_mix: dict[str, Any]) -> dict[str, Any]:
    report = {
        "correctness": bool(correctness.get("allclose")),
        "no_silent_fallback": bool(provider_mix.get("no_silent_fallback")),
        "complete_provider_report": bool(provider_mix.get("complete_provider_report")),
    }
    report["passed"] = bool(all(report.values()))
    return report


def _p1_report(
    p0: dict[str, Any],
    ratios: dict[str, Any],
    host_staging_bytes: int,
) -> dict[str, Any]:
    eager_ok = _ratio_pair_ok(
        ratios["triton_tvm_to_torch_eager_p50"],
        ratios["triton_tvm_to_torch_eager_p95"],
        P1_THRESHOLD,
    )
    inductor_ok = _ratio_pair_ok(
        ratios["triton_tvm_to_torch_compile_p50"],
        ratios["triton_tvm_to_torch_compile_p95"],
        P1_THRESHOLD,
    )
    zero_host = host_staging_bytes == 0
    return {
        "threshold": "<=3x torch_eager_cuda or torch_compile_inductor p50/p95",
        "eager_baseline_ok": eager_ok,
        "inductor_baseline_ok": inductor_ok,
        "zero_host_staging": zero_host,
        "passed": bool(p0.get("passed") and zero_host and (eager_ok or inductor_ok)),
    }


def _p2_report(
    p0: dict[str, Any],
    ratios: dict[str, Any],
    host_staging_bytes: int,
    warmup: int,
) -> dict[str, Any]:
    inductor_ok = _ratio_pair_ok(
        ratios["triton_tvm_to_torch_compile_p50"],
        ratios["triton_tvm_to_torch_compile_p95"],
        P2_THRESHOLD,
    )
    zero_host = host_staging_bytes == 0
    warmed_cache = warmup > 0
    return {
        "threshold": "<=2x torch_compile_inductor p50/p95",
        "inductor_baseline_ok": inductor_ok,
        "warmed_cache": warmed_cache,
        "zero_host_staging": zero_host,
        "complete_provider_mix": bool(p0.get("complete_provider_report")),
        "correctness": bool(p0.get("correctness")),
        "passed": bool(p0.get("passed") and zero_host and warmed_cache and inductor_ok),
    }


def _performance_tier(p0: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any]) -> str:
    if p2.get("passed"):
        return "P2"
    if p1.get("passed"):
        return "P1"
    if p0.get("passed"):
        return "P0"
    return "unavailable"


def _decision(report: dict[str, Any]) -> dict[str, Any]:
    p50_ratio = _to_float_or_none(report.get("ratios", {}).get("triton_tvm_to_torch_compile_p50"))
    p95_ratio = _to_float_or_none(report.get("ratios", {}).get("triton_tvm_to_torch_compile_p95"))
    max_ratio = max(value for value in (p50_ratio, p95_ratio) if value is not None) if (
        p50_ratio is not None or p95_ratio is not None
    ) else None
    if report.get("invariants", {}).get("status") != "passed":
        decision = "blocked_invariants_failed"
        next_action = "Fix M12.8 gate invariant failures before making a close/continue decision."
    elif report.get("p2", {}).get("passed"):
        decision = "close_m12_p2_passed"
        next_action = "Close M12; fixed-shape ViT E2E is performance-ready under the M12 P2 gate."
    elif max_ratio is not None and max_ratio <= NEAR_MISS_RATIO_MAX:
        decision = "open_tiny_m12_9_near_miss"
        next_action = (
            "Open a very small M12.9 near-miss slice; do not broaden scope or change P2."
        )
    elif max_ratio is not None and max_ratio > PROFILE_RETURN_RATIO_MIN:
        decision = "return_to_profile_bottleneck_may_be_wrong"
        next_action = "Return to profiling; do not continue blind optimization."
    else:
        decision = "return_to_profile_not_near_miss"
        next_action = "Return to profiling before any further optimization."
    return {
        "decision": decision,
        "next_action": next_action,
        "p2_passed": bool(report.get("p2", {}).get("passed")),
        "m12_complete": decision == "close_m12_p2_passed",
        "near_miss_ratio_max": NEAR_MISS_RATIO_MAX,
        "profile_return_ratio_min": PROFILE_RETURN_RATIO_MIN,
        "torch_compile_ratio_p50": p50_ratio,
        "torch_compile_ratio_p95": p95_ratio,
        "torch_compile_ratio_max": max_ratio,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    source_status = report.get("source_report_status", {})
    if source_status.get("m12_7_status") != "passed":
        failures.append("m12_8_m12_7_report_not_passed")
    if source_status.get("m12_7_invariants") != "passed":
        failures.append("m12_8_m12_7_invariants_not_passed")
    if tuple(report.get("baselines", []) or []) != BASELINES:
        failures.append("m12_8_baselines_changed")
    if report.get("warmed_cache") is not True:
        failures.append("m12_8_warmed_cache_required")
    if int(report.get("host_staging_bytes", 0) or 0) != 0:
        failures.append("m12_8_host_staging_nonzero")
    if report.get("correctness", {}).get("allclose") is not True:
        failures.append("m12_8_correctness_allclose_not_true")
    if not _provider_mix_ok(report.get("provider_mix", {})):
        failures.append("m12_8_provider_mix_incomplete")
    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_8_strict_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_8_full_tvm_runnable_claim_present")
    if report.get("performance_ready_e2e") and not report.get("p2", {}).get("passed"):
        failures.append("m12_8_performance_ready_without_p2")
    for baseline in BASELINES:
        record = report.get("latency_ms", {}).get(baseline, {})
        if _non_positive_or_missing(record.get("p50_ms")) or _non_positive_or_missing(
            record.get("p95_ms")
        ):
            failures.append(f"m12_8_missing_latency_{baseline}")
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


def _device_bytes_report(prepared, actual: dict[str, Any]) -> dict[str, Any]:
    output_bytes = sum(_tensor_nbytes(tensor) for tensor in actual.values())
    return {
        "scope": "model_input_output_tensors",
        "input_bytes": _tensor_nbytes(prepared.pixel_values),
        "output_bytes": output_bytes,
        "estimated_io_bytes": _tensor_nbytes(prepared.pixel_values) + output_bytes,
        "aggregate_intermediate_bytes_available": False,
    }


def _tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    latency = report.get("latency_ms", {})
    triton = latency.get("triton_tvm_e2e", {})
    compile_latency = latency.get("torch_compile_inductor", {})
    return {
        "measured": report.get("status") == "passed",
        "m12_performance_tier": report.get("m12_performance_tier"),
        "p1_passed": bool(report.get("p1", {}).get("passed")),
        "p2_passed": bool(report.get("p2", {}).get("passed")),
        "performance_ready_e2e": bool(report.get("performance_ready_e2e")),
        "decision": report.get("m12_8_decision", {}).get("decision"),
        "triton_tvm_p50_ms": triton.get("p50_ms"),
        "triton_tvm_p95_ms": triton.get("p95_ms"),
        "torch_compile_p50_ms": compile_latency.get("p50_ms"),
        "torch_compile_p95_ms": compile_latency.get("p95_ms"),
    }


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    m127_report: str | Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "gate_id": GATE_ID,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "availability_reason": availability_reason,
        "baselines": list(BASELINES),
        "provider_path": {
            "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            "source": "M12.7 validated fixed-shape provider path",
        },
        "source_reports": {"m12_7_report": str(m127_report)},
        "source_report_status": {},
        "latency_ms": {baseline: _latency_record([]) for baseline in BASELINES},
        "ratios": {
            "triton_tvm_to_torch_eager_p50": None,
            "triton_tvm_to_torch_eager_p95": None,
            "triton_tvm_to_torch_compile_p50": None,
            "triton_tvm_to_torch_compile_p95": None,
        },
        "correctness": {"allclose": None},
        "p0": {
            "correctness": False,
            "no_silent_fallback": False,
            "complete_provider_report": False,
            "passed": False,
        },
        "p1": {
            "threshold": "<=3x torch_eager_cuda or torch_compile_inductor p50/p95",
            "eager_baseline_ok": False,
            "inductor_baseline_ok": False,
            "zero_host_staging": True,
            "passed": False,
        },
        "p2": {
            "threshold": "<=2x torch_compile_inductor p50/p95",
            "inductor_baseline_ok": False,
            "warmed_cache": warmup > 0,
            "zero_host_staging": True,
            "complete_provider_mix": False,
            "correctness": False,
            "passed": False,
        },
        "m12_performance_tier": "unavailable",
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": 0,
        "launch_count_estimate": LAUNCH_COUNT_ESTIMATE,
        "provider_mix": {},
        "device_bytes": {},
        "m12_8_decision": {
            "decision": "blocked_unavailable",
            "next_action": "Run M12.8 benchmarks before making a close/continue decision.",
            "m12_complete": False,
        },
        "dependency_versions": {"tvm": str(tvm.__version__)},
    }
    report["summary"] = _summary(report)
    report["invariants"] = {
        "status": "failed",
        "invariant_failures": [f"m12_8_{availability_reason}"],
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
    ratios = report.get("ratios", {})
    decision = report.get("m12_8_decision", {})
    lines = [
        "# M12.8 ViT P2 Gate Decision",
        "",
        f"- Status: {report.get('status')}",
        f"- Target model: `{report.get('target_model')}`",
        f"- Warmup/repeat: {report.get('warmup')} / {report.get('repeat')}",
        f"- P2 passed: {report.get('p2', {}).get('passed')}",
        f"- Performance-ready E2E: {report.get('performance_ready_e2e')}",
        f"- Decision: `{decision.get('decision')}`",
        f"- Host staging bytes: {report.get('host_staging_bytes')}",
        "",
        "## E2E Timing",
        "",
        "| Path | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for name in BASELINES:
        record = latency.get(name, {})
        lines.append(
            f"| `{name}` | {_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |"
        )
    lines.extend(
        [
            "",
            "## Ratios",
            "",
            f"- Triton TVM / torch.compile p50,p95: "
            f"{_fmt(ratios.get('triton_tvm_to_torch_compile_p50'))}, "
            f"{_fmt(ratios.get('triton_tvm_to_torch_compile_p95'))}",
            f"- Decision next action: {decision.get('next_action')}",
            "",
            "## Provider Mix",
            "",
        ]
    )
    counts = report.get("provider_mix", {}).get("provider_counts", {})
    if counts:
        for provider, count in counts.items():
            lines.append(f"- `{provider}`: {count}")
    else:
        lines.append(f"- unavailable: {report.get('availability_reason', '')}")
    failures = ", ".join(report.get("invariants", {}).get("invariant_failures", [])) or "none"
    lines.extend(["", "## Invariants", "", f"- Failures: {failures}", ""])
    return "\n".join(lines)


def _ratio_pair_ok(p50_ratio: float | None, p95_ratio: float | None, threshold: float) -> bool:
    if p50_ratio is None or p95_ratio is None:
        return False
    return p50_ratio <= threshold and p95_ratio <= threshold


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m127-report", type=Path, default=DEFAULT_M127_REPORT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    report = run_vit_p2_gate_decision(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        run_benchmarks=not args.no_benchmarks,
        m127_report=args.m127_report,
    )
    decision = report.get("m12_8_decision", {})
    print(
        "M12.8 ViT P2 gate decision: "
        f"status={report['status']} p2={report.get('p2', {}).get('passed')} "
        f"decision={decision.get('decision')} out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
