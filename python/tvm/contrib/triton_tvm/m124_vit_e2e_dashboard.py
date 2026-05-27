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
"""M12.4 fixed-shape ViT E2E performance dashboard."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tvm

from .m123_vit_e2e_runner import (
    TARGET_MODEL,
    _correctness_report,
    _dependency_versions,
    _provider_mix_report,
    _require_torch,
    execute_vit_fixed_shape_provider_path,
    prepare_vit_fixed_shape_e2e,
)


REPORT_KIND = "triton_tvm_m12_4_vit_fixed_shape_e2e_dashboard"
DASHBOARD_ID = "m12_4_vit_fixed_shape_e2e_dashboard_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_4_vit_e2e_dashboard"
)
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 20
LAUNCH_COUNT_ESTIMATE = 16
P1_THRESHOLD = 3.0
P2_THRESHOLD = 2.0


def run_vit_e2e_dashboard(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M12.4 ViT E2E dashboard and optionally write artifacts."""

    if not run_benchmarks:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
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
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        actual, patch_state = execute_vit_fixed_shape_provider_path(prepared)
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
                    lambda: execute_vit_fixed_shape_provider_path(prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        ratios = _ratio_report(latency_ms)
        host_staging_bytes = 0
        p0 = _p0_report(correctness, provider_mix)
        p1 = _p1_report(p0, ratios, host_staging_bytes)
        p2 = _p2_report(p0, ratios, host_staging_bytes, warmup)
        tier = _performance_tier(p0, p1, p2)

        report: dict[str, Any] = {
            "report_kind": REPORT_KIND,
            "schema_version": 1,
            "dashboard_id": DASHBOARD_ID,
            "status": "measured",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_model": TARGET_MODEL,
            "fixed_shape": True,
            "seed": seed,
            "warmup": warmup,
            "repeat": repeat,
            "warmed_cache": warmup > 0,
            "baselines": ["torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e"],
            "latency_ms": latency_ms,
            "ratios": ratios,
            "correctness": correctness,
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "m12_performance_tier": tier,
            "performance_ready_e2e": bool(p2["passed"]),
            "performance_claim": bool(p2["passed"]),
            "strict_full_tvm_native": False,
            "full_tvm_runnable_models": 0,
            "host_staging_bytes": host_staging_bytes,
            "launch_count_estimate": LAUNCH_COUNT_ESTIMATE,
            "provider_mix": provider_mix,
            "device_bytes": _device_bytes_report(prepared, actual),
            "notes": [
                "M12.4 measures the explicit M12.3 provider path with CUDA event timing.",
                "Strict full-native TVM remains false because wrapper providers remain explicit.",
            ],
            "dependency_versions": _dependency_versions(prepared.torch),
        }
        report["summary"] = _summary(report)
        report["invariants"] = _invariants(report)
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "vit_tiny_random_wrapper.py").write_text(
                prepared.wrapper_source, encoding="utf-8"
            )
        _maybe_write_report(report, out_dir)
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"benchmark_failed:{type(err).__name__}:{err}",
        )
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
    inductor = latency_ms["torch_compile_inductor"]
    return {
        "triton_tvm_to_torch_eager_p50": _safe_ratio(triton["p50_ms"], eager["p50_ms"]),
        "triton_tvm_to_torch_eager_p95": _safe_ratio(triton["p95_ms"], eager["p95_ms"]),
        "triton_tvm_to_torch_compile_p50": _safe_ratio(triton["p50_ms"], inductor["p50_ms"]),
        "triton_tvm_to_torch_compile_p95": _safe_ratio(triton["p95_ms"], inductor["p95_ms"]),
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


def _ratio_pair_ok(p50_ratio: float | None, p95_ratio: float | None, threshold: float) -> bool:
    if p50_ratio is None or p95_ratio is None:
        return False
    return p50_ratio <= threshold and p95_ratio <= threshold


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


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
    inductor = latency.get("torch_compile_inductor", {})
    return {
        "measured": report.get("status") == "measured",
        "m12_performance_tier": report.get("m12_performance_tier"),
        "p1_passed": bool(report.get("p1", {}).get("passed")),
        "p2_passed": bool(report.get("p2", {}).get("passed")),
        "performance_ready_e2e": bool(report.get("performance_ready_e2e")),
        "triton_tvm_p50_ms": triton.get("p50_ms"),
        "triton_tvm_p95_ms": triton.get("p95_ms"),
        "torch_compile_p50_ms": inductor.get("p50_ms"),
        "torch_compile_p95_ms": inductor.get("p95_ms"),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("strict_full_tvm_native") is not False:
        failures.append("m12_4_strict_full_native_claim_present")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_4_full_tvm_runnable_claim_present")
    if int(report.get("host_staging_bytes", 0) or 0) != 0:
        failures.append("m12_4_host_staging_nonzero")
    if report.get("performance_ready_e2e") and not report.get("p2", {}).get("passed"):
        failures.append("m12_4_performance_ready_without_p2")
    if report.get("status") == "measured":
        for baseline in ("torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e"):
            record = report.get("latency_ms", {}).get(baseline, {})
            if record.get("p50_ms") in (None, 0.0) or record.get("p95_ms") in (None, 0.0):
                failures.append(f"m12_4_missing_latency_{baseline}")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
) -> dict[str, Any]:
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "dashboard_id": DASHBOARD_ID,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "availability_reason": availability_reason,
        "baselines": ["torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e"],
        "latency_ms": {
            "torch_eager_cuda": _latency_record([]),
            "torch_compile_inductor": _latency_record([]),
            "triton_tvm_e2e": _latency_record([]),
        },
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
        "device_bytes": {
            "scope": "model_input_output_tensors",
            "input_bytes": None,
            "output_bytes": None,
            "estimated_io_bytes": None,
            "aggregate_intermediate_bytes_available": False,
        },
        "dependency_versions": {"tvm": str(tvm.__version__)},
    }
    report["summary"] = _summary(report)
    report["invariants"] = _invariants(report)
    return report


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
    lines = [
        "# M12.4 ViT E2E Performance Dashboard",
        "",
        f"- Status: {report.get('status')}",
        f"- Target model: `{report.get('target_model')}`",
        f"- Warmup/repeat: {report.get('warmup')} / {report.get('repeat')}",
        f"- Performance tier: {report.get('m12_performance_tier')}",
        f"- P1 passed: {report.get('p1', {}).get('passed')}",
        f"- P2 passed: {report.get('p2', {}).get('passed')}",
        f"- Performance-ready E2E: {report.get('performance_ready_e2e')}",
        f"- Host staging bytes: {report.get('host_staging_bytes')}",
        f"- Launch estimate: {report.get('launch_count_estimate')}",
        "",
        "| Baseline | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for name in ("torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e"):
        record = latency.get(name, {})
        lines.append(
            f"| `{name}` | {_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |"
        )
    lines.extend(
        [
            "",
            "## Ratios",
            "",
            f"- Triton TVM / eager p50,p95: "
            f"{_fmt(ratios.get('triton_tvm_to_torch_eager_p50'))}, "
            f"{_fmt(ratios.get('triton_tvm_to_torch_eager_p95'))}",
            f"- Triton TVM / torch.compile p50,p95: "
            f"{_fmt(ratios.get('triton_tvm_to_torch_compile_p50'))}, "
            f"{_fmt(ratios.get('triton_tvm_to_torch_compile_p95'))}",
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
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    report = run_vit_e2e_dashboard(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report["summary"]
    print(
        "M12.4 ViT E2E dashboard: "
        f"status={report['status']} tier={summary['m12_performance_tier']} "
        f"p1={summary['p1_passed']} p2={summary['p2_passed']} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"measured", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
