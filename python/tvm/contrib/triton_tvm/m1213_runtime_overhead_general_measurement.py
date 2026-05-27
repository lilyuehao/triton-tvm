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
"""M12.13 reusable runtime-overhead measurement report."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tvm

from .frontend import lower_to_ttir
from .m124_vit_e2e_dashboard import DEFAULT_REPEAT, DEFAULT_WARMUP
from .runtime import build_triton_tvm
from .translator import translate_ttir


REPORT_KIND = "triton_tvm_m12_13_runtime_overhead_general_measurement"
REPORT_ID = "m12_13_runtime_overhead_general_measurement_v1"
MILESTONE = "M12.13"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_13_runtime_overhead_general_measurement"
)
DEFAULT_M1211_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_11_backend_general_replan/report.json"
)
DEFAULT_M1212_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/"
    "m12_12_real_tl_dot_bridge_harness/report.json"
)
DEFAULT_N = 1024
DEFAULT_BLOCK = 128

REQUIRED_MEASUREMENT_BUCKETS = (
    "dlpack_conversion_ms",
    "packed_func_lookup_ms",
    "artifact_run_wall_ms",
    "prebound_packed_call_wall_ms",
    "kernel_event_ms",
    "dispatch_minus_kernel_estimate_ms",
    "launch_envelope_no_per_call_sync_ms",
)


def run_runtime_overhead_general_measurement(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    n: int = DEFAULT_N,
    block: int = DEFAULT_BLOCK,
    run_measurement: bool = True,
    m1211_report: str | Path = DEFAULT_M1211_REPORT,
    m1212_report: str | Path = DEFAULT_M1212_REPORT,
) -> dict[str, Any]:
    """Measure reusable runtime overhead and optionally write the M12.13 report."""

    if not run_measurement:
        report = _empty_report(
            status="not_run",
            availability_reason="measurement_disabled",
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            n=n,
            block=block,
            m1211_report=m1211_report,
            m1212_report=m1212_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        source_m1211 = _read_json_report(m1211_report)
        source_m1212 = _read_json_report(m1212_report)
    except FileNotFoundError as err:
        report = _empty_report(
            status="unavailable",
            availability_reason=f"source_report_missing:{err.filename}",
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            n=n,
            block=block,
            m1211_report=m1211_report,
            m1212_report=m1212_report,
        )
        _maybe_write_report(report, out_dir)
        return report
    except json.JSONDecodeError as err:
        report = _empty_report(
            status="failed",
            availability_reason=f"source_report_invalid_json:{err}",
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            n=n,
            block=block,
            m1211_report=m1211_report,
            m1212_report=m1212_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_report(
            status="unavailable",
            availability_reason="torch_unavailable",
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            n=n,
            block=block,
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            source_m1211=source_m1211,
            source_m1212=source_m1212,
        )
        _maybe_write_report(report, out_dir)
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_report(
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            n=n,
            block=block,
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            source_m1211=source_m1211,
            source_m1212=source_m1212,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        evidence = _collect_measurements(torch, seed=seed, warmup=warmup, repeat=repeat, n=n, block=block)
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_report(
            status="failed",
            availability_reason=f"m12_13_failed:{type(err).__name__}:{err}",
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            n=n,
            block=block,
            m1211_report=m1211_report,
            m1212_report=m1212_report,
            source_m1211=source_m1211,
            source_m1212=source_m1212,
        )
        _maybe_write_report(report, out_dir)
        return report

    return build_runtime_overhead_general_measurement_report(
        seed=seed,
        warmup=warmup,
        repeat=repeat,
        n=n,
        block=block,
        m1211_report=source_m1211,
        m1211_report_path=m1211_report,
        m1212_report=source_m1212,
        m1212_report_path=m1212_report,
        measurements=evidence["measurements"],
        artifact_metadata=evidence["artifact_metadata"],
        correctness=evidence["correctness"],
        dependency_versions=_dependency_versions(torch),
        out_dir=out_dir,
    )


def build_runtime_overhead_general_measurement_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    n: int,
    block: int,
    m1211_report: dict[str, Any],
    m1211_report_path: str | Path,
    m1212_report: dict[str, Any],
    m1212_report_path: str | Path,
    measurements: dict[str, dict[str, Any]],
    correctness: dict[str, Any],
    artifact_metadata: dict[str, Any] | None = None,
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build a machine-readable M12.13 report from measured or synthetic data."""

    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "measurement_surface": {
            "name": "standalone_pointwise_flat_runtime_overhead_microbench",
            "contract": "pointwise_flat",
            "source": "real_triton_jit_pointwise_flat",
            "model_specific": False,
            "uses_vit_wrapper": False,
            "uses_wrapper_buffer_replay": False,
            "n": n,
            "block": block,
            "grid": [(int(n) + int(block) - 1) // int(block)],
            "artifact": artifact_metadata or {},
        },
        "source_reports": {
            "m12_11_report": str(m1211_report_path),
            "m12_12_report": str(m1212_report_path),
        },
        "source_report_status": {
            "m12_11_status": m1211_report.get("status"),
            "m12_11_invariants": (m1211_report.get("invariants") or {}).get("status"),
            "m12_11_primary_route": m1211_report.get("primary_route"),
            "m12_11_auxiliary_order": m1211_report.get("auxiliary_order"),
            "m12_12_status": m1212_report.get("status"),
            "m12_12_invariants": (m1212_report.get("invariants") or {}).get("status"),
            "m12_12_bridge_complete": m1212_report.get("backend_general_bridge_complete"),
        },
        "overhead_targets": _overhead_targets(),
        "measurements": measurements,
        "attribution": _attribution(measurements),
        "correctness": correctness,
        "route_boundaries": _route_boundaries(),
        "next_default_action": {
            "action": "m12_14_real_tt_dot_schedule_handoff",
            "reason": "M12.13 measures reusable runtime overhead; schedule handoff is the next frozen auxiliary track.",
        },
        "performance_claim": False,
        "performance_ready_e2e": False,
        "p2_passed": False,
        "backend_general_runtime_measurement_complete": True,
        "backend_general_complete": False,
        "schedule_handoff_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
        },
        "dependency_versions": dependency_versions or {"tvm": str(tvm.__version__)},
    }
    report["invariants"] = _invariants(report)
    report["status"] = "passed" if report["invariants"]["status"] == "passed" else "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _collect_measurements(torch, *, seed: int, warmup: int, repeat: int, n: int, block: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    artifact = lower_to_ttir(_runtime_overhead_kernel(), _signature(), {"BLOCK": block})
    irmod, meta = translate_ttir(
        artifact,
        grid=((int(n) + int(block) - 1) // int(block),),
        target="cuda",
        contract="pointwise_flat",
    )
    built = build_triton_tvm(irmod, meta)

    x = torch.rand((n,), device="cuda", dtype=torch.float32).contiguous()
    y = torch.rand((n,), device="cuda", dtype=torch.float32).contiguous()
    out0 = torch.empty((n,), device="cuda", dtype=torch.float32).contiguous()
    out1 = torch.empty((n,), device="cuda", dtype=torch.float32).contiguous()
    torch_args = [x, y, out0, out1]
    packed = built.executable[meta.kernel_name]

    def _make_tvm_args() -> list[Any]:
        return [tvm.runtime.from_dlpack(arg) for arg in torch_args] + [int(n)]

    for _ in range(max(0, int(warmup))):
        tvm_args = _make_tvm_args()
        packed(*tvm_args)
    torch.cuda.synchronize()

    samples = {bucket: [] for bucket in REQUIRED_MEASUREMENT_BUCKETS}
    pending_tvm_args: list[list[Any]] = []
    for _ in range(max(1, int(repeat))):
        dlpack_start = time.perf_counter()
        tvm_args = _make_tvm_args()
        samples["dlpack_conversion_ms"].append(_elapsed_ms(dlpack_start))
        pending_tvm_args.append(tvm_args)

        lookup_start = time.perf_counter()
        looked_up = built.executable[meta.kernel_name]
        samples["packed_func_lookup_ms"].append(_elapsed_ms(lookup_start))

        artifact_start = time.perf_counter()
        built.run(tvm_args)
        samples["artifact_run_wall_ms"].append(_elapsed_ms(artifact_start))

        prebound_start = time.perf_counter()
        looked_up(*tvm_args)
        prebound_ms = _elapsed_ms(prebound_start)
        samples["prebound_packed_call_wall_ms"].append(prebound_ms)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        looked_up(*tvm_args)
        end_event.record()
        torch.cuda.synchronize()
        kernel_ms = float(start_event.elapsed_time(end_event))
        samples["kernel_event_ms"].append(kernel_ms)
        samples["dispatch_minus_kernel_estimate_ms"].append(max(0.0, prebound_ms - kernel_ms))

        envelope_start = time.perf_counter()
        looked_up(*tvm_args)
        torch.cuda.synchronize()
        samples["launch_envelope_no_per_call_sync_ms"].append(_elapsed_ms(envelope_start))

    pending_tvm_args.clear()
    correctness = {
        "allclose": bool(
            torch.allclose(out0, x + y, rtol=1e-5, atol=1e-5)
            and torch.allclose(out1, x - y, rtol=1e-5, atol=1e-5)
        ),
        "baseline": "torch_cuda_pointwise_dual_store",
    }
    measurements = {bucket: _stat_record(values) for bucket, values in samples.items()}
    artifact_metadata = {
        "kernel_name": meta.kernel_name,
        "contract": meta.contract,
        "canonical_contract": meta.canonical_contract,
        "target": meta.target,
        "execution_kind": meta.execution_kind,
    }
    return {
        "measurements": measurements,
        "artifact_metadata": artifact_metadata,
        "correctness": correctness,
    }


def _runtime_overhead_kernel():
    import triton  # pylint: disable=import-outside-toplevel
    import triton.language as tl  # pylint: disable=import-outside-toplevel

    @triton.jit
    def kernel(x, y, out0, out1, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        vx = tl.load(x + offsets, mask=mask, other=0.0)
        vy = tl.load(y + offsets, mask=mask, other=0.0)
        tl.store(out0 + offsets, vx + vy, mask=mask)
        tl.store(out1 + offsets, vx - vy, mask=mask)

    return kernel


def _signature() -> dict[str, str]:
    return {
        "x": "*fp32",
        "y": "*fp32",
        "out0": "*fp32",
        "out1": "*fp32",
        "n": "i64",
        "BLOCK": "constexpr",
    }


def _overhead_targets() -> list[dict[str, Any]]:
    return [
        {
            "target": "tvm_packed_call_overhead",
            "measurement_buckets": ["packed_func_lookup_ms", "prebound_packed_call_wall_ms"],
        },
        {
            "target": "dlpack_conversion_overhead",
            "measurement_buckets": ["dlpack_conversion_ms"],
        },
        {
            "target": "artifact_dispatch_overhead",
            "measurement_buckets": ["artifact_run_wall_ms", "dispatch_minus_kernel_estimate_ms"],
        },
        {
            "target": "launch_overhead_without_per_call_sync",
            "measurement_buckets": ["kernel_event_ms", "launch_envelope_no_per_call_sync_ms"],
        },
    ]


def _route_boundaries() -> dict[str, Any]:
    return {
        "no_vit_wrapper_replay": True,
        "no_wrapper_buffer_replay": True,
        "no_wrapper_line_replay": True,
        "no_fused_qkv_artifact": True,
        "no_model_specific_executor": True,
        "no_p2_or_performance_claim": True,
        "no_schedule_handoff_claim": True,
    }


def _attribution(measurements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for bucket in REQUIRED_MEASUREMENT_BUCKETS:
        p50 = _to_float_or_none((measurements.get(bucket) or {}).get("p50_ms"))
        if p50 is not None:
            candidates.append((bucket, p50))
    if not candidates:
        return {"dominant_bucket": None, "dominant_p50_ms": None}
    bucket, value = max(candidates, key=lambda item: item[1])
    return {
        "dominant_bucket": bucket,
        "dominant_p50_ms": value,
        "interpretation": (
            "M12.13 is attribution evidence only; it does not apply an optimization "
            "or change the M12 P2 gate."
        ),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("milestone") != MILESTONE:
        failures.append("m12_13_wrong_milestone")
    source_status = report.get("source_report_status") or {}
    if source_status.get("m12_11_status") != "route_frozen":
        failures.append("m12_13_m12_11_not_route_frozen")
    if source_status.get("m12_11_invariants") != "passed":
        failures.append("m12_13_m12_11_invariants_not_passed")
    if source_status.get("m12_11_primary_route") != "real_tl_dot_bridge":
        failures.append("m12_13_m12_11_primary_route_changed")
    if source_status.get("m12_11_auxiliary_order") != [
        "runtime_overhead",
        "tt_dot_schedule_handoff",
    ]:
        failures.append("m12_13_m12_11_auxiliary_order_changed")
    if source_status.get("m12_12_status") != "passed":
        failures.append("m12_13_m12_12_not_passed")
    if source_status.get("m12_12_invariants") != "passed":
        failures.append("m12_13_m12_12_invariants_not_passed")

    surface = report.get("measurement_surface") or {}
    if surface.get("contract") != "pointwise_flat":
        failures.append("m12_13_measurement_surface_not_pointwise_flat")
    if surface.get("model_specific") is not False or surface.get("uses_vit_wrapper") is not False:
        failures.append("m12_13_measurement_surface_model_specific")
    if surface.get("uses_wrapper_buffer_replay") is not False:
        failures.append("m12_13_measurement_surface_wrapper_replay")

    measurements = report.get("measurements") or {}
    for bucket in REQUIRED_MEASUREMENT_BUCKETS:
        record = measurements.get(bucket)
        if not isinstance(record, dict):
            failures.append(f"m12_13_missing_measurement_{bucket}")
            continue
        if int(record.get("samples", 0) or 0) <= 0:
            failures.append(f"m12_13_measurement_has_no_samples_{bucket}")
        if _to_float_or_none(record.get("p50_ms")) is None:
            failures.append(f"m12_13_measurement_missing_p50_{bucket}")
        if _to_float_or_none(record.get("p95_ms")) is None:
            failures.append(f"m12_13_measurement_missing_p95_{bucket}")

    if (report.get("correctness") or {}).get("allclose") is not True:
        failures.append("m12_13_correctness_allclose_not_true")

    boundaries = report.get("route_boundaries") or {}
    for key in (
        "no_vit_wrapper_replay",
        "no_wrapper_buffer_replay",
        "no_wrapper_line_replay",
        "no_fused_qkv_artifact",
        "no_model_specific_executor",
        "no_p2_or_performance_claim",
        "no_schedule_handoff_claim",
    ):
        if boundaries.get(key) is not True:
            failures.append(f"m12_13_boundary_violation_{key}")

    if (report.get("next_default_action") or {}).get(
        "action"
    ) != "m12_14_real_tt_dot_schedule_handoff":
        failures.append("m12_13_next_action_not_m12_14")
    for key in (
        "performance_claim",
        "performance_ready_e2e",
        "p2_passed",
        "backend_general_complete",
        "schedule_handoff_claim",
        "strict_full_tvm_native",
    ):
        if report.get(key) is not False:
            failures.append(f"m12_13_forbidden_claim_{key}")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_13_full_tvm_runnable_claim_present")
    completion = report.get("completion_gate") or {}
    if completion.get("m12_complete") is not False:
        failures.append("m12_13_m12_complete_claim_present")
    if completion.get("requires_fixed_shape_vit_p2") is not True:
        failures.append("m12_13_p2_gate_not_preserved")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "m12_13_complete": report.get("status") == "passed",
        "runtime_overhead_measurement_complete": bool(
            report.get("status") == "passed"
            and (report.get("invariants") or {}).get("status") == "passed"
        ),
        "dominant_bucket": (report.get("attribution") or {}).get("dominant_bucket"),
        "performance_claim": False,
        "m12_complete": False,
        "next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(
    *,
    status: str,
    availability_reason: str,
    seed: int,
    warmup: int,
    repeat: int,
    n: int,
    block: int,
    m1211_report: str | Path,
    m1212_report: str | Path,
    source_m1211: dict[str, Any] | None = None,
    source_m1212: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "measurement_surface": {
            "name": "standalone_pointwise_flat_runtime_overhead_microbench",
            "contract": "pointwise_flat",
            "source": "real_triton_jit_pointwise_flat",
            "model_specific": False,
            "uses_vit_wrapper": False,
            "uses_wrapper_buffer_replay": False,
            "n": n,
            "block": block,
            "grid": [(int(n) + int(block) - 1) // int(block)],
            "artifact": {},
        },
        "source_reports": {
            "m12_11_report": str(m1211_report),
            "m12_12_report": str(m1212_report),
        },
        "source_report_status": {
            "m12_11_status": (source_m1211 or {}).get("status"),
            "m12_11_invariants": ((source_m1211 or {}).get("invariants") or {}).get("status"),
            "m12_11_primary_route": (source_m1211 or {}).get("primary_route"),
            "m12_11_auxiliary_order": (source_m1211 or {}).get("auxiliary_order"),
            "m12_12_status": (source_m1212 or {}).get("status"),
            "m12_12_invariants": ((source_m1212 or {}).get("invariants") or {}).get("status"),
            "m12_12_bridge_complete": (source_m1212 or {}).get("backend_general_bridge_complete"),
        },
        "overhead_targets": _overhead_targets(),
        "measurements": {},
        "attribution": {"dominant_bucket": None, "dominant_p50_ms": None},
        "correctness": {"allclose": None, "baseline": "torch_cuda_pointwise_dual_store"},
        "route_boundaries": _route_boundaries(),
        "next_default_action": {"action": "m12_14_real_tt_dot_schedule_handoff"},
        "performance_claim": False,
        "performance_ready_e2e": False,
        "p2_passed": False,
        "backend_general_runtime_measurement_complete": False,
        "backend_general_complete": False,
        "schedule_handoff_claim": False,
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
            "m12_13_complete": False,
            "runtime_overhead_measurement_complete": False,
            "dominant_bucket": None,
            "performance_claim": False,
            "m12_complete": False,
            "next_action": "m12_14_real_tt_dot_schedule_handoff",
        },
        "dependency_versions": {"tvm": str(tvm.__version__)},
    }
    return report


def _elapsed_ms(start: float, end: float | None = None) -> float:
    if end is None:
        end = time.perf_counter()
    return max(0.0, (end - start) * 1000.0)


def _stat_record(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "samples": 0,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "samples_ms": [],
        }
    return {
        "samples": len(ordered),
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": ordered,
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _to_float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _read_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dependency_versions(torch) -> dict[str, Any]:
    versions = {
        "torch": str(getattr(torch, "__version__", "")),
        "tvm": str(tvm.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        versions["cuda_device"] = torch.cuda.get_device_name(0)
    return versions


def _require_torch():
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError("PyTorch is required for M12.13 runtime-overhead measurement") from err
    return torch


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
    surface = report.get("measurement_surface") or {}
    attribution = report.get("attribution") or {}
    lines = [
        "# M12.13 Runtime Overhead General Measurement",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Surface: `{surface.get('name')}`",
        f"- Contract: `{surface.get('contract')}`",
        f"- Warmup/repeat: {report.get('warmup')} / {report.get('repeat')}",
        f"- Dominant bucket: `{attribution.get('dominant_bucket')}`",
        f"- Performance claim: `{report.get('performance_claim')}`",
        f"- P2 passed: `{report.get('p2_passed')}`",
        f"- Next action: `{(report.get('next_default_action') or {}).get('action')}`",
        "",
        "## Measurements",
        "",
        "| Bucket | Samples | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    measurements = report.get("measurements") or {}
    for bucket in REQUIRED_MEASUREMENT_BUCKETS:
        record = measurements.get(bucket) or {}
        lines.append(
            f"| `{bucket}` | {record.get('samples', 0)} | "
            f"{_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |"
        )
    lines.extend(["", "## Boundaries", ""])
    for key, value in (report.get("route_boundaries") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Invariants", ""])
    invariants = report.get("invariants") or {}
    lines.append(f"- Status: `{invariants.get('status')}`")
    for failure in invariants.get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m1213_runtime_overhead_general_measurement``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--no-measurement", action="store_true")
    args = parser.parse_args(argv)

    report = run_runtime_overhead_general_measurement(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        n=args.n,
        block=args.block,
        run_measurement=not args.no_measurement,
    )
    print(
        f"M12.13 runtime overhead: status={report['status']} "
        f"dominant_bucket={(report.get('attribution') or {}).get('dominant_bucket')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
