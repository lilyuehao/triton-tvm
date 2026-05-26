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
"""M11.6 vision Grid2D runtime/perf readiness dashboard."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tvm
from tvm.contrib.triton_tvm import build_triton_tvm
from tvm.contrib.triton_tvm.contracts import validate_pointwise_grid2d_static_contract
from tvm.contrib.triton_tvm.translator import TritonTVMMeta
from tvm.contrib.triton_tvm.vision import (
    M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
    M11_GRID2D_PROVIDER_NATIVE_TVM,
    M11_GRID2D_READINESS_VERSION,
    M11_GRID2D_RUNTIME_READY,
    M11_GRID_FAMILY_BN_SILU_FUSION,
    M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
    VISION_M11_6_INTERFACE_STATUS,
    build_native_grid2d_pointwise_tirx_source,
    VisionGrid2DPointwiseSemantics,
)

try:
    import torch
except ImportError:  # pragma: no cover - availability is reflected in reports.
    torch = None


REPORT_KIND = "triton_tvm_m11_6_vision_perf_dashboard"
BASELINE_ID = "m11_6_native_grid2d_pointwise_readiness_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m11/m11_6_vision_perf_dashboard"
)
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 20


@dataclass(frozen=True)
class VisionGrid2DDashboardCase:
    """One native Grid2D pointwise/fusion dashboard case."""

    case_name: str
    model: str
    grid_family: str
    x_extent: int
    y_extent: int
    op_kind: str
    block_size: int = 128

    @property
    def shape_hints(self) -> str:
        return f"x={self.x_extent}, y={self.y_extent}"


def default_grid2d_dashboard_suite() -> tuple[VisionGrid2DDashboardCase, ...]:
    """Return the fixed M11.6 native Grid2D readiness cases."""
    return (
        VisionGrid2DDashboardCase(
            case_name="vit_grid2d_conv_adjacent_add_1024x4",
            model="vit_tiny_random",
            grid_family=M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
            x_extent=1024,
            y_extent=4,
            op_kind="add",
        ),
        VisionGrid2DDashboardCase(
            case_name="yolo_grid2d_conv_adjacent_mul_4096x4",
            model="yolov8n_yaml_random",
            grid_family=M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
            x_extent=4096,
            y_extent=4,
            op_kind="mul",
        ),
        VisionGrid2DDashboardCase(
            case_name="yolo_grid2d_conv_adjacent_silu_16x512",
            model="yolov8n_yaml_random",
            grid_family=M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
            x_extent=16,
            y_extent=512,
            op_kind="silu",
        ),
        VisionGrid2DDashboardCase(
            case_name="yolo_grid2d_bn_silu_32x256",
            model="yolov8n_yaml_random",
            grid_family=M11_GRID_FAMILY_BN_SILU_FUSION,
            x_extent=32,
            y_extent=256,
            op_kind="bn_affine_silu",
        ),
        VisionGrid2DDashboardCase(
            case_name="yolo_grid2d_conv_adjacent_select_256x64",
            model="yolov8n_yaml_random",
            grid_family=M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
            x_extent=256,
            y_extent=64,
            op_kind="select_relu",
        ),
    )


def run_vision_grid2d_dashboard(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M11.6 native Grid2D dashboard and optionally write artifacts."""
    cases = [
        _run_case(case, warmup=warmup, repeat=repeat, run_benchmarks=run_benchmarks)
        for case in default_grid2d_dashboard_suite()
    ]
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "readiness_status": VISION_M11_6_INTERFACE_STATUS,
        "grid2d_readiness_version": M11_GRID2D_READINESS_VERSION,
        "claim_scope": (
            "native TVM Grid2D pointwise/fusion readiness dashboard; host-staged "
            "conv providers are excluded from performance guards"
        ),
        "host_staged_provider_policy": {
            "provider_kind": "python_torch_host_staged",
            "performance_claim": False,
            "excluded_from_perf_guard": True,
        },
        "warmup": warmup,
        "repeat": repeat,
        "cases": cases,
        "dependency_versions": _dependency_versions(),
    }
    report["summary"] = _summary(cases)
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


def _run_case(
    case: VisionGrid2DDashboardCase,
    *,
    warmup: int,
    repeat: int,
    run_benchmarks: bool,
) -> dict[str, Any]:
    semantics = VisionGrid2DPointwiseSemantics(
        case_name=case.case_name,
        model=case.model,
        grid_family=case.grid_family,
        x_extent=case.x_extent,
        y_extent=case.y_extent,
        op_kind=case.op_kind,
        block_size=case.block_size,
    )
    record: dict[str, Any] = {
        **asdict(case),
        "shape_hints": case.shape_hints,
        "contract": VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        "implementation_kind": M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        "runtime_status": M11_GRID2D_RUNTIME_READY,
        "provider_kind": M11_GRID2D_PROVIDER_NATIVE_TVM,
        "host_staging_bytes": 0,
        "performance_claim": True,
        "excluded_from_perf_guard": False,
        "native_inductor_us": None,
        "triton_tvm_us": None,
        "relative_to_inductor": None,
        "gbps_or_effective_bandwidth": None,
        "cache_hit": False,
        "correctness_allclose": None,
        "max_abs_error": None,
        "perf_guard_status": "unavailable",
        "availability_reason": "",
    }
    if not run_benchmarks:
        record["availability_reason"] = "benchmarks_disabled"
        return record
    if torch is None:
        record["availability_reason"] = "torch_unavailable"
        return record
    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        record["availability_reason"] = "cuda_unavailable"
        return record
    try:
        return _benchmark_case(record, semantics, warmup=warmup, repeat=repeat)
    except Exception as err:  # pylint: disable=broad-except
        record["availability_reason"] = f"benchmark_failed:{type(err).__name__}:{err}"
        return record


def _benchmark_case(
    record: dict[str, Any],
    semantics: VisionGrid2DPointwiseSemantics,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(_case_seed(semantics.case_name))
    shape = (semantics.y_extent, semantics.x_extent)
    x_np = rng.normal(size=shape).astype("float32")
    y_np = rng.normal(size=shape).astype("float32")
    expected_np = _numpy_grid2d_reference(semantics.op_kind, x_np, y_np)

    source = build_native_grid2d_pointwise_tirx_source(semantics)
    irmod = tvm.script.from_source(source)
    validate_pointwise_grid2d_static_contract(irmod)
    built = build_triton_tvm(irmod, _grid2d_meta(semantics))

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty(shape, "float32", dev)
    tvm_args = [
        tvm.runtime.tensor(x_np, dev),
        tvm.runtime.tensor(y_np, dev),
        out_tvm,
    ]
    built.run(tvm_args)
    tvm_out = out_tvm.numpy()
    for _ in range(warmup):
        built.run(tvm_args)
    timer = built.executable.mod.time_evaluator(
        semantics.case_name,
        dev,
        number=1,
        repeat=repeat,
        min_repeat_ms=0,
    )
    tvm_values = [float(value) * 1e6 for value in timer(*tvm_args).results]

    torch_x = torch.tensor(x_np, device="cuda")
    torch_y = torch.tensor(y_np, device="cuda")
    torch_values = _time_cuda_callable(
        lambda: _torch_grid2d_reference(semantics.op_kind, torch_x, torch_y),
        warmup=warmup,
        repeat=repeat,
    )
    tvm_p50 = _percentile(tvm_values, 50)
    torch_p50 = _percentile(torch_values, 50)
    max_abs_error = float(np.max(np.abs(tvm_out - expected_np)))
    allclose = bool(np.allclose(tvm_out, expected_np, rtol=1e-5, atol=1e-5))
    record.update(
        {
            "native_inductor_us": torch_p50,
            "triton_tvm_us": tvm_p50,
            "relative_to_inductor": tvm_p50 / torch_p50 if torch_p50 else None,
            "gbps_or_effective_bandwidth": _effective_gbps(semantics, tvm_p50),
            "correctness_allclose": allclose,
            "max_abs_error": max_abs_error,
            "perf_guard_status": "measured",
            "availability_reason": "",
        }
    )
    return record


def _grid2d_meta(semantics: VisionGrid2DPointwiseSemantics) -> TritonTVMMeta:
    return TritonTVMMeta(
        kernel_name=semantics.case_name,
        signature={},
        constexprs={},
        grid=(
            _ceildiv(semantics.x_extent, semantics.block_size),
            semantics.y_extent,
        ),
        target="cuda",
        target_kind="cuda",
        contract=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        canonical_contract=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        requested_contract=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        emit="tir",
        translator_version="m11_6_grid2d_dashboard",
        contract_version=VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        target_policy_version="m11_6_native_grid2d_pointwise",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=semantics.element_count,
        reduction_extent_param="",
        reduction_extent_kind="not_applicable",
        reduction_extent_value=None,
        buffer_extents={
            "x": f"T.int64({semantics.element_count})",
            "y": f"T.int64({semantics.element_count})",
            "out": f"T.int64({semantics.element_count})",
        },
        block_size=semantics.block_size,
        indexing_kind="grid2d_affine_pointwise",
        execution_kind=M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        accumulator_dtype_policy="fp32_elementwise",
        epsilon_policy="not_applicable",
        mask_policy="guarded_x_axis_store",
        axis_policy="program_id_x_y",
        layout_policy="rank2_row_major_static",
        launch_policy_id="m11_6_native_grid2d_static",
        abi=[
            {"name": "x", "kind": "pointer", "dtype": "float32"},
            {"name": "y", "kind": "pointer", "dtype": "float32"},
            {"name": "out", "kind": "pointer", "dtype": "float32"},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"{semantics.case_name}_m11_6_grid2d_dashboard",
        implementation_kind=M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        extern_symbol="",
    )


def _numpy_grid2d_reference(op_kind: str, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
    if op_kind == "add":
        return x_np + y_np
    if op_kind == "mul":
        return x_np * y_np
    if op_kind == "sub":
        return x_np - y_np
    if op_kind == "select_relu":
        return np.where(x_np > 0, x_np, y_np).astype("float32")
    if op_kind == "silu":
        return (x_np / (1.0 + np.exp(-x_np))).astype("float32")
    if op_kind == "bn_affine_silu":
        affine = x_np * y_np
        return (affine / (1.0 + np.exp(-affine))).astype("float32")
    raise ValueError(f"unsupported op_kind={op_kind!r}")


def _torch_grid2d_reference(op_kind: str, x_tensor, y_tensor):
    if op_kind == "add":
        return x_tensor + y_tensor
    if op_kind == "mul":
        return x_tensor * y_tensor
    if op_kind == "sub":
        return x_tensor - y_tensor
    if op_kind == "select_relu":
        return torch.where(x_tensor > 0, x_tensor, y_tensor)
    if op_kind == "silu":
        return x_tensor / (1.0 + torch.exp(-x_tensor))
    if op_kind == "bn_affine_silu":
        affine = x_tensor * y_tensor
        return affine / (1.0 + torch.exp(-affine))
    raise ValueError(f"unsupported op_kind={op_kind!r}")


def _time_cuda_callable(fn, *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [case for case in cases if case.get("perf_guard_status") == "measured"]
    return {
        "total_cases": len(cases),
        "measured_cases": len(measured),
        "allclose_cases": sum(1 for case in measured if case.get("correctness_allclose")),
        "native_tvm_grid2d_cases": sum(
            1
            for case in cases
            if case.get("implementation_kind") == M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM
        ),
        "host_staged_cases": sum(
            1 for case in cases if str(case.get("provider_kind", "")).startswith("python_torch")
        ),
        "performance_claim_cases": sum(1 for case in cases if case.get("performance_claim")),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for case in report.get("cases", []):
        if case.get("provider_kind") != M11_GRID2D_PROVIDER_NATIVE_TVM:
            failures.append("m11_6_grid2d_provider_not_native")
        if case.get("host_staging_bytes") != 0:
            failures.append("m11_6_grid2d_host_staging_nonzero")
        if case.get("performance_claim") is not True:
            failures.append("m11_6_grid2d_performance_claim_missing")
        if case.get("implementation_kind") != M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM:
            failures.append("m11_6_grid2d_implementation_not_native")
    if report.get("host_staged_provider_policy", {}).get("performance_claim") is not False:
        failures.append("m11_6_host_staged_provider_performance_claim_not_false")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M11.6 Vision Grid2D Performance Dashboard",
        "",
        f"- Report kind: `{report['report_kind']}`",
        f"- Baseline id: `{report['baseline_id']}`",
        f"- Readiness status: `{report['readiness_status']}`",
        f"- Measured cases: {report['summary']['measured_cases']} / {report['summary']['total_cases']}",
        f"- Allclose cases: {report['summary']['allclose_cases']}",
        f"- Invariants: {report['invariants']['status']}",
        "",
        "| Case | Model | Family | Shape hints | Contract | Runtime | Provider | TVM us | Ref us | Relative | GB/s |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["case_name"]),
                    str(case["model"]),
                    str(case["grid_family"]),
                    str(case["shape_hints"]),
                    str(case["contract"]),
                    str(case["runtime_status"]),
                    str(case["provider_kind"]),
                    _fmt(case.get("triton_tvm_us")),
                    _fmt(case.get("native_inductor_us")),
                    _fmt(case.get("relative_to_inductor")),
                    _fmt(case.get("gbps_or_effective_bandwidth")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Host-staged convolution providers are excluded from this performance guard.",
            "- Full TVM model runnable remains a separate stricter gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _dependency_versions() -> dict[str, Any]:
    info: dict[str, Any] = {
        "tvm": tvm.__version__,
        "torch": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device": "",
    }
    if torch is not None and torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
    return info


def _effective_gbps(semantics: VisionGrid2DPointwiseSemantics, us: float) -> float:
    if us <= 0:
        return 0.0
    return semantics.total_io_bytes / (us * 1e-6) / 1e9


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _ceildiv(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _case_seed(name: str) -> int:
    return sum(ord(ch) for ch in name) % (2**31)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    report = run_vision_grid2d_dashboard(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report["summary"]
    print(
        "M11.6 Grid2D dashboard: "
        f"measured={summary['measured_cases']}/{summary['total_cases']} "
        f"allclose={summary['allclose_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
