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
"""M11.7 vision native/device hotpath dashboard."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tvm
from tvm.contrib.triton_tvm import build_triton_tvm
from tvm.contrib.triton_tvm.contracts import validate_vision_conv2d_contract
from tvm.contrib.triton_tvm.m116_vision_dashboard import run_vision_grid2d_dashboard
from tvm.contrib.triton_tvm.translator import TritonTVMMeta
from tvm.contrib.triton_tvm.vision import (
    M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
    VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_M11_7_INTERFACE_STATUS,
    VISION_PROVIDER_DEVICE_TORCH_CUDA,
    VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    TargetVisionPolicy,
    VisionConv2DSemantics,
    build_extern_conv2d_tirx_source,
    register_device_torch_cuda_extern_conv2d,
)

try:
    import torch
except ImportError:  # pragma: no cover - availability is reflected in reports.
    torch = None


REPORT_KIND = "triton_tvm_m11_7_vision_hotpath_dashboard"
BASELINE_ID = "m11_7_vision_native_hotpath_dashboard_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m11/m11_7_vision_hotpath_dashboard"
)
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 20


@dataclass(frozen=True)
class VisionHotpathCase:
    """One M11.7 vision hotpath dashboard case."""

    case_name: str
    model: str
    op_family: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1


def default_hotpath_suite() -> tuple[VisionHotpathCase, ...]:
    """Return the fixed M11.7 conv hotpath dashboard cases."""
    return (
        VisionHotpathCase(
            "yolo_conv1x1_as_matmul_16x16x32",
            "yolov8n_yaml_random",
            "conv1x1_as_matmul",
            (1, 16, 16, 16),
            (32, 16, 1, 1),
            (1, 32, 16, 16),
        ),
        VisionHotpathCase(
            "yolo_conv1x1_as_matmul_32x8x64",
            "yolov8n_yaml_random",
            "conv1x1_as_matmul",
            (1, 32, 8, 8),
            (64, 32, 1, 1),
            (1, 64, 8, 8),
        ),
        VisionHotpathCase(
            "vit_conv1x1_as_matmul_64x2x64",
            "vit_tiny_random",
            "conv1x1_as_matmul",
            (1, 64, 2, 2),
            (64, 64, 1, 1),
            (1, 64, 2, 2),
        ),
        VisionHotpathCase(
            "vit_patch_device_conv2d_32_to_2",
            "vit_tiny_random",
            "device_conv2d_provider",
            (1, 3, 32, 32),
            (64, 3, 16, 16),
            (1, 64, 2, 2),
            stride=(16, 16),
        ),
        VisionHotpathCase(
            "yolo_device_conv2d_stride2_64_to_32",
            "yolov8n_yaml_random",
            "device_conv2d_provider",
            (1, 3, 64, 64),
            (16, 3, 3, 3),
            (1, 16, 32, 32),
            stride=(2, 2),
            padding=(1, 1),
        ),
        VisionHotpathCase(
            "yolo_device_conv2d_stride1_32_to_32",
            "yolov8n_yaml_random",
            "device_conv2d_provider",
            (1, 16, 32, 32),
            (16, 16, 3, 3),
            (1, 16, 32, 32),
            padding=(1, 1),
        ),
    )


def run_vision_hotpath_dashboard(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M11.7 vision hotpath dashboard and optionally write artifacts."""
    conv_cases = [
        _run_conv_case(case, warmup=warmup, repeat=repeat, run_benchmarks=run_benchmarks)
        for case in default_hotpath_suite()
    ]
    grid_report = run_vision_grid2d_dashboard(
        out_dir=None,
        warmup=warmup,
        repeat=repeat,
        run_benchmarks=run_benchmarks,
    )
    grid_cases = [_m117_grid_case(case) for case in grid_report.get("cases", [])[:4]]
    cases = conv_cases + grid_cases
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "hotpath_status": VISION_M11_7_INTERFACE_STATUS,
        "claim_scope": (
            "M11.7 dashboard for device-side conv2d provider, 1x1 conv hotpath "
            "classification, and native TVM Grid2D cases; host-staged providers are "
            "excluded from performance claims"
        ),
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


def _run_conv_case(
    case: VisionHotpathCase,
    *,
    warmup: int,
    repeat: int,
    run_benchmarks: bool,
) -> dict[str, Any]:
    semantics = _conv_semantics(case)
    decision = TargetVisionPolicy(vision_runtime_provider=VISION_PROVIDER_DEVICE_TORCH_CUDA).decide(
        semantics,
        vision_contract_ok=True,
    )
    record: dict[str, Any] = {
        **asdict(case),
        "contract": semantics.vision_contract,
        "implementation_kind": case.op_family,
        "provider_kind": decision.vision_provider_kind,
        "runtime_status": decision.vision_runtime_status,
        "runtime_claim": decision.vision_runtime_claim,
        "performance_claim": decision.vision_performance_claim,
        "host_staging_bytes": decision.vision_host_staging_bytes,
        "runtime_launch_count": decision.vision_runtime_launch_count,
        "artifact_call_count": decision.vision_artifact_call_count,
        "total_io_bytes": decision.vision_total_io_bytes,
        "triton_tvm_us": None,
        "torch_or_inductor_us": None,
        "relative_to_torch": None,
        "correctness_allclose": None,
        "max_abs_error": None,
        "perf_guard_status": "unavailable",
        "availability_reason": "",
    }
    if not run_benchmarks:
        record["availability_reason"] = "benchmarks_disabled"
        return record
    if torch is None or not torch.cuda.is_available() or not tvm.cuda(0).exist:
        record["availability_reason"] = "cuda_or_torch_unavailable"
        return record
    try:
        return _benchmark_conv_case(record, semantics, decision, warmup=warmup, repeat=repeat)
    except Exception as err:  # pylint: disable=broad-except
        record["availability_reason"] = f"benchmark_failed:{type(err).__name__}:{err}"
        return record


def _benchmark_conv_case(
    record: dict[str, Any],
    semantics: VisionConv2DSemantics,
    decision,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(_case_seed(semantics.kernel_name))
    input_np = rng.normal(size=semantics.input_shape).astype("float32")
    weight_np = rng.normal(size=semantics.weight_shape).astype("float32")
    source = build_extern_conv2d_tirx_source(semantics, decision)
    irmod = tvm.script.from_source(source)
    validate_vision_conv2d_contract(irmod)
    built = build_triton_tvm(irmod, _conv_meta(semantics))

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty(semantics.output_shape, "float32", dev)
    tvm_args = [
        tvm.runtime.tensor(input_np, dev),
        tvm.runtime.tensor(weight_np, dev),
        out_tvm,
    ]
    with register_device_torch_cuda_extern_conv2d():
        built.run(tvm_args)
        tvm_out = out_tvm.numpy()
        for _ in range(warmup):
            built.run(tvm_args)
        timer = built.executable.mod.time_evaluator(
            semantics.kernel_name,
            dev,
            number=1,
            repeat=repeat,
            min_repeat_ms=0,
        )
        tvm_values = [float(value) * 1e6 for value in timer(*tvm_args).results]

    torch_input = torch.tensor(input_np, device="cuda")
    torch_weight = torch.tensor(weight_np, device="cuda")
    expected = torch.nn.functional.conv2d(
        torch_input,
        torch_weight,
        stride=semantics.stride,
        padding=semantics.padding,
        dilation=semantics.dilation,
        groups=semantics.groups,
    )
    torch_values = _time_cuda_callable(
        lambda: torch.nn.functional.conv2d(
            torch_input,
            torch_weight,
            stride=semantics.stride,
            padding=semantics.padding,
            dilation=semantics.dilation,
            groups=semantics.groups,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    tvm_p50 = _percentile(tvm_values, 50)
    torch_p50 = _percentile(torch_values, 50)
    expected_np = expected.detach().cpu().numpy()
    record.update(
        {
            "triton_tvm_us": tvm_p50,
            "torch_or_inductor_us": torch_p50,
            "relative_to_torch": tvm_p50 / torch_p50 if torch_p50 else None,
            "correctness_allclose": bool(np.allclose(tvm_out, expected_np, rtol=1e-4, atol=1e-4)),
            "max_abs_error": float(np.max(np.abs(tvm_out - expected_np))),
            "perf_guard_status": "measured",
            "availability_reason": "",
        }
    )
    return record


def _m117_grid_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_name": case.get("case_name", ""),
        "model": case.get("model", ""),
        "op_family": "native_tvm_grid2d",
        "contract": case.get("contract", ""),
        "implementation_kind": M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        "provider_kind": case.get("provider_kind", ""),
        "runtime_status": case.get("runtime_status", ""),
        "runtime_claim": "native_tvm_grid2d_correctness_and_perf_scaffold",
        "performance_claim": bool(case.get("performance_claim", False)),
        "host_staging_bytes": int(case.get("host_staging_bytes", 0) or 0),
        "runtime_launch_count": 1,
        "artifact_call_count": 1,
        "total_io_bytes": None,
        "triton_tvm_us": case.get("triton_tvm_us"),
        "torch_or_inductor_us": case.get("native_inductor_us"),
        "relative_to_torch": case.get("relative_to_inductor"),
        "correctness_allclose": case.get("correctness_allclose"),
        "max_abs_error": case.get("max_abs_error"),
        "perf_guard_status": case.get("perf_guard_status", "unavailable"),
        "availability_reason": case.get("availability_reason", ""),
    }


def _conv_semantics(case: VisionHotpathCase) -> VisionConv2DSemantics:
    return VisionConv2DSemantics(
        source_kind="wrapper_extern_convolution",
        source_name="extern_kernels.convolution",
        kernel_name=case.case_name,
        vision_contract=(
            VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC
            if case.weight_shape[2:] == (1, 1)
            else VISION_CONTRACT_CONV2D_NCHW_STATIC
        ),
        vision_op_family="convolution",
        input_param="input",
        weight_param="weight",
        output_param="output",
        input_shape=case.input_shape,
        weight_shape=case.weight_shape,
        output_shape=case.output_shape,
        input_stride=_compact_rank4_stride(case.input_shape),
        weight_stride=_compact_rank4_stride(case.weight_shape),
        output_stride=_compact_rank4_stride(case.output_shape),
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=case.groups,
        bias_policy="none",
        transposed=False,
        output_padding=(0, 0),
    )


def _conv_meta(semantics: VisionConv2DSemantics) -> TritonTVMMeta:
    return TritonTVMMeta(
        kernel_name=semantics.kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract=semantics.vision_contract,
        canonical_contract=semantics.vision_contract,
        requested_contract=semantics.vision_contract,
        emit="tir",
        translator_version="m117_vision_hotpath_dashboard",
        contract_version=semantics.vision_contract,
        target_policy_version="m117_device_conv2d_provider",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=_numel(semantics.output_shape),
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=(
            semantics.weight_shape[1] * semantics.weight_shape[2] * semantics.weight_shape[3]
        ),
        buffer_extents={
            semantics.input_param: f"T.int64({_numel(semantics.input_shape)})",
            semantics.weight_param: f"T.int64({_numel(semantics.weight_shape)})",
            semantics.output_param: f"T.int64({_numel(semantics.output_shape)})",
        },
        block_size=1,
        indexing_kind="rank4_conv2d_device_provider",
        execution_kind="extern_conv2d_device_torch_cuda",
        accumulator_dtype_policy="fp32_conv2d_device_provider",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="nchw_oihw_static",
        layout_policy="rank4_static_conv2d_device_provider",
        launch_policy_id="extern_conv2d_device_provider",
        abi=[
            {"name": semantics.input_param, "kind": "pointer", "dtype": "float32"},
            {"name": semantics.weight_param, "kind": "pointer", "dtype": "float32"},
            {"name": semantics.output_param, "kind": "pointer", "dtype": "float32"},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"{semantics.kernel_name}_m117_hotpath",
        implementation_kind="extern_conv2d",
        extern_symbol="extern_kernels.convolution",
    )


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [case for case in cases if case.get("perf_guard_status") == "measured"]
    return {
        "total_cases": len(cases),
        "measured_cases": len(measured),
        "allclose_cases": sum(1 for case in measured if case.get("correctness_allclose")),
        "conv1x1_as_matmul_cases": sum(
            1 for case in cases if case.get("op_family") == "conv1x1_as_matmul"
        ),
        "device_conv2d_provider_cases": sum(
            1 for case in cases if case.get("provider_kind") == VISION_PROVIDER_DEVICE_TORCH_CUDA
        ),
        "native_tvm_grid2d_cases": sum(
            1 for case in cases if case.get("op_family") == "native_tvm_grid2d"
        ),
        "host_staged_cases": sum(
            1 for case in cases if str(case.get("provider_kind", "")).startswith("python_torch")
        ),
        "performance_claim_cases": sum(1 for case in cases if case.get("performance_claim")),
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for case in report.get("cases", []):
        if str(case.get("provider_kind", "")).startswith("python_torch"):
            failures.append("m11_7_hotpath_host_staged_provider_present")
        if int(case.get("host_staging_bytes", 0) or 0) != 0:
            failures.append("m11_7_hotpath_host_staging_nonzero")
        if case.get("performance_claim") is not True:
            failures.append("m11_7_hotpath_performance_claim_missing")
    return {
        "status": "passed" if not failures else "failed",
        "invariant_failures": sorted(set(failures)),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M11.7 Vision Hotpath Dashboard",
        "",
        f"- Report kind: `{report['report_kind']}`",
        f"- Baseline id: `{report['baseline_id']}`",
        f"- Hotpath status: `{report['hotpath_status']}`",
        f"- Measured cases: {report['summary']['measured_cases']} / {report['summary']['total_cases']}",
        f"- Allclose cases: {report['summary']['allclose_cases']}",
        f"- Invariants: {report['invariants']['status']}",
        "",
        "| Case | Model | Family | Contract | Provider | Runtime | TVM us | Ref us | Relative |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["case_name"]),
                    str(case["model"]),
                    str(case["op_family"]),
                    str(case["contract"]),
                    str(case["provider_kind"]),
                    str(case["runtime_status"]),
                    _fmt(case.get("triton_tvm_us")),
                    _fmt(case.get("torch_or_inductor_us")),
                    _fmt(case.get("relative_to_torch")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Host-staged providers are excluded from M11.7 performance claims.",
            "- Full TVM runnable model closure remains a stricter corpus gate.",
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


def _compact_rank4_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (channels * height * width, height * width, width, 1)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


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
    report = run_vision_hotpath_dashboard(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report["summary"]
    print(
        "M11.7 hotpath dashboard: "
        f"measured={summary['measured_cases']}/{summary['total_cases']} "
        f"allclose={summary['allclose_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
