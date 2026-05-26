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
"""M11.P vision conv2d diagnostic baseline runner.

This report is intentionally separate from corpus provider metadata.  The M11.4
conv2d provider remains correctness-only in corpus records; this runner records
same-machine timing for the first admitted ViT patch-embedding conv.
"""

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
from tvm.contrib.triton_tvm.translator import TritonTVMMeta
from tvm.contrib.triton_tvm.vision import (
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_EXTERN_SYMBOL,
    VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
    VISION_M11_4_VIT_PATCH_DILATION,
    VISION_M11_4_VIT_PATCH_INPUT_SHAPE,
    VISION_M11_4_VIT_PATCH_OUTPUT_SHAPE,
    VISION_M11_4_VIT_PATCH_PADDING,
    VISION_M11_4_VIT_PATCH_STRIDE,
    VISION_M11_4_VIT_PATCH_WEIGHT_SHAPE,
    VISION_M11_5_HARDENING_STATUS,
    VISION_M11_5_RUNTIME_SCOPE_STATUS,
    VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    TargetVisionPolicy,
    build_extern_conv2d_tirx_source,
    extract_vision_conv2d_semantics_from_wrapper_extern,
    register_python_torch_extern_conv2d,
)

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - availability is reflected in reports.
    torch = None
    torch_functional = None


REPORT_KIND = "triton_tvm_m11_vision_performance_baseline"
BASELINE_ID = "m11_vit_patch_conv_provider_vs_torch_conv2d_v1"
TORCH_CONV2D_BASELINE_ID = "torch_cuda_conv2d_framework_default_v1"
TRITON_TVM_CONV2D_BASELINE_ID = "triton_tvm_extern_conv2d_python_torch_host_staged_v1"
DEFAULT_OUT_DIR = Path("/home/liyh/xdb/triton-tvm-workbench/reports/m11/m11p_vit_patch_conv_baseline")
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 20


@dataclass(frozen=True)
class VisionConv2DBaselineCase:
    """A fixed shape measured by the M11.P vision baseline."""

    name: str
    contract: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    notes: str = ""

    @property
    def input_stride(self) -> tuple[int, int, int, int]:
        return _channels_last_stride(self.input_shape)

    @property
    def weight_stride(self) -> tuple[int, int, int, int]:
        return _channels_last_stride(self.weight_shape)

    @property
    def output_stride(self) -> tuple[int, int, int, int]:
        return _channels_last_stride(self.output_shape)


def default_vision_conv2d_baseline_suite() -> tuple[VisionConv2DBaselineCase, ...]:
    """Return the fixed M11.P vision baseline cases."""
    return (
        VisionConv2DBaselineCase(
            name="vit_patch_embedding_conv2d_1x3x32x32_64x3x16x16_stride16",
            contract=VISION_CONTRACT_CONV2D_NCHW_STATIC,
            input_shape=VISION_M11_4_VIT_PATCH_INPUT_SHAPE,
            weight_shape=VISION_M11_4_VIT_PATCH_WEIGHT_SHAPE,
            output_shape=VISION_M11_4_VIT_PATCH_OUTPUT_SHAPE,
            stride=VISION_M11_4_VIT_PATCH_STRIDE,
            padding=VISION_M11_4_VIT_PATCH_PADDING,
            dilation=VISION_M11_4_VIT_PATCH_DILATION,
            groups=1,
            notes="M11.4 correctness-only runtime-resolved ViT patch embedding conv",
        ),
    )


def run_vision_baseline(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M11.P vision baseline and optionally write JSON/Markdown reports."""
    cases = [
        _run_case(case, warmup=warmup, repeat=repeat, run_benchmarks=run_benchmarks)
        for case in default_vision_conv2d_baseline_suite()
    ]
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "baseline_kind": "diagnostic_vision_conv2d_performance_baseline_v1",
        "hardening_status": VISION_M11_5_HARDENING_STATUS,
        "baseline_performance_claim": True,
        "vision_runtime_provider_performance_claim": False,
        "claim_scope": (
            "same-machine timing baseline for the M11.4 ViT patch conv provider; "
            "not a full-model runnable or provider performance claim"
        ),
        "baseline_ids": {
            "torch": TORCH_CONV2D_BASELINE_ID,
            "triton_tvm": TRITON_TVM_CONV2D_BASELINE_ID,
        },
        "provider_boundary": {
            "provider": VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
            "runtime_provider_claim": "correctness_only",
            "vision_performance_claim_in_corpus": False,
            "full_model_runnable_claim": False,
        },
        "warmup": warmup,
        "repeat": repeat,
        "cases": cases,
        "dependency_versions": _dependency_versions(),
    }
    report["summary"] = _summary(cases)
    report["hardening_checks"] = _hardening_checks(report)
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
    case: VisionConv2DBaselineCase,
    *,
    warmup: int,
    repeat: int,
    run_benchmarks: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        **asdict(case),
        "input_shape": list(case.input_shape),
        "weight_shape": list(case.weight_shape),
        "output_shape": list(case.output_shape),
        "input_stride": list(case.input_stride),
        "weight_stride": list(case.weight_stride),
        "output_stride": list(case.output_stride),
        "baseline_torch": {
            "baseline_id": TORCH_CONV2D_BASELINE_ID,
            "backend": "torch.nn.functional.conv2d",
            "backend_policy": "framework_default",
            "performance_claim": True,
        },
        "baseline_triton_tvm": {
            "baseline_id": TRITON_TVM_CONV2D_BASELINE_ID,
            "provider": VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
            "runtime_provider_claim": "correctness_only",
            "performance_claim": True,
        },
        "torch_conv2d_us": None,
        "torch_conv2d_p50_us": None,
        "torch_conv2d_p95_us": None,
        "triton_tvm_us": None,
        "triton_tvm_p50_us": None,
        "triton_tvm_p95_us": None,
        "relative_to_torch": None,
        "conv_gflops": None,
        "perf_guard_status": "unavailable",
        "correctness_allclose": None,
        "max_abs_error": None,
        "availability_reason": "",
    }
    if not run_benchmarks:
        record["availability_reason"] = "benchmarks_disabled"
        return record
    if torch is None or torch_functional is None:
        record["availability_reason"] = "torch_unavailable"
        return record
    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        record["availability_reason"] = "cuda_unavailable"
        return record

    try:
        return _benchmark_case(record, case, warmup=warmup, repeat=repeat)
    except Exception as err:  # pylint: disable=broad-except
        record["availability_reason"] = f"benchmark_failed:{type(err).__name__}:{err}"
        return record


def _benchmark_case(
    record: dict[str, Any],
    case: VisionConv2DBaselineCase,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(_case_seed(case.name))
    input_np = rng.normal(size=case.input_shape).astype("float32")
    weight_np = rng.normal(size=case.weight_shape).astype("float32")

    torch_input = torch.tensor(input_np, device="cuda")
    torch_weight = torch.tensor(weight_np, device="cuda")
    expected = torch_functional.conv2d(
        torch_input,
        torch_weight,
        bias=None,
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=case.groups,
    )

    semantics = extract_vision_conv2d_semantics_from_wrapper_extern(
        _wrapper_source(case),
        case_name=case.name,
        kernel_name=f"{case.name}_m11_vision_baseline",
    )
    decision = TargetVisionPolicy(
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, vision_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_conv2d_tirx_source(semantics, decision))
    validate_vision_conv2d_contract(irmod)
    built = build_triton_tvm(irmod, _vision_meta(case, semantics.kernel_name, semantics))

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty(case.output_shape, "float32", dev)
    tvm_args: list[Any] = [
        tvm.runtime.tensor(input_np, dev),
        tvm.runtime.tensor(weight_np, dev),
        out_tvm,
    ]
    with register_python_torch_extern_conv2d():
        built.run(tvm_args)
        tvm_out = out_tvm.numpy()
        for _ in range(warmup):
            built.run(tvm_args)
        tvm_timer = built.executable.mod.time_evaluator(
            semantics.kernel_name,
            dev,
            number=1,
            repeat=repeat,
            min_repeat_ms=0,
        )
        tvm_values = [float(value) * 1e6 for value in tvm_timer(*tvm_args).results]

    expected_np = expected.detach().cpu().numpy()
    max_abs_error = float(np.max(np.abs(tvm_out - expected_np)))
    correctness_allclose = bool(np.allclose(tvm_out, expected_np, rtol=1e-5, atol=1e-5))
    torch_values = _time_cuda_callable(
        lambda: torch_functional.conv2d(
            torch_input,
            torch_weight,
            bias=None,
            stride=case.stride,
            padding=case.padding,
            dilation=case.dilation,
            groups=case.groups,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    tvm_p50 = _percentile(tvm_values, 50)
    tvm_p95 = _percentile(tvm_values, 95)
    torch_p50 = _percentile(torch_values, 50)
    torch_p95 = _percentile(torch_values, 95)
    record.update(
        {
            "torch_conv2d_us": torch_p50,
            "torch_conv2d_p50_us": torch_p50,
            "torch_conv2d_p95_us": torch_p95,
            "triton_tvm_us": tvm_p50,
            "triton_tvm_p50_us": tvm_p50,
            "triton_tvm_p95_us": tvm_p95,
            "relative_to_torch": tvm_p50 / torch_p50 if torch_p50 else None,
            "conv_gflops": _conv_gflops(case, tvm_p50),
            "perf_guard_status": "measured",
            "correctness_allclose": correctness_allclose,
            "max_abs_error": max_abs_error,
            "availability_reason": "",
        }
    )
    return record


def _wrapper_source(case: VisionConv2DBaselineCase) -> str:
    return f"""
def call(arg0, arg1):
    buf0 = empty_strided_cuda({_tuple_literal(case.input_shape)}, {_tuple_literal(case.input_stride)}, torch.float32)
    buf1 = empty_strided_cuda({_tuple_literal(case.weight_shape)}, {_tuple_literal(case.weight_stride)}, torch.float32)
    buf2 = extern_kernels.convolution(
        buf0,
        buf1,
        stride={_tuple_literal(case.stride)},
        padding={_tuple_literal(case.padding)},
        dilation={_tuple_literal(case.dilation)},
        transposed=False,
        output_padding=(0, 0),
        groups={case.groups},
        bias=None,
    )
    assert_size_stride(buf2, {_tuple_literal(case.output_shape)}, {_tuple_literal(case.output_stride)}, 'torch.ops.aten.convolution.default')
    return buf2
"""


def _vision_meta(
    case: VisionConv2DBaselineCase,
    kernel_name: str,
    semantics,
) -> TritonTVMMeta:
    return TritonTVMMeta(
        kernel_name=kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract=case.contract,
        canonical_contract=case.contract,
        requested_contract=case.contract,
        emit="tir",
        translator_version="m11_vision_performance_baseline",
        contract_version="conv2d_nchw_static_m11_v1",
        target_policy_version="m11_vision_performance_baseline",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=_numel(case.output_shape),
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=case.weight_shape[1] * case.weight_shape[2] * case.weight_shape[3],
        buffer_extents={
            semantics.input_param: f"T.int64({_numel(case.input_shape)})",
            semantics.weight_param: f"T.int64({_numel(case.weight_shape)})",
            semantics.output_param: f"T.int64({_numel(case.output_shape)})",
        },
        block_size=1,
        indexing_kind="rank4_conv2d_baseline",
        execution_kind="extern_conv2d_python_torch_host_staged",
        accumulator_dtype_policy="fp32_conv2d_baseline",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="nchw_oihw_static",
        layout_policy="rank4_static_conv2d_baseline",
        launch_policy_id="extern_conv2d_runtime_provider_baseline",
        abi=[
            {"name": semantics.input_param, "kind": "pointer", "dtype": "float32"},
            {"name": semantics.weight_param, "kind": "pointer", "dtype": "float32"},
            {"name": semantics.output_param, "kind": "pointer", "dtype": "float32"},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"{kernel_name}_vision_baseline",
        implementation_kind=VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
        extern_symbol=VISION_EXTERN_SYMBOL,
    )


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


def _conv_gflops(case: VisionConv2DBaselineCase, us: float) -> float:
    if us <= 0:
        return 0.0
    batch, output_channels, output_h, output_w = case.output_shape
    _, input_channels_per_group, kernel_h, kernel_w = case.weight_shape
    ops = (
        2.0
        * batch
        * output_channels
        * output_h
        * output_w
        * input_channels_per_group
        * kernel_h
        * kernel_w
    )
    return ops / (us * 1e3)


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [case for case in cases if case.get("perf_guard_status") == "measured"]
    unavailable = [case for case in cases if case.get("perf_guard_status") == "unavailable"]
    return {
        "total_cases": len(cases),
        "measured_cases": len(measured),
        "unavailable_cases": len(unavailable),
        "allclose_cases": sum(1 for case in measured if case.get("correctness_allclose") is True),
        "contracts": {
            contract: sum(1 for case in cases if case.get("contract") == contract)
            for contract in sorted({str(case.get("contract")) for case in cases})
        },
    }


def _hardening_checks(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    provider_boundary = report.get("provider_boundary", {})
    if provider_boundary.get("provider") != VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED:
        failures.append("m11p_provider_boundary_mismatch")
    if provider_boundary.get("runtime_provider_claim") != "correctness_only":
        failures.append("m11p_provider_claim_not_correctness_only")
    if provider_boundary.get("vision_performance_claim_in_corpus") is not False:
        failures.append("m11p_corpus_provider_performance_claim_not_false")
    if provider_boundary.get("full_model_runnable_claim") is not False:
        failures.append("m11p_full_model_runnable_claim_not_false")
    for case in report.get("cases", []):
        if case.get("contract") != VISION_CONTRACT_CONV2D_NCHW_STATIC:
            failures.append("m11p_contract_scope_mismatch")
            break
        if tuple(case.get("input_shape", ())) != VISION_M11_4_VIT_PATCH_INPUT_SHAPE:
            failures.append("m11p_input_shape_scope_mismatch")
            break
        if tuple(case.get("weight_shape", ())) != VISION_M11_4_VIT_PATCH_WEIGHT_SHAPE:
            failures.append("m11p_weight_shape_scope_mismatch")
            break
        if tuple(case.get("output_shape", ())) != VISION_M11_4_VIT_PATCH_OUTPUT_SHAPE:
            failures.append("m11p_output_shape_scope_mismatch")
            break
    return {
        "status": "passed" if not failures else "failed",
        "hardening_status": VISION_M11_5_HARDENING_STATUS,
        "runtime_scope_status": VISION_M11_5_RUNTIME_SCOPE_STATUS,
        "invariant_failures": sorted(set(failures)),
    }


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


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M11.P Vision Conv2D Performance Baseline",
        "",
        f"- Report kind: `{report['report_kind']}`",
        f"- Baseline id: `{report['baseline_id']}`",
        f"- Measured cases: {report['summary']['measured_cases']} / {report['summary']['total_cases']}",
        f"- Allclose cases: {report['summary']['allclose_cases']}",
        f"- Baseline performance claim: {report['baseline_performance_claim']}",
        f"- Runtime provider performance claim: {report['vision_runtime_provider_performance_claim']}",
        f"- Hardening status: `{report['hardening_status']}`",
        "",
        "## M11.5 Baseline Hardening",
        "",
        f"- Status: {report['hardening_checks']['status']}",
        f"- Runtime scope: {report['hardening_checks']['runtime_scope_status']}",
        "- Invariant failures: "
        f"{', '.join(report['hardening_checks']['invariant_failures']) or 'none'}",
        "",
        "## Cases",
        "",
        "| Case | Contract | Input | Weight | Status | TVM us | Torch conv2d us | Relative | Conv GFLOP/s |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["name"]),
                    str(case["contract"]),
                    "x".join(str(v) for v in case["input_shape"]),
                    "x".join(str(v) for v in case["weight_shape"]),
                    str(case["perf_guard_status"]),
                    _fmt(case.get("triton_tvm_us")),
                    _fmt(case.get("torch_conv2d_us")),
                    _fmt(case.get("relative_to_torch")),
                    _fmt(case.get("conv_gflops")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a same-machine vision baseline report, not a full-model runnable claim.",
            "- M11 corpus `vision_performance_claim` remains false for the provider.",
            "- Broader convolution coverage and Grid2D lowering are not measured here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _channels_last_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (channels * height * width, 1, width * channels, channels)


def _tuple_literal(values: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(value) for value in values) + ("," if len(values) == 1 else "") + ")"


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _case_seed(name: str) -> int:
    return sum(ord(ch) for ch in name) % (2**31)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    report = run_vision_baseline(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report["summary"]
    print(
        "M11.P vision baseline: "
        f"measured={summary['measured_cases']}/{summary['total_cases']} "
        f"allclose={summary['allclose_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
