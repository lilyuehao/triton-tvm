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
"""M9.P matmul-only shape dashboard and baseline runner."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tvm
from tvm.contrib.triton_tvm import build_triton_tvm, translate_ttir
from tvm.contrib.triton_tvm.matmul import (
    MATMUL_PERF_ENVELOPE_ID,
    MATMUL_SCHEDULE_REGISTRY_VERSION,
    MATMUL_TUNE_KEY_VERSION,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TORCH_CUDA_CUBLAS_BASELINE_ID,
    TRITON_NATIVE_MATMUL_BASELINE_ID,
    build_matmul_tune_key_payload,
    matmul_schedule_candidate_records,
)

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by report availability fields.
    torch = None
    triton = None
    tl = None


REPORT_KIND = "triton_tvm_m9p_matmul_phase1_dashboard"
DEFAULT_OUT_DIR = Path("/home/liyh/xdb/triton-tvm-workbench/reports/m09p/m9p_phase1_dashboard")
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 20


@dataclass(frozen=True)
class MatmulShapeCase:
    """A dashboard shape case, not an attention semantic."""

    name: str
    family: str
    m: int
    n: int
    k: int
    dtype: str = "float16"
    coverage_only: bool = False
    notes: str = ""


def default_shape_suite() -> tuple[MatmulShapeCase, ...]:
    """Return the fixed M9.P shape suite."""
    return (
        MatmulShapeCase(
            "tinymnist_mlp_fc1_original",
            "tinymlp_original",
            100,
            128,
            784,
            coverage_only=True,
            notes="original M9.8 fc1 shape; outside matmul_perf_core_v1 because M is not multiple of 16",
        ),
        MatmulShapeCase(
            "tinymnist_mlp_fc2_original",
            "tinymlp_original",
            100,
            10,
            128,
            coverage_only=True,
            notes="original M9.8 fc2 shape; outside matmul_perf_core_v1 because M/N are not multiples of 16",
        ),
        MatmulShapeCase("qk_t_matmul_16x16x64", "qk_t_shape", 16, 16, 64),
        MatmulShapeCase("av_matmul_16x64x16", "av_shape", 16, 64, 16),
        MatmulShapeCase("projection_matmul_32x32x64", "projection_shape", 32, 32, 64),
        MatmulShapeCase("mlp_ffn_up_matmul_32x64x64", "mlp_ffn_shape", 32, 64, 64),
        MatmulShapeCase("mlp_ffn_down_matmul_32x32x64", "mlp_ffn_shape", 32, 32, 64),
    )


def run_dashboard(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M9.P Phase 1 dashboard and optionally write JSON/Markdown reports."""
    cases = [_run_case(case, warmup=warmup, repeat=repeat, run_benchmarks=run_benchmarks) for case in default_shape_suite()]
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "matmul_perf_envelope": MATMUL_PERF_ENVELOPE_ID,
        "schedule_registry_version": MATMUL_SCHEDULE_REGISTRY_VERSION,
        "tune_key_version": MATMUL_TUNE_KEY_VERSION,
        "selected_phase": "M9.P Phase 0 + Phase 1 + Phase 2 + Phase 3",
        "package_status": {
            "M9.PA": "implemented_fp16_ptx_mma_tensorcore_candidate",
            "M9.PB": "implemented_torch_cuda_cublas_baseline_no_tvm_packed_provider",
            "M9.PD": "implemented_shape_dashboard",
            "M9.PC": "implemented_bf16_capable_16x16_simt_fallback_candidate",
            "M9.PE": "implemented_wrapper_extern_addmm_bias_artifact_slice",
        },
        "extern_provider_status": {
            "provider_id": "tvm_packed_cublas_or_cublaslt_runtime",
            "status": "unavailable",
            "reason": "phase1_pb_reports_torch_cuda_cublas_baseline_without_tvm_packed_runtime_replacement",
            "performance_claim": False,
        },
        "baseline_ids": {
            "triton": TRITON_NATIVE_MATMUL_BASELINE_ID,
            "torch": TORCH_CUDA_CUBLAS_BASELINE_ID,
            "triton_tvm": TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
        },
        "schedule_candidates": list(matmul_schedule_candidate_records()),
        "cases": cases,
    }
    report["summary"] = _summary(cases)
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
    case: MatmulShapeCase,
    *,
    warmup: int,
    repeat: int,
    run_benchmarks: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        **asdict(case),
        "matmul_perf_envelope": MATMUL_PERF_ENVELOPE_ID,
        "candidate_schedule_ids": [item["schedule_id"] for item in matmul_schedule_candidate_records()],
        "selected_schedule_id": "",
        "schedule_reject_reasons": {},
        "baseline_triton_us": None,
        "baseline_torch_us": None,
        "triton_tvm_us": None,
        "triton_tvm_p50_us": None,
        "triton_tvm_p95_us": None,
        "tflops": None,
        "relative_to_triton": None,
        "relative_to_torch": None,
        "perf_guard_status": "unavailable",
        "correctness_allclose": None,
        "max_abs_error": None,
        "tune_key": {},
        "availability_reason": "",
        "baseline_torch": {
            "baseline_id": TORCH_CUDA_CUBLAS_BASELINE_ID,
            "performance_claim": True,
        },
        "baseline_triton": {
            "baseline_id": TRITON_NATIVE_MATMUL_BASELINE_ID,
            "performance_claim": True,
        },
    }
    if case.coverage_only or not _shape_in_envelope(case):
        record["availability_reason"] = "shape_outside_matmul_perf_core_v1"
        return record
    if not run_benchmarks:
        record["availability_reason"] = "benchmarks_disabled"
        return record
    if torch is None or triton is None or tl is None:
        record["availability_reason"] = "torch_or_triton_unavailable"
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
    case: MatmulShapeCase,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1, 1, (case.m, case.k)).astype("float16")
    b_np = rng.uniform(-1, 1, (case.k, case.n)).astype("float16")
    expected = a_np.astype("float32") @ b_np.astype("float32")

    irmod, meta = translate_ttir(_dot_ttir(case), grid=(1,), contract="matmul_minimal")
    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    tvm_args = [
        tvm.runtime.tensor(a_np, dev),
        tvm.runtime.tensor(b_np, dev),
        tvm.runtime.empty((case.m, case.n), "float32", dev),
    ]
    built.run(tvm_args)
    tvm_out = tvm_args[-1].numpy()
    max_abs_error = float(np.max(np.abs(tvm_out - expected)))
    correctness_allclose = bool(np.allclose(tvm_out, expected, rtol=1e-2, atol=1e-2))

    timer = built.executable.mod.time_evaluator(
        meta.kernel_name,
        dev,
        number=1,
        repeat=repeat,
        min_repeat_ms=0,
    )
    tvm_us_values = [float(value) * 1e6 for value in timer(*tvm_args).results]
    tvm_p50 = _percentile(tvm_us_values, 50)
    tvm_p95 = _percentile(tvm_us_values, 95)

    torch_a = torch.tensor(a_np, device="cuda")
    torch_b = torch.tensor(b_np, device="cuda")
    torch_holder: dict[str, Any] = {}
    torch_us_values = _time_cuda_callable(
        lambda: torch_holder.__setitem__("out", torch.matmul(torch_a, torch_b)),
        warmup=warmup,
        repeat=repeat,
    )

    triton_c = torch.empty((case.m, case.n), dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(case.m, 16) * triton.cdiv(case.n, 16),)
    triton_us_values = _time_cuda_callable(
        lambda: _triton_matmul_kernel[grid](
            torch_a,
            torch_b,
            triton_c,
            case.m,
            case.n,
            case.k,
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=16,
        ),
        warmup=warmup,
        repeat=repeat,
    )

    triton_us = _percentile(triton_us_values, 50)
    torch_us = _percentile(torch_us_values, 50)
    record.update(
        {
            "selected_schedule_id": meta.selected_schedule_id,
            "schedule_reject_reasons": dict(meta.schedule_reject_reasons or {}),
            "baseline_triton_us": triton_us,
            "baseline_torch_us": torch_us,
            "triton_tvm_us": tvm_p50,
            "triton_tvm_p50_us": tvm_p50,
            "triton_tvm_p95_us": tvm_p95,
            "tflops": _tflops(case.m, case.n, case.k, tvm_p50),
            "relative_to_triton": tvm_p50 / triton_us if triton_us else None,
            "relative_to_torch": tvm_p50 / torch_us if torch_us else None,
            "perf_guard_status": "measured",
            "correctness_allclose": correctness_allclose,
            "max_abs_error": max_abs_error,
            "tune_key": dict(meta.tune_key or {}),
            "availability_reason": "",
        }
    )
    record["tune_key"].update(
        build_matmul_tune_key_payload(
            _semantics_from_meta(case, meta),
            schedule_id=meta.selected_schedule_id,
            tune_params=(meta.tune_key or {}).get("tune_params", {}),
            target_arch=torch.cuda.get_device_capability_string()
            if hasattr(torch.cuda, "get_device_capability_string")
            else str(torch.cuda.get_device_capability(0)),
            device_name=torch.cuda.get_device_name(0),
            framework_versions={
                "tvm": tvm.__version__,
                "triton": triton.__version__,
                "torch": torch.__version__,
            },
            cuda_target_metadata={"target": meta.target, "target_attrs": meta.target_attrs},
        )
    )
    return record


if triton is not None and tl is not None:

    @triton.jit
    def _triton_matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        pid_m = pid // tl.cdiv(N, BLOCK_N)
        pid_n = pid % tl.cdiv(N, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a_vals = tl.load(a + offs_m[:, None] * K + (k0 + offs_k[None, :]))
            b_vals = tl.load(b + (k0 + offs_k[:, None]) * N + offs_n[None, :])
            acc += tl.dot(a_vals, b_vals, out_dtype=tl.float32)
        tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc)

else:
    _triton_matmul_kernel = None


def _semantics_from_meta(case: MatmulShapeCase, meta: Any):
    from tvm.contrib.triton_tvm.matmul import MatmulSemantics

    return MatmulSemantics(
        source_kind="tt_dot",
        source_name="dashboard",
        kernel_name=case.name,
        m=case.m,
        n=case.n,
        k=case.k,
        batch_dims=(),
        a_param="a",
        b_param="b",
        c_param="out",
        a_dtype="float16",
        b_dtype="float16",
        accumulator_dtype="float32",
        output_dtype="float32",
        a_layout="row_major",
        b_layout="row_major",
        c_layout="row_major",
        a_stride=(case.k, 1),
        b_stride=(case.n, 1),
        c_stride=(case.n, 1),
        transpose_a=False,
        transpose_b=False,
        block_m=case.m,
        block_n=case.n,
        block_k=case.k,
        input_precision="tf32",
        tf32_policy="tf32",
        bounds_policy="exact",
        mask_kind="none",
        epilogue_kind="none",
        alias_policy="distinct_buffers",
        noalias=True,
        target_kind=meta.target_kind,
        implementation_kind="native_tir_schedule",
        fallback_reason="",
    )


def _shape_in_envelope(case: MatmulShapeCase) -> bool:
    return (
        case.dtype == "float16"
        and case.m % 16 == 0
        and case.n % 16 == 0
        and case.k % 16 == 0
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


def _tflops(m: int, n: int, k: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return (2.0 * m * n * k) / (us * 1e6)


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [case for case in cases if case.get("perf_guard_status") == "measured"]
    unavailable = [case for case in cases if case.get("perf_guard_status") == "unavailable"]
    return {
        "total_cases": len(cases),
        "measured_cases": len(measured),
        "unavailable_cases": len(unavailable),
        "allclose_cases": sum(1 for case in measured if case.get("correctness_allclose") is True),
        "tensorcore_selected_cases": sum(
            1 for case in measured if case.get("selected_schedule_id") == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M9.P Matmul Dashboard",
        "",
        f"- Report kind: `{report['report_kind']}`",
        f"- Envelope: `{report['matmul_perf_envelope']}`",
        f"- Registry: `{report['schedule_registry_version']}`",
        f"- Measured cases: {report['summary']['measured_cases']} / {report['summary']['total_cases']}",
        f"- TensorCore selected cases: {report['summary']['tensorcore_selected_cases']}",
        "",
        "## Package Status",
        "",
    ]
    for package_id, status in sorted(report["package_status"].items()):
        lines.append(f"- `{package_id}`: {status}")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Family | Shape | Status | Schedule | TVM us | Triton us | Torch us | TFLOPS |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["name"]),
                    str(case["family"]),
                    f"{case['m']}x{case['n']}x{case['k']}",
                    str(case["perf_guard_status"]),
                    str(case.get("selected_schedule_id") or case.get("availability_reason") or ""),
                    _fmt(case.get("triton_tvm_us")),
                    _fmt(case.get("baseline_triton_us")),
                    _fmt(case.get("baseline_torch_us")),
                    _fmt(case.get("tflops")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Provider Boundary",
            "",
            "- `python_torch_host_staged` remains excluded from performance baselines.",
            "- TVM packed cuBLAS/cuBLASLt runtime replacement is unavailable in M9.P.",
            "- Torch CUDA/cuBLAS is recorded only as an external baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dot_ttir(case: MatmulShapeCase) -> str:
    dtype = "f16" if case.dtype == "float16" else case.dtype
    return f"""
module {{
  tt.func @_m9p_dashboard_dot(%a:!tt.ptr<{dtype}>,%b:!tt.ptr<{dtype}>,%out:!tt.ptr<f32>) {{
    %a_zero = arith.constant dense<0> : tensor<{case.m}x{case.k}xi32>
    %b_zero = arith.constant dense<0> : tensor<{case.k}x{case.n}xi32>
    %c_zero = arith.constant dense<0> : tensor<{case.m}x{case.n}xi32>
    %a_splat = tt.splat %a : !tt.ptr<{dtype}> -> tensor<{case.m}x{case.k}x!tt.ptr<{dtype}>>
    %a_ptr = tt.addptr %a_splat, %a_zero : tensor<{case.m}x{case.k}x!tt.ptr<{dtype}>>, tensor<{case.m}x{case.k}xi32>
    %va = tt.load %a_ptr : tensor<{case.m}x{case.k}x!tt.ptr<{dtype}>>
    %b_splat = tt.splat %b : !tt.ptr<{dtype}> -> tensor<{case.k}x{case.n}x!tt.ptr<{dtype}>>
    %b_ptr = tt.addptr %b_splat, %b_zero : tensor<{case.k}x{case.n}x!tt.ptr<{dtype}>>, tensor<{case.k}x{case.n}xi32>
    %vb = tt.load %b_ptr : tensor<{case.k}x{case.n}x!tt.ptr<{dtype}>>
    %dot = tt.dot %va, %vb {{inputPrecision = tf32}} : tensor<{case.m}x{case.k}x{dtype}>, tensor<{case.k}x{case.n}x{dtype}> -> tensor<{case.m}x{case.n}xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<{case.m}x{case.n}x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %c_zero : tensor<{case.m}x{case.n}x!tt.ptr<f32>>, tensor<{case.m}x{case.n}xi32>
    tt.store %out_ptr, %dot : tensor<{case.m}x{case.n}x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    report = run_dashboard(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report["summary"]
    print(
        "M9.P dashboard: "
        f"measured={summary['measured_cases']}/{summary['total_cases']} "
        f"tensorcore_selected={summary['tensorcore_selected_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
