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
"""Performance baseline and regression guard for Triton TVM pointwise kernels.

This file is intentionally opt-in.  The numbers are hardware- and driver-sensitive, so
the default pytest run skips it.  Use it to establish a same-machine baseline and to
guard later translator/contract changes against large regressions.  The case set covers
the pre-M5 canonical pointwise contracts and pointwise_flat capability variants.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import build_triton_tvm, lower_to_ttir, translate_ttir
from tvm.contrib.triton_tvm.m9p_matmul_dashboard import run_dashboard
from tvm.contrib.triton_tvm.matmul import NATIVE_TIR_MATMUL_SCHEDULE_ID

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    pytestmark = pytest.skip("Triton or PyTorch is not available", allow_module_level=True)


_RUN_PERF = os.environ.get("TRITON_TVM_RUN_PERF_BASELINE") == "1"
_RUN_M9_MATMUL_PERF = os.environ.get("TRITON_TVM_RUN_M9_MATMUL_PERF_GUARD") == "1"
_RUN_M9P_MATMUL_DASHBOARD = os.environ.get("TRITON_TVM_RUN_M9P_MATMUL_DASHBOARD") == "1"
_DEFAULT_N = int(os.environ.get("TRITON_TVM_PERF_N", str(2**22)))
_DEFAULT_BLOCK = int(os.environ.get("TRITON_TVM_PERF_BLOCK", "256"))
_TVM_NUMBER = int(os.environ.get("TRITON_TVM_PERF_TVM_NUMBER", "20"))
_TVM_REPEAT = int(os.environ.get("TRITON_TVM_PERF_TVM_REPEAT", "7"))
_TVM_MIN_REPEAT_MS = int(os.environ.get("TRITON_TVM_PERF_TVM_MIN_REPEAT_MS", "100"))
_TRITON_WARMUP = int(os.environ.get("TRITON_TVM_PERF_TRITON_WARMUP", "25"))
_TRITON_REP = int(os.environ.get("TRITON_TVM_PERF_TRITON_REP", "100"))
_REGRESSION_TOLERANCE = float(os.environ.get("TRITON_TVM_PERF_TOLERANCE", "0.35"))
_BROADCAST_FEATURE = int(os.environ.get("TRITON_TVM_PERF_BROADCAST_FEATURE", "1024"))
_TVM_TO_TRITON_GBPS_FLOOR = float(
    os.environ.get("TRITON_TVM_PERF_TVM_TO_TRITON_GBPS_FLOOR", "0.10")
)
_M9_MATMUL_NUMBER = int(os.environ.get("TRITON_TVM_M9_MATMUL_PERF_NUMBER", "20"))
_M9_MATMUL_REPEAT = int(os.environ.get("TRITON_TVM_M9_MATMUL_PERF_REPEAT", "7"))
_M9_MATMUL_MIN_REPEAT_MS = int(
    os.environ.get("TRITON_TVM_M9_MATMUL_PERF_MIN_REPEAT_MS", "50")
)


@triton.jit
def _perf_vector_add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, vx + vy, mask=mask)


@triton.jit
def _perf_pointwise_chain_kernel(x, y, z, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    vz = tl.load(z + offsets, mask=mask, other=0.0)
    value = vx * 2.0 + vy - vz
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _perf_dual_store_kernel(x, y, out0, out1, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out0 + offsets, vx + vy, mask=mask)
    tl.store(out1 + offsets, vx - vy, mask=mask)


@triton.jit
def _perf_scalar_broadcast_kernel(x, out0, out1, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    shifted = alpha + 1.0
    tl.store(out0 + offsets, vx + shifted, mask=mask)
    tl.store(out1 + offsets, vx * alpha, mask=mask)


@triton.jit
def _perf_indexed_broadcast_relu_kernel(x, y, out, n, BLOCK: tl.constexpr, FEATURE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    y_offsets = offsets % FEATURE
    vx = tl.load(x + offsets, mask=mask)
    vy = tl.load(y + y_offsets, mask=mask)
    relu = tl.where(vx > 0.0, vx, 0.0)
    tl.store(out + offsets, relu + vy, mask=mask)


@triton.jit
def _perf_indexed_strided_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets * 2, mask=mask)
    tl.store(out + offsets, vx * 2.0, mask=mask)


@dataclass(frozen=True)
class _PerfCase:
    name: str
    jit_fn: Any
    signature: dict[str, str]
    contract: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    bytes_per_element: int
    scalars: dict[str, Any]
    constexprs: dict[str, Any] = field(default_factory=dict)
    array_size_factors: dict[str, int] = field(default_factory=dict)
    array_size_constants: dict[str, int] = field(default_factory=dict)


_PERF_CASES = (
    _PerfCase(
        name="vector_add",
        jit_fn=_perf_vector_add_kernel,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "out": "*fp32",
            "n": "i64",
            "BLOCK": "constexpr",
        },
        contract="pointwise_minimal",
        inputs=("x", "y"),
        outputs=("out",),
        bytes_per_element=12,
        scalars={},
    ),
    _PerfCase(
        name="pointwise_chain",
        jit_fn=_perf_pointwise_chain_kernel,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "z": "*fp32",
            "out": "*fp32",
            "n": "i64",
            "BLOCK": "constexpr",
        },
        contract="pointwise_minimal",
        inputs=("x", "y", "z"),
        outputs=("out",),
        bytes_per_element=16,
        scalars={},
    ),
    _PerfCase(
        name="dual_store",
        jit_fn=_perf_dual_store_kernel,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "out0": "*fp32",
            "out1": "*fp32",
            "n": "i64",
            "BLOCK": "constexpr",
        },
        contract="pointwise_flat",
        inputs=("x", "y"),
        outputs=("out0", "out1"),
        bytes_per_element=16,
        scalars={},
    ),
    _PerfCase(
        name="scalar_broadcast",
        jit_fn=_perf_scalar_broadcast_kernel,
        signature={
            "x": "*fp32",
            "out0": "*fp32",
            "out1": "*fp32",
            "n": "i64",
            "alpha": "fp32",
            "BLOCK": "constexpr",
        },
        contract="pointwise_flat",
        inputs=("x",),
        outputs=("out0", "out1"),
        bytes_per_element=12,
        scalars={"alpha": 2.5},
    ),
    _PerfCase(
        name="indexed_broadcast_relu",
        jit_fn=_perf_indexed_broadcast_relu_kernel,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "out": "*fp32",
            "n": "i64",
            "BLOCK": "constexpr",
            "FEATURE": "constexpr",
        },
        contract="pointwise_flat",
        inputs=("x", "y"),
        outputs=("out",),
        bytes_per_element=12,
        scalars={},
        constexprs={"FEATURE": _BROADCAST_FEATURE},
        array_size_constants={"y": _BROADCAST_FEATURE},
    ),
    _PerfCase(
        name="indexed_strided",
        jit_fn=_perf_indexed_strided_kernel,
        signature={
            "x": "*fp32",
            "out": "*fp32",
            "n": "i64",
            "BLOCK": "constexpr",
        },
        contract="pointwise_flat",
        inputs=("x",),
        outputs=("out",),
        bytes_per_element=8,
        scalars={},
        array_size_factors={"x": 2},
    ),
)


@pytest.mark.skipif(
    not _RUN_PERF,
    reason="Set TRITON_TVM_RUN_PERF_BASELINE=1 to run Triton TVM perf baseline",
)
@tvm.testing.requires_cuda
def test_pointwise_perf_baseline_regression_guard():
    results = _run_perf_baseline()
    print("TRITON_TVM_PERF_RESULT " + json.dumps(results, indent=2, sort_keys=True))

    write_path = os.environ.get("TRITON_TVM_PERF_WRITE_JSON")
    if write_path:
        Path(write_path).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    baseline_path = os.environ.get("TRITON_TVM_PERF_BASELINE_JSON")
    if baseline_path:
        _assert_no_perf_regression(results, Path(baseline_path))
    _assert_perf_smoke_no_order_of_magnitude_drop(results)


@pytest.mark.skipif(
    not _RUN_M9_MATMUL_PERF,
    reason="Set TRITON_TVM_RUN_M9_MATMUL_PERF_GUARD=1 to run the M9 matmul perf scaffold",
)
@tvm.testing.requires_cuda
def test_m9_matmul_perf_guard_scaffold():
    results = _run_m9_matmul_perf_scaffold()
    print("TRITON_TVM_M9_MATMUL_PERF_RESULT " + json.dumps(results, indent=2, sort_keys=True))

    write_path = os.environ.get("TRITON_TVM_M9_MATMUL_PERF_WRITE_JSON")
    if write_path:
        Path(write_path).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    assert results["implementation_kind"] == "native_tir_schedule"
    assert results["schedule_id"] == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert results["tvm_us"] > 0.0


@pytest.mark.skipif(
    not _RUN_M9P_MATMUL_DASHBOARD,
    reason="Set TRITON_TVM_RUN_M9P_MATMUL_DASHBOARD=1 to run the M9.P matmul dashboard",
)
@tvm.testing.requires_cuda_compute_version(7)
def test_m9p_matmul_dashboard_perf(tmp_path):
    report = run_dashboard(out_dir=tmp_path, warmup=2, repeat=5, run_benchmarks=True)
    print("TRITON_TVM_M9P_MATMUL_DASHBOARD " + json.dumps(report, indent=2, sort_keys=True))

    assert report["report_kind"] == "triton_tvm_m9p_matmul_phase1_dashboard"
    assert report["summary"]["measured_cases"] >= 3
    assert report["summary"]["tensorcore_selected_cases"] >= 3
    assert report["extern_provider_status"]["performance_claim"] is False
    for case in report["cases"]:
        if case["perf_guard_status"] != "measured":
            continue
        assert case["baseline_torch_us"] > 0.0
        assert case["baseline_triton_us"] > 0.0
        assert case["triton_tvm_us"] > 0.0
        assert case["tflops"] > 0.0
        assert case["correctness_allclose"] is True


def _run_perf_baseline() -> dict[str, Any]:
    dev = tvm.cuda(0)
    rng = np.random.default_rng(0)

    payload = {
        "schema_version": 1,
        "purpose": "pointwise performance baseline/regression guard",
        "n": _DEFAULT_N,
        "block": _DEFAULT_BLOCK,
        "tvm_number": _TVM_NUMBER,
        "tvm_repeat": _TVM_REPEAT,
        "tvm_min_repeat_ms": _TVM_MIN_REPEAT_MS,
        "triton_warmup": _TRITON_WARMUP,
        "triton_rep": _TRITON_REP,
        "broadcast_feature": _BROADCAST_FEATURE,
        "tvm_to_triton_gbps_floor": _TVM_TO_TRITON_GBPS_FLOOR,
        "device": torch.cuda.get_device_name(0),
        "tvm_version": tvm.__version__,
        "triton_version": triton.__version__,
        "cases": [],
    }

    for case in _PERF_CASES:
        payload["cases"].append(_benchmark_case(case, dev, rng))

    return payload


def _run_m9_matmul_perf_scaffold() -> dict[str, Any]:
    dev = tvm.cuda(0)
    rng = np.random.default_rng(0)
    m, n, k = 8, 4, 16
    a_np = rng.random((m, k), dtype=np.float32).astype("float16")
    b_np = rng.random((k, n), dtype=np.float32).astype("float16")

    irmod, meta = translate_ttir(
        _m9_perf_dot_ttir(),
        grid=(1,),
        target="cuda",
        contract="matmul_minimal",
    )
    built = build_triton_tvm(irmod, meta)
    args = [
        tvm.runtime.tensor(a_np, dev),
        tvm.runtime.tensor(b_np, dev),
        tvm.runtime.empty((m, n), "float32", dev),
    ]
    built.run(args)
    timer = built.executable.mod.time_evaluator(
        meta.kernel_name,
        dev,
        number=_M9_MATMUL_NUMBER,
        repeat=_M9_MATMUL_REPEAT,
        min_repeat_ms=_M9_MATMUL_MIN_REPEAT_MS,
    )
    tvm_seconds = statistics.median(timer(*args).results)
    return {
        "schema_version": 1,
        "purpose": "m9 matmul minimal perf scaffold",
        "m": m,
        "n": n,
        "k": k,
        "dtype": "float16",
        "contract": meta.contract,
        "implementation_kind": meta.implementation_kind,
        "schedule_id": meta.schedule_id,
        "tvm_number": _M9_MATMUL_NUMBER,
        "tvm_repeat": _M9_MATMUL_REPEAT,
        "tvm_min_repeat_ms": _M9_MATMUL_MIN_REPEAT_MS,
        "tvm_us": tvm_seconds * 1e6,
        "device": torch.cuda.get_device_name(0),
        "tvm_version": tvm.__version__,
        "triton_version": triton.__version__,
    }


def _m9_perf_dot_ttir() -> str:
    return """
module {
  tt.func @_m9_perf_dot(%a:!tt.ptr<f16>,%b:!tt.ptr<f16>,%out:!tt.ptr<f32>) {
    %a_zero = arith.constant dense<0> : tensor<8x16xi32>
    %b_zero = arith.constant dense<0> : tensor<16x4xi32>
    %c_zero = arith.constant dense<0> : tensor<8x4xi32>
    %a_splat = tt.splat %a : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
    %a_ptr = tt.addptr %a_splat, %a_zero : tensor<8x16x!tt.ptr<f16>>, tensor<8x16xi32>
    %va = tt.load %a_ptr : tensor<8x16x!tt.ptr<f16>>
    %b_splat = tt.splat %b : !tt.ptr<f16> -> tensor<16x4x!tt.ptr<f16>>
    %b_ptr = tt.addptr %b_splat, %b_zero : tensor<16x4x!tt.ptr<f16>>, tensor<16x4xi32>
    %vb = tt.load %b_ptr : tensor<16x4x!tt.ptr<f16>>
    %dot = tt.dot %va, %vb {inputPrecision = tf32} : tensor<8x16xf16>, tensor<16x4xf16> -> tensor<8x4xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<8x4x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %c_zero : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    tt.store %out_ptr, %dot : tensor<8x4x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _benchmark_case(case: _PerfCase, dev, rng: np.random.Generator) -> dict[str, Any]:
    n = _DEFAULT_N
    block = _DEFAULT_BLOCK
    grid = (triton.cdiv(n, block),)
    arrays = _make_arrays(case, n, rng)
    scalar_values = {"n": n, **case.scalars}

    constexprs = {"BLOCK": block, **case.constexprs}
    artifact = lower_to_ttir(case.jit_fn, case.signature, constexprs)
    irmod, meta = translate_ttir(
        artifact,
        grid=grid,
        target="cuda",
        contract=case.contract,
    )
    built = build_triton_tvm(irmod, meta)

    tvm_args = _make_tvm_args(case, arrays, scalar_values, dev)
    built.run(tvm_args)
    timer = built.executable.mod.time_evaluator(
        meta.kernel_name,
        dev,
        number=_TVM_NUMBER,
        repeat=_TVM_REPEAT,
        min_repeat_ms=_TVM_MIN_REPEAT_MS,
    )
    tvm_result = timer(*tvm_args)
    tvm_seconds = statistics.median(tvm_result.results)

    torch_args = _make_torch_args(case, arrays, scalar_values)
    triton_ms = triton.testing.do_bench(
        lambda: case.jit_fn[grid](*torch_args, **constexprs),
        warmup=_TRITON_WARMUP,
        rep=_TRITON_REP,
    )
    triton_seconds = float(triton_ms) / 1e3

    traffic_bytes = n * case.bytes_per_element
    tvm_gbps = _gbps(traffic_bytes, tvm_seconds)
    triton_gbps = _gbps(traffic_bytes, triton_seconds)
    return {
        "case": case.name,
        "contract": case.contract,
        "traffic_bytes": traffic_bytes,
        "tvm_us": tvm_seconds * 1e6,
        "tvm_gbps": tvm_gbps,
        "triton_us": triton_seconds * 1e6,
        "triton_gbps": triton_gbps,
        "tvm_to_triton_gbps": tvm_gbps / triton_gbps if triton_gbps else None,
    }


def _make_arrays(case: _PerfCase, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in case.inputs:
        arrays[name] = rng.random(_array_size(case, name, n), dtype=np.float32)
    for name in case.outputs:
        arrays[name] = np.empty(_array_size(case, name, n), dtype=np.float32)
    return arrays


def _array_size(case: _PerfCase, name: str, n: int) -> int:
    if name in case.array_size_constants:
        return int(case.array_size_constants[name])
    return n * int(case.array_size_factors.get(name, 1))


def _make_tvm_args(case: _PerfCase, arrays, scalars, dev) -> list[Any]:
    args = []
    for name, ty in case.signature.items():
        if ty == "constexpr":
            continue
        if name in arrays:
            if name in case.inputs:
                args.append(tvm.runtime.tensor(arrays[name], dev))
            else:
                args.append(tvm.runtime.empty(arrays[name].shape, "float32", dev))
        else:
            args.append(scalars[name])
    return args


def _make_torch_args(case: _PerfCase, arrays, scalars) -> list[Any]:
    args = []
    for name, ty in case.signature.items():
        if ty == "constexpr":
            continue
        if name in arrays:
            if name in case.inputs:
                args.append(torch.tensor(arrays[name], device="cuda"))
            else:
                args.append(torch.empty(arrays[name].shape, dtype=torch.float32, device="cuda"))
        else:
            args.append(scalars[name])
    return args


def _gbps(num_bytes: int, seconds: float) -> float:
    return num_bytes / seconds / 1e9


def _assert_no_perf_regression(current: dict[str, Any], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_cases = {case["case"]: case for case in baseline["cases"]}
    regressions = []

    for key in ("device", "n", "block"):
        if baseline.get(key) != current.get(key):
            regressions.append(
                f"{key}: current {current.get(key)!r} does not match "
                f"baseline {baseline.get(key)!r}"
            )

    for current_case in current["cases"]:
        name = current_case["case"]
        if name not in baseline_cases:
            continue

        baseline_gbps = float(baseline_cases[name]["tvm_gbps"])
        current_gbps = float(current_case["tvm_gbps"])
        floor_gbps = baseline_gbps * (1.0 - _REGRESSION_TOLERANCE)
        if current_gbps < floor_gbps:
            regressions.append(
                f"{name}: {current_gbps:.2f} GB/s below floor {floor_gbps:.2f} GB/s "
                f"(baseline {baseline_gbps:.2f} GB/s, tolerance {_REGRESSION_TOLERANCE:.0%})"
            )

    assert not regressions, "Triton TVM perf regression detected:\n" + "\n".join(regressions)


def _assert_perf_smoke_no_order_of_magnitude_drop(current: dict[str, Any]) -> None:
    regressions = []
    for current_case in current["cases"]:
        ratio = current_case.get("tvm_to_triton_gbps")
        if ratio is None:
            continue
        if float(ratio) < _TVM_TO_TRITON_GBPS_FLOOR:
            regressions.append(
                f"{current_case['case']}: TVM/Triton throughput ratio "
                f"{float(ratio):.3f} below smoke floor {_TVM_TO_TRITON_GBPS_FLOOR:.3f}"
            )
    assert not regressions, "Triton TVM perf smoke guard failed:\n" + "\n".join(regressions)


if __name__ == "__main__":
    tvm.testing.main()
