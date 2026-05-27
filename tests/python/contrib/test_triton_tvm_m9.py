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
"""M9 matmul correctness coverage for the Triton-to-TVM prototype."""

from types import SimpleNamespace

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import (
    M129OptimizeNativeWrapperMatmul,
    build_triton_tvm,
    register_python_torch_extern_gemm,
    translate_ttir,
    validate_matmul_minimal_contract,
)
from tvm.contrib.triton_tvm.matmul import (
    EXTERN_GEMM_PACKED_FUNC,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
    SIMT_TIR_MATMUL_SCHEDULE_ID,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_SCHEDULE_ID,
    TargetMatmulPolicy,
    build_extern_addmm_bias_tirx_source,
    build_extern_gemm_tirx_source,
    build_native_wrapper_matmul_tirx_source,
    extract_matmul_semantics_from_wrapper_extern_addmm,
    extract_matmul_semantics_from_wrapper_extern,
    is_m12_vit_native_wrapper_matmul_scope,
    register_python_torch_extern_addmm_bias,
)
from tvm.contrib.triton_tvm.m98_toy_mlp import (
    M98_BASELINE_KIND,
    M98_REPORT_KIND,
    _build_report,
    _extract_tiny_mlp_wrapper_plan,
    run_synthetic_tiny_mlp_smoke,
)
from tvm.contrib.triton_tvm.translator import TritonTVMMeta

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

try:
    import torch
except ImportError:
    torch = None


def _m9_dot_ttir(dtype: str) -> str:
    return f"""
module {{
  tt.func @_m9_dot(%a:!tt.ptr<{dtype}>,%b:!tt.ptr<{dtype}>,%out:!tt.ptr<f32>) {{
    %a_zero = arith.constant dense<0> : tensor<8x16xi32>
    %b_zero = arith.constant dense<0> : tensor<16x4xi32>
    %c_zero = arith.constant dense<0> : tensor<8x4xi32>
    %a_splat = tt.splat %a : !tt.ptr<{dtype}> -> tensor<8x16x!tt.ptr<{dtype}>>
    %a_ptr = tt.addptr %a_splat, %a_zero : tensor<8x16x!tt.ptr<{dtype}>>, tensor<8x16xi32>
    %va = tt.load %a_ptr : tensor<8x16x!tt.ptr<{dtype}>>
    %b_splat = tt.splat %b : !tt.ptr<{dtype}> -> tensor<16x4x!tt.ptr<{dtype}>>
    %b_ptr = tt.addptr %b_splat, %b_zero : tensor<16x4x!tt.ptr<{dtype}>>, tensor<16x4xi32>
    %vb = tt.load %b_ptr : tensor<16x4x!tt.ptr<{dtype}>>
    %dot = tt.dot %va, %vb {{inputPrecision = tf32}} : tensor<8x16x{dtype}>, tensor<16x4x{dtype}> -> tensor<8x4xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<8x4x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %c_zero : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    tt.store %out_ptr, %dot : tensor<8x4x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def _m97_dot_ttir(dtype: str) -> str:
    return (
        _m9_dot_ttir(dtype)
        .replace("tensor<16x4xi32>", "tensor<16x8xi32>")
        .replace("tensor<8x4xi32>", "tensor<8x8xi32>")
        .replace("tensor<16x4x!tt.ptr", "tensor<16x8x!tt.ptr")
        .replace("tensor<8x4x!tt.ptr", "tensor<8x8x!tt.ptr")
        .replace("tensor<16x4x" + dtype + ">", "tensor<16x8x" + dtype + ">")
        .replace("tensor<8x4xf32>", "tensor<8x8xf32>")
    )


def _m9p_dot_ttir(m: int, n: int, k: int, dtype: str = "f16") -> str:
    return f"""
module {{
  tt.func @_m9p_dot(%a:!tt.ptr<{dtype}>,%b:!tt.ptr<{dtype}>,%out:!tt.ptr<f32>) {{
    %a_zero = arith.constant dense<0> : tensor<{m}x{k}xi32>
    %b_zero = arith.constant dense<0> : tensor<{k}x{n}xi32>
    %c_zero = arith.constant dense<0> : tensor<{m}x{n}xi32>
    %a_splat = tt.splat %a : !tt.ptr<{dtype}> -> tensor<{m}x{k}x!tt.ptr<{dtype}>>
    %a_ptr = tt.addptr %a_splat, %a_zero : tensor<{m}x{k}x!tt.ptr<{dtype}>>, tensor<{m}x{k}xi32>
    %va = tt.load %a_ptr : tensor<{m}x{k}x!tt.ptr<{dtype}>>
    %b_splat = tt.splat %b : !tt.ptr<{dtype}> -> tensor<{k}x{n}x!tt.ptr<{dtype}>>
    %b_ptr = tt.addptr %b_splat, %b_zero : tensor<{k}x{n}x!tt.ptr<{dtype}>>, tensor<{k}x{n}xi32>
    %vb = tt.load %b_ptr : tensor<{k}x{n}x!tt.ptr<{dtype}>>
    %dot = tt.dot %va, %vb {{inputPrecision = tf32}} : tensor<{m}x{k}x{dtype}>, tensor<{k}x{n}x{dtype}> -> tensor<{m}x{n}xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<{m}x{n}x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %c_zero : tensor<{m}x{n}x!tt.ptr<f32>>, tensor<{m}x{n}xi32>
    tt.store %out_ptr, %dot : tensor<{m}x{n}x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def _extern_gemm_meta(kernel_name: str, m: int, n: int, k: int) -> TritonTVMMeta:
    abi = [
        {"name": "a", "kind": "pointer", "dtype": "float32"},
        {"name": "b", "kind": "pointer", "dtype": "float32"},
        {"name": "out", "kind": "pointer", "dtype": "float32"},
    ]
    return TritonTVMMeta(
        kernel_name=kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract="matmul_minimal",
        canonical_contract="matmul_minimal",
        requested_contract="matmul_minimal",
        emit="tir",
        translator_version="m96_extern_gemm_runtime_proof",
        contract_version="matmul_minimal_m9_v1",
        target_policy_version="m96_extern_gemm_runtime_proof",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=m,
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=k,
        buffer_extents={
            "a": f"T.int64({m * k})",
            "b": f"T.int64({k * n})",
            "out": f"T.int64({m * n})",
        },
        block_size=m,
        indexing_kind="rank2_matmul",
        execution_kind="extern_gemm_runtime_proof",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="spatial_mn_reduce_k",
        layout_policy="rank2_row_major",
        launch_policy_id="extern_gemm_runtime_proof",
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key="m96_extern_gemm_runtime_proof",
        matmul_source_kind="wrapper_extern_gemm",
        matmul_m=m,
        matmul_n=n,
        matmul_k=k,
        matmul_contract_ok=True,
        implementation_kind="extern_gemm",
        schedule_id="",
        extern_symbol="extern_kernels.mm",
        unsupported_matmul_reason="",
    )


def _extern_addmm_bias_meta(kernel_name: str, m: int, n: int, k: int) -> TritonTVMMeta:
    abi = [
        {"name": "bias", "kind": "pointer", "dtype": "float32"},
        {"name": "a", "kind": "pointer", "dtype": "float32"},
        {"name": "b", "kind": "pointer", "dtype": "float32"},
        {"name": "out", "kind": "pointer", "dtype": "float32"},
    ]
    meta = _extern_gemm_meta(kernel_name, m, n, k)
    return TritonTVMMeta(
        **{
            **meta.__dict__,
            "abi": abi,
            "buffer_extents": {
                "bias": f"T.int64({n})",
                "a": f"T.int64({m * k})",
                "b": f"T.int64({k * n})",
                "out": f"T.int64({m * n})",
            },
            "execution_kind": "extern_addmm_bias_runtime_proof",
            "launch_policy_id": "extern_addmm_bias_runtime_proof",
            "cache_key": "m9p_extern_addmm_bias_runtime_proof",
            "matmul_source_kind": "wrapper_extern_addmm_bias",
            "implementation_kind": "extern_addmm_bias",
            "extern_symbol": "extern_kernels.addmm",
        }
    )


_M98_WRAPPER_FIXTURE = """
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK):
    return
''')

class Runner:
    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        assert_size_stride(arg0_1, (100, 1, 28, 28), (784, 784, 28, 1))
        assert_size_stride(arg1_1, (128, 784), (784, 1))
        arg0_1 = copy_misaligned(arg0_1)
        buf0 = empty_strided_cuda((100, 128), (128, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(arg0_1, (100, 784), (784, 1), 0), reinterpret_tensor(arg1_1, (784, 128), (1, 784), 0), out=buf0)
        buf1 = buf0; del buf0
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(buf1, 12800, stream=raw_stream0)
        assert_size_stride(arg2_1, (10, 128), (128, 1))
        buf2 = empty_strided_cuda((100, 10), (10, 1), torch.float32)
        extern_kernels.mm(buf1, reinterpret_tensor(arg2_1, (128, 10), (1, 128), 0), out=buf2)
        return (buf2,)
"""


@tvm.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_m9_native_matmul_minimal_build_run(dtype):
    if dtype == "bf16" and ml_dtypes is None:
        pytest.skip("ml_dtypes is required for bfloat16 numpy conversion")

    a_f32 = np.arange(8 * 16, dtype="float32").reshape(8, 16) / 13.0
    b_f32 = np.arange(16 * 4, dtype="float32").reshape(16, 4) / 17.0
    if dtype == "bf16":
        a_np = a_f32.astype(ml_dtypes.bfloat16)
        b_np = b_f32.astype(ml_dtypes.bfloat16)
    else:
        a_np = a_f32.astype("float16")
        b_np = b_f32.astype("float16")
    expected = a_np.astype("float32") @ b_np.astype("float32")

    irmod, meta = translate_ttir(_m9_dot_ttir(dtype), grid=(1,), contract="matmul_minimal")
    validate_matmul_minimal_contract(irmod)
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.schedule_id == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert meta.unsupported_matmul_reason == ""

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((8, 4), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(7)
@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 32, 32)])
def test_m9p_tensorcore_matmul_build_run(shape):
    m, n, k = shape
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1, 1, (m, k)).astype("float16")
    b_np = rng.uniform(-1, 1, (k, n)).astype("float16")
    expected = a_np.astype("float32") @ b_np.astype("float32")

    irmod, meta = translate_ttir(
        _m9p_dot_ttir(m, n, k),
        grid=(1,),
        contract="matmul_minimal",
    )
    validate_matmul_minimal_contract(irmod)
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.schedule_id == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert meta.selected_schedule_id == TENSORCORE_TIR_MATMUL_SCHEDULE_ID

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((m, n), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-2, atol=1e-2)


@tvm.testing.requires_cuda
def test_m9p_simt_bf16_matmul_build_run():
    if ml_dtypes is None:
        pytest.skip("ml_dtypes is required for bfloat16 numpy conversion")

    m, n, k = 16, 16, 16
    a_f32 = np.arange(m * k, dtype="float32").reshape(m, k) / 13.0
    b_f32 = np.arange(k * n, dtype="float32").reshape(k, n) / 17.0
    a_np = a_f32.astype(ml_dtypes.bfloat16)
    b_np = b_f32.astype(ml_dtypes.bfloat16)
    expected = a_np.astype("float32") @ b_np.astype("float32")

    irmod, meta = translate_ttir(
        _m9p_dot_ttir(m, n, k, dtype="bf16"),
        grid=(1,),
        contract="matmul_minimal",
    )
    validate_matmul_minimal_contract(irmod)
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.schedule_id == SIMT_TIR_MATMUL_SCHEDULE_ID
    assert meta.selected_schedule_id == SIMT_TIR_MATMUL_SCHEDULE_ID

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((m, n), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-2, atol=1e-2)


def test_m98_wrapper_plan_extracts_two_runtime_resolved_gemms_and_relu():
    plan = _extract_tiny_mlp_wrapper_plan(_M98_WRAPPER_FIXTURE, batch_size=100)

    assert len(plan.gemm_records) == 2
    assert len(plan.relu_sources) == 1
    assert [record.matmul_m for record in plan.gemm_records] == [100, 100]
    assert [record.matmul_n for record in plan.gemm_records] == [128, 10]
    assert [record.matmul_k for record in plan.gemm_records] == [784, 128]
    assert all(
        record.extern_gemm_provider_kind == "python_torch_host_staged"
        for record in plan.gemm_records
    )
    assert all(record.extern_gemm_runtime_status == "runtime_resolved" for record in plan.gemm_records)
    assert all(record.extern_gemm_performance_claim is False for record in plan.gemm_records)
    assert "reinterpret_tensor(buf1, (100, 128), (128, 1), 0)" in (
        plan.gemm_records[1].normalized_source
    )


def test_m98_report_marks_host_staged_path_as_diagnostic_not_perf(tmp_path):
    plan = _extract_tiny_mlp_wrapper_plan(_M98_WRAPPER_FIXTURE, batch_size=100)
    executable = SimpleNamespace(
        wrapper_plan=plan,
        relu_record={
            "kernel_name": "triton_poi_fused_relu_0",
            "translate_status": {"ok": True, "bucket": "translated", "fallback_reason": ""},
        },
    )
    report = _build_report(
        mnist_root=tmp_path / "mnist",
        sample_count=10000,
        batch_size=100,
        timed_epochs=3,
        download_mnist=True,
        seed=0,
        executable=executable,
        correctness={
            "samples": 10000,
            "correct": 1000,
            "top1_accuracy": 0.1,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "allclose": True,
            "allclose_rtol": 1e-4,
            "allclose_atol": 1e-4,
            "accuracy_note": "random untrained TinyMNISTMLP",
        },
        performance={
            "timed_batches": 300,
            "timed_images": 30000,
            "batch_latency_ms_min": 1.0,
            "batch_latency_ms_p50": 1.5,
            "batch_latency_ms_p95": 2.0,
            "batch_latency_ms_mean": 1.6,
            "batch_latency_ms_max": 2.5,
            "per_image_latency_us_p50": 15.0,
            "per_image_latency_us_mean": 16.0,
            "images_per_second_mean": 62500.0,
            "images_per_second_p50": 66666.0,
        },
    )

    assert report["report_kind"] == M98_REPORT_KIND
    assert report["baseline_kind"] == M98_BASELINE_KIND
    assert report["performance_claim"] is False
    provider = report["artifacts"]["extern_gemm_provider"]
    assert provider["kind"] == "python_torch_host_staged"
    assert provider["performance_claim"] is False
    assert provider["uses_host_staging"] is True
    assert report["artifacts"]["wrapper"]["runtime_resolved_extern_gemm_count"] == 2
    assert report["diagnostics"]["silent_fallback_records"] == []
    assert report["diagnostics"]["no_silent_fallback"] is True


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m98_synthetic_staged_tiny_mlp_matches_pytorch():
    result = run_synthetic_tiny_mlp_smoke(batch_size=8)

    assert result["allclose"] is True
    assert result["max_abs_error"] <= 1e-4
    assert result["wrapper_plan"]["runtime_resolved_extern_gemm_count"] == 2
    assert result["wrapper_plan"]["relu_triton_kernel_count"] == 1


@tvm.testing.requires_cuda
def test_m96_extern_gemm_runtime_provider_row_major_build_run():
    source_line = (
        "extern_kernels.mm(reinterpret_tensor(a, (8, 16), (16, 1), 0), "
        "reinterpret_tensor(b, (16, 4), (4, 1), 0), out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        source_line,
        kernel_name="m96_extern_gemm_row_major",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, matmul_contract_ok=True)
    assert decision.extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED

    irmod = tvm.script.from_source(build_extern_gemm_tirx_source(semantics, decision))
    validate_matmul_minimal_contract(irmod)
    built = build_triton_tvm(
        irmod,
        _extern_gemm_meta("m96_extern_gemm_row_major", 8, 4, 16),
    )

    dev = tvm.cuda(0)
    a_np = np.arange(8 * 16, dtype="float32").reshape(8, 16) / 13.0
    b_np = np.arange(16 * 4, dtype="float32").reshape(16, 4) / 17.0
    out_tvm = tvm.runtime.empty((8, 4), "float32", dev)

    before = tvm.get_global_func(EXTERN_GEMM_PACKED_FUNC, allow_missing=True)
    with register_python_torch_extern_gemm():
        built.run(
            [
                tvm.runtime.tensor(a_np, dev),
                tvm.runtime.tensor(b_np, dev),
                out_tvm,
            ]
        )
    after = tvm.get_global_func(EXTERN_GEMM_PACKED_FUNC, allow_missing=True)
    assert (after is None) if before is None else (after is not None)
    tvm.testing.assert_allclose(out_tvm.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_m96_extern_gemm_runtime_provider_transposed_weight_build_run():
    source_line = (
        "extern_kernels.mm(reinterpret_tensor(a, (8, 16), (16, 1), 0), "
        "reinterpret_tensor(b, (16, 4), (1, 16), 0), out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        source_line,
        kernel_name="m96_extern_gemm_transposed_weight",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_gemm_tirx_source(semantics, decision))
    validate_matmul_minimal_contract(irmod)
    built = build_triton_tvm(
        irmod,
        _extern_gemm_meta("m96_extern_gemm_transposed_weight", 8, 4, 16),
    )

    dev = tvm.cuda(0)
    a_np = np.arange(8 * 16, dtype="float32").reshape(8, 16) / 13.0
    b_storage_np = np.arange(4 * 16, dtype="float32").reshape(4, 16) / 17.0
    out_tvm = tvm.runtime.empty((8, 4), "float32", dev)

    with register_python_torch_extern_gemm():
        built.run(
            [
                tvm.runtime.tensor(a_np, dev),
                tvm.runtime.tensor(b_storage_np, dev),
                out_tvm,
            ]
        )
    tvm.testing.assert_allclose(
        out_tvm.numpy(),
        a_np @ b_storage_np.T,
        rtol=1e-5,
        atol=1e-5,
    )


@tvm.testing.requires_cuda
def test_m9p_extern_addmm_bias_runtime_provider_build_run():
    source_line = (
        "extern_kernels.addmm(bias, reinterpret_tensor(a, (8, 16), (16, 1), 0), "
        "reinterpret_tensor(b, (16, 4), (4, 1), 0), alpha=1, beta=1, out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern_addmm(
        source_line,
        kernel_name="m9p_extern_addmm_bias",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, matmul_contract_ok=True)
    assert decision.implementation_kind == "extern_addmm_bias"
    assert decision.extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert decision.extern_gemm_performance_claim is False

    irmod = tvm.script.from_source(build_extern_addmm_bias_tirx_source(semantics, decision))
    validate_matmul_minimal_contract(irmod)
    built = build_triton_tvm(
        irmod,
        _extern_addmm_bias_meta("m9p_extern_addmm_bias", 8, 4, 16),
    )

    dev = tvm.cuda(0)
    bias_np = np.arange(4, dtype="float32") / 19.0
    a_np = np.arange(8 * 16, dtype="float32").reshape(8, 16) / 13.0
    b_np = np.arange(16 * 4, dtype="float32").reshape(16, 4) / 17.0
    out_tvm = tvm.runtime.empty((8, 4), "float32", dev)

    with register_python_torch_extern_addmm_bias():
        built.run(
            [
                tvm.runtime.tensor(bias_np, dev),
                tvm.runtime.tensor(a_np, dev),
                tvm.runtime.tensor(b_np, dev),
                out_tvm,
            ]
        )
    tvm.testing.assert_allclose(out_tvm.numpy(), a_np @ b_np + bias_np, rtol=1e-5, atol=1e-5)


def test_m12_native_wrapper_matmul_policy_and_contract_validation():
    gemm_source = (
        "extern_kernels.mm(reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), out=out)"
    )
    non_square_gemm_source = (
        "extern_kernels.mm(reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 128), (1, 64), 0), out=out)"
    )
    addmm_source = (
        "extern_kernels.addmm(bias, reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), alpha=1, beta=1, out=out)"
    )
    for source, extractor, kernel_name in (
        (gemm_source, extract_matmul_semantics_from_wrapper_extern, "m12_native_gemm"),
        (
            non_square_gemm_source,
            extract_matmul_semantics_from_wrapper_extern,
            "m12_native_gemm_non_square",
        ),
        (
            addmm_source,
            extract_matmul_semantics_from_wrapper_extern_addmm,
            "m12_native_addmm",
        ),
    ):
        semantics = extractor(source, kernel_name=kernel_name)
        assert is_m12_vit_native_wrapper_matmul_scope(semantics) is True
        decision = TargetMatmulPolicy(
            extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM
        ).decide(semantics, matmul_contract_ok=True)

        assert decision.implementation_kind == "native_tir_schedule"
        assert decision.schedule_id == NATIVE_TIR_MATMUL_SCHEDULE_ID
        assert decision.extern_gemm_provider_kind == EXTERN_GEMM_PROVIDER_NATIVE_TVM
        assert decision.extern_gemm_runtime_claim == (
            EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE
        )
        assert decision.extern_gemm_uses_host_staging is False
        assert decision.extern_gemm_performance_claim is False

        irmod = tvm.script.from_source(
            build_native_wrapper_matmul_tirx_source(semantics, decision)
        )
        validate_matmul_minimal_contract(irmod)


def test_m12_native_wrapper_matmul_rejects_stale_host_staging_metadata():
    source = (
        "extern_kernels.mm(reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        source,
        kernel_name="m12_native_bad_host_staging",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM
    ).decide(semantics, matmul_contract_ok=True)
    source_text = build_native_wrapper_matmul_tirx_source(semantics, decision).replace(
        '"triton_tvm.extern_gemm_uses_host_staging": False',
        '"triton_tvm.extern_gemm_uses_host_staging": True',
    )
    irmod = tvm.script.from_source(source_text)

    with pytest.raises(Exception, match="must not host stage"):
        validate_matmul_minimal_contract(irmod)


def test_m129_backend_pass_rewrites_native_wrapper_matmul_schedule():
    source_line = (
        "extern_kernels.mm(reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        source_line,
        kernel_name="m129_backend_pass_gemm",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_native_wrapper_matmul_tirx_source(semantics, decision))
    optimized = M129OptimizeNativeWrapperMatmul()(irmod)
    validate_matmul_minimal_contract(optimized)

    func = next(iter(optimized.functions.values()))
    assert str(func.attrs["triton_tvm.schedule_id"]) == (
        M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
    )
    script = optimized.script()
    assert 'thread="blockIdx.x"' in script
    assert 'thread="threadIdx.x"' in script
    assert "T.thread_binding(5" in script
    assert "T.thread_binding(64" in script


@tvm.testing.requires_cuda
def test_m12_native_wrapper_gemm_transposed_weight_build_run():
    source_line = (
        "extern_kernels.mm(reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        source_line,
        kernel_name="m12_native_wrapper_gemm",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_native_wrapper_matmul_tirx_source(semantics, decision))
    validate_matmul_minimal_contract(irmod)
    built = build_triton_tvm(irmod, _extern_gemm_meta("m12_native_wrapper_gemm", 5, 64, 64))

    dev = tvm.cuda(0)
    a_np = np.arange(5 * 64, dtype="float32").reshape(5, 64) / 13.0
    b_storage_np = np.arange(64 * 64, dtype="float32").reshape(64, 64) / 17.0
    out_tvm = tvm.runtime.empty((5, 64), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_storage_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), a_np @ b_storage_np.T, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_m12_native_wrapper_addmm_bias_build_run():
    source_line = (
        "extern_kernels.addmm(bias, reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), alpha=1, beta=1, out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern_addmm(
        source_line,
        kernel_name="m12_native_wrapper_addmm",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_native_wrapper_matmul_tirx_source(semantics, decision))
    validate_matmul_minimal_contract(irmod)
    built = build_triton_tvm(
        irmod,
        _extern_addmm_bias_meta("m12_native_wrapper_addmm", 5, 64, 64),
    )

    dev = tvm.cuda(0)
    bias_np = np.arange(64, dtype="float32") / 19.0
    a_np = np.arange(5 * 64, dtype="float32").reshape(5, 64) / 13.0
    b_storage_np = np.arange(64 * 64, dtype="float32").reshape(64, 64) / 17.0
    out_tvm = tvm.runtime.empty((5, 64), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(bias_np, dev),
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_storage_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(
        out_tvm.numpy(),
        a_np @ b_storage_np.T + bias_np,
        rtol=1e-5,
        atol=1e-5,
    )


@tvm.testing.requires_cuda
def test_m129_backend_pass_native_wrapper_addmm_bias_build_run():
    source_line = (
        "extern_kernels.addmm(bias, reinterpret_tensor(a, (5, 64), (64, 1), 0), "
        "reinterpret_tensor(b, (64, 64), (1, 64), 0), alpha=1, beta=1, out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern_addmm(
        source_line,
        kernel_name="m129_native_wrapper_addmm",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_native_wrapper_matmul_tirx_source(semantics, decision))
    built = build_triton_tvm(
        irmod,
        _extern_addmm_bias_meta("m129_native_wrapper_addmm", 5, 64, 64),
        passes=[M129OptimizeNativeWrapperMatmul()],
    )

    dev = tvm.cuda(0)
    bias_np = np.arange(64, dtype="float32") / 19.0
    a_np = np.arange(5 * 64, dtype="float32").reshape(5, 64) / 13.0
    b_storage_np = np.arange(64 * 64, dtype="float32").reshape(64, 64) / 17.0
    out_tvm = tvm.runtime.empty((5, 64), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(bias_np, dev),
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_storage_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(
        out_tvm.numpy(),
        a_np @ b_storage_np.T + bias_np,
        rtol=1e-5,
        atol=1e-5,
    )


@tvm.testing.requires_cuda
def test_m96_extern_gemm_runtime_provider_rejects_unsupported_dtype():
    source_line = (
        "extern_kernels.mm(reinterpret_tensor(a, (8, 16), (16, 1), 0), "
        "reinterpret_tensor(b, (16, 4), (4, 1), 0), out=out)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        source_line,
        kernel_name="m96_extern_gemm_reject_dtype",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_gemm_tirx_source(semantics, decision))
    built = build_triton_tvm(
        irmod,
        _extern_gemm_meta("m96_extern_gemm_reject_dtype", 8, 4, 16),
    )

    dev = tvm.cuda(0)
    with register_python_torch_extern_gemm():
        with pytest.raises(TypeError, match="expected float32"):
            built.run(
                [
                    tvm.runtime.empty((8, 16), "float16", dev),
                    tvm.runtime.empty((16, 4), "float32", dev),
                    tvm.runtime.empty((8, 4), "float32", dev),
                ]
            )


@tvm.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_m97_tiled_matmul_minimal_build_run(dtype):
    if dtype == "bf16" and ml_dtypes is None:
        pytest.skip("ml_dtypes is required for bfloat16 numpy conversion")

    a_f32 = np.arange(8 * 16, dtype="float32").reshape(8, 16) / 13.0
    b_f32 = np.arange(16 * 8, dtype="float32").reshape(16, 8) / 17.0
    if dtype == "bf16":
        a_np = a_f32.astype(ml_dtypes.bfloat16)
        b_np = b_f32.astype(ml_dtypes.bfloat16)
    else:
        a_np = a_f32.astype("float16")
        b_np = b_f32.astype("float16")
    expected = a_np.astype("float32") @ b_np.astype("float32")

    irmod, meta = translate_ttir(_m97_dot_ttir(dtype), grid=(1,), contract="matmul_minimal")
    validate_matmul_minimal_contract(irmod)
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.schedule_id == TILED_TIR_MATMUL_SCHEDULE_ID

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((8, 8), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(a_np, dev),
            tvm.runtime.tensor(b_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-3, atol=1e-3)
