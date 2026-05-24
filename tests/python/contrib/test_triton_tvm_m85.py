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
"""M8.5 numerics and dynamic shape hardening tests."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import (
    build_triton_tvm,
    translate_ttir,
    validate_masked_softmax_row_contract,
    validate_norm_row_contract,
    validate_row_reduction_contract,
)
from tvm.contrib.triton_tvm.model_corpus import build_model_corpus_report
from tvm.contrib.triton_tvm.reporting import build_capability_report, make_report_status

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


_STABLE_REPORT_BUCKETS = {
    "translated",
    "unsupported_ttir_op",
    "contract_error",
    "target_policy_error",
    "unsupported_stream",
    "input_error",
    "collection_error",
    "triton_tvm_error",
    "internal_error",
}


def _row_sum_ttir(dtype: str = "f32") -> str:
    return f"""
module {{
  tt.func @_m85_row_sum(%x:!tt.ptr<{dtype}>,%out:!tt.ptr<{dtype}>) {{
    %zero = arith.constant dense<0.000000e+00> : tensor<2x4x{dtype}>
    %xmask_c = arith.constant dense<2> : tensor<2x1xi32>
    %rmask_c = arith.constant dense<4> : tensor<1x4xi32>
    %c2_i32 = arith.constant 2 : i32
    %row_stride = arith.constant dense<4> : tensor<2x1xi32>
    %pid = tt.get_program_id x : i32
    %xbase = arith.muli %pid, %c2_i32 : i32
    %xrange = tt.make_range {{end = 2 : i32, start = 0 : i32}} : tensor<2xi32>
    %xrange_e = tt.expand_dims %xrange {{axis = 1 : i32}} : tensor<2xi32> -> tensor<2x1xi32>
    %xbase_s = tt.splat %xbase : i32 -> tensor<2x1xi32>
    %xidx = arith.addi %xbase_s, %xrange_e : tensor<2x1xi32>
    %xmask = arith.cmpi slt, %xidx, %xmask_c : tensor<2x1xi32>
    %rrange = tt.make_range {{end = 4 : i32, start = 0 : i32}} : tensor<4xi32>
    %ridx = tt.expand_dims %rrange {{axis = 0 : i32}} : tensor<4xi32> -> tensor<1x4xi32>
    %rmask = arith.cmpi slt, %ridx, %rmask_c : tensor<1x4xi32>
    %xmask_b = tt.broadcast %xmask : tensor<2x1xi1> -> tensor<2x4xi1>
    %rmask_b = tt.broadcast %rmask : tensor<1x4xi1> -> tensor<2x4xi1>
    %mask = arith.andi %xmask_b, %rmask_b : tensor<2x4xi1>
    %base = arith.muli %xidx, %row_stride : tensor<2x1xi32>
    %base_b = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x4xi32>
    %ridx_b = tt.broadcast %ridx : tensor<1x4xi32> -> tensor<2x4xi32>
    %idx = arith.addi %base_b, %ridx_b : tensor<2x4xi32>
    %x_s = tt.splat %x : !tt.ptr<{dtype}> -> tensor<2x4x!tt.ptr<{dtype}>>
    %x_ptr = tt.addptr %x_s, %idx : tensor<2x4x!tt.ptr<{dtype}>>, tensor<2x4xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<2x4x!tt.ptr<{dtype}>>
    %acc = "tt.reduce"(%vx) <{{axis = 1 : i32}}> ({{
    ^bb0(%lhs: {dtype}, %rhs: {dtype}):
      %sum = arith.addf %lhs, %rhs : {dtype}
      tt.reduce.return %sum : {dtype}
    }}) : (tensor<2x4x{dtype}>) -> tensor<2x{dtype}>
    %acc_e = tt.expand_dims %acc {{axis = 1 : i32}} : tensor<2x{dtype}> -> tensor<2x1x{dtype}>
    %out_s = tt.splat %out : !tt.ptr<{dtype}> -> tensor<2x1x!tt.ptr<{dtype}>>
    %out_ptr = tt.addptr %out_s, %xidx : tensor<2x1x!tt.ptr<{dtype}>>, tensor<2x1xi32>
    tt.store %out_ptr, %acc_e, %xmask : tensor<2x1x!tt.ptr<{dtype}>>
    tt.return
  }}
}}
"""


def _dynamic_row_sum_ttir(param_dtype: str = "i32") -> str:
    return f"""
module {{
  tt.func @_m85_row_sum_dyn(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>,%m:{param_dtype},%n:{param_dtype}) {{
    %zero = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
    %c2_i32 = arith.constant 2 : i32
    %pid = tt.get_program_id x : i32
    %xbase = arith.muli %pid, %c2_i32 : i32
    %xrange = tt.make_range {{end = 2 : i32, start = 0 : i32}} : tensor<2xi32>
    %xrange_e = tt.expand_dims %xrange {{axis = 1 : i32}} : tensor<2xi32> -> tensor<2x1xi32>
    %xbase_s = tt.splat %xbase : i32 -> tensor<2x1xi32>
    %xidx = arith.addi %xbase_s, %xrange_e : tensor<2x1xi32>
    %m_s = tt.splat %m : {param_dtype} -> tensor<2x1x{param_dtype}>
    %xmask = arith.cmpi slt, %xidx, %m_s : tensor<2x1x{param_dtype}>
    %rrange = tt.make_range {{end = 4 : i32, start = 0 : i32}} : tensor<4xi32>
    %ridx = tt.expand_dims %rrange {{axis = 0 : i32}} : tensor<4xi32> -> tensor<1x4xi32>
    %n_s_r = tt.splat %n : {param_dtype} -> tensor<1x4x{param_dtype}>
    %rmask = arith.cmpi slt, %ridx, %n_s_r : tensor<1x4x{param_dtype}>
    %xmask_b = tt.broadcast %xmask : tensor<2x1xi1> -> tensor<2x4xi1>
    %rmask_b = tt.broadcast %rmask : tensor<1x4xi1> -> tensor<2x4xi1>
    %mask = arith.andi %xmask_b, %rmask_b : tensor<2x4xi1>
    %n_s_x = tt.splat %n : {param_dtype} -> tensor<2x1x{param_dtype}>
    %base = arith.muli %xidx, %n_s_x : tensor<2x1x{param_dtype}>
    %base_b = tt.broadcast %base : tensor<2x1x{param_dtype}> -> tensor<2x4x{param_dtype}>
    %ridx_b = tt.broadcast %ridx : tensor<1x4xi32> -> tensor<2x4xi32>
    %idx = arith.addi %base_b, %ridx_b : tensor<2x4x{param_dtype}>
    %x_s = tt.splat %x : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_s, %idx : tensor<2x4x!tt.ptr<f32>>, tensor<2x4x{param_dtype}>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<2x4x!tt.ptr<f32>>
    %acc = "tt.reduce"(%vx) <{{axis = 1 : i32}}> ({{
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }}) : (tensor<2x4xf32>) -> tensor<2xf32>
    %acc_e = tt.expand_dims %acc {{axis = 1 : i32}} : tensor<2xf32> -> tensor<2x1xf32>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_s, %xidx : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    tt.store %out_ptr, %acc_e, %xmask : tensor<2x1x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def _masked_softmax_ttir(*, rmask_extent: int = 4, causal: bool = False) -> str:
    mask_setup = "%mask = arith.andi %xmask_b, %rmask_b : tensor<2x4xi1>"
    if causal:
        mask_setup = """
    %xidx_b = tt.broadcast %xidx : tensor<2x1xi32> -> tensor<2x4xi32>
    %causal = arith.cmpi sle, %ridx_b, %xidx_b : tensor<2x4xi32>
    %row_rmask = arith.andi %rmask_b, %causal : tensor<2x4xi1>
    %mask = arith.andi %xmask_b, %row_rmask : tensor<2x4xi1>""".rstrip()
    return f"""
module {{
  tt.func @_m85_masked_softmax(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {{
    %neg_inf = arith.constant dense<0xFF800000> : tensor<2x4xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
    %xmask_c = arith.constant dense<2> : tensor<2x1xi32>
    %rmask_c = arith.constant dense<{rmask_extent}> : tensor<1x4xi32>
    %c2_i32 = arith.constant 2 : i32
    %row_stride = arith.constant dense<4> : tensor<2x1xi32>
    %pid = tt.get_program_id x : i32
    %xbase = arith.muli %pid, %c2_i32 : i32
    %xrange = tt.make_range {{end = 2 : i32, start = 0 : i32}} : tensor<2xi32>
    %xrange_e = tt.expand_dims %xrange {{axis = 1 : i32}} : tensor<2xi32> -> tensor<2x1xi32>
    %xbase_s = tt.splat %xbase : i32 -> tensor<2x1xi32>
    %xidx = arith.addi %xbase_s, %xrange_e : tensor<2x1xi32>
    %xmask = arith.cmpi slt, %xidx, %xmask_c : tensor<2x1xi32>
    %rrange = tt.make_range {{end = 4 : i32, start = 0 : i32}} : tensor<4xi32>
    %ridx = tt.expand_dims %rrange {{axis = 0 : i32}} : tensor<4xi32> -> tensor<1x4xi32>
    %rmask = arith.cmpi slt, %ridx, %rmask_c : tensor<1x4xi32>
    %xmask_b = tt.broadcast %xmask : tensor<2x1xi1> -> tensor<2x4xi1>
    %rmask_b = tt.broadcast %rmask : tensor<1x4xi1> -> tensor<2x4xi1>
    %base = arith.muli %xidx, %row_stride : tensor<2x1xi32>
    %base_b = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x4xi32>
    %ridx_b = tt.broadcast %ridx : tensor<1x4xi32> -> tensor<2x4xi32>
{mask_setup}
    %idx = arith.addi %base_b, %ridx_b : tensor<2x4xi32>
    %x_s = tt.splat %x : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_s, %idx : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<2x4x!tt.ptr<f32>>
    %masked_vx = arith.select %mask, %vx, %neg_inf : tensor<2x4xi1>, tensor<2x4xf32>
    %maxv = "tt.reduce"(%masked_vx) <{{axis = 1 : i32}}> ({{
    ^bb0(%lhs: f32, %rhs: f32):
      %gt = arith.cmpf ogt, %lhs, %rhs : f32
      %nan = arith.cmpf une, %lhs, %lhs : f32
      %take_lhs = arith.ori %gt, %nan : i1
      %mx = arith.select %take_lhs, %lhs, %rhs : i1, f32
      tt.reduce.return %mx : f32
    }}) : (tensor<2x4xf32>) -> tensor<2xf32>
    %max_e = tt.expand_dims %maxv {{axis = 1 : i32}} : tensor<2xf32> -> tensor<2x1xf32>
    %max_b = tt.broadcast %max_e : tensor<2x1xf32> -> tensor<2x4xf32>
    %shift = arith.subf %vx, %max_b : tensor<2x4xf32>
    %exp = tt.extern_elementwise %shift {{libname = "", libpath = "", pure = true, symbol = "__nv_expf"}} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %masked_exp = arith.select %mask, %exp, %zero : tensor<2x4xi1>, tensor<2x4xf32>
    %den = "tt.reduce"(%masked_exp) <{{axis = 1 : i32}}> ({{
    ^bb0(%lhs2: f32, %rhs2: f32):
      %sum = arith.addf %lhs2, %rhs2 : f32
      tt.reduce.return %sum : f32
    }}) : (tensor<2x4xf32>) -> tensor<2xf32>
    %den_e = tt.expand_dims %den {{axis = 1 : i32}} : tensor<2xf32> -> tensor<2x1xf32>
    %den_b = tt.broadcast %den_e : tensor<2x1xf32> -> tensor<2x4xf32>
    %soft = arith.divf %exp, %den_b : tensor<2x4xf32>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_s, %idx : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptr, %soft, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


_RMSNORM_ROW_TTIR = """
module {
  tt.func @_m85_rmsnorm_row(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
    %eps = arith.constant dense<1.000000e-05> : tensor<2xf32>
    %denom = arith.constant dense<4.000000e+00> : tensor<2xf32>
    %xmask_c = arith.constant dense<2> : tensor<2x1xi32>
    %rmask_c = arith.constant dense<4> : tensor<1x4xi32>
    %c2_i32 = arith.constant 2 : i32
    %row_stride = arith.constant dense<4> : tensor<2x1xi32>
    %pid = tt.get_program_id x : i32
    %xbase = arith.muli %pid, %c2_i32 : i32
    %xrange = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %xrange_e = tt.expand_dims %xrange {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %xbase_s = tt.splat %xbase : i32 -> tensor<2x1xi32>
    %xidx = arith.addi %xbase_s, %xrange_e : tensor<2x1xi32>
    %xmask = arith.cmpi slt, %xidx, %xmask_c : tensor<2x1xi32>
    %rrange = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %ridx = tt.expand_dims %rrange {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %rmask = arith.cmpi slt, %ridx, %rmask_c : tensor<1x4xi32>
    %xmask_b = tt.broadcast %xmask : tensor<2x1xi1> -> tensor<2x4xi1>
    %rmask_b = tt.broadcast %rmask : tensor<1x4xi1> -> tensor<2x4xi1>
    %mask = arith.andi %xmask_b, %rmask_b : tensor<2x4xi1>
    %base = arith.muli %xidx, %row_stride : tensor<2x1xi32>
    %base_b = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x4xi32>
    %ridx_b = tt.broadcast %ridx : tensor<1x4xi32> -> tensor<2x4xi32>
    %idx = arith.addi %base_b, %ridx_b : tensor<2x4xi32>
    %x_s = tt.splat %x : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_s, %idx : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<2x4x!tt.ptr<f32>>
    %sq = arith.mulf %vx, %vx : tensor<2x4xf32>
    %ss = "tt.reduce"(%sq) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<2x4xf32>) -> tensor<2xf32>
    %mean = arith.divf %ss, %denom : tensor<2xf32>
    %var_eps = arith.addf %mean, %eps : tensor<2xf32>
    %scale = tt.extern_elementwise %var_eps {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<2xf32>) -> tensor<2xf32>
    %scale_e = tt.expand_dims %scale {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
    %scale_b = tt.broadcast %scale_e : tensor<2x1xf32> -> tensor<2x4xf32>
    %outv = arith.mulf %vx, %scale_b : tensor<2x4xf32>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_s, %idx : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptr, %outv, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _to_tvm(array: np.ndarray, dev):
    return tvm.runtime.tensor(array.reshape(-1), dev)


def _softmax_expected(x_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expected = np.full_like(x_np, -9.0)
    for row in range(x_np.shape[0]):
        values = x_np[row, mask[row]]
        shifted = values - values.max()
        expected[row, mask[row]] = np.exp(shifted) / np.exp(shifted).sum()
    return expected


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    ("dtype", "np_dtype", "rtol", "atol"),
    [
        ("f32", np.float32, 1e-5, 1e-5),
        ("f16", np.float16, 1e-3, 1e-3),
    ],
)
def test_m85_row_sum_fp32_fp16_build_run(dtype, np_dtype, rtol, atol):
    x_np = (np.arange(8, dtype="float32").reshape(2, 4) / 10.0).astype(np_dtype)
    expected = x_np.astype("float32").sum(axis=1).astype(np_dtype)
    irmod, meta = translate_ttir(_row_sum_ttir(dtype), grid=(1,), contract="row_reduction")
    validate_row_reduction_contract(irmod)
    assert meta.reduction_extent_kind == "constant"
    assert meta.reduction_extent_value == 4

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((2,), str(np.dtype(np_dtype)), dev)
    built.run([_to_tvm(x_np, dev), out_tvm])
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=rtol, atol=atol)


@tvm.testing.requires_cuda
def test_m85_row_sum_bfloat16_build_run_after_storage_legalizer_fix():
    if ml_dtypes is None:
        pytest.skip("ml_dtypes is required for bfloat16 numpy conversion")

    x_np = np.array([[1.0, 2.0, 3.0, 4.0], [-1.5, 0.5, 1.0, 2.5]], dtype="float32")
    x_bf16 = x_np.astype(ml_dtypes.bfloat16)
    expected = x_bf16.astype("float32").sum(axis=1).astype(ml_dtypes.bfloat16)

    irmod, meta = translate_ttir(_row_sum_ttir("bf16"), grid=(1,), contract="row_reduction")
    validate_row_reduction_contract(irmod)
    assert meta.abi[0]["dtype"] == "bfloat16"
    assert meta.reduction_extent_value == 4

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((2,), "bfloat16", dev)
    built.run([_to_tvm(x_bf16, dev), out_tvm])
    tvm.testing.assert_allclose(
        out_tvm.numpy().astype("float32"),
        expected.astype("float32"),
        rtol=0,
        atol=0,
    )


@tvm.testing.requires_cuda
def test_m85_dynamic_i32_rank2_row_sum_reuses_one_compiled_shape_policy():
    irmod, meta = translate_ttir(
        _dynamic_row_sum_ttir("i32"), grid=(1,), contract="row_reduction"
    )
    validate_row_reduction_contract(irmod)
    assert meta.extent_param == "m"
    assert meta.extent_kind == "runtime_param"
    assert meta.reduction_extent_param == "n"
    assert meta.reduction_extent_kind == "runtime_param"
    assert meta.buffer_extents == {
        "x": '(T.Cast("int64", m) * T.Cast("int64", n))',
        "out": 'T.Cast("int64", m)',
    }

    script = irmod.script()
    assert 'T.Cast("int64", m)' in script
    assert 'T.Cast("int64", n)' in script

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    for m, n in [(2, 3), (1, 4), (2, 4)]:
        x_np = np.arange(m * n, dtype="float32").reshape(m, n)
        out_tvm = tvm.runtime.empty((m,), "float32", dev)
        built.run([_to_tvm(x_np, dev), out_tvm, m, n])
        tvm.testing.assert_allclose(out_tvm.numpy(), x_np.sum(axis=1), rtol=1e-5, atol=1e-5)


def test_m85_cache_key_records_structural_reduction_extent_policy():
    static_mod, static_meta = translate_ttir(
        _row_sum_ttir("f32"), grid=(1,), contract="row_reduction"
    )
    static_again_mod, static_again_meta = translate_ttir(
        _row_sum_ttir("f32"), grid=(1,), contract="row_reduction"
    )
    dynamic_mod, dynamic_meta = translate_ttir(
        _dynamic_row_sum_ttir("i32"), grid=(1,), contract="row_reduction"
    )
    fp16_mod, fp16_meta = translate_ttir(
        _row_sum_ttir("f16"), grid=(1,), contract="row_reduction"
    )
    norm_mod, norm_meta = translate_ttir(_row_sum_ttir("f32"), grid=(1,), contract="norm_row")

    assert static_mod.script() == static_again_mod.script()
    assert static_meta.cache_key == static_again_meta.cache_key
    assert static_meta.reduction_extent_kind == "constant"
    assert dynamic_meta.reduction_extent_kind == "runtime_param"
    assert static_meta.cache_key != dynamic_meta.cache_key
    assert static_meta.cache_key != fp16_meta.cache_key
    assert static_meta.cache_key != norm_meta.cache_key
    assert all(len(meta.cache_key) == 64 for meta in (dynamic_meta, fp16_meta, norm_meta))
    assert dynamic_mod.script()
    assert fp16_mod.script()
    assert norm_mod.script()


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    ("ttir", "mask"),
    [
        (
            _masked_softmax_ttir(rmask_extent=3),
            np.array([[True, True, True, False], [True, True, True, False]]),
        ),
        (
            _masked_softmax_ttir(causal=True),
            np.array([[True, False, False, False], [True, True, False, False]]),
        ),
    ],
)
def test_m85_masked_softmax_partial_and_causal_masks_build_run(ttir, mask):
    x_np = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], dtype="float32")
    expected = _softmax_expected(x_np, mask)
    irmod, meta = translate_ttir(ttir, grid=(1,), contract="masked_softmax_row")
    validate_masked_softmax_row_contract(irmod)
    assert "if " in irmod.script()

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.tensor(np.full((8,), -9.0, dtype="float32"), dev)
    built.run([_to_tvm(x_np, dev), out_tvm])
    tvm.testing.assert_allclose(
        out_tvm.numpy().reshape(2, 4), expected, rtol=1e-5, atol=1e-5
    )


@tvm.testing.requires_cuda
def test_m85_norm_row_rmsnorm_covers_rsqrt_and_epsilon():
    x_np = np.array([[1.0, 2.0, -3.0, 4.0], [-1.5, 0.5, 1.0, 2.5]], dtype="float32")
    eps = np.float32(1e-5)
    scale = 1.0 / np.sqrt(np.sum(x_np * x_np, axis=1, keepdims=True) / 4.0 + eps)
    expected = x_np * scale.astype("float32")
    irmod, meta = translate_ttir(_RMSNORM_ROW_TTIR, grid=(1,), contract="norm_row")
    validate_norm_row_contract(irmod)
    assert meta.reduction_extent_value == 4
    assert "T.rsqrt" in irmod.script()

    built = build_triton_tvm(irmod, meta)
    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((8,), "float32", dev)
    built.run([_to_tvm(x_np, dev), out_tvm])
    tvm.testing.assert_allclose(
        out_tvm.numpy().reshape(2, 4), expected, rtol=1e-4, atol=1e-5
    )


def test_m85_report_and_model_guards_keep_schema_v1_clean():
    records = [
        {
            "corpus": "m85",
            "case_name": "row_sum",
            "kernel_name": "row_sum_kernel",
            "contract": "row_reduction",
            "translate_status": make_report_status(ok=True, bucket="translated"),
            "op_counts": {"tt.reduce": 1, "tt.load": 1, "tt.store": 1},
            "types": ["tensor<2x4xf32>", "tensor<2xf32>"],
            "indexing_summary": {"index_kinds": {"flat": 2}},
        },
        {
            "corpus": "m85",
            "case_name": "deferred_grid",
            "kernel_name": "deferred_grid_kernel",
            "contract": "row_reduction",
            "translate_status": make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="grid",
                message="deferred grid blocker",
            ),
            "blocker_class": "grid",
            "op_counts": {"tt.load": 1},
            "types": ["tensor<2x4xbf16>"],
            "indexing_summary": {},
        },
    ]
    report = build_capability_report(
        records,
        purpose="m8.5 report guard",
        corpus="m85",
        generated_at="2026-05-24T00:00:00+00:00",
    )
    assert set(report["summary"]["status_buckets"]) <= _STABLE_REPORT_BUCKETS
    assert report["supported_kernel_report"]["legacy_contract_records"] == []
    assert report["supported_kernel_report"]["silent_fallback_records"] == []
    assert report["supported_kernel_report"]["ok"] is True

    model_report = build_model_corpus_report(
        records,
        [
            {
                "model_family": "fixture",
                "model_case": "fixture_case",
                "status": "completed",
                "kernel_count": 2,
                "translated_kernels": 1,
                "native_fallback_kernels": 1,
                "status_buckets": {"contract_error": 1, "translated": 1},
                "blocker_classes": {"grid": 1},
                "wrapper_paths": [],
                "full_tvm_runnable": False,
            }
        ],
        generated_at="2026-05-24T00:00:00+00:00",
    )
    assert set(model_report["summary"]["status_buckets"]) <= _STABLE_REPORT_BUCKETS
    assert model_report["supported_kernel_report"]["legacy_contract_records"] == []
    assert model_report["supported_kernel_report"]["silent_fallback_records"] == []
