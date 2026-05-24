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
"""M8 rank-2 reduction, norm, and softmax contract tests."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import (
    UnsupportedTTIROpError,
    build_triton_tvm,
    get_triton_tvm_contract,
    translate_ttir,
    validate_masked_softmax_row_contract,
    validate_row_reduction_contract,
)
from tvm.contrib.triton_tvm.ttir import TTIRReader


_M8_ROW_SUM_TTIR = """
module {
  tt.func @_m8_row_sum(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
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
    %acc = "tt.reduce"(%vx) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<2x4xf32>) -> tensor<2xf32>
    %acc_e = tt.expand_dims %acc {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_s, %xidx : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    tt.store %out_ptr, %acc_e, %xmask : tensor<2x1x!tt.ptr<f32>>
    tt.return
  }
}
"""


_M8_MASKED_SOFTMAX_TTIR = """
module {
  tt.func @_m8_masked_softmax(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
    %neg_inf = arith.constant dense<0xFF800000> : tensor<2x4xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
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
    %masked_vx = arith.select %mask, %vx, %neg_inf : tensor<2x4xi1>, tensor<2x4xf32>
    %maxv = "tt.reduce"(%masked_vx) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %gt = arith.cmpf ogt, %lhs, %rhs : f32
      %nan = arith.cmpf une, %lhs, %lhs : f32
      %take_lhs = arith.ori %gt, %nan : i1
      %mx = arith.select %take_lhs, %lhs, %rhs : i1, f32
      tt.reduce.return %mx : f32
    }) : (tensor<2x4xf32>) -> tensor<2xf32>
    %max_e = tt.expand_dims %maxv {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
    %max_b = tt.broadcast %max_e : tensor<2x1xf32> -> tensor<2x4xf32>
    %shift = arith.subf %vx, %max_b : tensor<2x4xf32>
    %exp = tt.extern_elementwise %shift {libname = "", libpath = "", pure = true, symbol = "__nv_expf"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %masked_exp = arith.select %mask, %exp, %zero : tensor<2x4xi1>, tensor<2x4xf32>
    %den = "tt.reduce"(%masked_exp) <{axis = 1 : i32}> ({
    ^bb0(%lhs2: f32, %rhs2: f32):
      %sum = arith.addf %lhs2, %rhs2 : f32
      tt.reduce.return %sum : f32
    }) : (tensor<2x4xf32>) -> tensor<2xf32>
    %den_e = tt.expand_dims %den {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
    %den_b = tt.broadcast %den_e : tensor<2x1xf32> -> tensor<2x4xf32>
    %soft = arith.divf %exp, %den_b : tensor<2x4xf32>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_s, %idx : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptr, %soft, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_m8_static_contract_metadata_and_reader_snapshot():
    row = get_triton_tvm_contract("row_reduction")
    masked = get_triton_tvm_contract("masked_softmax_row")
    assert row.execution_kind == "serial_m8_rank2_lane"
    assert row.axis_policy == "axis_1_only"
    assert masked.mask_policy == "rank2_masks_explicit_or_dominated"

    graph = TTIRReader().read(_M8_MASKED_SOFTMAX_TTIR)
    assert "tt.expand_dims" in [op.name for op in graph.ops]
    assert "tt.broadcast" in [op.name for op in graph.ops]
    assert graph.op_by_result()["neg_inf"].attrs["value"] == 0xFF800000
    assert graph.op_by_result()["maxv"].attrs["axis"] == 1


def test_m8_static_rank2_contract_translation_without_cuda_runtime():
    row_mod, row_meta = translate_ttir(_M8_ROW_SUM_TTIR, grid=(1,), contract="row_reduction")
    validate_row_reduction_contract(row_mod)
    assert row_meta.contract == "row_reduction"
    assert row_meta.execution_kind == "serial_m8_rank2_lane"
    assert row_meta.axis_policy == "axis_1_only"
    assert "for rk in range(4)" in row_mod.script()

    softmax_mod, softmax_meta = translate_ttir(
        _M8_MASKED_SOFTMAX_TTIR, grid=(1,), contract="masked_softmax_row"
    )
    validate_masked_softmax_row_contract(softmax_mod)
    script = softmax_mod.script()
    assert softmax_meta.contract == "masked_softmax_row"
    assert "T.exp" in script
    assert "T.max" in script


def test_m8_negative_axis0_reduction_rejected():
    bad = _M8_ROW_SUM_TTIR.replace("<{axis = 1 : i32}>", "<{axis = 0 : i32}>", 1)
    with pytest.raises(UnsupportedTTIROpError, match="axis=1"):
        translate_ttir(bad, grid=(1,), contract="row_reduction")


@tvm.testing.requires_cuda
def test_m8_row_sum_rank2_build_run():
    x_np = np.arange(8, dtype="float32").reshape(2, 4)
    expected = x_np.sum(axis=1)
    irmod, meta = translate_ttir(_M8_ROW_SUM_TTIR, grid=(1,), contract="row_reduction")
    built = build_triton_tvm(irmod, meta)

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_np.reshape(-1), dev)
    out_tvm = tvm.runtime.empty((2,), "float32", dev)
    built.run([x_tvm, out_tvm])
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_m8_masked_softmax_rank2_build_run():
    x_np = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], dtype="float32")
    shifted = x_np - x_np.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    expected = exp / exp.sum(axis=1, keepdims=True)
    irmod, meta = translate_ttir(
        _M8_MASKED_SOFTMAX_TTIR, grid=(1,), contract="masked_softmax_row"
    )
    built = build_triton_tvm(irmod, meta)

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_np.reshape(-1), dev)
    out_tvm = tvm.runtime.empty((8,), "float32", dev)
    built.run([x_tvm, out_tvm])
    tvm.testing.assert_allclose(
        out_tvm.numpy().reshape(2, 4), expected, rtol=1e-5, atol=1e-5
    )
