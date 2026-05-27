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
"""Static M2.5 hardening tests for TTIR reader and contract boundaries.

These tests intentionally use hand-written TTIR.  They exercise translation and contract
validation without invoking Triton compilation, TVM build, or a CUDA runtime device.
"""

import pytest

import tvm
import tvm.testing
import tvm.contrib.triton_tvm as triton_tvm_pkg
from tvm.contrib.triton_tvm import (
    TritonTVMContractError,
    UnsupportedTTIROpError,
    UnsupportedTargetPolicyError,
    ValidateTritonKernelTIR,
    MatmulSemantics,
    TargetMatmulDecision,
    TargetMatmulPolicy,
    get_triton_tvm_contract,
    translate_ttir,
    validate_cuda_minimal_contract,
    validate_cuda_pointwise_flat_contract,
    validate_matmul_minimal_contract,
    validate_norm_single_row_contract,
    validate_pointwise_flat_contract,
    validate_reduction_minimal_contract,
    validate_triton_tvm_contract,
)
from tvm.contrib.triton_tvm.indexing import summarize_ttir_indexing
from tvm.contrib.triton_tvm.inductor import audit_inductor_ttir
from tvm.contrib.triton_tvm.matmul import (
    EXTERN_GEMM_PACKED_FUNC,
    EXTERN_GEMM_PROVIDER_ABI_VERSION,
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    EXTERN_GEMM_RUNTIME_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
    EXTERN_GEMM_RUNTIME_REPLACEMENT,
    EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
    EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    EXTERN_GEMM_SYMBOL,
    MATMUL_PERF_ENVELOPE_ID,
    MATMUL_SCHEDULE_REGISTRY_VERSION,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
    SIMT_TIR_MATMUL_SCHEDULE_ID,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_SCHEDULE_ID,
    build_extern_addmm_bias_tirx_source,
    build_extern_gemm_tirx_source,
    build_matmul_tune_key_payload,
    extract_matmul_semantics_from_wrapper_extern_addmm,
    extract_matmul_semantics_from_ttir,
    extract_matmul_semantics_from_wrapper_extern,
    matmul_schedule_candidate_ids,
    matmul_schedule_candidate_records,
    register_python_torch_extern_gemm,
)
from tvm.contrib.triton_tvm.ttir import TTIRReader


_STATIC_DUAL_STORE_TTIR = """
module {
  tt.func @_sd(%x:!tt.ptr<f32>,%y:!tt.ptr<f32>,%o0:!tt.ptr<f32>,%o1:!tt.ptr<f32>,%n:i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %idx64 = arith.extsi %idx : tensor<64xi32> to tensor<64xi64>
    %n_splat = tt.splat %n : i64 -> tensor<64xi64>
    %mask = arith.cmpi slt, %idx64, %n_splat : tensor<64xi64>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask, %cst : tensor<64x!tt.ptr<f32>>
    %y_splat = tt.splat %y : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %y_ptr = tt.addptr %y_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vy = tt.load %y_ptr, %mask, %cst : tensor<64x!tt.ptr<f32>>
    %o0_splat = tt.splat %o0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %o0_ptr = tt.addptr %o0_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %sum = arith.addf %vx, %vy : tensor<64xf32>
    tt.store %o0_ptr, %sum, %mask : tensor<64x!tt.ptr<f32>>
    %o1_splat = tt.splat %o1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %o1_ptr = tt.addptr %o1_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %diff = arith.subf %vx, %vy : tensor<64xf32>
    tt.store %o1_ptr, %diff, %mask : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


_STATIC_SCALAR_BEFORE_EXTENT_TTIR = """
module {
  tt.func @_scalar_before_extent(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>,%alpha:f32,%n:i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %idx64 = arith.extsi %idx : tensor<64xi32> to tensor<64xi64>
    %n_splat = tt.splat %n : i64 -> tensor<64xi64>
    %mask = arith.cmpi slt, %idx64, %n_splat : tensor<64xi64>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask, %cst : tensor<64x!tt.ptr<f32>>
    %alpha_splat = tt.splat %alpha : f32 -> tensor<64xf32>
    %value = arith.addf %vx, %alpha_splat : tensor<64xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_ptr, %value, %mask : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


_STATIC_TWO_SCALARS_BEFORE_EXTENT_TTIR = _STATIC_SCALAR_BEFORE_EXTENT_TTIR.replace(
    "%x:!tt.ptr<f32>,%out:!tt.ptr<f32>,%alpha:f32,%n:i64",
    "%x:!tt.ptr<f32>,%out:!tt.ptr<f32>,%alpha:f32,%beta:f32,%n:i64",
).replace(
    "%alpha_splat = tt.splat %alpha : f32 -> tensor<64xf32>\n"
    "    %value = arith.addf %vx, %alpha_splat : tensor<64xf32>",
    "%alpha_splat = tt.splat %alpha : f32 -> tensor<64xf32>\n"
    "    %beta_splat = tt.splat %beta : f32 -> tensor<64xf32>\n"
    "    %shift = arith.addf %alpha_splat, %beta_splat : tensor<64xf32>\n"
    "    %value = arith.addf %vx, %shift : tensor<64xf32>",
)


_STATIC_M35_INDEXED_TTIR = """
module {
  @M35_INDEXED_SIG@
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %x0 = arith.constant dense<16> : tensor<64xi32>
    %xmask = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %xmask : tensor<64xi32>
    %yidx = arith.remsi %idx, %x0 : tensor<64xi32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask : tensor<64x!tt.ptr<f32>>
    %y_splat = tt.splat %y : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %y_ptr = tt.addptr %y_splat, %yidx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vy = tt.load %y_ptr, %mask evictionPolicy = evict_last : tensor<64x!tt.ptr<f32>>
    %is_pos = arith.cmpf ogt, %vx, %cst : tensor<64xf32>
    %relu = arith.select %is_pos, %vx, %cst : tensor<64xi1>, tensor<64xf32>
    %sum = arith.addf %relu, %vy : tensor<64xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_ptr, %sum, %mask : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
""".replace(
    "@M35_INDEXED_SIG@",
    "tt.func public @_m35_indexed("
    "%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
    "%y: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
    "%out: !tt.ptr<f32> {tt.divisibility = 16 : i32}, "
    "%xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {",
)


_STATIC_M35_BOOL_BITCAST_TTIR = """
module {
  tt.func @_m35_bool(%x:!tt.ptr<f32>,%y:!tt.ptr<f32>,%out:!tt.ptr<i1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %xmask = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %xmask : tensor<64xi32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask : tensor<64x!tt.ptr<f32>>
    %y_splat = tt.splat %y : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %y_ptr = tt.addptr %y_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vy = tt.load %y_ptr, %mask : tensor<64x!tt.ptr<f32>>
    %x_gt = arith.cmpf ogt, %vx, %cst : tensor<64xf32>
    %y_gt = arith.cmpf ogt, %vy, %cst : tensor<64xf32>
    %both = arith.andi %x_gt, %y_gt : tensor<64xi1>
    %out_splat = tt.splat %out : !tt.ptr<i1> -> tensor<64x!tt.ptr<i1>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<i1>>, tensor<64xi32>
    %out_i8 = tt.bitcast %out_ptr : tensor<64x!tt.ptr<i1>> -> tensor<64x!tt.ptr<i8>>
    %both_i8 = arith.extui %both : tensor<64xi1> to tensor<64xi8>
    tt.store %out_i8, %both_i8, %mask : tensor<64x!tt.ptr<i8>>
    tt.return
  }
}
"""


_STATIC_M35_BOOL_BITCAST_STORE_ONLY_TTIR = """
module {
  tt.func @_m35_bool_store_only(%out:!tt.ptr<i1>) {
    %xmask = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %xmask : tensor<64xi32>
    %out_splat = tt.splat %out : !tt.ptr<i1> -> tensor<64x!tt.ptr<i1>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<i1>>, tensor<64xi32>
    %out_i8 = tt.bitcast %out_ptr : tensor<64x!tt.ptr<i1>> -> tensor<64x!tt.ptr<i8>>
    %mask_i8 = arith.extui %mask : tensor<64xi1> to tensor<64xi8>
    tt.store %out_i8, %mask_i8, %mask : tensor<64x!tt.ptr<i8>>
    tt.return
  }
}
"""


_STATIC_M7_COMPOSED_INDEX_TTIR = """
module {
  tt.func @_m7_composed_index(%x:!tt.ptr<f32>,%y:!tt.ptr<f32>,%out:!tt.ptr<f32>,%n:i64) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c16 = arith.constant dense<16> : tensor<64xi32>
    %c32 = arith.constant dense<32> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %idx64 = arith.extsi %idx : tensor<64xi32> to tensor<64xi64>
    %n_splat = tt.splat %n : i64 -> tensor<64xi64>
    %mask = arith.cmpi slt, %idx64, %n_splat : tensor<64xi64>
    %row = arith.divsi %idx, %c16 : tensor<64xi32>
    %col = arith.remsi %idx, %c16 : tensor<64xi32>
    %row_stride = arith.muli %row, %c32 : tensor<64xi32>
    %yidx = arith.addi %row_stride, %col : tensor<64xi32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<64x!tt.ptr<f32>>
    %y_splat = tt.splat %y : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %y_ptr = tt.addptr %y_splat, %yidx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vy = tt.load %y_ptr, %mask, %zero : tensor<64x!tt.ptr<f32>>
    %sum = arith.addf %vx, %vy : tensor<64xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_ptr, %sum, %mask : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


_STATIC_M7_CAST_ERF_TTIR = """
module {
  tt.func @_m7_cast_erf(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>,%n:i64) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %idx64 = arith.extsi %idx : tensor<64xi32> to tensor<64xi64>
    %n_splat = tt.splat %n : i64 -> tensor<64xi64>
    %mask = arith.cmpi slt, %idx64, %n_splat : tensor<64xi64>
    %idxf = arith.sitofp %idx : tensor<64xi32> to tensor<64xf32>
    %idxi = arith.fptosi %idxf : tensor<64xf32> to tensor<64xi32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idxi : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<64x!tt.ptr<f32>>
    %erf = tt.extern_elementwise %vx {libname = "", libpath = "", pure = true, symbol = "__nv_erff"} : (tensor<64xf32>) -> tensor<64xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_ptr, %erf, %mask : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


_STATIC_REDUCTION_ROWSUM_TTIR = """
module {
  tt.func @_row_sum(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>,%n:i64) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %pid = tt.get_program_id x : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %range64 = arith.extsi %range : tensor<64xi32> to tensor<64xi64>
    %n_splat = tt.splat %n : i64 -> tensor<64xi64>
    %mask = arith.cmpi slt, %range64, %n_splat : tensor<64xi64>
    %row = arith.extsi %pid : i32 to i64
    %row_base = arith.muli %row, %n : i64
    %x_row = tt.addptr %x, %row_base : !tt.ptr<f32>, i64
    %x_splat = tt.splat %x_row : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<64x!tt.ptr<f32>>
    %acc = "tt.reduce"(%vx) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<64xf32>) -> f32
    %out_ptr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %out_ptr, %acc : !tt.ptr<f32>
    tt.return
  }
}
"""


_STATIC_PRE_M7_DOT_TTIR = """
module {
  tt.func @_pre_m7_dot(%a:!tt.ptr<f32>,%b:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %a_splat = tt.splat %a : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %b_splat = tt.splat %b : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %b_ptr = tt.addptr %b_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %va = tt.load %a_ptr : tensor<64x!tt.ptr<f32>>
    %vb = tt.load %b_ptr : tensor<64x!tt.ptr<f32>>
    %dot = tt.dot %va, %vb : tensor<64xf32>, tensor<64xf32>
    %out_ptr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %out_ptr, %dot : !tt.ptr<f32>
    tt.return
  }
}
"""


def _static_m9_dot_ttir(dtype: str = "f16") -> str:
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


def _static_m97_tiled_dot_ttir(dtype: str = "f16") -> str:
    return (
        _static_m9_dot_ttir(dtype)
        .replace("tensor<16x4xi32>", "tensor<16x8xi32>")
        .replace("tensor<8x4xi32>", "tensor<8x8xi32>")
        .replace("tensor<16x4x!tt.ptr", "tensor<16x8x!tt.ptr")
        .replace("tensor<8x4x!tt.ptr", "tensor<8x8x!tt.ptr")
        .replace("tensor<16x4x" + dtype + ">", "tensor<16x8x" + dtype + ">")
        .replace("tensor<8x4xf32>", "tensor<8x8xf32>")
    )


def _static_m9p_dot_ttir(m: int, n: int, k: int, dtype: str = "f16") -> str:
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


_STATIC_PRE_M7_ATOMIC_GRID_TTIR = """
module {
  tt.func @_pre_m7_atomic_grid(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %pid_y = tt.get_program_id y : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %mask = arith.cmpf ogt, %zero, %zero : tensor<64xf32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask, %zero : tensor<64x!tt.ptr<f32>>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.atomic_add %out_ptr, %vx, %mask : tensor<64x!tt.ptr<f32>>, tensor<64xf32>
    tt.return
  }
}
"""


_STATIC_PRE_M7_ATTENTION_ADJACENT_TTIR = """
module {
  tt.func @_pre_m7_attention_adjacent(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr : tensor<64x!tt.ptr<f32>>
    %trans = tt.trans %vx : tensor<64xf32>
    %soft = tt.softmax %trans : tensor<64xf32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_ptr, %soft : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_static_m25_reader_op_graph_snapshot():
    graph = TTIRReader().read(_STATIC_DUAL_STORE_TTIR)

    assert graph.function_name == "_sd"
    params = [
        (param.name, param.type.raw, param.type.dtype, param.type.is_pointer)
        for param in graph.params
    ]
    assert params == [
        ("x", "!tt.ptr<f32>", "float32", True),
        ("y", "!tt.ptr<f32>", "float32", True),
        ("o0", "!tt.ptr<f32>", "float32", True),
        ("o1", "!tt.ptr<f32>", "float32", True),
        ("n", "i64", "int64", False),
    ]

    assert _op_snapshot(graph) == [
        ("arith.constant", ("cst",), ()),
        ("arith.constant", ("c64_i32",), ()),
        ("tt.get_program_id", ("pid",), ()),
        ("arith.muli", ("offsets",), ("pid", "c64_i32")),
        ("tt.make_range", ("range",), ()),
        ("tt.splat", ("s_offsets",), ("offsets",)),
        ("arith.addi", ("idx",), ("s_offsets", "range")),
        ("arith.extsi", ("idx64",), ("idx",)),
        ("tt.splat", ("n_splat",), ("n",)),
        ("arith.cmpi", ("mask",), ("idx64", "n_splat")),
        ("tt.splat", ("x_splat",), ("x",)),
        ("tt.addptr", ("x_ptr",), ("x_splat", "idx")),
        ("tt.load", ("vx",), ("x_ptr", "mask", "cst")),
        ("tt.splat", ("y_splat",), ("y",)),
        ("tt.addptr", ("y_ptr",), ("y_splat", "idx")),
        ("tt.load", ("vy",), ("y_ptr", "mask", "cst")),
        ("tt.splat", ("o0_splat",), ("o0",)),
        ("tt.addptr", ("o0_ptr",), ("o0_splat", "idx")),
        ("arith.addf", ("sum",), ("vx", "vy")),
        ("tt.store", (), ("o0_ptr", "sum", "mask")),
        ("tt.splat", ("o1_splat",), ("o1",)),
        ("tt.addptr", ("o1_ptr",), ("o1_splat", "idx")),
        ("arith.subf", ("diff",), ("vx", "vy")),
        ("tt.store", (), ("o1_ptr", "diff", "mask")),
        ("tt.return", (), ()),
    ]
    assert _type_snapshot(graph, "mask") == ("tensor<64xi1>", "bool", (64,), False)
    assert _type_snapshot(graph, "x_ptr") == (
        "tensor<64x!tt.ptr<f32>>",
        "float32",
        (64,),
        True,
    )
    assert _type_snapshot(graph, "vx") == ("tensor<64xf32>", "float32", (64,), False)

    attr_graph = TTIRReader().read(
        _STATIC_DUAL_STORE_TTIR.replace(
            "tt.load %x_ptr, %mask, %cst :",
            'tt.load %x_ptr, %mask, %cst {cache_modifier = "", '
            'eviction_policy = "evict_last"} :',
            1,
        )
    )
    load = attr_graph.op_by_result()["vx"]
    assert load.attrs["raw_attrs"] == '{cache_modifier = "", eviction_policy = "evict_last"}'
    assert load.attrs["unknown_attrs"] == {
        "cache_modifier": '""',
        "eviction_policy": '"evict_last"',
    }


def test_static_pre_m5_public_pass_surface():
    assert hasattr(triton_tvm_pkg, "ValidateTritonKernelTIR")
    assert not hasattr(triton_tvm_pkg, "NormalizeTritonKernelTIR")


def test_static_pre_m5_public_api_freeze():
    assert set(triton_tvm_pkg.__all__) == {
        "MATMUL_PERF_ENVELOPE_ID",
        "MATMUL_SCHEDULE_REGISTRY_VERSION",
        "EXTERN_GEMM_PROVIDER_NATIVE_TVM",
        "TTIRArtifact",
        "AttentionSemantics",
        "MatmulSemantics",
        "MatmulScheduleCandidate",
        "SIMT_TIR_MATMUL_SCHEDULE_ID",
        "TENSORCORE_TIR_MATMUL_SCHEDULE_ID",
        "ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED",
        "ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL",
        "ATTENTION_PROVIDER_NATIVE_DECOMPOSED",
        "ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED",
        "M11_GRID_FAMILIES",
        "M11_GRID2D_ARTIFACT_READY",
        "M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM",
        "M11_GRID2D_PROVIDER_NATIVE_TVM",
        "M11_GRID2D_READINESS_VERSION",
        "M11_GRID2D_RUNTIME_READY",
        "M11_GRID_FAMILY_BN_SILU_FUSION",
        "M11_GRID_FAMILY_CONCAT_SPLIT",
        "M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE",
        "M11_GRID_FAMILY_OTHER_MULTIDIM_POINTWISE",
        "M11_GRID_FAMILY_POOL_OR_SOFTMAX",
        "M11_GRID_FAMILY_YOLO_DECODE_POSTPROCESS",
        "M11_GRID_REPORT_FIELDS",
        "M11_GRID_STATUS_CLASSIFIED",
        "M11_GRID_TAXONOMY_VERSION",
        "TargetAttentionDecision",
        "TargetAttentionPolicy",
        "TargetMatmulDecision",
        "TargetMatmulPolicy",
        "TargetVisionDecision",
        "TargetVisionPolicy",
        "TritonTVMArtifact",
        "TritonTVMContract",
        "TritonTVMContractError",
        "TritonTVMError",
        "TritonTVMMeta",
        "UnsupportedContractError",
        "UnsupportedStreamError",
        "UnsupportedTTIROpError",
        "UnsupportedTargetPolicyError",
        "ValidateTritonKernelTIR",
        "VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC",
        "VISION_CONTRACT_CONV2D_NCHW_STATIC",
        "VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC",
        "VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC",
        "VISION_CONTRACT_POINTWISE_GRID2D_STATIC",
        "VISION_EXTERN_PACKED_FUNC",
        "VISION_M11_4_INTERFACE_STATUS",
        "VISION_M11_5_CORPUS_DIFF_BASELINE_ID",
        "VISION_M11_5_HARDENING_STATUS",
        "VISION_M11_5_RUNTIME_SCOPE_STATUS",
        "VISION_M11_6_CORPUS_DIFF_BASELINE_ID",
        "VISION_M11_6_INTERFACE_STATUS",
        "VISION_M11_6_RUNTIME_SCOPE_STATUS",
        "VISION_M11_7_CORPUS_DIFF_BASELINE_ID",
        "VISION_M11_7_INTERFACE_STATUS",
        "VISION_M11_7_RUNTIME_SCOPE_STATUS",
        "VISION_PROVIDER_DEVICE_TORCH_CUDA",
        "VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED",
        "VisionConv2DSemantics",
        "VisionGrid2DPointwiseSemantics",
        "build_attention_sdpa_tirx_source",
        "build_extern_addmm_bias_tirx_source",
        "build_extern_conv2d_tirx_source",
        "build_native_decomposed_attention_tirx_source",
        "build_native_grid2d_pointwise_tirx_source",
        "build_native_wrapper_matmul_tirx_source",
        "build_matmul_tune_key_payload",
        "build_triton_tvm",
        "classify_m11_captured_grid_record",
        "extract_attention_semantics_from_wrapper_sdpa",
        "extract_matmul_semantics_from_wrapper_extern_addmm",
        "extract_vision_conv2d_semantics_from_wrapper_extern",
        "get_triton_tvm_contract",
        "lower_to_ttir",
        "matmul_schedule_candidate_ids",
        "matmul_schedule_candidate_records",
        "normalize_triton_tvm_contract",
        "is_m11_6_static_conv2d_runtime_scope",
        "is_m11_7_device_conv2d_runtime_scope",
        "is_m12_vit_native_wrapper_matmul_scope",
        "register_python_torch_attention_sdpa",
        "register_device_torch_cuda_extern_conv2d",
        "register_python_torch_extern_addmm_bias",
        "register_python_torch_extern_conv2d",
        "register_python_torch_extern_gemm",
        "translate_ttir",
        "validate_attention_llama_causal_prefill_contract",
        "validate_attention_vit_full_contract",
        "validate_matmul_minimal_contract",
        "validate_masked_softmax_row_contract",
        "validate_norm_single_row_contract",
        "validate_norm_row_contract",
        "validate_pointwise_grid2d_static_contract",
        "validate_pointwise_flat_contract",
        "validate_pointwise_minimal_contract",
        "validate_reduction_minimal_contract",
        "validate_row_reduction_contract",
        "validate_softmax_row_contract",
        "validate_triton_tvm_contract",
        "validate_vision_conv2d_contract",
    }
    assert "TTIRReader" not in triton_tvm_pkg.__all__
    assert "validate_cuda_minimal_contract" not in triton_tvm_pkg.__all__
    assert "validate_cuda_pointwise_flat_contract" not in triton_tvm_pkg.__all__
    assert hasattr(triton_tvm_pkg, "validate_cuda_minimal_contract")
    assert hasattr(triton_tvm_pkg, "validate_cuda_pointwise_flat_contract")


def test_static_pre_m5_ttir_parse_boundary():
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    repo_root = Path(__file__).resolve().parents[3]
    translator_source = (
        repo_root / "python" / "tvm" / "contrib" / "triton_tvm" / "translator.py"
    ).read_text(encoding="utf-8")
    assert "TTIRReader" not in translator_source
    assert "normalize_ttir_input" in translator_source


def test_static_pre_m7_builder_boundary_keeps_single_tvmscript_source_builder():
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    repo_root = Path(__file__).resolve().parents[3]
    translator_source = (
        repo_root / "python" / "tvm" / "contrib" / "triton_tvm" / "translator.py"
    ).read_text(encoding="utf-8")
    assert translator_source.count("tvm.script.from_source(") == 1
    assert "source = builder.build_source()" in translator_source
    assert "normalize_ttir_input(ttir_or_graph)" in translator_source
    assert "TTIRReader" not in translator_source


def test_static_m4_reader_reduce_region_snapshot():
    graph = TTIRReader().read(_STATIC_REDUCTION_ROWSUM_TTIR)
    ops = [op.name for op in graph.ops]

    assert "tt.reduce" in ops
    assert "tt.store" in ops[ops.index("tt.reduce") :]
    assert ops.count("arith.addf") == 0

    reduce_op = graph.op_by_result()["acc"]
    assert reduce_op.attrs["axis"] == 0
    assert reduce_op.result_types[0].dtype == "float32"
    assert [[op.name for op in region] for region in reduce_op.regions] == [
        ["arith.addf", "tt.reduce.return"]
    ]


def test_static_pre_m7_reader_snapshot_classes_without_translation():
    snapshots = {
        "pointwise": TTIRReader().read(_STATIC_DUAL_STORE_TTIR),
        "broadcast_view_index": TTIRReader().read(_STATIC_M35_INDEXED_TTIR),
        "reduction": TTIRReader().read(_STATIC_REDUCTION_ROWSUM_TTIR),
        "matmul_dot": TTIRReader().read(_STATIC_PRE_M7_DOT_TTIR),
        "atomic_grid": TTIRReader().read(_STATIC_PRE_M7_ATOMIC_GRID_TTIR),
        "attention_adjacent": TTIRReader().read(_STATIC_PRE_M7_ATTENTION_ADJACENT_TTIR),
    }

    assert "tt.store" in [op.name for op in snapshots["pointwise"].ops]
    assert "arith.remsi" in [op.name for op in snapshots["broadcast_view_index"].ops]
    assert "tt.reduce" in [op.name for op in snapshots["reduction"].ops]
    assert "tt.dot" in [op.name for op in snapshots["matmul_dot"].ops]
    atomic_grid_ops = snapshots["atomic_grid"].op_by_result()
    assert atomic_grid_ops["pid_y"].attrs["axis"] == "y"
    assert "tt.atomic_add" in [op.name for op in snapshots["atomic_grid"].ops]
    assert {"tt.trans", "tt.softmax"} <= {
        op.name for op in snapshots["attention_adjacent"].ops
    }


def test_static_m9_reader_extracts_rank2_dot_semantics():
    graph = TTIRReader().read(_static_m9_dot_ttir("f16"))
    dot = graph.op_by_result()["dot"]

    assert dot.attrs["inputPrecision"] == "tf32"
    assert [(ty.raw, ty.dtype, ty.shape) for ty in dot.result_types] == [
        ("tensor<8x4xf32>", "float32", (8, 4))
    ]

    semantics = extract_matmul_semantics_from_ttir(graph)
    assert isinstance(semantics, MatmulSemantics)
    assert semantics.source_kind == "tt_dot"
    assert semantics.kernel_name == "_m9_dot"
    assert (semantics.m, semantics.n, semantics.k) == (8, 4, 16)
    assert (semantics.a_param, semantics.b_param, semantics.c_param) == ("a", "b", "out")
    assert semantics.a_dtype == "float16"
    assert semantics.b_dtype == "float16"
    assert semantics.accumulator_dtype == "float32"
    assert semantics.output_dtype == "float32"
    assert semantics.a_stride == (16, 1)
    assert semantics.b_stride == (4, 1)
    assert semantics.c_stride == (4, 1)
    assert semantics.mask_kind == "none"
    assert semantics.epilogue_kind == "none"
    assert semantics.implementation_kind == "unresolved"

    bf16_semantics = extract_matmul_semantics_from_ttir(
        TTIRReader().read(_static_m9_dot_ttir("bf16"))
    )
    assert bf16_semantics.a_dtype == "bfloat16"
    assert bf16_semantics.b_dtype == "bfloat16"
    assert bf16_semantics.output_dtype == "float32"


def test_static_m9_target_matmul_policy_selects_native_schedule():
    semantics = extract_matmul_semantics_from_ttir(
        TTIRReader().read(_static_m9_dot_ttir("f16"))
    )
    decision = TargetMatmulPolicy().decide(
        semantics,
        matmul_contract_ok=True,
    )

    assert isinstance(decision, TargetMatmulDecision)
    assert decision.matmul_contract_ok is True
    assert decision.implementation_kind == "native_tir_schedule"
    assert decision.schedule_id == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert decision.extern_symbol == ""
    assert decision.unsupported_matmul_reason == ""
    assert decision.cache_payload()["implementation_kind"] == "native_tir_schedule"


def test_static_m9_wrapper_extern_gemm_semantics_from_real_source_lines():
    transposed_weight_source = (
        "extern_kernels.mm(reinterpret_tensor(buf1, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg4_1, (64, 64), (1, 64), 0), out=buf2)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern(
        transposed_weight_source,
        case_name="llama_tiny",
    )

    assert semantics.source_kind == "wrapper_extern_gemm"
    assert semantics.source_name == EXTERN_GEMM_SYMBOL
    assert semantics.kernel_name == "llama_tiny"
    assert (semantics.m, semantics.n, semantics.k) == (16, 64, 64)
    assert (semantics.a_param, semantics.b_param, semantics.c_param) == (
        "buf1",
        "arg4_1",
        "buf2",
    )
    assert semantics.a_dtype == "float32"
    assert semantics.b_dtype == "float32"
    assert semantics.accumulator_dtype == "float32"
    assert semantics.output_dtype == "float32"
    assert semantics.a_stride == (64, 1)
    assert semantics.b_stride == (1, 64)
    assert semantics.c_stride == (64, 1)
    assert semantics.a_layout == "row_major"
    assert semantics.b_layout == "transposed_weight_view"
    assert semantics.c_layout == "row_major"
    assert semantics.epilogue_kind == "none"

    row_major_source = (
        "extern_kernels.mm(reinterpret_tensor(buf15, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg9_1, (64, 128), (128, 1), 0), out=buf16)"
    )
    row_major = extract_matmul_semantics_from_wrapper_extern(
        row_major_source,
        case_name="vit_tiny",
    )
    assert (row_major.m, row_major.n, row_major.k) == (16, 128, 64)
    assert row_major.b_stride == (128, 1)
    assert row_major.b_layout == "row_major"


def test_static_m9_wrapper_extern_gemm_policy_and_artifact_without_runtime():
    semantics = extract_matmul_semantics_from_wrapper_extern(
        "extern_kernels.mm(reinterpret_tensor(buf1, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg4_1, (64, 64), (1, 64), 0), out=buf2)",
        case_name="llama_tiny",
    )
    decision = TargetMatmulPolicy().decide(
        semantics,
        matmul_contract_ok=True,
    )

    assert decision.matmul_contract_ok is True
    assert decision.implementation_kind == "extern_gemm"
    assert decision.schedule_id == ""
    assert decision.extern_symbol == EXTERN_GEMM_SYMBOL
    assert decision.extern_packed_func == EXTERN_GEMM_PACKED_FUNC
    assert decision.extern_runtime_kind == EXTERN_GEMM_RUNTIME_KIND
    assert decision.extern_runtime_replacement == EXTERN_GEMM_RUNTIME_REPLACEMENT
    assert decision.extern_runtime_replacement_available is False
    assert (
        decision.extern_runtime_replacement_reason
        == EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON
    )
    assert decision.unsupported_matmul_reason == ""

    source = build_extern_gemm_tirx_source(semantics, decision)
    irmod = tvm.script.from_source(source)
    validate_matmul_minimal_contract(irmod)
    attrs = next(iter(irmod.functions.values())).attrs
    script = irmod.script()

    assert str(attrs["triton_tvm.contract"]) == "matmul_minimal"
    assert str(attrs["triton_tvm.matmul_source_kind"]) == "wrapper_extern_gemm"
    assert str(attrs["triton_tvm.implementation_kind"]) == "extern_gemm"
    assert str(attrs["triton_tvm.extern_symbol"]) == EXTERN_GEMM_SYMBOL
    assert str(attrs["triton_tvm.extern_packed_func"]) == EXTERN_GEMM_PACKED_FUNC
    assert str(attrs["triton_tvm.extern_runtime_kind"]) == EXTERN_GEMM_RUNTIME_KIND
    assert (
        str(attrs["triton_tvm.extern_runtime_replacement"])
        == EXTERN_GEMM_RUNTIME_REPLACEMENT
    )
    assert (
        str(attrs["triton_tvm.extern_runtime_replacement_reason"])
        == EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON
    )
    assert str(attrs["triton_tvm.a_dtype"]) == "float32"
    assert str(attrs["triton_tvm.b_dtype"]) == "float32"
    assert str(attrs["triton_tvm.output_dtype"]) == "float32"
    assert str(attrs["triton_tvm.accumulator_dtype"]) == "float32"
    assert str(attrs["triton_tvm.input_precision"]) == "extern_fp32"
    assert str(attrs["triton_tvm.bounds_policy"]) == "exact"
    assert str(attrs["triton_tvm.mask_kind"]) == "none"
    assert str(attrs["triton_tvm.epilogue_kind"]) == "none"
    assert int(attrs["triton_tvm.matmul_m"]) == 16
    assert int(attrs["triton_tvm.matmul_n"]) == 64
    assert int(attrs["triton_tvm.matmul_k"]) == 64
    assert str(attrs["triton_tvm.b_layout"]) == "transposed_weight_view"
    assert EXTERN_GEMM_PACKED_FUNC in script
    assert "call_packed" in script
    assert "call_extern" not in script
    assert 'with T.sblock("matmul")' not in script
    assert 'thread="blockIdx.x"' not in script
    assert 'thread="threadIdx.x"' not in script


def test_static_m96_wrapper_extern_gemm_runtime_provider_metadata():
    semantics = extract_matmul_semantics_from_wrapper_extern(
        "extern_kernels.mm(reinterpret_tensor(buf1, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg4_1, (64, 64), (1, 64), 0), out=buf2)",
        case_name="llama_tiny",
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(
        semantics,
        matmul_contract_ok=True,
    )

    assert decision.extern_runtime_kind == EXTERN_GEMM_RUNTIME_PROVIDER_KIND
    assert decision.extern_runtime_replacement == EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    assert decision.extern_runtime_replacement_available is True
    assert decision.extern_runtime_replacement_reason == EXTERN_GEMM_RUNTIME_PROVIDER_REASON
    assert decision.extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert decision.extern_gemm_provider_kind == EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    assert decision.extern_gemm_provider_abi_version == EXTERN_GEMM_PROVIDER_ABI_VERSION
    assert decision.extern_gemm_runtime_claim == EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY
    assert decision.extern_gemm_performance_claim is False
    assert decision.extern_gemm_uses_host_staging is True

    source = build_extern_gemm_tirx_source(semantics, decision)
    irmod = tvm.script.from_source(source)
    validate_matmul_minimal_contract(irmod)
    attrs = next(iter(irmod.functions.values())).attrs
    script = irmod.script()

    assert str(attrs["triton_tvm.extern_gemm_runtime_status"]) == (
        EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
    )
    assert str(attrs["triton_tvm.extern_gemm_provider_kind"]) == (
        EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    )
    assert int(attrs["triton_tvm.extern_gemm_provider_abi_version"]) == 1
    assert str(attrs["triton_tvm.extern_gemm_runtime_claim"]) == "correctness_only"
    assert bool(attrs["triton_tvm.extern_gemm_performance_claim"]) is False
    assert bool(attrs["triton_tvm.extern_gemm_uses_host_staging"]) is True
    assert bool(attrs["triton_tvm.transposed_b"]) is True
    assert str(attrs["triton_tvm.b_stride"]) == "1, 64"
    assert str(attrs["triton_tvm.b_storage_shape"]) == "64, 64"
    assert str(attrs["triton_tvm.b_storage_stride"]) == "64, 1"
    assert "T.bool(True)" in script


def test_static_m96_extern_gemm_rejects_transposed_storage_mismatch():
    semantics = extract_matmul_semantics_from_wrapper_extern(
        "extern_kernels.mm(reinterpret_tensor(buf1, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg4_1, (64, 128), (1, 64), 0), out=buf2)",
        case_name="llama_tiny",
    )
    decision = TargetMatmulPolicy().decide(
        semantics,
        matmul_contract_ok=True,
    )
    source = build_extern_gemm_tirx_source(semantics, decision)
    validate_matmul_minimal_contract(tvm.script.from_source(source))

    bad_storage_shape = source.replace(
        '"triton_tvm.b_storage_shape": "128, 64"',
        '"triton_tvm.b_storage_shape": "64, 128"',
    )
    with pytest.raises(TritonTVMContractError, match="b_storage_shape"):
        validate_matmul_minimal_contract(tvm.script.from_source(bad_storage_shape))

    bad_transposed_flag = source.replace(
        '"triton_tvm.transposed_b": True',
        '"triton_tvm.transposed_b": False',
    )
    with pytest.raises(TritonTVMContractError, match="transposed_b"):
        validate_matmul_minimal_contract(tvm.script.from_source(bad_transposed_flag))


def test_static_m95_matmul_contract_rejects_erased_semantic_surface():
    irmod, _ = translate_ttir(
        _static_m9_dot_ttir("f16"),
        grid=(1,),
        contract="matmul_minimal",
    )
    source = irmod.script()

    bad_func_dim = source.replace(
        '"triton_tvm.matmul_k": 16',
        '"triton_tvm.matmul_k": 15',
        1,
    )
    with pytest.raises(TritonTVMContractError, match="dims must match"):
        validate_matmul_minimal_contract(tvm.script.from_source(bad_func_dim))

    bad_axis_extent = source.replace(
        "vk = T.axis.reduce(16, k)",
        "vk = T.axis.reduce(15, k)",
    )
    with pytest.raises(TritonTVMContractError, match="k axis extent"):
        validate_matmul_minimal_contract(tvm.script.from_source(bad_axis_extent))

    bad_output_shape = source.replace(
        "out: T.Buffer((8, 4), \"float32\")",
        "out: T.Buffer((8, 5), \"float32\")",
    )
    with pytest.raises(TritonTVMContractError, match="C write buffer shape"):
        validate_matmul_minimal_contract(tvm.script.from_source(bad_output_shape))


def test_static_m95_extern_gemm_artifact_requires_runtime_gate_metadata():
    semantics = extract_matmul_semantics_from_wrapper_extern(
        "extern_kernels.mm(reinterpret_tensor(buf1, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg4_1, (64, 64), (1, 64), 0), out=buf2)",
        case_name="llama_tiny",
    )
    decision = TargetMatmulPolicy().decide(
        semantics,
        matmul_contract_ok=True,
    )
    source = build_extern_gemm_tirx_source(semantics, decision)

    missing_packed_func = source.replace(
        f'"triton_tvm.extern_packed_func": "{EXTERN_GEMM_PACKED_FUNC}", ',
        "",
    )
    with pytest.raises(TritonTVMContractError, match="extern_packed_func"):
        validate_matmul_minimal_contract(tvm.script.from_source(missing_packed_func))

    claims_runtime_replacement = source.replace(
        f'"triton_tvm.extern_runtime_replacement": "{EXTERN_GEMM_RUNTIME_REPLACEMENT}"',
        '"triton_tvm.extern_runtime_replacement": "available"',
    )
    with pytest.raises(TritonTVMContractError, match="extern_runtime_replacement"):
        validate_matmul_minimal_contract(
            tvm.script.from_source(claims_runtime_replacement)
        )

    claims_replacement_available = source.replace(
        '"triton_tvm.extern_runtime_replacement_available": False',
        '"triton_tvm.extern_runtime_replacement_available": True',
    )
    with pytest.raises(TritonTVMContractError, match="runtime replacement unavailable"):
        validate_matmul_minimal_contract(
            tvm.script.from_source(claims_replacement_available)
        )


def test_static_m9p_extern_addmm_bias_minimal_artifact():
    source_line = (
        "extern_kernels.addmm(arg0, reinterpret_tensor(arg1, (16, 64), "
        "(64, 1), 0), reinterpret_tensor(arg2, (64, 64), (1, 64), 0), "
        "alpha=1, beta=1, out=buf0)"
    )
    semantics = extract_matmul_semantics_from_wrapper_extern_addmm(
        {
            "op_name": "extern_kernels.addmm",
            "source": source_line,
            "case_name": "vit_tiny",
        }
    )
    same_semantics = extract_matmul_semantics_from_wrapper_extern(source_line)
    decision = TargetMatmulPolicy().decide(semantics, matmul_contract_ok=True)

    assert semantics.source_kind == "wrapper_extern_addmm_bias"
    assert semantics.epilogue_kind == "bias_add"
    assert semantics.bias_param == "arg0"
    assert semantics.bias_shape == (64,)
    assert semantics.bias_stride == (1,)
    assert semantics.b_layout == "transposed_weight_view"
    assert same_semantics.source_kind == "wrapper_extern_addmm_bias"
    assert decision.implementation_kind == "extern_addmm_bias"
    assert decision.extern_symbol == "extern_kernels.addmm"
    assert decision.extern_gemm_performance_claim is False

    artifact = build_extern_addmm_bias_tirx_source(semantics, decision)
    irmod = tvm.script.from_source(artifact)
    validate_matmul_minimal_contract(irmod)
    attrs = next(iter(irmod.functions.values())).attrs
    script = irmod.script()

    assert str(attrs["triton_tvm.matmul_source_kind"]) == "wrapper_extern_addmm_bias"
    assert str(attrs["triton_tvm.implementation_kind"]) == "extern_addmm_bias"
    assert str(attrs["triton_tvm.epilogue_kind"]) == "bias_add"
    assert str(attrs["triton_tvm.bias_shape"]) == "64"
    assert str(attrs["triton_tvm.bias_stride"]) == "1"
    assert int(attrs["triton_tvm.bias_rank"]) == 1
    assert "tvm.contrib.triton_tvm.extern_addmm_bias" in script
    assert 'with T.sblock("matmul")' not in script


def test_static_m9p_extern_addmm_bias_rejects_unsupported_subset():
    bad_alpha = (
        "extern_kernels.addmm(arg0, reinterpret_tensor(arg1, (16, 64), "
        "(64, 1), 0), reinterpret_tensor(arg2, (64, 64), (1, 64), 0), "
        "alpha=2, beta=1, out=buf0)"
    )
    with pytest.raises(UnsupportedTTIROpError, match="alpha=1 and beta=1"):
        extract_matmul_semantics_from_wrapper_extern_addmm(bad_alpha)

    bad_bias = (
        "extern_kernels.addmm(reinterpret_tensor(arg0, (16, 32), (32, 1), 0), "
        "reinterpret_tensor(arg1, (16, 64), (64, 1), 0), "
        "reinterpret_tensor(arg2, (64, 64), (1, 64), 0), "
        "alpha=1, beta=1, out=buf0)"
    )
    with pytest.raises(UnsupportedTTIROpError, match="rank-1 N or rank-2 MxN"):
        extract_matmul_semantics_from_wrapper_extern_addmm(bad_bias)

    bad_layout = (
        "extern_kernels.addmm(arg0, reinterpret_tensor(arg1, (16, 64), "
        "(1, 16), 0), reinterpret_tensor(arg2, (64, 64), (1, 64), 0), "
        "alpha=1, beta=1, out=buf0)"
    )
    with pytest.raises(UnsupportedTTIROpError, match="lhs must be row-major"):
        extract_matmul_semantics_from_wrapper_extern_addmm(bad_layout)


def test_static_m9_matmul_minimal_semantics_negative_boundaries():
    mismatch = _static_m9_dot_ttir("f16").replace(
        "tensor<16x4x!tt.ptr<f16>>",
        "tensor<15x4x!tt.ptr<f16>>",
    ).replace("tensor<16x4xf16>", "tensor<15x4xf16>")
    with pytest.raises(UnsupportedTTIROpError, match="K mismatch"):
        extract_matmul_semantics_from_ttir(TTIRReader().read(mismatch))

    masked = _static_m9_dot_ttir("f16").replace(
        "    %b_zero = arith.constant dense<0> : tensor<16x4xi32>",
        "    %b_zero = arith.constant dense<0> : tensor<16x4xi32>\n"
        "    %mask = arith.constant dense<true> : tensor<8x16xi1>",
    ).replace(
        "    %va = tt.load %a_ptr : tensor<8x16x!tt.ptr<f16>>",
        "    %va = tt.load %a_ptr, %mask : tensor<8x16x!tt.ptr<f16>>",
    )
    with pytest.raises(UnsupportedTTIROpError, match="unmasked tt.load"):
        extract_matmul_semantics_from_ttir(TTIRReader().read(masked))

    with pytest.raises(UnsupportedTTIROpError, match="result type|rank-2"):
        translate_ttir(_STATIC_PRE_M7_DOT_TTIR, grid=(1,), contract="matmul_minimal")


def test_static_m9_matmul_minimal_contract_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _static_m9_dot_ttir("f16"),
        grid=(1,),
        contract="matmul_minimal",
    )
    validate_matmul_minimal_contract(irmod)
    ValidateTritonKernelTIR(contract="matmul_minimal")(irmod)

    func = next(iter(irmod.functions.values()))
    sch = tvm.s_tir.Schedule(func)
    block = sch.get_sblock("matmul")
    block_node = sch.get(block)
    attrs = block_node.annotations
    script = irmod.script()

    assert len(sch.get_loops(block)) == 3
    assert [int(iter_var.iter_type) for iter_var in block_node.iter_vars] == [0, 0, 2]
    assert block_node.init is not None
    assert len(block_node.reads) == 2
    assert len(block_node.writes) == 1
    assert str(attrs["triton_tvm.contract"]) == "matmul_minimal"
    assert str(attrs["triton_tvm.matmul_source_kind"]) == "tt_dot"
    assert str(attrs["triton_tvm.implementation_kind"]) == "native_tir_schedule"
    assert str(attrs["triton_tvm.schedule_id"]) == NATIVE_TIR_MATMUL_SCHEDULE_ID

    assert meta.contract == "matmul_minimal"
    assert meta.contract_version == "matmul_minimal_m9_v1"
    assert meta.indexing_kind == "rank2_matmul"
    assert meta.execution_kind == "semantic_then_native_m9_matmul_sblock"
    assert meta.accumulator_dtype_policy == "fp32_accumulate"
    assert meta.mask_policy == "exact_unmasked"
    assert meta.axis_policy == "spatial_mn_reduce_k"
    assert meta.layout_policy == "rank2_row_major"
    assert meta.matmul_source_kind == "tt_dot"
    assert (meta.matmul_m, meta.matmul_n, meta.matmul_k) == (8, 4, 16)
    assert meta.matmul_contract_ok is True
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.schedule_id == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert meta.extern_symbol == ""
    assert meta.unsupported_matmul_reason == ""
    assert meta.buffer_extents == {
        "a": "T.int64(128)",
        "b": "T.int64(64)",
        "out": "T.int64(32)",
    }
    assert len(meta.cache_key) == 64

    assert 'with T.sblock("matmul")' in script
    assert "with T.init()" in script
    assert "call_extern" not in script
    assert 'thread="blockIdx.x"' in script
    assert 'thread="threadIdx.x"' in script


def test_static_m97_tiled_matmul_schedule_candidate_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _static_m97_tiled_dot_ttir("f16"),
        grid=(1,),
        contract="matmul_minimal",
    )
    validate_matmul_minimal_contract(irmod)
    func = next(iter(irmod.functions.values()))
    sch = tvm.s_tir.Schedule(func)
    block = sch.get_sblock("matmul")
    loops = sch.get_loops(block)
    block_node = sch.get(block)
    attrs = block_node.annotations

    assert meta.schedule_id == TILED_TIR_MATMUL_SCHEDULE_ID
    assert str(attrs["triton_tvm.schedule_id"]) == TILED_TIR_MATMUL_SCHEDULE_ID
    assert int(attrs["triton_tvm.tile_m"]) == 8
    assert int(attrs["triton_tvm.tile_n"]) == 8
    assert [int(sch.get(loop).extent) for loop in loops] == [1, 64, 16]

    bad_tile_attr = irmod.script().replace(
        '"triton_tvm.tile_n": 8',
        '"triton_tvm.tile_n": 4',
    )
    with pytest.raises(TritonTVMContractError, match="8x8 tile"):
        validate_matmul_minimal_contract(tvm.script.from_source(bad_tile_attr))

    unknown_schedule = irmod.script().replace(
        TILED_TIR_MATMUL_SCHEDULE_ID,
        "cuda_unknown_matmul_schedule",
    )
    with pytest.raises(TritonTVMContractError, match="unknown"):
        validate_matmul_minimal_contract(tvm.script.from_source(unknown_schedule))


def test_static_m9p_schedule_registry_and_tune_key_freeze():
    candidate_ids = matmul_schedule_candidate_ids()
    records = matmul_schedule_candidate_records()

    assert candidate_ids[0] == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert candidate_ids[-1] == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert len(records) == len(candidate_ids)
    assert all(record["schedule_id"] for record in records)
    assert all(record["implementation_kind"] == "native_tir_schedule" for record in records)
    assert records[0]["package_id"] == "M9.PA"
    assert records[1]["schedule_id"] == SIMT_TIR_MATMUL_SCHEDULE_ID
    assert records[1]["package_id"] == "M9.PC"
    assert records[0]["performance_claim"] is True
    assert MATMUL_SCHEDULE_REGISTRY_VERSION == "m9p_schedule_registry_v1"

    semantics = extract_matmul_semantics_from_ttir(
        TTIRReader().read(_static_m9p_dot_ttir(16, 16, 16))
    )
    tune_key = build_matmul_tune_key_payload(
        semantics,
        schedule_id=TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
        tune_params={"tile_m": 16, "tile_n": 16, "tile_k": 4},
        target_arch="sm_90",
        device_name="unit-test-device",
        framework_versions={"tvm": "unit", "triton": "unit"},
        cuda_target_metadata={"target": "cuda"},
    )
    assert tune_key["matmul_perf_envelope"] == MATMUL_PERF_ENVELOPE_ID
    assert tune_key["registry_version"] == MATMUL_SCHEDULE_REGISTRY_VERSION
    assert (tune_key["m"], tune_key["n"], tune_key["k"]) == (16, 16, 16)
    assert tune_key["schedule_id"] == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert tune_key["target_arch"] == "sm_90"
    assert tune_key["device_name"] == "unit-test-device"


def test_static_m9p_tensorcore_candidate_selection_and_report_fields():
    irmod, meta = translate_ttir(
        _static_m9p_dot_ttir(16, 16, 16),
        grid=(1,),
        contract="matmul_minimal",
    )
    validate_matmul_minimal_contract(irmod)
    func = next(iter(irmod.functions.values()))
    attrs = func.attrs
    script = irmod.script()

    assert meta.schedule_id == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert meta.selected_schedule_id == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert meta.matmul_perf_envelope == MATMUL_PERF_ENVELOPE_ID
    assert meta.candidate_schedule_ids[0] == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert meta.perf_guard_status == "not_measured"
    assert meta.schedule_reject_reasons[NATIVE_TIR_MATMUL_SCHEDULE_ID]
    assert meta.tune_key["registry_version"] == MATMUL_SCHEDULE_REGISTRY_VERSION
    assert meta.tune_key["schedule_id"] == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert str(attrs["triton_tvm.schedule_id"]) == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert str(attrs["triton_tvm.matmul_perf_envelope"]) == MATMUL_PERF_ENVELOPE_ID
    assert int(attrs["triton_tvm.tile_m"]) == 16
    assert int(attrs["triton_tvm.tile_n"]) == 16
    assert int(attrs["triton_tvm.tile_k"]) == 4
    assert "T.ptx_mma" in script or "tirx.ptx_mma" in script
    assert "m8n8k4" in script


def test_static_m9p_tensorcore_rejects_bf16_and_selects_simt_fallback():
    irmod, meta = translate_ttir(
        _static_m9p_dot_ttir(16, 16, 16, dtype="bf16"),
        grid=(1,),
        contract="matmul_minimal",
    )
    validate_matmul_minimal_contract(irmod)
    func = next(iter(irmod.functions.values()))
    sch = tvm.s_tir.Schedule(func)
    block = sch.get_sblock("matmul")
    attrs = sch.get(block).annotations

    assert meta.schedule_id == SIMT_TIR_MATMUL_SCHEDULE_ID
    assert meta.selected_schedule_id == SIMT_TIR_MATMUL_SCHEDULE_ID
    assert meta.matmul_perf_envelope == MATMUL_PERF_ENVELOPE_ID
    assert (
        meta.schedule_reject_reasons[TENSORCORE_TIR_MATMUL_SCHEDULE_ID]
        == "tensorcore_candidate_requires_fp16_inputs"
    )
    assert meta.schedule_reject_reasons[TILED_TIR_MATMUL_SCHEDULE_ID]
    assert meta.tune_key["schedule_id"] == SIMT_TIR_MATMUL_SCHEDULE_ID
    assert int(attrs["triton_tvm.tile_m"]) == 16
    assert int(attrs["triton_tvm.tile_n"]) == 16


def test_static_m9_audit_record_reports_native_matmul_details():
    record = audit_inductor_ttir(
        _static_m9_dot_ttir("f16"),
        case_name="m9_static",
        kernel_name="_m9_dot",
        contract="matmul_minimal",
    )

    assert record["translate_status"]["ok"] is True
    assert record["translate_status"]["bucket"] == "translated"
    assert record["matmul_source_kind"] == "tt_dot"
    assert (record["matmul_m"], record["matmul_n"], record["matmul_k"]) == (
        8,
        4,
        16,
    )
    assert record["matmul_contract_ok"] is True
    assert record["implementation_kind"] == "native_tir_schedule"
    assert record["schedule_id"] == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert record["extern_symbol"] == ""
    assert record["unsupported_matmul_reason"] == ""


def test_static_m4_reduction_minimal_contract_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _STATIC_REDUCTION_ROWSUM_TTIR,
        grid=(3,),
        contract="reduction_minimal",
    )
    validate_reduction_minimal_contract(irmod)
    ValidateTritonKernelTIR(contract="reduction_minimal")(irmod)

    script = irmod.script()
    assert meta.contract == "reduction_minimal"
    assert meta.contract_version == "reduction_minimal_v1"
    assert meta.indexing_kind == "block_reduction"
    assert meta.execution_kind == "serial_m4_single_lane"
    assert meta.accumulator_dtype_policy == "preserve_ttir_reduction_dtype"
    assert meta.epsilon_policy == "not_applicable"
    assert meta.mask_policy == "masked_reduction_loads_require_zero_other"
    assert meta.axis_policy == "axis_0_only"
    assert meta.layout_policy == "row_major_only"
    assert meta.extent_param == "n"
    assert meta.block_size == 64
    assert meta.buffer_extents == {
        "x": "(T.int64(3) * n)",
        "out": "T.int64(3)",
    }
    assert "red_0 = T.alloc_buffer" in script
    assert "for rk in range(64)" in script
    assert "if i == T.int64(0):" in script
    assert "out[row] = red_0[0]" in script


def test_static_pre_m5_norm_single_row_contract_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _STATIC_REDUCTION_ROWSUM_TTIR,
        grid=(3,),
        contract="norm_single_row",
    )
    validate_norm_single_row_contract(irmod)
    ValidateTritonKernelTIR(contract="norm_single_row")(irmod)

    assert meta.contract == "norm_single_row"
    assert meta.contract_version == "norm_single_row_v1"
    assert meta.indexing_kind == "single_row_norm"
    assert meta.execution_kind == "serial_m4_single_lane"
    assert meta.epsilon_policy == "runtime_and_constexpr_eps_supported"
    assert meta.layout_policy == "single_row_row_major"

    with pytest.raises(UnsupportedTTIROpError, match="requires at least one tt.reduce"):
        translate_ttir(_STATIC_DUAL_STORE_TTIR, grid=(1,), contract="norm_single_row")


def test_static_pre_m8_reduction_and_norm_contract_policy_metadata():
    reduction = get_triton_tvm_contract("reduction_minimal")
    norm = get_triton_tvm_contract("norm_single_row")

    assert reduction.execution_kind == "serial_m4_single_lane"
    assert reduction.accumulator_dtype_policy == "preserve_ttir_reduction_dtype"
    assert reduction.mask_policy == "masked_reduction_loads_require_zero_other"
    assert reduction.axis_policy == "axis_0_only"
    assert reduction.layout_policy == "row_major_only"

    assert norm.execution_kind == "serial_m4_single_lane"
    assert norm.accumulator_dtype_policy == "preserve_ttir_reduction_dtype"
    assert norm.epsilon_policy == "runtime_and_constexpr_eps_supported"
    assert norm.axis_policy == "axis_0_only"
    assert norm.layout_policy == "single_row_row_major"


def test_static_m25_tirscript_golden_shape():
    irmod, meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(irmod)

    script = irmod.script()
    assert meta.contract == "pointwise_flat"
    assert meta.canonical_contract == "pointwise_flat"
    assert meta.requested_contract == "pointwise_flat"
    assert meta.target_kind == "cuda"
    assert meta.extent_param == "n"
    assert meta.block_size == 64
    assert meta.indexing_kind == "pointwise"
    assert meta.launch_policy_id == "cuda_block_thread"
    assert meta.translator_version == "triton_tvm_python_pre_m5_contracts_v1"
    assert meta.contract_version == "pointwise_pre_m5_v1"
    assert meta.target_policy_version == "cuda_thread_binding_v1"
    assert meta.extent_kind == "runtime_param"
    assert meta.extent_value is None
    assert meta.buffer_extents == {
        "x": "n",
        "y": "n",
        "o0": "n",
        "o1": "n",
    }
    assert '"kind":"cuda"' in meta.target_attrs
    assert meta.abi == [
        {"name": "x", "kind": "pointer", "dtype": "float32"},
        {"name": "y", "kind": "pointer", "dtype": "float32"},
        {"name": "o0", "kind": "pointer", "dtype": "float32"},
        {"name": "o1", "kind": "pointer", "dtype": "float32"},
        {"name": "n", "kind": "scalar", "dtype": "int64"},
    ]
    assert meta.cache_policy == "disabled"
    assert meta.disk_cache_enabled is False
    assert meta.fallback_reason == ""
    assert "def _sd(" in script
    assert script.count("T.match_buffer") == 4
    assert 'thread="blockIdx.x"' in script
    assert 'thread="threadIdx.x"' in script
    assert script.count("if mask:") == 2
    assert "o0[i] = T.if_then_else(mask, x[i], T.float32(0.0))" in script
    assert " + T.if_then_else(mask, y[i], T.float32(0.0))" in script
    assert "o1[i] = T.if_then_else(mask, x[i], T.float32(0.0))" in script
    assert " - T.if_then_else(mask, y[i], T.float32(0.0))" in script


def test_static_m25_contract_boundaries_without_cuda_runtime():
    with pytest.raises(UnsupportedTTIROpError, match="contract='pointwise_flat'"):
        translate_ttir(_STATIC_DUAL_STORE_TTIR, grid=(1,))

    with pytest.warns(FutureWarning, match="cuda_pointwise_flat"):
        irmod, meta = translate_ttir(
            _STATIC_DUAL_STORE_TTIR,
            grid=(1,),
            contract="cuda_pointwise_flat",
        )
    assert meta.contract == "pointwise_flat"
    assert meta.canonical_contract == "pointwise_flat"
    assert meta.requested_contract == "pointwise_flat"
    with pytest.warns(FutureWarning, match="cuda_pointwise_flat"):
        validate_cuda_pointwise_flat_contract(irmod)
    with pytest.warns(FutureWarning, match="cuda_pointwise_flat"):
        ValidateTritonKernelTIR(contract="cuda_pointwise_flat")(irmod)

    with pytest.raises(TritonTVMContractError, match="exactly one BufferStore"):
        with pytest.warns(FutureWarning, match="cuda_minimal"):
            validate_cuda_minimal_contract(irmod)
    with pytest.raises(ValueError, match="Unsupported Triton TVM contract"):
        validate_triton_tvm_contract(irmod, "cuda_unknown")


def test_static_m25_alias_cache_key_and_validator_equivalence():
    canonical_mod, canonical_meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        contract="pointwise_flat",
    )
    with pytest.warns(FutureWarning, match="cuda_pointwise_flat"):
        alias_mod, alias_meta = translate_ttir(
            _STATIC_DUAL_STORE_TTIR,
            grid=(1,),
            contract="cuda_pointwise_flat",
        )

    assert canonical_meta.canonical_contract == "pointwise_flat"
    assert alias_meta.canonical_contract == "pointwise_flat"
    assert alias_meta.requested_contract == "pointwise_flat"
    assert canonical_meta.cache_key == alias_meta.cache_key
    assert canonical_mod.script() == alias_mod.script()
    validate_pointwise_flat_contract(alias_mod)
    with pytest.warns(FutureWarning, match="cuda_pointwise_flat"):
        validate_cuda_pointwise_flat_contract(alias_mod)


def test_static_pre_m5_pointwise_flat_indexed_capability_without_cuda_runtime():
    graph = TTIRReader().read(_STATIC_M35_INDEXED_TTIR)
    assert graph.params[0].type.raw == "!tt.ptr<f32>"
    load = graph.op_by_result()["vy"]
    assert load.attrs["raw_attrs"] == "evictionPolicy = evict_last"
    assert load.attrs["unknown_attrs"] == {"evictionPolicy": "evict_last"}

    irmod, meta = translate_ttir(
        graph,
        grid=(1,),
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(irmod)
    ValidateTritonKernelTIR(contract="pointwise_flat")(irmod)

    script = irmod.script()
    assert meta.contract == "pointwise_flat"
    assert meta.requested_contract == "pointwise_flat"
    assert meta.contract_version == "pointwise_pre_m5_v1"
    assert meta.extent_kind == "constant"
    assert meta.extent_value == 64
    assert meta.extent_param == ""
    assert meta.buffer_extents == {
        "x": "T.int64(64)",
        "y": "T.int64(16)",
        "out": "T.int64(64)",
    }
    assert 'y: T.Buffer((T.int64(16),), "float32")' in script
    assert "T.Select" in script
    assert "y[i % T.int64(16)]" in script
    assert "T.if_then_else(mask" not in script


def test_static_pre_m5_bool_bitcast_store_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _STATIC_M35_BOOL_BITCAST_TTIR,
        grid=(1,),
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(irmod)

    script = irmod.script()
    assert meta.abi[-1] == {"name": "out", "kind": "pointer", "dtype": "bool"}
    assert 'out: T.Buffer((T.int64(64),), "bool")' in script
    assert "out[i] = " in script
    assert 'T.Cast("int8"' not in script


def test_static_pre_m5_pointwise_flat_accepts_indexed_capability_surface():
    indexed_with_other = _STATIC_M35_INDEXED_TTIR.replace(
        "%vx = tt.load %x_ptr, %mask :",
        "%vx = tt.load %x_ptr, %mask, %cst :",
        1,
    ).replace(
        "%vy = tt.load %y_ptr, %mask evictionPolicy = evict_last :",
        "%vy = tt.load %y_ptr, %mask, %cst :",
        1,
    )
    indexed_mod, indexed_meta = translate_ttir(
        indexed_with_other, grid=(1,), contract="pointwise_flat"
    )
    validate_pointwise_flat_contract(indexed_mod)
    assert indexed_meta.indexing_kind == "pointwise"

    bool_mod, bool_meta = translate_ttir(
        _STATIC_M35_BOOL_BITCAST_STORE_ONLY_TTIR,
        grid=(1,),
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(bool_mod)
    assert bool_meta.abi[0] == {"name": "out", "kind": "pointer", "dtype": "bool"}


def test_static_m7_composed_index_classifier_without_cuda_runtime():
    graph = TTIRReader().read(_STATIC_M7_COMPOSED_INDEX_TTIR)
    summary = summarize_ttir_indexing(graph)
    assert summary["classifier_version"] == 1
    assert summary["unsupported_index_count"] == 0
    assert "affine" in " ".join(summary["index_kinds"])

    irmod, meta = translate_ttir(
        graph,
        grid=(1,),
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(irmod)
    script = irmod.script()
    assert meta.buffer_extents["y"].startswith("((T.ceildiv(n, T.int64(16))")
    assert "i // T.int64(16)" in script
    assert "i % T.int64(16)" in script


def test_static_m7_dtype_cast_and_erf_activation_without_cuda_runtime():
    graph = TTIRReader().read(_STATIC_M7_CAST_ERF_TTIR)
    assert "arith.sitofp" in [op.name for op in graph.ops]
    assert "arith.fptosi" in [op.name for op in graph.ops]

    irmod, meta = translate_ttir(
        graph,
        grid=(1,),
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(irmod)
    script = irmod.script()
    assert meta.contract == "pointwise_flat"
    assert "T.erf" in script
    assert 'T.Cast("int32"' in script


def test_static_pre_m5_pointwise_flat_validator_rejects_shape_only_matches():
    irmod, _ = translate_ttir(
        _STATIC_M35_INDEXED_TTIR,
        grid=(1,),
        contract="pointwise_flat",
    )
    script = irmod.script()
    old_store = (
        "                if mask:\n"
        "                    out[i] = T.Select(x[i] > T.float32(0.0), x[i], "
        "T.float32(0.0)) + y[i % T.int64(16)]"
    )
    assert old_store in script

    leaked_load = script.replace(
        old_store,
        "                leaked: T.float32 = x[i]\n"
        "                if mask:\n"
        "                    out[i] = leaked",
        1,
    )
    with pytest.raises(TritonTVMContractError, match="BufferLoad"):
        validate_pointwise_flat_contract(_tirx_from_source(leaked_load))

    unsupported_index = script.replace("y[i % T.int64(16)]", "y[i * T.int64(-1)]", 1)
    with pytest.raises(TritonTVMContractError, match="classified M7"):
        validate_pointwise_flat_contract(_tirx_from_source(unsupported_index))

    different_guard = script.replace(
        old_store,
        old_store
        + "\n"
        + "                other_mask: T.bool = i < T.int64(64)\n"
        + "                if other_mask:\n"
        + "                    out[i] = x[i]",
        1,
    )
    with pytest.raises(TritonTVMContractError, match="same mask guard"):
        validate_pointwise_flat_contract(_tirx_from_source(different_guard))


def test_static_pre_m5_unsafe_masked_load_without_other_rejected():
    unguarded_store = _STATIC_M35_INDEXED_TTIR.replace(
        "tt.store %out_ptr, %sum, %mask :",
        "tt.store %out_ptr, %sum :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="escapes the guarded store value"):
        translate_ttir(unguarded_store, grid=(1,), contract="pointwise_flat")

    bitcast_load = _STATIC_M35_BOOL_BITCAST_TTIR.replace(
        "tt.store %out_i8, %both_i8, %mask :",
        "%bad = tt.load %out_i8, %mask : tensor<64x!tt.ptr<i8>>\n"
        "    tt.store %out_i8, %bad, %mask :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="unsupported tt.bitcast"):
        translate_ttir(bitcast_load, grid=(1,), contract="pointwise_flat")


def test_static_m75_no_other_load_cannot_escape_through_masks_or_return():
    returned_load = _STATIC_M35_INDEXED_TTIR.replace(
        "tt.return",
        "tt.return %vx : tensor<64xf32>",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="cannot escape through tt.return"):
        translate_ttir(returned_load, grid=(1,), contract="pointwise_flat")

    load_derived_store_mask = _STATIC_M35_INDEXED_TTIR.replace(
        "%sum = arith.addf %relu, %vy : tensor<64xf32>\n"
        "    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%sum = arith.addf %relu, %vy : tensor<64xf32>\n"
        "    %load_mask = arith.cmpf ogt, %vx, %cst : tensor<64xf32>\n"
        "    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        1,
    ).replace(
        "tt.store %out_ptr, %sum, %mask :",
        "tt.store %out_ptr, %sum, %load_mask :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="same flat extent predicate"):
        translate_ttir(load_derived_store_mask, grid=(1,), contract="pointwise_flat")


def test_static_m75_stride_misclassification_negative_cases():
    negative_stride = _STATIC_M7_COMPOSED_INDEX_TTIR.replace(
        "%yidx = arith.addi %row_stride, %col : tensor<64xi32>",
        "%neg = arith.constant dense<-2> : tensor<64xi32>\n"
        "    %yidx = arith.muli %idx, %neg : tensor<64xi32>",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="negative stride"):
        translate_ttir(negative_stride, grid=(1,), contract="pointwise_flat")

    data_dependent_stride = _STATIC_M7_COMPOSED_INDEX_TTIR.replace(
        "%yidx = arith.addi %row_stride, %col : tensor<64xi32>",
        "%yidx = arith.muli %idx, %idx : tensor<64xi32>",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="unsupported composed indexing"):
        translate_ttir(data_dependent_stride, grid=(1,), contract="pointwise_flat")


def test_static_m25_target_policy_dispatch_without_cuda_runtime():
    for target in ("llvm", tvm.target.Target("llvm"), "rocm"):
        with pytest.raises(UnsupportedTargetPolicyError, match="No backend policy registered"):
            translate_ttir(_STATIC_DUAL_STORE_TTIR, grid=(1,), target=target)

    for target in ("cuda", tvm.target.Target("cuda"), "cuda -arch=sm_80"):
        irmod, meta = translate_ttir(
            _STATIC_DUAL_STORE_TTIR,
            grid=(1,),
            target=target,
            contract="pointwise_flat",
        )
        validate_pointwise_flat_contract(irmod)
        assert meta.target_kind == "cuda"
        assert meta.launch_policy_id == "cuda_block_thread"
        assert meta.target_policy_version == "cuda_thread_binding_v1"

    _, legacy_meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        target="cuda -arch=sm_80",
        contract="pointwise_flat",
    )
    assert "sm_80" in legacy_meta.target_attrs


def test_static_pre_m5_target_spelling_and_cache_key_canonicalization():
    from tvm import target as _target  # pylint: disable=import-outside-toplevel

    from_string_mod, from_string_meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        target="cuda -arch=sm_80",
        contract="pointwise_flat",
    )
    from_object_mod, from_object_meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        target=_target.Target({"kind": "cuda", "arch": "sm_80"}),
        contract="pointwise_flat",
    )

    assert from_string_meta.target_attrs == from_object_meta.target_attrs
    assert from_string_meta.cache_key == from_object_meta.cache_key
    assert from_string_mod.script() == from_object_mod.script()
    assert "cuda_" not in from_string_meta.canonical_contract
    assert "cuda_" not in from_string_meta.requested_contract


def test_static_pre_m6_cache_key_policy_excludes_graph_and_runtime_identity(monkeypatch):
    import tvm.contrib.triton_tvm.translator as translator_mod  # pylint: disable=import-outside-toplevel

    captured = {}
    original_cache_key = translator_mod._cache_key  # pylint: disable=protected-access

    def capture_cache_key(**kwargs):
        captured.update(kwargs)
        return original_cache_key(**kwargs)

    monkeypatch.setattr(
        translator_mod,
        "_cache_key",
        capture_cache_key,
    )
    translate_ttir(_STATIC_DUAL_STORE_TTIR, grid=(1,), contract="pointwise_flat")

    assert {
        "compile_region_name",
        "graph_id",
        "graph_kernel_index",
        "run_count",
        "session_id",
    }.isdisjoint(captured)
    assert {
        "abi",
        "canonical_contract",
        "cache_policy",
        "source_hash",
        "target",
        "ttir_hash",
    } <= set(captured)


def test_static_m4_reduction_minimal_negative_boundaries():
    with pytest.raises(UnsupportedTTIROpError, match="requires at least one tt.reduce"):
        translate_ttir(_STATIC_DUAL_STORE_TTIR, grid=(1,), contract="reduction_minimal")

    with pytest.raises(ValueError, match="static 1D grid"):
        translate_ttir(
            _STATIC_REDUCTION_ROWSUM_TTIR,
            grid=(1, 1),
            contract="reduction_minimal",
        )

    with pytest.raises(ValueError, match="static 1D grid"):
        translate_ttir(
            _STATIC_REDUCTION_ROWSUM_TTIR,
            grid=lambda _meta: (1,),
            contract="reduction_minimal",
        )

    bad_axis = _STATIC_REDUCTION_ROWSUM_TTIR.replace("axis = 0 : i32", "axis = 1 : i32", 1)
    with pytest.raises(UnsupportedTTIROpError, match="axis=0"):
        translate_ttir(bad_axis, grid=(1,), contract="reduction_minimal")

    bad_combiner = _STATIC_REDUCTION_ROWSUM_TTIR.replace("arith.addf", "arith.mulf", 1)
    with pytest.raises(UnsupportedTTIROpError, match="sum reduction"):
        translate_ttir(bad_combiner, grid=(1,), contract="reduction_minimal")

    load_without_other = _STATIC_REDUCTION_ROWSUM_TTIR.replace(
        "tt.load %x_ptr, %mask, %zero :",
        "tt.load %x_ptr, %mask :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="masked tt.load without"):
        translate_ttir(load_without_other, grid=(1,), contract="reduction_minimal")

    strided_pointer = _STATIC_REDUCTION_ROWSUM_TTIR.replace(
        "%x_ptr = tt.addptr %x_splat, %range :",
        "%two = arith.constant dense<2> : tensor<64xi32>\n"
        "    %strided = arith.muli %range, %two : tensor<64xi32>\n"
        "    %x_ptr = tt.addptr %x_splat, %strided :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="non-row-major"):
        translate_ttir(strided_pointer, grid=(1,), contract="reduction_minimal")


def test_static_m25_extent_inference_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _STATIC_TWO_SCALARS_BEFORE_EXTENT_TTIR,
        grid=(1,),
        contract="pointwise_minimal",
    )
    validate_triton_tvm_contract(irmod, "pointwise_minimal")
    assert meta.extent_param == "n"
    assert meta.abi == [
        {"name": "x", "kind": "pointer", "dtype": "float32"},
        {"name": "out", "kind": "pointer", "dtype": "float32"},
        {"name": "alpha", "kind": "scalar", "dtype": "float32"},
        {"name": "beta", "kind": "scalar", "dtype": "float32"},
        {"name": "n", "kind": "scalar", "dtype": "int64"},
    ]

    irmod, meta = translate_ttir(
        _STATIC_SCALAR_BEFORE_EXTENT_TTIR,
        grid=(1,),
        contract="pointwise_minimal",
    )
    validate_triton_tvm_contract(irmod, "pointwise_minimal")

    script = irmod.script()
    assert meta.abi == [
        {"name": "x", "kind": "pointer", "dtype": "float32"},
        {"name": "out", "kind": "pointer", "dtype": "float32"},
        {"name": "alpha", "kind": "scalar", "dtype": "float32"},
        {"name": "n", "kind": "scalar", "dtype": "int64"},
    ]
    assert meta.extent_param == "n"
    assert "x = T.match_buffer(x_handle, (n,)" in script
    assert "out = T.match_buffer(out_handle, (n,)" in script
    assert "(n + T.int64(63)) // T.int64(64)" in script
    assert "alpha" in script


def test_static_m25_rejects_inconsistent_memory_masks_without_cuda_runtime():
    different_store_mask = _STATIC_DUAL_STORE_TTIR.replace(
        "%diff = arith.subf %vx, %vy : tensor<64xf32>\n"
        "    tt.store %o1_ptr, %diff, %mask :",
        "%n_splat_2 = tt.splat %n : i64 -> tensor<64xi64>\n"
        "    %mask_2 = arith.cmpi slt, %idx64, %n_splat_2 : tensor<64xi64>\n"
        "    %diff = arith.subf %vx, %vy : tensor<64xf32>\n"
        "    tt.store %o1_ptr, %diff, %mask_2 :",
        1,
    )
    irmod, _ = translate_ttir(different_store_mask, grid=(1,), contract="pointwise_flat")
    validate_pointwise_flat_contract(irmod)

    different_extent = _STATIC_DUAL_STORE_TTIR.replace(
        "%n:i64) {",
        "%n:i64,%m:i64) {",
        1,
    ).replace(
        "%diff = arith.subf %vx, %vy : tensor<64xf32>\n"
        "    tt.store %o1_ptr, %diff, %mask :",
        "%m_splat = tt.splat %m : i64 -> tensor<64xi64>\n"
        "    %mask_m = arith.cmpi slt, %idx64, %m_splat : tensor<64xi64>\n"
        "    %diff = arith.subf %vx, %vy : tensor<64xf32>\n"
        "    tt.store %o1_ptr, %diff, %mask_m :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="ambiguous extent candidates"):
        translate_ttir(different_extent, grid=(1,), contract="pointwise_flat")


def test_static_m25_negative_translate_without_cuda_runtime():
    masked_load_without_other = _STATIC_SCALAR_BEFORE_EXTENT_TTIR.replace(
        "tt.load %x_ptr, %mask, %cst :",
        "tt.load %x_ptr, %mask :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="masked tt.load without other"):
        translate_ttir(masked_load_without_other, grid=(1,), contract="pointwise_minimal")

    reduction_ttir = _STATIC_DUAL_STORE_TTIR.replace(
        "    tt.return",
        "    %bad = tt.reduce %vx : tensor<64xf32>\n    tt.return",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="tt.reduce"):
        translate_ttir(reduction_ttir, grid=(1,), contract="pointwise_flat")

    non_contiguous_ttir = _STATIC_DUAL_STORE_TTIR.replace(
        "%x_ptr = tt.addptr %x_splat, %idx :",
        "%x_ptr = tt.addptr %x_splat, %offsets :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="unsupported composed indexing pattern"):
        translate_ttir(non_contiguous_ttir, grid=(1,), contract="pointwise_flat")

    maskless_store_ttir = _STATIC_DUAL_STORE_TTIR.replace(
        "tt.store %o1_ptr, %diff, %mask :",
        "tt.store %o1_ptr, %diff :",
        1,
    )
    irmod, _ = translate_ttir(maskless_store_ttir, grid=(1,), contract="pointwise_flat")
    script = irmod.script()
    assert "if mask:" in script


def _op_snapshot(graph):
    return [(op.name, tuple(op.results), tuple(op.operands)) for op in graph.ops]


def _type_snapshot(graph, result: str):
    op = graph.op_by_result()[result]
    assert len(op.result_types) == 1
    ty = op.result_types[0]
    return ty.raw, ty.dtype, ty.shape, ty.is_pointer


def _tirx_from_source(script: str):
    from tvm.script import ir as I  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    return tvm.script.from_source(script, {"I": I, "T": T})


if __name__ == "__main__":
    tvm.testing.main()
