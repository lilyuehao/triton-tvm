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
from tvm.contrib.triton_tvm import (
    NormalizeTritonKernelTIR,
    TTIRReader,
    TritonTVMContractError,
    UnsupportedTTIROpError,
    UnsupportedTargetPolicyError,
    translate_ttir,
    validate_cuda_minimal_contract,
    validate_cuda_pointwise_indexed_contract,
    validate_cuda_pointwise_flat_contract,
    validate_pointwise_indexed_contract,
    validate_pointwise_flat_contract,
    validate_reduction_minimal_contract,
    validate_triton_tvm_contract,
)


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


def test_static_m4_reduction_minimal_contract_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _STATIC_REDUCTION_ROWSUM_TTIR,
        grid=(3,),
        contract="reduction_minimal",
    )
    validate_reduction_minimal_contract(irmod)
    NormalizeTritonKernelTIR(contract="reduction_minimal")(irmod)

    script = irmod.script()
    assert meta.contract == "reduction_minimal"
    assert meta.contract_version == "reduction_minimal_v1"
    assert meta.indexing_kind == "block_reduction"
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
    assert meta.indexing_kind == "flat_contiguous"
    assert meta.launch_policy_id == "cuda_block_thread"
    assert meta.translator_version == "triton_tvm_python_m35_hardened_v1"
    assert meta.contract_version == "pointwise_v1"
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

    irmod, meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        contract="cuda_pointwise_flat",
    )
    assert meta.contract == "pointwise_flat"
    assert meta.canonical_contract == "pointwise_flat"
    assert meta.requested_contract == "cuda_pointwise_flat"
    validate_cuda_pointwise_flat_contract(irmod)
    NormalizeTritonKernelTIR(contract="cuda_pointwise_flat")(irmod)

    with pytest.raises(TritonTVMContractError, match="exactly one BufferStore"):
        validate_cuda_minimal_contract(irmod)
    with pytest.raises(ValueError, match="Unsupported Triton TVM contract"):
        validate_triton_tvm_contract(irmod, "cuda_unknown")


def test_static_m25_alias_cache_key_and_validator_equivalence():
    canonical_mod, canonical_meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        contract="pointwise_flat",
    )
    alias_mod, alias_meta = translate_ttir(
        _STATIC_DUAL_STORE_TTIR,
        grid=(1,),
        contract="cuda_pointwise_flat",
    )

    assert canonical_meta.canonical_contract == "pointwise_flat"
    assert alias_meta.canonical_contract == "pointwise_flat"
    assert alias_meta.requested_contract == "cuda_pointwise_flat"
    assert canonical_meta.cache_key == alias_meta.cache_key
    assert canonical_mod.script() == alias_mod.script()
    validate_pointwise_flat_contract(alias_mod)
    validate_cuda_pointwise_flat_contract(alias_mod)


def test_static_m35_indexed_contract_without_cuda_runtime():
    graph = TTIRReader().read(_STATIC_M35_INDEXED_TTIR)
    assert graph.params[0].type.raw == "!tt.ptr<f32>"
    load = graph.op_by_result()["vy"]
    assert load.attrs["raw_attrs"] == "evictionPolicy = evict_last"
    assert load.attrs["unknown_attrs"] == {"evictionPolicy": "evict_last"}

    irmod, meta = translate_ttir(
        graph,
        grid=(1,),
        contract="cuda_pointwise_indexed",
    )
    validate_pointwise_indexed_contract(irmod)
    validate_cuda_pointwise_indexed_contract(irmod)
    NormalizeTritonKernelTIR(contract="pointwise_indexed")(irmod)

    script = irmod.script()
    assert meta.contract == "pointwise_indexed"
    assert meta.requested_contract == "cuda_pointwise_indexed"
    assert meta.contract_version == "pointwise_indexed_v2"
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


def test_static_m35_bool_bitcast_store_without_cuda_runtime():
    irmod, meta = translate_ttir(
        _STATIC_M35_BOOL_BITCAST_TTIR,
        grid=(1,),
        contract="pointwise_indexed",
    )
    validate_pointwise_indexed_contract(irmod)

    script = irmod.script()
    assert meta.abi[-1] == {"name": "out", "kind": "pointer", "dtype": "bool"}
    assert 'out: T.Buffer((T.int64(64),), "bool")' in script
    assert "out[i] = " in script
    assert 'T.Cast("int8"' not in script


def test_static_m35_hardening_pointwise_flat_rejects_indexed_only_surface():
    indexed_with_other = _STATIC_M35_INDEXED_TTIR.replace(
        "%vx = tt.load %x_ptr, %mask :",
        "%vx = tt.load %x_ptr, %mask, %cst :",
        1,
    ).replace(
        "%vy = tt.load %y_ptr, %mask evictionPolicy = evict_last :",
        "%vy = tt.load %y_ptr, %mask, %cst :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="arith.remsi.*pointwise_flat"):
        translate_ttir(indexed_with_other, grid=(1,), contract="pointwise_flat")

    with pytest.raises(UnsupportedTTIROpError, match="tt.bitcast.*pointwise_flat"):
        translate_ttir(
            _STATIC_M35_BOOL_BITCAST_STORE_ONLY_TTIR,
            grid=(1,),
            contract="pointwise_flat",
        )


def test_static_m35_hardening_indexed_validator_rejects_shape_only_matches():
    irmod, _ = translate_ttir(
        _STATIC_M35_INDEXED_TTIR,
        grid=(1,),
        contract="pointwise_indexed",
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
        validate_pointwise_indexed_contract(_tirx_from_source(leaked_load))

    unsupported_index = script.replace("y[i % T.int64(16)]", "y[i + T.int64(1)]", 1)
    with pytest.raises(TritonTVMContractError, match="buffer indices"):
        validate_pointwise_indexed_contract(_tirx_from_source(unsupported_index))

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
        validate_pointwise_indexed_contract(_tirx_from_source(different_guard))


def test_static_m35_unsafe_masked_load_without_other_rejected():
    unguarded_store = _STATIC_M35_INDEXED_TTIR.replace(
        "tt.store %out_ptr, %sum, %mask :",
        "tt.store %out_ptr, %sum :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="escapes the guarded store value"):
        translate_ttir(unguarded_store, grid=(1,), contract="pointwise_indexed")

    bitcast_load = _STATIC_M35_BOOL_BITCAST_TTIR.replace(
        "tt.store %out_i8, %both_i8, %mask :",
        "%bad = tt.load %out_i8, %mask : tensor<64x!tt.ptr<i8>>\n"
        "    tt.store %out_i8, %bad, %mask :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="unsupported tt.bitcast"):
        translate_ttir(bitcast_load, grid=(1,), contract="pointwise_indexed")


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
    with pytest.raises(UnsupportedTTIROpError, match="same flat extent predicate"):
        translate_ttir(different_store_mask, grid=(1,), contract="pointwise_flat")

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
    with pytest.raises(UnsupportedTTIROpError, match="same flat extent predicate"):
        translate_ttir(different_extent, grid=(1,), contract="pointwise_flat")


def test_static_m25_negative_translate_without_cuda_runtime():
    masked_load_without_other = _STATIC_DUAL_STORE_TTIR.replace(
        "tt.load %x_ptr, %mask, %cst :",
        "tt.load %x_ptr, %mask :",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="masked tt.load without other"):
        translate_ttir(masked_load_without_other, grid=(1,), contract="pointwise_flat")

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
    with pytest.raises(UnsupportedTTIROpError, match="non-contiguous pointer pattern"):
        translate_ttir(non_contiguous_ttir, grid=(1,), contract="pointwise_flat")

    unguarded_store_ttir = _STATIC_DUAL_STORE_TTIR.replace(
        "tt.store %o1_ptr, %diff, %mask :",
        "tt.store %o1_ptr, %diff :",
        1,
    )
    with pytest.raises(TritonTVMContractError, match="guard every BufferStore"):
        translate_ttir(unguarded_store_ttir, grid=(1,), contract="pointwise_flat")


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
