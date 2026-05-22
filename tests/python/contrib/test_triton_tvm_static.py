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
    UnsupportedContractError,
    UnsupportedTTIROpError,
    UnsupportedTargetPolicyError,
    translate_ttir,
    validate_cuda_minimal_contract,
    validate_cuda_pointwise_flat_contract,
    validate_pointwise_flat_contract,
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
    assert meta.translator_version == "triton_tvm_python_m25_policy_v1"
    assert meta.contract_version == "pointwise_v1"
    assert meta.target_policy_version == "cuda_thread_binding_v1"
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


def test_static_m25_unsupported_contract_on_cuda_policy():
    with pytest.raises(UnsupportedContractError, match="does not support"):
        translate_ttir(_STATIC_DUAL_STORE_TTIR, grid=(1,), contract="reduction_minimal")


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


if __name__ == "__main__":
    tvm.testing.main()
