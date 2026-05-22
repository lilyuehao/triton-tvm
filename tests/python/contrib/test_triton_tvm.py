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

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import (
    NormalizeTritonKernelTIR,
    TTIRReader,
    UnsupportedTTIROpError,
    build_triton_tvm,
    lower_to_ttir,
    translate_ttir,
    validate_cuda_pointwise_flat_contract,
    validate_pointwise_flat_contract,
    validate_pointwise_minimal_contract,
)

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    pytestmark = pytest.skip("Triton or PyTorch is not available", allow_module_level=True)


@triton.jit
def _vector_add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, vx + vy, mask=mask)


@triton.jit
def _pointwise_chain_kernel(x, y, z, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    vz = tl.load(z + offsets, mask=mask, other=0.0)
    value = vx * 2.0 + vy - vz
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _dual_store_kernel(x, y, out0, out1, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out0 + offsets, vx + vy, mask=mask)
    tl.store(out1 + offsets, vx - vy, mask=mask)


@triton.jit
def _scalar_broadcast_kernel(x, out0, out1, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    shifted = alpha + 1.0
    tl.store(out0 + offsets, vx + shifted, mask=mask)
    tl.store(out1 + offsets, vx * alpha, mask=mask)


@triton.jit
def _strided_pointer_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    strided = offsets * 2
    vx = tl.load(x + strided, mask=mask, other=0.0)
    tl.store(out + offsets, vx, mask=mask)


def _signature():
    return {
        "x": "*fp32",
        "y": "*fp32",
        "out": "*fp32",
        "n": "i64",
        "BLOCK": "constexpr",
    }


def _chain_signature():
    return {
        "x": "*fp32",
        "y": "*fp32",
        "z": "*fp32",
        "out": "*fp32",
        "n": "i64",
        "BLOCK": "constexpr",
    }


def _dual_store_signature():
    return {
        "x": "*fp32",
        "y": "*fp32",
        "out0": "*fp32",
        "out1": "*fp32",
        "n": "i64",
        "BLOCK": "constexpr",
    }


def _scalar_broadcast_signature():
    return {
        "x": "*fp32",
        "out0": "*fp32",
        "out1": "*fp32",
        "n": "i64",
        "alpha": "fp32",
        "BLOCK": "constexpr",
    }


def _strided_pointer_signature():
    return {
        "x": "*fp32",
        "out": "*fp32",
        "n": "i64",
        "BLOCK": "constexpr",
    }


def _lower(block=128):
    return lower_to_ttir(_vector_add_kernel, _signature(), {"BLOCK": block})


def _lower_chain(block=64):
    return lower_to_ttir(_pointwise_chain_kernel, _chain_signature(), {"BLOCK": block})


def _lower_dual_store(block=64):
    return lower_to_ttir(_dual_store_kernel, _dual_store_signature(), {"BLOCK": block})


def _lower_scalar_broadcast(block=64):
    return lower_to_ttir(
        _scalar_broadcast_kernel, _scalar_broadcast_signature(), {"BLOCK": block}
    )


def _native_triton_add(x_np, y_np, block):
    x_torch = torch.tensor(x_np, device="cuda")
    y_torch = torch.tensor(y_np, device="cuda")
    out_torch = torch.empty_like(x_torch)
    n = x_np.shape[0]
    _vector_add_kernel[(triton.cdiv(n, block),)](x_torch, y_torch, out_torch, n, BLOCK=block)
    torch.cuda.synchronize()
    return out_torch.cpu().numpy()


@tvm.testing.requires_cuda
def test_lower_to_ttir_dump_and_reader(tmp_path):
    dump_path = tmp_path / "vector_add.ttir"
    artifact = lower_to_ttir(
        _vector_add_kernel,
        _signature(),
        {"BLOCK": 128},
        dump_path=dump_path,
    )

    assert artifact.triton_version == "3.7.0"
    assert dump_path.read_text(encoding="utf-8") == artifact.ttir
    assert "tt.load" in artifact.ttir

    graph = TTIRReader().read(artifact.ttir)
    assert graph.function_name == "_vector_add_kernel"
    assert [param.name for param in graph.params] == ["x", "y", "out", "n"]
    assert {op.name for op in graph.ops} >= {
        "tt.get_program_id",
        "tt.make_range",
        "tt.load",
        "tt.store",
        "arith.addf",
    }


@tvm.testing.requires_cuda
def test_ttir_reader_pointwise_corpus_before_m2():
    chain = lower_to_ttir(_pointwise_chain_kernel, _chain_signature(), {"BLOCK": 64})
    chain_graph = TTIRReader().read(chain.ttir)
    chain_ops = {op.name for op in chain_graph.ops}
    assert chain_graph.function_name == "_pointwise_chain_kernel"
    assert {"arith.mulf", "arith.addf", "arith.subf", "tt.load", "tt.store"} <= chain_ops

    dual = _lower_dual_store(block=64)
    dual_graph = TTIRReader().read(dual.ttir)
    assert dual_graph.function_name == "_dual_store_kernel"
    assert [op.name for op in dual_graph.ops].count("tt.store") == 2


@tvm.testing.requires_cuda
def test_translate_ttir_contract_and_metadata():
    artifact = _lower(block=128)
    irmod, meta = translate_ttir(artifact, grid=(8,), target="cuda")
    validate_pointwise_minimal_contract(irmod)

    script = irmod.script()
    assert "T.thread_binding" in script
    assert 'thread="blockIdx.x"' in script
    assert 'thread="threadIdx.x"' in script
    assert "T.if_then_else" in script
    assert irmod.attrs is None or irmod.attrs.get("external_mods", None) is None

    assert meta.kernel_name == "_vector_add_kernel"
    assert meta.constexprs == {"BLOCK": 128}
    assert meta.signature["x"] == "*fp32"
    assert meta.contract == "pointwise_minimal"
    assert meta.canonical_contract == "pointwise_minimal"
    assert meta.requested_contract == "pointwise_minimal"
    assert meta.target_kind == "cuda"
    assert meta.extent_param == "n"
    assert meta.block_size == 128
    assert meta.indexing_kind == "flat_contiguous"
    assert meta.launch_policy_id == "cuda_block_thread"
    assert meta.abi == [
        {"name": "x", "kind": "pointer", "dtype": "float32"},
        {"name": "y", "kind": "pointer", "dtype": "float32"},
        {"name": "out", "kind": "pointer", "dtype": "float32"},
        {"name": "n", "kind": "scalar", "dtype": "int64"},
    ]
    assert len(meta.cache_key) == 64


@tvm.testing.requires_cuda
@pytest.mark.parametrize("n", [1024, 1000])
def test_build_run_and_compare_numpy_and_native_triton(n):
    block = 128
    x_np = np.random.rand(n).astype("float32")
    y_np = np.random.rand(n).astype("float32")
    expected_np = x_np + y_np
    expected_triton = _native_triton_add(x_np, y_np, block)

    artifact = _lower(block=block)
    irmod, meta = translate_ttir(artifact, grid=(triton.cdiv(n, block),), target="cuda")

    @tvm.ir.transform.module_pass(opt_level=0, name="triton_tvm.TestUserPass")
    def mark_user_pass(mod, _ctx):  # pylint: disable=unused-argument
        return mod.with_attr("triton_tvm.user_pass", "seen")

    built = build_triton_tvm(irmod, meta, passes=[mark_user_pass], target="cuda")
    assert built.irmod.attrs["triton_tvm.user_pass"] == "seen"

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_np, dev)
    y_tvm = tvm.runtime.tensor(y_np, dev)
    out_tvm = tvm.runtime.empty((n,), "float32", dev)
    built.run([x_tvm, y_tvm, out_tvm, n])

    tvm.testing.assert_allclose(expected_triton, expected_np, rtol=1e-5, atol=1e-5)
    tvm.testing.assert_allclose(out_tvm.numpy(), expected_np, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_m2a_pointwise_chain_build_run_odd_size():
    block = 64
    n = 1000
    rng = np.random.default_rng(0)
    x_np = rng.random(n, dtype=np.float32)
    y_np = rng.random(n, dtype=np.float32)
    z_np = rng.random(n, dtype=np.float32)
    expected_np = x_np * np.float32(2.0) + y_np - z_np

    artifact = _lower_chain(block=block)
    graph = TTIRReader().read(artifact.ttir)
    ops = [op.name for op in graph.ops]
    assert ops.count("tt.load") == 3
    assert {"arith.mulf", "arith.addf", "arith.subf"} <= set(ops)

    irmod, meta = translate_ttir(artifact, grid=(triton.cdiv(n, block),), target="cuda")
    validate_pointwise_minimal_contract(irmod)
    built = build_triton_tvm(irmod, meta)

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_np, dev)
    y_tvm = tvm.runtime.tensor(y_np, dev)
    z_tvm = tvm.runtime.tensor(z_np, dev)
    out_tvm = tvm.runtime.empty((n,), "float32", dev)
    built.run([x_tvm, y_tvm, z_tvm, out_tvm, n])

    tvm.testing.assert_allclose(out_tvm.numpy(), expected_np, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_m25_dual_store_pointwise_flat_build_run():
    block = 64
    n = 1000
    rng = np.random.default_rng(1)
    x_np = rng.random(n, dtype=np.float32)
    y_np = rng.random(n, dtype=np.float32)

    artifact = _lower_dual_store(block=block)
    irmod, meta = translate_ttir(
        artifact,
        grid=(triton.cdiv(n, block),),
        target="cuda",
        contract="pointwise_flat",
    )
    validate_pointwise_flat_contract(irmod)
    assert meta.contract == "pointwise_flat"

    built = build_triton_tvm(
        irmod,
        meta,
        passes=[NormalizeTritonKernelTIR(contract="pointwise_flat")],
    )

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_np, dev)
    y_tvm = tvm.runtime.tensor(y_np, dev)
    out0_tvm = tvm.runtime.empty((n,), "float32", dev)
    out1_tvm = tvm.runtime.empty((n,), "float32", dev)
    built.run([x_tvm, y_tvm, out0_tvm, out1_tvm, n])

    tvm.testing.assert_allclose(out0_tvm.numpy(), x_np + y_np, rtol=1e-5, atol=1e-5)
    tvm.testing.assert_allclose(out1_tvm.numpy(), x_np - y_np, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_m25_scalar_broadcast_pointwise_flat_build_run():
    block = 64
    n = 777
    alpha = np.float32(2.5)
    rng = np.random.default_rng(2)
    x_np = rng.random(n, dtype=np.float32)

    artifact = _lower_scalar_broadcast(block=block)
    graph = TTIRReader().read(artifact.ttir)
    ops = [op.name for op in graph.ops]
    assert ops.count("tt.store") == 2
    assert "tt.splat" in ops

    irmod, meta = translate_ttir(
        artifact,
        grid=(triton.cdiv(n, block),),
        target="cuda",
        contract="cuda_pointwise_flat",
    )
    validate_cuda_pointwise_flat_contract(irmod)
    assert meta.contract == "pointwise_flat"
    assert meta.requested_contract == "cuda_pointwise_flat"
    built = build_triton_tvm(irmod, meta)

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_np, dev)
    out0_tvm = tvm.runtime.empty((n,), "float32", dev)
    out1_tvm = tvm.runtime.empty((n,), "float32", dev)
    built.run([x_tvm, out0_tvm, out1_tvm, n, float(alpha)])

    tvm.testing.assert_allclose(out0_tvm.numpy(), x_np + alpha + 1.0, rtol=1e-5, atol=1e-5)
    tvm.testing.assert_allclose(out1_tvm.numpy(), x_np * alpha, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cuda
def test_negative_unsupported_dot():
    artifact = _lower(block=128)
    bad_ttir = artifact.ttir.replace(
        "    tt.return",
        "    %bad = tt.dot %vx_6, %vy_8 : tensor<128xf32>\n    tt.return",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="tt.dot"):
        translate_ttir(bad_ttir, grid=(1,), target="cuda")


@tvm.testing.requires_cuda
def test_negative_masked_load_without_other():
    artifact = _lower(block=128)
    bad_ttir = re.sub(
        r"(tt\.load\s+%[\w$.]+,\s+%[\w$.]+),\s+%[\w$.]+\s+:",
        r"\1 :",
        artifact.ttir,
        count=1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="masked tt.load without other"):
        translate_ttir(bad_ttir, grid=(1,), target="cuda")


@tvm.testing.requires_cuda
def test_m25_negative_multiple_stores_require_pointwise_flat_contract():
    artifact = _lower_dual_store(block=64)
    with pytest.raises(UnsupportedTTIROpError, match="multiple tt.store"):
        translate_ttir(artifact, grid=(1,), target="cuda")


@tvm.testing.requires_cuda
def test_m25_negative_unsupported_reduction_and_atomic():
    artifact = _lower(block=128)
    reduction_ttir = artifact.ttir.replace(
        "    tt.return",
        "    %bad = tt.reduce %vx_6 : tensor<128xf32>\n    tt.return",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="tt.reduce"):
        translate_ttir(reduction_ttir, grid=(1,), target="cuda")

    atomic_ttir = artifact.ttir.replace(
        "    tt.return",
        "    tt.atomic_rmw add, %vx_5, %vx_6, %mask_4 : tensor<128x!tt.ptr<f32>>\n"
        "    tt.return",
        1,
    )
    with pytest.raises(UnsupportedTTIROpError, match="tt.atomic_rmw"):
        translate_ttir(atomic_ttir, grid=(1,), target="cuda")


@tvm.testing.requires_cuda
def test_m25_negative_non_contiguous_pointer_pattern():
    artifact = lower_to_ttir(
        _strided_pointer_kernel,
        _strided_pointer_signature(),
        {"BLOCK": 64},
    )
    with pytest.raises(UnsupportedTTIROpError, match="non-contiguous pointer pattern"):
        translate_ttir(
            artifact,
            grid=(1,),
            target="cuda",
            contract="cuda_pointwise_flat",
        )


@tvm.testing.requires_cuda
def test_negative_unsupported_emit_and_target():
    artifact = _lower(block=128)
    with pytest.raises(ValueError, match="emit='tirx'"):
        translate_ttir(artifact, grid=(1,), emit="tir")
    with pytest.raises(ValueError, match="No backend policy registered"):
        translate_ttir(artifact, grid=(1,), target="llvm")


@tvm.testing.requires_cuda
def test_negative_runtime_abi_and_grid_validation():
    artifact = _lower(block=128)
    irmod, meta = translate_ttir(artifact, grid=(1,), target="cuda")
    built = build_triton_tvm(irmod, meta)

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(np.ones((128,), dtype="float32"), dev)
    y_tvm = tvm.runtime.tensor(np.ones((128,), dtype="float32"), dev)
    out_tvm = tvm.runtime.empty((128,), "float32", dev)
    bad_out_tvm = tvm.runtime.empty((128,), "float16", dev)

    with pytest.raises(ValueError, match="Expected 4 runtime arguments"):
        built.run([x_tvm, y_tvm, out_tvm])
    with pytest.raises(ValueError, match="does not match translated grid"):
        built.run([x_tvm, y_tvm, out_tvm, 128], grid=(2,))
    with pytest.raises(TypeError, match="has dtype float16"):
        built.run([x_tvm, y_tvm, bad_out_tvm, 128])
    with pytest.raises(TypeError, match="must be an integer scalar"):
        built.run([x_tvm, y_tvm, out_tvm, 128.0])


@tvm.testing.requires_cuda
def test_negative_build_target_mismatch():
    artifact = _lower(block=128)
    irmod, meta = translate_ttir(artifact, grid=(1,), target="cuda")
    with pytest.raises(ValueError, match="does not match translated target"):
        build_triton_tvm(irmod, meta, target="llvm")


@tvm.testing.requires_cuda
def test_negative_stream_not_supported():
    artifact = _lower(block=128)
    irmod, meta = translate_ttir(artifact, grid=(1,), target="cuda")
    built = build_triton_tvm(irmod, meta)
    with pytest.raises(NotImplementedError, match="stream is not supported"):
        built.run([], stream=object())


if __name__ == "__main__":
    tvm.testing.main()
