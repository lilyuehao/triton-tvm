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
"""M12.12 real Triton JIT tl.dot bridge coverage."""

import pytest

import tvm.testing
from tvm.contrib.triton_tvm import lower_to_ttir, translate_ttir, validate_matmul_minimal_contract
from tvm.contrib.triton_tvm.m1212_real_tl_dot_bridge import NEGATIVE_CASES
from tvm.contrib.triton_tvm.matmul import (
    GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
    REAL_JIT_TT_DOT_SOURCE_KIND,
    TargetMatmulPolicy,
    extract_matmul_semantics_from_ttir,
)
from tvm.contrib.triton_tvm.ttir import TTIRReader

try:
    import triton
    import triton.language as tl
except ImportError:
    pytestmark = pytest.skip("Triton is not available", allow_module_level=True)


@triton.jit
def _m1212_real_dot(a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
    bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
    acc = tl.dot(av, bv, input_precision="tf32")
    tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


@triton.jit
def _m1212_real_dot_f32(
    a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
    bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
    acc = tl.dot(av, bv, input_precision="tf32")
    tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


@triton.jit
def _m1212_real_dot_transposed_b(
    a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
    bv = tl.load(b + n[None, :] * BLOCK_K + k[:, None])
    acc = tl.dot(av, bv, input_precision="tf32")
    tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


@triton.jit
def _m1212_real_dot_masked(
    a,
    b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(
        a + m[:, None] * K + k[None, :],
        mask=(m[:, None] < M) & (k[None, :] < K),
        other=0.0,
    )
    bv = tl.load(
        b + k[:, None] * N + n[None, :],
        mask=(k[:, None] < K) & (n[None, :] < N),
        other=0.0,
    )
    acc = tl.dot(av, bv, input_precision="tf32")
    tl.store(
        out + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _m1212_real_dot_epilogue(
    a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
    bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
    acc = tl.dot(av, bv, input_precision="tf32") + 1.0
    tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


@triton.jit
def _m13_generated_dot_bridge_f32(
    a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
    bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
    acc = tl.dot(av, bv, input_precision="tf32")
    tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


def _signature(dtype="*fp16"):
    return {
        "a": dtype,
        "b": dtype,
        "out": "*fp32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def _constexprs():
    return {"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 16}


@tvm.testing.requires_cuda
def test_m1212_real_tl_dot_bridge_lower_reader_semantics_and_contract():
    artifact = lower_to_ttir(_m1212_real_dot, _signature(), _constexprs())
    assert "tt.dot" in artifact.ttir

    graph = TTIRReader().read(artifact.ttir)
    dots = [op for op in graph.ops if op.name == "tt.dot"]
    assert len(dots) == 1
    dot = dots[0]
    assert len(dot.operands) == 3
    assert len(dot.results) == 1
    assert dot.attrs["inputPrecision"] == "tf32"
    assert [(ty.dtype, ty.shape) for ty in dot.result_types] == [("float32", (8, 8))]

    semantics = extract_matmul_semantics_from_ttir(
        graph, source_kind=REAL_JIT_TT_DOT_SOURCE_KIND
    )
    assert semantics.source_kind == REAL_JIT_TT_DOT_SOURCE_KIND
    assert (semantics.m, semantics.n, semantics.k) == (8, 8, 16)
    assert semantics.a_dtype == "float16"
    assert semantics.b_dtype == "float16"
    assert semantics.a_layout == "row_major"
    assert semantics.b_layout == "row_major"
    assert semantics.c_layout == "row_major"

    irmod, meta = translate_ttir(artifact, grid=(1,), contract="matmul_minimal")
    validate_matmul_minimal_contract(irmod)
    assert meta.matmul_source_kind == REAL_JIT_TT_DOT_SOURCE_KIND
    assert meta.matmul_contract_ok is True
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.unsupported_matmul_reason == ""


@tvm.testing.requires_cuda
def test_m13_generated_tl_dot_bridge_admits_fp32_without_real_jit_credit():
    artifact = lower_to_ttir(
        _m13_generated_dot_bridge_f32,
        _signature("*fp32"),
        {"BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 64},
    )
    graph = TTIRReader().read(artifact.ttir)
    semantics = extract_matmul_semantics_from_ttir(
        graph, source_kind=GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
    )
    assert semantics.source_kind == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
    assert (semantics.m, semantics.n, semantics.k) == (8, 64, 64)
    assert semantics.a_dtype == "float32"
    assert semantics.b_dtype == "float32"

    irmod, meta = translate_ttir(
        artifact,
        grid=(1,),
        contract="matmul_minimal",
        matmul_source_kind=GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
    )
    validate_matmul_minimal_contract(irmod)
    decision = TargetMatmulPolicy().decide(semantics, matmul_contract_ok=True)
    assert meta.matmul_source_kind == GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND
    assert meta.matmul_source_kind != REAL_JIT_TT_DOT_SOURCE_KIND
    assert meta.matmul_contract_ok is True
    assert decision.implementation_kind == "native_tir_schedule"
    assert decision.unsupported_matmul_reason == ""


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    "case_name,jit_fn,signature,constexprs,expected_reason",
    [
        (
            "unsupported_dtype",
            _m1212_real_dot_f32,
            _signature("*fp32"),
            _constexprs(),
            NEGATIVE_CASES["unsupported_dtype"],
        ),
        (
            "unsupported_layout",
            _m1212_real_dot_transposed_b,
            _signature(),
            _constexprs(),
            NEGATIVE_CASES["unsupported_layout"],
        ),
        (
            "masked_bounds",
            _m1212_real_dot_masked,
            {**_signature(), "M": "constexpr", "N": "constexpr", "K": "constexpr"},
            {"M": 7, "N": 7, "K": 15, **_constexprs()},
            NEGATIVE_CASES["masked_bounds"],
        ),
        (
            "unsupported_epilogue",
            _m1212_real_dot_epilogue,
            _signature(),
            _constexprs(),
            NEGATIVE_CASES["unsupported_epilogue"],
        ),
    ],
)
def test_m1212_real_tl_dot_negative_cases_are_explicit(
    case_name, jit_fn, signature, constexprs, expected_reason
):
    artifact = lower_to_ttir(jit_fn, signature, constexprs)
    assert "tt.dot" in artifact.ttir

    observed_reason = ""
    try:
        _, meta = translate_ttir(artifact, grid=(1,), contract="matmul_minimal")
        observed_reason = meta.unsupported_matmul_reason
    except Exception as err:  # pylint: disable=broad-except
        message = str(err)
        if expected_reason in message:
            observed_reason = expected_reason
        else:
            raise

    assert case_name in NEGATIVE_CASES
    assert observed_reason == expected_reason
