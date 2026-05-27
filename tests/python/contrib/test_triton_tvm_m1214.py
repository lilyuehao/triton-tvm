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
"""M12.14 real ``tt.dot`` schedule-handoff smoke coverage."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import lower_to_ttir, translate_ttir, validate_matmul_minimal_contract
from tvm.contrib.triton_tvm.matmul import (
    M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
    REAL_JIT_TT_DOT_SOURCE_KIND,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_SCHEDULE_ID,
)
from tvm.contrib.triton_tvm.runtime import build_triton_tvm

try:
    import triton
    import triton.language as tl
except ImportError:
    pytestmark = pytest.skip("Triton is not available", allow_module_level=True)


@triton.jit
def _m1214_real_dot(
    a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
    bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
    acc = tl.dot(av, bv, input_precision="tf32")
    tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)


def _signature():
    return {
        "a": "*fp16",
        "b": "*fp16",
        "out": "*fp32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def _lower_translate(block_m, block_n, block_k):
    artifact = lower_to_ttir(
        _m1214_real_dot,
        _signature(),
        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
    )
    assert "tt.dot" in artifact.ttir
    irmod, meta = translate_ttir(artifact, grid=(1,), contract="matmul_minimal")
    validate_matmul_minimal_contract(irmod)
    assert meta.matmul_source_kind == REAL_JIT_TT_DOT_SOURCE_KIND
    assert meta.implementation_kind == "native_tir_schedule"
    assert meta.schedule_id != M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
    return irmod, meta


@tvm.testing.requires_cuda
def test_m1214_real_tt_dot_tiled_schedule_handoff():
    _, meta = _lower_translate(8, 8, 16)

    assert meta.schedule_id == TILED_TIR_MATMUL_SCHEDULE_ID
    assert meta.selected_schedule_id == TILED_TIR_MATMUL_SCHEDULE_ID


@tvm.testing.requires_cuda_compute_version(7)
def test_m1214_real_tt_dot_tensorcore_schedule_handoff_build_run():
    m, n, k = 16, 16, 16
    irmod, meta = _lower_translate(m, n, k)

    assert meta.schedule_id == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert meta.selected_schedule_id == TENSORCORE_TIR_MATMUL_SCHEDULE_ID

    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1, 1, (m, k)).astype("float16")
    b_np = rng.uniform(-1, 1, (k, n)).astype("float16")
    expected = a_np.astype("float32") @ b_np.astype("float32")

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
