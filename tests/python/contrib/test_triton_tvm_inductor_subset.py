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
"""M3 TorchInductor pointwise TTIR audit tests."""

import json

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import build_triton_tvm, translate_ttir
from tvm.contrib.triton_tvm.inductor import (
    _builtin_cases,
    _load_builtin_corpus_module,
    audit_inductor_ttir,
    build_audit_report,
    extract_inductor_triton_sources,
    is_inductor_pointwise_kernel,
    load_inductor_kernel,
    lower_inductor_kernel_to_ttir,
    render_audit_markdown,
    write_audit_report,
)

try:
    import torch
except ImportError:
    torch = None


_WRAPPER_SOURCE = """
from torch._inductor.async_compile import AsyncCompile
async_compile = AsyncCompile()
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_heuristics

@triton_heuristics.pointwise(
    size_hints={'x': 64},
    filename=__file__,
    triton_meta={
        'signature': {
            'in_ptr0': '*fp32',
            'in_ptr1': '*fp32',
            'out_ptr0': '*fp32',
            'xnumel': 'i32',
            'XBLOCK': 'constexpr',
        },
        'constants': {},
        'configs': [],
    },
    inductor_meta={
        'grid_type': 'Grid1D',
        'num_reduction': 0,
        'atomic_add_found': False,
        'kernel_name': 'triton_poi_fused_add_0',
    },
)
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tmp1 = tl.load(in_ptr1 + xindex, xmask)
    tl.store(out_ptr0 + xindex, tmp0 + tmp1, xmask)
''', device_str='cuda')
"""


_UNSUPPORTED_MASKED_LOAD_TTIR = """
module {
  tt.func @_inductor_like(%x:!tt.ptr<f32>,%y:!tt.ptr<f32>,%out:!tt.ptr<f32>,%n:i32) {
    %c64_i32 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %offsets = arith.muli %pid, %c64_i32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s_offsets = tt.splat %offsets : i32 -> tensor<64xi32>
    %idx = arith.addi %s_offsets, %range : tensor<64xi32>
    %n_splat = tt.splat %n : i32 -> tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %n_splat : tensor<64xi32>
    %x_splat = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %x_ptr = tt.addptr %x_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vx = tt.load %x_ptr, %mask : tensor<64x!tt.ptr<f32>>
    %y_splat = tt.splat %y : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %y_ptr = tt.addptr %y_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %vy = tt.load %y_ptr, %mask : tensor<64x!tt.ptr<f32>>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_splat, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %sum = arith.addf %vx, %vy : tensor<64xf32>
    tt.store %out_ptr, %sum, %mask : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_m3_extracts_triton_source_from_inductor_wrapper():
    sources = extract_inductor_triton_sources(_WRAPPER_SOURCE, case_name="static_add")

    assert len(sources) == 1
    assert sources[0].case_name == "static_add"
    assert sources[0].kernel_name == "triton_poi_fused_add_0"
    assert sources[0].device_str == "cuda"
    assert "triton_heuristics.pointwise" in sources[0].source
    assert "@triton.jit" in sources[0].source


def test_m3_static_audit_report_for_pointwise_flat_masked_load(tmp_path):
    record = audit_inductor_ttir(
        _UNSUPPORTED_MASKED_LOAD_TTIR,
        case_name="static_masked_load",
        kernel_name="_inductor_like",
        signature={"x": "*fp32", "y": "*fp32", "out": "*fp32", "n": "i32"},
        constexprs={"XBLOCK": 64},
    )

    assert record["unique_ops"] == [
        "arith.addf",
        "arith.addi",
        "arith.cmpi",
        "arith.constant",
        "arith.muli",
        "tt.addptr",
        "tt.get_program_id",
        "tt.load",
        "tt.make_range",
        "tt.return",
        "tt.splat",
        "tt.store",
    ]
    assert record["load_count"] == 2
    assert record["store_count"] == 1
    assert record["translate_status"] == {
        "ok": True,
        "bucket": "translated",
        "fallback_reason": "",
    }

    report = build_audit_report([record])
    assert report["total_kernels"] == 1
    assert report["unsupported_buckets"] == {"translated": 1}
    markdown = render_audit_markdown(report)
    assert "M3 Inductor Pointwise TTIR Audit" in markdown
    assert "Pre-M5 Boundary" in markdown

    write_audit_report(report, tmp_path)
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert "cuda_" not in json.dumps(written)
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# M3")


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m3_cuda_collects_real_inductor_pointwise_ttir():
    import torch._dynamo
    import torch._inductor.config as inductor_config
    from torch._inductor.graph import GraphLowering

    def fn(x, y):
        return x + y * 2.0

    captured = []
    old_save_output_code = GraphLowering.save_output_code
    old_fx_graph_cache = inductor_config.fx_graph_cache
    GraphLowering.save_output_code = captured.append
    inductor_config.fx_graph_cache = False
    try:
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="inductor")
        x = torch.randn((128,), device="cuda")
        y = torch.randn((128,), device="cuda")
        compiled(x, y)
        torch.cuda.synchronize()
    finally:
        GraphLowering.save_output_code = old_save_output_code
        inductor_config.fx_graph_cache = old_fx_graph_cache
        torch._dynamo.reset()

    sources = [
        source
        for wrapper_source in captured
        for source in extract_inductor_triton_sources(wrapper_source, case_name="cuda_add_mul")
    ]
    assert sources

    kernel = load_inductor_kernel(sources[0])
    assert is_inductor_pointwise_kernel(kernel)
    artifact = lower_inductor_kernel_to_ttir(kernel)
    record = audit_inductor_ttir(
        artifact,
        case_name=kernel.case_name,
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        signature=kernel.signature,
        constexprs=kernel.constexprs,
    )

    assert "tt.load" in record["unique_ops"]
    assert "tt.store" in record["unique_ops"]
    assert record["translate_status"]["bucket"] in {
        "translated",
        "unsupported_ttir_op",
        "contract_error",
        "translate_other_error",
    }


_M35_CUDA_PROMOTED_CASES = [
    "add",
    "tuple_two_outputs",
    "relu_add",
    "where_cmp",
    "sigmoid_mul",
    "sin_cos",
    "broadcast_row",
    "broadcast_col",
    "strided_input",
    "slice_even",
    "fp16_add",
    "int32_add",
    "bool_mask",
]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
@pytest.mark.parametrize("case_name", _M35_CUDA_PROMOTED_CASES)
def test_m35_cuda_builds_and_runs_promoted_inductor_pointwise(case_name):
    source, torch_args, expected = _compile_builtin_case_to_source(case_name)
    kernel = load_inductor_kernel(source)
    assert is_inductor_pointwise_kernel(kernel)

    artifact = lower_inductor_kernel_to_ttir(kernel)
    irmod, meta = translate_ttir(
        artifact,
        grid=(1,),
        target="cuda",
        contract="pointwise_flat",
    )
    assert meta.contract == "pointwise_flat"

    built = build_triton_tvm(irmod, meta)
    tvm_args, outputs = _make_tvm_runtime_args(meta, torch_args, expected)
    built.run(tvm_args)

    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    for out_tvm, expected_torch in zip(outputs, expected_outputs, strict=True):
        expected_np = _torch_numpy(expected_torch).reshape(-1)
        actual_np = out_tvm.numpy()
        if expected_np.dtype == np.bool_ or np.issubdtype(expected_np.dtype, np.integer):
            np.testing.assert_array_equal(actual_np, expected_np)
        else:
            tvm.testing.assert_allclose(actual_np, expected_np, rtol=1e-3, atol=1e-3)


def _compile_builtin_case_to_source(case_name):
    import torch._dynamo
    import torch._inductor.config as inductor_config
    from torch._inductor.graph import GraphLowering

    builtin_module = _load_builtin_corpus_module()
    case = next(case for case in _builtin_cases(torch, builtin_module) if case.name == case_name)
    captured = []
    old_save_output_code = GraphLowering.save_output_code
    old_fx_graph_cache = inductor_config.fx_graph_cache
    GraphLowering.save_output_code = captured.append
    inductor_config.fx_graph_cache = False
    try:
        torch._dynamo.reset()
        args = case.make_args(torch)
        compiled = torch.compile(case.fn, backend="inductor")
        with torch.no_grad():
            expected = compiled(*args)
        torch.cuda.synchronize()
    finally:
        GraphLowering.save_output_code = old_save_output_code
        inductor_config.fx_graph_cache = old_fx_graph_cache
        torch._dynamo.reset()

    sources = [
        source
        for wrapper_source in captured
        for source in extract_inductor_triton_sources(wrapper_source, case_name=case_name)
    ]
    assert sources
    return sources[0], args, expected


def _make_tvm_runtime_args(meta, torch_args, expected):
    dev = tvm.cuda(0)
    input_arrays = [_torch_numpy(arg).reshape(-1) for arg in torch_args]
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    output_specs = iter(expected_outputs)
    tvm_args = []
    outputs = []
    for spec in meta.abi:
        name = spec["name"]
        if spec["kind"] == "pointer" and name.startswith("in_ptr"):
            index = int(name.removeprefix("in_ptr"))
            tvm_args.append(tvm.runtime.tensor(np.ascontiguousarray(input_arrays[index]), dev))
        elif spec["kind"] == "pointer" and name.startswith("out_ptr"):
            expected_torch = next(output_specs)
            expected_np = _torch_numpy(expected_torch).reshape(-1)
            out = tvm.runtime.empty(expected_np.shape, spec["dtype"], dev)
            tvm_args.append(out)
            outputs.append(out)
        elif spec["kind"] == "scalar":
            tvm_args.append(_runtime_extent_value(meta, expected_outputs))
        else:
            raise AssertionError(f"Unsupported ABI spec in M3.5 test: {spec}")
    return tvm_args, outputs


def _runtime_extent_value(meta, expected_outputs):
    if meta.extent_value is not None:
        return int(meta.extent_value)
    return int(_torch_numpy(expected_outputs[0]).size)


def _torch_numpy(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    tvm.testing.main()
