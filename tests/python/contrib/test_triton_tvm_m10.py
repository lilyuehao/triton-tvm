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
"""M10 attention runtime-entry correctness coverage."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import (
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    build_attention_sdpa_tirx_source,
    build_native_decomposed_attention_tirx_source,
    build_triton_tvm,
    register_python_torch_attention_sdpa,
    validate_attention_llama_causal_prefill_contract,
    validate_attention_vit_full_contract,
)
from tvm.contrib.triton_tvm.attention import (
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_EXTERN_PACKED_FUNC,
    ATTENTION_EXTERN_SYMBOL,
    ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
    ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY,
    ATTENTION_RUNTIME_STATUS_DEFERRED,
    ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    TargetAttentionPolicy,
    extract_attention_semantics_from_wrapper_sdpa,
)
from tvm.contrib.triton_tvm.m10_attention_baseline import run_attention_baseline
from tvm.contrib.triton_tvm.translator import TritonTVMMeta

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:
    torch = None
    torch_functional = None


def _sdpa_source(
    stride: tuple[int, int, int, int] = (320, 16, 64, 1),
    *,
    scale: float = 0.125,
) -> str:
    stride_text = ", ".join(str(value) for value in stride)
    return (
        f"{ATTENTION_EXTERN_SYMBOL}("
        "reinterpret_tensor(q, (1, 4, 5, 16), "
        f"({stride_text}), 0), "
        "reinterpret_tensor(k, (1, 4, 5, 16), "
        f"({stride_text}), 0), "
        "reinterpret_tensor(v, (1, 4, 5, 16), "
        f"({stride_text}), 0), "
        "None, False, scale=0.125)"
    ).replace("scale=0.125", f"scale={scale:g}")


def _llama_prefill_source(
    shape: tuple[int, int, int, int] = (1, 4, 16, 16),
    value_stride: tuple[int, int, int, int] = (1024, 16, 64, 1),
    mask_stride: tuple[int, int, int, int] = (256, 0, 16, 1),
    *,
    scale: float = 0.25,
) -> str:
    shape_text = ", ".join(str(value) for value in shape)
    value_stride_text = ", ".join(str(value) for value in value_stride)
    mask_stride_text = ", ".join(str(value) for value in mask_stride)
    return (
        f"{ATTENTION_EXTERN_SYMBOL}("
        "q, k, "
        f"reinterpret_tensor(v, ({shape_text}), ({value_stride_text}), 0), "
        f"reinterpret_tensor(mask, ({shape[0]}, {shape[1]}, {shape[2]}, {shape[2]}), "
        f"({mask_stride_text}), 0), "
        f"False, scale={scale:g})"
    )


def _attention_meta(
    kernel_name: str,
    *,
    contract: str = ATTENTION_CONTRACT_VIT_FULL,
    implementation_kind: str = "extern_attention_sdpa",
    execution_kind: str = "extern_attention_sdpa_runtime_proof",
    include_mask: bool = False,
) -> TritonTVMMeta:
    abi = [
        {"name": "q", "kind": "pointer", "dtype": "float32"},
        {"name": "k", "kind": "pointer", "dtype": "float32"},
        {"name": "v", "kind": "pointer", "dtype": "float32"},
    ]
    if include_mask:
        abi.append({"name": "mask", "kind": "pointer", "dtype": "float32"})
    abi.append({"name": "out", "kind": "pointer", "dtype": "float32"})
    contract_version = "attention_vit_full_m10_v1"
    mask_policy = "none_or_padding"
    if contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        contract_version = "attention_llama_causal_prefill_m10_v1"
        mask_policy = "causal_additive_mask"
    return TritonTVMMeta(
        kernel_name=kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract=contract,
        canonical_contract=contract,
        requested_contract=contract,
        emit="tir",
        translator_version="m103_attention_sdpa_runtime_entry",
        contract_version=contract_version,
        target_policy_version="m103_attention_sdpa_runtime_entry",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=5,
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=16,
        buffer_extents={
            "q": "T.int64(320)",
            "k": "T.int64(320)",
            "v": "T.int64(320)",
            "out": "T.int64(320)",
        },
        block_size=1,
        indexing_kind="rank4_sdpa",
        execution_kind=execution_kind,
        accumulator_dtype_policy="torch_sdpa_provider_default",
        epsilon_policy="not_applicable",
        mask_policy=mask_policy,
        axis_policy="batch_head_sequence_head_dim",
        layout_policy="rank4_static",
        launch_policy_id="extern_attention_sdpa_runtime_proof",
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"{kernel_name}_runtime_proof",
        implementation_kind=implementation_kind,
        extern_symbol=ATTENTION_EXTERN_SYMBOL,
    )


def test_m103_attention_semantics_and_artifact_metadata():
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _sdpa_source(),
        case_name="vit_tiny_random",
        kernel_name="m103_attention_sdpa_artifact",
    )
    decision = TargetAttentionPolicy().decide(semantics, attention_contract_ok=True)

    assert semantics.q_shape == (1, 4, 5, 16)
    assert semantics.q_stride == (320, 16, 64, 1)
    assert semantics.scale == 0.125
    assert semantics.causal is False
    assert decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY
    assert decision.attention_runtime_launch_count == 0
    assert decision.attention_artifact_call_count == 1
    assert decision.attention_qkv_bytes == 3840
    assert decision.attention_output_bytes == 1280
    assert decision.attention_total_io_bytes == 5120
    assert decision.attention_intermediate_buffer_bytes == 0
    assert decision.attention_host_staging_bytes == 0

    irmod = tvm.script.from_source(build_attention_sdpa_tirx_source(semantics, decision))
    validate_attention_vit_full_contract(irmod)
    attrs = next(iter(irmod.functions.values())).attrs
    script = irmod.script()

    assert str(attrs["triton_tvm.contract"]) == ATTENTION_CONTRACT_VIT_FULL
    assert str(attrs["triton_tvm.extern_packed_func"]) == ATTENTION_EXTERN_PACKED_FUNC
    assert str(attrs["triton_tvm.attention_runtime_status"]) == (
        ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY
    )
    assert int(attrs["triton_tvm.attention_runtime_launch_count"]) == 0
    assert int(attrs["triton_tvm.attention_artifact_call_count"]) == 1
    assert int(attrs["triton_tvm.attention_total_io_bytes"]) == 5120
    assert "call_packed" in script
    assert ATTENTION_EXTERN_PACKED_FUNC in script


def test_m10_attention_baseline_report_schema_no_benchmarks(tmp_path):
    report = run_attention_baseline(out_dir=tmp_path, run_benchmarks=False)

    assert report["report_kind"] == "triton_tvm_m10_attention_performance_baseline"
    assert report["baseline_performance_claim"] is True
    assert report["attention_runtime_provider_performance_claim"] is False
    assert report["summary"]["total_cases"] == 2
    assert report["summary"]["measured_cases"] == 0
    assert report["summary"]["unavailable_cases"] == 2
    assert {case["contract"] for case in report["cases"]} == {
        ATTENTION_CONTRACT_VIT_FULL,
        ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    }
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m103_attention_runtime_provider_build_run():
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _sdpa_source((320, 80, 16, 1)),
        case_name="vit_tiny_random",
        kernel_name="m103_attention_sdpa_provider",
    )
    decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, attention_contract_ok=True)
    assert decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED

    irmod = tvm.script.from_source(build_attention_sdpa_tirx_source(semantics, decision))
    validate_attention_vit_full_contract(irmod)
    built = build_triton_tvm(irmod, _attention_meta("m103_attention_sdpa_provider"))

    rng = np.random.default_rng(0)
    q_np = rng.normal(size=(1, 4, 5, 16)).astype("float32")
    k_np = rng.normal(size=(1, 4, 5, 16)).astype("float32")
    v_np = rng.normal(size=(1, 4, 5, 16)).astype("float32")
    expected = torch_functional.scaled_dot_product_attention(
        torch.from_numpy(q_np),
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=0.125,
    ).numpy()

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((1, 4, 5, 16), "float32", dev)
    with register_python_torch_attention_sdpa():
        built.run(
            [
                tvm.runtime.tensor(q_np, dev),
                tvm.runtime.tensor(k_np, dev),
                tvm.runtime.tensor(v_np, dev),
                out_tvm,
            ]
        )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_m104_native_decomposed_attention_metadata():
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _sdpa_source(scale=0.25),
        case_name="vit_tiny_random",
        kernel_name="m104_native_decomposed_attention",
    )
    decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    ).decide(semantics, attention_contract_ok=True)

    assert decision.implementation_kind == ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED
    assert decision.extern_runtime_kind == ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED
    assert decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert decision.attention_provider_kind == ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    assert decision.attention_performance_claim is False
    assert decision.attention_uses_host_staging is False
    assert decision.attention_runtime_launch_count == 1
    assert decision.attention_artifact_call_count == 1
    assert decision.attention_total_io_bytes == 5120
    assert decision.attention_intermediate_buffer_bytes == 15360
    assert decision.attention_total_accounted_bytes == 20480

    source = build_native_decomposed_attention_tirx_source(semantics, decision)
    irmod = tvm.script.from_source(source)
    validate_attention_vit_full_contract(irmod)
    attrs = next(iter(irmod.functions.values())).attrs
    script = irmod.script()

    assert str(attrs["triton_tvm.implementation_kind"]) == (
        ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED
    )
    assert str(attrs["triton_tvm.qk_matmul_boundary"]) == "matmul_minimal"
    assert str(attrs["triton_tvm.softmax_boundary"]) == "softmax_row"
    assert str(attrs["triton_tvm.av_matmul_boundary"]) == "matmul_minimal"
    assert int(attrs["triton_tvm.attention_runtime_launch_count"]) == 1
    assert int(attrs["triton_tvm.attention_total_accounted_bytes"]) == 20480
    assert "attention_qk_matmul" in script
    assert "attention_row_softmax" in script
    assert "attention_av_matmul" in script
    assert "call_packed" not in script


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m104_native_decomposed_attention_build_run_contiguous_vit():
    stride = (320, 80, 16, 1)
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _sdpa_source(stride, scale=0.25),
        case_name="vit_tiny_random",
        kernel_name="m104_native_decomposed_attention",
    )
    decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    ).decide(semantics, attention_contract_ok=True)

    irmod = tvm.script.from_source(build_native_decomposed_attention_tirx_source(semantics, decision))
    validate_attention_vit_full_contract(irmod)
    built = build_triton_tvm(
        irmod,
        _attention_meta(
            "m104_native_decomposed_attention",
            implementation_kind=ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
            execution_kind="native_decomposed_attention_qk_softmax_av",
        ),
    )

    rng = np.random.default_rng(1)
    q_logical = rng.normal(size=(1, 4, 5, 16)).astype("float32")
    k_logical = rng.normal(size=(1, 4, 5, 16)).astype("float32")
    v_logical = rng.normal(size=(1, 4, 5, 16)).astype("float32")
    expected = torch_functional.scaled_dot_product_attention(
        torch.from_numpy(q_logical),
        torch.from_numpy(k_logical),
        torch.from_numpy(v_logical),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=0.25,
    ).numpy()

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty((1, 4, 5, 16), "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(q_logical, dev),
            tvm.runtime.tensor(k_logical, dev),
            tvm.runtime.tensor(v_logical, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_m105_llama_prefill_semantics_and_native_metadata():
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _llama_prefill_source(),
        case_name="llama_tiny_random",
        kernel_name="m105_llama_prefill_native_decomposed_attention",
    )

    assert semantics.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
    assert semantics.phase == "causal_prefill"
    assert semantics.causal is True
    assert semantics.mask_kind == "causal"
    assert semantics.q_shape == (1, 4, 16, 16)
    assert semantics.q_stride == (1024, 16, 64, 1)
    assert semantics.mask_param == "mask"
    assert semantics.mask_shape == (1, 4, 16, 16)
    assert semantics.mask_stride == (256, 0, 16, 1)
    assert semantics.scale == 0.25

    default_decision = TargetAttentionPolicy().decide(
        semantics,
        attention_contract_ok=True,
    )
    assert default_decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED
    assert default_decision.unsupported_attention_runtime_reason == (
        "attention_llama_prefill_native_decomposed_provider_required_m10_5"
    )
    assert default_decision.attention_runtime_launch_count == 0
    assert default_decision.attention_total_io_bytes == 20480

    host_staged_decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, attention_contract_ok=True)
    assert host_staged_decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED

    native_decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    ).decide(semantics, attention_contract_ok=True)
    assert native_decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert native_decision.attention_provider_kind == ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    assert native_decision.attention_performance_claim is False
    assert native_decision.attention_runtime_launch_count == 1
    assert native_decision.attention_mask_bytes == 4096
    assert native_decision.attention_total_io_bytes == 20480
    assert native_decision.attention_intermediate_buffer_bytes == 139264
    assert native_decision.attention_total_accounted_bytes == 159744

    source = build_native_decomposed_attention_tirx_source(semantics, native_decision)
    irmod = tvm.script.from_source(source)
    validate_attention_llama_causal_prefill_contract(irmod)
    attrs = next(iter(irmod.functions.values())).attrs
    script = irmod.script()

    assert str(attrs["triton_tvm.contract"]) == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
    assert str(attrs["triton_tvm.softmax_boundary"]) == "masked_softmax_row"
    assert str(attrs["triton_tvm.mask_shape"]) == "1, 4, 16, 16"
    assert int(attrs["triton_tvm.attention_total_io_bytes"]) == 20480
    assert int(attrs["triton_tvm.attention_intermediate_buffer_bytes"]) == 139264
    assert "attention_mask_add" in script
    assert "call_packed" not in script


def test_m10_hardening_rejects_unknown_attention_provider():
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _sdpa_source(),
        case_name="vit_tiny_random",
        kernel_name="m10_hardening_unknown_provider",
    )
    decision = TargetAttentionPolicy(
        attention_runtime_provider="python_torch_host_stagged"
    ).decide(semantics, attention_contract_ok=True)

    assert decision.attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED
    assert decision.implementation_kind == "unsupported"
    assert decision.unsupported_attention_runtime_reason == (
        "attention_runtime_provider_unknown_m10_hardening"
    )
    assert decision.attention_provider_kind == "python_torch_host_stagged"
    assert decision.attention_runtime_launch_count == 0
    assert decision.attention_total_io_bytes == 5120


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m105_native_decomposed_attention_build_run_llama_prefill():
    shape = (1, 2, 4, 8)
    stride = (64, 32, 8, 1)
    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _llama_prefill_source(
            shape,
            value_stride=stride,
            mask_stride=(32, 16, 4, 1),
            scale=0.25,
        ),
        case_name="llama_tiny_random",
        kernel_name="m105_llama_prefill_native_decomposed_attention",
    )
    decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    ).decide(semantics, attention_contract_ok=True)

    irmod = tvm.script.from_source(build_native_decomposed_attention_tirx_source(semantics, decision))
    validate_attention_llama_causal_prefill_contract(irmod)
    built = build_triton_tvm(
        irmod,
        _attention_meta(
            "m105_llama_prefill_native_decomposed_attention",
            contract=ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
            implementation_kind=ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
            execution_kind="native_decomposed_attention_qk_masked_softmax_av",
            include_mask=True,
        ),
    )

    rng = np.random.default_rng(2)
    q_np = rng.normal(size=shape).astype("float32")
    k_np = rng.normal(size=shape).astype("float32")
    v_np = rng.normal(size=shape).astype("float32")
    causal = np.triu(np.ones((shape[2], shape[2]), dtype=bool), k=1)
    mask_np = np.where(causal, -1.0e9, 0.0).astype("float32")
    mask_np = np.broadcast_to(mask_np, (shape[0], shape[1], shape[2], shape[2])).copy()
    expected = torch_functional.scaled_dot_product_attention(
        torch.from_numpy(q_np),
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        attn_mask=torch.from_numpy(mask_np),
        dropout_p=0.0,
        is_causal=False,
        scale=0.25,
    ).numpy()

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty(shape, "float32", dev)
    built.run(
        [
            tvm.runtime.tensor(q_np, dev),
            tvm.runtime.tensor(k_np, dev),
            tvm.runtime.tensor(v_np, dev),
            tvm.runtime.tensor(mask_np, dev),
            out_tvm,
        ]
    )
    tvm.testing.assert_allclose(out_tvm.numpy(), expected, rtol=1e-5, atol=1e-5)
