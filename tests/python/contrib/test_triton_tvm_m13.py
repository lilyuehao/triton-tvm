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
"""M13 strict TVM-owned operator path report coverage."""

import json

import tvm

from tvm.contrib.triton_tvm.contracts import validate_pointwise_flat_contract
from tvm.contrib.triton_tvm.m13_strict_tvm_owned_operator_path import (
    GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
    M13_4_BIAS_EPILOGUE_IMPLEMENTATION,
    M13_4_RUNTIME_KIND,
    M13_5_CONV_RUNTIME_KIND,
    M13_5_CONV_SHAPE_SCOPE,
    M13_5_CONV_SCHEDULE_ID,
    M13_6_ATTENTION_PROVIDER_KIND,
    M13_6_ATTENTION_RUNTIME_KIND,
    M13_7_ALLCLOSE_ATOL,
    M13_7_ALLCLOSE_RTOL,
    M13_8_STRICT_MODE,
    M13_8_TORCH_COMPILE_MODE,
    M13_P1_REPORT_ID,
    M13_P2_DIRECT_RUNTIME_KIND,
    M13_P2_DIRECT_SCHEDULE_ID,
    _build_m13_4_a_copy_tirx_source,
    _build_m13_4_a_zero_tail_tirx_source,
    _build_m13_4_b_adapter_tirx_source,
    _build_m13_4_bias_commit_tirx_source,
    _build_m13_4_matmul_addmm_runtime_replacement_report,
    _build_m13_4_output_commit_tirx_source,
    _build_m13_5_native_conv2d_tirx_source,
    _build_m13_5_native_conv_slice_report,
    _build_m13_6_native_attention_slice_report,
    _build_m13_7_strict_correctness_gate_report,
    _build_m13_8_p2_dashboard_decision_report,
    _build_m13_p1_strict_surface_profile_report,
    _build_m13_p2_direct_io_matmul_tirx_source,
    _m13_4_b_storage_shape,
    _m13_4_b_storage_stride,
)
from tvm.contrib.triton_tvm.matmul import (
    extract_matmul_semantics_from_wrapper_extern,
    extract_matmul_semantics_from_wrapper_extern_addmm,
)
from tvm.contrib.triton_tvm.vision import (
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_IMPLEMENTATION_KIND_NATIVE_TVM_CONV2D,
    VISION_OP_FAMILY_CONVOLUTION,
    VISION_PROVIDER_NATIVE_TVM_CONV2D,
    VISION_SOURCE_KIND_WRAPPER_CONV2D,
    VisionConv2DSemantics,
)
from tvm.contrib.triton_tvm.contracts import validate_vision_conv2d_contract


ADDMM_SOURCE = (
    "extern_kernels.addmm(arg12_1, reinterpret_tensor(buf7, (5, 64), (64, 1), 0), "
    "reinterpret_tensor(arg11_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf8)"
)
GEMM_SOURCE = (
    "extern_kernels.mm(reinterpret_tensor(buf28, (1, 64), (64, 1), 0), "
    "reinterpret_tensor(arg23_1, (64, 64), (1, 64), 0), out=buf29)"
)


def test_m13_4_native_pointwise_adapters_validate_contract():
    addmm = extract_matmul_semantics_from_wrapper_extern_addmm(
        {
            "op_name": "extern_kernels.addmm",
            "source": ADDMM_SOURCE,
            "case_name": "vit_tiny_random",
            "kernel_name": "m13_4_test_addmm",
        }
    )
    gemm = extract_matmul_semantics_from_wrapper_extern(
        {
            "op_name": "extern_kernels.mm",
            "source": GEMM_SOURCE,
            "case_name": "vit_tiny_random",
            "kernel_name": "m13_4_test_gemm",
        }
    )

    sources = [
        _build_m13_4_a_copy_tirx_source(0, addmm),
        _build_m13_4_a_zero_tail_tirx_source(0, addmm, 8),
        _build_m13_4_b_adapter_tirx_source(0, addmm),
        _build_m13_4_bias_commit_tirx_source(0, addmm, 8),
        _build_m13_4_output_commit_tirx_source(6, gemm, 8),
    ]
    for source in sources:
        validate_pointwise_flat_contract(tvm.script.from_source(source))


def test_m13_4_report_invariants(tmp_path):
    records = [_fake_m13_4_record(i, "extern_addmm_bias", bias=True) for i in range(3)]
    records.extend(_fake_m13_4_record(i, "extern_gemm", bias=False) for i in range(3, 7))

    report = _build_m13_4_matmul_addmm_runtime_replacement_report(
        records=records,
        wrapper_source="",
        correctness={"allclose": True},
        seed=0,
        dependency_versions={},
        out_dir=tmp_path,
    )

    assert report["status"] == "passed"
    assert report["summary"]["generated_real_tl_dot_bridge_runtime_count"] == 7
    assert report["summary"]["bias_epilogue_native_artifact_count"] == 3
    assert report["summary"]["native_tvm_matmul_legacy_wrapper_provider_count"] == 0
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["invariants"]["status"] == "passed"


def test_m13_5_native_conv_artifact_validates_contract():
    semantics = _fake_m13_5_conv_semantics()
    irmod = tvm.script.from_source(_build_m13_5_native_conv2d_tirx_source(semantics))

    validate_vision_conv2d_contract(irmod)
    script = irmod.script()
    assert VISION_IMPLEMENTATION_KIND_NATIVE_TVM_CONV2D in script
    assert M13_5_CONV_SHAPE_SCOPE in script
    assert M13_5_CONV_SCHEDULE_ID in script
    assert "call_packed" not in script


def test_m13_5_report_invariants(tmp_path):
    report = _build_m13_5_native_conv_slice_report(
        records=[_fake_m13_5_conv_record()],
        wrapper_source="",
        conv_correctness={"allclose": True, "max_abs_error": 0.0},
        e2e_correctness={"allclose": True},
        seed=0,
        dependency_versions={},
        out_dir=tmp_path,
    )

    assert report["status"] == "passed"
    assert report["summary"]["native_conv_artifact_correctness"] is True
    assert report["summary"]["device_torch_cuda_conv_count"] == 0
    assert report["summary"]["host_staging_bytes"] == 0
    assert report["summary"]["performance_claim"] == "diagnostic_only"
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["shape_scope"] == M13_5_CONV_SHAPE_SCOPE


def test_m13_6_report_uses_unambiguous_attention_provider_kind(tmp_path):
    report = _build_m13_6_native_attention_slice_report(
        records=[_fake_m13_6_attention_record()],
        wrapper_source="",
        attention_correctness={"allclose": True, "max_abs_error": 0.0},
        e2e_correctness={"allclose": True},
        seed=0,
        dependency_versions={},
        out_dir=tmp_path,
    )

    assert report["status"] == "passed"
    assert report["implementation_level"] == "tvm_decomposed_artifact"
    assert report["provider_kind"] == M13_6_ATTENTION_PROVIDER_KIND
    assert report["provider_kind"] == "native_tvm_attention_decomposed"
    assert report["torch_replay_count"] == 0
    assert report["summary"]["torch_replay_count"] == 0
    assert report["native_attention_records"][0]["provider_kind"] == (
        M13_6_ATTENTION_PROVIDER_KIND
    )
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["provider_kind"] == "native_tvm_attention_decomposed"
    assert written["provider_kind"] != "native_decomposed"
    assert "native_decomposed" not in json.dumps(written)


def test_m13_7_strict_correctness_gate_report_invariants(tmp_path):
    captured = [_fake_m13_7_captured_record(i) for i in range(7)]
    matmul = [_fake_m13_4_record(i, "extern_addmm_bias", bias=True) for i in range(3)]
    matmul.extend(_fake_m13_4_record(i, "extern_gemm", bias=False) for i in range(3, 7))
    report = _build_m13_7_strict_correctness_gate_report(
        captured_records=captured,
        matmul_records=matmul,
        conv_records=[_fake_m13_5_conv_record()],
        attention_records=[_fake_m13_6_attention_record()],
        wrapper_source="",
        correctness={
            "allclose": True,
            "allclose_rtol": M13_7_ALLCLOSE_RTOL,
            "allclose_atol": M13_7_ALLCLOSE_ATOL,
            "outputs": {},
        },
        seed=0,
        dependency_versions={},
        out_dir=tmp_path,
    )

    assert report["status"] == "passed"
    assert report["m13_c_status"] == "passed"
    assert report["m13_p2_status"] == "not_run"
    assert report["m13_completion_status"] == "not_complete_m13_p2_not_run"
    assert report["summary"]["captured_tvm"] == 7
    assert report["summary"]["generated_bridge_matmul_addmm"] == 7
    assert report["summary"]["native_conv"] == 1
    assert report["summary"]["native_attention"] == 1
    assert report["summary"]["legacy_provider"] == 0
    assert report["summary"]["operator_counts_match_inventory"] is True
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["m13_gate_summary"]["m13_c_status"] == "passed"


def test_m13_8_p2_dashboard_records_completion_decision(tmp_path):
    report = _build_m13_8_p2_dashboard_decision_report(
        latency_ms={
            M13_8_TORCH_COMPILE_MODE: {
                "samples_ms": [1.0],
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "min_ms": 1.0,
                "max_ms": 1.0,
            },
            M13_8_STRICT_MODE: {
                "samples_ms": [1.5],
                "p50_ms": 1.5,
                "p95_ms": 1.5,
                "min_ms": 1.5,
                "max_ms": 1.5,
            },
        },
        correctness={"allclose": True},
        strict_counts={
            "host_staging_bytes": 0,
            "legacy_provider_count": 0,
            "silent_fallback_count": 0,
        },
        seed=0,
        warmup=1,
        repeat=1,
        dependency_versions={},
        out_dir=tmp_path / "m13_8_p2_dashboard_decision",
        m13_c_status="passed",
    )

    assert report["status"] == "passed"
    assert report["m13_p2_status"] == "passed"
    assert report["m13_completion_status"] == "complete"
    assert report["p2"]["passed"] is True
    assert report["p2"]["provider_relaxation_allowed"] is False
    assert report["p2"]["fused_qkv_credit_allowed"] is False
    written = json.loads(
        (tmp_path / "m13_8_p2_dashboard_decision" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["summary"]["m13_complete"] is True


def test_m13_p1_report_records_profile_loop_decision(tmp_path):
    report = _build_m13_p1_strict_surface_profile_report(
        entry_report={
            "report_id": "m13_8_p2_dashboard_decision_v1",
            "m13_c_status": "passed",
            "m13_p2_status": "failed",
            "ratios": {
                "strict_to_torch_compile_p50": 4.0,
                "strict_to_torch_compile_p95": 4.5,
            },
        },
        profiler_evidence={
            "record_count": 2,
            "bottleneck_ranking": [
                {
                    "bucket_id": "native_attention",
                    "launch_count": 1,
                    "total_ms": 1.0,
                    "p50_ms": 1.0,
                    "p95_ms": 1.0,
                    "max_ms": 1.0,
                }
            ],
        },
        p2_dashboard={
            "m13_p2_status": "failed",
            "p2": {"passed": False},
        },
        correctness={"allclose": True},
        strict_counts={
            "captured_tvm": 7,
            "generated_bridge_matmul_addmm": 7,
            "native_conv": 1,
            "native_attention": 1,
            "host_staging_bytes": 0,
            "legacy_provider_count": 0,
            "silent_fallback_count": 0,
            "provider_relaxation_count": 0,
            "fused_qkv_credit_count": 0,
        },
        seed=0,
        warmup=1,
        repeat=1,
        dependency_versions={},
        out_dir=tmp_path / "p1_strict_surface_profile",
    )

    assert report["report_id"] == M13_P1_REPORT_ID
    assert report["status"] == "passed"
    assert report["optimization_delta"]["provider_surface_changed"] is False
    assert report["next_entry_decision"]["decision"] == "continue_m13_p"
    assert report["next_entry_decision"]["next_loop_id"] == "M13.P2"
    written = json.loads(
        (tmp_path / "p1_strict_surface_profile" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["strict_surface_invariants"]["legacy_provider_count"] == 0


def test_m13_p2_direct_io_matmul_artifact_is_single_kernel():
    addmm = extract_matmul_semantics_from_wrapper_extern_addmm(
        {
            "op_name": "extern_kernels.addmm",
            "source": ADDMM_SOURCE,
            "case_name": "vit_tiny_random",
            "kernel_name": "m13_p2_test_addmm",
        }
    )
    source = _build_m13_p2_direct_io_matmul_tirx_source(
        0,
        addmm,
        b_storage_shape=_m13_4_b_storage_shape(addmm),
        b_storage_stride=_m13_4_b_storage_stride(addmm),
    )
    script = tvm.script.from_source(source).script()

    assert M13_P2_DIRECT_RUNTIME_KIND in script
    assert M13_P2_DIRECT_SCHEDULE_ID in script
    assert "call_packed" not in script
    assert "m13_p2_direct_io_matmul" in script
    assert "bias[vn]" in script


def _fake_m13_4_record(index, op_family, *, bias):
    return {
        "index": index,
        "status": "passed",
        "op_family": op_family,
        "matmul_core": GENERATED_REAL_TL_DOT_BRIDGE_SOURCE_KIND,
        "runtime_kind": M13_4_RUNTIME_KIND,
        "runtime_replacement_connected": True,
        "epilogue_kind": "bias_add" if bias else "none",
        "epilogue_implementation": M13_4_BIAS_EPILOGUE_IMPLEMENTATION if bias else "none",
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
    }


def _fake_m13_7_captured_record(index):
    return {
        "index": index,
        "status": "passed",
        "kernel_name": f"triton_fake_{index}",
        "implementation_level": "triton_language_lowering",
        "runtime_kind": "tvm_artifact_run",
        "run_count": 1,
        "harness_fallback_count": 0,
        "native_triton_launch_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "pytorch_fallback_count": 0,
    }


def _fake_m13_5_conv_semantics():
    return VisionConv2DSemantics(
        source_kind=VISION_SOURCE_KIND_WRAPPER_CONV2D,
        source_name="extern_kernels.convolution",
        kernel_name="m13_5_native_conv2d_patch_embedding",
        vision_contract=VISION_CONTRACT_CONV2D_NCHW_STATIC,
        vision_op_family=VISION_OP_FAMILY_CONVOLUTION,
        input_param="buf0",
        weight_param="buf1",
        output_param="buf2",
        input_shape=(1, 3, 32, 32),
        weight_shape=(64, 3, 16, 16),
        output_shape=(1, 64, 2, 2),
        input_stride=(3072, 1, 96, 3),
        weight_stride=(768, 1, 48, 3),
        output_stride=(256, 1, 128, 64),
        stride=(16, 16),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        bias_policy="none",
        transposed=False,
        output_padding=(0, 0),
    )


def _fake_m13_5_conv_record():
    return {
        "index": 0,
        "status": "passed",
        "op_family": "wrapper_conv",
        "contract": VISION_CONTRACT_CONV2D_NCHW_STATIC,
        "shape_scope": M13_5_CONV_SHAPE_SCOPE,
        "implementation_level": "tvm_native_wrapper_lowering",
        "runtime_kind": M13_5_CONV_RUNTIME_KIND,
        "provider_kind": VISION_PROVIDER_NATIVE_TVM_CONV2D,
        "device_torch_cuda_conv_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "performance_claim": "diagnostic_only",
    }


def _fake_m13_6_attention_record():
    return {
        "index": 0,
        "status": "passed",
        "op_family": "wrapper_attention",
        "contract": "attention_vit_full_v1",
        "shape_scope": "exact_vit_tiny_random_full_attention_m13",
        "implementation_level": "tvm_decomposed_artifact",
        "runtime_kind": M13_6_ATTENTION_RUNTIME_KIND,
        "provider_kind": M13_6_ATTENTION_PROVIDER_KIND,
        "torch_replay_count": 0,
        "host_staging_bytes": 0,
        "silent_fallback_count": 0,
        "performance_claim": "diagnostic_only",
    }
