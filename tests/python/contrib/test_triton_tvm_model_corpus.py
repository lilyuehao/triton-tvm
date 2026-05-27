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
"""M6/M6.5 external model corpus audit tests."""

import json
import os

import pytest

import tvm.contrib.triton_tvm as triton_tvm_pkg
import tvm.testing
from tvm.contrib.triton_tvm.attention import (
    ATTENTION_ABI_VERSION,
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_LLAMA_DECODE,
    ATTENTION_CONTRACT_UNCLASSIFIED,
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    ATTENTION_SEMANTICS_STATUS_ACCEPTED,
    ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
    ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY,
    ATTENTION_RUNTIME_STATUS_DEFERRED,
    ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
)
from tvm.contrib.triton_tvm.model_corpus import (
    TritonTVMModelAuditConfig,
    _is_model_kernel_replaceable,
    _maybe_materialize_m11_grid2d_artifact,
    _select_model_kernel_contract,
    build_model_corpus_report,
    builtin_model_audit_cases,
    diff_capability_reports,
    run_model_corpus_audit,
    write_model_corpus_report,
)
from tvm.contrib.triton_tvm.m123_vit_e2e_runner import run_vit_fixed_shape_e2e_runner
from tvm.contrib.triton_tvm.m124_vit_e2e_dashboard import run_vit_e2e_dashboard
from tvm.contrib.triton_tvm.m125_vit_optimization_review import run_optimization_review
from tvm.contrib.triton_tvm.m1255_vit_perf_optimization_freeze import (
    run_perf_optimization_freeze,
)
from tvm.contrib.triton_tvm.m126_vit_e2e_hardening import run_vit_e2e_hardening
from tvm.contrib.triton_tvm.m127_vit_matmul_provider_cost import (
    build_vit_native_matmul_provider_cost_report,
)
from tvm.contrib.triton_tvm.m128_vit_p2_gate_decision import build_vit_p2_gate_decision_report
from tvm.contrib.triton_tvm.m129_vit_provider_runtime_optimization import (
    BACKEND_PASS_MODE,
    CURRENT_MODE,
    FUSED_QKV_MODE,
    build_vit_provider_runtime_optimization_report,
)
from tvm.contrib.triton_tvm.m1210_vit_route_cleanup import (
    build_vit_route_cleanup_report,
)
from tvm.contrib.triton_tvm.m1211_backend_general_replan import (
    build_backend_general_replan_report,
)
from tvm.contrib.triton_tvm.m1212_real_tl_dot_bridge import (
    NEGATIVE_CASES,
    build_real_tl_dot_bridge_report,
)
from tvm.contrib.triton_tvm.m1213_runtime_overhead_general_measurement import (
    REQUIRED_MEASUREMENT_BUCKETS,
    _invariants as _m1213_report_invariants,
    build_runtime_overhead_general_measurement_report,
)
from tvm.contrib.triton_tvm.m1214_real_tt_dot_schedule_handoff import (
    NEXT_DEFAULT_ACTION as M1214_NEXT_DEFAULT_ACTION,
    TENSORCORE_CASE,
    TILED_CASE,
    _invariants as _m1214_report_invariants,
    build_real_tt_dot_schedule_handoff_report,
)
from tvm.contrib.triton_tvm.m1215_vit_p2_reentry import (
    NEXT_ACTION_AFTER_P2_MISS,
    NEXT_ACTION_AFTER_P2_PASS,
    P2_ELIGIBLE_MODE,
    _invariants as _m1215_report_invariants,
    build_vit_p2_reentry_report,
)
from tvm.contrib.triton_tvm.m1216_backend_general_p2_gap_analysis import (
    NEXT_ACTION_AFTER_M1216,
    REQUIRED_ANALYSIS_BUCKETS,
    _invariants as _m1216_report_invariants,
    build_backend_general_p2_gap_analysis_report,
)
from tvm.contrib.triton_tvm.m12p_profile_optimization_loop import (
    ITERATION_ID as M12P_ITERATION_ID,
    MINIMAL_RECORD_ITERATION_ID as M12P_MINIMAL_RECORD_ITERATION_ID,
    M12P_MINIMAL_RECORD_MODE,
    M12P_PREBOUND_MODE,
    NEXT_ACTION_CONTINUE as M12P_NEXT_ACTION_CONTINUE,
    _invariants as _m12p_report_invariants,
    build_m12p_profile_optimization_loop_report,
)
from tvm.contrib.triton_tvm.m12p_residual_profile import (
    ITERATION_ID as M12P_RESIDUAL_PROFILE_ITERATION_ID,
    _invariants as _m12p_residual_profile_invariants,
    build_m12p_residual_profile_report,
)
from tvm.contrib.triton_tvm.m12p_provider_session_reuse import (
    ITERATION_ID as M12P_PROVIDER_SESSION_REUSE_ITERATION_ID,
    M12P_SESSION_REUSE_MODE,
    _invariants as _m12p_provider_session_reuse_invariants,
    build_m12p_provider_session_reuse_report,
)
from tvm.contrib.triton_tvm.inductor import InductorKernel, extract_inductor_wrapper_extern_calls
from tvm.contrib.triton_tvm.matmul import (
    EXTERN_ADDMM_BIAS_PACKED_FUNC,
    EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON,
    EXTERN_GEMM_PACKED_FUNC,
    EXTERN_GEMM_PROVIDER_ABI_VERSION,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_PROVIDER_UNKNOWN_REASON,
    EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
    EXTERN_GEMM_RUNTIME_KIND,
    EXTERN_GEMM_RUNTIME_REPLACEMENT,
    EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
    EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_FAILED,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
    REAL_JIT_TT_DOT_SOURCE_KIND,
    TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
    TILED_TIR_MATMUL_SCHEDULE_ID,
)
from tvm.contrib.triton_tvm.reporting import make_report_status, render_capability_markdown
from tvm.contrib.triton_tvm.vision import (
    M11_GRID2D_ARTIFACT_READY,
    M11_GRID2D_RUNTIME_READY,
    M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
    VISION_PROVIDER_DEVICE_TORCH_CUDA,
)

try:
    import torch
except ImportError:
    torch = None


def test_m6_model_corpus_api_is_experimental_not_top_level_public_api():
    assert "TritonTVMModelAuditConfig" not in triton_tvm_pkg.__all__
    assert "run_model_corpus_audit" not in triton_tvm_pkg.__all__


def test_m8_auto_contract_selection_keeps_deferred_grid_explicit():
    pointwise = _fake_inductor_kernel("triton_poi_fused_add_0")
    layernorm = _fake_inductor_kernel("triton_per_fused_native_layer_norm_0", num_reduction=2)
    rms = _fake_inductor_kernel("triton_per_fused_embedding_mean_mul_pow_rsqrt_0", num_reduction=1)
    pool = _fake_inductor_kernel("triton_per_fused_max_pool2d_with_indices_40", num_reduction=1)
    softmax = _fake_inductor_kernel("triton_per_fused__softmax_prepare_64", num_reduction=2)
    grid = _fake_inductor_kernel("triton_poi_fused_convolution_0", grid_type="Grid2D")

    assert _select_model_kernel_contract(pointwise, "auto_m8") == "pointwise_flat"
    assert _select_model_kernel_contract(layernorm, "auto_m8") == "norm_row"
    assert _select_model_kernel_contract(rms, "auto_m8") == "norm_row"
    assert _select_model_kernel_contract(pool, "auto_m8") == "row_reduction"
    assert _select_model_kernel_contract(softmax, "auto_m8") == "masked_softmax_row"

    assert _is_model_kernel_replaceable(pointwise, "pointwise_flat")
    assert _is_model_kernel_replaceable(layernorm, "norm_row")
    assert not _is_model_kernel_replaceable(grid, "pointwise_flat")


def test_m117_grid2d_ready_record_materializes_native_artifact():
    kernel = _fake_inductor_kernel(
        "triton_poi_fused_convolution_silu_0",
        grid_type="Grid2D",
    )
    record = _kernel_record(
        "yolo",
        "yolo_tiny",
        kernel.kernel_name,
        make_report_status(
            ok=False,
            bucket="contract_error",
            fallback_reason="unsupported_inductor_kernel",
        ),
        grid_type="Grid2D",
    )
    record["size_hints"] = {"x": 16, "y": 64}
    record["blocker_class"] = "grid"

    _maybe_materialize_m11_grid2d_artifact(record, kernel)

    assert record["translate_status"] == make_report_status(ok=True, bucket="translated")
    assert record["contract"] == VISION_CONTRACT_POINTWISE_GRID2D_STATIC
    assert record["m11_grid_family"] == M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE
    assert record["m11_grid_artifact_status"] == M11_GRID2D_ARTIFACT_READY
    assert record["m11_grid_runtime_status"] == M11_GRID2D_RUNTIME_READY
    assert record["m11_grid_native_artifact_status"] == "translated"
    assert record["m11_grid_native_artifact_op_kind"] == "silu"


def test_m117_grid2d_materialization_moves_ready_records_to_translated_bucket():
    ready_kernel = _fake_inductor_kernel(
        "triton_poi_fused_convolution_silu_0",
        grid_type="Grid2D",
    )
    concat_kernel = _fake_inductor_kernel(
        "triton_poi_fused_convolution_silu_split_6",
        grid_type="Grid2D",
    )
    ready_record = _kernel_record(
        "yolo",
        "yolo_tiny",
        ready_kernel.kernel_name,
        make_report_status(
            ok=False,
            bucket="contract_error",
            fallback_reason="unsupported_inductor_kernel",
        ),
        grid_type="Grid2D",
    )
    concat_record = _kernel_record(
        "yolo",
        "yolo_tiny",
        concat_kernel.kernel_name,
        make_report_status(
            ok=False,
            bucket="contract_error",
            fallback_reason="unsupported_inductor_kernel",
        ),
        grid_type="Grid2D",
    )
    for record, kernel in [(ready_record, ready_kernel), (concat_record, concat_kernel)]:
        record["size_hints"] = {"x": 16, "y": 64}
        record["blocker_class"] = "grid"
        _maybe_materialize_m11_grid2d_artifact(record, kernel)
        record["blocker_class"] = "none" if record["translate_status"]["ok"] else "grid"
        record["pre_m8_family"] = (
            "translated" if record["translate_status"]["ok"] else "deferred_grid"
        )

    report = build_model_corpus_report(
        [ready_record, concat_record],
        [_model_record("yolo", "yolo_tiny", kernel_count=2, translated=1, fallback=1)],
        generated_at="2026-05-26T00:00:00+00:00",
    )

    assert report["summary"]["status_buckets"] == {
        "contract_error": 1,
        "translated": 1,
    }
    assert report["summary"]["translated_kernels"] == 1
    assert report["blockers"][0]["fallback_reason"] == (
        "grid2d_concat_split_multi_output_layout_deferred_m11_6"
    )


def test_pre_m9_wrapper_extern_extraction_classifies_matmul_conv_attention():
    wrapper_source = """
def call(arg0, arg1):
    buf0 = extern_kernels.mm(reinterpret_tensor(buf_a, (16, 64), (64, 1), 0), reinterpret_tensor(weight, (64, 64), (1, 64), 0), out=buf_out)
    buf1 = extern_kernels.addmm(bias, reinterpret_tensor(buf_a, (16, 64), (64, 1), 0), reinterpret_tensor(weight, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf_bias)
    buf2 = extern_kernels.convolution(arg0, arg1, stride=(1, 1), padding=(0, 0))
    buf3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(
        arg0, arg1, arg1, None, False, scale=0.25
    )
    return buf0, buf1, buf2, buf3
"""

    calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="toy",
        wrapper_path="/tmp/toy.py",
    )

    assert [(call.op_name, call.op_family) for call in calls] == [
        ("extern_kernels.mm", "extern_gemm"),
        ("extern_kernels.addmm", "extern_addmm_bias"),
        ("extern_kernels.convolution", "deferred_convolution"),
        (
            "torch.ops.aten._scaled_dot_product_efficient_attention.default",
            "deferred_attention",
        ),
    ]
    assert calls[0].case_name == "toy"
    assert calls[0].wrapper_path == "/tmp/toy.py"
    assert calls[0].matmul_source_kind == "wrapper_extern_gemm"
    assert (calls[0].matmul_m, calls[0].matmul_n, calls[0].matmul_k) == (16, 64, 64)
    assert calls[0].matmul_contract == "matmul_minimal"
    assert calls[0].matmul_contract_ok is True
    assert calls[0].implementation_kind == "extern_gemm"
    assert calls[0].extern_symbol == "extern_kernels.mm"
    assert calls[0].extern_packed_func == EXTERN_GEMM_PACKED_FUNC
    assert calls[0].extern_runtime_kind == EXTERN_GEMM_RUNTIME_KIND
    assert calls[0].extern_runtime_replacement == EXTERN_GEMM_RUNTIME_REPLACEMENT
    assert calls[0].extern_runtime_replacement_available is False
    assert (
        calls[0].extern_runtime_replacement_reason
        == EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON
    )
    assert calls[0].extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
    assert calls[0].extern_gemm_provider_kind == EXTERN_GEMM_PROVIDER_NONE
    assert calls[0].extern_gemm_provider_abi_version == 0
    assert calls[0].extern_gemm_performance_claim is False
    assert calls[0].extern_gemm_uses_host_staging is False
    assert calls[0].matmul_epilogue_kind == "none"
    assert calls[0].matmul_b_layout == "transposed_weight_view"
    assert calls[0].matmul_b_stride == (1, 64)
    assert calls[1].matmul_source_kind == "wrapper_extern_addmm_bias"
    assert calls[1].matmul_epilogue_kind == "bias_add"
    assert calls[1].matmul_contract_ok is True
    assert calls[1].implementation_kind == "extern_addmm_bias"
    assert calls[1].extern_symbol == "extern_kernels.addmm"
    assert calls[1].extern_packed_func == EXTERN_ADDMM_BIAS_PACKED_FUNC
    assert calls[1].extern_runtime_replacement_reason == (
        EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON
    )
    assert calls[1].extern_gemm_performance_claim is False
    assert calls[1].unsupported_matmul_reason == ""
    assert calls[3].attention_contract == ATTENTION_CONTRACT_UNCLASSIFIED
    assert calls[3].attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED
    assert "scale=0.25" in calls[3].source


def test_m96_wrapper_extern_provider_marks_runtime_resolved():
    wrapper_source = """
def call(arg0, arg1):
    buf0 = extern_kernels.mm(reinterpret_tensor(buf_a, (16, 64), (64, 1), 0), reinterpret_tensor(weight, (64, 64), (1, 64), 0), out=buf_out)
    return buf0
"""

    calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="toy",
        wrapper_path="/tmp/toy.py",
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    )

    assert len(calls) == 1
    assert calls[0].extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert calls[0].extern_gemm_provider_kind == EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    assert calls[0].extern_gemm_provider_abi_version == EXTERN_GEMM_PROVIDER_ABI_VERSION
    assert calls[0].extern_gemm_runtime_claim == EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY
    assert calls[0].extern_gemm_performance_claim is False
    assert calls[0].extern_gemm_uses_host_staging is True


def test_m126_wrapper_extern_provider_rejects_unknown_provider_id():
    wrapper_source = """
def call(arg0, arg1):
    buf0 = extern_kernels.mm(reinterpret_tensor(buf_a, (16, 64), (64, 1), 0), reinterpret_tensor(weight, (64, 64), (1, 64), 0), out=buf_out)
    return buf0
"""

    calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="vit_tiny_random",
        wrapper_path="/tmp/vit.py",
        extern_gemm_runtime_provider="native_tvm_matmul_stale",
    )

    assert len(calls) == 1
    assert calls[0].implementation_kind == "unsupported"
    assert calls[0].extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_FAILED
    assert calls[0].extern_gemm_provider_kind == "native_tvm_matmul_stale"
    assert calls[0].extern_runtime_replacement_reason == (
        EXTERN_GEMM_RUNTIME_PROVIDER_UNKNOWN_REASON
    )
    assert calls[0].unsupported_matmul_reason == EXTERN_GEMM_RUNTIME_PROVIDER_UNKNOWN_REASON
    assert calls[0].extern_gemm_performance_claim is False
    assert calls[0].extern_gemm_uses_host_staging is False


def test_pre_m10_multiline_sdpa_extraction_preserves_source_and_classifies_vit():
    wrapper_source = """
def call(buf_q, buf_k, buf_v):
    out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
        reinterpret_tensor(buf_q, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_k, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_v, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        None,
        False,
        scale=0.125,
    )
    return out
"""

    calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="vit_tiny_random",
        wrapper_path="/tmp/vit.py",
    )

    assert len(calls) == 1
    assert calls[0].op_family == "deferred_attention"
    assert "\n" in calls[0].source
    assert "scale=0.125" in calls[0].source
    assert calls[0].attention_contract == ATTENTION_CONTRACT_VIT_FULL
    assert calls[0].attention_abi_version == ATTENTION_ABI_VERSION
    assert calls[0].attention_causal is False
    assert calls[0].attention_mask_kind == "none_or_padding"
    assert calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY
    assert calls[0].attention_semantics_status == "attention_semantics_accepted"
    assert calls[0].attention_q_shape == "1, 4, 5, 16"
    assert calls[0].attention_q_stride == "320, 16, 64, 1"
    assert calls[0].attention_scale == 0.125
    assert calls[0].attention_performance_claim is False
    assert calls[0].attention_runtime_launch_count == 0
    assert calls[0].attention_artifact_call_count == 1
    assert calls[0].attention_total_io_bytes == 5120
    assert calls[0].attention_total_accounted_bytes == 5120


def test_pre_m10_llama_prefill_and_decode_attention_classification():
    prefill_source = """
def call(q, k, v, mask):
    return torch.ops.aten._scaled_dot_product_efficient_attention.default(
        q,
        k,
        reinterpret_tensor(v, (1, 4, 16, 16), (1024, 16, 64, 1), 0),
        reinterpret_tensor(mask, (1, 4, 16, 16), (256, 0, 16, 1), 0),
        False,
        scale=0.25,
    )
"""
    decode_source = """
def call(q, k, v, past_key_values):
    return torch.ops.aten._scaled_dot_product_efficient_attention.default(
        q, k, v, past_key_values, True, scale=0.125
    )
"""

    prefill = extract_inductor_wrapper_extern_calls(
        prefill_source,
        case_name="llama_tiny_random",
        wrapper_path="/tmp/llama_prefill.py",
    )[0]
    decode = extract_inductor_wrapper_extern_calls(
        decode_source,
        case_name="llama_decode_single_token_cache",
        wrapper_path="/tmp/llama_decode.py",
    )[0]

    assert prefill.attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
    assert prefill.attention_phase == "causal_prefill"
    assert prefill.attention_causal is True
    assert prefill.attention_kv_cache_policy == "prefill_no_cache_update"
    assert prefill.attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED
    assert prefill.attention_semantics_status == "attention_semantics_accepted"
    assert prefill.attention_q_shape == "1, 4, 16, 16"
    assert prefill.attention_mask_shape == "1, 4, 16, 16"
    assert prefill.attention_mask_stride == "256, 0, 16, 1"
    assert prefill.attention_runtime_launch_count == 0
    assert prefill.attention_total_io_bytes == 20480
    assert prefill.unsupported_attention_runtime_reason == (
        "attention_llama_prefill_native_decomposed_provider_required_m10_5"
    )
    assert decode.attention_contract == ATTENTION_CONTRACT_LLAMA_DECODE
    assert decode.attention_phase == "decode"
    assert decode.attention_kv_cache_policy == "kv_cache_layout_deferred"
    assert decode.attention_sequence_policy == "single_token_decode"


def test_pre_m10_report_section_splits_attention_abi_without_changing_pre_m9():
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_0",
            make_report_status(ok=True, bucket="translated"),
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_llama_0",
            make_report_status(ok=True, bucket="translated"),
        ),
    ]
    sdpa_op = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
    models = [
        _model_record(
            "vit",
            "vit_tiny",
            kernel_count=1,
            translated=1,
            fallback=0,
            extern_calls=[
                _extern_call(
                    "vit",
                    "vit_tiny",
                    sdpa_op,
                    "deferred_attention",
                    source=(
                        f"{sdpa_op}("
                        "reinterpret_tensor(buf_q, (1, 4, 5, 16), (320, 16, 64, 1), 0), "
                        "reinterpret_tensor(buf_k, (1, 4, 5, 16), (320, 16, 64, 1), 0), "
                        "reinterpret_tensor(buf_v, (1, 4, 5, 16), (320, 16, 64, 1), 0), "
                        "None, False, scale=0.125)"
                    ),
                )
            ],
        ),
        _model_record(
            "llama",
            "llama_tiny",
            kernel_count=1,
            translated=1,
            fallback=0,
            extern_calls=[
                _extern_call(
                    "llama",
                    "llama_tiny",
                    sdpa_op,
                    "deferred_attention",
                    source=(
                        f"{sdpa_op}("
                        "q, k, "
                        "reinterpret_tensor(buf_v, (1, 4, 16, 16), "
                        "(1024, 16, 64, 1), 0), "
                        "reinterpret_tensor(buf_mask, (1, 4, 16, 16), "
                        "(256, 0, 16, 1), 0), "
                        "False, scale=0.25)"
                    ),
                )
            ],
        ),
    ]

    report = build_model_corpus_report(
        records,
        models,
        generated_at="2026-05-26T00:00:00+00:00",
    )

    pre_m9 = report["pre_m9"]
    assert pre_m9["extern_family_classes"]["deferred_attention"]["call_count"] == 2
    assert {entry["op_family"] for entry in pre_m9["deferred_debt"]} == {
        "deferred_attention"
    }
    assert report["model_summary"]["full_tvm_runnable_models"] == 0

    pre_m10 = report["pre_m10"]
    assert pre_m10["taxonomy_version"] == 1
    assert pre_m10["observed_attention_call_count"] == 2
    assert pre_m10["attention_runtime_deferred_count"] == 1
    assert pre_m10["contract_classes"][ATTENTION_CONTRACT_VIT_FULL]["call_count"] == 1
    assert pre_m10["contract_classes"][ATTENTION_CONTRACT_VIT_FULL]["runtime_status"] == {
        ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY: 1
    }
    assert (
        pre_m10["contract_classes"][ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL][
            "call_count"
        ]
        == 1
    )
    assert pre_m10["contract_classes"][ATTENTION_CONTRACT_LLAMA_DECODE]["call_count"] == 0
    assert len(pre_m10["runtime_deferred_debt"]) == 2

    m10 = report["m10"]
    assert m10["artifact_count"] == 1
    assert m10["artifact_only_count"] == 1
    assert m10["runtime_resolved_count"] == 0
    assert m10["hardening_status"] == "m10_runtime_hardened_v1"
    assert m10["runtime_launch_count"] == 0
    assert m10["artifact_call_count"] == 1
    assert m10["total_io_bytes"] == 25600
    assert m10["intermediate_buffer_bytes"] == 0
    assert m10["host_staging_bytes"] == 0
    assert m10["unsupported_runtime_reasons"] == {
        "attention_llama_prefill_native_decomposed_provider_required_m10_5": 1
    }
    assert m10["status_counts"] == {
        ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY: 1,
        ATTENTION_RUNTIME_STATUS_DEFERRED: 1,
    }

    markdown = render_capability_markdown(report, title="Pre-M10 Snapshot")
    assert "## Pre-M10 Gate" in markdown
    assert "## M10 Attention Runtime Entry" in markdown
    assert ATTENTION_CONTRACT_VIT_FULL in markdown
    assert ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL in markdown
    assert ATTENTION_CONTRACT_LLAMA_DECODE in markdown


def test_m103_attention_provider_marks_only_vit_runtime_resolved():
    sdpa_op = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
    wrapper_source = f"""
def call(buf_q, buf_k, buf_v):
    out0 = {sdpa_op}(
        reinterpret_tensor(buf_q, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_k, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_v, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        None,
        False,
        scale=0.125,
    )
    out1 = {sdpa_op}(buf_q, buf_k, buf_v, None, True, scale=0.125)
    return out0, out1
"""

    calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="vit_tiny_random",
        attention_runtime_provider=ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    )

    assert calls[0].attention_contract == ATTENTION_CONTRACT_VIT_FULL
    assert calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert calls[0].attention_provider_kind == ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    assert calls[0].attention_runtime_claim == "correctness_only"
    assert calls[0].attention_performance_claim is False
    assert calls[0].attention_uses_host_staging is True
    assert calls[0].attention_runtime_launch_count == 1
    assert calls[0].attention_host_staging_bytes == 5120
    assert calls[1].attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED


def test_m104_native_decomposed_provider_marks_only_vit_runtime_resolved():
    sdpa_op = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
    wrapper_source = f"""
def call(buf_q, buf_k, buf_v):
    out0 = {sdpa_op}(
        reinterpret_tensor(buf_q, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_k, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_v, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        None,
        False,
        scale=0.25,
    )
    out1 = {sdpa_op}(buf_q, buf_k, buf_v, None, True, scale=0.125)
    return out0, out1
"""

    calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="vit_tiny_random",
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    )

    assert calls[0].attention_contract == ATTENTION_CONTRACT_VIT_FULL
    assert calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert calls[0].attention_provider_kind == ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    assert calls[0].attention_runtime_kind == ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED
    assert calls[0].attention_runtime_claim == "correctness_only"
    assert calls[0].attention_performance_claim is False
    assert calls[0].attention_uses_host_staging is False
    assert calls[0].attention_runtime_launch_count == 1
    assert calls[0].attention_intermediate_buffer_bytes == 15360
    assert calls[1].attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED

    llama_calls = extract_inductor_wrapper_extern_calls(
        f"def call(q, k, v):\n    return {sdpa_op}(q, k, v, None, True, scale=0.125)\n",
        case_name="llama_tiny_random",
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    )
    assert llama_calls[0].attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
    assert llama_calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED

    report = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny_random",
                "triton_vit_0",
                make_report_status(ok=True, bucket="translated"),
            )
        ],
        [
            _model_record(
                "vit",
                "vit_tiny_random",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[dict(vars(call)) for call in calls],
            ),
            _model_record(
                "llama",
                "llama_tiny_random",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[dict(vars(call)) for call in llama_calls],
            )
        ],
        generated_at="2026-05-26T00:00:00+00:00",
    )

    assert report["m10"]["runtime_resolved_count"] == 1
    assert report["m10"]["artifact_only_count"] == 0
    assert report["m10"]["performance_claim"] is False
    assert report["m10"]["runtime_launch_count"] == 1
    assert report["m10"]["intermediate_buffer_bytes"] == 15360
    assert sum(report["m10"]["unsupported_runtime_reasons"].values()) == 2
    assert report["model_summary"]["full_tvm_runnable_models"] == 0


def test_m105_native_decomposed_provider_resolves_vit_and_llama_prefill():
    sdpa_op = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
    vit_calls = extract_inductor_wrapper_extern_calls(
        f"""
def call(buf_q, buf_k, buf_v):
    return {sdpa_op}(
        reinterpret_tensor(buf_q, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_k, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        reinterpret_tensor(buf_v, (1, 4, 5, 16), (320, 16, 64, 1), 0),
        None,
        False,
        scale=0.25,
    )
""",
        case_name="vit_tiny_random",
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    )
    llama_calls = extract_inductor_wrapper_extern_calls(
        f"""
def call(q, k, v, mask):
    return {sdpa_op}(
        q,
        k,
        reinterpret_tensor(v, (1, 4, 16, 16), (1024, 16, 64, 1), 0),
        reinterpret_tensor(mask, (1, 4, 16, 16), (256, 0, 16, 1), 0),
        False,
        scale=0.25,
    )
""",
        case_name="llama_tiny_random",
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    )
    decode_calls = extract_inductor_wrapper_extern_calls(
        f"""
def call(q, k, v, past_key_values):
    return {sdpa_op}(q, k, v, past_key_values, True, scale=0.125)
""",
        case_name="llama_decode_single_token_cache",
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    )

    assert vit_calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert llama_calls[0].attention_contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL
    assert llama_calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert llama_calls[0].attention_runtime_kind == ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED
    assert llama_calls[0].attention_provider_kind == ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    assert llama_calls[0].attention_uses_host_staging is False
    assert llama_calls[0].attention_performance_claim is False
    assert llama_calls[0].attention_mask_shape == "1, 4, 16, 16"
    assert llama_calls[0].attention_runtime_launch_count == 1
    assert llama_calls[0].attention_mask_bytes == 4096
    assert llama_calls[0].attention_intermediate_buffer_bytes == 139264
    assert decode_calls[0].attention_contract == ATTENTION_CONTRACT_LLAMA_DECODE
    assert decode_calls[0].attention_runtime_status == ATTENTION_RUNTIME_STATUS_DEFERRED
    assert decode_calls[0].unsupported_attention_runtime_reason == (
        "attention_llama_decode_synthetic_only_m10_6"
    )

    report = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny_random",
                "triton_vit_0",
                make_report_status(ok=True, bucket="translated"),
            ),
            _kernel_record(
                "llama",
                "llama_tiny_random",
                "triton_llama_0",
                make_report_status(ok=True, bucket="translated"),
            ),
        ],
        [
            _model_record(
                "vit",
                "vit_tiny_random",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[dict(vars(call)) for call in vit_calls],
            ),
            _model_record(
                "llama",
                "llama_tiny_random",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[dict(vars(call)) for call in llama_calls],
            ),
            _model_record(
                "llama",
                "llama_decode_single_token_cache",
                kernel_count=0,
                translated=0,
                fallback=0,
                extern_calls=[dict(vars(call)) for call in decode_calls],
            ),
        ],
        generated_at="2026-05-26T00:00:00+00:00",
    )

    assert report["m10"]["runtime_resolved_count"] == 2
    assert report["m10"]["artifact_only_count"] == 0
    assert report["m10"]["performance_claim"] is False
    assert report["m10"]["runtime_launch_count"] == 2
    assert report["m10"]["total_io_bytes"] == 25600
    assert report["m10"]["intermediate_buffer_bytes"] == 154624
    assert report["m10"]["unsupported_runtime_reasons"] == {
        "attention_llama_decode_synthetic_only_m10_6": 1
    }
    assert report["pre_m10"]["contract_classes"][ATTENTION_CONTRACT_LLAMA_DECODE][
        "runtime_status"
    ] == {ATTENTION_RUNTIME_STATUS_DEFERRED: 1}
    assert report["model_summary"]["full_tvm_runnable_models"] == 0


def test_m6_model_report_groups_models_and_ranks_blockers(tmp_path):
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_0",
            make_report_status(ok=True, bucket="translated"),
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_llama_reduce_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
                message="not a replaceable pointwise kernel",
            ),
            op_counts={"tt.reduce": 1, "tt.load": 2},
            num_reduction=1,
        ),
    ]
    models = [
        _model_record("vit", "vit_tiny", kernel_count=1, translated=1, fallback=0),
        _model_record("llama", "llama_tiny", kernel_count=1, translated=0, fallback=1),
    ]

    report = build_model_corpus_report(
        records,
        models,
        dependency_versions={"torch": {"available": True, "version": "test"}},
        generated_at="2026-05-23T00:00:00+00:00",
    )
    write_model_corpus_report(report, tmp_path)

    assert report["report_kind"] == "triton_tvm_capability_report"
    assert report["model_summary"]["total_models"] == 2
    assert report["model_summary"]["fallback_kernels"] == 1
    assert report["models"][1]["full_tvm_runnable"] is False
    assert report["blockers"][0]["blocker_class"] == "reduction"
    assert report["blockers"][0]["models_impacted"] == 1
    assert report["blockers"][0]["fallback_reason"] == "unsupported_inductor_kernel"
    assert report["kernels"][1]["translate_status"]["fallback_reason"]

    markdown = render_capability_markdown(report, title="M6 Snapshot")
    assert "## Model Summary" in markdown
    assert "## Blockers" in markdown
    assert "## Pre-M7 Gate" in markdown
    assert "## Pre-M9 Gate" in markdown
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["model_summary"] == report["model_summary"]
    assert written["pre_m7"] == report["pre_m7"]
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# M6")


def test_pre_m7_report_section_preserves_buckets_and_orders_m7_blockers():
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_index_0",
            make_report_status(
                ok=False,
                bucket="unsupported_ttir_op",
                fallback_reason="unsupported_ttir_op",
                message="unsupported composed indexing pattern for pointwise_flat",
            ),
            op_counts={"tt.load": 2, "tt.store": 1, "tt.addptr": 2},
            indexing_summary={"kind": "flat_or_broadcast_candidate", "addptr_count": 2},
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_llama_reduce_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            op_counts={"tt.reduce": 1, "tt.load": 2},
            num_reduction=1,
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_llama_dot_0",
            make_report_status(
                ok=False,
                bucket="unsupported_ttir_op",
                fallback_reason="unsupported_ttir_op",
                message="tt.dot",
            ),
            op_counts={"tt.dot": 1, "tt.load": 2},
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_yolo_grid_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            grid_type="Grid2D",
        ),
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_attention_0",
            make_report_status(
                ok=False,
                bucket="unsupported_ttir_op",
                fallback_reason="unsupported_ttir_op",
                message="attention-adjacent softmax reshape",
            ),
            op_counts={"tt.trans": 1, "tt.load": 1},
        ),
    ]
    models = [
        _model_record("vit", "vit_tiny", kernel_count=2, translated=0, fallback=2),
        _model_record("llama", "llama_tiny", kernel_count=2, translated=0, fallback=2),
        _model_record("yolo", "yolo_tiny", kernel_count=1, translated=0, fallback=1),
    ]

    report = build_model_corpus_report(
        records,
        models,
        errors=[
            {
                "model_family": "vit",
                "model_case": "vit_zero",
                "case_name": "vit_zero",
                "bucket": "collection_error",
                "fallback_reason": "zero_triton_kernels",
                "message": "no kernels",
            }
        ],
        generated_at="2026-05-23T00:00:00+00:00",
    )

    pre_m7 = report["pre_m7"]
    assert pre_m7["taxonomy_version"] == 1
    assert pre_m7["builder_decision"] == "keep_tvmscript_source_builder_for_m7_entry"
    assert pre_m7["unsupported_taxonomy"]["top_level_bucket_policy"] == (
        "preserve_existing_buckets"
    )
    assert pre_m7["unsupported_taxonomy"]["detail_fields"] == [
        "fallback_reason",
        "blocker_class",
    ]
    assert report["summary"]["status_buckets"] == {
        "contract_error": 2,
        "unsupported_ttir_op": 3,
    }

    classes = pre_m7["reader_snapshot_classes"]
    assert classes["broadcast_view_index"]["kernel_count"] == 1
    assert classes["reduction"]["kernel_count"] == 1
    assert classes["matmul_dot"]["kernel_count"] == 1
    assert classes["atomic_grid"]["kernel_count"] == 1
    assert classes["attention_adjacent"]["kernel_count"] == 1

    entry_blockers = pre_m7["m7_entry_blockers"]
    assert [blocker["example_kernel"] for blocker in entry_blockers] == [
        "triton_vit_index_0"
    ]
    assert entry_blockers[0]["bucket"] == "unsupported_ttir_op"
    assert entry_blockers[0]["models_impacted"] == 1


def test_pre_m8_report_section_splits_reduction_family_debt_without_new_buckets():
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_per_fused_native_layer_norm_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            op_counts={"tt.reduce": 4, "tt.load": 4},
            num_reduction=4,
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_per_fused_add_embedding_mean_mul_pow_rsqrt_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            op_counts={"tt.reduce": 1, "tt.load": 3},
            num_reduction=1,
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_per_fused_max_pool2d_with_indices_40",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            op_counts={"tt.reduce": 1, "tt.load": 2},
            num_reduction=1,
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_per_fused__softmax_prepare_softmax_online_transpose_view_64",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            op_counts={"tt.reduce": 4, "tt.load": 4},
            num_reduction=4,
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_poi_fused_attention_masked_load_1",
            make_report_status(
                ok=False,
                bucket="unsupported_ttir_op",
                fallback_reason="unsupported_ttir_op",
                message="masked tt.load without other in attention softmax",
            ),
            op_counts={"tt.load": 2, "tt.store": 1},
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_poi_fused_convolution_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            grid_type="Grid2D",
        ),
    ]
    models = [
        _model_record("vit", "vit_tiny", kernel_count=1, translated=0, fallback=1),
        _model_record("llama", "llama_tiny", kernel_count=2, translated=0, fallback=2),
        _model_record("yolo", "yolo_tiny", kernel_count=3, translated=0, fallback=3),
    ]

    report = build_model_corpus_report(
        records,
        models,
        generated_at="2026-05-24T00:00:00+00:00",
    )

    assert report["summary"]["status_buckets"] == {
        "contract_error": 5,
        "unsupported_ttir_op": 1,
    }
    pre_m8 = report["pre_m8"]
    assert pre_m8["taxonomy_version"] == 1
    assert pre_m8["unsupported_taxonomy"]["top_level_bucket_policy"] == (
        "preserve_existing_buckets"
    )
    assert pre_m8["unsupported_taxonomy"]["detail_fields"] == [
        "fallback_reason",
        "blocker_class",
        "pre_m8_family",
    ]

    families = pre_m8["family_classes"]
    assert families["norm_layernorm"]["kernel_count"] == 1
    assert families["norm_rmsnorm"]["kernel_count"] == 1
    assert families["pooling_reduction"]["kernel_count"] == 1
    assert families["softmax_like"]["kernel_count"] == 1
    assert families["masked_attention_adjacent"]["kernel_count"] == 1
    assert families["deferred_grid"]["kernel_count"] == 1
    assert {entry["pre_m8_family"] for entry in pre_m8["deferred_debt"]} == {
        "deferred_grid"
    }
    assert report["model_summary"]["pre_m8_families"]["softmax_like"] == 1
    assert report["kernels"][0]["pre_m8_family"] == "norm_layernorm"

    markdown = render_capability_markdown(report, title="Pre-M8 Snapshot")
    assert "## Pre-M8 Gate" in markdown
    assert "softmax_like" in markdown


def test_pre_m9_report_section_splits_extern_gemm_from_deferred_grid_conv_attention():
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_pointwise_0",
            make_report_status(ok=True, bucket="translated"),
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_yolo_grid_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            grid_type="Grid2D",
        ),
    ]
    models = [
        _model_record(
            "vit",
            "vit_tiny",
            kernel_count=1,
            translated=1,
            fallback=0,
            extern_calls=[
                _extern_call(
                    "vit",
                    "vit_tiny",
                    "extern_kernels.addmm",
                    "extern_addmm_bias",
                    matmul_source_kind="wrapper_extern_addmm_bias",
                    matmul_m=16,
                    matmul_n=64,
                    matmul_k=64,
                    matmul_contract="matmul_minimal",
                    matmul_contract_ok=True,
                    matmul_a_dtype="float32",
                    matmul_b_dtype="float32",
                    matmul_accumulator_dtype="float32",
                    matmul_output_dtype="float32",
                    matmul_epilogue_kind="bias_add",
                    implementation_kind="extern_addmm_bias",
                    extern_symbol="extern_kernels.addmm",
                    extern_packed_func=EXTERN_ADDMM_BIAS_PACKED_FUNC,
                    extern_runtime_kind=EXTERN_GEMM_RUNTIME_KIND,
                    extern_runtime_replacement=EXTERN_GEMM_RUNTIME_REPLACEMENT,
                    extern_runtime_replacement_available=False,
                    extern_runtime_replacement_reason=(
                        EXTERN_ADDMM_BIAS_RUNTIME_REPLACEMENT_REASON
                    ),
                    source=(
                        "extern_kernels.addmm(arg12_1, reinterpret_tensor(buf7, "
                        "(16, 64), (64, 1), 0), reinterpret_tensor(arg11_1, "
                        "(64, 64), (1, 64), 0), alpha=1, beta=1, out=buf8)"
                    ),
                ),
                _extern_call(
                    "vit",
                    "vit_tiny",
                    "extern_kernels.mm",
                    "extern_gemm",
                    matmul_source_kind="wrapper_extern_gemm",
                    matmul_m=16,
                    matmul_n=64,
                    matmul_k=64,
                    matmul_contract="matmul_minimal",
                    matmul_contract_ok=True,
                    matmul_a_dtype="float32",
                    matmul_b_dtype="float32",
                    matmul_accumulator_dtype="float32",
                    matmul_output_dtype="float32",
                    matmul_epilogue_kind="none",
                    implementation_kind="extern_gemm",
                    extern_symbol="extern_kernels.mm",
                    extern_packed_func=EXTERN_GEMM_PACKED_FUNC,
                    extern_runtime_kind=EXTERN_GEMM_RUNTIME_KIND,
                    extern_runtime_replacement=EXTERN_GEMM_RUNTIME_REPLACEMENT,
                    extern_runtime_replacement_available=False,
                    extern_runtime_replacement_reason=(
                        EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON
                    ),
                    source=(
                        "extern_kernels.mm(reinterpret_tensor(buf1, (16, 64), "
                        "(64, 1), 0), reinterpret_tensor(arg4_1, (64, 64), "
                        "(1, 64), 0), out=buf2)"
                    ),
                ),
                _extern_call(
                    "vit",
                    "vit_tiny",
                    "torch.ops.aten._scaled_dot_product_efficient_attention.default",
                    "deferred_attention",
                ),
            ],
        ),
        _model_record(
            "yolo",
            "yolo_tiny",
            kernel_count=1,
            translated=0,
            fallback=1,
            extern_calls=[
                _extern_call(
                    "yolo",
                    "yolo_tiny",
                    "extern_kernels.convolution",
                    "deferred_convolution",
                )
            ],
        ),
    ]

    report = build_model_corpus_report(
        records,
        models,
        generated_at="2026-05-24T00:00:00+00:00",
    )

    assert report["model_summary"]["extern_op_count"] == 4
    assert report["model_summary"]["full_tvm_runnable_models"] == 0
    assert report["model_summary"]["triton_kernel_runnable_models"] == 1
    assert report["extern_ops"][0]["op_family"] == "extern_addmm_bias"
    gemm_record = next(
        record for record in report["extern_ops"] if record["op_family"] == "extern_gemm"
    )
    assert gemm_record["matmul_source_kind"] == "wrapper_extern_gemm"
    assert (gemm_record["matmul_m"], gemm_record["matmul_n"], gemm_record["matmul_k"]) == (
        16,
        64,
        64,
    )
    assert gemm_record["matmul_contract_ok"] is True
    assert gemm_record["matmul_epilogue_kind"] == "none"
    assert gemm_record["implementation_kind"] == "extern_gemm"
    assert gemm_record["extern_symbol"] == "extern_kernels.mm"
    assert gemm_record["extern_packed_func"] == EXTERN_GEMM_PACKED_FUNC
    assert gemm_record["extern_runtime_kind"] == EXTERN_GEMM_RUNTIME_KIND
    assert gemm_record["extern_runtime_replacement"] == EXTERN_GEMM_RUNTIME_REPLACEMENT
    assert gemm_record["extern_runtime_replacement_available"] is False
    assert (
        gemm_record["extern_runtime_replacement_reason"]
        == EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON
    )
    assert gemm_record["unsupported_matmul_reason"] == ""

    pre_m9 = report["pre_m9"]
    assert pre_m9["taxonomy_version"] == 1
    assert pre_m9["observed_ttir_dot_kernels"] == 0
    assert pre_m9["observed_grid_fallback_kernels"] == 1
    assert pre_m9["extern_family_classes"]["extern_gemm"]["call_count"] == 1
    assert pre_m9["extern_family_classes"]["extern_addmm_bias"]["call_count"] == 1
    assert pre_m9["extern_family_classes"]["deferred_convolution"]["call_count"] == 1
    assert pre_m9["extern_family_classes"]["deferred_attention"]["call_count"] == 1
    assert pre_m9["m96_extern_gemm_runtime"]["runtime_resolved_count"] == 0
    assert pre_m9["m96_extern_gemm_runtime"]["status_counts"] == {
        EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY: 1
    }
    assert pre_m9["m96_extern_gemm_runtime"]["provider_counts"] == {
        EXTERN_GEMM_PROVIDER_NONE: 1
    }
    assert len(pre_m9["m9_materialized_artifact_candidates"]) == 2
    candidate = next(
        entry
        for entry in pre_m9["m9_materialized_artifact_candidates"]
        if entry["op_family"] == "extern_gemm"
    )
    assert candidate["op_family"] == "extern_gemm"
    assert candidate["matmul_source_kind"] == "wrapper_extern_gemm"
    assert (candidate["matmul_m"], candidate["matmul_n"], candidate["matmul_k"]) == (
        16,
        64,
        64,
    )
    assert candidate["implementation_kind"] == "extern_gemm"
    assert candidate["extern_symbol"] == "extern_kernels.mm"
    assert candidate["extern_packed_func"] == EXTERN_GEMM_PACKED_FUNC
    assert candidate["extern_runtime_kind"] == EXTERN_GEMM_RUNTIME_KIND
    assert candidate["extern_runtime_replacement"] == EXTERN_GEMM_RUNTIME_REPLACEMENT
    assert candidate["extern_runtime_replacement_available"] is False
    assert candidate["extern_gemm_runtime_status"] == EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
    assert candidate["extern_gemm_provider_kind"] == EXTERN_GEMM_PROVIDER_NONE
    addmm_candidate = next(
        entry
        for entry in pre_m9["m9_materialized_artifact_candidates"]
        if entry["op_family"] == "extern_addmm_bias"
    )
    assert addmm_candidate["matmul_source_kind"] == "wrapper_extern_addmm_bias"
    assert addmm_candidate["matmul_epilogue_kind"] == "bias_add"
    assert addmm_candidate["implementation_kind"] == "extern_addmm_bias"
    assert addmm_candidate["extern_packed_func"] == EXTERN_ADDMM_BIAS_PACKED_FUNC
    assert pre_m9["m9_entry_debt"] == []
    assert {entry["op_family"] for entry in pre_m9["deferred_debt"]} == {
        "deferred_attention",
        "deferred_convolution",
    }

    markdown = render_capability_markdown(report, title="Pre-M9 Snapshot")
    assert "## Pre-M9 Gate" in markdown
    assert "extern_gemm" in markdown
    assert "deferred_convolution" in markdown


def test_pre_m11_report_section_classifies_vision_debt_and_attention_boundary():
    sdpa_op = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_grid_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            grid_type="Grid2D",
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_yolo_grid_0",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
            ),
            grid_type="Grid2D",
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_llama_0",
            make_report_status(ok=True, bucket="translated"),
        ),
    ]
    models = [
        _model_record(
            "vit",
            "vit_tiny",
            kernel_count=1,
            translated=0,
            fallback=1,
            extern_calls=[
                _extern_call(
                    "vit",
                    "vit_tiny",
                    "extern_kernels.convolution",
                    "deferred_convolution",
                ),
                _extern_call(
                    "vit",
                    "vit_tiny",
                    sdpa_op,
                    "deferred_attention",
                    attention_contract=ATTENTION_CONTRACT_VIT_FULL,
                    attention_semantics_status=ATTENTION_SEMANTICS_STATUS_ACCEPTED,
                    attention_implementation_kind=(
                        ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED
                    ),
                    attention_runtime_kind=ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
                    attention_provider_kind=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
                    attention_runtime_status=ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
                    attention_performance_claim=False,
                    attention_uses_host_staging=False,
                ),
            ],
        ),
        _model_record(
            "yolo",
            "yolo_tiny",
            kernel_count=1,
            translated=0,
            fallback=1,
            extern_calls=[
                _extern_call(
                    "yolo",
                    "yolo_tiny",
                    "extern_kernels.convolution",
                    "deferred_convolution",
                )
            ],
        ),
        _model_record("llama", "llama_tiny", kernel_count=1, translated=1, fallback=0),
    ]

    report = build_model_corpus_report(
        records,
        models,
        generated_at="2026-05-26T00:00:00+00:00",
    )

    pre_m11 = report["pre_m11"]
    assert pre_m11["taxonomy_version"] == 1
    assert pre_m11["gate_status"] == "pre_m11_debt_classified_v1"
    assert pre_m11["primary_debt_counts"] == {
        "captured_grid": 2,
        "deferred_convolution": 2,
    }
    debt_by_kind = {entry["debt_kind"]: entry for entry in pre_m11["entry_debt"]}
    assert debt_by_kind["wrapper_convolution"]["op_family"] == "deferred_convolution"
    assert debt_by_kind["wrapper_convolution"]["call_count"] == 2
    assert debt_by_kind["captured_grid"]["pre_m11_family"] == "captured_grid"
    assert debt_by_kind["captured_grid"]["kernel_count"] == 2
    assert debt_by_kind["captured_grid"]["models"] == ["vit_tiny", "yolo_tiny"]
    assert (
        pre_m11["vision_op_policy"]["implementation_policy"]["implicit_pytorch_fallback"]
        == "disallowed"
    )
    assert "nms" in pre_m11["vision_op_policy"]["operator_families"]["yolo_postprocess"]
    attention_boundary = pre_m11["attention_boundary"]
    assert attention_boundary["m11_entry_debt"] is False
    assert attention_boundary["historical_deferred_attention_family_count"] == 1
    assert attention_boundary["runtime_resolved_attention_count"] == 1
    assert attention_boundary["runtime_deferred_attention_count"] == 0
    assert attention_boundary["boundary_status"] == "m10_runtime_closed"
    assert "rope_runtime" in pre_m11["still_deferred_elsewhere"]

    markdown = render_capability_markdown(report, title="Pre-M11 Snapshot")
    assert "## Pre-M11 Gate" in markdown
    assert "deferred_convolution" in markdown
    assert "captured_grid" in markdown


def test_pre_m12_debt_gate_separates_frozen_m11_entry_from_current_residuals():
    translated_grid = _kernel_record(
        "vit",
        "vit_tiny",
        "triton_vit_grid_0",
        make_report_status(ok=True, bucket="translated"),
        grid_type="Grid2D",
    )
    translated_grid.update(
        {
            "m11_grid_status": "m11_grid_classified",
            "m11_grid_family": M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
            "m11_grid_launch_kind": "Grid2D",
            "m11_grid_runtime_status": M11_GRID2D_RUNTIME_READY,
            "m11_grid_artifact_status": M11_GRID2D_ARTIFACT_READY,
            "unsupported_m11_grid_runtime_reason": "",
        }
    )
    concat_split_grid = _kernel_record(
        "yolo",
        "yolo_tiny",
        "triton_yolo_grid_split_0",
        make_report_status(
            ok=False,
            bucket="contract_error",
            fallback_reason="grid2d_concat_split_multi_output_layout_deferred_m11_6",
        ),
        grid_type="Grid2D",
    )
    concat_split_grid.update(
        {
            "blocker_class": "grid",
            "pre_m8_family": "deferred_grid",
            "m11_grid_status": "m11_grid_classified",
            "m11_grid_family": "grid_concat_split",
            "m11_grid_launch_kind": "Grid2D",
            "m11_grid_runtime_status": "unsupported",
            "m11_grid_artifact_status": "m11_6_grid2d_artifact_rejected",
            "unsupported_m11_grid_runtime_reason": (
                "grid2d_concat_split_multi_output_layout_deferred_m11_6"
            ),
        }
    )
    gemm_artifact = _extern_call(
        "vit",
        "vit_tiny",
        "extern_kernels.mm",
        "extern_gemm",
        extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
    )

    report = build_model_corpus_report(
        [translated_grid, concat_split_grid],
        [
            _model_record(
                "vit",
                "vit_tiny",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[gemm_artifact],
            ),
            _model_record("yolo", "yolo_tiny", kernel_count=1, translated=0, fallback=1),
        ],
        generated_at="2026-05-26T00:00:00+00:00",
    )

    pre_m11 = report["pre_m11"]
    assert pre_m11["primary_debt_counts"]["captured_grid"] == 2
    assert pre_m11["grid_policy"]["debt"]["current_remaining_kernel_count"] == 1
    assert pre_m11["grid_policy"]["debt"]["entry_count_policy"] == (
        "frozen_m11_entry_count_from_classified_grid_records"
    )

    pre_m12 = report["pre_m12"]
    assert pre_m12["gate_status"] == "pre_m12_debt_cleaned_v1"
    assert pre_m12["entry_baseline"]["captured_fallback_count"] == 1
    assert pre_m12["residual_debt_counts"]["captured_grid_concat_split"] == 1
    assert pre_m12["residual_debt_counts"]["wrapper_extern_gemm_artifact_only"] == 1
    assert pre_m12["model_entry"][0]["model_case"] == "vit_tiny"
    assert pre_m12["model_entry"][0]["m12_entry_role"] == (
        "captured_clean_but_wrapper_matmul_artifact_only"
    )

    markdown = render_capability_markdown(report, title="Pre-M12 Snapshot")
    assert "## Pre-M12 Debt Gate" in markdown
    assert "captured_grid_concat_split" in markdown


def test_m12_native_vit_matmul_scope_resolves_only_vit_wrapper_calls():
    wrapper_source = """
def call(a, b, out):
    extern_kernels.mm(reinterpret_tensor(a, (5, 64), (64, 1), 0), reinterpret_tensor(b, (64, 64), (1, 64), 0), out=out)
"""
    vit_calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="vit_tiny_random",
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        extern_gemm_runtime_model_case="vit_tiny_random",
    )
    llama_calls = extract_inductor_wrapper_extern_calls(
        wrapper_source,
        case_name="llama_tiny_random",
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        extern_gemm_runtime_model_case="vit_tiny_random",
    )

    assert vit_calls[0].implementation_kind == "native_tir_schedule"
    assert vit_calls[0].schedule_id == NATIVE_TIR_MATMUL_SCHEDULE_ID
    assert vit_calls[0].extern_gemm_runtime_status == (
        EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
    )
    assert vit_calls[0].extern_gemm_provider_kind == EXTERN_GEMM_PROVIDER_NATIVE_TVM
    assert vit_calls[0].extern_gemm_uses_host_staging is False
    assert vit_calls[0].extern_gemm_performance_claim is False

    assert llama_calls[0].implementation_kind == "extern_gemm"
    assert llama_calls[0].extern_gemm_runtime_status == (
        EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
    )
    assert llama_calls[0].extern_gemm_provider_kind == EXTERN_GEMM_PROVIDER_NONE


def test_m12_report_closes_vit_native_matmul_without_promoting_full_native():
    kernels = [
        _kernel_record(
            "vit",
            "vit_tiny_random",
            f"triton_vit_{idx}",
            make_report_status(ok=True, bucket="translated"),
        )
        for idx in range(7)
    ]
    kernels.append(
        _kernel_record(
            "yolo",
            "yolov8n_yaml_random",
            "triton_yolo_concat_split",
            make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="grid2d_concat_split_multi_output_layout_deferred_m11_6",
            ),
            grid_type="Grid2D",
        )
    )
    kernels[-1].update(
        {
            "blocker_class": "grid",
            "pre_m8_family": "deferred_grid",
            "m11_grid_status": "m11_grid_classified",
            "m11_grid_family": "grid_concat_split",
            "m11_grid_launch_kind": "Grid2D",
            "m11_grid_runtime_status": "unsupported",
            "m11_grid_artifact_status": "m11_6_grid2d_artifact_rejected",
            "unsupported_m11_grid_runtime_reason": (
                "grid2d_concat_split_multi_output_layout_deferred_m11_6"
            ),
        }
    )
    vit_extern_calls = [
        _extern_call(
            "vit",
            "vit_tiny_random",
            "extern_kernels.convolution",
            "deferred_convolution",
            line_no=10,
            vision_runtime_status="runtime_resolved",
            vision_provider_kind="device_torch_cuda",
            vision_implementation_kind="extern_conv2d",
            vision_host_staging_bytes=0,
            vision_performance_claim=True,
        ),
        _extern_call(
            "vit",
            "vit_tiny_random",
            "torch.ops.aten._scaled_dot_product_efficient_attention.default",
            "deferred_attention",
            line_no=50,
            attention_runtime_status="runtime_resolved",
            attention_provider_kind="native_decomposed",
            attention_implementation_kind="native_decomposed",
            attention_host_staging_bytes=0,
            attention_performance_claim=False,
        ),
    ]
    for idx in range(3):
        vit_extern_calls.append(
            _native_m12_matmul_call(
                "vit",
                "vit_tiny_random",
                "extern_addmm_bias",
                line_no=20 + idx,
                m=5,
                n=64,
                k=64,
            )
        )
    for idx, shape in enumerate(((5, 64, 64), (5, 128, 64), (5, 64, 128), (1, 64, 64))):
        vit_extern_calls.append(
            _native_m12_matmul_call(
                "vit",
                "vit_tiny_random",
                "extern_gemm",
                line_no=60 + idx,
                m=shape[0],
                n=shape[1],
                k=shape[2],
            )
        )
    llama_gemm = _extern_call(
        "llama",
        "llama_tiny_random",
        "extern_kernels.mm",
        "extern_gemm",
        matmul_source_kind="wrapper_extern_gemm",
        matmul_contract_ok=True,
        implementation_kind="extern_gemm",
        extern_symbol="extern_kernels.mm",
        extern_packed_func=EXTERN_GEMM_PACKED_FUNC,
        extern_runtime_kind=EXTERN_GEMM_RUNTIME_KIND,
        extern_runtime_replacement=EXTERN_GEMM_RUNTIME_REPLACEMENT,
        extern_runtime_replacement_available=False,
        extern_runtime_replacement_reason=EXTERN_GEMM_RUNTIME_REPLACEMENT_REASON,
        extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
        extern_gemm_provider_kind=EXTERN_GEMM_PROVIDER_NONE,
    )

    report = build_model_corpus_report(
        kernels,
        [
            _model_record(
                "vit",
                "vit_tiny_random",
                kernel_count=7,
                translated=7,
                fallback=0,
                extern_calls=vit_extern_calls,
            ),
            _model_record(
                "llama",
                "llama_tiny_random",
                kernel_count=0,
                translated=0,
                fallback=0,
                extern_calls=[llama_gemm],
            ),
            _model_record(
                "yolo",
                "yolov8n_yaml_random",
                kernel_count=1,
                translated=0,
                fallback=1,
            ),
        ],
        generated_at="2026-05-27T00:00:00+00:00",
    )

    assert report["model_summary"]["full_tvm_runnable_models"] == 0
    assert report["pre_m12"]["residual_debt_counts"]["wrapper_extern_gemm_artifact_only"] == 1
    vit_entry = next(
        entry for entry in report["pre_m12"]["model_entry"] if entry["model_case"] == "vit_tiny_random"
    )
    assert vit_entry["artifact_only_extern_gemm"] == 0
    assert vit_entry["artifact_only_extern_addmm_bias"] == 0

    m12 = report["m12"]
    assert m12["policy"]["performance_ready_e2e"] is False
    assert m12["policy"]["strict_full_tvm_native"] is False
    plan = m12["vit_fixed_shape_execution_plan"]
    assert plan["captured_kernel_count"] == 7
    assert plan["wrapper_call_count"] == 9
    assert plan["artifact_only_matmul_blockers"] == 0
    closure = m12["m12_2_native_matmul_closure"]
    assert closure["status"] == "passed"
    assert closure["runtime_resolved_extern_gemm"] == 4
    assert closure["runtime_resolved_extern_addmm_bias"] == 3
    assert closure["host_staging_bytes"] == 0
    assert closure["performance_claim"] is False

    markdown = render_capability_markdown(report, title="M12 Snapshot")
    assert "## M12 ViT Fixed-Shape E2E" in markdown
    assert "native_tvm_matmul" in markdown


def test_m123_vit_e2e_runner_schema_without_execution(tmp_path):
    report = run_vit_fixed_shape_e2e_runner(out_dir=tmp_path, run_execution=False)

    assert report["report_kind"] == "triton_tvm_m12_3_vit_fixed_shape_e2e_runner"
    assert report["status"] == "not_run"
    assert report["target_model"] == "vit_tiny_random"
    assert report["performance_ready_e2e"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["p0"]["passed"] is False
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_m124_vit_e2e_dashboard_schema_without_benchmarks(tmp_path):
    report = run_vit_e2e_dashboard(
        out_dir=tmp_path,
        warmup=1,
        repeat=1,
        run_benchmarks=False,
    )

    assert report["report_kind"] == "triton_tvm_m12_4_vit_fixed_shape_e2e_dashboard"
    assert report["status"] == "not_run"
    assert report["target_model"] == "vit_tiny_random"
    assert report["latency_ms"]["triton_tvm_e2e"]["p50_ms"] is None
    assert report["p1"]["passed"] is False
    assert report["p2"]["passed"] is False
    assert report["performance_ready_e2e"] is False
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["host_staging_bytes"] == 0
    assert report["invariants"]["status"] == "passed"
    assert (tmp_path / "report.json").exists()
    assert "M12.4 ViT E2E Performance Dashboard" in (tmp_path / "report.md").read_text()


def test_m125_optimization_review_from_synthetic_dashboard(tmp_path):
    dashboard = {
        "report_kind": "triton_tvm_m12_4_vit_fixed_shape_e2e_dashboard",
        "status": "measured",
        "target_model": "vit_tiny_random",
        "m12_performance_tier": "P1",
        "performance_ready_e2e": False,
        "launch_count_estimate": 16,
        "p1": {"passed": True},
        "p2": {"passed": False},
        "provider_mix": {
            "provider_counts": {
                "torch_inductor_triton_captured_harness": 7,
                "device_torch_cuda": 1,
                "native_decomposed": 1,
                "native_tvm_matmul": 7,
            }
        },
    }
    dashboard_path = tmp_path / "m12_4_report.json"
    dashboard_path.write_text(json.dumps(dashboard), encoding="utf-8")

    out_dir = tmp_path / "review"
    report = run_optimization_review(dashboard_report=dashboard_path, out_dir=out_dir)

    assert report["report_kind"] == "triton_tvm_m12_5_vit_optimization_review"
    assert report["status"] == "passed"
    assert report["disposition"] == "open_m12_7_p2_recovery"
    assert report["focused_follow_up_recommended"] is True
    assert report["p3_recommended"] is False
    assert report["primary_target"] == "native_tvm_matmul_provider_cost"
    assert report["invariants"]["status"] == "passed"
    assert len(report["opportunities"]) == 7
    assert (out_dir / "report.json").exists()
    assert "M12.5 Optimization Opportunity Review" in (out_dir / "report.md").read_text()


def test_m1255_perf_optimization_freeze_from_synthetic_reports(tmp_path):
    dashboard = {
        "report_kind": "triton_tvm_m12_4_vit_fixed_shape_e2e_dashboard",
        "status": "measured",
        "target_model": "vit_tiny_random",
        "m12_performance_tier": "P1",
        "performance_ready_e2e": False,
        "host_staging_bytes": 0,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "launch_count_estimate": 16,
        "correctness": {"allclose": True},
        "p1": {"passed": True},
        "p2": {"passed": False},
        "latency_ms": {
            "torch_compile_inductor": {"p50_ms": 0.33, "p95_ms": 0.36},
            "triton_tvm_e2e": {"p50_ms": 0.91, "p95_ms": 0.96},
        },
        "ratios": {
            "triton_tvm_to_torch_compile_p50": 2.75,
            "triton_tvm_to_torch_compile_p95": 2.67,
        },
        "provider_mix": {
            "provider_counts": {
                "torch_inductor_triton_captured_harness": 7,
                "device_torch_cuda": 1,
                "native_decomposed": 1,
                "native_tvm_matmul": 7,
            }
        },
    }
    review = {
        "report_kind": "triton_tvm_m12_5_vit_optimization_review",
        "status": "passed",
        "disposition": "open_m12_7_p2_recovery",
        "focused_follow_up_recommended": True,
        "p3_recommended": False,
        "primary_target": "native_tvm_matmul_provider_cost",
        "opportunities": [
            {"category": "native_tvm_matmul_provider_cost", "priority": "primary"},
            {"category": "launch_count", "priority": "secondary"},
        ],
    }
    dashboard_path = tmp_path / "dashboard.json"
    review_path = tmp_path / "review.json"
    dashboard_path.write_text(json.dumps(dashboard), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")

    out_dir = tmp_path / "freeze"
    report = run_perf_optimization_freeze(
        dashboard_report=dashboard_path,
        review_report=review_path,
        out_dir=out_dir,
    )

    assert report["report_kind"] == "triton_tvm_m12_5_5_vit_perf_optimization_freeze"
    assert report["status"] == "passed"
    assert report["primary_target"] == "native_tvm_matmul_provider_cost"
    assert report["current_gate_state"]["p1_passed"] is True
    assert report["current_gate_state"]["p2_passed"] is False
    assert report["p2_gap"]["p50_excess_ms"] > 0.0
    assert report["p2_gap"]["p95_excess_ms"] > 0.0
    selected = [target for target in report["frozen_targets"] if target["selected_by_m12_5"]]
    assert len(selected) == 1
    assert selected[0]["target_id"] == "native_tvm_matmul_provider_cost"
    assert report["invariants"]["status"] == "passed"
    assert (out_dir / "report.json").exists()
    assert "M12.5.5 Performance Optimization Freeze" in (out_dir / "report.md").read_text()


def test_m126_vit_e2e_hardening_from_synthetic_reports(tmp_path):
    corpus = _m126_synthetic_corpus()
    dashboard = _m126_synthetic_dashboard()
    review = _m126_synthetic_review()
    freeze = _m126_synthetic_freeze()
    corpus_path = tmp_path / "corpus.json"
    dashboard_path = tmp_path / "dashboard.json"
    review_path = tmp_path / "review.json"
    freeze_path = tmp_path / "freeze.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    dashboard_path.write_text(json.dumps(dashboard), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    out_dir = tmp_path / "hardening"
    report = run_vit_e2e_hardening(
        corpus_report=corpus_path,
        dashboard_report=dashboard_path,
        review_report=review_path,
        freeze_report=freeze_path,
        out_dir=out_dir,
    )

    assert report["report_kind"] == "triton_tvm_m12_6_vit_e2e_surface_hardening"
    assert report["status"] == "passed"
    assert report["performance_optimization_introduced"] is False
    assert report["checks"]["provider_id_validation"]["status"] == "passed"
    assert report["checks"]["dashboard_schema_guards"]["status"] == "passed"
    assert report["checks"]["no_hidden_host_staging"]["status"] == "passed"
    assert report["checks"]["no_accidental_strict_native_claim"]["status"] == "passed"
    assert report["primary_next_target"] == "native_tvm_matmul_provider_cost"
    assert report["next_action"] == "M12.7 P2 Recovery - Native Matmul Provider Cost"
    assert (out_dir / "report.json").exists()
    assert "M12.6 E2E Surface Hardening" in (out_dir / "report.md").read_text()


def test_m126_vit_e2e_hardening_rejects_stale_surface_claims(tmp_path):
    corpus = _m126_synthetic_corpus()
    dashboard = _m126_synthetic_dashboard()
    review = _m126_synthetic_review()
    freeze = _m126_synthetic_freeze()
    dashboard["host_staging_bytes"] = 4
    dashboard["performance_ready_e2e"] = True
    dashboard["strict_full_tvm_native"] = True
    dashboard["provider_mix"]["wrapper_matmul"]["unsupported_matmul_reason"] = "old_scope"
    corpus["m12"]["m12_2_native_matmul_closure"]["records"][0][
        "unsupported_matmul_reason"
    ] = "old_scope"
    corpus_path = tmp_path / "corpus.json"
    dashboard_path = tmp_path / "dashboard.json"
    review_path = tmp_path / "review.json"
    freeze_path = tmp_path / "freeze.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    dashboard_path.write_text(json.dumps(dashboard), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    report = run_vit_e2e_hardening(
        corpus_report=corpus_path,
        dashboard_report=dashboard_path,
        review_report=review_path,
        freeze_report=freeze_path,
        out_dir=None,
    )

    failures = set(report["invariants"]["invariant_failures"])
    assert report["status"] == "failed"
    assert "m12_6_host_staging_nonzero" in failures
    assert "m12_6_dashboard_performance_ready_without_p2" in failures
    assert "m12_6_strict_native_claim_present_dashboard" in failures
    assert "m12_6_runtime_resolved_stale_fallback_reason" in failures


def test_m127_native_matmul_provider_cost_from_synthetic_profiles(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_vit_native_matmul_provider_cost_report(
        seed=0,
        warmup=5,
        repeat=20,
        profile_repeat=3,
        hardening_report={"status": "passed", "invariants": {"status": "passed"}},
        hardening_report_path=tmp_path / "hardening.json",
        freeze_report=_m126_synthetic_freeze(),
        freeze_report_path=tmp_path / "freeze.json",
        latency_ms=_m127_synthetic_latency(),
        legacy_correctness={"allclose": True},
        legacy_provider_mix=provider_mix,
        legacy_profile_runs=_m127_profile_runs("legacy_per_call_sync", sync_ms=0.060),
        optimized_correctness={"allclose": True},
        optimized_provider_mix=provider_mix,
        optimized_profile_runs=_m127_profile_runs("no_per_call_sync", sync_ms=0.0),
        out_dir=tmp_path / "m127",
    )

    assert report["report_kind"] == "triton_tvm_m12_7_native_matmul_provider_cost"
    assert report["status"] == "passed"
    assert report["performance_ready_e2e"] is False
    assert report["optimization"]["validated"] is True
    assert report["next_action"] == "M12.8 P2 Gate Re-run and Close/Continue Decision"
    assert report["dominant_cost_category"]["category"] == "synchronization_wall_ms"
    assert report["provider_mix"]["legacy_per_call_sync"]["provider_counts"][
        EXTERN_GEMM_PROVIDER_NATIVE_TVM
    ] == 7
    assert report["native_matmul_profile"]["policies"]["legacy_per_call_sync"][
        "complete_call_count"
    ] is True
    assert report["p2_gap_recovery"]["recovered_p50_ms"] > 0.0
    assert (tmp_path / "m127" / "report.json").exists()
    assert "M12.7 Native Matmul Provider Cost" in (
        tmp_path / "m127" / "report.md"
    ).read_text()


def test_m127_native_matmul_provider_cost_reports_invariant_failures(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    broken_provider_mix = json.loads(json.dumps(provider_mix))
    broken_provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6

    report = build_vit_native_matmul_provider_cost_report(
        seed=0,
        warmup=5,
        repeat=20,
        profile_repeat=3,
        hardening_report={"status": "failed", "invariants": {"status": "failed"}},
        hardening_report_path=tmp_path / "hardening.json",
        freeze_report=_m126_synthetic_freeze(),
        freeze_report_path=tmp_path / "freeze.json",
        latency_ms=_m127_synthetic_latency(),
        legacy_correctness={"allclose": True},
        legacy_provider_mix=provider_mix,
        legacy_profile_runs=_m127_profile_runs("legacy_per_call_sync", sync_ms=0.060),
        optimized_correctness={"allclose": False},
        optimized_provider_mix=broken_provider_mix,
        optimized_profile_runs=_m127_profile_runs("no_per_call_sync", sync_ms=0.0),
        out_dir=None,
    )

    failures = set(report["invariants"]["invariant_failures"])
    assert report["status"] == "failed"
    assert report["optimization"]["validated"] is False
    assert "m12_7_hardening_report_not_passed" in failures
    assert "m12_7_hardening_invariants_not_passed" in failures
    assert "m12_7_correctness_failed_no_per_call_sync" in failures
    assert "m12_7_provider_mix_mismatch_no_per_call_sync" in failures


def test_m128_vit_p2_gate_decision_closes_m12_on_p2_pass(tmp_path):
    report = build_vit_p2_gate_decision_report(
        seed=0,
        warmup=5,
        repeat=20,
        m127_report=_m128_synthetic_m127_report(),
        m127_report_path=tmp_path / "m127.json",
        latency_ms=_m128_synthetic_latency(triton_p50=0.640, triton_p95=0.700),
        correctness={"allclose": True},
        provider_mix=_m126_synthetic_dashboard()["provider_mix"],
        out_dir=tmp_path / "m128",
    )

    assert report["report_kind"] == "triton_tvm_m12_8_vit_p2_gate_decision"
    assert report["status"] == "passed"
    assert report["baselines"] == [
        "torch_eager_cuda",
        "torch_compile_inductor",
        "triton_tvm_e2e",
    ]
    assert report["p2"]["passed"] is True
    assert report["performance_ready_e2e"] is True
    assert report["performance_claim"] is True
    assert report["host_staging_bytes"] == 0
    assert report["correctness"]["allclose"] is True
    assert report["provider_mix"]["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] == 7
    assert report["m12_8_decision"]["decision"] == "close_m12_p2_passed"
    assert report["m12_8_decision"]["m12_complete"] is True
    assert (tmp_path / "m128" / "report.json").exists()
    assert "M12.8 ViT P2 Gate Decision" in (
        tmp_path / "m128" / "report.md"
    ).read_text()


def test_m128_vit_p2_gate_decision_near_miss_opens_tiny_m129(tmp_path):
    report = build_vit_p2_gate_decision_report(
        seed=0,
        warmup=5,
        repeat=20,
        m127_report=_m128_synthetic_m127_report(),
        m127_report_path=tmp_path / "m127.json",
        latency_ms=_m128_synthetic_latency(triton_p50=0.680, triton_p95=0.752),
        correctness={"allclose": True},
        provider_mix=_m126_synthetic_dashboard()["provider_mix"],
        out_dir=None,
    )

    assert report["p2"]["passed"] is False
    assert report["performance_ready_e2e"] is False
    assert report["m12_8_decision"]["decision"] == "open_tiny_m12_9_near_miss"
    assert report["m12_8_decision"]["m12_complete"] is False


def test_m128_vit_p2_gate_decision_far_miss_returns_to_profile(tmp_path):
    report = build_vit_p2_gate_decision_report(
        seed=0,
        warmup=5,
        repeat=20,
        m127_report=_m128_synthetic_m127_report(),
        m127_report_path=tmp_path / "m127.json",
        latency_ms=_m128_synthetic_latency(triton_p50=0.830, triton_p95=0.900),
        correctness={"allclose": True},
        provider_mix=_m126_synthetic_dashboard()["provider_mix"],
        out_dir=None,
    )

    assert report["p2"]["passed"] is False
    assert report["m12_8_decision"]["decision"] == "return_to_profile_bottleneck_may_be_wrong"
    assert report["m12_8_decision"]["torch_compile_ratio_max"] > 2.3


def test_m128_vit_p2_gate_decision_blocks_on_invariant_failure(tmp_path):
    broken_provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    broken_provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6

    report = build_vit_p2_gate_decision_report(
        seed=0,
        warmup=0,
        repeat=20,
        m127_report={"status": "failed", "invariants": {"status": "failed"}},
        m127_report_path=tmp_path / "m127.json",
        latency_ms=_m128_synthetic_latency(triton_p50=0.640, triton_p95=0.700),
        correctness={"allclose": False},
        provider_mix=broken_provider_mix,
        out_dir=None,
    )

    failures = set(report["invariants"]["invariant_failures"])
    assert report["status"] == "failed"
    assert report["p2"]["passed"] is False
    assert report["m12_8_decision"]["decision"] == "blocked_invariants_failed"
    assert "m12_8_m12_7_report_not_passed" in failures
    assert "m12_8_warmed_cache_required" in failures
    assert "m12_8_correctness_allclose_not_true" in failures
    assert "m12_8_provider_mix_incomplete" in failures


def test_m129_provider_runtime_optimization_status_not_p2_gated(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_vit_provider_runtime_optimization_report(
        seed=0,
        warmup=5,
        repeat=20,
        m128_report={"status": "passed", "m12_8_decision": {"decision": "return_to_profile"}},
        m128_report_path=tmp_path / "m128.json",
        latency_ms=_m129_synthetic_latency(current_p50=0.840, backend_p50=0.820),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
        },
        backend_pass_summary=_m129_backend_pass_summary(transformed=7, expected=7),
        provider_overhead_profile={},
        out_dir=tmp_path / "m129",
    )

    assert report["report_kind"] == "triton_tvm_m12_9_vit_provider_runtime_optimization"
    assert report["status"] == "passed"
    assert report["performance_ready_e2e"] is False
    assert report["p2_context"]["best_mode_p2_passed"] is False
    assert report["best_mode"]["mode"] == BACKEND_PASS_MODE
    assert report["backend_pass"]["schedule_id"] == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
    assert report["optimization_modes"][BACKEND_PASS_MODE]["p50_recovery_ms"] > 0.0
    assert report["optimization_modes"][BACKEND_PASS_MODE]["backend_general"] is True
    assert (
        report["optimization_modes"][BACKEND_PASS_MODE][
            "counts_for_triton_tvm_backend_progress"
        ]
        is True
    )
    assert report["optimization_modes"][FUSED_QKV_MODE]["model_specific_artifact"] is True
    assert (
        report["optimization_modes"][FUSED_QKV_MODE][
            "counts_for_triton_tvm_backend_progress"
        ]
        is False
    )
    assert (tmp_path / "m129" / "report.json").exists()


def test_m129_provider_runtime_optimization_reports_invariant_failures(tmp_path):
    provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    broken_provider_mix = json.loads(json.dumps(provider_mix))
    broken_provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6
    report = build_vit_provider_runtime_optimization_report(
        seed=0,
        warmup=5,
        repeat=20,
        m128_report={"status": "passed", "m12_8_decision": {"decision": "return_to_profile"}},
        m128_report_path=tmp_path / "m128.json",
        latency_ms=_m129_synthetic_latency(current_p50=0.840, backend_p50=0.820),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": False},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: broken_provider_mix,
        },
        backend_pass_summary=_m129_backend_pass_summary(transformed=6, expected=7),
        provider_overhead_profile={},
        out_dir=None,
    )

    failures = set(report["invariants"]["invariant_failures"])
    assert report["status"] == "failed"
    assert "m12_9_correctness_failed_triton_tvm_e2e_m129_backend_pass" in failures
    assert "m12_9_backend_pass_did_not_transform_all_calls" in failures
    assert any("provider_count_mismatch" in failure for failure in failures)


def test_m1210_route_cleanup_labels_model_specific_artifacts(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    m129 = build_vit_provider_runtime_optimization_report(
        seed=0,
        warmup=5,
        repeat=20,
        m128_report={"status": "passed", "m12_8_decision": {"decision": "return_to_profile"}},
        m128_report_path=tmp_path / "m128.json",
        latency_ms=_m129_synthetic_latency(current_p50=0.840, backend_p50=0.820),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
            FUSED_QKV_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
            FUSED_QKV_MODE: provider_mix,
        },
        backend_pass_summary=_m129_backend_pass_summary(transformed=7, expected=7),
        provider_overhead_profile={},
        out_dir=None,
    )
    report = build_vit_route_cleanup_report(
        m129_report=m129,
        m129_report_path=tmp_path / "m129.json",
        out_dir=tmp_path / "m1210",
    )

    claims = report["route_claims"]
    assert report["report_kind"] == "triton_tvm_m12_10_route_cleanup"
    assert report["status"] == "passed"
    assert claims[BACKEND_PASS_MODE]["backend_general"] is True
    assert claims[BACKEND_PASS_MODE]["counts_for_triton_tvm_backend_progress"] is True
    assert claims[FUSED_QKV_MODE]["model_specific_artifact"] is True
    assert claims[FUSED_QKV_MODE]["route_diagnostic_label"] == "model_specific_diagnostic_only"
    assert claims[FUSED_QKV_MODE]["counts_for_triton_tvm_backend_progress"] is False
    assert report["next_default_action"]["action"] == "m12_11_backend_general_replan"
    assert report["performance_ready_e2e"] is False
    assert report["completion_gate"]["m12_complete"] is False
    assert (tmp_path / "m1210" / "report.json").exists()


def test_m1211_backend_general_replan_freezes_real_tl_dot_route(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    m129 = build_vit_provider_runtime_optimization_report(
        seed=0,
        warmup=5,
        repeat=20,
        m128_report={"status": "passed", "m12_8_decision": {"decision": "return_to_profile"}},
        m128_report_path=tmp_path / "m128.json",
        latency_ms=_m129_synthetic_latency(current_p50=0.840, backend_p50=0.820),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
            FUSED_QKV_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
            FUSED_QKV_MODE: provider_mix,
        },
        backend_pass_summary=_m129_backend_pass_summary(transformed=7, expected=7),
        provider_overhead_profile={},
        out_dir=None,
    )
    m1210 = build_vit_route_cleanup_report(
        m129_report=m129,
        m129_report_path=tmp_path / "m129.json",
        out_dir=None,
    )
    report = build_backend_general_replan_report(
        m1210_report=m1210,
        m1210_report_path=tmp_path / "m1210.json",
        out_dir=tmp_path / "m1211",
    )

    matrix = report["route_matrix"]
    assert report["report_kind"] == "triton_tvm_m12_11_backend_general_replan"
    assert report["milestone"] == "M12.11"
    assert report["status"] == "route_frozen"
    assert report["primary_route"] == "real_tl_dot_bridge"
    assert report["auxiliary_order"] == ["runtime_overhead", "tt_dot_schedule_handoff"]
    assert report["next_default_action"]["action"] == "m12_12_real_tl_dot_bridge_harness"
    assert report["invariants"]["status"] == "passed"
    assert matrix["real_tl_dot_bridge"]["next_milestone"] == "M12.12"
    assert matrix["runtime_overhead"]["next_milestone"] == "M12.13"
    assert matrix["tt_dot_schedule_handoff"]["next_milestone"] == "M12.14"
    assert (
        matrix["m129_fused_qkv_artifact"]["readiness_disposition"]
        == "rejected_for_backend_readiness"
    )
    assert matrix["m129_fused_qkv_artifact"]["route_diagnostic_label"] == (
        "model_specific_diagnostic_only"
    )
    assert (
        matrix["m129_fused_qkv_artifact"]["counts_for_triton_tvm_backend_progress"]
        is False
    )
    assert report["performance_ready_e2e"] is False
    assert report["p2_passed"] is False
    assert report["backend_general_complete"] is False
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert (tmp_path / "m1211" / "report.json").exists()


def test_m1212_real_tl_dot_bridge_report_invariants(tmp_path):
    accepted_case = {
        "case_name": "real_jit_tl_dot_fp16_row_major",
        "lower_to_ttir_ok": True,
        "ttir_contains_tt_dot": True,
        "reader_dot_snapshot": {
            "name": "tt.dot",
            "results": ["acc"],
            "operands": ["a", "b", "zero"],
            "attrs": {"inputPrecision": "tf32"},
            "result_types": [{"raw": "tensor<8x8xf32>", "dtype": "float32", "shape": [8, 8]}],
        },
        "matmul_source_kind": REAL_JIT_TT_DOT_SOURCE_KIND,
        "matmul_contract_ok": True,
        "implementation_kind": "native_tir_schedule",
        "schedule_id": NATIVE_TIR_MATMUL_SCHEDULE_ID,
        "observed_reason": "",
    }
    negative_cases = [
        {
            "case_name": case_name,
            "expected_reason": reason,
            "observed_reason": reason,
            "status": "passed",
        }
        for case_name, reason in NEGATIVE_CASES.items()
    ]

    report = build_real_tl_dot_bridge_report(
        accepted_case=accepted_case,
        negative_cases=negative_cases,
        out_dir=tmp_path / "m1212",
    )

    taxonomy = report["source_taxonomy"]
    assert report["report_kind"] == "triton_tvm_m12_12_real_tl_dot_bridge_harness"
    assert report["milestone"] == "M12.12"
    assert report["status"] == "passed"
    assert report["invariants"]["status"] == "passed"
    assert taxonomy["synthetic_tt_dot"]["source_kind"] == "tt_dot"
    assert taxonomy["wrapper_extern_gemm"]["source_kind"] == "wrapper_extern_gemm"
    assert taxonomy["real_jit_tt_dot"]["source_kind"] == REAL_JIT_TT_DOT_SOURCE_KIND
    assert report["accepted_case"]["matmul_source_kind"] == REAL_JIT_TT_DOT_SOURCE_KIND
    assert {case["case_name"] for case in report["negative_cases"]} == set(NEGATIVE_CASES)
    assert report["performance_claim"] is False
    assert report["p2_passed"] is False
    assert report["backend_general_complete"] is False
    assert report["route_boundaries"]["no_vit_wrapper_replay"] is True
    assert report["route_boundaries"]["no_fused_qkv_artifact"] is True
    assert (tmp_path / "m1212" / "report.json").exists()


def test_m1213_runtime_overhead_general_measurement_report_invariants(tmp_path):
    report = build_runtime_overhead_general_measurement_report(
        seed=0,
        warmup=5,
        repeat=20,
        n=1024,
        block=128,
        m1211_report=_m1213_synthetic_m1211_report(),
        m1211_report_path=tmp_path / "m1211.json",
        m1212_report=_m1213_synthetic_m1212_report(),
        m1212_report_path=tmp_path / "m1212.json",
        measurements=_m1213_synthetic_measurements(),
        correctness={"allclose": True, "baseline": "torch_cuda_pointwise_dual_store"},
        out_dir=tmp_path / "m1213",
    )

    assert report["report_kind"] == "triton_tvm_m12_13_runtime_overhead_general_measurement"
    assert report["milestone"] == "M12.13"
    assert report["status"] == "passed"
    assert report["invariants"]["status"] == "passed"
    assert report["measurement_surface"]["contract"] == "pointwise_flat"
    assert report["measurement_surface"]["model_specific"] is False
    assert report["measurement_surface"]["uses_vit_wrapper"] is False
    assert set(report["measurements"]) == set(REQUIRED_MEASUREMENT_BUCKETS)
    assert report["attribution"]["dominant_bucket"] in REQUIRED_MEASUREMENT_BUCKETS
    assert report["route_boundaries"]["no_vit_wrapper_replay"] is True
    assert report["route_boundaries"]["no_fused_qkv_artifact"] is True
    assert report["route_boundaries"]["no_schedule_handoff_claim"] is True
    assert report["next_default_action"]["action"] == "m12_14_real_tt_dot_schedule_handoff"
    assert report["performance_ready_e2e"] is False
    assert report["performance_claim"] is False
    assert report["p2_passed"] is False
    assert report["backend_general_complete"] is False
    assert report["schedule_handoff_claim"] is False
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["completion_gate"]["m12_complete"] is False
    assert (tmp_path / "m1213" / "report.json").exists()
    assert "M12.13 Runtime Overhead General Measurement" in (
        tmp_path / "m1213" / "report.md"
    ).read_text()


def test_m1213_runtime_overhead_general_measurement_reports_invariant_failures(tmp_path):
    measurements = _m1213_synthetic_measurements()
    measurements.pop("dlpack_conversion_ms")
    report = build_runtime_overhead_general_measurement_report(
        seed=0,
        warmup=5,
        repeat=20,
        n=1024,
        block=128,
        m1211_report=_m1213_synthetic_m1211_report(),
        m1211_report_path=tmp_path / "m1211.json",
        m1212_report={"status": "failed", "invariants": {"status": "failed"}},
        m1212_report_path=tmp_path / "m1212.json",
        measurements=measurements,
        correctness={"allclose": False},
        out_dir=None,
    )
    report["performance_claim"] = True
    report["route_boundaries"]["no_fused_qkv_artifact"] = False
    report["invariants"] = _m1213_report_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert report["invariants"]["status"] == "failed"
    assert "m12_13_m12_12_not_passed" in failures
    assert "m12_13_m12_12_invariants_not_passed" in failures
    assert "m12_13_missing_measurement_dlpack_conversion_ms" in failures
    assert "m12_13_correctness_allclose_not_true" in failures
    assert "m12_13_forbidden_claim_performance_claim" in failures
    assert "m12_13_boundary_violation_no_fused_qkv_artifact" in failures


def test_m1214_real_tt_dot_schedule_handoff_report_invariants(tmp_path):
    report = build_real_tt_dot_schedule_handoff_report(
        m1211_report=_m1213_synthetic_m1211_report(),
        m1211_report_path=tmp_path / "m1211.json",
        m1212_report=_m1213_synthetic_m1212_report(),
        m1212_report_path=tmp_path / "m1212.json",
        m1213_report=_m1214_synthetic_m1213_report(),
        m1213_report_path=tmp_path / "m1213.json",
        schedule_handoff_cases=_m1214_synthetic_handoff_cases(),
        out_dir=tmp_path / "m1214",
    )

    assert report["report_kind"] == "triton_tvm_m12_14_real_tt_dot_schedule_handoff"
    assert report["milestone"] == "M12.14"
    assert report["status"] == "passed"
    assert report["invariants"]["status"] == "passed"
    assert report["handoff_surface"]["source_kind"] == REAL_JIT_TT_DOT_SOURCE_KIND
    assert report["handoff_surface"]["contract"] == "matmul_minimal"
    assert report["handoff_surface"]["mutates_schedule_registry"] is False
    cases = {case["case_name"]: case for case in report["schedule_handoff_cases"]}
    assert cases[TENSORCORE_CASE]["schedule_id"] == TENSORCORE_TIR_MATMUL_SCHEDULE_ID
    assert cases[TENSORCORE_CASE]["counts_as_handoff_readiness"] is True
    assert cases[TILED_CASE]["schedule_id"] == TILED_TIR_MATMUL_SCHEDULE_ID
    assert cases[TILED_CASE]["counts_as_handoff_readiness"] is True
    wrapper = report["wrapper_specific_schedule_boundary"]
    assert wrapper["schedule_id"] == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
    assert wrapper["selected_for_real_tt_dot"] is False
    assert wrapper["counts_as_handoff_readiness"] is False
    assert report["route_boundaries"]["no_wrapper_specific_schedule_readiness"] is True
    assert report["real_tt_dot_schedule_handoff_complete"] is True
    assert report["next_default_action"]["action"] == M1214_NEXT_DEFAULT_ACTION
    assert report["performance_ready_e2e"] is False
    assert report["performance_claim"] is False
    assert report["p2_passed"] is False
    assert report["backend_general_complete"] is False
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["completion_gate"]["m12_complete"] is False
    assert (tmp_path / "m1214" / "report.json").exists()
    assert "M12.14 Real `tt.dot` Schedule Handoff" in (
        tmp_path / "m1214" / "report.md"
    ).read_text()


def test_m1214_real_tt_dot_schedule_handoff_reports_invariant_failures(tmp_path):
    cases = _m1214_synthetic_handoff_cases()
    cases[0]["schedule_id"] = NATIVE_TIR_MATMUL_SCHEDULE_ID
    cases[0]["selected_schedule_id"] = NATIVE_TIR_MATMUL_SCHEDULE_ID
    cases[0]["counts_as_handoff_readiness"] = False
    report = build_real_tt_dot_schedule_handoff_report(
        m1211_report=_m1213_synthetic_m1211_report(),
        m1211_report_path=tmp_path / "m1211.json",
        m1212_report={"status": "failed", "invariants": {"status": "failed"}},
        m1212_report_path=tmp_path / "m1212.json",
        m1213_report={"status": "failed", "invariants": {"status": "failed"}},
        m1213_report_path=tmp_path / "m1213.json",
        schedule_handoff_cases=cases,
        out_dir=None,
    )
    report["performance_claim"] = True
    report["wrapper_specific_schedule_boundary"]["selected_for_real_tt_dot"] = True
    report["wrapper_specific_schedule_boundary"]["counts_as_handoff_readiness"] = True
    for candidate in report["reusable_tt_dot_schedule_candidates"]:
        if candidate["schedule_id"] == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
            candidate["selected_for_real_tt_dot"] = True
            candidate["counts_as_handoff_readiness"] = True
    report["invariants"] = _m1214_report_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert report["invariants"]["status"] == "failed"
    assert "m12_14_m12_12_not_passed" in failures
    assert "m12_14_m12_12_invariants_not_passed" in failures
    assert "m12_14_m12_13_not_passed" in failures
    assert "m12_14_m12_13_invariants_not_passed" in failures
    assert "m12_14_tensorcore_handoff_missing" in failures
    assert "m12_14_wrapper_schedule_selected_for_real_tt_dot" in failures
    assert "m12_14_wrapper_schedule_counts_as_handoff" in failures
    assert "m12_14_wrapper_candidate_selected" in failures
    assert "m12_14_wrapper_candidate_counts_as_handoff" in failures
    assert "m12_14_forbidden_claim_performance_claim" in failures


def test_m1215_vit_p2_reentry_closes_m12_on_p2_pass(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_vit_p2_reentry_report(
        seed=0,
        warmup=5,
        repeat=20,
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(),
        m1210_report_path=tmp_path / "m1210.json",
        m1214_report=_m1215_synthetic_m1214_report(),
        m1214_report_path=tmp_path / "m1214.json",
        latency_ms=_m1215_synthetic_latency(current_p50=0.840, backend_p50=0.650),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
        },
        backend_pass_summary=_m1215_backend_pass_summary(transformed=7, expected=7),
        out_dir=tmp_path / "m1215",
    )

    assert report["report_kind"] == "triton_tvm_m12_15_vit_p2_reentry"
    assert report["milestone"] == "M12.15"
    assert report["status"] == "passed"
    assert report["p2_eligible_mode"] == P2_ELIGIBLE_MODE
    assert report["p2"]["passed"] is True
    assert report["p2_passed"] is True
    assert report["performance_ready_e2e"] is True
    assert report["performance_claim"] is True
    assert report["completion_gate"]["m12_complete"] is True
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["next_default_action"]["action"] == NEXT_ACTION_AFTER_P2_PASS
    assert (tmp_path / "m1215" / "report.json").exists()
    assert "M12.15 Fixed-Shape ViT P2 Re-entry" in (
        tmp_path / "m1215" / "report.md"
    ).read_text()


def test_m1215_vit_p2_reentry_p2_miss_keeps_readiness_false(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_vit_p2_reentry_report(
        seed=0,
        warmup=5,
        repeat=20,
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(),
        m1210_report_path=tmp_path / "m1210.json",
        m1214_report=_m1215_synthetic_m1214_report(),
        m1214_report_path=tmp_path / "m1214.json",
        latency_ms=_m1215_synthetic_latency(current_p50=0.850, backend_p50=0.820),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
        },
        backend_pass_summary=_m1215_backend_pass_summary(transformed=7, expected=7),
        out_dir=None,
    )

    assert report["status"] == "passed"
    assert report["p2"]["passed"] is False
    assert report["p2_passed"] is False
    assert report["performance_ready_e2e"] is False
    assert report["performance_claim"] is False
    assert report["completion_gate"]["m12_complete"] is False
    assert report["next_default_action"]["action"] == NEXT_ACTION_AFTER_P2_MISS


def test_m1215_vit_p2_reentry_fused_qkv_cannot_count_for_p2(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    latency = _m1215_synthetic_latency(current_p50=0.850, backend_p50=0.820)
    latency[FUSED_QKV_MODE] = _m127_latency_record([0.500, 0.510, 0.520])
    report = build_vit_p2_reentry_report(
        seed=0,
        warmup=5,
        repeat=20,
        m129_report=_m1215_synthetic_m129_report(
            best_mode=FUSED_QKV_MODE,
            best_mode_p2_passed=True,
        ),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(),
        m1210_report_path=tmp_path / "m1210.json",
        m1214_report=_m1215_synthetic_m1214_report(),
        m1214_report_path=tmp_path / "m1214.json",
        latency_ms=latency,
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
        },
        backend_pass_summary=_m1215_backend_pass_summary(transformed=7, expected=7),
        out_dir=None,
    )

    assert report["status"] == "passed"
    assert report["p2_eligible_mode"] == BACKEND_PASS_MODE
    assert report["p2"]["passed"] is False
    assert report["performance_ready_e2e"] is False
    assert report["diagnostic_exclusions"][FUSED_QKV_MODE]["excluded_from_p2"] is True
    assert (
        report["diagnostic_exclusions"][FUSED_QKV_MODE][
            "counts_for_triton_tvm_backend_progress"
        ]
        is False
    )


def test_m1215_vit_p2_reentry_blocks_on_m1214_source_failure(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    source_m1214 = _m1215_synthetic_m1214_report()
    source_m1214["status"] = "failed"
    source_m1214["invariants"] = {"status": "failed", "invariant_failures": []}
    report = build_vit_p2_reentry_report(
        seed=0,
        warmup=5,
        repeat=20,
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(),
        m1210_report_path=tmp_path / "m1210.json",
        m1214_report=source_m1214,
        m1214_report_path=tmp_path / "m1214.json",
        latency_ms=_m1215_synthetic_latency(current_p50=0.840, backend_p50=0.650),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": True},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
        },
        backend_pass_summary=_m1215_backend_pass_summary(transformed=7, expected=7),
        out_dir=None,
    )

    failures = set(report["invariants"]["invariant_failures"])
    assert report["status"] == "failed"
    assert report["p2_passed"] is False
    assert report["performance_ready_e2e"] is False
    assert "m12_15_m12_14_not_passed" in failures
    assert "m12_15_m12_14_invariants_not_passed" in failures


def test_m1215_vit_p2_reentry_reports_invariant_failures(tmp_path):
    provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    broken_provider_mix = json.loads(json.dumps(provider_mix))
    broken_provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6
    report = build_vit_p2_reentry_report(
        seed=0,
        warmup=0,
        repeat=20,
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(fused_counts_for_progress=True),
        m1210_report_path=tmp_path / "m1210.json",
        m1214_report=_m1215_synthetic_m1214_report(),
        m1214_report_path=tmp_path / "m1214.json",
        latency_ms=_m1215_synthetic_latency(current_p50=0.840, backend_p50=0.650),
        correctness={
            CURRENT_MODE: {"allclose": True},
            BACKEND_PASS_MODE: {"allclose": False},
        },
        provider_mix={
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: broken_provider_mix,
        },
        backend_pass_summary=_m1215_backend_pass_summary(transformed=6, expected=7),
        out_dir=None,
    )
    report["host_staging_bytes"] = 1
    report["strict_full_tvm_native"] = True
    report["full_tvm_runnable_models"] = 1
    report["backend_general_complete"] = True
    report["performance_ready_e2e"] = True
    report["performance_claim"] = True
    report["completion_gate"]["m12_complete"] = True
    report["completion_gate"]["strict_full_tvm_native"] = True
    report["completion_gate"]["full_tvm_runnable_models"] = 1
    report["invariants"] = _m1215_report_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert report["invariants"]["status"] == "failed"
    assert "m12_15_warmed_cache_required" in failures
    assert "m12_15_host_staging_nonzero" in failures
    assert "m12_15_correctness_failed_triton_tvm_e2e_m129_backend_pass" in failures
    assert "m12_15_provider_mix_incomplete_triton_tvm_e2e_m129_backend_pass" in failures
    assert "m12_15_backend_pass_did_not_transform_all_calls" in failures
    assert "m12_15_fused_qkv_counts_for_backend_progress" in failures
    assert "m12_15_strict_native_claim_present" in failures
    assert "m12_15_full_tvm_runnable_claim_present" in failures
    assert "m12_15_backend_general_complete_claim_present" in failures
    assert "m12_15_performance_ready_without_p2" in failures
    assert "m12_15_performance_claim_without_p2" in failures
    assert "m12_15_m12_complete_without_p2" in failures
    assert "m12_15_completion_strict_native_claim_present" in failures
    assert "m12_15_completion_full_runnable_claim_present" in failures


def test_m1216_backend_general_p2_gap_analysis_selects_provider_recovery(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_backend_general_p2_gap_analysis_report(
        seed=0,
        m127_report=_m1216_synthetic_m127_report(),
        m127_report_path=tmp_path / "m127.json",
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(),
        m1210_report_path=tmp_path / "m1210.json",
        m1213_report=_m1216_synthetic_m1213_report(),
        m1213_report_path=tmp_path / "m1213.json",
        m1214_report=_m1215_synthetic_m1214_report(),
        m1214_report_path=tmp_path / "m1214.json",
        m1215_report=_m1216_synthetic_m1215_report(provider_mix=provider_mix),
        m1215_report_path=tmp_path / "m1215.json",
        out_dir=tmp_path / "m1216",
    )

    assert report["report_kind"] == "triton_tvm_m12_16_backend_general_p2_gap_analysis"
    assert report["milestone"] == "M12.16"
    assert report["status"] == "passed"
    assert report["p2_passed"] is False
    assert report["performance_ready_e2e"] is False
    assert report["completion_gate"]["m12_complete"] is False
    assert report["summary"]["dominant_measured_bucket"] == "native_tvm_matmul_provider"
    assert report["p2_gap_budget"]["p50_excess_ms"] > 0.0
    assert report["selected_next_slice"]["action"] == NEXT_ACTION_AFTER_M1216
    assert report["next_default_action"]["action"] == NEXT_ACTION_AFTER_M1216
    assert {bucket["bucket_id"] for bucket in report["gap_buckets"]} == set(
        REQUIRED_ANALYSIS_BUCKETS
    )
    assert (tmp_path / "m1216" / "report.json").exists()
    assert "M12.16 Backend-General P2 Gap Analysis" in (
        tmp_path / "m1216" / "report.md"
    ).read_text()


def test_m1216_backend_general_p2_gap_analysis_blocks_p2_pass_source(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    source_m1215 = _m1216_synthetic_m1215_report(provider_mix=provider_mix)
    source_m1215["p2_passed"] = True
    source_m1215["p2"]["passed"] = True
    source_m1215["performance_ready_e2e"] = True
    source_m1215["next_default_action"] = {"action": "close_m12_p2_passed"}
    report = build_backend_general_p2_gap_analysis_report(
        seed=0,
        m127_report=_m1216_synthetic_m127_report(),
        m127_report_path=tmp_path / "m127.json",
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(),
        m1210_report_path=tmp_path / "m1210.json",
        m1213_report=_m1216_synthetic_m1213_report(),
        m1213_report_path=tmp_path / "m1213.json",
        m1214_report=_m1215_synthetic_m1214_report(),
        m1214_report_path=tmp_path / "m1214.json",
        m1215_report=source_m1215,
        m1215_report_path=tmp_path / "m1215.json",
        out_dir=None,
    )

    failures = set(report["invariants"]["invariant_failures"])
    assert report["status"] == "failed"
    assert "m12_16_m12_15_not_p2_miss" in failures
    assert "m12_16_m12_15_performance_ready_unexpected" in failures
    assert "m12_16_m12_15_next_action_mismatch" in failures


def test_m1216_backend_general_p2_gap_analysis_reports_invariant_failures(tmp_path):
    provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6
    source_m1214 = _m1215_synthetic_m1214_report()
    source_m1214["p2_passed"] = True
    source_m1214["performance_ready_e2e"] = True
    report = build_backend_general_p2_gap_analysis_report(
        seed=0,
        m127_report=_m1216_synthetic_m127_report(optimization_validated=False),
        m127_report_path=tmp_path / "m127.json",
        m129_report=_m1215_synthetic_m129_report(),
        m129_report_path=tmp_path / "m129.json",
        m1210_report=_m1215_synthetic_m1210_report(fused_counts_for_progress=True),
        m1210_report_path=tmp_path / "m1210.json",
        m1213_report=_m1216_synthetic_m1213_report(runtime_complete=False),
        m1213_report_path=tmp_path / "m1213.json",
        m1214_report=source_m1214,
        m1214_report_path=tmp_path / "m1214.json",
        m1215_report=_m1216_synthetic_m1215_report(provider_mix=provider_mix),
        m1215_report_path=tmp_path / "m1215.json",
        out_dir=None,
    )
    report["gap_buckets"] = [
        bucket
        for bucket in report["gap_buckets"]
        if bucket["bucket_id"] != "native_tvm_matmul_provider"
    ]
    report["p2_gap_budget"]["ratio_p50"] = 999.0
    report["analysis_scope"]["counts_m12_14_as_e2e_performance_evidence"] = True
    report["backend_pass"]["counts_for_triton_tvm_backend_progress"] = False
    for bucket in report["gap_buckets"]:
        if bucket["bucket_id"] == "remaining_e2e_residual":
            bucket["actionable"] = True
        if bucket["bucket_id"] == "torch_inductor_captured_harness":
            bucket["estimated_ms"] = 0.1
    report["selected_next_slice"]["primary_bucket"] = "torch_inductor_captured_harness"
    report["selected_next_slice"]["model_specific"] = True
    report["performance_ready_e2e"] = True
    report["performance_claim"] = True
    report["p2_passed"] = True
    report["backend_general_complete"] = True
    report["strict_full_tvm_native"] = True
    report["full_tvm_runnable_models"] = 1
    report["completion_gate"]["m12_complete"] = True
    report["completion_gate"]["strict_full_tvm_native"] = True
    report["completion_gate"]["full_tvm_runnable_models"] = 1
    report["invariants"] = _m1216_report_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert "m12_16_m12_7_optimization_not_validated" in failures
    assert "m12_16_fused_qkv_counts_for_backend_progress" in failures
    assert "m12_16_m12_13_runtime_measurement_incomplete" in failures
    assert "m12_16_m12_14_treated_as_p2" in failures
    assert "m12_16_m12_14_treated_as_performance_ready" in failures
    assert "m12_16_provider_counts_mismatch" in failures
    assert "m12_16_scope_counts_m12_14_as_e2e" in failures
    assert "m12_16_gap_budget_p50_ratio_mismatch" in failures
    assert "m12_16_backend_pass_not_backend_progress" in failures
    assert "m12_16_missing_gap_bucket_native_tvm_matmul_provider" in failures
    assert "m12_16_residual_bucket_actionable" in failures
    assert (
        "m12_16_count_only_bucket_has_estimate_torch_inductor_captured_harness"
        in failures
    )
    assert "m12_16_wrong_primary_bucket" in failures
    assert "m12_16_selected_slice_model_specific" in failures
    assert "m12_16_forbidden_claim_performance_ready_e2e" in failures
    assert "m12_16_forbidden_claim_performance_claim" in failures
    assert "m12_16_forbidden_claim_p2_passed" in failures
    assert "m12_16_forbidden_claim_backend_general_complete" in failures
    assert "m12_16_forbidden_claim_strict_full_tvm_native" in failures
    assert "m12_16_full_tvm_runnable_claim_present" in failures
    assert "m12_16_m12_complete_claim_present" in failures
    assert "m12_16_completion_strict_native_claim_present" in failures
    assert "m12_16_completion_full_runnable_claim_present" in failures


def test_m12p_profile_optimization_loop_iteration_records_prebound_dispatch(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_m12p_profile_optimization_loop_report(
        seed=0,
        warmup=5,
        repeat=20,
        m1215_report=_m1216_synthetic_m1215_report(provider_mix=provider_mix),
        m1215_report_path=tmp_path / "m1215.json",
        m1216_report=_m12p_synthetic_m1216_report(),
        m1216_report_path=tmp_path / "m1216.json",
        latency_ms=_m12p_synthetic_latency(candidate_p50=0.790),
        correctness={
            BACKEND_PASS_MODE: {"allclose": True},
            M12P_PREBOUND_MODE: {"allclose": True},
        },
        provider_mix={
            BACKEND_PASS_MODE: provider_mix,
            M12P_PREBOUND_MODE: provider_mix,
        },
        backend_pass_summary=_m1216_synthetic_backend_pass_summary(),
        profile=_m12p_synthetic_profile(),
        out_dir=tmp_path / "m12p",
    )

    assert report["report_kind"] == "triton_tvm_m12_p_profile_optimization_loop"
    assert report["milestone"] == "M12.P"
    assert report["iteration_id"] == M12P_ITERATION_ID
    assert report["status"] == "passed"
    assert report["candidate_mode"] == M12P_PREBOUND_MODE
    assert report["selected_primary_bucket"] == "native_tvm_matmul_runtime_glue"
    assert report["optimization_scope"]["optimization"] == "prebound_packed_call"
    assert report["p2_passed"] is False
    assert report["performance_ready_e2e"] is False
    assert report["performance_claim"] is False
    assert report["backend_general_complete"] is False
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["next_default_action"]["action"] == M12P_NEXT_ACTION_CONTINUE
    assert report["next_default_action"]["next_loop_target"] == "native_tvm_matmul_kernel"
    assert report["summary"]["recovered_p50_ms"] > 0.0
    assert (tmp_path / "m12p" / "report.json").exists()
    assert "M12.P Profile Optimization Loop Iteration" in (
        tmp_path / "m12p" / "report.md"
    ).read_text()


def test_m12p_profile_optimization_loop_closes_only_on_unchanged_p2_pass(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_m12p_profile_optimization_loop_report(
        seed=0,
        warmup=5,
        repeat=20,
        m1215_report=_m1216_synthetic_m1215_report(provider_mix=provider_mix),
        m1215_report_path=tmp_path / "m1215.json",
        m1216_report=_m12p_synthetic_m1216_report(),
        m1216_report_path=tmp_path / "m1216.json",
        latency_ms=_m12p_synthetic_latency(candidate_p50=0.650),
        correctness={
            BACKEND_PASS_MODE: {"allclose": True},
            M12P_PREBOUND_MODE: {"allclose": True},
        },
        provider_mix={
            BACKEND_PASS_MODE: provider_mix,
            M12P_PREBOUND_MODE: provider_mix,
        },
        backend_pass_summary=_m1216_synthetic_backend_pass_summary(),
        profile=_m12p_synthetic_profile(),
        out_dir=None,
    )

    assert report["status"] == "passed"
    assert report["p2_passed"] is True
    assert report["performance_ready_e2e"] is True
    assert report["performance_claim"] is True
    assert report["backend_general_complete"] is True
    assert report["completion_gate"]["m12_complete"] is True
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["next_default_action"]["action"] == "close_m12_p2_passed"


def test_m12p_profile_optimization_loop_second_iteration_uses_previous_candidate(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_m12p_profile_optimization_loop_report(
        seed=0,
        warmup=5,
        repeat=20,
        m1215_report=_m1216_synthetic_m1215_report(provider_mix=provider_mix),
        m1215_report_path=tmp_path / "m1215.json",
        m1216_report=_m12p_synthetic_m1216_report(),
        m1216_report_path=tmp_path / "m1216.json",
        previous_iteration_report=_m12p_synthetic_previous_iteration_report(),
        previous_iteration_report_path=tmp_path / "m12p_iter_001.json",
        report_id="m12_p_iter_002_minimal_record_v1",
        iteration_id=M12P_MINIMAL_RECORD_ITERATION_ID,
        baseline_mode=M12P_PREBOUND_MODE,
        candidate_mode=M12P_MINIMAL_RECORD_MODE,
        measured_provider_modes=(M12P_PREBOUND_MODE, M12P_MINIMAL_RECORD_MODE),
        latency_ms=_m12p_synthetic_minimal_record_latency(candidate_p50=0.480),
        correctness={
            M12P_PREBOUND_MODE: {"allclose": True},
            M12P_MINIMAL_RECORD_MODE: {"allclose": True},
        },
        provider_mix={
            M12P_PREBOUND_MODE: provider_mix,
            M12P_MINIMAL_RECORD_MODE: provider_mix,
        },
        backend_pass_summary=_m1216_synthetic_backend_pass_summary(),
        profile=_m12p_synthetic_minimal_record_profile(),
        optimization_scope=_m12p_minimal_record_optimization_scope(),
        out_dir=None,
    )

    assert report["status"] == "passed"
    assert report["iteration_id"] == M12P_MINIMAL_RECORD_ITERATION_ID
    assert report["iteration_id"] != M12P_ITERATION_ID
    assert report["baseline_mode"] == M12P_PREBOUND_MODE
    assert report["candidate_mode"] == M12P_MINIMAL_RECORD_MODE
    assert report["p2_eligible_mode"] == M12P_MINIMAL_RECORD_MODE
    assert report["source_reports"]["previous_iteration_report"].endswith("m12p_iter_001.json")
    assert report["source_report_status"]["previous_iteration_id"] == M12P_ITERATION_ID
    assert report["source_report_status"]["previous_iteration_p2_passed"] is True
    assert report["optimization_scope"]["optimization"] == "prebound_minimal_record"
    assert report["optimization_scope"]["dispatch_mode"] == "prebound_minimal_record"
    assert report["profile_buckets"][2]["bucket_id"] == "native_tvm_matmul_runtime_glue"
    assert report["profile_buckets"][2]["recovered_ms"] > 0.0
    assert report["p2_passed"] is True
    assert report["backend_general_complete"] is True
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0


def test_m12p_profile_optimization_loop_reports_invariant_failures(tmp_path):
    provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    broken_provider_mix = json.loads(json.dumps(provider_mix))
    broken_provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6
    source_m1216 = _m12p_synthetic_m1216_report()
    source_m1216["selected_next_slice"]["primary_bucket"] = "remaining_e2e_residual"
    report = build_m12p_profile_optimization_loop_report(
        seed=0,
        warmup=0,
        repeat=20,
        m1215_report=_m1216_synthetic_m1215_report(provider_mix=provider_mix),
        m1215_report_path=tmp_path / "m1215.json",
        m1216_report=source_m1216,
        m1216_report_path=tmp_path / "m1216.json",
        latency_ms=_m12p_synthetic_latency(candidate_p50=0.790),
        correctness={
            BACKEND_PASS_MODE: {"allclose": True},
            M12P_PREBOUND_MODE: {"allclose": False},
        },
        provider_mix={
            BACKEND_PASS_MODE: provider_mix,
            M12P_PREBOUND_MODE: broken_provider_mix,
        },
        backend_pass_summary=_m1216_synthetic_backend_pass_summary(),
        profile={BACKEND_PASS_MODE: {"record_count": 0}, M12P_PREBOUND_MODE: {}},
        out_dir=None,
    )
    report["host_staging_bytes"] = 1
    report["optimization_scope"]["model_specific"] = True
    report["performance_ready_e2e"] = True
    report["performance_claim"] = True
    report["backend_general_complete"] = True
    report["strict_full_tvm_native"] = True
    report["full_tvm_runnable_models"] = 1
    report["completion_gate"]["m12_complete"] = True
    report["invariants"] = _m12p_report_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert "m12_p_wrong_entry_primary_bucket" in failures
    assert "m12_p_warmed_cache_required" in failures
    assert "m12_p_host_staging_nonzero" in failures
    assert "m12_p_optimization_model_specific" in failures
    assert f"m12_p_correctness_failed_{M12P_PREBOUND_MODE}" in failures
    assert f"m12_p_provider_mix_incomplete_{M12P_PREBOUND_MODE}" in failures
    assert f"m12_p_profile_record_count_{BACKEND_PASS_MODE}" in failures
    assert "m12_p_forbidden_claim_without_p2_performance_ready_e2e" in failures
    assert "m12_p_forbidden_claim_without_p2_performance_claim" in failures
    assert "m12_p_forbidden_claim_without_p2_backend_general_complete" in failures
    assert "m12_p_m12_complete_without_p2" in failures
    assert "m12_p_strict_native_claim_present" in failures
    assert "m12_p_full_tvm_runnable_claim_present" in failures


def test_m12p_residual_profile_records_residual_provider_buckets(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_m12p_residual_profile_report(
        seed=0,
        warmup=5,
        repeat=20,
        previous_iteration_report=_m12p_synthetic_minimal_record_iteration_report(),
        previous_iteration_report_path=tmp_path / "m12p_minimal_record.json",
        latency_ms=_m12p_synthetic_residual_profile_latency(candidate_p50=0.472),
        correctness={M12P_MINIMAL_RECORD_MODE: {"allclose": True}},
        provider_mix={M12P_MINIMAL_RECORD_MODE: provider_mix},
        native_matmul_profile=_m12p_synthetic_minimal_record_profile()[
            M12P_MINIMAL_RECORD_MODE
        ],
        residual_provider_profile=_m12p_synthetic_residual_provider_profile(),
        out_dir=tmp_path / "m12p_residual",
    )

    buckets = {bucket["bucket_id"]: bucket for bucket in report["profile_buckets"]}
    assert report["report_kind"] == "triton_tvm_m12_p_residual_profile"
    assert report["iteration_id"] == M12P_RESIDUAL_PROFILE_ITERATION_ID
    assert report["loop_phase"] == "profile_residual_only"
    assert report["status"] == "passed"
    assert report["p2_passed"] is True
    assert report["performance_ready_e2e"] is True
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert report["previous_iteration_snapshot"]["iteration_id"] == M12P_MINIMAL_RECORD_ITERATION_ID
    assert report["profile"]["residual_providers"]["record_count"] == 2
    assert buckets["device_torch_cuda_conv"]["candidate_ms"] == 0.031
    assert buckets["native_decomposed_attention"]["candidate_ms"] == 0.052
    assert buckets["remaining_unattributed_e2e_residual"]["estimated_p50_ms"] > 0.0
    assert report["selected_next_slice"]["action"] == "continue_residual_profile"
    assert (tmp_path / "m12p_residual" / "report.json").exists()
    assert "M12.P Residual Profile" in (
        tmp_path / "m12p_residual" / "report.md"
    ).read_text()


def test_m12p_residual_profile_reports_invariant_failures(tmp_path):
    provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    provider_mix["provider_counts"][VISION_PROVIDER_DEVICE_TORCH_CUDA] = 0
    report = build_m12p_residual_profile_report(
        seed=0,
        warmup=0,
        repeat=20,
        previous_iteration_report=_m12p_synthetic_previous_iteration_report(),
        previous_iteration_report_path=tmp_path / "m12p_prebound.json",
        latency_ms=_m12p_synthetic_residual_profile_latency(candidate_p50=0.800),
        correctness={M12P_MINIMAL_RECORD_MODE: {"allclose": False}},
        provider_mix={M12P_MINIMAL_RECORD_MODE: provider_mix},
        native_matmul_profile={"record_count": 0, "aggregate_sum_ms": {}},
        residual_provider_profile={"record_count": 0, "by_provider": {}},
        out_dir=None,
    )
    report["strict_full_tvm_native"] = True
    report["full_tvm_runnable_models"] = 1
    report["invariants"] = _m12p_residual_profile_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert "residual_profile_previous_iteration_mismatch" in failures
    assert "residual_profile_correctness_failed" in failures
    assert "residual_profile_conv_count" in failures
    assert "residual_profile_native_profile_count" in failures
    assert "residual_profile_conv_profile_count" in failures
    assert "residual_profile_attention_profile_count" in failures
    assert "residual_profile_strict_native_claim" in failures
    assert "residual_profile_full_runnable_claim" in failures


def test_m12p_provider_session_reuse_report_records_runtime_glue_recovery(tmp_path):
    provider_mix = _m126_synthetic_dashboard()["provider_mix"]
    report = build_m12p_provider_session_reuse_report(
        seed=0,
        warmup=5,
        repeat=20,
        previous_iteration_report=_m12p_synthetic_minimal_record_iteration_report(),
        previous_iteration_report_path=tmp_path / "m12p_minimal_record.json",
        residual_profile_report=_m12p_synthetic_residual_profile_report(),
        residual_profile_report_path=tmp_path / "m12p_residual.json",
        latency_ms=_m12p_synthetic_provider_session_reuse_latency(),
        correctness={
            M12P_MINIMAL_RECORD_MODE: {"allclose": True},
            M12P_SESSION_REUSE_MODE: {"allclose": True},
        },
        provider_mix={
            M12P_MINIMAL_RECORD_MODE: provider_mix,
            M12P_SESSION_REUSE_MODE: provider_mix,
        },
        native_matmul_profile=_m12p_synthetic_minimal_record_profile()[
            M12P_MINIMAL_RECORD_MODE
        ],
        residual_provider_profile=_m12p_synthetic_residual_provider_profile(),
        out_dir=tmp_path / "m12p_provider_session_reuse",
    )

    assert report["report_kind"] == "triton_tvm_m12_p_provider_session_reuse"
    assert report["iteration_id"] == M12P_PROVIDER_SESSION_REUSE_ITERATION_ID
    assert report["candidate_mode"] == M12P_SESSION_REUSE_MODE
    assert report["status"] == "passed"
    assert report["p2_passed"] is True
    assert report["optimization_scope"]["optimization"] == "provider_session_reuse"
    assert report["optimization_scope"]["changes_lowering"] is False
    assert report["optimization_scope"]["changes_schedule"] is False
    assert report["optimization_result"]["recovered_p50_ms"] > 0.0
    assert report["optimization_result"]["improved_p95"] is True
    assert report["strict_full_tvm_native"] is False
    assert report["full_tvm_runnable_models"] == 0
    assert (tmp_path / "m12p_provider_session_reuse" / "report.json").exists()
    assert "M12.P Provider Session Reuse" in (
        tmp_path / "m12p_provider_session_reuse" / "report.md"
    ).read_text()


def test_m12p_provider_session_reuse_report_invariant_failures(tmp_path):
    provider_mix = json.loads(json.dumps(_m126_synthetic_dashboard()["provider_mix"]))
    provider_mix["provider_counts"][EXTERN_GEMM_PROVIDER_NATIVE_TVM] = 6
    report = build_m12p_provider_session_reuse_report(
        seed=0,
        warmup=0,
        repeat=20,
        previous_iteration_report=_m12p_synthetic_previous_iteration_report(),
        previous_iteration_report_path=tmp_path / "m12p_prebound.json",
        residual_profile_report=None,
        residual_profile_report_path=tmp_path / "m12p_residual_missing.json",
        latency_ms=_m12p_synthetic_provider_session_reuse_latency(),
        correctness={
            M12P_MINIMAL_RECORD_MODE: {"allclose": True},
            M12P_SESSION_REUSE_MODE: {"allclose": False},
        },
        provider_mix={
            M12P_MINIMAL_RECORD_MODE: provider_mix,
            M12P_SESSION_REUSE_MODE: provider_mix,
        },
        native_matmul_profile={"record_count": 0, "aggregate_sum_ms": {}},
        residual_provider_profile={"record_count": 0, "by_provider": {}},
        out_dir=None,
    )
    report["optimization_scope"]["changes_lowering"] = True
    report["optimization_scope"]["model_specific"] = True
    report["host_staging_bytes"] = 1
    report["strict_full_tvm_native"] = True
    report["full_tvm_runnable_models"] = 1
    report["invariants"] = _m12p_provider_session_reuse_invariants(report)

    failures = set(report["invariants"]["invariant_failures"])
    assert "provider_session_reuse_previous_iteration_mismatch" in failures
    assert "provider_session_reuse_correctness_failed" in failures
    assert "provider_session_reuse_native_matmul_count" in failures
    assert "provider_session_reuse_forbidden_lowering_or_schedule_change" in failures
    assert "provider_session_reuse_model_specific" in failures
    assert "provider_session_reuse_host_staging_nonzero" in failures
    assert "provider_session_reuse_strict_native_claim" in failures
    assert "provider_session_reuse_full_runnable_claim" in failures


@tvm.testing.requires_cuda
def test_m123_vit_e2e_runner_cuda_correctness(tmp_path):
    if torch is None:
        pytest.skip("PyTorch is required for M12.3 ViT E2E runner")
    pytest.importorskip("transformers")

    report = run_vit_fixed_shape_e2e_runner(out_dir=tmp_path, seed=0)

    assert report["status"] == "passed"
    assert report["correctness"]["allclose"] is True
    assert report["p0"]["passed"] is True
    assert report["provider_mix"]["provider_counts"]["native_tvm_matmul"] == 7
    assert report["provider_mix"]["wrapper_matmul"]["extern_gemm"] == 4
    assert report["provider_mix"]["wrapper_matmul"]["extern_addmm_bias"] == 3
    assert report["provider_mix"]["wrapper_attention"]["provider_kind"] == "native_decomposed"
    assert report["provider_mix"]["wrapper_convolution"]["provider_kind"] == "device_torch_cuda"
    assert report["host_staging_bytes"] == 0
    assert report["performance_ready_e2e"] is False
    assert report["full_tvm_runnable_models"] == 0


def test_m96_provider_enabled_report_keeps_captured_kernel_counts_stable():
    report = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny",
                "triton_vit_0",
                make_report_status(ok=True, bucket="translated"),
            )
        ],
        [
            _model_record(
                "vit",
                "vit_tiny",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[
                    _extern_call(
                        "vit",
                        "vit_tiny",
                        "extern_kernels.mm",
                        "extern_gemm",
                        matmul_source_kind="wrapper_extern_gemm",
                        matmul_m=16,
                        matmul_n=64,
                        matmul_k=64,
                        matmul_contract="matmul_minimal",
                        matmul_contract_ok=True,
                        matmul_epilogue_kind="none",
                        implementation_kind="extern_gemm",
                        extern_symbol="extern_kernels.mm",
                        extern_packed_func=EXTERN_GEMM_PACKED_FUNC,
                        extern_runtime_kind="runtime_provider",
                        extern_runtime_replacement=(
                            EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
                        ),
                        extern_runtime_replacement_available=True,
                        extern_gemm_runtime_status=(
                            EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
                        ),
                        extern_gemm_provider_kind=(
                            EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
                        ),
                        extern_gemm_provider_abi_version=EXTERN_GEMM_PROVIDER_ABI_VERSION,
                        extern_gemm_runtime_claim=EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                        extern_gemm_performance_claim=False,
                        extern_gemm_uses_host_staging=True,
                    )
                ],
            )
        ],
        generated_at="2026-05-24T00:00:00+00:00",
    )

    assert report["model_summary"]["total_kernels"] == 1
    assert report["model_summary"]["translated_kernels"] == 1
    assert report["model_summary"]["fallback_kernels"] == 0
    assert report["model_summary"]["full_tvm_runnable_models"] == 0
    assert report["summary"]["status_buckets"] == {"translated": 1}
    runtime = report["pre_m9"]["m96_extern_gemm_runtime"]
    assert runtime["extern_gemm_runtime_resolved_count"] == 1
    assert runtime["status_counts"] == {
        EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED: 1
    }
    assert runtime["provider_counts"] == {
        EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED: 1
    }
    assert report["pre_m9"]["m9_materialized_artifact_candidates"] == []
    assert report["pre_m9"]["m9_entry_debt"] == []


def test_m6_zero_kernel_model_is_a_stable_collection_blocker():
    report = build_model_corpus_report(
        [],
        [
            {
                "model_family": "vit",
                "model_case": "vit_zero",
                "status": "zero_kernels",
                "kernel_count": 0,
                "translated_kernels": 0,
                "native_fallback_kernels": 0,
                "full_tvm_runnable": False,
            }
        ],
        errors=[
            {
                "model_family": "vit",
                "model_case": "vit_zero",
                "case_name": "vit_zero",
                "kernel_name": "",
                "bucket": "collection_error",
                "fallback_reason": "zero_triton_kernels",
                "error_type": "RuntimeError",
                "message": "no captured Triton kernels",
            }
        ],
        generated_at="2026-05-23T00:00:00+00:00",
    )

    assert report["model_summary"]["zero_kernel_models"] == 1
    assert report["full_tvm_runnable"] is False
    assert report["collection_errors"][0]["fallback_reason"] == "zero_triton_kernels"
    assert report["blockers"][0]["model_error_count"] == 1
    assert report["blockers"][0]["models"] == ["vit_zero"]


def test_m65_report_diff_ignores_timestamps_and_paths():
    before = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny",
                "triton_vit_0",
                make_report_status(ok=True, bucket="translated"),
                kernel_source_path="/tmp/a/kernel.py",
            )
        ],
        [_model_record("vit", "vit_tiny", kernel_count=1, translated=1, fallback=0)],
        generated_at="2026-05-23T00:00:00+00:00",
    )
    same_after = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny",
                "triton_vit_0",
                make_report_status(ok=True, bucket="translated"),
                kernel_source_path="/different/path/kernel.py",
            )
        ],
        [_model_record("vit", "vit_tiny", kernel_count=1, translated=1, fallback=0)],
        generated_at="2026-05-24T00:00:00+00:00",
    )

    stable_diff = diff_capability_reports(before, same_after)
    assert stable_diff["bucket_delta"] == {}
    assert stable_diff["blocker_delta"] == {}
    assert stable_diff["extern_family_delta"] == {}
    assert stable_diff["translated_delta"] == 0

    regressed = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny",
                "triton_vit_0",
                make_report_status(
                    ok=False,
                    bucket="unsupported_ttir_op",
                    fallback_reason="unsupported_ttir_op",
                    message="tt.dot",
                ),
                op_counts={"tt.dot": 1},
            )
        ],
        [_model_record("vit", "vit_tiny", kernel_count=1, translated=0, fallback=1)],
        generated_at="2026-05-24T00:00:00+00:00",
    )
    regression_diff = diff_capability_reports(before, regressed)
    assert regression_diff["bucket_delta"] == {
        "translated": -1,
        "unsupported_ttir_op": 1,
    }
    assert regression_diff["translated_delta"] == -1
    assert regression_diff["blocker_delta"] == {
        "unsupported_ttir_op|unsupported_ttir_op|matmul_dot": 1
    }
    assert regression_diff["full_tvm_runnable_delta"] == -1

    after_with_extern = build_model_corpus_report(
        [
            _kernel_record(
                "vit",
                "vit_tiny",
                "triton_vit_0",
                make_report_status(ok=True, bucket="translated"),
            )
        ],
        [
            _model_record(
                "vit",
                "vit_tiny",
                kernel_count=1,
                translated=1,
                fallback=0,
                extern_calls=[
                    _extern_call(
                        "vit",
                        "vit_tiny",
                        "extern_kernels.mm",
                        "extern_gemm",
                        matmul_source_kind="wrapper_extern_gemm",
                        matmul_m=16,
                        matmul_n=64,
                        matmul_k=64,
                        matmul_contract="matmul_minimal",
                        matmul_contract_ok=True,
                        matmul_epilogue_kind="none",
                        implementation_kind="extern_gemm",
                        extern_symbol="extern_kernels.mm",
                        extern_packed_func=EXTERN_GEMM_PACKED_FUNC,
                        extern_runtime_kind=EXTERN_GEMM_RUNTIME_KIND,
                        extern_runtime_replacement=EXTERN_GEMM_RUNTIME_REPLACEMENT,
                        extern_runtime_replacement_available=False,
                    )
                ],
            )
        ],
        generated_at="2026-05-24T00:00:00+00:00",
    )
    extern_diff = diff_capability_reports(before, after_with_extern)
    assert extern_diff["extern_family_delta"] == {"extern_gemm": 1}
    assert extern_diff["m9_materialized_artifact_candidate_delta"] == 1


def test_m75_model_supported_kernel_report_is_golden_guard():
    records = [
        _kernel_record(
            "vit",
            "vit_tiny",
            "triton_vit_fp32_flat",
            make_report_status(ok=True, bucket="translated"),
            indexing_summary={"index_kinds": {"flat": 3}},
            types=["tensor<64xf32>", "tensor<64xi1>"],
        ),
        _kernel_record(
            "yolo",
            "yolo_tiny",
            "triton_yolo_fp16_strided",
            make_report_status(ok=True, bucket="translated"),
            indexing_summary={"index_kinds": {"flat": 1, "mul": 1}},
            types=["tensor<128xf16>", "tensor<128xbf16>"],
        ),
        _kernel_record(
            "llama",
            "llama_tiny",
            "triton_llama_attention_no_other",
            make_report_status(
                ok=False,
                bucket="unsupported_ttir_op",
                fallback_reason="unsupported_ttir_op",
                message="masked tt.load without other requires load/store masks to use the same flat extent predicate",
            ),
            op_counts={"tt.load": 2, "tt.store": 1},
        ),
    ]
    models = [
        _model_record("vit", "vit_tiny", kernel_count=1, translated=1, fallback=0),
        _model_record("yolo", "yolo_tiny", kernel_count=1, translated=1, fallback=0),
        _model_record("llama", "llama_tiny", kernel_count=1, translated=0, fallback=1),
    ]

    report = build_model_corpus_report(
        records,
        models,
        generated_at="2026-05-24T00:00:00+00:00",
    )
    supported = report["supported_kernel_report"]

    assert supported["kernel_count"] == 2
    assert supported["contracts"] == {"pointwise_flat": 2}
    assert supported["dtypes"]["float32"] == 1
    assert supported["dtypes"]["float16"] == 1
    assert supported["dtypes"]["bfloat16"] == 1
    assert supported["tensor_shapes"] == {"128": 2, "64": 2}
    assert supported["layout_index_kinds"] == {"flat": 4, "mul": 1}
    assert supported["legacy_contract_records"] == []
    assert supported["silent_fallback_records"] == []
    assert supported["ok"] is True


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@pytest.mark.skipif(
    os.environ.get("TRITON_TVM_RUN_MODEL_CORPUS") != "1",
    reason="set TRITON_TVM_RUN_MODEL_CORPUS=1 to run the M6 model corpus",
)
@tvm.testing.requires_cuda
def test_m6_cuda_builtin_model_corpus_writes_reproducible_report(tmp_path):
    report = run_model_corpus_audit(
        builtin_model_audit_cases(),
        out_dir=tmp_path,
        config=TritonTVMModelAuditConfig(seed=0, min_models=3),
    )

    families = {model["model_family"] for model in report["models"]}
    assert {"vit", "llama", "yolo"} <= families
    assert report["model_summary"]["total_models"] == 3
    assert "dependency_versions" in report
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    for blocker in report["blockers"]:
        assert blocker["fallback_reason"]
        assert blocker["models_impacted"] >= 1


def _kernel_record(
    family,
    model_case,
    kernel_name,
    status,
    *,
    op_counts=None,
    grid_type="Grid1D",
    num_reduction=0,
    atomic_add_found=False,
    kernel_source_path="",
    indexing_summary=None,
    types=None,
):
    op_counts = op_counts or {"tt.load": 1, "tt.store": 1}
    return {
        "corpus": "m6_model_corpus",
        "model_family": family,
        "model_case": model_case,
        "case_name": model_case,
        "kernel_name": kernel_name,
        "contract": "pointwise_flat",
        "unique_ops": sorted(op_counts),
        "op_counts": op_counts,
        "types": types or ["tensor<64xf32>"],
        "load_count": op_counts.get("tt.load", 0),
        "store_count": op_counts.get("tt.store", 0),
        "grid_type": grid_type,
        "num_reduction": num_reduction,
        "atomic_add_found": atomic_add_found,
        "size_hints": {},
        "indexing_summary": indexing_summary or {},
        "kernel_source_path": kernel_source_path,
        "translate_status": status,
    }


def _m126_synthetic_corpus():
    records = [
        _m126_matmul_record("extern_addmm_bias", line_no=598 + index * 7)
        for index in range(3)
    ] + [
        _m126_matmul_record("extern_gemm", line_no=line_no)
        for line_no in (627, 647, 658, 675)
    ]
    return {
        "summary": {
            "total_kernels": 89,
            "translated_kernels": 72,
            "status_buckets": {"contract_error": 17, "translated": 72},
        },
        "model_summary": {
            "fallback_kernels": 17,
            "full_tvm_runnable_models": 0,
        },
        "pre_m12": {
            "residual_debt_counts": {
                "captured_grid_concat_split": 17,
                "wrapper_extern_gemm_artifact_only": 7,
                "wrapper_extern_addmm_bias_artifact_only": 0,
                "runtime_resolved_attention": 2,
                "runtime_resolved_convolution": 65,
            }
        },
        "m12": {
            "status": "m12_2_native_vit_matmul_closed",
            "policy": {
                "performance_ready_e2e": False,
                "strict_full_tvm_native": False,
                "full_tvm_runnable_models": 0,
            },
            "vit_fixed_shape_execution_plan": {
                "captured_kernel_count": 7,
                "translated_captured_kernel_count": 7,
                "wrapper_call_count": 9,
                "wrapper_matmul_call_count": 7,
                "artifact_only_matmul_blockers": 0,
                "native_runtime_resolved_matmul_count": 7,
                "no_hidden_fallback": True,
                "nodes": [
                    {
                        "node_kind": "captured_kernel",
                        "translate_ok": True,
                        "fallback_reason": "",
                        "provider_kind": "native_tvm_tirx",
                        "host_staging_bytes": 0,
                    }
                    for _ in range(7)
                ],
            },
            "m12_2_native_matmul_closure": {
                "status": "passed",
                "target_model": "vit_tiny_random",
                "provider_kind": EXTERN_GEMM_PROVIDER_NATIVE_TVM,
                "runtime_claim": EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
                "schedule_id": NATIVE_TIR_MATMUL_SCHEDULE_ID,
                "runtime_resolved_total": 7,
                "runtime_resolved_extern_gemm": 4,
                "runtime_resolved_extern_addmm_bias": 3,
                "host_staging_bytes": 0,
                "performance_claim": False,
                "performance_claim_count": 0,
                "provider_counts": {EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7},
                "records": records,
            },
        },
    }


def _m126_matmul_record(op_family, *, line_no):
    return {
        "op_family": op_family,
        "op_name": (
            "extern_kernels.addmm" if op_family == "extern_addmm_bias" else "extern_kernels.mm"
        ),
        "line_no": line_no,
        "matmul_m": 5,
        "matmul_n": 64,
        "matmul_k": 64,
        "implementation_kind": "native_tir_schedule",
        "schedule_id": NATIVE_TIR_MATMUL_SCHEDULE_ID,
        "runtime_status": EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
        "provider_kind": EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        "runtime_claim": EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
        "uses_host_staging": False,
        "performance_claim": False,
        "unsupported_matmul_reason": "",
    }


def _m126_synthetic_dashboard():
    latency = {
        "torch_eager_cuda": {"samples_ms": [0.36], "p50_ms": 0.36, "p95_ms": 0.43},
        "torch_compile_inductor": {"samples_ms": [0.33], "p50_ms": 0.33, "p95_ms": 0.36},
        "triton_tvm_e2e": {"samples_ms": [0.91], "p50_ms": 0.91, "p95_ms": 0.96},
    }
    return {
        "report_kind": "triton_tvm_m12_4_vit_fixed_shape_e2e_dashboard",
        "schema_version": 1,
        "dashboard_id": "m12_4_vit_fixed_shape_e2e_dashboard_v1",
        "status": "measured",
        "target_model": "vit_tiny_random",
        "fixed_shape": True,
        "warmup": 5,
        "repeat": 20,
        "warmed_cache": True,
        "baselines": ["torch_eager_cuda", "torch_compile_inductor", "triton_tvm_e2e"],
        "latency_ms": latency,
        "ratios": {
            "triton_tvm_to_torch_eager_p50": 2.53,
            "triton_tvm_to_torch_eager_p95": 2.23,
            "triton_tvm_to_torch_compile_p50": 2.75,
            "triton_tvm_to_torch_compile_p95": 2.67,
        },
        "correctness": {"allclose": True},
        "p0": {
            "correctness": True,
            "no_silent_fallback": True,
            "complete_provider_report": True,
            "passed": True,
        },
        "p1": {
            "threshold": "<=3x torch_eager_cuda or torch_compile_inductor p50/p95",
            "eager_baseline_ok": True,
            "inductor_baseline_ok": True,
            "zero_host_staging": True,
            "passed": True,
        },
        "p2": {
            "threshold": "<=2x torch_compile_inductor p50/p95",
            "inductor_baseline_ok": False,
            "warmed_cache": True,
            "zero_host_staging": True,
            "complete_provider_mix": True,
            "correctness": True,
            "passed": False,
        },
        "m12_performance_tier": "P1",
        "performance_ready_e2e": False,
        "performance_claim": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": 0,
        "launch_count_estimate": 16,
        "provider_mix": {
            "complete_provider_report": True,
            "no_silent_fallback": True,
            "provider_counts": {
                "torch_inductor_triton_captured_harness": 7,
                "device_torch_cuda": 1,
                "native_decomposed": 1,
                EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
            },
            "wrapper_convolution": {
                "count": 1,
                "provider_kind": "device_torch_cuda",
                "runtime_status": "runtime_resolved",
                "host_staging_bytes": 0,
            },
            "wrapper_attention": {
                "count": 1,
                "provider_kind": "native_decomposed",
                "runtime_status": "runtime_resolved",
                "host_staging_bytes": 0,
            },
            "wrapper_matmul": {
                "count": 7,
                "provider_kind": EXTERN_GEMM_PROVIDER_NATIVE_TVM,
                "runtime_status": "runtime_resolved",
                "host_staging_bytes": 0,
            },
        },
        "device_bytes": {"estimated_io_bytes": 1280},
        "summary": {"p1_passed": True, "p2_passed": False},
        "invariants": {"status": "passed", "invariant_failures": []},
    }


def _m126_synthetic_review():
    return {
        "report_kind": "triton_tvm_m12_5_vit_optimization_review",
        "status": "passed",
        "disposition": "open_m12_7_p2_recovery",
        "primary_target": "native_tvm_matmul_provider_cost",
        "performance_ready_e2e": False,
        "p2_passed": False,
    }


def _m126_synthetic_freeze():
    return {
        "report_kind": "triton_tvm_m12_5_5_vit_perf_optimization_freeze",
        "schema_version": 1,
        "freeze_id": "m12_5_5_vit_perf_optimization_freeze_v1",
        "status": "passed",
        "current_gate_state": {
            "dashboard_status": "measured",
            "m12_performance_tier": "P1",
            "p1_passed": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
            "host_staging_bytes": 0,
            "strict_full_tvm_native": False,
            "full_tvm_runnable_models": 0,
            "review_disposition": "open_m12_7_p2_recovery",
            "focused_follow_up_recommended": True,
        },
        "p2_gap": {
            "threshold": "triton_tvm_e2e p50/p95 <= 2x torch_compile_inductor",
            "torch_compile_p50_ms": 0.33,
            "torch_compile_p95_ms": 0.36,
            "triton_tvm_p50_ms": 0.91,
            "triton_tvm_p95_ms": 0.96,
            "p50_ceiling_ms": 0.66,
            "p95_ceiling_ms": 0.72,
            "p50_excess_ms": 0.25,
            "p95_excess_ms": 0.24,
        },
        "primary_target": "native_tvm_matmul_provider_cost",
        "frozen_targets": [{"target_id": "native_tvm_matmul_provider_cost"}],
        "acceptance_criteria": {
            "p2_threshold_unchanged": "p50 and p95 each <= 2x torch_compile_inductor"
        },
        "out_of_scope": ["YOLO closure"],
        "p2_recovery_sequence": [
            {
                "slice": "M12.6",
                "name": "E2E Surface Hardening",
                "performance_optimization": False,
            },
            {
                "slice": "M12.7",
                "name": "P2 Recovery - Native Matmul Provider Cost",
                "primary_target": "native_tvm_matmul_provider_cost",
            },
            {"slice": "M12.8", "name": "P2 Gate Re-run and Close/Continue Decision"},
        ],
        "invariants": {"status": "passed", "invariant_failures": []},
    }


def _m127_synthetic_latency():
    return {
        "torch_compile_inductor": _m127_latency_record([0.330, 0.332, 0.334]),
        "triton_tvm_e2e_legacy_sync": _m127_latency_record([0.910, 0.918, 0.922]),
        "triton_tvm_e2e_no_per_call_sync": _m127_latency_record([0.720, 0.724, 0.728]),
    }


def _m127_latency_record(samples):
    ordered = sorted(float(sample) for sample in samples)
    mid = len(ordered) // 2
    return {
        "samples_ms": ordered,
        "p50_ms": ordered[mid],
        "p95_ms": ordered[-1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _m127_profile_runs(sync_policy, *, sync_ms):
    return [
        [
            _m127_profile_record(index, sync_policy=sync_policy, sync_ms=sync_ms)
            for index in range(7)
        ]
        for _ in range(3)
    ]


def _m127_profile_record(index, *, sync_policy, sync_ms):
    op_family = "extern_addmm_bias" if index < 3 else "extern_gemm"
    record = _m126_matmul_record(op_family, line_no=598 + index * 7)
    record["index"] = index
    record["profile"] = {
        "sync_policy": sync_policy,
        "shape_abi_validation_ms": 0.002,
        "b_storage_resolution_ms": 0.003,
        "artifact_lookup_ms": 0.0,
        "dlpack_conversion_ms": 0.004,
        "artifact_dispatch_wall_ms": 0.010,
        "kernel_event_ms": 0.006,
        "dispatch_minus_kernel_estimate_ms": 0.004,
        "pre_sync_wall_ms": sync_ms / 2.0,
        "post_sync_wall_ms": sync_ms / 2.0,
        "synchronization_wall_ms": sync_ms,
        "provider_total_wall_ms": 0.019 + sync_ms,
    }
    return record


def _m128_synthetic_m127_report():
    return {
        "report_kind": "triton_tvm_m12_7_native_matmul_provider_cost",
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "p2_gap_recovery": {"optimized_p2_estimate_passed": False},
    }


def _m128_synthetic_latency(*, triton_p50, triton_p95):
    return {
        "torch_eager_cuda": _m127_latency_record([0.360, 0.365, 0.370]),
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        "triton_tvm_e2e": _m127_latency_record([triton_p50, triton_p50, triton_p95]),
    }


def _m129_synthetic_latency(*, current_p50, backend_p50):
    return {
        "torch_eager_cuda": _m127_latency_record([0.360, 0.365, 0.370]),
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        CURRENT_MODE: _m127_latency_record([current_p50, current_p50, current_p50 + 0.030]),
        BACKEND_PASS_MODE: _m127_latency_record([backend_p50, backend_p50, backend_p50 + 0.030]),
    }


def _m129_backend_pass_summary(*, transformed, expected):
    return {
        "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
        "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
        "transformed_call_count": transformed,
        "expected_call_count": expected,
        "loop_shape": "blockIdx.x=M rows, threadIdx.x=N columns, serial K",
        "per_call": [],
    }


def _m1213_synthetic_m1211_report():
    return {
        "status": "route_frozen",
        "primary_route": "real_tl_dot_bridge",
        "auxiliary_order": ["runtime_overhead", "tt_dot_schedule_handoff"],
        "invariants": {"status": "passed", "invariant_failures": []},
    }


def _m1213_synthetic_m1212_report():
    return {
        "status": "passed",
        "backend_general_bridge_complete": True,
        "accepted_case": {"matmul_source_kind": REAL_JIT_TT_DOT_SOURCE_KIND},
        "invariants": {"status": "passed", "invariant_failures": []},
    }


def _m1213_synthetic_measurements():
    values = {
        "dlpack_conversion_ms": [0.010, 0.011, 0.012],
        "packed_func_lookup_ms": [0.001, 0.001, 0.002],
        "artifact_run_wall_ms": [0.030, 0.031, 0.032],
        "prebound_packed_call_wall_ms": [0.020, 0.021, 0.022],
        "kernel_event_ms": [0.006, 0.007, 0.008],
        "dispatch_minus_kernel_estimate_ms": [0.013, 0.014, 0.015],
        "launch_envelope_no_per_call_sync_ms": [0.040, 0.041, 0.042],
    }
    return {bucket: _m1213_stat_record(samples) for bucket, samples in values.items()}


def _m1213_stat_record(samples):
    ordered = sorted(float(sample) for sample in samples)
    return {
        "samples": len(ordered),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[-1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": ordered,
    }


def _m1214_synthetic_m1213_report():
    return {
        "status": "passed",
        "summary": {"runtime_overhead_measurement_complete": True},
        "invariants": {"status": "passed", "invariant_failures": []},
    }


def _m1214_handoff_case(case_name, schedule_id):
    return {
        "case_name": case_name,
        "status": "passed",
        "lower_to_ttir_ok": True,
        "ttir_contains_tt_dot": True,
        "matmul_source_kind": REAL_JIT_TT_DOT_SOURCE_KIND,
        "matmul_contract_ok": True,
        "implementation_kind": "native_tir_schedule",
        "schedule_id": schedule_id,
        "selected_schedule_id": schedule_id,
        "expected_schedule_id": schedule_id,
        "candidate_schedule_ids": [
            TENSORCORE_TIR_MATMUL_SCHEDULE_ID,
            TILED_TIR_MATMUL_SCHEDULE_ID,
            M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            NATIVE_TIR_MATMUL_SCHEDULE_ID,
        ],
        "schedule_reject_reasons": {},
        "matmul_perf_envelope": "matmul_perf_core_v1",
        "perf_guard_status": "not_measured",
        "shape": {"m": 16, "n": 16, "k": 16},
        "observed_reason": "",
        "is_reusable_tt_dot_schedule": True,
        "uses_wrapper_specific_schedule": False,
        "counts_as_handoff_readiness": True,
    }


def _m1214_synthetic_handoff_cases():
    return [
        _m1214_handoff_case(TENSORCORE_CASE, TENSORCORE_TIR_MATMUL_SCHEDULE_ID),
        _m1214_handoff_case(TILED_CASE, TILED_TIR_MATMUL_SCHEDULE_ID),
    ]


def _m1215_synthetic_m129_report(
    *,
    best_mode=BACKEND_PASS_MODE,
    best_mode_p2_passed=False,
):
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "best_mode": {"mode": best_mode},
        "p2_context": {"best_mode_p2_passed": best_mode_p2_passed},
        "performance_ready_e2e": False,
        "provider_overhead_profile": {
            BACKEND_PASS_MODE: {
                "record_count": 7,
                "aggregate_sum_ms": {
                    "provider_total_wall_ms": 0.520,
                    "kernel_event_ms": 0.380,
                    "artifact_dispatch_wall_ms": 0.320,
                    "dlpack_conversion_ms": 0.022,
                    "b_storage_resolution_ms": 0.032,
                    "shape_abi_validation_ms": 0.004,
                },
            }
        },
        "optimization_modes": {
            CURRENT_MODE: {
                "model_specific_artifact": False,
                "counts_for_triton_tvm_backend_progress": False,
                "route_diagnostic_label": "baseline_only",
            },
            BACKEND_PASS_MODE: {
                "backend_general": True,
                "model_specific_artifact": False,
                "counts_for_triton_tvm_backend_progress": True,
                "route_diagnostic_label": "backend_general_progress",
            },
            FUSED_QKV_MODE: {
                "backend_general": False,
                "model_specific_artifact": True,
                "counts_for_triton_tvm_backend_progress": False,
                "route_diagnostic_label": "model_specific_diagnostic_only",
            },
        },
    }


def _m1215_synthetic_m1210_report(*, fused_counts_for_progress=False):
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "route_claims": {
            BACKEND_PASS_MODE: {
                "backend_general": True,
                "model_specific_artifact": False,
                "counts_for_triton_tvm_backend_progress": True,
                "route_diagnostic_label": "backend_general_progress",
            },
            FUSED_QKV_MODE: {
                "backend_general": False,
                "model_specific_artifact": True,
                "counts_for_triton_tvm_backend_progress": fused_counts_for_progress,
                "route_diagnostic_label": "model_specific_diagnostic_only",
            },
        },
    }


def _m1215_synthetic_m1214_report():
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "real_tt_dot_schedule_handoff_complete": True,
        "p2_passed": False,
        "performance_ready_e2e": False,
        "next_default_action": {"action": "m12_15_fixed_shape_vit_p2_reentry"},
    }


def _m1215_synthetic_latency(*, current_p50, backend_p50):
    return {
        "torch_eager_cuda": _m127_latency_record([0.360, 0.365, 0.370]),
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        CURRENT_MODE: _m127_latency_record([current_p50, current_p50, current_p50 + 0.030]),
        BACKEND_PASS_MODE: _m127_latency_record([backend_p50, backend_p50, backend_p50 + 0.030]),
    }


def _m1215_backend_pass_summary(*, transformed, expected):
    return {
        "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
        "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
        "transformed_call_count": transformed,
        "expected_call_count": expected,
        "p2_eligible": True,
        "model_specific_artifact": False,
        "counts_for_triton_tvm_backend_progress": True,
        "per_call": [],
    }


def _m1216_synthetic_m127_report(
    *,
    optimization_validated=True,
    optimized_p2_estimate_passed=False,
):
    aggregate = {
        "provider_total_wall_ms": _m1213_stat_record([0.500, 0.520, 0.540]),
        "kernel_event_ms": _m1213_stat_record([0.360, 0.380, 0.400]),
        "artifact_dispatch_wall_ms": _m1213_stat_record([0.300, 0.320, 0.340]),
        "dlpack_conversion_ms": _m1213_stat_record([0.020, 0.022, 0.024]),
        "b_storage_resolution_ms": _m1213_stat_record([0.030, 0.032, 0.034]),
        "shape_abi_validation_ms": _m1213_stat_record([0.003, 0.004, 0.005]),
    }
    per_call = []
    shapes = [
        ("extern_addmm_bias", [5, 64, 64]),
        ("extern_addmm_bias", [5, 64, 64]),
        ("extern_addmm_bias", [5, 64, 64]),
        ("extern_gemm", [5, 64, 64]),
        ("extern_gemm", [5, 128, 64]),
        ("extern_gemm", [5, 64, 128]),
        ("extern_gemm", [1, 64, 64]),
    ]
    for index, (op_family, shape) in enumerate(shapes):
        per_call.append(
            {
                "index": index,
                "op_family": op_family,
                "shape": shape,
                "schedule_id": "cuda_block_per_output_serial_k_v1",
                "categories": {
                    "provider_total_wall_ms": _m1213_stat_record([0.050, 0.060, 0.070]),
                    "kernel_event_ms": _m1213_stat_record([0.040, 0.045, 0.050]),
                    "artifact_dispatch_wall_ms": _m1213_stat_record([0.030, 0.035, 0.040]),
                },
            }
        )
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "optimization": {"validated": optimization_validated},
        "p2_gap_recovery": {"optimized_p2_estimate_passed": optimized_p2_estimate_passed},
        "native_matmul_profile": {
            "policies": {
                "no_per_call_sync": {
                    "aggregate_categories": aggregate,
                    "per_call": per_call,
                }
            }
        },
    }


def _m1216_synthetic_m1213_report(*, runtime_complete=True):
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "summary": {"runtime_overhead_measurement_complete": runtime_complete},
        "measurements": _m1213_synthetic_measurements(),
        "attribution": {
            "dominant_bucket": "artifact_run_wall_ms",
            "dominant_p50_ms": 0.031,
        },
    }


def _m1216_synthetic_backend_pass_summary():
    shapes = [
        ("extern_addmm_bias", [5, 64, 64]),
        ("extern_addmm_bias", [5, 64, 64]),
        ("extern_addmm_bias", [5, 64, 64]),
        ("extern_gemm", [5, 64, 64]),
        ("extern_gemm", [5, 128, 64]),
        ("extern_gemm", [5, 64, 128]),
        ("extern_gemm", [1, 64, 64]),
    ]
    return {
        "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
        "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
        "transformed_call_count": 7,
        "expected_call_count": 7,
        "p2_eligible": True,
        "model_specific_artifact": False,
        "counts_for_triton_tvm_backend_progress": True,
        "per_call": [
            {
                "index": index,
                "op_family": op_family,
                "shape_mnk": shape,
                "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            }
            for index, (op_family, shape) in enumerate(shapes)
        ],
    }


def _m1216_synthetic_m1215_report(*, provider_mix):
    latency = _m1215_synthetic_latency(current_p50=0.850, backend_p50=0.880)
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "target_model": "vit_tiny_random",
        "fixed_shape": True,
        "p2_eligible_mode": BACKEND_PASS_MODE,
        "latency_ms": latency,
        "p2": {"passed": False},
        "p2_passed": False,
        "performance_ready_e2e": False,
        "provider_mix": {
            CURRENT_MODE: provider_mix,
            BACKEND_PASS_MODE: provider_mix,
        },
        "backend_pass": _m1216_synthetic_backend_pass_summary(),
        "source_evidence_policy": {
            "m12_14_counts_as_p2_evidence": False,
            "p2_excluded_modes": [FUSED_QKV_MODE],
        },
        "next_default_action": {"action": "m12_16_backend_general_p2_gap_analysis"},
    }


def _m12p_synthetic_m1216_report():
    return {
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "p2_passed": False,
        "performance_ready_e2e": False,
        "p2_gap_budget": {
            "threshold": "eligible Triton TVM p50/p95 <= 2x torch_compile_inductor",
            "eligible_mode": BACKEND_PASS_MODE,
            "torch_compile_p50_ms": 0.334,
            "torch_compile_p95_ms": 0.350,
            "eligible_p50_ms": 0.880,
            "eligible_p95_ms": 0.910,
            "p50_ceiling_ms": 0.668,
            "p95_ceiling_ms": 0.700,
            "p50_excess_ms": 0.212,
            "p95_excess_ms": 0.210,
            "p2_passed": False,
        },
        "summary": {"dominant_measured_bucket": "native_tvm_matmul_provider"},
        "selected_next_slice": {
            "action": "m12_p_profile_optimization_loop",
            "primary_bucket": "native_tvm_matmul_provider",
            "model_specific": False,
            "counts_for_triton_tvm_backend_progress": True,
        },
        "next_default_action": {"action": "m12_p_profile_optimization_loop"},
    }


def _m12p_synthetic_previous_iteration_report():
    return {
        "report_kind": "triton_tvm_m12_p_profile_optimization_loop",
        "milestone": "M12.P",
        "iteration_id": M12P_ITERATION_ID,
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "p2_passed": True,
        "candidate_mode": M12P_PREBOUND_MODE,
        "latency_ms": {
            M12P_PREBOUND_MODE: _m127_latency_record([0.500, 0.505, 0.530]),
        },
    }


def _m12p_synthetic_minimal_record_iteration_report():
    return {
        "report_kind": "triton_tvm_m12_p_profile_optimization_loop",
        "milestone": "M12.P",
        "iteration_id": M12P_MINIMAL_RECORD_ITERATION_ID,
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "p2_passed": True,
        "candidate_mode": M12P_MINIMAL_RECORD_MODE,
        "latency_ms": {
            M12P_MINIMAL_RECORD_MODE: _m127_latency_record([0.472, 0.480, 0.503]),
        },
    }


def _m12p_synthetic_residual_profile_report():
    return {
        "report_kind": "triton_tvm_m12_p_residual_profile",
        "milestone": "M12.P",
        "iteration_id": M12P_RESIDUAL_PROFILE_ITERATION_ID,
        "status": "passed",
        "invariants": {"status": "passed", "invariant_failures": []},
        "candidate_mode": M12P_MINIMAL_RECORD_MODE,
        "summary": {"remaining_unattributed_e2e_residual_ms": 0.144},
    }


def _m12p_minimal_record_optimization_scope():
    return {
        "target_bucket": "native_tvm_matmul_runtime_glue",
        "optimization": "prebound_minimal_record",
        "description": "synthetic minimal-record runtime-glue optimization",
        "backend_general": True,
        "model_specific": False,
        "counts_for_triton_tvm_backend_progress": True,
        "changes_runtime_or_schedule": True,
        "dispatch_mode": "prebound_minimal_record",
        "sync_policy": "no_per_call_sync",
    }


def _m12p_synthetic_latency(*, candidate_p50):
    return {
        "torch_eager_cuda": _m127_latency_record([0.360, 0.365, 0.370]),
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        BACKEND_PASS_MODE: _m127_latency_record([0.850, 0.850, 0.880]),
        M12P_PREBOUND_MODE: _m127_latency_record(
            [candidate_p50, candidate_p50, candidate_p50 + 0.030]
        ),
    }


def _m12p_synthetic_minimal_record_latency(*, candidate_p50):
    return {
        "torch_eager_cuda": _m127_latency_record([0.360, 0.365, 0.370]),
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        M12P_PREBOUND_MODE: _m127_latency_record([0.500, 0.505, 0.530]),
        M12P_MINIMAL_RECORD_MODE: _m127_latency_record(
            [candidate_p50, candidate_p50, candidate_p50 + 0.020]
        ),
    }


def _m12p_synthetic_residual_profile_latency(*, candidate_p50):
    return {
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        M12P_MINIMAL_RECORD_MODE: _m127_latency_record(
            [candidate_p50, candidate_p50, candidate_p50 + 0.020]
        ),
    }


def _m12p_synthetic_provider_session_reuse_latency():
    return {
        "torch_compile_inductor": _m127_latency_record([0.330, 0.334, 0.350]),
        M12P_MINIMAL_RECORD_MODE: _m127_latency_record([0.500, 0.510, 0.540]),
        M12P_SESSION_REUSE_MODE: _m127_latency_record([0.460, 0.480, 0.505]),
    }


def _m12p_synthetic_profile():
    baseline = {
        "record_count": 7,
        "aggregate_sum_ms": {
            "provider_total_wall_ms": 0.520,
            "kernel_event_ms": 0.380,
            "dispatch_minus_kernel_estimate_ms": 0.0,
            "artifact_dispatch_wall_ms": 0.320,
            "dlpack_conversion_ms": 0.024,
            "b_storage_resolution_ms": 0.040,
            "shape_abi_validation_ms": 0.004,
        },
        "per_call": [],
    }
    candidate = {
        "record_count": 7,
        "aggregate_sum_ms": {
            "provider_total_wall_ms": 0.450,
            "kernel_event_ms": 0.378,
            "dispatch_minus_kernel_estimate_ms": 0.0,
            "artifact_dispatch_wall_ms": 0.250,
            "dlpack_conversion_ms": 0.024,
            "b_storage_resolution_ms": 0.040,
            "shape_abi_validation_ms": 0.004,
        },
        "per_call": [],
    }
    return {BACKEND_PASS_MODE: baseline, M12P_PREBOUND_MODE: candidate}


def _m12p_synthetic_minimal_record_profile():
    baseline = {
        "record_count": 7,
        "aggregate_sum_ms": {
            "provider_total_wall_ms": 0.198,
            "kernel_event_ms": 0.074,
            "dispatch_minus_kernel_estimate_ms": 0.0,
            "artifact_dispatch_wall_ms": 0.026,
            "dlpack_conversion_ms": 0.020,
            "b_storage_resolution_ms": 0.033,
            "shape_abi_validation_ms": 0.003,
        },
        "per_call": [],
    }
    candidate = {
        "record_count": 7,
        "aggregate_sum_ms": {
            "provider_total_wall_ms": 0.170,
            "kernel_event_ms": 0.073,
            "dispatch_minus_kernel_estimate_ms": 0.0,
            "artifact_dispatch_wall_ms": 0.025,
            "dlpack_conversion_ms": 0.020,
            "b_storage_resolution_ms": 0.032,
            "shape_abi_validation_ms": 0.003,
        },
        "per_call": [],
    }
    return {M12P_PREBOUND_MODE: baseline, M12P_MINIMAL_RECORD_MODE: candidate}


def _m12p_synthetic_residual_provider_profile():
    return {
        "record_count": 2,
        "by_provider": {
            VISION_PROVIDER_DEVICE_TORCH_CUDA: {
                "record_count": 1,
                "aggregate_sum_ms": {
                    "provider_total_wall_ms": 0.045,
                    "kernel_event_ms": 0.031,
                    "dispatch_minus_kernel_estimate_ms": 0.014,
                },
                "per_call": [],
            },
            ATTENTION_PROVIDER_NATIVE_DECOMPOSED: {
                "record_count": 1,
                "aggregate_sum_ms": {
                    "provider_total_wall_ms": 0.070,
                    "kernel_event_ms": 0.052,
                    "dispatch_minus_kernel_estimate_ms": 0.018,
                },
                "per_call": [],
            },
        },
    }


def _model_record(
    family,
    model_case,
    *,
    kernel_count,
    translated,
    fallback,
    extern_calls=None,
):
    return {
        "model_family": family,
        "model_case": model_case,
        "status": "completed",
        "kernel_count": kernel_count,
        "translated_kernels": translated,
        "native_fallback_kernels": fallback,
        "status_buckets": {},
        "blocker_classes": {},
        "wrapper_paths": [],
        "extern_calls": extern_calls or [],
        "triton_kernel_runnable": kernel_count > 0 and fallback == 0 and translated == kernel_count,
        "full_tvm_runnable": kernel_count > 0 and fallback == 0 and translated == kernel_count,
    }


def _extern_call(family, model_case, op_name, op_family, **kwargs):
    record = {
        "model_family": family,
        "model_case": model_case,
        "case_name": model_case,
        "op_name": op_name,
        "op_family": op_family,
        "line_no": 10,
        "wrapper_path": f"/tmp/{model_case}.py",
        "source": f"{op_name}(...)",
    }
    record.update(kwargs)
    return record


def _native_m12_matmul_call(family, model_case, op_family, *, line_no, m, n, k):
    is_addmm = op_family == "extern_addmm_bias"
    return _extern_call(
        family,
        model_case,
        "extern_kernels.addmm" if is_addmm else "extern_kernels.mm",
        op_family,
        line_no=line_no,
        matmul_source_kind="wrapper_extern_addmm_bias" if is_addmm else "wrapper_extern_gemm",
        matmul_m=m,
        matmul_n=n,
        matmul_k=k,
        matmul_contract="matmul_minimal",
        matmul_contract_ok=True,
        matmul_a_dtype="float32",
        matmul_b_dtype="float32",
        matmul_accumulator_dtype="float32",
        matmul_output_dtype="float32",
        matmul_epilogue_kind="bias_add" if is_addmm else "none",
        implementation_kind="native_tir_schedule",
        schedule_id=NATIVE_TIR_MATMUL_SCHEDULE_ID,
        extern_symbol="extern_kernels.addmm" if is_addmm else "extern_kernels.mm",
        extern_packed_func="",
        extern_runtime_kind="runtime_provider",
        extern_runtime_replacement=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        extern_runtime_replacement_available=True,
        extern_runtime_replacement_reason=(
            "extern_addmm_bias_native_tvm_fixed_shape_provider_enabled_m12_2"
            if is_addmm
            else "extern_gemm_native_tvm_fixed_shape_provider_enabled_m12_2"
        ),
        extern_gemm_runtime_status=EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
        extern_gemm_provider_kind=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        extern_gemm_provider_abi_version=EXTERN_GEMM_PROVIDER_ABI_VERSION,
        extern_gemm_runtime_claim=EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
        extern_gemm_performance_claim=False,
        extern_gemm_uses_host_staging=False,
        matmul_a_layout="row_major",
        matmul_b_layout="transposed_weight_view",
        matmul_c_layout="row_major",
        matmul_a_stride=(k, 1),
        matmul_b_stride=(1, k),
        matmul_c_stride=(n, 1),
    )


def _fake_inductor_kernel(
    kernel_name,
    *,
    grid_type="Grid1D",
    num_reduction=0,
    atomic_add_found=False,
):
    return InductorKernel(
        case_name="case",
        kernel_name=kernel_name,
        source=("triton_heuristics.pointwise " if num_reduction == 0 else "") + kernel_name,
        device_str="cuda",
        fn=object(),
        signature={},
        constexprs={},
        attrs=None,
        triton_meta={},
        inductor_meta={
            "grid_type": grid_type,
            "num_reduction": num_reduction,
            "atomic_add_found": atomic_add_found,
        },
        size_hints={},
    )


if __name__ == "__main__":
    tvm.testing.main()
