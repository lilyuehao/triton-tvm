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
"""M11 vision convolution artifact-boundary coverage."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.triton_tvm import (
    VISION_M11_4_INTERFACE_STATUS,
    VISION_M11_5_HARDENING_STATUS,
    VISION_M11_5_RUNTIME_SCOPE_STATUS,
    VISION_M11_6_INTERFACE_STATUS,
    VISION_M11_6_RUNTIME_SCOPE_STATUS,
    VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    TritonTVMContractError,
    build_extern_conv2d_tirx_source,
    build_native_grid2d_pointwise_tirx_source,
    build_triton_tvm,
    register_python_torch_extern_conv2d,
    validate_pointwise_grid2d_static_contract,
    validate_triton_tvm_contract,
    validate_vision_conv2d_contract,
)
from tvm.contrib.triton_tvm.inductor import extract_inductor_wrapper_extern_calls
from tvm.contrib.triton_tvm.model_corpus import build_model_corpus_report, diff_capability_reports
from tvm.contrib.triton_tvm.reporting import make_report_status, render_capability_markdown
from tvm.contrib.triton_tvm.vision import (
    M11_GRID_FAMILIES,
    M11_GRID2D_ARTIFACT_READY,
    M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
    M11_GRID2D_READINESS_VERSION,
    M11_GRID2D_RUNTIME_READY,
    M11_GRID_FAMILY_BN_SILU_FUSION,
    M11_GRID_FAMILY_CONCAT_SPLIT,
    M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
    M11_GRID_FAMILY_OTHER_MULTIDIM_POINTWISE,
    M11_GRID_FAMILY_POOL_OR_SOFTMAX,
    M11_GRID_FAMILY_YOLO_DECODE_POSTPROCESS,
    M11_GRID_TAXONOMY_VERSION,
    VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
    VISION_EXTERN_PACKED_FUNC,
    VISION_EXTERN_SYMBOL,
    VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
    VISION_PROVIDER_ABI_VERSION,
    VISION_PROVIDER_NONE,
    VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    VISION_RUNTIME_KIND_PROVIDER,
    VISION_RUNTIME_PROVIDER_REASON,
    VISION_RUNTIME_PROVIDER_SCOPE_REASON,
    VISION_RUNTIME_STATUS_ARTIFACT_ONLY,
    VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    VISION_SEMANTICS_STATUS_ACCEPTED,
    TargetVisionDecision,
    TargetVisionPolicy,
    VisionGrid2DPointwiseSemantics,
    classify_m11_captured_grid_record,
    extract_vision_conv2d_semantics_from_wrapper_extern,
    wrapper_conv2d_vision_report_fields,
)
from tvm.contrib.triton_tvm.m11_vision_baseline import run_vision_baseline
from tvm.contrib.triton_tvm.m116_vision_dashboard import run_vision_grid2d_dashboard
from tvm.contrib.triton_tvm.translator import TritonTVMMeta


def test_m112_extracts_vit_patch_embedding_conv_artifact_metadata():
    call = _extract_single_conv(
        _conv_wrapper_source(
            input_shape=(1, 3, 32, 32),
            weight_shape=(64, 3, 16, 16),
            output_shape=(1, 64, 2, 2),
            stride=(16, 16),
            padding=(0, 0),
        ),
        case_name="vit_tiny_random",
    )

    assert call.vision_semantics_status == VISION_SEMANTICS_STATUS_ACCEPTED
    assert call.vision_contract == VISION_CONTRACT_CONV2D_NCHW_STATIC
    assert call.vision_contract_version == "m11_v1"
    assert call.vision_layout == "nchw_channels_last"
    assert call.vision_input_shape == "1, 3, 32, 32"
    assert call.vision_weight_shape == "64, 3, 16, 16"
    assert call.vision_output_shape == "1, 64, 2, 2"
    assert call.vision_stride == "16, 16"
    assert call.vision_padding == "0, 0"
    assert call.vision_runtime_status == VISION_RUNTIME_STATUS_ARTIFACT_ONLY
    assert call.vision_runtime_launch_count == 0
    assert call.vision_artifact_call_count == 1
    assert call.vision_performance_claim is False
    assert call.unsupported_vision_reason == ""


def test_m114_vit_patch_embedding_conv_provider_metadata_and_contract():
    semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=(1, 3, 32, 32),
            weight_shape=(64, 3, 16, 16),
            output_shape=(1, 64, 2, 2),
            stride=(16, 16),
            padding=(0, 0),
        ),
        case_name="vit_tiny_random",
    )
    decision = TargetVisionPolicy(
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, vision_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_conv2d_tirx_source(semantics, decision))

    assert decision.vision_runtime_status == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert decision.vision_provider_kind == VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    assert decision.vision_provider_abi_version == 1
    assert decision.vision_runtime_claim == VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY
    assert decision.vision_uses_host_staging is True
    assert decision.vision_runtime_launch_count == 1
    assert decision.vision_artifact_call_count == 1
    assert decision.vision_input_bytes == 12288
    assert decision.vision_weight_bytes == 196608
    assert decision.vision_output_bytes == 1024
    assert decision.vision_total_io_bytes == 209920
    assert decision.vision_host_staging_bytes == 209920
    assert decision.vision_total_accounted_bytes == 419840
    validate_vision_conv2d_contract(irmod)
    validate_triton_tvm_contract(irmod, VISION_CONTRACT_CONV2D_NCHW_STATIC)
    assert irmod.script().count(f'T.call_packed("{VISION_EXTERN_PACKED_FUNC}"') == 1


@pytest.mark.parametrize(
    "input_shape, weight_shape, output_shape, stride, padding, expected_contract",
    [
        (
            (1, 3, 64, 64),
            (16, 3, 3, 3),
            (1, 16, 32, 32),
            (2, 2),
            (1, 1),
            VISION_CONTRACT_CONV2D_NCHW_STATIC,
        ),
        (
            (1, 32, 16, 16),
            (32, 32, 1, 1),
            (1, 32, 16, 16),
            (1, 1),
            (0, 0),
            VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
        ),
        (
            (1, 16, 16, 16),
            (16, 16, 3, 3),
            (1, 16, 16, 16),
            (1, 1),
            (1, 1),
            VISION_CONTRACT_CONV2D_NCHW_STATIC,
        ),
    ],
)
def test_m116_static_conv_provider_marks_yolo_representative_shapes_runtime_resolved(
    input_shape,
    weight_shape,
    output_shape,
    stride,
    padding,
    expected_contract,
):
    semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=input_shape,
            weight_shape=weight_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
        ),
        case_name="yolov8n_yaml_random",
    )
    decision = TargetVisionPolicy(
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, vision_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_conv2d_tirx_source(semantics, decision))

    assert semantics.vision_contract == expected_contract
    assert decision.vision_runtime_status == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED
    assert decision.vision_provider_kind == VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    assert decision.vision_runtime_claim == VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY
    assert decision.vision_performance_claim is False
    assert decision.vision_uses_host_staging is True
    assert decision.vision_runtime_launch_count == 1
    assert decision.vision_artifact_call_count == 1
    validate_vision_conv2d_contract(irmod)


def test_m116_out_of_scope_conv_remains_artifact_only_under_provider_scope():
    semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=(1, 3, 64, 64),
            weight_shape=(16, 3, 3, 3),
            output_shape=(1, 16, 64, 64),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(2, 2),
        ),
        case_name="yolov8n_yaml_random",
    )
    decision = TargetVisionPolicy(
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, vision_contract_ok=True)

    assert decision.vision_runtime_status == VISION_RUNTIME_STATUS_ARTIFACT_ONLY
    assert decision.vision_provider_kind == VISION_PROVIDER_NONE
    assert decision.vision_runtime_launch_count == 0
    assert decision.vision_artifact_call_count == 1
    assert decision.unsupported_vision_runtime_reason == VISION_RUNTIME_PROVIDER_SCOPE_REASON


def test_m115_contract_hardening_rejects_runtime_scope_and_accounting_regressions():
    vit_semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=(1, 3, 32, 32),
            weight_shape=(64, 3, 16, 16),
            output_shape=(1, 64, 2, 2),
            stride=(16, 16),
            padding=(0, 0),
        ),
        case_name="vit_tiny_random",
    )
    vit_decision = TargetVisionPolicy(
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(vit_semantics, vision_contract_ok=True)
    valid_source = build_extern_conv2d_tirx_source(vit_semantics, vit_decision)

    for broken_source, message in [
        (
            valid_source.replace(
                '"triton_tvm.vision_performance_claim": False',
                '"triton_tvm.vision_performance_claim": True',
            ),
            "must not claim performance",
        ),
        (
            valid_source.replace(
                '"triton_tvm.vision_total_io_bytes": 209920',
                '"triton_tvm.vision_total_io_bytes": 209916',
            ),
            "vision_total_io_bytes must be 209920",
        ),
        (
            valid_source.replace(
                '"triton_tvm.vision_runtime_launch_count": 1',
                '"triton_tvm.vision_runtime_launch_count": 0',
            ),
            "must record one runtime launch",
        ),
    ]:
        with pytest.raises(TritonTVMContractError, match=message):
            validate_vision_conv2d_contract(tvm.script.from_source(broken_source))

    yolo_semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=(1, 3, 64, 64),
            weight_shape=(16, 3, 3, 3),
            output_shape=(1, 16, 64, 64),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(2, 2),
        ),
        case_name="yolov8n_yaml_random",
    )
    out_of_scope_decision = TargetVisionDecision(
        vision_contract_ok=True,
        implementation_kind=VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
        extern_symbol=VISION_EXTERN_SYMBOL,
        extern_packed_func=VISION_EXTERN_PACKED_FUNC,
        vision_runtime_kind=VISION_RUNTIME_KIND_PROVIDER,
        vision_runtime_replacement=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
        vision_runtime_replacement_available=True,
        vision_runtime_replacement_reason=VISION_RUNTIME_PROVIDER_REASON,
        vision_runtime_status=VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
        vision_provider_kind=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
        vision_provider_abi_version=VISION_PROVIDER_ABI_VERSION,
        vision_runtime_claim=VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
        vision_performance_claim=False,
        vision_uses_host_staging=True,
        vision_artifact_call_count=1,
    ).with_accounting(yolo_semantics)
    with pytest.raises(TritonTVMContractError, match="M11.6 runtime-resolved conv2d"):
        validate_vision_conv2d_contract(
            tvm.script.from_source(
                build_extern_conv2d_tirx_source(yolo_semantics, out_of_scope_decision)
            )
        )


def test_m114_unknown_conv_provider_reports_stable_reason():
    semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=(1, 3, 32, 32),
            weight_shape=(64, 3, 16, 16),
            output_shape=(1, 64, 2, 2),
            stride=(16, 16),
            padding=(0, 0),
        ),
        case_name="vit_tiny_random",
    )
    decision = TargetVisionPolicy(vision_runtime_provider="unknown_provider").decide(
        semantics,
        vision_contract_ok=True,
    )

    assert decision.vision_runtime_status == "unsupported"
    assert decision.vision_provider_kind == "unknown_provider"
    assert decision.unsupported_vision_runtime_reason == (
        "vision_conv2d_runtime_provider_unknown_m11_4"
    )


@pytest.mark.parametrize(
    "input_shape, weight_shape, output_shape, stride, padding, groups, expected_contract",
    [
        (
            (1, 3, 64, 64),
            (16, 3, 3, 3),
            (1, 16, 32, 32),
            (2, 2),
            (1, 1),
            1,
            VISION_CONTRACT_CONV2D_NCHW_STATIC,
        ),
        (
            (1, 32, 16, 16),
            (32, 32, 1, 1),
            (1, 32, 16, 16),
            (1, 1),
            (0, 0),
            1,
            VISION_CONTRACT_CONV2D_1X1_NCHW_STATIC,
        ),
        (
            (1, 4, 8, 8),
            (4, 1, 3, 3),
            (1, 4, 8, 8),
            (1, 1),
            (1, 1),
            4,
            VISION_CONTRACT_DEPTHWISE_CONV2D_NCHW_STATIC,
        ),
        (
            (1, 4, 8, 8),
            (8, 2, 3, 3),
            (1, 8, 8, 8),
            (1, 1),
            (1, 1),
            2,
            VISION_CONTRACT_GROUPED_CONV2D_NCHW_STATIC,
        ),
    ],
)
def test_m112_artifact_builder_validates_all_initial_conv_contracts(
    input_shape,
    weight_shape,
    output_shape,
    stride,
    padding,
    groups,
    expected_contract,
):
    semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=input_shape,
            weight_shape=weight_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            groups=groups,
        ),
        case_name="m112_conv_contract",
    )
    decision = TargetVisionPolicy().decide(semantics, vision_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_conv2d_tirx_source(semantics, decision))

    assert semantics.vision_contract == expected_contract
    validate_vision_conv2d_contract(irmod)
    validate_triton_tvm_contract(irmod, expected_contract)
    assert irmod.script().count(f'T.call_packed("{VISION_EXTERN_PACKED_FUNC}"') == 1


def test_m112_unsupported_conv_forms_report_stable_reasons():
    cases = [
        (
            _conv_wrapper_source(bias="bias"),
            "vision_conv2d_bias_not_supported_m11_2",
        ),
        (
            _conv_wrapper_source(transposed=True),
            "vision_conv2d_transposed_not_supported_m11_2",
        ),
        (
            _conv_wrapper_source(output_padding=(1, 0)),
            "vision_conv2d_output_padding_not_supported_m11_2",
        ),
        (
            "def call(buf0, buf1):\n"
            "    return extern_kernels.convolution(buf0, buf1, stride=(1, 1), "
            "padding=(0, 0), dilation=(1, 1), transposed=False, "
            "output_padding=(0, 0), groups=1, bias=None)\n",
            "vision_conv2d_output_assignment_required",
        ),
        (
            _conv_wrapper_source(output_shape=(1, 16, 31, 32)),
            "vision_conv2d_output_shape_mismatch",
        ),
    ]
    for source, reason in cases:
        fields = wrapper_conv2d_vision_report_fields(
            op_name="extern_kernels.convolution",
            op_family="deferred_convolution",
            source="extern_kernels.convolution(...)",
            full_source=source,
            case_name="m112_unsupported",
        )

        assert fields["vision_semantics_status"] == "vision_semantics_unsupported"
        assert fields["unsupported_vision_reason"] == reason


@pytest.mark.parametrize(
    "kernel_name, model_family, expected_family",
    [
        (
            "triton_poi_fused_convolution_0",
            "vit",
            M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
        ),
        (
            "triton_poi_fused__native_batch_norm_legit_no_training_silu_5",
            "yolo",
            M11_GRID_FAMILY_BN_SILU_FUSION,
        ),
        (
            "triton_poi_fused_cat_convolution_silu_view_74",
            "yolo",
            M11_GRID_FAMILY_CONCAT_SPLIT,
        ),
        (
            "triton_per_fused_max_pool2d_with_indices_40",
            "yolo",
            M11_GRID_FAMILY_POOL_OR_SOFTMAX,
        ),
        (
            "triton_poi_fused_add_sigmoid_stack_decode_postprocess_75",
            "yolo",
            M11_GRID_FAMILY_YOLO_DECODE_POSTPROCESS,
        ),
        (
            "triton_poi_fused_arbitrary_multidim_99",
            "vit",
            M11_GRID_FAMILY_OTHER_MULTIDIM_POINTWISE,
        ),
    ],
)
def test_m113_captured_grid_classifier_reports_stable_families(
    kernel_name,
    model_family,
    expected_family,
):
    fields = classify_m11_captured_grid_record(
        _grid_record(kernel_name=kernel_name, model_family=model_family)
    )

    assert fields["m11_grid_status"] == "m11_grid_classified"
    assert fields["m11_grid_family"] == expected_family
    assert fields["m11_grid_launch_kind"] == "Grid2D"
    assert fields["m11_grid_program_axes"] == "x, y"
    assert fields["m11_grid_size_hints"] == "x=16, y=64"
    assert fields["unsupported_m11_grid_reason"] == "m11_3_taxonomy_only_no_grid_runtime"
    if expected_family in {
        M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
        M11_GRID_FAMILY_BN_SILU_FUSION,
    }:
        assert fields["m11_grid_contract"] == VISION_CONTRACT_POINTWISE_GRID2D_STATIC
        assert fields["m11_grid_artifact_status"] == M11_GRID2D_ARTIFACT_READY
        assert fields["m11_grid_runtime_status"] == M11_GRID2D_RUNTIME_READY
        assert fields["m11_grid_implementation_kind"] == M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM
        assert fields["m11_grid_performance_claim"] is True
    if expected_family == M11_GRID_FAMILY_CONCAT_SPLIT:
        assert fields["m11_grid_artifact_status"] == "m11_6_grid2d_artifact_rejected"
        assert fields["m11_grid_runtime_status"] == "unsupported"


def test_m113_captured_grid_classifier_leaves_non_grid_records_empty():
    translated = classify_m11_captured_grid_record(
        _grid_record(
            kernel_name="triton_poi_fused_convolution_0",
            ok=True,
            blocker_class="none",
        )
    )
    non_grid = classify_m11_captured_grid_record(
        _grid_record(
            kernel_name="triton_per_fused_max_pool2d_with_indices_40",
            grid_type="Grid1D",
            blocker_class="reduction",
        )
    )

    assert all(value == "" for value in translated.values())
    assert all(value == "" for value in non_grid.values())


def test_m112_model_corpus_section_reports_artifact_only_conv_without_full_model_claim():
    call = _extract_single_conv(
        _conv_wrapper_source(
            input_shape=(1, 3, 64, 64),
            weight_shape=(16, 3, 3, 3),
            output_shape=(1, 16, 32, 32),
            stride=(2, 2),
            padding=(1, 1),
        ),
        case_name="yolov8n_yaml_random",
    )
    report = build_model_corpus_report(
        [
            _grid_record(
                kernel_name="triton_poi_fused_convolution_0",
                model_family="yolo",
            )
        ],
        [
            {
                "model_family": "yolo",
                "model_case": "yolov8n_yaml_random",
                "status": "completed",
                "kernel_count": 1,
                "translated_kernels": 0,
                "native_fallback_kernels": 1,
                "status_buckets": {},
                "blocker_classes": {},
                "wrapper_paths": ["/tmp/yolov8n_yaml_random.py"],
                "extern_calls": [dict(vars(call))],
                "triton_kernel_runnable": False,
                "full_tvm_runnable": False,
            }
        ],
        generated_at="2026-05-26T00:00:00+00:00",
    )

    m11 = report["m11"]
    assert m11["interface_status"] == VISION_M11_6_INTERFACE_STATUS
    assert m11["convolution_interface_status"] == "m11_2_artifact_only_convolution_boundary_v1"
    assert m11["observed_convolution_call_count"] == 1
    assert m11["artifact_only_count"] == 1
    assert m11["runtime_resolved_count"] == 0
    assert m11["runtime_launch_count"] == 0
    assert m11["artifact_call_count"] == 1
    assert m11["model_full_tvm_runnable_after_m11_2_gate"] == 0
    assert m11["model_full_tvm_runnable_after_m11_3_gate"] == 0
    assert report["full_tvm_runnable"] is False
    assert report["summary"]["status_buckets"] == {"contract_error": 1}
    assert m11["contract_counts"] == {VISION_CONTRACT_CONV2D_NCHW_STATIC: 1}
    grid_taxonomy = m11["captured_grid_taxonomy"]
    assert grid_taxonomy["taxonomy_version"] == M11_GRID_TAXONOMY_VERSION
    assert grid_taxonomy["readiness_version"] == M11_GRID2D_READINESS_VERSION
    assert grid_taxonomy["kernel_count"] == 1
    assert grid_taxonomy["artifact_ready_count"] == 1
    assert grid_taxonomy["native_runtime_ready_count"] == 1
    assert m11["captured_grid_readiness"]["native_runtime_ready_count"] == 1
    assert set(grid_taxonomy["family_counts"]) == set(M11_GRID_FAMILIES)
    assert grid_taxonomy["family_counts"][M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE] == 1
    assert grid_taxonomy["family_counts"][M11_GRID_FAMILY_BN_SILU_FUSION] == 0
    assert grid_taxonomy["grid_type_counts"] == {"Grid2D": 1}
    assert grid_taxonomy["records"][0]["m11_grid_family"] == (
        M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE
    )

    markdown = render_capability_markdown(report, title="M11.2 Snapshot")
    assert "## M11 Vision Convolution Entry" in markdown
    assert "## M11 Captured Grid Taxonomy" in markdown
    assert "Artifact-only convolution calls: 1" in markdown
    assert VISION_CONTRACT_CONV2D_NCHW_STATIC in markdown
    assert M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE in markdown


def test_m114_model_corpus_section_reports_single_runtime_resolved_vit_conv():
    vit_call = _extract_single_conv(
        _conv_wrapper_source(
            input_shape=(1, 3, 32, 32),
            weight_shape=(64, 3, 16, 16),
            output_shape=(1, 64, 2, 2),
            stride=(16, 16),
            padding=(0, 0),
        ),
        case_name="vit_tiny_random",
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    )
    yolo_call = _extract_single_conv(
        _conv_wrapper_source(
            input_shape=(1, 3, 64, 64),
            weight_shape=(16, 3, 3, 3),
            output_shape=(1, 16, 32, 32),
            stride=(2, 2),
            padding=(1, 1),
        ),
        case_name="yolov8n_yaml_random",
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    )
    report = build_model_corpus_report(
        [
            _grid_record(
                kernel_name="triton_poi_fused_convolution_0",
                model_family="yolo",
            )
        ],
        [
            _model_record_with_call("vit", "vit_tiny_random", vit_call),
            _model_record_with_call("yolo", "yolov8n_yaml_random", yolo_call),
        ],
        generated_at="2026-05-26T00:00:00+00:00",
    )

    m11 = report["m11"]
    assert m11["interface_status"] == VISION_M11_6_INTERFACE_STATUS
    assert m11["convolution_interface_status"] == "m11_2_artifact_only_convolution_boundary_v1"
    assert m11["runtime_interface_status"] == VISION_M11_6_INTERFACE_STATUS
    assert m11["hardening_status"] == VISION_M11_5_HARDENING_STATUS
    assert m11["runtime_scope_status"] == VISION_M11_6_RUNTIME_SCOPE_STATUS
    assert m11["observed_convolution_call_count"] == 2
    assert m11["artifact_count"] == 2
    assert m11["artifact_only_count"] == 0
    assert m11["runtime_resolved_count"] == 2
    assert m11["runtime_launch_count"] == 2
    assert m11["artifact_call_count"] == 2
    assert m11["provider_counts"] == {VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED: 2}
    assert m11["unsupported_runtime_reasons"] == {}
    assert m11["runtime_resolved_records"][0]["vision_total_io_bytes"] == 209920
    assert m11["captured_grid_taxonomy"]["kernel_count"] == 1
    assert m11["captured_grid_readiness"]["native_runtime_ready_count"] == 1
    assert m11["model_full_tvm_runnable_after_m11_4_gate"] == 0
    assert m11["model_full_tvm_runnable_after_m11_5_gate"] == 0
    assert m11["model_full_tvm_runnable_after_m11_6_gate"] == 0
    assert m11["report_cache_invariants"]["status"] == "passed"
    assert m11["report_cache_invariants"]["invariant_failures"] == []
    assert m11["corpus_diff_guard"]["baseline_id"] == (
        "m11_6_vision_runtime_grid2d_readiness_baseline_v1"
    )
    assert m11["corpus_diff_guard"]["vision_runtime_resolved_count"] == 2
    assert m11["m10_attention_boundary"]["status"] == "m10_attention_boundary_closed"
    assert m11["runtime_scope"]["status"] == VISION_M11_6_RUNTIME_SCOPE_STATUS
    assert report["full_tvm_runnable"] is False

    baseline_report = build_model_corpus_report(
        [
            _grid_record(
                kernel_name="triton_poi_fused_convolution_0",
                model_family="yolo",
            )
        ],
        [
            _model_record_with_call(
                "vit",
                "vit_tiny_random",
                _extract_single_conv(
                    _conv_wrapper_source(
                        input_shape=(1, 3, 32, 32),
                        weight_shape=(64, 3, 16, 16),
                        output_shape=(1, 64, 2, 2),
                        stride=(16, 16),
                        padding=(0, 0),
                    ),
                    case_name="vit_tiny_random",
                ),
            ),
            _model_record_with_call(
                "yolo",
                "yolov8n_yaml_random",
                _extract_single_conv(
                    _conv_wrapper_source(
                        input_shape=(1, 3, 64, 64),
                        weight_shape=(16, 3, 3, 3),
                        output_shape=(1, 16, 32, 32),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    case_name="yolov8n_yaml_random",
                ),
            ),
        ],
        generated_at="2026-05-26T00:00:00+00:00",
    )
    diff = diff_capability_reports(baseline_report, report)
    assert diff["m11_vision_delta"]["runtime_resolved_count"] == 2
    assert diff["m11_vision_delta"]["artifact_only_count"] == -2
    assert diff["m11_vision_delta"]["runtime_launch_count"] == 2
    assert diff["m11_vision_delta"]["grid_family_delta"] == {}
    assert diff["m11_vision_delta"]["grid2d_native_runtime_ready_delta"] == 0

    markdown = render_capability_markdown(report, title="M11.4 Snapshot")
    assert "### M11.5 Hardening" in markdown
    assert VISION_M11_5_HARDENING_STATUS in markdown
    assert "### M11.6 Vision Readiness" in markdown
    assert "### M11 Runtime-Resolved Vision" in markdown
    assert VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED in markdown


def test_m11p_vision_baseline_schema_without_benchmarks(tmp_path):
    report = run_vision_baseline(out_dir=tmp_path, warmup=1, repeat=1, run_benchmarks=False)

    assert report["report_kind"] == "triton_tvm_m11_vision_performance_baseline"
    assert report["baseline_id"] == "m11_vit_patch_conv_provider_vs_torch_conv2d_v1"
    assert report["summary"]["total_cases"] == 1
    assert report["summary"]["measured_cases"] == 0
    assert report["hardening_status"] == VISION_M11_5_HARDENING_STATUS
    assert report["hardening_checks"]["status"] == "passed"
    assert report["hardening_checks"]["runtime_scope_status"] == (
        VISION_M11_5_RUNTIME_SCOPE_STATUS
    )
    assert report["vision_runtime_provider_performance_claim"] is False
    assert report["cases"][0]["availability_reason"] == "benchmarks_disabled"
    assert (tmp_path / "report.json").exists()
    markdown = (tmp_path / "report.md").read_text()
    assert "M11.P Vision Conv2D Performance Baseline" in markdown
    assert "M11.5 Baseline Hardening" in markdown


def test_m116_grid2d_native_artifact_contract_and_dashboard_schema(tmp_path):
    semantics = VisionGrid2DPointwiseSemantics(
        case_name="m116_grid2d_add_smoke",
        model="yolov8n_yaml_random",
        grid_family=M11_GRID_FAMILY_CONV_ADJACENT_POINTWISE,
        x_extent=64,
        y_extent=8,
        op_kind="add",
        block_size=32,
    )
    source = build_native_grid2d_pointwise_tirx_source(semantics)
    irmod = tvm.script.from_source(source)
    validate_pointwise_grid2d_static_contract(irmod)
    validate_triton_tvm_contract(irmod, VISION_CONTRACT_POINTWISE_GRID2D_STATIC)
    script = irmod.script()
    assert "blockIdx.x" in script
    assert "blockIdx.y" in script
    assert "threadIdx.x" in script
    assert "call_packed" not in script

    report = run_vision_grid2d_dashboard(
        out_dir=tmp_path,
        warmup=1,
        repeat=1,
        run_benchmarks=False,
    )
    assert report["report_kind"] == "triton_tvm_m11_6_vision_perf_dashboard"
    assert report["readiness_status"] == VISION_M11_6_INTERFACE_STATUS
    assert report["summary"]["total_cases"] == 5
    assert report["summary"]["native_tvm_grid2d_cases"] == 5
    assert report["summary"]["host_staged_cases"] == 0
    assert report["invariants"]["status"] == "passed"
    assert report["cases"][0]["contract"] == VISION_CONTRACT_POINTWISE_GRID2D_STATIC
    assert report["cases"][0]["implementation_kind"] == M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM
    assert report["cases"][0]["host_staging_bytes"] == 0
    assert report["cases"][0]["performance_claim"] is True
    assert report["cases"][0]["availability_reason"] == "benchmarks_disabled"
    assert (tmp_path / "report.json").exists()
    assert "M11.6 Vision Grid2D Performance Dashboard" in (tmp_path / "report.md").read_text()


@tvm.testing.requires_cuda
def test_m114_extern_conv2d_provider_build_run_matches_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is required for the M11.4 conv2d provider")

    semantics = _extract_semantics(
        _conv_wrapper_source(
            input_shape=(1, 3, 32, 32),
            weight_shape=(64, 3, 16, 16),
            output_shape=(1, 64, 2, 2),
            stride=(16, 16),
            padding=(0, 0),
        ),
        case_name="vit_tiny_random",
    )
    decision = TargetVisionPolicy(
        vision_runtime_provider=VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, vision_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_conv2d_tirx_source(semantics, decision))
    validate_vision_conv2d_contract(irmod)
    built = build_triton_tvm(irmod, _extern_conv2d_meta(semantics))

    dev = tvm.cuda(0)
    input_np = np.arange(1 * 3 * 32 * 32, dtype="float32").reshape(1, 3, 32, 32) / 97.0
    weight_np = (
        np.arange(64 * 3 * 16 * 16, dtype="float32").reshape(64, 3, 16, 16) / 101.0
    )
    out_tvm = tvm.runtime.empty((1, 64, 2, 2), "float32", dev)

    with register_python_torch_extern_conv2d():
        built.run(
            [
                tvm.runtime.tensor(input_np, dev),
                tvm.runtime.tensor(weight_np, dev),
                out_tvm,
            ]
        )
    expected = torch.nn.functional.conv2d(
        torch.tensor(input_np, device="cuda"),
        torch.tensor(weight_np, device="cuda"),
        stride=(16, 16),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    tvm.testing.assert_allclose(
        out_tvm.numpy(),
        expected.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def _extract_single_conv(
    source: str,
    *,
    case_name: str,
    vision_runtime_provider: str = VISION_PROVIDER_NONE,
):
    calls = extract_inductor_wrapper_extern_calls(
        source,
        case_name=case_name,
        wrapper_path=f"/tmp/{case_name}.py",
        vision_runtime_provider=vision_runtime_provider,
    )
    conv_calls = [call for call in calls if call.op_family == "deferred_convolution"]
    assert len(conv_calls) == 1
    return conv_calls[0]


def _grid_record(
    *,
    kernel_name,
    model_family="yolo",
    grid_type="Grid2D",
    blocker_class="grid",
    ok=False,
):
    return {
        "model_family": model_family,
        "model_case": f"{model_family}_tiny_random",
        "kernel_name": kernel_name,
        "grid_type": grid_type,
        "blocker_class": blocker_class,
        "size_hints": {"x": 16, "y": 64},
        "indexing_summary": {"program_id_axes": ["x", "y"]},
        "translate_status": make_report_status(
            ok=ok,
            bucket="translated" if ok else "contract_error",
            fallback_reason=None if ok else "unsupported_inductor_kernel",
        ),
    }


def _extract_semantics(source: str, *, case_name: str):
    call = _extract_single_conv(source, case_name=case_name)
    return extract_vision_conv2d_semantics_from_wrapper_extern(
        {
            "op_name": call.op_name,
            "source": call.source,
            "full_source": source,
            "line_no": call.line_no,
            "case_name": case_name,
            "kernel_name": case_name,
        }
    )


def _model_record_with_call(model_family: str, case_name: str, call) -> dict:
    return {
        "model_family": model_family,
        "model_case": case_name,
        "status": "completed",
        "kernel_count": 0,
        "translated_kernels": 0,
        "native_fallback_kernels": 0,
        "status_buckets": {},
        "blocker_classes": {},
        "wrapper_paths": [f"/tmp/{case_name}.py"],
        "extern_calls": [dict(vars(call))],
        "triton_kernel_runnable": False,
        "full_tvm_runnable": False,
    }


def _extern_conv2d_meta(semantics) -> TritonTVMMeta:
    return TritonTVMMeta(
        kernel_name=semantics.kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract=semantics.vision_contract,
        canonical_contract=semantics.vision_contract,
        requested_contract=semantics.vision_contract,
        emit="tir",
        translator_version="m114_extern_conv2d_runtime_proof",
        contract_version="conv2d_nchw_static_m11_v1",
        target_policy_version="m114_extern_conv2d_runtime_proof",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=_numel(semantics.output_shape),
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=(
            semantics.weight_shape[1] * semantics.weight_shape[2] * semantics.weight_shape[3]
        ),
        buffer_extents={
            semantics.input_param: f"T.int64({_numel(semantics.input_shape)})",
            semantics.weight_param: f"T.int64({_numel(semantics.weight_shape)})",
            semantics.output_param: f"T.int64({_numel(semantics.output_shape)})",
        },
        block_size=1,
        indexing_kind="rank4_conv2d_runtime_proof",
        execution_kind="extern_conv2d_python_torch_host_staged",
        accumulator_dtype_policy="fp32_conv2d_runtime_proof",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="nchw_oihw_static",
        layout_policy="rank4_static_conv2d_runtime_proof",
        launch_policy_id="extern_conv2d_runtime_provider_proof",
        abi=[
            {"name": semantics.input_param, "kind": "pointer", "dtype": "float32"},
            {"name": semantics.weight_param, "kind": "pointer", "dtype": "float32"},
            {"name": semantics.output_param, "kind": "pointer", "dtype": "float32"},
        ],
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"{semantics.kernel_name}_m114_runtime_proof",
        implementation_kind="extern_conv2d",
        extern_symbol="extern_kernels.convolution",
    )


def _conv_wrapper_source(
    *,
    input_shape=(1, 3, 64, 64),
    weight_shape=(16, 3, 3, 3),
    output_shape=(1, 16, 32, 32),
    stride=(2, 2),
    padding=(1, 1),
    dilation=(1, 1),
    groups=1,
    bias="None",
    transposed=False,
    output_padding=(0, 0),
) -> str:
    input_stride = _channels_last_stride(input_shape)
    weight_stride = _channels_last_stride(weight_shape)
    output_stride = _channels_last_stride(output_shape)
    transposed_text = "True" if transposed else "False"
    return f"""
def call(arg0, arg1, bias):
    buf0 = empty_strided_cuda({_tuple_literal(input_shape)}, {_tuple_literal(input_stride)}, torch.float32)
    buf1 = empty_strided_cuda({_tuple_literal(weight_shape)}, {_tuple_literal(weight_stride)}, torch.float32)
    buf2 = extern_kernels.convolution(
        buf0,
        buf1,
        stride={_tuple_literal(stride)},
        padding={_tuple_literal(padding)},
        dilation={_tuple_literal(dilation)},
        transposed={transposed_text},
        output_padding={_tuple_literal(output_padding)},
        groups={groups},
        bias={bias},
    )
    assert_size_stride(buf2, {_tuple_literal(output_shape)}, {_tuple_literal(output_stride)}, 'torch.ops.aten.convolution.default')
    return buf2
"""


def _channels_last_stride(shape):
    _, channels, height, width = shape
    return (channels * height * width, 1, width * channels, channels)


def _tuple_literal(values):
    return "(" + ", ".join(str(value) for value in values) + ("," if len(values) == 1 else "") + ")"


def _numel(shape):
    result = 1
    for value in shape:
        result *= int(value)
    return result


if __name__ == "__main__":
    tvm.testing.main()
