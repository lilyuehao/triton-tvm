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
from tvm.contrib.triton_tvm.model_corpus import (
    TritonTVMModelAuditConfig,
    _is_model_kernel_replaceable,
    _select_model_kernel_contract,
    build_model_corpus_report,
    builtin_model_audit_cases,
    diff_capability_reports,
    run_model_corpus_audit,
    write_model_corpus_report,
)
from tvm.contrib.triton_tvm.inductor import InductorKernel, extract_inductor_wrapper_extern_calls
from tvm.contrib.triton_tvm.reporting import make_report_status, render_capability_markdown

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


def test_pre_m9_wrapper_extern_extraction_classifies_matmul_conv_attention():
    wrapper_source = """
def call(arg0, arg1):
    buf0 = extern_kernels.mm(arg0, arg1, out=None)
    buf1 = extern_kernels.addmm(arg0, arg1, arg1, alpha=1, beta=1, out=None)
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
                _extern_call("vit", "vit_tiny", "extern_kernels.addmm", "extern_addmm_bias"),
                _extern_call("vit", "vit_tiny", "extern_kernels.mm", "extern_gemm"),
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

    pre_m9 = report["pre_m9"]
    assert pre_m9["taxonomy_version"] == 1
    assert pre_m9["observed_ttir_dot_kernels"] == 0
    assert pre_m9["observed_grid_fallback_kernels"] == 1
    assert pre_m9["extern_family_classes"]["extern_gemm"]["call_count"] == 1
    assert pre_m9["extern_family_classes"]["extern_addmm_bias"]["call_count"] == 1
    assert pre_m9["extern_family_classes"]["deferred_convolution"]["call_count"] == 1
    assert pre_m9["extern_family_classes"]["deferred_attention"]["call_count"] == 1
    assert {entry["op_family"] for entry in pre_m9["m9_entry_debt"]} == {
        "extern_addmm_bias",
        "extern_gemm",
    }
    assert {entry["op_family"] for entry in pre_m9["deferred_debt"]} == {
        "deferred_attention",
        "deferred_convolution",
    }

    markdown = render_capability_markdown(report, title="Pre-M9 Snapshot")
    assert "## Pre-M9 Gate" in markdown
    assert "extern_gemm" in markdown
    assert "deferred_convolution" in markdown


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


def _extern_call(family, model_case, op_name, op_family):
    return {
        "model_family": family,
        "model_case": model_case,
        "case_name": model_case,
        "op_name": op_name,
        "op_family": op_family,
        "line_no": 10,
        "wrapper_path": f"/tmp/{model_case}.py",
        "source": f"{op_name}(...)",
    }


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
