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
    build_model_corpus_report,
    builtin_model_audit_cases,
    diff_capability_reports,
    run_model_corpus_audit,
    write_model_corpus_report,
)
from tvm.contrib.triton_tvm.reporting import make_report_status, render_capability_markdown

try:
    import torch
except ImportError:
    torch = None


def test_m6_model_corpus_api_is_experimental_not_top_level_public_api():
    assert "TritonTVMModelAuditConfig" not in triton_tvm_pkg.__all__
    assert "run_model_corpus_audit" not in triton_tvm_pkg.__all__


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
        "types": ["tensor<64xf32>"],
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


def _model_record(family, model_case, *, kernel_count, translated, fallback):
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
        "full_tvm_runnable": kernel_count > 0 and fallback == 0 and translated == kernel_count,
    }


if __name__ == "__main__":
    tvm.testing.main()
