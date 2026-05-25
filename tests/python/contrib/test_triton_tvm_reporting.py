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
"""Pre-M5 capability report schema tests."""

import json

import tvm.testing
from tvm.contrib.triton_tvm import (
    TritonTVMContractError,
    UnsupportedContractError,
    UnsupportedStreamError,
    UnsupportedTTIROpError,
    UnsupportedTargetPolicyError,
)
from tvm.contrib.triton_tvm.reporting import (
    build_capability_report,
    bucket_for_exception,
    make_report_status,
    render_capability_markdown,
    status_from_exception,
    write_capability_report,
)


def test_reporting_schema_snapshot_for_pointwise_reduction_and_norm():
    records = [
        _record("m3", "add", "pointwise_flat", make_report_status(ok=True, bucket="translated")),
        _record(
            "m35",
            "indexed",
            "pointwise_flat",
            make_report_status(ok=True, bucket="translated"),
            op_counts={"tt.load": 2, "tt.store": 1, "arith.remsi": 1},
        ),
        _record(
            "m4",
            "row_sum",
            "reduction_minimal",
            status_from_exception(UnsupportedTTIROpError("tt.max is unsupported")),
            op_counts={"tt.reduce": 1, "tt.load": 1},
        ),
        _record(
            "m4",
            "layernorm",
            "norm_single_row",
            make_report_status(ok=True, bucket="translated"),
            op_counts={"tt.reduce": 2, "tt.load": 3, "tt.store": 1},
        ),
    ]

    report = build_capability_report(
        records,
        purpose="snapshot",
        corpus="pre_m5",
        generated_at="2026-05-22T00:00:00+00:00",
    )

    assert report["schema_version"] == 1
    assert report["report_kind"] == "triton_tvm_capability_report"
    assert report["summary"] == {
        "contracts": {
            "norm_single_row": 1,
            "pointwise_flat": 2,
            "reduction_minimal": 1,
        },
        "corpora": {"m3": 1, "m35": 1, "m4": 2},
        "status_buckets": {"translated": 3, "unsupported_ttir_op": 1},
        "total_kernels": 4,
        "translated_kernels": 3,
    }
    assert report["unsupported_buckets"] == report["summary"]["status_buckets"]
    assert report["op_histogram"] == {
        "arith.remsi": 1,
        "tt.load": 8,
        "tt.reduce": 3,
        "tt.store": 3,
    }
    assert report["kernels"][2]["translate_status"]["fallback_reason"] == "unsupported_ttir_op"
    assert report["supported_kernel_report"]["kernel_count"] == 3
    assert report["supported_kernel_report"]["contracts"] == {
        "norm_single_row": 1,
        "pointwise_flat": 2,
    }
    assert report["supported_kernel_report"]["legacy_contract_records"] == []
    assert report["supported_kernel_report"]["silent_fallback_records"] == []


def test_reporting_error_bucket_mapping_is_stable():
    cases = [
        (UnsupportedTTIROpError("bad op"), "unsupported_ttir_op"),
        (TritonTVMContractError("bad contract shape"), "contract_error"),
        (UnsupportedContractError("bad contract"), "contract_error"),
        (UnsupportedTargetPolicyError("bad target"), "target_policy_error"),
        (UnsupportedStreamError("bad stream"), "unsupported_stream"),
        (ValueError("bad input"), "input_error"),
        (RuntimeError("unexpected"), "internal_error"),
    ]

    for err, bucket in cases:
        assert bucket_for_exception(err) == bucket
        status = status_from_exception(err)
        assert status["bucket"] == bucket
        assert status["fallback_reason"] == bucket


def test_reporting_markdown_and_json_are_consistent(tmp_path):
    report = build_capability_report(
        [
            _record(
                "m4",
                "bad_axis",
                "reduction_minimal",
                status_from_exception(UnsupportedTTIROpError("axis=1")),
                op_counts={"tt.reduce": 1},
            )
        ],
        purpose="consistency",
        corpus="m4",
        generated_at="2026-05-22T00:00:00+00:00",
    )

    markdown = render_capability_markdown(report, title="Snapshot")
    assert "Total kernels: 1" in markdown
    assert "unsupported_ttir_op=1" in markdown
    assert "`reduction_minimal`" in markdown
    assert "axis=1" in markdown

    write_capability_report(report, tmp_path, markdown_title="Snapshot")
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    written_markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert written["summary"] == report["summary"]
    assert "unsupported_ttir_op=1" in written_markdown
    assert "`reduction_minimal`" in written_markdown


def test_m75_supported_kernel_report_tracks_shape_dtype_layout_and_guards():
    records = [
        _record(
            "m7",
            "fp16_broadcast",
            "pointwise_flat",
            make_report_status(ok=True, bucket="translated"),
            op_counts={"tt.load": 2, "tt.store": 1, "arith.remsi": 1},
        )
        | {
            "signature": {"x": "*fp16", "y": "*bf16", "out": "*fp32", "n": "i64"},
            "types": [
                "tensor<8x64xf16>",
                "tensor<64xbf16>",
                "tensor<64xi1>",
                "tensor<8x!tt.ptr<f16>>",
            ],
            "indexing_summary": {"index_kinds": {"flat": 2, "rem": 1}},
        },
        _record(
            "m7",
            "legacy_alias_input",
            "cuda_pointwise_flat",
            make_report_status(ok=True, bucket="translated"),
        ),
        _record(
            "m7",
            "missing_reason",
            "pointwise_flat",
            {"ok": False, "bucket": "unsupported_ttir_op"},
        ),
    ]

    report = build_capability_report(
        records,
        purpose="m7.5 supported kernel guard",
        corpus="m7",
        generated_at="2026-05-24T00:00:00+00:00",
    )
    supported = report["supported_kernel_report"]

    assert "cuda_" not in json.dumps(report)
    assert report["kernels"][1]["contract"] == "pointwise_flat"
    assert supported["kernel_count"] == 2
    assert supported["dtypes"]["float16"] >= 1
    assert supported["dtypes"]["bfloat16"] >= 1
    assert supported["dtypes"]["float32"] >= 1
    assert all(not dtype.startswith("!tt.ptr") for dtype in supported["dtypes"])
    assert supported["tensor_shapes"]["8x64"] == 1
    assert supported["layout_index_kinds"] == {"flat": 2, "rem": 1}
    assert supported["legacy_contract_records"] == [
        {
            "corpus": "m7",
            "case_name": "legacy_alias_input",
            "kernel_name": "legacy_alias_input_kernel",
            "contract": "pointwise_flat",
            "bucket": "translated",
            "fallback_reason": "",
        }
    ]
    assert supported["silent_fallback_records"] == [
        {
            "corpus": "m7",
            "case_name": "missing_reason",
            "kernel_name": "missing_reason_kernel",
            "contract": "pointwise_flat",
            "bucket": "unsupported_ttir_op",
            "fallback_reason": "unsupported_ttir_op",
        }
    ]
    assert supported["ok"] is False


def test_m9_report_preserves_matmul_policy_detail_fields():
    record = _record(
        "m9",
        "dot",
        "matmul_minimal",
        make_report_status(
            ok=False,
            bucket="target_policy_error",
            fallback_reason="matmul_policy_unsupported",
            message="matmul_no_enabled_implementation",
        ),
        op_counts={"tt.load": 2, "tt.dot": 1, "tt.store": 1},
    ) | {
        "types": ["tensor<8x16xf16>", "tensor<16x4xf16>", "tensor<8x4xf32>"],
        "matmul_source_kind": "tt_dot",
        "matmul_m": 8,
        "matmul_n": 4,
        "matmul_k": 16,
        "matmul_contract_ok": True,
        "implementation_kind": "unsupported",
        "schedule_id": "",
        "extern_symbol": "",
        "unsupported_matmul_reason": "matmul_no_enabled_implementation",
    }

    report = build_capability_report(
        [record],
        purpose="m9 matmul policy detail guard",
        corpus="m9",
        generated_at="2026-05-25T00:00:00+00:00",
    )

    kernel = report["kernels"][0]
    assert report["summary"]["status_buckets"] == {"target_policy_error": 1}
    assert report["supported_kernel_report"]["kernel_count"] == 0
    assert kernel["translate_status"]["fallback_reason"] == "matmul_policy_unsupported"
    assert kernel["matmul_source_kind"] == "tt_dot"
    assert (kernel["matmul_m"], kernel["matmul_n"], kernel["matmul_k"]) == (
        8,
        4,
        16,
    )
    assert kernel["matmul_contract_ok"] is True
    assert kernel["implementation_kind"] == "unsupported"
    assert kernel["schedule_id"] == ""
    assert kernel["extern_symbol"] == ""
    assert kernel["unsupported_matmul_reason"] == "matmul_no_enabled_implementation"


def _record(corpus, case, contract, status, op_counts=None):
    op_counts = op_counts or {"tt.load": 2, "tt.store": 1}
    return {
        "corpus": corpus,
        "case_name": case,
        "kernel_name": f"{case}_kernel",
        "contract": contract,
        "op_counts": op_counts,
        "types": ["tensor<64xf32>"],
        "load_count": op_counts.get("tt.load", 0),
        "store_count": op_counts.get("tt.store", 0),
        "translate_status": status,
    }


if __name__ == "__main__":
    tvm.testing.main()
