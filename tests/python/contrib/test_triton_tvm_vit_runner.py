import pytest

from tvm.contrib.triton_tvm.atomic import build_atomic_dag
from tvm.contrib.triton_tvm.models.vit import (
    build_vit_atomic_dag_generated_tir_artifacts,
    build_vit_tiny_fixed_shape_adapter,
    run_vit_tiny_fixed_shape_route,
    vit_execution_requests_from_report,
    vit_tiny_fixed_shape_top_level_specs,
    vit_tiny_fixed_shape_specs,
)


def test_vit_runner_materializes_fixed_shape_inventory_without_support_claims():
    specs = vit_tiny_fixed_shape_specs()
    top_specs = vit_tiny_fixed_shape_top_level_specs()
    adapter = build_vit_tiny_fixed_shape_adapter()

    assert len(specs) == 16
    assert len(top_specs) == 14
    assert len(adapter.operators()) == 16
    assert {item.source.source_origin.value for item in adapter.operators()} == {"torch_inductor"}
    assert {item.source.artifact_kind.value for item in adapter.operators()} == {
        "captured_jit_kernel"
    }


def test_vit_runner_compiles_and_runs_active_route(tmp_path):
    report = run_vit_tiny_fixed_shape_route(out_dir=tmp_path)

    assert report["status"] == "passed"
    assert report["performance_claim"] is False
    assert report["operator_count"] == 14
    assert report["top_level_operator_count"] == 14
    assert report["source_route_record_count"] == 16
    assert report["atomic_dag_built_operator_count"] == 16
    assert report["semantic_region_count"] == 14
    assert report["source_route_semantic_region_count"] == 16
    assert report["runtime_admitted_operator_count"] == 14
    assert report["gap_count"] == 0
    assert {record["runtime_admission_status"] for record in report["operators"]} == {
        "admitted"
    }
    assert (tmp_path / "report.json").exists()


def test_vit_runner_e2e_scaffold_records_placeholders(tmp_path):
    report = run_vit_tiny_fixed_shape_route(out_dir=tmp_path, enable_e2e_scaffold=True)
    diagnostic = report["diagnostic_e2e"]

    assert report["performance_claim"] is False
    assert diagnostic["execution_mode"] == "diagnostic_e2e"
    assert diagnostic["status"] == "planned_not_executed"
    assert diagnostic["performance_claim"] is False
    assert diagnostic["planned_operator_count"] == 14
    assert diagnostic["placeholder_operator_count"] == 14
    assert diagnostic["executed_operator_count"] == 0
    assert "p50" not in diagnostic
    assert "p95" not in diagnostic


def test_vit_runner_synthetic_tir_reference_and_tvm_allclose_14_of_14():
    report = run_vit_tiny_fixed_shape_route(target="llvm", enable_synthetic_tir=True)
    diagnostic = report["diagnostic_e2e"]

    assert diagnostic["status"] == "allclose"
    assert diagnostic["planned_operator_count"] == 14
    assert diagnostic["reference_executed_count"] == 14
    assert diagnostic["tvm_executed_count"] == 14
    assert diagnostic["allclose_count"] == 14
    assert diagnostic["not_run_count"] == 0
    assert diagnostic["performance_claim"] is False
    assert diagnostic["runtime_claim"] == "synthetic_runtime_channel_only"
    assert diagnostic["e2e_pass"] is False
    assert diagnostic["atomic_dag_lowering_coverage_claim"] is False
    assert diagnostic["artifact_source_breakdown"] == {"synthetic_tir": 14}
    assert diagnostic["model_comparison_status"] == "allclose"
    assert diagnostic["model_output_count"] == 2
    assert diagnostic["model_allclose_count"] == 2


def test_vit_inventory_uses_canonical_region_keys_and_statuses():
    report = run_vit_tiny_fixed_shape_route()
    operators = {record["operator_id"]: record for record in report["operators"]}

    assert operators["patch_embed"]["semantic_region_key"] == "conv_patchify"
    assert operators["patch_embed"]["lowering_contract_ids"] == ("conv2d_nchw_static",)
    assert operators["patch_embed"]["lowering_contract_status"] == "specialized_lowering"
    assert operators["patch_embed"]["lowering_contract_warning"] is None
    assert operators["encoder_norm0"]["semantic_region_key"] == "norm_row"
    assert operators["encoder_norm0"]["lowering_contract_ids"] == ("pointwise_flat",)
    assert operators["encoder_norm0"]["lowering_contract_status"] == "temporary_surrogate"
    assert operators["encoder_norm0"]["semantic_proof_status"] == "surrogate_validated"
    assert operators["class_token_add"]["semantic_region_key"] == "pointwise_grid2d"
    assert operators["class_token_add"]["lowering_contract_status"] == "flattened_surrogate"
    assert operators["qkv_projection"]["semantic_region_key"] == "matmul"
    assert operators["pooler"]["semantic_region_key"] == "matmul"


def test_vit_inventory_collapses_attention_internal_records():
    report = run_vit_tiny_fixed_shape_route()
    attention = next(record for record in report["operators"] if record["operator_id"] == "attention")

    assert report["top_level_operator_count"] == 14
    assert report["source_route_record_count"] == 16
    assert attention["semantic_region_key"] == "attention_decomposed"
    assert attention["source_route_record_ids"] == (
        "attention_qk",
        "attention_softmax",
        "attention_av",
    )


def test_vit_synthetic_tir_single_entry_reports_not_run_without_gap():
    report = run_vit_tiny_fixed_shape_route(
        target="llvm",
        enable_synthetic_tir=True,
        synthetic_tir_operator_ids=("residual_after_mlp",),
    )
    diagnostic = report["diagnostic_e2e"]

    assert diagnostic["planned_operator_count"] == 14
    assert diagnostic["tvm_executed_count"] == 1
    assert diagnostic["allclose_count"] == 1
    assert diagnostic["not_run_count"] == 13
    assert diagnostic["gap_count"] == 0
    assert diagnostic["failed_count"] == 0


def test_vit_runner_atomic_dag_generated_tir_replaces_synthetic_and_model_outputs_allclose():
    report = run_vit_tiny_fixed_shape_route(target="llvm", enable_atomic_dag_tir=True)
    diagnostic = report["diagnostic_e2e"]

    assert report["runtime_claim"] == "atomic_dag_generated_correctness_only"
    assert report["e2e_pass"] is True
    assert report["performance_claim"] is False
    assert diagnostic["status"] == "allclose"
    assert diagnostic["planned_operator_count"] == 14
    assert diagnostic["reference_executed_count"] == 14
    assert diagnostic["tvm_executed_count"] == 14
    assert diagnostic["allclose_count"] == 14
    assert diagnostic["artifact_source_breakdown"] == {"atomic_dag_generated_tir": 14}
    assert diagnostic["runtime_claim"] == "atomic_dag_generated_correctness_only"
    assert diagnostic["e2e_pass"] is True
    assert diagnostic["atomic_dag_lowering_coverage_claim"] is False
    assert diagnostic["model_comparison_status"] == "allclose"
    assert diagnostic["model_output_count"] == 2
    assert diagnostic["model_allclose_count"] == 2
    assert {record["buffer_id"] for record in diagnostic["model_outputs"]} == {
        "last_hidden_state",
        "pooler_output",
    }


def test_vit_runner_mixed_tir_artifact_sources_allclose_but_do_not_pass_e2e_gate():
    report = run_vit_tiny_fixed_shape_route(
        target="llvm",
        enable_synthetic_tir=True,
        enable_atomic_dag_tir=True,
        atomic_dag_tir_operator_ids=("patch_embed",),
    )
    diagnostic = report["diagnostic_e2e"]

    assert diagnostic["status"] == "allclose"
    assert diagnostic["allclose_count"] == 14
    assert diagnostic["artifact_source_breakdown"] == {
        "atomic_dag_generated_tir": 1,
        "synthetic_tir": 13,
    }
    assert diagnostic["model_comparison_status"] == "allclose"
    assert diagnostic["model_allclose_count"] == 2
    assert report["runtime_claim"] == "mixed_tir_artifact_correctness_only"
    assert report["e2e_pass"] is False
    assert diagnostic["e2e_pass"] is False


def test_atomic_dag_generated_tir_requires_matching_atomic_hash_bundle():
    report = run_vit_tiny_fixed_shape_route(target="llvm")
    requests = vit_execution_requests_from_report(report)
    adapter = build_vit_tiny_fixed_shape_adapter(target="llvm")
    atomics_by_route = {
        item.operator_id: build_atomic_dag(item.source, item.ttir)
        for item in adapter.operators()
    }
    patch_request = next(request for request in requests if request.operator_id == "patch_embed")
    bad_request = type(patch_request)(
        **{
            **patch_request.__dict__,
            "atomic_dag_hashes": ("not-the-patch-embed-hash",),
        }
    )

    with pytest.raises(ValueError, match="hash bundle"):
        build_vit_atomic_dag_generated_tir_artifacts(
            (bad_request,),
            atomics_by_route=atomics_by_route,
            target="llvm",
        )
