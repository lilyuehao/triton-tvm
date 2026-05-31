import pytest

from tvm.contrib.triton_tvm.atomic import build_atomic_dag
from tvm.contrib.triton_tvm.autotune import (
    AutotuneConfig,
    AutotuneWorkloadSpec,
    tune_or_load_workload,
)
from tvm.contrib.triton_tvm.models.vit import (
    VIT_S_SOURCE_ROUTE_RECORD_COUNT,
    VIT_S_TOP_LEVEL_OPERATOR_COUNT,
    build_vit_s_autotune_plan,
    build_vit_atomic_dag_gated_te_tir_artifacts,
    build_vit_s_atomic_dag_gated_te_tir_artifacts,
    build_vit_s_fixed_shape_adapter,
    load_vit_s_run_config,
    load_vit_s_weight_source,
    main as vit_main,
    register_vit_s_weight_source,
    build_vit_tiny_fixed_shape_adapter,
    run_vit_s_fixed_shape_route,
    run_vit_tiny_fixed_shape_route,
    VitSWeightBundle,
    VitSWeightSourceSpec,
    _latency_sample_summary,
    vit_s_weight_source_registry,
    vit_execution_requests_from_report,
    vit_s_fixed_shape_specs,
    vit_s_fixed_shape_top_level_specs,
    vit_tiny_fixed_shape_top_level_specs,
    vit_tiny_fixed_shape_specs,
)


VIT_S_TEMPLATE_CONFIG = (
    "python/tvm/contrib/triton_tvm/configs/vit_s_16_224_autotune_template.json"
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


def test_vit_s_runner_materializes_standard_inventory_without_support_claims():
    specs = vit_s_fixed_shape_specs()
    top_specs = vit_s_fixed_shape_top_level_specs()
    adapter = build_vit_s_fixed_shape_adapter()

    assert len(specs) == VIT_S_SOURCE_ROUTE_RECORD_COUNT
    assert len(top_specs) == VIT_S_TOP_LEVEL_OPERATOR_COUNT
    assert len(adapter.operators()) == VIT_S_SOURCE_ROUTE_RECORD_COUNT
    assert {item.source.source_origin.value for item in adapter.operators()} == {"torch_inductor"}
    assert {item.source.artifact_kind.value for item in adapter.operators()} == {
        "captured_jit_kernel"
    }


def test_vit_s_json_config_template_loads_and_builds_autotune_plan():
    kwargs = load_vit_s_run_config(VIT_S_TEMPLATE_CONFIG)
    config = kwargs["autotune_config"]
    plan = build_vit_s_autotune_plan(autotune_config=config)

    assert kwargs["target"] == "cuda"
    assert kwargs["vit_s_backend"] == "triton-tvm"
    assert kwargs["weight_source"] == "timm"
    assert config.enabled is True
    assert config.config_path == VIT_S_TEMPLATE_CONFIG
    assert config.config_hash
    assert plan.selected_operator_count == VIT_S_TOP_LEVEL_OPERATOR_COUNT
    assert plan.selected_workload_count == 13


def test_vit_s_json_config_rejects_invalid_operator_id(tmp_path):
    import json

    with open(VIT_S_TEMPLATE_CONFIG, encoding="utf-8") as file:
        config = json.load(file)
    config["autotune"]["operator_ids"] = ["patch_embed", "not_a_vit_s_operator"]
    path = tmp_path / "bad_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown autotune operator_id"):
        load_vit_s_run_config(path)


def test_vit_s_autotune_plan_dedupes_operator_ids_by_workload():
    config = AutotuneConfig(
        enabled=True,
        work_dir="/tmp/unit_test_vit_s_autotune",
        operator_ids=(
            "encoder00_qkv_projection",
            "encoder01_qkv_projection",
            "encoder00_qkv_projection",
            "encoder00_attention",
        ),
    )
    plan = build_vit_s_autotune_plan(autotune_config=config)

    assert plan.selected_operator_ids == (
        "encoder00_qkv_projection",
        "encoder01_qkv_projection",
        "encoder00_attention",
    )
    assert plan.selected_workload_count == 4
    assert plan.workload_reuse_map["tokens197_k384_n1152"] == (
        "encoder00_qkv_projection",
        "encoder01_qkv_projection",
    )
    assert {
        "attention_score_b1_heads6_tokens197_head64",
        "attention_softmax_b1_heads6_tokens197",
        "attention_apply_b1_heads6_tokens197_hidden384",
    }.issubset(set(plan.workload_reuse_map))


def test_tune_or_load_workload_reuses_existing_database_record(monkeypatch):
    import tvm.s_tir.meta_schedule as ms

    class FakeRecord:
        def as_json(self):
            return {"trace": "already_tuned"}

    class FakeDatabase:
        def __init__(self, *_, **__):
            self.record = FakeRecord()

        def query_ir_module(self, *_, **__):
            return "tuned_mod"

        def query_tuning_record(self, *_, **__):
            return self.record

        def get_all_tuning_records(self):
            return [self.record]

    def fail_tune(*_, **__):
        raise AssertionError("tune_tir should not run when a database record exists")

    monkeypatch.setattr(ms.database, "JSONDatabase", FakeDatabase)
    monkeypatch.setattr(ms, "tune_tir", fail_tune)

    result = tune_or_load_workload(
        AutotuneWorkloadSpec(
            workload_key="unit_workload",
            operator_id="unit_operator",
            operator_family="unit",
            tir_module=object(),
        ),
        target="cuda",
        config=AutotuneConfig(
            enabled=True,
            work_dir="/tmp/unit_test_autotune_reuse",
            max_trials_global=128,
            force_retune=False,
        ),
    )

    assert result.tuned_module == "tuned_mod"
    assert result.metadata["record_count"] == 1


def test_vit_cli_requires_config_and_rejects_legacy_flags():
    with pytest.raises(SystemExit) as error:
        vit_main(["--variant", "vit-s"])

    assert error.value.code == 2


def test_vit_s_runner_compiles_and_runs_standard_active_route():
    report = run_vit_s_fixed_shape_route(target="cuda")

    assert report["status"] == "passed"
    assert report["vit_s_backend"] == "triton-tvm"
    assert report["variant"] == "vit_s_16_224"
    assert report["performance_claim"] is False
    assert report["standard_vit_shape"]["image"] == 224
    assert report["standard_vit_shape"]["patch"] == 16
    assert report["standard_vit_shape"]["tokens"] == 197
    assert report["standard_vit_shape"]["hidden"] == 384
    assert report["standard_vit_shape"]["heads"] == 6
    assert report["standard_vit_shape"]["layers"] == 12
    assert report["top_level_operator_count"] == VIT_S_TOP_LEVEL_OPERATOR_COUNT
    assert report["source_route_record_count"] == VIT_S_SOURCE_ROUTE_RECORD_COUNT
    assert report["runtime_admitted_operator_count"] == VIT_S_TOP_LEVEL_OPERATOR_COUNT
    assert report["gap_count"] == 0
    assert report["buffer_plan"]["valid"] is True
    assert report["buffer_plan"]["final_output_buffer_ids"] == ("last_hidden_state", "logits")
    assert report["weight_source"]["source_id"] == "deterministic_random"


def test_vit_s_atomic_dag_gated_te_tir_artifact_builder_generates_125_artifacts():
    report = run_vit_s_fixed_shape_route(target="cuda", vit_s_backend="reference")
    requests = vit_execution_requests_from_report(report)
    adapter = build_vit_s_fixed_shape_adapter(target="cuda")
    atomics_by_route = {
        item.operator_id: build_atomic_dag(item.source, item.ttir)
        for item in adapter.operators()
    }
    bundle = load_vit_s_weight_source("deterministic_random", parameter_seed=0)

    artifacts = build_vit_s_atomic_dag_gated_te_tir_artifacts(
        requests,
        atomics_by_route=atomics_by_route,
        target="cuda",
        weight_bundle=bundle,
    )

    assert len(artifacts) == VIT_S_TOP_LEVEL_OPERATOR_COUNT
    assert {artifact.artifact_source for artifact in artifacts} == {
        "atomic_dag_gated_te_tir"
    }
    assert {artifact.model_id for artifact in artifacts} == {
        "vit_s_16_224_fixed_shape"
    }


def test_vit_s_weight_sources_are_registered_and_extensible():
    registry = vit_s_weight_source_registry()
    assert "deterministic_random" in registry.source_ids()
    assert "timm" in registry.source_ids()

    base = load_vit_s_weight_source("deterministic_random", parameter_seed=3)

    def custom_loader(**_: object):
        return VitSWeightBundle(
            source_id="unit_test_custom",
            provider_id="unit_test",
            values=base.values,
            metadata={"from_test": True},
        )

    register_vit_s_weight_source(
        VitSWeightSourceSpec(
            source_id="unit_test_custom",
            provider_id="unit_test",
            description="unit test source",
            default_config={},
        ),
        custom_loader,
    )
    custom = load_vit_s_weight_source("unit_test_custom")
    assert custom.metadata["from_test"] is True
    assert custom.values["image"].shape == (1, 3, 224, 224)


def test_vit_s_runner_torch_compile_outputs_allclose():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = run_vit_s_fixed_shape_route(
        target="cuda" if device == "cuda" else "llvm",
        enable_torch_compile=True,
        torch_compile_device=device,
    )
    comparison = report["torch_compile_comparison"]

    assert report["torch_compile_allclose"] is True
    assert report["model_comparison_status"] == "allclose"
    assert comparison["torch_compile_used"] is True
    assert comparison["status"] == "allclose"
    if device == "cuda":
        assert comparison["reference_source"] == "triton_tvm_atomic_dag_gated_te_tir"
    assert comparison["allclose_count"] == 2
    assert {record["buffer_id"] for record in comparison["outputs"]} == {
        "last_hidden_state",
        "logits",
    }


def test_vit_s_latency_summary_records_distribution():
    summary = _latency_sample_summary([1.0, 2.0, 4.0, 8.0])

    assert summary["count"] == 4
    assert summary["min_ms"] == 1.0
    assert summary["median_ms"] == 3.0
    assert summary["p90_ms"] == pytest.approx(6.8)
    assert summary["p95_ms"] == pytest.approx(7.4)
    assert summary["p99_ms"] == pytest.approx(7.88)
    assert summary["max_ms"] == 8.0
    assert summary["stddev_ms"] > 0.0


def test_vit_s_runner_loads_timm_weights_and_runs_classification():
    pytest.importorskip("timm")
    report = run_vit_s_fixed_shape_route(
        target="cuda",
        weight_source="timm",
        weight_source_config={
            "model_name": "vit_small_patch16_224",
            "pretrained": False,
            "image_source": "synthetic_gradient",
        },
        enable_torch_compile=True,
        enable_classification=True,
    )
    comparison = report["torch_compile_comparison"]
    classification = report["classification"]

    assert report["status"] == "passed"
    assert report["weight_source"]["source_id"] == "timm"
    assert report["weight_source"]["metadata"]["model_name"] == "vit_small_patch16_224"
    assert report["weight_source"]["metadata"]["hf_endpoint"] == "https://hf-mirror.com"
    assert comparison["compiled_source"] == "torch.compile(timm)"
    assert comparison["status"] == "allclose"
    assert comparison["reference_source"] == "triton_tvm_atomic_dag_gated_te_tir"
    assert classification["status"] == "classified"
    assert classification["backend"] == "triton-tvm"
    assert 0 <= classification["top1"]["index"] < 1000
    assert classification["logits_shape"] == (1, 1000)


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
    assert diagnostic["e2e_correctness_pass"] is False
    assert diagnostic["e2e_pass"] is False
    assert diagnostic["atomic_dag_lowering_coverage_claim"] is False
    assert diagnostic["dag_node_lowering_coverage_claim"] is False
    assert diagnostic["atomic_dag_role"] == "none"
    assert diagnostic["tir_generation_mode"] == "synthetic_operator_template_te"
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


def test_vit_runner_atomic_dag_gated_te_tir_replaces_synthetic_and_model_outputs_allclose():
    report = run_vit_tiny_fixed_shape_route(
        target="llvm", enable_atomic_dag_gated_te_tir=True
    )
    diagnostic = report["diagnostic_e2e"]

    assert report["runtime_claim"] == "atomic_dag_gated_te_correctness_only"
    assert report["e2e_correctness_pass"] is True
    assert report["e2e_pass"] is True
    assert report["e2e_pass_alias_for"] == "e2e_correctness_pass"
    assert report["performance_claim"] is False
    assert diagnostic["status"] == "allclose"
    assert diagnostic["planned_operator_count"] == 14
    assert diagnostic["reference_executed_count"] == 14
    assert diagnostic["tvm_executed_count"] == 14
    assert diagnostic["allclose_count"] == 14
    assert diagnostic["artifact_source_breakdown"] == {"atomic_dag_gated_te_tir": 14}
    assert diagnostic["tir_artifact_source_breakdown"] == {"atomic_dag_gated_te_tir": 14}
    assert diagnostic["runtime_claim"] == "atomic_dag_gated_te_correctness_only"
    assert diagnostic["e2e_correctness_pass"] is True
    assert diagnostic["e2e_pass"] is True
    assert diagnostic["e2e_pass_alias_for"] == "e2e_correctness_pass"
    assert diagnostic["e2e_gate_kind"] == "fixed_shape_vit_correctness"
    assert diagnostic["atomic_dag_role"] == "admission_gate"
    assert diagnostic["tir_generation_mode"] == "operator_template_te"
    assert diagnostic["atomic_dag_lowering_coverage_claim"] is False
    assert diagnostic["dag_node_lowering_coverage_claim"] is False
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
        enable_atomic_dag_gated_te_tir=True,
        atomic_dag_gated_te_operator_ids=("patch_embed",),
    )
    diagnostic = report["diagnostic_e2e"]

    assert diagnostic["status"] == "allclose"
    assert diagnostic["allclose_count"] == 14
    assert diagnostic["artifact_source_breakdown"] == {
        "atomic_dag_gated_te_tir": 1,
        "synthetic_tir": 13,
    }
    assert diagnostic["model_comparison_status"] == "allclose"
    assert diagnostic["model_allclose_count"] == 2
    assert report["runtime_claim"] == "mixed_tir_artifact_correctness_only"
    assert report["e2e_pass"] is False
    assert diagnostic["e2e_pass"] is False


def test_atomic_dag_gated_te_tir_requires_matching_atomic_hash_bundle():
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
        build_vit_atomic_dag_gated_te_tir_artifacts(
            (bad_request,),
            atomics_by_route=atomics_by_route,
            target="llvm",
        )
