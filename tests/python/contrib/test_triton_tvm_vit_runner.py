from tvm.contrib.triton_tvm.models.vit import (
    build_vit_tiny_fixed_shape_adapter,
    run_vit_tiny_fixed_shape_route,
    vit_tiny_fixed_shape_specs,
)


def test_vit_runner_materializes_fixed_shape_inventory_without_support_claims():
    specs = vit_tiny_fixed_shape_specs()
    adapter = build_vit_tiny_fixed_shape_adapter()

    assert len(specs) == 16
    assert len(adapter.operators()) == 16
    assert {item.source.source_origin.value for item in adapter.operators()} == {"torch_inductor"}
    assert {item.source.artifact_kind.value for item in adapter.operators()} == {
        "captured_jit_kernel"
    }


def test_vit_runner_compiles_and_runs_active_route(tmp_path):
    report = run_vit_tiny_fixed_shape_route(out_dir=tmp_path)

    assert report["status"] == "passed"
    assert report["performance_claim"] is False
    assert report["operator_count"] == 16
    assert report["atomic_dag_built_operator_count"] == 16
    assert report["semantic_region_count"] == 16
    assert report["runtime_admitted_operator_count"] == 16
    assert report["gap_count"] == 0
    assert {record["runtime_admission_status"] for record in report["operators"]} == {
        "admitted"
    }
    assert (tmp_path / "report.json").exists()
