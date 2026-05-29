from tvm.contrib.triton_tvm.atomic import build_atomic_dag, validate_source_atomic_join
from tvm.contrib.triton_tvm.contracts import validate_atomic_contract
from tvm.contrib.triton_tvm.manifest import build_operator_manifest_record
from tvm.contrib.triton_tvm.registry import CapabilityRecord, CapabilityRegistry
from tvm.contrib.triton_tvm.reports import build_alpha_smoke_report
from tvm.contrib.triton_tvm.runtime import admit_runtime
from tvm.contrib.triton_tvm.semantic import build_semantic_region
from tvm.contrib.triton_tvm.source import SourceOrigin, captured_inductor_jit_kernel


TTIR = """
module {
  tt.func public @matmul(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %0 = tt.dot %a, %b : (!tt.ptr<f32>, !tt.ptr<f32>) -> tensor<16x16xf32>
    tt.store %c, %0 : !tt.ptr<f32>, tensor<16x16xf32>
  }
}
"""


def test_alpha_smoke_path_has_no_silent_fallback_and_no_performance_claim():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validate_source_atomic_join(source, atomic)
    validation = validate_atomic_contract(source, atomic, contract_id="matmul")
    semantic = build_semantic_region(source, atomic, validation)
    manifest = build_operator_manifest_record(
        model_id="vit",
        operator_id="addmm_0",
        execution_order=0,
        source=source,
        atomic=atomic,
        semantic_region=semantic,
    )
    registry = CapabilityRegistry(
        [
            CapabilityRecord(
                capability_id="matmul.cuda",
                semantic_region_key="matmul",
                required_atomic_families=("dot",),
                shape_constraints={},
                dtype_constraints={},
                layout_constraints={},
                allowed_source_origins=(SourceOrigin.TORCH_INDUCTOR,),
                runtime_available=True,
            )
        ]
    )
    resolution = registry.resolve(source=source, atomic=atomic, semantic_region=semantic)
    admission = admit_runtime(
        source=source,
        atomic=atomic,
        validation=validation,
        semantic_region=semantic,
        manifest_record=manifest,
        registry_resolution=resolution,
    )
    report = build_alpha_smoke_report(
        sources=[source],
        atomics=[atomic],
        semantics=[semantic],
        manifests=[manifest],
    )
    assert admission.admitted
    assert report["performance_claim"] is False
    assert report["manifest_summary"]["supported_records"] == 1

