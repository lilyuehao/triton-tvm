from tvm.contrib.triton_tvm.atomic import build_atomic_dag
from tvm.contrib.triton_tvm.contracts import validate_atomic_contract
from tvm.contrib.triton_tvm.manifest import build_operator_manifest_record
from tvm.contrib.triton_tvm.registry import CapabilityRecord, CapabilityRegistry
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


def _registry():
    return CapabilityRegistry(
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


def test_runtime_admission_accepts_only_final_validated_route():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
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
    resolution = _registry().resolve(source=source, atomic=atomic, semantic_region=semantic)
    admission = admit_runtime(
        source=source,
        atomic=atomic,
        validation=validation,
        semantic_region=semantic,
        manifest_record=manifest,
        registry_resolution=resolution,
    )
    assert admission.admitted


def test_runtime_admission_turns_failed_validation_into_gap():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validation = validate_atomic_contract(source, atomic, contract_id="unknown_contract")
    manifest = build_operator_manifest_record(
        model_id="vit",
        operator_id="addmm_0",
        execution_order=0,
        source=source,
        atomic=atomic,
        semantic_region=None,
        gap_reason=validation.failure_reason,
    )
    admission = admit_runtime(
        source=source,
        atomic=atomic,
        validation=validation,
        semantic_region=None,
        manifest_record=manifest,
        registry_resolution=_registry().resolve(
            source=source,
            atomic=atomic,
            semantic_region=build_semantic_region(
                source,
                atomic,
                validate_atomic_contract(source, atomic, contract_id="matmul"),
            ),
        ),
    )
    assert not admission.admitted
    assert admission.status == "gap"

