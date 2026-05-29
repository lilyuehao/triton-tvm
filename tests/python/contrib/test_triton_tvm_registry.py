from tvm.contrib.triton_tvm.atomic import build_atomic_dag
from tvm.contrib.triton_tvm.contracts import validate_atomic_contract
from tvm.contrib.triton_tvm.registry import CapabilityRecord, CapabilityRegistry, LOOKUP_ORDER
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


def _route():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validation = validate_atomic_contract(source, atomic, contract_id="matmul")
    semantic = build_semantic_region(source, atomic, validation)
    return source, atomic, semantic


def test_registry_resolves_in_required_order():
    source, atomic, semantic = _route()
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
    result = registry.resolve(source=source, atomic=atomic, semantic_region=semantic)
    assert result.supported
    assert result.query_trace == LOOKUP_ORDER


def test_source_origin_policy_is_checked_after_semantic_and_atomic_requirements():
    source, atomic, semantic = _route()
    registry = CapabilityRegistry(
        [
            CapabilityRecord(
                capability_id="matmul.cuda",
                semantic_region_key="matmul",
                required_atomic_families=("dot",),
                shape_constraints={},
                dtype_constraints={},
                layout_constraints={},
                allowed_source_origins=(SourceOrigin.NATIVE_TRITON_LANGUAGE,),
                runtime_available=True,
            )
        ]
    )
    result = registry.resolve(source=source, atomic=atomic, semantic_region=semantic)
    assert not result.supported
    assert result.query_trace == LOOKUP_ORDER[:4]
    assert result.failure_reason == "source origin policy rejected input"

