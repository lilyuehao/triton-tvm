from tvm.contrib.triton_tvm.atomic import build_atomic_dag
from tvm.contrib.triton_tvm.contracts import validate_atomic_contract
from tvm.contrib.triton_tvm.manifest import (
    build_manifest_gap,
    build_operator_manifest_record,
    recover_manifest_proof_facts,
)
from tvm.contrib.triton_tvm.semantic import build_semantic_region
from tvm.contrib.triton_tvm.source import captured_inductor_jit_kernel


TTIR = """
module {
  tt.func public @matmul(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %0 = tt.dot %a, %b : (!tt.ptr<f32>, !tt.ptr<f32>) -> tensor<16x16xf32>
    tt.store %c, %0 : !tt.ptr<f32>, tensor<16x16xf32>
  }
}
"""


def _records():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validation = validate_atomic_contract(source, atomic, contract_id="matmul")
    semantic = build_semantic_region(source, atomic, validation)
    return source, atomic, semantic


def test_manifest_identity_is_model_id_plus_operator_id():
    source, atomic, semantic = _records()
    first = build_operator_manifest_record(
        model_id="vit",
        operator_id="addmm_0",
        execution_order=0,
        source=source,
        atomic=atomic,
        semantic_region=semantic,
    )
    second = build_operator_manifest_record(
        model_id="vit",
        operator_id="addmm_1",
        execution_order=1,
        source=source,
        atomic=atomic,
        semantic_region=semantic,
    )
    assert first.identity == ("vit", "addmm_0")
    assert second.identity == ("vit", "addmm_1")
    assert first.semantic_region_key == second.semantic_region_key == "matmul"


def test_manifest_proof_facts_are_recovered_by_joins():
    source, atomic, semantic = _records()
    record = build_operator_manifest_record(
        model_id="vit",
        operator_id="addmm_0",
        execution_order=0,
        source=source,
        atomic=atomic,
        semantic_region=semantic,
    )
    assert "source_hash" not in record.__dataclass_fields__
    facts = recover_manifest_proof_facts(
        record,
        sources={source.source_id: source},
        atomics={atomic.atomic_dag_hash: atomic},
        semantics={semantic.semantic_region_hash: semantic},
    )
    assert facts["source_origin"] == "torch_inductor"
    assert facts["matched_contract_id"] == "matmul"


def test_unsupported_cases_become_manifest_gaps():
    gap = build_manifest_gap(
        model_id="vit",
        operator_id="conv_0",
        source_id="source:conv",
        reason="missing contract validation",
    )
    assert gap.reason == "missing contract validation"

