import pytest

from tvm.contrib.triton_tvm.atomic import build_atomic_dag
from tvm.contrib.triton_tvm.contracts import validate_atomic_contract
from tvm.contrib.triton_tvm.semantic import build_graph_optimization_input, build_semantic_region
from tvm.contrib.triton_tvm.source import captured_inductor_jit_kernel


TTIR = """
module {
  tt.func public @matmul(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %0 = tt.dot %a, %b : (!tt.ptr<f32>, !tt.ptr<f32>) -> tensor<16x16xf32>
    tt.store %c, %0 : !tt.ptr<f32>, tensor<16x16xf32>
  }
}
"""


def test_contract_validator_promotes_valid_atomic_dag_to_semantic_region():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validation = validate_atomic_contract(source, atomic, contract_id="matmul")
    semantic = build_semantic_region(source, atomic, validation)
    assert validation.passed
    assert semantic.semantic_region_key == "matmul"
    assert semantic.input_atomic_dag_hash == atomic.atomic_dag_hash


def test_failed_contract_validation_does_not_create_semantic_region():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validation = validate_atomic_contract(source, atomic, contract_id="unknown_contract")
    assert not validation.passed
    with pytest.raises(ValueError):
        build_semantic_region(source, atomic, validation)


def test_future_graph_optimization_input_receives_semantic_and_atomic_records():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, TTIR)
    validation = validate_atomic_contract(source, atomic, contract_id="matmul")
    semantic = build_semantic_region(source, atomic, validation)
    opt_input = build_graph_optimization_input(semantic, atomic)
    assert opt_input.semantic_region is semantic
    assert opt_input.atomic_dag_hash == atomic.atomic_dag_hash

