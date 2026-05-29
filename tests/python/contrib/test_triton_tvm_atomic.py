import dataclasses

import pytest

from tvm.contrib.triton_tvm.atomic import (
    AtomicDAGRecord,
    build_atomic_dag,
    validate_atomic_dag_consistency,
    validate_source_atomic_join,
)
from tvm.contrib.triton_tvm.source import captured_inductor_jit_kernel


TTIR = """
module {
  tt.func public @matmul(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %0 = tt.load %a : !tt.ptr<f32> -> tensor<16x16xf32>
    %1 = tt.load %b : !tt.ptr<f32> -> tensor<16x16xf32>
    %2 = tt.dot %0, %1 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    tt.store %c, %2 : !tt.ptr<f32>, tensor<16x16xf32>
  }
}
"""


def _source():
    return captured_inductor_jit_kernel(ttir=TTIR, function_name="matmul", target="cuda")


def test_atomic_dag_hash_is_stable_for_identical_ttir():
    source = _source()
    first = build_atomic_dag(source, TTIR)
    second = build_atomic_dag(source, TTIR)
    assert first.atomic_dag_hash == second.atomic_dag_hash
    assert first.atomic_family_histogram["dot"] == 1


def test_atomic_producer_consumer_consistency_and_source_join():
    source = _source()
    atomic = build_atomic_dag(source, TTIR)
    assert atomic.producer_consumer_consistent is True
    assert validate_atomic_dag_consistency(atomic.atomic_ops) is True
    validate_source_atomic_join(source, atomic)


def test_atomic_join_mismatch_is_hard_failure():
    source = _source()
    atomic = build_atomic_dag(source, TTIR)
    changed = dataclasses.replace(atomic, source_hash="different")
    with pytest.raises(ValueError):
        validate_source_atomic_join(source, changed)


def test_unsupported_ttir_ops_are_explicit_atomic_records():
    ttir = TTIR.replace("tt.dot", "tt.future_op")
    source = captured_inductor_jit_kernel(ttir=ttir, function_name="matmul", target="cuda")
    atomic = build_atomic_dag(source, ttir)
    assert atomic.unsupported_atomic_ops
    assert atomic.unsupported_atomic_ops[0].atomic_family == "unsupported_atomic_op"


def test_atomic_dag_record_does_not_own_semantic_identity():
    fields = set(AtomicDAGRecord.__dataclass_fields__)
    assert "semantic_region" not in fields
    assert "semantic_region_hash" not in fields
    assert "runtime_backend" not in fields

