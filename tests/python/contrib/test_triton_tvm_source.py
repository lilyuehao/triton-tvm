import pytest

from tvm.contrib.triton_tvm.reports import build_alpha_smoke_report
from tvm.contrib.triton_tvm.source import (
    SourceArtifactKind,
    SourceOrigin,
    SourceRecord,
    captured_inductor_jit_kernel,
    native_triton_source,
    native_triton_ttir,
)


TTIR = """
module {
  tt.func public @pointwise(%x: !tt.ptr<f32>) {
    %0 = tt.load %x : !tt.ptr<f32> -> tensor<16xf32>
    tt.store %x, %0 : !tt.ptr<f32>, tensor<16xf32>
  }
}
"""


def test_source_origin_taxonomy_and_artifact_kinds():
    assert [origin.value for origin in SourceOrigin] == [
        "torch_inductor",
        "native_triton_language",
        "test_fixture",
    ]
    assert SourceOrigin("torch_inductor") is SourceOrigin.TORCH_INDUCTOR
    with pytest.raises(ValueError):
        SourceOrigin("project_generated")
    assert {kind.value for kind in SourceArtifactKind} == {
        "triton_python_source",
        "ttir_module",
        "captured_jit_kernel",
        "exported_inductor_kernel",
        "normalized_ttir_graph",
    }


def test_source_record_is_identity_only():
    record = captured_inductor_jit_kernel(ttir=TTIR, function_name="pointwise", target="cuda")
    assert record.source_origin is SourceOrigin.TORCH_INDUCTOR
    assert record.artifact_kind is SourceArtifactKind.CAPTURED_JIT_KERNEL
    for forbidden in ("schedule_id", "runtime_provider", "model_family", "capability_credit"):
        assert forbidden not in record.__dataclass_fields__


def test_normalized_ttir_graph_requires_hash_producer_and_parent_provenance():
    kwargs = dict(
        source_id="normalized:missing",
        source_origin=SourceOrigin.TORCH_INDUCTOR,
        artifact_kind=SourceArtifactKind.NORMALIZED_TTIR_GRAPH,
        source_hash=None,
        ttir_hash="abc",
        function_name="pointwise",
        target="cuda",
    )
    with pytest.raises(ValueError):
        SourceRecord(**kwargs)
    record = SourceRecord(
        **kwargs,
        producer_name="normalizer",
        producer_version="1",
        parent_source_id="parent",
    )
    assert record.producer_name == "normalizer"


def test_native_triton_language_requires_auditable_provenance_for_ttir():
    source = native_triton_source(
        source="@triton.jit\ndef pointwise(x):\n    return x\n",
        function_name="pointwise",
        target="cuda",
    )
    ttir = native_triton_ttir(
        ttir=TTIR,
        function_name="pointwise",
        target="cuda",
        parent_source_id=source.source_id,
    )
    assert ttir.parent_source_id == source.source_id
    with pytest.raises(ValueError):
        native_triton_ttir(ttir=TTIR, function_name="pointwise", target="cuda", parent_source_id=None)


def test_alpha_report_rejects_test_fixture_as_production_support():
    source = SourceRecord(
        source_id="fixture:pointwise",
        source_origin=SourceOrigin.TEST_FIXTURE,
        artifact_kind=SourceArtifactKind.TTIR_MODULE,
        source_hash=None,
        ttir_hash="abc",
        function_name="pointwise",
        target="cuda",
    )
    report = build_alpha_smoke_report(sources=[source], atomics=[], semantics=[], manifests=[])
    assert report["status"] == "non_production_input_rejected"
    assert report["performance_claim"] is False

