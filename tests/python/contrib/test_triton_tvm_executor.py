from dataclasses import dataclass, replace

import numpy as np
import pytest

from tvm.contrib.triton_tvm.models.vit import (
    build_vit_synthetic_execution_context,
    build_vit_synthetic_tir_artifacts,
    run_vit_tiny_fixed_shape_route,
    vit_execution_requests_from_report,
)
from tvm.contrib.triton_tvm.runtime import (
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_GAP,
    ExecutionContext,
    ExecutorRegistry,
    OperatorExecutionRequest,
    OperatorExecutionResult,
    ReferenceTensorExecutor,
    TIRArtifactBuildCache,
    TIRArtifactRegistry,
    TvmPackedExecutor,
)


@dataclass
class _NamedExecutor:
    backend_id: str
    status: str = "executed"
    reason: str | None = None

    def can_execute(self, request):
        return True

    def execute(self, request, context):
        return OperatorExecutionResult(
            status=self.status,
            operator_id=request.operator_id,
            backend_id=self.backend_id,
            reason=self.reason,
        )


def test_executor_registry_prefers_capability_match_over_semantic_match():
    request = _request(
        semantic_region_key="pointwise_flat",
        capability_id="cap.exact",
    )
    registry = (
        ExecutorRegistry.empty()
        .with_executor(_NamedExecutor("semantic"), semantic_region_key="pointwise_flat")
        .with_executor(_NamedExecutor("capability"), capability_id="cap.exact")
    )

    result = registry.execute(request, ExecutionContext(buffer_table={}, target="cuda"))

    assert result.backend_id == "capability"


def test_executor_registry_returns_explicit_gap_when_no_executor_matches():
    result = ExecutorRegistry.empty().execute(
        _request(semantic_region_key="pointwise_flat"),
        ExecutionContext(buffer_table={}, target="cuda"),
    )

    assert result.status == EXECUTION_STATUS_GAP
    assert result.reason == "no executor registered"


def test_failed_executor_does_not_mark_operator_executed():
    registry = ExecutorRegistry.empty().with_executor(
        _NamedExecutor("failing", status=EXECUTION_STATUS_FAILED, reason="boom"),
        semantic_region_key="pointwise_flat",
    )
    result = registry.execute(
        _request(semantic_region_key="pointwise_flat"),
        ExecutionContext(buffer_table={}, target="cuda"),
    )

    assert result.status == EXECUTION_STATUS_FAILED
    assert not result.executed


def test_reference_tensor_executor_pointwise_copy():
    context = ExecutionContext(
        buffer_table={"x": np.array([1.0, 2.0], dtype=np.float32)},
        target="cuda",
    )
    result = ReferenceTensorExecutor().execute(
        _request(
            operator_id="copy",
            semantic_region_key="pointwise_flat",
            input_buffer_ids=("x",),
            output_buffer_ids=("y",),
        ),
        context,
    )

    assert result.executed
    np.testing.assert_allclose(context.buffer_table["y"], np.array([1.0, 2.0], dtype=np.float32))


def test_reference_tensor_executor_matmul_with_bias():
    context = ExecutionContext(
        buffer_table={
            "a": np.array([[1.0, 2.0]], dtype=np.float32),
            "b": np.array([[3.0], [4.0]], dtype=np.float32),
            "bias": np.array([[5.0]], dtype=np.float32),
        },
        target="cuda",
    )
    result = ReferenceTensorExecutor().execute(
        _request(
            operator_id="matmul_bias",
            semantic_region_key="matmul_bias_epilogue",
            input_buffer_ids=("a", "b", "bias"),
            output_buffer_ids=("out",),
        ),
        context,
    )

    assert result.executed
    np.testing.assert_allclose(context.buffer_table["out"], np.array([[16.0]], dtype=np.float32))


def test_tvm_packed_executor_rejects_python_callable_fallback():
    with pytest.raises(ValueError, match="fallback is disabled"):
        TvmPackedExecutor({"cap.copy": lambda x: x + 1.0})


def test_reference_executor_requires_operator_spec():
    result = ReferenceTensorExecutor(include_legacy_defaults=False).execute(
        _request(
            operator_id="missing",
            semantic_region_key="pointwise_flat",
            input_buffer_ids=("x",),
            output_buffer_ids=("y",),
        ),
        ExecutionContext(buffer_table={"x": np.array([1.0], dtype=np.float32)}, target="llvm"),
    )

    assert result.status == EXECUTION_STATUS_GAP
    assert result.reason == "missing ReferenceOperatorSpec"


def test_tvm_packed_executor_uses_model_operator_artifact_lookup_and_build_cache():
    request = _vit_request("patch_embed")
    artifacts = build_vit_synthetic_tir_artifacts((request,), target="llvm")
    cache = TIRArtifactBuildCache()
    executor = TvmPackedExecutor(
        artifact_registry=TIRArtifactRegistry(artifacts),
        build_cache=cache,
    )
    context = build_vit_synthetic_execution_context(target="llvm")

    first = executor.execute(
        replace(request, capability_id="not.used.for.lookup"),
        context,
    )
    second = executor.execute(
        replace(request, capability_id="another.unused.capability"),
        context,
    )

    assert first.executed
    assert second.executed
    assert cache.build_count == 1
    assert cache.hit_count == 1
    assert "patch_embed.out" in context.buffer_table


def test_tvm_packed_executor_rejects_semantic_region_mismatch():
    request = _vit_request("patch_embed")
    artifact = build_vit_synthetic_tir_artifacts((request,), target="llvm")[0]
    executor = TvmPackedExecutor(artifact_registry=TIRArtifactRegistry((artifact,)))

    result = executor.execute(
        replace(request, semantic_region_key="pointwise_flat"),
        build_vit_synthetic_execution_context(target="llvm"),
    )

    assert result.status == EXECUTION_STATUS_FAILED
    assert "Semantic Region key" in result.reason


def test_tvm_packed_executor_rejects_target_mismatch():
    request = _vit_request("patch_embed")
    artifact = build_vit_synthetic_tir_artifacts((request,), target="llvm")[0]
    executor = TvmPackedExecutor(artifact_registry=TIRArtifactRegistry((artifact,)))

    result = executor.execute(
        request,
        build_vit_synthetic_execution_context(target="cuda"),
    )

    assert result.status == EXECUTION_STATUS_FAILED
    assert "target" in result.reason


def test_tir_region_artifact_validator_rejects_missing_entry():
    request = _vit_request("patch_embed")
    artifact = build_vit_synthetic_tir_artifacts((request,), target="llvm")[0]
    bad_step = replace(artifact.lowering_plan[0], primfunc_name="missing_entry")
    bad_artifact = replace(artifact, lowering_plan=(bad_step,))
    executor = TvmPackedExecutor(artifact_registry=TIRArtifactRegistry((bad_artifact,)))

    result = executor.execute(
        request,
        build_vit_synthetic_execution_context(target="llvm"),
    )

    assert result.status == EXECUTION_STATUS_FAILED
    assert "missing entry PrimFunc" in result.reason


def _request(
    *,
    operator_id: str = "op0",
    semantic_region_key: str | None = None,
    capability_id: str | None = None,
    input_buffer_ids: tuple[str, ...] = (),
    output_buffer_ids: tuple[str, ...] = (),
):
    return OperatorExecutionRequest(
        operator_id=operator_id,
        execution_order=0,
        semantic_region_key=semantic_region_key,
        capability_id=capability_id,
        source_id="source",
        atomic_dag_hash="atomic",
        shape_signature={},
        dtype_signature={},
        layout_signature={},
        input_buffer_ids=input_buffer_ids,
        output_buffer_ids=output_buffer_ids,
    )


def _vit_request(operator_id):
    report = run_vit_tiny_fixed_shape_route(target="llvm")
    return next(
        request
        for request in vit_execution_requests_from_report(report)
        if request.operator_id == operator_id
    )
