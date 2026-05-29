import numpy as np

from tvm.contrib.triton_tvm.runtime import (
    E2EOperatorPlan,
    EXECUTION_STATUS_EXECUTED,
    ExecutionContext,
    ExecutorRegistry,
    OperatorExecutionRequest,
    OperatorExecutionResult,
    PLACEHOLDER_REASON,
    ReferenceTensorExecutor,
    run_reference_tvm_e2e,
    run_e2e_scaffold,
)


def test_e2e_scaffold_empty_plan_has_no_performance_claim():
    result = run_e2e_scaffold([])

    assert result.status == "empty"
    assert result.planned_operator_count == 0
    assert result.placeholder_operator_count == 0
    assert result.executed_operator_count == 0
    assert result.performance_claim is False


def test_e2e_scaffold_defaults_admitted_operators_to_placeholders():
    result = run_e2e_scaffold(
        [
            E2EOperatorPlan(
                operator_id="op0",
                execution_order=0,
                semantic_region_key="pointwise_flat",
                capability_id="cap.pointwise",
                admission_status="admitted",
            )
        ]
    )

    assert result.status == "planned_not_executed"
    assert result.planned_operator_count == 1
    assert result.placeholder_operator_count == 1
    assert result.executed_operator_count == 0
    assert result.operators[0]["reason"] == PLACEHOLDER_REASON


def test_e2e_scaffold_does_not_execute_non_admitted_operators():
    result = run_e2e_scaffold(
        [
            E2EOperatorPlan(
                operator_id="op0",
                execution_order=0,
                semantic_region_key="pointwise_flat",
                capability_id="cap.pointwise",
                admission_status="gap",
            )
        ],
        executor_registry=ExecutorRegistry.empty().with_executor(
            ReferenceTensorExecutor(),
            semantic_region_key="pointwise_flat",
        ),
    )

    assert result.status == "gap"
    assert result.executed_operator_count == 0
    assert result.gap_count == 1


def test_e2e_scaffold_executes_reference_operator_when_buffers_are_supplied():
    context = ExecutionContext(
        buffer_table={"x": np.array([1.0, 2.0], dtype=np.float32)},
        target="cuda",
    )
    result = run_e2e_scaffold(
        [
            E2EOperatorPlan(
                operator_id="copy",
                execution_order=0,
                semantic_region_key="pointwise_flat",
                capability_id="cap.pointwise",
                admission_status="admitted",
                input_buffer_ids=("x",),
                output_buffer_ids=("y",),
            )
        ],
        executor_registry=ExecutorRegistry.empty().with_executor(
            ReferenceTensorExecutor(),
            semantic_region_key="pointwise_flat",
        ),
        context=context,
    )

    assert result.status == "executed"
    assert result.executed_operator_count == 1
    np.testing.assert_allclose(context.buffer_table["y"], np.array([1.0, 2.0], dtype=np.float32))


def test_reference_tvm_report_distinguishes_imported_tirx_artifact_source():
    request = OperatorExecutionRequest(
        model_id="vit",
        operator_id="copy",
        execution_order=0,
        semantic_region_key="pointwise_flat",
        lowering_contract_ids=("pointwise_flat",),
        source_route_record_ids=("copy",),
        source_ids=("source",),
        atomic_dag_hashes=("atomic",),
        atomic_dag_bundle_hash="bundle",
        input_buffer_ids=("x",),
        output_buffer_ids=("y",),
    )
    reference_context = ExecutionContext(
        buffer_table={"x": np.array([1.0, 2.0], dtype=np.float32)},
        target="llvm",
    )
    tvm_context = ExecutionContext(
        buffer_table={"x": np.array([1.0, 2.0], dtype=np.float32)},
        target="llvm",
    )

    class ImportedTirxExecutor:
        backend_id = "tvm_packed"

        def execute(self, request, context):
            context.buffer_table["y"] = context.buffer_table["x"].copy()
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_EXECUTED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason=None,
                output_buffer_ids=request.output_buffer_ids,
                artifact_source="imported_tirx",
                correctness_readback=True,
            )

    result = run_reference_tvm_e2e(
        (request,),
        reference_executor=ReferenceTensorExecutor(),
        tvm_executor=ImportedTirxExecutor(),
        reference_context=reference_context,
        tvm_context=tvm_context,
        source_route_record_count=1,
    )

    assert result.artifact_source_breakdown == {"imported_tirx": 1}
    assert result.allclose_count == 1
    assert result.e2e_pass is False
