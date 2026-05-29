# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Reusable E2E scaffold for admitted active-route operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .executor import (
    EXECUTION_STATUS_EXECUTED,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_GAP,
    ExecutionContext,
    ExecutorRegistry,
    OperatorExecutionRequest,
    OperatorExecutionResult,
    PLACEHOLDER_REASON,
    PlaceholderExecutor,
    ReferenceTensorExecutor,
    TvmPackedExecutor,
    _as_numpy,
)


@dataclass(frozen=True)
class E2EOperatorPlan:
    """One operator planned for diagnostic E2E execution."""

    operator_id: str
    execution_order: int
    model_id: str = ""
    declared_semantic_region_key: str | None = None
    semantic_region_key: str | None = None
    semantic_region_hash: str | None = None
    lowering_contract_ids: tuple[str, ...] = ()
    lowering_contract_status: str = "direct_lowering"
    source_route_record_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    atomic_dag_hashes: tuple[str, ...] = ()
    atomic_dag_bundle_hash: str = ""
    capability_id: str | None = None
    admission_status: str = "admitted"
    source_id: str | None = None
    atomic_dag_hash: str | None = None
    shape_signature: Mapping[str, Any] | None = None
    dtype_signature: Mapping[str, Any] | None = None
    layout_signature: Mapping[str, Any] | None = None
    input_buffer_ids: tuple[str, ...] = ()
    output_buffer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class E2EScaffoldResult:
    """Diagnostic E2E scaffold summary."""

    status: str
    planned_operator_count: int
    placeholder_operator_count: int
    executed_operator_count: int
    gap_count: int
    performance_claim: bool
    execution_mode: str
    operators: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ReferenceTvmE2EResult:
    """Reference-vs-TVM correctness report."""

    status: str
    planned_operator_count: int
    reference_executed_count: int
    tvm_executed_count: int
    allclose_count: int
    not_run_count: int
    gap_count: int
    failed_count: int
    top_level_operator_count: int
    source_route_record_count: int
    artifact_source_breakdown: Mapping[str, int]
    lowering_contract_status_breakdown: Mapping[str, int]
    lowering_contract_warning_count: int
    semantic_proof_status_breakdown: Mapping[str, int]
    proof_source_breakdown: Mapping[str, int]
    runtime_claim: str
    e2e_pass: bool
    performance_claim: bool
    graph_optimization_used: bool
    atomic_dag_lowering_coverage_claim: bool
    model_comparison_status: str
    model_output_count: int
    model_allclose_count: int
    model_outputs: tuple[Mapping[str, Any], ...]
    execution_mode: str
    operators: tuple[Mapping[str, Any], ...]


def run_e2e_scaffold(
    plans: Iterable[E2EOperatorPlan],
    *,
    executor_registry: ExecutorRegistry | None = None,
    context: ExecutionContext | None = None,
    target: str = "cuda",
) -> E2EScaffoldResult:
    """Run admitted operators through a diagnostic scaffold.

    The default registry uses a placeholder executor and performs no real
    computation. This keeps v1 useful as an E2E shape without making a runtime
    or performance claim.
    """

    plan_list = tuple(sorted(plans, key=lambda plan: plan.execution_order))
    registry = executor_registry or ExecutorRegistry.empty().with_executor(
        PlaceholderExecutor(), semantic_region_key="*"
    )
    execution_context = context or ExecutionContext(buffer_table={}, target=target)
    operator_results = []

    for plan in plan_list:
        if plan.admission_status != "admitted":
            result = OperatorExecutionResult(
                status=EXECUTION_STATUS_GAP,
                operator_id=plan.operator_id,
                backend_id=None,
                reason=f"operator admission status is {plan.admission_status}",
            )
        else:
            result = registry.execute(_request_from_plan(plan), execution_context)
        operator_results.append(_operator_result_record(plan, result))

    executed = sum(1 for record in operator_results if record["status"] == EXECUTION_STATUS_EXECUTED)
    placeholders = sum(1 for record in operator_results if record["reason"] == PLACEHOLDER_REASON)
    gaps = sum(1 for record in operator_results if record["status"] == EXECUTION_STATUS_GAP)
    status = (
        "executed"
        if executed == len(operator_results) and operator_results
        else "planned_not_executed"
        if placeholders
        else "partial"
        if executed and gaps
        else "gap"
        if gaps
        else "empty"
    )
    return E2EScaffoldResult(
        status=status,
        planned_operator_count=len(operator_results),
        placeholder_operator_count=placeholders,
        executed_operator_count=executed,
        gap_count=gaps,
        performance_claim=False,
        execution_mode="diagnostic_e2e",
        operators=tuple(operator_results),
    )


def e2e_result_to_dict(result: E2EScaffoldResult) -> dict[str, Any]:
    """Convert an E2E result to a report-friendly dictionary."""

    return asdict(result)


def reference_tvm_e2e_result_to_dict(result: ReferenceTvmE2EResult) -> dict[str, Any]:
    """Convert a reference-vs-TVM result to a report-friendly dictionary."""

    return asdict(result)


def run_reference_tvm_e2e(
    requests: Iterable[OperatorExecutionRequest],
    *,
    reference_executor: ReferenceTensorExecutor,
    tvm_executor: TvmPackedExecutor,
    reference_context: ExecutionContext,
    tvm_context: ExecutionContext,
    source_route_record_count: int,
    semantic_proof_status_by_operator: Mapping[str, str] | None = None,
    proof_source_by_operator: Mapping[str, str] | None = None,
    tvm_operator_ids: Iterable[str] | None = None,
    final_output_buffer_ids: tuple[str, ...] = (),
    runtime_claim: str = "synthetic_runtime_channel_only",
    e2e_gate_artifact_sources: tuple[str, ...] = (
        "atomic_dag_generated_tir",
        "imported_tirx",
    ),
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
) -> ReferenceTvmE2EResult:
    """Run reference and TVM operator paths and compare declared outputs."""

    request_list = tuple(sorted(requests, key=lambda request: request.execution_order))
    tvm_scope = set(tvm_operator_ids) if tvm_operator_ids is not None else None
    semantic_statuses = dict(semantic_proof_status_by_operator or {})
    proof_sources = dict(proof_source_by_operator or {})
    operator_records: list[Mapping[str, Any]] = []

    for request in request_list:
        reference_result = reference_executor.execute(request, reference_context)
        if tvm_scope is not None and request.operator_id not in tvm_scope:
            tvm_result = OperatorExecutionResult(
                status="not_run",
                operator_id=request.operator_id,
                backend_id=None,
                reason="outside TVM artifact scope",
            )
        else:
            _seed_missing_tvm_inputs_from_reference(request, reference_context, tvm_context)
            tvm_result = tvm_executor.execute(request, tvm_context)
        comparison = _compare_request_outputs(
            request,
            reference_context,
            tvm_context,
            reference_result,
            tvm_result,
            rtol=rtol,
            atol=atol,
        )
        operator_records.append(
            {
                "model_id": request.model_id,
                "operator_id": request.operator_id,
                "execution_order": request.execution_order,
                "declared_semantic_region_key": request.declared_semantic_region_key,
                "semantic_region_key": request.semantic_region_key,
                "semantic_region_hash": request.semantic_region_hash,
                "matched_contract_id": (
                    request.lowering_contract_ids[0] if request.lowering_contract_ids else None
                ),
                "contract_validation_status": "passed",
                "semantic_proof_status": semantic_statuses.get(
                    request.operator_id, "validated"
                ),
                "proof_source": proof_sources.get(
                    request.operator_id, "atomic_dag_contract_validation"
                ),
                "support_claim_source": "validated_contract",
                "wrapper_provider_claim": False,
                "lowering_contract_ids": request.lowering_contract_ids,
                "lowering_contract_status": request.lowering_contract_status,
                "lowering_contract_warning": _lowering_contract_warning(
                    request.lowering_contract_status
                ),
                "source_route_record_ids": request.source_route_record_ids,
                "reference_status": reference_result.status,
                "tvm_status": tvm_result.status,
                "comparison_status": comparison["comparison_status"],
                "allclose": comparison["allclose"],
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": comparison["max_abs_error"],
                "max_rel_error": comparison["max_rel_error"],
                "backend_id": tvm_result.backend_id,
                "target": tvm_context.target,
                "tir_artifact_source": tvm_result.artifact_source,
                "correctness_readback": tvm_result.correctness_readback,
                "failure_reason": reference_result.reason or tvm_result.reason,
                "runtime_claim": runtime_claim,
                "e2e_pass": False,
            }
        )

    reference_executed = sum(
        1 for record in operator_records if record["reference_status"] == EXECUTION_STATUS_EXECUTED
    )
    tvm_executed = sum(
        1 for record in operator_records if record["tvm_status"] == EXECUTION_STATUS_EXECUTED
    )
    allclose_count = sum(1 for record in operator_records if record["comparison_status"] == "allclose_passed")
    not_run_count = sum(1 for record in operator_records if record["tvm_status"] == "not_run")
    gap_count = sum(
        1
        for record in operator_records
        if record["reference_status"] == EXECUTION_STATUS_GAP
        or record["tvm_status"] == EXECUTION_STATUS_GAP
    )
    failed_count = sum(
        1
        for record in operator_records
        if record["reference_status"] == EXECUTION_STATUS_FAILED
        or record["tvm_status"] == EXECUTION_STATUS_FAILED
        or record["comparison_status"] == "allclose_failed"
    )
    artifact_source_breakdown = _count_nonempty(
        record["tir_artifact_source"] for record in operator_records
    )
    lowering_status_breakdown = _count_nonempty(
        record["lowering_contract_status"] for record in operator_records
    )
    semantic_status_breakdown = _count_nonempty(
        record["semantic_proof_status"] for record in operator_records
    )
    proof_source_breakdown = _count_nonempty(record["proof_source"] for record in operator_records)
    model_outputs = _compare_model_outputs(
        final_output_buffer_ids,
        reference_context,
        tvm_context,
        rtol=rtol,
        atol=atol,
    )
    model_allclose_count = sum(
        1 for record in model_outputs if record["comparison_status"] == "allclose_passed"
    )
    model_comparison_status = (
        "not_requested"
        if not model_outputs
        else "allclose"
        if model_allclose_count == len(model_outputs)
        else "failed"
    )
    allowed_artifact_sources = set(e2e_gate_artifact_sources)
    operator_sources = tuple(record["tir_artifact_source"] for record in operator_records)
    e2e_pass = (
        bool(operator_records)
        and allclose_count == len(operator_records)
        and bool(model_outputs)
        and model_allclose_count == len(model_outputs)
        and all(source in allowed_artifact_sources for source in operator_sources)
    )
    status = (
        "allclose"
        if allclose_count == len(operator_records) and operator_records
        else "partial"
        if reference_executed or tvm_executed
        else "gap"
        if gap_count
        else "empty"
    )
    return ReferenceTvmE2EResult(
        status=status,
        planned_operator_count=len(operator_records),
        reference_executed_count=reference_executed,
        tvm_executed_count=tvm_executed,
        allclose_count=allclose_count,
        not_run_count=not_run_count,
        gap_count=gap_count,
        failed_count=failed_count,
        top_level_operator_count=len(operator_records),
        source_route_record_count=source_route_record_count,
        artifact_source_breakdown=artifact_source_breakdown,
        lowering_contract_status_breakdown=lowering_status_breakdown,
        lowering_contract_warning_count=sum(
            1
            for record in operator_records
            if record["lowering_contract_status"]
            in {"temporary_surrogate", "flattened_surrogate"}
        ),
        semantic_proof_status_breakdown=semantic_status_breakdown,
        proof_source_breakdown=proof_source_breakdown,
        runtime_claim=runtime_claim,
        e2e_pass=e2e_pass,
        performance_claim=False,
        graph_optimization_used=False,
        atomic_dag_lowering_coverage_claim=False,
        model_comparison_status=model_comparison_status,
        model_output_count=len(model_outputs),
        model_allclose_count=model_allclose_count,
        model_outputs=tuple(model_outputs),
        execution_mode="reference_vs_tvm_op_by_op",
        operators=tuple(operator_records),
    )


def _request_from_plan(plan: E2EOperatorPlan) -> OperatorExecutionRequest:
    return OperatorExecutionRequest(
        model_id=plan.model_id,
        operator_id=plan.operator_id,
        execution_order=plan.execution_order,
        declared_semantic_region_key=plan.declared_semantic_region_key,
        semantic_region_key=plan.semantic_region_key,
        semantic_region_hash=plan.semantic_region_hash,
        lowering_contract_ids=plan.lowering_contract_ids,
        lowering_contract_status=plan.lowering_contract_status,
        source_route_record_ids=plan.source_route_record_ids,
        source_ids=plan.source_ids,
        atomic_dag_hashes=plan.atomic_dag_hashes,
        atomic_dag_bundle_hash=plan.atomic_dag_bundle_hash,
        capability_id=plan.capability_id,
        source_id=plan.source_id or "",
        atomic_dag_hash=plan.atomic_dag_hash or "",
        shape_signature=dict(plan.shape_signature or {}),
        dtype_signature=dict(plan.dtype_signature or {}),
        layout_signature=dict(plan.layout_signature or {}),
        input_buffer_ids=plan.input_buffer_ids,
        output_buffer_ids=plan.output_buffer_ids,
    )


def _operator_result_record(
    plan: E2EOperatorPlan, result: OperatorExecutionResult
) -> dict[str, Any]:
    return {
        "operator_id": plan.operator_id,
        "execution_order": plan.execution_order,
        "semantic_region_key": plan.semantic_region_key,
        "capability_id": plan.capability_id,
        "admission_status": plan.admission_status,
        "model_id": plan.model_id,
        "declared_semantic_region_key": plan.declared_semantic_region_key,
        "semantic_region_hash": plan.semantic_region_hash,
        "lowering_contract_ids": plan.lowering_contract_ids,
        "lowering_contract_status": plan.lowering_contract_status,
        "source_route_record_ids": plan.source_route_record_ids,
        "source_ids": plan.source_ids,
        "atomic_dag_hashes": plan.atomic_dag_hashes,
        "atomic_dag_bundle_hash": plan.atomic_dag_bundle_hash,
        "status": result.status,
        "backend_id": result.backend_id,
        "reason": result.reason,
    }


def _compare_request_outputs(
    request: OperatorExecutionRequest,
    reference_context: ExecutionContext,
    tvm_context: ExecutionContext,
    reference_result: OperatorExecutionResult,
    tvm_result: OperatorExecutionResult,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if not reference_result.executed or not tvm_result.executed:
        return {
            "comparison_status": "not_compared",
            "allclose": False,
            "max_abs_error": None,
            "max_rel_error": None,
        }
    max_abs = 0.0
    max_rel = 0.0
    allclose = True
    for output_buffer_id in request.output_buffer_ids:
        reference = _as_numpy(_read_context(reference_context, output_buffer_id))
        tvm = _as_numpy(_read_context(tvm_context, output_buffer_id))
        diff = abs(reference - tvm)
        max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
        denom = abs(reference)
        rel = diff / (denom + 1.0e-12)
        max_rel = max(max_rel, float(rel.max()) if rel.size else 0.0)
        allclose = bool(allclose and __import__("numpy").allclose(reference, tvm, rtol=rtol, atol=atol))
    return {
        "comparison_status": "allclose_passed" if allclose else "allclose_failed",
        "allclose": allclose,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


def _compare_model_outputs(
    final_output_buffer_ids: tuple[str, ...],
    reference_context: ExecutionContext,
    tvm_context: ExecutionContext,
    *,
    rtol: float,
    atol: float,
) -> list[Mapping[str, Any]]:
    records = []
    for buffer_id in final_output_buffer_ids:
        if not _has_context(reference_context, buffer_id) or not _has_context(tvm_context, buffer_id):
            records.append(
                {
                    "buffer_id": buffer_id,
                    "comparison_status": "not_compared",
                    "allclose": False,
                    "max_abs_error": None,
                    "max_rel_error": None,
                    "reference_source": "reference_tensor_executor",
                    "tvm_source": "tvm_packed_executor",
                    "correctness_readback": False,
                }
            )
            continue
        reference = _as_numpy(_read_context(reference_context, buffer_id))
        tvm = _as_numpy(_read_context(tvm_context, buffer_id))
        diff = abs(reference - tvm)
        max_abs = float(diff.max()) if diff.size else 0.0
        rel = diff / (abs(reference) + 1.0e-12)
        max_rel = float(rel.max()) if rel.size else 0.0
        allclose = bool(__import__("numpy").allclose(reference, tvm, rtol=rtol, atol=atol))
        records.append(
            {
                "buffer_id": buffer_id,
                "comparison_status": "allclose_passed" if allclose else "allclose_failed",
                "allclose": allclose,
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "reference_source": "reference_tensor_executor",
                "tvm_source": "tvm_packed_executor",
                "correctness_readback": True,
            }
        )
    return records


def _read_context(context: ExecutionContext, buffer_id: str) -> Any:
    if hasattr(context.buffer_table, "get") and not isinstance(context.buffer_table, dict):
        return context.buffer_table.get(buffer_id)
    return context.buffer_table[buffer_id]


def _write_context(context: ExecutionContext, buffer_id: str, value: Any) -> None:
    if hasattr(context.buffer_table, "set") and not isinstance(context.buffer_table, dict):
        context.buffer_table.set(buffer_id, value)
    else:
        context.buffer_table[buffer_id] = value


def _has_context(context: ExecutionContext, buffer_id: str) -> bool:
    if hasattr(context.buffer_table, "has") and not isinstance(context.buffer_table, dict):
        return context.buffer_table.has(buffer_id)
    return buffer_id in context.buffer_table


def _seed_missing_tvm_inputs_from_reference(
    request: OperatorExecutionRequest,
    reference_context: ExecutionContext,
    tvm_context: ExecutionContext,
) -> None:
    for buffer_id in request.input_buffer_ids:
        if not _has_context(tvm_context, buffer_id):
            _write_context(tvm_context, buffer_id, _as_numpy(_read_context(reference_context, buffer_id)).copy())


def _count_nonempty(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _lowering_contract_warning(status: str | None) -> str | None:
    if status == "temporary_surrogate":
        return "temporary lowering surrogate; not a full top-level semantic support claim"
    if status == "flattened_surrogate":
        return "flattened lowering surrogate; grid/indexing/mask/broadcast metadata is preserved in the report"
    return None
