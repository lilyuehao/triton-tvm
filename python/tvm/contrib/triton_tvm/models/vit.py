# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Fixed-shape ViT runner for the active Triton-TVM alpha route."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..atomic import (
    ATOMIC_FAMILY_DOT,
    ATOMIC_FAMILY_MEMORY,
    build_atomic_dag,
    validate_source_atomic_join,
)
from ..contracts import ContractValidationResult, validate_atomic_contract
from ..manifest import build_manifest_gap, build_operator_manifest_record
from ..registry import CapabilityRecord, CapabilityRegistry
from ..reports import build_alpha_smoke_report, write_json_report
from ..runtime import (
    E2EOperatorPlan,
    ExecutionContext,
    ExecutorRegistry,
    LoweringStep,
    OperatorExecutionRequest,
    ReferenceOperatorSpec,
    ReferenceTensorExecutor,
    RuntimeAdmissionResult,
    RuntimeBuffer,
    RuntimeBufferEdge,
    RuntimeBufferPlan,
    TIRArtifactRegistry,
    TIRRegionArtifact,
    TvmPackedExecutor,
    admit_runtime,
    e2e_result_to_dict,
    reference_tvm_e2e_result_to_dict,
    run_e2e_scaffold,
    run_reference_tvm_e2e,
    validate_runtime_buffer_plan,
)
from ..semantic import (
    LOWERING_CONTRACT_STATUS_COMPOSITE,
    LOWERING_CONTRACT_STATUS_DIRECT,
    LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE,
    LOWERING_CONTRACT_STATUS_SPECIALIZED,
    LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE,
    build_semantic_region,
)
from ..semantic.schema import SemanticRegionRecord
from ..source import SourceOrigin, captured_inductor_jit_kernel
from ..source.identity import stable_hash
from .adapters import ModelAdapter, ModelOperatorInput


VIT_TINY_FIXED_SHAPE_MODEL_ID = "vit_tiny_fixed_shape"
VIT_RUNNER_ID = "vit_fixed_shape_active_route_v2"
VIT_SOURCE_ROUTE_RECORD_COUNT = 16
VIT_TOP_LEVEL_OPERATOR_COUNT = 14
VIT_OPERATOR_COUNT = VIT_TOP_LEVEL_OPERATOR_COUNT
RUNTIME_CLAIM_SYNTHETIC_ONLY = "synthetic_runtime_channel_only"
RUNTIME_CLAIM_ATOMIC_DAG_GATED_TE = "atomic_dag_gated_te_correctness_only"
TIR_ARTIFACT_SOURCE_SYNTHETIC = "synthetic_tir"
TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE = "atomic_dag_gated_te_tir"
TIR_ARTIFACT_SOURCE_IMPORTED_TIRX = "imported_tirx"
E2E_GATE_ARTIFACT_SOURCES = (
    TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE,
    TIR_ARTIFACT_SOURCE_IMPORTED_TIRX,
)

_TOKENS = 4
_HIDDEN = 4
_MLP = 8
_CHANNELS = 3
_IMAGE = 4
_PATCH = 2
_PATCH_GRID = _IMAGE // _PATCH


@dataclass(frozen=True)
class VitOperatorSpec:
    """One fixed-shape ViT source/lowering route input."""

    operator_id: str
    top_level_operator_id: str
    top_level_execution_order: int
    semantic_region_key: str
    lowering_contract_id: str
    lowering_contract_status: str
    function_name: str
    ttir: str
    shape_signature: Mapping[str, Any]
    dtype_signature: Mapping[str, Any]
    layout_signature: Mapping[str, Any]

    @property
    def contract_id(self) -> str:
        """Backward-compatible alias for the lowering contract."""

        return self.lowering_contract_id


@dataclass(frozen=True)
class VitTopLevelOperatorSpec:
    """Canonical top-level ViT semantic operator."""

    operator_id: str
    execution_order: int
    semantic_region_key: str
    lowering_contract_ids: tuple[str, ...]
    lowering_contract_status: str
    source_route_record_ids: tuple[str, ...]
    input_buffer_ids: tuple[str, ...]
    output_buffer_ids: tuple[str, ...]
    shape_signature: Mapping[str, Any]
    dtype_signature: Mapping[str, Any]
    layout_signature: Mapping[str, Any]
    reference_impl_id: str
    reference_attrs: Mapping[str, Any]


@dataclass(frozen=True)
class VitSourceRouteRecord:
    """Per-source-route result from source materialization through validation."""

    operator_id: str
    top_level_operator_id: str
    execution_order: int
    top_level_execution_order: int
    source_id: str
    atomic_dag_hash: str
    declared_semantic_region_key: str
    semantic_region_key: str | None
    lowering_contract_id: str
    lowering_contract_status: str
    matched_contract_id: str | None
    validation_status: str
    semantic_region_hash: str | None
    semantic_proof_status: str | None
    proof_source: str | None
    support_claim_source: str | None
    wrapper_provider_claim: bool
    gap_reason: str | None


@dataclass(frozen=True)
class VitTopLevelOperatorRecord:
    """Canonical top-level operator report and execution-request source."""

    model_id: str
    operator_id: str
    execution_order: int
    declared_semantic_region_key: str
    semantic_region_key: str
    semantic_region_hash: str | None
    matched_contract_id: str | None
    contract_validation_status: str
    semantic_proof_status: str
    proof_source: str
    support_claim_source: str
    wrapper_provider_claim: bool
    lowering_contract_ids: tuple[str, ...]
    lowering_contract_status: str
    lowering_contract_warning: str | None
    source_route_record_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    atomic_dag_hashes: tuple[str, ...]
    atomic_dag_bundle_hash: str
    shape_signature: Mapping[str, Any]
    dtype_signature: Mapping[str, Any]
    layout_signature: Mapping[str, Any]
    input_buffer_ids: tuple[str, ...]
    output_buffer_ids: tuple[str, ...]
    registry_supported: bool
    capability_id: str | None
    runtime_admission_status: str
    gap_reason: str | None


def build_vit_tiny_fixed_shape_adapter(
    *, target: str = "cuda", producer_version: str | None = None
) -> ModelAdapter:
    """Materialize 16 source/lowering route records without support claims."""

    operators = []
    for index, spec in enumerate(vit_tiny_fixed_shape_specs()):
        source = captured_inductor_jit_kernel(
            ttir=spec.ttir,
            function_name=spec.function_name,
            target=target,
            producer_version=producer_version,
        )
        operators.append(
            ModelOperatorInput(
                model_id=VIT_TINY_FIXED_SHAPE_MODEL_ID,
                operator_id=spec.operator_id,
                execution_order=index,
                source=source,
                ttir=spec.ttir,
                optional_model_anchor=VIT_RUNNER_ID,
            )
        )
    return ModelAdapter(VIT_TINY_FIXED_SHAPE_MODEL_ID, tuple(operators))


def run_vit_tiny_fixed_shape_route(
    *,
    target: str = "cuda",
    producer_version: str | None = None,
    out_dir: str | Path | None = None,
    enable_e2e_scaffold: bool = False,
    executor_registry: ExecutorRegistry | None = None,
    enable_synthetic_tir: bool = False,
    synthetic_tir_operator_ids: Iterable[str] | None = None,
    enable_atomic_dag_gated_te_tir: bool = False,
    atomic_dag_gated_te_operator_ids: Iterable[str] | None = None,
    tir_artifact_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compile and run the fixed-shape ViT inventory through the active route."""

    adapter = build_vit_tiny_fixed_shape_adapter(
        target=target,
        producer_version=producer_version,
    )
    source_specs = {spec.operator_id: spec for spec in vit_tiny_fixed_shape_specs()}
    top_specs = {spec.operator_id: spec for spec in vit_tiny_fixed_shape_top_level_specs()}
    registry = default_vit_capability_registry()

    sources_by_route: dict[str, Any] = {}
    atomics_by_route: dict[str, Any] = {}
    validations_by_route: dict[str, ContractValidationResult] = {}
    semantics_by_route: dict[str, SemanticRegionRecord] = {}
    source_route_records: list[VitSourceRouteRecord] = []

    for item in adapter.operators():
        spec = source_specs[item.operator_id]
        source = item.source
        atomic = build_atomic_dag(source, item.ttir)
        validate_source_atomic_join(source, atomic)
        validation = validate_atomic_contract(
            source,
            atomic,
            contract_id=spec.lowering_contract_id,
            shape_constraints=spec.shape_signature,
            dtype_constraints=spec.dtype_signature,
            layout_constraints=spec.layout_signature,
        )

        semantic = None
        gap_reason = validation.failure_reason
        if validation.passed:
            semantic = build_semantic_region(
                source,
                atomic,
                validation,
                semantic_region_key=spec.semantic_region_key,
                declared_semantic_region_key=spec.semantic_region_key,
                lowering_contract_ids=(spec.lowering_contract_id,),
                lowering_contract_status=spec.lowering_contract_status,
            )
            semantics_by_route[spec.operator_id] = semantic
            gap_reason = None

        sources_by_route[spec.operator_id] = source
        atomics_by_route[spec.operator_id] = atomic
        validations_by_route[spec.operator_id] = validation
        source_route_records.append(
            VitSourceRouteRecord(
                operator_id=spec.operator_id,
                top_level_operator_id=spec.top_level_operator_id,
                execution_order=item.execution_order,
                top_level_execution_order=spec.top_level_execution_order,
                source_id=source.source_id,
                atomic_dag_hash=atomic.atomic_dag_hash,
                declared_semantic_region_key=spec.semantic_region_key,
                semantic_region_key=semantic.semantic_region_key if semantic is not None else None,
                lowering_contract_id=spec.lowering_contract_id,
                lowering_contract_status=spec.lowering_contract_status,
                matched_contract_id=validation.matched_contract_id,
                validation_status=validation.validation_status,
                semantic_region_hash=(
                    semantic.semantic_region_hash if semantic is not None else None
                ),
                semantic_proof_status=(
                    semantic.semantic_proof_status if semantic is not None else None
                ),
                proof_source=semantic.proof_source if semantic is not None else None,
                support_claim_source=(
                    semantic.support_claim_source if semantic is not None else None
                ),
                wrapper_provider_claim=(
                    semantic.wrapper_provider_claim if semantic is not None else False
                ),
                gap_reason=gap_reason,
            )
        )

    top_records, top_semantics, manifests, gaps = _top_level_records_from_source_routes(
        top_specs=top_specs,
        sources_by_route=sources_by_route,
        atomics_by_route=atomics_by_route,
        validations_by_route=validations_by_route,
        semantics_by_route=semantics_by_route,
        registry=registry,
    )
    execution_requests = vit_execution_requests_from_top_records(top_records)
    buffer_plan = build_vit_synthetic_buffer_plan(target=target)
    buffer_plan_validation = validate_runtime_buffer_plan(buffer_plan, execution_requests)

    alpha_report = build_alpha_smoke_report(
        sources=sources_by_route.values(),
        atomics=atomics_by_route.values(),
        semantics=top_semantics,
        manifests=manifests,
        gaps=gaps,
    )
    admitted_count = sum(
        1 for record in top_records if record.runtime_admission_status == "admitted"
    )
    report = {
        "report_kind": "triton_tvm_vit_fixed_shape_runner",
        "runner_id": VIT_RUNNER_ID,
        "model_id": VIT_TINY_FIXED_SHAPE_MODEL_ID,
        "target": target,
        "fixed_shape": True,
        "compile_route": [
            "source_record",
            "atomic_dag",
            "contract_validation",
            "semantic_region",
            "manifest",
            "capability_registry",
            "runtime_admission",
        ],
        "operator_count": len(top_records),
        "expected_operator_count": VIT_TOP_LEVEL_OPERATOR_COUNT,
        "top_level_operator_count": len(top_records),
        "source_route_record_count": len(source_route_records),
        "expected_source_route_record_count": VIT_SOURCE_ROUTE_RECORD_COUNT,
        "atomic_dag_built_operator_count": len(atomics_by_route),
        "semantic_region_count": len(top_semantics),
        "source_route_semantic_region_count": len(semantics_by_route),
        "runtime_admitted_operator_count": admitted_count,
        "gap_count": len(gaps),
        "status": (
            "passed"
            if admitted_count == VIT_TOP_LEVEL_OPERATOR_COUNT and not gaps else "gap"
        ),
        "performance_claim": False,
        "graph_optimization_used": False,
        "atomic_dag_lowering_coverage_claim": False,
        "runtime_claim": "op_by_op_correctness_only",
        "operators": [asdict(record) for record in top_records],
        "source_route_records": [asdict(record) for record in source_route_records],
        "buffer_plan": {
            "valid": buffer_plan_validation.valid,
            "reason": buffer_plan_validation.reason,
            "buffer_count": len(buffer_plan.buffers),
            "final_output_buffer_ids": buffer_plan.final_output_buffer_ids,
        },
        "alpha_report": alpha_report,
    }

    artifact_sources_by_operator = _resolve_vit_tir_artifact_sources(
        top_records,
        enable_synthetic_tir=enable_synthetic_tir,
        synthetic_tir_operator_ids=synthetic_tir_operator_ids,
        enable_atomic_dag_gated_te_tir=enable_atomic_dag_gated_te_tir,
        atomic_dag_gated_te_operator_ids=atomic_dag_gated_te_operator_ids,
        tir_artifact_sources=tir_artifact_sources,
    )

    if artifact_sources_by_operator:
        requested_operator_ids = tuple(artifact_sources_by_operator)
        runtime_claim = _runtime_claim_for_artifact_sources(
            artifact_sources_by_operator.values()
        )
        artifact_registry = TIRArtifactRegistry(
            build_vit_tir_artifacts(
                execution_requests,
                target=target,
                artifact_sources_by_operator=artifact_sources_by_operator,
                atomics_by_route=atomics_by_route,
            )
        )
        reference_context = build_vit_synthetic_execution_context(target=target)
        tvm_context = build_vit_synthetic_execution_context(target=target)
        diagnostic = run_reference_tvm_e2e(
            execution_requests,
            reference_executor=ReferenceTensorExecutor(vit_reference_operator_specs()),
            tvm_executor=TvmPackedExecutor(artifact_registry=artifact_registry),
            reference_context=reference_context,
            tvm_context=tvm_context,
            source_route_record_count=len(source_route_records),
            semantic_proof_status_by_operator={
                record.operator_id: record.semantic_proof_status for record in top_records
            },
            proof_source_by_operator={
                record.operator_id: record.proof_source for record in top_records
            },
            tvm_operator_ids=requested_operator_ids,
            final_output_buffer_ids=buffer_plan.final_output_buffer_ids,
            runtime_claim=runtime_claim,
            e2e_gate_artifact_sources=E2E_GATE_ARTIFACT_SOURCES,
        )
        report["diagnostic_e2e"] = reference_tvm_e2e_result_to_dict(diagnostic)
        report["runtime_claim"] = runtime_claim
        report["e2e_pass"] = diagnostic.e2e_pass
        report["e2e_pass_alias_for"] = diagnostic.e2e_pass_alias_for
        report["e2e_correctness_pass"] = diagnostic.e2e_correctness_pass
        report["e2e_gate_kind"] = diagnostic.e2e_gate_kind
        report["atomic_dag_role"] = diagnostic.atomic_dag_role
        report["tir_generation_mode"] = diagnostic.tir_generation_mode
        report["dag_node_lowering_coverage_claim"] = (
            diagnostic.dag_node_lowering_coverage_claim
        )
    elif enable_e2e_scaffold or executor_registry is not None:
        report["diagnostic_e2e"] = e2e_result_to_dict(
            run_e2e_scaffold(
                _e2e_plans_from_top_records(top_records),
                executor_registry=executor_registry,
                context=None,
                target=target,
            )
        )
    if out_dir is not None:
        write_json_report(Path(out_dir) / "report.json", report)
    return report


def default_vit_capability_registry() -> CapabilityRegistry:
    """Capability records for canonical fixed-shape ViT top-level operators."""

    return CapabilityRegistry(
        [
            CapabilityRecord(
                capability_id=f"vit_fixed_shape.{semantic_key}.cuda",
                semantic_region_key=semantic_key,
                required_atomic_families=required,
                shape_constraints={},
                dtype_constraints={},
                layout_constraints={},
                allowed_source_origins=(SourceOrigin.TORCH_INDUCTOR,),
                runtime_available=True,
                lowering_contract_ids=lowering_ids,
            )
            for semantic_key, lowering_ids, required in (
                ("pointwise_grid2d", ("pointwise_flat",), (ATOMIC_FAMILY_MEMORY,)),
                ("pointwise_flat", ("pointwise_flat",), (ATOMIC_FAMILY_MEMORY,)),
                ("norm_row", ("pointwise_flat",), (ATOMIC_FAMILY_MEMORY,)),
                ("matmul", ("matmul_bias_epilogue",), (ATOMIC_FAMILY_DOT,)),
                ("conv_patchify", ("conv2d_nchw_static",), (ATOMIC_FAMILY_MEMORY,)),
                (
                    "attention_decomposed",
                    ("attention_decomposed",),
                    (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY),
                ),
            )
        ]
    )


def vit_tiny_fixed_shape_specs() -> tuple[VitOperatorSpec, ...]:
    """Return the active 16 source/lowering route inventory."""

    top_specs = {spec.operator_id: spec for spec in vit_tiny_fixed_shape_top_level_specs()}

    def from_top(
        route_id: str,
        top_id: str,
        lowering_contract_id: str,
        function_name: str,
        ttir: str,
        layout: str,
    ) -> VitOperatorSpec:
        top = top_specs[top_id]
        return VitOperatorSpec(
            operator_id=route_id,
            top_level_operator_id=top_id,
            top_level_execution_order=top.execution_order,
            semantic_region_key=top.semantic_region_key,
            lowering_contract_id=lowering_contract_id,
            lowering_contract_status=top.lowering_contract_status,
            function_name=function_name,
            ttir=ttir,
            shape_signature=top.shape_signature,
            dtype_signature=top.dtype_signature,
            layout_signature={"layout": layout},
        )

    return (
        from_top("patch_embed", "patch_embed", "conv2d_nchw_static", "vit_patch_embed", _memory_ttir("vit_patch_embed"), "nchw"),
        from_top("class_token_add", "class_token_add", "pointwise_flat", "vit_class_token_add", _memory_ttir("vit_class_token_add"), "grid2d"),
        from_top("position_add", "position_add", "pointwise_flat", "vit_position_add", _memory_ttir("vit_position_add"), "grid2d"),
        from_top("encoder_norm0", "encoder_norm0", "pointwise_flat", "vit_encoder_norm0", _memory_ttir("vit_encoder_norm0"), "row"),
        from_top("qkv_projection", "qkv_projection", "matmul_bias_epilogue", "vit_qkv_projection", _dot_store_ttir("vit_qkv_projection"), "row_major"),
        from_top("attention_qk", "attention", "attention_decomposed", "vit_attention_qk", _dot_store_ttir("vit_attention_qk"), "row_major"),
        from_top("attention_softmax", "attention", "pointwise_flat", "vit_attention_softmax", _memory_ttir("vit_attention_softmax"), "row_major"),
        from_top("attention_av", "attention", "attention_decomposed", "vit_attention_av", _dot_store_ttir("vit_attention_av"), "row_major"),
        from_top("attention_projection", "attention_projection", "matmul_bias_epilogue", "vit_attention_projection", _dot_store_ttir("vit_attention_projection"), "row_major"),
        from_top("residual_after_attention", "residual_after_attention", "pointwise_flat", "vit_residual_after_attention", _memory_ttir("vit_residual_after_attention"), "flat"),
        from_top("encoder_norm1", "encoder_norm1", "pointwise_flat", "vit_encoder_norm1", _memory_ttir("vit_encoder_norm1"), "row"),
        from_top("mlp_fc0", "mlp_fc0", "matmul_bias_epilogue", "vit_mlp_fc0", _dot_store_ttir("vit_mlp_fc0"), "row_major"),
        from_top("mlp_gelu", "mlp_gelu", "pointwise_flat", "vit_mlp_gelu", _memory_ttir("vit_mlp_gelu"), "flat"),
        from_top("mlp_fc1", "mlp_fc1", "matmul_bias_epilogue", "vit_mlp_fc1", _dot_store_ttir("vit_mlp_fc1"), "row_major"),
        from_top("residual_after_mlp", "residual_after_mlp", "pointwise_flat", "vit_residual_after_mlp", _memory_ttir("vit_residual_after_mlp"), "flat"),
        from_top("pooler", "pooler", "matmul_bias_epilogue", "vit_pooler", _dot_store_ttir("vit_pooler"), "row_major"),
    )


def vit_tiny_fixed_shape_top_level_specs() -> tuple[VitTopLevelOperatorSpec, ...]:
    """Return the canonical 14 top-level ViT semantic operators."""

    shape = {"batch": 1, "tokens": _TOKENS, "image": _IMAGE, "patch": _PATCH, "hidden": _HIDDEN, "mlp": _MLP}
    dtype = {"activation": "float32", "parameter": "float32"}

    def spec(
        operator_id: str,
        order: int,
        semantic_key: str,
        lowering_ids: tuple[str, ...],
        lowering_status: str,
        source_ids: tuple[str, ...],
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        layout: str,
        reference_impl_id: str,
        attrs: Mapping[str, Any] | None = None,
    ) -> VitTopLevelOperatorSpec:
        return VitTopLevelOperatorSpec(
            operator_id=operator_id,
            execution_order=order,
            semantic_region_key=semantic_key,
            lowering_contract_ids=lowering_ids,
            lowering_contract_status=lowering_status,
            source_route_record_ids=source_ids,
            input_buffer_ids=inputs,
            output_buffer_ids=outputs,
            shape_signature=shape,
            dtype_signature=dtype,
            layout_signature={"layout": layout},
            reference_impl_id=reference_impl_id,
            reference_attrs=dict(attrs or {}),
        )

    return (
        spec("patch_embed", 0, "conv_patchify", ("conv2d_nchw_static",), LOWERING_CONTRACT_STATUS_SPECIALIZED, ("patch_embed",), ("image", "patch_embed.weight", "patch_embed.bias"), ("patch_embed.out",), "nchw_to_tokens", "conv_patchify", {"patch": _PATCH}),
        spec("class_token_add", 1, "pointwise_grid2d", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE, ("class_token_add",), ("patch_embed.out", "class_token"), ("class_token_add.out",), "tokens_hidden_grid", "class_token_add"),
        spec("position_add", 2, "pointwise_grid2d", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE, ("position_add",), ("class_token_add.out", "position_embed"), ("position_add.out",), "tokens_hidden_grid", "position_add"),
        spec("encoder_norm0", 3, "norm_row", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE, ("encoder_norm0",), ("position_add.out", "encoder_norm0.scale", "encoder_norm0.bias"), ("encoder_norm0.out",), "row_major", "norm_row", {"epsilon": 1.0e-5}),
        spec("qkv_projection", 4, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, ("qkv_projection",), ("encoder_norm0.out", "qkv.weight", "qkv.bias"), ("qkv_projection.out",), "row_major", "matmul_bias"),
        spec("attention", 5, "attention_decomposed", ("attention_decomposed",), LOWERING_CONTRACT_STATUS_COMPOSITE, ("attention_qk", "attention_softmax", "attention_av"), ("qkv_projection.out",), ("attention.out",), "row_major", "attention_decomposed", {"hidden": _HIDDEN}),
        spec("attention_projection", 6, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, ("attention_projection",), ("attention.out", "attention_projection.weight", "attention_projection.bias"), ("attention_projection.out",), "row_major", "matmul_bias"),
        spec("residual_after_attention", 7, "pointwise_flat", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_DIRECT, ("residual_after_attention",), ("position_add.out", "attention_projection.out"), ("residual_after_attention.out",), "flat", "residual_add"),
        spec("encoder_norm1", 8, "norm_row", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE, ("encoder_norm1",), ("residual_after_attention.out", "encoder_norm1.scale", "encoder_norm1.bias"), ("encoder_norm1.out",), "row_major", "norm_row", {"epsilon": 1.0e-5}),
        spec("mlp_fc0", 9, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, ("mlp_fc0",), ("encoder_norm1.out", "mlp_fc0.weight", "mlp_fc0.bias"), ("mlp_fc0.out",), "row_major", "matmul_bias"),
        spec("mlp_gelu", 10, "pointwise_flat", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_DIRECT, ("mlp_gelu",), ("mlp_fc0.out",), ("mlp_gelu.out",), "flat", "gelu"),
        spec("mlp_fc1", 11, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, ("mlp_fc1",), ("mlp_gelu.out", "mlp_fc1.weight", "mlp_fc1.bias"), ("mlp_fc1.out",), "row_major", "matmul_bias"),
        spec("residual_after_mlp", 12, "pointwise_flat", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_DIRECT, ("residual_after_mlp",), ("residual_after_attention.out", "mlp_fc1.out"), ("last_hidden_state",), "flat", "residual_add"),
        spec("pooler", 13, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, ("pooler",), ("last_hidden_state", "pooler.weight", "pooler.bias"), ("pooler_output",), "row_major", "pooler"),
    )


def vit_reference_operator_specs() -> tuple[ReferenceOperatorSpec, ...]:
    """Reference specs for the 14 top-level synthetic ViT operators."""

    return tuple(
        ReferenceOperatorSpec(
            operator_id=spec.operator_id,
            semantic_region_key=spec.semantic_region_key,
            input_buffer_ids=spec.input_buffer_ids,
            output_buffer_ids=spec.output_buffer_ids,
            attrs=spec.reference_attrs,
            reference_impl_id=spec.reference_impl_id,
        )
        for spec in vit_tiny_fixed_shape_top_level_specs()
    )


def vit_execution_requests_from_top_records(
    top_records: Iterable[VitTopLevelOperatorRecord],
) -> tuple[OperatorExecutionRequest, ...]:
    """Build top-level OperatorExecutionRequest records."""

    requests = []
    for record in sorted(top_records, key=lambda item: item.execution_order):
        requests.append(
            OperatorExecutionRequest(
                model_id=record.model_id,
                operator_id=record.operator_id,
                execution_order=record.execution_order,
                declared_semantic_region_key=record.declared_semantic_region_key,
                semantic_region_key=record.semantic_region_key,
                semantic_region_hash=record.semantic_region_hash,
                lowering_contract_ids=record.lowering_contract_ids,
                lowering_contract_status=record.lowering_contract_status,
                source_route_record_ids=record.source_route_record_ids,
                source_ids=record.source_ids,
                atomic_dag_hashes=record.atomic_dag_hashes,
                atomic_dag_bundle_hash=record.atomic_dag_bundle_hash,
                capability_id=record.capability_id,
                shape_signature=record.shape_signature,
                dtype_signature=record.dtype_signature,
                layout_signature=record.layout_signature,
                input_buffer_ids=record.input_buffer_ids,
                output_buffer_ids=record.output_buffer_ids,
            )
        )
    return tuple(requests)


def vit_execution_requests_from_report(report: Mapping[str, Any]) -> tuple[OperatorExecutionRequest, ...]:
    """Rebuild top-level OperatorExecutionRequest records from a ViT report."""

    return tuple(
        OperatorExecutionRequest(
            model_id=record["model_id"],
            operator_id=record["operator_id"],
            execution_order=record["execution_order"],
            declared_semantic_region_key=record["declared_semantic_region_key"],
            semantic_region_key=record["semantic_region_key"],
            semantic_region_hash=record["semantic_region_hash"],
            lowering_contract_ids=tuple(record["lowering_contract_ids"]),
            lowering_contract_status=record["lowering_contract_status"],
            source_route_record_ids=tuple(record["source_route_record_ids"]),
            source_ids=tuple(record["source_ids"]),
            atomic_dag_hashes=tuple(record["atomic_dag_hashes"]),
            atomic_dag_bundle_hash=record["atomic_dag_bundle_hash"],
            capability_id=record["capability_id"],
            shape_signature=record["shape_signature"],
            dtype_signature=record["dtype_signature"],
            layout_signature=record["layout_signature"],
            input_buffer_ids=tuple(record["input_buffer_ids"]),
            output_buffer_ids=tuple(record["output_buffer_ids"]),
        )
        for record in sorted(report["operators"], key=lambda item: item["execution_order"])
    )


def build_vit_synthetic_buffer_plan(*, target: str = "llvm") -> RuntimeBufferPlan:
    """Build a connected fixed-shape synthetic ViT buffer plan."""

    values = _initial_buffer_values()
    specs = vit_tiny_fixed_shape_top_level_specs()
    buffers: dict[str, RuntimeBuffer] = {}
    edges: dict[str, RuntimeBufferEdge] = {}

    def add_buffer(
        buffer_id: str,
        shape: tuple[int, ...],
        *,
        producer: str | None,
        consumers: tuple[str, ...],
        is_model_input: bool = False,
        is_parameter: bool = False,
        is_model_output: bool = False,
    ) -> None:
        buffers[buffer_id] = RuntimeBuffer(
            buffer_id=buffer_id,
            shape=shape,
            dtype="float32",
            layout="row_major",
            device=target,
            ndarray=values.get(buffer_id),
            producer_operator_id=producer,
            consumer_operator_ids=consumers,
        )
        edges[buffer_id] = RuntimeBufferEdge(
            buffer_id=buffer_id,
            producer_operator_id=producer,
            consumer_operator_ids=consumers,
            is_model_input=is_model_input,
            is_parameter=is_parameter,
            is_model_output=is_model_output,
        )

    add_buffer("image", (_CHANNELS, _IMAGE, _IMAGE), producer=None, consumers=("patch_embed",), is_model_input=True)
    for buffer_id, shape, consumers in _parameter_buffer_specs():
        add_buffer(buffer_id, shape, producer=None, consumers=consumers, is_parameter=True)
    for spec in specs:
        for output_buffer_id in spec.output_buffer_ids:
            add_buffer(
                output_buffer_id,
                _buffer_shape(output_buffer_id),
                producer=spec.operator_id,
                consumers=_consumers_for_buffer(output_buffer_id, specs),
                is_model_output=output_buffer_id in {"last_hidden_state", "pooler_output"},
            )
    return RuntimeBufferPlan(
        buffers=buffers,
        edges=edges,
        final_output_buffer_ids=("last_hidden_state", "pooler_output"),
    )


def build_vit_synthetic_execution_context(*, target: str = "llvm") -> ExecutionContext:
    """Create a fresh ExecutionContext containing only model inputs and parameters."""

    values = _initial_buffer_values()
    return ExecutionContext(
        buffer_table={buffer_id: value.copy() for buffer_id, value in values.items()},
        target=target,
    )


def build_vit_synthetic_tir_artifacts(
    requests: Iterable[OperatorExecutionRequest],
    *,
    target: str = "llvm",
    operator_ids: Iterable[str] | None = None,
) -> tuple[TIRRegionArtifact, ...]:
    """Generate synthetic TIR artifacts for selected top-level ViT operators."""

    request_list = tuple(requests)
    selected = set(operator_ids) if operator_ids is not None else None
    return build_vit_tir_artifacts(
        request_list,
        target=target,
        artifact_sources_by_operator={
            request.operator_id: TIR_ARTIFACT_SOURCE_SYNTHETIC
            for request in request_list
            if selected is None or request.operator_id in selected
        },
    )


def build_vit_atomic_dag_gated_te_tir_artifacts(
    requests: Iterable[OperatorExecutionRequest],
    *,
    atomics_by_route: Mapping[str, Any],
    target: str = "llvm",
    operator_ids: Iterable[str] | None = None,
) -> tuple[TIRRegionArtifact, ...]:
    """Generate TE/TIR artifacts admitted by validated Atomic DAG route evidence."""

    request_list = tuple(requests)
    selected = set(operator_ids) if operator_ids is not None else None
    return build_vit_tir_artifacts(
        request_list,
        target=target,
        artifact_sources_by_operator={
            request.operator_id: TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE
            for request in request_list
            if selected is None or request.operator_id in selected
        },
        atomics_by_route=atomics_by_route,
    )


def build_vit_tir_artifacts(
    requests: Iterable[OperatorExecutionRequest],
    *,
    target: str = "llvm",
    artifact_sources_by_operator: Mapping[str, str],
    atomics_by_route: Mapping[str, Any] | None = None,
) -> tuple[TIRRegionArtifact, ...]:
    """Build top-level ViT TIR artifacts for the requested artifact sources."""

    buffer_plan = build_vit_synthetic_buffer_plan(target=target)
    buffer_meta = {
        buffer_id: {"shape": buffer.shape, "dtype": buffer.dtype, "layout": buffer.layout}
        for buffer_id, buffer in buffer_plan.buffers.items()
    }
    artifacts = []
    for request in sorted(requests, key=lambda item: item.execution_order):
        artifact_source = artifact_sources_by_operator.get(request.operator_id)
        if artifact_source is None:
            continue
        _validate_tir_artifact_source(artifact_source)
        generation_meta = _tir_generation_meta(
            request,
            artifact_source=artifact_source,
            target=target,
            atomics_by_route=atomics_by_route,
        )
        tir_module, primfunc_name = _tir_module_for_operator(
            request.operator_id,
            name_prefix=_primfunc_prefix_for_artifact_source(artifact_source),
        )
        step = LoweringStep(
            name=f"{request.operator_id}.{artifact_source}",
            contract_id=request.lowering_contract_ids[0],
            primfunc_name=primfunc_name,
            input_buffer_ids=request.input_buffer_ids,
            output_buffer_ids=request.output_buffer_ids,
            intermediate_buffer_ids=(),
            arg_buffer_order=request.input_buffer_ids + request.output_buffer_ids,
            attrs={"artifact_source": artifact_source},
        )
        region_meta = {
            "semantic_region_hash": request.semantic_region_hash,
            "source_route_record_ids": request.source_route_record_ids,
            "atomic_dag_hashes": request.atomic_dag_hashes,
            "atomic_dag_bundle_hash": request.atomic_dag_bundle_hash,
            "semantic_region_key": request.semantic_region_key,
            "shape": dict(request.shape_signature),
            "dtype": dict(request.dtype_signature),
            "layout": dict(request.layout_signature),
            "input_buffer_ids": request.input_buffer_ids,
            "output_buffer_ids": request.output_buffer_ids,
            "producer_consumer_hints": {},
            "lowering_contract_ids": request.lowering_contract_ids,
            "schedule_id": _schedule_id_for_artifact_source(artifact_source),
            "fusion_eligibility": False,
            "tir_artifact_source": artifact_source,
            "tir_generation": generation_meta,
            "buffers": buffer_meta,
        }
        artifacts.append(
            TIRRegionArtifact(
                model_id=request.model_id,
                operator_id=request.operator_id,
                semantic_region_key=request.semantic_region_key or "",
                lowering_contract_ids=request.lowering_contract_ids,
                lowering_contract_status=request.lowering_contract_status,
                source_route_record_ids=request.source_route_record_ids,
                atomic_dag_hashes=request.atomic_dag_hashes,
                atomic_dag_bundle_hash=request.atomic_dag_bundle_hash,
                tir_module=tir_module,
                target=target,
                lowering_plan=(step,),
                region_meta=region_meta,
                input_buffer_ids=request.input_buffer_ids,
                output_buffer_ids=request.output_buffer_ids,
                artifact_source=artifact_source,
                build_cache_key=stable_hash(
                    {
                        "model_id": request.model_id,
                        "operator_id": request.operator_id,
                        "target": target,
                        "source": artifact_source,
                        "atomic_dag_bundle_hash": request.atomic_dag_bundle_hash,
                    }
                ),
            )
        )
    return tuple(artifacts)


def _resolve_vit_tir_artifact_sources(
    top_records: Iterable[VitTopLevelOperatorRecord],
    *,
    enable_synthetic_tir: bool,
    synthetic_tir_operator_ids: Iterable[str] | None,
    enable_atomic_dag_gated_te_tir: bool,
    atomic_dag_gated_te_operator_ids: Iterable[str] | None,
    tir_artifact_sources: Mapping[str, str] | None,
) -> dict[str, str]:
    top_operator_ids = tuple(record.operator_id for record in top_records)
    sources: dict[str, str] = {}
    if tir_artifact_sources is not None:
        for operator_id, artifact_source in tir_artifact_sources.items():
            if operator_id not in top_operator_ids:
                raise ValueError(f"unknown ViT top-level operator for TIR artifact: {operator_id}")
            _validate_tir_artifact_source(artifact_source)
            sources[operator_id] = artifact_source
    if enable_synthetic_tir:
        selected = (
            tuple(synthetic_tir_operator_ids)
            if synthetic_tir_operator_ids is not None
            else top_operator_ids
        )
        for operator_id in selected:
            if operator_id not in top_operator_ids:
                raise ValueError(f"unknown synthetic TIR operator: {operator_id}")
            sources.setdefault(operator_id, TIR_ARTIFACT_SOURCE_SYNTHETIC)
    if enable_atomic_dag_gated_te_tir:
        selected = (
            tuple(atomic_dag_gated_te_operator_ids)
            if atomic_dag_gated_te_operator_ids is not None
            else top_operator_ids
        )
        for operator_id in selected:
            if operator_id not in top_operator_ids:
                raise ValueError(f"unknown Atomic-DAG-gated TE/TIR operator: {operator_id}")
            sources[operator_id] = TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE
    return dict(sorted(sources.items(), key=lambda item: top_operator_ids.index(item[0])))


def _runtime_claim_for_artifact_sources(artifact_sources: Iterable[str]) -> str:
    source_set = set(artifact_sources)
    if source_set and source_set.issubset(E2E_GATE_ARTIFACT_SOURCES):
        return RUNTIME_CLAIM_ATOMIC_DAG_GATED_TE
    if source_set == {TIR_ARTIFACT_SOURCE_SYNTHETIC}:
        return RUNTIME_CLAIM_SYNTHETIC_ONLY
    return "mixed_tir_artifact_correctness_only"


def _validate_tir_artifact_source(artifact_source: str) -> None:
    if artifact_source not in {
        TIR_ARTIFACT_SOURCE_SYNTHETIC,
        TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE,
        TIR_ARTIFACT_SOURCE_IMPORTED_TIRX,
    }:
        raise ValueError(f"unsupported TIR artifact source: {artifact_source}")


def _tir_generation_meta(
    request: OperatorExecutionRequest,
    *,
    artifact_source: str,
    target: str,
    atomics_by_route: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if artifact_source == TIR_ARTIFACT_SOURCE_SYNTHETIC:
        return {
            "generator_id": "vit_synthetic_te_codegen_v1",
            "proof_source": "synthetic_runtime_channel_debug",
            "tir_generation_mode": "synthetic_operator_template_te",
            "atomic_dag_role": "none",
            "atomic_dag_gated": False,
            "atomic_dag_lowered": False,
            "dag_node_lowering_coverage_claim": False,
        }
    if artifact_source == TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE:
        if atomics_by_route is None:
            raise ValueError("Atomic-DAG-gated TE/TIR requires route Atomic DAG records")
        return _atomic_dag_gated_te_tir_generation_meta(
            request, atomics_by_route, target=target
        )
    if artifact_source == TIR_ARTIFACT_SOURCE_IMPORTED_TIRX:
        raise ValueError("imported_tirx artifacts require imported IRModule inputs")
    raise ValueError(f"unsupported TIR artifact source: {artifact_source}")


def _atomic_dag_gated_te_tir_generation_meta(
    request: OperatorExecutionRequest,
    atomics_by_route: Mapping[str, Any],
    *,
    target: str,
) -> dict[str, Any]:
    route_atomics = []
    for route_id in request.source_route_record_ids:
        atomic = atomics_by_route.get(route_id)
        if atomic is None:
            raise ValueError(f"missing Atomic DAG for route record: {route_id}")
        route_atomics.append(atomic)

    atomic_hashes = tuple(atomic.atomic_dag_hash for atomic in route_atomics)
    if atomic_hashes != request.atomic_dag_hashes:
        raise ValueError("Atomic-DAG-gated TE/TIR hash bundle does not match request")
    if any(atomic.target != target for atomic in route_atomics):
        raise ValueError("Atomic-DAG-gated TE/TIR target does not match request target")
    unsupported = sum(len(atomic.unsupported_atomic_ops) for atomic in route_atomics)
    if unsupported:
        raise ValueError("Atomic-DAG-gated TE/TIR refuses unsupported atomic ops")
    if not all(atomic.producer_consumer_consistent for atomic in route_atomics):
        raise ValueError("Atomic-DAG-gated TE/TIR requires producer/consumer consistency")

    family_histogram: dict[str, int] = {}
    for atomic in route_atomics:
        for family, count in atomic.atomic_family_histogram.items():
            family_histogram[family] = family_histogram.get(family, 0) + int(count)
    required = _required_atomic_families_for_tir_generation(request)
    missing = tuple(family for family in required if family not in family_histogram)
    if missing:
        raise ValueError(
            "Atomic-DAG-gated TE/TIR missing required atomic families: "
            + ", ".join(missing)
        )

    return {
        "generator_id": "vit_atomic_dag_gated_te_codegen_v1",
        "proof_source": "atomic_dag_contract_validation",
        "tir_generation_mode": "operator_template_te",
        "atomic_dag_role": "admission_gate",
        "atomic_dag_gated": True,
        "atomic_dag_lowered": False,
        "atomic_dag_hash_checked": True,
        "atomic_family_requirements_checked": True,
        "producer_consumer_consistency_checked": True,
        "dag_node_lowering_coverage_claim": False,
        "input_atomic_dag_hashes": atomic_hashes,
        "input_atomic_dag_bundle_hash": request.atomic_dag_bundle_hash,
        "input_atomic_family_histogram": dict(sorted(family_histogram.items())),
        "required_atomic_families": required,
    }


def _required_atomic_families_for_tir_generation(
    request: OperatorExecutionRequest,
) -> tuple[str, ...]:
    if request.semantic_region_key == "attention_decomposed":
        return (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY)
    if "matmul_bias_epilogue" in request.lowering_contract_ids:
        return (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY)
    if request.semantic_region_key == "matmul":
        return (ATOMIC_FAMILY_DOT,)
    return (ATOMIC_FAMILY_MEMORY,)


def _primfunc_prefix_for_artifact_source(artifact_source: str) -> str:
    if artifact_source == TIR_ARTIFACT_SOURCE_SYNTHETIC:
        return "synthetic"
    if artifact_source == TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE:
        return "atomic_dag_gated_te"
    if artifact_source == TIR_ARTIFACT_SOURCE_IMPORTED_TIRX:
        return "imported_tirx"
    raise ValueError(f"unsupported TIR artifact source: {artifact_source}")


def _schedule_id_for_artifact_source(artifact_source: str) -> str:
    if artifact_source == TIR_ARTIFACT_SOURCE_SYNTHETIC:
        return "synthetic_te_default"
    if artifact_source == TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE:
        return "atomic_dag_gated_te_default"
    if artifact_source == TIR_ARTIFACT_SOURCE_IMPORTED_TIRX:
        return "imported_tirx"
    raise ValueError(f"unsupported TIR artifact source: {artifact_source}")


def _top_level_records_from_source_routes(
    *,
    top_specs: Mapping[str, VitTopLevelOperatorSpec],
    sources_by_route: Mapping[str, Any],
    atomics_by_route: Mapping[str, Any],
    validations_by_route: Mapping[str, ContractValidationResult],
    semantics_by_route: Mapping[str, SemanticRegionRecord],
    registry: CapabilityRegistry,
) -> tuple[
    list[VitTopLevelOperatorRecord],
    list[SemanticRegionRecord],
    list[Any],
    list[Any],
]:
    top_records = []
    top_semantics = []
    manifests = []
    gaps = []

    for top_spec in sorted(top_specs.values(), key=lambda spec: spec.execution_order):
        route_ids = top_spec.source_route_record_ids
        route_sources = tuple(sources_by_route[route_id] for route_id in route_ids)
        route_atomics = tuple(atomics_by_route[route_id] for route_id in route_ids)
        route_validations = tuple(validations_by_route[route_id] for route_id in route_ids)
        atomic_hashes = tuple(atomic.atomic_dag_hash for atomic in route_atomics)
        bundle_hash = stable_hash({"atomic_dag_hashes": atomic_hashes})
        semantic_hash = stable_hash(
            {
                "model_id": VIT_TINY_FIXED_SHAPE_MODEL_ID,
                "operator_id": top_spec.operator_id,
                "semantic_region_key": top_spec.semantic_region_key,
                "source_route_record_ids": route_ids,
                "atomic_dag_hashes": atomic_hashes,
            }
        )
        all_valid = all(validation.passed for validation in route_validations)
        representative_semantic = semantics_by_route.get(route_ids[0])
        resolution = None
        admission = RuntimeAdmissionResult(False, "gap", "contract validation failed")
        gap_reason = None
        if all_valid and representative_semantic is not None:
            resolution = registry.resolve(
                source=route_sources[0],
                atomic=route_atomics[0],
                semantic_region=representative_semantic,
            )
            manifest_for_admission = build_operator_manifest_record(
                model_id=VIT_TINY_FIXED_SHAPE_MODEL_ID,
                operator_id=top_spec.operator_id,
                execution_order=top_spec.execution_order,
                source=route_sources[0],
                atomic=route_atomics[0],
                semantic_region=representative_semantic,
                validation=route_validations[0],
                shape_signature=top_spec.shape_signature,
                dtype_signature=top_spec.dtype_signature,
                layout_signature=top_spec.layout_signature,
                runtime_admission_status="not_checked",
                optional_model_anchor=VIT_RUNNER_ID,
                source_route_record_ids=route_ids,
                source_ids=tuple(source.source_id for source in route_sources),
                atomic_dag_hashes=atomic_hashes,
                atomic_dag_bundle_hash=bundle_hash,
                lowering_contract_ids=top_spec.lowering_contract_ids,
                lowering_contract_status=top_spec.lowering_contract_status,
            )
            admission = admit_runtime(
                source=route_sources[0],
                atomic=route_atomics[0],
                validation=route_validations[0],
                semantic_region=representative_semantic,
                manifest_record=manifest_for_admission,
                registry_resolution=resolution,
            )
            if len(route_ids) > 1 and not all_valid:
                admission = RuntimeAdmissionResult(False, "gap", "composite validation failed")
            gap_reason = admission.reason
            top_semantics.append(representative_semantic)
        else:
            failing = next(
                (validation for validation in route_validations if not validation.passed),
                route_validations[0],
            )
            gap_reason = failing.failure_reason or "contract validation failed"

        manifest = build_operator_manifest_record(
            model_id=VIT_TINY_FIXED_SHAPE_MODEL_ID,
            operator_id=top_spec.operator_id,
            execution_order=top_spec.execution_order,
            source=route_sources[0],
            atomic=route_atomics[0],
            semantic_region=representative_semantic if all_valid else None,
            validation=route_validations[0],
            shape_signature=top_spec.shape_signature,
            dtype_signature=top_spec.dtype_signature,
            layout_signature=top_spec.layout_signature,
            runtime_admission_status=admission.status,
            gap_reason=gap_reason,
            optional_model_anchor=VIT_RUNNER_ID,
            declared_semantic_region_key=top_spec.semantic_region_key,
            source_route_record_ids=route_ids,
            source_ids=tuple(source.source_id for source in route_sources),
            atomic_dag_hashes=atomic_hashes,
            atomic_dag_bundle_hash=bundle_hash,
            lowering_contract_ids=top_spec.lowering_contract_ids,
            lowering_contract_status=top_spec.lowering_contract_status,
        )
        manifests.append(manifest)
        if gap_reason is not None:
            gaps.append(
                build_manifest_gap(
                    model_id=VIT_TINY_FIXED_SHAPE_MODEL_ID,
                    operator_id=top_spec.operator_id,
                    source_id=route_sources[0].source_id,
                    reason=gap_reason,
                )
            )
        top_records.append(
            VitTopLevelOperatorRecord(
                model_id=VIT_TINY_FIXED_SHAPE_MODEL_ID,
                operator_id=top_spec.operator_id,
                execution_order=top_spec.execution_order,
                declared_semantic_region_key=top_spec.semantic_region_key,
                semantic_region_key=top_spec.semantic_region_key,
                semantic_region_hash=semantic_hash if all_valid else None,
                matched_contract_id=top_spec.lowering_contract_ids[0],
                contract_validation_status="passed" if all_valid else "failed",
                semantic_proof_status=_semantic_proof_status(top_spec.lowering_contract_status),
                proof_source="atomic_dag_contract_validation",
                support_claim_source="validated_contract" if all_valid else "explicit_gap",
                wrapper_provider_claim=False,
                lowering_contract_ids=top_spec.lowering_contract_ids,
                lowering_contract_status=top_spec.lowering_contract_status,
                lowering_contract_warning=_lowering_contract_warning(
                    top_spec.lowering_contract_status
                ),
                source_route_record_ids=route_ids,
                source_ids=tuple(source.source_id for source in route_sources),
                atomic_dag_hashes=atomic_hashes,
                atomic_dag_bundle_hash=bundle_hash,
                shape_signature=top_spec.shape_signature,
                dtype_signature=top_spec.dtype_signature,
                layout_signature=top_spec.layout_signature,
                input_buffer_ids=top_spec.input_buffer_ids,
                output_buffer_ids=top_spec.output_buffer_ids,
                registry_supported=bool(resolution.supported) if resolution is not None else False,
                capability_id=resolution.capability_id if resolution is not None else None,
                runtime_admission_status=admission.status,
                gap_reason=gap_reason,
            )
        )
    return top_records, top_semantics, manifests, gaps


def _e2e_plans_from_top_records(
    top_records: Iterable[VitTopLevelOperatorRecord],
) -> tuple[E2EOperatorPlan, ...]:
    plans = []
    for record in top_records:
        plans.append(
            E2EOperatorPlan(
                model_id=record.model_id,
                operator_id=record.operator_id,
                execution_order=record.execution_order,
                declared_semantic_region_key=record.declared_semantic_region_key,
                semantic_region_key=record.semantic_region_key,
                semantic_region_hash=record.semantic_region_hash,
                lowering_contract_ids=record.lowering_contract_ids,
                lowering_contract_status=record.lowering_contract_status,
                source_route_record_ids=record.source_route_record_ids,
                source_ids=record.source_ids,
                atomic_dag_hashes=record.atomic_dag_hashes,
                atomic_dag_bundle_hash=record.atomic_dag_bundle_hash,
                capability_id=record.capability_id,
                admission_status=record.runtime_admission_status,
                shape_signature=record.shape_signature,
                dtype_signature=record.dtype_signature,
                layout_signature=record.layout_signature,
                input_buffer_ids=record.input_buffer_ids,
                output_buffer_ids=record.output_buffer_ids,
            )
        )
    return tuple(plans)


def _initial_buffer_values() -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    values = {
        "image": _scaled_array((_CHANNELS, _IMAGE, _IMAGE), 0.01, -0.2),
        "patch_embed.weight": _scaled_array((_HIDDEN, _CHANNELS, _PATCH, _PATCH), 0.005, -0.03),
        "patch_embed.bias": _scaled_array((_HIDDEN,), 0.002, 0.01),
        "class_token": _scaled_array((1, _HIDDEN), 0.003, 0.02),
        "position_embed": _scaled_array((_TOKENS, _HIDDEN), 0.002, -0.015),
        "encoder_norm0.scale": np.ones((_HIDDEN,), dtype=np.float32),
        "encoder_norm0.bias": _scaled_array((_HIDDEN,), 0.001, 0.0),
        "qkv.weight": _scaled_array((_HIDDEN, 3 * _HIDDEN), 0.004, -0.04),
        "qkv.bias": _scaled_array((3 * _HIDDEN,), 0.002, -0.01),
        "attention_projection.weight": _scaled_array((_HIDDEN, _HIDDEN), 0.004, -0.02),
        "attention_projection.bias": _scaled_array((_HIDDEN,), 0.002, 0.005),
        "encoder_norm1.scale": np.ones((_HIDDEN,), dtype=np.float32),
        "encoder_norm1.bias": _scaled_array((_HIDDEN,), 0.001, 0.0),
        "mlp_fc0.weight": _scaled_array((_HIDDEN, _MLP), 0.003, -0.025),
        "mlp_fc0.bias": _scaled_array((_MLP,), 0.002, 0.01),
        "mlp_fc1.weight": _scaled_array((_MLP, _HIDDEN), 0.003, -0.02),
        "mlp_fc1.bias": _scaled_array((_HIDDEN,), 0.002, -0.005),
        "pooler.weight": _scaled_array((_HIDDEN, _HIDDEN), 0.003, -0.015),
        "pooler.bias": _scaled_array((_HIDDEN,), 0.002, 0.0),
    }
    return values


def _scaled_array(shape: tuple[int, ...], scale: float, offset: float):
    import numpy as np  # pylint: disable=import-outside-toplevel

    return (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) * scale + offset).astype(
        np.float32
    )


def _parameter_buffer_specs() -> tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...]:
    return (
        ("patch_embed.weight", (_HIDDEN, _CHANNELS, _PATCH, _PATCH), ("patch_embed",)),
        ("patch_embed.bias", (_HIDDEN,), ("patch_embed",)),
        ("class_token", (1, _HIDDEN), ("class_token_add",)),
        ("position_embed", (_TOKENS, _HIDDEN), ("position_add",)),
        ("encoder_norm0.scale", (_HIDDEN,), ("encoder_norm0",)),
        ("encoder_norm0.bias", (_HIDDEN,), ("encoder_norm0",)),
        ("qkv.weight", (_HIDDEN, 3 * _HIDDEN), ("qkv_projection",)),
        ("qkv.bias", (3 * _HIDDEN,), ("qkv_projection",)),
        ("attention_projection.weight", (_HIDDEN, _HIDDEN), ("attention_projection",)),
        ("attention_projection.bias", (_HIDDEN,), ("attention_projection",)),
        ("encoder_norm1.scale", (_HIDDEN,), ("encoder_norm1",)),
        ("encoder_norm1.bias", (_HIDDEN,), ("encoder_norm1",)),
        ("mlp_fc0.weight", (_HIDDEN, _MLP), ("mlp_fc0",)),
        ("mlp_fc0.bias", (_MLP,), ("mlp_fc0",)),
        ("mlp_fc1.weight", (_MLP, _HIDDEN), ("mlp_fc1",)),
        ("mlp_fc1.bias", (_HIDDEN,), ("mlp_fc1",)),
        ("pooler.weight", (_HIDDEN, _HIDDEN), ("pooler",)),
        ("pooler.bias", (_HIDDEN,), ("pooler",)),
    )


def _buffer_shape(buffer_id: str) -> tuple[int, ...]:
    if buffer_id in {
        "patch_embed.out",
        "class_token_add.out",
        "position_add.out",
        "encoder_norm0.out",
        "attention.out",
        "attention_projection.out",
        "residual_after_attention.out",
        "encoder_norm1.out",
        "mlp_fc1.out",
        "last_hidden_state",
    }:
        return (_TOKENS, _HIDDEN)
    if buffer_id == "qkv_projection.out":
        return (_TOKENS, 3 * _HIDDEN)
    if buffer_id in {"mlp_fc0.out", "mlp_gelu.out"}:
        return (_TOKENS, _MLP)
    if buffer_id == "pooler_output":
        return (1, _HIDDEN)
    raise KeyError(buffer_id)


def _consumers_for_buffer(
    buffer_id: str, specs: Iterable[VitTopLevelOperatorSpec]
) -> tuple[str, ...]:
    return tuple(
        spec.operator_id for spec in specs if buffer_id in spec.input_buffer_ids
    )


def _semantic_proof_status(lowering_status: str) -> str:
    if lowering_status in {
        LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE,
        LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE,
    }:
        return "surrogate_validated"
    return "validated"


def _lowering_contract_warning(status: str | None) -> str | None:
    if status == LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE:
        return "temporary surrogate validated by Atomic DAG; not a full norm_row support claim"
    if status == LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE:
        return "flattened surrogate validated by Atomic DAG; grid/indexing/mask/broadcast metadata is preserved"
    return None


def _synthetic_tir_module_for_operator(operator_id: str) -> tuple[Any, str]:
    return _tir_module_for_operator(operator_id, name_prefix="synthetic")


def _tir_module_for_operator(operator_id: str, *, name_prefix: str) -> tuple[Any, str]:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    name = f"{name_prefix}_{operator_id}"
    if operator_id == "patch_embed":
        image = te.placeholder((_CHANNELS, _IMAGE, _IMAGE), name="image")
        weight = te.placeholder((_HIDDEN, _CHANNELS, _PATCH, _PATCH), name="weight")
        bias = te.placeholder((_HIDDEN,), name="bias")
        rc = te.reduce_axis((0, _CHANNELS), name="rc")
        ry = te.reduce_axis((0, _PATCH), name="ry")
        rx = te.reduce_axis((0, _PATCH), name="rx")
        conv = te.compute(
            (_TOKENS, _HIDDEN),
            lambda token, hidden: te.sum(
                image[
                    rc,
                    (token // _PATCH_GRID) * _PATCH + ry,
                    (token % _PATCH_GRID) * _PATCH + rx,
                ]
                * weight[hidden, rc, ry, rx],
                axis=(rc, ry, rx),
            ),
            name="conv",
        )
        out = te.compute((_TOKENS, _HIDDEN), lambda i, j: conv[i, j] + bias[j], name="out")
        args = [image, weight, bias, out]
    elif operator_id == "class_token_add":
        x = te.placeholder((_TOKENS, _HIDDEN), name="x")
        cls = te.placeholder((1, _HIDDEN), name="cls")
        out = te.compute(
            (_TOKENS, _HIDDEN),
            lambda i, j: x[i, j] + te.if_then_else(i == 0, cls[0, j], 0.0),
            name="out",
        )
        args = [x, cls, out]
    elif operator_id in {"position_add", "residual_after_attention", "residual_after_mlp"}:
        a = te.placeholder(_buffer_shape(_binary_input_shape_id(operator_id)), name="a")
        b = te.placeholder(_buffer_shape(_binary_input_shape_id(operator_id)), name="b")
        out = te.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="out")
        args = [a, b, out]
    elif operator_id in {"encoder_norm0", "encoder_norm1"}:
        x = te.placeholder((_TOKENS, _HIDDEN), name="x")
        scale = te.placeholder((_HIDDEN,), name="scale")
        bias = te.placeholder((_HIDDEN,), name="bias")
        r_mean = te.reduce_axis((0, _HIDDEN), name="r_mean")
        mean = te.compute(
            (_TOKENS,),
            lambda i: te.sum(x[i, r_mean] * (1.0 / _HIDDEN), axis=r_mean),
            name="mean",
        )
        r_var = te.reduce_axis((0, _HIDDEN), name="r_var")
        var = te.compute(
            (_TOKENS,),
            lambda i: te.sum(
                (x[i, r_var] - mean[i])
                * (x[i, r_var] - mean[i])
                * (1.0 / _HIDDEN),
                axis=r_var,
            ),
            name="var",
        )
        out = te.compute(
            (_TOKENS, _HIDDEN),
            lambda i, j: (x[i, j] - mean[i]) / te.sqrt(var[i] + 1.0e-5) * scale[j] + bias[j],
            name="out",
        )
        args = [x, scale, bias, out]
    elif operator_id == "qkv_projection":
        args = _matmul_bias_te_args(te, _HIDDEN, 3 * _HIDDEN)
    elif operator_id == "attention":
        qkv = te.placeholder((_TOKENS, 3 * _HIDDEN), name="qkv")
        r = te.reduce_axis((0, _HIDDEN), name="r")
        scores = te.compute(
            (_TOKENS, _TOKENS),
            lambda i, j: te.sum(qkv[i, r] * qkv[j, _HIDDEN + r] * (1.0 / (_HIDDEN**0.5)), axis=r),
            name="scores",
        )
        exp_scores = te.compute(
            (_TOKENS, _TOKENS), lambda i, j: te.exp(scores[i, j]), name="exp_scores"
        )
        rk = te.reduce_axis((0, _TOKENS), name="rk")
        denom = te.compute(
            (_TOKENS,), lambda i: te.sum(exp_scores[i, rk], axis=rk), name="denom"
        )
        weights = te.compute(
            (_TOKENS, _TOKENS), lambda i, j: exp_scores[i, j] / denom[i], name="weights"
        )
        rv = te.reduce_axis((0, _TOKENS), name="rv")
        out = te.compute(
            (_TOKENS, _HIDDEN),
            lambda i, h: te.sum(weights[i, rv] * qkv[rv, 2 * _HIDDEN + h], axis=rv),
            name="out",
        )
        args = [qkv, out]
    elif operator_id == "attention_projection":
        args = _matmul_bias_te_args(te, _HIDDEN, _HIDDEN)
    elif operator_id == "mlp_fc0":
        args = _matmul_bias_te_args(te, _HIDDEN, _MLP)
    elif operator_id == "mlp_gelu":
        x = te.placeholder((_TOKENS, _MLP), name="x")
        out = te.compute(
            (_TOKENS, _MLP),
            lambda i, j: 0.5
            * x[i, j]
            * (
                1.0
                + te.tanh(
                    (2.0 / 3.141592653589793) ** 0.5
                    * (x[i, j] + 0.044715 * x[i, j] * x[i, j] * x[i, j])
                )
            ),
            name="out",
        )
        args = [x, out]
    elif operator_id == "mlp_fc1":
        lhs = te.placeholder((_TOKENS, _MLP), name="lhs")
        rhs = te.placeholder((_MLP, _HIDDEN), name="rhs")
        bias = te.placeholder((_HIDDEN,), name="bias")
        rk = te.reduce_axis((0, _MLP), name="rk")
        dot = te.compute(
            (_TOKENS, _HIDDEN), lambda i, j: te.sum(lhs[i, rk] * rhs[rk, j], axis=rk), name="dot"
        )
        out = te.compute((_TOKENS, _HIDDEN), lambda i, j: dot[i, j] + bias[j], name="out")
        args = [lhs, rhs, bias, out]
    elif operator_id == "pooler":
        lhs = te.placeholder((_TOKENS, _HIDDEN), name="lhs")
        rhs = te.placeholder((_HIDDEN, _HIDDEN), name="rhs")
        bias = te.placeholder((_HIDDEN,), name="bias")
        rk = te.reduce_axis((0, _HIDDEN), name="rk")
        dot = te.compute(
            (1, _HIDDEN), lambda i, j: te.sum(lhs[0, rk] * rhs[rk, j], axis=rk), name="dot"
        )
        out = te.compute((1, _HIDDEN), lambda i, j: dot[i, j] + bias[j], name="out")
        args = [lhs, rhs, bias, out]
    else:
        raise ValueError(f"unsupported synthetic TIR operator: {operator_id}")

    prim = te.create_prim_func(args).with_attr("global_symbol", name)
    return tvm.IRModule({name: prim}), name


def _matmul_bias_te_args(te: Any, input_dim: int, output_dim: int) -> list[Any]:
    lhs = te.placeholder((_TOKENS, input_dim), name="lhs")
    rhs = te.placeholder((input_dim, output_dim), name="rhs")
    bias = te.placeholder((output_dim,), name="bias")
    rk = te.reduce_axis((0, input_dim), name="rk")
    dot = te.compute(
        (_TOKENS, output_dim), lambda i, j: te.sum(lhs[i, rk] * rhs[rk, j], axis=rk), name="dot"
    )
    out = te.compute((_TOKENS, output_dim), lambda i, j: dot[i, j] + bias[j], name="out")
    return [lhs, rhs, bias, out]


def _binary_input_shape_id(operator_id: str) -> str:
    if operator_id == "position_add":
        return "position_add.out"
    if operator_id == "residual_after_attention":
        return "residual_after_attention.out"
    if operator_id == "residual_after_mlp":
        return "last_hidden_state"
    raise KeyError(operator_id)


def _memory_ttir(function_name: str) -> str:
    return f"""
module {{
  tt.func public @{function_name}(%x: !tt.ptr<f32>) {{
    %0 = tt.load %x : !tt.ptr<f32> -> tensor<16xf32>
    tt.store %x, %0 : !tt.ptr<f32>, tensor<16xf32>
  }}
}}
"""


def _dot_store_ttir(function_name: str) -> str:
    return f"""
module {{
  tt.func public @{function_name}(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {{
    %0 = tt.dot %a, %b : (!tt.ptr<f32>, !tt.ptr<f32>) -> tensor<16x16xf32>
    tt.store %c, %0 : !tt.ptr<f32>, tensor<16x16xf32>
  }}
}}
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m tvm.contrib.triton_tvm.models.vit``."""

    parser = argparse.ArgumentParser(description="Run fixed-shape ViT through the active route")
    parser.add_argument("--target", default="cuda")
    parser.add_argument("--producer-version", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--e2e-scaffold", action="store_true")
    parser.add_argument("--synthetic-tir", action="store_true")
    parser.add_argument("--atomic-dag-gated-te-tir", action="store_true")
    args = parser.parse_args(argv)
    report = run_vit_tiny_fixed_shape_route(
        target=args.target,
        producer_version=args.producer_version,
        out_dir=args.out_dir,
        enable_e2e_scaffold=args.e2e_scaffold,
        enable_synthetic_tir=args.synthetic_tir,
        enable_atomic_dag_gated_te_tir=args.atomic_dag_gated_te_tir,
    )
    diagnostic = report.get("diagnostic_e2e")
    diagnostic_suffix = (
        f" diagnostic_e2e={diagnostic['status']}" if diagnostic is not None else ""
    )
    print(
        "ViT active route "
        f"status={report['status']} "
        f"admitted={report['runtime_admitted_operator_count']}/{report['operator_count']} "
        f"source_routes={report['source_route_record_count']} "
        f"gaps={report['gap_count']}"
        f"{diagnostic_suffix}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
