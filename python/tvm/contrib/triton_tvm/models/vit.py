# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Fixed-shape ViT runner for the active Triton-TVM alpha route."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..atomic import (
    ATOMIC_FAMILY_DOT,
    ATOMIC_FAMILY_MEMORY,
    build_atomic_dag,
    validate_source_atomic_join,
)
from ..autotune import (
    AUTOTUNE_CODEGEN_SOURCE,
    AutotuneConfig,
    AutotunePlan,
    AutotuneWorkloadSpec,
    autotune_config_from_mapping,
    autotune_plan_report,
    build_autotune_plan,
    load_json_config,
    meta_schedule_target,
    parse_tuning_cores,
    tune_or_load_workload,
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
    TIRArtifactBuildCache,
    TIRArtifactRegistry,
    TIRRegionArtifact,
    TvmPackedExecutionSession,
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

VIT_S_FIXED_SHAPE_MODEL_ID = "vit_s_16_224_fixed_shape"
VIT_S_RUNNER_ID = "vit_s_16_224_active_route_v1"
VIT_S_SOURCE_ROUTE_RECORD_COUNT = 149
VIT_S_TOP_LEVEL_OPERATOR_COUNT = 125
VIT_S_OPERATOR_COUNT = VIT_S_TOP_LEVEL_OPERATOR_COUNT

_VIT_S_BATCH = 1
_VIT_S_CHANNELS = 3
_VIT_S_IMAGE = 224
_VIT_S_PATCH = 16
_VIT_S_PATCH_GRID = _VIT_S_IMAGE // _VIT_S_PATCH
_VIT_S_PATCH_TOKENS = _VIT_S_PATCH_GRID * _VIT_S_PATCH_GRID
_VIT_S_TOKENS = _VIT_S_PATCH_TOKENS + 1
_VIT_S_HIDDEN = 384
_VIT_S_HEADS = 6
_VIT_S_HEAD_DIM = _VIT_S_HIDDEN // _VIT_S_HEADS
_VIT_S_MLP = 1536
_VIT_S_LAYERS = 12
_VIT_S_CLASSES = 1000
VIT_S_WEIGHT_SOURCE_DETERMINISTIC = "deterministic_random"
VIT_S_WEIGHT_SOURCE_TIMM = "timm"
VIT_S_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
VIT_S_BACKEND_REFERENCE = "reference"
VIT_S_BACKEND_TRITON_TVM = "triton-tvm"
VIT_S_MATMUL_CODEGEN_TE_WORKLOAD = "te_workload"
VIT_S_MATMUL_CODEGEN_TE_AUTOTUNE = AUTOTUNE_CODEGEN_SOURCE
VIT_S_DEFAULT_MATMUL_AUTOTUNE_WORK_DIR = "/tmp/triton_tvm_vit_s_matmul_autotune"
VIT_S_OPERATOR_CODEGEN_TE_AUTOTUNE = AUTOTUNE_CODEGEN_SOURCE
VIT_S_DEFAULT_OPERATOR_AUTOTUNE_WORK_DIR = "/tmp/triton_tvm_vit_s_operator_autotune"
_VIT_S_CUDA_THREADS_PER_BLOCK = 256


@dataclass(frozen=True)
class VitSWeightSourceSpec:
    """Registered source of ViT-S input weights and synthetic inputs."""

    source_id: str
    provider_id: str
    description: str
    default_config: Mapping[str, Any]
    supports_pretrained: bool = False


@dataclass(frozen=True)
class VitSWeightBundle:
    """Materialized ViT-S inputs from a configured weight source."""

    source_id: str
    provider_id: str
    values: Mapping[str, Any]
    metadata: Mapping[str, Any]
    torch_model: Any | None = None


@dataclass(frozen=True)
class VitSMatmulAutotuneConfig:
    """MetaSchedule settings for ViT-S matmul TE workloads."""

    enabled: bool = False
    work_dir: str | None = None
    max_trials_global: int = 0
    max_trials_per_task: int | None = None
    num_trials_per_iter: int = 1
    cost_model: str = "random"
    num_tuning_cores: int | str = 1
    task_scheduler: str = "gradient"
    strategy: str = "evolutionary"
    seed: int | None = None
    force_retune: bool = False


@dataclass(frozen=True)
class VitSOperatorAutotuneConfig:
    """MetaSchedule settings for non-matmul ViT-S TE workloads."""

    enabled: bool = False
    work_dir: str | None = None
    max_trials_global: int = 0
    max_trials_per_task: int | None = None
    num_trials_per_iter: int = 1
    cost_model: str = "random"
    num_tuning_cores: int | str = 1
    task_scheduler: str = "gradient"
    strategy: str = "evolutionary"
    seed: int | None = None
    force_retune: bool = False


class VitSWeightSourceRegistry:
    """Registry for extensible ViT-S input weight providers."""

    def __init__(self):
        self._entries: dict[str, tuple[VitSWeightSourceSpec, Callable[..., VitSWeightBundle]]] = {}

    def register(
        self,
        spec: VitSWeightSourceSpec,
        loader: Callable[..., VitSWeightBundle],
    ) -> None:
        if not spec.source_id:
            raise ValueError("weight source id is required")
        self._entries[spec.source_id] = (spec, loader)

    def load(self, source_id: str, **config: Any) -> VitSWeightBundle:
        entry = self._entries.get(source_id)
        if entry is None:
            raise ValueError(f"unknown ViT-S weight source: {source_id}")
        spec, loader = entry
        merged = dict(spec.default_config)
        merged.update({key: value for key, value in config.items() if value is not None})
        return loader(**merged)

    def specs(self) -> tuple[VitSWeightSourceSpec, ...]:
        return tuple(entry[0] for entry in self._entries.values())

    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


_VIT_S_WEIGHT_SOURCE_REGISTRY: VitSWeightSourceRegistry | None = None


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


def default_vit_s_capability_registry() -> CapabilityRegistry:
    """Capability records for canonical fixed-shape ViT-S top-level operators."""

    return CapabilityRegistry(
        [
            CapabilityRecord(
                capability_id=f"vit_s_fixed_shape.{semantic_key}.cuda",
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


def default_vit_s_weight_source_registry() -> VitSWeightSourceRegistry:
    """Build the default registry for ViT-S input weight sources."""

    registry = VitSWeightSourceRegistry()
    registry.register(
        VitSWeightSourceSpec(
            source_id=VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
            provider_id="local_numpy",
            description="Deterministic synthetic ViT-S weights and input image",
            default_config={"parameter_seed": 0},
            supports_pretrained=False,
        ),
        _load_vit_s_deterministic_weight_source,
    )
    registry.register(
        VitSWeightSourceSpec(
            source_id=VIT_S_WEIGHT_SOURCE_TIMM,
            provider_id="timm",
            description="Weights loaded from a timm VisionTransformer state_dict",
            default_config={
                "model_name": "vit_small_patch16_224",
                "pretrained": False,
                "checkpoint_path": None,
                "cache_dir": None,
                "hf_endpoint": VIT_S_DEFAULT_HF_ENDPOINT,
                "image_source": "synthetic_gradient",
            },
            supports_pretrained=True,
        ),
        _load_vit_s_timm_weight_source,
    )
    return registry


def vit_s_weight_source_registry() -> VitSWeightSourceRegistry:
    """Return the process-local ViT-S weight source registry."""

    global _VIT_S_WEIGHT_SOURCE_REGISTRY
    if _VIT_S_WEIGHT_SOURCE_REGISTRY is None:
        _VIT_S_WEIGHT_SOURCE_REGISTRY = default_vit_s_weight_source_registry()
    return _VIT_S_WEIGHT_SOURCE_REGISTRY


def register_vit_s_weight_source(
    spec: VitSWeightSourceSpec,
    loader: Callable[..., VitSWeightBundle],
) -> None:
    """Register a custom ViT-S weight source provider."""

    vit_s_weight_source_registry().register(spec, loader)


def load_vit_s_weight_source(
    source_id: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    **config: Any,
) -> VitSWeightBundle:
    """Materialize a configured ViT-S weight source."""

    return vit_s_weight_source_registry().load(source_id, **config)


def build_vit_s_fixed_shape_adapter(
    *, target: str = "cuda", producer_version: str | None = None
) -> ModelAdapter:
    """Materialize standard ViT-S/16 fixed-shape source routes without support claims."""

    operators = []
    for index, spec in enumerate(vit_s_fixed_shape_specs()):
        source = captured_inductor_jit_kernel(
            ttir=spec.ttir,
            function_name=spec.function_name,
            target=target,
            producer_version=producer_version,
        )
        operators.append(
            ModelOperatorInput(
                model_id=VIT_S_FIXED_SHAPE_MODEL_ID,
                operator_id=spec.operator_id,
                execution_order=index,
                source=source,
                ttir=spec.ttir,
                optional_model_anchor=VIT_S_RUNNER_ID,
            )
        )
    return ModelAdapter(VIT_S_FIXED_SHAPE_MODEL_ID, tuple(operators))


def run_vit_s_fixed_shape_route(
    *,
    target: str = "cuda",
    producer_version: str | None = None,
    out_dir: str | Path | None = None,
    enable_torch_compile: bool = False,
    enable_classification: bool = False,
    enable_e2e_benchmark: bool = False,
    enable_atomic_dag_gated_te_tir: bool = False,
    vit_s_backend: str = VIT_S_BACKEND_TRITON_TVM,
    torch_compile_device: str | None = None,
    torch_compile_rtol: float = 1.0e-3,
    torch_compile_atol: float = 1.0e-3,
    benchmark_warmup: int = 5,
    benchmark_repeat: int = 20,
    benchmark_device: str | None = None,
    enable_cuda_graph_replay: bool = False,
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    matmul_autotune_config: VitSMatmulAutotuneConfig | None = None,
    operator_autotune_config: VitSOperatorAutotuneConfig | None = None,
    autotune_config: AutotuneConfig | None = None,
) -> dict[str, Any]:
    """Compile the standard ViT-S/16 inventory through the active evidence route."""

    if vit_s_backend not in {VIT_S_BACKEND_REFERENCE, VIT_S_BACKEND_TRITON_TVM}:
        raise ValueError(f"unsupported ViT-S backend: {vit_s_backend}")

    resolved_matmul_autotune_config = _resolve_vit_s_matmul_autotune_config(
        target=target,
        config=matmul_autotune_config,
    )
    resolved_operator_autotune_config = _resolve_vit_s_operator_autotune_config(
        target=target,
        config=operator_autotune_config,
    )
    matmul_report_autotune_config = autotune_config or resolved_matmul_autotune_config
    operator_report_autotune_config = autotune_config or resolved_operator_autotune_config
    weight_bundle = _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    if autotune_config is not None and autotune_config.enabled and not _is_cuda_target(target):
        raise ValueError(f"ViT-S autotune requires a CUDA target, got {target}")
    vit_s_autotune_plan = (
        build_vit_s_autotune_plan(
            autotune_config=autotune_config,
            weight_bundle=weight_bundle,
        )
        if autotune_config is not None and autotune_config.enabled
        else None
    )
    adapter = build_vit_s_fixed_shape_adapter(
        target=target,
        producer_version=producer_version,
    )
    source_specs = {spec.operator_id: spec for spec in vit_s_fixed_shape_specs()}
    top_specs = {spec.operator_id: spec for spec in vit_s_fixed_shape_top_level_specs()}
    registry = default_vit_s_capability_registry()

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
        model_id=VIT_S_FIXED_SHAPE_MODEL_ID,
        runner_id=VIT_S_RUNNER_ID,
        top_specs=top_specs,
        sources_by_route=sources_by_route,
        atomics_by_route=atomics_by_route,
        validations_by_route=validations_by_route,
        semantics_by_route=semantics_by_route,
        registry=registry,
    )
    execution_requests = vit_execution_requests_from_top_records(top_records)
    buffer_plan = build_vit_s_buffer_plan(
        target=target,
        parameter_seed=parameter_seed,
        weight_source=weight_source,
        weight_source_config=weight_source_config,
        weight_bundle=weight_bundle,
    )
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
        "report_kind": "triton_tvm_vit_s_fixed_shape_runner",
        "runner_id": VIT_S_RUNNER_ID,
        "model_id": VIT_S_FIXED_SHAPE_MODEL_ID,
        "variant": "vit_s_16_224",
        "target": target,
        "fixed_shape": True,
        "vit_s_backend": vit_s_backend,
        "weight_source": _vit_s_weight_source_report(weight_bundle),
        "standard_vit_shape": {
            "batch": _VIT_S_BATCH,
            "image": _VIT_S_IMAGE,
            "patch": _VIT_S_PATCH,
            "patch_tokens": _VIT_S_PATCH_TOKENS,
            "tokens": _VIT_S_TOKENS,
            "hidden": _VIT_S_HIDDEN,
            "heads": _VIT_S_HEADS,
            "head_dim": _VIT_S_HEAD_DIM,
            "mlp": _VIT_S_MLP,
            "layers": _VIT_S_LAYERS,
            "classes": _VIT_S_CLASSES,
        },
        "compile_route": [
            "source_record",
            "atomic_dag",
            "contract_validation",
            "semantic_region",
            "manifest",
            "capability_registry",
            "runtime_admission",
            "torch_compile_comparison",
        ],
        "operator_count": len(top_records),
        "expected_operator_count": VIT_S_TOP_LEVEL_OPERATOR_COUNT,
        "top_level_operator_count": len(top_records),
        "source_route_record_count": len(source_route_records),
        "expected_source_route_record_count": VIT_S_SOURCE_ROUTE_RECORD_COUNT,
        "atomic_dag_built_operator_count": len(atomics_by_route),
        "semantic_region_count": len(top_semantics),
        "source_route_semantic_region_count": len(semantics_by_route),
        "runtime_admitted_operator_count": admitted_count,
        "gap_count": len(gaps),
        "status": (
            "passed"
            if admitted_count == VIT_S_TOP_LEVEL_OPERATOR_COUNT and not gaps else "gap"
        ),
        "performance_claim": False,
        "graph_optimization_used": False,
        "autotune_config_path": autotune_config.config_path if autotune_config else None,
        "autotune_config_hash": autotune_config.config_hash if autotune_config else None,
        **autotune_plan_report(vit_s_autotune_plan),
        "vit_s_matmul_codegen": {
            "source": VIT_S_MATMUL_CODEGEN_TE_AUTOTUNE,
            "te_workload": "original_matmul_bias_epilogue",
            "autotune_enabled": matmul_report_autotune_config.enabled,
            "autotune_work_dir": matmul_report_autotune_config.work_dir,
            "autotune_max_trials_global": (
                matmul_report_autotune_config.max_trials_global
            ),
            "autotune_max_trials_per_task": (
                matmul_report_autotune_config.max_trials_per_task
            ),
            "autotune_num_trials_per_iter": (
                matmul_report_autotune_config.num_trials_per_iter
            ),
            "cost_model": matmul_report_autotune_config.cost_model,
            "num_tuning_cores": matmul_report_autotune_config.num_tuning_cores,
            "task_scheduler": matmul_report_autotune_config.task_scheduler,
            "strategy": matmul_report_autotune_config.strategy,
            "seed": matmul_report_autotune_config.seed,
            "force_retune": matmul_report_autotune_config.force_retune,
        },
        "vit_s_operator_autotune": {
            "source": VIT_S_OPERATOR_CODEGEN_TE_AUTOTUNE,
            "te_workload": "original_operator_te_workloads",
            "autotune_enabled": operator_report_autotune_config.enabled,
            "autotune_work_dir": operator_report_autotune_config.work_dir,
            "autotune_max_trials_global": (
                operator_report_autotune_config.max_trials_global
            ),
            "autotune_max_trials_per_task": (
                operator_report_autotune_config.max_trials_per_task
            ),
            "autotune_num_trials_per_iter": (
                operator_report_autotune_config.num_trials_per_iter
            ),
            "cost_model": operator_report_autotune_config.cost_model,
            "num_tuning_cores": operator_report_autotune_config.num_tuning_cores,
            "task_scheduler": operator_report_autotune_config.task_scheduler,
            "strategy": operator_report_autotune_config.strategy,
            "seed": operator_report_autotune_config.seed,
            "force_retune": operator_report_autotune_config.force_retune,
        },
        "atomic_dag_lowering_coverage_claim": False,
        "runtime_claim": (
            RUNTIME_CLAIM_ATOMIC_DAG_GATED_TE
            if vit_s_backend == VIT_S_BACKEND_TRITON_TVM
            else "standard_vit_s_reference_correctness_only"
        ),
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
    backend_execution = None
    backend_outputs = None
    artifact_registry = None
    build_cache = TIRArtifactBuildCache()
    needs_tvm_backend = (
        vit_s_backend == VIT_S_BACKEND_TRITON_TVM
        and (
            enable_atomic_dag_gated_te_tir
            or enable_torch_compile
            or enable_classification
            or enable_e2e_benchmark
        )
    )
    if needs_tvm_backend:
        tir_artifacts = build_vit_s_atomic_dag_gated_te_tir_artifacts(
            execution_requests,
            atomics_by_route=atomics_by_route,
            target=target,
            weight_bundle=weight_bundle,
            matmul_autotune_config=resolved_matmul_autotune_config,
            operator_autotune_config=resolved_operator_autotune_config,
            autotune_config=autotune_config,
        )
        artifact_registry = TIRArtifactRegistry(tir_artifacts)
        report["vit_s_matmul_codegen"]["artifacts"] = _vit_s_matmul_codegen_summary(
            tir_artifacts
        )
        report["vit_s_operator_autotune"]["artifacts"] = _vit_s_operator_codegen_summary(
            tir_artifacts
        )
        backend_execution = run_vit_s_triton_tvm_backend(
            execution_requests,
            atomics_by_route=atomics_by_route,
            target=target,
            weight_bundle=weight_bundle,
            artifact_registry=artifact_registry,
            build_cache=build_cache,
            matmul_autotune_config=resolved_matmul_autotune_config,
            operator_autotune_config=resolved_operator_autotune_config,
            autotune_config=autotune_config,
        )
        backend_outputs = backend_execution["final_outputs"]
        report["tvm_backend_execution"] = _strip_runtime_arrays(backend_execution)
    if enable_atomic_dag_gated_te_tir:
        if artifact_registry is None:
            artifact_registry = TIRArtifactRegistry(
                build_vit_s_atomic_dag_gated_te_tir_artifacts(
                    execution_requests,
                    atomics_by_route=atomics_by_route,
                    target=target,
                    weight_bundle=weight_bundle,
                    matmul_autotune_config=resolved_matmul_autotune_config,
                    operator_autotune_config=resolved_operator_autotune_config,
                    autotune_config=autotune_config,
                )
            )
        reference_context = build_vit_s_reference_execution_context(
            target=target,
            weight_bundle=weight_bundle,
        )
        tvm_context = build_vit_s_tvm_execution_context(
            target=target,
            weight_bundle=weight_bundle,
        )
        diagnostic = run_reference_tvm_e2e(
            execution_requests,
            reference_executor=ReferenceTensorExecutor(
                vit_s_reference_operator_specs(
                    gelu_approximation=str(
                        weight_bundle.metadata.get("gelu_approximation", "tanh")
                    ),
                    layer_norm_epsilon=float(
                        weight_bundle.metadata.get("layer_norm_epsilon", 1.0e-5)
                    ),
                )
            ),
            tvm_executor=TvmPackedExecutor(
                artifact_registry=artifact_registry,
                build_cache=build_cache,
            ),
            reference_context=reference_context,
            tvm_context=tvm_context,
            source_route_record_count=len(source_route_records),
            semantic_proof_status_by_operator={
                record.operator_id: record.semantic_proof_status for record in top_records
            },
            proof_source_by_operator={
                record.operator_id: record.proof_source for record in top_records
            },
            tvm_operator_ids=tuple(request.operator_id for request in execution_requests),
            final_output_buffer_ids=buffer_plan.final_output_buffer_ids,
            runtime_claim=RUNTIME_CLAIM_ATOMIC_DAG_GATED_TE,
            e2e_gate_artifact_sources=(TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE,),
            rtol=torch_compile_rtol,
            atol=torch_compile_atol,
        )
        report["diagnostic_e2e"] = reference_tvm_e2e_result_to_dict(diagnostic)
        report["runtime_claim"] = RUNTIME_CLAIM_ATOMIC_DAG_GATED_TE
        report["e2e_pass"] = diagnostic.e2e_pass
        report["e2e_pass_alias_for"] = diagnostic.e2e_pass_alias_for
        report["e2e_correctness_pass"] = diagnostic.e2e_correctness_pass
        report["e2e_gate_kind"] = diagnostic.e2e_gate_kind
        report["atomic_dag_role"] = diagnostic.atomic_dag_role
        report["tir_generation_mode"] = diagnostic.tir_generation_mode
        report["dag_node_lowering_coverage_claim"] = (
            diagnostic.dag_node_lowering_coverage_claim
        )
    if enable_torch_compile:
        comparison = run_vit_s_torch_compile_comparison(
            device=torch_compile_device,
            rtol=torch_compile_rtol,
            atol=torch_compile_atol,
            parameter_seed=parameter_seed,
            weight_source=weight_source,
            weight_source_config=weight_source_config,
            weight_bundle=weight_bundle,
            reference_outputs=backend_outputs,
            reference_source=(
                "triton_tvm_atomic_dag_gated_te_tir"
                if backend_outputs is not None
                else "numpy_standard_vit_s"
            ),
        )
        report["torch_compile_comparison"] = comparison
        report["torch_compile_allclose"] = comparison["allclose"]
        report["model_comparison_status"] = comparison["status"]
        report["model_output_count"] = comparison["output_count"]
        report["model_allclose_count"] = comparison["allclose_count"]
    if enable_classification:
        classification = run_vit_s_classification_task(
            parameter_seed=parameter_seed,
            weight_source=weight_source,
            weight_source_config=weight_source_config,
            weight_bundle=weight_bundle,
            backend_outputs=backend_outputs,
            backend=(
                VIT_S_BACKEND_TRITON_TVM
                if backend_outputs is not None
                else VIT_S_BACKEND_REFERENCE
            ),
        )
        report["classification"] = classification
        report["classification_status"] = classification["status"]
        report["classification_top1_index"] = classification["top1"]["index"]
    if enable_e2e_benchmark:
        benchmark = run_vit_s_e2e_latency_benchmark(
            device=benchmark_device or torch_compile_device,
            warmup=benchmark_warmup,
            repeat=benchmark_repeat,
            parameter_seed=parameter_seed,
            weight_source=weight_source,
            weight_source_config=weight_source_config,
            weight_bundle=weight_bundle,
            vit_s_backend=vit_s_backend,
            execution_requests=execution_requests,
            atomics_by_route=atomics_by_route,
            artifact_registry=artifact_registry,
            build_cache=build_cache,
            matmul_autotune_config=resolved_matmul_autotune_config,
            operator_autotune_config=resolved_operator_autotune_config,
            autotune_config=autotune_config,
            enable_cuda_graph_replay=enable_cuda_graph_replay,
        )
        report["e2e_latency_benchmark"] = benchmark
        report["e2e_latency_benchmark_status"] = benchmark["status"]
    if out_dir is not None:
        write_json_report(Path(out_dir) / "report.json", report)
    return report


def vit_s_fixed_shape_specs() -> tuple[VitOperatorSpec, ...]:
    """Return the standard ViT-S/16 fixed-shape source/lowering route inventory."""

    top_specs = {spec.operator_id: spec for spec in vit_s_fixed_shape_top_level_specs()}

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

    records = [
        from_top("patch_embed", "patch_embed", "conv2d_nchw_static", "vit_s_patch_embed", _memory_ttir("vit_s_patch_embed"), "nchw_to_tokens"),
        from_top("class_token_concat", "class_token_concat", "pointwise_flat", "vit_s_class_token_concat", _memory_ttir("vit_s_class_token_concat"), "tokens_hidden_grid"),
        from_top("position_add", "position_add", "pointwise_flat", "vit_s_position_add", _memory_ttir("vit_s_position_add"), "tokens_hidden_grid"),
    ]
    for layer in range(_VIT_S_LAYERS):
        prefix = _vit_s_layer_prefix(layer)
        records.extend(
            [
                from_top(f"{prefix}_norm0", f"{prefix}_norm0", "pointwise_flat", f"vit_s_{prefix}_norm0", _memory_ttir(f"vit_s_{prefix}_norm0"), "row_major"),
                from_top(f"{prefix}_qkv_projection", f"{prefix}_qkv_projection", "matmul_bias_epilogue", f"vit_s_{prefix}_qkv_projection", _dot_store_ttir(f"vit_s_{prefix}_qkv_projection"), "row_major"),
                from_top(f"{prefix}_attention_qk", f"{prefix}_attention", "attention_decomposed", f"vit_s_{prefix}_attention_qk", _dot_store_ttir(f"vit_s_{prefix}_attention_qk"), "row_major"),
                from_top(f"{prefix}_attention_softmax", f"{prefix}_attention", "pointwise_flat", f"vit_s_{prefix}_attention_softmax", _memory_ttir(f"vit_s_{prefix}_attention_softmax"), "row_major"),
                from_top(f"{prefix}_attention_av", f"{prefix}_attention", "attention_decomposed", f"vit_s_{prefix}_attention_av", _dot_store_ttir(f"vit_s_{prefix}_attention_av"), "row_major"),
                from_top(f"{prefix}_attention_projection", f"{prefix}_attention_projection", "matmul_bias_epilogue", f"vit_s_{prefix}_attention_projection", _dot_store_ttir(f"vit_s_{prefix}_attention_projection"), "row_major"),
                from_top(f"{prefix}_residual_after_attention", f"{prefix}_residual_after_attention", "pointwise_flat", f"vit_s_{prefix}_residual_after_attention", _memory_ttir(f"vit_s_{prefix}_residual_after_attention"), "flat"),
                from_top(f"{prefix}_norm1", f"{prefix}_norm1", "pointwise_flat", f"vit_s_{prefix}_norm1", _memory_ttir(f"vit_s_{prefix}_norm1"), "row_major"),
                from_top(f"{prefix}_mlp_fc0", f"{prefix}_mlp_fc0", "matmul_bias_epilogue", f"vit_s_{prefix}_mlp_fc0", _dot_store_ttir(f"vit_s_{prefix}_mlp_fc0"), "row_major"),
                from_top(f"{prefix}_mlp_gelu", f"{prefix}_mlp_gelu", "pointwise_flat", f"vit_s_{prefix}_mlp_gelu", _memory_ttir(f"vit_s_{prefix}_mlp_gelu"), "flat"),
                from_top(f"{prefix}_mlp_fc1", f"{prefix}_mlp_fc1", "matmul_bias_epilogue", f"vit_s_{prefix}_mlp_fc1", _dot_store_ttir(f"vit_s_{prefix}_mlp_fc1"), "row_major"),
                from_top(f"{prefix}_residual_after_mlp", f"{prefix}_residual_after_mlp", "pointwise_flat", f"vit_s_{prefix}_residual_after_mlp", _memory_ttir(f"vit_s_{prefix}_residual_after_mlp"), "flat"),
            ]
        )
    records.extend(
        [
            from_top("final_norm", "final_norm", "pointwise_flat", "vit_s_final_norm", _memory_ttir("vit_s_final_norm"), "row_major"),
            from_top("classifier_head", "classifier_head", "matmul_bias_epilogue", "vit_s_classifier_head", _dot_store_ttir("vit_s_classifier_head"), "row_major"),
        ]
    )
    return tuple(records)


def vit_s_fixed_shape_top_level_specs() -> tuple[VitTopLevelOperatorSpec, ...]:
    """Return canonical top-level operators for standard ViT-S/16 at 224x224."""

    shape = _vit_s_shape_signature()
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

    specs: list[VitTopLevelOperatorSpec] = [
        spec("patch_embed", 0, "conv_patchify", ("conv2d_nchw_static",), LOWERING_CONTRACT_STATUS_SPECIALIZED, ("patch_embed",), ("image", "patch_embed.weight", "patch_embed.bias"), ("patch_embed.out",), "nchw_to_tokens", "conv_patchify", {"patch": _VIT_S_PATCH}),
        spec("class_token_concat", 1, "pointwise_grid2d", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE, ("class_token_concat",), ("patch_embed.out", "class_token"), ("class_token_concat.out",), "tokens_hidden_grid", "class_token_concat"),
        spec("position_add", 2, "pointwise_grid2d", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_FLATTENED_SURROGATE, ("position_add",), ("class_token_concat.out", "position_embed"), ("position_add.out",), "tokens_hidden_grid", "position_add"),
    ]
    order = 3
    previous = "position_add.out"
    for layer in range(_VIT_S_LAYERS):
        prefix = _vit_s_layer_prefix(layer)
        norm0 = f"{prefix}.norm0.out"
        qkv = f"{prefix}.qkv.out"
        attention = f"{prefix}.attention.out"
        attention_projection = f"{prefix}.attention_projection.out"
        residual_attention = f"{prefix}.residual_after_attention.out"
        norm1 = f"{prefix}.norm1.out"
        mlp_fc0 = f"{prefix}.mlp_fc0.out"
        mlp_gelu = f"{prefix}.mlp_gelu.out"
        mlp_fc1 = f"{prefix}.mlp_fc1.out"
        residual_mlp = f"{prefix}.out"
        specs.extend(
            [
                spec(f"{prefix}_norm0", order, "norm_row", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE, (f"{prefix}_norm0",), (previous, f"{prefix}.norm0.scale", f"{prefix}.norm0.bias"), (norm0,), "row_major", "norm_row", {"epsilon": 1.0e-5}),
                spec(f"{prefix}_qkv_projection", order + 1, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_qkv_projection",), (norm0, f"{prefix}.qkv.weight", f"{prefix}.qkv.bias"), (qkv,), "row_major", "matmul_bias"),
                spec(f"{prefix}_attention", order + 2, "attention_decomposed", ("attention_decomposed",), LOWERING_CONTRACT_STATUS_COMPOSITE, (f"{prefix}_attention_qk", f"{prefix}_attention_softmax", f"{prefix}_attention_av"), (qkv,), (attention,), "row_major", "attention_decomposed", {"hidden": _VIT_S_HIDDEN, "heads": _VIT_S_HEADS}),
                spec(f"{prefix}_attention_projection", order + 3, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_attention_projection",), (attention, f"{prefix}.attention_projection.weight", f"{prefix}.attention_projection.bias"), (attention_projection,), "row_major", "matmul_bias"),
                spec(f"{prefix}_residual_after_attention", order + 4, "pointwise_flat", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_residual_after_attention",), (previous, attention_projection), (residual_attention,), "flat", "residual_add"),
                spec(f"{prefix}_norm1", order + 5, "norm_row", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE, (f"{prefix}_norm1",), (residual_attention, f"{prefix}.norm1.scale", f"{prefix}.norm1.bias"), (norm1,), "row_major", "norm_row", {"epsilon": 1.0e-5}),
                spec(f"{prefix}_mlp_fc0", order + 6, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_mlp_fc0",), (norm1, f"{prefix}.mlp_fc0.weight", f"{prefix}.mlp_fc0.bias"), (mlp_fc0,), "row_major", "matmul_bias"),
                spec(f"{prefix}_mlp_gelu", order + 7, "pointwise_flat", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_mlp_gelu",), (mlp_fc0,), (mlp_gelu,), "flat", "gelu_tanh"),
                spec(f"{prefix}_mlp_fc1", order + 8, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_mlp_fc1",), (mlp_gelu, f"{prefix}.mlp_fc1.weight", f"{prefix}.mlp_fc1.bias"), (mlp_fc1,), "row_major", "matmul_bias"),
                spec(f"{prefix}_residual_after_mlp", order + 9, "pointwise_flat", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_DIRECT, (f"{prefix}_residual_after_mlp",), (residual_attention, mlp_fc1), (residual_mlp,), "flat", "residual_add"),
            ]
        )
        previous = residual_mlp
        order += 10
    specs.extend(
        [
            spec("final_norm", order, "norm_row", ("pointwise_flat",), LOWERING_CONTRACT_STATUS_TEMPORARY_SURROGATE, ("final_norm",), (previous, "final_norm.scale", "final_norm.bias"), ("last_hidden_state",), "row_major", "norm_row", {"epsilon": 1.0e-5}),
            spec("classifier_head", order + 1, "matmul", ("matmul_bias_epilogue",), LOWERING_CONTRACT_STATUS_DIRECT, ("classifier_head",), ("last_hidden_state", "head.weight", "head.bias"), ("logits",), "row_major", "classifier_head"),
        ]
    )
    return tuple(specs)


def build_vit_s_buffer_plan(
    *,
    target: str = "cuda",
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    weight_bundle: VitSWeightBundle | None = None,
) -> RuntimeBufferPlan:
    """Build a connected fixed-shape standard ViT-S buffer plan."""

    bundle = weight_bundle or _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    values = bundle.values
    specs = vit_s_fixed_shape_top_level_specs()
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

    add_buffer("image", (_VIT_S_BATCH, _VIT_S_CHANNELS, _VIT_S_IMAGE, _VIT_S_IMAGE), producer=None, consumers=("patch_embed",), is_model_input=True)
    for buffer_id, shape, consumers in _vit_s_parameter_buffer_specs():
        add_buffer(buffer_id, shape, producer=None, consumers=consumers, is_parameter=True)
    for top_spec in specs:
        for output_buffer_id in top_spec.output_buffer_ids:
            add_buffer(
                output_buffer_id,
                _vit_s_buffer_shape(output_buffer_id),
                producer=top_spec.operator_id,
                consumers=_vit_s_consumers_for_buffer(output_buffer_id, specs),
                is_model_output=output_buffer_id in {"last_hidden_state", "logits"},
            )
    return RuntimeBufferPlan(
        buffers=buffers,
        edges=edges,
        final_output_buffer_ids=("last_hidden_state", "logits"),
    )


def build_vit_s_reference_execution_context(
    *,
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    weight_bundle: VitSWeightBundle | None = None,
    target: str = "cuda",
) -> ExecutionContext:
    """Create a host reference context containing ViT-S inputs and parameters."""

    bundle = weight_bundle or _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    return ExecutionContext(
        buffer_table={buffer_id: value.copy() for buffer_id, value in bundle.values.items()},
        target=target,
    )


def build_vit_s_tvm_execution_context(
    *,
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    weight_bundle: VitSWeightBundle | None = None,
    target: str = "cuda",
) -> ExecutionContext:
    """Create a TVM device context containing ViT-S inputs and parameters."""

    import tvm  # pylint: disable=import-outside-toplevel

    bundle = weight_bundle or _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    device = tvm.cuda(0) if target.startswith("cuda") else tvm.cpu(0)
    return ExecutionContext(
        buffer_table={
            buffer_id: tvm.runtime.tensor(value.copy(), device)
            for buffer_id, value in bundle.values.items()
        },
        target=target,
    )


def vit_s_reference_operator_specs(
    *,
    gelu_approximation: str = "tanh",
    layer_norm_epsilon: float = 1.0e-5,
) -> tuple[ReferenceOperatorSpec, ...]:
    """Reference specs for the 125 top-level standard ViT-S operators."""

    specs = []
    gelu_impl = "gelu_exact" if gelu_approximation == "none" else "gelu_tanh"
    for spec in vit_s_fixed_shape_top_level_specs():
        attrs = dict(spec.reference_attrs)
        impl_id = spec.reference_impl_id
        if impl_id == "norm_row":
            attrs["epsilon"] = layer_norm_epsilon
        elif impl_id == "gelu_tanh":
            impl_id = gelu_impl
        specs.append(
            ReferenceOperatorSpec(
                operator_id=spec.operator_id,
                semantic_region_key=spec.semantic_region_key,
                input_buffer_ids=spec.input_buffer_ids,
                output_buffer_ids=spec.output_buffer_ids,
                attrs=attrs,
                reference_impl_id=impl_id,
            )
        )
    return tuple(specs)


def build_vit_s_autotune_plan(
    *,
    autotune_config: AutotuneConfig,
    weight_bundle: VitSWeightBundle | None = None,
) -> AutotunePlan:
    """Build a deduplicated ViT-S workload plan from selected top-level operators."""

    gelu_approximation = (
        str(weight_bundle.metadata.get("gelu_approximation", "tanh"))
        if weight_bundle is not None
        else "tanh"
    )
    layer_norm_epsilon = (
        float(weight_bundle.metadata.get("layer_norm_epsilon", 1.0e-5))
        if weight_bundle is not None
        else 1.0e-5
    )
    available = tuple(spec.operator_id for spec in vit_s_fixed_shape_top_level_specs())
    return build_autotune_plan(
        autotune_config.operator_ids,
        available_operator_ids=available,
        workload_resolver=lambda operator_id: _vit_s_autotune_workload_specs(
            operator_id,
            gelu_approximation=gelu_approximation,
            layer_norm_epsilon=layer_norm_epsilon,
        ),
    )


def build_vit_s_atomic_dag_gated_te_tir_artifacts(
    requests: Iterable[OperatorExecutionRequest],
    *,
    atomics_by_route: Mapping[str, Any],
    target: str = "cuda",
    weight_bundle: VitSWeightBundle | None = None,
    operator_ids: Iterable[str] | None = None,
    matmul_autotune_config: VitSMatmulAutotuneConfig | None = None,
    operator_autotune_config: VitSOperatorAutotuneConfig | None = None,
    autotune_config: AutotuneConfig | None = None,
) -> tuple[TIRRegionArtifact, ...]:
    """Generate ViT-S TE/TIR artifacts admitted by validated Atomic DAG evidence."""

    request_list = tuple(requests)
    selected = set(operator_ids) if operator_ids is not None else None
    return build_vit_s_tir_artifacts(
        request_list,
        target=target,
        artifact_sources_by_operator={
            request.operator_id: TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE
            for request in request_list
            if selected is None or request.operator_id in selected
        },
        atomics_by_route=atomics_by_route,
        weight_bundle=weight_bundle,
        matmul_autotune_config=matmul_autotune_config,
        operator_autotune_config=operator_autotune_config,
        autotune_config=autotune_config,
    )


def build_vit_s_tir_artifacts(
    requests: Iterable[OperatorExecutionRequest],
    *,
    target: str = "cuda",
    artifact_sources_by_operator: Mapping[str, str],
    atomics_by_route: Mapping[str, Any],
    weight_bundle: VitSWeightBundle | None = None,
    matmul_autotune_config: VitSMatmulAutotuneConfig | None = None,
    operator_autotune_config: VitSOperatorAutotuneConfig | None = None,
    autotune_config: AutotuneConfig | None = None,
) -> tuple[TIRRegionArtifact, ...]:
    """Build top-level ViT-S TIR artifacts for the requested artifact sources."""

    buffer_plan = build_vit_s_buffer_plan(target=target, weight_bundle=weight_bundle)
    buffer_meta = {
        buffer_id: {"shape": buffer.shape, "dtype": buffer.dtype, "layout": buffer.layout}
        for buffer_id, buffer in buffer_plan.buffers.items()
    }
    gelu_approximation = (
        str(weight_bundle.metadata.get("gelu_approximation", "tanh"))
        if weight_bundle is not None
        else "tanh"
    )
    layer_norm_epsilon = (
        float(weight_bundle.metadata.get("layer_norm_epsilon", 1.0e-5))
        if weight_bundle is not None
        else 1.0e-5
    )
    matmul_codegen_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
    operator_codegen_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
    selected_autotune_operator_ids = (
        set(autotune_config.operator_ids)
        if autotune_config is not None and autotune_config.enabled
        else None
    )
    if autotune_config is not None and autotune_config.enabled:
        build_vit_s_autotune_plan(
            autotune_config=autotune_config,
            weight_bundle=weight_bundle,
        )
    resolved_operator_autotune_config = _resolve_vit_s_operator_autotune_config(
        target=target,
        config=operator_autotune_config,
    )
    artifacts = []
    for request in sorted(requests, key=lambda item: item.execution_order):
        artifact_source = artifact_sources_by_operator.get(request.operator_id)
        if artifact_source is None:
            continue
        if artifact_source != TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE:
            raise ValueError("ViT-S TIR artifacts currently require atomic_dag_gated_te_tir")
        generation_meta = _tir_generation_meta(
            request,
            artifact_source=artifact_source,
            target=target,
            atomics_by_route=atomics_by_route,
        )
        extra_buffer_meta = {}
        operator_codegen_records: tuple[dict[str, Any], ...] = ()
        if request.semantic_region_key == "attention_decomposed":
            (
                tir_module,
                lowering_plan,
                extra_buffer_meta,
                operator_codegen_records,
            ) = _vit_s_attention_tir_module_and_plan(
                request,
                name_prefix=_primfunc_prefix_for_artifact_source(artifact_source),
                artifact_source=artifact_source,
                target=target,
                operator_autotune_config=_effective_vit_s_operator_autotune_config(
                    target=target,
                    operator_id=request.operator_id,
                    legacy_config=resolved_operator_autotune_config,
                    autotune_config=autotune_config,
                    selected_operator_ids=selected_autotune_operator_ids,
                ),
                operator_codegen_cache=operator_codegen_cache,
            )
            matmul_codegen = None
        else:
            tir_module, primfunc_name, codegen_record = _vit_s_tir_module_for_operator(
                request.operator_id,
                name_prefix=_primfunc_prefix_for_artifact_source(artifact_source),
                target=target,
                gelu_approximation=gelu_approximation,
                layer_norm_epsilon=layer_norm_epsilon,
                matmul_autotune_config=_effective_vit_s_matmul_autotune_config(
                    target=target,
                    operator_id=request.operator_id,
                    legacy_config=matmul_autotune_config,
                    autotune_config=autotune_config,
                    selected_operator_ids=selected_autotune_operator_ids,
                ),
                matmul_codegen_cache=matmul_codegen_cache,
                operator_autotune_config=_effective_vit_s_operator_autotune_config(
                    target=target,
                    operator_id=request.operator_id,
                    legacy_config=resolved_operator_autotune_config,
                    autotune_config=autotune_config,
                    selected_operator_ids=selected_autotune_operator_ids,
                ),
                operator_codegen_cache=operator_codegen_cache,
            )
            matmul_codegen = (
                codegen_record
                if codegen_record is not None
                and codegen_record.get("operator_family") == "matmul"
                else None
            )
            if codegen_record is not None:
                operator_codegen_records = (codegen_record,)
            lowering_plan = (
                LoweringStep(
                    name=f"{request.operator_id}.{artifact_source}",
                    contract_id=request.lowering_contract_ids[0],
                    primfunc_name=primfunc_name,
                    input_buffer_ids=request.input_buffer_ids,
                    output_buffer_ids=request.output_buffer_ids,
                    intermediate_buffer_ids=(),
                    arg_buffer_order=request.input_buffer_ids + request.output_buffer_ids,
                    attrs={"artifact_source": artifact_source},
                ),
            )
        if not operator_codegen_records:
            operator_codegen_records = _vit_s_baseline_codegen_records(
                request.operator_id,
                gelu_approximation=gelu_approximation,
                layer_norm_epsilon=layer_norm_epsilon,
            )
        artifact_buffer_meta = dict(buffer_meta)
        artifact_buffer_meta.update(extra_buffer_meta)
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
            "buffers": artifact_buffer_meta,
            "gelu_approximation": gelu_approximation,
            "layer_norm_epsilon": layer_norm_epsilon,
        }
        if matmul_codegen is not None:
            region_meta["matmul_codegen"] = matmul_codegen
        if operator_codegen_records:
            region_meta["operator_codegen"] = operator_codegen_records
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
                lowering_plan=lowering_plan,
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
                        "gelu_approximation": gelu_approximation,
                        "layer_norm_epsilon": layer_norm_epsilon,
                        "matmul_codegen_id": (
                            matmul_codegen["codegen_id"]
                            if matmul_codegen is not None
                            else None
                        ),
                        "operator_codegen_ids": tuple(
                            record["codegen_id"] for record in operator_codegen_records
                        ),
                    }
                ),
            )
        )
    return tuple(artifacts)


def run_vit_s_triton_tvm_backend(
    execution_requests: Iterable[OperatorExecutionRequest],
    *,
    atomics_by_route: Mapping[str, Any],
    target: str = "cuda",
    weight_bundle: VitSWeightBundle,
    artifact_registry: TIRArtifactRegistry | None = None,
    build_cache: TIRArtifactBuildCache | None = None,
    context: ExecutionContext | None = None,
    session: TvmPackedExecutionSession | None = None,
    collect_operator_records: bool = True,
    read_final_outputs: bool = True,
    matmul_autotune_config: VitSMatmulAutotuneConfig | None = None,
    operator_autotune_config: VitSOperatorAutotuneConfig | None = None,
    autotune_config: AutotuneConfig | None = None,
) -> dict[str, Any]:
    """Execute ViT-S through the Triton-TVM packed executor."""

    request_list = tuple(sorted(execution_requests, key=lambda item: item.execution_order))
    registry = artifact_registry or TIRArtifactRegistry(
        build_vit_s_atomic_dag_gated_te_tir_artifacts(
            request_list,
            atomics_by_route=atomics_by_route,
            target=target,
            weight_bundle=weight_bundle,
            matmul_autotune_config=_resolve_vit_s_matmul_autotune_config(
                target=target,
                config=matmul_autotune_config,
            ),
            operator_autotune_config=_resolve_vit_s_operator_autotune_config(
                target=target,
                config=operator_autotune_config,
            ),
            autotune_config=autotune_config,
        )
    )
    cache = build_cache or TIRArtifactBuildCache()
    executor = TvmPackedExecutor(artifact_registry=registry, build_cache=cache)
    execution_context = context or build_vit_s_tvm_execution_context(
        target=target,
        weight_bundle=weight_bundle,
    )
    execution_session = session or executor.create_session(request_list, execution_context)
    results = execution_session.run(collect_operator_records=collect_operator_records)
    operator_records = [
        {
            "operator_id": result.operator_id,
            "execution_order": request.execution_order,
            "status": result.status,
            "backend_id": result.backend_id,
            "reason": result.reason,
            "elapsed_ms": result.elapsed_ms,
            "artifact_source": result.artifact_source,
            "correctness_readback": result.correctness_readback,
            "output_buffer_ids": result.output_buffer_ids,
        }
        for request, result in zip(request_list, results)
    ]
    if not collect_operator_records:
        operator_records = ()
    tvm_executed_count = sum(1 for record in operator_records if record["status"] == "executed")
    failed_count = sum(1 for record in operator_records if record["status"] == "failed")
    gap_count = sum(1 for record in operator_records if record["status"] == "gap")
    artifact_breakdown: dict[str, int] = {}
    for record in operator_records:
        source = record["artifact_source"]
        if source:
            artifact_breakdown[source] = artifact_breakdown.get(source, 0) + 1
    if not collect_operator_records:
        tvm_executed_count = len(request_list)
        artifact_breakdown = dict(execution_session.artifact_source_breakdown)
    final_outputs = {}
    if read_final_outputs:
        final_outputs = {
            buffer_id: _context_buffer_to_numpy(execution_context, buffer_id)
            for buffer_id in ("last_hidden_state", "logits")
            if buffer_id in execution_context.buffer_table
        }
    return {
        "status": "executed" if tvm_executed_count == len(request_list) else "failed",
        "backend_id": "tvm_packed",
        "target": target,
        "tvm_executed_count": tvm_executed_count,
        "planned_operator_count": len(request_list),
        "failed_count": failed_count,
        "gap_count": gap_count,
        "artifact_source_breakdown": dict(sorted(artifact_breakdown.items())),
        "build_cache": {
            "build_count": cache.build_count,
            "hit_count": cache.hit_count,
        },
        "final_outputs": final_outputs,
        "operators": tuple(operator_records),
        "performance_claim": False,
        "runtime_session": {
            "enabled": True,
            "collect_operator_records": collect_operator_records,
            "bound_operator_count": len(execution_session.operators),
        },
    }


def _context_buffer_to_numpy(context: ExecutionContext, buffer_id: str) -> Any:
    value = context.buffer_table[buffer_id]
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _strip_runtime_arrays(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "final_outputs"}


def run_vit_s_torch_compile_comparison(
    *,
    device: str | None = None,
    rtol: float = 1.0e-3,
    atol: float = 1.0e-3,
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    weight_bundle: VitSWeightBundle | None = None,
    reference_outputs: Mapping[str, Any] | None = None,
    reference_source: str = "numpy_standard_vit_s",
) -> dict[str, Any]:
    """Compare a ViT-S reference/backend path with a torch.compile model."""

    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = weight_bundle or _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    if reference_outputs is None:
        reference_outputs = _execute_vit_s_numpy(
            bundle.values,
            gelu_approximation=str(bundle.metadata.get("gelu_approximation", "tanh")),
            layer_norm_epsilon=float(bundle.metadata.get("layer_norm_epsilon", 1.0e-5)),
        )
    model, image, compiled_source = _build_vit_s_torch_model_and_image(
        bundle,
        selected_device=selected_device,
    )

    old_matmul_tf32, old_cudnn_tf32 = _configure_torch_float32_precision(
        torch,
        selected_device=selected_device,
    )
    try:
        with torch.no_grad():
            compiled = torch.compile(model)
            torch_last_hidden_state, torch_logits = compiled(image)
    finally:
        _restore_torch_float32_precision(torch, old_matmul_tf32, old_cudnn_tf32)

    torch_outputs = {
        "last_hidden_state": torch_last_hidden_state.detach().cpu().numpy(),
        "logits": torch_logits.detach().cpu().numpy(),
    }
    output_records = []
    for buffer_id, torch_value in torch_outputs.items():
        reference = reference_outputs[buffer_id]
        diff = np.abs(reference - torch_value)
        rel = diff / (np.abs(reference) + 1.0e-12)
        allclose = bool(np.allclose(reference, torch_value, rtol=rtol, atol=atol))
        output_records.append(
            {
                "buffer_id": buffer_id,
                "comparison_status": "allclose_passed" if allclose else "allclose_failed",
                "allclose": allclose,
                "shape": tuple(int(dim) for dim in reference.shape),
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": float(diff.max()) if diff.size else 0.0,
                "max_rel_error": float(rel.max()) if rel.size else 0.0,
            }
        )
    allclose_count = sum(1 for record in output_records if record["allclose"])
    return {
        "status": "allclose" if allclose_count == len(output_records) else "failed",
        "allclose": allclose_count == len(output_records),
        "output_count": len(output_records),
        "allclose_count": allclose_count,
        "device": selected_device,
        "weight_source": _vit_s_weight_source_report(bundle),
        "torch_version": torch.__version__,
        "torch_compile_used": True,
        "reference_source": reference_source,
        "compiled_source": compiled_source,
        "performance_claim": False,
        "outputs": tuple(output_records),
    }


def run_vit_s_e2e_latency_benchmark(
    *,
    device: str | None = None,
    warmup: int = 5,
    repeat: int = 20,
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    weight_bundle: VitSWeightBundle | None = None,
    vit_s_backend: str = VIT_S_BACKEND_REFERENCE,
    execution_requests: Iterable[OperatorExecutionRequest] | None = None,
    atomics_by_route: Mapping[str, Any] | None = None,
    artifact_registry: TIRArtifactRegistry | None = None,
    build_cache: TIRArtifactBuildCache | None = None,
    matmul_autotune_config: VitSMatmulAutotuneConfig | None = None,
    operator_autotune_config: VitSOperatorAutotuneConfig | None = None,
    autotune_config: AutotuneConfig | None = None,
    enable_cuda_graph_replay: bool = False,
) -> dict[str, Any]:
    """Measure local E2E forward latency for ViT-S backend and torch.compile."""

    import platform  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel

    if warmup < 0:
        raise ValueError("benchmark warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("benchmark repeat must be positive")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    selected_device_name = None
    if selected_device.startswith("cuda") and torch.cuda.is_available():
        selected_device_name = torch.cuda.get_device_name(torch.device(selected_device))
    bundle = weight_bundle or _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    backend_first_call_ms = None
    backend_session_init_ms = None
    backend_cuda_graph_capture_ms = None
    backend_cuda_graph_enabled = False
    backend_cuda_graph_state: dict[str, Any] | None = None
    backend_cuda_graph_requested = bool(enable_cuda_graph_replay)
    backend_samples_ms = []
    backend_execution_mode = "numpy_standard_vit_s"
    backend_path_key = "numpy_reference"
    if vit_s_backend == VIT_S_BACKEND_TRITON_TVM:
        if execution_requests is None or atomics_by_route is None:
            raise ValueError("triton-tvm benchmark requires execution requests and Atomic DAGs")
        request_list = tuple(execution_requests)
        resolved_matmul_autotune_config = _resolve_vit_s_matmul_autotune_config(
            target=selected_device,
            config=matmul_autotune_config,
        )
        resolved_operator_autotune_config = _resolve_vit_s_operator_autotune_config(
            target=selected_device,
            config=operator_autotune_config,
        )
        registry = artifact_registry or TIRArtifactRegistry(
            build_vit_s_atomic_dag_gated_te_tir_artifacts(
                request_list,
                atomics_by_route=atomics_by_route,
                target=selected_device,
                weight_bundle=bundle,
                matmul_autotune_config=resolved_matmul_autotune_config,
                operator_autotune_config=resolved_operator_autotune_config,
                autotune_config=autotune_config,
            )
        )
        cache = build_cache or TIRArtifactBuildCache()
        backend_execution_mode = "triton_tvm_atomic_dag_gated_te_tir"
        backend_path_key = "triton_tvm"
        persistent_context = build_vit_s_tvm_execution_context(
            target=selected_device,
            weight_bundle=bundle,
        )
        executor = TvmPackedExecutor(artifact_registry=registry, build_cache=cache)
        _synchronize_tvm_target(selected_device)
        session_init_start = time.perf_counter()
        backend_session = executor.create_session(request_list, persistent_context)
        backend_session.synchronize()
        backend_session_init_ms = (time.perf_counter() - session_init_start) * 1000.0

        backend_session.synchronize()
        start = time.perf_counter()
        backend_session.run(collect_operator_records=False)
        backend_session.synchronize()
        backend_first_call_ms = (time.perf_counter() - start) * 1000.0
        if enable_cuda_graph_replay:
            backend_session.synchronize()
            capture_start = time.perf_counter()
            backend_cuda_graph_enabled = backend_session.capture_cuda_graph()
            backend_session.synchronize()
            backend_cuda_graph_capture_ms = (time.perf_counter() - capture_start) * 1000.0
            backend_cuda_graph_state = backend_session.cuda_graph_state()
        else:
            backend_cuda_graph_state = backend_session.cuda_graph_state()
        for _ in range(warmup):
            backend_session.run(
                collect_operator_records=False,
                use_cuda_graph=backend_cuda_graph_enabled,
            )
        backend_session.synchronize()
        for _ in range(repeat):
            backend_session.synchronize()
            start = time.perf_counter()
            backend_session.run(
                collect_operator_records=False,
                use_cuda_graph=backend_cuda_graph_enabled,
            )
            backend_session.synchronize()
            backend_samples_ms.append((time.perf_counter() - start) * 1000.0)
        backend_cuda_graph_state = backend_session.cuda_graph_state()
    else:
        gelu_approximation = str(bundle.metadata.get("gelu_approximation", "tanh"))
        layer_norm_epsilon = float(bundle.metadata.get("layer_norm_epsilon", 1.0e-5))
        for _ in range(warmup):
            _execute_vit_s_numpy(
                bundle.values,
                gelu_approximation=gelu_approximation,
                layer_norm_epsilon=layer_norm_epsilon,
            )
        for _ in range(repeat):
            start = time.perf_counter()
            _execute_vit_s_numpy(
                bundle.values,
                gelu_approximation=gelu_approximation,
                layer_norm_epsilon=layer_norm_epsilon,
            )
            backend_samples_ms.append((time.perf_counter() - start) * 1000.0)

    model, image, compiled_source = _build_vit_s_torch_model_and_image(
        bundle,
        selected_device=selected_device,
    )
    old_matmul_tf32, old_cudnn_tf32 = _configure_torch_float32_precision(
        torch,
        selected_device=selected_device,
    )
    first_call_ms = None
    torch_samples_ms = []
    try:
        with torch.no_grad():
            compiled = torch.compile(model)
            _synchronize_torch_device(torch, selected_device)
            start = time.perf_counter()
            compiled(image)
            _synchronize_torch_device(torch, selected_device)
            first_call_ms = (time.perf_counter() - start) * 1000.0
            for _ in range(warmup):
                compiled(image)
            _synchronize_torch_device(torch, selected_device)
            for _ in range(repeat):
                _synchronize_torch_device(torch, selected_device)
                start = time.perf_counter()
                compiled(image)
                _synchronize_torch_device(torch, selected_device)
                torch_samples_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        _restore_torch_float32_precision(torch, old_matmul_tf32, old_cudnn_tf32)

    backend_summary = _latency_sample_summary(backend_samples_ms)
    torch_summary = _latency_sample_summary(torch_samples_ms)
    speedup = (
        backend_summary["median_ms"] / torch_summary["median_ms"]
        if torch_summary["median_ms"] > 0.0
        else None
    )
    return {
        "status": "benchmarked",
        "benchmark_kind": "vit_s_e2e_forward_latency",
        "variant": "vit_s_16_224",
        "device": selected_device,
        "device_name": selected_device_name,
        "weight_source": _vit_s_weight_source_report(bundle),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "performance_claim": False,
        "result_scope": "local_measurement_only",
        "statistical_method": {
            "latency_unit": "milliseconds",
            "timed_runs": repeat,
            "warmup_runs_per_path": warmup,
            "weight_loading_excluded": True,
            "backend_first_call_excluded_from_latency": backend_first_call_ms is not None,
            "backend_first_call_ms": backend_first_call_ms,
            "backend_session_init_excluded_from_latency": backend_session_init_ms is not None,
            "backend_session_init_ms": backend_session_init_ms,
            "backend_session_runtime_fast_path": (
                "precompiled_prevalidated_bound_args_no_per_op_records"
                if backend_session_init_ms is not None
                else None
            ),
            "backend_cuda_graph_replay_requested": backend_cuda_graph_requested,
            "backend_cuda_graph_replay_enabled": backend_cuda_graph_enabled,
            "backend_cuda_graph_capture_excluded_from_latency": (
                backend_cuda_graph_capture_ms is not None
            ),
            "backend_cuda_graph_capture_ms": backend_cuda_graph_capture_ms,
            "backend_cuda_graph_replay_state": backend_cuda_graph_state,
            "torch_compile_first_call_excluded_from_latency": True,
            "torch_compile_first_call_ms": first_call_ms,
            "torch_compile_warmup_runs_after_first_call": warmup,
            "measurement_scope": (
                "single batch=1 forward returning last_hidden_state and logits"
            ),
            "timer": "time.perf_counter",
            "gpu_synchronization": (
                "device synchronize before and after timed GPU backend calls"
                if selected_device.startswith("cuda") and torch.cuda.is_available()
                else "not applicable"
            ),
            "percentile_method": "linear_interpolation_on_sorted_samples",
            "stddev_kind": "sample_stddev",
            "tf32_enabled_during_torch_timing": False,
        },
        "paths": {
            backend_path_key: {
                "execution_mode": backend_execution_mode,
                "summary": backend_summary,
                "samples_ms": tuple(backend_samples_ms),
                "cuda_graph_replay": backend_cuda_graph_state,
            },
            "torch_compile": {
                "execution_mode": compiled_source,
                "summary": torch_summary,
                "samples_ms": tuple(torch_samples_ms),
            },
        },
        "torch_compile_vs_backend_median_speedup": speedup,
        "torch_compile_vs_numpy_median_speedup": speedup
        if backend_path_key == "numpy_reference"
        else None,
    }


def _build_vit_s_torch_model_and_image(
    bundle: VitSWeightBundle,
    *,
    selected_device: str,
) -> tuple[Any, Any, str]:
    import torch  # pylint: disable=import-outside-toplevel

    if bundle.torch_model is not None:
        model = _TimmVitSFeatureAndLogits(bundle.torch_model).eval().to(selected_device)
        compiled_source = "torch.compile(timm)"
    else:
        model = _build_torch_vit_s_model(bundle.values).eval().to(selected_device)
        compiled_source = "torch.compile(internal_torch_vit_s)"
    image = torch.from_numpy(bundle.values["image"].copy()).to(selected_device)
    return model, image, compiled_source


def _configure_torch_float32_precision(
    torch_module: Any,
    *,
    selected_device: str,
) -> tuple[Any | None, Any | None]:
    old_matmul_tf32 = None
    old_cudnn_tf32 = None
    if selected_device.startswith("cuda") and torch_module.cuda.is_available():
        old_matmul_tf32 = torch_module.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch_module.backends.cudnn.allow_tf32
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False
        torch_module.set_float32_matmul_precision("highest")
    return old_matmul_tf32, old_cudnn_tf32


def _restore_torch_float32_precision(
    torch_module: Any,
    old_matmul_tf32: Any | None,
    old_cudnn_tf32: Any | None,
) -> None:
    if old_matmul_tf32 is not None:
        torch_module.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    if old_cudnn_tf32 is not None:
        torch_module.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _synchronize_torch_device(torch_module: Any, selected_device: str) -> None:
    if selected_device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _synchronize_tvm_target(target: str) -> None:
    if not target.startswith("cuda"):
        return
    try:
        import tvm  # pylint: disable=import-outside-toplevel

        device = tvm.cuda(0)
        if hasattr(device, "sync"):
            device.sync()
    except Exception:
        return


def _latency_sample_summary(samples_ms: Iterable[float]) -> dict[str, float | int]:
    import statistics  # pylint: disable=import-outside-toplevel

    samples = sorted(float(sample) for sample in samples_ms)
    if not samples:
        raise ValueError("latency sample summary requires at least one sample")
    return {
        "count": len(samples),
        "min_ms": samples[0],
        "mean_ms": statistics.fmean(samples),
        "median_ms": _latency_percentile(samples, 50.0),
        "p90_ms": _latency_percentile(samples, 90.0),
        "p95_ms": _latency_percentile(samples, 95.0),
        "p99_ms": _latency_percentile(samples, 99.0),
        "max_ms": samples[-1],
        "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _latency_percentile(sorted_samples: list[float], percentile: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (percentile / 100.0) * (len(sorted_samples) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_samples[lower]
    weight = rank - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


def run_vit_s_classification_task(
    *,
    topk: int = 5,
    parameter_seed: int = 0,
    weight_source: str = VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
    weight_source_config: Mapping[str, Any] | None = None,
    weight_bundle: VitSWeightBundle | None = None,
    backend_outputs: Mapping[str, Any] | None = None,
    backend: str = VIT_S_BACKEND_REFERENCE,
) -> dict[str, Any]:
    """Run a simple ViT-S classification task from the configured weight source."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    bundle = weight_bundle or _resolve_vit_s_weight_bundle(
        weight_source=weight_source,
        parameter_seed=parameter_seed,
        weight_source_config=weight_source_config,
    )
    buffers = backend_outputs or _execute_vit_s_numpy(
        bundle.values,
        gelu_approximation=str(bundle.metadata.get("gelu_approximation", "tanh")),
        layer_norm_epsilon=float(bundle.metadata.get("layer_norm_epsilon", 1.0e-5)),
    )
    logits = buffers["logits"][0]
    k = max(1, min(int(topk), logits.shape[-1]))
    indices = np.argsort(logits)[-k:][::-1]
    probabilities = _softmax_numpy(logits[indices])
    records = []
    for rank, (index, score, probability) in enumerate(
        zip(indices, logits[indices], probabilities), start=1
    ):
        label_name, description = _imagenet_label(index)
        records.append(
            {
                "rank": rank,
                "index": int(index),
                "label_name": label_name,
                "description": description,
                "logit": float(score),
                "topk_probability": float(probability),
            }
        )
    return {
        "status": "classified",
        "backend": backend,
        "weight_source": _vit_s_weight_source_report(bundle),
        "input_source": bundle.metadata.get("image_source", "synthetic"),
        "top1": records[0],
        "topk": tuple(records),
        "logits_shape": tuple(int(dim) for dim in buffers["logits"].shape),
        "performance_claim": False,
    }


def _resolve_vit_s_weight_bundle(
    *,
    weight_source: str,
    parameter_seed: int,
    weight_source_config: Mapping[str, Any] | None,
) -> VitSWeightBundle:
    config = dict(weight_source_config or {})
    config.setdefault("parameter_seed", parameter_seed)
    return load_vit_s_weight_source(weight_source, **config)


def _vit_s_weight_source_report(bundle: VitSWeightBundle) -> dict[str, Any]:
    return {
        "source_id": bundle.source_id,
        "provider_id": bundle.provider_id,
        "metadata": dict(bundle.metadata),
    }


def _load_vit_s_deterministic_weight_source(
    *, parameter_seed: int = 0, **_: Any
) -> VitSWeightBundle:
    values = _vit_s_random_buffer_values(parameter_seed=parameter_seed)
    return VitSWeightBundle(
        source_id=VIT_S_WEIGHT_SOURCE_DETERMINISTIC,
        provider_id="local_numpy",
        values=values,
        metadata={
            "parameter_seed": parameter_seed,
            "pretrained": False,
            "image_source": "deterministic_random",
            "class_names": "imagenet_1k",
            "gelu_approximation": "tanh",
            "layer_norm_epsilon": 1.0e-5,
        },
    )


def _load_vit_s_timm_weight_source(
    *,
    model_name: str = "vit_small_patch16_224",
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    cache_dir: str | None = None,
    hf_endpoint: str | None = VIT_S_DEFAULT_HF_ENDPOINT,
    image_source: str = "synthetic_gradient",
    parameter_seed: int = 0,
    **_: Any,
) -> VitSWeightBundle:
    _configure_huggingface_endpoint(hf_endpoint)
    import timm  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    create_kwargs: dict[str, Any] = {"pretrained": bool(pretrained)}
    if checkpoint_path:
        create_kwargs["checkpoint_path"] = checkpoint_path
    if cache_dir:
        create_kwargs["cache_dir"] = cache_dir
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(parameter_seed))
        model = timm.create_model(model_name, **create_kwargs).eval()
    _configure_timm_vit_s_for_numpy_comparison(model)
    _validate_timm_vit_s_model(model, model_name)
    values = _vit_s_values_from_timm_model(
        model,
        image_source=image_source,
        parameter_seed=parameter_seed,
    )
    default_cfg = getattr(model, "default_cfg", {}) or {}
    return VitSWeightBundle(
        source_id=VIT_S_WEIGHT_SOURCE_TIMM,
        provider_id="timm",
        values=values,
        torch_model=model,
        metadata={
            "model_name": model_name,
            "pretrained": bool(pretrained),
            "checkpoint_path": checkpoint_path,
            "cache_dir": cache_dir,
            "hf_endpoint": hf_endpoint,
            "image_source": image_source,
            "parameter_seed": parameter_seed,
            "num_classes": int(getattr(model, "num_classes", _VIT_S_CLASSES)),
            "default_cfg": {
                key: default_cfg.get(key)
                for key in ("input_size", "mean", "std", "crop_pct", "interpolation")
            },
            "class_names": "imagenet_1k",
            "gelu_approximation": "none",
            "layer_norm_epsilon": float(getattr(model.norm, "eps", 1.0e-6)),
            "fused_attention": False,
        },
    )


def _configure_huggingface_endpoint(hf_endpoint: str | None) -> None:
    """Configure the Hugging Face endpoint before timm resolves pretrained weights."""

    if not hf_endpoint:
        return
    os.environ["HF_ENDPOINT"] = hf_endpoint
    try:
        import huggingface_hub.constants as hf_constants  # pylint: disable=import-outside-toplevel

        endpoint = hf_endpoint.rstrip("/")
        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            endpoint + "/{repo_id}/resolve/{revision}/{filename}"
        )
    except Exception:
        return


def _configure_timm_vit_s_for_numpy_comparison(model: Any) -> None:
    """Use explicit attention math so timm and NumPy follow the same operator route."""

    for block in getattr(model, "blocks", ()):
        attention = getattr(block, "attn", None)
        if attention is not None and hasattr(attention, "fused_attn"):
            attention.fused_attn = False


def _validate_timm_vit_s_model(model: Any, model_name: str) -> None:
    state = model.state_dict()
    expected = {
        "cls_token": (_VIT_S_BATCH, 1, _VIT_S_HIDDEN),
        "pos_embed": (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
        "patch_embed.proj.weight": (_VIT_S_HIDDEN, _VIT_S_CHANNELS, _VIT_S_PATCH, _VIT_S_PATCH),
        "head.weight": (_VIT_S_CLASSES, _VIT_S_HIDDEN),
    }
    for key, shape in expected.items():
        tensor = state.get(key)
        if tensor is None:
            raise ValueError(f"timm model {model_name} is missing state key: {key}")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"timm model {model_name} has incompatible {key} shape: "
                f"{tuple(tensor.shape)} != {shape}"
            )
    if len(getattr(model, "blocks", ())) != _VIT_S_LAYERS:
        raise ValueError(f"timm model {model_name} is not a 12-layer ViT-S model")


def _vit_s_values_from_timm_model(
    model: Any,
    *,
    image_source: str,
    parameter_seed: int,
) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    state = model.state_dict()

    def tensor(key: str) -> Any:
        return state[key].detach().cpu().numpy().astype(np.float32, copy=True)

    values = {
        "image": _vit_s_input_image(
            image_source=image_source,
            default_cfg=getattr(model, "default_cfg", {}) or {},
            parameter_seed=parameter_seed,
        ),
        "patch_embed.weight": tensor("patch_embed.proj.weight"),
        "patch_embed.bias": tensor("patch_embed.proj.bias"),
        "class_token": tensor("cls_token"),
        "position_embed": tensor("pos_embed"),
    }
    for layer in range(_VIT_S_LAYERS):
        prefix = _vit_s_layer_prefix(layer)
        timm_prefix = f"blocks.{layer}"
        values[f"{prefix}.norm0.scale"] = tensor(f"{timm_prefix}.norm1.weight")
        values[f"{prefix}.norm0.bias"] = tensor(f"{timm_prefix}.norm1.bias")
        values[f"{prefix}.qkv.weight"] = tensor(f"{timm_prefix}.attn.qkv.weight").T.copy()
        values[f"{prefix}.qkv.bias"] = tensor(f"{timm_prefix}.attn.qkv.bias")
        values[f"{prefix}.attention_projection.weight"] = tensor(f"{timm_prefix}.attn.proj.weight").T.copy()
        values[f"{prefix}.attention_projection.bias"] = tensor(f"{timm_prefix}.attn.proj.bias")
        values[f"{prefix}.norm1.scale"] = tensor(f"{timm_prefix}.norm2.weight")
        values[f"{prefix}.norm1.bias"] = tensor(f"{timm_prefix}.norm2.bias")
        values[f"{prefix}.mlp_fc0.weight"] = tensor(f"{timm_prefix}.mlp.fc1.weight").T.copy()
        values[f"{prefix}.mlp_fc0.bias"] = tensor(f"{timm_prefix}.mlp.fc1.bias")
        values[f"{prefix}.mlp_fc1.weight"] = tensor(f"{timm_prefix}.mlp.fc2.weight").T.copy()
        values[f"{prefix}.mlp_fc1.bias"] = tensor(f"{timm_prefix}.mlp.fc2.bias")
    values["final_norm.scale"] = tensor("norm.weight")
    values["final_norm.bias"] = tensor("norm.bias")
    values["head.weight"] = tensor("head.weight").T.copy()
    values["head.bias"] = tensor("head.bias")
    return values


def _vit_s_input_image(
    *,
    image_source: str,
    default_cfg: Mapping[str, Any],
    parameter_seed: int,
) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    if image_source == "synthetic_gradient":
        y = np.linspace(0.0, 1.0, _VIT_S_IMAGE, dtype=np.float32)
        x = np.linspace(0.0, 1.0, _VIT_S_IMAGE, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
        image = np.stack((grid_x, grid_y, 1.0 - grid_x), axis=0)[None, :, :, :]
    elif image_source == "deterministic_random":
        rng = np.random.default_rng(parameter_seed)
        image = rng.random(
            (_VIT_S_BATCH, _VIT_S_CHANNELS, _VIT_S_IMAGE, _VIT_S_IMAGE),
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unsupported ViT-S image source: {image_source}")
    mean = np.asarray(default_cfg.get("mean", (0.5, 0.5, 0.5)), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(default_cfg.get("std", (0.5, 0.5, 0.5)), dtype=np.float32).reshape(1, 3, 1, 1)
    return ((image - mean) / std).astype(np.float32)


def _imagenet_label(index: int) -> tuple[str | None, str | None]:
    try:
        from timm.data import ImageNetInfo  # pylint: disable=import-outside-toplevel

        info = ImageNetInfo()
        return info.index_to_label_name(int(index)), info.index_to_description(int(index))
    except Exception:  # pragma: no cover - optional label metadata
        return None, None


def _softmax_numpy(values: Any) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _vit_s_shape_signature() -> dict[str, int]:
    return {
        "batch": _VIT_S_BATCH,
        "channels": _VIT_S_CHANNELS,
        "image": _VIT_S_IMAGE,
        "patch": _VIT_S_PATCH,
        "patch_tokens": _VIT_S_PATCH_TOKENS,
        "tokens": _VIT_S_TOKENS,
        "hidden": _VIT_S_HIDDEN,
        "heads": _VIT_S_HEADS,
        "head_dim": _VIT_S_HEAD_DIM,
        "mlp": _VIT_S_MLP,
        "layers": _VIT_S_LAYERS,
        "classes": _VIT_S_CLASSES,
    }


def _vit_s_layer_prefix(layer: int) -> str:
    return f"encoder{layer:02d}"


def _vit_s_parameter_buffer_specs() -> tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...]:
    specs: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        ("patch_embed.weight", (_VIT_S_HIDDEN, _VIT_S_CHANNELS, _VIT_S_PATCH, _VIT_S_PATCH), ("patch_embed",)),
        ("patch_embed.bias", (_VIT_S_HIDDEN,), ("patch_embed",)),
        ("class_token", (_VIT_S_BATCH, 1, _VIT_S_HIDDEN), ("class_token_concat",)),
        ("position_embed", (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), ("position_add",)),
    ]
    for layer in range(_VIT_S_LAYERS):
        prefix = _vit_s_layer_prefix(layer)
        specs.extend(
            [
                (f"{prefix}.norm0.scale", (_VIT_S_HIDDEN,), (f"{prefix}_norm0",)),
                (f"{prefix}.norm0.bias", (_VIT_S_HIDDEN,), (f"{prefix}_norm0",)),
                (f"{prefix}.qkv.weight", (_VIT_S_HIDDEN, 3 * _VIT_S_HIDDEN), (f"{prefix}_qkv_projection",)),
                (f"{prefix}.qkv.bias", (3 * _VIT_S_HIDDEN,), (f"{prefix}_qkv_projection",)),
                (f"{prefix}.attention_projection.weight", (_VIT_S_HIDDEN, _VIT_S_HIDDEN), (f"{prefix}_attention_projection",)),
                (f"{prefix}.attention_projection.bias", (_VIT_S_HIDDEN,), (f"{prefix}_attention_projection",)),
                (f"{prefix}.norm1.scale", (_VIT_S_HIDDEN,), (f"{prefix}_norm1",)),
                (f"{prefix}.norm1.bias", (_VIT_S_HIDDEN,), (f"{prefix}_norm1",)),
                (f"{prefix}.mlp_fc0.weight", (_VIT_S_HIDDEN, _VIT_S_MLP), (f"{prefix}_mlp_fc0",)),
                (f"{prefix}.mlp_fc0.bias", (_VIT_S_MLP,), (f"{prefix}_mlp_fc0",)),
                (f"{prefix}.mlp_fc1.weight", (_VIT_S_MLP, _VIT_S_HIDDEN), (f"{prefix}_mlp_fc1",)),
                (f"{prefix}.mlp_fc1.bias", (_VIT_S_HIDDEN,), (f"{prefix}_mlp_fc1",)),
            ]
        )
    specs.extend(
        [
            ("final_norm.scale", (_VIT_S_HIDDEN,), ("final_norm",)),
            ("final_norm.bias", (_VIT_S_HIDDEN,), ("final_norm",)),
            ("head.weight", (_VIT_S_HIDDEN, _VIT_S_CLASSES), ("classifier_head",)),
            ("head.bias", (_VIT_S_CLASSES,), ("classifier_head",)),
        ]
    )
    return tuple(specs)


def _vit_s_buffer_shape(buffer_id: str) -> tuple[int, ...]:
    if buffer_id == "patch_embed.out":
        return (_VIT_S_BATCH, _VIT_S_PATCH_TOKENS, _VIT_S_HIDDEN)
    if buffer_id in {"class_token_concat.out", "position_add.out", "last_hidden_state"}:
        return (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN)
    if buffer_id == "logits":
        return (_VIT_S_BATCH, _VIT_S_CLASSES)
    if buffer_id.endswith(".qkv.out"):
        return (_VIT_S_BATCH, _VIT_S_TOKENS, 3 * _VIT_S_HIDDEN)
    if buffer_id.endswith(".mlp_fc0.out") or buffer_id.endswith(".mlp_gelu.out"):
        return (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_MLP)
    if (
        buffer_id.endswith(".norm0.out")
        or buffer_id.endswith(".attention.out")
        or buffer_id.endswith(".attention_projection.out")
        or buffer_id.endswith(".residual_after_attention.out")
        or buffer_id.endswith(".norm1.out")
        or buffer_id.endswith(".mlp_fc1.out")
        or buffer_id.endswith(".out")
    ):
        return (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN)
    raise KeyError(buffer_id)


def _vit_s_consumers_for_buffer(
    buffer_id: str, specs: Iterable[VitTopLevelOperatorSpec]
) -> tuple[str, ...]:
    return tuple(
        spec.operator_id for spec in specs if buffer_id in spec.input_buffer_ids
    )


def _vit_s_initial_buffer_values(*, parameter_seed: int = 0) -> dict[str, Any]:
    return _vit_s_random_buffer_values(parameter_seed=parameter_seed)


def _vit_s_random_buffer_values(*, parameter_seed: int = 0) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(parameter_seed)

    def normal(shape: tuple[int, ...], scale: float) -> Any:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    def zeros(shape: tuple[int, ...]) -> Any:
        return np.zeros(shape, dtype=np.float32)

    values = {
        "image": normal((_VIT_S_BATCH, _VIT_S_CHANNELS, _VIT_S_IMAGE, _VIT_S_IMAGE), 0.2),
        "patch_embed.weight": normal((_VIT_S_HIDDEN, _VIT_S_CHANNELS, _VIT_S_PATCH, _VIT_S_PATCH), 0.02),
        "patch_embed.bias": zeros((_VIT_S_HIDDEN,)),
        "class_token": normal((_VIT_S_BATCH, 1, _VIT_S_HIDDEN), 0.02),
        "position_embed": normal((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), 0.02),
    }
    for layer in range(_VIT_S_LAYERS):
        prefix = _vit_s_layer_prefix(layer)
        values[f"{prefix}.norm0.scale"] = np.ones((_VIT_S_HIDDEN,), dtype=np.float32)
        values[f"{prefix}.norm0.bias"] = zeros((_VIT_S_HIDDEN,))
        values[f"{prefix}.qkv.weight"] = normal((_VIT_S_HIDDEN, 3 * _VIT_S_HIDDEN), 0.02)
        values[f"{prefix}.qkv.bias"] = zeros((3 * _VIT_S_HIDDEN,))
        values[f"{prefix}.attention_projection.weight"] = normal((_VIT_S_HIDDEN, _VIT_S_HIDDEN), 0.02)
        values[f"{prefix}.attention_projection.bias"] = zeros((_VIT_S_HIDDEN,))
        values[f"{prefix}.norm1.scale"] = np.ones((_VIT_S_HIDDEN,), dtype=np.float32)
        values[f"{prefix}.norm1.bias"] = zeros((_VIT_S_HIDDEN,))
        values[f"{prefix}.mlp_fc0.weight"] = normal((_VIT_S_HIDDEN, _VIT_S_MLP), 0.02)
        values[f"{prefix}.mlp_fc0.bias"] = zeros((_VIT_S_MLP,))
        values[f"{prefix}.mlp_fc1.weight"] = normal((_VIT_S_MLP, _VIT_S_HIDDEN), 0.02)
        values[f"{prefix}.mlp_fc1.bias"] = zeros((_VIT_S_HIDDEN,))
    values["final_norm.scale"] = np.ones((_VIT_S_HIDDEN,), dtype=np.float32)
    values["final_norm.bias"] = zeros((_VIT_S_HIDDEN,))
    values["head.weight"] = normal((_VIT_S_HIDDEN, _VIT_S_CLASSES), 0.02)
    values["head.bias"] = zeros((_VIT_S_CLASSES,))
    return values


def _execute_vit_s_numpy(
    values: Mapping[str, Any],
    *,
    gelu_approximation: str = "tanh",
    layer_norm_epsilon: float = 1.0e-5,
) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    buffers = {buffer_id: np.array(value, copy=True) for buffer_id, value in values.items()}
    patches = _vit_s_patch_embed_numpy(
        buffers["image"], buffers["patch_embed.weight"], buffers["patch_embed.bias"]
    )
    buffers["patch_embed.out"] = patches
    x = np.concatenate([buffers["class_token"], patches], axis=1).astype(np.float32)
    buffers["class_token_concat.out"] = x
    x = (x + buffers["position_embed"]).astype(np.float32)
    buffers["position_add.out"] = x
    for layer in range(_VIT_S_LAYERS):
        prefix = _vit_s_layer_prefix(layer)
        norm0 = _vit_s_layer_norm_numpy(
            x,
            buffers[f"{prefix}.norm0.scale"],
            buffers[f"{prefix}.norm0.bias"],
            epsilon=layer_norm_epsilon,
        )
        buffers[f"{prefix}.norm0.out"] = norm0
        qkv = (norm0 @ buffers[f"{prefix}.qkv.weight"] + buffers[f"{prefix}.qkv.bias"]).astype(np.float32)
        buffers[f"{prefix}.qkv.out"] = qkv
        attention = _vit_s_attention_numpy(qkv)
        buffers[f"{prefix}.attention.out"] = attention
        attention_projection = (
            attention @ buffers[f"{prefix}.attention_projection.weight"]
            + buffers[f"{prefix}.attention_projection.bias"]
        ).astype(np.float32)
        buffers[f"{prefix}.attention_projection.out"] = attention_projection
        residual_attention = (x + attention_projection).astype(np.float32)
        buffers[f"{prefix}.residual_after_attention.out"] = residual_attention
        norm1 = _vit_s_layer_norm_numpy(
            residual_attention,
            buffers[f"{prefix}.norm1.scale"],
            buffers[f"{prefix}.norm1.bias"],
            epsilon=layer_norm_epsilon,
        )
        buffers[f"{prefix}.norm1.out"] = norm1
        mlp_fc0 = (
            norm1 @ buffers[f"{prefix}.mlp_fc0.weight"] + buffers[f"{prefix}.mlp_fc0.bias"]
        ).astype(np.float32)
        buffers[f"{prefix}.mlp_fc0.out"] = mlp_fc0
        mlp_gelu = _vit_s_gelu_numpy(mlp_fc0, approximation=gelu_approximation)
        buffers[f"{prefix}.mlp_gelu.out"] = mlp_gelu
        mlp_fc1 = (
            mlp_gelu @ buffers[f"{prefix}.mlp_fc1.weight"] + buffers[f"{prefix}.mlp_fc1.bias"]
        ).astype(np.float32)
        buffers[f"{prefix}.mlp_fc1.out"] = mlp_fc1
        x = (residual_attention + mlp_fc1).astype(np.float32)
        buffers[f"{prefix}.out"] = x
    last_hidden_state = _vit_s_layer_norm_numpy(
        x,
        buffers["final_norm.scale"],
        buffers["final_norm.bias"],
        epsilon=layer_norm_epsilon,
    )
    buffers["last_hidden_state"] = last_hidden_state
    buffers["logits"] = (
        last_hidden_state[:, 0, :] @ buffers["head.weight"] + buffers["head.bias"]
    ).astype(np.float32)
    return buffers


def _vit_s_patch_embed_numpy(image: Any, weight: Any, bias: Any) -> Any:
    batch = image.shape[0]
    windows = image.reshape(
        batch,
        _VIT_S_CHANNELS,
        _VIT_S_PATCH_GRID,
        _VIT_S_PATCH,
        _VIT_S_PATCH_GRID,
        _VIT_S_PATCH,
    )
    windows = windows.transpose(0, 2, 4, 1, 3, 5).reshape(
        batch, _VIT_S_PATCH_TOKENS, _VIT_S_CHANNELS * _VIT_S_PATCH * _VIT_S_PATCH
    )
    return (windows @ weight.reshape(_VIT_S_HIDDEN, -1).T + bias).astype("float32")


def _vit_s_layer_norm_numpy(x: Any, scale: Any, bias: Any, epsilon: float = 1.0e-5) -> Any:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / ((var + epsilon) ** 0.5) * scale + bias).astype("float32")


def _vit_s_attention_numpy(qkv: Any) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    batch, tokens, _ = qkv.shape
    packed = qkv.reshape(batch, tokens, 3, _VIT_S_HEADS, _VIT_S_HEAD_DIM).transpose(2, 0, 3, 1, 4)
    q, k, v = packed[0], packed[1], packed[2]
    scores = (q @ np.swapaxes(k, -1, -2)) * np.float32(1.0 / math.sqrt(_VIT_S_HEAD_DIM))
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores).astype(np.float32)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    out = weights @ v
    return out.transpose(0, 2, 1, 3).reshape(batch, tokens, _VIT_S_HIDDEN).astype(np.float32)


def _vit_s_gelu_tanh_numpy(x: Any) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    coeff = np.float32(math.sqrt(2.0 / math.pi))
    return (
        np.float32(0.5)
        * x
        * (np.float32(1.0) + np.tanh(coeff * (x + np.float32(0.044715) * x * x * x)))
    ).astype(np.float32)


def _vit_s_gelu_exact_numpy(x: Any) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    try:
        from scipy import special  # pylint: disable=import-outside-toplevel

        erf = special.erf
    except Exception:  # pragma: no cover - fallback for minimal environments
        erf = np.vectorize(math.erf)
    return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)


def _vit_s_gelu_numpy(x: Any, *, approximation: str) -> Any:
    if approximation == "none":
        return _vit_s_gelu_exact_numpy(x)
    if approximation == "tanh":
        return _vit_s_gelu_tanh_numpy(x)
    raise ValueError(f"unsupported ViT-S GELU approximation: {approximation}")


class _TimmVitSFeatureAndLogits:
    """Adapter that makes a timm ViT return hidden states and logits."""

    def __new__(cls, model: Any) -> Any:
        import torch  # pylint: disable=import-outside-toplevel

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped: Any):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, image: Any) -> tuple[Any, Any]:
                features = self.wrapped.forward_features(image)
                logits = self.wrapped.forward_head(features)
                return features, logits

        return Wrapper(model)


def _build_torch_vit_s_model(values: Mapping[str, Any]) -> Any:
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as torch_functional  # pylint: disable=import-outside-toplevel

    def frozen_parameter(name: str) -> Any:
        return torch.nn.Parameter(torch.from_numpy(values[name].copy()), requires_grad=False)

    class TorchVitSBlock(torch.nn.Module):
        def __init__(self, layer: int):
            super().__init__()
            prefix = _vit_s_layer_prefix(layer)
            self.norm0_scale = frozen_parameter(f"{prefix}.norm0.scale")
            self.norm0_bias = frozen_parameter(f"{prefix}.norm0.bias")
            self.qkv_weight = frozen_parameter(f"{prefix}.qkv.weight")
            self.qkv_bias = frozen_parameter(f"{prefix}.qkv.bias")
            self.attention_projection_weight = frozen_parameter(f"{prefix}.attention_projection.weight")
            self.attention_projection_bias = frozen_parameter(f"{prefix}.attention_projection.bias")
            self.norm1_scale = frozen_parameter(f"{prefix}.norm1.scale")
            self.norm1_bias = frozen_parameter(f"{prefix}.norm1.bias")
            self.mlp_fc0_weight = frozen_parameter(f"{prefix}.mlp_fc0.weight")
            self.mlp_fc0_bias = frozen_parameter(f"{prefix}.mlp_fc0.bias")
            self.mlp_fc1_weight = frozen_parameter(f"{prefix}.mlp_fc1.weight")
            self.mlp_fc1_bias = frozen_parameter(f"{prefix}.mlp_fc1.bias")

        def forward(self, x: Any) -> Any:
            norm0 = torch_functional.layer_norm(
                x, (_VIT_S_HIDDEN,), self.norm0_scale, self.norm0_bias, eps=1.0e-5
            )
            qkv = norm0 @ self.qkv_weight + self.qkv_bias
            batch, tokens, _ = qkv.shape
            packed = qkv.reshape(batch, tokens, 3, _VIT_S_HEADS, _VIT_S_HEAD_DIM).permute(2, 0, 3, 1, 4)
            q, k, v = packed[0], packed[1], packed[2]
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(float(_VIT_S_HEAD_DIM)))
            weights = torch_functional.softmax(scores, dim=-1)
            attention = (weights @ v).transpose(1, 2).reshape(batch, tokens, _VIT_S_HIDDEN)
            attention_projection = attention @ self.attention_projection_weight + self.attention_projection_bias
            x = x + attention_projection
            norm1 = torch_functional.layer_norm(
                x, (_VIT_S_HIDDEN,), self.norm1_scale, self.norm1_bias, eps=1.0e-5
            )
            mlp = norm1 @ self.mlp_fc0_weight + self.mlp_fc0_bias
            mlp = torch_functional.gelu(mlp, approximate="tanh")
            mlp = mlp @ self.mlp_fc1_weight + self.mlp_fc1_bias
            return x + mlp

    class TorchVitS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed_weight = frozen_parameter("patch_embed.weight")
            self.patch_embed_bias = frozen_parameter("patch_embed.bias")
            self.class_token = frozen_parameter("class_token")
            self.position_embed = frozen_parameter("position_embed")
            self.layers = torch.nn.ModuleList(
                TorchVitSBlock(layer) for layer in range(_VIT_S_LAYERS)
            )
            self.final_norm_scale = frozen_parameter("final_norm.scale")
            self.final_norm_bias = frozen_parameter("final_norm.bias")
            self.head_weight = frozen_parameter("head.weight")
            self.head_bias = frozen_parameter("head.bias")

        def forward(self, image: Any) -> tuple[Any, Any]:
            x = torch_functional.conv2d(
                image,
                self.patch_embed_weight,
                self.patch_embed_bias,
                stride=_VIT_S_PATCH,
            )
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat((self.class_token.expand(image.shape[0], -1, -1), x), dim=1)
            x = x + self.position_embed
            for layer in self.layers:
                x = layer(x)
            x = torch_functional.layer_norm(
                x, (_VIT_S_HIDDEN,), self.final_norm_scale, self.final_norm_bias, eps=1.0e-5
            )
            logits = x[:, 0, :] @ self.head_weight + self.head_bias
            return x, logits

    return TorchVitS()


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
    model_id: str = VIT_TINY_FIXED_SHAPE_MODEL_ID,
    runner_id: str = VIT_RUNNER_ID,
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
                "model_id": model_id,
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
                model_id=model_id,
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
                optional_model_anchor=runner_id,
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
            model_id=model_id,
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
            optional_model_anchor=runner_id,
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
                    model_id=model_id,
                    operator_id=top_spec.operator_id,
                    source_id=route_sources[0].source_id,
                    reason=gap_reason,
                )
            )
        top_records.append(
            VitTopLevelOperatorRecord(
                model_id=model_id,
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


def _vit_s_tir_module_for_operator(
    operator_id: str,
    *,
    name_prefix: str,
    target: str,
    gelu_approximation: str,
    layer_norm_epsilon: float,
    matmul_autotune_config: VitSMatmulAutotuneConfig | None,
    matmul_codegen_cache: dict[str, tuple[Any, dict[str, Any]]],
    operator_autotune_config: VitSOperatorAutotuneConfig | None,
    operator_codegen_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[Any, str, dict[str, Any] | None]:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    threads = _VIT_S_CUDA_THREADS_PER_BLOCK

    def blocks(total: int) -> int:
        return (total + threads - 1) // threads

    name = f"{name_prefix}_{operator_id}"
    if (
        operator_autotune_config is not None
        and operator_autotune_config.enabled
        and _is_cuda_target(target)
        and _vit_s_is_non_matmul_te_operator(operator_id)
    ):
        return _vit_s_operator_te_tir_module(
            operator_id=operator_id,
            name=name,
            target=target,
            gelu_approximation=gelu_approximation,
            layer_norm_epsilon=layer_norm_epsilon,
            autotune_config=operator_autotune_config,
            codegen_cache=operator_codegen_cache,
        )
    if operator_id == "patch_embed":
        total = _VIT_S_PATCH_TOKENS * _VIT_S_HIDDEN
        nblocks = blocks(total)

        @T.prim_func
        def prim(
            image: T.Buffer((_VIT_S_BATCH, _VIT_S_CHANNELS, _VIT_S_IMAGE, _VIT_S_IMAGE), "float32"),
            weight: T.Buffer((_VIT_S_HIDDEN, _VIT_S_CHANNELS, _VIT_S_PATCH, _VIT_S_PATCH), "float32"),
            bias: T.Buffer((_VIT_S_HIDDEN,), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_PATCH_TOKENS, _VIT_S_HIDDEN), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            acc = T.alloc_buffer((1,), scope="local")
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = idx // _VIT_S_HIDDEN
                        hidden = idx % _VIT_S_HIDDEN
                        py = token // _VIT_S_PATCH_GRID
                        px = token % _VIT_S_PATCH_GRID
                        acc[0] = T.float32(0.0)
                        for rc, ry, rx in T.grid(_VIT_S_CHANNELS, _VIT_S_PATCH, _VIT_S_PATCH):
                            acc[0] = (
                                acc[0]
                                + image[0, rc, py * _VIT_S_PATCH + ry, px * _VIT_S_PATCH + rx]
                                * weight[hidden, rc, ry, rx]
                            )
                        out[0, token, hidden] = acc[0] + bias[hidden]

    elif operator_id == "class_token_concat":
        total = _VIT_S_TOKENS * _VIT_S_HIDDEN
        nblocks = blocks(total)

        @T.prim_func
        def prim(
            x: T.Buffer((_VIT_S_BATCH, _VIT_S_PATCH_TOKENS, _VIT_S_HIDDEN), "float32"),
            cls: T.Buffer((_VIT_S_BATCH, 1, _VIT_S_HIDDEN), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = idx // _VIT_S_HIDDEN
                        hidden = idx % _VIT_S_HIDDEN
                        if token == 0:
                            out[0, token, hidden] = cls[0, 0, hidden]
                        else:
                            out[0, token, hidden] = x[0, token - 1, hidden]

    elif operator_id == "position_add" or operator_id.endswith("_residual_after_attention") or operator_id.endswith("_residual_after_mlp"):
        total = _VIT_S_TOKENS * _VIT_S_HIDDEN
        nblocks = blocks(total)

        @T.prim_func
        def prim(
            a: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
            b: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = idx // _VIT_S_HIDDEN
                        hidden = idx % _VIT_S_HIDDEN
                        out[0, token, hidden] = a[0, token, hidden] + b[0, token, hidden]

    elif operator_id.endswith("_norm0") or operator_id.endswith("_norm1") or operator_id == "final_norm":
        total = _VIT_S_TOKENS * _VIT_S_HIDDEN
        nblocks = blocks(total)
        epsilon = T.float32(layer_norm_epsilon)

        @T.prim_func
        def prim(
            x: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
            scale: T.Buffer((_VIT_S_HIDDEN,), "float32"),
            bias: T.Buffer((_VIT_S_HIDDEN,), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            mean = T.alloc_buffer((1,), scope="local")
            var = T.alloc_buffer((1,), scope="local")
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = idx // _VIT_S_HIDDEN
                        hidden = idx % _VIT_S_HIDDEN
                        mean[0] = T.float32(0.0)
                        for h in T.serial(_VIT_S_HIDDEN):
                            mean[0] = mean[0] + x[0, token, h] * T.float32(1.0 / _VIT_S_HIDDEN)
                        var[0] = T.float32(0.0)
                        for h in T.serial(_VIT_S_HIDDEN):
                            diff = x[0, token, h] - mean[0]
                            var[0] = var[0] + diff * diff * T.float32(1.0 / _VIT_S_HIDDEN)
                        out[0, token, hidden] = (
                            (x[0, token, hidden] - mean[0])
                            / T.sqrt(var[0] + epsilon)
                            * scale[hidden]
                            + bias[hidden]
                        )

    elif operator_id.endswith("_qkv_projection"):
        return _vit_s_matmul_te_tir_module(
            operator_id=operator_id,
            input_dim=_VIT_S_HIDDEN,
            output_dim=3 * _VIT_S_HIDDEN,
            name=name,
            output_kind="tokens",
            target=target,
            autotune_config=matmul_autotune_config,
            codegen_cache=matmul_codegen_cache,
        )
    elif operator_id.endswith("_attention_projection"):
        return _vit_s_matmul_te_tir_module(
            operator_id=operator_id,
            input_dim=_VIT_S_HIDDEN,
            output_dim=_VIT_S_HIDDEN,
            name=name,
            output_kind="tokens",
            target=target,
            autotune_config=matmul_autotune_config,
            codegen_cache=matmul_codegen_cache,
        )
    elif operator_id.endswith("_mlp_fc0"):
        return _vit_s_matmul_te_tir_module(
            operator_id=operator_id,
            input_dim=_VIT_S_HIDDEN,
            output_dim=_VIT_S_MLP,
            name=name,
            output_kind="tokens",
            target=target,
            autotune_config=matmul_autotune_config,
            codegen_cache=matmul_codegen_cache,
        )
    elif operator_id.endswith("_mlp_fc1"):
        return _vit_s_matmul_te_tir_module(
            operator_id=operator_id,
            input_dim=_VIT_S_MLP,
            output_dim=_VIT_S_HIDDEN,
            name=name,
            output_kind="tokens",
            target=target,
            autotune_config=matmul_autotune_config,
            codegen_cache=matmul_codegen_cache,
        )
    elif operator_id == "classifier_head":
        return _vit_s_matmul_te_tir_module(
            operator_id=operator_id,
            input_dim=_VIT_S_HIDDEN,
            output_dim=_VIT_S_CLASSES,
            name=name,
            output_kind="classifier",
            target=target,
            autotune_config=matmul_autotune_config,
            codegen_cache=matmul_codegen_cache,
        )

    elif operator_id.endswith("_mlp_gelu"):
        total = _VIT_S_TOKENS * _VIT_S_MLP
        nblocks = blocks(total)
        exact = gelu_approximation == "none"

        @T.prim_func
        def prim(
            x: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_MLP), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_MLP), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = idx // _VIT_S_MLP
                        hidden = idx % _VIT_S_MLP
                        value = x[0, token, hidden]
                        if exact:
                            out[0, token, hidden] = (
                                T.float32(0.5)
                                * value
                                * (T.float32(1.0) + T.erf(value / T.sqrt(T.float32(2.0))))
                            )
                        else:
                            out[0, token, hidden] = (
                                T.float32(0.5)
                                * value
                                * (
                                    T.float32(1.0)
                                    + T.tanh(
                                        T.float32(math.sqrt(2.0 / math.pi))
                                        * (
                                            value
                                            + T.float32(0.044715) * value * value * value
                                        )
                                    )
                                )
                            )

    elif operator_id.endswith("_attention"):
        total = _VIT_S_TOKENS * _VIT_S_HEADS
        nblocks = blocks(total)
        scale = T.float32(1.0 / math.sqrt(float(_VIT_S_HEAD_DIM)))

        @T.prim_func
        def prim(
            qkv: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, 3 * _VIT_S_HIDDEN), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            max_score = T.alloc_buffer((1,), scope="local")
            denom = T.alloc_buffer((1,), scope="local")
            acc = T.alloc_buffer((1,), scope="local")
            score = T.alloc_buffer((1,), scope="local")
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = idx // _VIT_S_HEADS
                        head = idx % _VIT_S_HEADS
                        max_score[0] = T.float32(-3.4e38)
                        for key_token in T.serial(_VIT_S_TOKENS):
                            score[0] = T.float32(0.0)
                            for k in T.serial(_VIT_S_HEAD_DIM):
                                score[0] = (
                                    score[0]
                                    + qkv[0, token, head * _VIT_S_HEAD_DIM + k]
                                    * qkv[
                                        0,
                                        key_token,
                                        _VIT_S_HIDDEN + head * _VIT_S_HEAD_DIM + k,
                                    ]
                                )
                            score[0] = score[0] * scale
                            max_score[0] = T.max(max_score[0], score[0])
                        for dim in T.serial(_VIT_S_HEAD_DIM):
                            denom[0] = T.float32(0.0)
                            acc[0] = T.float32(0.0)
                            for key_token in T.serial(_VIT_S_TOKENS):
                                score[0] = T.float32(0.0)
                                for k in T.serial(_VIT_S_HEAD_DIM):
                                    score[0] = (
                                        score[0]
                                        + qkv[0, token, head * _VIT_S_HEAD_DIM + k]
                                        * qkv[
                                            0,
                                            key_token,
                                            _VIT_S_HIDDEN + head * _VIT_S_HEAD_DIM + k,
                                        ]
                                    )
                                score[0] = T.exp(score[0] * scale - max_score[0])
                                denom[0] = denom[0] + score[0]
                                acc[0] = (
                                    acc[0]
                                    + score[0]
                                    * qkv[
                                        0,
                                        key_token,
                                        2 * _VIT_S_HIDDEN + head * _VIT_S_HEAD_DIM + dim,
                                    ]
                                )
                            out[0, token, head * _VIT_S_HEAD_DIM + dim] = acc[0] / denom[0]

    else:
        raise ValueError(f"unsupported ViT-S TIR operator: {operator_id}")

    return tvm.IRModule({name: prim.with_attr("global_symbol", name)}), name, None


def _vit_s_attention_tir_module_and_plan(
    request: OperatorExecutionRequest,
    *,
    name_prefix: str,
    artifact_source: str,
    target: str,
    operator_autotune_config: VitSOperatorAutotuneConfig | None,
    operator_codegen_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[Any, tuple[LoweringStep, ...], dict[str, Mapping[str, Any]], tuple[dict[str, Any], ...]]:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    if (
        operator_autotune_config is not None
        and operator_autotune_config.enabled
        and _is_cuda_target(target)
    ):
        return _vit_s_attention_te_tir_module_and_plan(
            request,
            name_prefix=name_prefix,
            artifact_source=artifact_source,
            target=target,
            autotune_config=operator_autotune_config,
            codegen_cache=operator_codegen_cache,
        )

    threads = _VIT_S_CUDA_THREADS_PER_BLOCK
    score_total = _VIT_S_HEADS * _VIT_S_TOKENS * _VIT_S_TOKENS
    score_blocks = (score_total + threads - 1) // threads
    softmax_total = _VIT_S_HEADS * _VIT_S_TOKENS
    softmax_blocks = (softmax_total + threads - 1) // threads
    apply_total = _VIT_S_TOKENS * _VIT_S_HIDDEN
    apply_blocks = (apply_total + threads - 1) // threads
    scale = T.float32(1.0 / math.sqrt(float(_VIT_S_HEAD_DIM)))
    score_name = f"{name_prefix}_{request.operator_id}_score"
    softmax_name = f"{name_prefix}_{request.operator_id}_softmax"
    apply_name = f"{name_prefix}_{request.operator_id}_apply"
    scores_id = f"{request.operator_id}.attention_scores"

    @T.prim_func
    def score_prim(
        qkv: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, 3 * _VIT_S_HIDDEN), "float32"),
        scores: T.Buffer((_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        score = T.alloc_buffer((1,), scope="local")
        for bx in T.thread_binding(score_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                idx = bx * threads + tx
                if idx < score_total:
                    head = idx // (_VIT_S_TOKENS * _VIT_S_TOKENS)
                    rem = idx % (_VIT_S_TOKENS * _VIT_S_TOKENS)
                    token = rem // _VIT_S_TOKENS
                    key_token = rem % _VIT_S_TOKENS
                    score[0] = T.float32(0.0)
                    for k in T.serial(_VIT_S_HEAD_DIM):
                        score[0] = (
                            score[0]
                            + qkv[0, token, head * _VIT_S_HEAD_DIM + k]
                            * qkv[
                                0,
                                key_token,
                                _VIT_S_HIDDEN + head * _VIT_S_HEAD_DIM + k,
                            ]
                        )
                    scores[0, head, token, key_token] = score[0] * scale

    @T.prim_func
    def softmax_prim(
        scores: T.Buffer((_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        max_score = T.alloc_buffer((1,), scope="local")
        denom = T.alloc_buffer((1,), scope="local")
        for bx in T.thread_binding(softmax_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                idx = bx * threads + tx
                if idx < softmax_total:
                    head = idx // _VIT_S_TOKENS
                    token = idx % _VIT_S_TOKENS
                    max_score[0] = T.float32(-3.4e38)
                    for key_token in T.serial(_VIT_S_TOKENS):
                        max_score[0] = T.max(
                            max_score[0],
                            scores[0, head, token, key_token],
                        )
                    denom[0] = T.float32(0.0)
                    for key_token in T.serial(_VIT_S_TOKENS):
                        scores[0, head, token, key_token] = T.exp(
                            scores[0, head, token, key_token] - max_score[0]
                        )
                        denom[0] = denom[0] + scores[0, head, token, key_token]
                    for key_token in T.serial(_VIT_S_TOKENS):
                        scores[0, head, token, key_token] = (
                            scores[0, head, token, key_token] / denom[0]
                        )

    @T.prim_func
    def apply_prim(
        qkv: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, 3 * _VIT_S_HIDDEN), "float32"),
        weights: T.Buffer((_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS), "float32"),
        out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        acc = T.alloc_buffer((1,), scope="local")
        for bx in T.thread_binding(apply_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                idx = bx * threads + tx
                if idx < apply_total:
                    token = idx // _VIT_S_HIDDEN
                    hidden = idx % _VIT_S_HIDDEN
                    head = hidden // _VIT_S_HEAD_DIM
                    dim = hidden % _VIT_S_HEAD_DIM
                    acc[0] = T.float32(0.0)
                    for key_token in T.serial(_VIT_S_TOKENS):
                        acc[0] = (
                            acc[0]
                            + weights[0, head, token, key_token]
                            * qkv[
                                0,
                                key_token,
                                2 * _VIT_S_HIDDEN + head * _VIT_S_HEAD_DIM + dim,
                            ]
                        )
                    out[0, token, hidden] = acc[0]

    qkv_id = request.input_buffer_ids[0]
    out_id = request.output_buffer_ids[0]
    lowering_plan = (
        LoweringStep(
            name=f"{request.operator_id}.{artifact_source}.score",
            contract_id=request.lowering_contract_ids[0],
            primfunc_name=score_name,
            input_buffer_ids=(qkv_id,),
            output_buffer_ids=(),
            intermediate_buffer_ids=(scores_id,),
            arg_buffer_order=(qkv_id, scores_id),
            attrs={"artifact_source": artifact_source, "attention_step": "score"},
        ),
        LoweringStep(
            name=f"{request.operator_id}.{artifact_source}.softmax",
            contract_id=request.lowering_contract_ids[0],
            primfunc_name=softmax_name,
            input_buffer_ids=(scores_id,),
            output_buffer_ids=(),
            intermediate_buffer_ids=(scores_id,),
            arg_buffer_order=(scores_id,),
            attrs={"artifact_source": artifact_source, "attention_step": "softmax"},
        ),
        LoweringStep(
            name=f"{request.operator_id}.{artifact_source}.apply",
            contract_id=request.lowering_contract_ids[0],
            primfunc_name=apply_name,
            input_buffer_ids=(qkv_id, scores_id),
            output_buffer_ids=(out_id,),
            intermediate_buffer_ids=(),
            arg_buffer_order=(qkv_id, scores_id, out_id),
            attrs={"artifact_source": artifact_source, "attention_step": "apply"},
        ),
    )
    return (
        tvm.IRModule(
            {
                score_name: score_prim.with_attr("global_symbol", score_name),
                softmax_name: softmax_prim.with_attr("global_symbol", softmax_name),
                apply_name: apply_prim.with_attr("global_symbol", apply_name),
            }
        ),
        lowering_plan,
        {
            scores_id: {
                "shape": (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
                "dtype": "float32",
                "layout": "row_major",
            }
        },
        (),
    )


def _resolve_vit_s_operator_autotune_config(
    *,
    target: str,
    config: VitSOperatorAutotuneConfig | None,
) -> VitSOperatorAutotuneConfig:
    if config is not None:
        return config
    if _is_cuda_target(target):
        return VitSOperatorAutotuneConfig(
            enabled=False,
            work_dir=VIT_S_DEFAULT_OPERATOR_AUTOTUNE_WORK_DIR,
            max_trials_global=0,
            max_trials_per_task=0,
            num_trials_per_iter=1,
            cost_model="random",
            num_tuning_cores=1,
            task_scheduler="gradient",
            strategy="evolutionary",
            seed=None,
            force_retune=False,
        )
    return VitSOperatorAutotuneConfig(enabled=False)


def _vit_s_is_non_matmul_te_operator(operator_id: str) -> bool:
    return (
        operator_id in {"patch_embed", "class_token_concat", "position_add"}
        or operator_id.endswith("_residual_after_attention")
        or operator_id.endswith("_residual_after_mlp")
        or operator_id.endswith("_norm0")
        or operator_id.endswith("_norm1")
        or operator_id == "final_norm"
        or operator_id.endswith("_mlp_gelu")
    )


def _vit_s_is_matmul_operator(operator_id: str) -> bool:
    return (
        operator_id.endswith("_qkv_projection")
        or operator_id.endswith("_attention_projection")
        or operator_id.endswith("_mlp_fc0")
        or operator_id.endswith("_mlp_fc1")
        or operator_id == "classifier_head"
    )


def _vit_s_autotune_workload_specs(
    operator_id: str,
    *,
    gelu_approximation: str,
    layer_norm_epsilon: float,
) -> tuple[AutotuneWorkloadSpec, ...]:
    if _vit_s_is_matmul_operator(operator_id):
        input_dim, output_dim, output_kind = _vit_s_matmul_dims_for_operator(operator_id)
        workload_key = _vit_s_matmul_shape_class(
            output_kind=output_kind,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return (
            AutotuneWorkloadSpec(
                workload_key=workload_key,
                operator_id=operator_id,
                operator_family="matmul",
            ),
        )
    if _vit_s_is_non_matmul_te_operator(operator_id):
        workload_key, operator_family = _vit_s_non_matmul_workload_key(
            operator_id,
            gelu_approximation=gelu_approximation,
            layer_norm_epsilon=layer_norm_epsilon,
        )
        return (
            AutotuneWorkloadSpec(
                workload_key=workload_key,
                operator_id=operator_id,
                operator_family=operator_family,
            ),
        )
    if operator_id.endswith("_attention"):
        return tuple(
            AutotuneWorkloadSpec(
                workload_key=_vit_s_attention_step_workload_key(step_name),
                operator_id=operator_id,
                operator_family=f"attention_{step_name}",
                attrs={"attention_step": step_name},
            )
            for step_name in ("score", "softmax", "apply")
        )
    raise ValueError(f"unsupported ViT-S autotune operator: {operator_id}")


def _vit_s_matmul_dims_for_operator(operator_id: str) -> tuple[int, int, str]:
    if operator_id.endswith("_qkv_projection"):
        return _VIT_S_HIDDEN, 3 * _VIT_S_HIDDEN, "tokens"
    if operator_id.endswith("_attention_projection"):
        return _VIT_S_HIDDEN, _VIT_S_HIDDEN, "tokens"
    if operator_id.endswith("_mlp_fc0"):
        return _VIT_S_HIDDEN, _VIT_S_MLP, "tokens"
    if operator_id.endswith("_mlp_fc1"):
        return _VIT_S_MLP, _VIT_S_HIDDEN, "tokens"
    if operator_id == "classifier_head":
        return _VIT_S_HIDDEN, _VIT_S_CLASSES, "classifier"
    raise ValueError(f"unsupported ViT-S matmul operator: {operator_id}")


def _vit_s_non_matmul_workload_key(
    operator_id: str,
    *,
    gelu_approximation: str,
    layer_norm_epsilon: float,
) -> tuple[str, str]:
    if operator_id == "patch_embed":
        return "patch_embed_b1_c3_i224_p16_h384", "patch_embed"
    if operator_id == "class_token_concat":
        return "class_token_concat_b1_tokens197_hidden384", "class_token_concat"
    if (
        operator_id == "position_add"
        or operator_id.endswith("_residual_after_attention")
        or operator_id.endswith("_residual_after_mlp")
    ):
        return "add_b1_tokens197_hidden384", "add"
    if (
        operator_id.endswith("_norm0")
        or operator_id.endswith("_norm1")
        or operator_id == "final_norm"
    ):
        return (
            "layer_norm_b1_tokens197_hidden384_eps"
            f"{_vit_s_float_token(layer_norm_epsilon)}",
            "layer_norm",
        )
    if operator_id.endswith("_mlp_gelu"):
        return f"gelu_{gelu_approximation}_b1_tokens197_mlp1536", "gelu"
    raise ValueError(f"unsupported ViT-S non-matmul autotune operator: {operator_id}")


def _vit_s_attention_step_workload_key(step_name: str) -> str:
    if step_name == "score":
        return "attention_score_b1_heads6_tokens197_head64"
    if step_name == "softmax":
        return "attention_softmax_b1_heads6_tokens197"
    if step_name == "apply":
        return "attention_apply_b1_heads6_tokens197_hidden384"
    raise ValueError(f"unsupported ViT-S attention TE step: {step_name}")


def _effective_vit_s_matmul_autotune_config(
    *,
    target: str,
    operator_id: str,
    legacy_config: VitSMatmulAutotuneConfig | None,
    autotune_config: AutotuneConfig | None,
    selected_operator_ids: set[str] | None,
) -> VitSMatmulAutotuneConfig | None:
    if autotune_config is None:
        return legacy_config
    if (
        autotune_config.enabled
        and selected_operator_ids is not None
        and operator_id in selected_operator_ids
        and _vit_s_is_matmul_operator(operator_id)
    ):
        return _autotune_config_to_vit_s_matmul_config(autotune_config)
    return VitSMatmulAutotuneConfig(enabled=False)


def _effective_vit_s_operator_autotune_config(
    *,
    target: str,
    operator_id: str,
    legacy_config: VitSOperatorAutotuneConfig | None,
    autotune_config: AutotuneConfig | None,
    selected_operator_ids: set[str] | None,
) -> VitSOperatorAutotuneConfig | None:
    if autotune_config is None:
        return legacy_config
    if (
        autotune_config.enabled
        and selected_operator_ids is not None
        and operator_id in selected_operator_ids
        and _is_cuda_target(target)
    ):
        return _autotune_config_to_vit_s_operator_config(autotune_config)
    return VitSOperatorAutotuneConfig(enabled=False)


def _autotune_config_to_vit_s_matmul_config(
    config: AutotuneConfig,
) -> VitSMatmulAutotuneConfig:
    return VitSMatmulAutotuneConfig(
        enabled=config.enabled,
        work_dir=config.work_dir,
        max_trials_global=config.max_trials_global,
        max_trials_per_task=config.max_trials_per_task,
        num_trials_per_iter=config.num_trials_per_iter,
        cost_model=config.cost_model,
        num_tuning_cores=config.num_tuning_cores,
        task_scheduler=config.task_scheduler,
        strategy=config.strategy,
        seed=config.seed,
        force_retune=config.force_retune,
    )


def _autotune_config_to_vit_s_operator_config(
    config: AutotuneConfig,
) -> VitSOperatorAutotuneConfig:
    return VitSOperatorAutotuneConfig(
        enabled=config.enabled,
        work_dir=config.work_dir,
        max_trials_global=config.max_trials_global,
        max_trials_per_task=config.max_trials_per_task,
        num_trials_per_iter=config.num_trials_per_iter,
        cost_model=config.cost_model,
        num_tuning_cores=config.num_tuning_cores,
        task_scheduler=config.task_scheduler,
        strategy=config.strategy,
        seed=config.seed,
        force_retune=config.force_retune,
    )


def _vit_s_baseline_codegen_records(
    operator_id: str,
    *,
    gelu_approximation: str,
    layer_norm_epsilon: float,
) -> tuple[dict[str, Any], ...]:
    records = []
    for spec in _vit_s_autotune_workload_specs(
        operator_id,
        gelu_approximation=gelu_approximation,
        layer_norm_epsilon=layer_norm_epsilon,
    ):
        record = {
            "codegen_source": "baseline_tir",
            "workload_class": spec.workload_key,
            "operator_family": spec.operator_family,
            "operator_id": operator_id,
            "tuned": False,
            "autotune_enabled": False,
            "work_dir": None,
            "record_hash": None,
            "record_count": 0,
        }
        record["codegen_id"] = stable_hash(
            {
                "operator_id": operator_id,
                "workload_class": spec.workload_key,
                "source": record["codegen_source"],
            }
        )
        records.append(record)
    return tuple(records)


def _vit_s_operator_te_tir_module(
    *,
    operator_id: str,
    name: str,
    target: str,
    gelu_approximation: str,
    layer_norm_epsilon: float,
    autotune_config: VitSOperatorAutotuneConfig,
    codegen_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[Any, str, dict[str, Any]]:
    canonical_mod, workload_class, operator_family = _vit_s_non_matmul_te_workload_module(
        operator_id=operator_id,
        gelu_approximation=gelu_approximation,
        layer_norm_epsilon=layer_norm_epsilon,
    )
    cache_key = stable_hash(
        {
            "workload_class": workload_class,
            "target": target,
            "autotune_enabled": autotune_config.enabled,
            "work_dir": autotune_config.work_dir,
            "max_trials_global": autotune_config.max_trials_global,
            "max_trials_per_task": autotune_config.max_trials_per_task,
            "num_trials_per_iter": autotune_config.num_trials_per_iter,
            "cost_model": autotune_config.cost_model,
            "num_tuning_cores": autotune_config.num_tuning_cores,
            "task_scheduler": autotune_config.task_scheduler,
            "strategy": autotune_config.strategy,
            "seed": autotune_config.seed,
            "force_retune": autotune_config.force_retune,
        }
    )
    cached = codegen_cache.get(cache_key)
    if cached is None:
        scheduled_mod, codegen_meta = _vit_s_apply_meta_schedule_to_operator_te(
            canonical_mod,
            workload_class=workload_class,
            operator_family=operator_family,
            target=target,
            config=autotune_config,
        )
        codegen_cache[cache_key] = (scheduled_mod, codegen_meta)
    else:
        scheduled_mod, codegen_meta = cached
    operator_meta = dict(codegen_meta)
    operator_meta["operator_id"] = operator_id
    operator_meta["codegen_id"] = stable_hash(
        {
            "operator_id": operator_id,
            "workload_class": workload_class,
            "source": operator_meta["codegen_source"],
            "record_hash": operator_meta.get("record_hash"),
        }
    )
    return _rename_single_primfunc_module(scheduled_mod, name), name, operator_meta


def _vit_s_non_matmul_te_workload_module(
    *,
    operator_id: str,
    gelu_approximation: str,
    layer_norm_epsilon: float,
) -> tuple[Any, str, str]:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    if operator_id == "patch_embed":
        image = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_CHANNELS, _VIT_S_IMAGE, _VIT_S_IMAGE),
            name="image",
            dtype="float32",
        )
        weight = te.placeholder(
            (_VIT_S_HIDDEN, _VIT_S_CHANNELS, _VIT_S_PATCH, _VIT_S_PATCH),
            name="weight",
            dtype="float32",
        )
        bias = te.placeholder((_VIT_S_HIDDEN,), name="bias", dtype="float32")
        rc = te.reduce_axis((0, _VIT_S_CHANNELS), name="rc")
        ry = te.reduce_axis((0, _VIT_S_PATCH), name="ry")
        rx = te.reduce_axis((0, _VIT_S_PATCH), name="rx")
        conv = te.compute(
            (_VIT_S_BATCH, _VIT_S_PATCH_TOKENS, _VIT_S_HIDDEN),
            lambda batch, token, hidden: te.sum(
                image[
                    batch,
                    rc,
                    (token // _VIT_S_PATCH_GRID) * _VIT_S_PATCH + ry,
                    (token % _VIT_S_PATCH_GRID) * _VIT_S_PATCH + rx,
                ]
                * weight[hidden, rc, ry, rx],
                axis=(rc, ry, rx),
            ),
            name="conv",
        )
        out = te.compute(
            (_VIT_S_BATCH, _VIT_S_PATCH_TOKENS, _VIT_S_HIDDEN),
            lambda batch, token, hidden: conv[batch, token, hidden] + bias[hidden],
            name="out",
        )
        args = [image, weight, bias, out]
        workload_class = "patch_embed_b1_c3_i224_p16_h384"
        operator_family = "patch_embed"
    elif operator_id == "class_token_concat":
        x = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_PATCH_TOKENS, _VIT_S_HIDDEN),
            name="x",
            dtype="float32",
        )
        cls = te.placeholder(
            (_VIT_S_BATCH, 1, _VIT_S_HIDDEN),
            name="cls",
            dtype="float32",
        )
        out = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            lambda batch, token, hidden: te.if_then_else(
                token == 0,
                cls[batch, 0, hidden],
                x[batch, token - 1, hidden],
            ),
            name="out",
        )
        args = [x, cls, out]
        workload_class = "class_token_concat_b1_tokens197_hidden384"
        operator_family = "class_token_concat"
    elif (
        operator_id == "position_add"
        or operator_id.endswith("_residual_after_attention")
        or operator_id.endswith("_residual_after_mlp")
    ):
        a = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            name="a",
            dtype="float32",
        )
        b = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            name="b",
            dtype="float32",
        )
        out = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            lambda batch, token, hidden: a[batch, token, hidden] + b[batch, token, hidden],
            name="out",
        )
        args = [a, b, out]
        workload_class = "add_b1_tokens197_hidden384"
        operator_family = "add"
    elif (
        operator_id.endswith("_norm0")
        or operator_id.endswith("_norm1")
        or operator_id == "final_norm"
    ):
        x = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            name="x",
            dtype="float32",
        )
        scale = te.placeholder((_VIT_S_HIDDEN,), name="scale", dtype="float32")
        bias = te.placeholder((_VIT_S_HIDDEN,), name="bias", dtype="float32")
        r_mean = te.reduce_axis((0, _VIT_S_HIDDEN), name="r_mean")
        mean = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS),
            lambda batch, token: te.sum(
                x[batch, token, r_mean] * (1.0 / _VIT_S_HIDDEN),
                axis=r_mean,
            ),
            name="mean",
        )
        r_var = te.reduce_axis((0, _VIT_S_HIDDEN), name="r_var")
        var = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS),
            lambda batch, token: te.sum(
                (x[batch, token, r_var] - mean[batch, token])
                * (x[batch, token, r_var] - mean[batch, token])
                * (1.0 / _VIT_S_HIDDEN),
                axis=r_var,
            ),
            name="var",
        )
        out = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            lambda batch, token, hidden: (
                (x[batch, token, hidden] - mean[batch, token])
                / te.sqrt(var[batch, token] + layer_norm_epsilon)
                * scale[hidden]
                + bias[hidden]
            ),
            name="out",
        )
        args = [x, scale, bias, out]
        workload_class = (
            "layer_norm_b1_tokens197_hidden384_eps"
            f"{_vit_s_float_token(layer_norm_epsilon)}"
        )
        operator_family = "layer_norm"
    elif operator_id.endswith("_mlp_gelu"):
        x = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_MLP),
            name="x",
            dtype="float32",
        )
        if gelu_approximation == "none":
            out = te.compute(
                (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_MLP),
                lambda batch, token, hidden: 0.5
                * x[batch, token, hidden]
                * (
                    1.0
                    + te.erf(x[batch, token, hidden] * (1.0 / math.sqrt(2.0)))
                ),
                name="out",
            )
        else:
            out = te.compute(
                (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_MLP),
                lambda batch, token, hidden: 0.5
                * x[batch, token, hidden]
                * (
                    1.0
                    + te.tanh(
                        math.sqrt(2.0 / math.pi)
                        * (
                            x[batch, token, hidden]
                            + 0.044715
                            * x[batch, token, hidden]
                            * x[batch, token, hidden]
                            * x[batch, token, hidden]
                        )
                    )
                ),
                name="out",
            )
        args = [x, out]
        workload_class = f"gelu_{gelu_approximation}_b1_tokens197_mlp1536"
        operator_family = "gelu"
    else:
        raise ValueError(f"unsupported ViT-S TE autotune operator: {operator_id}")
    prim = te.create_prim_func(args).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": prim}), workload_class, operator_family


def _vit_s_attention_te_tir_module_and_plan(
    request: OperatorExecutionRequest,
    *,
    name_prefix: str,
    artifact_source: str,
    target: str,
    autotune_config: VitSOperatorAutotuneConfig,
    codegen_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[Any, tuple[LoweringStep, ...], dict[str, Mapping[str, Any]], tuple[dict[str, Any], ...]]:
    import tvm  # pylint: disable=import-outside-toplevel

    qkv_id = request.input_buffer_ids[0]
    out_id = request.output_buffer_ids[0]
    scores_id = f"{request.operator_id}.attention_scores"
    weights_id = f"{request.operator_id}.attention_weights"
    step_specs = (
        (
            "score",
            f"{name_prefix}_{request.operator_id}_score",
            (qkv_id,),
            (),
            (scores_id,),
            (qkv_id, scores_id),
        ),
        (
            "softmax",
            f"{name_prefix}_{request.operator_id}_softmax",
            (scores_id,),
            (),
            (weights_id,),
            (scores_id, weights_id),
        ),
        (
            "apply",
            f"{name_prefix}_{request.operator_id}_apply",
            (qkv_id, weights_id),
            (out_id,),
            (),
            (qkv_id, weights_id, out_id),
        ),
    )
    functions = {}
    lowering_steps = []
    codegen_records = []
    for step_name, primfunc_name, input_ids, output_ids, intermediate_ids, arg_order in step_specs:
        canonical_mod, workload_class, operator_family = _vit_s_attention_step_te_workload_module(
            step_name
        )
        cache_key = stable_hash(
            {
                "workload_class": workload_class,
                "target": target,
                "autotune_enabled": autotune_config.enabled,
                "work_dir": autotune_config.work_dir,
                "max_trials_global": autotune_config.max_trials_global,
                "max_trials_per_task": autotune_config.max_trials_per_task,
                "num_trials_per_iter": autotune_config.num_trials_per_iter,
                "cost_model": autotune_config.cost_model,
                "num_tuning_cores": autotune_config.num_tuning_cores,
                "task_scheduler": autotune_config.task_scheduler,
                "strategy": autotune_config.strategy,
                "seed": autotune_config.seed,
                "force_retune": autotune_config.force_retune,
            }
        )
        cached = codegen_cache.get(cache_key)
        if cached is None:
            scheduled_mod, codegen_meta = _vit_s_apply_meta_schedule_to_operator_te(
                canonical_mod,
                workload_class=workload_class,
                operator_family=operator_family,
                target=target,
                config=autotune_config,
            )
            codegen_cache[cache_key] = (scheduled_mod, codegen_meta)
        else:
            scheduled_mod, codegen_meta = cached
        renamed = _rename_single_primfunc_module(scheduled_mod, primfunc_name)
        functions[primfunc_name] = renamed[primfunc_name]
        operator_meta = dict(codegen_meta)
        operator_meta["operator_id"] = request.operator_id
        operator_meta["attention_step"] = step_name
        operator_meta["codegen_id"] = stable_hash(
            {
                "operator_id": request.operator_id,
                "attention_step": step_name,
                "workload_class": workload_class,
                "source": operator_meta["codegen_source"],
                "record_hash": operator_meta.get("record_hash"),
            }
        )
        codegen_records.append(operator_meta)
        lowering_steps.append(
            LoweringStep(
                name=f"{request.operator_id}.{artifact_source}.{step_name}",
                contract_id=request.lowering_contract_ids[0],
                primfunc_name=primfunc_name,
                input_buffer_ids=input_ids,
                output_buffer_ids=output_ids,
                intermediate_buffer_ids=intermediate_ids,
                arg_buffer_order=arg_order,
                attrs={"artifact_source": artifact_source, "attention_step": step_name},
            )
        )
    extra_buffer_meta = {
        scores_id: {
            "shape": (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            "dtype": "float32",
            "layout": "row_major",
        },
        weights_id: {
            "shape": (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            "dtype": "float32",
            "layout": "row_major",
        },
    }
    return tvm.IRModule(functions), tuple(lowering_steps), extra_buffer_meta, tuple(codegen_records)


def _vit_s_attention_step_te_workload_module(step_name: str) -> tuple[Any, str, str]:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    if step_name == "score":
        qkv = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_TOKENS, 3 * _VIT_S_HIDDEN),
            name="qkv",
            dtype="float32",
        )
        rk = te.reduce_axis((0, _VIT_S_HEAD_DIM), name="rk")
        scores = te.compute(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            lambda batch, head, token, key_token: te.sum(
                qkv[batch, token, head * _VIT_S_HEAD_DIM + rk]
                * qkv[
                    batch,
                    key_token,
                    _VIT_S_HIDDEN + head * _VIT_S_HEAD_DIM + rk,
                ]
                * (1.0 / math.sqrt(float(_VIT_S_HEAD_DIM))),
                axis=rk,
            ),
            name="out",
        )
        args = [qkv, scores]
        workload_class = "attention_score_b1_heads6_tokens197_head64"
        operator_family = "attention_score"
    elif step_name == "softmax":
        scores = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            name="scores",
            dtype="float32",
        )
        r_max = te.reduce_axis((0, _VIT_S_TOKENS), name="r_max")
        score_max = te.compute(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS),
            lambda batch, head, token: te.max(scores[batch, head, token, r_max], axis=r_max),
            name="score_max",
        )
        exp_scores = te.compute(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            lambda batch, head, token, key_token: te.exp(
                scores[batch, head, token, key_token] - score_max[batch, head, token]
            ),
            name="exp_scores",
        )
        r_denom = te.reduce_axis((0, _VIT_S_TOKENS), name="r_denom")
        denom = te.compute(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS),
            lambda batch, head, token: te.sum(
                exp_scores[batch, head, token, r_denom],
                axis=r_denom,
            ),
            name="denom",
        )
        weights = te.compute(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            lambda batch, head, token, key_token: exp_scores[
                batch, head, token, key_token
            ]
            / denom[batch, head, token],
            name="out",
        )
        args = [scores, weights]
        workload_class = "attention_softmax_b1_heads6_tokens197"
        operator_family = "attention_softmax"
    elif step_name == "apply":
        qkv = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_TOKENS, 3 * _VIT_S_HIDDEN),
            name="qkv",
            dtype="float32",
        )
        weights = te.placeholder(
            (_VIT_S_BATCH, _VIT_S_HEADS, _VIT_S_TOKENS, _VIT_S_TOKENS),
            name="weights",
            dtype="float32",
        )
        r_token = te.reduce_axis((0, _VIT_S_TOKENS), name="r_token")
        out = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS, _VIT_S_HIDDEN),
            lambda batch, token, hidden: te.sum(
                weights[batch, hidden // _VIT_S_HEAD_DIM, token, r_token]
                * qkv[
                    batch,
                    r_token,
                    2 * _VIT_S_HIDDEN
                    + (hidden // _VIT_S_HEAD_DIM) * _VIT_S_HEAD_DIM
                    + hidden % _VIT_S_HEAD_DIM,
                ],
                axis=r_token,
            ),
            name="out",
        )
        args = [qkv, weights, out]
        workload_class = "attention_apply_b1_heads6_tokens197_hidden384"
        operator_family = "attention_apply"
    else:
        raise ValueError(f"unsupported ViT-S attention TE step: {step_name}")
    prim = te.create_prim_func(args).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": prim}), workload_class, operator_family


def _vit_s_apply_meta_schedule_to_operator_te(
    canonical_mod: Any,
    *,
    workload_class: str,
    operator_family: str,
    target: str,
    config: VitSOperatorAutotuneConfig,
) -> tuple[Any, dict[str, Any]]:
    autotune_result = tune_or_load_workload(
        AutotuneWorkloadSpec(
            workload_key=workload_class,
            operator_id=workload_class,
            operator_family=operator_family,
            tir_module=canonical_mod,
        ),
        target=target,
        config=AutotuneConfig(
            enabled=config.enabled,
            work_dir=config.work_dir or VIT_S_DEFAULT_OPERATOR_AUTOTUNE_WORK_DIR,
            max_trials_global=config.max_trials_global,
            max_trials_per_task=config.max_trials_per_task,
            num_trials_per_iter=config.num_trials_per_iter,
            cost_model=config.cost_model,
            num_tuning_cores=config.num_tuning_cores,
            task_scheduler=config.task_scheduler,
            strategy=config.strategy,
            seed=config.seed,
            force_retune=config.force_retune,
        ),
    )
    return autotune_result.tuned_module, dict(autotune_result.metadata)


def _vit_s_float_token(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def _resolve_vit_s_matmul_autotune_config(
    *,
    target: str,
    config: VitSMatmulAutotuneConfig | None,
) -> VitSMatmulAutotuneConfig:
    if config is not None:
        return config
    if _is_cuda_target(target):
        return VitSMatmulAutotuneConfig(
            enabled=True,
            work_dir=VIT_S_DEFAULT_MATMUL_AUTOTUNE_WORK_DIR,
            max_trials_global=1,
            max_trials_per_task=1,
            num_trials_per_iter=1,
            cost_model="random",
            num_tuning_cores=1,
            task_scheduler="gradient",
            strategy="evolutionary",
            seed=None,
            force_retune=False,
        )
    return VitSMatmulAutotuneConfig(enabled=False)


def _is_cuda_target(target: str) -> bool:
    return str(target).startswith("cuda")


def _vit_s_matmul_te_tir_module(
    *,
    operator_id: str,
    input_dim: int,
    output_dim: int,
    name: str,
    output_kind: str,
    target: str,
    autotune_config: VitSMatmulAutotuneConfig | None,
    codegen_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[Any, str, dict[str, Any]]:
    config = autotune_config or VitSMatmulAutotuneConfig(enabled=False)
    shape_class = _vit_s_matmul_shape_class(
        output_kind=output_kind,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    cache_key = stable_hash(
        {
            "shape_class": shape_class,
            "target": target,
            "autotune_enabled": config.enabled,
            "work_dir": config.work_dir,
            "max_trials_global": config.max_trials_global,
            "max_trials_per_task": config.max_trials_per_task,
            "num_trials_per_iter": config.num_trials_per_iter,
            "cost_model": config.cost_model,
            "num_tuning_cores": config.num_tuning_cores,
            "task_scheduler": config.task_scheduler,
            "strategy": config.strategy,
            "seed": config.seed,
            "force_retune": config.force_retune,
        }
    )
    cached = codegen_cache.get(cache_key)
    if cached is None:
        canonical_mod = _vit_s_matmul_te_workload_module(
            input_dim=input_dim,
            output_dim=output_dim,
            output_kind=output_kind,
            name="main",
        )
        if config.enabled and _is_cuda_target(target):
            scheduled_mod, codegen_meta = _vit_s_apply_meta_schedule_to_matmul_te(
                canonical_mod,
                shape_class=shape_class,
                target=target,
                config=config,
            )
        else:
            scheduled_mod = (
                _vit_s_matmul_cuda_baseline_tir_module(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    output_kind=output_kind,
                )
                if _is_cuda_target(target)
                else canonical_mod
            )
            codegen_meta = {
                "codegen_source": VIT_S_MATMUL_CODEGEN_TE_WORKLOAD,
                "shape_class": shape_class,
                "tuned": False,
                "autotune_enabled": config.enabled,
                "work_dir": config.work_dir,
                "record_hash": None,
                "record_count": 0,
            }
        codegen_cache[cache_key] = (scheduled_mod, codegen_meta)
    else:
        scheduled_mod, codegen_meta = cached
    operator_meta = dict(codegen_meta)
    operator_meta["operator_id"] = operator_id
    operator_meta["workload_class"] = shape_class
    operator_meta["operator_family"] = "matmul"
    operator_meta["codegen_id"] = stable_hash(
        {
            "operator_id": operator_id,
            "shape_class": shape_class,
            "source": operator_meta["codegen_source"],
            "record_hash": operator_meta.get("record_hash"),
        }
    )
    return _rename_single_primfunc_module(scheduled_mod, name), name, operator_meta


def _vit_s_matmul_cuda_baseline_tir_module(
    *,
    input_dim: int,
    output_dim: int,
    output_kind: str,
) -> Any:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel

    threads = _VIT_S_CUDA_THREADS_PER_BLOCK
    token_count = 1 if output_kind == "classifier" else _VIT_S_TOKENS
    total = _VIT_S_BATCH * token_count * output_dim
    nblocks = (total + threads - 1) // threads

    if output_kind == "classifier":

        @T.prim_func
        def prim(
            lhs: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, input_dim), "float32"),
            rhs: T.Buffer((input_dim, output_dim), "float32"),
            bias: T.Buffer((output_dim,), "float32"),
            out: T.Buffer((_VIT_S_BATCH, output_dim), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            acc = T.alloc_buffer((1,), scope="local")
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        hidden = idx % output_dim
                        acc[0] = T.float32(0.0)
                        for k in T.serial(input_dim):
                            acc[0] = acc[0] + lhs[0, 0, k] * rhs[k, hidden]
                        out[0, hidden] = acc[0] + bias[hidden]

    else:

        @T.prim_func
        def prim(
            lhs: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, input_dim), "float32"),
            rhs: T.Buffer((input_dim, output_dim), "float32"),
            bias: T.Buffer((output_dim,), "float32"),
            out: T.Buffer((_VIT_S_BATCH, _VIT_S_TOKENS, output_dim), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            acc = T.alloc_buffer((1,), scope="local")
            for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
                for tx in T.thread_binding(threads, thread="threadIdx.x"):
                    idx = bx * threads + tx
                    if idx < total:
                        token = (idx // output_dim) % _VIT_S_TOKENS
                        hidden = idx % output_dim
                        acc[0] = T.float32(0.0)
                        for k in T.serial(input_dim):
                            acc[0] = acc[0] + lhs[0, token, k] * rhs[k, hidden]
                        out[0, token, hidden] = acc[0] + bias[hidden]

    return tvm.IRModule({"main": prim.with_attr("global_symbol", "main")})


def _vit_s_matmul_shape_class(
    *,
    output_kind: str,
    input_dim: int,
    output_dim: int,
) -> str:
    if output_kind == "classifier":
        return f"classifier_m1_k{input_dim}_n{output_dim}"
    return f"tokens{_VIT_S_TOKENS}_k{input_dim}_n{output_dim}"


def _vit_s_matmul_te_workload_module(
    *,
    input_dim: int,
    output_dim: int,
    output_kind: str,
    name: str,
) -> Any:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    lhs = te.placeholder(
        (_VIT_S_BATCH, _VIT_S_TOKENS, input_dim),
        name="lhs",
        dtype="float32",
    )
    rhs = te.placeholder((input_dim, output_dim), name="rhs", dtype="float32")
    bias = te.placeholder((output_dim,), name="bias", dtype="float32")
    rk = te.reduce_axis((0, input_dim), name="k")
    if output_kind == "classifier":
        matmul = te.compute(
            (_VIT_S_BATCH, output_dim),
            lambda batch, hidden: te.sum(lhs[batch, 0, rk] * rhs[rk, hidden], axis=rk),
            name="matmul",
        )
        out = te.compute(
            (_VIT_S_BATCH, output_dim),
            lambda batch, hidden: matmul[batch, hidden] + bias[hidden],
            name="out",
        )
    else:
        matmul = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS, output_dim),
            lambda batch, token, hidden: te.sum(
                lhs[batch, token, rk] * rhs[rk, hidden],
                axis=rk,
            ),
            name="matmul",
        )
        out = te.compute(
            (_VIT_S_BATCH, _VIT_S_TOKENS, output_dim),
            lambda batch, token, hidden: matmul[batch, token, hidden] + bias[hidden],
            name="out",
        )
    prim = te.create_prim_func([lhs, rhs, bias, out]).with_attr("global_symbol", name)
    return tvm.IRModule({name: prim})


def _vit_s_apply_meta_schedule_to_matmul_te(
    canonical_mod: Any,
    *,
    shape_class: str,
    target: str,
    config: VitSMatmulAutotuneConfig,
) -> tuple[Any, dict[str, Any]]:
    autotune_result = tune_or_load_workload(
        AutotuneWorkloadSpec(
            workload_key=shape_class,
            operator_id=shape_class,
            operator_family="matmul",
            tir_module=canonical_mod,
        ),
        target=target,
        config=AutotuneConfig(
            enabled=config.enabled,
            work_dir=config.work_dir or VIT_S_DEFAULT_MATMUL_AUTOTUNE_WORK_DIR,
            max_trials_global=config.max_trials_global,
            max_trials_per_task=config.max_trials_per_task,
            num_trials_per_iter=config.num_trials_per_iter,
            cost_model=config.cost_model,
            num_tuning_cores=config.num_tuning_cores,
            task_scheduler=config.task_scheduler,
            strategy=config.strategy,
            seed=config.seed,
            force_retune=config.force_retune,
        ),
    )
    codegen_meta = dict(autotune_result.metadata)
    codegen_meta["shape_class"] = shape_class
    return autotune_result.tuned_module, codegen_meta


def _vit_s_meta_schedule_target(target: str) -> Any:
    return meta_schedule_target(target)


def _rename_single_primfunc_module(module: Any, name: str) -> Any:
    import tvm  # pylint: disable=import-outside-toplevel

    primfuncs = [
        func for _, func in module.functions.items() if "PrimFunc" in type(func).__name__
    ]
    if len(primfuncs) != 1:
        raise ValueError("expected a single PrimFunc module")
    return tvm.IRModule({name: primfuncs[0].with_attr("global_symbol", name)})


def _vit_s_matmul_codegen_summary(artifacts: Iterable[TIRRegionArtifact]) -> dict[str, Any]:
    matmul_records = [
        dict(artifact.region_meta["matmul_codegen"])
        for artifact in artifacts
        if "matmul_codegen" in artifact.region_meta
    ]
    shape_records: dict[str, dict[str, Any]] = {}
    for record in matmul_records:
        shape_records.setdefault(
            str(record["shape_class"]),
            {
                "shape_class": record["shape_class"],
                "codegen_source": record["codegen_source"],
                "tuned": record["tuned"],
                "work_dir": record["work_dir"],
                "record_hash": record["record_hash"],
                "record_count": record["record_count"],
            },
        )
    return {
        "matmul_operator_count": len(matmul_records),
        "shape_class_count": len(shape_records),
        "shape_classes": tuple(shape_records.values()),
        "codegen_source_breakdown": {
            source: sum(1 for record in matmul_records if record["codegen_source"] == source)
            for source in sorted({record["codegen_source"] for record in matmul_records})
        },
    }


def _vit_s_operator_codegen_summary(artifacts: Iterable[TIRRegionArtifact]) -> dict[str, Any]:
    operator_records = []
    for artifact in artifacts:
        records = artifact.region_meta.get("operator_codegen", ())
        if isinstance(records, Mapping):
            operator_records.append(dict(records))
        else:
            operator_records.extend(dict(record) for record in records)
    workload_records: dict[str, dict[str, Any]] = {}
    for record in operator_records:
        workload_records.setdefault(
            str(record["workload_class"]),
            {
                "workload_class": record["workload_class"],
                "operator_family": record["operator_family"],
                "codegen_source": record["codegen_source"],
                "tuned": record["tuned"],
                "work_dir": record["work_dir"],
                "record_hash": record["record_hash"],
                "record_count": record["record_count"],
            },
        )
    return {
        "operator_step_count": len(operator_records),
        "workload_class_count": len(workload_records),
        "workload_classes": tuple(workload_records.values()),
        "codegen_source_breakdown": {
            source: sum(1 for record in operator_records if record["codegen_source"] == source)
            for source in sorted({record["codegen_source"] for record in operator_records})
        },
        "operator_family_breakdown": {
            family: sum(1 for record in operator_records if record["operator_family"] == family)
            for family in sorted({record["operator_family"] for record in operator_records})
        },
    }


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


def _parse_tuning_cores(value: str) -> int | str:
    return parse_tuning_cores(value)


def load_vit_s_run_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the ViT-S model-specific JSON run config."""

    raw_config, config_hash = load_json_config(path)
    config_path = str(Path(path))
    required_top = {
        "schema_version",
        "model",
        "target",
        "route",
        "weight_source",
        "checks",
        "benchmark",
        "autotune",
        "out_dir",
    }
    missing = sorted(required_top - set(raw_config))
    if missing:
        raise ValueError(f"missing required config field(s): {', '.join(missing)}")
    if int(raw_config["schema_version"]) != 1:
        raise ValueError("unsupported ViT-S config schema_version")
    model = _require_mapping(raw_config, "model")
    if model.get("name") != "vit-s" or model.get("variant") != "vit_s_16_224":
        raise ValueError("ViT-S config requires model.name='vit-s' and variant='vit_s_16_224'")
    target = str(raw_config["target"])
    route = _require_mapping(raw_config, "route")
    if route.get("artifact_source") != TIR_ARTIFACT_SOURCE_ATOMIC_DAG_GATED_TE:
        raise ValueError("ViT-S config currently requires artifact_source='atomic_dag_gated_te_tir'")
    backend = str(route.get("backend"))
    if backend not in {VIT_S_BACKEND_TRITON_TVM, VIT_S_BACKEND_REFERENCE}:
        raise ValueError(f"unsupported ViT-S backend in config: {backend}")

    weight_source = _require_mapping(raw_config, "weight_source")
    weight_source_type = str(weight_source.get("type"))
    if weight_source_type not in vit_s_weight_source_registry().source_ids():
        raise ValueError(f"unknown ViT-S weight source in config: {weight_source_type}")
    weight_source_config: dict[str, Any] = {}
    if weight_source_type == VIT_S_WEIGHT_SOURCE_TIMM:
        timm_model_name = weight_source.get("timm_model_name", weight_source.get("model_name"))
        if timm_model_name is None:
            raise ValueError("timm weight_source requires timm_model_name")
        weight_source_config = {
            "model_name": timm_model_name,
            "pretrained": bool(weight_source.get("pretrained", False)),
            "checkpoint_path": weight_source.get("checkpoint_path"),
            "cache_dir": weight_source.get("cache_dir"),
            "hf_endpoint": weight_source.get("hf_endpoint", VIT_S_DEFAULT_HF_ENDPOINT),
            "image_source": weight_source.get("image_source", "synthetic_gradient"),
        }

    checks = _require_mapping(raw_config, "checks")
    torch_compile = checks.get("torch_compile", {"enabled": False})
    if not isinstance(torch_compile, Mapping):
        raise ValueError("checks.torch_compile must be an object")
    benchmark = _require_mapping(raw_config, "benchmark")
    autotune_section = _require_mapping(raw_config, "autotune")
    autotune_config = autotune_config_from_mapping(
        autotune_section,
        config_path=config_path,
        config_hash=config_hash,
    )
    if autotune_config.enabled and not _is_cuda_target(target):
        raise ValueError(f"ViT-S autotune requires a CUDA target, got {target}")
    if autotune_config.enabled:
        build_vit_s_autotune_plan(
            autotune_config=autotune_config,
            weight_bundle=None,
        )

    return {
        "target": target,
        "producer_version": raw_config.get("producer_version"),
        "out_dir": raw_config.get("out_dir"),
        "enable_torch_compile": bool(torch_compile.get("enabled", False)),
        "enable_classification": bool(checks.get("classification", False)),
        "enable_e2e_benchmark": bool(benchmark.get("enabled", False)),
        "enable_atomic_dag_gated_te_tir": bool(checks.get("diagnostic_e2e", False)),
        "vit_s_backend": backend,
        "torch_compile_device": torch_compile.get("device"),
        "torch_compile_rtol": float(torch_compile.get("rtol", 1.0e-3)),
        "torch_compile_atol": float(torch_compile.get("atol", 1.0e-3)),
        "benchmark_warmup": int(benchmark.get("warmup", 5)),
        "benchmark_repeat": int(benchmark.get("repeat", 20)),
        "benchmark_device": benchmark.get("device"),
        "enable_cuda_graph_replay": bool(benchmark.get("cuda_graph_replay", False)),
        "parameter_seed": int(raw_config.get("parameter_seed", 0)),
        "weight_source": weight_source_type,
        "weight_source_config": weight_source_config,
        "autotune_config": autotune_config,
    }


def run_vit_s_config_file(path: str | Path) -> dict[str, Any]:
    """Run the ViT-S route from a model-specific JSON config file."""

    kwargs = load_vit_s_run_config(path)
    return run_vit_s_fixed_shape_route(**kwargs)


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m tvm.contrib.triton_tvm.models.vit``."""

    parser = argparse.ArgumentParser(description="Run fixed-shape ViT through the active route")
    parser.add_argument("--config", required=True, help="Path to a ViT-S JSON run config.")
    args = parser.parse_args(argv)
    report = run_vit_s_config_file(args.config)
    comparison = report.get("torch_compile_comparison")
    comparison_suffix = (
        f" torch_compile={comparison['status']}" if comparison is not None else ""
    )
    classification = report.get("classification")
    classification_suffix = (
        f" top1={classification['top1']['index']}" if classification is not None else ""
    )
    benchmark = report.get("e2e_latency_benchmark")
    benchmark_suffix = ""
    if benchmark is not None:
        backend_key = "triton_tvm" if "triton_tvm" in benchmark["paths"] else "numpy_reference"
        backend_summary = benchmark["paths"][backend_key]["summary"]
        torch_summary = benchmark["paths"]["torch_compile"]["summary"]
        benchmark_suffix = (
            f" {backend_key}_p50_ms={backend_summary['median_ms']:.3f}"
            f" torch_compile_p50_ms={torch_summary['median_ms']:.3f}"
            f" speedup={benchmark['torch_compile_vs_backend_median_speedup']:.2f}x"
        )
    print(
        "ViT-S active route "
        f"status={report['status']} "
        f"admitted={report['runtime_admitted_operator_count']}/{report['operator_count']} "
        f"source_routes={report['source_route_record_count']} "
        f"gaps={report['gap_count']}"
        f"{comparison_suffix}"
        f"{classification_suffix}"
        f"{benchmark_suffix}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
