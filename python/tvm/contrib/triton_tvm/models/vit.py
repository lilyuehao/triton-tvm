# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Fixed-shape ViT runner for the active Triton-TVM alpha route."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..atomic import (
    ATOMIC_FAMILY_DOT,
    ATOMIC_FAMILY_MEMORY,
    build_atomic_dag,
    validate_source_atomic_join,
)
from ..contracts import validate_atomic_contract
from ..manifest import build_manifest_gap, build_operator_manifest_record
from ..registry import CapabilityRecord, CapabilityRegistry
from ..reports import build_alpha_smoke_report, write_json_report
from ..runtime import RuntimeAdmissionResult, admit_runtime
from ..semantic import build_semantic_region
from ..semantic.schema import SemanticRegionRecord
from ..source import SourceOrigin, captured_inductor_jit_kernel
from .adapters import ModelAdapter, ModelOperatorInput


VIT_TINY_FIXED_SHAPE_MODEL_ID = "vit_tiny_fixed_shape"
VIT_RUNNER_ID = "vit_fixed_shape_active_route_v1"
VIT_OPERATOR_COUNT = 16


@dataclass(frozen=True)
class VitOperatorSpec:
    """One fixed-shape ViT operator input for the active route."""

    operator_id: str
    contract_id: str
    function_name: str
    ttir: str
    shape_signature: Mapping[str, Any]
    dtype_signature: Mapping[str, Any]
    layout_signature: Mapping[str, Any]


@dataclass(frozen=True)
class VitRouteRecord:
    """Per-operator result from source materialization through admission."""

    operator_id: str
    execution_order: int
    source_id: str
    atomic_dag_hash: str
    contract_id: str
    validation_status: str
    semantic_region_key: str | None
    registry_supported: bool
    capability_id: str | None
    runtime_admission_status: str
    gap_reason: str | None


def build_vit_tiny_fixed_shape_adapter(
    *, target: str = "cuda", producer_version: str | None = None
) -> ModelAdapter:
    """Materialize fixed-shape ViT operator inputs without support claims."""

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
) -> dict[str, Any]:
    """Compile and run the fixed-shape ViT inventory through the active route."""

    adapter = build_vit_tiny_fixed_shape_adapter(
        target=target,
        producer_version=producer_version,
    )
    specs = {spec.operator_id: spec for spec in vit_tiny_fixed_shape_specs()}
    registry = default_vit_capability_registry()

    sources = []
    atomics = []
    semantics: list[SemanticRegionRecord] = []
    manifests = []
    gaps = []
    route_records = []

    for item in adapter.operators():
        spec = specs[item.operator_id]
        source = item.source
        sources.append(source)
        atomic = build_atomic_dag(source, item.ttir)
        validate_source_atomic_join(source, atomic)
        atomics.append(atomic)

        validation = validate_atomic_contract(
            source,
            atomic,
            contract_id=spec.contract_id,
            shape_constraints=spec.shape_signature,
            dtype_constraints=spec.dtype_signature,
            layout_constraints=spec.layout_signature,
        )

        semantic = None
        resolution = None
        admission = RuntimeAdmissionResult(False, "gap", validation.failure_reason)
        gap_reason = validation.failure_reason

        if validation.passed:
            semantic = build_semantic_region(source, atomic, validation)
            semantics.append(semantic)
            manifest_for_admission = build_operator_manifest_record(
                model_id=item.model_id,
                operator_id=item.operator_id,
                execution_order=item.execution_order,
                source=source,
                atomic=atomic,
                semantic_region=semantic,
                shape_signature=spec.shape_signature,
                dtype_signature=spec.dtype_signature,
                layout_signature=spec.layout_signature,
                runtime_admission_status="not_checked",
                optional_model_anchor=item.optional_model_anchor,
            )
            resolution = registry.resolve(
                source=source,
                atomic=atomic,
                semantic_region=semantic,
            )
            admission = admit_runtime(
                source=source,
                atomic=atomic,
                validation=validation,
                semantic_region=semantic,
                manifest_record=manifest_for_admission,
                registry_resolution=resolution,
            )
            gap_reason = admission.reason

        manifest = build_operator_manifest_record(
            model_id=item.model_id,
            operator_id=item.operator_id,
            execution_order=item.execution_order,
            source=source,
            atomic=atomic,
            semantic_region=semantic,
            shape_signature=spec.shape_signature,
            dtype_signature=spec.dtype_signature,
            layout_signature=spec.layout_signature,
            runtime_admission_status=admission.status,
            gap_reason=gap_reason,
            optional_model_anchor=item.optional_model_anchor,
        )
        manifests.append(manifest)
        if gap_reason is not None:
            gaps.append(
                build_manifest_gap(
                    model_id=item.model_id,
                    operator_id=item.operator_id,
                    source_id=source.source_id,
                    reason=gap_reason,
                )
            )

        route_records.append(
            VitRouteRecord(
                operator_id=item.operator_id,
                execution_order=item.execution_order,
                source_id=source.source_id,
                atomic_dag_hash=atomic.atomic_dag_hash,
                contract_id=spec.contract_id,
                validation_status=validation.validation_status,
                semantic_region_key=semantic.semantic_region_key if semantic is not None else None,
                registry_supported=bool(resolution.supported) if resolution is not None else False,
                capability_id=resolution.capability_id if resolution is not None else None,
                runtime_admission_status=admission.status,
                gap_reason=gap_reason,
            )
        )

    alpha_report = build_alpha_smoke_report(
        sources=sources,
        atomics=atomics,
        semantics=semantics,
        manifests=manifests,
        gaps=gaps,
    )
    admitted_count = sum(1 for record in route_records if record.runtime_admission_status == "admitted")
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
        "operator_count": len(route_records),
        "expected_operator_count": VIT_OPERATOR_COUNT,
        "atomic_dag_built_operator_count": len(atomics),
        "semantic_region_count": len(semantics),
        "runtime_admitted_operator_count": admitted_count,
        "gap_count": len(gaps),
        "status": "passed" if admitted_count == VIT_OPERATOR_COUNT and not gaps else "gap",
        "performance_claim": False,
        "operators": [asdict(record) for record in route_records],
        "alpha_report": alpha_report,
    }
    if out_dir is not None:
        write_json_report(Path(out_dir) / "report.json", report)
    return report


def default_vit_capability_registry() -> CapabilityRegistry:
    """Capability records for the fixed-shape ViT alpha runner."""

    return CapabilityRegistry(
        [
            CapabilityRecord(
                capability_id=f"vit_fixed_shape.{contract_id}.cuda",
                semantic_region_key=contract_id,
                required_atomic_families=required,
                shape_constraints={},
                dtype_constraints={},
                layout_constraints={},
                allowed_source_origins=(SourceOrigin.TORCH_INDUCTOR,),
                runtime_available=True,
            )
            for contract_id, required in (
                ("pointwise_flat", (ATOMIC_FAMILY_MEMORY,)),
                ("matmul", (ATOMIC_FAMILY_DOT,)),
                ("matmul_bias_epilogue", (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY)),
                ("conv2d_nchw_static", (ATOMIC_FAMILY_MEMORY,)),
                ("attention_decomposed", (ATOMIC_FAMILY_DOT, ATOMIC_FAMILY_MEMORY)),
            )
        ]
    )


def vit_tiny_fixed_shape_specs() -> tuple[VitOperatorSpec, ...]:
    """Return the active fixed-shape ViT operator inventory."""

    return (
        _spec("patch_embed", "conv2d_nchw_static", "vit_patch_embed", _memory_ttir("vit_patch_embed"), "nchw"),
        _spec("class_token_add", "pointwise_flat", "vit_class_token_add", _memory_ttir("vit_class_token_add"), "flat"),
        _spec("position_add", "pointwise_flat", "vit_position_add", _memory_ttir("vit_position_add"), "flat"),
        _spec("encoder_norm0", "pointwise_flat", "vit_encoder_norm0", _memory_ttir("vit_encoder_norm0"), "flat"),
        _spec("qkv_projection", "matmul_bias_epilogue", "vit_qkv_projection", _dot_store_ttir("vit_qkv_projection"), "row_major"),
        _spec("attention_qk", "attention_decomposed", "vit_attention_qk", _dot_store_ttir("vit_attention_qk"), "row_major"),
        _spec("attention_softmax", "pointwise_flat", "vit_attention_softmax", _memory_ttir("vit_attention_softmax"), "flat"),
        _spec("attention_av", "attention_decomposed", "vit_attention_av", _dot_store_ttir("vit_attention_av"), "row_major"),
        _spec("attention_projection", "matmul_bias_epilogue", "vit_attention_projection", _dot_store_ttir("vit_attention_projection"), "row_major"),
        _spec("residual_after_attention", "pointwise_flat", "vit_residual_after_attention", _memory_ttir("vit_residual_after_attention"), "flat"),
        _spec("encoder_norm1", "pointwise_flat", "vit_encoder_norm1", _memory_ttir("vit_encoder_norm1"), "flat"),
        _spec("mlp_fc0", "matmul_bias_epilogue", "vit_mlp_fc0", _dot_store_ttir("vit_mlp_fc0"), "row_major"),
        _spec("mlp_gelu", "pointwise_flat", "vit_mlp_gelu", _memory_ttir("vit_mlp_gelu"), "flat"),
        _spec("mlp_fc1", "matmul_bias_epilogue", "vit_mlp_fc1", _dot_store_ttir("vit_mlp_fc1"), "row_major"),
        _spec("residual_after_mlp", "pointwise_flat", "vit_residual_after_mlp", _memory_ttir("vit_residual_after_mlp"), "flat"),
        _spec("pooler", "matmul_bias_epilogue", "vit_pooler", _dot_store_ttir("vit_pooler"), "row_major"),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m tvm.contrib.triton_tvm.models.vit``."""

    parser = argparse.ArgumentParser(description="Run fixed-shape ViT through the active route")
    parser.add_argument("--target", default="cuda")
    parser.add_argument("--producer-version", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    report = run_vit_tiny_fixed_shape_route(
        target=args.target,
        producer_version=args.producer_version,
        out_dir=args.out_dir,
    )
    print(
        "ViT active route "
        f"status={report['status']} "
        f"admitted={report['runtime_admitted_operator_count']}/{report['operator_count']} "
        f"gaps={report['gap_count']}"
    )
    return 0 if report["status"] == "passed" else 1


def _spec(
    operator_id: str,
    contract_id: str,
    function_name: str,
    ttir: str,
    layout: str,
) -> VitOperatorSpec:
    return VitOperatorSpec(
        operator_id=operator_id,
        contract_id=contract_id,
        function_name=function_name,
        ttir=ttir,
        shape_signature={"batch": 1, "image": 32, "patch": 16, "hidden": 192},
        dtype_signature={"activation": "f32"},
        layout_signature={"layout": layout},
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
