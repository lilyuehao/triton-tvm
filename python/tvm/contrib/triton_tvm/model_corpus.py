# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""M6/M6.5 model-level TorchInductor corpus audit helpers.

The model corpus path intentionally audits native TorchInductor output instead
of promising fallback-free model execution.  Its job is to make blockers stable,
ranked, and reproducible for small external-library model fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .attention import (
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_EXTERN_PACKED_FUNC,
    ATTENTION_EXTERN_SYMBOL,
    ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA,
    ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_NONE,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    ATTENTION_REPORT_FIELDS,
    ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY,
    ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED,
    ATTENTION_RUNTIME_KIND_PROVIDER,
    ATTENTION_RUNTIME_REPLACEMENT,
    ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY,
    ATTENTION_RUNTIME_STATUS_DEFERRED,
    ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    ATTENTION_TARGET_CONTRACTS,
    wrapper_sdpa_attention_report_fields,
)
from .inductor import (
    InductorKernel,
    InductorTritonSource,
    audit_inductor_kernel,
    extract_inductor_triton_sources,
    extract_inductor_wrapper_extern_calls,
    is_inductor_pointwise_kernel,
    load_inductor_kernel,
)
from .matmul import (
    EXTERN_ADDMM_BIAS_PACKED_FUNC,
    EXTERN_ADDMM_BIAS_SYMBOL,
    EXTERN_GEMM_PROVIDER_NONE,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    EXTERN_GEMM_SYMBOL,
)
from .reporting import (
    build_capability_report,
    make_report_status,
    write_capability_report,
)
from .vision import (
    M11_GRID_FAMILIES,
    M11_GRID2D_ARTIFACT_READY,
    M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
    M11_GRID2D_PROVIDER_NATIVE_TVM,
    M11_GRID2D_READINESS_VERSION,
    M11_GRID2D_RUNTIME_READY,
    M11_GRID_REPORT_FIELDS,
    M11_GRID_TAXONOMY_VERSION,
    VISION_CONTRACT_CONV2D_NCHW_STATIC,
    VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
    VISION_CONTRACT_VERSION,
    VISION_EXTERN_PACKED_FUNC,
    VISION_EXTERN_SYMBOL,
    VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D,
    VISION_M11_4_INTERFACE_STATUS,
    VISION_M11_4_VIT_PATCH_DILATION,
    VISION_M11_4_VIT_PATCH_INPUT_SHAPE,
    VISION_M11_4_VIT_PATCH_OUTPUT_SHAPE,
    VISION_M11_4_VIT_PATCH_PADDING,
    VISION_M11_4_VIT_PATCH_STRIDE,
    VISION_M11_4_VIT_PATCH_WEIGHT_SHAPE,
    VISION_M11_5_CORPUS_DIFF_BASELINE_ID,
    VISION_M11_5_HARDENING_STATUS,
    VISION_M11_5_RUNTIME_SCOPE_STATUS,
    VISION_M11_6_CORPUS_DIFF_BASELINE_ID,
    VISION_M11_6_INTERFACE_STATUS,
    VISION_M11_6_RUNTIME_SCOPE_STATUS,
    VISION_PROVIDER_NONE,
    VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    VISION_REPORT_FIELDS,
    VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    VISION_RUNTIME_KIND_ARTIFACT_ONLY,
    VISION_RUNTIME_KIND_PROVIDER,
    VISION_RUNTIME_REPLACEMENT,
    VISION_RUNTIME_PROVIDER_SCOPE_REASON,
    VISION_RUNTIME_STATUS_ARTIFACT_ONLY,
    VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
    VISION_SEMANTICS_STATUS_ACCEPTED,
    classify_m11_captured_grid_record,
    wrapper_conv2d_vision_report_fields,
)


MODEL_CORPUS = "m6_model_corpus"
PRE_M7_TAXONOMY_VERSION = 1
PRE_M7_BUILDER_DECISION = "keep_tvmscript_source_builder_for_m7_entry"
PRE_M8_TAXONOMY_VERSION = 1
PRE_M9_TAXONOMY_VERSION = 1
PRE_M10_TAXONOMY_VERSION = 1
PRE_M11_TAXONOMY_VERSION = 1
_PRE_M7_READER_CLASSES = (
    "pointwise",
    "broadcast_view_index",
    "reduction",
    "matmul_dot",
    "atomic_grid",
    "attention_adjacent",
)
_PRE_M7_EXCLUDED_M7_BLOCKER_CLASSES = {
    "attention_adjacent",
    "atomic",
    "grid",
    "matmul_dot",
    "reduction",
    "zero_triton_kernels",
}
_PRE_M8_IN_SCOPE_FAMILIES = {
    "norm_layernorm",
    "norm_rmsnorm",
    "pooling_reduction",
    "row_reduction",
    "softmax_like",
    "masked_attention_adjacent",
}
_PRE_M9_ENTRY_EXTERN_FAMILIES = {"extern_gemm", "extern_addmm_bias"}
_PRE_M9_DEFERRED_EXTERN_FAMILIES = {"deferred_convolution", "deferred_attention"}


@dataclass(frozen=True)
class TritonTVMModelAuditConfig:
    """Configuration for the experimental M6/M6.5 model corpus audit."""

    seed: int = 0
    contract: str = "pointwise_flat"
    target: str = "cuda"
    min_models: int = 1
    extern_gemm_runtime_provider: str = EXTERN_GEMM_PROVIDER_NONE
    attention_runtime_provider: str = ATTENTION_PROVIDER_NONE
    vision_runtime_provider: str = VISION_PROVIDER_NONE


@dataclass(frozen=True)
class TritonTVMModelAuditCase:
    """One model fixture in the M6/M6.5 audit corpus."""

    model_family: str
    case_name: str
    make_model: Callable[[Any], Any]
    make_inputs: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]]


def builtin_model_audit_cases() -> list[TritonTVMModelAuditCase]:
    """Return the external-library tiny model fixtures used by M6/M6.5."""
    return [
        TritonTVMModelAuditCase(
            model_family="vit",
            case_name="vit_tiny_random",
            make_model=_make_vit_tiny,
            make_inputs=_make_vit_inputs,
        ),
        TritonTVMModelAuditCase(
            model_family="llama",
            case_name="llama_tiny_random",
            make_model=_make_llama_tiny,
            make_inputs=_make_llama_inputs,
        ),
        TritonTVMModelAuditCase(
            model_family="yolo",
            case_name="yolov8n_yaml_random",
            make_model=_make_yolo_tiny,
            make_inputs=_make_yolo_inputs,
        ),
    ]


def run_model_corpus_audit(
    cases: list[TritonTVMModelAuditCase] | None = None,
    *,
    out_dir: str | Path,
    config: TritonTVMModelAuditConfig | None = None,
) -> dict[str, Any]:
    """Run the M6/M6.5 model corpus audit and write report artifacts."""
    cfg = config or TritonTVMModelAuditConfig()
    selected_cases = list(cases or builtin_model_audit_cases())
    if len(selected_cases) < cfg.min_models:
        raise ValueError(
            f"M6 model corpus requires at least {cfg.min_models} cases, "
            f"got {len(selected_cases)}"
        )

    out_path = Path(out_dir)
    wrapper_dir = out_path / "wrappers"
    kernel_dir = out_path / "kernels"
    ttir_dir = out_path / "ttir"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    ttir_dir.mkdir(parents=True, exist_ok=True)

    kernel_records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    collection_errors: list[dict[str, Any]] = []

    for case in selected_cases:
        model_record, records, errors = _run_one_model_case(
            case,
            cfg,
            wrapper_dir=wrapper_dir,
            kernel_dir=kernel_dir,
            ttir_dir=ttir_dir,
        )
        model_records.append(model_record)
        kernel_records.extend(records)
        collection_errors.extend(errors)

    report = build_model_corpus_report(
        kernel_records,
        model_records,
        dependency_versions=_dependency_versions(),
        errors=collection_errors,
    )
    write_model_corpus_report(report, out_path)
    return report


def build_model_corpus_report(
    kernel_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    *,
    dependency_versions: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable M6/M6.5 model corpus report."""
    normalized_models = [_normalize_model_record(record) for record in model_records]
    normalized_kernels = [_normalize_model_kernel_record(record) for record in kernel_records]
    extern_records = _extern_records_from_models(normalized_models)
    report = build_capability_report(
        normalized_kernels,
        purpose="m6/m6.5 external model corpus audit",
        corpus=MODEL_CORPUS,
        generated_at=generated_at,
        errors=errors,
    )
    report["dependency_versions"] = dict(sorted((dependency_versions or {}).items()))
    report["model_summary"] = _model_summary(normalized_models, normalized_kernels)
    report["models"] = normalized_models
    report["blockers"] = _rank_blockers(normalized_kernels, normalized_models, errors or [])
    report["pre_m7"] = _pre_m7_report_section(
        normalized_kernels,
        report["blockers"],
        errors or [],
    )
    report["pre_m8"] = _pre_m8_report_section(normalized_kernels, report["blockers"])
    report["pre_m9"] = _pre_m9_report_section(
        normalized_kernels,
        normalized_models,
        extern_records,
    )
    report["pre_m10"] = _pre_m10_report_section(extern_records)
    report["m10"] = _m10_attention_runtime_section(extern_records)
    report["pre_m11"] = _pre_m11_report_section(
        normalized_kernels,
        normalized_models,
        extern_records,
    )
    report["m11"] = _m11_vision_runtime_section(
        extern_records,
        normalized_models,
        normalized_kernels,
    )
    report["extern_ops"] = extern_records
    report["full_tvm_runnable"] = (
        bool(normalized_models)
        and all(model.get("full_tvm_runnable", False) for model in normalized_models)
    )
    return report


def write_model_corpus_report(report: dict[str, Any], out_dir: str | Path) -> None:
    """Write ``report.json`` and ``report.md`` for a model corpus audit."""
    write_capability_report(
        report,
        out_dir,
        markdown_title="M6/M6.5 Model Corpus Audit",
        footer_lines=_model_corpus_footer_lines(),
    )


def diff_capability_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare two capability reports while ignoring unstable timestamps and paths."""
    before_buckets = Counter((before.get("summary") or {}).get("status_buckets", {}))
    after_buckets = Counter((after.get("summary") or {}).get("status_buckets", {}))
    before_blockers = _blocker_counter(before)
    after_blockers = _blocker_counter(after)
    before_model_status = Counter(
        str(model.get("status", ""))
        for model in before.get("models", [])
        if model.get("status")
    )
    after_model_status = Counter(
        str(model.get("status", ""))
        for model in after.get("models", [])
        if model.get("status")
    )
    return {
        "schema_version": 1,
        "report_kind": "triton_tvm_capability_report_diff",
        "before": _diff_summary(before),
        "after": _diff_summary(after),
        "bucket_delta": _counter_delta(before_buckets, after_buckets),
        "blocker_delta": _counter_delta(before_blockers, after_blockers),
        "model_status_delta": _counter_delta(before_model_status, after_model_status),
        "translated_delta": int(after.get("translated_kernels", 0))
        - int(before.get("translated_kernels", 0)),
        "total_kernel_delta": int(after.get("total_kernels", 0))
        - int(before.get("total_kernels", 0)),
        "extern_family_delta": _counter_delta(
            Counter((before.get("model_summary") or {}).get("extern_op_families", {})),
            Counter((after.get("model_summary") or {}).get("extern_op_families", {})),
        ),
        "full_tvm_runnable_delta": int(
            (after.get("model_summary") or {}).get("full_tvm_runnable_models", 0)
        )
        - int((before.get("model_summary") or {}).get("full_tvm_runnable_models", 0)),
        "triton_kernel_runnable_delta": int(
            (after.get("model_summary") or {}).get("triton_kernel_runnable_models", 0)
        )
        - int(
            (before.get("model_summary") or {}).get("triton_kernel_runnable_models", 0)
        ),
        "m9_materialized_artifact_candidate_delta": len(
            (after.get("pre_m9") or {}).get("m9_materialized_artifact_candidates", [])
        )
        - len((before.get("pre_m9") or {}).get("m9_materialized_artifact_candidates", [])),
        "m11_vision_delta": _m11_vision_diff_delta(before, after),
    }


def _m11_vision_diff_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_m11 = before.get("m11") or {}
    after_m11 = after.get("m11") or {}
    before_grid = before_m11.get("captured_grid_taxonomy") or {}
    after_grid = after_m11.get("captured_grid_taxonomy") or {}
    numeric_fields = (
        "observed_convolution_call_count",
        "artifact_count",
        "artifact_only_count",
        "runtime_resolved_count",
        "runtime_launch_count",
        "artifact_call_count",
        "total_io_bytes",
        "host_staging_bytes",
        "total_accounted_bytes",
        "model_full_tvm_runnable_after_m11_5_gate",
        "model_full_tvm_runnable_after_m11_6_gate",
    )
    return {
        field: int(after_m11.get(field, 0) or 0) - int(before_m11.get(field, 0) or 0)
        for field in numeric_fields
    } | {
        "status_delta": _counter_delta(
            Counter(before_m11.get("status_counts", {})),
            Counter(after_m11.get("status_counts", {})),
        ),
        "provider_delta": _counter_delta(
            Counter(before_m11.get("provider_counts", {})),
            Counter(after_m11.get("provider_counts", {})),
        ),
        "contract_delta": _counter_delta(
            Counter(before_m11.get("contract_counts", {})),
            Counter(after_m11.get("contract_counts", {})),
        ),
        "grid_family_delta": _counter_delta(
            Counter(before_grid.get("family_counts", {})),
            Counter(after_grid.get("family_counts", {})),
        ),
        "grid_kernel_delta": int(after_grid.get("kernel_count", 0) or 0)
        - int(before_grid.get("kernel_count", 0) or 0),
        "grid2d_native_runtime_ready_delta": int(
            (after_m11.get("captured_grid_readiness") or {}).get(
                "native_runtime_ready_count", 0
            )
            or 0
        )
        - int(
            (before_m11.get("captured_grid_readiness") or {}).get(
                "native_runtime_ready_count", 0
            )
            or 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m tvm.contrib.triton_tvm.model_corpus``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builtin-model-corpus", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract", default="pointwise_flat")
    parser.add_argument("--min-models", type=int, default=1)
    parser.add_argument(
        "--extern-gemm-runtime-provider",
        choices=[EXTERN_GEMM_PROVIDER_NONE, EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED],
        default=EXTERN_GEMM_PROVIDER_NONE,
    )
    parser.add_argument(
        "--attention-runtime-provider",
        choices=[
            ATTENTION_PROVIDER_NONE,
            ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED,
            ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
        ],
        default=ATTENTION_PROVIDER_NONE,
    )
    parser.add_argument(
        "--vision-runtime-provider",
        choices=[VISION_PROVIDER_NONE, VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED],
        default=VISION_PROVIDER_NONE,
    )
    args = parser.parse_args(argv)

    if not args.builtin_model_corpus:
        raise RuntimeError("M6 model corpus CLI currently requires --builtin-model-corpus")
    report = run_model_corpus_audit(
        builtin_model_audit_cases(),
        out_dir=args.out_dir,
        config=TritonTVMModelAuditConfig(
            seed=args.seed,
            contract=args.contract,
            min_models=args.min_models,
            extern_gemm_runtime_provider=args.extern_gemm_runtime_provider,
            attention_runtime_provider=args.attention_runtime_provider,
            vision_runtime_provider=args.vision_runtime_provider,
        ),
    )
    print(
        "Wrote M6/M6.5 model corpus audit with "
        f"{len(report['models'])} models and {report['total_kernels']} kernels "
        f"to {args.out_dir}"
    )
    return 0


def _run_one_model_case(
    case: TritonTVMModelAuditCase,
    cfg: TritonTVMModelAuditConfig,
    *,
    wrapper_dir: Path,
    kernel_dir: Path,
    ttir_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    kernel_records: list[dict[str, Any]] = []
    wrappers: list[str] = []
    model_record = _base_model_record(case)
    try:
        wrappers = _capture_inductor_wrappers(case, cfg)
    except Exception as err:  # pylint: disable=broad-except
        entry = _collection_error(case, "compile_error", err)
        errors.append(entry)
        model_record.update(
            {
                "status": "compile_error",
                "error_type": entry["error_type"],
                "message": entry["message"],
                "full_tvm_runnable": False,
            }
        )
        return model_record, kernel_records, errors

    sources: list[InductorTritonSource] = []
    for wrapper_index, wrapper_source in enumerate(wrappers):
        wrapper_path = wrapper_dir / f"{_safe_name(case.case_name)}_{wrapper_index}.py"
        wrapper_path.write_text(wrapper_source, encoding="utf-8")
        model_record["wrapper_paths"].append(str(wrapper_path))
        model_record["extern_calls"].extend(
            _extern_call_to_record(call, case, cfg)
            for call in extract_inductor_wrapper_extern_calls(
                wrapper_source,
                case_name=case.case_name,
                wrapper_path=str(wrapper_path),
                extern_gemm_runtime_provider=cfg.extern_gemm_runtime_provider,
                attention_runtime_provider=cfg.attention_runtime_provider,
                vision_runtime_provider=cfg.vision_runtime_provider,
            )
        )
        sources.extend(
            extract_inductor_triton_sources(
                wrapper_source,
                case_name=case.case_name,
                wrapper_path=str(wrapper_path),
            )
        )

    for source in sources:
        kernel_path = kernel_dir / f"{_safe_name(case.case_name)}_{source.kernel_name}.py"
        kernel_path.write_text(source.source, encoding="utf-8")
        record = _audit_one_model_kernel(
            case,
            source,
            kernel_path=kernel_path,
            ttir_dir=ttir_dir,
            contract=cfg.contract,
        )
        kernel_records.append(record)

    _finalize_model_record(model_record, kernel_records)
    if not sources:
        entry = {
            "model_family": case.model_family,
            "model_case": case.case_name,
            "case_name": case.case_name,
            "kernel_name": "",
            "bucket": "collection_error",
            "fallback_reason": "zero_triton_kernels",
            "error_type": "RuntimeError",
            "message": "TorchInductor produced no captured Triton kernels for this model case",
        }
        errors.append(entry)
        model_record["status"] = "zero_kernels"
        model_record["full_tvm_runnable"] = False
    return model_record, kernel_records, errors


def _capture_inductor_wrappers(
    case: TritonTVMModelAuditCase,
    cfg: TritonTVMModelAuditConfig,
) -> list[str]:
    import torch  # pylint: disable=import-outside-toplevel
    import torch._dynamo  # pylint: disable=import-outside-toplevel
    import torch._inductor.config as inductor_config  # pylint: disable=import-outside-toplevel
    from torch._inductor.graph import GraphLowering  # pylint: disable=import-outside-toplevel

    if cfg.target != "cuda":
        raise ValueError(f"M6 model corpus currently supports target='cuda', got {cfg.target!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("M6 model corpus audit requires CUDA")

    _set_offline_env()
    _seed_torch(torch, cfg.seed)
    captured: list[str] = []
    old_save_output_code = GraphLowering.save_output_code
    old_fx_graph_cache = inductor_config.fx_graph_cache
    GraphLowering.save_output_code = captured.append
    inductor_config.fx_graph_cache = False
    try:
        torch._dynamo.reset()
        model = case.make_model(torch)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model = model.to("cuda")
        args, kwargs = case.make_inputs(torch)
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            compiled(*args, **kwargs)
        torch.cuda.synchronize()
    finally:
        GraphLowering.save_output_code = old_save_output_code
        inductor_config.fx_graph_cache = old_fx_graph_cache
        torch._dynamo.reset()
    return captured


def _audit_one_model_kernel(
    case: TritonTVMModelAuditCase,
    source: InductorTritonSource,
    *,
    kernel_path: Path,
    ttir_dir: Path,
    contract: str,
) -> dict[str, Any]:
    kernel: InductorKernel | None = None
    try:
        kernel = load_inductor_kernel(source)
        selected_contract = _select_model_kernel_contract(kernel, contract)
        record = audit_inductor_kernel(kernel, ttir_dir=ttir_dir, contract=selected_contract)
    except Exception as err:  # pylint: disable=broad-except
        record = {
            "corpus": MODEL_CORPUS,
            "case_name": case.case_name,
            "kernel_name": source.kernel_name,
            "contract": contract,
            "source_hash": "",
            "ttir_hash": "",
            "signature": {},
            "constexprs": {},
            "unique_ops": [],
            "op_counts": {},
            "types": [],
            "raw_load_store_attrs": [],
            "load_count": 0,
            "store_count": 0,
            "mask_forms": [],
            "indexing_summary": {},
            "translate_status": make_report_status(
                ok=False,
                bucket="internal_error",
                fallback_reason="load_or_audit_error",
                error_type=type(err).__name__,
                message=str(err),
            ),
        }

    record["corpus"] = MODEL_CORPUS
    record["case_name"] = case.case_name
    record["model_family"] = case.model_family
    record["model_case"] = case.case_name
    record["kernel_source_path"] = str(kernel_path)
    if kernel is not None:
        _attach_inductor_metadata(record, kernel)
        if not _is_model_kernel_replaceable(kernel, str(record.get("contract", contract))):
            record["candidate_translate_status"] = dict(record["translate_status"])
            record["translate_status"] = make_report_status(
                ok=False,
                bucket="contract_error",
                fallback_reason="unsupported_inductor_kernel",
                message=(
                    "M8 model audit only marks Grid1D non-atomic pointwise and "
                    "approved reduction-family Inductor kernels as TVM-replaceable"
                ),
            )
    else:
        record.setdefault("grid_type", "")
        record.setdefault("num_reduction", 0)
        record.setdefault("atomic_add_found", False)
        record.setdefault("size_hints", {})
    record["blocker_class"] = _blocker_class(record)
    record["pre_m8_family"] = _pre_m8_family(record)
    return record


def _select_model_kernel_contract(kernel: InductorKernel, contract: str) -> str:
    if contract != "auto_m8":
        return contract
    meta = kernel.inductor_meta
    if int(meta.get("num_reduction", 0) or 0) <= 0:
        return "pointwise_flat"
    text = f"{kernel.kernel_name} {kernel.source}".lower()
    if "softmax" in text:
        return "masked_softmax_row"
    if any(token in text for token in ("layer_norm", "layernorm", "native_layer_norm")):
        return "norm_row"
    if any(token in text for token in ("rms", "rsqrt", "embedding_mean_mul_pow_rsqrt")):
        return "norm_row"
    if "max_pool" in text or "pool" in text:
        return "row_reduction"
    return "row_reduction"


def _is_model_kernel_replaceable(kernel: InductorKernel, contract: str) -> bool:
    meta = kernel.inductor_meta
    if meta.get("grid_type") != "Grid1D" or bool(meta.get("atomic_add_found", False)):
        return False
    num_reduction = int(meta.get("num_reduction", 0) or 0)
    if contract == "pointwise_flat":
        return is_inductor_pointwise_kernel(kernel)
    if contract in {"row_reduction", "norm_row", "softmax_row", "masked_softmax_row"}:
        return num_reduction > 0
    return False


def _attach_inductor_metadata(record: dict[str, Any], kernel: InductorKernel) -> None:
    meta = kernel.inductor_meta
    record.update(
        {
            "grid_type": meta.get("grid_type", ""),
            "num_reduction": int(meta.get("num_reduction", 0) or 0),
            "atomic_add_found": bool(meta.get("atomic_add_found", False)),
            "size_hints": dict(kernel.size_hints),
        }
    )


def _finalize_model_record(model_record: dict[str, Any], records: list[dict[str, Any]]) -> None:
    model_records = [
        record for record in records if record.get("model_case") == model_record["model_case"]
    ]
    status_buckets = Counter(
        record["translate_status"].get("bucket", "unknown") for record in model_records
    )
    blockers = Counter(
        record.get("blocker_class", "unknown")
        for record in model_records
        if not record["translate_status"].get("ok", False)
    )
    model_record.update(
        {
            "status": "completed",
            "kernel_count": len(model_records),
            "translated_kernels": sum(
                1 for record in model_records if record["translate_status"].get("ok", False)
            ),
            "native_fallback_kernels": sum(
                1 for record in model_records if not record["translate_status"].get("ok", False)
            ),
            "status_buckets": dict(sorted(status_buckets.items())),
            "blocker_classes": dict(sorted(blockers.items())),
        }
    )
    triton_kernel_runnable = (
        model_record["kernel_count"] > 0
        and model_record["native_fallback_kernels"] == 0
        and model_record["translated_kernels"] == model_record["kernel_count"]
    )
    model_record["triton_kernel_runnable"] = triton_kernel_runnable
    model_record["full_tvm_runnable"] = (
        triton_kernel_runnable and not model_record.get("extern_calls")
    )


def _base_model_record(case: TritonTVMModelAuditCase) -> dict[str, Any]:
    return {
        "model_family": case.model_family,
        "model_case": case.case_name,
        "status": "not_run",
        "kernel_count": 0,
        "translated_kernels": 0,
        "native_fallback_kernels": 0,
        "status_buckets": {},
        "blocker_classes": {},
        "wrapper_paths": [],
        "extern_calls": [],
        "triton_kernel_runnable": False,
        "full_tvm_runnable": False,
    }


def _normalize_model_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("model_family", "")
    normalized.setdefault("model_case", normalized.get("case_name", ""))
    normalized.setdefault("status", "not_run")
    normalized.setdefault("kernel_count", 0)
    normalized.setdefault("translated_kernels", 0)
    normalized.setdefault("native_fallback_kernels", 0)
    normalized.setdefault("status_buckets", {})
    normalized.setdefault("blocker_classes", {})
    normalized.setdefault("wrapper_paths", [])
    normalized.setdefault("extern_calls", [])
    normalized["extern_calls"] = [
        _normalize_extern_call_record(call, normalized) for call in normalized["extern_calls"]
    ]
    normalized["extern_call_count"] = len(normalized["extern_calls"])
    normalized.setdefault(
        "triton_kernel_runnable",
        bool(
            normalized.get("kernel_count", 0)
            and normalized.get("native_fallback_kernels", 0) == 0
            and normalized.get("translated_kernels", 0) == normalized.get("kernel_count", 0)
        ),
    )
    normalized.setdefault("full_tvm_runnable", False)
    if normalized["extern_calls"] and normalized.get("full_tvm_runnable"):
        normalized["full_tvm_runnable"] = False
    return normalized


def _normalize_model_kernel_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("corpus", MODEL_CORPUS)
    normalized.setdefault("model_family", "")
    normalized.setdefault("model_case", normalized.get("case_name", ""))
    normalized.setdefault("grid_type", "")
    normalized.setdefault("num_reduction", 0)
    normalized.setdefault("atomic_add_found", False)
    normalized.setdefault("size_hints", {})
    normalized.setdefault("blocker_class", _blocker_class(normalized))
    normalized.setdefault("pre_m8_family", _pre_m8_family(normalized))
    for field, value in classify_m11_captured_grid_record(normalized).items():
        normalized.setdefault(field, value)
    return normalized


def _extern_call_to_record(
    call,
    case: TritonTVMModelAuditCase,
    cfg: TritonTVMModelAuditConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or TritonTVMModelAuditConfig()
    record = {
        "model_family": case.model_family,
        "model_case": case.case_name,
        "case_name": case.case_name,
        "op_name": call.op_name,
        "op_family": call.op_family,
        "line_no": call.line_no,
        "wrapper_path": call.wrapper_path,
        "source": call.source,
    }
    for field_name in _EXTERN_MATMUL_FIELDS:
        record[field_name] = getattr(call, field_name, _extern_matmul_default(field_name))
    for field_name in _EXTERN_ATTENTION_FIELDS:
        record[field_name] = getattr(
            call,
            field_name,
            _extern_attention_default(field_name),
        )
    for field_name in _EXTERN_VISION_FIELDS:
        record[field_name] = getattr(call, field_name, _extern_vision_default(field_name))
    if record["op_family"] == "deferred_attention":
        record.update(
            wrapper_sdpa_attention_report_fields(
                op_name=record["op_name"],
                op_family=record["op_family"],
                source=record["source"],
                model_family=case.model_family,
                model_case=case.case_name,
                case_name=case.case_name,
                attention_runtime_provider=cfg.attention_runtime_provider,
            )
        )
    if record["op_family"] == "deferred_convolution" and not record.get(
        "vision_semantics_status"
    ):
        record.update(
            wrapper_conv2d_vision_report_fields(
                op_name=record["op_name"],
                op_family=record["op_family"],
                source=record["source"],
                line_no=int(record.get("line_no", 0) or 0),
                model_case=case.case_name,
                case_name=case.case_name,
                vision_runtime_provider=cfg.vision_runtime_provider,
            )
        )
    return record


def _normalize_extern_call_record(
    record: dict[str, Any],
    model_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_record = model_record or {}
    normalized = dict(record)
    normalized.setdefault("model_family", model_record.get("model_family", ""))
    normalized.setdefault("model_case", model_record.get("model_case", ""))
    normalized.setdefault("case_name", normalized.get("model_case", ""))
    normalized.setdefault("op_name", "")
    normalized.setdefault("op_family", _extern_family_from_op_name(str(normalized["op_name"])))
    normalized.setdefault("line_no", 0)
    normalized.setdefault("wrapper_path", "")
    normalized.setdefault("source", "")
    for field_name in _EXTERN_MATMUL_FIELDS:
        normalized.setdefault(field_name, _extern_matmul_default(field_name))
    for field_name in _EXTERN_ATTENTION_FIELDS:
        normalized.setdefault(field_name, _extern_attention_default(field_name))
    for field_name in _EXTERN_VISION_FIELDS:
        normalized.setdefault(field_name, _extern_vision_default(field_name))
    if normalized.get("op_family") == "deferred_attention":
        classification = wrapper_sdpa_attention_report_fields(
            op_name=str(normalized.get("op_name", "")),
            op_family=str(normalized.get("op_family", "")),
            source=str(normalized.get("source", "")),
            model_family=str(normalized.get("model_family", "")),
            model_case=str(normalized.get("model_case", "")),
            case_name=str(normalized.get("case_name", "")),
        )
        for field_name, value in classification.items():
            if (
                field_name == "unsupported_attention_runtime_reason"
                and str(normalized.get("attention_runtime_status", ""))
                == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
            ):
                continue
            if not normalized.get(field_name):
                normalized[field_name] = value
    if normalized.get("op_family") == "deferred_convolution" and not normalized.get(
        "vision_semantics_status"
    ):
        classification = wrapper_conv2d_vision_report_fields(
            op_name=str(normalized.get("op_name", "")),
            op_family=str(normalized.get("op_family", "")),
            source=str(normalized.get("source", "")),
            line_no=int(normalized.get("line_no", 0) or 0),
            model_case=str(normalized.get("model_case", "")),
            case_name=str(normalized.get("case_name", "")),
        )
        for field_name, value in classification.items():
            if not normalized.get(field_name):
                normalized[field_name] = value
    if (
        normalized.get("op_family") in ("extern_gemm", "extern_addmm_bias")
        and normalized.get("implementation_kind") in ("extern_gemm", "extern_addmm_bias")
    ):
        normalized["extern_gemm_runtime_status"] = (
            normalized.get("extern_gemm_runtime_status")
            or EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
        )
        normalized["extern_gemm_provider_kind"] = (
            normalized.get("extern_gemm_provider_kind") or EXTERN_GEMM_PROVIDER_NONE
        )
        if normalized.get("extern_gemm_provider_abi_version") is None:
            normalized["extern_gemm_provider_abi_version"] = 0
    return normalized


_EXTERN_MATMUL_FIELDS = (
    "matmul_source_kind",
    "matmul_m",
    "matmul_n",
    "matmul_k",
    "matmul_contract",
    "matmul_contract_ok",
    "matmul_a_dtype",
    "matmul_b_dtype",
    "matmul_accumulator_dtype",
    "matmul_output_dtype",
    "matmul_epilogue_kind",
    "implementation_kind",
    "schedule_id",
    "extern_symbol",
    "extern_packed_func",
    "extern_runtime_kind",
    "extern_runtime_replacement",
    "extern_runtime_replacement_available",
    "extern_runtime_replacement_reason",
    "extern_gemm_runtime_status",
    "extern_gemm_provider_kind",
    "extern_gemm_provider_abi_version",
    "extern_gemm_runtime_claim",
    "extern_gemm_performance_claim",
    "extern_gemm_uses_host_staging",
    "unsupported_matmul_reason",
    "matmul_a_layout",
    "matmul_b_layout",
    "matmul_c_layout",
    "matmul_a_stride",
    "matmul_b_stride",
    "matmul_c_stride",
)

_EXTERN_ATTENTION_FIELDS = ATTENTION_REPORT_FIELDS
_EXTERN_VISION_FIELDS = VISION_REPORT_FIELDS


def _extern_matmul_default(field_name: str) -> Any:
    if field_name in {
        "matmul_m",
        "matmul_n",
        "matmul_k",
        "extern_gemm_provider_abi_version",
    }:
        return None
    if field_name in {
        "matmul_contract_ok",
        "extern_runtime_replacement_available",
        "extern_gemm_performance_claim",
        "extern_gemm_uses_host_staging",
    }:
        return False
    if field_name.endswith("_stride"):
        return None
    return ""


def _extern_attention_default(field_name: str) -> Any:
    if field_name in {"attention_abi_version", "attention_provider_abi_version"}:
        return 0
    if field_name in {
        "attention_runtime_launch_count",
        "attention_artifact_call_count",
        "attention_qkv_bytes",
        "attention_mask_bytes",
        "attention_output_bytes",
        "attention_total_io_bytes",
        "attention_intermediate_buffer_bytes",
        "attention_host_staging_bytes",
        "attention_total_accounted_bytes",
    }:
        return 0
    if field_name in {
        "attention_causal",
        "attention_runtime_replacement_available",
        "attention_performance_claim",
        "attention_uses_host_staging",
    }:
        return False
    if field_name == "attention_scale":
        return None
    return ""


def _extern_vision_default(field_name: str) -> Any:
    if field_name in {
        "vision_groups",
        "vision_provider_abi_version",
        "vision_runtime_launch_count",
        "vision_artifact_call_count",
        "vision_input_bytes",
        "vision_weight_bytes",
        "vision_output_bytes",
        "vision_total_io_bytes",
        "vision_host_staging_bytes",
        "vision_total_accounted_bytes",
    }:
        return 0
    if field_name in {
        "vision_transposed",
        "vision_runtime_replacement_available",
        "vision_performance_claim",
        "vision_uses_host_staging",
    }:
        return False
    return ""


def _extern_family_from_op_name(op_name: str) -> str:
    if op_name in {"extern_kernels.mm", "extern_kernels.bmm"}:
        return "extern_gemm"
    if op_name == "extern_kernels.addmm":
        return "extern_addmm_bias"
    if op_name == "extern_kernels.convolution":
        return "deferred_convolution"
    if op_name == "torch.ops.aten._scaled_dot_product_efficient_attention.default":
        return "deferred_attention"
    return "extern_other"


def _extern_records_from_models(model_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in model_records:
        for call in model.get("extern_calls", []) or []:
            records.append(_normalize_extern_call_record(call, model))
    return sorted(
        records,
        key=lambda item: (
            str(item.get("model_case", "")),
            str(item.get("wrapper_path", "")),
            int(item.get("line_no", 0) or 0),
            str(item.get("op_name", "")),
        ),
    )


def _model_summary(
    model_records: list[dict[str, Any]],
    kernel_records: list[dict[str, Any]],
) -> dict[str, Any]:
    families = Counter(record["model_family"] for record in model_records)
    model_status = Counter(record["status"] for record in model_records)
    extern_records = _extern_records_from_models(model_records)
    extern_families = Counter(record.get("op_family", "unknown") for record in extern_records)
    extern_ops = Counter(record.get("op_name", "unknown") for record in extern_records)
    blocker_classes = Counter(
        record.get("blocker_class", "unknown")
        for record in kernel_records
        if not record.get("translate_status", {}).get("ok", False)
    )
    pre_m8_families = Counter(
        record.get("pre_m8_family", "unknown")
        for record in kernel_records
        if not record.get("translate_status", {}).get("ok", False)
    )
    return {
        "total_models": len(model_records),
        "completed_models": sum(
            1 for record in model_records if record.get("status") == "completed"
        ),
        "failed_models": sum(
            1
            for record in model_records
            if record.get("status") not in ("completed", "not_run")
        ),
        "zero_kernel_models": sum(
            1 for record in model_records if record.get("status") == "zero_kernels"
        ),
        "full_tvm_runnable_models": sum(
            1 for record in model_records if record.get("full_tvm_runnable")
        ),
        "triton_kernel_runnable_models": sum(
            1 for record in model_records if record.get("triton_kernel_runnable")
        ),
        "extern_op_count": len(extern_records),
        "total_kernels": len(kernel_records),
        "translated_kernels": sum(
            1 for record in kernel_records if record.get("translate_status", {}).get("ok")
        ),
        "fallback_kernels": sum(
            1 for record in kernel_records if not record.get("translate_status", {}).get("ok")
        ),
        "families": dict(sorted(families.items())),
        "model_status": dict(sorted(model_status.items())),
        "extern_op_families": dict(sorted(extern_families.items())),
        "extern_ops": dict(sorted(extern_ops.items())),
        "blocker_classes": dict(sorted(blocker_classes.items())),
        "pre_m8_families": dict(sorted(pre_m8_families.items())),
    }


def _pre_m7_report_section(
    kernel_records: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "taxonomy_version": PRE_M7_TAXONOMY_VERSION,
        "builder_decision": PRE_M7_BUILDER_DECISION,
        "builder_decision_reason": (
            "Direct node construction was evaluated for the Pre-M7 gate but is "
            "not adopted before M7 because it would duplicate the current "
            "pointwise/reduction TVMScript template builders without yet "
            "reducing M7 blocker risk."
        ),
        "unsupported_taxonomy": _pre_m7_unsupported_taxonomy(),
        "reader_snapshot_classes": _pre_m7_reader_snapshot_classes(kernel_records),
        "m7_entry_blockers": _pre_m7_entry_blockers(blockers),
        "collection_error_count": len(errors),
    }


def _pre_m7_unsupported_taxonomy() -> dict[str, Any]:
    return {
        "top_level_bucket_policy": "preserve_existing_buckets",
        "detail_fields": ["fallback_reason", "blocker_class"],
        "stable_top_level_buckets": [
            "translated",
            "unsupported_ttir_op",
            "contract_error",
            "target_policy_error",
            "unsupported_stream",
            "input_error",
            "collection_error",
            "triton_tvm_error",
            "internal_error",
        ],
        "blocker_class_policy": {
            "atomic": "classify through blocker_class without adding a top-level bucket",
            "broadcast_view_index": (
                "keep the existing exception bucket and use fallback_reason or "
                "blocker_class for M7 ordering"
            ),
            "collection": "keep non-kernel failures in collection_error",
            "grid": "classify through blocker_class without adding a top-level bucket",
            "matmul_dot": "classify through blocker_class without adding a top-level bucket",
            "reduction": "classify through blocker_class without adding a top-level bucket",
        },
    }


def _pre_m8_report_section(
    kernel_records: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    family_classes = _pre_m8_family_classes(kernel_records)
    return {
        "taxonomy_version": PRE_M8_TAXONOMY_VERSION,
        "unsupported_taxonomy": {
            "top_level_bucket_policy": "preserve_existing_buckets",
            "detail_fields": ["fallback_reason", "blocker_class", "pre_m8_family"],
            "stable_top_level_buckets": _pre_m7_unsupported_taxonomy()[
                "stable_top_level_buckets"
            ],
            "family_policy": {
                "norm_layernorm": "M8 norm-family candidate",
                "norm_rmsnorm": "M8 norm-family candidate",
                "pooling_reduction": "separate reduction-family detail class",
                "row_reduction": "M8 reduction-family candidate",
                "softmax_like": "M8 softmax-family candidate",
                "masked_attention_adjacent": (
                    "masked softmax or causal-mask policy input; attention runtime "
                    "remains out of scope"
                ),
                "deferred_grid": "deferred outside Pre-M8/M8 reduction-family scope",
            },
        },
        "contract_policy": {
            "execution_kind": "serial_m4_single_lane for current reduction/norm contracts",
            "future_parallel_metadata": (
                "future parallel reductions must use a distinct execution_kind"
            ),
            "accumulator_dtype_policy": "preserve_ttir_reduction_dtype",
            "epsilon_policy": "runtime_and_constexpr_eps_supported for norm_single_row",
            "mask_policy": "masked reduction loads require explicit zero other",
            "axis_policy": "axis_0_only",
            "layout_policy": "row_major_only or single_row_row_major",
        },
        "family_classes": family_classes,
        "m8_entry_debt": [
            _pre_m8_entry_from_family(name, entry)
            for name, entry in family_classes.items()
            if name in _PRE_M8_IN_SCOPE_FAMILIES
        ],
        "deferred_debt": [
            _pre_m8_entry_from_family(name, entry)
            for name, entry in family_classes.items()
            if name.startswith("deferred_")
        ],
        "blocker_count": len(blockers),
    }


def _pre_m8_family_classes(
    kernel_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}

    def get_entry(name: str) -> dict[str, Any]:
        if name not in classes:
            classes[name] = {
                "kernel_count": 0,
                "buckets": Counter(),
                "blocker_classes": Counter(),
                "example_kernel": "",
                "example_message": "",
            }
        return classes[name]

    for record in kernel_records:
        if record.get("translate_status", {}).get("ok", False):
            continue
        family = str(record.get("pre_m8_family", "")) or _pre_m8_family(record)
        entry = get_entry(family)
        status = record.get("translate_status", {})
        entry["kernel_count"] += 1
        entry["buckets"][str(status.get("bucket", "unknown"))] += 1
        entry["blocker_classes"][str(record.get("blocker_class", "unknown"))] += 1
        if not entry["example_kernel"]:
            entry["example_kernel"] = str(record.get("kernel_name", ""))
            entry["example_message"] = str(status.get("message", ""))

    normalized = {}
    for name, entry in sorted(classes.items()):
        normalized[name] = {
            "kernel_count": int(entry["kernel_count"]),
            "buckets": dict(sorted(entry["buckets"].items())),
            "blocker_classes": dict(sorted(entry["blocker_classes"].items())),
            "example_kernel": entry["example_kernel"],
            "example_message": entry["example_message"],
        }
    return normalized


def _pre_m8_entry_from_family(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "pre_m8_family": name,
        "kernel_count": int(entry.get("kernel_count", 0)),
        "buckets": dict(entry.get("buckets", {})),
        "blocker_classes": dict(entry.get("blocker_classes", {})),
        "example_kernel": entry.get("example_kernel", ""),
        "example_message": entry.get("example_message", ""),
    }


def _pre_m9_report_section(
    kernel_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    extern_records: list[dict[str, Any]],
) -> dict[str, Any]:
    extern_family_classes = _pre_m9_extern_family_classes(extern_records)
    ttir_dot_records = [
        record for record in kernel_records if "tt.dot" in set(record.get("unique_ops", []))
    ]
    grid_records = [
        record
        for record in kernel_records
        if not record.get("translate_status", {}).get("ok", False)
        and record.get("blocker_class") == "grid"
    ]
    return {
        "taxonomy_version": PRE_M9_TAXONOMY_VERSION,
        "report_policy": {
            "extern_visibility": (
                "Wrapper-level extern calls are explicit Pre-M9 observations and "
                "are not counted as translated Triton kernels."
            ),
            "extern_gemm_runtime_replacement_gate": (
                "M9.5 keeps extern_gemm artifact-only by default. M9.6 may mark "
                "wrapper GEMM runtime_resolved only under an explicit "
                "correctness-only provider; provider resolution is reported "
                "separately from captured Triton kernel translations."
            ),
            "full_tvm_runnable": (
                "A model is full_tvm_runnable only when all captured Triton kernels "
                "translate and no wrapper-level extern calls remain."
            ),
            "grid_boundary": (
                "M7.5/M8.5 deferred grid debt remains separate from M9 matmul "
                "acceptance unless a later ADR changes the milestone scope."
            ),
        },
        "detail_fields": [
            "fallback_reason",
            "blocker_class",
            "pre_m8_family",
            "op_family",
            "matmul_source_kind",
            "matmul_epilogue_kind",
            "implementation_kind",
            "extern_runtime_kind",
            "extern_runtime_replacement",
            "extern_gemm_runtime_status",
            "extern_gemm_provider_kind",
            "unsupported_matmul_reason",
        ],
        "contract_policy": {
            "plain_gemm": (
                "M9 entry admits explicit GEMM policy before lowering; initial "
                "runtime path may be a TVM artifact extern call."
            ),
            "batched_gemm": "classified during M9 but lower after plain GEMM policy is stable",
            "gemm_epilogue": (
                "bias/add/activation epilogues are classified separately from the "
                "core GEMM path"
            ),
            "qkv_projection": (
                "QKV projections are GEMM candidates; attention runtime remains "
                "deferred to M10"
            ),
            "tensorcore_dtype_layout": (
                "fp16/bf16 TensorCore dtype and layout policy must be documented "
                "before native schedule expansion"
            ),
        },
        "extern_policy": {
            "extern_gemm": (
                "Allowed only as an explicit TVM artifact extern call; it is not "
                "native fallback."
            ),
            "extern_addmm_bias": (
                "GEMM+bias candidate; bias/epilogue must be visible in reports."
            ),
            "deferred_convolution": "Deferred to the vision/conv stack unless M9 admits 1x1 conv-to-GEMM.",
            "deferred_attention": "Deferred to the attention runtime milestone.",
        },
        "extern_family_classes": extern_family_classes,
        "m96_extern_gemm_runtime": _m96_extern_gemm_runtime_section(extern_records),
        "m9_materialized_artifact_candidates": _pre_m9_materialized_artifact_candidates(
            extern_records
        ),
        "m9_entry_debt": [
            _pre_m9_entry_from_extern_family(name, entry)
            for name, entry in _pre_m9_entry_debt_classes(extern_records).items()
        ],
        "deferred_debt": [
            _pre_m9_entry_from_extern_family(name, entry)
            for name, entry in extern_family_classes.items()
            if name in _PRE_M9_DEFERRED_EXTERN_FAMILIES
        ],
        "observed_ttir_dot_kernels": len(ttir_dot_records),
        "observed_grid_fallback_kernels": len(grid_records),
        "model_full_tvm_runnable_after_extern_gate": sum(
            1 for record in model_records if record.get("full_tvm_runnable")
        ),
    }


def _pre_m10_report_section(extern_records: list[dict[str, Any]]) -> dict[str, Any]:
    attention_records = [
        record
        for record in extern_records
        if str(record.get("op_family", "")) == "deferred_attention"
    ]
    contract_classes = _pre_m10_attention_contract_classes(attention_records)
    runtime_deferred_debt = [
        _pre_m10_entry_from_attention_contract(contract, entry)
        for contract, entry in contract_classes.items()
        if int(entry.get("call_count", 0) or 0) > 0
    ]
    return {
        "taxonomy_version": PRE_M10_TAXONOMY_VERSION,
        "report_policy": {
            "attention_abi_gate": (
                "Pre-M10 classifies wrapper-level attention calls before M10 "
                "runtime work. It does not count attention calls as translated "
                "Triton kernels or runtime-resolved model progress."
            ),
            "schema_policy": (
                "Report schema v1 top-level buckets remain stable; attention "
                "ABI details are additive fields on deferred_attention records."
            ),
            "runtime_policy": (
                "All Pre-M10 attention contracts use deferred_attention_runtime "
                "until an explicit M10 runtime milestone admits lowering."
            ),
        },
        "target_contracts": list(ATTENTION_TARGET_CONTRACTS),
        "detail_fields": list(ATTENTION_REPORT_FIELDS),
        "contract_policy": {
            "attention_vit_full_v1": (
                "ViT full attention ABI with no RoPE or KV cache requirement; "
                "runtime remains deferred."
            ),
            "attention_llama_causal_prefill_v1": (
                "Llama causal prefill ABI with RoPE reported as deferred "
                "pointwise policy and no KV cache update requirement."
            ),
            "attention_llama_decode_v1": (
                "Llama single-token decode ABI with RoPE and KV cache layout "
                "reported but not implemented in Pre-M10."
            ),
        },
        "contract_classes": contract_classes,
        "observed_attention_call_count": len(attention_records),
        "attention_runtime_deferred_count": sum(
            1
            for record in attention_records
            if str(record.get("attention_runtime_status", ""))
            == ATTENTION_RUNTIME_STATUS_DEFERRED
        ),
        "runtime_deferred_debt": runtime_deferred_debt,
    }


def _pre_m10_attention_contract_classes(
    attention_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}

    def get_entry(contract: str) -> dict[str, Any]:
        if contract not in classes:
            classes[contract] = {
                "call_count": 0,
                "models": set(),
                "abi_status": Counter(),
                "runtime_status": Counter(),
                "phases": Counter(),
                "mask_kinds": Counter(),
                "rope_policies": Counter(),
                "kv_cache_policies": Counter(),
                "sequence_policies": Counter(),
                "unsupported_reasons": Counter(),
                "example_op": "",
                "example_model": "",
                "example_source": "",
            }
        return classes[contract]

    for contract in ATTENTION_TARGET_CONTRACTS:
        get_entry(contract)

    for record in attention_records:
        contract = str(record.get("attention_contract", "")) or "attention_unclassified"
        entry = get_entry(contract)
        entry["call_count"] += 1
        if record.get("model_case"):
            entry["models"].add(str(record.get("model_case", "")))
        _counter_add(entry["abi_status"], record.get("attention_abi_status", ""))
        _counter_add(entry["runtime_status"], record.get("attention_runtime_status", ""))
        _counter_add(entry["phases"], record.get("attention_phase", ""))
        _counter_add(entry["mask_kinds"], record.get("attention_mask_kind", ""))
        _counter_add(entry["rope_policies"], record.get("attention_rope_policy", ""))
        _counter_add(entry["kv_cache_policies"], record.get("attention_kv_cache_policy", ""))
        _counter_add(entry["sequence_policies"], record.get("attention_sequence_policy", ""))
        _counter_add(entry["unsupported_reasons"], record.get("unsupported_attention_reason", ""))
        if not entry["example_op"]:
            entry["example_op"] = str(record.get("op_name", ""))
            entry["example_model"] = str(record.get("model_case", ""))
            entry["example_source"] = str(record.get("source", ""))

    normalized = {}
    for contract, entry in sorted(classes.items()):
        models = sorted(entry["models"])
        normalized[contract] = {
            "call_count": int(entry["call_count"]),
            "models_impacted": len(models),
            "models": models,
            "abi_status": dict(sorted(entry["abi_status"].items())),
            "runtime_status": dict(sorted(entry["runtime_status"].items())),
            "phases": dict(sorted(entry["phases"].items())),
            "mask_kinds": dict(sorted(entry["mask_kinds"].items())),
            "rope_policies": dict(sorted(entry["rope_policies"].items())),
            "kv_cache_policies": dict(sorted(entry["kv_cache_policies"].items())),
            "sequence_policies": dict(sorted(entry["sequence_policies"].items())),
            "unsupported_reasons": dict(sorted(entry["unsupported_reasons"].items())),
            "example_op": entry["example_op"],
            "example_model": entry["example_model"],
            "example_source": entry["example_source"],
        }
    return normalized


def _counter_add(counter: Counter, value: Any) -> None:
    text = str(value or "")
    if text:
        counter[text] += 1


def _pre_m10_entry_from_attention_contract(
    contract: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "attention_contract": contract,
        "call_count": int(entry.get("call_count", 0)),
        "models_impacted": int(entry.get("models_impacted", 0)),
        "models": list(entry.get("models", [])),
        "runtime_status": dict(entry.get("runtime_status", {})),
        "unsupported_reasons": dict(entry.get("unsupported_reasons", {})),
        "example_op": entry.get("example_op", ""),
        "example_model": entry.get("example_model", ""),
        "example_source": entry.get("example_source", ""),
    }


def _m10_attention_runtime_section(extern_records: list[dict[str, Any]]) -> dict[str, Any]:
    attention_records = [
        record
        for record in extern_records
        if str(record.get("op_family", "")) == "deferred_attention"
    ]
    artifact_only_records = [
        record for record in attention_records if _is_materialized_attention_artifact(record)
    ]
    runtime_resolved = [
        record for record in attention_records if _is_runtime_resolved_attention_record(record)
    ]
    artifact_records = artifact_only_records + runtime_resolved
    status_counts = Counter(
        str(record.get("attention_runtime_status", ""))
        or ATTENTION_RUNTIME_STATUS_DEFERRED
        for record in attention_records
    )
    provider_counts = Counter(
        str(record.get("attention_provider_kind", "")) or ATTENTION_PROVIDER_NONE
        for record in attention_records
    )
    unsupported_runtime_reasons = Counter(
        str(record.get("unsupported_attention_runtime_reason", ""))
        for record in attention_records
        if str(record.get("unsupported_attention_runtime_reason", ""))
    )
    total_runtime_launches = sum(
        int(record.get("attention_runtime_launch_count", 0) or 0)
        for record in attention_records
    )
    total_artifact_calls = sum(
        int(record.get("attention_artifact_call_count", 0) or 0)
        for record in attention_records
    )
    total_io_bytes = sum(
        int(record.get("attention_total_io_bytes", 0) or 0)
        for record in attention_records
    )
    total_intermediate_bytes = sum(
        int(record.get("attention_intermediate_buffer_bytes", 0) or 0)
        for record in attention_records
    )
    total_host_staging_bytes = sum(
        int(record.get("attention_host_staging_bytes", 0) or 0)
        for record in attention_records
    )
    total_accounted_bytes = sum(
        int(record.get("attention_total_accounted_bytes", 0) or 0)
        for record in attention_records
    )
    return {
        "taxonomy_version": 1,
        "provider_policy": "opt_in_correctness_only",
        "performance_claim": False,
        "hardening_status": "m10_runtime_hardened_v1",
        "rope_runtime_policy": "deferred_explicit",
        "kv_cache_runtime_policy": "deferred_explicit",
        "decode_runtime_policy": "synthetic_report_only",
        "artifact_count": len(artifact_records),
        "artifact_only_count": len(artifact_only_records),
        "runtime_resolved_count": len(runtime_resolved),
        "runtime_launch_count": total_runtime_launches,
        "artifact_call_count": total_artifact_calls,
        "total_io_bytes": total_io_bytes,
        "intermediate_buffer_bytes": total_intermediate_bytes,
        "host_staging_bytes": total_host_staging_bytes,
        "total_accounted_bytes": total_accounted_bytes,
        "status_counts": dict(sorted(status_counts.items())),
        "provider_counts": dict(sorted(provider_counts.items())),
        "unsupported_runtime_reasons": dict(sorted(unsupported_runtime_reasons.items())),
        "runtime_resolved_records": [
            {
                "model_case": record.get("model_case", ""),
                "op_name": record.get("op_name", ""),
                "line_no": int(record.get("line_no", 0) or 0),
                "attention_contract": record.get("attention_contract", ""),
                "attention_provider_kind": record.get("attention_provider_kind", ""),
                "attention_provider_abi_version": record.get(
                    "attention_provider_abi_version",
                ),
                "attention_runtime_claim": record.get("attention_runtime_claim", ""),
                "attention_performance_claim": bool(
                    record.get("attention_performance_claim", False)
                ),
                "attention_uses_host_staging": bool(
                    record.get("attention_uses_host_staging", False)
                ),
                "attention_phase": record.get("attention_phase", ""),
                "attention_mask_kind": record.get("attention_mask_kind", ""),
                "attention_q_shape": record.get("attention_q_shape", ""),
                "attention_mask_shape": record.get("attention_mask_shape", ""),
                "attention_scale": record.get("attention_scale"),
                "attention_runtime_launch_count": int(
                    record.get("attention_runtime_launch_count", 0) or 0
                ),
                "attention_total_io_bytes": int(
                    record.get("attention_total_io_bytes", 0) or 0
                ),
                "attention_intermediate_buffer_bytes": int(
                    record.get("attention_intermediate_buffer_bytes", 0) or 0
                ),
                "attention_host_staging_bytes": int(
                    record.get("attention_host_staging_bytes", 0) or 0
                ),
                "attention_total_accounted_bytes": int(
                    record.get("attention_total_accounted_bytes", 0) or 0
                ),
            }
            for record in runtime_resolved
        ],
    }


def _m11_vision_runtime_section(
    extern_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    kernel_records: list[dict[str, Any]],
) -> dict[str, Any]:
    conv_records = [
        record
        for record in extern_records
        if str(record.get("op_family", "")) == "deferred_convolution"
    ]
    materialized_records = [
        record for record in conv_records if _is_materialized_vision_conv_record(record)
    ]
    artifact_only_records = [
        record for record in conv_records if _is_materialized_vision_conv_artifact(record)
    ]
    runtime_resolved = [
        record for record in conv_records if _is_runtime_resolved_vision_record(record)
    ]
    status_counts = Counter(
        str(record.get("vision_runtime_status", "")) or "unsupported"
        for record in conv_records
    )
    contract_counts = Counter(
        str(record.get("vision_contract", "")) or "vision_unclassified"
        for record in conv_records
    )
    semantics_status_counts = Counter(
        str(record.get("vision_semantics_status", "")) or "vision_semantics_unclassified"
        for record in conv_records
    )
    layout_counts = Counter(
        str(record.get("vision_layout", "")) or "unknown"
        for record in conv_records
    )
    unsupported_reasons = Counter(
        str(record.get("unsupported_vision_reason", ""))
        for record in conv_records
        if str(record.get("unsupported_vision_reason", ""))
    )
    unsupported_runtime_reasons = Counter(
        str(record.get("unsupported_vision_runtime_reason", ""))
        for record in conv_records
        if str(record.get("unsupported_vision_runtime_reason", ""))
    )
    provider_counts = Counter(
        str(record.get("vision_provider_kind", "")) or VISION_PROVIDER_NONE
        for record in conv_records
    )
    runnable_models = sum(1 for record in model_records if record.get("full_tvm_runnable"))
    runtime_launch_count = sum(
        int(record.get("vision_runtime_launch_count", 0) or 0)
        for record in conv_records
    )
    artifact_call_count = sum(
        int(record.get("vision_artifact_call_count", 0) or 0)
        for record in conv_records
    )
    provider_enabled = any(
        str(record.get("vision_provider_kind", "")) == VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
        for record in runtime_resolved
    )
    hardening = _m11_vision_hardening_section(
        conv_records,
        materialized_records,
        artifact_only_records,
        runtime_resolved,
        model_records,
        kernel_records,
        provider_enabled=provider_enabled,
    )
    grid_readiness = _m11_grid2d_readiness_section(kernel_records)
    model_smoke = _m11_runtime_resolved_model_smoke_section(
        model_records,
        kernel_records,
        extern_records,
    )
    m11_6_enabled = provider_enabled or int(grid_readiness.get("native_runtime_ready_count", 0))
    return {
        "taxonomy_version": 1,
        "interface_status": (
            VISION_M11_6_INTERFACE_STATUS
            if m11_6_enabled
            else M11_GRID_TAXONOMY_VERSION
        ),
        "convolution_interface_status": "m11_2_artifact_only_convolution_boundary_v1",
        "runtime_interface_status": (
            VISION_M11_6_INTERFACE_STATUS if m11_6_enabled else ""
        ),
        "hardening_status": VISION_M11_5_HARDENING_STATUS if provider_enabled else "",
        "runtime_scope_status": VISION_M11_6_RUNTIME_SCOPE_STATUS if provider_enabled else "",
        "provider_policy": (
            "opt_in_python_torch_host_staged_correctness_only"
            if provider_enabled
            else "artifact_only_no_runtime_provider"
        ),
        "performance_claim": False,
        "vision_contract_version": VISION_CONTRACT_VERSION,
        "detail_fields": list(VISION_REPORT_FIELDS) + list(M11_GRID_REPORT_FIELDS),
        "observed_convolution_call_count": len(conv_records),
        "artifact_count": len(materialized_records),
        "artifact_only_count": len(artifact_only_records),
        "runtime_resolved_count": len(runtime_resolved),
        "runtime_launch_count": runtime_launch_count,
        "artifact_call_count": artifact_call_count,
        "provider_counts": dict(sorted(provider_counts.items())),
        "total_io_bytes": sum(
            int(record.get("vision_total_io_bytes", 0) or 0) for record in conv_records
        ),
        "host_staging_bytes": sum(
            int(record.get("vision_host_staging_bytes", 0) or 0)
            for record in conv_records
        ),
        "total_accounted_bytes": sum(
            int(record.get("vision_total_accounted_bytes", 0) or 0)
            for record in conv_records
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "contract_counts": dict(sorted(contract_counts.items())),
        "semantics_status_counts": dict(sorted(semantics_status_counts.items())),
        "layout_counts": dict(sorted(layout_counts.items())),
        "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
        "unsupported_runtime_reasons": dict(sorted(unsupported_runtime_reasons.items())),
        "artifact_records": [
            {
                "model_case": record.get("model_case", ""),
                "op_name": record.get("op_name", ""),
                "line_no": int(record.get("line_no", 0) or 0),
                "vision_contract": record.get("vision_contract", ""),
                "vision_layout": record.get("vision_layout", ""),
                "vision_input_shape": record.get("vision_input_shape", ""),
                "vision_weight_shape": record.get("vision_weight_shape", ""),
                "vision_output_shape": record.get("vision_output_shape", ""),
                "vision_stride": record.get("vision_stride", ""),
                "vision_padding": record.get("vision_padding", ""),
                "vision_dilation": record.get("vision_dilation", ""),
                "vision_groups": int(record.get("vision_groups", 0) or 0),
                "vision_runtime_status": record.get("vision_runtime_status", ""),
                "vision_performance_claim": bool(
                    record.get("vision_performance_claim", False)
                ),
            }
            for record in materialized_records
        ],
        "runtime_resolved_records": [
            {
                "model_case": record.get("model_case", ""),
                "op_name": record.get("op_name", ""),
                "line_no": int(record.get("line_no", 0) or 0),
                "vision_contract": record.get("vision_contract", ""),
                "vision_provider_kind": record.get("vision_provider_kind", ""),
                "vision_provider_abi_version": int(
                    record.get("vision_provider_abi_version", 0) or 0
                ),
                "vision_runtime_claim": record.get("vision_runtime_claim", ""),
                "vision_performance_claim": bool(
                    record.get("vision_performance_claim", False)
                ),
                "vision_uses_host_staging": bool(
                    record.get("vision_uses_host_staging", False)
                ),
                "vision_runtime_launch_count": int(
                    record.get("vision_runtime_launch_count", 0) or 0
                ),
                "vision_artifact_call_count": int(
                    record.get("vision_artifact_call_count", 0) or 0
                ),
                "vision_input_bytes": int(record.get("vision_input_bytes", 0) or 0),
                "vision_weight_bytes": int(record.get("vision_weight_bytes", 0) or 0),
                "vision_output_bytes": int(record.get("vision_output_bytes", 0) or 0),
                "vision_total_io_bytes": int(
                    record.get("vision_total_io_bytes", 0) or 0
                ),
                "vision_host_staging_bytes": int(
                    record.get("vision_host_staging_bytes", 0) or 0
                ),
                "vision_total_accounted_bytes": int(
                    record.get("vision_total_accounted_bytes", 0) or 0
                ),
            }
            for record in runtime_resolved
        ],
        "captured_grid_taxonomy": _m11_captured_grid_taxonomy_section(kernel_records),
        "captured_grid_readiness": grid_readiness,
        "m11_6_readiness": {
            "status": VISION_M11_6_INTERFACE_STATUS if m11_6_enabled else "not_enabled",
            "conv_runtime_target": {
                "target_min_runtime_resolved": 10,
                "observed_runtime_resolved": len(runtime_resolved),
                "runtime_scope_status": (
                    VISION_M11_6_RUNTIME_SCOPE_STATUS if provider_enabled else ""
                ),
                "provider_kind": (
                    VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
                    if provider_enabled
                    else VISION_PROVIDER_NONE
                ),
                "runtime_claim": (
                    VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY if provider_enabled else ""
                ),
                "performance_claim": False,
            },
            "grid2d_target": {
                "target_classified": 48,
                "target_artifact_generated_min": 10,
                "target_native_runtime_min": 5,
                "classified_count": grid_readiness.get("classified_count", 0),
                "artifact_ready_count": grid_readiness.get("artifact_ready_count", 0),
                "native_runtime_ready_count": grid_readiness.get(
                    "native_runtime_ready_count", 0
                ),
                "silent_fallback_count": grid_readiness.get("silent_fallback_count", 0),
            },
        },
        "runtime_resolved_model_smoke": model_smoke["runtime_resolved_model_smoke"],
        "runtime_resolved_model_provider_mix": model_smoke[
            "runtime_resolved_model_provider_mix"
        ],
        "full_tvm_native_model": model_smoke["full_tvm_native_model"],
        "full_tvm_runnable_model": runnable_models,
        "report_cache_invariants": hardening["report_cache_invariants"],
        "corpus_diff_guard": hardening["corpus_diff_guard"],
        "m10_attention_boundary": hardening["m10_attention_boundary"],
        "runtime_scope": hardening["runtime_scope"],
        "model_full_tvm_runnable_after_m11_2_gate": runnable_models,
        "model_full_tvm_runnable_after_m11_3_gate": runnable_models,
        "model_full_tvm_runnable_after_m11_4_gate": runnable_models,
        "model_full_tvm_runnable_after_m11_5_gate": runnable_models,
        "model_full_tvm_runnable_after_m11_6_gate": runnable_models,
    }


def _m11_vision_hardening_section(
    conv_records: list[dict[str, Any]],
    materialized_records: list[dict[str, Any]],
    artifact_only_records: list[dict[str, Any]],
    runtime_resolved: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    kernel_records: list[dict[str, Any]],
    *,
    provider_enabled: bool,
) -> dict[str, Any]:
    attention_records = [
        record
        for record in _extern_records_from_models(model_records)
        if str(record.get("op_family", "")) == "deferred_attention"
    ]
    runtime_attention = [
        record for record in attention_records if _is_runtime_resolved_attention_record(record)
    ]
    runnable_models = sum(1 for record in model_records if record.get("full_tvm_runnable"))
    failures = _m11_vision_hardening_failures(
        conv_records,
        materialized_records,
        artifact_only_records,
        runtime_resolved,
        attention_records,
        runtime_attention,
        runnable_models,
        provider_enabled=provider_enabled,
    )
    invariant_status = (
        "passed"
        if provider_enabled and not failures
        else "failed"
        if provider_enabled
        else "not_applicable_no_vision_runtime_provider"
    )
    status_buckets = Counter(
        str((record.get("translate_status") or {}).get("bucket", ""))
        for record in kernel_records
        if (record.get("translate_status") or {}).get("bucket", "")
    )
    translated_kernel_count = sum(
        1 for record in kernel_records if (record.get("translate_status") or {}).get("ok")
    )
    captured_grid_count = sum(
        1
        for record in kernel_records
        if str(record.get("m11_grid_status", "")) == "m11_grid_classified"
    )
    return {
        "report_cache_invariants": {
            "status": invariant_status,
            "schema_version": 1,
            "hardening_status": (
                VISION_M11_5_HARDENING_STATUS if provider_enabled else ""
            ),
            "vision_report_fields_frozen": list(VISION_REPORT_FIELDS),
            "m11_grid_report_fields_frozen": list(M11_GRID_REPORT_FIELDS),
            "implicit_pytorch_fallback_allowed": False,
            "runtime_provider_opt_in": provider_enabled,
            "runtime_provider": (
                VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
                if provider_enabled
                else VISION_PROVIDER_NONE
            ),
            "artifact_call_count_matches_materialized": sum(
                int(record.get("vision_artifact_call_count", 0) or 0)
                for record in conv_records
            )
            == len(materialized_records),
            "runtime_launch_count_matches_runtime_records": sum(
                int(record.get("vision_runtime_launch_count", 0) or 0)
                for record in conv_records
            )
            == len(runtime_resolved),
            "runtime_provider_performance_claim": False,
            "full_model_runnable_claim": False,
            "invariant_failures": failures,
        },
        "corpus_diff_guard": {
            "baseline_id": (
                VISION_M11_6_CORPUS_DIFF_BASELINE_ID
                if provider_enabled
                else VISION_M11_5_CORPUS_DIFF_BASELINE_ID
            ),
            "baseline_status": "recorded" if provider_enabled else "not_applicable",
            "schema_version": 1,
            "status_buckets": dict(sorted(status_buckets.items())),
            "captured_kernel_count": len(kernel_records),
            "translated_kernel_count": translated_kernel_count,
            "captured_grid_fallback_count": captured_grid_count,
            "observed_convolution_call_count": len(conv_records),
            "vision_artifact_count": len(materialized_records),
            "vision_artifact_only_count": len(artifact_only_records),
            "vision_runtime_resolved_count": len(runtime_resolved),
            "vision_runtime_launch_count": sum(
                int(record.get("vision_runtime_launch_count", 0) or 0)
                for record in conv_records
            ),
            "vision_artifact_call_count": sum(
                int(record.get("vision_artifact_call_count", 0) or 0)
                for record in conv_records
            ),
            "runtime_resolved_attention_count": len(runtime_attention),
            "full_tvm_runnable_models": runnable_models,
        },
        "m10_attention_boundary": {
            "status": (
                "m10_attention_boundary_closed"
                if len(attention_records) == len(runtime_attention)
                else "m10_attention_boundary_not_closed"
            ),
            "historical_deferred_attention_family_count": len(attention_records),
            "runtime_resolved_attention_count": len(runtime_attention),
            "runtime_deferred_attention_count": len(attention_records)
            - len(runtime_attention),
            "boundary_policy": "do_not_reopen_m10_attention_in_m11_5",
        },
        "runtime_scope": {
            "status": VISION_M11_6_RUNTIME_SCOPE_STATUS if provider_enabled else "",
            "provider": (
                VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED
                if provider_enabled
                else VISION_PROVIDER_NONE
            ),
            "runtime_claim": (
                VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY if provider_enabled else ""
            ),
            "performance_claim": False,
            "supported_contracts": [
                VISION_CONTRACT_CONV2D_NCHW_STATIC,
                "conv2d_1x1_nchw_static_v1",
            ],
            "supported_batch": "N=1",
            "supported_stride": "1, 2, or 16",
            "supported_padding": "0 or 1",
            "supported_dilation": "1, 1",
            "supported_groups": 1,
            "unsupported_runtime_reason_for_other_convs": (
                VISION_RUNTIME_PROVIDER_SCOPE_REASON if provider_enabled else ""
            ),
            "observed_runtime_records": len(runtime_resolved),
        },
    }


def _m11_vision_hardening_failures(
    conv_records: list[dict[str, Any]],
    materialized_records: list[dict[str, Any]],
    artifact_only_records: list[dict[str, Any]],
    runtime_resolved: list[dict[str, Any]],
    attention_records: list[dict[str, Any]],
    runtime_attention: list[dict[str, Any]],
    runnable_models: int,
    *,
    provider_enabled: bool,
) -> list[str]:
    failures: list[str] = []
    artifact_call_count = sum(
        int(record.get("vision_artifact_call_count", 0) or 0) for record in conv_records
    )
    runtime_launch_count = sum(
        int(record.get("vision_runtime_launch_count", 0) or 0) for record in conv_records
    )
    if artifact_call_count != len(materialized_records):
        failures.append("vision_artifact_call_count_mismatch")
    if runtime_launch_count != len(runtime_resolved):
        failures.append("vision_runtime_launch_count_mismatch")
    if runnable_models:
        failures.append("m11_full_tvm_runnable_claim_not_allowed")
    if len(attention_records) != len(runtime_attention):
        failures.append("m10_attention_boundary_not_closed")

    for record in runtime_resolved:
        if not _m11_record_is_static_conv_runtime_scope(record):
            failures.append("vision_runtime_scope_not_m11_6_static_conv2d")
            break
        if str(record.get("vision_provider_kind", "")) != VISION_PROVIDER_PYTHON_TORCH_HOST_STAGED:
            failures.append("vision_runtime_provider_kind_mismatch")
            break
        if str(record.get("vision_runtime_claim", "")) != VISION_RUNTIME_CLAIM_CORRECTNESS_ONLY:
            failures.append("vision_runtime_claim_not_correctness_only")
            break
        if bool(record.get("vision_performance_claim", False)):
            failures.append("vision_runtime_performance_claim_not_allowed")
            break
        if not bool(record.get("vision_uses_host_staging", False)):
            failures.append("vision_runtime_host_staging_not_recorded")
            break
        if not _m11_record_accounting_is_consistent(record, runtime_resolved=True):
            failures.append("vision_runtime_byte_accounting_mismatch")
            break

    for record in artifact_only_records:
        if str(record.get("vision_provider_kind", "") or VISION_PROVIDER_NONE) != VISION_PROVIDER_NONE:
            failures.append("vision_artifact_only_provider_not_none")
            break
        if int(record.get("vision_runtime_launch_count", 0) or 0) != 0:
            failures.append("vision_artifact_only_runtime_launch_not_zero")
            break
        if bool(record.get("vision_uses_host_staging", False)):
            failures.append("vision_artifact_only_host_staging_not_allowed")
            break
        if bool(record.get("vision_performance_claim", False)):
            failures.append("vision_artifact_only_performance_claim_not_allowed")
            break
        if provider_enabled and not str(record.get("unsupported_vision_runtime_reason", "")):
            failures.append("vision_artifact_only_scope_reason_missing")
            break
        if not _m11_record_accounting_is_consistent(record, runtime_resolved=False):
            failures.append("vision_artifact_only_byte_accounting_mismatch")
            break
    return sorted(set(failures))


def _m11_record_is_static_conv_runtime_scope(record: dict[str, Any]) -> bool:
    contract = str(record.get("vision_contract", ""))
    stride = _m11_int_tuple_record(record, "vision_stride")
    padding = _m11_int_tuple_record(record, "vision_padding")
    dilation = _m11_int_tuple_record(record, "vision_dilation")
    input_shape = _m11_int_tuple_record(record, "vision_input_shape")
    weight_shape = _m11_int_tuple_record(record, "vision_weight_shape")
    if (
        len(input_shape) != 4
        or len(weight_shape) != 4
        or input_shape[0] != 1
        or int(record.get("vision_groups", 0) or 0) != 1
        or str(record.get("vision_bias_policy", "")) != "none"
        or bool(record.get("vision_transposed", False))
        or _m11_int_tuple_record(record, "vision_output_padding") != (0, 0)
        or dilation != (1, 1)
    ):
        return False
    if contract == "conv2d_1x1_nchw_static_v1":
        return weight_shape[2:] == (1, 1) and stride == (1, 1) and padding == (0, 0)
    if contract != VISION_CONTRACT_CONV2D_NCHW_STATIC:
        return False
    if len(stride) != 2 or len(padding) != 2:
        return False
    return stride[0] == stride[1] and padding[0] == padding[1] and stride[0] in {
        1,
        2,
        16,
    } and padding[0] in {0, 1}


def _m11_record_accounting_is_consistent(
    record: dict[str, Any],
    *,
    runtime_resolved: bool,
) -> bool:
    input_shape = _m11_int_tuple_record(record, "vision_input_shape")
    weight_shape = _m11_int_tuple_record(record, "vision_weight_shape")
    output_shape = _m11_int_tuple_record(record, "vision_output_shape")
    if not input_shape or not weight_shape or not output_shape:
        return False
    input_bytes = _m11_numel(input_shape) * 4
    weight_bytes = _m11_numel(weight_shape) * 4
    output_bytes = _m11_numel(output_shape) * 4
    total_io_bytes = input_bytes + weight_bytes + output_bytes
    host_staging_bytes = total_io_bytes if runtime_resolved else 0
    return (
        int(record.get("vision_input_bytes", 0) or 0) == input_bytes
        and int(record.get("vision_weight_bytes", 0) or 0) == weight_bytes
        and int(record.get("vision_output_bytes", 0) or 0) == output_bytes
        and int(record.get("vision_total_io_bytes", 0) or 0) == total_io_bytes
        and int(record.get("vision_host_staging_bytes", 0) or 0) == host_staging_bytes
        and int(record.get("vision_total_accounted_bytes", 0) or 0)
        == total_io_bytes + host_staging_bytes
    )


def _m11_int_tuple_record(record: dict[str, Any], field: str) -> tuple[int, ...]:
    value = record.get(field, "")
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _m11_tuple_text(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values)


def _m11_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _m11_captured_grid_taxonomy_section(
    kernel_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grid_records = [
        record
        for record in kernel_records
        if str(record.get("m11_grid_status", "")) == "m11_grid_classified"
    ]
    family_counts = Counter(str(record.get("m11_grid_family", "")) for record in grid_records)
    grid_type_counts = Counter(
        str(record.get("m11_grid_launch_kind", "")) or "unknown"
        for record in grid_records
    )
    model_counts = Counter(str(record.get("model_case", "")) for record in grid_records)
    operator_tag_counts: Counter[str] = Counter()
    for record in grid_records:
        operator_tag_counts.update(
            tag.strip()
            for tag in str(record.get("m11_grid_operator_tags", "")).split(",")
            if tag.strip()
        )
    artifact_ready_count = sum(
        1
        for record in grid_records
        if str(record.get("m11_grid_artifact_status", "")) == M11_GRID2D_ARTIFACT_READY
    )
    native_runtime_ready_count = sum(
        1
        for record in grid_records
        if str(record.get("m11_grid_runtime_status", "")) == M11_GRID2D_RUNTIME_READY
    )
    performance_claim_count = sum(
        1 for record in grid_records if bool(record.get("m11_grid_performance_claim", False))
    )
    unsupported_runtime_reasons = Counter(
        str(record.get("unsupported_m11_grid_runtime_reason", ""))
        for record in grid_records
        if str(record.get("unsupported_m11_grid_runtime_reason", ""))
    )

    family_counts_with_zeros = {
        family: int(family_counts.get(family, 0)) for family in M11_GRID_FAMILIES
    }
    return {
        "taxonomy_version": M11_GRID_TAXONOMY_VERSION,
        "status": "m11_6_captured_grid2d_readiness_classified_v1"
        if grid_records
        else "m11_3_captured_grid_taxonomy_classified_v1",
        "readiness_version": M11_GRID2D_READINESS_VERSION,
        "detail_fields": list(M11_GRID_REPORT_FIELDS),
        "kernel_count": len(grid_records),
        "artifact_ready_count": artifact_ready_count,
        "native_runtime_ready_count": native_runtime_ready_count,
        "performance_claim_count": performance_claim_count,
        "silent_fallback_count": 0,
        "unsupported_runtime_reasons": dict(sorted(unsupported_runtime_reasons.items())),
        "family_counts": family_counts_with_zeros,
        "grid_type_counts": dict(sorted(grid_type_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "operator_tag_counts": dict(sorted(operator_tag_counts.items())),
        "record_sample_limit": 16,
        "records": [
            {
                "model_case": record.get("model_case", ""),
                "kernel_name": record.get("kernel_name", ""),
                "m11_grid_family": record.get("m11_grid_family", ""),
                "m11_grid_launch_kind": record.get("m11_grid_launch_kind", ""),
                "m11_grid_program_axes": record.get("m11_grid_program_axes", ""),
                "m11_grid_size_hints": record.get("m11_grid_size_hints", ""),
                "m11_grid_operator_tags": record.get("m11_grid_operator_tags", ""),
                "m11_grid_contract": record.get("m11_grid_contract", ""),
                "m11_grid_artifact_status": record.get("m11_grid_artifact_status", ""),
                "m11_grid_runtime_status": record.get("m11_grid_runtime_status", ""),
                "m11_grid_implementation_kind": record.get(
                    "m11_grid_implementation_kind", ""
                ),
                "unsupported_m11_grid_reason": record.get(
                    "unsupported_m11_grid_reason", ""
                ),
                "unsupported_m11_grid_runtime_reason": record.get(
                    "unsupported_m11_grid_runtime_reason", ""
                ),
            }
            for record in grid_records[:16]
        ],
    }


def _m11_grid2d_readiness_section(kernel_records: list[dict[str, Any]]) -> dict[str, Any]:
    grid_records = [
        record
        for record in kernel_records
        if str(record.get("m11_grid_status", "")) == "m11_grid_classified"
    ]
    artifact_ready = [
        record
        for record in grid_records
        if str(record.get("m11_grid_artifact_status", "")) == M11_GRID2D_ARTIFACT_READY
    ]
    native_ready = [
        record
        for record in grid_records
        if str(record.get("m11_grid_runtime_status", "")) == M11_GRID2D_RUNTIME_READY
    ]
    unsupported_reasons = Counter(
        str(record.get("unsupported_m11_grid_runtime_reason", ""))
        for record in grid_records
        if str(record.get("unsupported_m11_grid_runtime_reason", ""))
    )
    return {
        "status": M11_GRID2D_READINESS_VERSION if grid_records else "not_applicable",
        "contract": VISION_CONTRACT_POINTWISE_GRID2D_STATIC,
        "implementation_kind": M11_GRID2D_IMPLEMENTATION_KIND_NATIVE_TVM,
        "provider_kind": M11_GRID2D_PROVIDER_NATIVE_TVM,
        "classified_count": len(grid_records),
        "artifact_ready_count": len(artifact_ready),
        "native_runtime_ready_count": len(native_ready),
        "runtime_launch_count": sum(
            int(record.get("m11_grid_runtime_launch_count", 0) or 0)
            for record in native_ready
        ),
        "artifact_call_count": sum(
            int(record.get("m11_grid_artifact_call_count", 0) or 0)
            for record in artifact_ready
        ),
        "host_staging_bytes": sum(
            int(record.get("m11_grid_host_staging_bytes", 0) or 0)
            for record in native_ready
        ),
        "performance_claim_count": sum(
            1 for record in native_ready if bool(record.get("m11_grid_performance_claim", False))
        ),
        "silent_fallback_count": 0,
        "unsupported_runtime_count": len(grid_records) - len(native_ready),
        "unsupported_runtime_reasons": dict(sorted(unsupported_reasons.items())),
        "family_counts": dict(
            sorted(Counter(str(record.get("m11_grid_family", "")) for record in grid_records).items())
        ),
        "records": [
            {
                "model_case": record.get("model_case", ""),
                "kernel_name": record.get("kernel_name", ""),
                "m11_grid_family": record.get("m11_grid_family", ""),
                "m11_grid_size_hints": record.get("m11_grid_size_hints", ""),
                "m11_grid_artifact_status": record.get("m11_grid_artifact_status", ""),
                "m11_grid_runtime_status": record.get("m11_grid_runtime_status", ""),
                "m11_grid_implementation_kind": record.get(
                    "m11_grid_implementation_kind", ""
                ),
                "unsupported_m11_grid_runtime_reason": record.get(
                    "unsupported_m11_grid_runtime_reason", ""
                ),
            }
            for record in grid_records[:16]
        ],
    }


def _m11_runtime_resolved_model_smoke_section(
    model_records: list[dict[str, Any]],
    kernel_records: list[dict[str, Any]],
    extern_records: list[dict[str, Any]],
) -> dict[str, Any]:
    kernels_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in kernel_records:
        kernels_by_model[str(record.get("model_case", ""))].append(record)
    extern_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in extern_records:
        extern_by_model[str(record.get("model_case", ""))].append(record)

    records: list[dict[str, Any]] = []
    provider_mix: dict[str, dict[str, int]] = {}
    for model in model_records:
        model_case = str(model.get("model_case", ""))
        model_family = str(model.get("model_family", ""))
        kernels = kernels_by_model.get(model_case, [])
        externs = extern_by_model.get(model_case, [])
        translated = sum(1 for record in kernels if (record.get("translate_status") or {}).get("ok"))
        grid_ready = sum(
            1
            for record in kernels
            if str(record.get("m11_grid_runtime_status", "")) == M11_GRID2D_RUNTIME_READY
        )
        grid_unsupported = sum(
            1
            for record in kernels
            if str(record.get("m11_grid_status", "")) == "m11_grid_classified"
            and str(record.get("m11_grid_runtime_status", "")) != M11_GRID2D_RUNTIME_READY
        )
        conv_records = [
            record for record in externs if str(record.get("op_family", "")) == "deferred_convolution"
        ]
        conv_runtime = sum(1 for record in conv_records if _is_runtime_resolved_vision_record(record))
        conv_artifact = sum(1 for record in conv_records if _is_materialized_vision_conv_artifact(record))
        attention_records = [
            record for record in externs if str(record.get("op_family", "")) == "deferred_attention"
        ]
        attention_runtime = sum(
            1 for record in attention_records if _is_runtime_resolved_attention_record(record)
        )
        attention_deferred = len(attention_records) - attention_runtime
        full_native = bool(model.get("full_tvm_runnable", False)) and not conv_artifact
        vision_model = model_family in {"vit", "yolo"}
        partial_smoke_ok = (
            vision_model
            and bool(conv_runtime or grid_ready)
            and attention_deferred == 0
        )
        smoke_ok = (
            partial_smoke_ok
            and grid_unsupported == 0
            and attention_deferred == 0
        )
        mix = {
            "translated_captured_kernels": translated,
            "native_grid2d_runtime_ready": grid_ready,
            "unsupported_grid2d_runtime": grid_unsupported,
            "conv_runtime_resolved": conv_runtime,
            "conv_artifact_only": conv_artifact,
            "attention_runtime_resolved": attention_runtime,
            "attention_deferred": attention_deferred,
        }
        provider_mix[model_case] = mix
        records.append(
            {
                "model_case": model_case,
                "model_family": model_family,
                "runtime_resolved_model_smoke": smoke_ok,
                "partial_runtime_resolved_model_smoke": partial_smoke_ok,
                "provider_mix": mix,
                "full_tvm_native_model": full_native,
                "full_tvm_runnable_model": bool(model.get("full_tvm_runnable", False)),
            }
        )
    return {
        "runtime_resolved_model_smoke": {
            "schema_version": 1,
            "definition": (
                "diagnostic vision readiness smoke: ViT/YOLO captured kernels are "
                "translated or native Grid2D-ready, wrapper calls have explicit runtime "
                "providers or explicit artifacts, and no silent fallback is counted as "
                "runnable closure"
            ),
            "model_count": sum(
                1 for record in records if record.get("runtime_resolved_model_smoke")
            ),
            "partial_model_count": sum(
                1
                for record in records
                if record.get("partial_runtime_resolved_model_smoke")
            ),
            "records": records,
        },
        "runtime_resolved_model_provider_mix": provider_mix,
        "full_tvm_native_model": sum(1 for record in records if record["full_tvm_native_model"]),
    }


def _pre_m11_report_section(
    kernel_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    extern_records: list[dict[str, Any]],
) -> dict[str, Any]:
    extern_family_classes = _pre_m9_extern_family_classes(extern_records)
    convolution_entry = extern_family_classes.get(
        "deferred_convolution",
        _empty_pre_m9_extern_family_entry(),
    )
    grid_entry = _pre_m11_grid_debt_entry(kernel_records)
    attention_records = [
        record
        for record in extern_records
        if str(record.get("op_family", "")) == "deferred_attention"
    ]
    runtime_resolved_attention = [
        record for record in attention_records if _is_runtime_resolved_attention_record(record)
    ]
    attention_status_counts = Counter(
        str(record.get("attention_runtime_status", ""))
        or ATTENTION_RUNTIME_STATUS_DEFERRED
        for record in attention_records
    )
    attention_provider_counts = Counter(
        str(record.get("attention_provider_kind", "")) or ATTENTION_PROVIDER_NONE
        for record in attention_records
    )
    entry_debt = []
    if int(convolution_entry.get("call_count", 0) or 0) > 0:
        conv_debt = _pre_m9_entry_from_extern_family(
            "deferred_convolution",
            convolution_entry,
        )
        conv_debt["debt_kind"] = "wrapper_convolution"
        entry_debt.append(conv_debt)
    if int(grid_entry.get("kernel_count", 0) or 0) > 0:
        grid_debt = dict(grid_entry)
        grid_debt["debt_kind"] = "captured_grid"
        entry_debt.append(grid_debt)

    return {
        "taxonomy_version": PRE_M11_TAXONOMY_VERSION,
        "gate_status": "pre_m11_debt_classified_v1",
        "report_policy": {
            "scope": (
                "Pre-M11 classifies vision/convolution wrapper debt and captured "
                "Triton grid blockers before M11 implementation."
            ),
            "schema_policy": (
                "Schema-v1 top-level buckets and captured-kernel accounting remain "
                "stable; Pre-M11 details are additive report fields."
            ),
            "fallback_policy": (
                "Implicit PyTorch fallback is disallowed. TVM externs are allowed "
                "only when explicit in reports."
            ),
            "full_tvm_runnable": (
                "full_tvm_runnable remains false until all captured-kernel "
                "fallbacks and wrapper-level extern calls are resolved."
            ),
            "attention_boundary": (
                "Historical deferred_attention wrapper-family counts remain "
                "visible, but M10 runtime-resolved observed SDPA is not M11 "
                "vision entry debt."
            ),
        },
        "detail_fields": [
            "fallback_reason",
            "blocker_class",
            "pre_m8_family",
            "op_family",
            "vision_op_policy_family",
            "vision_layout_policy",
            "vision_implementation_policy",
            "attention_runtime_status",
            "attention_provider_kind",
        ],
        "primary_debt_counts": {
            "deferred_convolution": int(convolution_entry.get("call_count", 0) or 0),
            "captured_grid": int(grid_entry.get("kernel_count", 0) or 0),
        },
        "entry_debt": entry_debt,
        "vision_op_policy": _pre_m11_vision_op_policy(),
        "grid_policy": {
            "entry_policy": "explicit_grid_blocker_until_m11_grid_or_launch_policy",
            "implementation_requirement": (
                "Expand supported grid/launch semantics or keep a stable "
                "unsupported taxonomy; do not silently translate unsupported "
                "multi-dimensional launch patterns."
            ),
            "debt": grid_entry,
        },
        "convolution_policy": {
            "entry_policy": "explicit_wrapper_debt_before_runtime_or_native_lowering",
            "implementation_requirement": (
                "Convolution/operator records must expose TVM extern or native "
                "TIR/TIRX implementation choices before affecting full-model "
                "runnable claims."
            ),
            "debt": _pre_m9_entry_from_extern_family(
                "deferred_convolution",
                convolution_entry,
            ),
        },
        "attention_boundary": {
            "m11_entry_debt": False,
            "historical_deferred_attention_family_count": len(attention_records),
            "runtime_resolved_attention_count": len(runtime_resolved_attention),
            "runtime_deferred_attention_count": int(
                attention_status_counts.get(ATTENTION_RUNTIME_STATUS_DEFERRED, 0)
            ),
            "status_counts": dict(sorted(attention_status_counts.items())),
            "provider_counts": dict(sorted(attention_provider_counts.items())),
            "boundary_status": (
                "m10_runtime_closed"
                if len(attention_records) == len(runtime_resolved_attention)
                else "attention_runtime_not_closed_for_this_snapshot"
            ),
        },
        "still_deferred_elsewhere": [
            "rope_runtime",
            "kv_cache_runtime",
            "decode_runtime",
            "arbitrary_attention_masks",
            "attention_performance_claim",
        ],
        "model_full_tvm_runnable_after_pre_m11_gate": sum(
            1 for record in model_records if record.get("full_tvm_runnable")
        ),
    }


def _pre_m11_vision_op_policy() -> dict[str, Any]:
    return {
        "operator_families": {
            "convolution": [
                "conv2d",
                "1x1_conv",
                "depthwise_conv",
                "grouped_conv",
            ],
            "structural": ["pool", "resize", "concat", "slice"],
            "yolo_postprocess": ["yolo_decode", "postprocess", "nms"],
        },
        "layout_policy": {
            "nchw": "supported_policy_target",
            "nhwc": "must_be_classified_before_lowering",
            "channels_last": "must_be_classified_before_lowering",
            "stride_padding_dilation": "must_be_reported_explicitly",
        },
        "implementation_policy": {
            "implicit_pytorch_fallback": "disallowed",
            "explicit_tvm_extern": "allowed_when_reported",
            "native_tir_or_tirx": "preferred_after_contract_boundary",
            "nms": "may_start_as_explicit_tvm_extern_then_move_native_later",
        },
    }


def _pre_m11_grid_debt_entry(
    kernel_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grid_records = [
        record
        for record in kernel_records
        if not record.get("translate_status", {}).get("ok", False)
        and record.get("blocker_class") == "grid"
    ]
    buckets: Counter = Counter()
    blocker_classes: Counter = Counter()
    grid_types: Counter = Counter()
    models: set[str] = set()
    families: set[str] = set()
    example_kernel = ""
    example_message = ""
    for record in grid_records:
        status = record.get("translate_status", {})
        buckets[str(status.get("bucket", "unknown"))] += 1
        blocker_classes[str(record.get("blocker_class", "unknown"))] += 1
        grid_types[str(record.get("grid_type", "unknown"))] += 1
        if record.get("model_case"):
            models.add(str(record.get("model_case", "")))
        if record.get("model_family"):
            families.add(str(record.get("model_family", "")))
        if not example_kernel:
            example_kernel = str(record.get("kernel_name", ""))
            example_message = str(status.get("message", ""))

    return {
        "pre_m11_family": "captured_grid",
        "kernel_count": len(grid_records),
        "buckets": dict(sorted(buckets.items())),
        "blocker_classes": dict(sorted(blocker_classes.items())),
        "grid_types": dict(sorted(grid_types.items())),
        "models_impacted": len(models),
        "models": sorted(models),
        "model_families": sorted(families),
        "example_kernel": example_kernel,
        "example_message": example_message,
    }


def _empty_pre_m9_extern_family_entry() -> dict[str, Any]:
    return {
        "call_count": 0,
        "op_names": {},
        "models_impacted": 0,
        "models": [],
        "example_op": "",
        "example_model": "",
        "example_source": "",
    }


def _pre_m9_materialized_artifact_candidates(
    extern_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for record in extern_records:
        if not _is_materialized_extern_artifact_record(record):
            continue
        candidates.append(
            {
                "op_family": record.get("op_family", ""),
                "op_name": record.get("op_name", ""),
                "model_case": record.get("model_case", ""),
                "line_no": int(record.get("line_no", 0) or 0),
                "matmul_contract": record.get("matmul_contract", ""),
                "matmul_source_kind": record.get("matmul_source_kind", ""),
                "matmul_m": record.get("matmul_m"),
                "matmul_n": record.get("matmul_n"),
                "matmul_k": record.get("matmul_k"),
                "matmul_contract_ok": bool(record.get("matmul_contract_ok", False)),
                "matmul_a_dtype": record.get("matmul_a_dtype", ""),
                "matmul_b_dtype": record.get("matmul_b_dtype", ""),
                "matmul_accumulator_dtype": record.get("matmul_accumulator_dtype", ""),
                "matmul_output_dtype": record.get("matmul_output_dtype", ""),
                "matmul_epilogue_kind": record.get("matmul_epilogue_kind", ""),
                "implementation_kind": record.get("implementation_kind", ""),
                "extern_symbol": record.get("extern_symbol", ""),
                "extern_packed_func": record.get("extern_packed_func", ""),
                "extern_runtime_kind": record.get("extern_runtime_kind", ""),
                "extern_runtime_replacement": record.get(
                    "extern_runtime_replacement",
                    "",
                ),
                "extern_runtime_replacement_available": bool(
                    record.get("extern_runtime_replacement_available", False)
                ),
                "extern_runtime_replacement_reason": record.get(
                    "extern_runtime_replacement_reason",
                    "",
                ),
                "extern_gemm_runtime_status": record.get(
                    "extern_gemm_runtime_status",
                    "",
                ),
                "extern_gemm_provider_kind": record.get("extern_gemm_provider_kind", ""),
                "extern_gemm_provider_abi_version": record.get(
                    "extern_gemm_provider_abi_version",
                ),
                "extern_gemm_runtime_claim": record.get("extern_gemm_runtime_claim", ""),
                "extern_gemm_performance_claim": bool(
                    record.get("extern_gemm_performance_claim", False)
                ),
                "extern_gemm_uses_host_staging": bool(
                    record.get("extern_gemm_uses_host_staging", False)
                ),
                "unsupported_matmul_reason": record.get("unsupported_matmul_reason", ""),
                "example_source": record.get("source", ""),
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            str(item.get("model_case", "")),
            int(item.get("line_no", 0) or 0),
            str(item.get("op_name", "")),
        ),
    )


def _pre_m9_entry_debt_classes(
    extern_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    debt_records = [
        record
        for record in extern_records
        if _is_pre_m9_entry_debt_record(record)
    ]
    return _pre_m9_extern_family_classes(debt_records)


def _is_pre_m9_entry_debt_record(record: dict[str, Any]) -> bool:
    family = str(record.get("op_family", "")) or _extern_family_from_op_name(
        str(record.get("op_name", ""))
    )
    if family == "extern_addmm_bias":
        return not _is_materialized_extern_addmm_bias_record(record)
    if family == "extern_gemm":
        return not (
            _is_materialized_extern_gemm_record(record)
            or _is_runtime_resolved_extern_gemm_record(record)
        )
    return False


def _is_materialized_extern_artifact_record(record: dict[str, Any]) -> bool:
    return _is_materialized_extern_gemm_record(record) or _is_materialized_extern_addmm_bias_record(
        record
    )


def _is_materialized_attention_artifact(record: dict[str, Any]) -> bool:
    return (
        str(record.get("op_family", "")) == "deferred_attention"
        and str(record.get("attention_contract", "")) == ATTENTION_CONTRACT_VIT_FULL
        and str(record.get("attention_semantics_status", "")) == "attention_semantics_accepted"
        and str(record.get("attention_implementation_kind", ""))
        == ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA
        and str(record.get("attention_extern_symbol", "")) == ATTENTION_EXTERN_SYMBOL
        and str(record.get("attention_extern_packed_func", "")) == ATTENTION_EXTERN_PACKED_FUNC
        and str(record.get("attention_runtime_kind", "") or ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY)
        == ATTENTION_RUNTIME_KIND_ARTIFACT_ONLY
        and str(record.get("attention_runtime_replacement", ""))
        == ATTENTION_RUNTIME_REPLACEMENT
        and not bool(record.get("attention_runtime_replacement_available", False))
        and str(
            record.get("attention_runtime_status", "")
            or ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY
        )
        == ATTENTION_RUNTIME_STATUS_ARTIFACT_ONLY
    )


def _is_materialized_vision_conv_artifact(record: dict[str, Any]) -> bool:
    return (
        _is_materialized_vision_conv_record(record)
        and str(record.get("vision_runtime_kind", "")) == VISION_RUNTIME_KIND_ARTIFACT_ONLY
        and str(record.get("vision_runtime_replacement", "")) == VISION_RUNTIME_REPLACEMENT
        and not bool(record.get("vision_runtime_replacement_available", False))
        and str(record.get("vision_runtime_status", ""))
        == VISION_RUNTIME_STATUS_ARTIFACT_ONLY
        and not bool(record.get("vision_performance_claim", False))
    )


def _is_materialized_vision_conv_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("op_family", "")) == "deferred_convolution"
        and str(record.get("vision_source_kind", "")) == "wrapper_extern_convolution"
        and str(record.get("vision_semantics_status", ""))
        == VISION_SEMANTICS_STATUS_ACCEPTED
        and str(record.get("vision_implementation_kind", ""))
        == VISION_IMPLEMENTATION_KIND_EXTERN_CONV2D
        and str(record.get("vision_extern_symbol", "")) == VISION_EXTERN_SYMBOL
        and str(record.get("vision_extern_packed_func", "")) == VISION_EXTERN_PACKED_FUNC
        and not bool(record.get("vision_performance_claim", False))
    )


def _is_runtime_resolved_vision_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("op_family", "")) == "deferred_convolution"
        and str(record.get("vision_runtime_kind", "")) == VISION_RUNTIME_KIND_PROVIDER
        and str(record.get("vision_runtime_status", ""))
        == VISION_RUNTIME_STATUS_RUNTIME_RESOLVED
    )


def _is_runtime_resolved_attention_record(record: dict[str, Any]) -> bool:
    contract = str(record.get("attention_contract", ""))
    native_decomposed = (
        str(record.get("attention_implementation_kind", ""))
        == ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED
        and str(record.get("attention_runtime_kind", "")) == ATTENTION_RUNTIME_KIND_NATIVE_DECOMPOSED
        and str(record.get("attention_provider_kind", ""))
        == ATTENTION_PROVIDER_NATIVE_DECOMPOSED
        and not bool(record.get("attention_performance_claim", False))
        and not bool(record.get("attention_uses_host_staging", False))
    )
    extern_provider = (
        str(record.get("attention_implementation_kind", ""))
        == ATTENTION_IMPLEMENTATION_KIND_EXTERN_SDPA
        and str(record.get("attention_extern_symbol", "")) == ATTENTION_EXTERN_SYMBOL
        and str(record.get("attention_extern_packed_func", "")) == ATTENTION_EXTERN_PACKED_FUNC
        and str(record.get("attention_runtime_kind", "")) == ATTENTION_RUNTIME_KIND_PROVIDER
        and str(record.get("attention_provider_kind", ""))
        == ATTENTION_PROVIDER_PYTHON_TORCH_HOST_STAGED
        and not bool(record.get("attention_performance_claim", False))
        and bool(record.get("attention_uses_host_staging", False))
    )
    return (
        str(record.get("op_family", "")) == "deferred_attention"
        and contract
        in (ATTENTION_CONTRACT_VIT_FULL, ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL)
        and str(record.get("attention_semantics_status", "")) == "attention_semantics_accepted"
        and str(record.get("attention_runtime_status", ""))
        == ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
        and ((contract == ATTENTION_CONTRACT_VIT_FULL and extern_provider) or native_decomposed)
    )


def _is_materialized_extern_gemm_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("op_family", "")) == "extern_gemm"
        and str(record.get("matmul_source_kind", "")) == "wrapper_extern_gemm"
        and bool(record.get("matmul_contract_ok", False))
        and str(record.get("implementation_kind", "")) == "extern_gemm"
        and str(record.get("extern_symbol", "")) == EXTERN_GEMM_SYMBOL
        and str(record.get("extern_packed_func", "")) != EXTERN_ADDMM_BIAS_PACKED_FUNC
        and str(record.get("extern_runtime_kind", "") or "artifact_only") == "artifact_only"
        and not bool(record.get("extern_runtime_replacement_available", False))
        and str(
            record.get("extern_gemm_runtime_status", "")
            or EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
        )
        == EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
    )


def _is_materialized_extern_addmm_bias_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("op_family", "")) == "extern_addmm_bias"
        and str(record.get("matmul_source_kind", "")) == "wrapper_extern_addmm_bias"
        and str(record.get("matmul_epilogue_kind", "")) == "bias_add"
        and bool(record.get("matmul_contract_ok", False))
        and str(record.get("implementation_kind", "")) == "extern_addmm_bias"
        and str(record.get("extern_symbol", "")) == EXTERN_ADDMM_BIAS_SYMBOL
        and str(record.get("extern_packed_func", "")) == EXTERN_ADDMM_BIAS_PACKED_FUNC
        and str(record.get("extern_runtime_kind", "") or "artifact_only") == "artifact_only"
        and not bool(record.get("extern_runtime_replacement_available", False))
        and str(
            record.get("extern_gemm_runtime_status", "")
            or EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
        )
        == EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
    )


def _is_runtime_resolved_extern_gemm_record(record: dict[str, Any]) -> bool:
    return (
        str(record.get("op_family", "")) == "extern_gemm"
        and str(record.get("matmul_source_kind", "")) == "wrapper_extern_gemm"
        and bool(record.get("matmul_contract_ok", False))
        and str(record.get("implementation_kind", "")) == "extern_gemm"
        and str(record.get("extern_symbol", "")) == "extern_kernels.mm"
        and str(record.get("extern_gemm_runtime_status", ""))
        == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
        and str(record.get("extern_gemm_provider_kind", ""))
        == EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    )


def _m96_extern_gemm_runtime_section(
    extern_records: list[dict[str, Any]],
) -> dict[str, Any]:
    gemm_records = [
        record for record in extern_records if str(record.get("op_family", "")) == "extern_gemm"
    ]
    runtime_resolved = [
        record for record in gemm_records if _is_runtime_resolved_extern_gemm_record(record)
    ]
    status_counts = Counter(
        str(record.get("extern_gemm_runtime_status", ""))
        or EXTERN_GEMM_RUNTIME_STATUS_ARTIFACT_ONLY
        for record in gemm_records
    )
    provider_counts = Counter(
        str(record.get("extern_gemm_provider_kind", "")) or EXTERN_GEMM_PROVIDER_NONE
        for record in gemm_records
    )
    return {
        "provider_policy": "opt_in_correctness_only",
        "performance_claim": False,
        "runtime_resolved_count": len(runtime_resolved),
        "extern_gemm_runtime_resolved_count": len(runtime_resolved),
        "status_counts": dict(sorted(status_counts.items())),
        "provider_counts": dict(sorted(provider_counts.items())),
        "runtime_resolved_records": [
            {
                "model_case": record.get("model_case", ""),
                "op_name": record.get("op_name", ""),
                "line_no": int(record.get("line_no", 0) or 0),
                "extern_gemm_provider_kind": record.get("extern_gemm_provider_kind", ""),
                "extern_gemm_provider_abi_version": record.get(
                    "extern_gemm_provider_abi_version",
                ),
                "extern_gemm_runtime_claim": record.get("extern_gemm_runtime_claim", ""),
                "extern_gemm_performance_claim": bool(
                    record.get("extern_gemm_performance_claim", False)
                ),
                "extern_gemm_uses_host_staging": bool(
                    record.get("extern_gemm_uses_host_staging", False)
                ),
            }
            for record in runtime_resolved
        ],
    }


def _pre_m9_extern_family_classes(
    extern_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}

    def get_entry(name: str) -> dict[str, Any]:
        if name not in classes:
            classes[name] = {
                "call_count": 0,
                "op_names": Counter(),
                "models": set(),
                "example_op": "",
                "example_model": "",
                "example_source": "",
            }
        return classes[name]

    for record in extern_records:
        family = str(record.get("op_family", "")) or _extern_family_from_op_name(
            str(record.get("op_name", ""))
        )
        entry = get_entry(family)
        entry["call_count"] += 1
        entry["op_names"][str(record.get("op_name", "unknown"))] += 1
        if record.get("model_case"):
            entry["models"].add(str(record.get("model_case", "")))
        if not entry["example_op"]:
            entry["example_op"] = str(record.get("op_name", ""))
            entry["example_model"] = str(record.get("model_case", ""))
            entry["example_source"] = str(record.get("source", ""))

    normalized = {}
    for name, entry in sorted(classes.items()):
        models = sorted(entry["models"])
        normalized[name] = {
            "call_count": int(entry["call_count"]),
            "op_names": dict(sorted(entry["op_names"].items())),
            "models_impacted": len(models),
            "models": models,
            "example_op": entry["example_op"],
            "example_model": entry["example_model"],
            "example_source": entry["example_source"],
        }
    return normalized


def _pre_m9_entry_from_extern_family(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "op_family": name,
        "call_count": int(entry.get("call_count", 0)),
        "op_names": dict(entry.get("op_names", {})),
        "models_impacted": int(entry.get("models_impacted", 0)),
        "models": list(entry.get("models", [])),
        "example_op": entry.get("example_op", ""),
        "example_model": entry.get("example_model", ""),
        "example_source": entry.get("example_source", ""),
    }


def _pre_m7_reader_snapshot_classes(
    kernel_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped = {
        name: {
            "kernel_count": 0,
            "example_kernel": "",
            "example_ops": [],
        }
        for name in _PRE_M7_READER_CLASSES
    }
    for record in kernel_records:
        name = _pre_m7_reader_class(record)
        entry = grouped[name]
        entry["kernel_count"] += 1
        if not entry["example_kernel"]:
            entry["example_kernel"] = str(record.get("kernel_name", ""))
            entry["example_ops"] = list(record.get("unique_ops", []))[:12]
    return grouped


def _pre_m7_reader_class(record: dict[str, Any]) -> str:
    unique_ops = set(record.get("unique_ops", []))
    blocker_class = str(record.get("blocker_class", ""))
    if bool(record.get("atomic_add_found", False)) or blocker_class in ("atomic", "grid"):
        return "atomic_grid"
    if record.get("grid_type") and record.get("grid_type") != "Grid1D":
        return "atomic_grid"
    if int(record.get("num_reduction", 0) or 0) > 0 or "tt.reduce" in unique_ops:
        return "reduction"
    if "tt.dot" in unique_ops or blocker_class == "matmul_dot":
        return "matmul_dot"
    if _pre_m7_attention_adjacent(record):
        return "attention_adjacent"
    if _pre_m7_broadcast_view_index(record):
        return "broadcast_view_index"
    return "pointwise"


def _pre_m7_attention_adjacent(record: dict[str, Any]) -> bool:
    text = _pre_m7_record_text(record)
    if any(token in text for token in ("attention", "softmax", "causal", "qkv", "rope")):
        return True
    unique_ops = set(record.get("unique_ops", []))
    return bool(unique_ops & {"tt.trans", "tt.expand_dims"})


def _pre_m7_broadcast_view_index(record: dict[str, Any]) -> bool:
    text = _pre_m7_record_text(record)
    if any(token in text for token in ("broadcast", "index", "stride", "view")):
        return True
    indexing = record.get("indexing_summary") or {}
    return int(indexing.get("addptr_count", 0) or 0) > 1


def _pre_m7_record_text(record: dict[str, Any]) -> str:
    status = record.get("translate_status", {})
    parts = [
        str(record.get("blocker_class", "")),
        str(record.get("kernel_name", "")),
        str(status.get("bucket", "")),
        str(status.get("fallback_reason", "")),
        str(status.get("message", "")),
    ]
    return " ".join(parts).lower()


def _pre_m7_entry_blockers(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for blocker in blockers:
        if not _is_pre_m7_entry_blocker(blocker):
            continue
        entries.append(
            {
                "bucket": blocker.get("bucket", ""),
                "fallback_reason": blocker.get("fallback_reason", ""),
                "blocker_class": blocker.get("blocker_class", ""),
                "models_impacted": blocker.get("models_impacted", 0),
                "kernel_count": blocker.get("kernel_count", 0),
                "models": list(blocker.get("models", [])),
                "example_kernel": blocker.get("example_kernel", ""),
                "example_message": blocker.get("example_message", ""),
                "example_ops": list(blocker.get("example_ops", [])),
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            -int(item.get("models_impacted", 0)),
            -int(item.get("kernel_count", 0)),
            str(item.get("blocker_class", "")),
            str(item.get("example_kernel", "")),
        ),
    )


def _is_pre_m7_entry_blocker(blocker: dict[str, Any]) -> bool:
    bucket = str(blocker.get("bucket", ""))
    reason = str(blocker.get("fallback_reason", ""))
    blocker_class = str(blocker.get("blocker_class", ""))
    if bucket == "collection_error":
        return False
    if blocker_class in _PRE_M7_EXCLUDED_M7_BLOCKER_CLASSES:
        return False
    text = " ".join(
        [
            bucket,
            reason,
            blocker_class,
            str(blocker.get("example_kernel", "")),
            str(blocker.get("example_message", "")),
        ]
    )
    text = text.lower()
    example_ops = set(blocker.get("example_ops", []))
    if example_ops & {"tt.trans", "tt.expand_dims"}:
        return False
    if any(token in text for token in ("attention", "softmax", "causal", "qkv", "rope")):
        return False
    return (
        bucket == "unsupported_ttir_op"
        or any(token in text for token in ("pointwise", "broadcast", "index", "stride", "view"))
    )


def _rank_blockers(
    kernel_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

    def get_entry(bucket: str, reason: str, blocker_class: str) -> dict[str, Any]:
        key = (bucket, reason, blocker_class)
        if key not in buckets:
            buckets[key] = {
                "bucket": bucket,
                "fallback_reason": reason,
                "blocker_class": blocker_class,
                "kernel_count": 0,
                "model_error_count": 0,
                "models": set(),
                "pre_m8_families": Counter(),
                "example_kernel": "",
                "example_message": "",
                "example_ops": [],
            }
        return buckets[key]

    for record in kernel_records:
        status = record.get("translate_status", {})
        if status.get("ok", False):
            continue
        bucket = str(status.get("bucket", "unknown"))
        reason = str(status.get("fallback_reason", "")) or bucket
        blocker_class = str(record.get("blocker_class", "")) or reason
        entry = get_entry(bucket, reason, blocker_class)
        entry["kernel_count"] += 1
        entry["models"].add(record.get("model_case", ""))
        pre_m8_family = str(record.get("pre_m8_family", "")) or _pre_m8_family(record)
        entry["pre_m8_families"][pre_m8_family] += 1
        if not entry["example_kernel"]:
            entry["example_kernel"] = record.get("kernel_name", "")
            entry["example_message"] = str(status.get("message", ""))
            entry["example_ops"] = list(record.get("unique_ops", []))[:12]

    known_models = {
        (record.get("model_family", ""), record.get("model_case", "")): record
        for record in model_records
    }
    for error in errors:
        model_case = error.get("model_case") or error.get("case_name", "")
        model_family = error.get("model_family", "")
        if not model_family and ("", model_case) not in known_models:
            model_family = "unknown"
        bucket = str(error.get("bucket", "collection_error"))
        reason = str(error.get("fallback_reason", "")) or bucket
        blocker_class = reason
        entry = get_entry(bucket, reason, blocker_class)
        entry["model_error_count"] += 1
        entry["models"].add(model_case)
        entry["pre_m8_families"]["collection_error"] += 1
        if not entry["example_message"]:
            entry["example_message"] = str(error.get("message", ""))

    ranked = []
    for entry in buckets.values():
        models = sorted(model for model in entry.pop("models") if model)
        families = entry.pop("pre_m8_families")
        entry["pre_m8_families"] = dict(sorted(families.items()))
        entry["models"] = models
        entry["models_impacted"] = len(models)
        ranked.append(entry)
    return sorted(
        ranked,
        key=lambda item: (
            -int(item.get("models_impacted", 0)),
            -int(item.get("kernel_count", 0)),
            str(item.get("bucket", "")),
            str(item.get("blocker_class", "")),
            item.get("models", [""])[0] if item.get("models") else "",
        ),
    )


def _blocker_class(record: dict[str, Any]) -> str:
    status = record.get("translate_status", {})
    if status.get("ok", False):
        return "none"
    unique_ops = set(record.get("unique_ops", []))
    if bool(record.get("atomic_add_found", False)) or any("atomic" in op for op in unique_ops):
        return "atomic"
    if int(record.get("num_reduction", 0) or 0) > 0 or "tt.reduce" in unique_ops:
        return "reduction"
    if "tt.dot" in unique_ops:
        return "matmul_dot"
    if record.get("grid_type") and record.get("grid_type") != "Grid1D":
        return "grid"
    if _pre_m7_attention_adjacent(record):
        return "attention_adjacent"
    reason = str(status.get("fallback_reason", "")) or str(status.get("bucket", "unknown"))
    if reason == "unsupported_inductor_kernel":
        return "unsupported_inductor_kernel"
    message = str(status.get("message", "")).lower()
    if "__nv_erff" in message:
        return "activation_erf"
    if "sitofp" in message or "fptosi" in message:
        return "dtype_cast"
    if any(
        token in message
        for token in (
            "composed indexing",
            "unique extent",
            "flat extent predicate",
            "indexing",
            "broadcast",
            "view",
        )
    ):
        return "broadcast_view_index"
    return reason


def _pre_m8_family(record: dict[str, Any]) -> str:
    status = record.get("translate_status", {})
    if status.get("ok", False):
        return "translated"

    blocker_class = str(record.get("blocker_class", ""))
    text = _pre_m7_record_text(record)
    kernel_name = str(record.get("kernel_name", "")).lower()

    if blocker_class == "grid":
        return "deferred_grid"
    if blocker_class == "atomic":
        return "deferred_atomic"
    if blocker_class == "matmul_dot":
        return "deferred_matmul_dot"
    if blocker_class == "attention_adjacent":
        return "masked_attention_adjacent"
    if "masked tt.load without other" in text and any(
        token in text for token in ("attention", "softmax", "causal")
    ):
        return "masked_attention_adjacent"

    if blocker_class == "reduction":
        if "softmax" in text:
            return "softmax_like"
        if any(token in text for token in ("layer_norm", "layernorm", "native_layer_norm")):
            return "norm_layernorm"
        if any(token in text for token in ("rms", "rsqrt", "embedding_mean_mul_pow_rsqrt")):
            return "norm_rmsnorm"
        if "max_pool" in text or "pool" in kernel_name:
            return "pooling_reduction"
        return "row_reduction"

    if blocker_class in {"unsupported_inductor_kernel", "broadcast_view_index"}:
        return "deferred_pointwise_or_grid"
    reason = str(status.get("fallback_reason", "")) or str(status.get("bucket", "unknown"))
    return reason


def _blocker_counter(report: dict[str, Any]) -> Counter:
    counter: Counter[str] = Counter()
    for blocker in report.get("blockers", []):
        key = "|".join(
            [
                str(blocker.get("bucket", "")),
                str(blocker.get("fallback_reason", "")),
                str(blocker.get("blocker_class", "")),
            ]
        )
        counter[key] += int(blocker.get("kernel_count", 0)) + int(
            blocker.get("model_error_count", 0)
        )
    return counter


def _counter_delta(before: Counter, after: Counter) -> dict[str, int]:
    delta = {}
    for key in sorted(set(before) | set(after)):
        value = int(after.get(key, 0)) - int(before.get(key, 0))
        if value:
            delta[key] = value
    return delta


def _diff_summary(report: dict[str, Any]) -> dict[str, Any]:
    model_summary = report.get("model_summary") or {}
    return {
        "total_models": int(model_summary.get("total_models", 0)),
        "total_kernels": int(report.get("total_kernels", 0)),
        "translated_kernels": int(report.get("translated_kernels", 0)),
        "status_buckets": dict(sorted((report.get("summary") or {}).get("status_buckets", {}).items())),
    }


def _collection_error(
    case: TritonTVMModelAuditCase,
    bucket: str,
    err: Exception,
) -> dict[str, Any]:
    return {
        "model_family": case.model_family,
        "model_case": case.case_name,
        "case_name": case.case_name,
        "kernel_name": "",
        "bucket": bucket,
        "fallback_reason": bucket,
        "error_type": type(err).__name__,
        "message": str(err),
    }


def _dependency_versions() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "triton", "transformers", "ultralytics", "torchvision", "tvm"):
        try:
            module = __import__(name)
        except Exception as err:  # pylint: disable=broad-except
            versions[name] = {"available": False, "error": f"{type(err).__name__}: {err}"}
            continue
        versions[name] = {
            "available": True,
            "version": str(getattr(module, "__version__", "")),
        }
    try:
        import torch  # pylint: disable=import-outside-toplevel

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:  # pylint: disable=broad-except
        versions["cuda_available"] = False
        versions["cuda_device_count"] = 0
    versions["generated_at"] = datetime.now(timezone.utc).isoformat()
    return versions


def _set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _seed_torch(torch, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_vit_tiny(torch):
    from transformers import ViTConfig, ViTModel  # pylint: disable=import-outside-toplevel

    config = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
    )
    return ViTModel(config)


def _make_vit_inputs(torch) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (torch.randn((1, 3, 32, 32), device="cuda"),), {}


def _make_llama_tiny(torch):
    from transformers import LlamaConfig, LlamaModel  # pylint: disable=import-outside-toplevel

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    return LlamaModel(config)


def _make_llama_inputs(torch) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (torch.randint(0, 128, (1, 16), device="cuda"),), {}


def _make_yolo_tiny(torch):
    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    return YOLO("yolov8n.yaml").model


def _make_yolo_inputs(torch) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (torch.randn((1, 3, 64, 64), device="cuda"),), {}


def _model_corpus_footer_lines() -> list[str]:
    return [
        "## M6/M6.5 Boundary",
        "",
        "- This report explains model blockers; it is not a fallback-free success claim.",
        "- External model libraries use random initialization and small fixed shapes.",
        "- Native Inductor capture is audited offline; M5/M5.5 hook tests own live replacement.",
    ]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


if __name__ == "__main__":
    raise SystemExit(main())
