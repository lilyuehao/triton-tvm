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
    ATTENTION_REPORT_FIELDS,
    ATTENTION_RUNTIME_STATUS_DEFERRED,
    ATTENTION_TARGET_CONTRACTS,
    classify_wrapper_sdpa_attention,
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


MODEL_CORPUS = "m6_model_corpus"
PRE_M7_TAXONOMY_VERSION = 1
PRE_M7_BUILDER_DECISION = "keep_tvmscript_source_builder_for_m7_entry"
PRE_M8_TAXONOMY_VERSION = 1
PRE_M9_TAXONOMY_VERSION = 1
PRE_M10_TAXONOMY_VERSION = 1
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
            _extern_call_to_record(call, case)
            for call in extract_inductor_wrapper_extern_calls(
                wrapper_source,
                case_name=case.case_name,
                wrapper_path=str(wrapper_path),
                extern_gemm_runtime_provider=cfg.extern_gemm_runtime_provider,
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
    return normalized


def _extern_call_to_record(call, case: TritonTVMModelAuditCase) -> dict[str, Any]:
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
    if record["op_family"] == "deferred_attention":
        record.update(
            classify_wrapper_sdpa_attention(
                op_name=record["op_name"],
                op_family=record["op_family"],
                source=record["source"],
                model_family=case.model_family,
                model_case=case.case_name,
                case_name=case.case_name,
            ).as_report_fields()
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
    if normalized.get("op_family") == "deferred_attention":
        classification = classify_wrapper_sdpa_attention(
            op_name=str(normalized.get("op_name", "")),
            op_family=str(normalized.get("op_family", "")),
            source=str(normalized.get("source", "")),
            model_family=str(normalized.get("model_family", "")),
            model_case=str(normalized.get("model_case", "")),
            case_name=str(normalized.get("case_name", "")),
        ).as_report_fields()
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
    if field_name == "attention_abi_version":
        return 0
    if field_name == "attention_causal":
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
