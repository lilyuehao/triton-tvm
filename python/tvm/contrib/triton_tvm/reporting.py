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
"""Unified capability reporting for Triton TVM audits and fallback decisions."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import (
    TritonTVMContractError,
    TritonTVMError,
    UnsupportedContractError,
    UnsupportedStreamError,
    UnsupportedTTIROpError,
    UnsupportedTargetPolicyError,
)


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "triton_tvm_capability_report"
_LEGACY_CONTRACT_ALIASES = {
    "cuda_minimal": "pointwise_minimal",
    "cuda_pointwise_flat": "pointwise_flat",
}


def make_report_status(
    *,
    ok: bool,
    bucket: str,
    error_type: str = "",
    message: str = "",
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Create a stable status payload for JSON and Markdown reports."""
    status = {
        "ok": bool(ok),
        "bucket": bucket,
        "fallback_reason": "" if ok else (fallback_reason or bucket),
    }
    if error_type:
        status["error_type"] = error_type
    if message:
        status["message"] = message
    return status


def status_from_exception(err: Exception) -> dict[str, Any]:
    """Map public unsupported/error classes to stable report buckets."""
    return make_report_status(
        ok=False,
        bucket=bucket_for_exception(err),
        error_type=type(err).__name__,
        message=str(err),
    )


def bucket_for_exception(err: Exception) -> str:
    """Return the stable fallback/report bucket for an exception."""
    if isinstance(err, UnsupportedTTIROpError):
        return "unsupported_ttir_op"
    if isinstance(err, (TritonTVMContractError, UnsupportedContractError)):
        return "contract_error"
    if isinstance(err, UnsupportedTargetPolicyError):
        return "target_policy_error"
    if isinstance(err, UnsupportedStreamError):
        return "unsupported_stream"
    if isinstance(err, (TypeError, ValueError)):
        return "input_error"
    if isinstance(err, TritonTVMError):
        return "triton_tvm_error"
    return "internal_error"


def build_capability_report(
    kernel_records: list[dict[str, Any]],
    *,
    purpose: str,
    corpus: str = "",
    generated_at: str | None = None,
    filtered: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the shared JSON report used by pointwise, reduction, and norm audits."""
    records = []
    diagnostics = {
        "legacy_contract_records": [],
        "silent_fallback_records": [],
    }
    for record in kernel_records:
        normalized = _normalize_record(record)
        if _is_legacy_contract(record.get("contract", "")):
            diagnostics["legacy_contract_records"].append(_record_ref(normalized))
        if _is_raw_silent_fallback(record):
            diagnostics["silent_fallback_records"].append(_record_ref(normalized))
        records.append(normalized)
    collection_errors = [_normalize_collection_error(error) for error in (errors or [])]
    op_histogram: Counter[str] = Counter()
    type_histogram: Counter[str] = Counter()
    bucket_histogram: Counter[str] = Counter()
    contract_histogram: Counter[str] = Counter()
    corpus_histogram: Counter[str] = Counter()

    for record in records:
        op_histogram.update(record.get("op_counts", {}))
        type_histogram.update(record.get("types", []))
        status = record["translate_status"]
        bucket_histogram[status.get("bucket", "unknown")] += 1
        contract = record.get("contract", "")
        if contract:
            contract_histogram[contract] += 1
        record_corpus = record.get("corpus", "")
        if record_corpus:
            corpus_histogram[record_corpus] += 1

    summary = {
        "total_kernels": len(records),
        "translated_kernels": sum(1 for record in records if record["translate_status"]["ok"]),
        "status_buckets": dict(sorted(bucket_histogram.items())),
        "contracts": dict(sorted(contract_histogram.items())),
        "corpora": dict(sorted(corpus_histogram.items())),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "purpose": purpose,
        "corpus": corpus,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "op_histogram": dict(sorted(op_histogram.items())),
        "type_histogram": dict(sorted(type_histogram.items())),
        "unsupported_buckets": summary["status_buckets"],
        "supported_kernel_report": _build_supported_kernel_report(records, diagnostics),
        "filtered_kernels": list(filtered or []),
        "collection_errors": collection_errors,
        "kernels": records,
    }
    report["total_kernels"] = summary["total_kernels"]
    report["translated_kernels"] = summary["translated_kernels"]
    return report


def render_capability_markdown(
    report: dict[str, Any],
    *,
    title: str = "Triton TVM Capability Report",
    footer_lines: list[str] | None = None,
) -> str:
    """Render the shared capability report as Markdown."""
    summary = report["summary"]
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Total kernels: {summary['total_kernels']}",
        f"- Translated kernels: {summary['translated_kernels']}",
        f"- Status buckets: {_format_histogram(summary['status_buckets'])}",
        f"- Contracts: {_format_histogram(summary['contracts'])}",
        "",
    ]
    if "graph_summary" in report:
        graph_summary = report.get("graph_summary") or {}
        lines.extend(
            [
                "## Graph Summary",
                "",
                f"- Total graphs: {graph_summary.get('total_graphs', 0)}",
                f"- Completed graphs: {graph_summary.get('completed_graphs', 0)}",
                f"- Failed graphs: {graph_summary.get('failed_graphs', 0)}",
                f"- Total kernels: {graph_summary.get('total_kernels', 0)}",
                f"- Translated kernels: {graph_summary.get('translated_kernels', 0)}",
                f"- Native fallback kernels: {graph_summary.get('native_fallback_kernels', 0)}",
                f"- Cache hits: {graph_summary.get('cache_hits', 0)}",
                f"- Cache misses: {graph_summary.get('cache_misses', 0)}",
                f"- TVM run count: {graph_summary.get('run_count', 0)}",
                "",
                "Markdown graph details are a summary; JSON `graphs` and `kernels` "
                "are authoritative for `kernel_record_indices` and reverse lookup.",
            ]
        )
        graphs = report.get("graphs") or []
        if graphs:
            lines.extend(
                [
                    "",
                    "| Graph | Status | Kernels | TVM | Native | Cache hit/miss | TVM runs |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for graph in graphs:
                lines.append(
                    "| {graph_id} | {status} | {total} | {translated} | {native} | "
                    "{cache_hits}/{cache_misses} | {runs} |".format(
                        graph_id=_escape_markdown_cell(str(graph.get("graph_id", ""))),
                        status=_escape_markdown_cell(str(graph.get("status", ""))),
                        total=graph.get("total_kernels", 0),
                        translated=graph.get("translated_kernels", 0),
                        native=graph.get("native_fallback_kernels", 0),
                        cache_hits=graph.get("cache_hits", 0),
                        cache_misses=graph.get("cache_misses", 0),
                        runs=graph.get("run_count", 0),
                    )
                )
        lines.append("")
    supported_kernel_report = report.get("supported_kernel_report") or {}
    if supported_kernel_report:
        lines.extend(
            [
                "## Supported Kernel Report",
                "",
                f"- Supported kernels: {supported_kernel_report.get('kernel_count', 0)}",
                f"- Contracts: {_format_histogram(supported_kernel_report.get('contracts', {}))}",
                f"- DTypes: {_format_histogram(supported_kernel_report.get('dtypes', {}))}",
                f"- Tensor shapes: {_format_histogram(supported_kernel_report.get('tensor_shapes', {}))}",
                f"- Layout/index kinds: {_format_histogram(supported_kernel_report.get('layout_index_kinds', {}))}",
                f"- Legacy contract records: {len(supported_kernel_report.get('legacy_contract_records', []))}",
                f"- Silent fallback records: {len(supported_kernel_report.get('silent_fallback_records', []))}",
                "",
            ]
        )
    if "model_summary" in report:
        model_summary = report.get("model_summary") or {}
        lines.extend(
            [
                "## Model Summary",
                "",
                f"- Total models: {model_summary.get('total_models', 0)}",
                f"- Completed models: {model_summary.get('completed_models', 0)}",
                f"- Failed models: {model_summary.get('failed_models', 0)}",
                f"- Full TVM runnable models: {model_summary.get('full_tvm_runnable_models', 0)}",
                f"- Triton-kernel runnable models: {model_summary.get('triton_kernel_runnable_models', 0)}",
                f"- Fallback kernels: {model_summary.get('fallback_kernels', 0)}",
                f"- Wrapper extern calls: {model_summary.get('extern_op_count', 0)}",
                f"- Model status: {_format_histogram(model_summary.get('model_status', {}))}",
                f"- Extern op families: {_format_histogram(model_summary.get('extern_op_families', {}))}",
                f"- Blocker classes: {_format_histogram(model_summary.get('blocker_classes', {}))}",
                "",
            ]
        )
        models = report.get("models") or []
        if models:
            lines.extend(
                [
                    "| Family | Model | Status | Kernels | TVM | Fallback | Full TVM |",
                    "|---|---|---|---:|---:|---:|---|",
                ]
            )
            for model in models:
                lines.append(
                    "| {family} | {case} | {status} | {kernels} | {translated} | "
                    "{fallback} | {full_tvm} |".format(
                        family=_escape_markdown_cell(str(model.get("model_family", ""))),
                        case=_escape_markdown_cell(str(model.get("model_case", ""))),
                        status=_escape_markdown_cell(str(model.get("status", ""))),
                        kernels=model.get("kernel_count", 0),
                        translated=model.get("translated_kernels", 0),
                        fallback=model.get("native_fallback_kernels", 0),
                        full_tvm="yes" if model.get("full_tvm_runnable") else "no",
                    )
                )
            lines.append("")
        blockers = report.get("blockers") or []
        if blockers:
            lines.extend(
                [
                    "## Blockers",
                    "",
                    "| Bucket | Reason | Class | Models | Kernels | Model Errors | Example |",
                    "|---|---|---|---:|---:|---:|---|",
                ]
            )
            for blocker in blockers:
                lines.append(
                    "| {bucket} | {reason} | {cls} | {models} | {kernels} | "
                    "{model_errors} | {example} |".format(
                        bucket=_escape_markdown_cell(str(blocker.get("bucket", ""))),
                        reason=_escape_markdown_cell(str(blocker.get("fallback_reason", ""))),
                        cls=_escape_markdown_cell(str(blocker.get("blocker_class", ""))),
                        models=blocker.get("models_impacted", 0),
                        kernels=blocker.get("kernel_count", 0),
                        model_errors=blocker.get("model_error_count", 0),
                        example=_escape_markdown_cell(
                            str(
                                blocker.get("example_kernel", "")
                                or blocker.get("example_message", "")
                            )[:160]
                        ),
                    )
                )
            lines.append("")
        dependency_versions = report.get("dependency_versions") or {}
        if dependency_versions:
            lines.extend(["## Dependency Versions", ""])
            for name, value in sorted(dependency_versions.items()):
                if isinstance(value, dict):
                    version = value.get("version", "")
                    available = value.get("available", "")
                    error = value.get("error", "")
                    text = f"available={available}"
                    if version:
                        text += f", version={version}"
                    if error:
                        text += f", error={error}"
                else:
                    text = str(value)
                lines.append(f"- `{name}`: {_escape_markdown_cell(text)}")
            lines.append("")
    if "pre_m7" in report:
        pre_m7 = report.get("pre_m7") or {}
        lines.extend(
            [
                "## Pre-M7 Gate",
                "",
                f"- Taxonomy version: {pre_m7.get('taxonomy_version', '')}",
                f"- Builder decision: `{pre_m7.get('builder_decision', '')}`",
                "",
            ]
        )
        entry_blockers = pre_m7.get("m7_entry_blockers") or []
        if entry_blockers:
            lines.extend(
                [
                    "| Bucket | Reason | Class | Models | Kernels | Example |",
                    "|---|---|---|---:|---:|---|",
                ]
            )
            for blocker in entry_blockers:
                lines.append(
                    "| {bucket} | {reason} | {cls} | {models} | {kernels} | {example} |".format(
                        bucket=_escape_markdown_cell(str(blocker.get("bucket", ""))),
                        reason=_escape_markdown_cell(str(blocker.get("fallback_reason", ""))),
                        cls=_escape_markdown_cell(str(blocker.get("blocker_class", ""))),
                        models=blocker.get("models_impacted", 0),
                        kernels=blocker.get("kernel_count", 0),
                        example=_escape_markdown_cell(
                            str(
                                blocker.get("example_kernel", "")
                                or blocker.get("example_message", "")
                            )[:160]
                        ),
                    )
            )
            lines.append("")
    if "pre_m8" in report:
        pre_m8 = report.get("pre_m8") or {}
        taxonomy = pre_m8.get("unsupported_taxonomy") or {}
        lines.extend(
            [
                "## Pre-M8 Gate",
                "",
                f"- Taxonomy version: {pre_m8.get('taxonomy_version', '')}",
                f"- Detail fields: {', '.join(taxonomy.get('detail_fields', []))}",
                f"- M8 entry debt families: {len(pre_m8.get('m8_entry_debt') or [])}",
                f"- Deferred debt families: {len(pre_m8.get('deferred_debt') or [])}",
                "",
            ]
        )
        family_classes = pre_m8.get("family_classes") or {}
        if family_classes:
            lines.extend(
                [
                    "| Family | Kernels | Buckets | Blocker classes | Example |",
                    "|---|---:|---|---|---|",
                ]
            )
            for family, entry in sorted(family_classes.items()):
                lines.append(
                    "| {family} | {kernels} | {buckets} | {classes} | {example} |".format(
                        family=_escape_markdown_cell(str(family)),
                        kernels=entry.get("kernel_count", 0),
                        buckets=_escape_markdown_cell(
                            _format_histogram(entry.get("buckets", {}))
                        ),
                        classes=_escape_markdown_cell(
                            _format_histogram(entry.get("blocker_classes", {}))
                        ),
                        example=_escape_markdown_cell(
                            str(entry.get("example_kernel", ""))[:160]
                        ),
                    )
                )
            lines.append("")
    if "pre_m9" in report:
        pre_m9 = report.get("pre_m9") or {}
        lines.extend(
            [
                "## Pre-M9 Gate",
                "",
                f"- Taxonomy version: {pre_m9.get('taxonomy_version', '')}",
                f"- Detail fields: {', '.join(pre_m9.get('detail_fields', []))}",
                "- Materialized artifact candidates: "
                f"{len(pre_m9.get('m9_materialized_artifact_candidates') or [])}",
                "- M9.6 extern GEMM runtime resolved: "
                f"{(pre_m9.get('m96_extern_gemm_runtime') or {}).get('runtime_resolved_count', 0)}",
                f"- M9 entry debt families: {len(pre_m9.get('m9_entry_debt') or [])}",
                f"- Deferred debt families: {len(pre_m9.get('deferred_debt') or [])}",
                f"- Observed TTIR dot kernels: {pre_m9.get('observed_ttir_dot_kernels', 0)}",
                f"- Observed grid fallback kernels: {pre_m9.get('observed_grid_fallback_kernels', 0)}",
                "",
            ]
        )
        extern_family_classes = pre_m9.get("extern_family_classes") or {}
        if extern_family_classes:
            lines.extend(
                [
                    "| Family | Calls | Models | Ops | Example model | Example op |",
                    "|---|---:|---:|---|---|---|",
                ]
            )
            for family, entry in sorted(extern_family_classes.items()):
                lines.append(
                    "| {family} | {calls} | {models} | {ops} | {example_model} | {example_op} |".format(
                        family=_escape_markdown_cell(str(family)),
                        calls=entry.get("call_count", 0),
                        models=entry.get("models_impacted", 0),
                        ops=_escape_markdown_cell(
                            _format_histogram(entry.get("op_names", {}))
                        ),
                        example_model=_escape_markdown_cell(
                            str(entry.get("example_model", ""))
                        ),
                        example_op=_escape_markdown_cell(str(entry.get("example_op", ""))),
                    )
                )
            lines.append("")
    if "pre_m10" in report:
        pre_m10 = report.get("pre_m10") or {}
        lines.extend(
            [
                "## Pre-M10 Gate",
                "",
                f"- Taxonomy version: {pre_m10.get('taxonomy_version', '')}",
                f"- Target contracts: {', '.join(pre_m10.get('target_contracts', []))}",
                f"- Detail fields: {', '.join(pre_m10.get('detail_fields', []))}",
                f"- Observed attention calls: {pre_m10.get('observed_attention_call_count', 0)}",
                "- Deferred attention runtime calls: "
                f"{pre_m10.get('attention_runtime_deferred_count', 0)}",
                "",
            ]
        )
        contract_classes = pre_m10.get("contract_classes") or {}
        if contract_classes:
            lines.extend(
                [
                    "| Contract | Calls | Models | Runtime | Phase | Mask | Example model |",
                    "|---|---:|---:|---|---|---|---|",
                ]
            )
            for contract, entry in sorted(contract_classes.items()):
                lines.append(
                    "| {contract} | {calls} | {models} | {runtime} | {phase} | "
                    "{mask} | {example_model} |".format(
                        contract=_escape_markdown_cell(str(contract)),
                        calls=entry.get("call_count", 0),
                        models=entry.get("models_impacted", 0),
                        runtime=_escape_markdown_cell(
                            _format_histogram(entry.get("runtime_status", {}))
                        ),
                        phase=_escape_markdown_cell(
                            _format_histogram(entry.get("phases", {}))
                        ),
                        mask=_escape_markdown_cell(
                            _format_histogram(entry.get("mask_kinds", {}))
                        ),
                        example_model=_escape_markdown_cell(
                            str(entry.get("example_model", ""))
                        ),
                    )
                )
            lines.append("")
    lines.extend(
        [
            "## TTIR Op Coverage",
            "",
            "| Op | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{op}` | {count} |" for op, count in report["op_histogram"].items())
    lines.extend(
        [
            "",
            "## Type Coverage",
            "",
            "| Type | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{ty}` | {count} |" for ty, count in report["type_histogram"].items())
    lines.extend(
        [
            "",
            "## Kernel Status",
            "",
            "| Corpus | Case | Kernel | Contract | Loads | Stores | Status | Fallback | Message |",
            "|---|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for record in report["kernels"]:
        status = record["translate_status"]
        lines.append(
            "| {corpus} | {case} | `{kernel}` | `{contract}` | {loads} | {stores} | "
            "{bucket} | {fallback} | {message} |".format(
                corpus=_escape_markdown_cell(record.get("corpus", "")),
                case=_escape_markdown_cell(record.get("case_name", "")),
                kernel=record.get("kernel_name", ""),
                contract=record.get("contract", ""),
                loads=record.get("load_count", 0),
                stores=record.get("store_count", 0),
                bucket=status.get("bucket", ""),
                fallback=status.get("fallback_reason", ""),
                message=_escape_markdown_cell(status.get("message", "")[:160]),
            )
        )
    if footer_lines:
        lines.extend(["", *footer_lines, ""])
    return "\n".join(lines)


def write_capability_report(
    report: dict[str, Any],
    out_dir: str | Path,
    *,
    markdown_title: str = "Triton TVM Capability Report",
    footer_lines: list[str] | None = None,
) -> None:
    """Write shared ``report.json`` and ``report.md`` files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "report.md").write_text(
        render_capability_markdown(
            report,
            title=markdown_title,
            footer_lines=footer_lines,
        ),
        encoding="utf-8",
    )


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    status = normalized.get("translate_status") or normalized.get("status")
    if status is None:
        status = make_report_status(ok=False, bucket="not_run")
    else:
        status = _normalize_status(status)
    normalized["translate_status"] = status
    normalized.setdefault("case_name", "")
    normalized.setdefault("kernel_name", "")
    normalized["contract"] = _canonical_report_contract(str(normalized.get("contract") or ""))
    normalized.setdefault("corpus", "")
    normalized.setdefault("op_counts", {})
    normalized.setdefault("types", [])
    normalized.setdefault("load_count", 0)
    normalized.setdefault("store_count", 0)
    return normalized


def _normalize_status(status: dict[str, Any]) -> dict[str, Any]:
    ok = bool(status.get("ok", False))
    return make_report_status(
        ok=ok,
        bucket=str(status.get("bucket", "translated" if ok else "unknown")),
        error_type=str(status.get("error_type", "")),
        message=str(status.get("message", "")),
        fallback_reason=str(status.get("fallback_reason", "")) or None,
    )


def _normalize_collection_error(error: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(error)
    bucket = str(normalized.get("bucket", "collection_error"))
    normalized.setdefault("fallback_reason", bucket)
    return normalized


def _build_supported_kernel_report(
    records: list[dict[str, Any]],
    diagnostics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    supported = [
        record for record in records if record.get("translate_status", {}).get("ok", False)
    ]
    contracts: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    tensor_shapes: Counter[str] = Counter()
    layout_index_kinds: Counter[str] = Counter()

    for record in supported:
        contract = str(record.get("contract", ""))
        if contract:
            contracts[contract] += 1
        for raw_type in record.get("types", []):
            dtype, shape = _dtype_shape_from_ttir_type(str(raw_type))
            if dtype:
                dtypes[dtype] += 1
            if shape:
                tensor_shapes[shape] += 1
        for dtype in _signature_dtypes(record.get("signature", {})):
            dtypes[dtype] += 1
        indexing = record.get("indexing_summary") or {}
        index_kinds = indexing.get("index_kinds") or {}
        if index_kinds:
            layout_index_kinds.update(index_kinds)
        else:
            kind = str(indexing.get("kind", ""))
            if kind:
                layout_index_kinds[kind] += 1

    return {
        "schema_version": 1,
        "purpose": "m7_supported_kernel_guard",
        "kernel_count": len(supported),
        "contracts": dict(sorted(contracts.items())),
        "dtypes": dict(sorted(dtypes.items())),
        "tensor_shapes": dict(sorted(tensor_shapes.items())),
        "layout_index_kinds": dict(sorted(layout_index_kinds.items())),
        "legacy_contract_records": list(diagnostics.get("legacy_contract_records", [])),
        "silent_fallback_records": list(diagnostics.get("silent_fallback_records", [])),
        "ok": not diagnostics.get("legacy_contract_records")
        and not diagnostics.get("silent_fallback_records"),
    }


def _record_ref(record: dict[str, Any]) -> dict[str, Any]:
    status = record.get("translate_status", {})
    return {
        "corpus": record.get("corpus", ""),
        "case_name": record.get("case_name", ""),
        "kernel_name": record.get("kernel_name", ""),
        "contract": record.get("contract", ""),
        "bucket": status.get("bucket", ""),
        "fallback_reason": status.get("fallback_reason", ""),
    }


def _is_legacy_contract(contract: Any) -> bool:
    return str(contract) in _LEGACY_CONTRACT_ALIASES


def _canonical_report_contract(contract: str) -> str:
    return _LEGACY_CONTRACT_ALIASES.get(contract, contract)


def _is_raw_silent_fallback(record: dict[str, Any]) -> bool:
    status = record.get("translate_status") or record.get("status")
    native_fallback_count = int(record.get("native_fallback_count", 0) or 0)
    if not isinstance(status, dict):
        return bool(native_fallback_count)
    fallback_reason = str(status.get("fallback_reason", ""))
    if native_fallback_count and not fallback_reason:
        return True
    return not bool(status.get("ok", False)) and not fallback_reason


def _dtype_shape_from_ttir_type(raw_type: str) -> tuple[str, str]:
    if raw_type.startswith("tensor<") and raw_type.endswith(">"):
        inner = raw_type[len("tensor<") : -1]
        parts = inner.split("x")
        if len(parts) >= 2:
            return _canonical_report_dtype(parts[-1]), "x".join(parts[:-1])
        return _canonical_report_dtype(inner), ""
    if raw_type.startswith("!tt.ptr<") and raw_type.endswith(">"):
        return _canonical_report_dtype(raw_type[len("!tt.ptr<") : -1]), ""
    return _canonical_report_dtype(raw_type), ""


def _signature_dtypes(signature: dict[str, Any]) -> list[str]:
    dtypes = []
    for raw_type in signature.values():
        text = str(raw_type)
        if text == "constexpr":
            continue
        if text.startswith("*"):
            text = text[1:]
        dtype = _canonical_report_dtype(text)
        if dtype:
            dtypes.append(dtype)
    return dtypes


def _canonical_report_dtype(dtype: str) -> str:
    if dtype.startswith("!tt.ptr<") and dtype.endswith(">"):
        return _canonical_report_dtype(dtype[len("!tt.ptr<") : -1])
    mapping = {
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "ui8": "uint8",
        "ui16": "uint16",
        "ui32": "uint32",
        "ui64": "uint64",
    }
    return mapping.get(dtype, dtype)


def _format_histogram(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _escape_markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_jsonable(v) for v in value]
        return repr(value)
