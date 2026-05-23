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
    records = [_normalize_record(record) for record in kernel_records]
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
                f"- Fallback kernels: {model_summary.get('fallback_kernels', 0)}",
                f"- Model status: {_format_histogram(model_summary.get('model_status', {}))}",
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
    normalized.setdefault("contract", "")
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
