# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Structured report helpers for active Triton-TVM evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import re
from typing import Any, Iterable

from ..atomic import AtomicDAGRecord
from ..manifest import ManifestGapRecord, OperatorManifestRecord
from ..semantic import SemanticRegionRecord
from ..source import SourceOrigin, SourceRecord, is_production_origin


def build_alpha_smoke_report(
    *,
    sources: Iterable[SourceRecord],
    atomics: Iterable[AtomicDAGRecord],
    semantics: Iterable[SemanticRegionRecord],
    manifests: Iterable[OperatorManifestRecord],
    gaps: Iterable[ManifestGapRecord] = (),
) -> dict[str, Any]:
    """Build a structured alpha report without making runtime decisions."""

    source_list = tuple(sources)
    test_inputs = [source.source_id for source in source_list if source.source_origin == SourceOrigin.TEST_FIXTURE]
    production_sources = [source for source in source_list if is_production_origin(source.source_origin)]
    manifest_list = tuple(manifests)
    return {
        "report_kind": "triton_tvm_alpha_path",
        "status": "ok" if not test_inputs else "non_production_input_rejected",
        "performance_claim": False,
        "source_summary": {
            "total": len(source_list),
            "production": len(production_sources),
            "test_input_rejections": test_inputs,
            "origins": sorted({source.source_origin.value for source in source_list}),
            "artifact_kinds": sorted({source.artifact_kind.value for source in source_list}),
        },
        "atomic_summary": {
            "total": len(tuple(atomics)),
            "join_consistency_required": True,
        },
        "semantic_summary": {
            "total": len(tuple(semantics)),
            "matching_key": "semantic_region_key",
        },
        "manifest_summary": {
            "total": len(manifest_list),
            "identity": "model_id+operator_id",
            "supported_records": sum(1 for record in manifest_list if record.gap_reason is None),
        },
        "gaps": [asdict(gap) for gap in gaps],
    }


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write a report in stable JSON format."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def active_inventory(
    *,
    package_dir: str | Path,
    test_dir: str | Path,
) -> dict[str, Any]:
    """Inventory active package modules and test filenames."""

    package_root = Path(package_dir)
    tests_root = Path(test_dir)
    milestone_module = re.compile(r"^m(?:9|10|11|12|13|14)(?:[0-9a-z_].*)?\.py$")
    milestone_test = re.compile(r"^test_triton_tvm_m(?:9|10|11|12|13|14)(?:[0-9a-z_].*)?\.py$")
    modules = sorted(path.relative_to(package_root).as_posix() for path in package_root.rglob("*.py"))
    tests = sorted(path.name for path in tests_root.glob("test_triton_tvm*.py"))
    return {
        "active_package_modules": modules,
        "active_test_files": tests,
        "milestone_named_active_modules": [name for name in modules if milestone_module.match(Path(name).name)],
        "milestone_named_active_tests": [name for name in tests if milestone_test.match(name)],
        "performance_claim": False,
    }

