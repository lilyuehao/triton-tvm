# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Atomic DAG extraction and validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Iterable

from ..source import SourceRecord, ttir_hash_from_text
from ..source.identity import stable_hash
from .schema import (
    ATOMIC_FAMILY_ARITH,
    ATOMIC_FAMILY_CAST_DTYPE,
    ATOMIC_FAMILY_COMPARE_SELECT,
    ATOMIC_FAMILY_DOT,
    ATOMIC_FAMILY_LAUNCH_INDEX,
    ATOMIC_FAMILY_MATH,
    ATOMIC_FAMILY_MEMORY,
    ATOMIC_FAMILY_POINTER_INDEX,
    ATOMIC_FAMILY_REDUCTION,
    ATOMIC_FAMILY_SHAPE_VIEW,
    ATOMIC_FAMILY_UNSUPPORTED,
    AtomicDAGRecord,
    AtomicOpRecord,
)
from .ttir import NormalizedTTIRGraph, TTIROp, parse_ttir


_DIRECT_FAMILY_BY_OP = {
    "tt.get_program_id": ATOMIC_FAMILY_LAUNCH_INDEX,
    "tt.make_range": ATOMIC_FAMILY_SHAPE_VIEW,
    "arith.constant": ATOMIC_FAMILY_SHAPE_VIEW,
    "arith.addi": ATOMIC_FAMILY_POINTER_INDEX,
    "arith.muli": ATOMIC_FAMILY_POINTER_INDEX,
    "arith.subi": ATOMIC_FAMILY_POINTER_INDEX,
    "arith.addf": ATOMIC_FAMILY_ARITH,
    "arith.subf": ATOMIC_FAMILY_ARITH,
    "arith.mulf": ATOMIC_FAMILY_ARITH,
    "arith.divf": ATOMIC_FAMILY_ARITH,
    "arith.cmpi": ATOMIC_FAMILY_COMPARE_SELECT,
    "arith.cmpf": ATOMIC_FAMILY_COMPARE_SELECT,
    "arith.select": ATOMIC_FAMILY_COMPARE_SELECT,
    "tt.splat": ATOMIC_FAMILY_SHAPE_VIEW,
    "tt.expand_dims": ATOMIC_FAMILY_SHAPE_VIEW,
    "tt.broadcast": ATOMIC_FAMILY_SHAPE_VIEW,
    "tt.reshape": ATOMIC_FAMILY_SHAPE_VIEW,
    "tt.view": ATOMIC_FAMILY_SHAPE_VIEW,
    "tt.load": ATOMIC_FAMILY_MEMORY,
    "tt.store": ATOMIC_FAMILY_MEMORY,
    "tt.bitcast": ATOMIC_FAMILY_CAST_DTYPE,
    "tt.reduce": ATOMIC_FAMILY_REDUCTION,
    "tt.reduce.return": ATOMIC_FAMILY_REDUCTION,
    "tt.dot": ATOMIC_FAMILY_DOT,
    "tt.extern_elementwise": ATOMIC_FAMILY_MATH,
}


def build_atomic_dag(source: SourceRecord, ttir: str | NormalizedTTIRGraph) -> AtomicDAGRecord:
    """Build a stable AtomicDAGRecord from source identity and TTIR."""

    graph = parse_ttir(ttir) if isinstance(ttir, str) else ttir
    records = _atomic_records_from_graph(graph)
    records = _with_producers_and_consumers(records)
    unsupported = tuple(record for record in records if record.unsupported_reason)
    histogram = dict(sorted(Counter(record.atomic_family for record in records).items()))
    module_hash = ttir_hash_from_text(graph.raw_ttir)
    atomic_hash = _atomic_dag_hash(records, graph.function_name)
    return AtomicDAGRecord(
        atomic_dag_hash=atomic_hash,
        source_id=source.source_id,
        source_origin=source.source_origin,
        artifact_kind=source.artifact_kind,
        function_name=graph.function_name,
        target=source.target,
        source_hash=source.source_hash,
        ttir_hash=source.ttir_hash or module_hash,
        atomic_ops=tuple(records),
        unsupported_atomic_ops=unsupported,
        atomic_family_histogram=histogram,
        producer_consumer_consistent=validate_atomic_dag_consistency(records),
    )


def validate_source_atomic_join(source: SourceRecord, atomic: AtomicDAGRecord) -> None:
    """Hard-fail when duplicated proof fields diverge."""

    checks = {
        "source_id": source.source_id == atomic.source_id,
        "source_origin": source.source_origin == atomic.source_origin,
        "artifact_kind": source.artifact_kind == atomic.artifact_kind,
        "source_hash": source.source_hash == atomic.source_hash,
        "ttir_hash": (source.ttir_hash is None or source.ttir_hash == atomic.ttir_hash),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"SourceRecord and AtomicDAGRecord mismatch: {', '.join(failed)}")


def validate_atomic_dag_consistency(records: Iterable[AtomicOpRecord]) -> bool:
    """Validate producer and consumer references are reciprocal."""

    by_id = {record.atomic_id: record for record in records}
    for record in by_id.values():
        for producer_id in record.producer_ids:
            if producer_id not in by_id or record.atomic_id not in by_id[producer_id].consumer_ids:
                return False
        for consumer_id in record.consumer_ids:
            if consumer_id not in by_id or record.atomic_id not in by_id[consumer_id].producer_ids:
                return False
    return True


def _atomic_records_from_graph(graph: NormalizedTTIRGraph) -> list[AtomicOpRecord]:
    records: list[AtomicOpRecord] = []
    for index, op in enumerate(graph.ops):
        if op.name in {"tt.return", "func.return"}:
            continue
        family, reason = _classify_ttir_op(op)
        result_type = op.result_types[0] if op.result_types else None
        records.append(
            AtomicOpRecord(
                atomic_id=f"a{len(records):04d}",
                atomic_family=family,
                ttir_op_name=op.name,
                operands=tuple(op.operands),
                results=tuple(op.results),
                dtype=result_type.dtype if result_type is not None else None,
                shape=result_type.shape if result_type is not None and result_type.shape else None,
                attrs=dict(op.attrs),
                source_location=f"{graph.function_name}:op[{index}]",
                producer_ids=(),
                consumer_ids=(),
                local_scope_id=None,
                unsupported_reason=reason,
            )
        )
    return records


def _with_producers_and_consumers(records: list[AtomicOpRecord]) -> tuple[AtomicOpRecord, ...]:
    result_to_id = {result: record.atomic_id for record in records for result in record.results}
    consumers: dict[str, list[str]] = {record.atomic_id: [] for record in records}
    producers_by_id: dict[str, list[str]] = {}
    for record in records:
        producers = sorted(
            {
                result_to_id[operand]
                for operand in record.operands
                if operand in result_to_id and result_to_id[operand] != record.atomic_id
            }
        )
        producers_by_id[record.atomic_id] = producers
        for producer in producers:
            consumers[producer].append(record.atomic_id)
    return tuple(
        AtomicOpRecord(
            atomic_id=record.atomic_id,
            atomic_family=record.atomic_family,
            ttir_op_name=record.ttir_op_name,
            operands=record.operands,
            results=record.results,
            dtype=record.dtype,
            shape=record.shape,
            attrs=record.attrs,
            source_location=record.source_location,
            producer_ids=tuple(producers_by_id[record.atomic_id]),
            consumer_ids=tuple(sorted(consumers[record.atomic_id])),
            local_scope_id=record.local_scope_id,
            unsupported_reason=record.unsupported_reason,
        )
        for record in records
    )


def _classify_ttir_op(op: TTIROp) -> tuple[str, str | None]:
    if op.name in _DIRECT_FAMILY_BY_OP:
        return _DIRECT_FAMILY_BY_OP[op.name], None
    if op.name.startswith("math."):
        return ATOMIC_FAMILY_MATH, None
    return ATOMIC_FAMILY_UNSUPPORTED, f"No active atomic family mapping for TTIR op {op.name}"


def _atomic_dag_hash(records: tuple[AtomicOpRecord, ...], function_name: str) -> str:
    payload = {
        "schema_version": "atomic_dag_v1",
        "function_name": function_name,
        "atomic_ops": [
            {
                key: value
                for key, value in asdict(record).items()
                if key not in {"source_location", "producer_ids", "consumer_ids"}
            }
            for record in records
        ],
    }
    return stable_hash(payload)

