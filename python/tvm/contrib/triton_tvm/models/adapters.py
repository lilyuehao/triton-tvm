# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Minimal model adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..atomic import AtomicDAGRecord, build_atomic_dag
from ..source import SourceRecord


@dataclass(frozen=True)
class ModelOperatorInput:
    """Materialized model operator input without support claims."""

    model_id: str
    operator_id: str
    execution_order: int
    source: SourceRecord
    ttir: str
    optional_model_anchor: str | None = None


class ModelAdapter:
    """Adapter that materializes inputs but cannot declare backend support."""

    def __init__(self, model_id: str, operators: tuple[ModelOperatorInput, ...]):
        self.model_id = model_id
        self._operators = operators

    def operators(self) -> tuple[ModelOperatorInput, ...]:
        return self._operators

    def atomic_inputs(self) -> tuple[AtomicDAGRecord, ...]:
        return tuple(build_atomic_dag(item.source, item.ttir) for item in self._operators)


def reject_adapter_support_declaration(**_: object) -> None:
    """Model adapters may not declare support."""

    raise ValueError("model adapters materialize inputs only; registry resolves support")

