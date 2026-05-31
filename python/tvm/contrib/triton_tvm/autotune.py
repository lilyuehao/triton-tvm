# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Package-level MetaSchedule autotuning helpers for Triton-TVM artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .source.identity import stable_hash


AUTOTUNE_CODEGEN_SOURCE = "te_workload_meta_schedule"


@dataclass(frozen=True)
class AutotuneConfig:
    """MetaSchedule settings plus model operator selection."""

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
    operator_ids: tuple[str, ...] = ()
    config_path: str | None = None
    config_hash: str | None = None


@dataclass(frozen=True)
class AutotuneWorkloadSpec:
    """One canonical workload that may be shared by many model operators."""

    workload_key: str
    operator_id: str
    operator_family: str
    tir_module: Any | None = None
    workload_name: str = "main"
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutotunePlan:
    """Deduplicated autotune workload plan for a model run."""

    selected_operator_ids: tuple[str, ...]
    workloads: tuple[AutotuneWorkloadSpec, ...]
    workload_reuse_map: Mapping[str, tuple[str, ...]]

    @property
    def selected_operator_count(self) -> int:
        return len(self.selected_operator_ids)

    @property
    def selected_workload_count(self) -> int:
        return len(self.workloads)


@dataclass(frozen=True)
class AutotuneResult:
    """Tuned TIR module and report metadata for one workload."""

    workload_key: str
    operator_family: str
    tuned_module: Any
    metadata: Mapping[str, Any]


def load_json_config(path: str | Path) -> tuple[Mapping[str, Any], str]:
    """Load a JSON config and return the parsed object plus stable file hash."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Triton-TVM config must be a JSON object")
    return payload, stable_hash(text)


def autotune_config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    config_path: str | None = None,
    config_hash: str | None = None,
) -> AutotuneConfig:
    """Parse the common ``autotune`` config section."""

    operator_ids = mapping.get("operator_ids", ())
    if operator_ids is None:
        operator_ids = ()
    if not isinstance(operator_ids, list) or not all(
        isinstance(item, str) for item in operator_ids
    ):
        raise ValueError("autotune.operator_ids must be a list of operator_id strings")
    return AutotuneConfig(
        enabled=bool(mapping.get("enabled", False)),
        work_dir=_optional_str(mapping.get("work_dir")),
        max_trials_global=int(mapping.get("max_trials_global", 0)),
        max_trials_per_task=_optional_int(mapping.get("max_trials_per_task")),
        num_trials_per_iter=int(mapping.get("num_trials_per_iter", 1)),
        cost_model=str(mapping.get("cost_model", "random")),
        num_tuning_cores=parse_tuning_cores(mapping.get("num_tuning_cores", 1)),
        task_scheduler=str(mapping.get("task_scheduler", "gradient")),
        strategy=str(mapping.get("strategy", "evolutionary")),
        seed=_optional_int(mapping.get("seed")),
        force_retune=bool(mapping.get("force_retune", False)),
        operator_ids=tuple(operator_ids),
        config_path=config_path,
        config_hash=config_hash,
    )


def build_autotune_plan(
    operator_ids: Iterable[str],
    *,
    available_operator_ids: Iterable[str],
    workload_resolver: Callable[[str], Iterable[AutotuneWorkloadSpec]],
) -> AutotunePlan:
    """Validate selected operators and deduplicate their model-provided workloads."""

    available = set(available_operator_ids)
    selected = []
    seen_selected = set()
    for operator_id in operator_ids:
        if operator_id not in available:
            raise ValueError(f"unknown autotune operator_id: {operator_id}")
        if operator_id not in seen_selected:
            selected.append(operator_id)
            seen_selected.add(operator_id)

    workloads_by_key: dict[str, AutotuneWorkloadSpec] = {}
    reuse: dict[str, list[str]] = {}
    for operator_id in selected:
        specs = tuple(workload_resolver(operator_id))
        if not specs:
            raise ValueError(f"operator has no autotune workload: {operator_id}")
        for spec in specs:
            workloads_by_key.setdefault(spec.workload_key, spec)
            reuse.setdefault(spec.workload_key, [])
            if operator_id not in reuse[spec.workload_key]:
                reuse[spec.workload_key].append(operator_id)

    ordered_workloads = tuple(workloads_by_key[key] for key in workloads_by_key)
    return AutotunePlan(
        selected_operator_ids=tuple(selected),
        workloads=ordered_workloads,
        workload_reuse_map={key: tuple(value) for key, value in reuse.items()},
    )


def tune_or_load_workload(
    spec: AutotuneWorkloadSpec,
    *,
    target: str,
    config: AutotuneConfig,
) -> AutotuneResult:
    """Tune a canonical TE/TIR workload or load an existing MetaSchedule record."""

    if not config.enabled:
        raise ValueError("autotune config is disabled")
    if not str(target).startswith("cuda"):
        raise ValueError(f"autoscheduler currently requires a CUDA target, got {target}")
    if spec.tir_module is None:
        raise ValueError(f"autotune workload has no TIR module: {spec.workload_key}")

    import tvm.s_tir.meta_schedule as ms  # pylint: disable=import-outside-toplevel

    work_dir = Path(config.work_dir or "/tmp/triton_tvm_autotune")
    workload_work_dir = work_dir / spec.workload_key
    workload_work_dir.mkdir(parents=True, exist_ok=True)
    tuning_target = meta_schedule_target(target)
    database = ms.database.JSONDatabase(work_dir=str(workload_work_dir), allow_missing=True)
    tuned_mod = database.query_ir_module(
        spec.tir_module,
        tuning_target,
        workload_name=spec.workload_name,
    )
    record = database.query_tuning_record(
        spec.tir_module,
        tuning_target,
        workload_name=spec.workload_name,
    )
    if config.max_trials_global > 0 and (
        config.force_retune or tuned_mod is None or record is None
    ):
        database = ms.tune_tir(
            spec.tir_module,
            target=tuning_target,
            work_dir=str(workload_work_dir),
            max_trials_global=config.max_trials_global,
            max_trials_per_task=config.max_trials_per_task,
            num_trials_per_iter=config.num_trials_per_iter,
            cost_model=config.cost_model,
            num_tuning_cores=config.num_tuning_cores,
            task_scheduler=config.task_scheduler,
            strategy=config.strategy,
            seed=config.seed,
        )
        tuned_mod = database.query_ir_module(
            spec.tir_module,
            tuning_target,
            workload_name=spec.workload_name,
        )
        record = database.query_tuning_record(
            spec.tir_module,
            tuning_target,
            workload_name=spec.workload_name,
        )

    record_hash = stable_hash(record.as_json()) if record is not None else None
    record_count = len(database.get_all_tuning_records())
    if tuned_mod is None or record is None:
        raise ValueError(
            "missing MetaSchedule record for "
            f"{spec.workload_key}; rerun with max_trials_global > 0"
        )
    metadata = {
        "codegen_source": AUTOTUNE_CODEGEN_SOURCE,
        "workload_class": spec.workload_key,
        "operator_family": spec.operator_family,
        "tuned": True,
        "autotune_enabled": True,
        "work_dir": str(workload_work_dir),
        "record_hash": record_hash,
        "record_count": record_count,
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
    return AutotuneResult(
        workload_key=spec.workload_key,
        operator_family=spec.operator_family,
        tuned_module=tuned_mod,
        metadata=metadata,
    )


def meta_schedule_target(target: str) -> Any:
    """Return a MetaSchedule target with CUDA resource attrs when needed."""

    import tvm  # pylint: disable=import-outside-toplevel

    if not str(target).startswith("cuda"):
        return tvm.target.Target(target)
    base = tvm.target.Target("cuda")
    attrs: dict[str, Any] = {
        "kind": "cuda",
        "max_threads_per_block": int(base.attrs.get("max_num_threads", 1024)),
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": int(base.attrs.get("thread_warp_size", 32)),
    }
    arch = base.attrs.get("arch")
    if arch is not None:
        attrs["arch"] = str(arch)
    return tvm.target.Target(attrs)


def parse_tuning_cores(value: Any) -> int | str:
    """Parse MetaSchedule tuning core count from JSON or CLI-compatible values."""

    if isinstance(value, str):
        if value in {"physical", "logical"}:
            return value
        parsed = int(value)
    else:
        parsed = int(value)
    if parsed <= 0:
        raise ValueError("tuning cores must be positive, 'physical', or 'logical'")
    return parsed


def autotune_plan_report(plan: AutotunePlan | None) -> Mapping[str, Any]:
    """Return report-friendly selected operator/workload metadata."""

    if plan is None:
        return {
            "selected_operator_count": 0,
            "selected_workload_count": 0,
            "workload_reuse_map": {},
        }
    return {
        "selected_operator_count": plan.selected_operator_count,
        "selected_workload_count": plan.selected_workload_count,
        "workload_reuse_map": dict(plan.workload_reuse_map),
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
