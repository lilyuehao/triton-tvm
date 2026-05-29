# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Operator execution ABI, reference execution, and TVM packed artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable, Mapping, MutableMapping, Protocol


EXECUTION_STATUS_EXECUTED = "executed"
EXECUTION_STATUS_GAP = "gap"
EXECUTION_STATUS_FAILED = "failed"
PLACEHOLDER_REASON = "placeholder_after_admission"


@dataclass(frozen=True)
class OperatorExecutionRequest:
    """One admitted top-level operator request for an executor."""

    operator_id: str
    execution_order: int
    model_id: str = ""
    declared_semantic_region_key: str | None = None
    semantic_region_key: str | None = None
    semantic_region_hash: str | None = None
    lowering_contract_ids: tuple[str, ...] = ()
    lowering_contract_status: str = "direct_lowering"
    source_route_record_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    atomic_dag_hashes: tuple[str, ...] = ()
    atomic_dag_bundle_hash: str = ""
    capability_id: str | None = None
    source_id: str = ""
    atomic_dag_hash: str = ""
    shape_signature: Mapping[str, Any] = field(default_factory=dict)
    dtype_signature: Mapping[str, Any] = field(default_factory=dict)
    layout_signature: Mapping[str, Any] = field(default_factory=dict)
    input_buffer_ids: tuple[str, ...] = ()
    output_buffer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "declared_semantic_region_key",
            self.declared_semantic_region_key or self.semantic_region_key,
        )
        if self.source_id and not self.source_ids:
            object.__setattr__(self, "source_ids", (self.source_id,))
        if self.atomic_dag_hash and not self.atomic_dag_hashes:
            object.__setattr__(self, "atomic_dag_hashes", (self.atomic_dag_hash,))
        if self.source_ids and not self.source_id:
            object.__setattr__(self, "source_id", self.source_ids[0])
        if self.atomic_dag_hashes and not self.atomic_dag_hash:
            object.__setattr__(self, "atomic_dag_hash", self.atomic_dag_hashes[0])
        if not self.source_route_record_ids:
            object.__setattr__(self, "source_route_record_ids", (self.operator_id,))


@dataclass(frozen=True)
class OperatorExecutionResult:
    """Result from one operator execution attempt."""

    status: str
    operator_id: str
    backend_id: str | None
    reason: str | None
    elapsed_ms: float | None = None
    output_buffer_ids: tuple[str, ...] = ()
    artifact_source: str | None = None
    correctness_readback: bool = False

    @property
    def executed(self) -> bool:
        return self.status == EXECUTION_STATUS_EXECUTED


@dataclass
class RuntimeBuffer:
    """One runtime buffer slot with explicit metadata and optional storage."""

    buffer_id: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    device: str
    ndarray: Any = None
    producer_operator_id: str | None = None
    consumer_operator_ids: tuple[str, ...] = ()


@dataclass
class RuntimeBufferTable:
    """Mutable buffer table shared by executors."""

    buffers: MutableMapping[str, RuntimeBuffer]

    def has(self, buffer_id: str) -> bool:
        return buffer_id in self.buffers and self.buffers[buffer_id].ndarray is not None

    def get(self, buffer_id: str) -> Any:
        return self.buffers[buffer_id].ndarray

    def set(self, buffer_id: str, value: Any) -> None:
        if buffer_id in self.buffers:
            self.buffers[buffer_id].ndarray = value
            return
        shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", "float32"))
        self.buffers[buffer_id] = RuntimeBuffer(
            buffer_id=buffer_id,
            shape=shape,
            dtype=dtype,
            layout="unknown",
            device="unknown",
            ndarray=value,
        )


@dataclass(frozen=True)
class RuntimeBufferEdge:
    """Producer/consumer edge for one top-level runtime buffer."""

    buffer_id: str
    producer_operator_id: str | None
    consumer_operator_ids: tuple[str, ...]
    is_model_input: bool = False
    is_parameter: bool = False
    is_model_output: bool = False


@dataclass(frozen=True)
class RuntimeBufferPlan:
    """Top-level op-by-op buffer plan."""

    buffers: Mapping[str, RuntimeBuffer]
    edges: Mapping[str, RuntimeBufferEdge]
    final_output_buffer_ids: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeBufferPlanValidationResult:
    """Validation result for a RuntimeBufferPlan."""

    valid: bool
    reason: str | None = None


@dataclass
class ExecutionContext:
    """Execution context shared by executors."""

    buffer_table: MutableMapping[str, Any] | RuntimeBufferTable
    target: str
    debug_flags: Mapping[str, Any] | None = None
    timing_enabled: bool = False


class OperatorExecutor(Protocol):
    """Executor protocol used by the registry."""

    backend_id: str

    def can_execute(self, request: OperatorExecutionRequest) -> bool:
        """Whether this executor accepts the request."""

    def execute(
        self, request: OperatorExecutionRequest, context: ExecutionContext
    ) -> OperatorExecutionResult:
        """Execute or return an explicit gap/failure."""


@dataclass(frozen=True)
class _ExecutorEntry:
    executor: OperatorExecutor
    capability_id: str | None
    semantic_region_key: str | None


class ExecutorRegistry:
    """Dispatch operators by capability first, then Semantic Region."""

    def __init__(self, entries: tuple[_ExecutorEntry, ...] | list[_ExecutorEntry] = ()):
        self._entries = tuple(entries)

    @classmethod
    def empty(cls) -> "ExecutorRegistry":
        return cls(())

    def with_executor(
        self,
        executor: OperatorExecutor,
        *,
        capability_id: str | None = None,
        semantic_region_key: str | None = None,
    ) -> "ExecutorRegistry":
        if capability_id is None and semantic_region_key is None:
            raise ValueError("executor registration requires a capability or Semantic Region key")
        return ExecutorRegistry(
            self._entries
            + (
                _ExecutorEntry(
                    executor=executor,
                    capability_id=capability_id,
                    semantic_region_key=semantic_region_key,
                ),
            )
        )

    def resolve(self, request: OperatorExecutionRequest) -> OperatorExecutor | None:
        for entry in self._entries:
            if (
                entry.capability_id is not None
                and entry.capability_id == request.capability_id
                and entry.executor.can_execute(request)
            ):
                return entry.executor
        for entry in self._entries:
            if (
                entry.semantic_region_key is not None
                and entry.semantic_region_key in {request.semantic_region_key, "*"}
                and entry.executor.can_execute(request)
            ):
                return entry.executor
        return None

    def execute(
        self, request: OperatorExecutionRequest, context: ExecutionContext
    ) -> OperatorExecutionResult:
        executor = self.resolve(request)
        if executor is None:
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_GAP,
                operator_id=request.operator_id,
                backend_id=None,
                reason="no executor registered",
            )
        return executor.execute(request, context)


class PlaceholderExecutor:
    """Executor placeholder used before real operator execution exists."""

    backend_id = "placeholder"

    def can_execute(self, request: OperatorExecutionRequest) -> bool:
        return True

    def execute(
        self, request: OperatorExecutionRequest, context: ExecutionContext
    ) -> OperatorExecutionResult:
        return OperatorExecutionResult(
            status=EXECUTION_STATUS_GAP,
            operator_id=request.operator_id,
            backend_id=self.backend_id,
            reason=PLACEHOLDER_REASON,
        )


@dataclass(frozen=True)
class ReferenceOperatorSpec:
    """Concrete per-operator reference implementation spec."""

    operator_id: str
    semantic_region_key: str
    input_buffer_ids: tuple[str, ...]
    output_buffer_ids: tuple[str, ...]
    attrs: Mapping[str, Any]
    reference_impl_id: str


class ReferenceTensorExecutor:
    """Numpy reference executor used only as a correctness oracle."""

    backend_id = "reference_tensor"

    def __init__(
        self,
        operator_specs: Iterable[ReferenceOperatorSpec] = (),
        *,
        include_legacy_defaults: bool = True,
    ):
        specs = list(operator_specs)
        if include_legacy_defaults:
            specs.extend(_legacy_reference_specs())
        self._operator_specs = {spec.operator_id: spec for spec in specs}

    def can_execute(self, request: OperatorExecutionRequest) -> bool:
        spec = self._operator_specs.get(request.operator_id)
        return spec is not None and spec.semantic_region_key == request.semantic_region_key

    def execute(
        self, request: OperatorExecutionRequest, context: ExecutionContext
    ) -> OperatorExecutionResult:
        spec = self._operator_specs.get(request.operator_id)
        if spec is None:
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_GAP,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason="missing ReferenceOperatorSpec",
            )
        if spec.semantic_region_key != request.semantic_region_key:
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_FAILED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason="ReferenceOperatorSpec Semantic Region mismatch",
            )
        if request.input_buffer_ids and tuple(request.input_buffer_ids) != spec.input_buffer_ids:
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_FAILED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason="ReferenceOperatorSpec input buffers mismatch",
            )
        if request.output_buffer_ids and tuple(request.output_buffer_ids) != spec.output_buffer_ids:
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_FAILED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason="ReferenceOperatorSpec output buffers mismatch",
            )
        try:
            start = time.perf_counter()
            self._execute_spec(spec, context)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except Exception as err:  # pragma: no cover - defensive ABI boundary
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_FAILED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason=str(err),
            )
        return OperatorExecutionResult(
            status=EXECUTION_STATUS_EXECUTED,
            operator_id=request.operator_id,
            backend_id=self.backend_id,
            reason=None,
            elapsed_ms=elapsed_ms,
            output_buffer_ids=spec.output_buffer_ids,
        )

    def _execute_spec(self, spec: ReferenceOperatorSpec, context: ExecutionContext) -> None:
        impl_id = spec.reference_impl_id
        if impl_id == "copy":
            result = _copy_array(_read(context, spec.input_buffer_ids[0]))
        elif impl_id in {"add", "residual_add"}:
            result = _as_numpy(_read(context, spec.input_buffer_ids[0])) + _as_numpy(
                _read(context, spec.input_buffer_ids[1])
            )
        elif impl_id == "class_token_add":
            x = _as_numpy(_read(context, spec.input_buffer_ids[0]))
            cls = _as_numpy(_read(context, spec.input_buffer_ids[1]))
            result = x.copy()
            result[0:1, :] = result[0:1, :] + cls
        elif impl_id == "position_add":
            result = _as_numpy(_read(context, spec.input_buffer_ids[0])) + _as_numpy(
                _read(context, spec.input_buffer_ids[1])
            )
        elif impl_id == "gelu":
            x = _as_numpy(_read(context, spec.input_buffer_ids[0]))
            result = 0.5 * x * (
                1.0 + _np().tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
            )
        elif impl_id == "norm_row":
            x = _as_numpy(_read(context, spec.input_buffer_ids[0]))
            scale = _as_numpy(_read(context, spec.input_buffer_ids[1]))
            bias = _as_numpy(_read(context, spec.input_buffer_ids[2]))
            eps = float(spec.attrs.get("epsilon", 1.0e-5))
            mean = x.mean(axis=-1, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            result = (x - mean) / _np().sqrt(var + eps) * scale + bias
        elif impl_id in {"matmul", "matmul_bias"}:
            lhs = _as_numpy(_read(context, spec.input_buffer_ids[0]))
            rhs = _as_numpy(_read(context, spec.input_buffer_ids[1]))
            result = lhs @ rhs
            if len(spec.input_buffer_ids) >= 3:
                result = result + _as_numpy(_read(context, spec.input_buffer_ids[2]))
        elif impl_id == "pooler":
            lhs = _as_numpy(_read(context, spec.input_buffer_ids[0]))[0:1, :]
            rhs = _as_numpy(_read(context, spec.input_buffer_ids[1]))
            result = lhs @ rhs + _as_numpy(_read(context, spec.input_buffer_ids[2]))
        elif impl_id == "conv_patchify":
            result = _reference_conv_patchify(spec, context)
        elif impl_id == "attention_decomposed":
            result = _reference_attention(spec, context)
        else:
            raise ValueError(f"unsupported reference implementation: {impl_id}")
        _write(context, spec.output_buffer_ids[0], _np().asarray(result, dtype=_np().float32))


@dataclass(frozen=True)
class LoweringStep:
    """One executable PrimFunc step in a region lowering plan."""

    name: str
    contract_id: str
    primfunc_name: str
    input_buffer_ids: tuple[str, ...]
    output_buffer_ids: tuple[str, ...]
    intermediate_buffer_ids: tuple[str, ...] = ()
    arg_buffer_order: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TIRRegionArtifact:
    """TIR artifact with top-level semantic-region metadata."""

    model_id: str
    operator_id: str
    semantic_region_key: str
    lowering_contract_ids: tuple[str, ...]
    lowering_contract_status: str
    source_route_record_ids: tuple[str, ...]
    atomic_dag_hashes: tuple[str, ...]
    atomic_dag_bundle_hash: str
    tir_module: Any
    target: str
    lowering_plan: tuple[LoweringStep, ...]
    region_meta: Mapping[str, Any]
    input_buffer_ids: tuple[str, ...]
    output_buffer_ids: tuple[str, ...]
    artifact_source: str
    build_cache_key: str

    @property
    def entry_names(self) -> tuple[str, ...]:
        return tuple(step.primfunc_name for step in self.lowering_plan)


@dataclass
class BuiltTIRRegionArtifact:
    """Built runtime module plus packed entry functions."""

    artifact: TIRRegionArtifact
    runtime_module: Any
    packed_funcs: Mapping[str, Any]
    device: Any


class TIRArtifactValidationError(ValueError):
    """Raised when a TIRRegionArtifact fails request-time validation."""


class TIRArtifactRegistry:
    """Lookup TIRRegionArtifact by ``(model_id, operator_id)``."""

    def __init__(self, artifacts: Iterable[TIRRegionArtifact] = ()):
        self._artifacts = {(artifact.model_id, artifact.operator_id): artifact for artifact in artifacts}

    def get(self, model_id: str, operator_id: str) -> TIRRegionArtifact | None:
        return self._artifacts.get((model_id, operator_id))

    def has_request(self, request: OperatorExecutionRequest) -> bool:
        return self.get(request.model_id, request.operator_id) is not None

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._artifacts))


class TIRArtifactBuildCache:
    """Small build cache for TVM runtime modules."""

    def __init__(self):
        self._built: dict[tuple[str, str], BuiltTIRRegionArtifact] = {}
        self.build_count = 0
        self.hit_count = 0

    def compile_or_get(self, artifact: TIRRegionArtifact) -> BuiltTIRRegionArtifact:
        key = (artifact.build_cache_key, artifact.target)
        cached = self._built.get(key)
        if cached is not None:
            self.hit_count += 1
            return cached
        tvm = _import_tvm()
        runtime_module = tvm.build(artifact.tir_module, target=artifact.target)
        packed_funcs = {entry_name: runtime_module[entry_name] for entry_name in artifact.entry_names}
        built = BuiltTIRRegionArtifact(
            artifact=artifact,
            runtime_module=runtime_module,
            packed_funcs=packed_funcs,
            device=_device_for_target(tvm, artifact.target),
        )
        self._built[key] = built
        self.build_count += 1
        return built


class TvmPackedExecutor:
    """TVM packed executor backed by TIRRegionArtifact lookup and validation."""

    backend_id = "tvm_packed"

    def __init__(
        self,
        packed_funcs: Mapping[str, Any] | None = None,
        *,
        artifact_registry: TIRArtifactRegistry | None = None,
        build_cache: TIRArtifactBuildCache | None = None,
    ):
        if packed_funcs:
            raise ValueError("TvmPackedExecutor Python callable fallback is disabled")
        self._artifact_registry = artifact_registry or TIRArtifactRegistry()
        self.build_cache = build_cache or TIRArtifactBuildCache()

    def can_execute(self, request: OperatorExecutionRequest) -> bool:
        return self._artifact_registry.has_request(request)

    def execute(
        self, request: OperatorExecutionRequest, context: ExecutionContext
    ) -> OperatorExecutionResult:
        artifact = self._artifact_registry.get(request.model_id, request.operator_id)
        if artifact is not None:
            return self._execute_artifact(request, context, artifact)
        return OperatorExecutionResult(
            status=EXECUTION_STATUS_GAP,
            operator_id=request.operator_id,
            backend_id=self.backend_id,
            reason="no TIR artifact registered",
        )

    def _execute_artifact(
        self,
        request: OperatorExecutionRequest,
        context: ExecutionContext,
        artifact: TIRRegionArtifact,
    ) -> OperatorExecutionResult:
        try:
            validate_tir_region_artifact(artifact, request, context)
            start = time.perf_counter()
            built = self.build_cache.compile_or_get(artifact)
            produced: set[str] = set()
            for step in artifact.lowering_plan:
                args = [
                    _get_or_create_tvm_buffer(
                        context,
                        artifact,
                        buffer_id,
                        built.device,
                        is_input=buffer_id in step.input_buffer_ids and buffer_id not in produced,
                    )
                    for buffer_id in step.arg_buffer_order
                ]
                built.packed_funcs[step.primfunc_name](*args)
                produced.update(step.output_buffer_ids)
                produced.update(step.intermediate_buffer_ids)
            for output_buffer_id in artifact.output_buffer_ids:
                if not _has(context, output_buffer_id):
                    raise ValueError(f"output buffer missing after TVM execution: {output_buffer_id}")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except TIRArtifactValidationError as err:
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_FAILED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason=str(err),
                artifact_source=artifact.artifact_source,
            )
        except Exception as err:  # pragma: no cover - defensive ABI boundary
            return OperatorExecutionResult(
                status=EXECUTION_STATUS_FAILED,
                operator_id=request.operator_id,
                backend_id=self.backend_id,
                reason=str(err),
                artifact_source=artifact.artifact_source,
            )
        return OperatorExecutionResult(
            status=EXECUTION_STATUS_EXECUTED,
            operator_id=request.operator_id,
            backend_id=self.backend_id,
            reason=None,
            elapsed_ms=elapsed_ms,
            output_buffer_ids=artifact.output_buffer_ids,
            artifact_source=artifact.artifact_source,
            correctness_readback=True,
        )


def validate_runtime_buffer_plan(
    plan: RuntimeBufferPlan,
    requests: Iterable[OperatorExecutionRequest],
) -> RuntimeBufferPlanValidationResult:
    """Validate that top-level operator requests form a connected buffer graph."""

    request_list = tuple(sorted(requests, key=lambda request: request.execution_order))
    request_by_id = {request.operator_id: request for request in request_list}
    order_by_operator = {request.operator_id: request.execution_order for request in request_list}

    for buffer_id, edge in plan.edges.items():
        if buffer_id not in plan.buffers:
            return RuntimeBufferPlanValidationResult(False, f"edge references unknown buffer: {buffer_id}")
        if (
            edge.producer_operator_id is None
            and not edge.is_model_input
            and not edge.is_parameter
        ):
            return RuntimeBufferPlanValidationResult(False, f"buffer has no producer: {buffer_id}")
        if edge.producer_operator_id is not None and edge.producer_operator_id not in request_by_id:
            return RuntimeBufferPlanValidationResult(
                False, f"producer references unknown operator: {edge.producer_operator_id}"
            )
        for consumer in edge.consumer_operator_ids:
            if consumer not in request_by_id:
                return RuntimeBufferPlanValidationResult(
                    False, f"consumer references unknown operator: {consumer}"
                )
            if (
                edge.producer_operator_id is not None
                and order_by_operator[edge.producer_operator_id] >= order_by_operator[consumer]
            ):
                return RuntimeBufferPlanValidationResult(
                    False,
                    f"producer must execute before consumer for buffer: {buffer_id}",
                )

    for request in request_list:
        for buffer_id in request.input_buffer_ids + request.output_buffer_ids:
            if buffer_id not in plan.buffers:
                return RuntimeBufferPlanValidationResult(
                    False, f"request references unknown buffer: {buffer_id}"
                )
        for output_buffer_id in request.output_buffer_ids:
            edge = plan.edges.get(output_buffer_id)
            if edge is None or edge.producer_operator_id != request.operator_id:
                return RuntimeBufferPlanValidationResult(
                    False, f"request output has no matching producer edge: {output_buffer_id}"
                )

    for buffer_id in plan.final_output_buffer_ids:
        edge = plan.edges.get(buffer_id)
        if edge is None or edge.producer_operator_id is None:
            return RuntimeBufferPlanValidationResult(
                False, f"final output cannot be traced to a producer: {buffer_id}"
            )
    return RuntimeBufferPlanValidationResult(True)


def validate_tir_region_artifact(
    artifact: TIRRegionArtifact,
    request: OperatorExecutionRequest,
    context: ExecutionContext,
) -> None:
    """Validate a TIR artifact against an execution request before building."""

    tvm = _import_tvm()
    if not isinstance(artifact.tir_module, tvm.IRModule):
        raise TIRArtifactValidationError("tir_module must be a tvm.IRModule")
    if not artifact.lowering_plan:
        raise TIRArtifactValidationError("lowering_plan must not be empty")
    if len(artifact.entry_names) != len(artifact.lowering_plan):
        raise TIRArtifactValidationError("entry_names and lowering_plan length mismatch")
    for entry_name, step in zip(artifact.entry_names, artifact.lowering_plan):
        if entry_name != step.primfunc_name:
            raise TIRArtifactValidationError("lowering step PrimFunc name mismatch")
        try:
            func = artifact.tir_module[entry_name]
        except Exception as err:
            raise TIRArtifactValidationError(f"missing entry PrimFunc: {entry_name}") from err
        if "PrimFunc" not in type(func).__name__:
            raise TIRArtifactValidationError(f"entry is not a PrimFunc: {entry_name}")
        declared = set(artifact.input_buffer_ids) | set(artifact.output_buffer_ids)
        declared |= set(step.input_buffer_ids) | set(step.output_buffer_ids)
        declared |= set(step.intermediate_buffer_ids)
        declared |= set(artifact.region_meta.get("buffers", {}))
        if any(buffer_id not in declared for buffer_id in step.arg_buffer_order):
            raise TIRArtifactValidationError("arg order references undeclared buffer")
        required_args = set(step.input_buffer_ids) | set(step.output_buffer_ids)
        if not required_args.issubset(set(step.arg_buffer_order)):
            raise TIRArtifactValidationError("arg order missing step input or output buffer")
    if artifact.model_id != request.model_id:
        raise TIRArtifactValidationError("artifact model_id does not match request")
    if artifact.operator_id != request.operator_id:
        raise TIRArtifactValidationError("artifact operator_id does not match request")
    if artifact.target != context.target:
        raise TIRArtifactValidationError("artifact target does not match execution context")
    if artifact.input_buffer_ids != request.input_buffer_ids:
        raise TIRArtifactValidationError("artifact input buffers do not match request")
    if artifact.output_buffer_ids != request.output_buffer_ids:
        raise TIRArtifactValidationError("artifact output buffers do not match request")
    if artifact.semantic_region_key != request.semantic_region_key:
        raise TIRArtifactValidationError("artifact Semantic Region key does not match request")
    if artifact.lowering_contract_ids != request.lowering_contract_ids:
        raise TIRArtifactValidationError("artifact lowering contracts do not match request")
    if artifact.lowering_contract_status != request.lowering_contract_status:
        raise TIRArtifactValidationError("artifact lowering contract status does not match request")
    if artifact.source_route_record_ids != request.source_route_record_ids:
        raise TIRArtifactValidationError("artifact source route records do not match request")
    if artifact.atomic_dag_hashes != request.atomic_dag_hashes:
        raise TIRArtifactValidationError("artifact Atomic DAG hashes do not match request")
    if artifact.atomic_dag_bundle_hash != request.atomic_dag_bundle_hash:
        raise TIRArtifactValidationError("artifact Atomic DAG bundle hash does not match request")
    if request.semantic_region_hash is not None:
        meta_hash = artifact.region_meta.get("semantic_region_hash")
        if meta_hash != request.semantic_region_hash:
            raise TIRArtifactValidationError("artifact Semantic Region hash does not match request")
    required_meta = {
        "semantic_region_hash",
        "source_route_record_ids",
        "atomic_dag_hashes",
        "atomic_dag_bundle_hash",
        "shape",
        "dtype",
        "layout",
    }
    missing_meta = sorted(key for key in required_meta if key not in artifact.region_meta)
    if missing_meta:
        raise TIRArtifactValidationError("artifact region_meta missing: " + ", ".join(missing_meta))


def _legacy_reference_specs() -> tuple[ReferenceOperatorSpec, ...]:
    return (
        ReferenceOperatorSpec(
            operator_id="copy",
            semantic_region_key="pointwise_flat",
            input_buffer_ids=("x",),
            output_buffer_ids=("y",),
            attrs={},
            reference_impl_id="copy",
        ),
        ReferenceOperatorSpec(
            operator_id="matmul_bias",
            semantic_region_key="matmul_bias_epilogue",
            input_buffer_ids=("a", "b", "bias"),
            output_buffer_ids=("out",),
            attrs={},
            reference_impl_id="matmul_bias",
        ),
    )


def _reference_conv_patchify(spec: ReferenceOperatorSpec, context: ExecutionContext) -> Any:
    image = _as_numpy(_read(context, spec.input_buffer_ids[0]))
    weight = _as_numpy(_read(context, spec.input_buffer_ids[1]))
    bias = _as_numpy(_read(context, spec.input_buffer_ids[2]))
    patch = int(spec.attrs.get("patch", 2))
    grid_h = image.shape[1] // patch
    grid_w = image.shape[2] // patch
    hidden = weight.shape[0]
    result = _np().zeros((grid_h * grid_w, hidden), dtype=_np().float32)
    for token in range(grid_h * grid_w):
        py = token // grid_w
        px = token % grid_w
        window = image[:, py * patch : (py + 1) * patch, px * patch : (px + 1) * patch]
        for h in range(hidden):
            result[token, h] = _np().sum(window * weight[h]) + bias[h]
    return result


def _reference_attention(spec: ReferenceOperatorSpec, context: ExecutionContext) -> Any:
    qkv = _as_numpy(_read(context, spec.input_buffer_ids[0]))
    hidden = int(spec.attrs.get("hidden", qkv.shape[1] // 3))
    q = qkv[:, :hidden]
    k = qkv[:, hidden : 2 * hidden]
    v = qkv[:, 2 * hidden :]
    scores = (q @ k.T) / math.sqrt(float(hidden))
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = _np().exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v


def _get_or_create_tvm_buffer(
    context: ExecutionContext,
    artifact: TIRRegionArtifact,
    buffer_id: str,
    device: Any,
    *,
    is_input: bool,
) -> Any:
    value_exists = _has(context, buffer_id)
    if value_exists:
        value = _read(context, buffer_id)
        if _is_tvm_tensor(value):
            return value
        if artifact.target != "llvm":
            raise ValueError("host ndarray cannot be passed to non-llvm TVM executor")
        tensor = _empty_tvm_tensor(artifact.target, device, value.shape, str(value.dtype))
        tensor.copyfrom(value)
        _write(context, buffer_id, tensor)
        return tensor
    if is_input:
        raise ValueError(f"input buffer missing: {buffer_id}")
    shape, dtype = _shape_dtype_for_buffer(artifact, buffer_id)
    tensor = _empty_tvm_tensor(artifact.target, device, shape, dtype)
    _write(context, buffer_id, tensor)
    return tensor


def _shape_dtype_for_buffer(artifact: TIRRegionArtifact, buffer_id: str) -> tuple[tuple[int, ...], str]:
    buffers = artifact.region_meta.get("buffers", {})
    meta = buffers.get(buffer_id)
    if meta is None:
        raise ValueError(f"buffer metadata missing: {buffer_id}")
    return tuple(int(dim) for dim in meta["shape"]), str(meta.get("dtype", "float32"))


def _read(context: ExecutionContext, buffer_id: str) -> Any:
    if isinstance(context.buffer_table, RuntimeBufferTable):
        return context.buffer_table.get(buffer_id)
    return context.buffer_table[buffer_id]


def _write(context: ExecutionContext, buffer_id: str, value: Any) -> None:
    if isinstance(context.buffer_table, RuntimeBufferTable):
        context.buffer_table.set(buffer_id, value)
    else:
        context.buffer_table[buffer_id] = value


def _has(context: ExecutionContext, buffer_id: str) -> bool:
    if isinstance(context.buffer_table, RuntimeBufferTable):
        return context.buffer_table.has(buffer_id)
    return buffer_id in context.buffer_table


def _as_numpy(value: Any):
    if hasattr(value, "numpy"):
        return value.numpy()
    return _np().asarray(value)


def _copy_array(value: Any):
    return _np().array(_as_numpy(value), copy=True)


def _np():
    import numpy as np  # pylint: disable=import-outside-toplevel

    return np


def _import_tvm():
    import tvm  # pylint: disable=import-outside-toplevel

    return tvm


def _device_for_target(tvm: Any, target: str) -> Any:
    if target.startswith("cuda"):
        return tvm.cuda(0)
    return tvm.cpu(0)


def _empty_tvm_tensor(target: str, device: Any, shape: tuple[int, ...], dtype: str) -> Any:
    tvm = _import_tvm()
    if hasattr(tvm, "runtime") and hasattr(tvm.runtime, "empty"):
        return tvm.runtime.empty(shape, dtype, device)
    return tvm.nd.empty(shape, dtype, device)


def _is_tvm_tensor(value: Any) -> bool:
    return hasattr(value, "copyfrom") and hasattr(value, "numpy") and hasattr(value, "device")
