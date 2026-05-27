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
"""M12.3 fixed-shape ViT E2E correctness runner.

This runner is a correctness/P0 gate, not the M12 performance dashboard.  It
executes the fixed `vit_tiny_random` Inductor wrapper under an explicit provider
mix: wrapper GEMM/addmm calls are replaced by M12.2 Native TVM matmul kernels,
the wrapper conv call remains the explicit `device_torch_cuda` provider, and
the wrapper SDPA call is replayed through the M10 native-decomposed primitive
sequence.  Captured Inductor Triton kernels are kept as an explicit harness
execution kind in this step; M12.3 does not claim strict full-native TVM.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tvm

from .attention import (
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
)
from .inductor import (
    extract_inductor_triton_sources,
    extract_inductor_wrapper_extern_calls,
)
from .matmul import (
    EXTERN_ADDMM_BIAS_SYMBOL,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM,
    EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    EXTERN_GEMM_SYMBOL,
    NATIVE_TIR_MATMUL_SCHEDULE_ID,
    TargetMatmulDecision,
    TargetMatmulPolicy,
    build_native_wrapper_matmul_tirx_source,
    extract_matmul_semantics_from_wrapper_extern,
    extract_matmul_semantics_from_wrapper_extern_addmm,
)
from .model_corpus import _make_vit_inputs, _make_vit_tiny, _set_offline_env, _seed_torch
from .runtime import TritonTVMArtifact, build_triton_tvm
from .translator import TritonTVMMeta
from .vision import (
    VISION_PROVIDER_DEVICE_TORCH_CUDA,
    VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
)


REPORT_KIND = "triton_tvm_m12_3_vit_fixed_shape_e2e_runner"
RUNNER_ID = "m12_3_vit_fixed_shape_e2e_correctness_v1"
TARGET_MODEL = "vit_tiny_random"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_3_vit_fixed_shape_e2e_runner"
)
ALLCLOSE_RTOL = 1e-4
ALLCLOSE_ATOL = 1e-4
NATIVE_MATMUL_SYNC_POLICY_LEGACY = "legacy_per_call_sync"
NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC = "no_per_call_sync"
NATIVE_MATMUL_SYNC_POLICIES = frozenset(
    {
        NATIVE_MATMUL_SYNC_POLICY_LEGACY,
        NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
    }
)
NATIVE_MATMUL_DISPATCH_MODE_ARTIFACT_RUN = "artifact_run"
NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL = "prebound_packed_call"
NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD = "prebound_minimal_record"
NATIVE_MATMUL_DISPATCH_MODES = frozenset(
    {
        NATIVE_MATMUL_DISPATCH_MODE_ARTIFACT_RUN,
        NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL,
        NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
    }
)


def _elapsed_ms(start: float, end: float | None = None) -> float:
    if end is None:
        end = time.perf_counter()
    return max(0.0, (end - start) * 1000.0)


def _rounded_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _positive_delta_ms(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return _rounded_ms(max(0.0, float(left) - float(right)))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NativeMatmulProfileConfig:
    """Opt-in profiling controls for M12 fixed-shape native matmul calls."""

    enabled: bool = False
    sync_policy: str = NATIVE_MATMUL_SYNC_POLICY_LEGACY
    cuda_event_timing: bool = True
    dispatch_mode: str = NATIVE_MATMUL_DISPATCH_MODE_ARTIFACT_RUN

    def __post_init__(self) -> None:
        if self.sync_policy not in NATIVE_MATMUL_SYNC_POLICIES:
            raise ValueError(f"unknown native matmul sync policy: {self.sync_policy}")
        if self.dispatch_mode not in NATIVE_MATMUL_DISPATCH_MODES:
            raise ValueError(f"unknown native matmul dispatch mode: {self.dispatch_mode}")


@dataclass(frozen=True)
class NativeMatmulRuntimeCall:
    """One M12.3 Native TVM wrapper matmul executable."""

    index: int
    op_family: str
    op_name: str
    line_no: int
    source: str
    artifact: TritonTVMArtifact
    packed_func: Any
    decision: TargetMatmulDecision
    matmul_m: int
    matmul_n: int
    matmul_k: int
    b_storage_shape: tuple[int, int]
    b_storage_stride: tuple[int, int]

    def report_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "op_family": self.op_family,
            "op_name": self.op_name,
            "line_no": self.line_no,
            "matmul_m": self.matmul_m,
            "matmul_n": self.matmul_n,
            "matmul_k": self.matmul_k,
            "implementation_kind": self.decision.implementation_kind,
            "schedule_id": self.decision.schedule_id,
            "runtime_status": self.decision.extern_gemm_runtime_status,
            "provider_kind": self.decision.extern_gemm_provider_kind,
            "runtime_claim": self.decision.extern_gemm_runtime_claim,
            "uses_host_staging": self.decision.extern_gemm_uses_host_staging,
            "performance_claim": self.decision.extern_gemm_performance_claim,
        }


@dataclass(frozen=True)
class VitFixedShapeE2EPrepared:
    """Captured fixed-shape ViT executable and provider artifacts."""

    torch: Any
    model: Any
    compiled: Any
    pixel_values: Any
    wrapper_source: str
    matmul_calls: list[NativeMatmulRuntimeCall]
    expected: dict[str, Any]


class _NativeMatmulPatchState:
    """Mutable execution state for the patched wrapper externs."""

    def __init__(
        self,
        calls: list[NativeMatmulRuntimeCall],
        torch_module,
        *,
        profile_config: NativeMatmulProfileConfig | None = None,
    ):
        self._torch = torch_module
        self._profile_config = profile_config or NativeMatmulProfileConfig()
        if self._profile_config.sync_policy not in NATIVE_MATMUL_SYNC_POLICIES:
            raise ValueError(
                f"unknown native matmul sync policy: {self._profile_config.sync_policy}"
            )
        self._calls = list(calls)
        self._pending_tvm_args: list[list[Any]] = []
        self.executed: list[dict[str, Any]] = []
        self.provider_profiles: list[dict[str, Any]] = []
        self.conv_runtime_calls = 0
        self.attention_runtime_calls = 0
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self._addmm = deque(
            call for call in self._calls if call.op_family == "extern_addmm_bias"
        )
        self._gemm = deque(call for call in self._calls if call.op_family == "extern_gemm")
        self._pending_tvm_args.clear()
        self.executed.clear()
        self.provider_profiles.clear()
        self.conv_runtime_calls = 0
        self.attention_runtime_calls = 0

    def addmm(self, bias, lhs, rhs, *args, **kwargs):
        timing_enabled = self._diagnostic_timing_enabled()
        validation_start = time.perf_counter() if timing_enabled else None
        out = kwargs.get("out")
        if out is None:
            raise RuntimeError("M12.3 native addmm patch requires out= buffer")
        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 1.0))
        if alpha != 1.0 or beta != 1.0:
            raise RuntimeError(f"M12.3 native addmm patch requires alpha=beta=1, got {alpha}, {beta}")
        call = self._addmm.popleft()
        validation_ms = _elapsed_ms(validation_start) if validation_start is not None else 0.0
        b_storage_start = time.perf_counter() if timing_enabled else None
        rhs_storage = self._b_storage(rhs, call)
        b_storage_ms = _elapsed_ms(b_storage_start) if b_storage_start is not None else 0.0
        self._run_native_matmul(
            call,
            [bias, lhs, rhs_storage, out],
            shape_abi_validation_ms=validation_ms,
            b_storage_resolution_ms=b_storage_ms,
        )
        return out

    def mm(self, lhs, rhs, *args, **kwargs):
        timing_enabled = self._diagnostic_timing_enabled()
        validation_start = time.perf_counter() if timing_enabled else None
        out = kwargs.get("out")
        if out is None:
            raise RuntimeError("M12.3 native mm patch requires out= buffer")
        call = self._gemm.popleft()
        validation_ms = _elapsed_ms(validation_start) if validation_start is not None else 0.0
        b_storage_start = time.perf_counter() if timing_enabled else None
        rhs_storage = self._b_storage(rhs, call)
        b_storage_ms = _elapsed_ms(b_storage_start) if b_storage_start is not None else 0.0
        self._run_native_matmul(
            call,
            [lhs, rhs_storage, out],
            shape_abi_validation_ms=validation_ms,
            b_storage_resolution_ms=b_storage_ms,
        )
        return out

    def wrap_conv(self, original):
        def _conv(*args, **kwargs):
            self.conv_runtime_calls += 1
            if not self._profile_config.enabled:
                return original(*args, **kwargs)
            return self._profile_provider_call(
                provider_kind=VISION_PROVIDER_DEVICE_TORCH_CUDA,
                op_family="extern_convolution",
                call_index=self.conv_runtime_calls - 1,
                fn=lambda: original(*args, **kwargs),
            )

        return _conv

    def native_decomposed_sdpa(self, query, key, value, *args, **kwargs):
        self.attention_runtime_calls += 1
        return self._profile_provider_call(
            provider_kind=ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
            op_family="aten_scaled_dot_product_attention",
            call_index=self.attention_runtime_calls - 1,
            fn=lambda: self._native_decomposed_sdpa_impl(query, key, value, *args, **kwargs),
        )

    def _native_decomposed_sdpa_impl(self, query, key, value, *args, **kwargs):
        scale = kwargs.get("scale")
        if scale is None:
            scale = float(query.shape[-1]) ** -0.5
        scores = self._torch.matmul(query, key.transpose(-2, -1)) * float(scale)
        probs = self._torch.softmax(scores, dim=-1)
        out = self._torch.matmul(probs, value)
        # The Inductor wrapper expects the efficient-attention output to share
        # the query layout: logical (B, H, S, D) over contiguous (B, S, H, D).
        out = out.transpose(1, 2).contiguous().transpose(1, 2)
        return (out,)

    def assert_all_calls_consumed(self) -> None:
        if self._addmm or self._gemm:
            raise RuntimeError(
                "M12.3 native matmul patch did not consume all calls: "
                f"addmm={len(self._addmm)}, gemm={len(self._gemm)}"
            )

    def finalize_profiles(self) -> None:
        """Resolve deferred CUDA-event timings and release no-sync DLPack handles."""

        if not self._profile_config.enabled:
            self._pending_tvm_args.clear()
            return
        self._finalize_event_profiles(self.executed)
        self._finalize_event_profiles(self.provider_profiles)
        self._pending_tvm_args.clear()

    def _finalize_event_profiles(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            profile = record.get("profile")
            if not isinstance(profile, dict):
                continue
            events = profile.pop("_cuda_events", None)
            if events is None:
                continue
            start_event, end_event = events
            try:
                kernel_ms = float(start_event.elapsed_time(end_event))
            except RuntimeError as err:
                profile["kernel_event_error"] = f"{type(err).__name__}:{err}"
                kernel_ms = None
            profile["kernel_event_ms"] = _rounded_ms(kernel_ms)
            profile["dispatch_minus_kernel_estimate_ms"] = _positive_delta_ms(
                profile.get("artifact_dispatch_wall_ms")
                if "artifact_dispatch_wall_ms" in profile
                else profile.get("provider_total_wall_ms"),
                kernel_ms,
            )

    def _profile_provider_call(self, *, provider_kind: str, op_family: str, call_index: int, fn):
        if not self._profile_config.enabled:
            return fn()
        total_start = time.perf_counter()
        start_event = None
        end_event = None
        if self._profile_config.cuda_event_timing:
            start_event = self._torch.cuda.Event(enable_timing=True)
            end_event = self._torch.cuda.Event(enable_timing=True)
            start_event.record()
        result = fn()
        if end_event is not None:
            end_event.record()
        record = {
            "index": call_index,
            "op_family": op_family,
            "provider_kind": provider_kind,
            "runtime_status": (
                VISION_RUNTIME_STATUS_RUNTIME_RESOLVED
                if provider_kind == VISION_PROVIDER_DEVICE_TORCH_CUDA
                else ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED
            ),
            "profile": {
                "sync_policy": self._profile_config.sync_policy,
                "dispatch_mode": self._profile_config.dispatch_mode,
                "provider_total_wall_ms": _rounded_ms(_elapsed_ms(total_start)),
                "kernel_event_ms": None,
                "dispatch_minus_kernel_estimate_ms": None,
            },
        }
        if start_event is not None and end_event is not None:
            record["profile"]["_cuda_events"] = (start_event, end_event)
        self.provider_profiles.append(record)
        return result

    def _diagnostic_timing_enabled(self) -> bool:
        return (
            self._profile_config.enabled
            or self._profile_config.dispatch_mode
            != NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD
        )

    def _run_native_matmul(
        self,
        call: NativeMatmulRuntimeCall,
        torch_args: list[Any],
        *,
        shape_abi_validation_ms: float,
        b_storage_resolution_ms: float,
    ) -> None:
        profile_enabled = self._profile_config.enabled
        if (
            not profile_enabled
            and self._profile_config.dispatch_mode
            == NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD
        ):
            self._run_native_matmul_minimal_record(call, torch_args)
            return

        total_start = time.perf_counter()
        pre_sync_ms = 0.0
        if self._profile_config.sync_policy == NATIVE_MATMUL_SYNC_POLICY_LEGACY:
            pre_sync_start = time.perf_counter()
            self._torch.cuda.synchronize()
            pre_sync_ms = _elapsed_ms(pre_sync_start)

        dlpack_start = time.perf_counter()
        tvm_args = [tvm.runtime.from_dlpack(arg) for arg in torch_args]
        dlpack_ms = _elapsed_ms(dlpack_start)
        if self._profile_config.sync_policy == NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC:
            self._pending_tvm_args.append(tvm_args)

        start_event = None
        end_event = None
        if profile_enabled and self._profile_config.cuda_event_timing:
            start_event = self._torch.cuda.Event(enable_timing=True)
            end_event = self._torch.cuda.Event(enable_timing=True)
            start_event.record()

        dispatch_start = time.perf_counter()
        if (
            self._profile_config.dispatch_mode
            in {
                NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_PACKED_CALL,
                NATIVE_MATMUL_DISPATCH_MODE_PREBOUND_MINIMAL_RECORD,
            }
        ):
            call.packed_func(*tvm_args)
        else:
            call.artifact.run(tvm_args)
        artifact_dispatch_ms = _elapsed_ms(dispatch_start)

        if end_event is not None:
            end_event.record()

        post_sync_ms = 0.0
        if self._profile_config.sync_policy == NATIVE_MATMUL_SYNC_POLICY_LEGACY:
            post_sync_start = time.perf_counter()
            self._torch.cuda.synchronize()
            post_sync_ms = _elapsed_ms(post_sync_start)

        record = call.report_record()
        if profile_enabled:
            profile = {
                "sync_policy": self._profile_config.sync_policy,
                "dispatch_mode": self._profile_config.dispatch_mode,
                "shape_abi_validation_ms": _rounded_ms(shape_abi_validation_ms),
                "b_storage_resolution_ms": _rounded_ms(b_storage_resolution_ms),
                "artifact_lookup_ms": 0.0,
                "dlpack_conversion_ms": _rounded_ms(dlpack_ms),
                "artifact_dispatch_wall_ms": _rounded_ms(artifact_dispatch_ms),
                "pre_sync_wall_ms": _rounded_ms(pre_sync_ms),
                "post_sync_wall_ms": _rounded_ms(post_sync_ms),
                "synchronization_wall_ms": _rounded_ms(pre_sync_ms + post_sync_ms),
                "kernel_event_ms": None,
                "dispatch_minus_kernel_estimate_ms": None,
                "provider_total_wall_ms": _rounded_ms(_elapsed_ms(total_start)),
            }
            if start_event is not None and end_event is not None:
                profile["_cuda_events"] = (start_event, end_event)
            record["profile"] = profile
        self.executed.append(record)

    def _run_native_matmul_minimal_record(
        self,
        call: NativeMatmulRuntimeCall,
        torch_args: list[Any],
    ) -> None:
        if self._profile_config.sync_policy == NATIVE_MATMUL_SYNC_POLICY_LEGACY:
            self._torch.cuda.synchronize()
        tvm_args = [tvm.runtime.from_dlpack(arg) for arg in torch_args]
        if self._profile_config.sync_policy == NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC:
            self._pending_tvm_args.append(tvm_args)
        call.packed_func(*tvm_args)
        if self._profile_config.sync_policy == NATIVE_MATMUL_SYNC_POLICY_LEGACY:
            self._torch.cuda.synchronize()
        self.executed.append(call.report_record())

    def _b_storage(self, rhs, call: NativeMatmulRuntimeCall):
        expected_shape = call.b_storage_shape
        expected_stride = call.b_storage_stride
        base = getattr(rhs, "_base", None)
        if base is not None and tuple(int(dim) for dim in base.shape) == expected_shape:
            return base
        if tuple(int(dim) for dim in rhs.shape) == expected_shape and tuple(
            int(stride) for stride in rhs.stride()
        ) == expected_stride:
            return rhs
        return self._torch.as_strided(
            rhs,
            expected_shape,
            expected_stride,
            storage_offset=int(rhs.storage_offset()),
        )

def run_vit_fixed_shape_e2e_runner(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    seed: int = 0,
    run_execution: bool = True,
) -> dict[str, Any]:
    """Run the M12.3 fixed-shape ViT correctness gate and write a report."""

    if not run_execution:
        report = _empty_report(seed=seed, status="not_run", availability_reason="execution_disabled")
        _maybe_write_report(report, out_dir)
        return report

    torch = _require_torch()
    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_report(
            seed=seed,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
        )
        _maybe_write_report(report, out_dir)
        return report

    prepared = prepare_vit_fixed_shape_e2e(seed=seed)
    actual, patch_state = execute_vit_fixed_shape_provider_path(prepared)
    correctness = _correctness_report(actual, prepared.expected)
    provider_mix = _provider_mix_report(
        wrapper_source=prepared.wrapper_source,
        matmul_calls=prepared.matmul_calls,
        patch_state=patch_state,
    )
    p0 = {
        "correctness": bool(correctness["allclose"]),
        "no_silent_fallback": bool(provider_mix["no_silent_fallback"]),
        "complete_provider_report": bool(provider_mix["complete_provider_report"]),
    }
    p0["passed"] = bool(all(p0.values()))

    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "passed" if p0["passed"] else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "correctness": correctness,
        "p0": p0,
        "provider_mix": provider_mix,
        "native_matmul_records": patch_state.executed,
        "performance_ready_e2e": False,
        "performance_claim": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": 0,
        "notes": [
            "M12.3 is correctness/P0 only; M12.4 owns p50/p95 and P1/P2 performance gates.",
            "Captured Inductor kernels are executed by the explicit Inductor Triton harness in this runner; strict full-native TVM remains false.",
        ],
        "dependency_versions": _dependency_versions(torch),
    }
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "vit_tiny_random_wrapper.py").write_text(
            prepared.wrapper_source, encoding="utf-8"
        )
    _maybe_write_report(report, out_dir)
    return report


def prepare_vit_fixed_shape_e2e(
    *,
    seed: int = 0,
    native_matmul_passes: list[Any] | tuple[Any, ...] | None = None,
) -> VitFixedShapeE2EPrepared:
    """Capture the fixed-shape ViT wrapper and build its explicit provider artifacts."""

    torch = _require_torch()
    model, compiled, pixel_values, wrapper_source = _capture_compiled_vit(torch, seed=seed)
    matmul_calls = _build_native_matmul_calls(
        wrapper_source,
        native_matmul_passes=native_matmul_passes,
    )
    expected = _run_torch_baseline(torch, model, pixel_values)
    return VitFixedShapeE2EPrepared(
        torch=torch,
        model=model,
        compiled=compiled,
        pixel_values=pixel_values,
        wrapper_source=wrapper_source,
        matmul_calls=matmul_calls,
        expected=expected,
    )


def execute_vit_fixed_shape_provider_path(
    prepared: VitFixedShapeE2EPrepared,
    *,
    profile_native_matmul: bool = False,
    native_matmul_sync_policy: str = NATIVE_MATMUL_SYNC_POLICY_LEGACY,
    native_matmul_dispatch_mode: str = NATIVE_MATMUL_DISPATCH_MODE_ARTIFACT_RUN,
) -> tuple[dict[str, Any], _NativeMatmulPatchState]:
    """Execute the prepared ViT path through the explicit M12 provider mix."""

    profile_config = NativeMatmulProfileConfig(
        enabled=profile_native_matmul,
        sync_policy=native_matmul_sync_policy,
        dispatch_mode=native_matmul_dispatch_mode,
    )
    with VitFixedShapeProviderSession(prepared, profile_config=profile_config) as session:
        return session.run()


class VitFixedShapeProviderSession:
    """Reusable patched-provider session for repeated fixed-shape executions."""

    def __init__(
        self,
        prepared: VitFixedShapeE2EPrepared,
        *,
        profile_config: NativeMatmulProfileConfig | None = None,
    ):
        self._prepared = prepared
        self._profile_config = profile_config or NativeMatmulProfileConfig()
        self._ctx = None
        self.patch_state: _NativeMatmulPatchState | None = None

    def __enter__(self):
        self._ctx = _patched_provider_mix(
            self._prepared.torch,
            self._prepared.matmul_calls,
            profile_config=self._profile_config,
        )
        self.patch_state = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._ctx is not None:
                return self._ctx.__exit__(exc_type, exc, tb)
            return None
        finally:
            self._ctx = None
            self.patch_state = None

    def run(self) -> tuple[dict[str, Any], _NativeMatmulPatchState]:
        if self.patch_state is None:
            raise RuntimeError("VitFixedShapeProviderSession must be used as a context manager")
        self.patch_state.reset_runtime_state()
        with self._prepared.torch.no_grad():
            actual_output = self._prepared.compiled(self._prepared.pixel_values)
        self._prepared.torch.cuda.synchronize()
        self.patch_state.finalize_profiles()
        self.patch_state.assert_all_calls_consumed()
        return _output_tensors(actual_output), self.patch_state


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m tvm.contrib.triton_tvm.m123_vit_e2e_runner``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-execution", action="store_true")
    args = parser.parse_args(argv)

    report = run_vit_fixed_shape_e2e_runner(
        out_dir=args.out_dir,
        seed=args.seed,
        run_execution=not args.no_execution,
    )
    correctness = report.get("correctness", {})
    print(
        f"M12.3 ViT E2E runner: status={report['status']} "
        f"allclose={correctness.get('allclose')} out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


def _capture_compiled_vit(torch, *, seed: int):
    import torch._dynamo  # pylint: disable=import-outside-toplevel
    import torch._inductor.config as inductor_config  # pylint: disable=import-outside-toplevel
    from torch._inductor.graph import GraphLowering  # pylint: disable=import-outside-toplevel

    _set_offline_env()
    _seed_torch(torch, seed)
    captured: list[str] = []
    old_save_output_code = GraphLowering.save_output_code
    old_fx_graph_cache = inductor_config.fx_graph_cache
    GraphLowering.save_output_code = captured.append
    inductor_config.fx_graph_cache = False
    torch._dynamo.reset()
    model = _make_vit_tiny(torch).eval().to("cuda")
    args, kwargs = _make_vit_inputs(torch)
    pixel_values = args[0].detach()
    try:
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            compiled(*args, **kwargs)
        torch.cuda.synchronize()
    finally:
        GraphLowering.save_output_code = old_save_output_code
        inductor_config.fx_graph_cache = old_fx_graph_cache
    if len(captured) != 1:
        raise RuntimeError(f"M12.3 expected exactly one ViT wrapper, got {len(captured)}")
    return model, compiled, pixel_values, captured[0]


def _run_torch_baseline(torch, model, pixel_values) -> dict[str, Any]:
    with torch.no_grad():
        output = model(pixel_values)
    torch.cuda.synchronize()
    return _output_tensors(output)


def _output_tensors(output) -> dict[str, Any]:
    return {
        "last_hidden_state": output.last_hidden_state.detach(),
        "pooler_output": output.pooler_output.detach(),
    }


def _build_native_matmul_calls(
    wrapper_source: str,
    *,
    native_matmul_passes: list[Any] | tuple[Any, ...] | None = None,
) -> list[NativeMatmulRuntimeCall]:
    calls = sorted(
        extract_inductor_wrapper_extern_calls(
            wrapper_source,
            case_name=TARGET_MODEL,
            extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
            extern_gemm_runtime_model_case=TARGET_MODEL,
        ),
        key=lambda call: call.line_no,
    )
    matmul_calls = [
        call for call in calls if call.op_family in {"extern_gemm", "extern_addmm_bias"}
    ]
    if len(matmul_calls) != 7:
        raise RuntimeError(f"M12.3 expected 7 ViT matmul/addmm calls, got {len(matmul_calls)}")

    built: list[NativeMatmulRuntimeCall] = []
    for index, call in enumerate(matmul_calls):
        kernel_name = f"m12_3_vit_{call.op_family}_{index}"
        if call.op_family == "extern_addmm_bias":
            semantics = extract_matmul_semantics_from_wrapper_extern_addmm(
                {
                    "op_name": call.op_name,
                    "source": call.source,
                    "case_name": TARGET_MODEL,
                    "kernel_name": kernel_name,
                }
            )
        else:
            semantics = extract_matmul_semantics_from_wrapper_extern(
                {
                    "op_name": call.op_name,
                    "source": call.source,
                    "case_name": TARGET_MODEL,
                    "kernel_name": kernel_name,
                }
            )
        decision = TargetMatmulPolicy(
            target_kind="cuda",
            extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_NATIVE_TVM,
        ).decide(semantics, matmul_contract_ok=True)
        if (
            decision.implementation_kind != "native_tir_schedule"
            or decision.extern_gemm_runtime_status != EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
            or decision.extern_gemm_provider_kind != EXTERN_GEMM_PROVIDER_NATIVE_TVM
        ):
            raise RuntimeError(
                "M12.3 expected native_tvm_matmul runtime-resolved decision, "
                f"got {decision}"
            )
        source = build_native_wrapper_matmul_tirx_source(semantics, decision)
        irmod = tvm.script.from_source(source)
        artifact = build_triton_tvm(
            irmod,
            _native_matmul_meta(semantics, decision),
            passes=native_matmul_passes,
        )
        transposed_b = semantics.b_layout == "transposed_weight_view"
        b_storage_shape = (semantics.n, semantics.k) if transposed_b else (semantics.k, semantics.n)
        b_storage_stride = (semantics.k, 1) if transposed_b else semantics.b_stride
        built.append(
            NativeMatmulRuntimeCall(
                index=index,
                op_family=call.op_family,
                op_name=call.op_name,
                line_no=call.line_no,
                source=call.source,
                artifact=artifact,
                packed_func=artifact.executable[artifact.meta.kernel_name],
                decision=decision,
                matmul_m=semantics.m,
                matmul_n=semantics.n,
                matmul_k=semantics.k,
                b_storage_shape=b_storage_shape,
                b_storage_stride=b_storage_stride,
            )
        )
    return built


def _native_matmul_meta(semantics, decision: TargetMatmulDecision) -> TritonTVMMeta:
    transposed_b = semantics.b_layout == "transposed_weight_view"
    b_storage_shape = (semantics.n, semantics.k) if transposed_b else (semantics.k, semantics.n)
    abi = []
    buffer_extents = {}
    if semantics.source_kind == "wrapper_extern_addmm_bias":
        abi.append({"name": semantics.bias_param, "kind": "pointer", "dtype": "float32"})
        buffer_extents[semantics.bias_param] = f"T.int64({int(np.prod(semantics.bias_shape))})"
    abi.extend(
        [
            {"name": semantics.a_param, "kind": "pointer", "dtype": semantics.a_dtype},
            {"name": semantics.b_param, "kind": "pointer", "dtype": semantics.b_dtype},
            {"name": semantics.c_param, "kind": "pointer", "dtype": semantics.output_dtype},
        ]
    )
    buffer_extents.update(
        {
            semantics.a_param: f"T.int64({semantics.m * semantics.k})",
            semantics.b_param: f"T.int64({b_storage_shape[0] * b_storage_shape[1]})",
            semantics.c_param: f"T.int64({semantics.m * semantics.n})",
        }
    )
    return TritonTVMMeta(
        kernel_name=semantics.kernel_name,
        signature={},
        constexprs={},
        grid=(semantics.m * semantics.n,),
        target="cuda",
        target_kind="cuda",
        contract="matmul_minimal",
        canonical_contract="matmul_minimal",
        requested_contract="matmul_minimal",
        emit="tirx",
        translator_version="m12_3_vit_e2e_runner",
        contract_version="matmul_minimal_m9_v1",
        target_policy_version="m12_3_vit_e2e_runner",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=semantics.m,
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=semantics.k,
        buffer_extents=buffer_extents,
        block_size=semantics.m,
        indexing_kind="rank2_wrapper_native_matmul",
        execution_kind="m12_3_vit_e2e_native_matmul",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="spatial_mn_reduce_k",
        layout_policy="rank2_row_major_or_transposed_weight_view",
        launch_policy_id=NATIVE_TIR_MATMUL_SCHEDULE_ID,
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=(
            f"m12_3_vit:{semantics.kernel_name}:{semantics.m}:"
            f"{semantics.n}:{semantics.k}:{semantics.source_kind}"
        ),
        matmul_source_kind=semantics.source_kind,
        matmul_m=semantics.m,
        matmul_n=semantics.n,
        matmul_k=semantics.k,
        matmul_contract_ok=decision.matmul_contract_ok,
        implementation_kind=decision.implementation_kind,
        schedule_id=decision.schedule_id,
        extern_symbol=EXTERN_ADDMM_BIAS_SYMBOL
        if semantics.source_kind == "wrapper_extern_addmm_bias"
        else EXTERN_GEMM_SYMBOL,
        unsupported_matmul_reason=decision.unsupported_matmul_reason,
    )


@contextlib.contextmanager
def _patched_provider_mix(
    torch,
    matmul_calls: list[NativeMatmulRuntimeCall],
    *,
    profile_config: NativeMatmulProfileConfig | None = None,
) -> Iterator[_NativeMatmulPatchState]:
    import torch._inductor.select_algorithm as select_algorithm  # pylint: disable=import-outside-toplevel

    state = _NativeMatmulPatchState(matmul_calls, torch, profile_config=profile_config)
    extern_kernels = select_algorithm.extern_kernels
    old_addmm = extern_kernels.addmm
    old_mm = extern_kernels.mm
    old_conv = extern_kernels.convolution
    sdpa_packet = torch.ops.aten._scaled_dot_product_efficient_attention
    old_sdpa_default = sdpa_packet.default
    extern_kernels.addmm = state.addmm
    extern_kernels.mm = state.mm
    extern_kernels.convolution = state.wrap_conv(old_conv)
    sdpa_packet.default = state.native_decomposed_sdpa
    try:
        yield state
    finally:
        extern_kernels.addmm = old_addmm
        extern_kernels.mm = old_mm
        extern_kernels.convolution = old_conv
        sdpa_packet.default = old_sdpa_default


def _correctness_report(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    allclose = True
    max_abs = 0.0
    max_rel = 0.0
    for name in ("last_hidden_state", "pooler_output"):
        actual_np = actual[name].detach().cpu().numpy()
        expected_np = expected[name].detach().cpu().numpy()
        abs_error = np.abs(actual_np - expected_np)
        denom = np.maximum(np.abs(expected_np), 1e-12)
        rel_error = abs_error / denom
        output_allclose = bool(
            np.allclose(actual_np, expected_np, rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL)
        )
        outputs[name] = {
            "shape": list(actual_np.shape),
            "allclose": output_allclose,
            "max_abs_error": float(abs_error.max()),
            "max_rel_error": float(rel_error.max()),
        }
        allclose = allclose and output_allclose
        max_abs = max(max_abs, outputs[name]["max_abs_error"])
        max_rel = max(max_rel, outputs[name]["max_rel_error"])
    return {
        "baseline": "torch_eager_cuda_vit_tiny_random",
        "allclose": bool(allclose),
        "allclose_rtol": ALLCLOSE_RTOL,
        "allclose_atol": ALLCLOSE_ATOL,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "outputs": outputs,
    }


def _provider_mix_report(
    *,
    wrapper_source: str,
    matmul_calls: list[NativeMatmulRuntimeCall],
    patch_state: _NativeMatmulPatchState,
) -> dict[str, Any]:
    captured = extract_inductor_triton_sources(wrapper_source, case_name=TARGET_MODEL)
    executed_counts = Counter(record["op_family"] for record in patch_state.executed)
    provider_counts = {
        "torch_inductor_triton_captured_harness": len(captured),
        VISION_PROVIDER_DEVICE_TORCH_CUDA: patch_state.conv_runtime_calls,
        ATTENTION_PROVIDER_NATIVE_DECOMPOSED: patch_state.attention_runtime_calls,
        EXTERN_GEMM_PROVIDER_NATIVE_TVM: len(patch_state.executed),
    }
    complete = (
        len(captured) == 7
        and patch_state.conv_runtime_calls == 1
        and patch_state.attention_runtime_calls == 1
        and executed_counts.get("extern_gemm", 0) == 4
        and executed_counts.get("extern_addmm_bias", 0) == 3
    )
    return {
        "complete_provider_report": bool(complete),
        "no_silent_fallback": bool(complete),
        "captured_kernel_count": len(captured),
        "captured_kernel_execution_kind": "torch_inductor_triton_captured_harness",
        "captured_kernel_native_tvm_artifact_status": "translated_but_not_launched_by_m12_3_runner",
        "wrapper_call_count": 9,
        "provider_counts": provider_counts,
        "wrapper_convolution": {
            "count": patch_state.conv_runtime_calls,
            "provider_kind": VISION_PROVIDER_DEVICE_TORCH_CUDA,
            "runtime_status": VISION_RUNTIME_STATUS_RUNTIME_RESOLVED,
            "host_staging_bytes": 0,
        },
        "wrapper_attention": {
            "count": patch_state.attention_runtime_calls,
            "provider_kind": ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
            "runtime_status": ATTENTION_RUNTIME_STATUS_RUNTIME_RESOLVED,
            "host_staging_bytes": 0,
            "runtime_claim": "native_decomposed_correctness_only",
        },
        "wrapper_matmul": {
            "count": len(matmul_calls),
            "runtime_resolved_count": len(patch_state.executed),
            "extern_gemm": executed_counts.get("extern_gemm", 0),
            "extern_addmm_bias": executed_counts.get("extern_addmm_bias", 0),
            "provider_kind": EXTERN_GEMM_PROVIDER_NATIVE_TVM,
            "runtime_status": EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
            "runtime_claim": EXTERN_GEMM_RUNTIME_CLAIM_NATIVE_TVM_FIXED_SHAPE,
            "host_staging_bytes": 0,
            "performance_claim": False,
        },
    }


def _empty_report(*, seed: int, status: str, availability_reason: str) -> dict[str, Any]:
    return {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "availability_reason": availability_reason,
        "correctness": {
            "baseline": "torch_eager_cuda_vit_tiny_random",
            "allclose": None,
            "allclose_rtol": ALLCLOSE_RTOL,
            "allclose_atol": ALLCLOSE_ATOL,
        },
        "p0": {
            "correctness": False,
            "no_silent_fallback": False,
            "complete_provider_report": False,
            "passed": False,
        },
        "provider_mix": {},
        "native_matmul_records": [],
        "performance_ready_e2e": False,
        "performance_claim": False,
        "full_tvm_runnable_models": 0,
        "host_staging_bytes": 0,
    }


def _maybe_write_report(report: dict[str, Any], out_dir: str | Path | None) -> None:
    if out_dir is None:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "report.md").write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    correctness = report.get("correctness", {})
    p0 = report.get("p0", {})
    provider_mix = report.get("provider_mix", {})
    lines = [
        "# M12.3 ViT Fixed-Shape E2E Runner",
        "",
        f"- Status: {report.get('status')}",
        f"- Target model: `{report.get('target_model')}`",
        f"- Correctness allclose: {correctness.get('allclose')}",
        f"- Max abs error: {correctness.get('max_abs_error')}",
        f"- P0 passed: {p0.get('passed')}",
        f"- Performance-ready E2E: {report.get('performance_ready_e2e')}",
        f"- Full TVM runnable models: {report.get('full_tvm_runnable_models')}",
        f"- Host staging bytes: {report.get('host_staging_bytes')}",
        "",
        "## Provider Mix",
        "",
    ]
    counts = provider_mix.get("provider_counts", {})
    if counts:
        for provider, count in counts.items():
            lines.append(f"- `{provider}`: {count}")
    else:
        lines.append(f"- unavailable: {report.get('availability_reason', '')}")
    records = report.get("native_matmul_records", [])
    if records:
        lines.extend(
            [
                "",
                "## Native Matmul Records",
                "",
                "| Family | Line | Shape | Provider | Status | Perf claim |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for record in records:
            shape = f"{record['matmul_m']}x{record['matmul_n']}x{record['matmul_k']}"
            lines.append(
                f"| {record['op_family']} | {record['line_no']} | {shape} | "
                f"{record['provider_kind']} | {record['runtime_status']} | "
                f"{record['performance_claim']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _dependency_versions(torch) -> dict[str, Any]:
    versions = {
        "torch": str(getattr(torch, "__version__", "")),
        "tvm": str(tvm.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        versions["cuda_device"] = torch.cuda.get_device_name(0)
    return versions


def _require_torch():
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as err:  # pragma: no cover - reflected in unavailable reports.
        raise RuntimeError("M12.3 ViT E2E runner requires PyTorch") from err
    return torch


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
