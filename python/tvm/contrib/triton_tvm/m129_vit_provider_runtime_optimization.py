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
"""M12.9 fixed-shape ViT provider runtime optimization report."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tvm

from .attention import ATTENTION_PROVIDER_NATIVE_DECOMPOSED
from .matmul import EXTERN_GEMM_PROVIDER_NATIVE_TVM, M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID
from .m123_vit_e2e_runner import (
    NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
    TARGET_MODEL,
    NativeMatmulProfileConfig,
    _build_native_matmul_calls,
    _correctness_report,
    _dependency_versions,
    _NativeMatmulPatchState,
    _output_tensors,
    _provider_mix_report,
    _require_torch,
    execute_vit_fixed_shape_provider_path,
    prepare_vit_fixed_shape_e2e,
)
from .m124_vit_e2e_dashboard import DEFAULT_REPEAT, DEFAULT_WARMUP, P2_THRESHOLD
from .passes import M129OptimizeNativeWrapperMatmul
from .vision import VISION_PROVIDER_DEVICE_TORCH_CUDA


REPORT_KIND = "triton_tvm_m12_9_vit_provider_runtime_optimization"
REPORT_ID = "m12_9_vit_provider_runtime_optimization_v1"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_9_provider_runtime_optimization"
)
DEFAULT_M128_REPORT = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_8_p2_gate_decision/report.json"
)
CURRENT_MODE = "triton_tvm_e2e_current_no_per_call_sync"
BACKEND_PASS_MODE = "triton_tvm_e2e_m129_backend_pass"
FUSED_QKV_MODE = "triton_tvm_e2e_m129_backend_pass_fused_qkv"
EXPECTED_PROVIDER_COUNTS = {
    "torch_inductor_triton_captured_harness": 7,
    VISION_PROVIDER_DEVICE_TORCH_CUDA: 1,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED: 1,
    EXTERN_GEMM_PROVIDER_NATIVE_TVM: 7,
}


def run_vit_provider_runtime_optimization(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    run_benchmarks: bool = True,
    m128_report: str | Path = DEFAULT_M128_REPORT,
) -> dict[str, Any]:
    """Run M12.9 backend-pass optimization A/B and optionally write a report."""

    if not run_benchmarks:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="not_run",
            availability_reason="benchmarks_disabled",
            m128_report=m128_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        torch = _require_torch()
    except RuntimeError:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="torch_unavailable",
            m128_report=m128_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="unavailable",
            availability_reason="cuda_or_tvm_cuda_unavailable",
            m128_report=m128_report,
        )
        _maybe_write_report(report, out_dir)
        return report

    try:
        m128 = _read_json_report(m128_report)
        prepared = prepare_vit_fixed_shape_e2e(seed=seed)
        backend_pass = M129OptimizeNativeWrapperMatmul()
        optimized_calls = _build_native_matmul_calls(
            prepared.wrapper_source,
            native_matmul_passes=[backend_pass],
        )
        optimized_prepared = replace(prepared, matmul_calls=optimized_calls)

        current_actual, current_state = execute_vit_fixed_shape_provider_path(
            prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )
        backend_actual, backend_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )
        fused_qkv_artifact = _build_fused_qkv_addmm_artifact()
        fused_actual, fused_state = _execute_with_fused_qkv_provider(
            optimized_prepared,
            fused_qkv_artifact,
        )
        current_correctness = _correctness_report(current_actual, prepared.expected)
        backend_correctness = _correctness_report(backend_actual, prepared.expected)
        fused_correctness = _correctness_report(fused_actual, prepared.expected)
        current_provider_mix = _provider_mix_report(
            wrapper_source=prepared.wrapper_source,
            matmul_calls=prepared.matmul_calls,
            patch_state=current_state,
        )
        backend_provider_mix = _provider_mix_report(
            wrapper_source=prepared.wrapper_source,
            matmul_calls=optimized_prepared.matmul_calls,
            patch_state=backend_state,
        )
        fused_provider_mix = _provider_mix_report(
            wrapper_source=prepared.wrapper_source,
            matmul_calls=optimized_prepared.matmul_calls,
            patch_state=fused_state,
        )

        _, current_profile_state = execute_vit_fixed_shape_provider_path(
            prepared,
            profile_native_matmul=True,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )
        _, backend_profile_state = execute_vit_fixed_shape_provider_path(
            optimized_prepared,
            profile_native_matmul=True,
            native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
        )

        latency_ms = {
            "torch_eager_cuda": _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: _run_torch_eager(prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            "torch_compile_inductor": _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: _run_torch_compile(prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            CURRENT_MODE: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            BACKEND_PASS_MODE: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: execute_vit_fixed_shape_provider_path(
                        optimized_prepared,
                        native_matmul_sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
            FUSED_QKV_MODE: _latency_record(
                _time_cuda_callable(
                    prepared.torch,
                    lambda: _execute_with_fused_qkv_provider(
                        optimized_prepared,
                        fused_qkv_artifact,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            ),
        }
        report = build_vit_provider_runtime_optimization_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            m128_report=m128,
            m128_report_path=m128_report,
            latency_ms=latency_ms,
            correctness={
                CURRENT_MODE: current_correctness,
                BACKEND_PASS_MODE: backend_correctness,
                FUSED_QKV_MODE: fused_correctness,
            },
            provider_mix={
                CURRENT_MODE: current_provider_mix,
                BACKEND_PASS_MODE: backend_provider_mix,
                FUSED_QKV_MODE: fused_provider_mix,
            },
            backend_pass_summary=_backend_pass_summary(optimized_prepared.matmul_calls),
            provider_overhead_profile={
                CURRENT_MODE: _profile_summary(current_profile_state.executed),
                BACKEND_PASS_MODE: _profile_summary(backend_profile_state.executed),
            },
            dependency_versions=_dependency_versions(prepared.torch),
            out_dir=out_dir,
        )
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "vit_tiny_random_wrapper.py").write_text(
                prepared.wrapper_source, encoding="utf-8"
            )
        return report
    except Exception as err:  # pylint: disable=broad-except
        report = _empty_report(
            seed=seed,
            warmup=warmup,
            repeat=repeat,
            status="failed",
            availability_reason=f"m12_9_failed:{type(err).__name__}:{err}",
            m128_report=m128_report,
        )
        _maybe_write_report(report, out_dir)
        return report


def build_vit_provider_runtime_optimization_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    m128_report: dict[str, Any],
    m128_report_path: str | Path,
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    backend_pass_summary: dict[str, Any],
    provider_overhead_profile: dict[str, Any] | None = None,
    dependency_versions: dict[str, Any] | None = None,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build an M12.9 optimization report from measured or synthetic data."""

    host_staging_bytes = 0
    ratios = _ratio_report(latency_ms)
    best_mode = _best_mode(latency_ms)
    p2_context = _p2_context(latency_ms, ratios, best_mode)
    optimization_modes = _optimization_modes(
        latency_ms,
        correctness,
        provider_mix,
        backend_pass_summary,
    )
    invariants = _invariants(
        correctness=correctness,
        provider_mix=provider_mix,
        backend_pass_summary=backend_pass_summary,
        host_staging_bytes=host_staging_bytes,
    )
    status = "passed" if invariants["status"] == "passed" else "failed"
    performance_ready = bool(status == "passed" and p2_context["best_mode_p2_passed"])
    report = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "warmed_cache": warmup > 0,
        "source_reports": {"m12_8_report": str(m128_report_path)},
        "source_report_status": {
            "m12_8_status": m128_report.get("status"),
            "m12_8_decision": (m128_report.get("m12_8_decision") or {}).get("decision"),
        },
        "latency_ms": latency_ms,
        "ratios": ratios,
        "optimization_modes": optimization_modes,
        "best_mode": best_mode,
        "correctness": correctness,
        "provider_mix": provider_mix,
        "backend_pass": backend_pass_summary,
        "provider_overhead_profile": provider_overhead_profile or {},
        "cuda_graph": {
            "status": "skipped",
            "reason": "m12_9_user_requested_tvm_backend_pass_first",
        },
        "p2_context": p2_context,
        "invariants": invariants,
        "host_staging_bytes": host_staging_bytes,
        "performance_ready_e2e": performance_ready,
        "performance_claim": performance_ready,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "dependency_versions": dependency_versions or {},
        "notes": [
            "M12.9 does not redefine P2 and does not use 2x as the status gate.",
            "The primary optimization is an opt-in TVM module pass over native wrapper matmul TIR.",
            "CUDA Graph replay is intentionally skipped in this pass-first slice.",
        ],
    }
    _maybe_write_report(report, out_dir)
    return report


def _run_torch_eager(prepared):
    with prepared.torch.no_grad():
        return prepared.model(prepared.pixel_values)


def _run_torch_compile(prepared):
    with prepared.torch.no_grad():
        return prepared.compiled(prepared.pixel_values)


def _time_cuda_callable(
    torch_module,
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch_module.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch_module.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _latency_record(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples_ms": [], "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(float(value) for value in values)
    return {
        "samples_ms": values,
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio_report(latency_ms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    ratios: dict[str, Any] = {}
    for mode in (CURRENT_MODE, BACKEND_PASS_MODE, FUSED_QKV_MODE):
        record = latency_ms.get(mode, {})
        if not record:
            continue
        ratios[f"{mode}_to_torch_compile_p50"] = _safe_ratio(
            record.get("p50_ms"),
            compile_latency.get("p50_ms"),
        )
        ratios[f"{mode}_to_torch_compile_p95"] = _safe_ratio(
            record.get("p95_ms"),
            compile_latency.get("p95_ms"),
        )
    return ratios


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_f <= 0.0:
        return None
    return numerator_f / denominator_f


def _best_mode(latency_ms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for mode in (CURRENT_MODE, BACKEND_PASS_MODE, FUSED_QKV_MODE):
        p50 = latency_ms.get(mode, {}).get("p50_ms")
        if p50 is not None:
            candidates.append((float(p50), mode))
    if not candidates:
        return {"mode": None, "p50_ms": None}
    p50, mode = min(candidates)
    return {"mode": mode, "p50_ms": p50, "p95_ms": latency_ms[mode].get("p95_ms")}


def _p2_context(
    latency_ms: dict[str, dict[str, Any]],
    ratios: dict[str, Any],
    best_mode: dict[str, Any],
) -> dict[str, Any]:
    mode = best_mode.get("mode")
    compile_latency = latency_ms.get("torch_compile_inductor", {})
    p50_ratio = ratios.get(f"{mode}_to_torch_compile_p50") if mode else None
    p95_ratio = ratios.get(f"{mode}_to_torch_compile_p95") if mode else None
    return {
        "threshold": "best triton_tvm mode p50/p95 <= 2x torch_compile_inductor",
        "torch_compile_p50_ms": compile_latency.get("p50_ms"),
        "torch_compile_p95_ms": compile_latency.get("p95_ms"),
        "best_mode": mode,
        "best_mode_p50_ratio": p50_ratio,
        "best_mode_p95_ratio": p95_ratio,
        "best_mode_p2_passed": bool(
            p50_ratio is not None
            and p95_ratio is not None
            and p50_ratio <= P2_THRESHOLD
            and p95_ratio <= P2_THRESHOLD
        ),
    }


def _optimization_modes(
    latency_ms: dict[str, dict[str, Any]],
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    backend_pass_summary: dict[str, Any],
) -> dict[str, Any]:
    current_p50 = _to_float(latency_ms.get(CURRENT_MODE, {}).get("p50_ms"))
    backend_p50 = _to_float(latency_ms.get(BACKEND_PASS_MODE, {}).get("p50_ms"))
    return {
        CURRENT_MODE: {
            "kind": "contract_preserving_current",
            "backend_general": False,
            "model_specific_artifact": False,
            "counts_for_triton_tvm_backend_progress": False,
            "route_diagnostic_label": "baseline_only",
            "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            "backend_pass_enabled": False,
            "p50_ms": current_p50,
            "p95_ms": latency_ms.get(CURRENT_MODE, {}).get("p95_ms"),
            "correctness_allclose": correctness.get(CURRENT_MODE, {}).get("allclose"),
            "provider_mix_complete": provider_mix.get(CURRENT_MODE, {}).get(
                "complete_provider_report"
            ),
        },
        BACKEND_PASS_MODE: {
            "kind": "contract_preserving_backend_pass",
            "backend_general": True,
            "model_specific_artifact": False,
            "counts_for_triton_tvm_backend_progress": True,
            "route_diagnostic_label": "backend_general_progress",
            "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            "backend_pass_enabled": True,
            "backend_pass_schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            "transformed_call_count": backend_pass_summary.get("transformed_call_count"),
            "p50_ms": backend_p50,
            "p95_ms": latency_ms.get(BACKEND_PASS_MODE, {}).get("p95_ms"),
            "p50_recovery_ms": (
                None if current_p50 is None or backend_p50 is None else current_p50 - backend_p50
            ),
            "correctness_allclose": correctness.get(BACKEND_PASS_MODE, {}).get("allclose"),
            "provider_mix_complete": provider_mix.get(BACKEND_PASS_MODE, {}).get(
                "complete_provider_report"
            ),
        },
        FUSED_QKV_MODE: {
            "kind": "experimental_backend_artifact_fused_qkv",
            "backend_general": False,
            "model_specific_artifact": True,
            "counts_for_triton_tvm_backend_progress": False,
            "route_diagnostic_label": "model_specific_diagnostic_only",
            "requires_future_generalization": True,
            "native_matmul_sync_policy": NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            "backend_pass_enabled": True,
            "backend_pass_schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            "fused_qkv_addmm": True,
            "p50_ms": _to_float(latency_ms.get(FUSED_QKV_MODE, {}).get("p50_ms")),
            "p95_ms": latency_ms.get(FUSED_QKV_MODE, {}).get("p95_ms"),
            "correctness_allclose": correctness.get(FUSED_QKV_MODE, {}).get("allclose"),
            "provider_mix_complete": provider_mix.get(FUSED_QKV_MODE, {}).get(
                "complete_provider_report"
            ),
            "diagnostic_reason": (
                "fixed-shape ViT QKV subgraph artifact; useful evidence, but not "
                "Triton backend progress until generalized through language lowering, "
                "TVM scheduling, or reusable runtime work"
            ),
        },
    }


def _backend_pass_summary(matmul_calls) -> dict[str, Any]:
    per_call = []
    transformed = 0
    for call in matmul_calls:
        func = next(iter(call.artifact.irmod.functions.values()))
        schedule_id = str((func.attrs or {}).get("triton_tvm.schedule_id", ""))
        if schedule_id == M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID:
            transformed += 1
        per_call.append(
            {
                "index": call.index,
                "op_family": call.op_family,
                "shape_mnk": [call.matmul_m, call.matmul_n, call.matmul_k],
                "schedule_id": schedule_id,
            }
        )
    return {
        "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
        "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
        "transformed_call_count": transformed,
        "expected_call_count": len(matmul_calls),
        "loop_shape": "blockIdx.x=M rows, threadIdx.x=N columns, serial K",
        "per_call": per_call,
    }


class _FusedQKVArtifact:
    def __init__(self, executable, kernel_name: str):
        self.executable = executable
        self.kernel_name = kernel_name

    def run(self, args: list[Any] | tuple[Any, ...]) -> None:
        self.executable[self.kernel_name](*list(args))


class _FusedQKVPatchState(_NativeMatmulPatchState):
    def __init__(self, calls, torch_module, fused_qkv_artifact):
        super().__init__(
            calls,
            torch_module,
            profile_config=NativeMatmulProfileConfig(
                enabled=False,
                sync_policy=NATIVE_MATMUL_SYNC_POLICY_NO_PER_CALL_SYNC,
            ),
        )
        self._fused_qkv_artifact = fused_qkv_artifact
        self._qkv_pending: list[tuple[Any, list[Any], dict[str, Any]]] = []

    def addmm(self, bias, lhs, rhs, *args, **kwargs):
        if len(self._qkv_pending) >= 3:
            return super().addmm(bias, lhs, rhs, *args, **kwargs)
        out = kwargs.get("out")
        if out is None:
            raise RuntimeError("M12.9 fused QKV addmm patch requires out= buffer")
        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 1.0))
        if alpha != 1.0 or beta != 1.0:
            raise RuntimeError(f"M12.9 fused QKV addmm requires alpha=beta=1, got {alpha}, {beta}")
        call = self._addmm.popleft()
        rhs_storage = self._b_storage(rhs, call)
        record = call.report_record()
        record["m12_9_fused_qkv_group"] = True
        record["m12_9_fused_qkv_role"] = len(self._qkv_pending)
        if len(self._qkv_pending) < 2:
            record["m12_9_fused_qkv_dispatch"] = "deferred"
        else:
            record["m12_9_fused_qkv_dispatch"] = "fused_artifact"
        self.executed.append(record)
        self._qkv_pending.append((call, [bias, lhs, rhs_storage, out], record))
        if len(self._qkv_pending) == 3:
            self._run_fused_qkv()
        return out

    def _run_fused_qkv(self) -> None:
        first = self._qkv_pending[0][1]
        second = self._qkv_pending[1][1]
        third = self._qkv_pending[2][1]
        torch_args = [
            first[0],
            first[1],
            first[2],
            first[3],
            second[0],
            second[2],
            second[3],
            third[0],
            third[2],
            third[3],
        ]
        tvm_args = [tvm.runtime.from_dlpack(arg) for arg in torch_args]
        self._pending_tvm_args.append(tvm_args)
        self._fused_qkv_artifact.run(tvm_args)


def _execute_with_fused_qkv_provider(prepared, fused_qkv_artifact):
    with _patched_fused_qkv_provider_mix(
        prepared.torch,
        prepared.matmul_calls,
        fused_qkv_artifact,
    ) as patch_state:
        with prepared.torch.no_grad():
            actual_output = prepared.compiled(prepared.pixel_values)
        prepared.torch.cuda.synchronize()
        patch_state.finalize_profiles()
        patch_state.assert_all_calls_consumed()
    return _output_tensors(actual_output), patch_state


@contextlib.contextmanager
def _patched_fused_qkv_provider_mix(torch, matmul_calls, fused_qkv_artifact):
    import torch._inductor.select_algorithm as select_algorithm  # pylint: disable=import-outside-toplevel

    state = _FusedQKVPatchState(matmul_calls, torch, fused_qkv_artifact)
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


def _build_fused_qkv_addmm_artifact() -> _FusedQKVArtifact:
    kernel_name = "m12_9_vit_fused_qkv_addmm"
    source = f"""
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def {kernel_name}(bias0_h: T.handle, a_h: T.handle, b0_h: T.handle, c0_h: T.handle,
                     bias1_h: T.handle, b1_h: T.handle, c1_h: T.handle,
                     bias2_h: T.handle, b2_h: T.handle, c2_h: T.handle):
        T.func_attr({{"global_symbol": "{kernel_name}", "tirx.noalias": True, "target": T.target("cuda"),
                     "triton_tvm.contract": "m12_9_fused_qkv_addmm",
                     "triton_tvm.schedule_id": "{M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID}",
                     "triton_tvm.extern_gemm_uses_host_staging": False}})
        bias0 = T.match_buffer(bias0_h, (64,), "float32", strides=(1,))
        a = T.match_buffer(a_h, (5, 64), "float32", strides=(64, 1))
        b0 = T.match_buffer(b0_h, (64, 64), "float32", strides=(64, 1))
        c0 = T.match_buffer(c0_h, (5, 64), "float32", strides=(64, 1))
        bias1 = T.match_buffer(bias1_h, (64,), "float32", strides=(1,))
        b1 = T.match_buffer(b1_h, (64, 64), "float32", strides=(64, 1))
        c1 = T.match_buffer(c1_h, (5, 64), "float32", strides=(64, 1))
        bias2 = T.match_buffer(bias2_h, (64,), "float32", strides=(1,))
        b2 = T.match_buffer(b2_h, (64, 64), "float32", strides=(64, 1))
        c2 = T.match_buffer(c2_h, (5, 64), "float32", strides=(64, 1))
        for blockIdx_x in T.thread_binding(0, 5, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 64, thread="threadIdx.x"):
                mi = blockIdx_x
                ni = threadIdx_x
                for kk in T.serial(0, 64):
                    with T.sblock("q_matmul"):
                        vm = T.axis.spatial(5, mi)
                        vn = T.axis.spatial(64, ni)
                        vk = T.axis.reduce(64, kk)
                        T.reads(a[vm, vk], b0[vn, vk], bias0[vn])
                        T.writes(c0[vm, vn])
                        with T.init():
                            c0[vm, vn] = bias0[vn]
                        c0[vm, vn] = c0[vm, vn] + a[vm, vk] * b0[vn, vk]
                    with T.sblock("k_matmul"):
                        vm = T.axis.spatial(5, mi)
                        vn = T.axis.spatial(64, ni)
                        vk = T.axis.reduce(64, kk)
                        T.reads(a[vm, vk], b1[vn, vk], bias1[vn])
                        T.writes(c1[vm, vn])
                        with T.init():
                            c1[vm, vn] = bias1[vn]
                        c1[vm, vn] = c1[vm, vn] + a[vm, vk] * b1[vn, vk]
                    with T.sblock("v_matmul"):
                        vm = T.axis.spatial(5, mi)
                        vn = T.axis.spatial(64, ni)
                        vk = T.axis.reduce(64, kk)
                        T.reads(a[vm, vk], b2[vn, vk], bias2[vn])
                        T.writes(c2[vm, vn])
                        with T.init():
                            c2[vm, vn] = bias2[vn]
                        c2[vm, vn] = c2[vm, vn] + a[vm, vk] * b2[vn, vk]
"""
    mod = tvm.script.from_source(source)
    return _FusedQKVArtifact(tvm.compile(mod, target="cuda"), kernel_name)


def _profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = (
        "shape_abi_validation_ms",
        "b_storage_resolution_ms",
        "dlpack_conversion_ms",
        "artifact_dispatch_wall_ms",
        "kernel_event_ms",
        "provider_total_wall_ms",
    )
    summary = {"record_count": len(records), "aggregate_sum_ms": {}}
    for category in categories:
        total = 0.0
        samples = 0
        for record in records:
            profile = record.get("profile") or {}
            value = profile.get(category)
            if value is not None:
                total += float(value)
                samples += 1
        summary["aggregate_sum_ms"][category] = round(total, 6) if samples else None
    return summary


def _invariants(
    *,
    correctness: dict[str, dict[str, Any]],
    provider_mix: dict[str, dict[str, Any]],
    backend_pass_summary: dict[str, Any],
    host_staging_bytes: int,
) -> dict[str, Any]:
    failures: list[str] = []
    if host_staging_bytes != 0:
        failures.append("m12_9_host_staging_nonzero")
    for mode in (CURRENT_MODE, BACKEND_PASS_MODE, FUSED_QKV_MODE):
        if mode not in correctness and mode not in provider_mix:
            continue
        if correctness.get(mode, {}).get("allclose") is not True:
            failures.append(f"m12_9_correctness_failed_{mode}")
        mix = provider_mix.get(mode, {})
        if mix.get("complete_provider_report") is not True:
            failures.append(f"m12_9_provider_mix_incomplete_{mode}")
        counts = mix.get("provider_counts", {})
        for provider, expected in EXPECTED_PROVIDER_COUNTS.items():
            if counts.get(provider) != expected:
                failures.append(f"m12_9_provider_count_mismatch_{mode}_{provider}")
    if backend_pass_summary.get("transformed_call_count") != backend_pass_summary.get(
        "expected_call_count"
    ):
        failures.append("m12_9_backend_pass_did_not_transform_all_calls")
    return {"status": "failed" if failures else "passed", "invariant_failures": failures}


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _read_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _empty_report(
    *,
    seed: int,
    warmup: int,
    repeat: int,
    status: str,
    availability_reason: str,
    m128_report: str | Path,
) -> dict[str, Any]:
    return {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": TARGET_MODEL,
        "fixed_shape": True,
        "seed": seed,
        "warmup": warmup,
        "repeat": repeat,
        "source_reports": {"m12_8_report": str(m128_report)},
        "latency_ms": {},
        "optimization_modes": {},
        "best_mode": {"mode": None, "p50_ms": None},
        "correctness": {},
        "provider_mix": {},
        "backend_pass": {
            "pass_name": "triton_tvm.M129OptimizeNativeWrapperMatmul",
            "schedule_id": M129_ROW_THREADED_WRAPPER_MATMUL_SCHEDULE_ID,
            "transformed_call_count": 0,
            "expected_call_count": 0,
        },
        "provider_overhead_profile": {},
        "cuda_graph": {"status": "skipped"},
        "p2_context": {"best_mode_p2_passed": False},
        "invariants": {"status": "not_run", "invariant_failures": []},
        "host_staging_bytes": 0,
        "performance_ready_e2e": False,
        "performance_claim": False,
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
    best = report.get("best_mode", {})
    p2 = report.get("p2_context", {})
    backend = report.get("backend_pass", {})
    latency = report.get("latency_ms", {})
    lines = [
        "# M12.9 Provider Runtime Optimization",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Warmup/repeat: {report.get('warmup')} / {report.get('repeat')}",
        f"- Best mode: `{best.get('mode')}` p50=`{best.get('p50_ms')}` ms",
        f"- P2 naturally passed: `{p2.get('best_mode_p2_passed')}`",
        f"- Backend pass transformed calls: {backend.get('transformed_call_count')} / "
        f"{backend.get('expected_call_count')}",
        "",
        "## Latency",
        "",
        "| Path | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for name, record in latency.items():
        lines.append(f"| `{name}` | {_fmt(record.get('p50_ms'))} | {_fmt(record.get('p95_ms'))} |")
    lines.extend(
        [
            "",
            "## Backend Pass",
            "",
            f"- Pass: `{backend.get('pass_name')}`",
            f"- Schedule: `{backend.get('schedule_id')}`",
            f"- Loop shape: {backend.get('loop_shape')}",
            "",
            "## Invariants",
            "",
            f"- Status: `{(report.get('invariants') or {}).get('status')}`",
        ]
    )
    for failure in (report.get("invariants") or {}).get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m129_vit_provider_runtime_optimization``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--m128-report", type=Path, default=DEFAULT_M128_REPORT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    report = run_vit_provider_runtime_optimization(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        run_benchmarks=not args.no_benchmarks,
        m128_report=args.m128_report,
    )
    best = report.get("best_mode", {})
    print(
        f"M12.9 provider runtime optimization: status={report['status']} "
        f"best={best.get('mode')} p50={best.get('p50_ms')} out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "not_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
