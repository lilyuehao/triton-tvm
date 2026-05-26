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
"""M10 attention performance baseline runner.

This report is intentionally separate from the M10 corpus runtime fields.  The
native-decomposed provider remains correctness-only in corpus records; this
runner records same-machine timing baselines for the currently admitted
attention contracts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tvm
from tvm.contrib.triton_tvm import build_triton_tvm
from tvm.contrib.triton_tvm.attention import (
    ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
    ATTENTION_CONTRACT_VIT_FULL,
    ATTENTION_EXTERN_SYMBOL,
    ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
    ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
    TargetAttentionPolicy,
    build_native_decomposed_attention_tirx_source,
    extract_attention_semantics_from_wrapper_sdpa,
)
from tvm.contrib.triton_tvm.contracts import (
    validate_attention_llama_causal_prefill_contract,
    validate_attention_vit_full_contract,
)
from tvm.contrib.triton_tvm.translator import TritonTVMMeta

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - availability is reflected in reports.
    torch = None
    torch_functional = None


REPORT_KIND = "triton_tvm_m10_attention_performance_baseline"
BASELINE_ID = "m10_attention_native_decomposed_vs_torch_sdpa_v1"
TORCH_SDPA_BASELINE_ID = "torch_cuda_sdpa_framework_default_v1"
TRITON_TVM_ATTENTION_BASELINE_ID = "triton_tvm_native_decomposed_attention_v1"
DEFAULT_OUT_DIR = Path("/home/liyh/xdb/triton-tvm-workbench/reports/m10/m10_attention_baseline")
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 20


@dataclass(frozen=True)
class AttentionBaselineCase:
    """A fixed shape measured by the M10 attention baseline."""

    name: str
    contract: str
    batch: int
    heads: int
    sequence: int
    head_dim: int
    scale: float
    mask_kind: str
    notes: str = ""

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.heads, self.sequence, self.head_dim)

    @property
    def qkv_stride(self) -> tuple[int, int, int, int]:
        return (
            self.heads * self.sequence * self.head_dim,
            self.sequence * self.head_dim,
            self.head_dim,
            1,
        )

    @property
    def mask_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.heads, self.sequence, self.sequence)

    @property
    def mask_stride(self) -> tuple[int, int, int, int]:
        return (
            self.heads * self.sequence * self.sequence,
            self.sequence * self.sequence,
            self.sequence,
            1,
        )


def default_attention_baseline_suite() -> tuple[AttentionBaselineCase, ...]:
    """Return the fixed M10 attention baseline cases."""
    return (
        AttentionBaselineCase(
            name="vit_full_contiguous_1x4x5x16",
            contract=ATTENTION_CONTRACT_VIT_FULL,
            batch=1,
            heads=4,
            sequence=5,
            head_dim=16,
            scale=0.125,
            mask_kind="none_or_padding",
            notes="M10.4 native-decomposed ViT full-attention correctness shape",
        ),
        AttentionBaselineCase(
            name="llama_prefill_contiguous_1x4x16x16",
            contract=ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL,
            batch=1,
            heads=4,
            sequence=16,
            head_dim=16,
            scale=0.25,
            mask_kind="causal",
            notes="M10.5 native-decomposed Llama causal prefill correctness shape",
        ),
    )


def run_attention_baseline(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    run_benchmarks: bool = True,
) -> dict[str, Any]:
    """Run the M10 attention baseline and optionally write JSON/Markdown reports."""
    cases = [
        _run_case(case, warmup=warmup, repeat=repeat, run_benchmarks=run_benchmarks)
        for case in default_attention_baseline_suite()
    ]
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "baseline_kind": "diagnostic_attention_performance_baseline_v1",
        "baseline_performance_claim": True,
        "attention_runtime_provider_performance_claim": False,
        "claim_scope": (
            "same-machine timing baseline for M10 native_decomposed attention contracts; "
            "not a full-model runnable or provider performance claim"
        ),
        "baseline_ids": {
            "torch": TORCH_SDPA_BASELINE_ID,
            "triton_tvm": TRITON_TVM_ATTENTION_BASELINE_ID,
        },
        "provider_boundary": {
            "provider": ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
            "runtime_provider_claim": "correctness_only",
            "attention_performance_claim_in_corpus": False,
            "full_model_runnable_claim": False,
        },
        "warmup": warmup,
        "repeat": repeat,
        "cases": cases,
        "dependency_versions": _dependency_versions(),
    }
    report["summary"] = _summary(cases)
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_path / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _run_case(
    case: AttentionBaselineCase,
    *,
    warmup: int,
    repeat: int,
    run_benchmarks: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        **asdict(case),
        "shape": list(case.shape),
        "qkv_stride": list(case.qkv_stride),
        "mask_shape": list(case.mask_shape) if case.mask_kind == "causal" else [],
        "mask_stride": list(case.mask_stride) if case.mask_kind == "causal" else [],
        "baseline_torch": {
            "baseline_id": TORCH_SDPA_BASELINE_ID,
            "backend": "torch.nn.functional.scaled_dot_product_attention",
            "backend_policy": "framework_default",
            "performance_claim": True,
        },
        "baseline_triton_tvm": {
            "baseline_id": TRITON_TVM_ATTENTION_BASELINE_ID,
            "provider": ATTENTION_PROVIDER_NATIVE_DECOMPOSED,
            "runtime_provider_claim": "correctness_only",
            "performance_claim": True,
        },
        "torch_sdpa_us": None,
        "torch_sdpa_p50_us": None,
        "torch_sdpa_p95_us": None,
        "triton_tvm_us": None,
        "triton_tvm_p50_us": None,
        "triton_tvm_p95_us": None,
        "relative_to_torch": None,
        "qk_av_gflops": None,
        "perf_guard_status": "unavailable",
        "correctness_allclose": None,
        "max_abs_error": None,
        "availability_reason": "",
    }
    if not run_benchmarks:
        record["availability_reason"] = "benchmarks_disabled"
        return record
    if torch is None or torch_functional is None:
        record["availability_reason"] = "torch_unavailable"
        return record
    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        record["availability_reason"] = "cuda_unavailable"
        return record

    try:
        return _benchmark_case(record, case, warmup=warmup, repeat=repeat)
    except Exception as err:  # pylint: disable=broad-except
        record["availability_reason"] = f"benchmark_failed:{type(err).__name__}:{err}"
        return record


def _benchmark_case(
    record: dict[str, Any],
    case: AttentionBaselineCase,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(_case_seed(case.name))
    q_np = rng.normal(size=case.shape).astype("float32")
    k_np = rng.normal(size=case.shape).astype("float32")
    v_np = rng.normal(size=case.shape).astype("float32")
    mask_np = _causal_mask_np(case) if case.mask_kind == "causal" else None

    torch_q = torch.tensor(q_np, device="cuda")
    torch_k = torch.tensor(k_np, device="cuda")
    torch_v = torch.tensor(v_np, device="cuda")
    torch_mask = torch.tensor(mask_np, device="cuda") if mask_np is not None else None
    expected = torch_functional.scaled_dot_product_attention(
        torch_q,
        torch_k,
        torch_v,
        attn_mask=torch_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=case.scale,
    )

    semantics = extract_attention_semantics_from_wrapper_sdpa(
        _wrapper_source(case),
        case_name=case.name,
        kernel_name=f"{case.name}_m10_attention_baseline",
    )
    decision = TargetAttentionPolicy(
        attention_runtime_provider=ATTENTION_PROVIDER_NATIVE_DECOMPOSED
    ).decide(semantics, attention_contract_ok=True)
    irmod = tvm.script.from_source(build_native_decomposed_attention_tirx_source(semantics, decision))
    if case.contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        validate_attention_llama_causal_prefill_contract(irmod)
    else:
        validate_attention_vit_full_contract(irmod)
    built = build_triton_tvm(irmod, _attention_meta(case, semantics.kernel_name))

    dev = tvm.cuda(0)
    out_tvm = tvm.runtime.empty(case.shape, "float32", dev)
    tvm_args: list[Any] = [
        tvm.runtime.tensor(q_np, dev),
        tvm.runtime.tensor(k_np, dev),
        tvm.runtime.tensor(v_np, dev),
    ]
    if mask_np is not None:
        tvm_args.append(tvm.runtime.tensor(mask_np, dev))
    tvm_args.append(out_tvm)
    built.run(tvm_args)

    tvm_out = out_tvm.numpy()
    expected_np = expected.detach().cpu().numpy()
    max_abs_error = float(np.max(np.abs(tvm_out - expected_np)))
    correctness_allclose = bool(np.allclose(tvm_out, expected_np, rtol=1e-5, atol=1e-5))

    for _ in range(warmup):
        built.run(tvm_args)
    tvm_timer = built.executable.mod.time_evaluator(
        semantics.kernel_name,
        dev,
        number=1,
        repeat=repeat,
        min_repeat_ms=0,
    )
    tvm_values = [float(value) * 1e6 for value in tvm_timer(*tvm_args).results]
    torch_values = _time_cuda_callable(
        lambda: torch_functional.scaled_dot_product_attention(
            torch_q,
            torch_k,
            torch_v,
            attn_mask=torch_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=case.scale,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    tvm_p50 = _percentile(tvm_values, 50)
    tvm_p95 = _percentile(tvm_values, 95)
    torch_p50 = _percentile(torch_values, 50)
    torch_p95 = _percentile(torch_values, 95)
    record.update(
        {
            "torch_sdpa_us": torch_p50,
            "torch_sdpa_p50_us": torch_p50,
            "torch_sdpa_p95_us": torch_p95,
            "triton_tvm_us": tvm_p50,
            "triton_tvm_p50_us": tvm_p50,
            "triton_tvm_p95_us": tvm_p95,
            "relative_to_torch": tvm_p50 / torch_p50 if torch_p50 else None,
            "qk_av_gflops": _qk_av_gflops(case, tvm_p50),
            "perf_guard_status": "measured",
            "correctness_allclose": correctness_allclose,
            "max_abs_error": max_abs_error,
            "availability_reason": "",
        }
    )
    return record


def _wrapper_source(case: AttentionBaselineCase) -> str:
    shape_text = ", ".join(str(value) for value in case.shape)
    stride_text = ", ".join(str(value) for value in case.qkv_stride)
    if case.contract == ATTENTION_CONTRACT_VIT_FULL:
        return (
            f"{ATTENTION_EXTERN_SYMBOL}("
            f"reinterpret_tensor(q, ({shape_text}), ({stride_text}), 0), "
            f"reinterpret_tensor(k, ({shape_text}), ({stride_text}), 0), "
            f"reinterpret_tensor(v, ({shape_text}), ({stride_text}), 0), "
            f"None, False, scale={case.scale:g})"
        )
    mask_shape_text = ", ".join(str(value) for value in case.mask_shape)
    mask_stride_text = ", ".join(str(value) for value in case.mask_stride)
    return (
        f"{ATTENTION_EXTERN_SYMBOL}("
        "q, k, "
        f"reinterpret_tensor(v, ({shape_text}), ({stride_text}), 0), "
        f"reinterpret_tensor(mask, ({mask_shape_text}), ({mask_stride_text}), 0), "
        f"False, scale={case.scale:g})"
    )


def _attention_meta(case: AttentionBaselineCase, kernel_name: str) -> TritonTVMMeta:
    abi = [
        {"name": "q", "kind": "pointer", "dtype": "float32"},
        {"name": "k", "kind": "pointer", "dtype": "float32"},
        {"name": "v", "kind": "pointer", "dtype": "float32"},
    ]
    buffer_extents = {
        "q": f"T.int64({_numel(case.shape)})",
        "k": f"T.int64({_numel(case.shape)})",
        "v": f"T.int64({_numel(case.shape)})",
        "out": f"T.int64({_numel(case.shape)})",
    }
    contract_version = "attention_vit_full_m10_v1"
    execution_kind = "native_decomposed_attention_qk_softmax_av"
    mask_policy = "none_or_padding"
    layout_policy = "rank4_static_attention_baseline"
    if case.contract == ATTENTION_CONTRACT_LLAMA_CAUSAL_PREFILL:
        abi.append({"name": "mask", "kind": "pointer", "dtype": "float32"})
        buffer_extents["mask"] = f"T.int64({_numel(case.mask_shape)})"
        contract_version = "attention_llama_causal_prefill_m10_v1"
        execution_kind = "native_decomposed_attention_qk_masked_softmax_av"
        mask_policy = "causal_additive_mask"
        layout_policy = "rank4_static_prefill_attention_baseline"
    abi.append({"name": "out", "kind": "pointer", "dtype": "float32"})
    return TritonTVMMeta(
        kernel_name=kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract=case.contract,
        canonical_contract=case.contract,
        requested_contract=case.contract,
        emit="tir",
        translator_version="m10_attention_performance_baseline",
        contract_version=contract_version,
        target_policy_version="m10_attention_performance_baseline",
        target_attrs="cuda",
        triton_version="",
        tvm_version=tvm.__version__,
        ttir_hash="",
        source_hash="",
        extent_param="",
        extent_kind="constant",
        extent_value=case.sequence,
        reduction_extent_param="",
        reduction_extent_kind="constant",
        reduction_extent_value=case.head_dim,
        buffer_extents=buffer_extents,
        block_size=1,
        indexing_kind="rank4_sdpa_attention_baseline",
        execution_kind=execution_kind,
        accumulator_dtype_policy="fp32_attention_baseline",
        epsilon_policy="not_applicable",
        mask_policy=mask_policy,
        axis_policy="batch_head_sequence_head_dim",
        layout_policy=layout_policy,
        launch_policy_id="native_decomposed_attention_baseline",
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=f"{kernel_name}_attention_baseline",
        implementation_kind=ATTENTION_IMPLEMENTATION_KIND_NATIVE_DECOMPOSED,
        extern_symbol=ATTENTION_EXTERN_SYMBOL,
    )


def _causal_mask_np(case: AttentionBaselineCase) -> np.ndarray:
    causal = np.triu(np.ones((case.sequence, case.sequence), dtype=bool), k=1)
    mask = np.where(causal, -1.0e9, 0.0).astype("float32")
    return np.broadcast_to(mask, case.mask_shape).copy()


def _time_cuda_callable(fn, *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _qk_av_gflops(case: AttentionBaselineCase, us: float) -> float:
    if us <= 0:
        return 0.0
    ops = 4.0 * case.batch * case.heads * case.sequence * case.sequence * case.head_dim
    return ops / (us * 1e3)


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [case for case in cases if case.get("perf_guard_status") == "measured"]
    unavailable = [case for case in cases if case.get("perf_guard_status") == "unavailable"]
    return {
        "total_cases": len(cases),
        "measured_cases": len(measured),
        "unavailable_cases": len(unavailable),
        "allclose_cases": sum(1 for case in measured if case.get("correctness_allclose") is True),
        "contracts": {
            contract: sum(1 for case in cases if case.get("contract") == contract)
            for contract in sorted({str(case.get("contract")) for case in cases})
        },
    }


def _dependency_versions() -> dict[str, Any]:
    info: dict[str, Any] = {
        "tvm": tvm.__version__,
        "torch": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device": "",
    }
    if torch is not None and torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
    return info


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M10 Attention Performance Baseline",
        "",
        f"- Report kind: `{report['report_kind']}`",
        f"- Baseline id: `{report['baseline_id']}`",
        f"- Measured cases: {report['summary']['measured_cases']} / {report['summary']['total_cases']}",
        f"- Allclose cases: {report['summary']['allclose_cases']}",
        f"- Baseline performance claim: {report['baseline_performance_claim']}",
        f"- Runtime provider performance claim: {report['attention_runtime_provider_performance_claim']}",
        "",
        "## Cases",
        "",
        "| Case | Contract | Shape | Mask | Status | TVM us | Torch SDPA us | Relative | QK/AV GFLOP/s |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["name"]),
                    str(case["contract"]),
                    "x".join(str(v) for v in case["shape"]),
                    str(case["mask_kind"]),
                    str(case["perf_guard_status"]),
                    _fmt(case.get("triton_tvm_us")),
                    _fmt(case.get("torch_sdpa_us")),
                    _fmt(case.get("relative_to_torch")),
                    _fmt(case.get("qk_av_gflops")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a same-machine attention baseline report, not a full-model runnable claim.",
            "- M10 corpus `attention_performance_claim` remains false for the native provider.",
            "- RoPE runtime, KV-cache runtime, and decode runtime are not measured here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _case_seed(name: str) -> int:
    return sum(ord(ch) for ch in name) % (2**31)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    report = run_attention_baseline(
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        run_benchmarks=not args.no_benchmarks,
    )
    summary = report["summary"]
    print(
        "M10 attention baseline: "
        f"measured={summary['measured_cases']}/{summary['total_cases']} "
        f"allclose={summary['allclose_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
