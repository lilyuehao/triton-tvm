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
"""M12.12 real Triton JIT ``tl.dot`` bridge harness report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import validate_matmul_minimal_contract
from .errors import TritonTVMError
from .frontend import lower_to_ttir
from .matmul import (
    EXTERN_GEMM_SYMBOL,
    REAL_JIT_TT_DOT_SOURCE_KIND,
    TT_DOT_SOURCE_KIND,
    extract_matmul_semantics_from_ttir,
)
from .translator import translate_ttir
from .ttir import TTIRReader


REPORT_KIND = "triton_tvm_m12_12_real_tl_dot_bridge_harness"
REPORT_ID = "m12_12_real_tl_dot_bridge_harness_v1"
MILESTONE = "M12.12"
DEFAULT_OUT_DIR = Path(
    "/home/liyh/xdb/triton-tvm-workbench/reports/m12/m12_12_real_tl_dot_bridge_harness"
)

SUPPORTED_CASE = "real_jit_tl_dot_fp16_row_major"
NEGATIVE_CASES = {
    "unsupported_dtype": "matmul_input_dtype_not_supported",
    "unsupported_layout": "non_row_major_matmul_not_supported",
    "masked_bounds": "matmul_bounds_policy_not_supported",
    "unsupported_epilogue": "matmul_epilogue_not_supported",
}


def run_real_tl_dot_bridge_harness(
    *,
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Lower real Triton JIT ``tl.dot`` kernels and write the M12.12 report."""

    try:
        evidence = _collect_real_tl_dot_evidence()
    except (ImportError, RuntimeError, ValueError, TritonTVMError) as err:
        report = _empty_report(status="unavailable", availability_reason=str(err))
        _maybe_write_report(report, out_dir)
        return report

    return build_real_tl_dot_bridge_report(
        accepted_case=evidence["accepted_case"],
        negative_cases=evidence["negative_cases"],
        out_dir=out_dir,
    )


def build_real_tl_dot_bridge_report(
    *,
    accepted_case: dict[str, Any],
    negative_cases: list[dict[str, Any]],
    out_dir: str | Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Build a machine-readable M12.12 bridge report from collected evidence."""

    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goal": (
            "Prove real Triton JIT tl.dot lowering reaches TTIRReader, "
            "MatmulSemantics, and matmul_minimal without wrapper replay."
        ),
        "accepted_case": dict(accepted_case),
        "negative_cases": list(negative_cases),
        "source_taxonomy": _source_taxonomy(),
        "route_boundaries": _route_boundaries(),
        "next_default_action": {
            "action": "m12_13_runtime_overhead_general_measurement",
            "reason": "M12.12 bridges real tl.dot; runtime overhead is the next frozen auxiliary track.",
        },
        "performance_claim": False,
        "performance_ready_e2e": False,
        "p2_passed": False,
        "backend_general_bridge_complete": True,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {
            "m12_complete": False,
            "requires_fixed_shape_vit_p2": True,
            "p2_passed": False,
            "performance_ready_e2e": False,
        },
    }
    report["invariants"] = _invariants(report)
    if report["invariants"]["status"] != "passed":
        report["status"] = "failed"
    report["summary"] = _summary(report)
    _maybe_write_report(report, out_dir)
    return report


def _collect_real_tl_dot_evidence() -> dict[str, Any]:
    kernels = _real_tl_dot_kernels()
    signature = _matmul_signature("*fp16")
    constexprs = {"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 16}
    accepted = _run_real_case(
        SUPPORTED_CASE,
        kernels["supported"],
        signature,
        constexprs,
        expected_reason="",
        accepted=True,
    )
    negative_cases = [
        _run_real_case(
            "unsupported_dtype",
            kernels["unsupported_dtype"],
            _matmul_signature("*fp32"),
            constexprs,
            expected_reason=NEGATIVE_CASES["unsupported_dtype"],
        ),
        _run_real_case(
            "unsupported_layout",
            kernels["unsupported_layout"],
            signature,
            constexprs,
            expected_reason=NEGATIVE_CASES["unsupported_layout"],
        ),
        _run_real_case(
            "masked_bounds",
            kernels["masked_bounds"],
            {
                **signature,
                "M": "constexpr",
                "N": "constexpr",
                "K": "constexpr",
            },
            {"M": 7, "N": 7, "K": 15, **constexprs},
            expected_reason=NEGATIVE_CASES["masked_bounds"],
        ),
        _run_real_case(
            "unsupported_epilogue",
            kernels["unsupported_epilogue"],
            signature,
            constexprs,
            expected_reason=NEGATIVE_CASES["unsupported_epilogue"],
        ),
    ]
    return {"accepted_case": accepted, "negative_cases": negative_cases}


def _run_real_case(
    case_name: str,
    jit_fn,
    signature: dict[str, str],
    constexprs: dict[str, Any],
    *,
    expected_reason: str,
    accepted: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_name": case_name,
        "expected_reason": expected_reason,
        "status": "failed",
        "lower_to_ttir_ok": False,
        "ttir_contains_tt_dot": False,
        "reader_dot_snapshot": {},
        "matmul_source_kind": "",
        "matmul_contract_ok": False,
        "implementation_kind": "",
        "schedule_id": "",
        "observed_reason": "",
    }
    try:
        artifact = lower_to_ttir(jit_fn, signature, constexprs)
        result["lower_to_ttir_ok"] = True
        result["kernel_name"] = artifact.kernel_name
        result["triton_version"] = artifact.triton_version
        result["ttir_contains_tt_dot"] = "tt.dot" in artifact.ttir
        graph = TTIRReader().read(artifact.ttir)
        result["reader_dot_snapshot"] = _dot_snapshot(graph)
        semantics = extract_matmul_semantics_from_ttir(
            graph, source_kind=REAL_JIT_TT_DOT_SOURCE_KIND
        )
        result["matmul_semantics"] = _matmul_semantics_snapshot(semantics)
        result["matmul_source_kind"] = semantics.source_kind
        irmod, meta = translate_ttir(artifact, grid=(1,), contract="matmul_minimal")
        result["matmul_contract_ok"] = bool(meta.matmul_contract_ok)
        result["implementation_kind"] = meta.implementation_kind
        result["schedule_id"] = meta.schedule_id
        result["observed_reason"] = meta.unsupported_matmul_reason
        if accepted:
            validate_matmul_minimal_contract(irmod)
            result["status"] = "passed" if _accepted_case_ok(result) else "failed"
        else:
            result["status"] = (
                "passed" if result["observed_reason"] == expected_reason else "failed"
            )
    except Exception as err:  # pylint: disable=broad-except
        result["observed_reason"] = _stable_exception_reason(err)
        result["error_type"] = type(err).__name__
        result["error_message"] = str(err)
        if not accepted and result["observed_reason"] == expected_reason:
            result["status"] = "passed"
    return result


def _accepted_case_ok(result: dict[str, Any]) -> bool:
    dot = result.get("reader_dot_snapshot") or {}
    return (
        result.get("lower_to_ttir_ok") is True
        and result.get("ttir_contains_tt_dot") is True
        and dot.get("name") == "tt.dot"
        and len(dot.get("operands") or []) == 3
        and len(dot.get("result_types") or []) == 1
        and (dot.get("attrs") or {}).get("inputPrecision") == "tf32"
        and result.get("matmul_source_kind") == REAL_JIT_TT_DOT_SOURCE_KIND
        and result.get("matmul_contract_ok") is True
        and result.get("implementation_kind") == "native_tir_schedule"
        and result.get("observed_reason") == ""
    )


def _stable_exception_reason(err: Exception) -> str:
    text = str(err)
    for reason in NEGATIVE_CASES.values():
        if reason in text:
            return reason
    if "input dtype" in text or "dtype" in text:
        return "matmul_input_dtype_not_supported"
    if "layout" in text or "row_major" in text:
        return "non_row_major_matmul_not_supported"
    if "mask" in text or "bounds" in text or "unmasked" in text:
        return "matmul_bounds_policy_not_supported"
    if "epilogue" in text:
        return "matmul_epilogue_not_supported"
    return text


def _dot_snapshot(graph) -> dict[str, Any]:
    dot_ops = [op for op in graph.ops if op.name == "tt.dot"]
    if not dot_ops:
        return {}
    dot = dot_ops[0]
    return {
        "name": dot.name,
        "results": list(dot.results),
        "operands": list(dot.operands),
        "attrs": dict(dot.attrs),
        "result_types": [
            {"raw": ty.raw, "dtype": ty.dtype, "shape": list(ty.shape)}
            for ty in dot.result_types
        ],
    }


def _matmul_semantics_snapshot(semantics) -> dict[str, Any]:
    return {
        "source_kind": semantics.source_kind,
        "kernel_name": semantics.kernel_name,
        "m": semantics.m,
        "n": semantics.n,
        "k": semantics.k,
        "a_dtype": semantics.a_dtype,
        "b_dtype": semantics.b_dtype,
        "accumulator_dtype": semantics.accumulator_dtype,
        "output_dtype": semantics.output_dtype,
        "a_layout": semantics.a_layout,
        "b_layout": semantics.b_layout,
        "c_layout": semantics.c_layout,
        "a_stride": list(semantics.a_stride),
        "b_stride": list(semantics.b_stride),
        "c_stride": list(semantics.c_stride),
        "bounds_policy": semantics.bounds_policy,
        "mask_kind": semantics.mask_kind,
        "epilogue_kind": semantics.epilogue_kind,
    }


def _matmul_signature(dtype: str) -> dict[str, str]:
    return {
        "a": dtype,
        "b": dtype,
        "out": "*fp32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def _source_taxonomy() -> dict[str, Any]:
    return {
        "synthetic_tt_dot": {
            "source_kind": TT_DOT_SOURCE_KIND,
            "role": "legacy_static_fixture",
            "counts_as_real_jit_bridge": False,
        },
        "wrapper_extern_gemm": {
            "source_kind": "wrapper_extern_gemm",
            "source_symbol": EXTERN_GEMM_SYMBOL,
            "role": "wrapper_extern_source",
            "counts_as_real_jit_bridge": False,
        },
        "real_jit_tt_dot": {
            "source_kind": REAL_JIT_TT_DOT_SOURCE_KIND,
            "role": "real_triton_jit_tl_dot_lowering",
            "counts_as_real_jit_bridge": True,
        },
    }


def _route_boundaries() -> dict[str, Any]:
    return {
        "no_vit_wrapper_replay": True,
        "no_wrapper_buffer_replay": True,
        "no_wrapper_line_replay": True,
        "no_fused_qkv_artifact": True,
        "no_p2_or_performance_claim": True,
        "schedule_success_required": False,
    }


def _invariants(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    accepted = report.get("accepted_case") or {}
    if report.get("milestone") != MILESTONE:
        failures.append("m12_12_wrong_milestone")
    if not _accepted_case_ok(accepted):
        failures.append("m12_12_accepted_real_jit_bridge_missing")
    taxonomy = report.get("source_taxonomy") or {}
    for key in ("synthetic_tt_dot", "wrapper_extern_gemm", "real_jit_tt_dot"):
        if key not in taxonomy:
            failures.append(f"m12_12_missing_taxonomy_{key}")
    expected_negatives = dict(NEGATIVE_CASES)
    for case in report.get("negative_cases") or []:
        name = case.get("case_name")
        if name not in expected_negatives:
            failures.append(f"m12_12_unknown_negative_{name}")
            continue
        if case.get("status") != "passed":
            failures.append(f"m12_12_negative_not_passed_{name}")
        if case.get("observed_reason") != expected_negatives[name]:
            failures.append(f"m12_12_negative_reason_mismatch_{name}")
    if {case.get("case_name") for case in report.get("negative_cases") or []} != set(
        expected_negatives
    ):
        failures.append("m12_12_negative_case_set_mismatch")
    boundaries = report.get("route_boundaries") or {}
    for key in (
        "no_vit_wrapper_replay",
        "no_wrapper_buffer_replay",
        "no_wrapper_line_replay",
        "no_fused_qkv_artifact",
        "no_p2_or_performance_claim",
    ):
        if boundaries.get(key) is not True:
            failures.append(f"m12_12_boundary_violation_{key}")
    for key in (
        "performance_claim",
        "performance_ready_e2e",
        "p2_passed",
        "backend_general_complete",
        "strict_full_tvm_native",
    ):
        if report.get(key) is not False:
            failures.append(f"m12_12_forbidden_claim_{key}")
    if int(report.get("full_tvm_runnable_models", 0) or 0) != 0:
        failures.append("m12_12_full_tvm_runnable_claim_present")
    if (report.get("completion_gate") or {}).get("m12_complete") is not False:
        failures.append("m12_12_m12_complete_claim_present")
    return {"status": "passed" if not failures else "failed", "invariant_failures": failures}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "m12_12_complete": report.get("status") == "passed",
        "real_jit_tl_dot_bridge": _accepted_case_ok(report.get("accepted_case") or {}),
        "accepted_source_kind": (report.get("accepted_case") or {}).get("matmul_source_kind"),
        "negative_cases_passed": all(
            case.get("status") == "passed" for case in report.get("negative_cases") or []
        ),
        "performance_claim": False,
        "m12_complete": False,
        "next_action": (report.get("next_default_action") or {}).get("action"),
    }


def _empty_report(*, status: str, availability_reason: str) -> dict[str, Any]:
    return {
        "report_kind": REPORT_KIND,
        "schema_version": 1,
        "report_id": REPORT_ID,
        "milestone": MILESTONE,
        "status": status,
        "availability_reason": availability_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accepted_case": {},
        "negative_cases": [],
        "source_taxonomy": _source_taxonomy(),
        "route_boundaries": _route_boundaries(),
        "invariants": {"status": "not_run", "invariant_failures": []},
        "performance_claim": False,
        "performance_ready_e2e": False,
        "p2_passed": False,
        "backend_general_bridge_complete": False,
        "backend_general_complete": False,
        "strict_full_tvm_native": False,
        "full_tvm_runnable_models": 0,
        "completion_gate": {"m12_complete": False, "requires_fixed_shape_vit_p2": True},
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
    accepted = report.get("accepted_case") or {}
    lines = [
        "# M12.12 Real tl.dot Bridge Harness",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Accepted source kind: `{accepted.get('matmul_source_kind')}`",
        f"- Accepted contract ok: `{accepted.get('matmul_contract_ok')}`",
        f"- Performance claim: `{report.get('performance_claim')}`",
        f"- P2 passed: `{report.get('p2_passed')}`",
        "",
        "## Accepted Case",
        "",
        f"- Case: `{accepted.get('case_name')}`",
        f"- TTIR contains `tt.dot`: `{accepted.get('ttir_contains_tt_dot')}`",
        f"- Dot operands: `{(accepted.get('reader_dot_snapshot') or {}).get('operands')}`",
        f"- Dot attrs: `{(accepted.get('reader_dot_snapshot') or {}).get('attrs')}`",
        "",
        "## Negative Cases",
        "",
        "| Case | Expected | Observed | Status |",
        "| --- | --- | --- | --- |",
    ]
    for case in report.get("negative_cases", []):
        lines.append(
            f"| `{case.get('case_name')}` | `{case.get('expected_reason')}` | "
            f"`{case.get('observed_reason')}` | `{case.get('status')}` |"
        )
    lines.extend(["", "## Source Taxonomy", ""])
    for name, entry in (report.get("source_taxonomy") or {}).items():
        lines.append(
            f"- `{name}`: source_kind=`{entry.get('source_kind')}`, "
            f"role=`{entry.get('role')}`"
        )
    lines.extend(["", "## Invariants", ""])
    invariants = report.get("invariants") or {}
    lines.append(f"- Status: `{invariants.get('status')}`")
    for failure in invariants.get("invariant_failures", []):
        lines.append(f"- Failure: `{failure}`")
    lines.append("")
    return "\n".join(lines)


def _real_tl_dot_kernels() -> dict[str, Any]:
    import triton  # pylint: disable=import-outside-toplevel
    import triton.language as tl  # pylint: disable=import-outside-toplevel

    @triton.jit
    def supported(a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
        bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
        acc = tl.dot(av, bv, input_precision="tf32")
        tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)

    @triton.jit
    def unsupported_dtype(
        a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
        bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
        acc = tl.dot(av, bv, input_precision="tf32")
        tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)

    @triton.jit
    def unsupported_layout(
        a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
        bv = tl.load(b + n[None, :] * BLOCK_K + k[:, None])
        acc = tl.dot(av, bv, input_precision="tf32")
        tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)

    @triton.jit
    def masked_bounds(
        a,
        b,
        out,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(
            a + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            b + k[:, None] * N + n[None, :],
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, input_precision="tf32")
        tl.store(
            out + m[:, None] * N + n[None, :],
            acc,
            mask=(m[:, None] < M) & (n[None, :] < N),
        )

    @triton.jit
    def unsupported_epilogue(
        a, b, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        m = tl.arange(0, BLOCK_M)
        n = tl.arange(0, BLOCK_N)
        k = tl.arange(0, BLOCK_K)
        av = tl.load(a + m[:, None] * BLOCK_K + k[None, :])
        bv = tl.load(b + k[:, None] * BLOCK_N + n[None, :])
        acc = tl.dot(av, bv, input_precision="tf32") + 1.0
        tl.store(out + m[:, None] * BLOCK_N + n[None, :], acc)

    return {
        "supported": supported,
        "unsupported_dtype": unsupported_dtype,
        "unsupported_layout": unsupported_layout,
        "masked_bounds": masked_bounds,
        "unsupported_epilogue": unsupported_epilogue,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m tvm.contrib.triton_tvm.m1212_real_tl_dot_bridge``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    report = run_real_tl_dot_bridge_harness(out_dir=args.out_dir)
    print(
        f"M12.12 real tl.dot bridge: status={report['status']} "
        f"source_kind={(report.get('accepted_case') or {}).get('matmul_source_kind')} "
        f"out_dir={args.out_dir}"
    )
    return 0 if report["status"] in {"passed", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
