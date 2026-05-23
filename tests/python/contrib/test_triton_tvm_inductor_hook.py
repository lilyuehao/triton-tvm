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
"""M5 experimental TorchInductor hook tests."""

import json

import pytest

import tvm.contrib.triton_tvm as triton_tvm_pkg
import tvm.testing
import tvm.contrib.triton_tvm.inductor as inductor_mod
from tvm.contrib.triton_tvm.inductor import (
    InductorTritonSource,
    TritonTVMInductorConfig,
    TritonTVMInductorSession,
    make_triton_tvm_inductor_backend,
)

try:
    import torch
except ImportError:
    torch = None


_STABLE_M5_BUCKETS = {
    "translated",
    "unsupported_ttir_op",
    "contract_error",
    "target_policy_error",
    "unsupported_stream",
    "input_error",
    "triton_tvm_error",
    "internal_error",
}


def test_m5_backend_factory_is_experimental_not_top_level_public_api():
    backend = make_triton_tvm_inductor_backend()

    assert callable(backend)
    assert isinstance(backend.triton_tvm_session, TritonTVMInductorSession)
    assert "make_triton_tvm_inductor_backend" not in triton_tvm_pkg.__all__
    assert "TritonTVMInductorSession" not in triton_tvm_pkg.__all__


def test_m5_compile_hook_falls_back_to_native_and_reports_stable_bucket(tmp_path):
    session = TritonTVMInductorSession(TritonTVMInductorConfig(report_dir=tmp_path))
    sentinel = object()
    calls = []

    def native_triton(async_compile, kernel_name, source_code, device_str="cuda"):
        calls.append((async_compile, kernel_name, source_code, device_str))
        return sentinel

    result = session.compile_triton(
        object(),
        native_triton,
        "bad_kernel",
        "not_a_triton_kernel = 1",
        "cuda",
    )

    assert result is sentinel
    assert calls and calls[0][1:] == ("bad_kernel", "not_a_triton_kernel = 1", "cuda")
    assert session.records
    status = session.records[-1]["translate_status"]
    assert status["ok"] is False
    assert status["bucket"] in _STABLE_M5_BUCKETS
    assert status["fallback_reason"]

    report = session.report()
    assert report["summary"]["status_buckets"][status["bucket"]] == 1
    assert report["counters"]["native_fallbacks"] == 1
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["report_kind"] == "triton_tvm_capability_report"
    assert written["kernels"][0]["translate_status"]["fallback_reason"]


def test_m55_graph_report_tracks_fallback_kernel_and_cache_lifecycle():
    session = TritonTVMInductorSession()
    graph = session.begin_graph(compile_region_name="unit")

    session._record_fallback(  # pylint: disable=protected-access
        InductorTritonSource(
            case_name="bad",
            kernel_name="bad_kernel",
            source="not_a_triton_kernel = 1",
        ),
        {
            "ok": False,
            "bucket": "unsupported_ttir_op",
            "fallback_reason": "unsupported_ttir_op",
        },
    )
    session.end_graph(graph, ok=True)

    report = session.report()
    assert report["artifact_cache_size"] == 0
    assert report["graph_summary"] == {
        "cache_hits": 0,
        "cache_misses": 0,
        "completed_graphs": 1,
        "failed_graphs": 0,
        "native_fallback_kernels": 1,
        "run_count": 0,
        "total_graphs": 1,
        "total_kernels": 1,
        "translated_kernels": 0,
    }
    assert report["graphs"][0]["compile_region_name"] == "unit"
    assert report["graphs"][0]["fallback_reasons"] == {"unsupported_ttir_op": 1}
    assert report["kernels"][0]["graph_id"] == report["graphs"][0]["graph_id"]


def test_pre_m6_inductor_report_schema_snapshot_and_graph_consistency(tmp_path):
    session = TritonTVMInductorSession()
    graph = session.begin_graph(compile_region_name="pre_m6")

    class FakeTTIRArtifact:
        ttir = "not valid ttir"

    class FakeMeta:
        cache_key = "a" * 64
        cache_policy = "disabled"
        disk_cache_enabled = False

    class FakeBuilt:
        meta = FakeMeta()

    translated = session._record_translated(  # pylint: disable=protected-access
        kernel=inductor_mod.InductorKernel(
            case_name="good",
            kernel_name="good_kernel",
            source="@triton.jit\ndef good_kernel():\n    pass\n",
            device_str="cuda",
            fn=None,
            signature={"x": "*fp32"},
            constexprs={"BLOCK": 64},
            attrs=None,
            triton_meta={},
            inductor_meta={},
            size_hints={},
        ),
        artifact=FakeTTIRArtifact(),
        built=FakeBuilt(),
        cache_hit=False,
    )
    session._record_run(translated)  # pylint: disable=protected-access
    session._record_fallback(  # pylint: disable=protected-access
        InductorTritonSource(
            case_name="bad",
            kernel_name="bad_kernel",
            source="not_a_triton_kernel = 1",
        ),
        {
            "ok": False,
            "bucket": "contract_error",
            "fallback_reason": "unsupported_inductor_kernel",
        },
    )
    session.end_graph(graph, ok=True)

    report = session.report()
    assert {
        "summary",
        "graph_summary",
        "graphs",
        "kernels",
        "private_api_guard",
        "runtime_counter_policy",
        "report_flush_policy",
    } <= set(report)
    assert report["graph_summary"]["cache_misses"] == 1
    assert report["graph_summary"]["run_count"] == 1
    assert report["runtime_counter_policy"] == {
        "native_fallback_count": "compile_time_native_fallback_decisions_only",
        "run_count": "successful_tvm_launcher_calls_only",
    }
    assert report["report_flush_policy"] == "eager_when_report_dir_set"

    kernels = report["kernels"]
    graph_record = report["graphs"][0]
    for expected_kernel_index, record_index in enumerate(graph_record["kernel_record_indices"]):
        assert 0 <= record_index < len(kernels)
        kernel_record = kernels[record_index]
        assert kernel_record["graph_id"] == graph_record["graph_id"]
        assert kernel_record["graph_kernel_index"] == expected_kernel_index
        assert {
            "cache_key",
            "cache_hit",
            "native_fallback_count",
            "run_count",
            "graph_id",
            "graph_kernel_index",
        } <= set(kernel_record)

    assert kernels[0]["cache_key"] == "a" * 64
    assert kernels[0]["run_count"] == 1
    assert kernels[1]["native_fallback_count"] == 1
    assert kernels[1]["run_count"] == 0

    session.write_report(tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Graph Summary" in markdown
    assert "JSON `graphs` and `kernels` are authoritative" in markdown
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["graphs"][0]["kernel_record_indices"] == [0, 1]


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_m55_hook_reentrancy_guard_and_restore():
    from torch._inductor.async_compile import AsyncCompile

    session = TritonTVMInductorSession()
    original = AsyncCompile.triton
    installed = inductor_mod._install_async_compile_triton_hook(  # pylint: disable=protected-access
        AsyncCompile,
        session,
    )
    try:
        assert AsyncCompile.triton is not original
        with pytest.raises(RuntimeError, match="reentrant_compile"):
            inductor_mod._install_async_compile_triton_hook(  # pylint: disable=protected-access
                AsyncCompile,
                session,
            )
    finally:
        inductor_mod._restore_async_compile_triton_hook(  # pylint: disable=protected-access
            AsyncCompile,
            installed,
        )
    assert AsyncCompile.triton is original


def test_pre_m6_private_api_guard_fails_fast_before_patch_and_reports_bucket():
    class BadAsyncCompile:
        def triton(self, kernel_name):  # pylint: disable=unused-argument
            return None

    session = TritonTVMInductorSession()
    original = BadAsyncCompile.triton

    with pytest.raises(ValueError, match="AsyncCompile.triton signature"):
        inductor_mod._install_async_compile_triton_hook(  # pylint: disable=protected-access
            BadAsyncCompile,
            session,
        )

    assert BadAsyncCompile.triton is original
    assert inductor_mod._ACTIVE_HOOK is None  # pylint: disable=protected-access
    report = session.report()
    assert report["collection_errors"][0]["bucket"] == "input_error"
    assert report["collection_errors"][0]["stage"] == "private_api_guard"
    assert report["private_api_guard"]["ok"] is False


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_m55_hook_restores_after_compile_exception(monkeypatch):
    from torch._inductor.async_compile import AsyncCompile
    import torch._inductor.compile_fx as compile_fx_mod

    original = AsyncCompile.triton

    def fail_compile_fx(*_args, **_kwargs):
        raise RuntimeError("forced compile failure")

    monkeypatch.setattr(compile_fx_mod, "compile_fx", fail_compile_fx)
    backend = make_triton_tvm_inductor_backend()

    with pytest.raises(RuntimeError, match="forced compile failure"):
        backend(object(), [])

    assert AsyncCompile.triton is original
    report = backend.triton_tvm_session.report()
    assert report["graph_summary"]["failed_graphs"] == 1
    assert report["graphs"][0]["status"] == "internal_error"


def test_m55_raw_stream_handoff_restores_default_stream(monkeypatch):
    session = TritonTVMInductorSession()
    record = {"run_count": 0}
    stream_events = []

    class FakeDevice:
        def set_raw_stream(self, stream):
            stream_events.append(stream)

    class FakeArtifact:
        meta = object()

        def __init__(self):
            self.received_args = None

        def run(self, args):
            self.received_args = args
            return "ok"

    artifact = FakeArtifact()
    monkeypatch.setattr(
        inductor_mod,
        "_torch_runtime_args_to_tvm",
        lambda args, _meta: (["tvm_arg", *args], 0),
    )
    monkeypatch.setattr(inductor_mod, "_tvm_cuda_device", lambda _device_index: FakeDevice())

    kernel = inductor_mod._TritonTVMInductorKernel(  # pylint: disable=protected-access
        session=session,
        record=record,
        artifact=artifact,
        async_compile=None,
        original_triton=None,
        kernel_name="fake",
        source_code="",
        device_str="cuda",
    )

    assert kernel.run("torch_arg", stream=12345) == "ok"
    assert artifact.received_args == ["tvm_arg", "torch_arg"]
    assert stream_events == [12345, 0]
    assert record["run_count"] == 1
    assert session.counters["runs"] == 1


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m5_torch_compile_pointwise_runs_tvm_and_reuses_cache(tmp_path):
    def fn(x, y):
        return x + y * 2.0

    session = TritonTVMInductorSession(TritonTVMInductorConfig(report_dir=tmp_path))
    backend = make_triton_tvm_inductor_backend(session)
    x = torch.randn((128,), device="cuda")
    y = torch.randn((128,), device="cuda")

    out = _compile_and_run(fn, (x, y), backend)
    torch.testing.assert_close(out, fn(x, y))

    x2 = torch.randn((128,), device="cuda")
    y2 = torch.randn((128,), device="cuda")
    out2 = _compile_and_run(fn, (x2, y2), backend)
    torch.testing.assert_close(out2, fn(x2, y2))

    translated = [
        record
        for record in session.records
        if record["translate_status"]["ok"] and record["contract"] == "pointwise_flat"
    ]
    assert translated
    assert any(record["run_count"] > 0 for record in translated)
    assert any(record["cache_hit"] for record in translated)
    assert all(record["translate_status"]["fallback_reason"] == "" for record in translated)
    assert session.report()["artifact_cache_size"] >= 1
    assert (tmp_path / "report.json").exists()


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m5_torch_compile_unsupported_kernel_uses_native_fallback():
    def fn(x):
        return torch.sum(x, dim=1)

    session = TritonTVMInductorSession()
    backend = make_triton_tvm_inductor_backend(session)
    x = torch.randn((16, 32), device="cuda")

    out = _compile_and_run(fn, (x,), backend)
    torch.testing.assert_close(out, fn(x))

    fallback_records = [
        record for record in session.records if not record["translate_status"]["ok"]
    ]
    assert fallback_records
    assert all(record["translate_status"]["fallback_reason"] for record in fallback_records)


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@tvm.testing.requires_cuda
def test_m55_torch_compile_graph_report_mixes_tvm_and_native_fallback(tmp_path):
    def fn(x, y):
        pointwise = y + 1.0
        reduction = torch.sum(x.reshape(16, 8), dim=1)
        return pointwise, reduction

    session = TritonTVMInductorSession(TritonTVMInductorConfig(report_dir=tmp_path))
    backend = make_triton_tvm_inductor_backend(session)
    x = torch.randn((128,), device="cuda")
    y = torch.randn((256,), device="cuda")

    out_pointwise, out_reduction = _compile_and_run(fn, (x, y), backend)
    expected_pointwise, expected_reduction = fn(x, y)
    torch.testing.assert_close(out_pointwise, expected_pointwise)
    torch.testing.assert_close(out_reduction, expected_reduction)

    graph = next(graph for graph in session.report()["graphs"] if graph["total_kernels"] >= 2)
    assert graph["translated_kernels"] >= 1
    assert graph["native_fallback_kernels"] >= 1
    assert graph["cache_misses"] >= 1
    assert graph["fallback_reasons"]
    assert graph["run_count"] >= 1


def _compile_and_run(fn, args, backend):
    import torch._dynamo  # pylint: disable=import-outside-toplevel
    import torch._inductor.config as inductor_config  # pylint: disable=import-outside-toplevel

    old_fx_graph_cache = inductor_config.fx_graph_cache
    inductor_config.fx_graph_cache = False
    try:
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=backend)
        with torch.no_grad():
            result = compiled(*args)
        torch.cuda.synchronize()
        return result
    finally:
        inductor_config.fx_graph_cache = old_fx_graph_cache
        torch._dynamo.reset()


if __name__ == "__main__":
    tvm.testing.main()
