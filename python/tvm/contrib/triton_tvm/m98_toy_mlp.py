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
"""M9.8 TinyMNISTMLP staged diagnostic baseline.

This module intentionally keeps M9.8 separate from the frozen 3-model corpus.
The staged path is a diagnostic integration proof: wrapper GEMMs run through
the M9.6 host-staged provider and the ReLU Inductor kernel runs through the
existing pointwise TVM path.  It is not a native performance claim.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tvm

from .inductor import (
    InductorTritonSource,
    extract_inductor_triton_sources,
    is_inductor_pointwise_kernel,
    load_inductor_kernel,
    lower_inductor_kernel_to_ttir,
)
from .matmul import (
    EXTERN_GEMM_PACKED_FUNC,
    EXTERN_GEMM_PROVIDER_ABI_VERSION,
    EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
    EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
    EXTERN_GEMM_RUNTIME_PROVIDER_KIND,
    EXTERN_GEMM_RUNTIME_PROVIDER_REASON,
    EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
    EXTERN_GEMM_SYMBOL,
    TargetMatmulDecision,
    TargetMatmulPolicy,
    build_extern_gemm_tirx_source,
    extract_matmul_semantics_from_wrapper_extern,
    register_python_torch_extern_gemm,
)
from .reporting import make_report_status
from .runtime import TritonTVMArtifact, build_triton_tvm
from .translator import TritonTVMMeta, translate_ttir


M98_REPORT_KIND = "triton_tvm_m98_tiny_mlp_diagnostic_baseline"
M98_PURPOSE = "m9.8 mnist tinymnistmlp staged diagnostic baseline"
M98_BASELINE_KIND = "diagnostic_host_staged_triton_tvm"
M98_MODEL_NAME = "TinyMNISTMLP"
M98_HIDDEN_FEATURES = 128
M98_INPUT_FEATURES = 784
M98_OUTPUT_FEATURES = 10
M98_DEFAULT_BATCH_SIZE = 100
M98_DEFAULT_TIMED_EPOCHS = 3
M98_ALLCLOSE_RTOL = 1e-4
M98_ALLCLOSE_ATOL = 1e-4


@dataclass(frozen=True)
class _TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class TinyMLPGemmRecord:
    """One normalized wrapper GEMM in the M9.8 staged plan."""

    index: int
    source_line: str
    normalized_source: str
    kernel_name: str
    matmul_m: int
    matmul_n: int
    matmul_k: int
    matmul_b_layout: str
    extern_gemm_runtime_status: str
    extern_gemm_provider_kind: str
    extern_gemm_provider_abi_version: int
    extern_gemm_runtime_claim: str
    extern_gemm_performance_claim: bool
    extern_gemm_uses_host_staging: bool
    extern_runtime_kind: str
    extern_runtime_replacement: str
    extern_runtime_replacement_available: bool
    extern_runtime_replacement_reason: str


@dataclass(frozen=True)
class TinyMLPWrapperPlan:
    """Captured TorchInductor wrapper facts needed by the staged runner."""

    wrapper_source: str
    gemm_records: tuple[TinyMLPGemmRecord, ...]
    relu_sources: tuple[InductorTritonSource, ...]


@dataclass
class TinyMLPStagedExecutable:
    """Compiled staged artifacts and persistent buffers for one batch shape."""

    model: Any
    batch_size: int
    wrapper_plan: TinyMLPWrapperPlan
    gemm_artifacts: tuple[TritonTVMArtifact, TritonTVMArtifact]
    relu_artifact: TritonTVMArtifact
    relu_record: dict[str, Any]
    weight1_torch: Any
    weight2_torch: Any
    weight1_tvm: Any
    weight2_tvm: Any
    hidden_tvm: Any
    hidden_flat_tvm: Any
    output_tvm: Any
    device: Any

    def run_cuda_batch(self, images_cuda) -> np.ndarray:
        """Run one fixed-size CUDA batch through GEMM, ReLU, GEMM."""
        if tuple(int(dim) for dim in images_cuda.shape) != (
            self.batch_size,
            1,
            28,
            28,
        ):
            raise ValueError(
                "M9.8 staged TinyMLP expects fixed full batches shaped "
                f"({self.batch_size}, 1, 28, 28), got {tuple(images_cuda.shape)}"
            )

        x_cuda = images_cuda.reshape(self.batch_size, M98_INPUT_FEATURES).contiguous()
        x_tvm = tvm.runtime.from_dlpack(x_cuda)
        self.gemm_artifacts[0].run([x_tvm, self.weight1_tvm, self.hidden_tvm])
        self.relu_artifact.run(
            [self.hidden_flat_tvm, self.batch_size * M98_HIDDEN_FEATURES]
        )
        self.gemm_artifacts[1].run([self.hidden_tvm, self.weight2_tvm, self.output_tvm])
        return self.output_tvm.numpy()


def make_tiny_mnist_mlp(torch_module, *, seed: int = 0):
    """Create the fixed M9.8 bias-free TinyMNISTMLP with deterministic weights."""
    torch_module.manual_seed(seed)
    dont_skip_tracing = getattr(getattr(torch_module, "_dynamo", None), "dont_skip_tracing", None)
    if dont_skip_tracing is None:
        dont_skip_tracing = lambda fn: fn

    class TinyMNISTMLP(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch_module.nn.Linear(
                M98_INPUT_FEATURES,
                M98_HIDDEN_FEATURES,
                bias=False,
            )
            self.fc2 = torch_module.nn.Linear(
                M98_HIDDEN_FEATURES,
                M98_OUTPUT_FEATURES,
                bias=False,
            )
            self.relu = torch_module.nn.ReLU()

        @dont_skip_tracing
        def forward(self, x):  # pylint: disable=arguments-differ
            x = x.flatten(1)
            x = self.fc1(x)
            x = self.relu(x)
            return self.fc2(x)

    return TinyMNISTMLP()


def build_tiny_mlp_staged_executable(
    *,
    batch_size: int = M98_DEFAULT_BATCH_SIZE,
    seed: int = 0,
) -> TinyMLPStagedExecutable:
    """Capture Inductor for TinyMNISTMLP and build the staged TVM artifacts."""
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("M9.8 TinyMLP staged diagnostic requires CUDA")

    model, wrappers = _capture_tiny_mlp_wrappers(torch, batch_size=batch_size, seed=seed)
    if len(wrappers) != 1:
        raise RuntimeError(f"M9.8 expected exactly one Inductor wrapper, got {len(wrappers)}")

    plan = _extract_tiny_mlp_wrapper_plan(wrappers[0], batch_size=batch_size)
    gemm_artifacts = tuple(_build_extern_gemm_artifact(record) for record in plan.gemm_records)
    if len(gemm_artifacts) != 2:
        raise RuntimeError(f"M9.8 expected two GEMM artifacts, got {len(gemm_artifacts)}")

    relu_artifact, relu_record = _build_relu_artifact(plan.relu_sources[0])
    dev = tvm.cuda(0)
    weight1_torch = model.fc1.weight.detach().contiguous()
    weight2_torch = model.fc2.weight.detach().contiguous()
    hidden_tvm = tvm.runtime.empty((batch_size, M98_HIDDEN_FEATURES), "float32", dev)
    return TinyMLPStagedExecutable(
        model=model,
        batch_size=batch_size,
        wrapper_plan=plan,
        gemm_artifacts=(gemm_artifacts[0], gemm_artifacts[1]),
        relu_artifact=relu_artifact,
        relu_record=relu_record,
        weight1_torch=weight1_torch,
        weight2_torch=weight2_torch,
        weight1_tvm=tvm.runtime.from_dlpack(weight1_torch),
        weight2_tvm=tvm.runtime.from_dlpack(weight2_torch),
        hidden_tvm=hidden_tvm,
        hidden_flat_tvm=hidden_tvm._create_view((batch_size * M98_HIDDEN_FEATURES,)),
        output_tvm=tvm.runtime.empty((batch_size, M98_OUTPUT_FEATURES), "float32", dev),
        device=dev,
    )


def run_synthetic_tiny_mlp_smoke(
    *,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a focused CUDA synthetic correctness smoke for tests and debugging."""
    torch = _require_torch()
    executable = build_tiny_mlp_staged_executable(batch_size=batch_size, seed=seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 1)
    images = torch.randn(
        (batch_size, 1, 28, 28),
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    )
    with register_python_torch_extern_gemm():
        staged = executable.run_cuda_batch(images)
    with torch.no_grad():
        expected = executable.model(images).detach().cpu().numpy()
    max_abs_error = float(np.max(np.abs(staged - expected)))
    return {
        "max_abs_error": max_abs_error,
        "allclose": bool(
            np.allclose(staged, expected, rtol=M98_ALLCLOSE_RTOL, atol=M98_ALLCLOSE_ATOL)
        ),
        "wrapper_plan": _wrapper_plan_summary(executable.wrapper_plan),
        "relu_record": executable.relu_record,
    }


def run_mnist_tiny_mlp_baseline(
    *,
    mnist_root: str | Path,
    out_dir: str | Path,
    batch_size: int = M98_DEFAULT_BATCH_SIZE,
    timed_epochs: int = M98_DEFAULT_TIMED_EPOCHS,
    download_mnist: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full MNIST test split through the staged diagnostic baseline."""
    torch = _require_torch()
    torchvision = _require_torchvision()
    if not torch.cuda.is_available():
        raise RuntimeError("M9.8 MNIST TinyMLP diagnostic baseline requires CUDA")

    mnist_root = Path(mnist_root)
    out_dir = Path(out_dir)
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root=str(mnist_root),
        train=False,
        transform=transform,
        download=download_mnist,
    )
    sample_count = len(dataset)
    if sample_count % batch_size != 0:
        raise ValueError(
            f"M9.8 fixed-batch runner requires sample_count % batch_size == 0, "
            f"got {sample_count} samples and batch_size={batch_size}"
        )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    executable = build_tiny_mlp_staged_executable(batch_size=batch_size, seed=seed)
    correctness = _run_correctness_epoch(torch, executable, data_loader)
    performance = _run_timed_epochs(
        torch,
        executable,
        data_loader,
        timed_epochs=timed_epochs,
    )
    report = _build_report(
        mnist_root=mnist_root,
        sample_count=sample_count,
        batch_size=batch_size,
        timed_epochs=timed_epochs,
        download_mnist=download_mnist,
        seed=seed,
        executable=executable,
        correctness=correctness,
        performance=performance,
    )
    _write_report(report, out_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m tvm.contrib.triton_tvm.m98_toy_mlp``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=M98_DEFAULT_BATCH_SIZE)
    parser.add_argument("--timed-epochs", type=int, default=M98_DEFAULT_TIMED_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = run_mnist_tiny_mlp_baseline(
        mnist_root=args.mnist_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        timed_epochs=args.timed_epochs,
        download_mnist=args.download_mnist,
        seed=args.seed,
    )
    perf = report["performance"]
    correctness = report["correctness"]
    print(
        "Wrote M9.8 TinyMNISTMLP diagnostic baseline to "
        f"{args.out_dir} with top1={correctness['top1_accuracy']:.4f}, "
        f"p50_batch_ms={perf['batch_latency_ms_p50']:.3f}, "
        f"images_per_second_mean={perf['images_per_second_mean']:.2f}"
    )
    return 0


def _capture_tiny_mlp_wrappers(torch, *, batch_size: int, seed: int):
    import torch._dynamo  # pylint: disable=import-outside-toplevel
    import torch._inductor.config as inductor_config  # pylint: disable=import-outside-toplevel
    from torch._inductor.graph import GraphLowering  # pylint: disable=import-outside-toplevel

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    captured: list[str] = []
    old_save_output_code = GraphLowering.save_output_code
    old_fx_graph_cache = inductor_config.fx_graph_cache
    GraphLowering.save_output_code = captured.append
    inductor_config.fx_graph_cache = False
    try:
        torch._dynamo.reset()
        model = make_tiny_mnist_mlp(torch, seed=seed).eval().to("cuda")
        example = torch.randn((batch_size, 1, 28, 28), device="cuda")
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            compiled(example)
        torch.cuda.synchronize()
    finally:
        GraphLowering.save_output_code = old_save_output_code
        inductor_config.fx_graph_cache = old_fx_graph_cache
        torch._dynamo.reset()
    return model, captured


def _extract_tiny_mlp_wrapper_plan(
    wrapper_source: str,
    *,
    batch_size: int,
) -> TinyMLPWrapperPlan:
    shape_env = _collect_tensor_specs(wrapper_source)
    tree = ast.parse(wrapper_source)
    lines = wrapper_source.splitlines()
    gemm_records: list[TinyMLPGemmRecord] = []
    for node in sorted(ast.walk(tree), key=lambda item: (getattr(item, "lineno", 0), getattr(item, "col_offset", 0))):
        if not isinstance(node, ast.Call) or _attribute_chain_ast(node.func) != EXTERN_GEMM_SYMBOL:
            continue
        source_line = lines[node.lineno - 1].strip()
        normalized_source = _normalize_extern_gemm_call(node, shape_env)
        semantics = extract_matmul_semantics_from_wrapper_extern(
            normalized_source,
            kernel_name=f"m98_tiny_mlp_gemm_{len(gemm_records)}",
        )
        decision = TargetMatmulPolicy(
            extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
        ).decide(semantics, matmul_contract_ok=True)
        if decision.implementation_kind != "extern_gemm":
            raise RuntimeError(
                "M9.8 TinyMLP expected wrapper GEMM to select extern_gemm, "
                f"got {decision.implementation_kind}: {decision.unsupported_matmul_reason}"
            )
        if decision.extern_gemm_runtime_status != EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED:
            raise RuntimeError(
                "M9.8 TinyMLP expected runtime-resolved extern GEMM provider, "
                f"got {decision.extern_gemm_runtime_status}"
            )
        gemm_records.append(_gemm_record(len(gemm_records), source_line, normalized_source, semantics, decision))

    relu_sources = tuple(
        source
        for source in extract_inductor_triton_sources(
            wrapper_source,
            case_name="m98_tiny_mlp",
        )
        if "relu" in source.kernel_name.lower()
    )
    if len(gemm_records) != 2:
        raise RuntimeError(f"M9.8 TinyMLP expected two wrapper GEMMs, got {len(gemm_records)}")
    if len(relu_sources) != 1:
        raise RuntimeError(f"M9.8 TinyMLP expected one ReLU Triton kernel, got {len(relu_sources)}")
    if tuple((record.matmul_m, record.matmul_n, record.matmul_k) for record in gemm_records) != (
        (batch_size, M98_HIDDEN_FEATURES, M98_INPUT_FEATURES),
        (batch_size, M98_OUTPUT_FEATURES, M98_HIDDEN_FEATURES),
    ):
        raise RuntimeError(
            "M9.8 TinyMLP wrapper GEMM shapes changed: "
            f"{[(record.matmul_m, record.matmul_n, record.matmul_k) for record in gemm_records]}"
        )
    return TinyMLPWrapperPlan(
        wrapper_source=wrapper_source,
        gemm_records=tuple(gemm_records),
        relu_sources=relu_sources,
    )


def _collect_tensor_specs(wrapper_source: str) -> dict[str, _TensorSpec]:
    tree = ast.parse(wrapper_source)
    specs: dict[str, _TensorSpec] = {}
    nodes = sorted(
        ast.walk(tree),
        key=lambda item: (getattr(item, "lineno", 0), getattr(item, "col_offset", 0)),
    )
    for node in nodes:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if _attribute_chain_ast(call.func) == "assert_size_stride" and len(call.args) >= 3:
                name = _expr_name(call.args[0])
                specs[name] = _TensorSpec(
                    shape=_int_tuple(call.args[1], "assert_size_stride shape"),
                    stride=_int_tuple(call.args[2], "assert_size_stride stride"),
                )
        elif isinstance(node, ast.Assign) and node.targets:
            if not isinstance(node.targets[0], ast.Name):
                continue
            target = _expr_name(node.targets[0])
            if isinstance(node.value, ast.Call):
                call_name = _attribute_chain_ast(node.value.func)
                if call_name == "empty_strided_cuda" and len(node.value.args) >= 3:
                    specs[target] = _TensorSpec(
                        shape=_int_tuple(node.value.args[0], "empty_strided_cuda shape"),
                        stride=_int_tuple(node.value.args[1], "empty_strided_cuda stride"),
                        dtype=_dtype_name(node.value.args[2]),
                    )
                elif call_name == "copy_misaligned" and node.value.args:
                    source_name = _expr_name(node.value.args[0])
                    if source_name in specs:
                        specs[target] = specs[source_name]
            elif isinstance(node.value, ast.Name) and node.value.id in specs:
                specs[target] = specs[node.value.id]
    return specs


def _normalize_extern_gemm_call(
    node: ast.Call,
    specs: dict[str, _TensorSpec],
) -> str:
    if len(node.args) < 2:
        raise RuntimeError("M9.8 wrapper extern GEMM requires lhs and rhs args")
    lhs = _normalize_gemm_operand(node.args[0], specs, "lhs")
    rhs = _normalize_gemm_operand(node.args[1], specs, "rhs")
    out_name = ""
    for keyword in node.keywords:
        if keyword.arg == "out":
            out_name = _expr_name(keyword.value)
            break
    if not out_name:
        raise RuntimeError("M9.8 wrapper extern GEMM requires out= buffer")
    return f"{EXTERN_GEMM_SYMBOL}({lhs}, {rhs}, out={out_name})"


def _normalize_gemm_operand(
    node: ast.AST,
    specs: dict[str, _TensorSpec],
    role: str,
) -> str:
    if isinstance(node, ast.Call) and _attribute_chain_ast(node.func) == "reinterpret_tensor":
        if len(node.args) < 4:
            raise RuntimeError(f"M9.8 {role} reinterpret_tensor is malformed")
        base = _expr_name(node.args[0])
        shape = _int_tuple(node.args[1], f"{role} shape")
        stride = _int_tuple(node.args[2], f"{role} stride")
        offset = _literal_int(node.args[3], f"{role} offset")
        if offset != 0:
            raise RuntimeError(f"M9.8 {role} reinterpret_tensor offset must be 0")
        return f"reinterpret_tensor({base}, {_tuple_text(shape)}, {_tuple_text(stride)}, 0)"
    name = _expr_name(node)
    spec = specs.get(name)
    if spec is None:
        raise RuntimeError(f"M9.8 cannot infer shape/stride for raw GEMM {role} {name!r}")
    if len(spec.shape) != 2 or len(spec.stride) != 2:
        raise RuntimeError(f"M9.8 raw GEMM {role} must be rank-2, got {spec.shape}")
    return f"reinterpret_tensor({name}, {_tuple_text(spec.shape)}, {_tuple_text(spec.stride)}, 0)"


def _gemm_record(
    index: int,
    source_line: str,
    normalized_source: str,
    semantics,
    decision: TargetMatmulDecision,
) -> TinyMLPGemmRecord:
    return TinyMLPGemmRecord(
        index=index,
        source_line=source_line,
        normalized_source=normalized_source,
        kernel_name=semantics.kernel_name,
        matmul_m=semantics.m,
        matmul_n=semantics.n,
        matmul_k=semantics.k,
        matmul_b_layout=semantics.b_layout,
        extern_gemm_runtime_status=decision.extern_gemm_runtime_status,
        extern_gemm_provider_kind=decision.extern_gemm_provider_kind,
        extern_gemm_provider_abi_version=decision.extern_gemm_provider_abi_version,
        extern_gemm_runtime_claim=decision.extern_gemm_runtime_claim,
        extern_gemm_performance_claim=decision.extern_gemm_performance_claim,
        extern_gemm_uses_host_staging=decision.extern_gemm_uses_host_staging,
        extern_runtime_kind=decision.extern_runtime_kind,
        extern_runtime_replacement=decision.extern_runtime_replacement,
        extern_runtime_replacement_available=decision.extern_runtime_replacement_available,
        extern_runtime_replacement_reason=decision.extern_runtime_replacement_reason,
    )


def _build_extern_gemm_artifact(record: TinyMLPGemmRecord) -> TritonTVMArtifact:
    semantics = extract_matmul_semantics_from_wrapper_extern(
        record.normalized_source,
        kernel_name=record.kernel_name,
    )
    decision = TargetMatmulPolicy(
        extern_gemm_runtime_provider=EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED
    ).decide(semantics, matmul_contract_ok=True)
    irmod = tvm.script.from_source(build_extern_gemm_tirx_source(semantics, decision))
    return build_triton_tvm(irmod, _extern_gemm_meta(semantics, decision))


def _extern_gemm_meta(semantics, decision: TargetMatmulDecision) -> TritonTVMMeta:
    abi = [
        {"name": semantics.a_param, "kind": "pointer", "dtype": semantics.a_dtype},
        {"name": semantics.b_param, "kind": "pointer", "dtype": semantics.b_dtype},
        {"name": semantics.c_param, "kind": "pointer", "dtype": semantics.output_dtype},
    ]
    transposed_b = semantics.b_layout == "transposed_weight_view"
    b_storage_shape = (semantics.n, semantics.k) if transposed_b else (semantics.k, semantics.n)
    return TritonTVMMeta(
        kernel_name=semantics.kernel_name,
        signature={},
        constexprs={},
        grid=(1,),
        target="cuda",
        target_kind="cuda",
        contract="matmul_minimal",
        canonical_contract="matmul_minimal",
        requested_contract="matmul_minimal",
        emit="tirx",
        translator_version="m98_tiny_mlp_diagnostic",
        contract_version="matmul_minimal_m9_v1",
        target_policy_version="m98_tiny_mlp_diagnostic",
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
        buffer_extents={
            semantics.a_param: f"T.int64({semantics.m * semantics.k})",
            semantics.b_param: f"T.int64({b_storage_shape[0] * b_storage_shape[1]})",
            semantics.c_param: f"T.int64({semantics.m * semantics.n})",
        },
        block_size=semantics.m,
        indexing_kind="rank2_matmul",
        execution_kind="m98_tiny_mlp_diagnostic_extern_gemm",
        accumulator_dtype_policy="fp32_accumulate",
        epsilon_policy="not_applicable",
        mask_policy="exact_unmasked",
        axis_policy="spatial_mn_reduce_k",
        layout_policy="rank2_row_major",
        launch_policy_id="m98_tiny_mlp_diagnostic",
        abi=abi,
        cache_policy="disabled",
        disk_cache_enabled=False,
        fallback_reason="",
        cache_key=(
            f"m98_tiny_mlp:{semantics.kernel_name}:{semantics.m}:"
            f"{semantics.n}:{semantics.k}:{semantics.b_layout}:"
            f"{decision.extern_gemm_provider_kind}"
        ),
        matmul_source_kind=semantics.source_kind,
        matmul_m=semantics.m,
        matmul_n=semantics.n,
        matmul_k=semantics.k,
        matmul_contract_ok=decision.matmul_contract_ok,
        implementation_kind=decision.implementation_kind,
        schedule_id=decision.schedule_id,
        extern_symbol=decision.extern_symbol,
        unsupported_matmul_reason=decision.unsupported_matmul_reason,
    )


def _build_relu_artifact(source: InductorTritonSource) -> tuple[TritonTVMArtifact, dict[str, Any]]:
    kernel = load_inductor_kernel(source)
    if not is_inductor_pointwise_kernel(kernel):
        raise RuntimeError("M9.8 ReLU kernel is not a supported pointwise Inductor kernel")
    artifact = lower_inductor_kernel_to_ttir(kernel)
    irmod, meta = translate_ttir(
        artifact,
        grid=(1,),
        target="cuda",
        contract="pointwise_flat",
    )
    built = build_triton_tvm(irmod, meta)
    record = {
        "kernel_name": meta.kernel_name,
        "contract": meta.contract,
        "translate_status": make_report_status(ok=True, bucket="translated"),
        "cache_key": meta.cache_key,
        "cache_hit": False,
        "cache_policy": meta.cache_policy,
        "disk_cache_enabled": meta.disk_cache_enabled,
        "extent_kind": meta.extent_kind,
        "extent_value": meta.extent_value,
        "block_size": meta.block_size,
        "abi": list(meta.abi),
    }
    return built, record


def _run_correctness_epoch(torch, executable: TinyMLPStagedExecutable, data_loader) -> dict[str, Any]:
    max_abs_error = 0.0
    max_rel_error = 0.0
    correct = 0
    total = 0
    allclose = True
    with register_python_torch_extern_gemm():
        with torch.no_grad():
            for images, labels in data_loader:
                images_cuda = images.to("cuda", dtype=torch.float32, non_blocking=True)
                staged = executable.run_cuda_batch(images_cuda)
                expected = executable.model(images_cuda).detach().cpu().numpy()
                abs_error = np.abs(staged - expected)
                max_abs_error = max(max_abs_error, float(abs_error.max()))
                denom = np.maximum(np.abs(expected), 1e-12)
                max_rel_error = max(max_rel_error, float((abs_error / denom).max()))
                allclose = allclose and bool(
                    np.allclose(staged, expected, rtol=M98_ALLCLOSE_RTOL, atol=M98_ALLCLOSE_ATOL)
                )
                predictions = np.argmax(staged, axis=1)
                labels_np = labels.numpy()
                correct += int((predictions == labels_np).sum())
                total += int(labels_np.size)
    return {
        "samples": total,
        "correct": correct,
        "top1_accuracy": float(correct / total) if total else 0.0,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "allclose": bool(allclose),
        "allclose_rtol": M98_ALLCLOSE_RTOL,
        "allclose_atol": M98_ALLCLOSE_ATOL,
        "accuracy_note": "random untrained TinyMNISTMLP; accuracy is a deterministic smoke metric",
    }


def _run_timed_epochs(
    torch,
    executable: TinyMLPStagedExecutable,
    data_loader,
    *,
    timed_epochs: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    total_images = 0
    with register_python_torch_extern_gemm():
        for _ in range(timed_epochs):
            for images, _labels in data_loader:
                images_cuda = images.to("cuda", dtype=torch.float32, non_blocking=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                executable.run_cuda_batch(images_cuda)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1e3
                latencies_ms.append(float(elapsed_ms))
                total_images += executable.batch_size
    if not latencies_ms:
        raise RuntimeError("M9.8 timed baseline produced no latency samples")
    total_seconds = sum(latencies_ms) / 1e3
    return {
        "timed_batches": len(latencies_ms),
        "timed_images": total_images,
        "batch_latency_ms_min": min(latencies_ms),
        "batch_latency_ms_p50": statistics.median(latencies_ms),
        "batch_latency_ms_p95": _percentile(latencies_ms, 95.0),
        "batch_latency_ms_mean": statistics.fmean(latencies_ms),
        "batch_latency_ms_max": max(latencies_ms),
        "per_image_latency_us_p50": statistics.median(latencies_ms) * 1000.0 / executable.batch_size,
        "per_image_latency_us_mean": statistics.fmean(latencies_ms) * 1000.0 / executable.batch_size,
        "images_per_second_mean": total_images / total_seconds if total_seconds else 0.0,
        "images_per_second_p50": executable.batch_size / (statistics.median(latencies_ms) / 1e3),
    }


def _build_report(
    *,
    mnist_root: Path,
    sample_count: int,
    batch_size: int,
    timed_epochs: int,
    download_mnist: bool,
    seed: int,
    executable: TinyMLPStagedExecutable,
    correctness: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_kind": M98_REPORT_KIND,
        "purpose": M98_PURPOSE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_kind": M98_BASELINE_KIND,
        "performance_claim": False,
        "model": {
            "name": M98_MODEL_NAME,
            "input_shape": [batch_size, 1, 28, 28],
            "layers": [
                "flatten/view",
                "Linear(784,128,bias=False)",
                "ReLU",
                "Linear(128,10,bias=False)",
            ],
            "trained": False,
            "seed": seed,
        },
        "dataset": {
            "name": "MNIST",
            "root": str(mnist_root),
            "split": "test",
            "sample_count": sample_count,
            "download_requested": bool(download_mnist),
        },
        "execution": {
            "batch_size": batch_size,
            "timed_epochs": timed_epochs,
            "full_batches": sample_count // batch_size,
            "path": "extern_gemm_provider -> pointwise_tvm_relu -> extern_gemm_provider",
        },
        "device": _device_info(),
        "dependency_versions": _dependency_versions(),
        "artifacts": {
            "extern_gemm_provider": {
                "kind": EXTERN_GEMM_PROVIDER_PYTHON_TORCH_HOST_STAGED,
                "abi_version": EXTERN_GEMM_PROVIDER_ABI_VERSION,
                "runtime_claim": EXTERN_GEMM_RUNTIME_CLAIM_CORRECTNESS_ONLY,
                "performance_claim": False,
                "uses_host_staging": True,
                "runtime_status": EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED,
            },
            "wrapper": _wrapper_plan_summary(executable.wrapper_plan),
            "gemm_records": [asdict(record) for record in executable.wrapper_plan.gemm_records],
            "relu_record": dict(executable.relu_record),
        },
        "diagnostics": {
            "silent_fallback_records": [],
            "native_fallback_records": [],
            "no_silent_fallback": True,
            "corpus_policy": "separate_m98_toy_report_does_not_change_3_model_corpus_89_41_48",
        },
        "correctness": correctness,
        "performance": performance,
        "boundary": {
            "official_performance_baseline": False,
            "host_staged_provider_is_diagnostic_only": True,
            "excluded": [
                "training",
                "MNIST CNN",
                "bias/addmm epilogue",
                "attention",
                "TensorCore defaulting",
                "autotune",
                "performance closure",
            ],
        },
    }


def _write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    correctness = report["correctness"]
    perf = report["performance"]
    artifacts = report["artifacts"]
    lines = [
        "# M9.8 MNIST TinyMLP Diagnostic Baseline",
        "",
        "This report is a staged diagnostic baseline, not an official native "
        "performance claim.",
        "",
        "## Summary",
        "",
        f"- Baseline kind: `{report['baseline_kind']}`",
        f"- Dataset: MNIST test, {report['dataset']['sample_count']} samples",
        f"- Batch size: {report['execution']['batch_size']}",
        f"- Correctness allclose: {correctness['allclose']}",
        f"- Top1 accuracy: {correctness['top1_accuracy']:.6f}",
        f"- Max abs error vs PyTorch: {correctness['max_abs_error']:.6e}",
        f"- Batch latency p50/p95: {perf['batch_latency_ms_p50']:.3f} / "
        f"{perf['batch_latency_ms_p95']:.3f} ms",
        f"- Images/s mean: {perf['images_per_second_mean']:.2f}",
        "",
        "## Artifacts",
        "",
        f"- Runtime-resolved extern GEMMs: {artifacts['wrapper']['extern_gemm_count']}",
        f"- ReLU translated kernels: {artifacts['wrapper']['relu_triton_kernel_count']}",
        f"- Silent fallback records: {len(report['diagnostics']['silent_fallback_records'])}",
        f"- Extern provider performance claim: "
        f"{artifacts['extern_gemm_provider']['performance_claim']}",
        "",
        "## Boundary",
        "",
        "- `python_torch_host_staged` uses host staging and is correctness-only.",
        "- The existing 3-model corpus metrics remain separate and unchanged.",
        "- Bias/addmm, CNN/convolution, attention, TensorCore defaulting, autotune, "
        "and performance closure remain out of scope.",
        "",
    ]
    return "\n".join(lines)


def _wrapper_plan_summary(plan: TinyMLPWrapperPlan) -> dict[str, Any]:
    return {
        "extern_gemm_count": len(plan.gemm_records),
        "runtime_resolved_extern_gemm_count": sum(
            1
            for record in plan.gemm_records
            if record.extern_gemm_runtime_status == EXTERN_GEMM_RUNTIME_STATUS_RUNTIME_RESOLVED
        ),
        "relu_triton_kernel_count": len(plan.relu_sources),
        "gemm_shapes": [
            [record.matmul_m, record.matmul_n, record.matmul_k] for record in plan.gemm_records
        ],
        "provider_kinds": sorted({record.extern_gemm_provider_kind for record in plan.gemm_records}),
        "extern_runtime_kinds": sorted({record.extern_runtime_kind for record in plan.gemm_records}),
    }


def _dependency_versions() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "torchvision", "triton", "tvm"):
        try:
            module = __import__(name)
        except Exception as err:  # pylint: disable=broad-except
            versions[name] = {"available": False, "error": f"{type(err).__name__}: {err}"}
            continue
        versions[name] = {
            "available": True,
            "version": str(getattr(module, "__version__", "")),
        }
    return versions


def _device_info() -> dict[str, Any]:
    try:
        torch = _require_torch()
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    except Exception as err:  # pylint: disable=broad-except
        return {"cuda_available": False, "error": f"{type(err).__name__}: {err}"}


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _require_torch():
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError("M9.8 TinyMLP requires PyTorch") from err
    return torch


def _require_torchvision():
    try:
        import torchvision  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError("M9.8 MNIST baseline requires torchvision") from err
    return torchvision


def _attribute_chain_ast(node: ast.AST) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _expr_name(node: ast.AST) -> str:
    name = _attribute_chain_ast(node)
    if name:
        return name
    raise RuntimeError(f"M9.8 expected a named tensor expression, got {ast.dump(node)}")


def _int_tuple(node: ast.AST, label: str) -> tuple[int, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise RuntimeError(f"M9.8 {label} must be a static tuple")
    return tuple(_literal_int(elt, label) for elt in node.elts)


def _literal_int(node: ast.AST, label: str) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_int(node.operand, label)
    raise RuntimeError(f"M9.8 {label} must be a static integer")


def _dtype_name(node: ast.AST) -> str:
    name = _attribute_chain_ast(node)
    if name == "torch.float32":
        return "float32"
    return name.rsplit(".", maxsplit=1)[-1] if name else ""


def _tuple_text(values: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(value) for value in values) + ")"


if __name__ == "__main__":
    raise SystemExit(main())
