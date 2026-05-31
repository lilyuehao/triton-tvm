# Triton-TVM

`tvm.contrib.triton_tvm` is an experimental integration for carrying
Triton/TorchInductor operator evidence into TVM. The package is organized around
auditable compiler records rather than wrapper names: source identity is
captured first, low-level TTIR operations are converted into an Atomic DAG, and
only validated contracts are promoted into executable semantic regions.

The current implementation focuses on correctness closure for fixed-shape
operator routes. It does not perform graph optimization, operator fusion, or
automatic backend selection.

For the public alpha boundary, see the repository-level
[`ALPHA_SCOPE.md`](../../../../ALPHA_SCOPE.md) and
[`ALPHA_RELEASE.md`](../../../../ALPHA_RELEASE.md).

## What It Does

The integration provides a structured path from Triton-like source material to
TVM runtime execution:

```text
SourceRecord
  -> AtomicDAGRecord
  -> ContractValidationResult
  -> SemanticRegionRecord
  -> OperatorManifestRecord
  -> CapabilityRegistry
  -> OperatorExecutionRequest
  -> TIRRegionArtifact
  -> TVM packed-function execution
```

Each stage has one responsibility:

- `SourceRecord` captures where the input came from and how it is identified.
- `AtomicDAGRecord` describes TTIR-derived low-level operations and their data
  dependencies.
- `ContractValidationResult` records whether the Atomic DAG satisfies a known
  lowering contract.
- `SemanticRegionRecord` gives a validated operator region a model-level
  semantic identity.
- `OperatorManifestRecord` joins model/operator identity with proof records.
- `CapabilityRegistry` checks whether a validated region is admissible for
  runtime execution.
- `OperatorExecutionRequest` describes the exact top-level operator, buffers,
  source routes, contracts, and Atomic DAG hashes to execute.
- `TIRRegionArtifact` packages a TVM `IRModule`, lowering plan, artifact source,
  target, and buffer metadata.

## Technical Route

The design separates model semantics from lowering contracts:

```text
semantic_region_key != lowering_contract_id
```

For example, a ViT patch embedding operator keeps the model-level semantic key
`conv_patchify` while the current lowering contract can be
`conv2d_nchw_static`. A layer normalization region keeps the semantic key
`norm_row` even when the temporary lowering contract is `pointwise_flat`.

This distinction matters because support is not inferred from a wrapper,
provider name, function name, or model label. Runtime admission is based on
validated proof facts:

- source identity and origin
- Atomic DAG operation families and dependencies
- shape, dtype, and layout constraints
- contract validation status
- semantic region identity
- lowering contract identity
- runtime availability

## Package Layout

```text
source/     Source identity, artifact kinds, origin policy, input adapters
atomic/     TTIR parsing, Atomic DAG records, hashing, consistency checks
contracts/  Contract validation records and validators
semantic/   Semantic region records and graph-optimization inputs
manifest/   Model/operator manifests and proof joins
registry/   Capability records and lookup policy
runtime/    Execution requests, buffer plans, reference and TVM executors
models/     Fixed-shape model routes and runnable examples
reports/    Structured report helpers
```

## Fixed-Shape ViT Example

The package includes a fixed-shape ViT route used to exercise the current
end-to-end correctness path. It tracks 14 top-level semantic operators while
preserving 16 source/lowering route records. The attention region is represented
as one top-level `attention_decomposed` operator with three internal source
routes.

Top-level operator inventory:

```text
1 x conv_patchify
2 x pointwise_grid2d
2 x norm_row
5 x matmul
1 x attention_decomposed
3 x pointwise_flat
```

The runnable ViT path supports:

- source and Atomic DAG construction for all 16 route records
- contract validation and semantic region construction
- manifest and capability lookup
- runtime admission for 14 top-level operators
- connected buffer planning for `last_hidden_state` and `pooler_output`
- reference tensor execution for all 14 operators
- TVM packed-function execution through `TIRRegionArtifact`
- artifact-source reporting for `synthetic_tir`, `atomic_dag_gated_te_tir`,
  and `imported_tirx`
- model-output comparison for `last_hidden_state` and `pooler_output`

## Usage

Run commands from the TVM repository root.

Set `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/python"
```

The CLI entry point is config-driven:

```bash
python -m tvm.contrib.triton_tvm.models.vit \
  --config python/tvm/contrib/triton_tvm/configs/vit_s_16_224_autotune_template.json
```

The config covers target, route, weight source, correctness checks,
classification, benchmark settings, and autoscheduler selection. Programmatic
helpers for the tiny fixed-shape scaffold remain available from
`tvm.contrib.triton_tvm.models.vit`, but the CLI is reserved for model-specific
JSON configs.

## Fixed-Shape ViT-S/16 Example

The default template targets the standard ViT-S/16 inventory at 224x224:

```text
batch=1, image=224, patch=16, tokens=197, hidden=384,
heads=6, layers=12, mlp=1536, classes=1000
```

The ViT-S report includes 125 top-level operators, 149 source/lowering route
records, runtime admission status, `tvm_backend_execution` records for the
`tvm_packed` backend, optional `diagnostic_e2e` reference-vs-TVM correctness,
and `torch_compile_comparison` records for `last_hidden_state` and `logits`.

ViT-S autotuning is configured through a model-specific JSON file instead of
per-demo CLI flags. The CLI entry point accepts only `--config`:

```bash
python -m tvm.contrib.triton_tvm.models.vit \
  --config python/tvm/contrib/triton_tvm/configs/vit_s_16_224_autotune_template.json
```

The template is runnable and contains target, weight source, checks,
benchmarking, and MetaSchedule settings. Edit `autotune.operator_ids` to choose
which top-level ViT-S operators are tuned. The runner deduplicates selected
operators by workload shape and attributes, so multiple layers can reuse one
MetaSchedule database record set. For example, all QKV projections share
`tokens197_k384_n1152`, residual adds share `add_b1_tokens197_hidden384`, and
each selected attention operator expands to `attention_score`,
`attention_softmax`, and `attention_apply` workloads.

The report includes `autotune_config_path`, `autotune_config_hash`,
`selected_operator_count`, `selected_workload_count`, and
`workload_reuse_map`. Operators selected for tuning report
`codegen_source=te_workload_meta_schedule`; unselected operators keep the
baseline TE/TIR path. Atomic operator boundaries remain intact and
`performance_claim` stays `false`.

ViT-S input weights are loaded through a small registry. The built-in sources
are:

- `deterministic_random`: local deterministic synthetic weights and input.
- `timm`: a `timm` VisionTransformer state dict, with optional pretrained or
  checkpoint-backed loading.

The template uses `timm` weights and a synthetic-image classification task.
Pretrained timm weights can be requested by setting
`weight_source.pretrained=true`, or supplied from a local checkpoint with
`weight_source.checkpoint_path`. The default Hugging Face endpoint is
`https://hf-mirror.com`.

To download the timm checkpoint explicitly through the Hugging Face CLI mirror,
unset proxy variables if the Hub metadata HEAD request is intercepted:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  conda run -n tvm-0.24.0 hf download \
    timm/vit_small_patch16_224.augreg_in1k \
    model.safetensors config.json \
    --local-dir /tmp/triton_tvm_hf_weights/vit_small_patch16_224_augreg_in1k \
    --max-workers 1
```

Then run the local checkpoint-backed classification path by pointing
`weight_source.checkpoint_path` in the JSON config at the downloaded
`model.safetensors` file:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  conda run -n tvm-0.24.0 python -m tvm.contrib.triton_tvm.models.vit \
    --config python/tvm/contrib/triton_tvm/configs/vit_s_16_224_autotune_template.json
```

For a local E2E latency comparison of the Triton-TVM backend against
`torch.compile`, set `benchmark.enabled=true` in the JSON config. The benchmark
excludes weight loading, excludes backend first-call/compile and the first
`torch.compile` call from timed latency samples, runs warmup iterations,
synchronizes GPU timing, and reports all timed samples plus
min/mean/median/p90/p95/p99/max/sample-stddev in `report.json`.

Set `benchmark.cuda_graph_replay=true` to capture the pre-bound Triton-TVM
ViT-S session once and replay it during the backend timed loop. The capture
uses a dedicated CUDA stream, keeps the 125 Atomic top-level operator sequence
unchanged, and excludes session init, first call, graph capture, and warmup
from timed latency. The report records `backend_cuda_graph_replay_requested`,
`backend_cuda_graph_replay_enabled`, capture time, capture error if any, and
the replay count under `e2e_latency_benchmark.statistical_method`.

## Testing

Run the Triton-TVM test set:

```bash
python -m pytest \
  tests/python/contrib/test_triton_tvm_source.py \
  tests/python/contrib/test_triton_tvm_atomic.py \
  tests/python/contrib/test_triton_tvm_semantic.py \
  tests/python/contrib/test_triton_tvm_manifest.py \
  tests/python/contrib/test_triton_tvm_registry.py \
  tests/python/contrib/test_triton_tvm_model_adapter.py \
  tests/python/contrib/test_triton_tvm_runtime_admission.py \
  tests/python/contrib/test_triton_tvm_runtime_e2e.py \
  tests/python/contrib/test_triton_tvm_executor.py \
  tests/python/contrib/test_triton_tvm_reports.py \
  tests/python/contrib/test_triton_tvm_alpha_path.py \
  tests/python/contrib/test_triton_tvm_vit_runner.py \
  tests/python/contrib/test_triton_tvm_static_hygiene.py
```

## Current Boundaries

This package is a correctness-oriented integration layer. In particular:

- `synthetic_tir` is a runtime-channel debug artifact source and is not an
  end-to-end support proof.
- `atomic_dag_gated_te_tir` artifacts use validated Atomic DAG route evidence
  as an admission gate before selecting a fixed-shape TE/TIR template. They can
  be used for the fixed-shape correctness gate, but they do not claim generic
  Atomic-DAG-node-to-TIR lowering coverage.
- `imported_tirx` is represented in the reporting ABI; importing external TIRX
  modules is a separate integration point.
- Graph optimization and fusion are intentionally outside the current path.
- Diagnostic runtime measurements can be recorded, but the package does not
  make production performance claims.

## Report Fields To Look For

The ViT report is designed to make support and execution claims auditable. The
most useful fields are:

- `top_level_operator_count`
- `source_route_record_count`
- `semantic_region_key`
- `lowering_contract_ids`
- `lowering_contract_status`
- `semantic_proof_status`
- `support_claim_source`
- `artifact_source_breakdown`
- `tvm_executed_count`
- `allclose_count`
- `model_comparison_status`
- `model_allclose_count`
- `performance_claim`
