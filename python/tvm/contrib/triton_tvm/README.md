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
- artifact-source reporting for `synthetic_tir`, `atomic_dag_generated_tir`,
  and `imported_tirx`
- model-output comparison for `last_hidden_state` and `pooler_output`

## Usage

Run commands from the TVM repository root.

Set `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/python"
```

Run the fixed-shape ViT route through source capture, Atomic DAG validation,
manifest construction, capability lookup, and runtime admission:

```bash
python -m tvm.contrib.triton_tvm.models.vit \
  --target llvm \
  --out-dir /tmp/triton_tvm_vit_route
```

Run the full TVM correctness path using Atomic-DAG-generated TIR artifacts:

```bash
python -m tvm.contrib.triton_tvm.models.vit \
  --target llvm \
  --atomic-dag-tir \
  --out-dir /tmp/triton_tvm_vit_atomic_dag_tir
```

The report written to `report.json` includes the operator inventory, source
route count, artifact source, per-operator allclose status, model-output
allclose status, and runtime claim fields.

For a scaffold-only post-admission report, use:

```bash
python -m tvm.contrib.triton_tvm.models.vit \
  --target llvm \
  --e2e-scaffold \
  --out-dir /tmp/triton_tvm_vit_scaffold
```

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
- `atomic_dag_generated_tir` artifacts are generated from validated route
  evidence and can be used for the fixed-shape correctness gate.
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
