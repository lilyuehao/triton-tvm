# Triton-TVM Architecture

This directory documents the active alpha architecture for
`tvm.contrib.triton_tvm`. Historical planning notes are intentionally not part
of the active path; the canonical implementation entry is the package README:

- `python/tvm/contrib/triton_tvm/README.md`

## Active Pipeline

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

Each boundary has a narrow responsibility:

- `source/` records origin, artifact kind, stable identity, and provenance.
- `atomic/` extracts low-level TTIR-derived operation evidence.
- `contracts/` validates whether an Atomic DAG satisfies a supported lowering
  contract.
- `semantic/` creates model-level semantic regions only after validation.
- `manifest/` connects model/operator identity with proof records.
- `registry/` decides whether a semantic region is admitted for runtime.
- `runtime/` builds execution requests, buffer plans, and TVM artifacts.
- `reports/` records evidence and claims without deciding support.

## Support Rule

Support is never inferred from wrapper names, provider labels, function names,
or model labels. A route is supported only when the source identity, Atomic DAG,
contract validation result, semantic region, manifest entry, capability record,
and runtime artifact are all present and consistent.

The design also keeps these identities separate:

```text
semantic_region_key != lowering_contract_id
```

For example, a ViT patch-embedding region can have semantic key
`conv_patchify` while lowering through the `conv2d_nchw_static` contract.

## Fixed-Shape ViT Alpha Route

The alpha route covers a fixed-shape ViT correctness path. It reports 14
top-level semantic operators while preserving 16 source/lowering route records.
The difference is intentional: the decomposed attention region is one
top-level semantic operator with three internal source routes.

Top-level inventory:

```text
1 x conv_patchify
2 x pointwise_grid2d
2 x norm_row
5 x matmul
1 x attention_decomposed
3 x pointwise_flat
```

The route validates source and Atomic DAG evidence, constructs semantic regions
and manifests, resolves capabilities, builds TVM artifacts, executes through
TVM packed functions, and compares `last_hidden_state` plus `pooler_output`
against the reference tensor model.

## Usage

Run from the repository root:

```bash
export PYTHONPATH="$PWD/python"

python -m tvm.contrib.triton_tvm.models.vit \
  --target llvm \
  --atomic-dag-tir \
  --out-dir /tmp/triton_tvm_vit_atomic_dag_tir
```

For a scaffold-only post-admission report:

```bash
python -m tvm.contrib.triton_tvm.models.vit \
  --target llvm \
  --e2e-scaffold \
  --out-dir /tmp/triton_tvm_vit_scaffold
```

## Validation

Run the focused Triton-TVM tests:

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

## Boundaries

The alpha route is correctness-oriented. It does not claim arbitrary Triton
kernel import, dynamic shapes, graph optimization, fusion, automatic backend
selection, general model import, or production performance.
