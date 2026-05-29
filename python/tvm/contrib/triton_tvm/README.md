# Triton-TVM

`tvm.contrib.triton_tvm` is an experimental Triton-to-TVM integration organized
around explicit source identity, low-level atomic graph evidence, semantic
contract validation, and capability lookup.

The package is intentionally split by responsibility. Source identity does not
claim support, Atomic DAGs do not own semantic meaning, model adapters do not
declare backend capability, and reports do not decide runtime admission.

## Active Pipeline

```text
SourceRecord
  -> AtomicDAGRecord
  -> ContractValidationResult
  -> SemanticRegionRecord
  -> OperatorManifestRecord
  -> CapabilityRegistry
  -> runtime admission
```

Each step adds one kind of proof:

- `SourceRecord`: where the input came from, what artifact kind it is, and how
  it is hashed or reproduced.
- `AtomicDAGRecord`: the TTIR-derived low-level operation graph, including
  unsupported atomic operations and producer/consumer consistency.
- `ContractValidationResult`: the validator result that proves whether an
  Atomic DAG satisfies a semantic contract.
- `SemanticRegionRecord`: a validator-created semantic sidecar. Failed
  validation does not create one.
- `OperatorManifestRecord`: model/operator identity and local proof joins.
- `CapabilityRegistry`: support resolution by semantic region, atomic
  requirements, constraints, source policy, and runtime availability.

## Package Layout

```text
source/     input origin, artifact kind, source/TTIR identity, adapters
atomic/     TTIR parsing, Atomic DAG records, hashing, consistency checks
contracts/  contract validation records and validators
semantic/   Semantic Region records and graph optimization inputs
manifest/   model/operator manifest records and proof joins
registry/   capability records and lookup policy
models/     model adapters that materialize inputs only
runtime/    final-route runtime admission helpers
reports/    structured evidence reports
```

## Source Identity

Production source origins are:

- `torch_inductor`
- `native_triton_language`

`test_fixture` is allowed only for tests and is excluded from production
capability reporting.

Artifact kinds are:

- `triton_python_source`
- `ttir_module`
- `captured_jit_kernel`
- `exported_inductor_kernel`
- `normalized_ttir_graph`

`source/` is not a semantic layer. It must not encode schedules, runtime
providers, model family support, registry keys, or capability credit.

## Semantic Boundary

Only `contracts/` may promote a `SourceRecord + AtomicDAGRecord` pair into a
`SemanticRegionRecord`, and only when validation passes. Unsupported TTIR,
missing atomic families, unknown contracts, and source/atomic mismatches remain
explicit validation failures or manifest gaps.

## Registry Lookup

Capability lookup is ordered:

```text
semantic region
  -> required atomic families / op attributes
  -> shape / dtype / layout constraints
  -> source-origin policy
  -> runtime admission availability
```

The manifest identity is `model_id + operator_id`. The semantic region key is a
capability matching key, not a unique model/operator identity.

## Fixed-Shape ViT Runner

The active package includes a fixed-shape ViT alpha runner that exercises the
current compile/admission route without importing archived milestone code:

```bash
PYTHONPATH=/home/liyh/xdb/tvm/python /home/liyh/miniconda3/envs/tvm-0.24.0/bin/python -m \
  tvm.contrib.triton_tvm.models.vit \
  --out-dir /home/liyh/xdb/triton-tvm-workbench/reports/m15/vit_fixed_shape_active_route
```

The runner materializes 16 fixed-shape ViT operator inputs as production
TorchInductor `SourceRecord`s, builds Atomic DAGs, validates contracts, creates
Semantic Region records, resolves capability records, and runs runtime
admission. It writes `report.json` and makes no performance claim.

## Development Checks

Run the active package tests with:

```bash
PYTHONPATH=/home/liyh/xdb/tvm/python /home/liyh/miniconda3/envs/tvm-0.24.0/bin/python -m pytest \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_source.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_atomic.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_semantic.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_manifest.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_registry.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_model_adapter.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_runtime_admission.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_reports.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_alpha_path.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_vit_runner.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_static_hygiene.py
```

This package currently makes no performance claim.
