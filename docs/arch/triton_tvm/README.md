# Triton-TVM Alpha Pipeline

This is the active engineering entry for `tvm.contrib.triton_tvm`.

The current package is an alpha semantic-boundary refactor. It keeps the
completed atomic route as correctness evidence, but active code no longer uses
development milestone module names, historical provider paths, or legacy source
labels as support signals.

## Canonical Pipeline

```text
SourceRecord(source_origin + artifact_kind + identity)
  -> Atomic DAG
  -> contract validator
  -> Semantic Region
  -> model manifest / capability registry
  -> runtime admission and reports
```

`source/` is only an input normalization and identity layer. It records where
input material came from and how it can be hashed or audited. It does not encode
schedules, runtime providers, model families, semantic support, or capability
credit.

## Package Map

- `source/`: `SourceRecord`, `SourceOrigin`, `SourceArtifactKind`, stable source
  and TTIR hashing, TorchInductor adapters, and native Triton language adapters.
- `atomic/`: Atomic DAG schema, textual TTIR extraction, atomic family
  classification, stable DAG hashing, unsupported atomic records, and
  SourceRecord/AtomicDAGRecord join consistency validation.
- `contracts/`: contract validators and validation-result records. This is the
  only active boundary that may promote an Atomic DAG into a Semantic Region.
- `semantic/`: Semantic Region records and graph-optimization input sidecars.
  It consumes Atomic DAG records and passing validator results; it does not
  parse TTIR directly.
- `manifest/`: model/operator manifest records. Identity is `model_id +
  operator_id`; capability matching uses `semantic_region_key`.
- `registry/`: capability lookup by Semantic Region, atomic requirements,
  shape/dtype/layout constraints, source-origin policy, then runtime admission.
- `models/`: minimal adapters that materialize SourceRecords and TTIR inputs
  without declaring backend support.
- `runtime/`: admission helpers for the final validated route only.
- `reports/`: structured evidence report helpers. Reports read facts; they do
  not decide support.

## Source Policy

Production source origins are:

- `torch_inductor`
- `native_triton_language`

`test_fixture` exists only for tests and is rejected by alpha capability
reports as production support.

Artifact kinds are:

- `triton_python_source`
- `ttir_module`
- `captured_jit_kernel`
- `exported_inductor_kernel`
- `normalized_ttir_graph`

`normalized_ttir_graph` records must include producer metadata, a stable content
hash, and parent source or TTIR provenance. Native Triton language production
records require auditable Triton source provenance or a provenance chain back to
such material.

## Atomic And Semantic Boundaries

Atomic DAG records are low-level TTIR-derived proof structures. They contain
atomic operation records, unsupported-operation records, source identity
snapshots, hashes, and producer/consumer consistency state. They do not own
semantic-region identity.

Semantic Region records are sidecars created only after a contract validator
passes. Failed validation produces a `ContractValidationResult` or a manifest
gap; it must not create a Semantic Region or registry-matchable identity.

## Manifest And Registry

Manifest identity is:

```text
model_id + operator_id
```

Capability matching key is:

```text
semantic_region_key
```

Proof facts such as source hashes, TTIR hashes, source origin, artifact kind,
validator id, and matched contract id are recovered by joining manifest records
to SourceRecord, AtomicDAGRecord, ContractValidationResult, and SemanticRegion
records. They are not independent manifest identities.

Registry lookup order is fixed:

```text
Semantic Region
  -> required atomic families / op attrs
  -> shape / dtype / layout constraints
  -> source-origin policy
  -> runtime / admission availability
```

## Adding A Model

1. Add a model adapter under `models/`.
2. The adapter emits SourceRecords and TTIR, or already-normalized TTIR graph
   inputs with producer/version/hash/provenance metadata.
3. Build Atomic DAG records from those inputs.
4. Validate SourceRecord/AtomicDAGRecord source snapshots.
5. Run contract validators.
6. Build Semantic Region sidecars only for passing validation results.
7. Build manifest records keyed by `model_id + operator_id`.
8. Resolve support through the capability registry.
9. Emit explicit manifest gaps for unsupported cases.

Model adapters must not add model-specific atomic families, source origins,
artifact kinds, Semantic Region keys, or registry keys. If the active vocabulary
is insufficient, emit a gap.

## Fixed-Shape ViT Runner

The active package includes a fixed-shape ViT runner for the current
compile/admission route:

```bash
PYTHONPATH=/home/liyh/xdb/tvm/python /home/liyh/miniconda3/envs/tvm-0.24.0/bin/python -m \
  tvm.contrib.triton_tvm.models.vit \
  --out-dir /home/liyh/xdb/triton-tvm-workbench/reports/m15/vit_fixed_shape_active_route
```

It materializes 16 fixed-shape ViT operator inputs, builds Atomic DAG records,
validates contracts, creates Semantic Region records, resolves capabilities,
and runs runtime admission. It is an alpha route runner and makes no performance
claim.

Add `--e2e-scaffold` to record the diagnostic post-admission E2E scaffold. The
scaffold records placeholders only; it does not execute model operators or
report latency.

Add `--target llvm --atomic-dag-tir` to build
`artifact_source=atomic_dag_generated_tir` artifacts from the validated Atomic
DAG route records, run 14/14 top-level operators through TVM packed functions,
and compare `last_hidden_state` plus `pooler_output` against the reference
tensor model. This remains a correctness gate only and makes no performance
claim.

## Legacy Boundary

Historical milestone modules, report scripts, provider experiments, wrapper
paths, fixed-shape dashboards, and performance loops are archived outside the
Python import path under:

```text
/home/liyh/xdb/triton-tvm-workbench/legacy_reference/m15_active_relocation
```

The archive is reference material only. Active package code and active tests do
not import it.

## Validation

Run the M15 active test gate:

```bash
PYTHONPATH=/home/liyh/xdb/tvm/python /home/liyh/miniconda3/envs/tvm-0.24.0/bin/python -m pytest \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_source.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_atomic.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_semantic.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_manifest.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_registry.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_model_adapter.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_runtime_admission.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_runtime_e2e.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_executor.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_reports.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_alpha_path.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_vit_runner.py \
  /home/liyh/xdb/tvm/tests/python/contrib/test_triton_tvm_static_hygiene.py
```

This alpha scope makes no performance claim.
