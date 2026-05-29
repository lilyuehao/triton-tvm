# Triton-TVM Design

`tvm.contrib.triton_tvm` is organized around auditable compiler records. The
design goal is to make every support claim reconstructable from source
identity, low-level operator evidence, contract validation, semantic identity,
capability admission, and runtime artifact execution.

## Core Invariants

- Source records identify input material; they do not imply backend support.
- Atomic DAG records describe low-level operation evidence; they do not own
  model semantics.
- Contract validators are the only boundary that can promote Atomic DAG
  evidence toward a semantic region.
- Semantic regions carry model-level operator identity.
- Manifest records join model/operator identity to source, Atomic DAG,
  validation, and semantic records.
- Capability lookup is explicit and ordered.
- Runtime execution happens only through admitted operator execution requests.
- Reports record evidence and claims; they do not decide support.

## Identity Model

Two identities are intentionally separate:

```text
semantic_region_key != lowering_contract_id
```

`semantic_region_key` names the model-level operator region. Examples include
`conv_patchify`, `norm_row`, `matmul`, `attention_decomposed`, and
`pointwise_flat`.

`lowering_contract_id` names the validated lowering surface used to construct
or execute a TVM artifact. Examples include static convolution, row norm,
matrix multiplication, decomposed attention, and pointwise contracts.

This separation prevents a wrapper name, provider id, function name, model
label, or historical route name from becoming a support signal.

## Admission Order

The capability registry resolves support in this order:

```text
Semantic region
  -> required atomic families and operation attributes
  -> shape, dtype, and layout constraints
  -> source-origin policy
  -> runtime and artifact availability
```

An unsupported route must produce an explicit gap or rejection reason. Silent
fallback is outside the alpha contract.

## Runtime Path

The runtime path is operator-oriented:

```text
OperatorExecutionRequest
  -> buffer plan
  -> TIRRegionArtifact
  -> TVM packed function
  -> output comparison
```

The current alpha route executes fixed-shape ViT operators through TVM packed
functions and compares model outputs against the reference tensor model.

## Artifact Sources

The report ABI distinguishes artifact sources:

- `atomic_dag_generated_tir`: generated from validated Atomic DAG route
  records and used by the alpha correctness gate.
- `synthetic_tir`: debug/runtime-channel artifact source, not an end-to-end
  support proof.
- `imported_tirx`: external TIRX ABI representation; general import is outside
  the current alpha.

## Public Alpha Boundary

The public alpha is a correctness and architecture snapshot. It does not claim
graph optimization, operator fusion, dynamic-shape support, automatic backend
selection, arbitrary Triton-kernel import, or production performance.
