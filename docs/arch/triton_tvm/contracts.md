# Triton-TVM Contract Policy

Contracts are the validation boundary between low-level Atomic DAG evidence and
model-level semantic regions. A contract does not name a model operator by
itself, and a semantic operator does not imply that a lowering contract is
available.

## Contract Validation

Contract validators check:

- operation family requirements
- required operation attributes
- shape, dtype, and layout constraints
- source-origin policy
- unsupported-operation records
- route-record consistency

Only passing validators can produce semantic-region sidecars. Failed validation
must produce an explicit validation result or manifest gap.

## Semantic vs Lowering Identity

The active design keeps this invariant:

```text
semantic_region_key != lowering_contract_id
```

Examples:

- `conv_patchify` can lower through a static convolution contract.
- `norm_row` can lower through row-oriented or temporary pointwise contracts.
- `attention_decomposed` can own several internal source routes while remaining
  one top-level semantic operator.

## Alpha Route Contracts

The fixed-shape ViT alpha route uses contracts for:

- patch embedding convolution
- grid-shaped pointwise operators
- row normalization
- matrix multiplication
- decomposed attention
- flat pointwise operators

The route is considered admitted only after contract validation, semantic
region construction, manifest construction, capability lookup, runtime request
construction, and TVM artifact execution all succeed.

## Unsupported Cases

Unsupported cases must remain explicit. The alpha contract policy does not
allow silent fallback, implicit provider substitution, wrapper-name support
credit, or model-label support credit.
