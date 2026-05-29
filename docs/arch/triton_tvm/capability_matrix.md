# Triton-TVM Alpha Capability Matrix

This matrix describes the active public alpha capability surface. It is not a
development log.

## Supported Alpha Route

| Area | Status | Notes |
| --- | --- | --- |
| Source identity | Supported | `SourceRecord` captures origin, artifact kind, stable content identity, and provenance. |
| Atomic DAG evidence | Supported | `AtomicDAGRecord` stores low-level operation families, dependencies, hashes, and unsupported-operation records. |
| Contract validation | Supported | Passing `ContractValidationResult` records are required before semantic-region construction. |
| Semantic regions | Supported | Semantic identity is separate from lowering contract identity. |
| Manifest records | Supported | Model/operator identity joins source, Atomic DAG, contract validation, and semantic evidence. |
| Capability registry | Supported | Lookup checks semantic region, required atomic families, constraints, source policy, and runtime availability. |
| TVM packed execution | Supported for the alpha route | `TIRRegionArtifact` objects execute through TVM packed functions for fixed-shape ViT correctness. |
| Report ABI | Supported | Reports expose source-route count, top-level operator count, artifact-source breakdown, allclose status, model comparison, and runtime claims. |

## Fixed-Shape ViT Operator Coverage

The alpha route tracks 14 top-level semantic operators over 16 source/lowering
route records:

| Semantic operator | Count | Notes |
| --- | ---: | --- |
| `conv_patchify` | 1 | Patch embedding convolution semantic region. |
| `pointwise_grid2d` | 2 | Grid-shaped pointwise regions. |
| `norm_row` | 2 | Row normalization regions. |
| `matmul` | 5 | Matrix multiplication regions. |
| `attention_decomposed` | 1 | One semantic attention region with three internal source routes. |
| `pointwise_flat` | 3 | Flat pointwise regions. |

## Explicit Non-Capabilities

| Area | Status |
| --- | --- |
| Arbitrary Triton kernels | Not supported |
| Dynamic shape | Not supported |
| Graph optimization | Not supported |
| Operator fusion | Not supported |
| Automatic backend selection | Not supported |
| General model import | Not supported |
| General external TIRX import | Not supported |
| Production CUDA performance claim | Not claimed |

## Contract Accounting

A route is counted as supported only when these records join successfully:

```text
SourceRecord
AtomicDAGRecord
ContractValidationResult
SemanticRegionRecord
OperatorManifestRecord
CapabilityRecord
OperatorExecutionRequest
TIRRegionArtifact
```

Records that fail validation or capability lookup must remain explicit gaps.
