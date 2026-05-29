# Triton-TVM Alpha Scope

This alpha is a public compiler-infrastructure snapshot. It is intended for
reviewing the architecture, running the fixed-shape ViT correctness route, and
auditing the evidence chain from source identity to TVM packed-function
execution.

## Supported

- Fixed-shape ViT alpha route.
- `SourceRecord` / `AtomicDAGRecord` / `ContractValidationResult` /
  `SemanticRegionRecord` evidence path.
- 14 top-level semantic operators over 16 source/lowering route records.
- TVM packed-function correctness execution through `TIRRegionArtifact`.
- Explicit runtime admission through `CapabilityRegistry`.
- Model-output allclose checks for `last_hidden_state` and `pooler_output`.
- No silent fallback on the alpha route.
- No production performance claim.

## Not Supported

- Arbitrary Triton kernels.
- Dynamic shape.
- Graph optimization or operator fusion.
- Production CUDA performance.
- Automatic backend selection.
- General model import.
- General external TIRX module import.

## Operator Accounting

The fixed-shape ViT route reports 14 top-level semantic operators while
preserving 16 source/lowering route records. The accounting differs because the
attention block is represented as one semantic operator,
`attention_decomposed`, with three internal source routes.

Top-level semantic operator inventory:

```text
1 x conv_patchify
2 x pointwise_grid2d
2 x norm_row
5 x matmul
1 x attention_decomposed
3 x pointwise_flat
```

## Runtime Claim

The alpha route is a correctness gate. Runtime measurements may be collected
for diagnostics, but this release does not claim production latency,
throughput, TensorCore parity, or general backend competitiveness.
