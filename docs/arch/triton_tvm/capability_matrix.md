# Triton TVM Capability Matrix

This file is the compact TVM-side implementation matrix. The full planning
matrix and historical reports live in
`/home/liyh/xdb/triton-tvm-workbench`.

## Supported Contracts

| Contract | Status | Notes |
| --- | --- | --- |
| `pointwise_minimal` | supported | minimal scalar/vector pointwise path |
| `pointwise_flat` | supported | M7 pointwise, broadcast, view, indexing surface |
| `reduction_minimal` | supported, Pre-M8 policy documented | correctness-first `serial_m4_single_lane` path |
| `norm_single_row` | supported, Pre-M8 policy documented | single-row LayerNorm/RMSNorm path over `serial_m4_single_lane` |
| `row_reduction` | supported in M8 | rank-2 Grid1D row sum/max reduction over `serial_m8_rank2_lane` |
| `norm_row` | supported in M8 | rank-2 LayerNorm/RMSNorm-family row reductions |
| `softmax_row` | supported in M8 | rank-2 row max/exp/sum softmax |
| `masked_softmax_row` | supported in M8 | rank-2 masked and causal-style row softmax policy |

## M7.5 Guard State

- Model corpus: ViT/YOLO/Llama tiny, 89 kernels.
- Translated kernels: 31.
- Remaining explicit fallbacks: 57 `contract_error`, 1 `unsupported_ttir_op`.
- Residual blocker split: 48 `grid`, 9 `reduction`, 1 `attention_adjacent`.
- Supported report contract histogram: `pointwise_flat: 31`.
- Supported report guards: `legacy_contract_records=[]`,
  `silent_fallback_records=[]`, `pre_m7.m7_entry_blockers=[]`.
- Perf smoke: 6 cases passed on NVIDIA RTX 6000D.

## M8 Guard State

- Model corpus: ViT/YOLO/Llama tiny, 89 kernels.
- Translated kernels: 41.
- Remaining explicit fallbacks: 48 `contract_error`.
- Residual blocker split: 48 `grid`.
- Supported report contract histogram: `pointwise_flat: 32`,
  `norm_row: 6`, `row_reduction: 2`, `masked_softmax_row: 1`.
- Supported report guards: `legacy_contract_records=[]`,
  `silent_fallback_records=[]`.
- M7.5 `grid` debt remains explicitly deferred.

## M8.5 Guard State

- Semantic coverage remains unchanged from M8: 41 translated kernels and
  48 explicit `grid` fallbacks.
- M8.5 adds focused hardening for rank-2 row-family `float32`/`float16`
  numerics, dynamic row/reduction extents, cache-key extent metadata, masked
  softmax partial masks, causal-style masks, and RMSNorm `rsqrt`/epsilon.
- M8 row-family `bfloat16` is no longer rejected before build. The original
  rank-2 bf16 build failure was traced to and fixed in TIRX
  `BF16StorageLegalize`; M8.5 now includes CUDA runtime coverage for bf16 row
  reduction. Pointwise bf16 reporting remains unchanged.
- Supported report guards remain clean:
  `legacy_contract_records=[]`, `silent_fallback_records=[]`.

## Stable Report Buckets

- `translated`
- `unsupported_ttir_op`
- `contract_error`
- `target_policy_error`
- `unsupported_stream`
- `input_error`
- `collection_error`
- `triton_tvm_error`
- `internal_error`

Use additive detail fields rather than introducing new top-level buckets without
an ADR or design update.

Pre-M8 additive detail fields are `fallback_reason`, `blocker_class`, and
`pre_m8_family`.

## Next Boundary

Pre-M9 adds wrapper-level extern observability while preserving the M8.5 captured
Triton-kernel surface:

- captured Triton kernels: 89 total, 41 translated, 48 explicit `grid`
  fallbacks
- observed TTIR `tt.dot` kernels: 0
- M9 entry wrapper debt: `extern_gemm: 11`, `extern_addmm_bias: 3`
- deferred wrapper debt: `deferred_convolution: 65`,
  `deferred_attention: 2`
- full TVM runnable models after the extern gate: 0

M9 owns matmul, GEMM, epilogue, performance, and autotune work. The observed
M7.5 `grid` blockers and Pre-M9 deferred convolution/attention wrapper calls
remain separate deferred debt classes and should not be counted as M9 matmul
acceptance unless a later ADR changes that boundary.
