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
| `matmul_minimal` | supported in M9.1-M9.8 semantic/static/native/extern-proof/toy-graph scope | exact unmasked rank-2 `tt.dot` -> unresolved semantic validation block -> native serial-K correctness path plus gated `cuda_block_tile_8x8_serial_k_v1` candidate; wrapper `extern_kernels.mm` -> explicit packed-call artifact with artifact-only default and opt-in correctness-only `python_torch_host_staged` runtime proof; M9.8 adds a separate TinyMNISTMLP diagnostic baseline with two runtime-resolved extern GEMMs and one TVM pointwise ReLU |

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
- M9.4 wrapper extern GEMM artifact candidates: `extern_gemm: 11`
- remaining M9 entry wrapper debt: `extern_addmm_bias: 3`
- deferred wrapper debt: `deferred_convolution: 65`,
  `deferred_attention: 2`
- full TVM runnable models after the extern gate: 0

M9 owns the matmul entry boundary first. M9 entry is MatmulSemantics-first:

- `tt_dot`, `wrapper_extern_gemm`, and future graph matmul sources converge on
  `MatmulContract`.
- M9.1 implements the first accepted semantic scope: synthetic/static exact
  unmasked rank-2 A[M,K] x B[K,N] -> C[M,N] `tt.dot` with fp16/bf16 inputs and
  fp32 accumulation/output.
- The first lowering is an unresolved semantic TIRX `matmul` `T.sblock` with
  `SSR` axes, reads/writes, init/update, and attrs, not final native schedule
  and not `call_extern`.
- `TargetMatmulPolicy` runs after contract validation and records additive
  policy/report detail fields. M9.3 now chooses the correctness-first native
  schedule `cuda_block_per_output_serial_k_v1` for supported `tt_dot`
  semantics. This schedule id is CUDA target-policy implementation metadata,
  not a `MatmulContract` requirement.
- M9.3 CUDA runtime coverage passes for fp16 and bf16 inputs with fp32 output.
- M9.4 chooses `implementation_kind="extern_gemm"` for valid wrapper
  `extern_kernels.mm` source records and emits the explicit packed-call
  artifact `tvm.contrib.triton_tvm.extern_gemm`.
- M9.4's extern artifact is not runtime replacement and does not make models
  `full_tvm_runnable`.
- M9.5 is complete as Matmul Surface Hardening: validator checks now cover
  M/N/K attr consistency, axis/loop extents, read/write shapes, one-element
  regions, init/update stores, and extern artifact metadata. Wrapper
  `extern_gemm` records carry `extern_runtime_kind="artifact_only"` and
  `extern_runtime_replacement="not_available"`; `extern_addmm_bias` records
  carry visible `bias_add` epilogue debt. Corpus diff coverage and an opt-in
  minimal perf scaffold are present. Runtime cublas/cublaslt replacement,
  bias/add epilogue lowering, TensorCore/vendor scheduling, performance
  optimization, and model closure remain follow-on work.
- M9.6 adds an opt-in correctness-only extern GEMM runtime proof provider,
  `python_torch_host_staged`. It validates the packed ABI and row-major or
  transposed-weight-view storage semantics without making performance,
  TVM-only backend, or full-model runnable claims.
- M9.7 adds native `cuda_block_tile_8x8_serial_k_v1` for supported `tt_dot`
  shapes with M/N multiples of 8. Other accepted shapes keep
  `cuda_block_per_output_serial_k_v1`. TensorCore/PTX-MMA remains a spike
  record, not a default path or completion gate.
- M9.8 is complete as a toy graph E2E diagnostic gate before M10 attention:
  TinyMNISTMLP with flatten/view, `Linear(784,128,bias=False)`, ReLU, and
  `Linear(128,10,bias=False)`. It validates matmul + pointwise + wrapper
  extern/runtime provider + report/cache consistency + no silent fallback on
  the full MNIST test split. The staged baseline is
  `diagnostic_host_staged_triton_tvm`, with `extern_gemm_performance_claim=false`;
  it is not autotune or performance closure.

The observed M7.5 `grid` blockers and Pre-M9 deferred convolution/attention
wrapper calls remain separate deferred debt classes and should not be counted as
M9 matmul acceptance unless a later ADR changes that boundary.
