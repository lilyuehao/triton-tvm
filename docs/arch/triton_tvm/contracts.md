# Triton TVM Contract Policy

This page records the compact implementation-side contract policy. Long-form
planning and generated artifacts live in
`/home/liyh/xdb/triton-tvm-workbench`.

## Pre-M8 Reduction and Norm Policy

| Contract | Execution | Accumulator dtype | Epsilon | Mask | Axis | Layout |
| --- | --- | --- | --- | --- | --- | --- |
| `reduction_minimal` | `serial_m4_single_lane` | Preserve TTIR reduction dtype. | Not applicable. | Masked reduction loads require explicit zero `other`; safe unmasked parameter loads are allowed. | `axis=0` only. | Row-major input and scalar row output. |
| `norm_single_row` | `serial_m4_single_lane` | Preserve TTIR reduction dtype. | Runtime and constexpr `eps` are supported by the current single-row norm path. | Same as `reduction_minimal`. | `axis=0` only. | Single-row row-major input/output with row-local parameter vectors. |

The current M4 path is correctness-first and serial. Future parallel reduction
paths must use a distinct `execution_kind` in metadata so reports and cache keys
do not conflate them with `serial_m4_single_lane`.

## M8 Row Reduction, Norm, and Softmax Policy

M8 adds a rank-2 Grid1D row-family surface while keeping the M4 contracts
unchanged.

| Contract | Execution | Accumulator dtype | Epsilon | Mask | Axis | Layout |
| --- | --- | --- | --- | --- | --- | --- |
| `row_reduction` | `serial_m8_rank2_lane` | Preserve TTIR reduction dtype. | Not applicable. | Rank-2 masks must be explicit or dominated by the store mask. | `axis=1` only. | Rank-2 row-major. |
| `norm_row` | `serial_m8_rank2_lane` | Preserve TTIR reduction dtype. | Runtime and constexpr `eps` supported. | Same as `row_reduction`. | `axis=1` only. | Rank-2 row-major. |
| `softmax_row` | `serial_m8_rank2_lane` | Preserve TTIR reduction dtype. | Not applicable. | Same as `row_reduction`. | `axis=1` only. | Rank-2 row-major. |
| `masked_softmax_row` | `serial_m8_rank2_lane` | Preserve TTIR reduction dtype. | Not applicable. | Same as `row_reduction`; causal-style masks are admitted as rank-2 boolean masks. | `axis=1` only. | Rank-2 row-major or causal mask. |

The M8 row path supports sum and max `tt.reduce` combiners, `tt.expand_dims`,
`tt.broadcast`, `tt.extern_elementwise` `exp`/`rsqrt`, and the observed
Inductor rank-2 row indexing patterns. It does not add grid, atomic,
convolution, dot/GEMM, or attention runtime semantics.

## M8.5 Numerics and Dynamic Shape Policy

M8.5 keeps the M8 semantic surface fixed and hardens its boundary:

- `float32`, `float16`, and `bfloat16` rank-2 row reductions are covered by
  focused CUDA regression tests.
- `bfloat16` remains supported for pointwise reporting and ABI metadata, and
  M8.5 no longer rejects M8 row-family `bfloat16` TTIR before TVM build. The
  original bf16 build failure was traced to TIRX `BF16StorageLegalize` missing
  an on-demand storage remap for bf16 `compute_scope`/local allocation buffers;
  that compiler bug is fixed in the TVM tree.
- M8.5's explicit bf16 runtime coverage is row reduction. Rank-2 norm/softmax
  bf16 inputs are accepted under the same row-family contract checks, with
  broader dtype-specific numerical coverage left to future corpus-driven
  hardening.
- Dynamic rank-2 row and reduction extents are represented as runtime scalar
  params in generated TIR, with integer scalar shape/index params cast to
  `int64` in expressions. This is the builder's normalization policy for
  TIR shape/index arithmetic: TTIR may carry i32 runtime shape scalars, but TIR
  buffer extents and generated row-major indices use int64 expressions. Without
  the normalization, mixed i32/i64 dynamic shapes fail during TVMScript parsing
  before contract validation can report a useful boundary.
- `TritonTVMMeta` records both row extent metadata and reduction-axis extent
  metadata. Cache keys include this structural metadata, buffer extent
  expressions, ABI, contract/version, source hashes, and execution policy, but
  not concrete runtime shape values.
- Static partial masks keep active reduction extent metadata separate from the
  physical row storage width, so masked softmax can preserve masked stores
  without shrinking backing buffers incorrectly.

## Pre-M8 Unsupported Detail Classification

Report schema v1 top-level buckets remain stable. Pre-M8 adds detail
classification through existing additive fields:

- `fallback_reason`
- `blocker_class`
- `pre_m8_family`

The M8 entry family classes are:

- `norm_layernorm`
- `norm_rmsnorm`
- `pooling_reduction`
- `row_reduction`
- `softmax_like`
- `masked_attention_adjacent`

Deferred classes such as `deferred_grid`, `deferred_atomic`, and
`deferred_matmul_dot` remain visible but are outside Pre-M8/M8
reduction/norm/softmax scope unless a later ADR changes the milestone boundary.

## Pre-M9 Matmul and Extern Policy

M8.5 did not expose captured TTIR `tt.dot` kernels in the model corpus. Pre-M9
therefore treats wrapper-level Inductor extern calls as first-class report
observations before matmul lowering:

- `extern_gemm`: `extern_kernels.mm`/batched-GEMM-family calls. M9 entry owns
  the policy for representing these as explicit TVM artifact extern calls.
- `extern_addmm_bias`: `extern_kernels.addmm` GEMM+bias calls. M9 entry owns
  bias/epilogue visibility before lowering.
- `deferred_convolution`: `extern_kernels.convolution` calls. These remain
  outside M9 unless a later ADR admits a specific 1x1 conv-to-GEMM boundary.
- `deferred_attention`: direct ATen scaled-dot-product attention calls. These
  remain outside M9 and belong to the attention runtime milestone.

Accepted extern GEMM must be explicit in the TVM artifact and must not be
counted as native fallback. Report schema v1 top-level buckets stay stable;
Pre-M9 adds `op_family` as wrapper-level detail metadata.

`full_tvm_runnable` means all captured Triton kernels translate and no
wrapper-level extern calls remain. `triton_kernel_runnable` records the narrower
condition where captured Triton kernels translate even if wrapper externs are
still present.
