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

## Pre-M10 Attention ABI Policy

Pre-M10 freezes attention ABI/report vocabulary before M10 runtime work. It does
not implement attention lowering, RoPE runtime, KV cache runtime, or attention
performance.

Target attention contract ids:

- `attention_vit_full_v1`
- `attention_llama_causal_prefill_v1`
- `attention_llama_decode_v1`

Wrapper-level
`torch.ops.aten._scaled_dot_product_efficient_attention.default` calls remain
`op_family="deferred_attention"` and receive additive ABI fields such as
`attention_contract`, `attention_phase`, `attention_mask_kind`,
`attention_rope_policy`, `attention_kv_cache_policy`, and
`attention_runtime_status`.

All Pre-M10 attention records use
`attention_runtime_status="deferred_attention_runtime"`. `full_tvm_runnable`
still requires no wrapper-level extern calls, so attention ABI classification is
not model runtime closure.

## M9 Matmul Semantics and Contract Policy

M9 uses a unified `MatmulSemantics` / `MatmulContract` boundary before target
implementation selection. Captured TTIR `tt.dot`, wrapper-level extern GEMM, and
future graph-level matmul sources should enter the same semantic contract.

M9.1-M9.4 implement `matmul_minimal` for synthetic/static exact unmasked rank-2
`tt.dot`: A[M,K] x B[K,N] -> C[M,N], row-major, fp16/bf16 inputs, fp32
accumulation/output. The translator first validates
`implementation_kind="unresolved"` and then, when target policy allows it,
returns `implementation_kind="native_tir_schedule"` with
`schedule_id="cuda_block_per_output_serial_k_v1"`.

M9.4 also admits wrapper `extern_kernels.mm` source records as
`source_kind="wrapper_extern_gemm"` when lhs is static rank-2 row-major
`reinterpret_tensor(A, (M,K), (K,1), 0)` and rhs is either row-major
`reinterpret_tensor(B, (K,N), (N,1), 0)` or the observed transposed-weight view
`reinterpret_tensor(B, (K,N), (1,K), 0)`. Target policy returns
`implementation_kind="extern_gemm"` with `extern_symbol="extern_kernels.mm"`.

Source boundaries:

- `tt_dot`: TTIR source from `TTIRReader` and `NormalizedTTIROpGraph`.
- `wrapper_extern_gemm`: Inductor wrapper extern source collected outside the
  `AsyncCompile.triton` TTIR path.
- `future_graph_matmul`: reserved for later ATen/Relax/graph matmul sources.

The semantic validation output is a `matmul` `T.sblock`, not `call_extern`:

- spatial axes `m`, `n`; reduction axis `k`
- reads `A[m, k]`, `B[k, n]`
- writes `C[m, n]`
- init zeroes `C[m, n]`
- update performs `C[m, n] += cast(A) * cast(B)`
- function/block attrs preserve source kind, M/N/K, dtype, layout, precision,
  epilogue, bounds, and implementation metadata

`TargetMatmulPolicy` runs only after `MatmulContract` validation. M9.3 enables
the first native decision, `native_tir_schedule`, using one CUDA block per
output element and serial K accumulation. M9.4 enables the explicit extern GEMM
artifact decision, represented by one packed call to
`tvm.contrib.triton_tvm.extern_gemm`. Unsupported decisions still use a stable
report bucket with non-empty matmul-specific detail. Extern GEMM must not be
counted as native fallback or runtime replacement.

M9.5 strengthens validation without changing the accepted semantic surface. The
matmul block validator now checks function/block M/N/K attr consistency,
positive axis extents, unresolved/native loop extents, read/write buffer
shapes, one-element regions, and exactly one init store plus one update store
to the declared output buffer. Extern GEMM artifacts must preserve dtype,
layout, stride, bounds, mask, epilogue, packed-func, and runtime-gate attrs.

The CUDA schedule id `cuda_block_per_output_serial_k_v1` is target-policy
implementation metadata, not part of `MatmulContract`. The contract preserves
semantic axes, read/write regions, init/update structure, dtype/layout fields,
and implementation metadata. Future target policies may map the same semantic
matmul shape to custom hardware cores, tiles, or backend intrinsics without
renaming the contract.

The M9.4/M9.5 extern GEMM artifact is also not a runtime replacement by
default. It records an explicit packed call artifact under `matmul_minimal`
with `extern_runtime_kind="artifact_only"`,
`extern_runtime_replacement="not_available"`, and
`extern_runtime_replacement_reason="extern_gemm_runtime_replacement_gate_closed"`.
M9.6 may explicitly opt into `python_torch_host_staged`, which marks
`extern_gemm_runtime_status="runtime_resolved"` and
`extern_gemm_runtime_claim="correctness_only"`. That provider uses host staging
and makes no performance, TVM-only backend, or full-model runnable claim.
`full_tvm_runnable` remains false while wrapper extern calls and captured
kernel fallbacks remain.

M9 report details distinguish `matmul_source_kind`, `matmul_contract_ok`,
`implementation_kind`, `extern_symbol`, `schedule_id`, dtype/layout/epilogue
metadata, `extern_runtime_kind`, `extern_runtime_replacement`,
`extern_gemm_runtime_status`, `extern_gemm_provider_kind`, and
`unsupported_matmul_reason` while preserving schema-v1 top-level bucket
stability unless a later ADR changes that policy.

M9.7 adds `cuda_block_tile_8x8_serial_k_v1` as target-policy implementation
metadata for native `tt_dot` schedules. It is selected only when the accepted
semantic shape has M/N multiples of 8. It does not add boundary masks or tail
semantics; unsupported or non-eligible accepted shapes continue to use
`cuda_block_per_output_serial_k_v1`.

M9.8 adds a separate TinyMNISTMLP diagnostic baseline without changing the
`matmul_minimal` semantic contract. The staged graph uses two wrapper
`extern_kernels.mm` records resolved by the correctness-only
`python_torch_host_staged` provider plus one TVM pointwise ReLU artifact. The
report is `diagnostic_host_staged_triton_tvm`, keeps
`extern_gemm_performance_claim=false`, and does not change captured corpus
accounting or full-model runnable status.

M9.P keeps the same `matmul_minimal` core semantic contract and adds a
target-policy performance envelope, `matmul_perf_core_v1`. Native decisions
carry the frozen schedule registry version `m9p_schedule_registry_v1`,
candidate schedule ids, selected schedule id, reject reasons, perf guard
status, and a tune-key payload covering M/N/K, dtype, layout, stride,
transpose, epilogue, target/device, schedule id, tune params, framework
versions, CUDA metadata, and registry version.

M9.PA selects
`cuda_ptx_mma_m8n8k4_warp_tile_16x16_fp16fp32_v1` only for eligible fp16
`tt_dot` shapes inside the envelope. M9.PC adds
`cuda_block_tile_16x16_simt_v1` as the non-TensorCore fallback candidate for
eligible fp16/bf16 envelope shapes when TensorCore is unavailable or rejected;
legacy serial-K schedules remain lower-priority correctness fallbacks. M9.PB's
`torch_cuda_cublas_baseline_v1` remains an external dashboard baseline, not a
TVM packed runtime replacement.

M9.PE admits only the minimal wrapper `extern_kernels.addmm` bias slice as
`source_kind="wrapper_extern_addmm_bias"` with `epilogue_kind="bias_add"`.
It materializes an explicit packed-call artifact
`tvm.contrib.triton_tvm.extern_addmm_bias` and optional correctness-only
host-staged provider metadata. This does not widen `matmul_perf_core_v1`;
native performance schedules still reject epilogues through target policy.
