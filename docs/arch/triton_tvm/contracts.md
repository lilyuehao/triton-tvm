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

Pre-M10 records used `attention_runtime_status="deferred_attention_runtime"`.
M10.1-M10.6 now consume the same ABI fields for the observed ViT full-attention
call and the observed Llama causal prefill call while leaving decode
synthetic/report-only. `full_tvm_runnable` still requires no wrapper-level
extern calls, so attention ABI classification and correctness runtime
resolution are not model runtime closure.

## M10 Attention Runtime Entry Policy

M10.1-M10.4 admit the observed ViT full-attention wrapper SDPA call:

- source op:
  `torch.ops.aten._scaled_dot_product_efficient_attention.default`
- contract: `attention_vit_full_v1`
- required semantics: rank-4 fp32 Q/K/V, matching static shapes, explicit
  shape/stride metadata, no mask, `causal=False`, and numeric scale
- artifact: explicit packed call
  `tvm.contrib.triton_tvm.extern_attention_sdpa`
- default runtime status: `artifact_only`
- opt-in provider:
  `--attention-runtime-provider python_torch_host_staged`
- provider claim: `correctness_only`, ABI v1, host-staged,
  `attention_performance_claim=false`
- opt-in native provider:
  `--attention-runtime-provider native_decomposed`
- native decomposition: QK^T matmul, row softmax, and AV matmul in one
  correctness-first CUDA TIR artifact, with explicit `matmul_minimal` and
  `softmax_row` boundary attrs
- native claim: `correctness_only`, ABI v1, no host staging,
  `attention_performance_claim=false`

The M10.4 native-provider corpus report keeps captured-kernel accounting
stable: 89 captured Triton kernels, 41 translated, 48 explicit `grid`
fallbacks, 2 wrapper SDPA calls, 1 ViT runtime-resolved native record, 1 Llama
deferred attention record, 1 materialized attention artifact, and 0 full TVM
runnable models.

M10.5 additionally admits only the observed Llama causal prefill wrapper SDPA
call under `--attention-runtime-provider native_decomposed`:

- source op:
  `torch.ops.aten._scaled_dot_product_efficient_attention.default`
- contract: `attention_llama_causal_prefill_v1`
- required semantics: rank-4 fp32 Q/K/V/output, Q/K treated as already-RoPE'd
  upstream tensors, rank-4 V `reinterpret_tensor`, rank-4 additive causal mask
  `reinterpret_tensor`, matching Q/K/V/output shapes, mask shape `[B,H,S,S]`,
  `is_causal=False`, and numeric scale
- mask metadata: `attention_mask_param`, `attention_mask_shape`,
  `attention_mask_stride`, and `attention_mask_dtype`
- default and `python_torch_host_staged` runtime status:
  `deferred_attention_runtime`
- native decomposition: QK^T matmul, mask add, masked row softmax, and AV
  matmul in one correctness-first CUDA TIR artifact, with explicit
  `matmul_minimal` and `masked_softmax_row` boundary attrs
- native claim: `correctness_only`, ABI v1, no host staging,
  `attention_performance_claim=false`

The M10.5 native-provider corpus report keeps captured-kernel accounting
stable: 89 captured Triton kernels, 41 translated, 48 explicit `grid`
fallbacks, 2 wrapper SDPA calls, 2 native runtime-resolved attention records, 0
artifact-only attention calls, and 0 full TVM runnable models.

M10.6 keeps `attention_llama_decode_v1` synthetic/report-only. Runtime support
remains explicitly unsupported until a real corpus decode wrapper call appears.

M10 also has a separate same-machine attention performance baseline report,
`m10_attention_native_decomposed_vs_torch_sdpa_v1`, for fixed ViT full-attention
and Llama causal-prefill shapes. That report is not a corpus runtime-provider
claim: M10 corpus records still keep `attention_performance_claim=false`.

M10 hardening is complete. Attention provider ids now fail explicitly when
unknown; attention records and TIR attrs include launch and byte accounting;
contract validators check accounting invariants; model reports aggregate
launch, memory, host-staging, and unsupported runtime reason counters. The
regenerated native-provider corpus reports `m10_runtime_hardened_v1`, 2 runtime
launches, 2 artifact calls, 25,600 total IO bytes, 154,624 intermediate buffer
bytes, 0 host-staging bytes, and 0 full TVM runnable models.

M10 hardening does not add arbitrary masks, RoPE runtime, KV cache
layout/update/read, decode runtime, or corpus provider performance claims.
Those remain behind later ADR decisions.

## Pre-M11 Vision Operator Policy

Pre-M11 freezes the M11 vision/convolution entry boundary without adding
lowering or runtime support.

Primary M11 entry debt:

- `deferred_convolution=65`: wrapper-level `extern_kernels.convolution` calls.
- `captured_grid=48`: explicit captured-kernel `contract_error/grid` blockers.

Vision operator families that must be classified before YOLO E2E claims:

- conv2d, 1x1 conv, depthwise conv, grouped conv
- pool, resize, concat, slice
- YOLO decode, postprocess, NMS

Layout and parameter policy:

- NCHW and NHWC must be reported explicitly.
- channels-last must be classified before lowering.
- stride, padding, and dilation must be part of the reported policy boundary.

Implementation policy:

- Implicit PyTorch fallback is disallowed.
- Explicit TVM extern is allowed when reported.
- Native TIR/TIRX schedules should be introduced only after the concrete
  operator contract/report boundary is defined.
- NMS may start as an explicit TVM extern and move to native TIR/TIRX later.

Historical `deferred_attention=2` wrapper-family counts remain visible in
reports, but both observed SDPA calls are runtime-resolved under M10
`native_decomposed`; attention is not M11 vision entry debt. RoPE runtime,
KV-cache runtime, decode runtime, arbitrary attention masks, and attention
performance claims remain deferred outside M11.

## M11.0-M11.2 Vision Convolution Artifact Policy

M11.0-M11.2 freeze the convolution report and artifact boundary without adding
convolution runtime support.

Frozen contract/version ids:

- `vision_contract_version="m11_v1"`
- `conv2d_nchw_static_v1`
- `conv2d_1x1_nchw_static_v1`
- `depthwise_conv2d_nchw_static_v1`
- `grouped_conv2d_nchw_static_v1`

Accepted wrapper convolution records must be static fp32 rank-4 logical
NCHW/OIHW conv2d with valid stride, padding, dilation, and groups; no bias;
`transposed=False`; `output_padding=(0, 0)`; and output shape matching the
standard conv formula. Channels-last memory format may be recorded from tensor
strides, but the semantic layout contract remains explicit.

The M11.2 artifact is an explicit packed call:

- packed function: `tvm.contrib.triton_tvm.extern_conv2d`
- runtime status: `vision_runtime_status="artifact_only"`
- runtime launches: `vision_runtime_launch_count=0`
- artifact calls: `vision_artifact_call_count=1`
- performance claim: `vision_performance_claim=false`
- runtime replacement: unavailable

Artifact-only convolution does not make a model full TVM runnable.

## M11.4-M11.6 Vision Runtime Provider Policy

M11.4 admits exactly one runtime-resolved vision path before hardening: the
ViT patch-embedding `conv2d_nchw_static_v1` case with input `(1,3,32,32)`,
weight `(64,3,16,16)`, output `(1,64,2,2)`, stride `(16,16)`, padding
`(0,0)`, dilation `(1,1)`, groups `1`, no bias, no transpose, and zero output
padding.

The runtime provider is explicit and opt-in:

- provider: `python_torch_host_staged`
- packed function: `tvm.contrib.triton_tvm.extern_conv2d`
- runtime kind: `runtime_provider`
- runtime status: `vision_runtime_status="runtime_resolved"`
- provider ABI: `vision_provider_abi_version=1`
- runtime claim: `vision_runtime_claim="correctness_only"`
- host staging: `vision_uses_host_staging=true`
- performance claim: `vision_performance_claim=false`

All other accepted M11 conv records remain artifact-only under M11.4 and carry
`unsupported_vision_runtime_reason="vision_conv2d_provider_scope_m11_4_vit_patch_only"`.
Unknown vision runtime provider ids fail explicitly with
`vision_conv2d_runtime_provider_unknown_m11_4`.

M11.P provides a same-machine diagnostic baseline for the admitted ViT patch
shape against Torch CUDA conv2d. The baseline is not a provider-wide
performance claim and does not make any model full TVM runnable.

M11.5 hardens this narrow surface without expanding it:

- `m11.interface_status="m11_5_vision_runtime_hardened_v1"` in provider
  corpus reports.
- `m11.report_cache_invariants` records the report/cache invariant status.
- `m11.corpus_diff_guard` records the M11 baseline
  `m11_5_vit_patch_provider_corpus_baseline_v1`.
- `m11.m10_attention_boundary` records that M10 attention remains closed.
- Vision conv2d validators reject performance claims, runtime launch count
  regressions, runtime scope escape from the exact ViT patch case, and exact
  byte-accounting mismatches.

M11.6 expands the opt-in conv correctness provider scope to
`m11_6_static_conv2d_runtime_scope_v1`:

- `N=1`, static fp32 NCHW input and OIHW weight.
- `groups=1`, `dilation=(1,1)`, no bias, no transpose, zero output padding.
- regular conv stride `1`, `2`, or `16`; padding `0` or `1`.
- 1x1 conv stride `1`, padding `0`.

This remains `python_torch_host_staged` and
`vision_runtime_claim="correctness_only"` with
`vision_performance_claim=false` and host-staging bytes recorded. The
regenerated M11.6 corpus observes 65/65 conv wrapper calls runtime-resolved
under that explicit provider, but this is not a native conv performance claim
and not full-model closure.

M11.7 adds the explicit device-side provider id `device_torch_cuda` for the
same static conv2d envelope. This provider consumes CUDA TVM tensors through
DLPack, invokes Torch CUDA conv2d on device, writes the CUDA output tensor
without host staging, and records:

- `vision_runtime_status="runtime_resolved"`
- `vision_provider_kind="device_torch_cuda"`
- `vision_runtime_claim="performance_eligible"`
- `vision_performance_claim=true`
- `vision_uses_host_staging=false`
- `vision_host_staging_bytes=0`
- `m11.interface_status="m11_7_vision_native_hotpath_runtime_closure_v1"`
- `m11.runtime_scope_status="m11_7_device_static_conv2d_hotpath_scope_v1"`

The regenerated M11.7 corpus reports 89 captured Triton kernels, 72 translated,
and 17 explicit Grid2D fallbacks. The 31 M11.6 Grid2D-ready records are now
materialized as validated `pointwise_grid2d_static_v1` native TVM artifacts in
the captured-kernel translated accounting, while the remaining 17 concat/split
records stay explicit `contract_error` blockers with
`grid2d_concat_split_multi_output_layout_deferred_m11_6`. The same corpus
resolves 65/65 wrapper conv records through `device_torch_cuda`, resolves
25/25 observed `conv2d_1x1_nchw_static_v1` records under the M11.7 hotpath
classification, sets `vit_tiny_random_runtime_resolved_smoke=true`, and keeps
`full_tvm_runnable=0`. The M11.7 dashboard is a hotpath dashboard; benchmark
measurements are opt-in and host-staged providers remain excluded from
performance claims.

## Pre-M12 E2E Entry Policy

Pre-M12 separates frozen historical entry debt from current residual M12 debt.
Pre-M11 historical debt remains `deferred_convolution=65` and
`captured_grid=48`; current residual debt is reported in the generated
`pre_m12` section.

M12 starts from:

- 89 captured kernels, 72 translated, and 17 explicit fallbacks.
- 17 YOLO concat/split Grid2D blockers with
  `grid2d_concat_split_multi_output_layout_deferred_m11_6`.
- 11 artifact-only `extern_gemm` calls and 3 artifact-only
  `extern_addmm_bias` calls across ViT/Llama.
- 65/65 wrapper conv records runtime-resolved under explicit
  `device_torch_cuda`.
- 2/2 observed wrapper SDPA calls runtime-resolved under M10
  `native_decomposed`.
- `full_tvm_runnable_models=0`.

M12 must freeze closure levels before broadening runtime:

- `runtime_resolved_smoke`: diagnostic provider/native mix evidence.
- `performance_ready_e2e`: every required captured and wrapper call has an
  explicit admitted runtime path, zero host staging, complete provider mix,
  correctness, and an E2E performance report.
- `strict_full_tvm_native`: no host-staged provider, no opaque fallback, and no
  provider path counted as native TVM schedule unless the milestone implements
  that native schedule.

M12 is scoped to `vit_tiny_random` fixed-shape E2E. Completion requires P2:
fixed ViT shape, warmed cache, `host_staging_bytes=0`, complete provider mix,
p50/p95 latency reported, and p50/p95 each `<= 2x`
`torch.compile`/Inductor. P3 is optional and only opens after optimization
opportunity review. YOLO closure, arbitrary ViT variants, dynamic shape, strict
full-native replacement claims, general concat/split Grid2D closure, new
attention semantics, and native TVM conv scheduling claims are non-goals for
M12.

Native TVM conv scheduling, RoPE runtime, KV-cache runtime, decode runtime,
arbitrary attention masks, and attention provider performance claims remain
outside the M12 entry unless an explicit M12 ADR or submilestone admits them.

M12.0-M12.14 are complete as a ViT-only fixed-shape E2E surface, targeted
provider-cost recovery step, P2 gate decision, provider-runtime optimization,
route cleanup, backend-general route freeze, real `tl.dot` bridge proof, and
runtime overhead measurement, plus real `tt.dot` schedule handoff.
M12.0 freezes the closure/performance vocabulary. M12.1 records the ordered
`vit_tiny_random` execution plan with 7 captured kernels and 9 wrapper calls.
M12.2 admits only the observed ViT fp32 wrapper matmul/addmm shapes through the
correctness-first `native_tvm_matmul` provider:

- 4/4 ViT `extern_gemm` records and 3/3 ViT `extern_addmm_bias` records are
  `runtime_resolved`.
- The schedule id is `cuda_block_per_output_serial_k_v1`.
- Runtime claim is `native_tvm_fixed_shape_correctness_only`.
- `host_staging_bytes=0` and `extern_gemm_performance_claim=false`.
- M12.3 runs the fixed-shape ViT correctness runner and passes P0 against
  Torch eager CUDA: allclose is true, provider reporting is complete, no silent
  fallback is true, host staging is 0, and provider counts are 7 explicit
  captured-kernel harness launches, 1 `device_torch_cuda` conv,
  1 `native_decomposed` attention replay, and 7 `native_tvm_matmul` wrapper
  matmul/addmm launches.
- M12.4 measures fixed-shape E2E latency and passes P1 but misses P2 at
  `2.7478x / 2.6755x` `torch.compile` p50/p95.
- M12.5/M12.5.5 identify and freeze `native_tvm_matmul_provider_cost` as the
  primary P2 recovery target.
- M12.6 hardens the surface without performance optimization. Unknown extern
  GEMM provider ids fail explicitly with
  `extern_gemm_runtime_provider_unknown_m12_6`; dashboard/freeze schema,
  report/cache, provider-id, no-stale-fallback, no-hidden-host-staging,
  no-strict-native, performance-ready, and corpus diff guards all pass.
- M12.7 profiles the 7 native matmul wrapper calls, validates the fixed-shape
  `no_per_call_sync` path, identifies `kernel_event_ms` as the dominant legacy
  provider-cost category, and recovers `0.095584 ms` p50 / `0.090501 ms` p95
  E2E while preserving all provider/correctness/host-staging invariants.
- M12.8 reruns the warmed-cache P2 gate and preserves correctness allclose,
  complete provider mix, no silent fallback, and `host_staging_bytes=0`, but
  P2 fails at `2.4731x / 2.4193x` `torch.compile` p50/p95. The decision is to
  return to profiling rather than continue blind optimization.
- M12.9 improves the best measured path to `2.2841x / 2.2618x`
  `torch.compile`, still above P2.
- M12.10 marks fixed-shape fused QKV as `model_specific_diagnostic_only`,
  keeps it out of Triton backend progress accounting, and hands off to M12.11.
- M12.11 freezes `primary_route=real_tl_dot_bridge`, orders
  `runtime_overhead` before `tt_dot_schedule_handoff`, and sets M12.12 real
  `tl.dot` bridge harness as the next implementation slice.
- M12.12 proves a real `@triton.jit` `tl.dot` kernel can lower through
  Triton JIT to textual three-operand `tt.dot`, parse through `TTIRReader`,
  and validate as `MatmulSemantics.source_kind=real_jit_tt_dot` under
  `matmul_minimal` for the supported row-major fp16/fp32 case.
- M12.13 measures reusable runtime overhead with a standalone `pointwise_flat`
  artifact, records DLPack, packed-function lookup, artifact dispatch,
  prebound packed-call, CUDA event, dispatch-minus-kernel, and no-per-call-sync
  launch-envelope buckets, and keeps P2/performance and schedule-handoff
  claims false.
- M12.14 completes real `tt.dot` schedule handoff. Real JIT 16x16x16 fp16
  `tt.dot` selects TensorCore schedule
  `cuda_ptx_mma_m8n8k4_warp_tile_16x16_fp16fp32_v1`, real JIT 8x8x16 fp16
  `tt.dot` selects reusable tiled schedule `cuda_block_tile_8x8_serial_k_v1`,
  and the M12.9 wrapper-specific schedule is excluded from handoff readiness.
- M12.P iteration `provider_runtime_glue_prebound_packed_call_v1` adds
  backend-general prebound packed-call dispatch for the fixed-shape native
  matmul provider and passes the unchanged P2 gate at `1.5043x / 1.4946x`
  `torch.compile` p50/p95. `performance_ready_e2e=true` for the M12 provider
  mix; `strict_full_tvm_native=false` and `full_tvm_runnable_models=0` remain.
- M12.P post-closure iteration `provider_runtime_glue_minimal_record_v1` keeps
  the same backend-general provider route and measures
  `triton_tvm_e2e_m12p_minimal_record` at `0.472496 / 0.503189 ms` p50/p95,
  or `1.4856x / 1.4179x` versus `torch.compile`, without expanding strict
  full-native, native-conv, dynamic-shape, YOLO, Llama, or fused-QKV claims.
- Llama GEMM, YOLO concat/split, dynamic shapes, and strict full-native
  replacement claims remain outside M12.2 scope.

## M11.3/M11.6 Captured-Grid Policy

M11.3 classifies captured Grid2D blockers without expanding supported grid or
runtime semantics.

Frozen taxonomy id:

- `m11_3_grid_taxonomy_v1`

Per-kernel additive report fields:

- `m11_grid_status`
- `m11_grid_family`
- `m11_grid_launch_kind`
- `m11_grid_program_axes`
- `m11_grid_size_hints`
- `m11_grid_operator_tags`
- `unsupported_m11_grid_reason`
- `m11_grid_contract`
- `m11_grid_contract_version`
- `m11_grid_artifact_status`
- `m11_grid_runtime_status`
- `m11_grid_implementation_kind`
- `m11_grid_provider_kind`
- `m11_grid_runtime_claim`
- `m11_grid_performance_claim`
- `m11_grid_runtime_launch_count`
- `m11_grid_artifact_call_count`
- `m11_grid_host_staging_bytes`
- `unsupported_m11_grid_runtime_reason`

M11.3 fields apply only to nontranslated records with
`blocker_class="grid"`. Translated kernels and non-grid blockers keep empty
defaults. The stable family ids are `grid_conv_adjacent_pointwise`,
`grid_bn_silu_fusion`, `grid_concat_split`, `grid_pool_or_softmax`,
`grid_yolo_decode_postprocess`, and `grid_other_multidim_pointwise`.

M11.6 adds the native static Grid2D readiness contract
`pointwise_grid2d_static_v1`. It admits static Grid2D affine f32 pointwise and
BN/SiLU-adjacent scaffold cases as `implementation_kind="native_tvm_grid2d"`
with zero host staging and a native performance dashboard. Complex
concat/split Grid2D records remain explicit unsupported runtime blockers with
`grid2d_concat_split_multi_output_layout_deferred_m11_6`. The regenerated
M11.6 corpus classifies all 48 Grid2D blockers, marks 31 artifact/native
runtime-ready, keeps 17 concat/split records unsupported, records zero silent
fallback, and keeps `full_tvm_runnable=0`.

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
- `real_jit_tt_dot`: real Triton JIT `tl.dot` artifact lowered by
  `lower_to_ttir` to textual `tt.dot`, parsed by `TTIRReader`, and admitted
  only for the supported exact unmasked row-major rank-2 case.
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

M12.2 adds a separate Native TVM wrapper matmul runtime surface for the fixed
ViT shape only. When `--extern-gemm-runtime-provider native_tvm_matmul` is
scoped with `--extern-gemm-runtime-model-case vit_tiny_random`, supported
`wrapper_extern_gemm` and `wrapper_extern_addmm_bias` records may select
`implementation_kind="native_tir_schedule"` with
`provider_kind="native_tvm_matmul"`, runtime status `runtime_resolved`, and
runtime claim `native_tvm_fixed_shape_correctness_only`. The generated TIRX
uses the serial-K per-output schedule
`cuda_block_per_output_serial_k_v1`, supports the observed ViT fp32 shapes
`(M,N,K)=(5,64,64)`, `(5,128,64)`, `(5,64,128)`, and `(1,64,64)` for GEMM,
and `(5,64,64)` with rank-1 bias for addmm. Transposed-weight views are
represented by B storage shaped `(N,K)` and reads `B[n, k]`. Contract
validation requires `extern_gemm_uses_host_staging=false`,
`extern_gemm_performance_claim=false`, no stale packed-call provider metadata,
and exact `epilogue_kind` matching (`none` for GEMM, `bias_add` for addmm).

M12.12 adds the real JIT `tl.dot` bridge without changing wrapper extern
behavior. `TTIRReader` records real textual `tt.dot` operands, result types,
and bare attrs such as `inputPrecision = tf32`. Real JIT dot artifacts use
`source_kind="real_jit_tt_dot"`; legacy raw/static fixtures keep
`source_kind="tt_dot"`, and wrapper externs keep
`source_kind="wrapper_extern_gemm"`. The admitted real JIT case requires
rank-2 A[M,K] x B[K,N] -> C[M,N], fp16/bf16 inputs, fp32 accumulator/output,
unmasked exact loads/stores, inferable row-major A/B/C pointer layout, and a
zero third `tt.dot` accumulator operand. Unsupported dtype, layout,
masked/bounds, and epilogue cases report explicit reason ids:
`matmul_input_dtype_not_supported`, `non_row_major_matmul_not_supported`,
`matmul_bounds_policy_not_supported`, and `matmul_epilogue_not_supported`.
M12.12 does not claim schedule handoff, TensorCore success, P2, performance
readiness, ViT wrapper replay, or fused-QKV readiness.

M12.13 adds reusable runtime-overhead measurement without changing wrapper
extern behavior. It builds a standalone `pointwise_flat` artifact through
`lower_to_ttir -> translate_ttir -> build_triton_tvm`, measures DLPack
conversion, packed-function lookup, artifact run wall time, prebound
packed-call wall time, CUDA event kernel time, dispatch-minus-kernel estimate,
and no-per-call-sync launch envelope, and keeps P2, performance readiness,
schedule handoff, strict-native, and model-specific executor claims false.

M12.14 adds real `tt.dot` schedule handoff without changing wrapper extern
behavior. It uses real Triton JIT `tl.dot` lowering-derived
`MatmulSemantics.source_kind=real_jit_tt_dot` to select reusable TVM GPU
schedules, including TensorCore for 16x16x16 fp16 and tiled schedule for
8x8x16 fp16. The M12.9 wrapper-specific schedule is explicitly excluded from
handoff readiness; P2, performance readiness, backend-complete, strict-native,
and full-runnable claims remain false.
