# Triton TVM Backend Design

## Scope

The implementation remains in the TVM tree under
`python/tvm/contrib/triton_tvm`. The companion workbench stores planning and
generated artifacts only.

## TTIR Boundary

`ttir.py` owns textual TTIR parsing and normalization. `translator.py` consumes
`NormalizedTTIROpGraph` and should not instantiate `TTIRReader` directly.

New TTIR op support should update reader or graph snapshots before translator
lowering behavior is expanded.

## Builder Policy

The backend keeps the TVMScript source builder through M7.5. The only controlled
source parsing boundary is `tvm.script.from_source`. A direct TIR node builder
rewrite is deferred until contract cleanup proves it is necessary.

## Report Policy

Report schema v1 keeps stable top-level buckets:

- `translated`
- `unsupported_ttir_op`
- `contract_error`
- `target_policy_error`
- `unsupported_stream`
- `input_error`
- `collection_error`
- `triton_tvm_error`
- `internal_error`

Milestone-specific metadata should be additive. Examples include
`fallback_reason`, `blocker_class`, `pre_m7`, and `supported_kernel_report`.

## Fallback Policy

Native fallback must be explicit. A failed translation or native fallback needs
a non-empty `fallback_reason`. M7.5 supported reports are clean only when
`legacy_contract_records=[]` and `silent_fallback_records=[]`.

## Current Milestone Boundary

M8 is complete for the Pre-M8-approved reduction, norm, softmax, masked
softmax, and causal-mask softmax surface. The model corpus now translates 41 of
89 kernels; the remaining 48 explicit fallbacks are deferred `grid` blockers.

The compact contract policy is in `contracts.md`. M4
`reduction_minimal`/`norm_single_row` lowering remains
`execution_kind=serial_m4_single_lane`. M8 row-family lowering uses
`execution_kind=serial_m8_rank2_lane` for `row_reduction`, `norm_row`,
`softmax_row`, and `masked_softmax_row`.

M8.5 is complete and keeps the same semantic coverage. It hardens row-family
fp32/fp16 numerics, opens bf16 row-family translation after fixing the TIRX
bf16 storage legalizer local-buffer remap bug, records reduction-axis extent
metadata in cache keys, and normalizes rank-2 dynamic integer shape/index
scalars to int64 expressions. Focused CUDA runtime coverage now includes bf16
row reduction; broader bf16 norm/softmax numerical matrices remain future
hardening work.

Pre-M9 is complete. The model corpus still has 41 translated captured Triton
kernels and 48 explicit `grid` fallbacks, with 0 observed TTIR `tt.dot` kernels.
Pre-M9 adds wrapper-level extern observability: 11 `extern_gemm` calls, 3
`extern_addmm_bias` calls, 65 `deferred_convolution` calls, and 2
`deferred_attention` calls. `full_tvm_runnable` now requires no captured Triton
fallbacks and no wrapper-level extern calls; `triton_kernel_runnable` preserves
the narrower captured-kernel-only condition.

M9 starts from a unified `MatmulSemantics` / `MatmulContract` boundary rather
than treating captured TTIR `tt.dot` support and wrapper-level extern GEMM
lowering as alternatives. The first semantic lowering target is an unresolved,
schedulable TIRX `matmul` block with attrs preserving source, dtype, layout,
M/N/K, precision, bounds, epilogue, and implementation metadata. Target policy
then chooses native schedule, explicit TVM artifact extern GEMM, or unsupported
fallback. M9 must not silently absorb deferred `grid`, convolution, or attention
debt.

M9.1 is implemented for synthetic/static exact unmasked rank-2 `tt.dot` through
`MatmulSemantics`, `matmul_minimal`, an unresolved semantic
`T.sblock("matmul")`, and validator coverage. M9.2 adds
`TargetMatmulPolicy` and additive report metadata. M9.3 enables the first
executable native schedule, `cuda_block_per_output_serial_k_v1`, and CUDA
correctness coverage for fp16/bf16 inputs to fp32 output. M9.4 materializes
wrapper `extern_kernels.mm` as an explicit packed-call artifact under
`matmul_minimal`, with `implementation_kind="extern_gemm"` and
`extern_symbol="extern_kernels.mm"`. M9.5 hardens this surface without
semantic expansion: contract validation now checks function/block M/N/K attrs,
axis and loop extents, read/write buffer shapes, one-element regions, and
init/update stores; extern GEMM artifacts must preserve dtype/layout/policy
attrs, the packed function name, and artifact-only runtime-gate metadata.

The M9.3 CUDA schedule is a `TargetMatmulPolicy` implementation, not part of
`MatmulContract`; future target policies may map the same semantic matmul axes
to custom hardware cores, tiles, or backend intrinsics. The M9.4 extern artifact
is not runtime replacement and does not make wrapper GEMM runnable. M9.5 records
that gate explicitly with `extern_runtime_kind="artifact_only"` and
`extern_runtime_replacement="not_available"`. M9.PE later admits only the
minimal `extern_addmm_bias` `bias_add` artifact slice; runtime
cublas/cublaslt replacement, arbitrary epilogues, activation epilogues,
attention, convolution, and model closure remain follow-on work.

The GEMM follow-on track is split to keep runtime proof, schedule candidates,
and graph integration concerns separate. M9.6 is an extern GEMM runtime proof:
an opt-in
`python_torch_host_staged` provider registers `tvm.contrib.triton_tvm.extern_gemm`
for correctness-only artifact execution, validates the packed ABI, and records
runtime-resolved wrapper extern coverage. It is not a performance path,
TVM-only backend claim, or full-model runnable closure. M9.7 owns native matmul
schedule quality: `cuda_block_tile_8x8_serial_k_v1` is selected only for
supported `tt_dot` shapes with M/N multiples of 8; other accepted shapes keep
`cuda_block_per_output_serial_k_v1`. TensorCore/PTX-MMA remains a spike record,
not a default path or completion gate. M9.8 is a toy graph E2E smoke before
attention: TinyMNISTMLP with flatten/view, two bias-free linear layers, ReLU,
extern provider/report/cache visibility, and no silent fallback. M9.8 is now
implemented as a separate full-MNIST diagnostic baseline with two
runtime-resolved `python_torch_host_staged` extern GEMMs and one TVM pointwise
ReLU artifact. It keeps `extern_gemm_performance_claim=false`; autotune,
algorithm selection, and performance closure remain later work.

These follow-ons should not change the captured Triton-kernel accounting by
themselves. The current `89/41/48` corpus metric counts captured Triton
kernels. Runtime-resolved wrapper `extern_gemm` calls should be reported as
wrapper extern runtime coverage unless a later report ADR explicitly changes
the model-level counting policy.

M9.P is the post-M9.8 Matmul Performance Track / Horizontal CUDA Matmul
Hardening track before M10 attention. It keeps `MatmulSemantics` stable and
uses `matmul_perf_core_v1` as a target-policy performance envelope for native
CUDA matmul schedules. The preferred native source is `tt_dot`; wrapper
`extern_gemm` remains explicit artifact/runtime-proof coverage and must not be
used as a performance path through `python_torch_host_staged`.

The M9.P core envelope is intentionally narrow: fp16/bf16 inputs, fp32
accumulation, fp32 output first, row-major A/C, row-major or
transposed-weight-view B, rank-2 exact bounds, M/N/K multiples of 16 or 32, and
compile-time-specialized tile sizes. Cases outside that envelope must be
explicitly unsupported by target policy rather than silently widening
`matmul_minimal`.

M9.P may move QK^T, AV, projection, and MLP/FFN shape suites forward as matmul
performance and schedule-selection cases only. It does not move masked
softmax, RoPE, KV cache, attention ABI, or attention runtime forward; those
remain M10 scope.

M9.P Phase 0-3 is implemented. The frozen registry version is
`m9p_schedule_registry_v1`, and native matmul decisions now carry
`matmul_perf_envelope`, candidate schedule ids, selected schedule id, reject
reasons, perf guard status, and tune-key payloads. The PA TensorCore candidate
is `cuda_ptx_mma_m8n8k4_warp_tile_16x16_fp16fp32_v1`, selected only for fp16
`tt_dot` shapes inside `matmul_perf_core_v1`. The PC SIMT fallback candidate is
`cuda_block_tile_16x16_simt_v1`, selected for eligible fp16/bf16 envelope
shapes when TensorCore is rejected. PB uses
`torch_cuda_cublas_baseline_v1` as an external dashboard baseline; TVM packed
cuBLAS/cuBLASLt runtime replacement remains explicitly unavailable.

M9.PE admits only wrapper
`extern_kernels.addmm(bias, reinterpret_tensor(A), reinterpret_tensor(B),
alpha=1, beta=1, out=C)` as `source_kind="wrapper_extern_addmm_bias"` with
`epilogue_kind="bias_add"`. The artifact is an explicit packed call to
`tvm.contrib.triton_tvm.extern_addmm_bias` with artifact-only default metadata
and optional correctness-only host-staged provider metadata. The native
`matmul_perf_core_v1` envelope still rejects epilogues, so this semantic
breadth does not become a hidden performance-schedule expansion.

Pre-M10 is now the attention ABI/report cleanup gate before M10 runtime work.
It adds `python/tvm/contrib/triton_tvm/attention.py` and classifies wrapper
SDPA calls without lowering them. The frozen attention contracts are
`attention_vit_full_v1`, `attention_llama_causal_prefill_v1`, and
`attention_llama_decode_v1`; every Pre-M10 attention record remains
`op_family="deferred_attention"` with
`attention_runtime_status="deferred_attention_runtime"`.

Pre-M10 preserves report schema-v1 buckets, captured-kernel accounting, and
`full_tvm_runnable` policy. M10.1-M10.6 now consume this surface for the
observed ViT full-attention SDPA call and the observed Llama causal prefill
SDPA call: `AttentionSemantics`, the explicit packed-call artifact
`tvm.contrib.triton_tvm.extern_attention_sdpa`, and the opt-in correctness-only
`python_torch_host_staged` ViT provider. The provider keeps
`attention_performance_claim=false` and does not change full-model runnable
status. M10.4 adds the opt-in correctness-only `native_decomposed` provider
for ViT QK^T, row softmax, and AV, with explicit `matmul_minimal` and
`softmax_row` boundary attrs and no performance claim. M10.5 extends
`native_decomposed` to Llama causal prefill with already-RoPE'd Q/K tensors,
rank-4 V, additive causal mask metadata, QK^T, masked row softmax, and AV.
M10.6 keeps decode synthetic/report-only until a real corpus decode wrapper
appears.

M10 also records a separate same-machine attention performance baseline,
`m10_attention_native_decomposed_vs_torch_sdpa_v1`, for fixed ViT
full-attention and Llama causal-prefill shapes against Torch CUDA SDPA. This
baseline does not change corpus runtime-provider metadata:
`attention_performance_claim` remains false in M10 corpus records.

M10 hardening is complete. Attention runtime records and TIR attrs now carry
launch/byte accounting, unknown attention providers fail explicitly, and model
reports aggregate launch, memory, host-staging, and unsupported runtime reason
counters. RoPE runtime, KV cache layout/update/read, causal decode runtime,
arbitrary mask support, and corpus provider performance claims remain behind
later ADR decisions.

Pre-M11 is complete as the vision/convolution and captured-grid debt cleanup
gate. The model-corpus report now emits a `pre_m11` section and the generated
native-provider corpus freezes M11 entry debt as `deferred_convolution=65`
wrapper-level `extern_kernels.convolution` calls plus `captured_grid=48`
explicit `contract_error/grid` captured-kernel blockers. M11 must define the
concrete convolution/operator contract and report fields before lowering or
runtime changes.

The Pre-M11 vision policy boundary includes conv2d, 1x1 conv,
depthwise/grouped conv, pool, resize, concat, slice, YOLO decode/postprocess,
NMS, NCHW/NHWC, channels-last, and stride/padding/dilation. Explicit TVM extern
is allowed when reported; implicit PyTorch fallback is disallowed. Historical
`deferred_attention=2` remains wrapper-family history only because both
observed SDPA calls are runtime-resolved under M10 `native_decomposed`.

M11.0-M11.2 are implemented as the artifact-only convolution entry. The new
`vision.py` surface freezes `vision_*` report fields, `m11_v1` contract
version metadata, and the conv contract ids `conv2d_nchw_static_v1`,
`conv2d_1x1_nchw_static_v1`, `depthwise_conv2d_nchw_static_v1`, and
`grouped_conv2d_nchw_static_v1`. Wrapper `extern_kernels.convolution` calls are
classified from full wrapper AST tensor metadata and accepted only for static
fp32 rank-4 NCHW/OIHW conv2d without bias, transpose, or output padding and
with formula-matching output shape.

Accepted M11.2 conv records materialize an explicit artifact-only packed call
to `tvm.contrib.triton_tvm.extern_conv2d` with zero runtime launches, one
artifact call, unavailable runtime replacement, and
`vision_performance_claim=false`. This preserves schema-v1 top-level buckets,
historical `op_family="deferred_convolution"` accounting, captured-kernel
counts, M10 attention runtime boundaries, and `full_tvm_runnable=0`. M11.3
is now implemented as taxonomy-only captured-grid classification.

M11.3 adds `m11_3_grid_taxonomy_v1` and per-kernel `m11_grid_*` fields for
nontranslated `blocker_class="grid"` records. The regenerated M11.3 corpus
keeps 89 captured kernels, 41 translated, 48 explicit Grid2D fallbacks, 65
artifact-only conv records, 2 M10 runtime-resolved attention records, and 0
full TVM runnable models. The 48 grid blockers split into 17
`grid_conv_adjacent_pointwise`, 14 `grid_bn_silu_fusion`, and 17
`grid_concat_split` records. M11.3 does not add Grid2D lowering, a runtime
provider, conv runtime, performance claims, or model closure.

M11.4 adds the first narrow runtime-resolved vision slice without broadening
the convolution contract family. Only the exact ViT patch-embedding conv
(`1x3x32x32` input, `64x3x16x16` weight, stride `16`, output `1x64x2x2`) is
runtime-resolved under the opt-in correctness-only
`python_torch_host_staged` provider for
`tvm.contrib.triton_tvm.extern_conv2d`. All other accepted conv records remain
artifact-only and report the M11.4 provider-scope unsupported reason. Provider
metadata records ABI v1, one runtime launch, one artifact call, host-staging
bytes, and `vision_performance_claim=false`.

M11.P adds a separate same-machine diagnostic baseline for that ViT patch
provider path against Torch CUDA conv2d. It reports p50/p95 latency and
allclose correctness, but remains outside corpus provider performance claims
and full-model runnable accounting.

M11.5 hardens that narrow runtime/perf evidence without widening the provider:
the vision conv2d validator checks exact byte accounting, runtime launch count,
runtime scope, and performance-claim boundaries; corpus reports add
`m11.report_cache_invariants`, `m11.corpus_diff_guard`, and
`m11.m10_attention_boundary`; M11.P reports hardening checks while keeping the
provider performance claim false.

M11.6 moves the milestone from artifact/hardening into runtime readiness. The
conv provider scope is now `m11_6_static_conv2d_runtime_scope_v1`: N=1 fp32
NCHW/OIHW regular conv with groups=1, dilation=1, stride 1/2/16, padding 0/1,
plus 1x1 stride-1 padding-0 conv. This provider is still
`python_torch_host_staged`, correctness-only, and excluded from performance
claims; the regenerated corpus observes 65/65 conv wrapper records
runtime-resolved only under that explicit provider.

M11.6 also introduces the native readiness contract
`pointwise_grid2d_static_v1` for static f32 Grid2D pointwise/fusion scaffold
artifacts. Conv-adjacent and BN/SiLU Grid2D blockers are marked
`native_tvm_grid2d` ready with zero host staging and covered by a separate
dashboard; concat/split Grid2D blockers remain explicit unsupported runtime
records. The regenerated corpus reports 31 Grid2D artifact/native
runtime-ready records, 17 concat/split blockers, zero silent fallback, and
`full_tvm_runnable=0`.
