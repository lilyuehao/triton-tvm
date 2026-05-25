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
`extern_runtime_replacement="not_available"`. Addmm/bias remains deferred as
`extern_addmm_bias` with visible `bias_add` epilogue metadata. Runtime
cublas/cublaslt replacement, TensorCore/vendor scheduling, performance
optimization, bias/add epilogue lowering, attention, convolution, and model
closure remain follow-on work.

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
