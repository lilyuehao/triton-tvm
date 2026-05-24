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

M9 starts matmul/GEMM/epilogue policy work from that observed extern GEMM
surface. M9 entry work must define contract boundaries before semantic expansion
and must not silently absorb deferred `grid`, convolution, or attention debt.
