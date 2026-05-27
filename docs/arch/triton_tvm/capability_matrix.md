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
| `matmul_minimal` | supported in M9.1-M9.8 semantic/static/native/extern-proof/toy-graph scope; M9.P Phase 0-3 hardening complete; M12.2 ViT native wrapper closure complete; M12.6 provider-id hardening complete; M12.12 real JIT `tl.dot` bridge complete; M12.14 real `tt.dot` schedule handoff complete; M12.P fixed-shape ViT P2 provider path complete | exact unmasked rank-2 `tt.dot` -> unresolved semantic validation block -> native schedule policy with TensorCore, SIMT16, and serial-K fallback candidates; wrapper `extern_kernels.mm` -> explicit packed-call artifact with artifact-only default and opt-in correctness-only `python_torch_host_staged` runtime proof; wrapper `extern_kernels.addmm` -> minimal `bias_add` packed-call artifact; M12.2 admits only observed `vit_tiny_random` fp32 wrapper GEMM/addmm shapes through correctness-first `native_tvm_matmul` serial-K TIR with zero host staging; M12.6 rejects unknown/stale wrapper GEMM provider ids explicitly; M12.12 admits real Triton JIT three-operand `tt.dot` as `source_kind=real_jit_tt_dot` for the supported row-major fp16/bf16 input and fp32 accumulator/output case; M12.14 proves real JIT `tt.dot` semantics can select reusable TensorCore/tiled schedules without wrapper-specific readiness; M12.P adds backend-general prebound packed-call dispatch for the fixed-shape ViT native matmul provider and passes the unchanged P2 gate without strict full-native TVM or full-runnable model claims |
| `attention_vit_full_v1` | supported in M10.1-M10.4 artifact/provider/native-decomposed scope | observed wrapper SDPA ViT full-attention call -> semantic extraction -> explicit packed-call artifact `tvm.contrib.triton_tvm.extern_attention_sdpa`; opt-in correctness-only `python_torch_host_staged` provider; opt-in correctness-only native decomposed QK^T, row softmax, and AV path via `native_decomposed`; no performance or full-model runnable claim |
| `attention_llama_causal_prefill_v1` | supported in M10.5 native-decomposed correctness scope | observed wrapper SDPA Llama causal prefill call -> semantic extraction for already-RoPE'd Q/K tensors, rank-4 V, additive causal mask metadata, `is_causal=False`, and numeric scale; opt-in correctness-only native decomposed QK^T, masked row softmax, and AV path via `native_decomposed`; no RoPE, KV-cache, decode, performance, or full-model runnable claim |
| `conv2d_nchw_static_v1` | supported in M11.7 device-provider hotpath scope | static fp32 rank-4 NCHW/OIHW wrapper `extern_kernels.convolution` metadata -> explicit packed-call artifact `tvm.contrib.triton_tvm.extern_conv2d`; no bias/transposed/output-padding and output shape formula validation; opt-in `python_torch_host_staged` remains correctness-only; opt-in `device_torch_cuda` marks the observed N=1 groups=1 dilation=1 stride 1/2/16 padding 0/1 regular conv scope runtime-resolved with zero host staging and `performance_eligible`; full TVM runnable closure remains false |
| `conv2d_1x1_nchw_static_v1` | supported in M11.7 device-provider hotpath scope | 1x1 static fp32 rank-4 NCHW/OIHW wrapper conv artifact metadata; `device_torch_cuda` admits N=1 groups=1 stride=1 padding=0 as the M11.7 1x1 hotpath/conv-as-matmul classification surface, resolving 25/25 observed 1x1 records with zero host staging while keeping full-model runnable closure false |
| `depthwise_conv2d_nchw_static_v1` | supported in M11.2 artifact-only scope | depthwise static fp32 rank-4 NCHW/OIHW wrapper conv artifact metadata; no runtime or performance claim |
| `grouped_conv2d_nchw_static_v1` | supported in M11.2 artifact-only scope | grouped static fp32 rank-4 NCHW/OIHW wrapper conv artifact metadata; no runtime or performance claim |
| `pointwise_grid2d_static_v1` | supported in M11.7 native Grid2D translated artifact scope | static Grid2D affine f32 pointwise/fusion scaffold for conv-adjacent and BN/SiLU families; `native_tvm_grid2d`, zero host staging, perf dashboard measured 5/5 allclose cases; M11.7 materializes 31 captured records as translated native artifacts; concat/split Grid2D remains explicitly unsupported |

## Pre-M12 Entry Baseline

- Generated workbench report: `reports/pre_m12/report.json`.
- Historical Pre-M11 entry debt remains frozen as `deferred_convolution=65`
  and `captured_grid=48`.
- Current M12 captured-kernel baseline is 89 total, 72 translated, and
  17 explicit YOLO concat/split Grid2D fallbacks.
- Strict/full closure still has 11 artifact-only `extern_gemm` calls and
  3 artifact-only `extern_addmm_bias` calls.
- Runtime-resolved provider evidence includes 65/65 `device_torch_cuda`
  conv records and 2/2 M10 `native_decomposed` attention records.
- Native TVM conv scheduling is strict-native debt only; `device_torch_cuda`
  remains an explicit provider through the packed-call boundary.
- `full_tvm_runnable_models=0`.
- M12 is scoped to `vit_tiny_random` fixed-shape E2E performance readiness.
  Completion requires P2: warmed cache, zero host staging, complete provider
  mix, p50/p95 reported, and p50/p95 each `<= 2x`
  `torch.compile`/Inductor. YOLO closure, arbitrary ViT variants, dynamic
  shape, general concat/split closure, strict full-native replacement claims,
  and native TVM conv scheduling claims are non-goals for M12.

## M12.0-M12.P ViT Fixed-Shape E2E Surface

- Generated workbench reports:
  `reports/m12/m12_0_policy_freeze/report.json`,
  `reports/m12/m12_1_vit_execution_plan/report.json`, and
  `reports/m12/m12_2_vit_native_matmul_closure_corpus/report.json`.
  M12.3 adds `reports/m12/m12_3_vit_fixed_shape_e2e_runner/report.json`,
  M12.4 adds `reports/m12/m12_4_vit_e2e_dashboard/report.json`, M12.5/M12.5.5
  add the optimization review/freeze reports, and M12.6 adds
  `reports/m12/m12_6_hardening/report.json`. M12.7 adds
  `reports/m12/m12_7_native_matmul_provider_cost/report.json`. M12.8 adds
  `reports/m12/m12_8_p2_gate_decision/report.json`. M12.9/M12.10/M12.11 add
  `reports/m12/m12_9_provider_runtime_optimization/report.json` and
  `reports/m12/m12_10_route_cleanup/report.json` and
  `reports/m12/m12_11_backend_general_replan/report.json`. M12.12 adds
  `reports/m12/m12_12_real_tl_dot_bridge_harness/report.json`. M12.13 adds
  `reports/m12/m12_13_runtime_overhead_general_measurement/report.json`.
  M12.14 adds `reports/m12/m12_14_real_tt_dot_schedule_handoff/report.json`.
  M12.15 adds `reports/m12/m12_15_fixed_shape_vit_p2_reentry/report.json`.
  M12.16 adds `reports/m12/m12_16_backend_general_p2_gap_analysis/report.json`.
  M12.P adds
  `reports/m12/m12_p_profile_optimization_loop/provider_runtime_glue_prebound_packed_call_v1/report.json`
  and
  `reports/m12/m12_p_profile_optimization_loop/provider_runtime_glue_minimal_record_v1/report.json`.
- M12.0 freezes the closure and performance vocabulary:
  `runtime_resolved_smoke`, `performance_ready_e2e`,
  `strict_full_tvm_native`, and P0/P1/P2/P3.
- M12.1 records the ordered fixed-shape ViT execution plan: 7 captured kernels,
  9 wrapper calls, 7 wrapper matmul/addmm calls, launch estimate 16, and
  `host_staging_bytes=0`.
- M12.2 resolves 4/4 ViT `extern_gemm` and 3/3 ViT `extern_addmm_bias` records
  through `native_tvm_matmul`, schedule
  `cuda_block_per_output_serial_k_v1`, runtime claim
  `native_tvm_fixed_shape_correctness_only`, and no performance claim.
- The native wrapper matmul runtime is scoped to
  `--extern-gemm-runtime-model-case vit_tiny_random`; Llama GEMM remains
  artifact-only unless a later milestone admits it.
- M12.3 runs fixed-shape `vit_tiny_random` through an explicit correctness
  runner and passes P0 against Torch eager CUDA. Provider counts are
  7 explicit captured-kernel harness launches, 1 `device_torch_cuda` conv,
  1 `native_decomposed` attention replay, and 7 `native_tvm_matmul` wrapper
  matmul/addmm launches. Host staging remains 0 and no performance claim is
  made.
- M12.4 measures fixed-shape E2E latency and passes P1 but misses P2:
  `triton_tvm_e2e` is `2.7478x / 2.6755x` `torch.compile` p50/p95.
- M12.5/M12.5.5 identify and freeze `native_tvm_matmul_provider_cost` as the
  primary P2 recovery target with p50/p95 gap
  `0.247040 / 0.240926 ms`.
- M12.6 hardens the E2E report/provider surface without optimizing
  performance. Unknown extern GEMM provider ids fail explicitly, and the
  generated hardening report guards provider ids, dashboard/freeze schema,
  report/cache, stale fallback reasons, hidden host staging, strict/full-native
  claims, performance-ready gating, and corpus diff baselines.
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
  `tl.dot` bridge harness as the next default.
- M12.12 proves the real `tl.dot` bridge: a real `@triton.jit` kernel lowers
  through `lower_to_ttir` to textual `tt.dot`, `TTIRReader` records dot
  operands/results/attrs, `MatmulSemantics.source_kind=real_jit_tt_dot`, and
  `matmul_minimal` accepts the supported row-major fp16/fp32 case. Unsupported
  dtype, layout, masked/bounds, and epilogue cases remain explicit negative
  boundaries.
- M12.13 measures reusable runtime overhead with a standalone `pointwise_flat`
  artifact instead of ViT wrapper replay, identifies `artifact_run_wall_ms` as
  the dominant bucket, and keeps P2/performance and schedule-handoff claims
  false.
- M12.14 completes real `tt.dot` schedule handoff. A real JIT 16x16x16 fp16
  `tt.dot` case selects TensorCore schedule
  `cuda_ptx_mma_m8n8k4_warp_tile_16x16_fp16fp32_v1`; a real JIT 8x8x16 fp16
  case selects reusable tiled schedule `cuda_block_tile_8x8_serial_k_v1`; the
  M12.9 wrapper-specific schedule is excluded from handoff readiness.
- M12.15 reruns the fixed-shape ViT P2 gate using
  `triton_tvm_e2e_m129_backend_pass` as the only P2-eligible Triton TVM path.
  It preserves correctness, complete provider mix, zero host staging, and
  fused-QKV exclusion, but P2 fails at `2.5293x / 2.4060x`
  `torch.compile` p50/p95.
- M12.16 analyzes that P2 miss without changing runtime/schedule code. The
  P2 excess is `0.183552 / 0.153272 ms` p50/p95, the dominant measured bucket
  is `native_tvm_matmul_provider` (`0.516780 ms` provider total,
  `0.378080 ms` kernel event, `0.138700 ms` provider glue), residual E2E is
  recorded as `0.360308 ms`, and M12.P is selected as the closure loop.
- M12.P iteration `provider_runtime_glue_prebound_packed_call_v1` adds
  backend-general prebound packed-call dispatch for the fixed-shape native
  matmul provider. It preserves correctness allclose, complete provider mix,
  no silent fallback, and `host_staging_bytes=0`, and passes the unchanged P2
  gate: `triton_tvm_e2e_m12p_prebound_packed_call` is
  `0.500944 / 0.530693 ms` p50/p95 versus `torch.compile`/Inductor
  `0.333008 / 0.355069 ms`, or `1.5043x / 1.4946x`.
- `performance_ready_e2e=true` for the M12 fixed-shape provider mix.
  `strict_full_tvm_native=false` and `full_tvm_runnable_models=0` remain.

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
- minimal addmm/bias artifact candidates: `extern_addmm_bias: 3`
- deferred wrapper debt: `deferred_convolution: 65`,
  `deferred_attention: 2`
- full TVM runnable models after the extern gate: 0

M9 owns the matmul entry boundary first. M9 entry is MatmulSemantics-first:

- `tt_dot`, `wrapper_extern_gemm`, and future graph matmul sources converge on
  `MatmulContract`.
- M12.12 later keeps synthetic/static `tt_dot`, wrapper `extern_gemm`, and real
  JIT `real_jit_tt_dot` separate in reports while converging the accepted
  native cases through the same `MatmulSemantics` contract.
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
  `extern_runtime_replacement="not_available"`; M9.PE later materializes
  `extern_addmm_bias` as a minimal `bias_add` packed-call artifact. Corpus diff
  coverage and an opt-in minimal perf scaffold are present. Runtime
  cublas/cublaslt replacement, activation epilogues, arbitrary epilogues,
  performance optimization, and model closure remain follow-on work.
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
- M9.P Phase 0-3 is complete after M9.8 as a horizontal CUDA matmul
  performance hardening track. It keeps `MatmulSemantics` stable and uses
  `matmul_perf_core_v1` as a target-policy envelope for native CUDA schedules:
  `tt_dot` native path preferred, fp16/bf16 inputs, fp32 accumulation/output,
  row-major A/C, row-major or transposed-weight-view B, rank-2 exact static
  shapes with M/N/K multiples of 16 or 32.
- M9.PA adds `cuda_ptx_mma_m8n8k4_warp_tile_16x16_fp16fp32_v1` for eligible
  fp16 `tt_dot` envelope shapes. M9.PC adds
  `cuda_block_tile_16x16_simt_v1` as the bf16-capable non-TensorCore fallback
  candidate under the same envelope. M9.PB reports
  `torch_cuda_cublas_baseline_v1` as a real CUDA dashboard baseline while TVM
  packed cuBLAS/cuBLASLt runtime replacement remains unavailable. M9.PD reports
  5 measured TensorCore-selected envelope cases and 2 TinyMLP original coverage
  records marked unavailable. M9.PE admits only the minimal wrapper
  `extern_kernels.addmm` `bias_add` artifact and keeps epilogues outside the
  native performance envelope.
- M9.P may benchmark QK^T, AV, projection, and MLP/FFN shape suites as matmul
  performance cases only. Masked softmax, RoPE, KV cache, attention ABI, and
  attention runtime remain M10 scope.
- Pre-M10 freezes attention ABI/report vocabulary without adding attention
  runtime support. Wrapper SDPA stays `deferred_attention` and is classified
  into `attention_vit_full_v1`, `attention_llama_causal_prefill_v1`, or
  `attention_llama_decode_v1`. Current report coverage observes 2 attention
  calls: one ViT full attention and one Llama causal prefill. Decode is covered
  by synthetic tests only.
- M10.1-M10.4 are complete for `attention_vit_full_v1`: wrapper SDPA semantic
  extraction, explicit packed-call artifact
  `tvm.contrib.triton_tvm.extern_attention_sdpa`, and opt-in
  correctness-only `python_torch_host_staged` provider. M10.4 adds the opt-in
  correctness-only `native_decomposed` provider for ViT QK^T, row softmax, and
  AV. The native-provider corpus reports 1 ViT runtime-resolved attention
  record, 1 Llama deferred attention record, 1 materialized attention artifact,
  and 0 full TVM runnable models while keeping captured-kernel accounting at
  89/41/48.
- M10.5 is complete for `attention_llama_causal_prefill_v1`: observed wrapper
  SDPA semantic extraction, additive causal mask metadata, and opt-in
  correctness-only `native_decomposed` provider for QK^T, masked row softmax,
  and AV. Q/K are treated as already-RoPE'd upstream tensors; RoPE and
  KV-cache runtime remain deferred.
- M10.6 keeps `attention_llama_decode_v1` synthetic/report-only until a real
  corpus decode wrapper call appears.
- The M10.5 native-provider corpus reports 2 native runtime-resolved attention
  records, 0 artifact-only attention calls, and 0 full TVM runnable models
  while keeping captured-kernel accounting at 89/41/48.
- M10 attention baseline is complete as a separate same-machine report:
  `m10_attention_native_decomposed_vs_torch_sdpa_v1` measures fixed
  `attention_vit_full_v1` and `attention_llama_causal_prefill_v1` shapes
  against Torch CUDA SDPA. Corpus provider records still keep
  `attention_performance_claim=false`.
- M10 hardening is complete. Unknown attention provider ids fail explicitly;
  attention records and TIR attrs carry launch/byte accounting; report
  summaries aggregate launch, memory, host-staging, and unsupported runtime
  reason counters; and runtime-resolved attention records are protected from
  stale fallback reason pollution during normalization.
- The regenerated native-provider corpus reports
  `m10_runtime_hardened_v1`, 2 runtime launches, 2 artifact calls, 25,600 total
  IO bytes, 154,624 intermediate buffer bytes, 0 host-staging bytes, and 0 full
  TVM runnable models.
- The old M10.5 label is retired.
- Pre-M11 is complete as the vision/convolution and captured-grid debt cleanup
  gate. The model-corpus report now emits a `pre_m11` section that freezes M11
  entry debt as `deferred_convolution=65` wrapper-level
  `extern_kernels.convolution` calls and `captured_grid=48` explicit
  captured-kernel `contract_error/grid` blockers. Both debt families affect
  `vit_tiny_random` and `yolov8n_yaml_random`.
- Pre-M11 freezes the vision policy boundary for conv2d, 1x1 conv,
  depthwise/grouped conv, pool, resize, concat, slice, YOLO
  decode/postprocess, NMS, NCHW/NHWC, channels-last, and
  stride/padding/dilation. Explicit TVM extern is allowed when reported;
  implicit PyTorch fallback is disallowed.
- Historical `deferred_attention=2` remains visible as wrapper-family history,
  but both observed SDPA calls are runtime-resolved under M10
  `native_decomposed`; attention is not M11 vision entry debt.
- M11.0-M11.2 are complete as artifact-only convolution entry. The report
  vocabulary uses `vision_*` fields and `vision_contract_version="m11_v1"`;
  accepted wrapper convs materialize exactly one packed call to
  `tvm.contrib.triton_tvm.extern_conv2d` with
  `vision_runtime_status="artifact_only"`, `vision_runtime_launch_count=0`,
  `vision_artifact_call_count=1`, and
  `vision_performance_claim=false`.
- The generated M11.2 corpus reports 89 captured kernels, 41 translated, 48
  captured-grid fallbacks, 65 observed wrapper conv records, 65 artifact-only
  conv records, 2 M10 runtime-resolved attention records, and 0 full TVM
  runnable models.
- M11.3 is complete as captured-grid taxonomy only. The report vocabulary adds
  `m11_3_grid_taxonomy_v1` and per-kernel `m11_grid_*` fields for nontranslated
  `blocker_class="grid"` records. The regenerated corpus keeps 89 captured
  kernels, 41 translated, 48 explicit Grid2D fallbacks, 65 artifact-only conv
  records, 2 M10 runtime-resolved attention records, and 0 full TVM runnable
  models.
- M11.3 captured-grid split: 17 `grid_conv_adjacent_pointwise`, 14
  `grid_bn_silu_fusion`, 17 `grid_concat_split`, 0 `grid_pool_or_softmax`, 0
  `grid_yolo_decode_postprocess`, and 0 `grid_other_multidim_pointwise`.
- M11.4/M11.P complete the first narrow runtime and diagnostic baseline slice:
  the exact ViT patch conv is runtime-resolved through the opt-in
  correctness-only `python_torch_host_staged`
  `tvm.contrib.triton_tvm.extern_conv2d` provider, and a same-machine baseline
  compares it against Torch CUDA conv2d. The regenerated corpus keeps 89
  captured kernels, 41 translated, 48 explicit Grid2D fallbacks, 65 wrapper
  conv records, 1 runtime-resolved ViT patch conv, 64 artifact-only convs, 65
  artifact calls, 1 vision runtime launch, 2 M10 runtime-resolved attention
  records, and 0 full TVM runnable models. The latest baseline measured 1/1
  allclose case with TVM p50 189.0550 us and Torch conv2d p50 36.4800 us;
  provider performance claim remains false.
- M11.5 hardening is complete without scope expansion: contract validators
  reject runtime scope, launch count, performance-claim, and exact
  byte-accounting regressions; reports include `m11.report_cache_invariants`,
  `m11.corpus_diff_guard`, and `m11.m10_attention_boundary`.
- M11.6 is complete as vision runtime/Grid2D readiness. The regenerated corpus
  keeps 89 captured kernels, 41 translated, and 48 explicit Grid2D blockers,
  but moves conv provider coverage to the static correctness envelope:
  65/65 wrapper conv records are runtime-resolved under
  `python_torch_host_staged`, with `vision_performance_claim=false`.
- M11.6 adds `pointwise_grid2d_static_v1` and native
  `native_tvm_grid2d` readiness for Grid2D pointwise/fusion records:
  48 Grid2D blockers classified, 31 artifact/native runtime-ready, 17
  concat/split records explicitly unsupported, and zero silent fallback. The
  separate dashboard reports 5/5 measured native Grid2D allclose cases with no
  host staging.
- M11.7 moves the 31 Grid2D-ready records into captured-kernel translated
  accounting by materializing validated `pointwise_grid2d_static_v1` native TVM
  artifacts. The regenerated M11.7 corpus now reports 89 captured kernels, 72
  translated, and 17 explicit Grid2D fallbacks, with the remaining blockers all
  carrying `grid2d_concat_split_multi_output_layout_deferred_m11_6`.
- Full-model closure remains future work: ViT has a full diagnostic readiness
  smoke, YOLO has a partial smoke but still has 17 concat/split Grid2D runtime
  blockers, and `full_tvm_runnable` remains 0.

The observed M7.5 `grid` blockers and Pre-M9 deferred convolution/attention
wrapper calls remain separate deferred debt classes and should not be counted as
M9 matmul acceptance unless a later ADR changes that boundary.
