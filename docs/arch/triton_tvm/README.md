# Triton TVM Backend

This directory is the compact engineering entry point for the Triton-to-TVM
backend implementation that lives in the TVM tree.

Implementation source of truth:

- `python/tvm/contrib/triton_tvm/`
- `tests/python/contrib/test_triton_tvm*.py`

Active planning, long devlogs, ADRs, experiment reports, generated corpus
outputs, and resume contexts live in the companion workspace:

- `/home/liyh/xdb/triton-tvm-workbench`

Read these TVM-side files when changing implementation:

- `design.md`
- `capability_matrix.md`
- `contracts.md`

Current implementation boundary:

- Historical M8.5 captured Triton-kernel coverage was 41/89 translated with 48
  explicit `grid` fallbacks. The current M11.7 corpus is 72/89 translated with
  17 explicit Grid2D concat/split fallbacks after native Grid2D artifact
  materialization.
- Pre-M9 wrapper-level extern visibility is active: GEMM/GEMM+bias extern
  calls are M9 entry debt, while convolution and attention wrapper calls remain
  deferred.
- M9 entry is MatmulSemantics-first: `tt.dot`, wrapper-level extern GEMM, and
  future graph matmul sources should converge on `MatmulContract` before
  target policy chooses native schedule, explicit TVM artifact extern GEMM, or
  unsupported fallback.
- M9.1-M9.8 implement `matmul_minimal` for synthetic/static exact unmasked
  rank-2 `tt.dot`: first validating an unresolved semantic
  `T.sblock("matmul")`, then selecting a correctness-first
  `native_tir_schedule` with fp16/bf16 to fp32 CUDA runtime coverage. M9.7
  adds gated `cuda_block_tile_8x8_serial_k_v1` for eligible M/N-multiple-of-8
  shapes while non-eligible accepted shapes keep
  `cuda_block_per_output_serial_k_v1`. Wrapper `extern_kernels.mm`
  materializes as an explicit `tvm.contrib.triton_tvm.extern_gemm` packed-call
  artifact. M9.6 adds an opt-in correctness-only
  `python_torch_host_staged` provider proof; the default remains artifact-only,
  and the provider makes no performance, TVM-only backend, or full-model
  runnable claim. M9.8 adds a separate TinyMNISTMLP staged diagnostic baseline:
  two runtime-resolved host-staged extern GEMMs plus one TVM pointwise ReLU on
  the full MNIST test split. This diagnostic report does not change captured
  corpus metrics or make a native performance claim.
- M12.2 adds a ViT-only Native TVM wrapper matmul closure: the observed
  fixed-shape fp32 `vit_tiny_random` 4/4 `extern_gemm` and 3/3
  `extern_addmm_bias` records can be scoped to `native_tvm_matmul`, using the
  correctness-first serial-K schedule `cuda_block_per_output_serial_k_v1` with
  zero host staging and no performance claim.
- M12.3 adds the fixed-shape ViT E2E correctness runner. It compares the
  admitted runner path against Torch eager CUDA and passes P0 with allclose
  correctness, complete provider reporting, no silent fallback, zero host
  staging, 7 Native TVM wrapper matmul/addmm launches, 1 `device_torch_cuda`
  conv, and 1 `native_decomposed` attention replay.
- M11.7 extends the vision stack with an opt-in `device_torch_cuda` conv2d
  provider for the observed static N=1 conv scope. It keeps the explicit
  `tvm.contrib.triton_tvm.extern_conv2d` packed-call boundary, moves runtime
  records off host staging, materializes the 31 Grid2D-ready pointwise/BN-SiLU
  captured kernels as validated `pointwise_grid2d_static_v1` native TVM
  artifacts, adds an M11.7 hotpath dashboard, and keeps strict full TVM
  runnable model closure at 0.
- Pre-M12 separates historical entry debt from current M12 debt. Pre-M11
  remains frozen as `deferred_convolution=65` and `captured_grid=48`; current
  M12 entry is 89/72/17 captured-kernel accounting, 17 YOLO concat/split
  Grid2D blockers, 11 artifact-only `extern_gemm` calls, 3 artifact-only
  `extern_addmm_bias` calls, explicit `device_torch_cuda` conv provider
  evidence, and `full_tvm_runnable_models=0`.
- M12.0-M12.14 are complete. Policy vocabulary is frozen, the fixed-shape ViT
  execution plan is reported, ViT matmul/addmm artifact blockers are closed,
  the fixed-shape correctness runner passes P0, the E2E dashboard passes P1
  but misses P2, M12.6 hardens the provider/report surface, and M12.7 profiles
  native matmul provider cost while validating the fixed-shape
  `no_per_call_sync` path. M12.8 reruns the P2 gate, preserves correctness and
  provider invariants, but fails P2 at `2.4731x / 2.4193x` `torch.compile`
  p50/p95. M12.9 improves the best measured path to
  `2.2841x / 2.2618x` `torch.compile` but still misses P2. M12.10 labels the
  fixed-shape fused-QKV artifact as diagnostic-only, and M12.11 freezes the
  next backend-general route as real `tl.dot` lowering first, runtime overhead
  second, and schedule handoff third. M12.12 proves the real `tl.dot` bridge,
  and M12.13 measures reusable runtime overhead with a standalone
  `pointwise_flat` artifact. M12.14 proves real JIT `tt.dot` schedule handoff:
  TensorCore selection for 16x16x16 fp16 and reusable tiled selection for
  8x8x16 fp16, with the M12.9 wrapper schedule excluded from handoff
  readiness. M12.P iteration `provider_runtime_glue_prebound_packed_call_v1`
  adds backend-general prebound packed-call dispatch for the fixed-shape native
  matmul provider and passes unchanged P2 at `1.5043x / 1.4946x`
  `torch.compile` p50/p95. Post-closure iteration
  `provider_runtime_glue_minimal_record_v1` keeps P2 passing at
  `1.4856x / 1.4179x` without expanding claims. The current report keeps
  `strict_full_tvm_native=false`, `full_tvm_runnable_models=0`, 7 Llama
  artifact-only GEMMs, and 17 YOLO concat/split blockers.
- Follow-on branch `m12p-real-ttdot-native-matmul-provider` is closed without
  implementation. The route remains a future contract-expansion reference for
  real Triton JIT `tl.dot` -> textual `tt.dot` -> `TTIRReader` ->
  `MatmulSemantics(source_kind=real_jit_tt_dot)` -> `TargetMatmulPolicy`,
  but current fixed-shape ViT wrapper fp32 `native_tvm_matmul` calls do not
  satisfy the real-`tt.dot` TensorCore envelope.

Read these workbench files when starting a new milestone:

- `/home/liyh/xdb/triton-tvm-workbench/CURRENT.md`
- The active context path named by `CURRENT.md`
- `/home/liyh/xdb/triton-tvm-workbench/CAPABILITY_MATRIX.md`
- `/home/liyh/xdb/triton-tvm-workbench/adr/*.md`
- `/home/liyh/xdb/triton-tvm-workbench/devlog/milestones/<milestone>.md`

The historical M0 to M7.5 devlog is frozen at
`/home/liyh/xdb/triton-tvm-workbench/devlog/archive/m00_to_m75_full.md`.
