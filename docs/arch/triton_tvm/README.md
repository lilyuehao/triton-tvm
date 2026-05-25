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

- M8.5 captured Triton-kernel coverage remains 41/89 translated with 48
  explicit `grid` fallbacks.
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

Read these workbench files when starting a new milestone:

- `/home/liyh/xdb/triton-tvm-workbench/CURRENT.md`
- The active context path named by `CURRENT.md`
- `/home/liyh/xdb/triton-tvm-workbench/CAPABILITY_MATRIX.md`
- `/home/liyh/xdb/triton-tvm-workbench/adr/*.md`
- `/home/liyh/xdb/triton-tvm-workbench/devlog/milestones/<milestone>.md`

The historical M0 to M7.5 devlog is frozen at
`/home/liyh/xdb/triton-tvm-workbench/devlog/archive/m00_to_m75_full.md`.
