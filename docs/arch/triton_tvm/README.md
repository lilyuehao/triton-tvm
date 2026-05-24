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

Read these workbench files when starting a new milestone:

- `/home/liyh/xdb/triton-tvm-workbench/CURRENT.md`
- The active context path named by `CURRENT.md`
- `/home/liyh/xdb/triton-tvm-workbench/CAPABILITY_MATRIX.md`
- `/home/liyh/xdb/triton-tvm-workbench/adr/*.md`
- `/home/liyh/xdb/triton-tvm-workbench/devlog/milestones/<milestone>.md`

The historical M0 to M7.5 devlog is frozen at
`/home/liyh/xdb/triton-tvm-workbench/devlog/archive/m00_to_m75_full.md`.
