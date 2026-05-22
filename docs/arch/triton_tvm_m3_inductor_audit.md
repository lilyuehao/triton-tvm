# Triton TVM M3 Inductor Pointwise TTIR Audit

This document records the M3 offline audit of TorchInductor-generated pointwise
Triton kernels.  M3 is a coverage and gap-discovery milestone only: it does not
build or run TVM output, does not hook Inductor runtime, and does not expand the
M2.5 translator contract.

## Environment

- TVM tree: `/home/liyh/xdb/tvm`
- Conda environment: `tvm-0.24.0`
- PyTorch version: `2.12.0+cu130`
- Triton version: `3.7.0`
- Target used to generate Inductor kernels: CUDA

## Command

```bash
conda run -n tvm-0.24.0 \
  python -m tvm.contrib.triton_tvm.inductor \
  --builtin-corpus \
  --out-dir /tmp/triton_tvm_m3 \
  --min-kernels 20
```

The local run wrote:

- `/tmp/triton_tvm_m3/kernels/*.py`: 24 extracted Inductor Triton source blocks.
- `/tmp/triton_tvm_m3/ttir/*.ttir`: 24 optimized TTIR dumps.
- `/tmp/triton_tvm_m3/report.json`: machine-readable M3 audit report.
- `/tmp/triton_tvm_m3/report.md`: full generated Markdown report.

## Summary

- Total pointwise kernels audited: 24.
- Kernels translated by the current M2.5 translator: 0.
- Unsupported buckets:
  - `unsupported_ttir_op`: 24.
- Collection errors: none.

The dominant blocker is Inductor's default masked `tl.load(..., mask)` lowering
without an explicit `other`.  This maps to TTIR `tt.load` without a third operand,
which M2.5 intentionally rejects to preserve Triton undefined-lane semantics.

## Builtin Corpus

The audit corpus covers the planned M3 pointwise shapes:

| Case | Kernel | Loads | Stores | First unsupported reason |
|---|---|---:|---:|---|
| `add` | `triton_poi_fused_add_0` | 2 | 1 | masked `tt.load` without `other` |
| `add_mul` | `triton_poi_fused_add_mul_0` | 2 | 1 | masked `tt.load` without `other` |
| `three_input_chain` | `triton_poi_fused_add_mul_sub_0` | 3 | 1 | masked `tt.load` without `other` |
| `scalar_alpha` | `triton_poi_fused_add_0` | 1 | 1 | masked `tt.load` without `other` |
| `two_scalars` | `triton_poi_fused_add_mul_0` | 1 | 1 | masked `tt.load` without `other` |
| `relu_add` | `triton_poi_fused_add_relu_0` | 2 | 1 | masked `tt.load` without `other` |
| `sigmoid_mul` | `triton_poi_fused_mul_sigmoid_0` | 2 | 1 | masked `tt.load` without `other` |
| `tanh_shift` | `triton_poi_fused_add_tanh_0` | 1 | 1 | masked `tt.load` without `other` |
| `exp_log_abs` | `triton_poi_fused_abs_add_exp_log_0` | 1 | 1 | masked `tt.load` without `other` |
| `sin_cos` | `triton_poi_fused_add_cos_sin_0` | 2 | 1 | masked `tt.load` without `other` |
| `where_cmp` | `triton_poi_fused_gt_where_0` | 2 | 1 | masked `tt.load` without `other` |
| `clamp_add` | `triton_poi_fused_add_clamp_0` | 2 | 1 | masked `tt.load` without `other` |
| `reciprocal_abs` | `triton_poi_fused_abs_add_reciprocal_0` | 1 | 1 | masked `tt.load` without `other` |
| `sqrt_rsqrt` | `triton_poi_fused_abs_add_rsqrt_sqrt_0` | 2 | 1 | masked `tt.load` without `other` |
| `pow2_add` | `triton_poi_fused_add_mul_0` | 2 | 1 | masked `tt.load` without `other` |
| `fp16_add` | `triton_poi_fused_add_0` | 2 | 1 | masked `tt.load` without `other` |
| `bf16_add` | `triton_poi_fused_add_0` | 2 | 1 | masked `tt.load` without `other` |
| `int32_add` | `triton_poi_fused_add_0` | 2 | 1 | masked `tt.load` without `other` |
| `bool_mask` | `triton_poi_fused_bitwise_and_gt_0` | 2 | 1 | masked `tt.load` without `other` |
| `broadcast_row` | `triton_poi_fused_add_0` | 2 | 1 | `arith.remsi` |
| `broadcast_col` | `triton_poi_fused_add_0` | 2 | 1 | `arith.divsi` |
| `tuple_two_outputs` | `triton_poi_fused_add_sub_0` | 2 | 2 | masked `tt.load` without `other` |
| `strided_input` | `triton_poi_fused_add_slice_0` | 2 | 1 | masked `tt.load` without `other` |
| `slice_even` | `triton_poi_fused_mul_slice_0` | 1 | 1 | masked `tt.load` without `other` |

## TTIR Coverage

Unique TTIR ops seen:

```text
arith.addf, arith.addi, arith.andi, arith.cmpf, arith.cmpi,
arith.constant, arith.divf, arith.divsi, arith.extf, arith.extui,
arith.mulf, arith.muli, arith.ori, arith.remsi, arith.select,
arith.subf, arith.truncf, math.absf, math.cos, math.exp, math.log,
math.sin, tt.addptr, tt.bitcast, tt.extern_elementwise,
tt.get_program_id, tt.load, tt.make_range, tt.precise_sqrt,
tt.return, tt.splat, tt.store
```

High-frequency ops:

| Op | Count |
|---|---:|
| `tt.splat` | 92 |
| `tt.addptr` | 68 |
| `arith.constant` | 68 |
| `tt.load` | 43 |
| `arith.muli` | 26 |
| `arith.addi` | 25 |
| `tt.store` | 25 |
| `arith.cmpi` | 24 |
| `tt.get_program_id` | 24 |
| `tt.make_range` | 24 |
| `tt.return` | 24 |
| `arith.addf` | 22 |

Notable ops outside the M2.5 translator subset:

- Mask/indexing: `arith.divsi`, `arith.remsi`, `arith.andi`, `arith.ori`,
  `arith.select`.
- Math: `math.absf`, `math.sin`, `math.cos`, `math.exp`, `math.log`,
  `tt.extern_elementwise`, `tt.precise_sqrt`.
- Type conversion / pointer reinterpretation: `arith.extf`, `arith.truncf`,
  `tt.bitcast`.

## M3.5 Gap List

Priority order for M3.5:

1. Support or safely reject Inductor masked load without `other` using explicit
   def-use proof for masked-out lanes.
2. Add translator coverage for `arith.select`, `arith.cmpf`, `arith.divf`,
   `arith.extf`, `arith.truncf`, and common integer index ops.
3. Add broadcast/strided indexing support for `arith.divsi` and `arith.remsi`
   patterns.
4. Add math op lowering or explicit unsupported buckets for `math.*`,
   `tt.extern_elementwise`, and `tt.precise_sqrt`.
5. Promote 3 to 5 audited kernels into offline Inductor pointwise E2E tests once
   the masked-load blocker is resolved.

## M3.5 Follow-up Result

M3.5 implemented `contract="pointwise_indexed"` and reran the same builtin corpus.

Command:

```bash
conda run -n tvm-0.24.0 \
  python -m tvm.contrib.triton_tvm.inductor \
  --builtin-corpus \
  --out-dir /tmp/triton_tvm_m35 \
  --min-kernels 20 \
  --contract pointwise_indexed
```

Result:

- Total pointwise kernels audited: 24.
- Kernels translated with `pointwise_indexed`: 24.
- Unsupported buckets: `translated=24`.
- CUDA build/run promoted regression cases: 13 representative kernels.
- Runtime integration status: offline only; no TorchInductor launcher/cache hook
  is implemented or implied.

The M3.5 implementation resolved the original masked-load blocker, added the
observed pointwise math/type/indexing ops, and promoted 13 representative
Inductor pointwise kernels into CUDA build/run tests.

## M3.5 Hardening / M4 Entry Gate

Before starting M4 reductions, the M3.5 surface was hardened rather than
expanded:

- `pointwise_minimal` and `pointwise_flat` retain the M2.5 flat pointwise TTIR
  capability table.
- `pointwise_indexed` exclusively owns the M3.5 no-`other` masked load proof,
  indexed pointer patterns, math/extern lowering, and observed bool bitcast-store
  collapse.
- `pointwise_indexed` contract validation now rejects shape-only matches where
  direct loads escape guarded store values, stores use different guards, or
  buffer indices leave the declared `i`, `i % C`, `i // C`, `i * C` set.
- `reduction_minimal` remains an explicit placeholder and is not routed through
  the pointwise translator.

## M4 Reduction Opening

M4 promotes `reduction_minimal` from placeholder to a correctness-first row-wise
reduction contract.  This does not change the M3/M3.5 Inductor pointwise audit
surface and does not add a TorchInductor runtime hook.

M4 reduction boundary:

- Parses Triton 3.7 quoted `"tt.reduce"` region ops without leaking combiner ops
  into the top-level TTIR op stream.
- Supports `tl.sum(axis=0)` with sum combiners only.
- Supports row sum, RMS core, RMSNorm weight, and single-row LayerNorm
  gamma/beta CUDA correctness cases.
- The full single-row LN/RMS pre-M5 surface is closed: runtime `n`,
  runtime/constexpr `eps`, masks, RMS weight, LayerNorm gamma/beta, and
  LayerNorm double reduction are covered.
- Requires explicit-zero masked loads and static 1D row-count grids.  Safe
  unmasked parameter loads are supported for `n == BLOCK` style cases.
- Leaves shared/allreduce lowering, cross-block reductions, Welford, `tl.max`,
  persistent kernels, and Inductor integration out of scope.
