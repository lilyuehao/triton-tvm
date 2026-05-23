# Triton TVM Contracts

This page is the Pre-M5 contract matrix.  Contract names are target-neutral.
Legacy `cuda_*` spellings are compatibility aliases only and must not appear in
metadata, cache keys, report buckets, or internal dispatch.

## Contract Matrix

| Contract | Purpose | TTIR Ops | Pointer / Index Pattern | Mask Semantics | Launch Policy | Expected TIRX Shape | Unsupported Cases | Tests |
|---|---|---|---|---|---|---|---|---|
| `pointwise_minimal` | Single-output basic pointwise. | M2.5 pointwise ops: `arith.add*`, `arith.sub*`, `arith.mul*`, `arith.cmpi`, casts, `tt.get_program_id`, `tt.make_range`, `tt.splat`, `tt.addptr`, `tt.load`, `tt.store`, `tt.return`. | Flat contiguous `base + i`, one output store. | Masked loads must provide explicit `other`; the single store must be guarded. | CUDA policy `cuda_block_thread`, `blockIdx.x` outer and `threadIdx.x` lane. | One `tirx.PrimFunc`, no `external_mods`, `tirx.noalias`, match buffers from ABI, nested block/lane thread bindings. | Multiple stores, no-`other` masked loads, non-contiguous indices, reductions, dot, atomics. | `test_translate_ttir_contract_and_metadata`, `test_static_m25_extent_inference_without_cuda_runtime`, `test_static_m25_negative_translate_without_cuda_runtime`. |
| `pointwise_flat` | Canonical pointwise path for flat, multi-output, and M3.5 Inductor pointwise capability variants. | `pointwise_minimal` ops plus observed M3.5 pointwise ops: `arith.select`, `arith.cmpf`, `arith.div*`, `arith.rem*`, `arith.extf`, `arith.truncf`, `math.*`, `tt.bitcast`, `tt.extern_elementwise`, `tt.precise_sqrt`. | Flat `i`, observed indexed forms `i % C`, `i // C`, `i * C`, and bool bitcast-store collapse. | Stores share one guard; no-`other` masked loads are accepted only when def-use proves the value cannot escape the guarded store. | CUDA policy `cuda_block_thread`. | Same shell as `pointwise_minimal`, one or more guarded stores, canonical contract metadata. | Unsafe no-`other` loads, unsupported composed indices, direct loads escaping guarded values, inconsistent masks, reductions, dot, atomics. | `test_static_m25_tirscript_golden_shape`, `test_static_pre_m5_pointwise_flat_indexed_capability_without_cuda_runtime`, `test_static_pre_m5_pointwise_flat_validator_rejects_shape_only_matches`, `test_m35_cuda_builds_and_runs_promoted_inductor_pointwise`. |
| `reduction_minimal` | Correctness-first row/block reduction. | Reduction subset: pointwise arithmetic needed by row reductions, `math.rsqrt`, `tt.reduce`, `tt.reduce.return`, load/store/range/splat/addptr. | Row-major `x + row * n + offsets`, scalar row output `out + row`, vector epilogue `out + row * n + offsets`. | Masked reduction loads require explicit zero `other`; safe unmasked parameter loads are allowed for `n == BLOCK` style vectors. | Static 1D CUDA row grid, `blockIdx.x` row, `threadIdx.x == 0` serial reduction body. | Thread-bound shell with serial reduction loop and local accumulator buffer. | Callable or multidim grid, non-sum combiners, axis other than 0, non-row-major pointers, cross-block reductions, atomics, Welford, max. | `test_static_m4_reduction_minimal_contract_without_cuda_runtime`, `test_static_m4_reduction_minimal_negative_boundaries`, `test_m4_row_sum_reduction_minimal_build_run`. |
| `norm_single_row` | Canonical single-row LN/RMS surface over the M4 reduction lowering. | Same op subset as `reduction_minimal`. | Same row-major input/output and per-row weight/gamma/beta vectors. | Same as `reduction_minimal`; runtime and constexpr `eps` are both ABI-covered. | Static 1D CUDA row grid. | Same correctness-first reduction shell; metadata uses `norm_single_row`. | Same reduction exclusions; no new reduction semantics beyond M4. | `test_static_pre_m5_norm_single_row_contract_without_cuda_runtime`, RMS/RMSNorm/LayerNorm CUDA cases in `test_triton_tvm.py`. |

## Runtime Boundary

`artifact.run(args, grid=None, stream=None)` is intentionally narrow before M5.
`args` must match `meta.abi`; `grid=None` means the translated metadata grid is
used, while an explicit static grid is only validated.  `stream=None` means the
TVM runtime current/default stream.  Non-default or raw PyTorch streams raise
`UnsupportedStreamError` with the `unsupported_stream` bucket.

## Cache Boundary

Pre-M5 disk cache is disabled.  `meta.cache_key` is a stable identity for the M5
in-memory cache prototype and includes TTIR/source hashes, canonical contract,
target policy and attrs, constexprs, ABI signature, TVM/Triton versions, and the
translator capability version.  It excludes legacy aliases, non-canonical target
spellings, and unstable object reprs.
