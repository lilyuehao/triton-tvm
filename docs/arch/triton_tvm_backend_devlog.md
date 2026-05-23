# Triton to TVM Backend Development Log

This log records implementation checkpoints for the Triton language to TVM backend prototype.
Keep it close to `triton_tvm_backend_plan.md` so future milestones can compare plan, code, tests,
and known gaps without rereading the whole patch history.

## 2026-05-22: M0-M1.5 Prototype

### Environment

- TVM tree: `/home/liyh/xdb/tvm`
- Conda environment: `tvm-0.24.0`
- TVM version: `0.24.0`
- Triton version: `3.7.0`
- Target validated: CUDA
- GPU present during validation: NVIDIA RTX 6000D

### Implemented Scope

- Added `python/tvm/contrib/triton_tvm/` as a pure Python prototype package.
- Implemented M0 TTIR extraction from `@triton.jit` via `triton.compiler.ASTSource` and
  `triton.compiler.compile(...).asm["ttir"]`.
- Implemented M0.5 textual TTIR reader into `NormalizedTTIROpGraph`.
- Implemented M1 vector-add TTIR subset translation to `tirx.PrimFunc`.
- Implemented M1.5 build/runtime smoke path with `tvm.compile` and `artifact.run(...)`.
- Avoided the existing black-box Triton integration path (`T.call_kernel` / `external_mods`).

### Public API Added

- `lower_to_ttir(jit_fn, signature, constexprs, *, dump_path=None) -> TTIRArtifact`
- `TTIRReader.read(ttir: str) -> NormalizedTTIROpGraph`
- `translate_ttir(ttir_or_graph, *, grid, target="cuda", emit="tirx", contract="cuda_minimal")`
- `build_triton_tvm(irmod, meta, passes=None, target="cuda") -> TritonTVMArtifact`
- `TritonTVMArtifact.run(args, grid=None, stream=None)`
- `NormalizeTritonKernelTIR(contract="cuda_minimal")`
- `validate_cuda_minimal_contract(irmod)`

### TTIR Subset Covered

- `tt.func`
- `arith.constant`
- `tt.get_program_id`
- `tt.make_range`
- `tt.splat`
- `tt.addptr`
- `tt.load`
- `tt.store`
- `arith.addi`, `arith.muli`, `arith.extsi`, `arith.cmpi`, `arith.addf`
- `tt.return`

### Translation Contract

- Emits `tirx.PrimFunc` with CUDA target, `tirx.noalias=True`, and `global_symbol`.
- Maps pointer parameters to flat 1D `T.match_buffer`.
- Keeps runtime scalar `n` dynamic.
- Maps Triton block tensor lanes to `blockIdx.x` and `threadIdx.x`.
- Lowers masked load with `other` to `T.if_then_else(mask, load, other)`.
- Lowers masked store with an outer `if mask` guard.
- Rejects masked load without `other` in M1.5.
- Rejects non-CUDA targets and non-`tirx` emission.
- Rejects unsupported TTIR ops such as `tt.dot`.

### Runtime and Metadata

- `TritonTVMMeta` records kernel name, signature, constexprs, grid, target, contract, emit mode,
  TVM/Triton versions, TTIR hash, source hash, ABI, and cache key.
- `build_triton_tvm` applies user-supplied TVM passes before `tvm.compile`.
- `stream` in `artifact.run(...)` is intentionally unsupported in M1.5 and raises explicitly.
- Cache support is metadata-only; no disk cache is implemented.

### Tests Added

- File: `tests/python/contrib/test_triton_tvm.py`
- Coverage:
  - TTIR extraction and dump path.
  - TTIR reader normalization.
  - TIRX contract and metadata.
  - User pass insertion before build.
  - CUDA correctness against NumPy and native Triton.
  - Divisible and non-divisible vector lengths.
  - Negative tests for unsupported op, masked load without `other`, unsupported emit/target,
    and unsupported stream.

### Validation

Command:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm*.py
```

Result:

```text
8 passed
```

Additional note:

- `ruff` is not installed in the `tvm-0.24.0` environment, so automated ruff validation was not run.
- New-file line lengths were checked manually against the repo's 100-column ruff setting.

### Known Gaps

- Parser is a conservative textual TTIR parser, not a full MLIR parser.
- Builder uses controlled TVMScript source plus `tvm.script.from_source`; direct node construction is
  deferred.
- Only vector-add-style pointwise structure is supported.
- No reductions, dot, atomics, TMA, async copy, tensor descriptors, block pointers, or Inductor hook.
- Only CUDA `tirx` output is supported.
- No explicit disk artifact cache.

### Suggested Next Checkpoint

For M2, extend this log with:

- New TTIR ops accepted.
- Mask semantics changes, especially masked load without `other`.
- Broadcast and multi-op pointwise examples added.
- New golden IR shape changes.
- Correctness and negative test deltas.

## 2026-05-22: Pre-M2 Hardening

### Motivation

After the M1.5 prototype passed, the remaining pre-M2 cleanup focused on tightening the
implementation around the current CUDA-minimal contract without changing
`triton_tvm_backend_plan.md`.

### Implementation Changes

- Strengthened `validate_cuda_minimal_contract`:
  - checks the outer loop is bound to `blockIdx.x`;
  - checks the inner loop is bound to `threadIdx.x`;
  - checks the M1.5 contract has exactly one `BufferStore`;
  - checks masked stores are represented with an `IfThenElse` guard.
- Strengthened `TritonTVMArtifact.run(...)`:
  - validates runtime argument count against the recorded ABI;
  - validates TVM tensor pointer arguments and dtypes;
  - validates scalar argument kinds;
  - validates explicit runtime grid against the translated grid when both are concrete;
  - still rejects non-default streams explicitly.
- Strengthened `build_triton_tvm(...)`:
  - rejects build targets whose target kind does not match the translated target.

### Tests Added

- Added a standalone TTIR reader corpus test before M2:
  - `_pointwise_chain_kernel` validates reader coverage for multi-op pointwise TTIR;
  - `_dual_store_kernel` validates reader visibility into multiple-store TTIR.
- Added negative runtime tests:
  - wrong runtime argument count;
  - mismatched explicit grid;
  - pointer dtype mismatch;
  - scalar kind mismatch.
- Added negative build test for target mismatch.

### Validation

Command:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm*.py
```

Result:

```text
11 passed
```

### Remaining M2 Entry Conditions

- `NormalizeTritonKernelTIR` is still a validation/anchor pass; it does not yet perform real
  lane-axis, addptr, or mask canonicalization.
- The translator still emits `cuda_minimal` TIRX directly.
- The runtime grid is validated when provided, but the generated TIRX launch extent is still
  computed from the runtime scalar extent.
- TTIR reader coverage is broader than vector add, but translator coverage remains intentionally
  vector-add-style.

## 2026-05-22: M2A Standalone Pointwise Expansion

### Implemented Scope

- Extended translator coverage from vector-add-style pointwise to standalone pointwise chains with
  multiple `tt.load` ops, one `tt.store`, and one output.
- Added `arith.mulf` to the accepted TTIR op subset.
- Centralized binary arithmetic lowering for `arith.addf/subf/mulf` and `arith.addi/subi/muli`.
- Kept masked load/store semantics unchanged:
  - masked load with `other` lowers through `T.if_then_else`;
  - masked load without `other` still raises `UnsupportedTTIROpError`;
  - masked store remains guarded by `IfThenElse`.
- Continued to reject multiple stores with an explicit `multiple tt.store is not supported yet`
  error.

### Tests Added

- `_pointwise_chain_kernel` now translates, builds, runs, and validates against NumPy for an odd
  vector size.
- The M2A correctness test covers three pointer inputs, one output, three masked loads,
  `arith.mulf`, `arith.addf`, and `arith.subf`.
- Added a negative test that `_dual_store_kernel` is rejected before TIRX generation.

### Remaining Gaps

- Multiple stores and multiple outputs remain unsupported.
- Reductions, dot, atomics, non-contiguous pointer patterns, and Inductor hook remain unsupported.
- No new disk cache, non-CUDA target, `emit="tir"`, or real normalization pass was added.

### Validation

Command:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm*.py
```

Result:

```text
13 passed
```

## 2026-05-22: M2.5 Standalone Pointwise Corpus

### Implemented Scope

- Added the `cuda_pointwise_flat` contract for standalone flat-contiguous pointwise kernels.
- Preserved `cuda_minimal` as the single-store contract; multiple stores now require
  `contract="cuda_pointwise_flat"` explicitly.
- Extended translator emission from a single `tt.store` template to store-list emission.
- Supported multiple independent masked store expressions in one kernel.
- Supported scalar/splat/block tensor broadcast in the covered pointwise subset.
- Tightened pointer lowering to the flat contiguous lane-index pattern
  `pid * BLOCK + tl.arange(0, BLOCK)`.
- Continued to reject masked load without `other`, reduction, dot, atomic, block pointer,
  non-contiguous pointer patterns, and non-CUDA / non-`tirx` emission.

### Contract and Runtime Changes

- Added `validate_cuda_pointwise_flat_contract`.
- Added `validate_triton_tvm_contract` as the shared contract dispatch helper.
- Updated `NormalizeTritonKernelTIR` and `build_triton_tvm` to validate the selected contract
  from the pass argument or metadata.
- `cuda_pointwise_flat` validator checks:
  - no `external_mods`;
  - each function is a `tirx.PrimFunc`;
  - target is CUDA and `tirx.noalias` / `global_symbol` are set;
  - outer loop binds `blockIdx.x`;
  - inner loop binds `threadIdx.x`;
  - at least one `BufferStore` exists;
  - every `BufferStore` is guarded by an `IfThenElse` mask.

### Tests Added

- `_dual_store_kernel` now translates, builds, runs, and validates two outputs:
  - `out0 = x + y`;
  - `out1 = x - y`.
- Added `_scalar_broadcast_kernel` correctness coverage for runtime scalar broadcast:
  - `out0 = x + alpha + 1.0`;
  - `out1 = x * alpha`.
- Kept vector add and M2A pointwise-chain correctness tests green.
- Added negative coverage for:
  - multiple stores under default `cuda_minimal`;
  - reduction op;
  - atomic op;
  - non-contiguous pointer pattern (`x + offsets * 2`);
  - existing masked load without `other` and `tt.dot` cases.

### Supported TTIR Subset at M2.5

- `tt.func`
- `arith.constant`
- `tt.get_program_id`
- `tt.make_range`
- `tt.splat`
- `tt.addptr`
- `tt.load`
- `tt.store`
- `arith.addi`, `arith.subi`, `arith.muli`
- `arith.addf`, `arith.subf`, `arith.mulf`
- `arith.extsi`, `arith.extui`
- `arith.cmpi`
- `tt.return`

### Remaining Gaps

- Reductions, dot, atomics, block pointers, tensor descriptors, TMA, async copy, and Inductor hook
  remain unsupported.
- `cuda_pointwise_flat` only covers the flat contiguous pointer pattern; strided and
  non-contiguous pointer expressions are rejected.
- `NormalizeTritonKernelTIR` remains a validation/anchor pass and does not yet perform real IR
  rewriting.
- Runtime launch still derives extent from the runtime scalar and validates an explicit grid only
  when one is provided.

### Validation

Command:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
```

Result:

```text
17 passed
```

## 2026-05-22: M2.5 Hardening: Pointwise Contract Cleanup

### Implemented Scope

- Added `tests/python/contrib/test_triton_tvm_static.py` as a static regression suite.
- The static suite uses hand-written TTIR and does not invoke Triton compilation, TVM build, or
  CUDA runtime launch.
- Added a reader op graph snapshot for a dual-store pointwise TTIR module.
- Added TVMScript golden-shape checks for `cuda_pointwise_flat`:
  - four pointer `T.match_buffer` bindings;
  - `blockIdx.x` outer thread binding;
  - `threadIdx.x` inner thread binding;
  - two independent masked stores;
  - masked load values lowered through `T.if_then_else`.
- Added pure translate/contract negative tests for:
  - multiple stores under default `cuda_minimal`;
  - validating a multi-store module as `cuda_minimal`;
  - unknown contract dispatch;
  - masked load without `other`;
  - reduction op;
  - non-contiguous pointer pattern;
  - unguarded store under `cuda_pointwise_flat`.

### Reader Cleanup

- Normalized `TTIRReader` result type parsing for:
  - `tt.splat` result types after `->`;
  - `arith.extsi` / `arith.extui` result types after `to`;
  - `tt.load` result types from pointer element type;
  - `tt.addptr` result type from the pointer operand;
  - `tt.store` / `tt.return` as no-result ops.

### Contract Boundary

- `cuda_minimal` is the legacy single-store CUDA pointwise contract.
- `cuda_pointwise_flat` is the M2.5 flat contiguous pointwise contract.  It keeps the same launch
  and lane shape, but allows multiple masked stores and multiple outputs.
- Both contracts remain validation targets today; `NormalizeTritonKernelTIR` still does not rewrite
  IR.

### Validation

Command:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
```

Result:

```text
4 passed
```

## 2026-05-22: M2.5 Hardening: Backend Decoupling / CUDA Policy Boundary

### Motivation

Adapting the prototype to non-CUDA backends is now an explicit pre-M3 priority.  This
checkpoint keeps the existing CUDA lowering and correctness behavior intact, but stops treating
CUDA-specific names and launch tags as the translator's long-term internal contract.

### Implementation Changes

- Added canonical pointwise contract names:
  - `pointwise_minimal` for the single-store pointwise contract;
  - `pointwise_flat` for the flat-contiguous multi-store pointwise contract.
- Kept compatibility aliases:
  - `cuda_minimal -> pointwise_minimal`;
  - `cuda_pointwise_flat -> pointwise_flat`.
- Added `validate_pointwise_minimal_contract` and `validate_pointwise_flat_contract`; the
  legacy `validate_cuda_*` validators remain as wrappers.
- Updated `translate_ttir`, `NormalizeTritonKernelTIR`, and `validate_triton_tvm_contract` to
  accept canonical names and aliases.
- Added an internal pointwise target policy boundary:
  - CUDA is currently the only implemented policy;
  - CUDA policy owns the `blockIdx.x` / `threadIdx.x` launch tags;
  - unsupported targets now fail with an unsupported target policy error instead of an
    inline `target == "cuda"` check.
- Added explicit contract and policy metadata structures so `pointwise_flat` describes the IR
  contract while the CUDA policy owns launch legalization.
- Added stable error classes for M3 audit bucketing:
  - `UnsupportedTargetPolicyError`;
  - `UnsupportedContractError`;
  - existing `UnsupportedTTIROpError` for unsupported TTIR subset cases.
- Supported target dispatch through both strings and `tvm.target.Target` objects.  Legacy
  `cuda -arch=sm_80` spelling is normalized to a TVM Target for policy selection.
- Changed runtime extent inference to use the flat-contiguous mask/index pattern such as
  `%idx < splat(%n)` instead of assuming the first scalar parameter is the extent.
- Tightened M2.5 extent/mask acceptance to one flat memory mask predicate shared by loads and
  stores.  Multiple extent candidates, inconsistent load/store masks, and compound masks remain
  unsupported.
- Extended `TritonTVMMeta` with target-neutral anchors:
  - canonical contract and requested contract;
  - target kind;
  - target attrs;
  - extent parameter;
  - block size;
  - indexing kind;
  - launch policy id;
  - translator version, contract version, and target policy version.
- Cache keys now use canonical contract, target attrs, launch/index metadata, and version fields,
  so a `cuda_*` alias and its canonical `pointwise_*` name map to the same compilation identity.
- `NormalizeTritonKernelTIR` still performs contract dispatch and validation only.  It is
  intentionally a no-op rewrite pass in M2.5.
- `TTIRReader` preserves raw/unknown load-store attribute dictionaries when textual TTIR exposes
  attributes such as cache modifier or eviction policy.  These attrs are not lowered yet.

### Static Coverage Added

- `contract="pointwise_flat"` golden-shape generation.
- `contract="cuda_pointwise_flat"` alias dispatch and legacy validator compatibility.
- Alias cache-key canonicalization between `pointwise_flat` and `cuda_pointwise_flat`.
- Unsupported non-CUDA target policy errors for string and `tvm.target.Target` inputs.
- CUDA policy dispatch for `target="cuda"`, `tvm.target.Target("cuda")`, and legacy
  `target="cuda -arch=sm_80"`.
- Unsupported contract-on-CUDA-policy error for `reduction_minimal`.
- Hand-written TTIR cases where scalar parameter order is `alpha, n` and `alpha, beta, n`,
  verifying that `n` is inferred as the runtime extent.
- Rejection coverage for inconsistent load/store masks and multiple extent candidates.
- Raw load/store attr preservation in the TTIR reader.
- Metadata assertions for canonical contract, requested contract, target kind, extent parameter,
  block size, indexing kind, launch policy id, target attrs, and version fields.

### Remaining Bounds

- Non-CUDA lowering and runtime execution remain unsupported in M2.5.
- `NormalizeTritonKernelTIR` is still a validation/anchor pass and does not rewrite IR yet.
- Alignment, cache modifier, eviction policy, volatile, and richer load/store metadata parsing
  remain future TTIR reader work.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Results:

```text
9 passed
17 passed
1 skipped
```

## 2026-05-22: M2.5 Performance Baseline / Regression Guard

### Positioning

This checkpoint is a performance baseline and regression-protection hook, not a performance
optimization target.  The measured numbers are hardware-, driver-, and load-sensitive.  They are
intended to catch obvious same-machine regressions after translator, contract, or lowering changes.

### Implementation Changes

- Added `tests/python/contrib/test_triton_tvm_perf.py`.
- The perf test is opt-in and skipped by default.
- It measures the M2.5 standalone pointwise corpus:
  - vector add (`cuda_minimal`);
  - pointwise chain (`cuda_minimal`);
  - dual store (`cuda_pointwise_flat`);
  - scalar broadcast (`cuda_pointwise_flat`).
- TVM timing uses the compiled runtime module's `time_evaluator`, so Python `artifact.run(...)`
  argument validation is outside the measured kernel runtime.
- Native Triton timing is recorded as same-machine context, not as the pass/fail criterion.
- The guard can:
  - write a JSON baseline via `TRITON_TVM_PERF_WRITE_JSON`;
  - compare against a JSON baseline via `TRITON_TVM_PERF_BASELINE_JSON`;
  - fail if current TVM GB/s is below the stored same-machine baseline by more than
    `TRITON_TVM_PERF_TOLERANCE` (default `0.35`).
- Added a local RTX 6000D sample baseline:
  `docs/arch/triton_tvm_m25_perf_baseline_rtx6000d.baseline`.

### Commands

Default test path, expected to skip:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Collect a local baseline:

```bash
TRITON_TVM_RUN_PERF_BASELINE=1 \
TRITON_TVM_PERF_WRITE_JSON=/tmp/triton_tvm_m25_perf_baseline.json \
conda run -n tvm-0.24.0 python -m pytest -q -s tests/python/contrib/test_triton_tvm_perf.py
```

Run regression protection against a baseline:

```bash
TRITON_TVM_RUN_PERF_BASELINE=1 \
TRITON_TVM_PERF_BASELINE_JSON=docs/arch/triton_tvm_m25_perf_baseline_rtx6000d.baseline \
conda run -n tvm-0.24.0 python -m pytest -q -s tests/python/contrib/test_triton_tvm_perf.py
```

### Local Baseline

Environment:

- GPU: NVIDIA RTX 6000D
- TVM version: `0.24.0`
- Triton version: `3.7.0`
- `n = 4194304`
- `BLOCK = 256`

| Case | Contract | TVM us | TVM GB/s | Triton us | Triton GB/s |
|---|---:|---:|---:|---:|---:|
| vector_add | cuda_minimal | 10.01 | 5029.79 | 48.96 | 1027.95 |
| pointwise_chain | cuda_minimal | 10.09 | 6651.81 | 64.65 | 1038.06 |
| dual_store | cuda_pointwise_flat | 12.46 | 5385.01 | 58.92 | 1139.00 |
| scalar_broadcast | cuda_pointwise_flat | 11.65 | 4321.74 | 42.29 | 1190.18 |

### Validation

Default skip:

```text
1 skipped
```

Opt-in baseline collection:

```text
1 passed
```

Opt-in regression guard against the generated baseline:

```text
1 passed
```

## 2026-05-22: M3 Inductor Pointwise Audit

### Implemented Scope

- Added `python/tvm/contrib/triton_tvm/inductor.py` as the M3 offline audit tool.
- The tool extracts `async_compile.triton(...)` source blocks from TorchInductor wrapper code
  using Python AST parsing.
- The tool loads extracted kernels through PyTorch `PyCodeCache`, force-reloading source blocks
  so the generated `CachingAutotuner` still has its Triton configs.
- The tool lowers loaded Inductor kernels to optimized TTIR with the existing Triton frontend path,
  preserving Inductor argument attrs from the autotuner metadata.
- The tool audits TTIR op/type/attr/indexing/mask coverage and then tries the existing
  `translate_ttir(..., contract="pointwise_flat")` path only to classify current gaps.
- No TVM build/run path, Inductor runtime hook, or translator semantic expansion was added.

### CLI Added

```bash
python -m tvm.contrib.triton_tvm.inductor \
  --builtin-corpus \
  --out-dir /tmp/triton_tvm_m3 \
  --min-kernels 20
```

Outputs:

- `kernels/*.py`
- `ttir/*.ttir`
- `report.json`
- `report.md`

`--input-dir PATH` can also scan existing Inductor wrapper/cache `.py` files.

### Builtin Corpus

The builtin corpus currently contains 24 pointwise cases:

- `add`
- `add_mul`
- `three_input_chain`
- `scalar_alpha`
- `two_scalars`
- `relu_add`
- `sigmoid_mul`
- `tanh_shift`
- `exp_log_abs`
- `sin_cos`
- `where_cmp`
- `clamp_add`
- `reciprocal_abs`
- `sqrt_rsqrt`
- `pow2_add`
- `fp16_add`
- `bf16_add`
- `int32_add`
- `bool_mask`
- `broadcast_row`
- `broadcast_col`
- `tuple_two_outputs`
- `strided_input`
- `slice_even`

### Local Audit Result

Command:

```bash
conda run -n tvm-0.24.0 \
  python -m tvm.contrib.triton_tvm.inductor \
  --builtin-corpus \
  --out-dir /tmp/triton_tvm_m3 \
  --min-kernels 20
```

Result:

```text
Wrote M3 Inductor audit with 24 kernels to /tmp/triton_tvm_m3
```

Summary:

- 24 pointwise kernels collected.
- 24 TTIR dumps generated.
- 0 kernels translated by the current M2.5 translator.
- Unsupported buckets: `unsupported_ttir_op=24`.
- Dominant blocker: Inductor emits masked `tt.load` without explicit `other`.
- Broadcast cases additionally exposed `arith.remsi` and `arith.divsi` indexing patterns.

Full compressed report:

- `docs/arch/triton_tvm_m3_inductor_audit.md`

### Tests Added

- Added `tests/python/contrib/test_triton_tvm_inductor_subset.py`.
- Static coverage:
  - AST extraction from hard-coded Inductor wrapper source.
  - Coverage/report generation from hand-written TTIR.
  - Unsupported bucket and Markdown/JSON report checks.
- CUDA coverage:
  - Compiles a small `torch.compile(..., backend="inductor")` pointwise case.
  - Captures Inductor wrapper source through `GraphLowering.save_output_code`.
  - Extracts, loads, lowers to TTIR, reads with `TTIRReader`, and runs audit classification.

### Remaining Bounds

- M3 intentionally does not implement unsafe masked-load semantics.
- M3 does not build/run translated TVM modules.
- M3 does not add an Inductor runtime or `torch.compile` hook.
- M3.5 should start with masked load semantics and high-frequency Inductor pointwise ops from the
  M3 coverage report.

## 2026-05-22: M3.5 Inductor Pointwise Coverage

### Implemented Scope

- Added the canonical `pointwise_indexed` contract and `cuda_pointwise_indexed` alias.
- Extended `TritonTVMMeta` with:
  - `extent_kind`;
  - `extent_value`;
  - `buffer_extents`.
- Hardened `TTIRReader` for real TorchInductor TTIR:
  - strips parameter attrs from types;
  - parses `arith.cmpf`, `arith.select`, `arith.extf`, `arith.truncf`;
  - parses `tt.bitcast`, `tt.extern_elementwise`, and bare load/store attrs such as
    `evictionPolicy = evict_last`.
- Implemented safe masked `tt.load` without `other` for `pointwise_indexed`:
  - load/store masks must share the same memory predicate;
  - no-`other` load values must flow only into store values guarded by that predicate;
  - pointer, load, return, unguarded-store, and unsupported uses are rejected explicitly.
- Added lowering for the M3 Inductor pointwise op surface:
  - `arith.select`, observed `arith.cmpf` predicates, `arith.divf`;
  - `arith.divsi`, `arith.remsi`, bool `arith.andi` / `arith.ori`;
  - `arith.extf`, `arith.truncf`;
  - `math.absf`, `math.sin`, `math.cos`, `math.exp`, `math.log`, `tt.precise_sqrt`;
  - `tt.extern_elementwise` symbols `__nv_expf`, `__nv_tanhf`, and `__nv_rsqrtf`.
- Added indexed pointer support for:
  - flat `idx`;
  - broadcast `idx % C` and `idx // C`;
  - strided `idx * C`.
- Added the observed bool output pattern:
  - `ptr<i1> -> ptr<i8>` plus `extui bool -> i8` stores are collapsed to bool buffer stores.
- Kept `NormalizeTritonKernelTIR` validation-only and did not add an Inductor runtime hook.

### Audit Result

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

```text
Wrote M3 Inductor audit with 24 kernels to /tmp/triton_tvm_m35
```

Summary:

- 24 pointwise kernels collected.
- 24 kernels translated with `contract="pointwise_indexed"`.
- Unsupported buckets: `translated=24`.

### Tests Added

- Static `pointwise_indexed` coverage for:
  - real Inductor parameter attrs and bare load attrs;
  - constant extent masks;
  - safe and unsafe masked loads without `other`;
  - indexed broadcast pointers;
  - bool bitcast-store collapse.
- M3.5 audit coverage proving the previous masked-load blocker translates under
  `pointwise_indexed`.
- CUDA build/run coverage for promoted Inductor pointwise cases:
  - `add`, `tuple_two_outputs`, `relu_add`, `where_cmp`, `sigmoid_mul`, `sin_cos`;
  - `broadcast_row`, `broadcast_col`, `strided_input`, `slice_even`;
  - `fp16_add`, `int32_add`, `bool_mask`.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_inductor_subset.py
```

Results:

```text
12 passed
17 passed
16 passed
```

## 2026-05-22: M3.5 Hardening / M4 Entry Gate

### Positioning

This checkpoint hardens the aggressive M3.5 Inductor pointwise coverage before
opening M4 reductions.  It does not add reduction lowering, does not expand the
Inductor pointwise corpus, and does not add an Inductor runtime hook.

### Implementation Changes

- Replaced the global translator op allowlist with per-contract TTIR capability
  tables:
  - `pointwise_minimal` and `pointwise_flat` keep the M2.5 flat pointwise op set.
  - `pointwise_indexed` exclusively enables M3.5 no-`other` masked loads,
    indexed pointer ops, math/extern ops, and the observed bool bitcast-store
    pattern.
- Made `reduction_minimal` fail before target-policy or pointwise-builder
  dispatch with an explicit not-implemented error for M4.
- Bumped metadata versions:
  - translator: `triton_tvm_python_m35_hardened_v1`;
  - `pointwise_indexed`: `pointwise_indexed_v2`.
- Strengthened `validate_pointwise_indexed_contract`:
  - every store must be guarded;
  - all stores must use the same guard;
  - direct `BufferLoad` nodes must appear only inside guarded store values;
  - store/load buffer indices are limited to `i`, `i % C`, `i // C`, and `i * C`.
- Kept `NormalizeTritonKernelTIR` validation-only as an explicit pre-M4 choice.
  M4 can continue the direct-to-contract prototype; shared lane/mask/index
  rewrites should move into this pass only once reduction lowering creates
  concrete reuse pressure.

### Tests Added

- Static hardening coverage that `pointwise_flat` rejects M3.5-only indexed ops
  and bool bitcast-store patterns.
- Static validator coverage that shape-only `pointwise_indexed` TIR is rejected
  when direct loads escape guarded store values, indexes leave the supported set,
  or stores use different guards.
- Static coverage that `reduction_minimal` reports `not implemented yet` instead
  of falling into pointwise-specific errors.

### M4 Entry Criteria

- `pointwise_indexed` safe no-`other` masked load semantics are locked by
  translator proof, contract validation, and negative tests.
- Pointwise contract op surfaces are contract-specific and no longer expand via
  a single global allowlist.
- M3.5 remains documented as offline pointwise E2E only.
- `reduction_minimal` remains a clean placeholder.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_inductor_subset.py
```

Results:

```text
14 passed
17 passed
16 passed
```

Audit recheck:

```bash
conda run -n tvm-0.24.0 \
  python -m tvm.contrib.triton_tvm.inductor \
  --builtin-corpus \
  --out-dir /tmp/triton_tvm_m35_hardening_m3 \
  --min-kernels 20

conda run -n tvm-0.24.0 \
  python -m tvm.contrib.triton_tvm.inductor \
  --builtin-corpus \
  --out-dir /tmp/triton_tvm_m35_hardening_m35 \
  --min-kernels 20 \
  --contract pointwise_indexed
```

Audit results:

```text
pointwise_flat: total=24, translated=0, unsupported_ttir_op=24
pointwise_indexed: total=24, translated=24, bucket translated=24
```

## 2026-05-22: M3.5 Pointwise Performance Baseline

### Positioning

This checkpoint extends the opt-in performance guard from the M2.5 standalone
pointwise corpus to the M3.5 `pointwise_indexed` contract.  It remains a
same-machine regression guard, not a portable performance claim.

### Test Changes

- `tests/python/contrib/test_triton_tvm_perf.py` now covers:
  - the existing M2.5 `cuda_minimal` / `cuda_pointwise_flat` standalone cases;
  - `indexed_broadcast_relu`, covering no-`other` masked loads, `%` indexing,
    compare/select, and `pointwise_indexed`;
  - `indexed_strided`, covering no-`other` masked loads and `idx * 2`
    `pointwise_indexed` pointer indexing.
- Older JSON baselines may omit newer cases; regression comparison now checks
  overlapping cases so the M2.5 baseline remains usable.
- Added local RTX 6000D sample baseline:
  `docs/arch/triton_tvm_m35_perf_baseline_rtx6000d.baseline`.

### Commands

Default test path, expected to skip:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Collect the M3.5 baseline:

```bash
TRITON_TVM_RUN_PERF_BASELINE=1 \
TRITON_TVM_PERF_WRITE_JSON=/tmp/triton_tvm_m35_perf_baseline.json \
conda run -n tvm-0.24.0 python -m pytest -q -s tests/python/contrib/test_triton_tvm_perf.py
```

### Local Baseline

Environment:

- GPU: NVIDIA RTX 6000D
- TVM version: `0.24.0`
- Triton version: `3.7.0`
- `n = 4194304`
- `BLOCK = 256`
- broadcast feature size: `1024`

| Case | Contract | TVM us | TVM GB/s | Triton us | Triton GB/s | TVM/Triton GB/s |
|---|---|---:|---:|---:|---:|---:|
| vector_add | cuda_minimal | 10.01 | 5029.32 | 49.32 | 1020.41 | 4.93x |
| pointwise_chain | cuda_minimal | 10.10 | 6644.53 | 66.12 | 1014.89 | 6.55x |
| dual_store | cuda_pointwise_flat | 12.47 | 5379.86 | 58.40 | 1149.09 | 4.68x |
| scalar_broadcast | cuda_pointwise_flat | 11.65 | 4318.75 | 42.30 | 1189.89 | 3.63x |
| indexed_broadcast_relu | pointwise_indexed | 9.99 | 5037.21 | 30.27 | 1662.65 | 3.03x |
| indexed_strided | pointwise_indexed | 9.98 | 3362.37 | 46.57 | 720.46 | 4.67x |

### Validation

Results:

```text
default perf test: 1 skipped
opt-in baseline collection: 1 passed
opt-in regression guard against M3.5 baseline: 1 passed
opt-in regression guard against old M2.5 baseline: 1 passed
```

## 2026-05-22: M4 Reduction Subset

### Positioning

M4 opens `reduction_minimal` as a correctness-first reduction contract.  It is
not a performance milestone and does not add an Inductor runtime hook.  The v0
lowering intentionally uses one lane per row (`threadIdx.x == 0`) to compute the
row reduction and epilogue serially, avoiding accumulator races until a later
shared/allreduce or TIR reduction-block path is designed.

### Implementation Changes

- Extended `TTIRReader` to parse quoted region ops such as `"tt.reduce"`:
  - the reduce op remains a top-level `TTIROp`;
  - combiner ops are stored in `TTIROp.regions`;
  - region-local `arith.addf` / `tt.reduce.return` no longer leak into the
    top-level op stream;
  - ops after the reduce region, including `tt.store`, continue to parse.
- Implemented the dedicated `reduction_minimal` translator path:
  - `tl.sum(axis=0)` with single-input sum combiner;
  - row-major pointers `x + row * n + offsets`, vector epilogues
    `out + row * n + offsets`, scalar row outputs `out + row`, and
    per-row weight/gamma/beta vectors;
  - explicit-zero masked loads only;
  - fp32 row sum, RMS core, RMSNorm weight, and LayerNorm gamma/beta formulas.
- Promoted `reduction_minimal` from placeholder to contract version
  `reduction_minimal_v1` and added `validate_reduction_minimal_contract`.
- Kept M4 grid support narrow: static 1D row count only.  Callable and multidim
  grids are rejected.

### Tests Added

- Static reader coverage for real Triton 3.7 `"tt.reduce"` region shape.
- Static reduction contract and negative coverage for unsupported axis,
  non-sum combiner, masked load without explicit zero `other`, non-row-major
  pointer patterns, callable grid, and multidim grid.
- CUDA correctness coverage for row sum, RMS core, RMSNorm with weight, and
  LayerNorm with gamma/beta and two reductions.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_inductor_subset.py
```

Results:

```text
16 passed
21 passed
16 passed
```

## 2026-05-22: M4 Full LN/RMS Closure

### Positioning

This closes the single-row full LN/RMS surface that was reserved as a pre-M5
capability.  It keeps the same correctness-first `reduction_minimal` lowering
and does not introduce a performance-oriented shared/allreduce path.

### Implementation Changes

- Kept masked loads strict: `tt.load(ptr, mask)` without `other` is still
  rejected for reductions.
- Added safe unmasked `tt.load(ptr)` support for reduction epilogues and
  parameter vectors, covering `n == BLOCK` gamma/beta/weight access patterns.
- Confirmed runtime `eps` flows through the scalar ABI for RMSNorm and
  LayerNorm, in addition to the earlier constexpr-eps cases.

### Tests Added

- CUDA RMSNorm with runtime `eps`, runtime `n`, mask, and weight vector.
- CUDA LayerNorm with runtime `eps`, runtime `n`, mask, gamma/beta vectors, and
  two reductions.
- CUDA LayerNorm with runtime `eps` and unmasked gamma/beta loads in the
  `n == BLOCK` case.

### Validation

Results after closure:

```text
test_triton_tvm_static.py: 16 passed
test_triton_tvm.py: 24 passed
test_triton_tvm_inductor_subset.py: 16 passed
```

## 2026-05-22: Pre-M5 Debt Sprint Workflows 1-4

### Scope

This sprint prepares the prototype for M5 Inductor integration without adding
new Triton semantics, op coverage, reduction capability, or Inductor hooks.

### Workflow 1: Pass Surface Cleanup

- Renamed the validation-only public pass from `NormalizeTritonKernelTIR` to
  `ValidateTritonKernelTIR`.
- Removed `NormalizeTritonKernelTIR` from the public `tvm.contrib.triton_tvm`
  API.  A pass with that name should return only when it performs real IR
  rewrites.
- No `Legalize*`, `Lower*`, or `Normalize*` public pass remains as a no-op.

### Workflow 2: Legacy Contract Alias Cleanup

- Canonical contracts are now `pointwise_minimal`, `pointwise_flat`,
  `reduction_minimal`, and `norm_single_row`.
- `cuda_minimal` and `cuda_pointwise_flat` remain only as compatibility aliases
  and emit `FutureWarning`.
- Metadata and cache keys use canonical contract names even when a legacy alias
  is requested in compatibility tests.
- Removed the public `pointwise_indexed` / `cuda_pointwise_indexed` contract
  surface.  The M3.5 indexed/no-`other` pointwise behavior is now a
  `pointwise_flat` capability, not a separate contract.

### Workflow 3: Contract Matrix

- Added `docs/arch/triton_tvm_contracts.md` as the Pre-M5 capability matrix.
- Added `norm_single_row` as the canonical name for the single-row LN/RMS M4
  surface.  It reuses the existing correctness-first reduction lowering and
  does not add new reduction semantics.
- Strengthened `pointwise_flat` validator coverage so the old indexed
  shape-only rejection remains attached to the canonical pointwise contract.

### Workflow 4: Runtime, Cache, Stream Boundary

- Cache key payload now includes TTIR/source hash, canonical contract, target
  policy, target attrs, constexprs, ABI signature, TVM/Triton versions, and
  translator capability version.
- Cache key payload rejects unstable object reprs and no longer includes legacy
  aliases or non-canonical target spelling.
- Pre-M5 disk cache is explicit disabled metadata: `cache_policy="disabled"`
  and `disk_cache_enabled=False`.
- `artifact.run(..., stream=None)` is documented as TVM current/default stream.
  Non-`None` streams raise `UnsupportedStreamError` with `unsupported_stream`.
- Inductor audit status records now carry `fallback_reason` so M5 fallback
  routing has a stable report bucket field.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_inductor_subset.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Results:

```text
test_triton_tvm_static.py: 19 passed
test_triton_tvm_inductor_subset.py: 16 passed
test_triton_tvm.py: 24 passed
test_triton_tvm_perf.py: 1 skipped
```

## 2026-05-22: Pre-M5 Debt Sprint Workflows 5-8

### Workflow 5: TTIRReader and Builder Debt

- Moved TTIR input normalization to `ttir.normalize_ttir_input`.
- Removed direct `TTIRReader` use from `translator.py`; translator code now
  consumes `NormalizedTTIROpGraph` after the controlled reader boundary.
- Added a static guard that `translator.py` does not instantiate `TTIRReader`.
- Kept existing `NormalizedTTIROpGraph` reader snapshots for pointwise and
  reduction TTIR.  New TTIR op support must update these snapshots first.
- Registered builder debt in `docs/arch/triton_tvm_capability_matrix.md`:
  current builder is TVMScript source plus `tvm.script.from_source`; M5 may keep
  it with golden coverage, while M6 must evaluate direct node construction or
  document why the source builder remains appropriate.

### Workflow 6: Unified Reporting

- Added `python/tvm/contrib/triton_tvm/reporting.py`.
- Added `tests/python/contrib/test_triton_tvm_reporting.py` snapshot coverage
  for a unified pointwise/reduction/norm report schema.
- Updated Inductor audit report construction to use the shared capability
  report schema while preserving `report.json` / `report.md` output paths.
- Stable buckets now include `translated`, `unsupported_ttir_op`,
  `contract_error`, `target_policy_error`, `unsupported_stream`, `input_error`,
  `triton_tvm_error`, and `internal_error`.
- Added `docs/arch/triton_tvm_capability_matrix.md`.

### Workflow 7: Test Structure

- Kept correctness tests default-runnable.
- CUDA correctness tests retain `tvm.testing.requires_cuda` gating.
- Perf regression guard remains opt-in and skipped by default.
- Added public API and parser-boundary static tests so milestone demo paths do
  not silently define the M5 surface.
- Existing negative tests cover public unsupported paths: unsupported TTIR ops,
  contract errors, target policy errors, runtime ABI/grid errors, and stream
  errors.

### Workflow 8: Public API Freeze

- Trimmed `tvm.contrib.triton_tvm.__all__` to stable M5-entry APIs.
- Kept legacy `validate_cuda_*` wrappers importable for compatibility warnings,
  but removed them from `__all__`.
- Removed `TTIRReader` and normalized graph structures from the top-level
  public export surface; they remain accessible from their implementation
  modules for tests and internal tooling.
- Public docstrings now state unsupported behavior explicitly; no path silently
  falls back to native Triton.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_reporting.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_static.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_inductor_subset.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Results:

```text
test_triton_tvm_reporting.py: 3 passed
test_triton_tvm_static.py: 21 passed
test_triton_tvm_inductor_subset.py: 16 passed
test_triton_tvm.py: 24 passed
test_triton_tvm_perf.py: 1 skipped
```

## 2026-05-23: M5 Inductor Integration Prototype

### Scope

This checkpoint adds the first live TorchInductor hook.  It does not expand
Triton semantic coverage: M5 routes only already-supported `pointwise_flat`
Inductor kernels through TVM, and records explicit native fallback for
unsupported or failed kernels.

### Implementation Changes

- Added experimental APIs under `tvm.contrib.triton_tvm.inductor`:
  `TritonTVMInductorConfig`, `TritonTVMInductorSession`, and
  `make_triton_tvm_inductor_backend`.
- The backend wraps TorchInductor by temporarily patching
  `torch._inductor.async_compile.AsyncCompile.triton` during a single
  `torch.compile` call.
- Supported `Grid1D` non-atomic pointwise kernels lower to TTIR, translate with
  `contract="pointwise_flat"`, build through TVM, and run with a TVM artifact.
- Unsupported kernels use native Triton fallback with a stable report bucket and
  non-empty `fallback_reason`.
- Added a process-local artifact cache keyed by `TritonTVMMeta.cache_key`.
  Disk cache remains disabled.
- Added a PyTorch launcher path that wraps CUDA tensors through DLPack and hands
  Inductor's raw CUDA stream to TVM with `tvm.cuda(device).set_raw_stream(...)`.
  Outputs remain owned by the Inductor wrapper.

### Tests Added

- Added `tests/python/contrib/test_triton_tvm_inductor_hook.py`.
- Static coverage checks that M5 hook APIs are experimental module APIs, not
  top-level `__all__` exports.
- Unit fallback coverage verifies unsupported source records stable report
  buckets and writes the shared report schema.
- CUDA E2E coverage verifies:
  - `torch.compile(..., backend=make_triton_tvm_inductor_backend(...))` runs a
    pointwise add/mul kernel through TVM;
  - repeated compilation in the same session hits the process-local cache;
  - an unsupported Inductor reduction keeps correctness through native fallback.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_reporting.py tests/python/contrib/test_triton_tvm_static.py tests/python/contrib/test_triton_tvm_inductor_subset.py tests/python/contrib/test_triton_tvm_inductor_hook.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Results:

```text
reporting/static/inductor_subset/inductor_hook: 44 passed
test_triton_tvm.py: 24 passed
test_triton_tvm_perf.py: 1 skipped
```

## 2026-05-23: M5.5 Inductor Hook Hardening Gate

### Scope

M5.5 is the mandatory gate before M6.  It does not add TTIR op semantics or
model coverage.  The checkpoint hardens the live Inductor hook lifecycle,
graph-level reporting, process-memory cache accounting, raw stream handoff,
fallback boundaries, and compile/runtime exception handling.

### Implementation Changes

- Added graph-level session records to `TritonTVMInductorSession`.  Reports now
  include `graph_summary` and `graphs` alongside kernel records.
- Each graph records `graph_id`, compile region, observed kernel order,
  TVM/native fallback counts, cache hits/misses, run count, fallback reason
  histogram, and compile failure status when applicable.
- Kernel records produced by the live hook now carry `graph_id` and
  `graph_kernel_index`, so one `torch.compile` graph can explain which kernels
  were replaced by TVM and which kernels used native Triton fallback.
- Added a process-local hook lifecycle guard around
  `AsyncCompile.triton`.  The patch is active only during the wrapped Inductor
  compile call, rejects nested active hooks, serializes hook installation, and
  restores the original method on success or exception.
- TVM launcher run accounting now updates both kernel-level `run_count` and the
  owning graph's `run_count`.
- Runtime errors still raise to the caller; they are recorded with stable
  status payloads and do not silently fall back to native Triton.

### Tests Added

- Added unit coverage for graph-level fallback records and process-local cache
  lifecycle reporting.
- Added hook reentrancy and restore tests, including a forced compile failure
  path that verifies `AsyncCompile.triton` is restored.
- Added raw stream handoff regression coverage that verifies the launcher sets
  Inductor's raw stream and restores TVM's stream to `0`.
- Added CUDA E2E coverage for a mixed graph: one supported pointwise Inductor
  Triton kernel runs through TVM while an unsupported reduction kernel in the
  same graph uses native Triton fallback.

### Documentation

- Promoted M5.5 to a mandatory M6 entry gate in the backend plan.
- Documented graph-level report fields, hook lifecycle rules, launcher stream
  behavior, and artifact cache lifecycle in the capability matrix.
- Cache remains process-local, session-local, in-memory only, and discarded
  with `TritonTVMInductorSession`; disk cache remains disabled.

### Validation

Commands:

```bash
conda run -n tvm-0.24.0 python -m py_compile python/tvm/contrib/triton_tvm/inductor.py tests/python/contrib/test_triton_tvm_inductor_hook.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_inductor_hook.py -k 'torch_compile'
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_reporting.py tests/python/contrib/test_triton_tvm_static.py tests/python/contrib/test_triton_tvm_inductor_subset.py tests/python/contrib/test_triton_tvm_inductor_hook.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm.py
conda run -n tvm-0.24.0 python -m pytest -q tests/python/contrib/test_triton_tvm_perf.py
```

Results:

```text
py_compile: passed
inductor_hook torch_compile subset: 3 passed
reporting/static/inductor_subset/inductor_hook: 49 passed
test_triton_tvm.py: 24 passed
test_triton_tvm_perf.py: 1 skipped
```
