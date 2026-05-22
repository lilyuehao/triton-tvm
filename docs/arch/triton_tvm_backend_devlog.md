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
