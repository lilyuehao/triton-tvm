# Triton TVM Capability Matrix

This is the unified capability and report matrix.  It is separate from the
contract definition page because M5 fallback decisions need one schema that can
describe pointwise, reduction, norm, and live Inductor hook results together.

## Report Schema

Reports are emitted as `triton_tvm_capability_report` with `schema_version = 1`.
The same schema is used for M3 audit, M3.5 promoted pointwise cases, M4
reduction cases, `norm_single_row` cases, M5/M5.5/Pre-M6 Inductor hook records,
M6/M6.5 model corpus audits, and the Pre-M7 debt gate.

Required top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Report schema version. |
| `report_kind` | Always `triton_tvm_capability_report`. |
| `purpose` | Human-readable report purpose. |
| `corpus` | Corpus name when the report has one dominant source. |
| `generated_at` | ISO timestamp. |
| `summary` | Counts by status bucket, contract, and corpus. |
| `op_histogram` | Aggregated TTIR op counts. |
| `type_histogram` | Aggregated TTIR type coverage. |
| `unsupported_buckets` | Backward-compatible alias of `summary.status_buckets`. |
| `collection_errors` | Non-kernel collection or hook-entry failures, including Pre-M6 private API guard failures. |
| `kernels` | Per-kernel records. |
| `graph_summary` | M5.5/Pre-M6 graph-level aggregate for live Inductor hook sessions. |
| `graphs` | M5.5/Pre-M6 per-graph records for one or more wrapped Inductor compile calls. |
| `private_api_guard` | Latest Pre-M6 hook-entry API/version/signature guard result. |
| `runtime_counter_policy` | Declares that `run_count` is TVM-launcher-only and native fallback counts are compile-time decisions. |
| `report_flush_policy` | Declares when live hook report files are written. |
| `dependency_versions` | M6/M6.5 model corpus dependency versions and CUDA availability. |
| `model_summary` | M6/M6.5 model-level aggregate: model counts, fallback kernels, and blocker classes. |
| `models` | M6/M6.5 per-model records for ViT/YOLO/Llama-like audit fixtures. |
| `blockers` | M6/M6.5 ranked blockers across kernel and model collection failures. |
| `pre_m7` | Pre-M7 reader/builder/taxonomy gate metadata for M7 entry ordering. |

Per-kernel records use:

| Field | Meaning |
|---|---|
| `corpus` | `m3`, `m35`, `m4`, `m5_inductor_hook`, or a more specific audit corpus name. |
| `case_name` | Stable case name. |
| `kernel_name` | TTIR / Triton kernel symbol. |
| `contract` | Canonical contract attempted. |
| `op_counts` | Per-kernel TTIR op counts. |
| `types` | Normalized TTIR type list. |
| `load_count` / `store_count` | Memory op counts for quick scanning. |
| `cache_key` / `cache_hit` | M5 process-local artifact cache metadata when applicable. |
| `run_count` | Number of successful TVM launcher calls observed by the M5 hook; this is not a full graph launch count. |
| `native_fallback_count` | Number of compile-time native fallback decisions recorded by the M5 hook; native `.run(...)` calls are not wrapped in Pre-M6. |
| `graph_id` / `graph_kernel_index` | M5.5 graph ownership and kernel order when the record came from a live hook. |
| `model_family` / `model_case` | M6/M6.5 model ownership when the record came from the model corpus. |
| `grid_type` / `num_reduction` / `atomic_add_found` | M6/M6.5 Inductor metadata used to classify model blockers. |
| `size_hints` | M6/M6.5 Inductor size hints, kept for shape/blocker diagnosis. |
| `blocker_class` | M6/M6.5 normalized class such as `reduction`, `matmul_dot`, `atomic`, `grid`, or a fallback reason. |
| `translate_status` | Stable status payload. |

Status payload:

| Field | Meaning |
|---|---|
| `ok` | Whether the selected path translated successfully. |
| `bucket` | Stable report bucket. |
| `fallback_reason` | Empty on success; otherwise the M5 fallback reason. |
| `error_type` | Python exception type when applicable. |
| `message` | Short diagnostic message when applicable. |

## Stable Buckets

| Bucket | Source |
|---|---|
| `translated` | Successful translation/build classification. |
| `unsupported_ttir_op` | `UnsupportedTTIROpError`. |
| `contract_error` | `TritonTVMContractError` or `UnsupportedContractError`. |
| `target_policy_error` | `UnsupportedTargetPolicyError`. |
| `unsupported_stream` | `UnsupportedStreamError`. |
| `input_error` | Invalid user input, bad runtime argument type, or Pre-M6 private API/version/signature guard failure. |
| `collection_error` | M6/M6.5 model collection produced no Triton kernels or hit a non-kernel collection failure. |
| `triton_tvm_error` | Other known Triton TVM errors. |
| `internal_error` | Unexpected exceptions. |

## Contract Capability Rows

| Contract | Corpus Coverage | Report Contract Name | Fallback Boundary |
|---|---|---|---|
| `pointwise_minimal` | Standalone single-output pointwise. | `pointwise_minimal` | Unsupported pointwise features must report a bucket; no fallback is silent. |
| `pointwise_flat` | Standalone multi-output plus M3.5/M5 Inductor pointwise capability variants. | `pointwise_flat` | M5 live hook routes supported kernels through TVM; unsafe no-`other` loads, unsupported indices, dot, atomics, reductions, and build failures are bucketed before native fallback. |
| `reduction_minimal` | M4 row sum and simple row-wise reductions. | `reduction_minimal` | Unsupported axes, combiners, grids, pointer patterns, and masked-load forms are bucketed. |
| `norm_single_row` | M4 single-row RMS/RMSNorm/LayerNorm surface. | `norm_single_row` | Shares M4 reduction buckets; no new reduction semantics are implied. |

## TTIR Reader Boundary

Textual TTIR parsing is a controlled temporary implementation:

- `ttir.py` owns textual parsing and `normalize_ttir_input`.
- `translator.py` consumes `NormalizedTTIROpGraph` and does not instantiate
  `TTIRReader`.
- `NormalizedTTIROpGraph` snapshot tests are the guardrail for reader changes.
- Adding support for a new TTIR op requires updating the reader snapshot before
  translator support is accepted.

## Builder Debt

Current builder: controlled TVMScript source plus `tvm.script.from_source`.

Risk: string-based IR construction grows fragile as more contracts and rewrites
are added.

M5 allowed: yes, as long as golden-shape tests cover all supported contracts.

M6 decision: keep the TVMScript source builder.  M6/M6.5 adds model corpus
audit/reporting only and does not expand TTIR semantics, contracts, or rewrite
passes.  Direct node construction is deferred to Pre-M7, where reader/builder
cleanup is the explicit milestone goal.

Pre-M7 decision: keep the TVMScript source builder for M7 entry.  Direct node
construction was evaluated, but replacing the builder before M7 would duplicate
the current pointwise/reduction template builders without reducing the immediate
M7 blocker risk.  The guardrail is a single `tvm.script.from_source` call behind
`translator.py`'s builder boundary, static tests that forbid `TTIRReader` use in
the translator, and TVMScript golden-shape coverage for supported contracts.

## M5 Inductor Hook Boundary

- `make_triton_tvm_inductor_backend(...)` is an experimental custom backend for
  `torch.compile`; it is not exported from top-level `tvm.contrib.triton_tvm`.
- It is not a stable ABI.  Pre-M6 freezes only `fallback="native"`, supported
  `pointwise_flat` replacement, and explicit unsupported native fallback policy.
- The hook temporarily patches `AsyncCompile.triton` only during the wrapped
  Inductor compile call.
- Before patching, the hook checks that `torch.compile`,
  `torch._inductor.compile_fx.compile_fx`, Triton version metadata, and
  `AsyncCompile.triton`'s callable signature match the experimental contract.
  Unsupported private API drift fails fast with an `input_error`
  `collection_errors` entry and does not enter a half-patched state.
- Supported `Grid1D` non-atomic pointwise kernels use `pointwise_flat`, TVM
  build, process-local artifact cache, and the TVM launcher.
- Unsupported or failed kernels use native Triton fallback with a non-empty
  `fallback_reason`; fallback is explicit in the report, not silent.
- Disk cache remains disabled in metadata.  M5 cache means only the session-local
  in-memory artifact map keyed by `TritonTVMMeta.cache_key`.
- Reports are written eagerly after compile/run accounting only when
  `report_dir` is configured or `write_report(out_dir=...)` is called.

## M5.5 Graph-Level Hook Hardening

M5.5 does not add TTIR op support or model coverage.  It stabilizes how the live
Inductor hook is installed, observed, and unwound.

Graph records use:

| Field | Meaning |
|---|---|
| `graph_id` | Session-local stable graph id such as `graph_0`. |
| `compile_region_name` | Optional name forwarded by Inductor backend kwargs. |
| `status` | `compiling`, `completed`, or `internal_error`. |
| `total_kernels` | Number of Inductor Triton kernels observed in the graph. |
| `translated_kernels` | Kernels replaced by TVM artifacts. |
| `native_fallback_kernels` | Kernels delegated to native Triton fallback. |
| `cache_hits` / `cache_misses` | Process-local artifact cache decisions for translated kernels. |
| `run_count` | Successful TVM launcher calls attributed to the graph. |
| `fallback_reasons` | Histogram of non-empty fallback reasons. |
| `kernel_names` | Observed Inductor Triton kernel order. |
| `kernel_record_indices` | Indices into the top-level `kernels` list. |
| `error_type` / `message` | Present when graph compilation exits through an exception. |

Hook lifecycle rules:

- `AsyncCompile.triton` is patched only inside one wrapped Inductor compile call.
- A process-local reentrancy/thread-safety guard prevents nested active hooks and
  serializes concurrent hook installation.
- `AsyncCompile.triton` must be restored on successful compile, compile-time
  exception, and native fallback paths.
- A single graph may contain multiple observed Triton kernels; supported
  pointwise kernels may run through TVM while unsupported kernels in the same
  graph use native Triton fallback.

Launcher and cache lifecycle:

- The TVM launcher accepts PyTorch CUDA tensors through DLPack and scalar args
  unchanged after ABI validation.
- If Inductor provides a raw CUDA stream, the launcher calls
  `tvm.cuda(device).set_raw_stream(int(stream))`, runs the TVM artifact, then
  restores TVM's stream to `0`.
- The launcher does not allocate outputs and does not synchronize; the Inductor
  wrapper owns output tensors and PyTorch stream ordering.
- The artifact cache is `TritonTVMInductorSession.artifacts`: process-local,
  session-local, in-memory only, and discarded when the session is discarded.
  There is no disk cache and no cross-process reuse in M5.5.

## Pre-M6 Integration Contract Freeze

Pre-M6 does not add TTIR op support, model coverage, or scheduling changes.  It
freezes the integration contract needed before M6 graph-level work:

- `TritonTVMInductorSession.report()` has snapshot coverage for `summary`,
  `graph_summary`, `graphs`, `kernels[*].graph_id`,
  `kernels[*].graph_kernel_index`, `cache_key`, `cache_hit`,
  `native_fallback_count`, and `run_count`.
- JSON graph consistency is bidirectional: every
  `graphs[*].kernel_record_indices` entry points at an existing kernel record,
  and that kernel record must carry the same `graph_id` and matching
  `graph_kernel_index`.
- Markdown includes a graph summary table and explicitly states that JSON
  `graphs` and `kernels` are authoritative for per-graph details.
- `TritonTVMMeta.cache_key` identifies a single kernel artifact only.  It does
  not include `graph_id`, compile region, `run_count`, or session transient
  identity; graph-level cache hits and misses only aggregate kernel-level cache
  decisions.
- Runtime counters are intentionally narrow: TVM-backed kernels increment
  `run_count`, while native fallback records use `native_fallback_count` as a
  compile-time fallback decision count.

## M6/M6.5 Model Corpus Audit

M6/M6.5 does not define model-level success as fallback-free execution.  It
defines success as a stable, reproducible audit that explains why selected
external-library models are not yet full TVM / fallback-free.

Dependency policy:

- Install the balanced external stack in `tvm-0.24.0`:
  `transformers>=4,<5` and `ultralytics>=8,<9`.
- Model weights are random initialization only; no pretrained weights are
  downloaded.
- Reports record dependency versions under `dependency_versions`.

Model records use:

| Field | Meaning |
|---|---|
| `model_family` | `vit`, `yolo`, `llama`, or a caller-provided family. |
| `model_case` | Stable fixed-shape case name. |
| `status` | `completed`, `compile_error`, `zero_kernels`, or `not_run`. |
| `kernel_count` | Number of captured Inductor Triton kernels. |
| `translated_kernels` | Kernels classified as TVM-replaceable under the selected contract. |
| `native_fallback_kernels` | Kernels that remain native fallback blockers. |
| `full_tvm_runnable` | True only when every captured kernel is TVM-replaceable and at least one kernel was captured. |
| `wrapper_paths` | Saved native Inductor wrapper sources. |

Blocker records are ranked by:

1. impacted model count, descending;
2. kernel count, descending;
3. bucket, blocker class, and first model name.

`python -m tvm.contrib.triton_tvm.model_corpus --builtin-model-corpus --out-dir
...` captures native Inductor wrappers, extracts Triton kernels, writes kernel
sources and TTIR dumps, and emits `report.json` / `report.md`.  The live
replacement path remains owned by the M5/M5.5 hook tests.

## Pre-M7 Debt Gate

Pre-M7 does not add TTIR op support, model coverage, scheduling changes, or live
model replacement behavior.  It closes the M7 entry debt gate:

- M6/M6.5 model corpus reports include additive `pre_m7` metadata with
  `taxonomy_version = 1` and
  `builder_decision = "keep_tvmscript_source_builder_for_m7_entry"`.
- Top-level status buckets remain stable.  Pre-M7 classifies unsupported detail
  through `fallback_reason` and `blocker_class`, so report diffs continue to use
  bucket/blocker/model deltas without migration churn.
- `reader_snapshot_classes` groups model-corpus kernels into `pointwise`,
  `broadcast_view_index`, `reduction`, `matmul_dot`, `atomic_grid`, and
  `attention_adjacent`.
- `m7_entry_blockers` filters ranked blockers to pointwise/broadcast/view/index
  candidates and excludes collection failures, reduction, matmul/dot, atomic,
  grid, and attention-adjacent blockers.
- Static Pre-M7 tests cover the reader snapshot classes with hand-written TTIR
  and confirm that the translator keeps one controlled TVMScript source-builder
  boundary.
