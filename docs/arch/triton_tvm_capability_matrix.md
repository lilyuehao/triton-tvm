# Triton TVM Capability Matrix

This is the unified capability and report matrix.  It is separate from the
contract definition page because M5 fallback decisions need one schema that can
describe pointwise, reduction, norm, and live Inductor hook results together.

## Report Schema

Reports are emitted as `triton_tvm_capability_report` with `schema_version = 1`.
The same schema is used for M3 audit, M3.5 promoted pointwise cases, M4
reduction cases, `norm_single_row` cases, and M5/M5.5 Inductor hook records.

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
| `kernels` | Per-kernel records. |
| `graph_summary` | M5.5 graph-level aggregate for live Inductor hook sessions. |
| `graphs` | M5.5 per-graph records for one or more wrapped Inductor compile calls. |

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
| `run_count` | Number of successful TVM launcher calls observed by the M5 hook. |
| `native_fallback_count` | Number of compile-time native fallback decisions recorded by the M5 hook. |
| `graph_id` / `graph_kernel_index` | M5.5 graph ownership and kernel order when the record came from a live hook. |
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
| `input_error` | Invalid user input, such as bad grid or argument type. |
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

Required by M6: evaluate a direct node builder, or keep the TVMScript source
builder with a documented rationale and matching golden coverage.

## M5 Inductor Hook Boundary

- `make_triton_tvm_inductor_backend(...)` is an experimental custom backend for
  `torch.compile`; it is not exported from top-level `tvm.contrib.triton_tvm`.
- The hook temporarily patches `AsyncCompile.triton` only during the wrapped
  Inductor compile call.
- Supported `Grid1D` non-atomic pointwise kernels use `pointwise_flat`, TVM
  build, process-local artifact cache, and the TVM launcher.
- Unsupported or failed kernels use native Triton fallback with a non-empty
  `fallback_reason`; fallback is explicit in the report, not silent.
- Disk cache remains disabled in metadata.  M5 cache means only the session-local
  in-memory artifact map keyed by `TritonTVMMeta.cache_key`.

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
