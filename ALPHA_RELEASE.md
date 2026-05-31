# Triton-TVM Alpha Changelog

## v0.2.0-alpha

- ViT-S support: fixed-shape ViT-S/16 224x224 execution now routes through
  Triton-TVM artifacts, with reference execution retained as the correctness
  oracle.
- Autoscheduler: operator tuning is exposed as a package-level Triton-TVM
  capability with workload deduplication and reusable MetaSchedule records.
- Config-file entrypoint: model runs now use a JSON config file to describe the
  route, weights, checks, benchmark settings, and explicit operator tuning
  selection.
- Runtime optimizations: packed TVM execution adds reusable execution sessions,
  device-resident tensor setup, reduced per-op dispatch overhead, and optional
  CUDA graph replay for benchmark runs.

## v0.1.0-alpha

- Atomic DAG + TE backend.
- ViT-like toy model correctness path.
