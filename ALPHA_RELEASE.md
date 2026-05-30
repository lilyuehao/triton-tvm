# v0.1.1-alpha Release Notes

`v0.1.1-alpha` is a public alpha snapshot of the Triton-TVM integration in this
fork. It is suitable for compiler-infrastructure review and fixed-shape
correctness reproduction, not for production inference deployment.

## Support Scope

- Fixed-shape ViT correctness route.
- Source identity capture for Triton/TorchInductor-derived inputs.
- Atomic DAG construction from low-level operator evidence.
- Contract validation before semantic-region promotion.
- Manifest and capability lookup before runtime admission.
- TVM packed-function execution through `TIRRegionArtifact`.
- 14 top-level semantic operators over 16 source/lowering route records.

## Verification

Run from the repository root:

```bash
export PYTHONPATH="$PWD/python"

python -m tvm.contrib.triton_tvm.models.vit \
  --target llvm \
  --atomic-dag-gated-te-tir \
  --out-dir /tmp/triton_tvm_vit_atomic_dag_gated_te_tir
```

Run the focused test set:

```bash
python -m pytest \
  tests/python/contrib/test_triton_tvm_source.py \
  tests/python/contrib/test_triton_tvm_atomic.py \
  tests/python/contrib/test_triton_tvm_semantic.py \
  tests/python/contrib/test_triton_tvm_manifest.py \
  tests/python/contrib/test_triton_tvm_registry.py \
  tests/python/contrib/test_triton_tvm_model_adapter.py \
  tests/python/contrib/test_triton_tvm_runtime_admission.py \
  tests/python/contrib/test_triton_tvm_runtime_e2e.py \
  tests/python/contrib/test_triton_tvm_executor.py \
  tests/python/contrib/test_triton_tvm_reports.py \
  tests/python/contrib/test_triton_tvm_alpha_path.py \
  tests/python/contrib/test_triton_tvm_vit_runner.py \
  tests/python/contrib/test_triton_tvm_static_hygiene.py
```

## Non-Goals

- No arbitrary Triton-kernel import.
- No dynamic-shape support.
- No graph optimization or operator fusion.
- No automatic backend selection.
- No production CUDA performance claim.
- No general model-import claim.

## Expected Report Signals

The ViT report should expose source route count, top-level operator count,
artifact-source breakdown, allclose status, model-output comparison status, and
explicit runtime-claim fields.

The alpha correctness route uses `atomic_dag_gated_te_tir`: Atomic DAG records
act as the admission and evidence gate, while the executable TVM IR is produced
from fixed-shape TE/TIR templates. This is a correctness claim, not a generic
Atomic-DAG-node-to-TIR lowering claim.
