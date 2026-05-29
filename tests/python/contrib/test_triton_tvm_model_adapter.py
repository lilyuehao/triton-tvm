import pytest

from tvm.contrib.triton_tvm.models import (
    ModelAdapter,
    ModelOperatorInput,
    reject_adapter_support_declaration,
)
from tvm.contrib.triton_tvm.source import captured_inductor_jit_kernel


TTIR = """
module {
  tt.func public @pointwise(%x: !tt.ptr<f32>) {
    %0 = tt.load %x : !tt.ptr<f32> -> tensor<16xf32>
    tt.store %x, %0 : !tt.ptr<f32>, tensor<16xf32>
  }
}
"""


def test_model_adapter_materializes_inputs_without_declaring_support():
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="pointwise", target="cuda")
    adapter = ModelAdapter(
        "toy",
        (
            ModelOperatorInput(
                model_id="toy",
                operator_id="pointwise_0",
                execution_order=0,
                source=source,
                ttir=TTIR,
            ),
        ),
    )
    assert adapter.operators()[0].source is source
    assert adapter.atomic_inputs()[0].source_id == source.source_id
    with pytest.raises(ValueError):
        reject_adapter_support_declaration(model_id="toy")

