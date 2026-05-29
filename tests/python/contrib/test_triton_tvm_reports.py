from pathlib import Path

from tvm.contrib.triton_tvm.reports import active_inventory, build_alpha_smoke_report, write_json_report
from tvm.contrib.triton_tvm.source import captured_inductor_jit_kernel


TTIR = """
module {
  tt.func public @pointwise(%x: !tt.ptr<f32>) {
    %0 = tt.load %x : !tt.ptr<f32> -> tensor<16xf32>
    tt.store %x, %0 : !tt.ptr<f32>, tensor<16xf32>
  }
}
"""


def test_reports_read_facts_and_keep_no_performance_claim(tmp_path):
    source = captured_inductor_jit_kernel(ttir=TTIR, function_name="pointwise", target="cuda")
    report = build_alpha_smoke_report(sources=[source], atomics=[], semantics=[], manifests=[])
    assert report["performance_claim"] is False
    assert report["source_summary"]["production"] == 1
    out = tmp_path / "report.json"
    write_json_report(out, report)
    assert out.exists()


def test_active_inventory_finds_active_package_and_tests():
    root = Path(__file__).resolve().parents[3]
    inventory = active_inventory(
        package_dir=root / "python/tvm/contrib/triton_tvm",
        test_dir=root / "tests/python/contrib",
    )
    assert "source/schema.py" in inventory["active_package_modules"]
    assert "test_triton_tvm_reports.py" in inventory["active_test_files"]
    assert inventory["performance_claim"] is False

