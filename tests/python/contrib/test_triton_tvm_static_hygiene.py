from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "python/tvm/contrib/triton_tvm"
TESTS = ROOT / "tests/python/contrib"
DOCS = ROOT / "docs/arch"

MILESTONE_MODULE = re.compile(r"^m(?:9|10|11|12|13|14)(?:[0-9a-z_].*)?\.py$")
MILESTONE_TEST = re.compile(r"^test_triton_tvm_m(?:9|10|11|12|13|14)(?:[0-9a-z_].*)?\.py$")

BANNED_ACTIVE_PATTERNS = (
    "captured_triton_jit",
    "canonical_generated_triton",
    "generated_tl_dot_bridge",
    "wrapper_semantics_direct",
    "synthetic_fixture",
    r"\bsource_kind\b",
    r"\bmodel_binding_origin\b",
    "python_torch_host_staged",
    "device_torch_cuda",
    "wrapper_extern",
    "host_staged",
    "strict_wrapper",
    "performance_dashboard",
    "perf_optimization",
    "provider_runtime_optimization",
    "p2_dashboard",
)

MILESTONE_IMPORT_PATTERNS = (
    r"tvm\.contrib\.triton_tvm\.m(?:9|10|11|12|13|14)(?:[0-9a-z_]\w*)?",
    r"from tvm\.contrib\.triton_tvm import m(?:9|10|11|12|13|14)(?:[0-9a-z_]\w*)?",
    r"from \.m(?:9|10|11|12|13|14)(?:[0-9a-z_]\w*)? import",
    r"from \. import m(?:9|10|11|12|13|14)(?:[0-9a-z_]\w*)?",
)


def test_active_package_has_no_milestone_named_modules_recursively():
    bad = sorted(path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py") if MILESTONE_MODULE.match(path.name))
    assert bad == []


def test_active_tests_have_only_current_filenames():
    bad = sorted(path.name for path in TESTS.glob("test_triton_tvm*.py") if MILESTONE_TEST.match(path.name))
    assert bad == []


def test_active_package_has_no_legacy_identity_literals_or_imports():
    patterns = tuple(re.compile(pattern) for pattern in BANNED_ACTIVE_PATTERNS + MILESTONE_IMPORT_PATTERNS)
    offenders = _scan_files(PACKAGE.rglob("*.py"), patterns)
    assert offenders == []


def test_active_tests_have_no_legacy_imports_or_fixture_credit_literals():
    patterns = tuple(re.compile(pattern) for pattern in BANNED_ACTIVE_PATTERNS + MILESTONE_IMPORT_PATTERNS)
    files = [path for path in TESTS.glob("test_triton_tvm*.py") if path.name != "test_triton_tvm_static_hygiene.py"]
    offenders = _scan_files(files, patterns)
    assert offenders == []


def test_no_legacy_reference_archive_in_active_package():
    assert not (PACKAGE / "legacy_reference").exists()


def test_public_triton_tvm_docs_do_not_contain_local_workstation_paths():
    patterns = (re.compile(r"/home/"), re.compile(r"triton-tvm-workbench"))
    doc_roots = (
        ROOT / "README.md",
        ROOT / "ALPHA_SCOPE.md",
        ROOT / "ALPHA_RELEASE.md",
        DOCS / "triton_tvm",
        DOCS / "triton_tvm_backend_plan.md",
        DOCS / "triton_tvm_backend_devlog.md",
        DOCS / "triton_tvm_backend_devlog_toc.md",
        DOCS / "triton_tvm_contracts.md",
        DOCS / "triton_tvm_capability_matrix.md",
        PACKAGE / "README.md",
    )
    files = []
    for root in doc_roots:
        if root.is_dir():
            files.extend(root.rglob("*.md"))
        else:
            files.append(root)
    offenders = _scan_files(files, patterns)
    assert offenders == []


def _scan_files(paths, patterns):
    offenders = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        for index, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                offenders.append(f"{path.relative_to(ROOT)}:{index}")
    return offenders
