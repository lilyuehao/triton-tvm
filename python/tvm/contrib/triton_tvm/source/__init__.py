"""Source identity and normalization APIs."""

from .identity import build_source_id, source_hash_from_text, stable_hash, ttir_hash_from_text
from .inductor import captured_inductor_jit_kernel, exported_inductor_source
from .normalize import normalize_source_record, validate_native_triton_audit
from .schema import SourceArtifactKind, SourceOrigin, SourceRecord, is_production_origin
from .triton_language import native_triton_source, native_triton_ttir

__all__ = [
    "SourceArtifactKind",
    "SourceOrigin",
    "SourceRecord",
    "build_source_id",
    "captured_inductor_jit_kernel",
    "exported_inductor_source",
    "is_production_origin",
    "native_triton_source",
    "native_triton_ttir",
    "normalize_source_record",
    "source_hash_from_text",
    "stable_hash",
    "ttir_hash_from_text",
    "validate_native_triton_audit",
]
