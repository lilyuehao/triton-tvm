# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Triton frontend entry points."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TTIRArtifact:
    """Optimized TTIR and the specialization metadata used to produce it."""

    ttir: str
    kernel_name: str
    signature: dict[str, str]
    constexprs: dict[str, Any]
    triton_version: str
    source_hash: str


def lower_to_ttir(
    jit_fn,
    signature: dict[str, str],
    constexprs: dict[str, Any] | None = None,
    *,
    dump_path: str | Path | None = None,
    attrs: Any | None = None,
) -> TTIRArtifact:
    """Lower a Triton JIT function to optimized textual TTIR.

    This pre-M5 API is intentionally narrow and pinned to Triton 3.7.0.  It
    does not compile TVM code, does not install an Inductor hook, and does not
    silently fall back to native Triton when lowering fails.

    Parameters
    ----------
    jit_fn:
        A ``triton.runtime.jit.JITFunction``.

    signature:
        Triton ASTSource signature, including ``"constexpr"`` entries.

    constexprs:
        Compile-time values used to specialize the Triton kernel.

    dump_path:
        Optional path where the textual TTIR should be written.

    attrs:
        Optional Triton argument attributes.  Standalone kernels usually do not
        need this, but TorchInductor kernels carry alignment/divisibility attrs
        through their generated autotuner metadata.
    """
    import triton  # pylint: disable=import-outside-toplevel

    if triton.__version__ != "3.7.0":
        raise RuntimeError(
            "tvm.contrib.triton_tvm prototype is pinned to triton 3.7.0, "
            f"but found {triton.__version__}"
        )

    constexprs = dict(constexprs or {})
    source = triton.compiler.ASTSource(
        fn=jit_fn,
        signature=dict(signature),
        constexprs=constexprs,
        attrs=attrs,
    )
    compiled = triton.compiler.compile(source, options={"constexprs": constexprs})
    ttir = compiled.asm["ttir"]

    if dump_path is not None:
        Path(dump_path).write_text(ttir, encoding="utf-8")

    return TTIRArtifact(
        ttir=ttir,
        kernel_name=getattr(jit_fn, "__name__", getattr(jit_fn, "name", compiled.name)),
        signature=dict(signature),
        constexprs=constexprs,
        triton_version=triton.__version__,
        source_hash=_source_hash(jit_fn),
    )


def _source_hash(jit_fn) -> str:
    raw_src = getattr(jit_fn, "raw_src", None) or getattr(jit_fn, "src", "")
    if not isinstance(raw_src, str):
        raw_src = repr(raw_src)
    return hashlib.sha256(raw_src.encode("utf-8")).hexdigest()
