# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0.
"""Stable hashing helpers for source records."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def stable_json(payload: Mapping[str, Any] | list[Any] | tuple[Any, ...]) -> str:
    """Serialize a payload with stable key order."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(payload: Mapping[str, Any] | list[Any] | tuple[Any, ...] | str | bytes) -> str:
    """Return a sha256 hash for structured or textual content."""

    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = stable_json(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def source_hash_from_text(text: str) -> str:
    """Hash auditable source text."""

    return stable_hash(text)


def ttir_hash_from_text(ttir: str) -> str:
    """Hash textual TTIR."""

    return stable_hash(ttir)


def build_source_id(*, origin: str, artifact: str, content_hash: str, function_name: str | None) -> str:
    """Build a readable, stable source identifier."""

    suffix = content_hash[:16]
    name = function_name or "module"
    return f"{origin}:{artifact}:{name}:{suffix}"

