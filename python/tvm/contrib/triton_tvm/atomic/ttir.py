# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information.
"""Small textual TTIR reader used by the atomic extractor."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TTIRType:
    raw: str
    dtype: str | None
    shape: tuple[int | str, ...] = ()
    is_pointer: bool = False


@dataclass(frozen=True)
class TTIRParam:
    name: str
    type: TTIRType


@dataclass
class TTIROp:
    name: str
    results: list[str]
    operands: list[str]
    result_types: list[TTIRType]
    attrs: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class NormalizedTTIRGraph:
    function_name: str
    params: list[TTIRParam]
    ops: list[TTIROp]
    raw_ttir: str


_FUNC_RE = re.compile(r"tt\.func\s+(?:public\s+)?@(?P<name>[A-Za-z_][\w$]*)\((?P<params>.*)\)")
_ASSIGN_RE = re.compile(r"(?P<results>(?:%[\w$.]+)(?:\s*,\s*%[\w$.]+)*)\s*=\s*(?P<body>.*)")
_OP_RE = re.compile(r'"?(?P<name>[A-Za-z_][\w.]+)"?(?P<rest>.*)')


def parse_ttir(ttir: str) -> NormalizedTTIRGraph:
    """Parse the TTIR subset needed for source-to-atomic evidence."""

    function_name = ""
    params: list[TTIRParam] = []
    ops: list[TTIROp] = []
    in_function = False
    for line in _normalized_lines(ttir):
        if line in {"module {", "{", "}"}:
            continue
        if not in_function:
            match = _FUNC_RE.search(line)
            if match:
                function_name = match.group("name")
                params = _parse_params(match.group("params"))
                in_function = True
            continue
        if line.startswith("}"):
            in_function = False
            continue
        op = _parse_op(line)
        if op is not None:
            ops.append(op)
    if not function_name:
        raise ValueError("No tt.func found in TTIR module")
    return NormalizedTTIRGraph(
        function_name=function_name,
        params=params,
        ops=ops,
        raw_ttir=ttir,
    )


def _normalized_lines(ttir: str) -> list[str]:
    lines: list[str] = []
    for raw in ttir.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\s+loc\([^)]*\)", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _parse_params(text: str) -> list[TTIRParam]:
    result: list[TTIRParam] = []
    depth = 0
    token = []
    items = []
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(token).strip())
            token = []
        else:
            token.append(char)
    if token:
        items.append("".join(token).strip())
    for item in items:
        if ":" not in item:
            continue
        name, type_text = item.split(":", 1)
        result.append(TTIRParam(_clean_name(name), _parse_type(type_text.strip())))
    return result


def _parse_op(line: str) -> TTIROp | None:
    results: list[str] = []
    body = line
    assign = _ASSIGN_RE.match(line)
    if assign:
        results = [_clean_name(item) for item in assign.group("results").split(",")]
        body = assign.group("body").strip()
    match = _OP_RE.match(body)
    if not match:
        return None
    name = match.group("name")
    rest = match.group("rest")
    operands = [_clean_name(item) for item in re.findall(r"%[\w$.]+", rest)]
    return TTIROp(
        name=name,
        results=results,
        operands=operands,
        result_types=_parse_result_types(rest),
        attrs=_parse_attrs(name, rest),
        raw=line,
    )


def _parse_result_types(rest: str) -> list[TTIRType]:
    if "->" not in rest:
        return []
    tail = rest.rsplit("->", 1)[1].strip()
    return [_parse_type(tail)]


def _parse_type(text: str) -> TTIRType:
    raw = text.strip()
    is_pointer = raw.startswith("!tt.ptr")
    dtype_match = re.search(r"\b(f16|f32|f64|bf16|i1|i8|i16|i32|i64)\b", raw)
    shape_match = re.search(r"tensor<([^x>]+(?:x[^x>]+)*)x", raw)
    shape: tuple[int | str, ...] = ()
    if shape_match:
        dims: list[int | str] = []
        for dim in shape_match.group(1).split("x"):
            dims.append(int(dim) if dim.isdigit() else dim)
        shape = tuple(dims)
    return TTIRType(raw=raw, dtype=dtype_match.group(1) if dtype_match else None, shape=shape, is_pointer=is_pointer)


def _parse_attrs(name: str, rest: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if name == "tt.get_program_id":
        axis = rest.split(":", 1)[0].strip()
        attrs["axis"] = axis
    for key, value in re.findall(r"([A-Za-z_][\w]*)\s*=\s*(-?\d+)", rest):
        attrs[key] = int(value)
    return attrs


def _clean_name(text: str) -> str:
    return text.strip().lstrip("%")

