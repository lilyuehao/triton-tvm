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
"""Textual TTIR reader for the Triton-to-TVM prototype."""

from __future__ import annotations

import re
from typing import Any

from .op_graph import NormalizedTTIROpGraph, TTIROp, TTIRParam, TTIRType


_FUNC_RE = re.compile(
    r"tt\.func\s+(?:public\s+)?@(?P<name>[A-Za-z_][\w$]*)"
    r"\((?P<params>.*)\)\s*(?:attributes\s+(?P<attrs>\{.*\}))?\s*\{"
)
_ASSIGN_RE = re.compile(r"(?P<results>(?:%[\w$.]+)(?:\s*,\s*%[\w$.]+)*)\s*=\s*(?P<body>.*)")
_OP_RE = re.compile(r"(?P<name>[A-Za-z_][\w.]+)(?P<rest>.*)")


class TTIRReader:
    """Read Triton textual TTIR into a normalized operation graph.

    The parser is intentionally small and conservative.  It is not a general
    MLIR parser; it only normalizes the textual forms needed by the standalone
    pointwise path, and keeps all textual parsing contained in this module.
    """

    def read(self, ttir: str) -> NormalizedTTIROpGraph:
        """Parse a textual TTIR module."""
        lines = self._normalized_lines(ttir)
        function_name = ""
        params: list[TTIRParam] = []
        function_attrs: dict[str, Any] = {}
        ops: list[TTIROp] = []
        in_func = False

        for line in lines:
            if line in ("module {", "{", "}"):
                continue
            if not in_func:
                match = _FUNC_RE.match(line)
                if match:
                    function_name = match.group("name")
                    params = self._parse_params(match.group("params"))
                    function_attrs = self._parse_attr_dict(match.group("attrs") or "")
                    in_func = True
                continue

            if line == "}":
                in_func = False
                continue
            if line.startswith("}"):
                in_func = False
                continue

            op = self._parse_op(line)
            if op is not None:
                ops.append(op)

        if not function_name:
            raise ValueError("No tt.func found in TTIR module")

        return NormalizedTTIROpGraph(
            function_name=function_name,
            params=params,
            ops=ops,
            attrs=function_attrs,
            raw_ttir=ttir,
        )

    def _normalized_lines(self, ttir: str) -> list[str]:
        lines: list[str] = []
        for raw_line in ttir.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#loc"):
                continue
            line = _strip_loc_annotations(line).strip()
            if not line:
                continue
            lines.append(line)
        return lines

    def _parse_params(self, text: str) -> list[TTIRParam]:
        params: list[TTIRParam] = []
        for item in _split_top_level(text, ","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Malformed TTIR function parameter: {item}")
            name, type_text = item.split(":", 1)
            params.append(
                TTIRParam(_clean_value_name(name.strip()), parse_ttir_type(type_text.strip()))
            )
        return params

    def _parse_op(self, line: str) -> TTIROp | None:
        results: list[str] = []
        body = line
        assign = _ASSIGN_RE.match(line)
        if assign:
            results = [_clean_value_name(x) for x in assign.group("results").split(",")]
            body = assign.group("body").strip()

        match = _OP_RE.match(body)
        if not match:
            return None

        name = match.group("name")
        rest = match.group("rest").strip()
        attrs = self._parse_op_attrs(name, rest)
        operands = [_clean_value_name(x) for x in re.findall(r"%[\w$.]+", rest)]
        result_types = self._parse_result_types(name, rest)
        return TTIROp(
            name=name,
            results=results,
            operands=operands,
            result_types=result_types,
            attrs=attrs,
            raw=line,
        )

    def _parse_op_attrs(self, name: str, rest: str) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if name == "arith.constant":
            value_text = rest.split(":", 1)[0].strip()
            attrs["value"] = _parse_constant_value(value_text)
        elif name == "tt.get_program_id":
            axis = rest.split(":", 1)[0].strip()
            attrs["axis"] = axis
        elif name == "tt.make_range":
            match = re.search(
                r"\{\s*end\s*=\s*(-?\d+)\s*:\s*i\d+\s*,\s*start\s*=\s*(-?\d+)\s*:\s*i\d+\s*\}",
                rest,
            )
            if match:
                attrs["end"] = int(match.group(1))
                attrs["start"] = int(match.group(2))
        elif name == "arith.cmpi":
            predicate = rest.split(",", 1)[0].strip()
            attrs["predicate"] = predicate
        if name in ("tt.load", "tt.store"):
            raw_attrs = _extract_first_attr_dict(rest)
            if raw_attrs:
                attrs["raw_attrs"] = raw_attrs
                attrs["unknown_attrs"] = self._parse_attr_dict(raw_attrs)
        return attrs

    def _parse_result_types(self, name: str, rest: str) -> list[TTIRType]:
        if name in ("tt.return", "tt.store"):
            return []
        if name == "tt.splat" and "->" in rest:
            return [parse_ttir_type(rest.rsplit("->", 1)[1].strip())]
        if name in ("arith.extsi", "arith.extui") and " to " in rest:
            return [parse_ttir_type(rest.rsplit(" to ", 1)[1].strip())]
        if ":" not in rest:
            return []
        type_text = rest.rsplit(":", 1)[1].strip()
        if not type_text:
            return []
        types = [parse_ttir_type(item.strip()) for item in _split_top_level(type_text, ",")]
        if name == "arith.cmpi" and types:
            ty = types[0]
            return [TTIRType(raw=_format_ttir_type("bool", ty.shape), dtype="bool", shape=ty.shape)]
        if name == "tt.load" and types:
            ty = types[0]
            if ty.is_pointer:
                return [_loaded_pointer_type(ty)]
        if name == "tt.addptr" and types:
            return [types[0]]
        return types

    def _parse_attr_dict(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        attrs: dict[str, Any] = {}
        for item in _split_top_level(text.strip("{} "), ","):
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            attrs[key.strip()] = value.strip()
        return attrs


def parse_ttir_type(type_text: str) -> TTIRType:
    """Parse the subset of TTIR type strings needed by the prototype."""
    type_text = type_text.strip()
    tensor_match = re.fullmatch(r"tensor<(?P<shape>\d+(?:x\d+)*)x(?P<elem>.+)>", type_text)
    if tensor_match:
        shape = tuple(int(x) for x in tensor_match.group("shape").split("x"))
        elem = tensor_match.group("elem")
        elem_type = parse_ttir_type(elem)
        return TTIRType(
            raw=type_text,
            dtype=elem_type.dtype,
            shape=shape,
            is_pointer=elem_type.is_pointer,
        )

    ptr_match = re.fullmatch(r"!tt\.ptr<(?P<elem>.+)>", type_text)
    if ptr_match:
        elem_type = parse_ttir_type(ptr_match.group("elem"))
        return TTIRType(raw=type_text, dtype=elem_type.dtype, is_pointer=True)

    return TTIRType(raw=type_text, dtype=_normalize_dtype(type_text))


def _normalize_dtype(dtype: str) -> str:
    dtype = dtype.strip()
    mapping = {
        "i1": "bool",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "bf16": "bfloat16",
    }
    if dtype in mapping:
        return mapping[dtype]
    int_match = re.fullmatch(r"([iu])(\d+)", dtype)
    if int_match:
        prefix = "int" if int_match.group(1) == "i" else "uint"
        return prefix + int_match.group(2)
    return dtype


def _loaded_pointer_type(ty: TTIRType) -> TTIRType:
    return TTIRType(raw=_format_ttir_type(ty.dtype, ty.shape), dtype=ty.dtype, shape=ty.shape)


def _format_ttir_type(dtype: str, shape: tuple[int, ...]) -> str:
    dtype_text = _denormalize_dtype(dtype)
    if shape:
        shape_text = "x".join(str(dim) for dim in shape)
        return f"tensor<{shape_text}x{dtype_text}>"
    return dtype_text


def _denormalize_dtype(dtype: str) -> str:
    mapping = {
        "bool": "i1",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "bfloat16": "bf16",
    }
    if dtype in mapping:
        return mapping[dtype]
    int_match = re.fullmatch(r"(u?int)(\d+)", dtype)
    if int_match:
        prefix = "u" if int_match.group(1) == "uint" else "i"
        return prefix + int_match.group(2)
    return dtype


def _clean_value_name(name: str) -> str:
    name = name.strip()
    if name.startswith("%"):
        name = name[1:]
    return name


def _parse_constant_value(text: str) -> int | float | bool | str:
    text = text.strip()
    dense_match = re.fullmatch(r"dense<(.+)>", text)
    if dense_match:
        text = dense_match.group(1)
    if text in ("true", "false"):
        return text == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text, 0)
    except ValueError:
        return text


def _strip_loc_annotations(text: str) -> str:
    result: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(" loc(", i):
            i += len(" loc(")
            depth = 1
            while i < len(text) and depth:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _extract_first_attr_dict(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _split_top_level(text: str, delimiter: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    pairs = {"<": ">", "(": ")", "{": "}", "[": "]"}
    closers = set(pairs.values())
    for i, ch in enumerate(text):
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth -= 1
        elif ch == delimiter and depth == 0:
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:])
    return items
