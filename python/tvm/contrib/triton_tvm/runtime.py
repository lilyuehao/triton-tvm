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
"""Build and runtime helpers for translated Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import tvm

from .contracts import validate_triton_tvm_contract
from .errors import TritonTVMContractError
from .translator import TritonTVMMeta


@dataclass
class TritonTVMArtifact:
    """Compiled artifact returned by ``build_triton_tvm``."""

    executable: tvm.runtime.Executable
    irmod: tvm.IRModule
    meta: TritonTVMMeta
    target: str

    def run(self, args: list[Any] | tuple[Any, ...], grid=None, stream=None):
        """Run the compiled kernel.

        ``grid`` is accepted to mirror the planned runtime API.  The current TIRX
        kernel computes its launch extent from the runtime scalar, so a provided
        grid is recorded by callers but not used by the launcher yet.
        """
        if stream is not None:
            raise NotImplementedError(
                "stream is not supported by TritonTVMArtifact.run in this prototype"
            )
        _validate_grid(grid, self.meta.grid)
        _validate_args(args, self.meta)
        return self.executable[self.meta.kernel_name](*list(args))


def build_triton_tvm(
    irmod: tvm.IRModule,
    meta: TritonTVMMeta,
    passes: list[Any] | tuple[Any, ...] | None = None,
    target: str = "cuda",
) -> TritonTVMArtifact:
    """Apply optional TVM passes and compile the translated IRModule."""
    if tvm.target.Target(target).kind.name != tvm.target.Target(meta.target).kind.name:
        raise ValueError(
            f"Build target {target!r} does not match translated target {meta.target!r}"
        )
    passes = list(passes or [])
    mod = irmod
    for tvm_pass in passes:
        mod = tvm_pass(mod)
    validate_triton_tvm_contract(mod, meta.contract)
    executable = tvm.compile(mod, target=target)
    return TritonTVMArtifact(executable=executable, irmod=mod, meta=meta, target=target)


def _validate_grid(grid, meta_grid) -> None:
    if grid is None or callable(meta_grid):
        return
    expected = _normalize_grid(meta_grid)
    actual = _normalize_grid(grid)
    if expected is not None and actual != expected:
        raise ValueError(f"Grid {actual} does not match translated grid {expected}")


def _normalize_grid(grid):
    if grid is None or callable(grid):
        return None
    if isinstance(grid, Integral):
        return (int(grid),)
    if isinstance(grid, list | tuple):
        return tuple(int(x) for x in grid)
    raise TypeError(f"Unsupported grid type: {type(grid)!r}")


def _validate_args(args: list[Any] | tuple[Any, ...], meta: TritonTVMMeta) -> None:
    if len(args) != len(meta.abi):
        raise ValueError(f"Expected {len(meta.abi)} runtime arguments, got {len(args)}")

    for i, (arg, spec) in enumerate(zip(args, meta.abi, strict=True)):
        kind = spec["kind"]
        dtype = spec["dtype"]
        name = spec["name"]
        if kind == "pointer":
            if not hasattr(arg, "dtype") or not hasattr(arg, "device"):
                raise TypeError(f"Argument {i} ({name}) must be a TVM tensor")
            if str(arg.dtype) != dtype:
                raise TypeError(
                    f"Argument {i} ({name}) has dtype {arg.dtype}, expected {dtype}"
                )
        elif kind == "scalar":
            _validate_scalar_arg(i, name, arg, dtype)
        else:
            raise TritonTVMContractError(f"Unknown ABI kind {kind!r} for argument {name}")


def _validate_scalar_arg(index: int, name: str, arg: Any, dtype: str) -> None:
    if dtype == "bool":
        if not isinstance(arg, bool):
            raise TypeError(f"Argument {index} ({name}) must be bool")
    elif dtype.startswith("int") or dtype.startswith("uint"):
        if isinstance(arg, bool) or not isinstance(arg, Integral):
            raise TypeError(f"Argument {index} ({name}) must be an integer scalar")
        if dtype.startswith("uint") and int(arg) < 0:
            raise ValueError(f"Argument {index} ({name}) must be non-negative for {dtype}")
    elif dtype.startswith("float") or dtype == "bfloat16":
        if isinstance(arg, bool) or not isinstance(arg, Real):
            raise TypeError(f"Argument {index} ({name}) must be a numeric scalar")
