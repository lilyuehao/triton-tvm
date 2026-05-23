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
"""Adapter passes for the Triton-to-TVM prototype."""

from __future__ import annotations

import tvm

from .contracts import validate_triton_tvm_contract


def ValidateTritonKernelTIR(contract: str = "pointwise_minimal"):
    """Return the validation pass for a Triton TVM contract.

    Pre-M5 keeps validation separate from normalization.  The translator emits
    canonical TIRX directly today; a pass named ``NormalizeTritonKernelTIR``
    should only return once it performs real IR rewrites.
    """

    @tvm.ir.transform.module_pass(opt_level=0, name="triton_tvm.ValidateTritonKernelTIR")
    def _pass(mod, _ctx):  # pylint: disable=unused-argument
        validate_triton_tvm_contract(mod, contract)
        return mod

    return _pass
