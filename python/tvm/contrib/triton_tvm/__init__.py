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
"""Prototype Triton language frontend backed by TVM TIRX."""

from .contracts import (
    TritonTVMContract,
    get_triton_tvm_contract,
    normalize_triton_tvm_contract,
    validate_cuda_minimal_contract,
    validate_cuda_pointwise_flat_contract,
    validate_norm_single_row_contract,
    validate_pointwise_flat_contract,
    validate_pointwise_minimal_contract,
    validate_reduction_minimal_contract,
    validate_triton_tvm_contract,
)
from .errors import (
    TritonTVMContractError,
    TritonTVMError,
    UnsupportedContractError,
    UnsupportedStreamError,
    UnsupportedTTIROpError,
    UnsupportedTargetPolicyError,
)
from .frontend import TTIRArtifact, lower_to_ttir
from .passes import ValidateTritonKernelTIR
from .runtime import TritonTVMArtifact, build_triton_tvm
from .translator import TritonTVMMeta, translate_ttir

__all__ = [
    "TTIRArtifact",
    "TritonTVMArtifact",
    "TritonTVMContract",
    "TritonTVMContractError",
    "TritonTVMError",
    "TritonTVMMeta",
    "UnsupportedContractError",
    "UnsupportedStreamError",
    "UnsupportedTTIROpError",
    "UnsupportedTargetPolicyError",
    "ValidateTritonKernelTIR",
    "build_triton_tvm",
    "get_triton_tvm_contract",
    "lower_to_ttir",
    "normalize_triton_tvm_contract",
    "translate_ttir",
    "validate_norm_single_row_contract",
    "validate_pointwise_flat_contract",
    "validate_pointwise_minimal_contract",
    "validate_reduction_minimal_contract",
    "validate_triton_tvm_contract",
]
