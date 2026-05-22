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
"""Errors raised by the Triton-to-TVM prototype."""


class TritonTVMError(RuntimeError):
    """Base class for Triton-to-TVM frontend errors."""


class UnsupportedTTIROpError(TritonTVMError):
    """Raised when the prototype sees TTIR outside its supported subset."""


class TritonTVMContractError(TritonTVMError):
    """Raised when translated TIR does not satisfy the selected contract."""


class UnsupportedContractError(TritonTVMError, ValueError):
    """Raised when no Triton TVM contract or alias matches a requested contract."""


class UnsupportedTargetPolicyError(TritonTVMError, ValueError):
    """Raised when no backend policy is registered for a target/contract pair."""
