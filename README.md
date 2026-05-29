<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Triton-TVM Alpha
================

This repository is a public alpha snapshot of an experimental Triton-to-TVM
integration built as a fork of Apache TVM. The active package lives in
[`tvm.contrib.triton_tvm`](python/tvm/contrib/triton_tvm/README.md).

Alpha entry points:

- [Alpha scope](ALPHA_SCOPE.md)
- [Alpha release notes](ALPHA_RELEASE.md)
- [Architecture overview](docs/arch/triton_tvm/README.md)
- [Package README](python/tvm/contrib/triton_tvm/README.md)

The current route carries Triton/TorchInductor operator evidence through:

```text
SourceRecord
  -> AtomicDAGRecord
  -> ContractValidationResult
  -> SemanticRegionRecord
  -> OperatorManifestRecord
  -> CapabilityRegistry
  -> OperatorExecutionRequest
  -> TIRRegionArtifact
  -> TVM packed-function execution
```

The alpha supports a fixed-shape ViT correctness route with 14 top-level
semantic operators over 16 source/lowering route records. The difference is
intentional: the decomposed attention region is one top-level semantic operator
with three internal source routes. This release makes no production performance
claim and does not claim general Triton kernel import.

Apache TVM Upstream
-------------------

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Apache TVM: Open Machine Learning Compiler Framework

[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

Apache TVM is an open machine learning compilation framework,
following the following principles:

- Python-first development that enables quick customization of machine learning compiler pipelines.
- Universal deployment to bring models into minimum deployable modules.

License
-------
TVM is licensed under the [Apache-2.0](LICENSE) license.

Getting Started
---------------
Check out the [TVM Documentation](https://tvm.apache.org/docs/) site for installation instructions, tutorials, examples, and more.
The [Getting Started with TVM](https://tvm.apache.org/docs/get_started/overview.html) tutorial is a great
place to start.

Contribute to TVM
-----------------
TVM adopts the Apache committer model. We aim to create an open-source project maintained and owned by the community.
Check out the [Contributor Guide](https://tvm.apache.org/docs/contribute/).

History and Acknowledgement
---------------------------
TVM started as a research project for deep learning compilation.
The first version of the project benefited a lot from the following projects:

- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
 originates from Halide. We also learned and adapted some parts of the lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.

Since then, the project has gone through several rounds of redesigns.
The current design is also drastically different from the initial design, following the
development trend of the ML compiler community.

The most recent version focuses on a cross-level design with TensorIR as the tensor-level representation
and Relax as the graph-level representation and Python-first transformations.
The project's current design goal is to make the ML compiler accessible by enabling most
transformations to be customizable in Python and bringing a cross-level representation that can jointly
optimize computational graphs, tensor programs, and libraries. The project is also a foundation
infra for building Python-first vertical compilers for domains, such as LLMs.
