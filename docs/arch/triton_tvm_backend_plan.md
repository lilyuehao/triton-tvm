# Triton to TVM Backend Plan

## 背景

目标不是让 TVM 调用 Triton 已经编译好的 kernel，也不是把 Triton 当作黑盒外部模块挂到 `external_mods`。目标是把 Triton language 作为 frontend，把 TVM 作为 backend/codegen：

```text
Triton language
  -> Triton frontend and semantic lowering
  -> Triton TTIR
  -> TVM TIR/TIRX
  -> user TVM IR passes
  -> TVM lowering and codegen
```

这样做的核心收益是：

- 复用 Triton 生态里的 kernel 表达方式，尤其是 TorchInductor 生成的 Triton kernel。
- 复用 TVM 侧已有 IR pass、schedule、analysis 和 codegen。
- 为后续非 Triton 原生 backend、实验 GPU codegen、定制硬件 codegen 留出统一入口。

本地约束：

- 使用 conda 环境 `tvm-0.24.0`。
- 本地 TVM 是 `/home/liyh/xdb/tvm`，版本为 `0.24.0`。
- 环境中有 `triton 3.7.0`。
- 本地 TVM 树已有 `tirx` 路径，以及早期 `T.call_kernel`/Triton 黑盒集成，但它不是本方案的最终形态。

## 非目标

第一阶段不追求完整 Triton 语义覆盖，也不追求完整 TorchInductor 无感替换。

明确不作为第一阶段目标的内容：

- 不让 Triton 自己生成 PTX/CUBIN 后由 TVM 调用。
- 不从 Triton Python AST 直接重写一套完整 frontend。
- 不优先支持 `tl.dot`、tensor descriptor、TMA、async copy、warp specialization、atomic、复杂 autotune。
- 不承诺对所有 TorchInductor kernel 兼容。
- 不一开始实现完整 out-of-tree Triton backend plugin。

## 总体方案

推荐以 TTIR 作为转译边界：

```text
@triton.jit function
  -> Triton ASTSource.make_ir
  -> optimized TTIR
  -> TTIRReader
  -> NormalizedTTIROpGraph
  -> Triton-flavored Kernel TIR
  -> NormalizeTritonKernelTIR
  -> backend-expected TIR/TIRX
  -> existing TVM pass pipeline
  -> tvm.compile
```

选择 TTIR 的理由：

- Triton 已经完成了 Python frontend、constexpr specialization、type inference、`tl.*` semantic lowering。
- TTIR 仍然相对 target-independent，比 TTGIR 更适合接入 TVM 自己的调度和 codegen。
- TTGIR 已经引入 TritonGPU layout、warp/CTA scheduling 和 backend-specific lowering，接入后 TVM 能发挥的空间会变小。

不建议从 Python AST 开始：

- 需要重做 Triton 的语义检查、类型系统、constexpr、broadcast、block tensor 规则。
- 和 TorchInductor 生成代码之间的兼容成本更高。

不建议第一阶段从 TTGIR 开始：

- TTGIR 绑定了 TritonGPU dialect 和具体 GPU lowering 策略。
- 对复用 TVM IR pass 不友好。

### TIR contract

复用已有 TVM pass 的关键不是“生成一个合法的 `PrimFunc`”，而是生成一个满足后续 pass pattern 和分析假设的 `PrimFunc`。TVM pass infra 支持用户组合自定义 IR-to-IR pipeline，因此 translator 输出必须有明确 contract：

```text
TTIR
  -> Triton-flavored Kernel TIR
  -> NormalizeTritonKernelTIR
  -> BackendExpectedTIRContract
  -> existing user passes
```

其中：

- `Triton-flavored Kernel TIR` 保留 Triton block tensor、lane axis、mask、load/store metadata 等信息，方便做语义保持的 normalization。
- `NormalizeTritonKernelTIR` 把 Triton 风格结构改写成目标 pass 能识别的 loop/block/buffer/thread-binding 形态。
- `BackendExpectedTIRContract` 是项目内需要显式记录和测试的契约，包括 loop nest、block scope、buffer flattening、launch axis、thread binding、dtype、mask lowering 和 metadata。

也就是说，translator 不应把“canonical TIR”当成唯一目标。不同后端或不同用户 pass 可以定义不同 contract；第一版可以选择一个最小 contract，但必须把它写清楚并用 TVMScript golden test 固化。

### TTIR reader boundary

TTIR textual MLIR parser 可以作为 MVP 的输入手段，但不应成为 translator 主体的长期接口。translator 内部不要直接消费 raw textual string，也不要把 regex/MLIR textual parsing 逻辑散落在 lowering 代码里。

建议固定为三层：

```text
TTIR textual / MLIR module
  -> TTIRReader
  -> NormalizedTTIROpGraph
  -> TVM/TIR builder
```

`NormalizedTTIROpGraph` 至少需要表达：

- op name and dialect
- operands and results
- result types
- attributes
- regions and blocks, if present
- source location, optional
- normalized metadata for load/store/cache/alignment

这样 Triton textual format、attribute spelling 或 Python binding API 变化时，主要修改 `TTIRReader`，不会污染 TVM builder 和后续 pass contract。

## IR 映射策略

第一版应生成 contract-driven TVM `tirx.PrimFunc` 或 `tir.PrimFunc`。合法性只是底线，真正目标是满足后续 TVM pass 的输入契约。为了兼顾语义保真和 pass 复用，可以先生成 Triton-flavored Kernel TIR，再通过 normalization pass 改写成后端期望的 TIR/TIRX。

建议输出形态：

```text
PrimFunc
  attrs:
    global_symbol
    target
    tir.noalias
    optional triton metadata
  params:
    handle or buffer args
  body:
    root block
    launch axes
    explicit lane/block tensor loops
    BufferLoad and BufferStore
    IfThenElse for masks
```

### Triton block tensor

Triton block tensor 是主要语义难点。MVP 先将 block tensor 显式展开为 TVM loop/lane：

```text
tl.arange(0, BLOCK)
  -> lane loop

tl.load(ptr + offsets, mask)
  -> for lane:
       if mask[lane]:
           value[lane] = BufferLoad(...)

tl.store(ptr + offsets, value, mask)
  -> for lane:
       if mask[lane]:
           BufferStore(...)
```

后续可以把 lane loop 改写成 vector lanes 或 thread binding，由 TVM pass 决定。

### Program id 和 launch grid

`tl.program_id(axis)` 第一层不直接绑定到 CUDA `blockIdx`，而是映射为 abstract launch axis：

```text
tl.program_id(0) -> launch_axis_0
tl.program_id(1) -> launch_axis_1
tl.program_id(2) -> launch_axis_2
```

随后由 `LegalizeLaunchAxis(target)` 决定具体 lowering：

```text
CUDA target:
  launch_axis_0 -> blockIdx.x
  launch_axis_1 -> blockIdx.y
  launch_axis_2 -> blockIdx.z

custom hardware target:
  launch_axis_* -> core id / tile id / cluster id / backend intrinsic
```

CUDA 验证路径中可以 lowering 成 TVM thread binding：

```text
for bx in T.thread_binding(0, grid_x, thread="blockIdx.x"):
  ...
```

lane/block tensor 维度可以先用普通 loop 或 abstract lane axis 表达，再由后续 pass 决定是否绑定到 `threadIdx.x`、vectorize 或 unroll。这样 frontend bridge 不会过早把 Triton 语义固定为 CUDA 线程模型。

### 指针和 buffer

Triton pointer 参数需要映射为 TVM buffer：

```text
*fp32 -> Buffer((dynamic_extent,), "float32")
```

第一阶段可以把 pointer 看成 flat 1D buffer。多维 shape、stride 和 contiguous/divisibility metadata 后续再补。

### Mask

Triton `mask` 需要保留 undefined semantics。Triton `tl.load` 的语义是：如果 `mask[idx]` 为 false 且 `other is None`，对应 lane 的结果是 undefined。因此 TVM lowering 不应默认补 0，也不应默认补任意确定值。

```text
tl.load(mask=m, other=o)
  -> Select(m, BufferLoad, cast(o))

tl.load(mask=m, other=None)
  -> value has poison/undef semantics
  -> only legal if proven not used when m is false
  -> otherwise UnsupportedTTIROpError
```

MVP 行为：

- load with mask and `other`: `Select(mask, load, other)`。
- load with mask and no `other`: 建立 def-use 检查；如果无法证明 masked-out lane 不会被使用，直接报 `UnsupportedTTIROpError`。
- store with mask: `IfThenElse(mask, BufferStore, no-op)`。

如果 TVM/TIRX 后续引入明确的 `undef` 或 poison 表达，可以再把这类 load 映射为 undef-aware IR。MVP 以拒绝不安全 case 为准。

### DType

先覆盖：

- `bool`
- signed/unsigned integer: `int1/int8/int16/int32/int64`, `uint8/uint16/uint32/uint64`
- floating: `float16`, `float32`, `bfloat16`

FP8、packed type、pointer-to-tensor type 后续扩展。

## MVP 支持范围

第一批 kernel：

- vector add
- elementwise chain
- masked load/store
- simple broadcasting within block tensor
- simple unary/binary math
- simple 1D and 2D grids

第一批 TTIR op 类别：

- function signature and attributes
- constants
- `tt.get_program_id`
- `tt.make_range` or equivalent arange op
- arithmetic: add/sub/mul/div/floordiv/rem
- compare
- cast
- splat/broadcast
- load
- store
- return

第一批 normalization/adapter pass 应随 M1/M2 一起设计，而不是等性能阶段再补：

```text
AnnotateTritonLaneAxis
InferContiguousAndDivisibility
LegalizeMaskLoadStore
CanonicalizeAddPtr
CanonicalizeSplatBroadcast
AttachTritonMetadata
LegalizeLaunchAxis
```

这些 pass 的目标不是做激进优化，而是避免 TTIR 中原本存在的 coalescing、contiguity、divisibility、alignment、cache policy 等信息在翻译成普通 loop/load/store 后丢失。Correctness-first 的 lane loop 展开可以保留，但必须给后续 TVM pass 留下足够可分析的结构和 metadata。

第二批再支持：

- reductions
- `tl.sum`, `tl.max`
- more complete `other` handling
- multi-dimensional block tensors
- static range and simple loops
- simple shared-memory-like allocation if TTIR exposes it before TTGIR

第三批再评估：

- `tl.dot`
- atomic ops
- tensor descriptor
- block pointer
- async copy
- TMA
- architecture-specific lowering

## 复用 TVM Pass

复用点位于 TTIR 转成 TVM IR 之后：

```text
Triton TTIR
  -> TVM TIR/TIRX
  -> custom TVM passes
  -> TVM codegen
```

不能直接在 TTIR 上复用 TVM pass，因为 TTIR 是 Triton/MLIR dialect，TVM pass 无法识别。

如果用户 pass 是：

- `tirx.PrimFunc` pass：优先支持，translator 第一版就应输出 `tirx.PrimFunc`。
- `tir.PrimFunc` pass：可以在 TIRX 和 TIR 之间加转换或直接生成对应 TIR。
- Relax pass：需要把 kernel 包装成 Relax `call_tir`。
- lower/codegen 前 pass：需要 translator 生成正确的 target attrs、thread binding、buffer 信息。

建议在 translator 中保留 minimal metadata：

```text
triton.kernel_name
triton.signature
triton.constexprs
triton.grid
triton.num_warps
triton.source_hash
triton.contiguity
triton.divisibility
triton.alignment
triton.cache_modifier
triton.eviction_policy
triton.volatile
```

metadata 只用于 debug 和二次优化，不应该成为 pass 匹配的唯一依据。

## TorchInductor 兼容路线

TorchInductor 的 GPU 主路径大致是：

```text
PyTorch program
  -> TorchDynamo / AOTAutograd
  -> TorchInductor loop-level IR and scheduler
  -> generated Triton code
  -> Triton compiler
  -> CUDA/HIP binary
```

本方案希望替换最后一段：

```text
Inductor generated Triton code
  -> Triton TTIR
  -> TVM TIR/TIRX
  -> TVM passes
  -> TVM codegen
```

兼容分三档：

1. Triton language 子集兼容
   - 能编译 TorchInductor 生成的某些 Triton kernel。
   - 第一阶段目标。

2. Inductor generated code 离线兼容
   - 从 Inductor debug/output code 中提取 Triton kernel。
   - 使用本 translator 离线编译并比对结果。
   - 第二阶段目标。

3. `torch.compile(backend="inductor")` 无感兼容
   - 需要接入 Inductor runtime/cache/autotune/launcher。
   - 难度高，且依赖 PyTorch 内部 API。
   - 后续阶段目标。

推荐路线：

```text
Phase 1:
  standalone @triton.jit -> TVM

Phase 2:
  Inductor pointwise TTIR coverage audit

Phase 3:
  offline Inductor Triton code -> TVM

Phase 4:
  Inductor compile hook or custom kernel backend

Phase 5:
  integrate into Inductor autotune candidate system
```

第一批 Inductor 目标 kernel：

- pointwise fusion
- simple broadcast
- layernorm/rmsnorm-like kernels after reduction support lands
- simple indexing kernels

不建议第一批目标：

- matmul templates
- attention kernels
- complex persistent kernels
- kernels relying heavily on TritonGPU layout assumptions

在接入 `torch.compile` 之前，需要明确 runtime/launcher 边界。PyTorch 侧用户自定义 Triton kernel 和 `torch.compile` 的组合不仅是 compiler 问题，还涉及 wrapper、autograd/fallback、cache、launcher 和 PyTorch subsystem 的可组合性。因此 M3.5 之前应先做 offline audit，不应直接承诺稳定 Inductor ABI。

## 代码组织建议

建议新增 Python 原型目录：

```text
python/tvm/contrib/triton_tvm/
  __init__.py
  frontend.py
  ttir.py
  op_graph.py
  translator.py
  contracts.py
  passes.py
  builder.py
  runtime.py
  testing.py
```

职责：

- `frontend.py`
  - 接收 `triton.runtime.jit.JITFunction`、signature、constexprs、grid。
  - 调用 Triton 生成 optimized TTIR。

- `ttir.py`
  - TTIR 文本或 MLIR module 的抽取、dump、normalization。
  - 提供 `TTIRReader`，把 textual MLIR 或 MLIR module 转成统一中间结构。
  - 初期可以以 TTIR textual MLIR 作为 reader 输入，降低对 Triton C++ Python binding 的依赖。

- `op_graph.py`
  - 定义 `NormalizedTTIROpGraph`。
  - 隔离 op/type/attr/region 的 normalized 表达。

- `translator.py`
  - `NormalizedTTIROpGraph` 到 Triton-flavored Kernel TIR 的转换核心。
  - 管理 SSA value map、type map、block tensor shape map。

- `contracts.py`
  - 定义 backend-expected TIR/TIRX contract。
  - 记录 loop/block/buffer/launch-axis/thread-binding 形态约束。

- `passes.py`
  - 实现 `NormalizeTritonKernelTIR`、`LegalizeLaunchAxis`、`LegalizeMaskLoadStore` 等 adapter pass。

- `builder.py`
  - 封装 TVM `tirx` IRBuilder 或直接构造 TVM nodes。

- `runtime.py`
  - 管理 build artifact、grid evaluation、argument ABI、kernel cache key、module loading 和 packed func calling convention。
  - 后续对接 Inductor launcher/cache。

- `testing.py`
  - 原生 Triton vs TVM 输出对比。
  - 提供随机输入、dtype、shape、mask 测试工具。

未来如果需要 C++ 实现，可以把稳定后的 translator 迁到：

```text
src/relax/frontend/
src/tirx/frontend/
src/contrib/triton_tvm/
```

第一阶段不建议直接 C++ 化，原因是 Triton TTIR 抽取和 TorchInductor 兼容都更适合 Python 快速迭代。

## API 草案

API 拆成三个层级，分别服务 IR debug、TVM pass 注入和 E2E runtime。不要让一个 `compile_triton(...)` 同时隐藏 TTIR 抽取、IR 翻译、pass pipeline、build、cache 和 launcher。

### IR/debug API

```python
import triton
import triton.language as tl
import tvm
from tvm.contrib.triton_tvm import lower_to_ttir, translate_ttir


@triton.jit
def add_kernel(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    vx = tl.load(x + offsets, mask=mask, other=0.0)
    vy = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, vx + vy, mask=mask)


ttir = lower_to_ttir(
    add_kernel,
    signature={
        "x": "*fp32",
        "y": "*fp32",
        "out": "*fp32",
        "n": "i64",
        "BLOCK": "constexpr",
    },
    constexprs={"BLOCK": 128},
)

irmod, meta = translate_ttir(
    ttir,
    grid=lambda n: ((n + 127) // 128,),
    target="cuda",
    emit="tirx",
)

print(ttir)
print(irmod.script())
```

### Build/pass API

```python
from tvm.contrib.triton_tvm import build_triton_tvm

artifact = build_triton_tvm(
    irmod,
    meta,
    passes=[
        NormalizeTritonKernelTIR(contract="pointwise_minimal"),
        my_custom_tvm_pass,
    ],
    target="cuda",
)
```

### Runtime API

```python
artifact.run(grid=(ceildiv(n, 128),), args=[x, y, out, n], stream=None)
```

`artifact` 至少需要携带：

- specialized signature and constexprs
- grid evaluator or concrete grid
- TVM IRModule hash and source hash
- target and contract id
- argument ABI
- buffer binding policy
- packed func name
- module loading handle
- cache key
- fallback reason, if compilation is unsupported

## 测试计划

### Unit tests

新增：

```text
tests/python/contrib/test_triton_tvm_frontend.py
tests/python/contrib/test_triton_tvm_ttir_reader.py
tests/python/contrib/test_triton_tvm_translate.py
tests/python/contrib/test_triton_tvm_inductor_subset.py
```

覆盖：

- dtype mapping
- signature mapping
- constexpr mapping
- arange mapping
- program id mapping
- mask mapping
- load/store mapping
- arithmetic mapping
- TTIRReader op/type/attr normalization
- TIR contract validation
- launch axis legalization

### Correctness tests

每个测试同时执行：

```text
native Triton result
TVM-translated result
NumPy or PyTorch reference
```

通过标准：

- `float32`: `rtol=1e-5`, `atol=1e-5`
- `float16/bfloat16`: 使用更宽松阈值
- integer/bool: exact match

### IR golden tests

对简单 kernel 保存 expected TVM Script，确保 translator 输出稳定。

另保存 TTIR golden 和 `NormalizedTTIROpGraph` golden。TTIR textual golden 用于发现 Triton 版本漂移；op graph golden 用于保证 reader 变化不会影响 translator contract。

### Coverage tests

对真实 TorchInductor pointwise TTIR dump 做 coverage audit：

```text
total kernels
unique TTIR ops
unique attrs
unique types
unsupported ops
unsupported attrs
unsupported type patterns
mask/load/store forms
indexing forms
```

Inductor audit 不要求全部跑通，但必须输出 capability matrix 和 unsupported report。

### Negative tests

遇到不支持的 TTIR op 时必须明确报错：

```text
UnsupportedTTIROpError: tt.dot is not supported yet
```

不要 silent fallback 到 Triton 原生 codegen，否则会掩盖 translator 覆盖率。

## 里程碑

### M0: TTIR 抽取

目标：

- 从 `@triton.jit` 生成 optimized TTIR。
- 保存 TTIR dump。
- 支持 signature 和 constexprs。
- pin `triton 3.7.0` 并记录版本。

产物：

- `lower_to_ttir(...)`
- vector add TTIR dump test
- TTIR textual golden

### M0.5: TTIR Reader

目标：

- textual MLIR 或 MLIR module 转成 `NormalizedTTIROpGraph`。
- 不生成 TVM IR。
- 做 op/type/attr coverage tests。

产物：

- `TTIRReader`
- `NormalizedTTIROpGraph`
- op graph golden tests

### M1: Vector Add IR

目标：

- TTIR op graph 转成 TVM `tirx.PrimFunc` 或 `tir.PrimFunc`。
- 输出 Triton-flavored Kernel TIR。
- 运行 `NormalizeTritonKernelTIR` 到一个明确的 backend contract。
- 输出 TVMScript golden。
- 不要求性能。

产物：

- `translate_ttir(...)`
- `NormalizeTritonKernelTIR`
- vector add TVMScript golden
- TIR contract validator

### M1.5: Runtime Smoke

目标：

- TVM build/compile。
- run + compare NumPy/Triton。
- 明确 ABI、grid、cache metadata。
- 证明用户 TVM pass 可以插入且不破坏语义。

产物：

- `build_triton_tvm(...)`
- `artifact.run(...)`
- vector add correctness test
- minimal cache key and ABI metadata test

### M2a: Standalone Pointwise Expansion

目标：

- 在 standalone Triton kernel 上扩展 pointwise translator，不接 TorchInductor。
- 支持多个 elementwise op chain。
- 支持多个 `tt.load`。
- 仍只支持单个 `tt.store` 和单个输出。
- 支持 `arith.addf/subf/mulf`、`arith.addi/subi/muli`。
- 支持必要的 cast/compare variants。
- 支持 mask load/store with `other`。
- 继续拒绝 mask without `other`，不在本阶段做 def-use 证明。
- 继续拒绝 multiple stores、multiple outputs、reduction、dot、atomic、non-contiguous 和 Inductor hook。

产物：

- `_pointwise_chain_kernel` 能 translate/build/run。
- vector add 旧测试不退化。
- odd-size pointwise chain correctness test。
- unsupported multiple store 明确报错。
- unsupported mask without `other` 继续报错。
- devlog 记录新增 op 覆盖与剩余缺口。

### M2.5: Standalone Pointwise Corpus

目标：

- 在 standalone Triton kernel 上打通 pointwise translator 的主要形态，不接
  TorchInductor runtime。
- 支持多个 `tt.load`、多个 `tt.store` 和多个输出。
- 支持一个 kernel 内多个 independent store expressions。
- 支持 flat contiguous pointer 模型下的常见 `tt.addptr` 和 lane index 表达式。
- 支持简单 scalar/splat/block tensor broadcast。
- 支持 pointwise `add/sub/mul`，`div/rem/floordiv` 作为 stretch goal。
- 新增 canonical `pointwise_flat` contract；保留 `cuda_pointwise_flat` 和 `cuda_minimal`
  作为 CUDA 兼容别名。
- 继续拒绝 reduction、dot、atomic、block pointer、non-contiguous pointer、高维 block
  tensor 和 Inductor hook。

产物：

- `_dual_store_kernel` 能 translate/build/run。
- 至少 3 到 5 个 standalone pointwise correctness tests：
  - vector add 旧用例；
  - M2a pointwise chain；
  - multiple-output dual-store；
  - simple scalar/broadcast pointwise。
- `pointwise_flat` / `cuda_pointwise_flat` contract validator：
  - 不使用 `external_mods`；
  - outer loop 绑定 `blockIdx.x`；
  - inner loop 绑定 `threadIdx.x`；
  - 允许多个 `BufferStore`；
  - 每个 store 都必须由 mask guard 保护。
- Negative tests 覆盖 masked load without `other`、`tt.dot`、reduction、atomic 和
  unsupported non-contiguous pointer pattern。
- M2.5 devlog checkpoint，记录 supported/unsupported TTIR op、type、attr、mask 和 indexing
  forms。

### M2.5 Hardening: Pointwise Contract Cleanup

目标：

- 把 Backend Decoupling / CUDA 解耦显式提前到 M2.5 hardening，作为 M3 Inductor audit
  前置条件，避免后续 coverage 和能力矩阵建立在 CUDA-only translator 结构上。
- 清理过时的 M1/M1.5 报错文案和 docstring，使 public API 文案描述 current prototype
  而不是旧 milestone。
- 补充 TVMScript golden、TTIR reader corpus 和 op graph snapshot，固定 M2.5 IR shape。
- 把 translator 内部从单 store 模板整理为 store list emission。
- 引入 canonical contract 名称 `pointwise_minimal` 和 `pointwise_flat`，并保留
  `cuda_minimal` / `cuda_pointwise_flat` 作为兼容 alias，避免后续 pass 匹配混用。
- 抽出 backend/launch policy 边界；M2.5 只实现 CUDA policy，其他 target 明确报
  unsupported target policy，不新增非 CUDA runtime。
- 修正 runtime extent policy：从 flat-contiguous mask/index pattern 识别 extent scalar，
  不再默认使用第一个 scalar 参数；M2.5 只接受单一 flat mask predicate，遇到多个
  extent、load/store mask 不一致或 compound mask 时明确报 unsupported。
- 为 M3 前的后端扩展补 metadata anchor：canonical contract、target kind、extent param、
  block size、indexing kind、launch policy id、translator version、contract version 和
  target policy version。
- 支持基于 `tvm.target.Target` 的 policy dispatch；字符串 target 只作为 Target 构造输入，
  不通过字符串 contains 判断 CUDA。
- cache key 使用 canonical contract、target policy id/version 和 target attrs；`cuda_*`
  alias 不产生独立 cache identity。
- 增加不依赖 CUDA runtime 的纯 translate/contract negative tests。

产物：

- M2.5 hardening devlog entry。
- `pointwise_minimal` / `pointwise_flat` 以及 `cuda_*` alias 的短 contract 说明。
- 稳定的 standalone pointwise regression suite。
- 静态 regression 覆盖 canonical contract、alias dispatch、unsupported target policy、
  alias cache-key canonicalization、TVM Target object dispatch、scalar-before-extent 参数顺序、
  mask/extent 保守拒绝、raw load/store attrs 保留和 target-neutral metadata。

完成标准：

- translator core 不再直接用 CUDA 字符串判断 target，CUDA 细节只存在于 CUDA policy。
- public contract 名称 target-neutral；旧 `cuda_*` 名称仅作为 alias。
- CUDA TVMScript golden 形态不退化。
- 非 CUDA target 通过 policy registry 失败，而不是硬编码 target 检查。
- extent param 只从明确的 flat mask/index pattern 推断。
- metadata/cache key 使用 canonical contract 和 target policy id/version。
- contract validator 尽量保持 target-neutral，CUDA thread tag 检查只位于 policy-specific
  validation boundary。

### M2.5 Perf Baseline: Performance Baseline / Regression Guard

目标：

- 定位为性能基线和回归保护，不作为性能优化目标。
- 在 M2.5 standalone pointwise corpus 上记录同机 runtime baseline。
- 默认不进入普通 correctness CI，避免 benchmark 噪声和耗时污染功能测试。
- 支持手动或夜间任务开启 benchmark。
- 支持写出 JSON baseline，并在后续同机运行中按 tolerance 对比 TVM runtime 吞吐。
- 原生 Triton 数据只作为同机参考，不作为 pass/fail criterion。

产物：

- opt-in perf test: `tests/python/contrib/test_triton_tvm_perf.py`
- baseline JSON 写出入口：`TRITON_TVM_PERF_WRITE_JSON`
- baseline JSON 对比入口：`TRITON_TVM_PERF_BASELINE_JSON`
- tolerance 配置：`TRITON_TVM_PERF_TOLERANCE`
- M2.5 perf devlog checkpoint。

### M3: Inductor Pointwise Audit

进入 M3 前置条件：

- M2.5 hardening 的 CUDA 解耦必须完成：canonical pointwise contract、backend launch
  policy boundary、extent inference policy、alias cache canonicalization、Target object dispatch
  和 metadata anchor 均有静态测试。
- M3 audit 只统计 Inductor TTIR coverage，不应重新依赖旧的硬编码 `cuda_*` contract
  作为 translator 内部主入口。

目标：

- 离线收集 20 到 50 个 TorchInductor pointwise TTIR dumps。
- 统计 TTIR op 覆盖率。
- 统计 unsupported op、attr、type 和 indexing pattern。
- 建立 capability matrix。
- 不要求全部跑通。

产物：

- Inductor TTIR corpus script
- coverage matrix
- unsupported report
- prioritized M3.5 gap list

### M3.5: Offline Inductor Pointwise E2E

目标：

- 从 TorchInductor debug generated Triton code 中提取 kernel。
- 离线走 TTIR -> TVM。
- 至少跑通 3 到 5 个真实 Inductor pointwise kernels。
- 覆盖 multiple inputs/outputs。
- 覆盖 odd sizes、dynamic n、部分 non-contiguous case。

产物：

- extraction doc
- offline Inductor kernel regression test
- capability matrix update

### M3.5 Hardening: M4 Entry Gate

目标：

- 在开启 M4 reduction 前收紧 M3.5 aggressive pointwise 覆盖的语义边界。
- 不新增 Inductor pointwise 覆盖，不实现 reduction lowering，不接 Inductor runtime。
- 将 pointwise op capability 从全局 allowlist 改成 per-contract allowlist。
- 将 `pointwise_indexed` 的 no-`other` masked load、indexed pointer 和 bool bitcast-store
  语义固化到 contract validator 和 negative tests。
- 保持 `NormalizeTritonKernelTIR` validation-only，作为 M4 前的显式选择；如果 M4 暴露
  mask/index/legalize 复用压力，再规划 rewrite pass。
- 让 `reduction_minimal` 保持 clean placeholder，明确报 not implemented，不进入 pointwise
  translator。

完成标准：

- `pointwise_minimal` / `pointwise_flat` 只接受 M2.5 flat pointwise op surface。
- `pointwise_indexed` 独占 M3.5 indexed/math/no-`other` 扩展。
- 手写 shape-only TIR 不能绕过 `pointwise_indexed` 的 load/guard/index 约束。
- 三组回归通过：static、standalone pointwise、Inductor promoted subset。

### M4: Reduction Subset

目标：

- 支持 `reduction_minimal` correctness-first row-wise reduction。
- 支持 Triton 3.7 textual `"tt.reduce"` region parsing，combiner 限定为
  `arith.addf` / `arith.addi` + `tt.reduce.return`。
- 支持 `tl.sum(axis=0)`、row sum、RMSNorm core、RMSNorm with weight，以及单行
  LayerNorm gamma/beta 形态。
- 单行完整 LN/RMS 已收口：runtime `n`、runtime/constexpr `eps`、mask、RMS
  weight、LayerNorm gamma/beta、LayerNorm 双 reduction 均有 CUDA correctness
  覆盖。
- 明确 M4 v0 不追求性能：每个 row 由 `threadIdx.x == 0` 串行完成 reduction 和
  epilogue，避免 shared/local accumulator race；shared/allreduce 或标准 TIR
  reduction block 留给后续优化。
- 明确 mask semantics：masked load 必须带 explicit zero `other`；安全的
  unmasked load 可用于 `n == BLOCK` 的 weight/gamma/beta 等参数；masked-out
  lanes 只有在 kernel 显式 `tl.where` 或 mask 后才不会参与后续 reduction。
- `grid` 必须是 static 1D row count；不支持 callable/multidim grid、persistent
  kernel、cross-block reduction、Welford、`tl.max`、atomic 或 Inductor runtime hook。

产物：

- reduction correctness tests
- row sum / RMS / RMSNorm / LayerNorm CUDA correctness cases
- `validate_reduction_minimal_contract`

M5 前保持：

- 单行完整 LN/RMS 作为 M5 前置能力维护，不再作为未完成项。
- 性能路径另起 hardening/optimization，不作为 M4 pass/fail criterion。

### Pre-M5 Debt Sprint: Integration Base Cleanup

目标：

- 不新增 Triton 语义、op coverage、reduction 能力或 Inductor hook。
- 清理 validation-only pass 命名，避免 public API 中出现 no-op
  `Normalize*` / `Legalize*` / `Lower*` pass。
- 收敛 contract matrix 到 canonical 名称：
  `pointwise_minimal`、`pointwise_flat`、`reduction_minimal`、
  `norm_single_row`。
- 将 legacy `cuda_minimal` / `cuda_pointwise_flat` 降级为 public
  compatibility alias，发 warning，且不进入 metadata、cache key、report
  bucket 或内部 dispatch。
- 将 M3.5 indexed/no-`other` pointwise 支持作为 `pointwise_flat` capability，
  不再保留独立 `pointwise_indexed` contract。
- 明确 cache/stream/runtime/fallback 边界，为 M5 hook、launcher 和 cache
  prototype 留出稳定接口。

产物：

- `ValidateTritonKernelTIR(contract=...)` 只负责 contract validation。
- `NormalizeTritonKernelTIR` 从 public API 移除；只有真实 rewrite pass 存在时
  才重新引入。
- `docs/arch/triton_tvm_contracts.md` 记录 contract 名称、TTIR op、index
  pattern、mask、launch、TIR shape、unsupported cases 和 tests。
- `TritonTVMMeta.cache_policy == "disabled"`，`disk_cache_enabled == False`；
  cache key 是稳定 in-memory prototype identity，不暗示 disk cache 已实现。
- `artifact.run(..., stream=None)` 表示 TVM current/default stream；
  非 `None` stream 当前报 `UnsupportedStreamError("unsupported_stream: ...")`。
- Inductor audit report status 带 `fallback_reason`。

完成标准：

- static / standalone CUDA / Inductor subset correctness 全绿。
- 主测试不再使用 `cuda_*` contract；仅 compatibility tests 覆盖 alias warning
  和 cache key canonicalization。
- metadata/cache key/report bucket 不出现 legacy contract 名称。

### Pre-M5 Debt Sprint: Reader, Reporting, Tests, API Freeze

目标：

- 将 textual TTIR parser 固定为受控临时实现，而不是散落在 translator 或
  builder 中。
- 建立 pointwise / reduction / norm 共用的 capability report schema，给 M5
  hook 的 fallback reason 使用。
- 把测试结构从 milestone demo 收敛成 default correctness、CUDA-gated
  correctness、opt-in perf、negative boundary、golden shape 这几类。
- 冻结 M5 前最小 public API。

产物：

- `ttir.normalize_ttir_input` 是唯一 raw TTIR text 到
  `NormalizedTTIROpGraph` 的入口。
- `translator.py` 不直接实例化 `TTIRReader`，只消费 normalized graph。
- Builder debt 登记：当前使用 controlled TVMScript source +
  `tvm.script.from_source`；M5 允许继续使用，M6 必须评估 direct node
  builder 或保留书面理由。
- `python/tvm/contrib/triton_tvm/reporting.py`
- `tests/python/contrib/test_triton_tvm_reporting.py`
- `docs/arch/triton_tvm_capability_matrix.md`
- `__all__` 只导出稳定 M5-entry API；legacy `cuda_*` wrappers 保持
  compatibility warning 但不在 `__all__` 中；`TTIRReader` 不再从 top-level
  package 导出。

完成标准：

- report schema 有 snapshot test。
- 所有 public unsupported error 都能映射到 stable bucket。
- M3/M3.5/M4 corpus 记录能进入同一 report schema。
- Markdown report 和 JSON report 的 summary/status/contract 信息一致。
- perf test 默认 skip；CUDA correctness 保持 skip 条件；每个 supported
  contract 保持 golden shape 或 CUDA correctness 覆盖。

### M5: Inductor Integration Prototype

目标：

- 提供 PyTorch/Inductor hook 原型。
- 对支持的 `pointwise_flat` Inductor kernel 走 TVM，不支持或编译失败的
  kernel 明确走 native Triton fallback，并记录 `fallback_reason`。
- 接入 process-local artifact cache 和 PyTorch Tensor / raw CUDA stream
  launcher。
- custom backend/hook 只做实验，不承诺稳定 ABI。

产物：

- `TritonTVMInductorConfig(fallback="native", target="cuda", passes=(),
  report_dir=None)`
- `TritonTVMInductorSession`：维护 report records、counters、process-local
  artifact cache。
- `make_triton_tvm_inductor_backend(config=None)`：返回可传给
  `torch.compile` 的 experimental backend；API 保持在
  `tvm.contrib.triton_tvm.inductor` 下，不进入 top-level `__all__`。
- hook 只在一次 Inductor compile 期间 patch
  `torch._inductor.async_compile.AsyncCompile.triton`。
- launcher 通过 DLPack 将 PyTorch CUDA Tensor 包装成 TVM Tensor；Inductor
  raw stream 通过 `tvm.cuda(device).set_raw_stream(...)` 交给 TVM，运行后恢复
  到 default stream。
- cache key 复用 `TritonTVMMeta.cache_key`；disk cache 继续保持 disabled。
- `tests/python/contrib/test_triton_tvm_inductor_hook.py`

完成标准：

- custom backend 对简单 Inductor pointwise add/mul 使用 TVM 路径并保持
  correctness。
- unsupported Inductor kernel 保持 PyTorch 输出 correctness，并有 native
  fallback record。
- 同一 session 重复编译同一 supported kernel 命中 process-local artifact
  cache。
- report 使用 `triton_tvm_capability_report` schema，记录 `cache_key`、
  `cache_hit`、`run_count`、`native_fallback_count` 和 stable status bucket。

### M5+ Roadmap: From Hook Prototype to Model E2E

路线原则：

- `.5` milestone 是 hardening 半代：只稳定已有能力，不新增大语义面。
- 每个关键大版本前设置 `Pre-Mx Debt`：先清 API、contract、report、cache、
  launcher、测试结构和文档债务，再扩 coverage。
- 模型级“完整编译”定义为：不走 opaque PyTorch/Inductor native fallback；
  TVM 负责 artifact/cache/launcher/report。允许 TVM artifact 内显式调用
  cuBLAS/cuDNN/cuDNN-frontend 等 extern，但必须可见、可统计、可测试。
- 推荐模型推进顺序：ViT -> YOLO -> Llama2-7B。ViT 更集中在
  matmul/norm/softmax/attention；YOLO 增加 conv/layout/decode/NMS；Llama2-7B
  对 attention、KV cache、动态 seq、显存和 kernel launch overhead 压力最大。

#### M5.5 Hardening: Inductor Hook Stabilization

Status: implemented as the mandatory M6 entry gate.

目标：

- 不新增 TTIR op 或模型 coverage。
- 稳定 M5 hook 的 lifecycle、report、cache、stream、fallback 和异常边界。
- 支持一个 graph 内多个 Inductor Triton kernel 被 hook 观察、部分 TVM
  替换、部分 native fallback，并生成 graph-level report。

产物：

- session-level graph report：`graph_id`、compile region、kernel order、
  TVM/native fallback summary、cache hit/miss、run count、fallback reason
  histogram。
- hook reentrancy/thread-safety guard：`AsyncCompile.triton` 只在一次
  Inductor compile 内被 patch；全局 hook 锁跨越 compile 生命周期；异常退出
  和多次 `torch.compile` 后都必须恢复原始 Triton path。
- raw stream handoff regression：TVM launcher 接收 Inductor raw CUDA stream，
  调用后恢复 TVM CUDA stream 到 `0`，且不分配输出、不同步。
- mixed-graph regression：同一个 `torch.compile` graph 内至少观察到两个
  Inductor Triton kernel，其中 supported pointwise 走 TVM，unsupported
  reduction 走 explicit native fallback。
- artifact cache lifecycle 文档：cache 只属于
  `TritonTVMInductorSession.artifacts`，只在当前 Python 进程和当前 session
  内有效；session 丢弃即失效；不写 disk cache；命中和未命中同时进入
  kernel-level 和 graph-level report。

完成标准：

- M5 hook tests、M3/M3.5/M4 corpus、standalone CUDA correctness 全绿。
- graph-level report 能说明每个 kernel 的 TVM/fallback 路径。
- runtime failure 不会 silent fallback；compile-time unsupported 可以 native
  fallback，但必须有稳定 bucket。
- 没有 graph-level report、reentrancy/thread-safety guard、raw stream handoff
  regression、cache lifecycle 文档时，禁止进入 M6。

#### Pre-M6 Debt: Integration Contract Cleanup

Reality check after M5.5:

- 与最初预估一致：Pre-M6 仍然应该是 integration contract cleanup，不新增
  TTIR op、模型 coverage 或调度优化。
- 已由 M5.5 提前清掉：experimental hook API 仍留在
  `tvm.contrib.triton_tvm.inductor`，没有进入 top-level `__all__`；graph-level
  report 已进入 capability matrix；cache lifecycle 已明确为 process-local、
  session-local、in-memory only；hook lifecycle 已有 reentrancy guard 和异常恢复
  回归。
- 与最初预估不同：graph fields 已经存在，但还没有正式 snapshot；JSON report
  已包含 graph 信息，Markdown report 仍主要是 kernel-level 视图；native fallback
  目前只记录 compile-time fallback，不记录 native fallback `.run(...)` 次数；
  PyTorch/Inductor private API 漂移风险比预估更突出，必须在 M6 前 fail fast。

目标：

- 固定 M5/M5.5 experimental Inductor API 边界，避免 M6 graph integration 扩散到
  top-level public API。
- 将 `m5_inductor_hook` 的 kernel-level 与 graph-level report 字段冻结成可测
  schema，并明确 JSON/Markdown 的一致性边界。
- 固定 PyTorch/Inductor/Triton version guard、private API signature guard 和
  unsupported message。
- 固定 graph identity、cache identity、runtime counter 三者的边界，避免 M6
  进入模型 corpus 后 report 语义变形。

产物：

- `TritonTVMInductorSession.report()` schema snapshot：覆盖 `summary`、
  `graph_summary`、`graphs`、`kernels[*].graph_id`、
  `kernels[*].graph_kernel_index`、`cache_key`、`cache_hit`、
  `native_fallback_count` 和 `run_count`。
- graph/kernels 两级 report consistency tests：
  - JSON 中每个 graph 的 `kernel_record_indices` 必须指向存在的 kernel record；
  - kernel record 的 `graph_id` / `graph_kernel_index` 必须能反查 graph；
  - Markdown 必须展示 graph summary，或明确声明 Markdown 是 kernel summary、
    graph details 以 JSON 为准。
- M5 hook API 文档 freeze：experimental、not stable ABI、only
  `fallback="native"`、supported `pointwise_flat` replacement、unsupported native
  fallback policy。
- private API guard：在 hook 入口检查 `torch.compile` / TorchInductor /
  Triton 版本和 `AsyncCompile.triton` callable signature；不支持时 fail fast，
  report bucket 使用 `input_error` 或新的稳定 version bucket，不允许进入半
  patch 状态。
- cache key / graph identity policy：`TritonTVMMeta.cache_key` 只标识单 kernel
  artifact；`graph_id`、compile region、run count 不进入 artifact cache key；
  graph-level cache hits/misses 只聚合 kernel-level cache decisions。
- runtime counter policy：TVM-backed kernel 的 `.run(...)` 已记录；native
  fallback kernel 运行态计数要么通过 wrapper 记录，要么在 report schema 中明确
  标为 compile-time-only，不能让 `run_count` 被误读为全 graph launch count。
- report flush policy：M5.5 当前在 compile/run 路径 eager write report；Pre-M6
  必须决定 M6 是否保留 eager write、改为 explicit flush，或只在 `report_dir`
  存在时写，以免模型 corpus 的 runtime hot path 被 report I/O 污染。

完成标准：

- `m5_inductor_hook` report schema snapshot 进入测试，且 capability matrix 与
  snapshot 字段一致。
- 不支持的 PyTorch/Inductor/Triton 版本或 private API signature 在 hook 入口
  fail fast，并给出稳定 bucket/message。
- JSON/Markdown report 对 graph/kernel summary 的关系有测试或明确文档。
- cache key 不包含 graph/run/session transient 字段，并有回归测试保护。
- M5.5 全部 gate 保持绿色后，才允许进入 M6。

#### M6: Graph-Level Inductor Integration

Entry gate:

- M5.5 必须保持绿色。尤其是 graph-level report、hook reentrancy/thread
  guard、raw stream handoff regression、mixed TVM/native graph regression 和
  cache lifecycle 文档。
- M6 可以扩展 graph integration 行为，但不能先扩大 TTIR op/model coverage
  来绕过 M5.5 的 hook 生命周期问题。

目标：

- 从单 kernel hook 提升到 graph-level integration prototype。
- 一个 `torch.compile` graph 内多个 supported Triton kernel 走 TVM artifact；
  unsupported kernel 继续显式 native fallback。
- 建立模型 corpus audit 入口，为后续 ViT/YOLO/Llama coverage 排序。

产物：

- graph session object：inputs、outputs、kernel list、artifact list、fallback
  list、runtime launch order。
- `torch.compile` graph smoke tests：multi-pointwise、pointwise+reduction、
  pointwise+unsupported。
- corpus collector：对 ViT、YOLO、Llama2 小 shape 运行 Inductor，收集
  wrapper source、TTIR、report、fallback buckets。

完成标准：

- multi-kernel graph correctness 通过。
- graph report 能定位每个 model blocker 的 TTIR op、contract、dtype、shape。
- native fallback 仍允许，但不能缺 report。

#### M6.5 Hardening: Model Corpus Audit

目标：

- 不扩语义，专注把真实模型 corpus 可重复、可比较、可排序。
- 建立 ViT/YOLO/Llama2 的最小 audit fixtures。

产物：

- ViT tiny/small、YOLO nano/tiny、Llama2-like tiny transformer 的
  Inductor corpus reports。
- blocker ranking：按出现次数、模型影响、实现风险排序。
- report diff 工具：比较两个 commit 的 translated/fallback bucket 变化。

完成标准：

- 每个模型族至少有一个固定 shape audit fixture。
- coverage regression 能在 CI 或 opt-in CUDA job 中复现。

#### Pre-M7 Debt: TTIR Reader and Builder Cleanup

目标：

- 解决继续扩 pointwise/broadcast/indexing 前的 reader/builder 债务。
- 评估 direct node builder；若继续使用 TVMScript source builder，必须写出保留理由。

产物：

- `NormalizedTTIROpGraph` snapshot 分类：pointwise、broadcast、view/index、
  reduction、matmul、attention。
- TVMScript source builder 风险登记和 direct builder 评估结论。
- unsupported indexing/broadcast bucket 细分，避免都落到 generic
  `unsupported_ttir_op`。

完成标准：

- 新增 pointwise/indexing coverage 前，reader snapshot 与 error bucket 都先稳定。

#### M7: Pointwise, Broadcast, View, and Indexing Completion

目标：

- 覆盖 ViT/YOLO/Llama 中高频 pointwise/broadcast/view/indexing kernel。
- 支持 fp32/fp16/bf16/int/bool 的常见 cast、compare、select、activation 和
  multi-output pointwise。

产物：

- broadcast row/col/general rank flattening policy。
- contiguous/strided/index expression classifier。
- 常见 activation：GELU、SiLU、sigmoid、tanh、relu、clamp、where。
- bool mask、dtype cast、bitcast-store、multi-output pointwise correctness。

完成标准：

- M6 corpus 中 pointwise/broadcast/view 类 blocker 基本清零。
- ViT/YOLO/Llama tiny corpus 的 pointwise kernel 不走 native fallback。

#### M7.5 Hardening: Pointwise Model Hardening

目标：

- 不新增大 op；稳定 M7 coverage 在真实模型中的 shape/dtype/layout 表现。
- 补负例和 perf guard，避免 unsafe no-`other` load、mask、stride 误判。

产物：

- model corpus pointwise golden reports。
- dtype-specific correctness：fp32/fp16/bf16/int/bool。
- pointwise perf smoke：不要求超过 Triton，但禁止数量级退化。

完成标准：

- M7 supported kernel report 无 legacy contract、无 silent fallback。

#### Pre-M8 Debt: Reduction and Norm Contract Cleanup

目标：

- 在扩 reduction/norm/softmax 前，把 `reduction_minimal` 和
  `norm_single_row` 的边界拆清。
- 明确 accumulator dtype、epsilon、mask、axis、row/column layout policy。

产物：

- reduction family contracts：row reduction、column reduction、softmax、
  norm、masked softmax。
- numerical policy：fp32 accumulate、fp16/bf16 output、epsilon handling。
- CUDA correctness fixture generator。

完成标准：

- reduction/norm unsupported case 都有稳定 bucket。
- M4 correctness-first serial path 与后续 parallel path 的 metadata 可区分。

#### M8: Reductions, Norms, and Softmax

目标：

- 支持 ViT/Llama attention 和 transformer block 所需的 reduction/norm/softmax。
- 支持 YOLO 中常见 reduction-like decode/postprocess 前置算子。

产物：

- row/column sum、mean、variance、max/min 选择性支持。
- LayerNorm、RMSNorm、softmax、masked softmax、causal mask softmax。
- simple parallel reduction path 或明确保留 correctness-first path 的性能风险。

完成标准：

- ViT tiny transformer block 中 norm/softmax 不 native fallback。
- Llama2-like tiny block 中 RMSNorm/causal softmax correctness 通过。

#### M8.5 Hardening: Numerics and Dynamic Shape

目标：

- 固定 transformer numerics 和动态 shape policy。
- 覆盖 batch/seq/image size 变化时 cache specialization 与 runtime scalar ABI。

产物：

- fp16/bf16 tolerance matrix。
- dynamic batch/seq/image shape smoke tests。
- cache specialization report：shape/constexpr/runtime scalar 哪些进入 key。

完成标准：

- ViT/LLM tiny fixtures 在至少两组 shape 下 correctness 通过。

#### Pre-M9 Debt: Matmul and Extern Policy Cleanup

目标：

- 在实现 `tt.dot`/GEMM 前明确 TVM native schedule、TensorCore path、extern
  GEMM 的边界。
- 明确 autotune/cache/report 怎么表达 matmul candidate。

产物：

- matmul contract：plain GEMM、batched GEMM、GEMM+epilogue、QKV projection。
- extern policy：允许 cuBLAS/cuDNN extern，但必须是 TVM artifact 内显式
  call，不算 native fallback。
- TensorCore dtype/layout policy。

完成标准：

- matmul fallback、extern、native schedule 三类路径在 report 中可区分。

#### M9: Matmul, GEMM, and Epilogue

目标：

- 支持 ViT/LLM 的 linear、QKV projection、MLP GEMM。
- 支持 YOLO head 中常见 1x1 conv lowering 到 GEMM 的路径，或显式延后到
  M11 conv stack。

产物：

- `tt.dot` reader/translator 支持，或 TVM extern GEMM lowering。
- GEMM + bias/add/activation epilogue。
- TensorCore fp16/bf16 correctness 和 baseline perf。

完成标准：

- ViT tiny/small transformer block 的 matmul/norm/softmax/pointwise 可编译。
- Llama2-like tiny block 的 MLP/QKV/O projection correctness 通过。

#### M9.5 Hardening: Matmul Performance and Autotune

目标：

- 稳定 TensorCore path、shape specialization、autotune cache 和 perf guard。
- 不扩 attention/conv 语义。

产物：

- matmul baseline：ViT shape、Llama MLP/QKV shape、YOLO 1x1-like shape。
- autotune record：candidate id、selected schedule/extern、cache key。
- launch overhead report。

完成标准：

- matmul 不出现数量级性能退化；正确性和 cache hit 可重复。

#### Pre-M10 Debt: Attention ABI Cleanup

目标：

- 在 attention E2E 前固定 mask、RoPE、KV cache、causal flag、sequence length
  的 ABI 和 report 表达。

产物：

- attention contracts：ViT full attention、Llama causal prefill、Llama decode。
- KV cache layout policy：contiguous/cache paged 是否支持，unsupported 如何 bucket。
- RoPE policy：pointwise fused 或单独 kernel。

完成标准：

- attention 相关 fallback reason 可直接映射到 mask/RoPE/KV/cache/layout。

#### M10: Attention Stack

目标：

- 支持 ViT attention 和 Llama causal attention prefill 的基本编译。
- 初版 decode path 可以 correctness-first，但必须显式报告性能风险。

产物：

- QK^T、masked softmax、AV、output projection 串联。
- RoPE pointwise support。
- KV cache update/read basic path。

完成标准：

- ViT tiny attention block fallback-free。
- Llama2-like tiny prefill attention correctness 通过；decode path 有明确
  fallback/perf report。

#### M10.5 Hardening: LLM Runtime Hardening

目标：

- 稳定 LLM prefill/decode runtime、KV cache、显存和 launch overhead。

产物：

- long-seq smoke：短 seq 与中等 seq 两档。
- KV cache correctness：prefill 后 decode 多 token。
- memory report：artifact memory、workspace、cache buffer。

完成标准：

- Llama2-like tiny end-to-end block prefill+decode correctness 通过。

#### Pre-M11 Debt: Vision Operator Policy Cleanup

目标：

- 在 YOLO E2E 前明确 conv/pool/resize/concat/NMS 的实现策略。
- 不允许隐式 PyTorch fallback；TVM extern 和 native TVM schedule 必须可见。

产物：

- vision op policy：conv2d、depthwise/grouped conv、pool、resize、concat、
  slice、decode、NMS。
- layout policy：NCHW/NHWC、channels-last、stride/padding/dilation。
- NMS policy：可先 TVM extern，后续再 native TIR/TIRX。

完成标准：

- YOLO corpus blocker 能分到 conv/layout/postprocess 三类，不再混在 generic bucket。

#### M11: Vision Convolution and Operator Stack

目标：

- 支持 YOLO backbone/head 所需基础视觉 op。
- ViT patch embedding conv 或 unfold/linear 路径要可编译。

产物：

- conv2d、1x1 conv、depthwise/grouped conv、pool、resize、concat/slice。
- SiLU/GELU/pointwise fusion 与 conv/GEMM epilogue 的 report。
- YOLO decode 前处理 kernel 支持。

完成标准：

- YOLO nano/tiny backbone+head 主体可编译；NMS 可通过显式 TVM extern。
- ViT patch embedding + transformer encoder tiny graph 可编译。

#### M11.5 Hardening: Vision Model Hardening

目标：

- 稳定 YOLO/ViT 真实 graph 的 shape、dtype、layout 和 fallback-free report。

产物：

- YOLO nano/tiny fixed-shape correctness。
- ViT small fixed-shape correctness。
- vision perf baseline 和 report diff。

完成标准：

- YOLO/ViT tiny/small fixtures 无 native fallback；允许显式 TVM extern。

#### M12: ViT E2E

目标：

- 完整编译 ViT-B/16 或 ViT-small inference。

产物：

- patch embedding、positional add、multi-head attention、MLP、LayerNorm、
  classifier head 全链路 TVM artifact。
- fixed-shape 和至少一个 dynamic batch smoke。
- accuracy/correctness 对齐 eager PyTorch 或 Inductor reference。

完成标准：

- ViT selected model 无 native fallback。
- report 完整列出所有 TVM kernels/externs/cache hits/runtime runs。

#### M13: YOLO E2E

目标：

- 完整编译 YOLOv5n/YOLOv8n 级别基础视觉模型 inference。

产物：

- backbone、neck、head、decode 全链路 TVM artifact。
- NMS 初版允许显式 TVM extern；native TVM NMS 可作为后续优化。
- fixed input size correctness 和 perf baseline。

完成标准：

- YOLO selected model 无 native fallback；NMS 若 extern 必须在 report 中明确。
- 输出 boxes/scores/classes 与 eager/Inductor reference 在 tolerance 内一致。

#### M14: Llama2-7B E2E

目标：

- 编译 Llama2-7B fp16/bf16 inference 的 prefill + decode 基本路径。

产物：

- embedding、RMSNorm、QKV/O projection、RoPE、causal attention、MLP、lm_head。
- KV cache allocation/update/read path。
- prefill/decode correctness、显存 report、基础 throughput baseline。

完成标准：

- Llama2-7B selected config 无 native fallback；允许显式 TVM extern GEMM。
- 单 batch prompt prefill + 多 token decode correctness 通过。
- report 能区分 TensorCore/extern/native TVM kernels、cache hits、launch 次数和
  memory footprint。

## 风险和对策

### Triton TTIR textual format 不稳定

风险：

- Triton 版本升级会改变 TTIR op 名、attribute、location、metadata 或 textual format。
- TorchInductor 生成的 kernel 会放大这些小差异。

对策：

- 第一阶段 pin `triton 3.7.0`。
- 增加 TTIR golden tests。
- 将 TTIR parsing 层隔离在 `ttir.py`。
- translator 内部只消费 `NormalizedTTIROpGraph`，不直接消费 raw textual string。
- textual parser 只作为 `TTIRReader` 的一种实现。

### Python binding 不够好遍历 MLIR module

风险：

- Triton C++ Python binding 可能不提供完整 op walk API。

对策：

- MVP 先使用 textual MLIR parser。
- 后续再评估直接绑定 MLIR op API。

### TVM IR 形态和用户 pass 不匹配

风险：

- translator 生成的 loop/block 结构不符合已有 pass 的 pattern。

对策：

- 先收集用户 pass 的输入期望。
- 提前定义 `BackendExpectedTIRContract`。
- 增加 `NormalizeTritonKernelTIR` 和 contract validator。
- 为 translator 提供 shape policy 和 binding policy。

### TorchInductor 内部 API 变化

风险：

- Inductor 不是稳定插件 ABI。

对策：

- 先做 offline generated Triton 兼容。
- 后做 hook。
- 维持版本矩阵，不承诺跨所有 PyTorch 版本。

### 性能不达预期

风险：

- naive block tensor loop 化后性能弱于原生 Triton。
- 如果输出 IR 过于普通，TVM 后续 pass 未必能恢复 Triton 原本隐含的 coalescing、contiguity、divisibility 和 vectorization 信息。

对策：

- 第一阶段以 correctness 和 pass 复用为目标。
- M1/M2 就引入 `AnnotateTritonLaneAxis`、`InferContiguousAndDivisibility`、`CanonicalizeAddPtr`、`CanonicalizeSplatBroadcast` 等 adapter pass。
- 保存 alignment、cache modifier、eviction policy、volatile 等 load/store metadata。
- 后续让 TVM pass 做 vectorize/thread binding/unroll/shared memory。
- 为常见 pattern 加 TTIR-aware canonicalization。

### Runtime 和 launcher 边界不清

风险：

- 只返回 `IRModule` 或 `tvm.compile` 产物不足以支撑 E2E kernel 运行。
- grid evaluation、constexpr specialization、argument ABI、buffer binding、device stream、cache key、module loading、packed func calling convention 和 fallback 都可能在接 Inductor 时暴露问题。

对策：

- API 拆分为 `lower_to_ttir`、`translate_ttir`、`build_triton_tvm` 和 `artifact.run`。
- M1.5 就建立 runtime smoke test。
- build artifact 必须携带 ABI/grid/cache/fallback metadata。

## 需要尽早确认的问题

1. 用户已有 TVM pass 是基于 `tirx`、`tir` 还是 Relax？
2. 这些 pass 期望的 loop/thread/buffer 形态是什么？
3. 第一批目标 TorchInductor kernel 是 pointwise、reduction、还是 matmul epilogue？
4. 是否要求第一阶段直接接入 `torch.compile`，还是可以先接受 offline Inductor kernel？
5. 目标 codegen 是 CUDA 为主，还是还要覆盖 CPU/LLVM、ROCm、其他 backend？

## 推荐第一步

先实现 M0、M0.5、M1 和 M1.5 的最小闭环：

```text
@triton.jit vector add
  -> optimized TTIR
  -> NormalizedTTIROpGraph
  -> Triton-flavored Kernel TIR
  -> BackendExpectedTIRContract
  -> user pass placeholder
  -> tvm.compile cuda
  -> artifact.run correctness test
```

这一步的成功标准不是“vector add 能跑”这么宽泛，而是同时证明：

- Triton frontend 能稳定产出 TTIR。
- TTIR 能通过 `TTIRReader` 转成稳定的 `NormalizedTTIROpGraph`。
- op graph 能转成满足后端 pass contract 的 TIR/TIRX。
- 用户 TVM pass 能插入并不破坏语义。
- TVM codegen/runtime 能以明确 ABI 启动 kernel。

这个闭环成立后，TorchInductor 只需要成为 Triton source/TTIR 的上游来源；是否接入 `torch.compile` runtime 可以作为后续独立问题推进。

## 参考资料

- Apache TVM PR 17395, TIR and Triton integration: https://github.com/apache/tvm/pull/17395
- Apache TVM PR 17434, source kernel integration via `T.call_kernel`: https://github.com/apache/tvm/pull/17434
- PyTorch `torch.compile` documentation: https://docs.pytorch.org/docs/2.9/generated/torch.compile.html
- PyTorch compiler overview: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html
- PyTorch 2.0 overview, TorchInductor and Triton codegen: https://docs.pytorch.org/get-started/pytorch-2.0/
- Apache TVM pass infrastructure: https://tvm.apache.org/docs/arch/pass_infra.html
- Triton `tl.load` API: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- PyTorch Triton kernel compilation stages: https://pytorch.org/blog/triton-kernel-compilation-stages/
- PyTorch user-defined Triton kernels with `torch.compile`: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- Triton Shared middle layer: https://github.com/microsoft/triton-shared
