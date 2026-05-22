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

### M4: Reduction Subset

目标：

- 支持简单 block reduction。
- 跑通 layernorm/rmsnorm-like kernel 的核心片段。
- 明确 reduction axis、init 和 mask semantics。

产物：

- reduction correctness tests
- one model-derived kernel case

### M5: Inductor Integration Prototype

目标：

- 提供 PyTorch/Inductor hook 原型。
- 对支持的 kernel 走 TVM，不支持的 kernel 明确 fallback 或跳过。
- 接入 cache 和 launcher。
- custom backend/hook 只做实验，不承诺稳定 ABI。

产物：

- experimental integration module
- compile/cache prototype

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
