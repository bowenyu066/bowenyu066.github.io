---
title: Gluon in Triton
date: 2025-6-30 10:08:04 +0800
excerpt: "Notes on the experimental feature, gluon, in Triton."
---

## Layout Utilities

`get_mma_instr_shape(shape: Tuple[int, int], element_ty: tl.dtype) -> Tuple[int, int, int]`

- 输入欲求得矩阵乘法结果的最终矩阵大小 `shape` 和元素类型 `element_ty`
- 输出三元组 `(M, N, K)`，代表单条指令处理的小矩阵乘法是 M x K 和 K x N 的矩阵乘法，处理最终结果的 `(M, N)` 一块
- 如果 `shape[0] >= 128`，则 `M = 128`，否则 `M = 64`；如果 `shape[1] >= 256`，则 `N = 256`，否则 `N = shape[1]`。`K = 256 // element_ty.bitwidth`，每次只处理 256 bit 宽度的矩阵乘法
- 不清楚这里的 128、256、64 是如何来的

`get_tmem_32x32b_reg_layout(instr_shape: Tuple[int, int, int], shape: Tuple[int, int], num_warps: int) -> gl.BlockedLayout`

- `instr_shape = M, N, K` 由 `get_mma_instr_shape` 得到
- `shape` 代表 2D tensor 的 shape
- `num_warps` 代表每个 cta block 里的 warp 数目，必须是 4 或 8
- 它返回一个 Blocked Layout。里面细节还蛮复杂的，但是莫得注释，搞不清楚。

`get_nvmma_layout(shape: Tuple[int, ...], element_ty: tl.dtype, order: List[int, ...]=[1, 0], fp4_padded=False)`

- 大致看了下 `NVMMASharedEncoding`，是一些特殊化的 swizzled layouts，用 `swizzle_byte_width` 128、64、32、0 标记，对应的 `vec` 等于 1 或者 2 用 `fp4_padded` 标记。具体实现就没看了

`get_mma_reg_layout(shape, num_warps, dtype=gl.float32)`

- 把 `get_mma_instr_shape` 和 `get_tmem_32x32b_reg_layout` 结合起来

## Data Abstraction

`Channel(T, alloc_fn)`

- 函数里定义了三个类（都用了 `@aggregate` 装饰器，虽然不清楚是干嘛的）：`ChannelType`，`Producer`，`Consumer`
  - `@aggregate` 用于把 python class 转换为 Triton jit-compiled 过程中类似于 C++ class/struct 的东西
- `T` 代表 `memory descriptor` 的类型，比如 `gl.shared_memory_descriptor` 或者 `gl.nvidia.hopper.tma.tensor_descriptor`，虽然不太清楚这是什么
- 两个概念：producer 和 consumer，producer 负责写入数据，consumer 负责读取数据
- `mem: T` 代表内存 block 的环形队列 (ring buffer)
- `ready_bars, empty_bars: gl.shared_memory_descriptor` 是两个 barriers；barrier 类似于 semaphore（上锁作用），用来确保环形队列中，只有 consumer 阅读完数据后 producer 才能覆写那片数据，只有 producer 写入数据后 consumer 才能读取那片数据。`ready_bars` 用来指代 consumer 是否可以读取数据，`empty_bars` 用来指代 producer 是否可以写入数据。但是它具体如何运转的不太清楚，因为几个 AI 给了完全相反的说法，我研究了一下午（花了好长时间）也没搞清楚，也没有找到相关的文档，所以这里只能先放过去了
- `num_buffers: gl.constexpr` 代表环形队列的 buffer 数目，用于 double/triple buffering 及其他有多个 buffer 的场景；`gl.constexpr` 代表在编译期确定的常量
- `num_consumers: gl.constexpr` 代表 consumer 的数目，用于多个 consumer 同时存在的情形

`LoadContext`, `PipelinedLoadProducer`, `MMAContext`, `MMAProducer`

## Gluon Attention

`AttentionConfig`

据说使用 `config` 是项目里一个通用的惯例。它的作用是定义一些配置参数，供后续的 attention kernel 使用。这里的 **kernel** 是指 GPU kernel，也就是一个编译好的、可以在 GPU 上并行执行的函数。

## Entry Point

主函数 entry point 是 `attention_forward(q, k, v, causal, sm_scale)`，里面调用 `_gluon_attn` 这个 JITFunction。

```python
_gluon_attn[grid](
    sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
    desc_q, desc_k, desc_v, desc_o,  #
    HEAD_DIM_K, BLOCK_M, BLOCK_N,  #
    stage, torch_dtype_to_triton(q.dtype),  #
    num_warps=4)
```

传入的不是裸 tensor，而是所谓的 tensor descriptor，它们通过 `make_tensor_desc` 函数创建，返回一个 `TensorDescriptor` 对象。里面除了包含 tensor 原有的数据之外，还包含 shape、stride、block shape（对于 blocked layout）、dtype、memory layout 等等。有了 tensor descriptor，可以通过 `load_tensor_desc_to_smem` 函数把该 tensor 加载到 shared memory 中。

在 `_gluon_attn` 函数中，主要的计算调用的是 `_attn_fwd_inner` 函数。这个函数接收 `desc_k` 和 `desc_v`，但是要把 `desc_q` 先采用 `load_tensor_desc_to_smem` 函数加载到 shared memory 中，还要分成两个 tile 分开加载两次（原因不太理解）。它的结果貌似是通过 `info0` 和 `info1` 这两个 `InnerLoopInfo` 对象来描述的，它有一个 `consume_result` 方法，可以用来直接把结果写入到输出 tensor 中。在 `_attn_fwd_inner` 函数中，主要的计算用的 `gl.warp_specialize` 函数来进行，里面除了接收 config 等信息之外，还接收 `_attn_fwd_correction`, `_attn_fwd_softmax0`, `_attn_fwd_softmax1`, 
`_attn_fwd_mma`, `_attn_fwd_load` 这些真正用来做计算的函数。这些函数具体做计算时，就用到了我们前面提到的 producer 和 consumer 协同工作的机制。