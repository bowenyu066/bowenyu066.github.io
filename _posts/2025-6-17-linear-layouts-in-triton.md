---
title: Linear Layouts in Triton
date: 2025-6-17 17:10:00 +0800
permalink: /posts/2025/06/linear-layouts-in-triton/
excerpt: "Notes on linear layouts in Triton and its conversion with various traditional layout types."
tags: 
    - Computer Science
categories: 
    - Computer Science
---

本帖暂时用中文写，稍微方便点。后面有可能会转换为英文版。

## Linear Layouts

### Concepts

### Multiplication

## Convertion to Traditional Layouts

### CTA Layouts

<!-- An explanation of what CTA, CGA, blocks, warps, threads and registers are in Triton. -->

硬件（GPU）最小的计算单元是 thread，每个 thread 有若干 registers 用来存储数据，各个 thread 之间可以并行运算。一个 warp 由若干 threads 组成，通常是 32 个。一个 CTA block 由若干 warps 组成。最后，一个 CGA block 由若干 CTA blocks 组成。如果这些组成的倍数通常都是 2 的幂次，下面所介绍的 CTA layouts 就都可以用 linear layouts 来表示。

CTA layout 的目的是将硬件资源的布局映射到对应的 logical tensor 坐标上（比如，这块 CTA block 负责 logical tensor 的哪一块数据，或者这个 register 存储了 logical tensor 的哪个元素）。根据[文档（LinearLayoutConversions.cpp）](https://github.com/triton-lang/triton/blob/d9facf3/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)里的定义，我们可以将 CTA layout 按照层次的不同分为两类：

- cgaLayout：一块 CGA block 被划分为若干 CTA blocks，相应地，所对应的 logical tensor 也被划分为若干 blocks。该 layout 将 CTA block 的坐标映射到 logical tensor 的 block 坐标上。
- ctaLayout：一块 CTA block 内部有许多 warps，每个 warp 内部有许多 threads，每个 thread 内部有若干 registers。该 layout 将（warp_index, thread_index, register_index）映射到对应的 logical tensor block 内部的相对坐标上。

[文档](https://github.com/triton-lang/triton/blob/d9facf3/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)同时指出，不同文档的命名法多有混淆和差别。比如，该[文档（TritonGPUAttrDefs.td）](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L56)中定义的 `CTALayoutAttr` 实际上是我们这里的 `cgaLayout`。

#### cgaLayout (CTALayoutAttr)

声明一个 cgaLayout 需要指定以下参数：

- `CTAsPerCGA`：每个 CGA block 是如何分割为若干 CTA blocks 的。比如，`CTAsPerCGA = [2, 4]` 表示每个 CGA block 被分割为 2x4 个 CTA blocks。
- `CTASplitNum`：logical tensor 是如何分割为若干 blocks 的。比如，`CTASplitNum = [2, 4]` 表示 logical tensor 被分割为 2x4 个 blocks。
- `CTAOrder`：对于多维的 CTA blocks，当展平为一维时的排列顺序。比如，`CTAOrder = [1, 0]` 表示先按 dim1（一整行从左到右）排列，再按 dim0（再从上到下）排列。

多个 CTA block 可以对应同一块 logical tensor block。在转换为 linear layout 时，函数将默认检查在每个维度上的 CTA block 数量是否是 logical tensor block 数量的整数倍数。

```cpp

int rank = layout.getCTAOrder().size();
for (int i = 0; i < rank; i++) {
    // Start with the most minor dimension, which is order[0].
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];
    assert(ctas % split == 0);
  }
```

<!-- TODO: Explain how to convert cgaLayout to linear layout. -->

#### ctaLayout

<!-- TODO: Explain how to combine a ctaLayout with a cgaLayout. -->

