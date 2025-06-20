---
title: CuTe Layouts
date: 2025-6-20 11:48:03 +0800
excerpt: "Notes on CuTe Layouts in NVIDIA cutlass documentations."
---

本帖暂时用中文写，稍微方便点。后面有可能会转换为英文版。

与 Linear layout 类似，CuTe layout 的本质也是实现 logical tensor 的坐标和硬件内存地址布局的映射。相比于 Linear layout，CuTe layout 并不要求各个维度的长度都是 2 的幂次，这使得它比 Linear layout 更加灵活；但是，并不是所有的 linear layout 都可以用 CuTe layout 来实现，比如 `swizzledSharedLayout` 因涉及 xor 操作而难以用 CuTe layout 来表达。本文将介绍 CuTe layout 的基本概念和一些使用实例。

## CuTe Layout 的基本概念

一整块 logical tensor 在分配到内存中存储时通常不是任意分布的，而是采用一定的周期性模式，这样也更便于后续使用这些数据来进行计算。我们先以一块 2 维的 logical tensor 为例。最简单的且最自然的分布方式即为行/列优先（row/column-major），如下图所示。

![Row/Column-major layout](/images/posts/cute-layouts/row-and-col-major.png)


## CuTe Layout 的代数运算


