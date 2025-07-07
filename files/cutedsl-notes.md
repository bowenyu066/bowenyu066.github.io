---
title: Notes on CUTLASS DSL (CuTeDSL)
date: 2025-7-6 22:07:12 +0800
excerpt: "Basic, introductory notes on CuTeDSL, a domain-specific language for CUDA programming, which allows users to write CUDA kernels in Python."
tags: 
    - Computer Science
    - CUDA
    - Python
    - CuTeDSL
    - GPU
categories: 
    - Notes
---

# `elementwise_add.py` for Ampere GPU in CuTeDSL

## TV Layout

From: [NVIDIA CUTLASS Documentation (0t_mma_atom)](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0t_mma_atom.html)

TV layout is a short for "thread-value layout". First introduced in Volta GPU (spanning Turing and Ampere), this layout is used to describe how *threads* within a QP (quadpair, a group of 8 threads) and *values* (accumulators, registers) within a thread, labeled by `(logical_thr_id, logical_val_id)`, map to the logical tensor indices `(m, n)` [^1].

Let's investigate an example below, of an 8x8 matrix:

![TV Layout Example](/images/posts/cutedsl-notes/tv-layout.png)

Each thread owns 8 values. To describe the layout, first focus on changing `logical_thr_id` while keeping `logical_val_id = 0` fixed:

```text
(T=0, V=0) -> (0, 0) = 0
(T=1, V=0) -> (1, 0) = 1
(T=2, V=0) -> (0, 2) = 16
(T=3, V=0) -> (1, 2) = 17
(T=4, V=0) -> (4, 0) = 4
(T=5, V=0) -> (5, 0) = 5
(T=6, V=0) -> (4, 2) = 20
(T=7, V=0) -> (5, 2) = 21
```

where `T=4,5,6,7` are the 4th, 5th, 6th, 7th logical thread id of the MMA corresponding to thread indices of `16`,`17`,`18`,`19` of the warp. Such mapping between logical and real thread indices is to be recorded in `ThrID` mapping (and this is why we call the above thread indices as "*logical* thread id"). We may infer from `T=0` to `T=7` data that there exist three types of periodicity: `T=0 -> T=1` with stride `1`, `T=0 -> T=2` with stride `16`, and `T=0 -> T=4` with stride `4`. The layout of `logical_thr_id` is thus described as:

```cpp
using ThreadLayout = Layout<Shape<_2, _2, _2>, Stride<_1, _16, _4>>;
```

Next, fix `logical_thr_id = 0` and change `logical_val_id`. But first, we need to specify how values are ordered within a thread. The picture below illustrates the value ordering:

![Value Ordering Example](/images/posts/cutedsl-notes/tv-layout-2.png)

Given such ordering, we can now describe the mapping of `logical_val_id` to `(m, n)` indices:

```text
(T=0, V=0) -> (0, 0) = 0
(T=0, V=1) -> (0, 1) = 8
(T=0, V=2) -> (2, 0) = 2
(T=0, V=3) -> (2, 1) = 10
(T=0, V=4) -> (0, 4) = 32
(T=0, V=5) -> (0, 5) = 40
(T=0, V=6) -> (2, 4) = 34
(T=0, V=7) -> (2, 5) = 42
```

The rule is clear: there also exist three types of periodicity: `V=0 -> V=1` with stride `8`, `V=0 -> V=2` with stride `2`, and `V=0 -> V=4` with stride `32`. The layout of `logical_val_id` can thus described as:

```cpp
using ValLayout = Layout<Shape<_2, _2, _2>, Stride<_8, _2, _32>>;
```

Finally, we can combine the two layouts to get the TV layout:

```cpp
using TVLayout = Layout<Shape <Shape <_2,  _2, _2>, Shape <_2, _2,  _2>>,
                        Stride<Stride<_1, _16, _4>, Stride<_8, _2, _32>>>;
```

# `dense_gemm.py` for Hopper GPU in CuTeDSL

We'll investigate `dense_gemm.py` as a specific example of CuTeDSL usage. This script demonstrates how to use CuTeDSL to implement a dense matrix multiplication (GEMM) operation for the NVIDIA Hopper architecture.

Basic parameters description:

- Matrix `A` dimension: `M x K x L`, `L` is the batch dim; `A` can be row-major (`K`) or column-major (`M`)
- Matrix `B` dimension: `K x N x L`, `B` can be row-major (`N`) or column-major (`K`)
- Matrix `C` dimension: `M x N x L`, `C` can be row-major (`N`) or column-major (`M`)

The workflow is as follows:

1. Load `A` and `B` matrices from the global memory, GMEM, to shared memory, SMEM, using Tensor Memory Access (TMA) operations
2. Perform matrix multiply-accumulate (MMA) operations using Hopper's WGMMA instructions; results are stores in accumulators (registers, RMEM)
3. Store results from RMEM back to SMEM, then to GMEM with TMA operations

The following diagram illustrates the above workflow.

![Overall workflow](/images/posts/cutedsl-notes/workflow.png)

To run the script, the following arguments are required:

- `--mnkl`: the dimensions `M, N, K, L` of the matrices
- `--tile_shape_mnk`: the Hopper WGMMA **tile shape** `M x N x K`, namely, the shape of the CTA tile
  - This tile shape stands for: one CTA block will handle one `M x N` sub-matrix of `C` (portionally), by computing the matrix multiplication of `A` and `B` with the shape `M x K` and `K x N`, respectively
  - Tile shapes `M`, `N`, and `K` do not need to match the `--mnkl` dimensions
  - The following constraints are checked:
    - Tile shape `M` must be 64/128
    - Tile shape `N` must be 64/128/256
    - Tile shape `K` must be 64
- `--cluster_shape_mn`: the Hopper WGMMA **cluster shape** `M x N`; I am not sure what this means so far, in the given example it's set to `1 x 1`, so we can ignore it for now
  - It is inferred that this cluster shape refers to the CTA layout: `self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)`
  - Note that it's `cluster_shape_mnk` instead of `cluster_shape_mn`; the `k` dimension is always `1`: `cluster_shape_mnk = (*cluster_shape_mn, 1)`
- `--a_dtype`, `--b_dtype`, `--c_dtype`, `--acc_dtype`: the data types of matrices `A`, `B`, `C`, and accumulators, respectively
- `--a_major`, `--b_major`, `--c_major`: the memory layout of matrices `A`, `B`, and `C` (row-major or column-major)

Other than the above attributes, the `HopperWgmmaGemmKernel` class also contains the following ones:

- `self.atom_layout_mnk`: either `(2, 1, 1)` or `(1, 1, 1)`. When `tile_shape_mnk[0] > 64` (equivalently, `tile_shape_mnk[0] == 128`) and `tile_shape_mnk[1] > 128` (equivalently, `tile_shape_mnk[1] == 256`), it is set to `(2, 1, 1)`, which means using 2 warp groups per CTA.

The main computation is performed in `HopperWgmmaGemmKernel.__call__(self, a, b, c, stream)`. `a`, `b`, and `c` are `cute.Tensor`'s; `stream` is called a "CUDA stream for asynchronous execution", of which I don't know the exact meaning. The layout information (cute layout) is already stored in `cute.tensor` objects, and can be accessed via `cutlass.utils.LayoutEnum.from_tensor(a)`.

[^1]: By default, the logical tensor indices are encoded in column-major order. See https://github.com/NVIDIA/cutlass/discussions/2197.
