---
title: Notes on CUTLASS DSL (CuTeDSL)
date: 2025-7-6 22:07:12 +0800
excerpt: "Basic, introductory notes on CuTeDSL, a domain-specific language for CUDA programming, which allows users to write CUDA kernels in Python."
tags: 
    - Computer Science
    - CUDA
    - Python
    - CuTeDSL
categories: 
    - Notes
---

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
- `--tile_shape_mnk`: the Hopper WGMMA **tile shape** `...x...x...`, namely, the shape of the CTA tile
- `--cluster_shape_mn`: the Hopper WGMMA **cluster shape** `...x...`; I am not sure what this means so far
- `--a_dtype`, `--b_dtype`, `--c_dtype`, `--acc_dtype`: the data types of matrices `A`, `B`, `C`, and accumulators, respectively
- `--a_major`, `--b_major`, `--c_major`: the memory layout of matrices `A`, `B`, and `C` (row-major or column-major)

The main computation is performed in `HopperWgmmaGemmKernel.__call__(self, a, b, c, stream)`. `a`, `b`, and `c` are `cute.Tensor`'s; `stream` is called a "CUDA stream for asynchronous execution", of which I don't know the exact meaning. The layout information (cute layout) is already stored in `cute.tensor` objects, and can be accessed via `cutlass.utils.LayoutEnum.from_tensor(a)`.
