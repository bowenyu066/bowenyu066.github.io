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

> In Triton, a linear layout (LL) is a function that maps from a "hardware location" to a "logical tensor index".

简而言之，linear layout 是一个函数 $$\mathcal{L}$$，将硬件资源的布局映射到对应的 logical tensor 坐标上。其之所以被称为*线性的*（linear），是在以下两个意义上线性：

1. 硬件资源的布局和 logical tensor 的坐标都可以使用**二进制数**来表示。这得益于硬件资源（如 registers、threads、warps、CTA blocks 等）和 logical tensor 坐标的各个维度通常都是 2 的幂次。
2. layout 函数在**异或**（xor，通常记为⊕）运算下是线性的。由于硬件坐标和 logical tensor 坐标都可以用二进制数表示，xor 运算可以被视为对二进制数的每一位进行独立的无进位加法运算：
   $$
   0\oplus0=0,\ 0\oplus1=1,\ 1\oplus0=1,\ 1\oplus1=0
   $$

使用 xor 运算（而不是通常的加法运算）来描述线性性质，看起来有些难以理解。其一大好处是可以用来描述 swizzling（混洗）操作。[Triton Linear Layout: Concept](https://www.lei.chat/posts/triton-linear-layout-concept/) 一文中提到的第二个例子就是 swizzling layout 的一个例子，其中便使用了 xor 操作。

有了以上概念，linear layout $$\mathcal{L}$$ 的线性性质可以用数学语言描述为：

$$
\mathcal{L}(x_0\oplus y_0, x_1\oplus y_1, \cdots, x_n\oplus y_n) = \mathcal{L}(x_0, x_1, \cdots, x_n) \oplus \mathcal{L}(y_0, y_1, \cdots, y_n)
$$

由于 linear layout 的每个输入坐标 $$x_i$$ 都是二进制数，我们可以自然地把多维坐标 $$(x_0, x_1, \cdots, x_n)$$ 前后连接到一起展平成一维，形成一个更长的二进制数。因此，所有的 linear layout 都可以被视为一个一维的函数 $$\mathcal{L}(x=b_0b_1...b_n)$$，其中 $$b_i$$ 是二进制数的每一位。

对于满足上述线性性质的 linear layout $$\mathcal{L}$$，我们只需要部分的输入输出坐标对 $$x_k \mapsto y_k$$，就可以利用线性性质求出任意的输入输出坐标对。最简单也是最自然的输入坐标即为 2 的幂次的整数 $$x_k = 2^k=(00...010...0)_2$$。由于 xor 运算是无进位加法，任意输入坐标 $$x_k$$ 都可以被表示为若干个 2 的幂次的整数的异或和，相应的输出坐标 $$y_k$$ 也可以被表示为若干个输出坐标的异或和。以下面例子为例，输入坐标 $$(t, w)$$ 均在 $$[0, 3]$$ 范围内，给出了 $$(0, 1)$$、$$(0, 2)$$、$$(1, 0)$$ 和 $$(2, 0)$$ 四个输入坐标的输出坐标：

$$
\mathcal{L} (0, 1) = (0, 1) \\
\mathcal{L} (0, 2) = (0, 2) \\
\mathcal{L} (1, 0) = (1, 1) \\
\mathcal{L} (2, 0) = (2, 2) \\
$$

任意输入坐标 $$(t, w)$$ 都可以表示为 $$t = 2^0 \cdot t_0 \oplus 2^1 \cdot t_1$$ 和 $$w = 2^0 \cdot w_0 \oplus 2^1 \cdot w_1$$，其中 $$t_i, w_i \in \{0, 1\}$$。这样，我们就能计算出任意输入坐标对应的输出坐标：

$$
\begin{align*}
    \mathcal{L} (1, 3) &= \mathcal{L} (1, 0) \oplus \mathcal{L} (0, 1) \oplus \mathcal{L} (0, 2)
    \\&= (1, 1) \oplus (0, 1) \oplus (0, 2) = (1, 2)
\end{align*}
$$

因此，我们可以把 $$(0, 1)$$、$$(0, 2)$$、$$(1, 0)$$ 和 $$(2, 0)$$ 四个输入坐标看成是 linear layout 的**基向量**（basis vectors），他们都对应 2 的幂次的整数。

Triton 里最基本的 linear layout 有两个：identity layout 和 zero layout。

- `LinearLayout::identity1D`：将 $$[0, \text{size})$$ 范围内的整数进行恒等映射 $$\mathcal{L}(x) = x$$。
  ```cpp

  static LinearLayout identity1D(int32_t size, StringAttr inDim, StringAttr outDim);
  ```
- `LinearLayout::zeros1D`：将 $$[0, \text{size})$$ 范围内的整数映射为 0，即 $$\mathcal{L}(x) = 0$$。
  ```cpp

  static LinearLayout zeros1D(int32_t size, StringAttr inDim, StringAttr outDim);
  ```
  
这两种最基本的 linear layout 可以通过乘法构造出更复杂的 linear layout。除此以外，Triton 提供了方便的 constructor 来构造 linear layout。

- `LinearLayout(BasesT, ArrayRef<StringAttr>)`：通过一组完备的基向量输入输出对来构造 linear layout。这种构造会自动检查该 linear layout 是否为满射（surjective），即是否能覆盖所有的 logical tensor 坐标。
  ```cpp

  #include "mlir/IR/MLIRContext.h"
  #include "llvm/ADT/MapVector.h"
  #include "LinearLayout.h"

  // Setup necessary MLIR components
  mlir::MLIRContext context;
  auto inDim1 = mlir::StringAttr::get(&context, "in1");
  auto outDim1 = mlir::StringAttr::get(&context, "out1");
  auto outDim2 = mlir::StringAttr::get(&context, "out2");

  // Create a layout by defining the bases directly.
  using BasesT = llvm::MapVector<mlir::StringAttr,
                              std::vector<std::vector<int32_t>>>;
  BasesT bases;
  bases[inDim1] = {
      {1, 0}, // L(in1=1) = (out1=1, out2=0)
      {5, 1}, // L(in1=2) = (out1=5, out2=1)
      {2, 2}  // L(in1=4) = (out1=2, out2=2)
  };

  std::vector<mlir::StringAttr> outDimNames = {outDim1, outDim2};

  // This constructor infers output sizes and requires the map to be surjective.
  LinearLayout layout(bases, outDimNames);

  // layout.getOutDimSize("out1") would be 8 (next power of 2 >= 5)
  // layout.getOutDimSize("out2") would be 4 (next power of 2 >= 2)
  ```
  - 可以使用 C++ initializer lists ({...}) 来简化初始化：
  ```cpp

  #include "mlir/IR/MLIRContext.h"
  #include "LinearLayout.h"

  // Setup
  mlir::MLIRContext context;
  auto in1 = mlir::StringAttr::get(&context, "in1");
  auto in2 = mlir::StringAttr::get(&context, "in2");
  auto out1 = mlir::StringAttr::get(&context, "out1");
  auto out2 = mlir::StringAttr::get(&context, "out2");

  // Use the initializer-list friendly constructor.
  LinearLayout layout(
      /* bases */
      {
          {in1, { {0, 1}, {0, 2} } }, // L(in1=1)={0,1}, L(in1=2)={0,2}
          {in2, { {0, 4}, {0, 8}, {1, 1} } } // L(in2=1)={0,4}, L(in2=2)={0,8}, L(in2=4)={1,1}
      },
      /* outDimNames */
      {out1, out2}
  );
  ```
- `LinearLayout(BasesT, ArrayRef<pair<StringAttr, int32_t>>, bool)`：同样通过一组完备的基向量输入输出对来构造 linear layout，并且可以选择是否检查该 linear layout 是否为满射。由于非满射情况下无法自动推断出输出维度的大小，因此需要手动指定输出维度的大小。
  ```cpp

  #include "mlir/IR/MLIRContext.h"
  #include "llvm/ADT/MapVector.h"
  #include "LinearLayout.h"

  // Setup necessary MLIR components
  mlir::MLIRContext context;
  auto inDim1 = mlir::StringAttr::get(&context, "in1");
  auto outDim1 = mlir::StringAttr::get(&context, "out1");

  // Create a non-surjective layout with explicit output sizes.
  using BasesT = llvm::MapVector<mlir::StringAttr,
                                std::vector<std::vector<int32_t>>>;
  BasesT bases;
  bases[inDim1] = {
      {1}, // L(in1=1) = (out1=1)
      {4}  // L(in1=2) = (out1=4)
  };

  // Explicitly define the output dimension and its size.
  std::vector<std::pair<mlir::StringAttr, int32_t>> outDims = {
      {outDim1, 32} // The codomain for out1 is [0, 32), even though we only
                    // produce values up to 5.
  };

  // Create the layout, specifying that it doesn't need to be surjective.
  LinearLayout layout(bases, outDims, /*requireSurjective=*/false);
  ```
  - 同样，可以使用 C++ initializer lists ({...}) 来简化初始化：
  ```cpp

  #include "mlir/IR/MLIRContext.h"
  #include "path/to/your/LinearLayout.h"

  // Setup
  mlir::MLIRContext context;
  auto inDim1 = mlir::StringAttr::get(&context, "in1");
  auto outDim1 = mlir::StringAttr::get(&context, "out1");

  //  Use the initializer-list friendly constructor for a non-surjective layout.
  LinearLayout layout(
      /* bases */
      {
          {inDim1, { {1}, {4} } } // L(in1=1) = {1}, L(in1=2) = {4}
      },
      /* outDims */
      {
          {outDim1, 32} // Explicitly specify size of out1 is 32.
      },
      /* requireSurjective */
      false
  );
  ```

### Multiplication

两个 linear layout 可以通过乘法构造出新的 linear layout。在讲解乘法之前，我们应当强调，linear layout 的输入和输出坐标是命名的，从上面给的几个例子里也能看出这一点。对于单独的一个 layout，输入和输出的名称是任意的，只是为了表示有不同输入和输出的通道；但是，在两个 linear layout 相乘时，他们可以有相同的输入/输出名称，也可以是不同的——相同的输入/输出名称表示两个 linear layout 使用相同的输入/输出通道。我们下面分情况进行讨论。

- 输入和输出的名称均不重合：直接将两个 linear layout 的输入输出坐标进行拼接。

  $$
  \mathcal{L}_1(x_1; \text{`i1'}; \text{`o1'}) * \mathcal{L}_2(x_2; \text{`i2'}; \text{`o2'})\mapsto \mathcal{L}(x_1, x_2; \text{`i1'}, \text{`i2'}; \text{`o1'}, \text{`o2'}) \\
  \text{such that } \mathcal{L}(x_1, x_2) = (\mathcal{L}_1(x_1), \mathcal{L}_2(x_2))
  $$

  如下图所示，图示中所有的方格代表二进制表示的一个 bit，$$\mathcal{L}_1$$ 和 $$\mathcal{L}_2$$ 的输入输出坐标分别用不同的颜色表示。可以看到，$$\mathcal{L}_1 * \mathcal{L}_2$$ 的输入输出坐标仅仅只是 $$\mathcal{L}_1$$ 和 $$\mathcal{L}_2$$ 的输入输出坐标的拼接。

  ![](/images/posts/2025-6-17-linear-layouts-in-triton/1.png)

- 输入的名称重合，输出的名称不重合：合成的 layout 将输入合成为一个通道，将输出拼接。

  $$
  \mathcal{L}_1(x_1; \text{`i'}; \text{`o1'}) * \mathcal{L}_2(x_2; \text{`i'}; \text{`o2'})\mapsto \mathcal{L}(x; \text{`i'}; \text{`o1'}, \text{`o2'}) \\
  \text{such that } \mathcal{L}(x) = (\mathcal{L}_1(x_1), \mathcal{L}_2(x_2)), \text{ where } x = \text{concat}(x_2, x_1) 
  $$

  如下图所示，将输入合成为一个通道时，乘号前 layout（$$\mathcal{L}_1$$）的输入放在低位，乘号后 layout（$$\mathcal{L}_2$$）的输入放在高位。

  ![](/images/posts/2025-6-17-linear-layouts-in-triton/2.png)

除此以外，还有两种可能的情况，分别是输出名称重合、输入输出名称均重合。合成通道和拼接的规则和上面完全相同，在此不再赘述。

- 输入的名称不重合，输出的名称重合：合成的 layout 将输出合成为一个通道，将输入拼接。
- 输入输出的名称均重合：合成的 layout 将输入输出均合成为一个通道。

有了以上定义的乘法规则，我们就可以用最基本的 `identity1D` 和 `zeros1D` 来构造出更复杂的 linear layout。我们可以看几个例子。

1. $$\mathcal{L}(x) = x / 4$$，$$x \in [0, 8)$$；该 layout 可视作 `zeros1D(4, "i", "o") * identity1D(2, "i", "o")`，因其等效于直接舍弃 $$x$$ 的低两位。

2. $$\mathcal{L}(x) = x \text{ \% } 4$$，$$x \in [0, 8)$$；该 layout 可视作 `identity1D(4, "i", "o") * zeros1D(2, "i", "o")`，因其等效于直接舍弃 $$x$$ 的最高位。

3. $$\mathcal{L}(x) = (x \text{ \% } 4,\  x / 4)$$，$$x \in [0, 32)$$；该 layout 可视作 `identity1D(4, "i", "o1") * identity1D(8, "i", "o2")`，因其等效于直接将 $$x$$ 的低两位和高三位分别作为两个输出。

## Convertion to Traditional Layouts

### CTA Layouts

<!-- An explanation of what CTA, CGA, blocks, warps, threads and registers are in Triton. -->

硬件（GPU）最小的计算单元是 thread，每个 thread 有若干 registers 用来存储数据，各个 thread 之间可以并行运算。一个 warp 由若干 threads 组成，通常是 32 个。一个 CTA block 由若干 warps 组成。最后，一个 CGA block 由若干 CTA blocks 组成，完成一个完整的计算任务。通常而言，这些组成的倍数都是 2 的幂次。

以矩阵乘法为例，假设有两个矩阵 $$A$$ 和 $$B$$，大小分别是 $$M \times K$$ 和 $$K \times N$$，现在想要计算矩阵 $$C = AB$$。每对元素的乘法 $$a_{ij} \cdot b_{jk}$$ 由一个 thread 来完成；一整行与一整列之间的乘法 $$c_{ik}=\sum_{j}a_{ij} \cdot b_{jk}$$ 由一个或多个 warp 来完成；一个 CTA block 则负责计算 $$C$$ 的一块小区域；最后，一个 CGA block 通过若干 CTA blocks 来计算整个矩阵 $$C$$。

CTA layout 的目的是将硬件资源的布局映射到对应的 logical tensor 坐标上；比如，这块 CTA block 负责 logical tensor 的哪一块数据，或者这个 register 存储了 logical tensor 的哪个元素。根据[文档（LinearLayoutConversions.cpp）](https://github.com/triton-lang/triton/blob/d9facf3/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)里的定义，我们可以将 CTA layout 按照层次的不同分为两类：

- cgaLayout：一块 CGA block 被划分为若干 CTA blocks，相应地，所对应的 logical tensor 也被划分为若干 blocks。该 layout 将 CTA block 的坐标映射到 logical tensor 的 block 坐标上。
- ctaLayout：一块 CTA block 内部有许多 warps，每个 warp 内部有许多 threads，每个 thread 内部有若干 registers。该 layout 将（warp_index, thread_index, register_index）映射到对应的 logical tensor block 内部的相对坐标上。

[文档](https://github.com/triton-lang/triton/blob/d9facf3/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)同时指出，不同文档的命名法多有混淆和差别。比如，该[文档（TritonGPUAttrDefs.td）](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L56)中定义的 `CTALayoutAttr` 实际上是我们这里的 `cgaLayout`。

#### cgaLayout (CTALayoutAttr)

声明一个 cgaLayout 需要指定以下参数：

- `CTAsPerCGA`：每个 CGA block 是如何分割为若干 CTA blocks 的。比如，`CTAsPerCGA = [2, 4]` 表示每个 CGA block 被分割为 2x4 个 CTA blocks。
- `CTASplitNum`：logical tensor 是如何分割为若干 blocks 的。比如，`CTASplitNum = [2, 4]` 表示 logical tensor 被分割为 2x4 个 blocks。
- `CTAOrder`：对于多维的 CTA blocks，当展平为一维时的排列顺序。比如，`CTAOrder = [1, 0]` 表示先按 dim1（一整行从左到右）排列，再按 dim0（再从上到下）排列。

多个 CTA block 可以对应同一块 logical tensor block。比如，如果 `CTAsPerCGA[0] = 8`，`CTASplitNum[0] = 2`，则 CTA 在第 0 个维度上被分成 8 份，交替映射到 logical tensor 被分成的 2 份上，分配的方式为 (0, 1, 0, 1, 0, 1, 0, 1)。从这个例子也可以看出，`CTAsPerCGA` 和 `CTASplitNum` 的每个维度上的值必须是整数倍数关系。在转换为 linear layout 时，函数将默认检查上述条件。在该条件满足时，cgaLayout 的数学表达式可以写为：

$$
\mathcal{L}(x_0, x_1, \cdots, x_n)= \left( x_0 \text{ \% split[0]}, x_1 \text{ \% split[1]}, \cdots, x_n \text{ \% split} [n]   \right)
$$

其中，$$(x_0, x_1, \cdots, x_n)$$ 是 CTA block 的坐标，坐标限定在 `CTAsPerCGA` 给定的范围内；$$\text{split}[i]$$ 是 `CTASplitNum` 在第 $$i$$ 个维度上的值。

如果每个维度上的 `CTAsPerCGA` 和 `CTASplitNum` 都是 2 的幂次，则可以将 cgaLayout 转换为 linear layout。这时，cgaLayout 的输入坐标 $$(x_0, x_1, \cdots, x_n)$$ 每一个都可以用二进制来表示，再按照 `CTAOrder` 给出的顺序从低位到高位排列。比如，如果 `CTAsPerCGA = [2, 4]`，`CTASplitNum = [2, 4]`，`CTAOrder = [1, 0]`，则输入坐标 $$(x_0, x_1)$$ 可以分别表示为一个 1-bit 和一个 2-bit 的二进制数 $$(b_0, b_1)$$；再按照 `CTAOrder` 的顺序排列，得到的 linear layout 输入坐标为 $$b_0b_1$$，即 $$b_1$$ 在低位，$$b_0$$ 在高位。坐标 $$(1, 1)$$ 按照上述规则转换为 linear layout 输入坐标为 $$b_0b_1 = (101)_2$$。同时，为了能够表示取余操作，我们可以利用 layout 的乘法（参见上文），将 $$x \text{ \% } a$$ 转换为 `identity1D(a, "i", "o") * zeros1D(size/a, "i", "o")`。具体 `makeCgaLayout(layout)` 实现的代码梗概如下：

```cpp

LinearLayout ret = LinearLayout::empty(); // Initialization
for (int i = 0; i < rank; i++) {
  // Start with the most minor dimension, which is order[0].
  // Check the divisibility condition
  int dim = layout.getCTAOrder()[i];
  int split = layout.getCTASplitNum()[dim];
  int ctas = layout.getCTAsPerCGA()[dim];
  assert(ctas % split == 0);
  // Create the linear layout for this dimension
  ret *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
         LinearLayout::zeros1D(ctas / split, kBlock, outDimNames[dim]);
}
// Transpose to standard order (dim0, dim1, ...).
return ret.transposeOuts(outDimNames);
```

#### ctaLayout

<!-- TODO: Explain how to combine a ctaLayout with a cgaLayout. -->

