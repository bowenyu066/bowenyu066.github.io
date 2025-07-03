---
title: Some Questions on Cutlass Layout
date: 2025-6-6 11:43:00 +0800
permalink: /posts/2025/06/questions-on-cutlass-layout
excerpt: "My own notes and questions on the layout of CUTLASS."
tags: 
    - Computer Science
    - CUDA
    - CUTLASS
categories: 
    - Notes
---

## Layout Composition

The composition of two layouts `A` and `B` is defined as the function composition of the two layouts, i.e., `(A o B)(x) = A(B(x))`. According to the [CUTLASS documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md), we can compute the composition of two layouts by making use of the left-distributive property: `A o (B1, B2, ..., Bn) = (A o B1, A o B2, ..., A o Bn)`, so long as `B` is injective.

Although not stated explicitly, the documentation suggests two other conditions are necessary for the left-distributive property to hold: the **stride divisibility condition** and the **shape divisibility condition**. The stride divisibility condition ensures that every sub-stride of the layout `B` is divisible by the shapes of the layout `A`. Here are some examples that satisfy the stride divisibility condition:

```text

(6,2) /  2 => (3,2)
(6,2) /  3 => (2,2)
(6,2) /  6 => (1,2)
(6,2) / 12 => (1,1)
(3,6,2,8) /  3 => (1,6,2,8)
(3,6,2,8) /  6 => (1,3,2,8)
(3,6,2,8) /  9 => (1,2,2,8)
(3,6,2,8) / 72 => (1,1,1,4)
```

While the shape divisibility condition ensures that the every sub-shape (sorry for coining this term) of the layout `B` is divisible by the reduced shape of the layout `A` after applying the stride divisibility condition. Here are also some examples that satisfy the shape divisibility condition; notice, the shapes on the left hand side correspond to the ones after dividing the strides:

```text

(6,2) %  2 => (2,1)
(6,2) %  3 => (3,1)
(6,2) %  6 => (6,1)
(6,2) % 12 => (6,2)
(3,6,2,8) %  6 => (3,2,1,1)
(3,6,2,8) %  9 => (3,3,1,1)
(1,2,2,8) %  2 => (1,2,1,1)
(1,2,2,8) % 16 => (1,2,2,4)
```

With this in mind, I once assumed that the left-distributive property holds for all layouts that satisfy the stride divisibility condition and shape divisibility condition. However, I was not able to prove this property mathematically. If left-distributive property holds, this suggests that the layout `A` should be linear with respect to the layout `B`. This is because of the following reasoning:

Let \\(x\\) be an arbitrary index in the domain of layout `B`, denoted as \\((s_0, s_1, ..., s_n):(d_0, d_1, ..., d_n) \equiv (B_0, B_1, ..., B_n)\\). To compute the corresponding output index of this layout, we first decompose the index `x` into its natural coordinates with respect to the shape of layout `B`, i.e., \\((x_0, x_1, ..., x_n)\\). Then, we apply to each coordinate the sub-layout `B_i` of `B`, and sum them up to get the final output index. This can be expressed as:

$$
B(x) = \sum_{i=0}^{n} B_i(x_i) = \sum_{i=0}^{n} d_i \cdot x_i
$$

Therefore, the left-distributive property `A o (B1, B2, ..., Bn) = (A o B1, A o B2, ..., A o Bn)` can be expressed in mathematical terms as follows:

$$
A((B_0, B_1, ..., B_n)(x)) = (A \circ B_0, A \circ B_1, ..., A \circ B_n)(x)
$$

which is equivalent to:

$$
A \left(\sum_i B_i(x_i) \right) = \sum_i A(B_i(x_i))
$$

This means that the layout `A` is indeed linear with respect to the output of each sub-layout `B_i`, which, I believe, is highly non-trivial in that `A` itself is far from being a linear function. What I am not sure is what conditions are imposed by the stride divisibility condition and shape divisibility condition--maybe they are sufficient to ensure the above linearity.

The documentation does not provide a proof of this property. Therefore, I decided to first test it with some examples that satisfy the stride divisibility condition and shape divisibility condition, in hope of obtaining some insights. Here's a function that I wrote to test the built-in `cute::composition` function in CUTLASS:

```cpp

#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
using namespace cute;
#define PRINT(x) print(#x); print(": "); print(x); print("\n");

template <class Shape1, class Stride1, class Shape2, class Stride2>
bool test_composition(Layout<Shape1, Stride1> const& a, Layout<Shape2, Stride2> const& b){
  auto c = composition(a, b);
  PRINT(a);
  PRINT(b);
  PRINT(c);
  for (int i = 0; i < size(c); i++) {
    if (c(i) != a(b(i))) {
      printf("Mismatch at index %d: c(%d) = %d, a(b(%d)) = %d\n", i, i, c(i), i, a(b(i)));
      return false;
    }
  }
  printf("Composition test passed!\n");
  return true;
}
```

The `cute::composition` provided automatically checks the stride divisibility condition and shape divisibility condition, so I don't need to worry about that. To my surprise, I did find some counterexamples where the left-distributive property does not hold. Here are some examples that I tested:

```text

a: (_10,_2):(_16,_4)
b: (_4,_5):(_5,_2)
c: ((_2,_2),_5):((_80,_4),_32)
Mismatch at index 13: c(13) = 176, a(b(13)) = 20
```

```text

a: (_10,_2):(_16,_4)
b: (_2,_5):(_5,_2)
c: (_2,_5):(_80,_32)
Mismatch at index 7: c(7) = 176, a(b(7)) = 20
```

```text

a: (_6,_2):(_8,_2)
b: (_4,_3):(_3,_2)
c: ((_2,_2),_3):((_24,_2),_16)
Mismatch at index 9: c(9) = 56, a(b(9)) = 10
```

Of course, there are also many examples where the property does hold:

```text

a: (_6,_2):(_8,_2)
b: (_4,_3):(_3,_1)
c: ((_2,_2),_3):((_24,_2),_8)
Composition test passed!
```

```text

a: (_10,_2):(_16,_4)
b: (_10,_2):(_2,_1)
c: ((_5,_2),_2):((_32,_4),_16)
Composition test passed!
```

```text

a: (_10,_2):(_16,_4)
b: (_5,_4):(_1,_5)
c: (_5,(_2,_2)):(_16,(_80,_4))
Composition test passed!
```

```text

a: (_10,_2):(_16,_4)
b: (_5,_2):(_2,_1)
c: (_5,_2):(_32,_16)
Composition test passed!
```

```text

a: (_10,_2):(_16,_4)
b: (_4,_5):(_5,_1)
c: ((_2,_2),_5):((_80,_4),_16)
Composition test passed!
```

```text

a: (_10,_2):(_16,_4)
b: (_2,_10):(_10,_2)
c: (_2,(_5,_2)):(_4,(_32,_4))
Composition test passed!
```

So, here's everything I have so far. I am still not sure what the conditions are for the left-distributive property to hold, but I do know that it does not hold in general. The CUTLASS library has been released for over two years (maybe), but I have not seen any discussions on this topic. Maybe it is indeed a non-trivial problem, but maybe it isn't a big deal and does not really affect anything.
