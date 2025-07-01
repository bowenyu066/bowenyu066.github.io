---
title: Random Notes
date: 2025-6-17 14:48:00 +0800
permalink: /posts/2025/06/random-notes/
excerpt: "A personal memo containing various random notes and thoughts."
tags: 
    - Computer Science
categories: 
    - Memo
---

https://www.coursera.org/learn/build-a-computer

llvm::outs()<<"retIsF16"<<retIsF16<<"\n";

ttir -> ttgir

layout 转换的函数

MLIR_ENABLE_DUMP = 1

TODO list for this week:
- 看 triton python tutorials 怎么用的
- 看一下 gluon 到底是干什么的，里面有没有 linear 或者 blocked layouts；如果有 blocked layouts，看看能否转换为 linear
- emitIndices 函数是与 layout 转换相关的，研究一下里面的细节（https://github.com/triton-lang/triton/blob/main/lib/Conversion/TritonGPUToLLVM/Utility.cpp#L310）
- 学会使用 llvm::outs() 打印调试信息
- 学会使用 MLIR_ENABLE_DUMP = 1 来看下 python code 是如何一步步转换为 IR 的，里面是怎么用 layout 的（在哪一个层次，可能是 ttir -> ttgir 层次？）

Decorator: 

```python
add_kernel = triton.jit(add_kernel) # of type JITFunction[type(add_kernel)]
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
# equivalent to triton.jit(add_kernel)[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

JITFunction has implemented its `__getitem__` method, such that `triton.jit(add_kernel)[grid]` is equivalent to `triton.jit(add_kernel).__getitem__(grid)`.

```python
class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)

class JITFunction(KernelInterface[T]): ...
```

Therefore, `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)` is equivalent to `triton.jit(add_kernel).__getitem__(grid)(x, y, output, n_elements, BLOCK_SIZE=1024)`, which is `triton.jit(add_kernel).run(grid=grid, warmup=False, *(x, y, output, n_elements, BLOCK_SIZE=1024))`. The core computation is done in `kernel.run`, which is too complicated that I actually haven't fully figured out how it functions yet.

`typing` is so freaking strange. It makes a python script look like a C++ template code.

Positional and keyword arguments in Python:
 
- `*args` collects all positional arguments into a tuple, while `**kwargs` collects all keyword arguments into a dictionary.
- In a function definition, there are five types of parameters:
  1. Positional-only parameters (e.g., `def func(a, b, /)`): Arguments listed before a forward slash (`/`).
  2. Positional-or-keyword parameters (e.g., `def func(a, b)`): The "normal" arguments we use every day. They appear after any positional-only arguments but before `*args`.
  3. Keyword-only parameters (e.g., `def func(*, a, b)`): Any argument that appears after `*args` or a bare `*`.
  4. Variable positional parameters (e.g., `def func(*args)`).
  5. Variable keyword parameters (e.g., `def func(**kwargs)`).  
- In a general function call:
  - positional arguments are passed first, followed by keyword arguments. This is non-negotiable. 
  - However, when collecting variable positional arguments (`*args`), they can be passed after keyword arguments, where python will cleverly unpack and place them in the correct place. They cannot, nevertheless, be passed after variable keyword arguments (`**kwargs`).

```python
def my_func(pos_only=None, /, std_arg=None, *args, kw_only=None, **kwargs):
    print(f"pos_only: {pos_only}, std_arg: {std_arg}, args: {args}, kw_only: {kw_only}, kwargs: {kwargs}")
```

In the example above, `my_func(kw_only=10, *(30, 40, 50))` would work, but `my_func(**{'kw_only': 10}, *(30, ))` would raise a `SyntaxError`.

By the way, I just found out python checks the order of arguments in a function call during **compilation** step, even before **runtime**.

- The **compilation** step checks `SyntaxError`, `IndentationError`, and `TabError`.
  - `SyntaxError`:
    - *Invalid structure*: Using a keyword in the wrong place, such as `for = 10  # 'for' is a keyword, not a variable name`
    - *Malformed expressions*: Unbalanced parentheses, brackets, or quotes, such as `my_list = [1, 2, (3, 4]  # Mismatched () and []`
    - *Argument order violations*: The rules we've discussed above.
    - *Invalid assignment*: Trying to assign a value to something that can't hold one, such as `"hello" = 12 # Can't assign to a string literal`
  - `IndentationError`:
    - *Unexpected indentation*: An indent that doesn't follow a colon (`:`).
    - *Unmatched indentation*: Code that is expected to be indented but isn't.
  - `TabError`:  If tabs and spaces are used interchangeably within the same block, Python can't reliably determine the indentation level, resulting in a TabError.
    ```python
    def example_function():
        print("Indented with tabs")
        print("This line has a mixture of tabs and spaces")
    ```
    In modern code editors like VSCode, this error is automatically fixed by code editors silently converting all tabs to spaces.

- The **runtime** step checks the rest of errors, such as `NameError`, `TypeError`, `ValueError`, etc.

The conventional `try-except` block can only capture those errors that occur during the **runtime** step. In VSCode, these two types of errors are underlined in different colors: errors occurred during compilation time (can't be captured by `try-except`) are colored red, while errors occurred during runtime (can be captured by `try-except`) are colored yellow.

## July 1

我们这一个月工作的心得：

1. 看了很多东西，也写了一些文档，但是并没有真正上手做什么。
2. 感觉目标不够明确。在看 code 的过程中经常会纠结于一些细节，但是后来发现这些细节并不是我们需要关注的重点，导致浪费了很多时间。
  - 像 triton 这样一个很大的项目，阅读的过程中通常会觉得有些 overwhelming，有很多地方细节实在是太多，不知道究竟要了解到什么程度，经常迷失在细节里面
3. 希望能有更明确的一个可以执行的任务，让我上手写写代码

_cute_ops_gen.py

今天开会的时候跟我说要我重新做回 cuteDsl 的工作。大致我理解的意思是：

- CUTLASS 有两套代码，一套是 C++ 的，另一套是 Python 的
- Python 的代码执行过程是，Python 代码先被翻译为 CuTe IR，然后是 optimized CuTe IR，再到 LLVM IR，最后被编译为机器码
- CUTLASS 是开源的，但是上面所述的转换过程中，Python 到 CuTe 基本已知，CuTe 到 llvm 和 kernel binary 代码并没有开源，只有 Python 写的接口（API，不知道这样叫对不对）和生成的 IR （被叫做 dump）是开源的
- 既然他们想在 Intel 的 GPU 上弄这一套，具体弄到什么程度我也不知道，但是他们想要把这套转换过程弄出来
- 整个工程很繁杂（一个 MLIR 的代码接近几十万行），他们说一个月内弄完不太现实，所以还是让我以看文档为主，对照着 C++ 的代码搞清楚各个函数的预期表现是什么，向下转换的逻辑是什么样的，可能让我在一些最简单的 operator 操作上写一些单元测试

说实话我有些不太开心，感觉自己有点被耍，前面让我看的 linear layout 和 gluon 之类现在都不提了，上一个月花的时间好像直接被浪费掉了。关键是那几个人说的也不清楚，感觉他们自己也没想好要干什么，东一榔头西一棒的。现在又让我回去看 cuteDsl 的文档，感觉有点像是把我当成了一个文档阅读器，搞半天两个月全部都是在看些可能以后永远也不会用到的文档，也不说要看到什么程度，就只是说看看看。（这段是 Copilot 自动生成的，但我觉得说得太好了）

但说回来，这么大的一个工程项目，确实想想都繁杂，可能最简单的任务上手 ramp up 都得花好久才行，更别提我这种从来没接触过 C++、从来没写过 cuda、完全不懂 LLVM 和 IR、从来没真正做过一个项目的人了。虽然我很想要有点代码产出，但是也许现在得换个心态。一个很重要的能学习的点就是观察和阅读庞大项目的代码究竟是如何组织的，为什么要有这么多文件夹（好多都还是同名的），如何高效率地在并不清楚全部细节的情况下阅读并理解代码、不陷入细节漩涡里，这可能比直接上手写代码更重要。我们还可以在读代码的过程中获得工程代码如何书写的一手资料。这样看的话，也不能说上个月就完全一无所获，至少看了不少 Python 代码后知道了 `__init__.py` 到底干什么的，学会了各种奇妙的 Python 语法糖，对 `typing` 库里的类型注解有了更深的理解，还有些杂七杂八的 `abc`、`@builtin`、`@aggregate`、`@triton.jit` 等等的装饰器的用法。如果从这个角度看，可学的可就太多了，不管是 C++ 里各种奇妙的模板元编程，还有 CMake、pyproject.toml 等等配置文件，大项目的文档组织架构、注释风格，Python 代码与 low-level IR 究竟在哪里转换的，什么是 GPUDialect，等等等等。

我也是刚刚才意识到，之前做 UROP 时其实都是当调包侠，这是我第一次需要这么关心底层的东西，所以不知道该怎么把握细节信息量。学吧……

今天实在无心工作。明天 sync 下他们究竟想要我干什么，再 get down。
