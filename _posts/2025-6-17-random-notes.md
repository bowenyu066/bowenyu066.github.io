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
- emitIndices 函数是与 layout 转换相关的，研究一下里面的细节
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

Therefore, `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)` is equivalent to `triton.jit(add_kernel).__getitem__(grid)(x, y, output, n_elements, BLOCK_SIZE=1024)`, which is `triton.jit(add_kernel).run(grid=grid, warmup=False, *(x, y, output, n_elements, BLOCK_SIZE=1024))`.

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