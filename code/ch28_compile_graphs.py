"""讲义：第 28 章 训练里用 compile 与 CUDA Graphs。

运行：uv run python ch28_compile_graphs.py     # CPU 计时；CUDA Graph 部分需要 GPU [GPU]
"""

import re
import time

import torch
from torch._inductor.utils import run_and_get_code


def many_small_ops(x):
    for _ in range(20):
        x = x * 1.01 + 0.01
    return x


def per_call_us(fn, x, n: int = 2000) -> float:
    for _ in range(50):
        fn(x)
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t) / n * 1e6


def fused_names(fn, x) -> set:
    torch._dynamo.reset()
    _, code = run_and_get_code(torch.compile(fn), x)
    return set(re.findall(r"cpp_fused\w*", code[0]))


def main():
    x = torch.randn(64)
    cf = torch.compile(many_small_ops)
    print("\n第一步 40 个小算子：eager", round(per_call_us(many_small_ops, x), 1), "µs，compile",
          round(per_call_us(cf, x), 1), "µs")
    print("融合后的 kernel：", fused_names(many_small_ops, x))


if __name__ == "__main__":
    main()
