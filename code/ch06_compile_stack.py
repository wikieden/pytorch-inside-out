"""讲义：第 6 章 eager 与 graph：torch.compile 里面的三道工序。

运行：uv run python ch06_compile_stack.py     # 本机 CPU 上 Inductor 生成 C++ 代码，第一次要编译几秒
"""

import re

import torch
from torch._inductor.utils import run_and_get_code


def f(x):
    return (x.sin().cos() * 2).sum(-1)


def g(x):
    y = x.sin()
    if y.sum() > 0:
        y = y * 2
    return y.cos()


def fused_kernel_names(fn, *args) -> set:
    """编译 fn，返回生成代码里的融合 kernel 名字。"""
    torch._dynamo.reset()
    _, code = run_and_get_code(torch.compile(fn), *args)
    return set(re.findall(r"cpp_fused\w*", code[0]))


def graph_breaks(fn, *args):
    e = torch._dynamo.explain(fn)(*args)
    return e.graph_count, e.graph_break_count


def main():
    x = torch.randn(64, 64)
    print("\n第二步 四个算子融成：", fused_kernel_names(f, x))
    print("\n第三步 断图：f 的 (图数, 断图数) =", graph_breaks(f, x), "；g =", graph_breaks(g, x))
    print("\n第四步 形状变化：TORCH_LOGS=recompiles 运行可看到只重编译一次")


if __name__ == "__main__":
    main()
