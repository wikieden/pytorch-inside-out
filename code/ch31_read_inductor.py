"""讲义：第 31 章 读 Inductor 生成的代码。

运行：uv run python ch31_read_inductor.py
"""

import re

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code


def softmax2(x):
    return torch.softmax(x * 2, -1)


def mlp(x, w, b):
    return F.gelu(x @ w + b)


def generated(fn, *args) -> str:
    torch._dynamo.reset()
    _, code = run_and_get_code(torch.compile(fn), *args)
    return code[0]


def summarize(src: str) -> dict:
    return dict(fused=sorted(set(re.findall(r"cpp_fused\w*", src))),
                extern=sorted(set(re.findall(r"extern_kernels\.(\w+)", src))),
                vectorized="at::vec::Vectorized" in src)


def main():
    print("\n第二步 softmax(x * 2)：", summarize(generated(softmax2, torch.randn(8, 64))))
    print("gelu(x @ w + b)：", summarize(generated(mlp, torch.randn(8, 64), torch.randn(64, 32), torch.randn(32))))


if __name__ == "__main__":
    main()
