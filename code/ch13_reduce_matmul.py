"""讲义：第 13 章 规约、矩阵乘与数值稳定。

运行：uv run python ch13_reduce_matmul.py
"""

import torch


def reductions():
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return t.sum(0).tolist(), t.sum(1).tolist(), tuple(t.sum(1, keepdim=True).shape), t.max(-1)


def einsum_checks():
    A, B = torch.randn(4, 2, 3), torch.randn(4, 3, 5)
    dot = torch.einsum("i,i->", torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6])).item()
    return dot, torch.allclose(torch.einsum("bij,bjk->bik", A, B), A @ B)


def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def logsumexp_demo():
    big = torch.tensor([1000.0, 1001.0, 1002.0])
    return torch.exp(big).sum().log().item(), torch.logsumexp(big, 0).item(), torch.softmax(big, 0).tolist()


def bf16_running_sum(n: int = 1000):
    s = torch.zeros((), dtype=torch.bfloat16)
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    for _ in range(n):
        s = s + one
    return s.item(), torch.ones(4096, dtype=torch.bfloat16).sum().item()


def main():
    print("\n第一步 sum(0) / sum(1) / keepdim 形状 / max：", reductions())
    print("\n第二步 einsum 点积、和 @ 一致：", einsum_checks(), " 4096³ FLOPs =", f"{matmul_flops(4096, 4096, 4096):.3e}")
    print("\n第三步 直接算 / logsumexp / softmax：", logsumexp_demo())
    print("\n第四步 bf16 逐个加 1000 次 / sum 4096 个 1：", bf16_running_sum())


if __name__ == "__main__":
    main()
