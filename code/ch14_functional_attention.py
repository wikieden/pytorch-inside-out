"""讲义：第 14 章 nn.functional：归一化、激活与注意力。

运行：uv run python ch14_functional_attention.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norms():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    return F.layer_norm(x, (4,)), nn.RMSNorm(4)(x).detach()


def activations():
    z = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
    return F.relu(z), F.gelu(z), F.silu(z)


def swiglu(x, w1, w2, w3):
    return (F.silu(x @ w1) * (x @ w3)) @ w2


def manual_attention(q, k, v):
    T, d = q.shape[-2], q.shape[-1]
    s = q @ k.transpose(-1, -2) / math.sqrt(d)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    return s.softmax(-1) @ v


def sdpa_vs_manual(seed: int = 0) -> float:
    torch.manual_seed(seed)
    q, k, v = (torch.randn(1, 2, 4, 8) for _ in range(3))
    return (F.scaled_dot_product_attention(q, k, v, is_causal=True) - manual_attention(q, k, v)).abs().max().item()


def scores_bytes(B: int, H: int, T: int, bytes_per: int = 2) -> int:
    """手写 attention 的分数表字节数。"""
    return B * H * T * T * bytes_per


def main():
    print("\n第二步 LayerNorm / RMSNorm：", norms())
    print("\n第三步 ReLU / GELU / SiLU：", activations())
    print("\n第四步 SDPA 与手写差：", sdpa_vs_manual())
    print("分数表（B=1, H=32, T=16384, bf16）：", scores_bytes(1, 32, 16384) / 1e9, "GB")


if __name__ == "__main__":
    main()
