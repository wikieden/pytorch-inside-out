"""讲义：第 27 章 显存：谁占了、怎么省、怎么看。

运行：uv run python ch27_memory.py
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def layer_activation_bytes(s: int, b: int, h: int, a: int = 0) -> float:
    """每层激活字节估算 s·b·h·(34 + 5as/h)；a = 0 表示只算第一项（用了 FlashAttention）。"""
    return s * b * h * (34 + 5 * a * s / h)


def saved_activation_mb(use_checkpoint: bool, B=32, D=1024, L=8) -> float:
    """统计为反向存下的张量字节（不算参数本身）。"""
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) for _ in range(L)])
    params = {p.untyped_storage().data_ptr() for p in blocks.parameters()}
    x = torch.randn(B, D, requires_grad=True)
    saved = [0]

    def pack(t):
        if t.untyped_storage().data_ptr() not in params:
            saved[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h = x
        for blk in blocks:
            h = checkpoint(blk, h, use_reentrant=False) if use_checkpoint else blk(h)
        loss = h.pow(2).mean()
    loss.backward()
    return saved[0] / 2**20


def main():
    per = layer_activation_bytes(4096, 1, 4096)
    print("\n第一步 7B（h=4096）、s=4096、b=1：每层", round(per / 1e6), "MB，32 层", round(per * 32 / 1e9, 1), "GB")
    print("\n第二步 存下的激活：不重计算", saved_activation_mb(False), "MB，checkpoint", saved_activation_mb(True), "MB")


if __name__ == "__main__":
    main()
