"""讲义：第 32 章 Triton 编程模型：一个 program 处理一块数据。

运行：uv run python ch32_triton_model.py
Mac 上没有 Triton（只有 Linux 版），这里用 PyTorch 在 CPU 上模拟 program / offs / mask 的规则；
真正的 Triton kernel 见讲义，标 [GPU]。
"""

import torch


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def programs(n: int, block: int) -> list:
    """每个 program 的 (pid, offs 起点, offs 终点, mask 为真的个数)。"""
    return [(pid, pid * block, pid * block + block - 1, min(block, n - pid * block)) for pid in range(cdiv(n, block))]


def add_relu_blocked(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """按 Triton 的方式一块一块算：offs = pid × BLOCK + arange，mask 挡越界。"""
    n = x.numel()
    out = torch.empty_like(x)
    for pid in range(cdiv(n, block)):
        offs = pid * block + torch.arange(block)
        mask = offs < n
        o = offs[mask]
        out[o] = torch.clamp(x[o] + y[o], min=0)
    return out


def row_softmax_blocked(x: torch.Tensor, block: int) -> torch.Tensor:
    """一行一个 program；越界位置补 −∞，exp 后正好是 0。"""
    rows, N = x.shape
    out = torch.empty_like(x)
    cols = torch.arange(block)
    for pid in range(rows):
        row = torch.full((block,), float("-inf"))
        row[cols < N] = x[pid]
        e = torch.exp(row - row.max())
        out[pid] = (e / e.sum())[:N]
    return out


def main():
    print("\n第二步 n = 2500、BLOCK = 1024：", programs(2500, 1024))
    x, y = torch.randn(2500), torch.randn(2500)
    print("分块算的 add_relu 与 torch 一致：", torch.equal(add_relu_blocked(x, y), torch.relu(x + y)))
    a = torch.randn(4, 1000)
    print("\n第三步 一行一个 program 的 softmax（BLOCK = 1024）与 torch 一致：",
          torch.allclose(row_softmax_blocked(a, 1024), torch.softmax(a, -1), atol=1e-6))


if __name__ == "__main__":
    main()
