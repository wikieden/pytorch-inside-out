"""讲义：第 29 章 为什么融合能变快：算力受限与访存受限。

运行：uv run python ch29_why_fusion.py     # Mac 上用 MPS 计时；没有 GPU 时用 CPU
"""

import time

import torch
import torch.nn.functional as F


def ridge(peak_flops: float, bandwidth: float) -> float:
    return peak_flops / bandwidth


def intensity_matmul(n: int, bytes_per: int = 2) -> float:
    return 2 * n**3 / (3 * n * n * bytes_per)


def f(x, b, s):
    return F.gelu(x + b) * s


def fusion_benchmark(n: int = 1 << 24, reps: int = 20):
    """返回 [(名字, 毫秒, 读写字节, 有效 GB/s)]：eager 读写 8 个张量，compile 4 个。"""
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev, lambda: None)
    x, b, s = (torch.randn(n, device=dev) for _ in range(3))
    out = []
    for name, fn, tensors in (("eager", f, 8), ("compile", torch.compile(f), 4)):
        for _ in range(5):
            fn(x, b, s)
        sync()
        t = time.perf_counter()
        for _ in range(reps):
            fn(x, b, s)
        sync()
        ms = (time.perf_counter() - t) / reps * 1e3
        nbytes = tensors * n * 4
        out.append((name, round(ms, 2), nbytes, round(nbytes / ms / 1e6)))
    return dev, out


def main():
    print("\n第一步 H100 拐点 ≈", round(ridge(989e12, 3.35e12)), "FLOP/字节")
    print("\n第二步 fp32 add 算术强度 1/12 ≈", round(1 / 12, 3), "；矩阵乘 n=4096（bf16）≈", round(intensity_matmul(4096)))
    print("\n第三步 gelu(x+b)*s 融合前后：", fusion_benchmark())


if __name__ == "__main__":
    main()
