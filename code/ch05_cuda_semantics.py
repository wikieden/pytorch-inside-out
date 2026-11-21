"""讲义：第 5 章 GPU 语义：下单就走、何时会等、显存怎么借。

运行：uv run python ch05_cuda_semantics.py      # 在 Mac 上用 MPS 演示；没有 GPU 时跳过计时
"""

import time

import torch


def gpu_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.synchronize
    if torch.backends.mps.is_available():
        return "mps", torch.mps.synchronize
    return None, None


def launch_vs_done(n: int = 4096):
    """返回 (下单毫秒, 做完毫秒)：不同步掐表只测到下单。"""
    dev, sync = gpu_device()
    if dev is None:
        return None
    a = torch.randn(n, n, device=dev)
    for _ in range(3):
        a @ a
    sync()
    t0 = time.perf_counter()
    a @ a
    t1 = time.perf_counter()
    sync()
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t0) * 1e3


def allocator_demo():
    """返回 (两个 64 MB 张量时在用 MB, del 一个后在用 MB)。只在 MPS / CUDA 上有意义。"""
    dev, sync = gpu_device()
    if dev is None:
        return None
    used = torch.mps.current_allocated_memory if dev == "mps" else torch.cuda.memory_allocated
    base = used()
    a = torch.randn(4096, 4096, device=dev)
    c = torch.randn(4096, 4096, device=dev)
    sync()
    two = (used() - base) / 2**20
    del c
    sync()
    one = (used() - base) / 2**20
    del a
    return two, one


def main():
    print("\n第一步 下单 vs 做完（ms）：", launch_vs_done())
    print("\n第四步 缓存分配器：在用（MB）", allocator_demo())


if __name__ == "__main__":
    main()
