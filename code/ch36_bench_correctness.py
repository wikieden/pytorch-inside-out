"""讲义：第 36 章 benchmark 与正确性：测准、对齐、再上线。

运行：uv run python ch36_bench_correctness.py
"""

import time

import torch
from torch.utils import benchmark


def naive_timings(n: int = 1 << 22, reps: int = 30):
    x, y = torch.randn(n), torch.randn(n)
    t = time.perf_counter()
    torch.relu(x + y)
    first = (time.perf_counter() - t) * 1e3
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        torch.relu(x + y)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return first, ts[len(ts) // 2]


def timer_median_ms(n: int = 1 << 22, threads: int = 1) -> float:
    x, y = torch.randn(n), torch.randn(n)
    m = benchmark.Timer("torch.relu(x + y)", globals=dict(x=x, y=y, torch=torch), num_threads=threads)
    return m.blocked_autorange(min_run_time=0.5).median * 1e3


def effective_gbs(nbytes: float, ms: float) -> float:
    return nbytes / ms / 1e6


def tolerance_checks():
    """在 1.0 上加 d，assert_close 是否通过。"""
    res = {}
    for dt, ds in ((torch.float32, (1e-7, 1e-5, 1e-3)), (torch.bfloat16, (0.0078, 0.016, 0.03))):
        b = torch.tensor([1.0], dtype=dt)
        for d in ds:
            a = (b.float() + d).to(dt)
            try:
                torch.testing.assert_close(a, b)
                res[(str(dt), d)] = True
            except AssertionError:
                res[(str(dt), d)] = False
    return res


def noncontiguous_bug():
    t = torch.arange(6.0).view(2, 3).t()
    return t.reshape(-1).tolist(), torch.as_strided(t, (6,), (1,)).tolist()


def main():
    print("\n第一步 自己掐表 (首次 ms, 中位数 ms)：", naive_timings())
    print("benchmark.Timer 默认 1 线程中位数", round(timer_median_ms(), 2), "ms；用全部线程", round(timer_median_ms(threads=torch.get_num_threads()), 2), "ms")
    print("\n第二步 50.3 MB / 0.70 ms ≈", round(effective_gbs(3 * (1 << 22) * 4, 0.70)), "GB/s")
    print("\n第三步 容差：", tolerance_checks())
    print("\n第四步 不连续输入 正确 vs 按 storage 读：", noncontiguous_bug())


if __name__ == "__main__":
    main()
