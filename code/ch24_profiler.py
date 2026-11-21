"""讲义：第 24 章 profiler：给训练拍一段"录像"。

运行：uv run python ch24_profiler.py     # CPU 上录一个小 MLP，比较默认 AdamW 与 fused
"""

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule


def make(fused: bool):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512), nn.LayerNorm(512))
    return m, torch.optim.AdamW(m.parameters(), fused=fused), torch.randn(64, 512)


def profile_step(fused: bool, active: int = 10):
    """返回 (每步总 ms, 每步优化器 ms, 优化器占比, 按 Self CPU 排的前 5 个算子名)。"""
    m, opt, x = make(fused)

    def step():
        with record_function("train_step"):
            loss = m(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait=1, warmup=2, active=active)) as prof:
        for _ in range(3 + active):
            step()
            prof.step()
    ka = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in ka)
    opt_ev = next(e for e in ka if e.key.startswith("Optimizer.step"))
    top = [e.key for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:5]]
    return total / active / 1e3, opt_ev.cpu_time_total / active / 1e3, opt_ev.cpu_time_total / total, top


def amdahl(p: float, s: float) -> float:
    return 1 / ((1 - p) + p / s)


def main():
    for fused in (False, True):
        t, o, share, top = profile_step(fused)
        print(f"\nfused={fused}: 每步 {t:.2f} ms，优化器 {o:.2f} ms（{share:.0%}）；Self 前 5：{top}")
    print("\nAmdahl：占 47.9% 的部分快 2.45 倍 → 总加速", round(amdahl(0.479, 2.45), 2))


if __name__ == "__main__":
    main()
