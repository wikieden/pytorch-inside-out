"""讲义：第 9 章 训练循环：五行主干 + 三件护具。

运行：uv run python ch09_training_loop.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_line(steps: int = 21, lr: float = 0.5, seed: int = 0):
    """用 nn.Linear(1, 1) + SGD 拟合 y = 3x + 1；返回 {步: (更新前 loss, 更新后 w, b)}。"""
    torch.manual_seed(seed)
    X = torch.linspace(-1, 1, 32).unsqueeze(1)
    Y = 3 * X + 1
    m = nn.Linear(1, 1)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    log = {}
    for step in range(steps):
        loss = F.mse_loss(m(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log[step] = (loss.item(), m.weight.item(), m.bias.item())
    return log


def clip_demo(grad=(3.0, 4.0), max_norm: float = 1.0):
    p = nn.Parameter(torch.zeros(len(grad)))
    p.grad = torch.tensor(grad)
    n = torch.nn.utils.clip_grad_norm_([p], max_norm)
    return n.item(), p.grad.tolist()


def warmup_cosine(step: int, base: float = 3e-4, warmup: int = 100, total: int = 1000) -> float:
    if step < warmup:
        return base * step / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def scheduler_lrs(base: float = 3e-4, warmup: int = 100, total: int = 1000, at=(0, 50, 100, 550, 1000)):
    """用 LambdaLR 真跑一遍，取指定步的学习率。"""
    opt = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=base)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, base, warmup, total) / base)
    lrs = []
    for _ in range(total + 1):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return [lrs[s] for s in at]


def main():
    log = fit_line()
    print("\n第一步 拟合 y = 3x + 1：")
    for s in (0, 1, 2, 5, 10, 20):
        loss, w, b = log[s]
        print(f"  步 {s:2d}: loss {loss:.4f}  w {w:.3f}  b {b:.3f}")
    print("\n第三步 裁剪 [3, 4] 到 1：", clip_demo())
    print("\n第四步 warmup + cosine 第 0/50/100/550/1000 步：", [f"{v:.2e}" for v in scheduler_lrs()])


if __name__ == "__main__":
    main()
