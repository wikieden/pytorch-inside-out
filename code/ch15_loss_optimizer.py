"""讲义：第 15 章 损失与优化器：cross_entropy 的细节和 AdamW 的一步。

运行：uv run python ch15_loss_optimizer.py
"""

import torch
import torch.nn.functional as F


def ce_examples():
    lg = torch.tensor([[2.0, 1.0, 0.1]])
    lg2 = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.0]])
    return dict(
        softmax=torch.softmax(lg, -1).tolist(),
        ce=F.cross_entropy(lg, torch.tensor([0])).item(),
        ignore=F.cross_entropy(lg2, torch.tensor([0, -100])).item(),
        both=F.cross_entropy(lg2, torch.tensor([0, 1])).item(),
        smooth=F.cross_entropy(lg, torch.tensor([0]), label_smoothing=0.1).item(),
    )


def adamw_steps(theta=1.0, g=0.5, lr=0.1, wd=0.01, steps=3):
    p = torch.nn.Parameter(torch.tensor([theta]))
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=wd)
    out = []
    for _ in range(steps):
        p.grad = torch.tensor([g])
        opt.step()
        st = opt.state[p]
        out.append((round(p.item(), 6), st["exp_avg"].item(), st["exp_avg_sq"].item()))
    return out


def adamw_first_step_small_grad(g=0.0005):
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.0)
    p.grad = torch.tensor([g])
    opt.step()
    return p.item()


def main():
    print("\n第一、二步 cross_entropy：", ce_examples())
    print("\n第三步 AdamW 三步 (θ, m, v)：", adamw_steps())
    print("梯度 0.0005 时第一步后 θ：", adamw_first_step_small_grad())


if __name__ == "__main__":
    main()
