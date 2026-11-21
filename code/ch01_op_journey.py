"""讲义：第 1 章 PyTorch 做了什么：一行 y = x @ w 的旅程。

运行：uv run python ch01_op_journey.py
"""

import torch


def journey(requires_grad: bool, device: str = "cpu"):
    """算一次 y = x @ w，返回 y 和它身上的 grad_fn（账本上的那一笔）。"""
    torch.manual_seed(0)
    x = torch.randn(2, 3, device=device)
    w = torch.randn(3, 4, device=device, requires_grad=requires_grad)
    y = x @ w
    return y, y.grad_fn


def journey_inference(device: str = "cpu"):
    """在 inference_mode 里算同一行：整段都不记账。"""
    torch.manual_seed(0)
    x = torch.randn(2, 3, device=device)
    w = torch.randn(3, 4, device=device, requires_grad=True)
    with torch.inference_mode():
        y = x @ w
    return y, y.grad_fn


def main():
    print("\n第一步 numpy 能算，PyTorch 多了 GPU 和自动求导")
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("本机可用的 GPU 后端：", dev)
    print("\n第二步 四站：前端 → 分派器 → autograd 记一笔 → kernel")
    y, fn = journey(requires_grad=True)
    print("w 要梯度：y.grad_fn =", type(fn).__name__)
    y, fn = journey(requires_grad=False)
    print("w 不要梯度：y.grad_fn =", fn)
    y, fn = journey_inference()
    print("inference_mode 里：y.grad_fn =", fn)


if __name__ == "__main__":
    main()
