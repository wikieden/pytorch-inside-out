"""讲义：第 4 章 Autograd：账本怎么记、怎么倒着算。

运行：uv run python ch04_autograd.py
"""

import torch


def numeric_slope(w: float = 3.0, h: float = 1e-3) -> float:
    """L = w² 的数值导数：拧一点点看变化。"""
    return ((w + h) ** 2 - w**2) / h


def chain_example(x=2.0, w=3.0, b=1.0):
    """z = w·x + b，L = z²；返回 (z, L, w.grad, b.grad, 账本节点名)。"""
    xt = torch.tensor(x)
    wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    z = wt * xt + bt
    L = z**2
    names = [type(L.grad_fn).__name__, type(L.grad_fn.next_functions[0][0]).__name__]
    names += [type(f).__name__ for f, _ in L.grad_fn.next_functions[0][0].next_functions]
    L.backward(retain_graph=True)
    g1 = (wt.grad.item(), bt.grad.item())
    L.backward()
    g2 = (wt.grad.item(), bt.grad.item())
    return z.item(), L.item(), g1, g2, names


def inplace_error() -> str:
    w = torch.tensor(3.0, requires_grad=True)
    a = w.exp()
    a.add_(1)
    try:
        a.sum().backward()
        return "没报错"
    except RuntimeError as e:
        return str(e).split(":")[0]


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * (x > 0)


def myrelu_grad():
    t = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
    MyReLU.apply(t).sum().backward()
    ok = torch.autograd.gradcheck(MyReLU.apply, (torch.tensor([-1.0, 2.0, 3.0], dtype=torch.double, requires_grad=True),))
    return t.grad.tolist(), ok


def main():
    print("\n第一步 L = w²，w = 3 的数值斜率 ≈", round(numeric_slope(), 3))
    z, L, g1, g2, names = chain_example()
    print("\n第二步 z =", z, " L =", L, " (w.grad, b.grad) =", g1)
    print("\n第三步 账本：", names)
    print("\n第四步 再反向一次（不清零）：", g2)
    print("原地修改：", inplace_error())
    print("\n第五步 MyReLU 梯度与 gradcheck：", myrelu_grad())


if __name__ == "__main__":
    main()
