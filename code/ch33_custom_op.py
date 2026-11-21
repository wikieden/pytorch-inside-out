"""讲义：第 33 章 注册自定义算子。

运行：uv run python ch33_custom_op.py
"""

import torch
from torch import Tensor


@torch.library.custom_op("ptdd::add_relu", mutates_args=())
def add_relu(x: Tensor, y: Tensor) -> Tensor:
    return torch.relu(x + y)          # 真正的实现：可以换成 Triton / C++ / 任何东西


@add_relu.register_fake
def _(x, y):
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(output)


def _backward(ctx, grad):
    (out,) = ctx.saved_tensors
    g = grad * (out > 0)
    return g, g


add_relu.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("ptdd::times2_nofake", mutates_args=())
def times2_nofake(x: Tensor) -> Tensor:
    return x * 2


def forward_backward():
    x = torch.tensor([-2.0, 1.0, 3.0], requires_grad=True)
    y = torch.tensor([1.0, -3.0, 1.0], requires_grad=True)
    out = add_relu(x, y)
    out.sum().backward()
    return out.tolist(), x.grad.tolist(), y.grad.tolist(), type(out.grad_fn).__name__


def opcheck():
    return torch.library.opcheck(add_relu, (torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)))


def graph_breaks():
    def f(a, b):
        return add_relu(a * 2, b).sin()

    e = torch._dynamo.explain(f)(torch.randn(4), torch.randn(4))
    return e.graph_count, e.graph_break_count


def compile_without_fake() -> str:
    torch._dynamo.reset()
    try:
        torch.compile(lambda a: times2_nofake(a) + 1, fullgraph=True)(torch.randn(3))
        return "编译成功"
    except Exception as e:
        return type(e).__name__


def main():
    print("\n第二步 没有 fake 的算子去编译：", compile_without_fake())
    print("有 fake 的算子在图里：(图数, 断图数) =", graph_breaks())
    print("\n第三步 前向、两个输入的梯度、grad_fn：", forward_backward())
    print("\n第四步 opcheck：", opcheck())


if __name__ == "__main__":
    main()
