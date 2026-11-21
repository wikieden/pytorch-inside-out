"""讲义：第 11 章 形状与广播：从右往左对齐，1 可以拉伸。

运行：uv run python ch11_shape_broadcast.py
"""

import torch
from einops import rearrange


def broadcast_shape(a: tuple, b: tuple):
    """能广播返回结果形状，不能返回报错信息。"""
    try:
        return tuple((torch.ones(a) + torch.ones(b)).shape)
    except RuntimeError as e:
        return str(e)


def addition_table():
    return torch.tensor([[1.0], [2.0], [3.0]]) + torch.tensor([10.0, 20.0])


def keepdim_trap(n: int = 3):
    """B = D 时 x - x.mean(1) 不报错但错；keepdim 才对。返回两者是否一样。"""
    torch.manual_seed(0)
    x = torch.randn(n, n)
    wrong = x - x.mean(dim=1)
    right = x - x.mean(dim=1, keepdim=True)
    return torch.allclose(wrong, right), torch.allclose(right.mean(1), torch.zeros(n), atol=1e-6)


def split_heads(B=2, T=5, D=8, H=2):
    q = torch.randn(B, T, D)
    a = q.view(B, T, H, D // H).transpose(1, 2)
    b = rearrange(q, "b t (h d) -> b h t d", h=H)
    return tuple(a.shape), a.stride(), torch.equal(a, b)


def main():
    print("\n第一步 广播：")
    for a, b in [((2, 3, 4), (4,)), ((2, 3, 4), (3, 1)), ((2, 3, 4), (3,)), ((3, 1), (2,))]:
        print(f"  {a} + {b} → {broadcast_shape(a, b)}")
    print("加法表：\n", addition_table())
    print("\n第二步 keepdim 陷阱（错的等于对的？对的每行均值为 0？）：", keepdim_trap())
    print("\n第三、四步 拆头 (形状, stride, 与 einops 一致)：", split_heads())


if __name__ == "__main__":
    main()
