"""讲义：第 2 章 Tensor 内部：一块内存 + 一张"怎么读"的说明书。

运行：uv run python ch02_tensor_stride.py
"""

import torch


def describe(t: torch.Tensor) -> dict:
    """说明书：shape、stride、offset、是否连续。"""
    return dict(shape=tuple(t.shape), stride=t.stride(), offset=t.storage_offset(), contiguous=t.is_contiguous())


def position(offset: int, stride: tuple, index: tuple) -> int:
    """元素在 storage 里的位置 = offset + Σ i_k × stride_k。"""
    return offset + sum(i * s for i, s in zip(index, stride))


def try_view_flat(t: torch.Tensor):
    """不连续时 view 会报错；返回 (是否成功, 结果或报错开头)。"""
    try:
        return True, t.view(-1).tolist()
    except RuntimeError as e:
        return False, str(e).split(".")[0]


def main():
    x = torch.arange(6).view(2, 3)
    print("\n第一步 一排数 + 说明书")
    print("x 的说明书：", describe(x))
    print("x[1, 2] 的位置 = 0 + 1×3 + 2×1 =", position(0, x.stride(), (1, 2)), "→ 值", x[1, 2].item())
    print("\n第二步 view 只改说明书")
    t = x.t()
    print("x.t() 的说明书：", describe(t))
    print("x.t()[2, 1] 位置 =", position(0, t.stride(), (2, 1)))
    s = x[:, 1:]
    print("x[:, 1:] 的说明书：", describe(s))
    print("共用内存：", x.data_ptr() == t.data_ptr())
    print("\n第三步 contiguous")
    print("x.t().view(6)：", try_view_flat(t))
    print("x.t().contiguous().view(6)：", t.contiguous().view(6).tolist())
    print("x.t().reshape(6)：", t.reshape(6).tolist())
    print("\n第四步 stride = 0")
    b = torch.tensor([10, 20, 30])
    print("b.expand(4, 3) 的 stride：", b.expand(4, 3).stride(), "；repeat 的 stride：", b.repeat(4, 1).stride())


if __name__ == "__main__":
    main()
