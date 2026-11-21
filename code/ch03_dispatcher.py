"""讲义：第 3 章 分派器：按标签一层层处理。

运行：uv run python ch03_dispatcher.py
"""

import torch


def keys(t: torch.Tensor) -> str:
    """张量身上的 dispatch key（内部 API，只用来学习）。"""
    return str(torch._C._dispatch_keys(t))


def autocast_matmul():
    """autocast 里做 x @ w：返回 dtype、grad_fn 名字、上游节点名字。"""
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    w = torch.randn(3, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = x @ w
    ups = [None if f is None else type(f).__name__ for f, _ in y.grad_fn.next_functions]
    return y.dtype, type(y.grad_fn).__name__, ups


def meta_matmul(n: int = 10**6):
    """meta 设备上的 n × n 矩阵乘：只算形状，返回 (名义字节数, 结果形状)。"""
    m = torch.empty(n, n, device="meta")
    return m.numel() * m.element_size(), tuple((m @ m.T).shape)


def main():
    print("\n第一步 看一眼真实的标签")
    print("CPU 张量：", keys(torch.randn(2)))
    if torch.backends.mps.is_available():
        print("MPS 张量：", keys(torch.randn(2, device="mps")))
    print("\n第二步 autocast → autograd → 厨房")
    print("dtype / grad_fn / 上游：", autocast_matmul())
    print("\n第四步 Meta 与 inference_mode")
    nbytes, shape = meta_matmul()
    print(f"meta 张量名义上 {nbytes / 1e12:.0f} TB，m @ m.T 形状 {shape}")
    with torch.inference_mode():
        z = torch.randn(2)
    print("inference_mode 里的张量：", keys(z))


if __name__ == "__main__":
    main()
