"""讲义：第 10 章 保存与加载：存清单、不存对象，断点要存全。

运行：uv run python ch10_save_load.py
"""

import io
import tempfile

import torch
import torch.nn as nn


def size_mb(obj) -> float:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell() / 1e6


def checkpoint_sizes(n: int = 1000):
    """nn.Linear(n, n) 训一步后，模型和 AdamW 存档各多少 MB。"""
    m = nn.Linear(n, n)
    opt = torch.optim.AdamW(m.parameters())
    m(torch.randn(2, n)).sum().backward()
    opt.step()
    return size_mb(m.state_dict()), size_mb(opt.state_dict()), list(opt.state_dict()["state"][0].keys())


def strict_false():
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    sd = dict(m.state_dict())
    sd.pop("1.bias")
    return m.load_state_dict(sd, strict=False)


class Evil:
    def __reduce__(self):
        return (print, ("!!! 存档里的代码被执行了",))


def weights_only_blocks() -> str:
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"x": Evil()}, f.name)
        try:
            torch.load(f.name)
            return "加载成功（危险）"
        except Exception as e:
            return type(e).__name__


def main():
    print("\n第一步 strict=False 返回：", strict_false())
    print("\n第二步 weights_only 默认挡住了：", weights_only_blocks())
    print("\n第三步 1000×1000 Linear：模型 / AdamW / 状态键 =", checkpoint_sizes())


if __name__ == "__main__":
    main()
