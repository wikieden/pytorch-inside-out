"""讲义：第 26 章 混合精度：bf16、fp16、TF32、FP8 怎么选。

运行：uv run python ch26_mixed_precision.py
"""

import torch

FORMATS = [torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]


def finfo_table():
    return {str(d).replace("torch.", ""): (torch.finfo(d).max, torch.finfo(d).eps, torch.finfo(d).tiny) for d in FORMATS}


def rounding_examples():
    return dict(bf16=torch.tensor(1.001, dtype=torch.bfloat16).item(),
                fp16=round(torch.tensor(1.001, dtype=torch.float16).item(), 4),
                e4m3=torch.tensor(3.14159).to(torch.float8_e4m3fn).float().item(),
                fp16_overflow=(torch.tensor([1e4], dtype=torch.float16) * 10).item())


def loss_scaling(g: float = 1e-8, scale: float = 65536):
    return torch.tensor(g).half().item(), (torch.tensor(g) * scale).half().item()


def main():
    print("\n第一步 finfo (max, eps, tiny)：")
    for k, v in finfo_table().items():
        print(f"  {k:14s} {v}")
    print("舍入例子：", rounding_examples())
    print("\n第二步 梯度 1e-8 转 fp16 / 放大 65536 倍后转 fp16：", loss_scaling())
    print("\n第三步 fp32 矩阵乘默认精度：", torch.get_float32_matmul_precision())


if __name__ == "__main__":
    main()
