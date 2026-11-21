"""讲义：第 30 章 推理模式与 KV cache。

运行：uv run python ch30_inference_mode.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kv_cache_matches(T: int = 6, d: int = 8, seed: int = 0) -> bool:
    """一步步用 KV cache 算的输出 == 一次性因果 attention。"""
    torch.manual_seed(seed)
    x = torch.randn(T, d)
    Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))
    full = F.scaled_dot_product_attention((x @ Wq)[None], (x @ Wk)[None], (x @ Wv)[None], is_causal=True)[0]
    K, V, outs = [], [], []
    for t in range(T):
        q = x[t : t + 1] @ Wq
        K.append(x[t : t + 1] @ Wk)
        V.append(x[t : t + 1] @ Wv)
        outs.append(F.scaled_dot_product_attention(q[None], torch.cat(K)[None], torch.cat(V)[None])[0])
    return torch.allclose(torch.cat(outs), full, atol=1e-5)


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    return 2 * layers * kv_heads * head_dim * bytes_per


def decode_ceiling(bandwidth: float, weight_bytes: float) -> float:
    return bandwidth / weight_bytes


def inference_tensor_error() -> str:
    m = nn.Linear(10, 10)
    with torch.inference_mode():
        b = m(torch.randn(2, 10))
    try:
        m(b).sum().backward()
        return "没报错"
    except RuntimeError as e:
        return str(e).split(".")[0]


def main():
    print("\n第一步 inference 张量拿去训练：", inference_tensor_error())
    print("\n第二步 KV cache 与整段计算一致：", kv_cache_matches())
    llama2 = kv_bytes_per_token(32, 32, 128)
    print("\n第三步 Llama-2-7B 每 token", llama2, "字节，4096 token", llama2 * 4096 / 2**30, "GiB；GQA 8 个 KV 头",
          kv_bytes_per_token(32, 8, 128), "字节")
    print("\n第四步 7B bf16 在 H100 上 decode 上限 ≈", round(decode_ceiling(3.35e12, 14e9)), "tok/s")


if __name__ == "__main__":
    main()
